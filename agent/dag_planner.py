"""
DAG Task Planner — 多Agent协同框架第一层

将用户需求拆解为有向无环图（DAG），每个节点包含：
- task_id: 唯一标识
- goal: 任务目标
- executor: 执行者（hermes/cc/codex/agency-agent角色）
- dependencies: 依赖的前置任务ID列表
- token_budget: 预估token消耗
- risk_level: 风险等级（low/medium/high/critical）
- parallel_group: 可并行执行的组号
- status: 任务状态

底层逻辑：拓扑排序 + 并行分组 + 关键路径分析
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ExecutorType(str, Enum):
    HERMES = "hermes"           # Hermes直接执行
    CLAUDE_CODE = "cc"          # Claude Code执行
    CODEX = "codex"             # Codex执行
    AGENCY_AGENT = "agency"     # agency-agents角色执行
    DELEGATE = "delegate"       # 子Agent执行


@dataclass
class TaskNode:
    """DAG中的一个任务节点"""
    task_id: str
    goal: str
    executor: ExecutorType
    dependencies: List[str] = field(default_factory=list)
    token_budget: int = 5000
    risk_level: RiskLevel = RiskLevel.MEDIUM
    parallel_group: int = 0
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskNode:
        data['executor'] = ExecutorType(data.get('executor', 'hermes'))
        data['risk_level'] = RiskLevel(data.get('risk_level', 'medium'))
        return cls(**data)


@dataclass
class DAGPlan:
    """完整的DAG执行计划"""
    plan_id: str
    root_goal: str
    nodes: Dict[str, TaskNode] = field(default_factory=dict)
    adjacency: Dict[str, List[str]] = field(default_factory=dict)  # task_id -> [dependent_task_ids]
    reverse_adjacency: Dict[str, List[str]] = field(default_factory=dict)  # task_id -> [dependency_task_ids]
    parallel_groups: Dict[int, List[str]] = field(default_factory=dict)  # group_id -> [task_ids]
    critical_path: List[str] = field(default_factory=list)
    total_token_budget: int = 0
    created_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'plan_id': self.plan_id,
            'root_goal': self.root_goal,
            'nodes': {k: v.to_dict() for k, v in self.nodes.items()},
            'adjacency': self.adjacency,
            'reverse_adjacency': self.reverse_adjacency,
            'parallel_groups': self.parallel_groups,
            'critical_path': self.critical_path,
            'total_token_budget': self.total_token_budget,
            'created_at': self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DAGPlan:
        nodes = {k: TaskNode.from_dict(v) for k, v in data.get('nodes', {}).items()}
        return cls(
            plan_id=data['plan_id'],
            root_goal=data['root_goal'],
            nodes=nodes,
            adjacency=data.get('adjacency', {}),
            reverse_adjacency=data.get('reverse_adjacency', {}),
            parallel_groups={int(k): v for k, v in data.get('parallel_groups', {}).items()},
            critical_path=data.get('critical_path', []),
            total_token_budget=data.get('total_token_budget', 0),
            created_at=data.get('created_at', 0.0),
        )


class DAGPlanner:
    """
    DAG任务拆解器
    
    核心职责：
    1. 将用户需求拆解为子任务
    2. 建立依赖关系（有向无环图）
    3. 识别可并行执行的任务组
    4. 计算关键路径和总token预算
    5. 分配执行者（根据任务类型选择最优Agent）
    """

    def __init__(self):
        self._plans: Dict[str, DAGPlan] = {}

    def create_plan(
        self,
        root_goal: str,
        subtasks: List[Dict[str, Any]],
    ) -> DAGPlan:
        """
        创建DAG执行计划
        
        Args:
            root_goal: 用户的原始需求
            subtasks: 子任务列表，每个子任务包含:
                - goal: 任务目标
                - executor: 执行者类型（可选，默认hermes）
                - dependencies: 依赖的任务索引列表（可选）
                - token_budget: token预算（可选）
                - risk_level: 风险等级（可选）
        
        Returns:
            DAGPlan: 完整的执行计划
        """
        import time
        
        plan_id = f"plan_{uuid.uuid4().hex[:12]}"
        
        # 创建任务节点
        nodes: Dict[str, TaskNode] = {}
        task_ids: List[str] = []
        
        for i, subtask in enumerate(subtasks):
            task_id = f"{plan_id}_t{i:03d}"
            task_ids.append(task_id)
            
            # 解析执行者
            executor_str = subtask.get('executor', 'hermes')
            try:
                executor = ExecutorType(executor_str)
            except ValueError:
                executor = ExecutorType.HERMES
            
            # 解析风险等级
            risk_str = subtask.get('risk_level', 'medium')
            try:
                risk_level = RiskLevel(risk_str)
            except ValueError:
                risk_level = RiskLevel.MEDIUM
            
            # 解析依赖（索引转ID）
            dep_indices = subtask.get('dependencies', [])
            dependencies = [task_ids[idx] for idx in dep_indices if 0 <= idx < i]
            
            nodes[task_id] = TaskNode(
                task_id=task_id,
                goal=subtask['goal'],
                executor=executor,
                dependencies=dependencies,
                token_budget=subtask.get('token_budget', 5000),
                risk_level=risk_level,
                metadata=subtask.get('metadata', {}),
            )
        
        # 建立邻接表
        adjacency: Dict[str, List[str]] = {tid: [] for tid in task_ids}
        reverse_adjacency: Dict[str, List[str]] = {tid: [] for tid in task_ids}
        
        for task_id, node in nodes.items():
            for dep_id in node.dependencies:
                adjacency[dep_id].append(task_id)
                reverse_adjacency[task_id].append(dep_id)
        
        # 拓扑排序 + 并行分组
        parallel_groups = self._compute_parallel_groups(nodes, adjacency)
        
        # 计算关键路径
        critical_path = self._compute_critical_path(nodes, adjacency, reverse_adjacency)
        
        # 计算总token预算
        total_token_budget = sum(n.token_budget for n in nodes.values())
        
        plan = DAGPlan(
            plan_id=plan_id,
            root_goal=root_goal,
            nodes=nodes,
            adjacency=adjacency,
            reverse_adjacency=reverse_adjacency,
            parallel_groups=parallel_groups,
            critical_path=critical_path,
            total_token_budget=total_token_budget,
            created_at=time.time(),
        )
        
        self._plans[plan_id] = plan
        
        logger.info(f"DAG plan created: {plan_id}, {len(nodes)} tasks, "
                    f"{len(parallel_groups)} parallel groups, "
                    f"critical path length: {len(critical_path)}")
        
        return plan

    def _compute_parallel_groups(
        self,
        nodes: Dict[str, TaskNode],
        adjacency: Dict[str, List[str]],
    ) -> Dict[int, List[str]]:
        """
        计算可并行执行的任务组
        
        使用BFS层级遍历：同一层级的任务可以并行执行
        """
        # 计算每个节点的入度
        in_degree: Dict[str, int] = {tid: 0 for tid in nodes}
        for task_id, node in nodes.items():
            for dep_id in node.dependencies:
                in_degree[task_id] += 1
        
        # BFS层级遍历
        queue: deque = deque()
        for tid, deg in in_degree.items():
            if deg == 0:
                queue.append(tid)
        
        groups: Dict[int, List[str]] = {}
        group_id = 0
        visited: Set[str] = set()
        
        while queue:
            current_level = list(queue)
            queue.clear()
            
            if current_level:
                groups[group_id] = current_level
                # 更新节点的parallel_group
                for tid in current_level:
                    nodes[tid].parallel_group = group_id
                    visited.add(tid)
                group_id += 1
            
            for tid in current_level:
                for neighbor in adjacency.get(tid, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        
        # 检查是否有环
        if len(visited) != len(nodes):
            logger.warning("DAG contains cycle! Some tasks may not be reachable.")
        
        return groups

    def _compute_critical_path(
        self,
        nodes: Dict[str, TaskNode],
        adjacency: Dict[str, List[str]],
        reverse_adjacency: Dict[str, List[str]],
    ) -> List[str]:
        """
        计算关键路径（最长路径）
        
        关键路径决定了任务的最短完成时间
        """
        # 计算每个节点的最早开始时间（ES）
        es: Dict[str, int] = {tid: 0 for tid in nodes}
        
        # 拓扑排序
        in_degree: Dict[str, int] = {tid: len(nodes[tid].dependencies) for tid in nodes}
        queue: deque = deque([tid for tid, deg in in_degree.items() if deg == 0])
        topo_order: List[str] = []
        
        while queue:
            tid = queue.popleft()
            topo_order.append(tid)
            for neighbor in adjacency.get(tid, []):
                es[neighbor] = max(es[neighbor], es[tid] + 1)
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 计算每个节点的最晚开始时间（LS）
        max_es = max(es.values()) if es else 0
        ls: Dict[str, int] = {tid: max_es for tid in nodes}
        
        for tid in reversed(topo_order):
            for neighbor in adjacency.get(tid, []):
                ls[tid] = min(ls[tid], ls[neighbor] - 1)
        
        # 关键路径：ES == LS 的节点
        critical_path = [tid for tid in topo_order if es[tid] == ls[tid]]
        
        return critical_path

    def get_plan(self, plan_id: str) -> Optional[DAGPlan]:
        """获取指定的执行计划"""
        return self._plans.get(plan_id)

    def get_ready_tasks(self, plan_id: str) -> List[TaskNode]:
        """获取当前可以执行的任务（所有依赖已完成）"""
        plan = self._plans.get(plan_id)
        if not plan:
            return []
        
        ready = []
        for task_id, node in plan.nodes.items():
            if node.status != "pending":
                continue
            
            # 检查所有依赖是否已完成
            all_deps_done = all(
                plan.nodes[dep_id].status == "completed"
                for dep_id in node.dependencies
                if dep_id in plan.nodes
            )
            
            if all_deps_done:
                ready.append(node)
        
        return ready

    def update_task_status(
        self,
        plan_id: str,
        task_id: str,
        status: str,
        result: Optional[str] = None,
        error: Optional[str] = None,
    ) -> bool:
        """更新任务状态"""
        plan = self._plans.get(plan_id)
        if not plan or task_id not in plan.nodes:
            return False
        
        import time
        
        node = plan.nodes[task_id]
        node.status = status
        
        if status == "executing" and node.started_at is None:
            node.started_at = time.time()
        elif status == "completed":
            node.completed_at = time.time()
            node.result = result
        elif status == "failed":
            node.completed_at = time.time()
            node.error = error
        
        logger.info(f"Task {task_id} status updated to {status}")
        return True

    def get_plan_summary(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """获取计划摘要"""
        plan = self._plans.get(plan_id)
        if not plan:
            return None
        
        total = len(plan.nodes)
        completed = sum(1 for n in plan.nodes.values() if n.status == "completed")
        failed = sum(1 for n in plan.nodes.values() if n.status == "failed")
        executing = sum(1 for n in plan.nodes.values() if n.status == "executing")
        pending = sum(1 for n in plan.nodes.values() if n.status == "pending")
        
        return {
            'plan_id': plan.plan_id,
            'root_goal': plan.root_goal,
            'total_tasks': total,
            'completed': completed,
            'failed': failed,
            'executing': executing,
            'pending': pending,
            'progress': f"{completed}/{total}" if total > 0 else "0/0",
            'parallel_groups': len(plan.parallel_groups),
            'critical_path_length': len(plan.critical_path),
            'total_token_budget': plan.total_token_budget,
        }

    def serialize_plan(self, plan_id: str) -> Optional[str]:
        """序列化计划为JSON"""
        plan = self._plans.get(plan_id)
        if not plan:
            return None
        return json.dumps(plan.to_dict(), ensure_ascii=False, indent=2)

    def deserialize_plan(self, data: str) -> Optional[DAGPlan]:
        """从JSON反序列化计划"""
        try:
            plan_data = json.loads(data)
            plan = DAGPlan.from_dict(plan_data)
            self._plans[plan.plan_id] = plan
            return plan
        except Exception as e:
            logger.error(f"Failed to deserialize plan: {e}")
            return None


# 全局单例
_dag_planner: Optional[DAGPlanner] = None


def get_dag_planner() -> DAGPlanner:
    """获取全局DAG拆解器实例"""
    global _dag_planner
    if _dag_planner is None:
        _dag_planner = DAGPlanner()
    return _dag_planner
