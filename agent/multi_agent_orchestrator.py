"""
Multi-Agent Orchestrator — 多Agent协同框架总编排器

连接四层架构：
    DAG任务拆解 → 状态机驱动 → 仲裁器裁决 → 评估闭环进化

以及支撑系统七件套。

这是整个框架的入口点，提供统一的API供Hermes调用。
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """编排器配置"""
    max_retries: int = 3
    quality_threshold: float = 0.7
    budget_warning_ratio: float = 0.8
    enable_eval_loop: bool = True
    enable_support_system: bool = True
    state_persistence: bool = True


class MultiAgentOrchestrator:
    """
    多Agent协同框架总编排器
    
    职责：
    1. 接收用户需求
    2. 调用DAG拆解器生成执行计划
    3. 通过状态机管理任务生命周期
    4. 通过仲裁器解决所有冲突
    5. 通过评估闭环驱动进化
    6. 通过支撑系统提供基础设施
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        
        # 延迟导入，避免循环依赖
        from agent.dag_planner import get_dag_planner
        from agent.arbiter import get_arbiter
        from agent.task_state_machine import get_state_machine_manager
        from agent.eval_loop import get_eval_loop
        from agent.support_system import get_support_system
        
        self.dag = get_dag_planner()
        self.arbiter = get_arbiter()
        self.state_machine = get_state_machine_manager(arbiter=self.arbiter)
        self.eval_loop = get_eval_loop()
        self.support = get_support_system()
        
        # 配置仲裁器
        self.arbiter.set_quality_threshold(self.config.quality_threshold)
        self.arbiter.set_budget_warning_ratio(self.config.budget_warning_ratio)
        
        # 注册状态转换回调
        self.state_machine.register_transition_callback(self._on_state_change)
        
        # 当前活跃计划
        self._active_plan_id: Optional[str] = None
        
        logger.info("Multi-Agent Orchestrator initialized")

    def _on_state_change(self, task_id: str, from_state: str, to_state: str, trigger: str):
        """状态变化回调"""
        logger.info(f"[Orchestrator] Task {task_id}: {from_state} → {to_state}")

    def process_request(
        self,
        request: str,
        subtasks: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        处理用户请求（主入口）
        
        Args:
            request: 用户需求描述
            subtasks: 预定义的子任务列表（可选，不提供则自动拆解）
            context: 附加上下文
        
        Returns:
            处理结果
        """
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        span = self.support.tracer.start_span(trace_id, "process_request")
        
        try:
            # 1. 自动拆解（如果未提供subtasks）
            if not subtasks:
                subtasks = self._auto_decompose(request, context)
            
            # 2. 创建DAG计划
            plan = self.dag.create_plan(root_goal=request, subtasks=subtasks)
            self._active_plan_id = plan.plan_id
            
            # 3. 为每个任务创建状态机
            for task_id in plan.nodes:
                self.state_machine.create_machine(
                    task_id=task_id,
                    max_retries=self.config.max_retries,
                )
                # 转换到decomposed状态
                self.state_machine.transition(
                    task_id=task_id,
                    trigger=self.state_machine._machines[task_id].__class__.__mro__[0].__class__.__mro__[0],
                    skip_arbiter=True,
                )
            
            # 4. 获取可执行的任务
            ready_tasks = self.dag.get_ready_tasks(plan.plan_id)
            
            # 5. 执行任务
            results = []
            for task in ready_tasks:
                result = self._execute_task(task, trace_id)
                results.append(result)
            
            # 6. 生成摘要
            summary = self.dag.get_plan_summary(plan.plan_id)
            
            span.finish("ok")
            
            return {
                'success': True,
                'plan_id': plan.plan_id,
                'trace_id': trace_id,
                'summary': summary,
                'results': results,
            }
            
        except Exception as e:
            span.finish("error")
            logger.error(f"Process request failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'trace_id': trace_id,
            }

    def _auto_decompose(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        自动拆解需求为子任务
        
        这是一个简化的拆解逻辑，实际使用时应该由LLM辅助拆解
        """
        # 简单拆解：将请求作为单个任务
        return [
            {
                'goal': request,
                'executor': 'hermes',
                'risk_level': 'medium',
                'token_budget': 10000,
            }
        ]

    def _execute_task(self, task, trace_id: str) -> Dict[str, Any]:
        """执行单个任务"""
        task_id = task.task_id
        
        # 更新状态为executing
        self.state_machine.transition(
            task_id=task_id,
            trigger=self.state_machine._machines[task_id].__class__.__mro__[0].__class__.__mro__[0],
            skip_arbiter=True,
        )
        
        # 检查熔断器
        if not self.support.circuit_breaker.can_execute(task.executor.value):
            return {
                'task_id': task_id,
                'status': 'circuit_open',
                'error': 'Circuit breaker is open',
            }
        
        # 记录开始
        span = self.support.tracer.start_span(
            trace_id, f"execute_{task_id}",
            metadata={'executor': task.executor.value}
        )
        
        try:
            # 根据executor类型选择执行方式
            if task.executor.value == 'hermes':
                result = self._execute_hermes(task)
            elif task.executor.value == 'cc':
                result = self._execute_claude_code(task)
            elif task.executor.value == 'codex':
                result = self._execute_codex(task)
            else:
                result = self._execute_delegate(task)
            
            # 记录成功
            self.support.circuit_breaker.record_success(task.executor.value)
            span.finish("ok")
            
            return result
            
        except Exception as e:
            # 记录失败
            self.support.circuit_breaker.record_failure(task.executor.value)
            span.finish("error")
            
            return {
                'task_id': task_id,
                'status': 'failed',
                'error': str(e),
            }

    def _execute_hermes(self, task) -> Dict[str, Any]:
        """Hermes直接执行"""
        return {
            'task_id': task.task_id,
            'status': 'completed',
            'executor': 'hermes',
            'result': f"Task '{task.goal}' executed by Hermes",
        }

    def _execute_claude_code(self, task) -> Dict[str, Any]:
        """Claude Code执行"""
        return {
            'task_id': task.task_id,
            'status': 'completed',
            'executor': 'cc',
            'result': f"Task '{task.goal}' delegated to Claude Code",
        }

    def _execute_codex(self, task) -> Dict[str, Any]:
        """Codex执行"""
        return {
            'task_id': task.task_id,
            'status': 'completed',
            'executor': 'codex',
            'result': f"Task '{task.goal}' delegated to Codex",
        }

    def _execute_delegate(self, task) -> Dict[str, Any]:
        """子Agent执行"""
        return {
            'task_id': task.task_id,
            'status': 'completed',
            'executor': 'delegate',
            'result': f"Task '{task.goal}' delegated to sub-agent",
        }

    def get_status(self) -> Dict[str, Any]:
        """获取编排器状态"""
        return {
            'active_plan_id': self._active_plan_id,
            'plan_summary': (
                self.dag.get_plan_summary(self._active_plan_id)
                if self._active_plan_id else None
            ),
            'executor_weights': self.eval_loop.get_executor_weights(),
            'arbiter_stats': self.arbiter.get_statistics(),
            'eval_stats': self.eval_loop.get_statistics(),
            'support_health': self.support.health_check(),
        }

    def save_state(self) -> str:
        """保存状态到JSON"""
        state = {
            'active_plan_id': self._active_plan_id,
            'dag_plans': {
                pid: self.dag.serialize_plan(pid)
                for pid in [self._active_plan_id]
                if pid
            },
            'state_machines': self.state_machine.serialize_all(),
            'executor_weights': self.eval_loop.get_executor_weights(),
            'arbiter_stats': self.arbiter.get_statistics(),
            'saved_at': time.time(),
        }
        return json.dumps(state, ensure_ascii=False, indent=2)

    def load_state(self, data: str) -> bool:
        """从JSON加载状态"""
        try:
            state = json.loads(data)
            self._active_plan_id = state.get('active_plan_id')
            
            # 恢复DAG计划
            for pid, plan_data in state.get('dag_plans', {}).items():
                if plan_data:
                    self.dag.deserialize_plan(plan_data)
            
            # 恢复状态机
            sm_data = state.get('state_machines', '{}')
            if sm_data:
                self.state_machine.deserialize_all(sm_data)
            
            logger.info("Orchestrator state loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False


# 全局单例
_orchestrator: Optional[MultiAgentOrchestrator] = None


def get_orchestrator(config: Optional[OrchestratorConfig] = None) -> MultiAgentOrchestrator:
    """获取全局编排器实例"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MultiAgentOrchestrator(config=config)
    return _orchestrator
