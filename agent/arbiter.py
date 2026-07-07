"""
Arbiter — 多Agent协同框架第三层（最核心）

仲裁器解决所有分歧：
- 预算超标了要不要砍质检？
- 质检发现问题要不要重做？
- 执行说没问题、质检说有问题——听谁的？

所有冲突不进模块内部协商，统一进 Arbiter。

优先级焊死（不打架）：
    安全 > 正确性 > 质检信号 > 成本预算 > 性能优化

收益函数：
    Utility = 任务价值 - 成本 - 风险惩罚
    系统不是被动省钱，而是主动判断值不值得做。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    """优先级枚举（数值越大优先级越高）"""
    PERFORMANCE = 1      # 性能优化
    COST_BUDGET = 2      # 成本预算
    QUALITY_SIGNAL = 3   # 质检信号
    CORRECTNESS = 4      # 正确性
    SAFETY = 5           # 安全


class ConflictType(str, Enum):
    """冲突类型"""
    BUDGET_EXCEEDED = "budget_exceeded"           # 预算超标
    QUALITY_FAILED = "quality_failed"             # 质检失败
    EXECUTOR_VS_CRITIC = "executor_vs_critic"     # 执行者vs质检者意见不一
    RETRY_DECISION = "retry_decision"             # 是否重试
    ROUTE_SELECTION = "route_selection"           # 路由选择
    STATE_TRANSITION = "state_transition"         # 状态转换审批
    RESOURCE_CONTENTION = "resource_contention"   # 资源竞争


class Decision(str, Enum):
    """仲裁决策"""
    APPROVE = "approve"           # 批准
    REJECT = "reject"             # 拒绝
    RETRY = "retry"               # 重试
    ESCALATE = "escalate"         # 升级（人工介入）
    DEFER = "defer"               # 延后
    ADJUST_BUDGET = "adjust_budget"  # 调整预算
    SKIP_QUALITY = "skip_quality"    # 跳过质检（高风险）


@dataclass
class ConflictContext:
    """冲突上下文"""
    conflict_type: ConflictType
    task_id: str
    description: str
    executor_claims: Dict[str, Any] = field(default_factory=dict)
    critic_claims: Dict[str, Any] = field(default_factory=dict)
    budget_info: Dict[str, Any] = field(default_factory=dict)
    quality_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArbitrationResult:
    """仲裁结果"""
    decision_id: str
    conflict_type: ConflictType
    decision: Decision
    priority_level: Priority
    utility_score: float
    reason: str
    actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'decision_id': self.decision_id,
            'conflict_type': self.conflict_type.value,
            'decision': self.decision.value,
            'priority_level': self.priority_level.value,
            'utility_score': self.utility_score,
            'reason': self.reason,
            'actions': self.actions,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
        }


class Arbiter:
    """
    仲裁器
    
    核心职责：
    1. 统一裁决所有冲突（不进模块内部协商）
    2. 按优先级排序：安全 > 正确性 > 质检信号 > 成本预算 > 性能优化
    3. 引入收益函数：Utility = 任务价值 - 成本 - 风险惩罚
    4. 审批状态转换
    5. 记录所有仲裁决策（可审计）
    """

    def __init__(self):
        self._decisions: List[ArbitrationResult] = []
        self._risk_penalty_weights: Dict[str, float] = {
            'low': 0.1,
            'medium': 0.3,
            'high': 0.6,
            'critical': 1.0,
        }
        self._quality_threshold: float = 0.7  # 质检通过阈值
        self._budget_warning_ratio: float = 0.8  # 预算警告阈值

    def set_quality_threshold(self, threshold: float):
        """设置质检通过阈值"""
        self._quality_threshold = max(0.0, min(1.0, threshold))

    def set_budget_warning_ratio(self, ratio: float):
        """设置预算警告阈值"""
        self._budget_warning_ratio = max(0.0, min(1.0, ratio))

    def calculate_utility(
        self,
        task_value: float,
        cost: float,
        risk_level: str,
        quality_score: float = 1.0,
    ) -> float:
        """
        计算收益函数
        
        Utility = 任务价值 * 质量分数 - 成本 - 风险惩罚
        
        Args:
            task_value: 任务价值（0-1）
            cost: 成本（token消耗）
            risk_level: 风险等级（low/medium/high/critical）
            quality_score: 质量分数（0-1）
        
        Returns:
            utility: 收益值（越高越值得做）
        """
        risk_penalty = self._risk_penalty_weights.get(risk_level, 0.3)
        
        # 归一化成本（假设10000 token为基准）
        normalized_cost = cost / 10000.0
        
        utility = (task_value * quality_score) - normalized_cost - risk_penalty
        
        return round(utility, 4)

    def arbitrate(self, context: ConflictContext) -> ArbitrationResult:
        """
        执行仲裁
        
        Args:
            context: 冲突上下文
        
        Returns:
            ArbitrationResult: 仲裁结果
        """
        decision_id = f"arb_{uuid.uuid4().hex[:12]}"
        
        # 根据冲突类型选择仲裁策略
        if context.conflict_type == ConflictType.BUDGET_EXCEEDED:
            result = self._arbitrate_budget(context, decision_id)
        elif context.conflict_type == ConflictType.QUALITY_FAILED:
            result = self._arbitrate_quality(context, decision_id)
        elif context.conflict_type == ConflictType.EXECUTOR_VS_CRITIC:
            result = self._arbitrate_executor_vs_critic(context, decision_id)
        elif context.conflict_type == ConflictType.RETRY_DECISION:
            result = self._arbitrate_retry(context, decision_id)
        elif context.conflict_type == ConflictType.ROUTE_SELECTION:
            result = self._arbitrate_route(context, decision_id)
        elif context.conflict_type == ConflictType.STATE_TRANSITION:
            result = self._arbitrate_state_transition(context, decision_id)
        elif context.conflict_type == ConflictType.RESOURCE_CONTENTION:
            result = self._arbitrate_resource(context, decision_id)
        else:
            result = ArbitrationResult(
                decision_id=decision_id,
                conflict_type=context.conflict_type,
                decision=Decision.ESCALATE,
                priority_level=Priority.SAFETY,
                utility_score=0.0,
                reason=f"Unknown conflict type: {context.conflict_type}",
            )
        
        self._decisions.append(result)
        
        logger.info(
            f"Arbitration {decision_id}: {context.conflict_type.value} → "
            f"{result.decision.value} (utility: {result.utility_score}, "
            f"priority: {result.priority_level.name})"
        )
        
        return result

    def _arbitrate_budget(self, ctx: ConflictContext, decision_id: str) -> ArbitrationResult:
        """仲裁预算超标"""
        budget_used = ctx.budget_info.get('used', 0)
        budget_limit = ctx.budget_info.get('limit', 10000)
        task_value = ctx.budget_info.get('task_value', 0.5)
        risk_level = ctx.metadata.get('risk_level', 'medium')
        
        usage_ratio = budget_used / budget_limit if budget_limit > 0 else 1.0
        
        utility = self.calculate_utility(
            task_value=task_value,
            cost=budget_used,
            risk_level=risk_level,
        )
        
        if usage_ratio > 1.5:
            # 严重超标：拒绝
            return ArbitrationResult(
                decision_id=decision_id,
                conflict_type=ctx.conflict_type,
                decision=Decision.REJECT,
                priority_level=Priority.COST_BUDGET,
                utility_score=utility,
                reason=f"Budget exceeded by {usage_ratio:.0%}. Rejecting to prevent resource waste.",
                actions=["cancel_task", "log_budget_overrun"],
            )
        elif usage_ratio > 1.0:
            # 轻微超标：调整预算或降级质检
            if utility > 0:
                return ArbitrationResult(
                    decision_id=decision_id,
                    conflict_type=ctx.conflict_type,
                    decision=Decision.ADJUST_BUDGET,
                    priority_level=Priority.COST_BUDGET,
                    utility_score=utility,
                    reason=f"Budget exceeded by {usage_ratio:.0%} but utility is positive. Adjusting budget.",
                    actions=["increase_budget", "log_adjustment"],
                )
            else:
                return ArbitrationResult(
                    decision_id=decision_id,
                    conflict_type=ctx.conflict_type,
                    decision=Decision.REJECT,
                    priority_level=Priority.COST_BUDGET,
                    utility_score=utility,
                    reason=f"Budget exceeded and utility is negative. Rejecting.",
                    actions=["cancel_task", "log_budget_overrun"],
                )
        else:
            # 未超标：批准
            return ArbitrationResult(
                decision_id=decision_id,
                conflict_type=ctx.conflict_type,
                decision=Decision.APPROVE,
                priority_level=Priority.COST_BUDGET,
                utility_score=utility,
                reason="Within budget. Approving.",
            )

    def _arbitrate_quality(self, ctx: ConflictContext, decision_id: str) -> ArbitrationResult:
        """仲裁质检失败"""
        quality_score = ctx.quality_info.get('score', 0.0)
        risk_level = ctx.metadata.get('risk_level', 'medium')
        retry_count = ctx.metadata.get('retry_count', 0)
        max_retries = ctx.metadata.get('max_retries', 3)
        
        utility = self.calculate_utility(
            task_value=ctx.quality_info.get('task_value', 0.5),
            cost=ctx.quality_info.get('cost', 0),
            risk_level=risk_level,
            quality_score=quality_score,
        )
        
        if quality_score >= self._quality_threshold:
            # 质检通过
            return ArbitrationResult(
                decision_id=decision_id,
                conflict_type=ctx.conflict_type,
                decision=Decision.APPROVE,
                priority_level=Priority.QUALITY_SIGNAL,
                utility_score=utility,
                reason=f"Quality score {quality_score:.2f} meets threshold {self._quality_threshold:.2f}.",
            )
        elif retry_count >= max_retries:
            # 重试次数用尽
            return ArbitrationResult(
                decision_id=decision_id,
                conflict_type=ctx.conflict_type,
                decision=Decision.ESCALATE,
                priority_level=Priority.QUALITY_SIGNAL,
                utility_score=utility,
                reason=f"Quality failed and retry limit reached ({retry_count}/{max_retries}). Escalating to human.",
                actions=["escalate_to_human", "log_quality_failure"],
            )
        elif risk_level == 'critical':
            # 关键风险：必须重试
            return ArbitrationResult(
                decision_id=decision_id,
                conflict_type=ctx.conflict_type,
                decision=Decision.RETRY,
                priority_level=Priority.SAFETY,
                utility_score=utility,
                reason=f"Critical risk task failed quality check. Must retry.",
                actions=["retry_task", "log_critical_retry"],
            )
        else:
            # 普通任务：根据utility决定
            if utility > 0:
                return ArbitrationResult(
                    decision_id=decision_id,
                    conflict_type=ctx.conflict_type,
                    decision=Decision.RETRY,
                    priority_level=Priority.QUALITY_SIGNAL,
                    utility_score=utility,
                    reason=f"Quality failed ({quality_score:.2f}) but utility is positive. Retrying.",
                    actions=["retry_task"],
                )
            else:
                return ArbitrationResult(
                    decision_id=decision_id,
                    conflict_type=ctx.conflict_type,
                    decision=Decision.REJECT,
                    priority_level=Priority.QUALITY_SIGNAL,
                    utility_score=utility,
                    reason=f"Quality failed and utility is negative. Rejecting.",
                    actions=["cancel_task", "log_quality_failure"],
                )

    def _arbitrate_executor_vs_critic(self, ctx: ConflictContext, decision_id: str) -> ArbitrationResult:
        """仲裁执行者vs质检者意见不一"""
        executor_says_ok = ctx.executor_claims.get('quality_ok', True)
        critic_says_ok = ctx.critic_claims.get('quality_ok', False)
        critic_score = ctx.critic_claims.get('score', 0.0)
        risk_level = ctx.metadata.get('risk_level', 'medium')
        
        utility = self.calculate_utility(
            task_value=ctx.metadata.get('task_value', 0.5),
            cost=ctx.metadata.get('cost', 0),
            risk_level=risk_level,
            quality_score=critic_score,
        )
        
        # 优先级：质检信号 > 执行者自评
        if not critic_says_ok:
            if risk_level in ('high', 'critical'):
                # 高风险：必须听质检的
                return ArbitrationResult(
                    decision_id=decision_id,
                    conflict_type=ctx.conflict_type,
                    decision=Decision.RETRY,
                    priority_level=Priority.QUALITY_SIGNAL,
                    utility_score=utility,
                    reason=f"Critic failed ({critic_score:.2f}) on {risk_level} risk task. Executor claims OK but overriding.",
                    actions=["retry_task", "log_critic_override"],
                )
            elif utility > 0:
                # utility为正：重试
                return ArbitrationResult(
                    decision_id=decision_id,
                    conflict_type=ctx.conflict_type,
                    decision=Decision.RETRY,
                    priority_level=Priority.QUALITY_SIGNAL,
                    utility_score=utility,
                    reason=f"Critic failed but utility positive. Retrying.",
                    actions=["retry_task"],
                )
            else:
                # utility为负：拒绝
                return ArbitrationResult(
                    decision_id=decision_id,
                    conflict_type=ctx.conflict_type,
                    decision=Decision.REJECT,
                    priority_level=Priority.QUALITY_SIGNAL,
                    utility_score=utility,
                    reason=f"Critic failed and utility negative. Rejecting.",
                    actions=["cancel_task"],
                )
        else:
            # 质检通过：批准
            return ArbitrationResult(
                decision_id=decision_id,
                conflict_type=ctx.conflict_type,
                decision=Decision.APPROVE,
                priority_level=Priority.QUALITY_SIGNAL,
                utility_score=utility,
                reason="Both executor and critic agree. Approving.",
            )

    def _arbitrate_retry(self, ctx: ConflictContext, decision_id: str) -> ArbitrationResult:
        """仲裁是否重试"""
        retry_count = ctx.metadata.get('retry_count', 0)
        max_retries = ctx.metadata.get('max_retries', 3)
        risk_level = ctx.metadata.get('risk_level', 'medium')
        
        utility = self.calculate_utility(
            task_value=ctx.metadata.get('task_value', 0.5),
            cost=ctx.metadata.get('cost', 0),
            risk_level=risk_level,
        )
        
        if retry_count >= max_retries:
            return ArbitrationResult(
                decision_id=decision_id,
                conflict_type=ctx.conflict_type,
                decision=Decision.ESCALATE,
                priority_level=Priority.CORRECTNESS,
                utility_score=utility,
                reason=f"Retry limit reached ({retry_count}/{max_retries}). Escalating.",
                actions=["escalate_to_human"],
            )
        elif utility > 0:
            return ArbitrationResult(
                decision_id=decision_id,
                conflict_type=ctx.conflict_type,
                decision=Decision.RETRY,
                priority_level=Priority.CORRECTNESS,
                utility_score=utility,
                reason=f"Utility positive. Retrying ({retry_count + 1}/{max_retries}).",
                actions=["retry_task"],
            )
        else:
            return ArbitrationResult(
                decision_id=decision_id,
                conflict_type=ctx.conflict_type,
                decision=Decision.REJECT,
                priority_level=Priority.COST_BUDGET,
                utility_score=utility,
                reason=f"Utility negative. Rejecting retry.",
                actions=["cancel_task"],
            )

    def _arbitrate_route(self, ctx: ConflictContext, decision_id: str) -> ArbitrationResult:
        """仲裁路由选择"""
        # 选择executor：根据任务特征选择最优Agent
        task_type = ctx.metadata.get('task_type', 'general')
        risk_level = ctx.metadata.get('risk_level', 'medium')
        
        # 路由规则
        if task_type == 'code' or risk_level in ('high', 'critical'):
            executor = 'cc'  # Claude Code
            reason = "Code/high-risk task → Claude Code"
        elif task_type == 'image' or task_type == 'video':
            executor = 'codex'  # Codex
            reason = "Media task → Codex"
        elif task_type == 'simple' or risk_level == 'low':
            executor = 'hermes'  # Hermes直接执行
            reason = "Simple/low-risk task → Hermes direct"
        else:
            executor = 'delegate'  # 子Agent
            reason = "General task → Delegate"
        
        return ArbitrationResult(
            decision_id=decision_id,
            conflict_type=ctx.conflict_type,
            decision=Decision.APPROVE,
            priority_level=Priority.CORRECTNESS,
            utility_score=0.5,
            reason=reason,
            actions=[f"route_to_{executor}"],
            metadata={'selected_executor': executor},
        )

    def _arbitrate_state_transition(self, ctx: ConflictContext, decision_id: str) -> ArbitrationResult:
        """仲裁状态转换"""
        from_state = ctx.metadata.get('from_state', '')
        to_state = ctx.metadata.get('to_state', '')
        trigger = ctx.metadata.get('trigger', '')
        
        # 所有状态转换都批准（合法性由状态机检查）
        return ArbitrationResult(
            decision_id=decision_id,
            conflict_type=ctx.conflict_type,
            decision=Decision.APPROVE,
            priority_level=Priority.CORRECTNESS,
            utility_score=1.0,
            reason=f"State transition approved: {from_state} → {to_state}",
        )

    def _arbitrate_resource(self, ctx: ConflictContext, decision_id: str) -> ArbitrationResult:
        """仲裁资源竞争"""
        # 按优先级分配资源
        return ArbitrationResult(
            decision_id=decision_id,
            conflict_type=ctx.conflict_type,
            decision=Decision.APPROVE,
            priority_level=Priority.PERFORMANCE,
            utility_score=0.5,
            reason="Resource allocated by priority.",
        )

    def approve_transition(
        self,
        task_id: str,
        from_state: str,
        to_state: str,
        trigger: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        审批状态转换（供状态机调用）
        
        Returns:
            (approved, reason)
        """
        context = ConflictContext(
            conflict_type=ConflictType.STATE_TRANSITION,
            task_id=task_id,
            description=f"State transition: {from_state} → {to_state}",
            metadata={
                'from_state': from_state,
                'to_state': to_state,
                'trigger': trigger,
                **(metadata or {}),
            },
        )
        
        result = self.arbitrate(context)
        
        return result.decision == Decision.APPROVE, result.reason

    def get_decision_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取仲裁决策历史"""
        return [d.to_dict() for d in self._decisions[-limit:]]

    def get_statistics(self) -> Dict[str, Any]:
        """获取仲裁统计"""
        if not self._decisions:
            return {'total_decisions': 0}
        
        from collections import Counter
        
        decision_counts = Counter(d.decision.value for d in self._decisions)
        conflict_counts = Counter(d.conflict_type.value for d in self._decisions)
        avg_utility = sum(d.utility_score for d in self._decisions) / len(self._decisions)
        
        return {
            'total_decisions': len(self._decisions),
            'decision_distribution': dict(decision_counts),
            'conflict_distribution': dict(conflict_counts),
            'average_utility': round(avg_utility, 4),
        }


# 全局单例
_arbiter: Optional[Arbiter] = None


def get_arbiter() -> Arbiter:
    """获取全局仲裁器实例"""
    global _arbiter
    if _arbiter is None:
        _arbiter = Arbiter()
    return _arbiter
