"""
Task State Machine — 多Agent协同框架第二层

任务生命周期状态机：received → decomposed → executing → quality_checking → memorizing → completed

状态转换规则：
- 所有状态转换必须经过仲裁器确认
- 不允许"状态漂移"或流程乱序
- 出错自动回退重做（retrying状态）
- 阻塞等待依赖（blocked状态）

事件驱动：每一步由事件自动推进，不需要人去标注走到哪了
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class TaskState(str, Enum):
    """任务状态枚举"""
    RECEIVED = "received"           # 已收到
    DECOMPOSED = "decomposed"       # 已拆解
    BLOCKED = "blocked"             # 阻塞（等待依赖/人工审批/其他）
    EXECUTING = "executing"         # 执行中
    QUALITY_CHECKING = "quality_checking"  # 质检中
    MEMORIZING = "memorizing"       # 记入记忆中
    COMPLETED = "completed"         # 已完成
    FAILED = "failed"               # 失败
    RETRYING = "retrying"           # 重试中


class TransitionTrigger(str, Enum):
    """状态转换触发事件"""
    TASK_RECEIVED = "task_received"
    DECOMPOSITION_COMPLETE = "decomposition_complete"
    DEPENDENCY_RESOLVED = "dependency_resolved"
    HUMAN_APPROVAL = "human_approval"
    BLOCK_REASON_RESOLVED = "block_reason_resolved"
    EXECUTION_START = "execution_start"
    EXECUTION_COMPLETE = "execution_complete"
    EXECUTION_FAILED = "execution_failed"
    QUALITY_CHECK_START = "quality_check_start"
    QUALITY_CHECK_PASS = "quality_check_pass"
    QUALITY_CHECK_FAIL = "quality_check_fail"
    MEMORY_WRITE_START = "memory_write_start"
    MEMORY_WRITE_COMPLETE = "memory_write_complete"
    RETRY_START = "retry_start"
    RETRY_LIMIT_REACHED = "retry_limit_reached"
    MANUAL_OVERRIDE = "manual_override"


# 合法的状态转换表
VALID_TRANSITIONS: Dict[TaskState, Dict[TransitionTrigger, TaskState]] = {
    TaskState.RECEIVED: {
        TransitionTrigger.TASK_RECEIVED: TaskState.RECEIVED,
        TransitionTrigger.DECOMPOSITION_COMPLETE: TaskState.DECOMPOSED,
    },
    TaskState.DECOMPOSED: {
        TransitionTrigger.EXECUTION_START: TaskState.EXECUTING,
        TransitionTrigger.DEPENDENCY_RESOLVED: TaskState.DECOMPOSED,
        TransitionTrigger.MANUAL_OVERRIDE: TaskState.EXECUTING,
    },
    TaskState.BLOCKED: {
        TransitionTrigger.BLOCK_REASON_RESOLVED: TaskState.DECOMPOSED,
        TransitionTrigger.DEPENDENCY_RESOLVED: TaskState.DECOMPOSED,
        TransitionTrigger.HUMAN_APPROVAL: TaskState.DECOMPOSED,
        TransitionTrigger.MANUAL_OVERRIDE: TaskState.EXECUTING,
    },
    TaskState.EXECUTING: {
        TransitionTrigger.EXECUTION_COMPLETE: TaskState.QUALITY_CHECKING,
        TransitionTrigger.EXECUTION_FAILED: TaskState.FAILED,
        TransitionTrigger.QUALITY_CHECK_START: TaskState.QUALITY_CHECKING,
        TransitionTrigger.MANUAL_OVERRIDE: TaskState.QUALITY_CHECKING,
    },
    TaskState.QUALITY_CHECKING: {
        TransitionTrigger.QUALITY_CHECK_PASS: TaskState.MEMORIZING,
        TransitionTrigger.QUALITY_CHECK_FAIL: TaskState.FAILED,
        TransitionTrigger.MEMORY_WRITE_START: TaskState.MEMORIZING,
        TransitionTrigger.MANUAL_OVERRIDE: TaskState.MEMORIZING,
    },
    TaskState.MEMORIZING: {
        TransitionTrigger.MEMORY_WRITE_COMPLETE: TaskState.COMPLETED,
        TransitionTrigger.MANUAL_OVERRIDE: TaskState.COMPLETED,
    },
    TaskState.COMPLETED: {
        # 终态，不允许转换
    },
    TaskState.FAILED: {
        TransitionTrigger.RETRY_START: TaskState.RETRYING,
        TransitionTrigger.MANUAL_OVERRIDE: TaskState.RETRYING,
        TransitionTrigger.RETRY_LIMIT_REACHED: TaskState.FAILED,  # 保持失败
    },
    TaskState.RETRYING: {
        TransitionTrigger.EXECUTION_START: TaskState.EXECUTING,
        TransitionTrigger.RETRY_LIMIT_REACHED: TaskState.FAILED,
        TransitionTrigger.MANUAL_OVERRIDE: TaskState.EXECUTING,
    },
}


@dataclass
class StateTransition:
    """状态转换记录"""
    from_state: TaskState
    to_state: TaskState
    trigger: TransitionTrigger
    timestamp: float
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'from_state': self.from_state.value,
            'to_state': self.to_state.value,
            'trigger': self.trigger.value,
            'timestamp': self.timestamp,
            'reason': self.reason,
            'metadata': self.metadata,
        }


@dataclass
class TaskStateMachine:
    """单个任务的状态机实例"""
    task_id: str
    current_state: TaskState = TaskState.RECEIVED
    history: List[StateTransition] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    blocked_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'current_state': self.current_state.value,
            'history': [t.to_dict() for t in self.history],
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'blocked_reason': self.blocked_reason,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskStateMachine:
        history = [
            StateTransition(
                from_state=TaskState(t['from_state']),
                to_state=TaskState(t['to_state']),
                trigger=TransitionTrigger(t['trigger']),
                timestamp=t['timestamp'],
                reason=t.get('reason'),
                metadata=t.get('metadata', {}),
            )
            for t in data.get('history', [])
        ]
        return cls(
            task_id=data['task_id'],
            current_state=TaskState(data['current_state']),
            history=history,
            retry_count=data.get('retry_count', 0),
            max_retries=data.get('max_retries', 3),
            blocked_reason=data.get('blocked_reason'),
            metadata=data.get('metadata', {}),
            created_at=data.get('created_at', 0.0),
            updated_at=data.get('updated_at', 0.0),
        )


class StateMachineManager:
    """
    状态机管理器
    
    核心职责：
    1. 管理所有任务的状态机实例
    2. 验证状态转换的合法性
    3. 执行状态转换并记录历史
    4. 处理重试逻辑
    5. 处理阻塞/解除阻塞
    """

    def __init__(self, arbiter=None):
        self._machines: Dict[str, TaskStateMachine] = {}
        self._arbiter = arbiter  # 仲裁器引用，用于验证转换
        self._transition_callbacks: List[Callable] = []

    def set_arbiter(self, arbiter):
        """设置仲裁器"""
        self._arbiter = arbiter

    def register_transition_callback(self, callback: Callable):
        """注册状态转换回调"""
        self._transition_callbacks.append(callback)

    def create_machine(self, task_id: str, max_retries: int = 3) -> TaskStateMachine:
        """创建新的状态机实例"""
        machine = TaskStateMachine(
            task_id=task_id,
            max_retries=max_retries,
        )
        self._machines[task_id] = machine
        logger.info(f"State machine created for task {task_id}")
        return machine

    def get_machine(self, task_id: str) -> Optional[TaskStateMachine]:
        """获取状态机实例"""
        return self._machines.get(task_id)

    def can_transition(
        self,
        task_id: str,
        trigger: TransitionTrigger,
    ) -> Tuple[bool, Optional[str]]:
        """
        检查状态转换是否合法
        
        Returns:
            (is_valid, reason): 是否合法，不合法时返回原因
        """
        machine = self._machines.get(task_id)
        if not machine:
            return False, f"Task {task_id} not found"
        
        current = machine.current_state
        valid_triggers = VALID_TRANSITIONS.get(current, {})
        
        if trigger not in valid_triggers:
            return False, (
                f"Invalid transition: {current.value} + {trigger.value}. "
                f"Valid triggers for {current.value}: "
                f"{[t.value for t in valid_triggers.keys()]}"
            )
        
        # 检查重试限制
        if trigger == TransitionTrigger.RETRY_START:
            if machine.retry_count >= machine.max_retries:
                return False, f"Retry limit reached ({machine.retry_count}/{machine.max_retries})"
        
        return True, None

    def transition(
        self,
        task_id: str,
        trigger: TransitionTrigger,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        skip_arbiter: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """
        执行状态转换
        
        Args:
            task_id: 任务ID
            trigger: 触发事件
            reason: 转换原因
            metadata: 附加数据
            skip_arbiter: 是否跳过仲裁器验证（用于系统内部转换）
        
        Returns:
            (success, error_message)
        """
        # 检查合法性
        can, err = self.can_transition(task_id, trigger)
        if not can:
            logger.warning(f"Transition rejected: {err}")
            return False, err
        
        machine = self._machines[task_id]
        from_state = machine.current_state
        to_state = VALID_TRANSITIONS[from_state][trigger]
        
        # 仲裁器验证（除非跳过）
        if not skip_arbiter and self._arbiter:
            approved, arbiter_reason = self._arbiter.approve_transition(
                task_id=task_id,
                from_state=from_state.value,
                to_state=to_state.value,
                trigger=trigger.value,
                metadata=metadata or {},
            )
            if not approved:
                logger.warning(f"Arbiter rejected transition: {arbiter_reason}")
                return False, f"Arbiter rejected: {arbiter_reason}"
        
        # 执行转换
        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            trigger=trigger,
            timestamp=time.time(),
            reason=reason,
            metadata=metadata or {},
        )
        
        machine.current_state = to_state
        machine.history.append(transition)
        machine.updated_at = time.time()
        
        # 处理重试计数
        if trigger == TransitionTrigger.RETRY_START:
            machine.retry_count += 1
            logger.info(f"Task {task_id} retry {machine.retry_count}/{machine.max_retries}")
        
        # 处理阻塞原因
        if to_state == TaskState.BLOCKED:
            machine.blocked_reason = reason
        elif from_state == TaskState.BLOCKED:
            machine.blocked_reason = None
        
        # 触发回调
        for callback in self._transition_callbacks:
            try:
                callback(task_id, from_state.value, to_state.value, trigger.value)
            except Exception as e:
                logger.error(f"Transition callback error: {e}")
        
        logger.info(
            f"Task {task_id}: {from_state.value} → {to_state.value} "
            f"(trigger: {trigger.value}, reason: {reason})"
        )
        
        return True, None

    def block_task(self, task_id: str, reason: str) -> Tuple[bool, Optional[str]]:
        """阻塞任务（等待依赖/人工审批/其他）"""
        machine = self._machines.get(task_id)
        if not machine:
            return False, f"Task {task_id} not found"
        
        # 只有特定状态可以被阻塞
        blockable_states = {TaskState.DECOMPOSED, TaskState.EXECUTING}
        if machine.current_state not in blockable_states:
            return False, f"Cannot block task in state {machine.current_state.value}"
        
        # 直接设置状态（跳过正常转换流程，因为BLOCKED是特殊状态）
        old_state = machine.current_state
        machine.current_state = TaskState.BLOCKED
        machine.blocked_reason = reason
        machine.updated_at = time.time()
        
        transition = StateTransition(
            from_state=old_state,
            to_state=TaskState.BLOCKED,
            trigger=TransitionTrigger.MANUAL_OVERRIDE,
            timestamp=time.time(),
            reason=reason,
        )
        machine.history.append(transition)
        
        logger.info(f"Task {task_id} blocked: {reason}")
        return True, None

    def unblock_task(self, task_id: str) -> Tuple[bool, Optional[str]]:
        """解除任务阻塞"""
        machine = self._machines.get(task_id)
        if not machine:
            return False, f"Task {task_id} not found"
        
        if machine.current_state != TaskState.BLOCKED:
            return False, f"Task {task_id} is not blocked"
        
        return self.transition(
            task_id=task_id,
            trigger=TransitionTrigger.BLOCK_REASON_RESOLVED,
            reason="Block resolved",
        )

    def get_all_tasks_by_state(self, state: TaskState) -> List[str]:
        """获取指定状态的所有任务ID"""
        return [
            task_id for task_id, machine in self._machines.items()
            if machine.current_state == state
        ]

    def get_blocked_tasks(self) -> List[Tuple[str, str]]:
        """获取所有阻塞的任务及其原因"""
        return [
            (task_id, machine.blocked_reason or "unknown")
            for task_id, machine in self._machines.items()
            if machine.current_state == TaskState.BLOCKED
        ]

    def get_retrying_tasks(self) -> List[Tuple[str, int, int]]:
        """获取所有重试中的任务及其重试次数"""
        return [
            (task_id, machine.retry_count, machine.max_retries)
            for task_id, machine in self._machines.items()
            if machine.current_state == TaskState.RETRYING
        ]

    def serialize_all(self) -> str:
        """序列化所有状态机为JSON"""
        data = {
            task_id: machine.to_dict()
            for task_id, machine in self._machines.items()
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    def deserialize_all(self, data: str) -> int:
        """从JSON反序列化所有状态机"""
        try:
            machines_data = json.loads(data)
            for task_id, machine_data in machines_data.items():
                self._machines[task_id] = TaskStateMachine.from_dict(machine_data)
            return len(machines_data)
        except Exception as e:
            logger.error(f"Failed to deserialize state machines: {e}")
            return 0


# 全局单例
_state_machine_manager: Optional[StateMachineManager] = None


def get_state_machine_manager(arbiter=None) -> StateMachineManager:
    """获取全局状态机管理器实例"""
    global _state_machine_manager
    if _state_machine_manager is None:
        _state_machine_manager = StateMachineManager(arbiter=arbiter)
    return _state_machine_manager
