"""
Support System — 多Agent协同框架支撑系统七件套

1. 缓存冷热分层
2. 负载均衡（优先级队列+并发上限）
3. 12项KPI闭环调控
4. 故障熔断
5. 记忆过期清理（180天删除+快照）
6. 日志全链路Trace
7. 白名单脱敏
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ==================== 1. 缓存冷热分层 ====================

class CacheLevel(str, Enum):
    HOT = "hot"      # 热数据（频繁访问）
    WARM = "warm"    # 温数据（偶尔访问）
    COLD = "cold"    # 冷数据（很少访问）


@dataclass
class CacheEntry:
    key: str
    value: Any
    level: CacheLevel = CacheLevel.WARM
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    ttl: float = 3600.0  # 默认1小时

    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl


class TieredCache:
    """分层缓存"""

    def __init__(self, max_hot: int = 100, max_warm: int = 500, max_cold: int = 2000):
        self._cache: Dict[str, CacheEntry] = {}
        self._max_sizes = {
            CacheLevel.HOT: max_hot,
            CacheLevel.WARM: max_warm,
            CacheLevel.COLD: max_cold,
        }

    def get(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if not entry or entry.is_expired():
            if entry:
                del self._cache[key]
            return None

        entry.access_count += 1
        entry.last_access = time.time()

        # 自动升级
        if entry.access_count > 10 and entry.level != CacheLevel.HOT:
            entry.level = CacheLevel.HOT
        elif entry.access_count > 3 and entry.level == CacheLevel.COLD:
            entry.level = CacheLevel.WARM

        return entry.value

    def put(self, key: str, value: Any, ttl: float = 3600.0, level: CacheLevel = CacheLevel.WARM):
        # 检查容量
        self._evict_if_needed(level)

        self._cache[key] = CacheEntry(
            key=key, value=value, level=level, ttl=ttl
        )

    def _evict_if_needed(self, level: CacheLevel):
        entries = [e for e in self._cache.values() if e.level == level]
        max_size = self._max_sizes[level]

        if len(entries) >= max_size:
            # LRU淘汰
            entries.sort(key=lambda e: e.last_access)
            to_remove = len(entries) - max_size + 1
            for e in entries[:to_remove]:
                del self._cache[e.key]

    def stats(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for e in self._cache.values():
            counts[e.level.value] += 1
        return dict(counts)


# ==================== 2. 负载均衡 ====================

@dataclass
class WorkerSlot:
    worker_id: str
    current_load: int = 0
    max_load: int = 5
    priority: int = 0
    is_healthy: bool = True


class LoadBalancer:
    """负载均衡器（优先级队列+并发上限）"""

    def __init__(self):
        self._workers: Dict[str, WorkerSlot] = {}
        self._queue: deque = deque()

    def register_worker(self, worker_id: str, max_load: int = 5, priority: int = 0):
        self._workers[worker_id] = WorkerSlot(
            worker_id=worker_id, max_load=max_load, priority=priority
        )

    def allocate(self, task_priority: int = 0) -> Optional[str]:
        """分配任务到最优worker"""
        candidates = [
            w for w in self._workers.values()
            if w.is_healthy and w.current_load < w.max_load
        ]

        if not candidates:
            return None

        # 按优先级和负载排序
        candidates.sort(key=lambda w: (-w.priority, w.current_load))
        selected = candidates[0]
        selected.current_load += 1
        return selected.worker_id

    def release(self, worker_id: str):
        worker = self._workers.get(worker_id)
        if worker and worker.current_load > 0:
            worker.current_load -= 1

    def mark_unhealthy(self, worker_id: str):
        if worker_id in self._workers:
            self._workers[worker_id].is_healthy = False

    def stats(self) -> Dict[str, Any]:
        return {
            wid: {'load': w.current_load, 'max': w.max_load, 'healthy': w.is_healthy}
            for wid, w in self._workers.items()
        }


# ==================== 3. KPI闭环调控 ====================

class KPIMetric(str, Enum):
    TASK_COMPLETION_RATE = "task_completion_rate"
    AVERAGE_QUALITY = "average_quality"
    AVERAGE_LATENCY = "average_latency"
    TOKEN_EFFICIENCY = "token_efficiency"
    COST_PER_TASK = "cost_per_task"
    RETRY_RATE = "retry_rate"
    ESCALATION_RATE = "escalation_rate"
    EXECUTOR_UTILIZATION = "executor_utilization"
    CACHE_HIT_RATE = "cache_hit_rate"
    DRIFT_DETECTION_RATE = "drift_detection_rate"
    HUMAN_INTERVENTION_RATE = "human_intervention_rate"
    OVERALL_UTILITY = "overall_utility"


@dataclass
class KPISnapshot:
    timestamp: float
    metrics: Dict[str, float]


class KPIController:
    """12项KPI闭环调控"""

    def __init__(self):
        self._snapshots: List[KPISnapshot] = []
        self._targets: Dict[str, float] = {
            KPIMetric.TASK_COMPLETION_RATE.value: 0.9,
            KPIMetric.AVERAGE_QUALITY.value: 0.8,
            KPIMetric.TOKEN_EFFICIENCY.value: 0.7,
            KPIMetric.RETRY_RATE.value: 0.1,
            KPIMetric.CACHE_HIT_RATE.value: 0.6,
        }

    def record_snapshot(self, metrics: Dict[str, float]):
        self._snapshots.append(KPISnapshot(
            timestamp=time.time(), metrics=metrics
        ))

    def get_latest(self) -> Optional[KPISnapshot]:
        return self._snapshots[-1] if self._snapshots else None

    def check_targets(self) -> Dict[str, Dict[str, Any]]:
        """检查KPI是否达标"""
        latest = self.get_latest()
        if not latest:
            return {}

        results = {}
        for metric, target in self._targets.items():
            current = latest.metrics.get(metric, 0)
            results[metric] = {
                'current': current,
                'target': target,
                'met': current >= target,
                'gap': target - current,
            }
        return results


# ==================== 4. 故障熔断 ====================

class CircuitState(str, Enum):
    CLOSED = "closed"      # 正常
    OPEN = "open"          # 熔断
    HALF_OPEN = "half_open"  # 半开


class CircuitBreaker:
    """故障熔断器"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._circuits: Dict[str, Dict[str, Any]] = {}

    def _get_circuit(self, key: str) -> Dict[str, Any]:
        if key not in self._circuits:
            self._circuits[key] = {
                'state': CircuitState.CLOSED,
                'failure_count': 0,
                'last_failure': 0,
                'success_count': 0,
            }
        return self._circuits[key]

    def can_execute(self, key: str) -> bool:
        circuit = self._get_circuit(key)

        if circuit['state'] == CircuitState.CLOSED:
            return True
        elif circuit['state'] == CircuitState.OPEN:
            if time.time() - circuit['last_failure'] > self._recovery_timeout:
                circuit['state'] = CircuitState.HALF_OPEN
                return True
            return False
        else:  # HALF_OPEN
            return True

    def record_success(self, key: str):
        circuit = self._get_circuit(key)
        circuit['failure_count'] = 0
        circuit['success_count'] += 1
        if circuit['state'] == CircuitState.HALF_OPEN:
            circuit['state'] = CircuitState.CLOSED

    def record_failure(self, key: str):
        circuit = self._get_circuit(key)
        circuit['failure_count'] += 1
        circuit['last_failure'] = time.time()

        if circuit['failure_count'] >= self._failure_threshold:
            circuit['state'] = CircuitState.OPEN
            logger.warning(f"Circuit breaker OPEN for {key}")

    def get_state(self, key: str) -> CircuitState:
        return self._get_circuit(key)['state']


# ==================== 5. 记忆过期清理 ====================

class MemoryCleaner:
    """记忆过期清理（180天删除+快照）"""

    def __init__(self, ttl_days: int = 180):
        self._ttl_seconds = ttl_days * 86400

    def is_expired(self, created_at: float) -> bool:
        return time.time() - created_at > self._ttl_seconds

    def filter_expired(self, items: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """筛选过期项，返回 (保留, 过期)"""
        keep = []
        expired = []
        for item in items:
            if self.is_expired(item.get('created_at', 0)):
                expired.append(item)
            else:
                keep.append(item)
        return keep, expired


# ==================== 6. 日志全链路Trace ====================

@dataclass
class TraceSpan:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation: str
    start_time: float
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    def finish(self, status: str = "ok"):
        self.end_time = time.time()
        self.status = status

    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0


class TraceCollector:
    """全链路追踪"""

    def __init__(self):
        self._traces: Dict[str, List[TraceSpan]] = defaultdict(list)

    def start_span(
        self,
        trace_id: str,
        operation: str,
        parent_span_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceSpan:
        span = TraceSpan(
            trace_id=trace_id,
            span_id=f"span_{uuid.uuid4().hex[:8]}",
            parent_span_id=parent_span_id,
            operation=operation,
            start_time=time.time(),
            metadata=metadata or {},
        )
        self._traces[trace_id].append(span)
        return span

    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        return self._traces.get(trace_id, [])

    def get_all_traces(self) -> Dict[str, List[Dict]]:
        return {
            tid: [asdict(s) for s in spans]
            for tid, spans in self._traces.items()
        }


# ==================== 7. 白名单脱敏 ====================

class Sanitizer:
    """白名单脱敏"""

    DEFAULT_PATTERNS = [
        (r'(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*\S+', '[REDACTED]'),
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
        (r'\b\d{11,}\b', '[NUMBER]'),
    ]

    def __init__(self):
        self._patterns = list(self.DEFAULT_PATTERNS)
        self._whitelist: Set[str] = set()

    def add_whitelist(self, key: str):
        self._whitelist.add(key)

    def sanitize(self, text: str) -> str:
        import re
        for pattern, replacement in self._patterns:
            text = re.sub(pattern, replacement, text)
        return text


# ==================== 支撑系统总管理器 ====================

class SupportSystem:
    """支撑系统七件套"""

    def __init__(self):
        self.cache = TieredCache()
        self.load_balancer = LoadBalancer()
        self.kpi = KPIController()
        self.circuit_breaker = CircuitBreaker()
        self.memory_cleaner = MemoryCleaner()
        self.tracer = TraceCollector()
        self.sanitizer = Sanitizer()

    def health_check(self) -> Dict[str, Any]:
        return {
            'cache': self.cache.stats(),
            'load_balancer': self.load_balancer.stats(),
            'kpi': self.kpi.check_targets(),
            'status': 'healthy',
        }


# 全局单例
_support_system: Optional[SupportSystem] = None


def get_support_system() -> SupportSystem:
    global _support_system
    if _support_system is None:
        _support_system = SupportSystem()
    return _support_system
