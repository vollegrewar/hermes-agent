"""
Eval Loop — 多Agent协同框架第四层

评估闭环驱动进化：
    output → 评分 → 反馈 → 策略更新

核心功能：
1. 质量评分（自动评估任务输出质量）
2. 漂移检测（Prompt结构漂移、反馈漂移、仲裁策略偏移）
3. 策略更新（根据评分自动调整路由权重、预算分配）
4. 离线测评（漂移超限时触发自动校准）

不用等人发现系统在退化，Eval Loop 自动出报告。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DriftType(str, Enum):
    """漂移类型"""
    PROMPT_STRUCTURE = "prompt_structure"    # Prompt结构漂移
    FEEDBACK_PATTERN = "feedback_pattern"    # L5反馈漂移
    ARBITRATION_POLICY = "arbitration_policy"  # 仲裁策略偏移
    QUALITY_DEGRADATION = "quality_degradation"  # 质量退化
    COST_DRIFT = "cost_drift"               # 成本漂移


class Severity(str, Enum):
    """严重程度"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class QualityScore:
    """质量评分"""
    task_id: str
    executor: str
    score: float           # 0-1
    dimensions: Dict[str, float] = field(default_factory=dict)  # 各维度分数
    feedback: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DriftReport:
    """漂移报告"""
    drift_id: str
    drift_type: DriftType
    severity: Severity
    description: str
    baseline_value: float
    current_value: float
    threshold: float
    recommendation: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'drift_id': self.drift_id,
            'drift_type': self.drift_type.value,
            'severity': self.severity.value,
            'description': self.description,
            'baseline_value': self.baseline_value,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'recommendation': self.recommendation,
            'timestamp': self.timestamp,
        }


@dataclass
class EvalReport:
    """评估报告"""
    report_id: str
    period_start: float
    period_end: float
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    average_quality_score: float
    executor_scores: Dict[str, float] = field(default_factory=dict)
    drift_reports: List[DriftReport] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'report_id': self.report_id,
            'period_start': self.period_start,
            'period_end': self.period_end,
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'average_quality_score': self.average_quality_score,
            'executor_scores': self.executor_scores,
            'drift_reports': [d.to_dict() for d in self.drift_reports],
            'recommendations': self.recommendations,
            'timestamp': self.timestamp,
        }


class EvalLoop:
    """
    评估闭环
    
    核心职责：
    1. 收集质量评分
    2. 检测漂移（Prompt结构、反馈模式、仲裁策略）
    3. 生成评估报告
    4. 触发策略更新
    5. 维护基线数据
    """

    def __init__(self):
        # 评分历史
        self._scores: List[QualityScore] = []
        
        # 漂移检测配置
        self._drift_thresholds: Dict[DriftType, float] = {
            DriftType.PROMPT_STRUCTURE: 0.3,      # 结构变化超过30%
            DriftType.FEEDBACK_PATTERN: 0.25,      # 反馈模式变化超过25%
            DriftType.ARBITRATION_POLICY: 0.2,     # 仲裁策略变化超过20%
            DriftType.QUALITY_DEGRADATION: 0.15,   # 质量下降超过15%
            DriftType.COST_DRIFT: 0.3,             # 成本漂移超过30%
        }
        
        # 基线数据
        self._baselines: Dict[str, float] = {
            'average_quality': 0.8,
            'average_cost': 5000.0,
            'success_rate': 0.9,
            'prompt_complexity': 0.5,
        }
        
        # 执行者权重（用于路由决策）
        self._executor_weights: Dict[str, float] = {
            'hermes': 1.0,
            'cc': 1.0,
            'codex': 1.0,
            'agency': 1.0,
            'delegate': 1.0,
        }
        
        # 滑动窗口
        self._window_size: int = 100
        self._recent_scores: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self._window_size))
        
        # 漂移报告
        self._drift_reports: List[DriftReport] = []
        
        # 策略更新回调
        self._update_callbacks: List[Callable] = []

    def register_update_callback(self, callback: Callable):
        """注册策略更新回调"""
        self._update_callbacks.append(callback)

    def set_drift_threshold(self, drift_type: DriftType, threshold: float):
        """设置漂移检测阈值"""
        self._drift_thresholds[drift_type] = max(0.0, min(1.0, threshold))

    def set_baseline(self, key: str, value: float):
        """设置基线值"""
        self._baselines[key] = value

    def record_score(self, score: QualityScore):
        """记录质量评分"""
        self._scores.append(score)
        self._recent_scores[score.executor].append(score.score)
        
        # 更新执行者权重
        self._update_executor_weight(score.executor, score.score)
        
        logger.info(
            f"Quality score recorded: task={score.task_id}, "
            f"executor={score.executor}, score={score.score:.2f}"
        )

    def _update_executor_weight(self, executor: str, score: float):
        """根据评分更新执行者权重"""
        if executor not in self._executor_weights:
            return
        
        recent = list(self._recent_scores[executor])
        if len(recent) < 5:
            return
        
        # 计算移动平均
        avg_score = sum(recent) / len(recent)
        
        # 权重调整：高分增加权重，低分降低权重
        if avg_score > 0.8:
            self._executor_weights[executor] = min(2.0, self._executor_weights[executor] * 1.05)
        elif avg_score < 0.5:
            self._executor_weights[executor] = max(0.3, self._executor_weights[executor] * 0.95)

    def get_executor_weight(self, executor: str) -> float:
        """获取执行者权重"""
        return self._executor_weights.get(executor, 1.0)

    def detect_drift(self) -> List[DriftReport]:
        """
        检测漂移
        
        三种漂移检测：
        1. Prompt结构漂移
        2. L5反馈漂移
        3. 仲裁策略偏移
        """
        reports = []
        
        # 检测质量退化
        quality_drift = self._detect_quality_drift()
        if quality_drift:
            reports.append(quality_drift)
        
        # 检测成本漂移
        cost_drift = self._detect_cost_drift()
        if cost_drift:
            reports.append(cost_drift)
        
        # 检测执行者性能漂移
        executor_drifts = self._detect_executor_drift()
        reports.extend(executor_drifts)
        
        self._drift_reports.extend(reports)
        
        if reports:
            logger.warning(f"Detected {len(reports)} drift reports")
            for report in reports:
                logger.warning(f"  - {report.drift_type.value}: {report.description}")
        
        return reports

    def _detect_quality_drift(self) -> Optional[DriftReport]:
        """检测质量退化"""
        if len(self._scores) < 20:
            return None
        
        # 比较最近20个评分与基线
        recent = [s.score for s in self._scores[-20:]]
        avg_recent = sum(recent) / len(recent)
        baseline = self._baselines['average_quality']
        
        drift_amount = baseline - avg_recent
        threshold = self._drift_thresholds[DriftType.QUALITY_DEGRADATION]
        
        if drift_amount > threshold:
            severity = Severity.CRITICAL if drift_amount > threshold * 2 else Severity.WARNING
            return DriftReport(
                drift_id=f"drift_{uuid.uuid4().hex[:8]}",
                drift_type=DriftType.QUALITY_DEGRADATION,
                severity=severity,
                description=f"Quality degraded by {drift_amount:.2%}",
                baseline_value=baseline,
                current_value=avg_recent,
                threshold=threshold,
                recommendation="Review recent task outputs. Consider retraining or adjusting prompts.",
            )
        
        return None

    def _detect_cost_drift(self) -> Optional[DriftReport]:
        """检测成本漂移"""
        if len(self._scores) < 20:
            return None
        
        # 计算最近任务的平均成本（如果有成本数据）
        recent_costs = [
            s.dimensions.get('cost', 0) for s in self._scores[-20:]
            if 'cost' in s.dimensions
        ]
        
        if not recent_costs:
            return None
        
        avg_cost = sum(recent_costs) / len(recent_costs)
        baseline_cost = self._baselines['average_cost']
        
        if baseline_cost == 0:
            return None
        
        drift_ratio = abs(avg_cost - baseline_cost) / baseline_cost
        threshold = self._drift_thresholds[DriftType.COST_DRIFT]
        
        if drift_ratio > threshold:
            severity = Severity.WARNING if drift_ratio < threshold * 2 else Severity.CRITICAL
            return DriftReport(
                drift_id=f"drift_{uuid.uuid4().hex[:8]}",
                drift_type=DriftType.COST_DRIFT,
                severity=severity,
                description=f"Cost drifted by {drift_ratio:.2%}",
                baseline_value=baseline_cost,
                current_value=avg_cost,
                threshold=threshold,
                recommendation="Review token usage patterns. Consider optimizing prompts or adjusting budgets.",
            )
        
        return None

    def _detect_executor_drift(self) -> List[DriftReport]:
        """检测执行者性能漂移"""
        reports = []
        
        for executor, scores in self._recent_scores.items():
            if len(scores) < 10:
                continue
            
            recent = list(scores)[-10:]
            avg_recent = sum(recent) / len(recent)
            
            # 与整体平均比较
            all_scores = [s.score for s in self._scores[-50:]]
            if not all_scores:
                continue
            
            avg_all = sum(all_scores) / len(all_scores)
            
            # 如果某个执行者持续低于平均值
            if avg_recent < avg_all * 0.7:
                reports.append(DriftReport(
                    drift_id=f"drift_{uuid.uuid4().hex[:8]}",
                    drift_type=DriftType.FEEDBACK_PATTERN,
                    severity=Severity.WARNING,
                    description=f"Executor '{executor}' performance below average",
                    baseline_value=avg_all,
                    current_value=avg_recent,
                    threshold=avg_all * 0.7,
                    recommendation=f"Review {executor} task assignments. Consider reducing weight or retraining.",
                ))
        
        return reports

    def generate_report(
        self,
        period_hours: float = 24.0,
    ) -> EvalReport:
        """
        生成评估报告
        
        Args:
            period_hours: 报告周期（小时）
        
        Returns:
            EvalReport: 评估报告
        """
        now = time.time()
        period_start = now - (period_hours * 3600)
        
        # 筛选周期内的评分
        period_scores = [
            s for s in self._scores
            if s.timestamp >= period_start
        ]
        
        # 计算统计
        total_tasks = len(period_scores)
        completed_tasks = sum(1 for s in period_scores if s.score >= 0.7)
        failed_tasks = sum(1 for s in period_scores if s.score < 0.5)
        
        avg_score = (
            sum(s.score for s in period_scores) / total_tasks
            if total_tasks > 0 else 0.0
        )
        
        # 按执行者统计
        executor_scores: Dict[str, float] = {}
        executor_counts: Dict[str, int] = defaultdict(int)
        executor_sums: Dict[str, float] = defaultdict(float)
        
        for s in period_scores:
            executor_counts[s.executor] += 1
            executor_sums[s.executor] += s.score
        
        for executor in executor_counts:
            executor_scores[executor] = (
                executor_sums[executor] / executor_counts[executor]
            )
        
        # 检测漂移
        drift_reports = self.detect_drift()
        
        # 生成建议
        recommendations = self._generate_recommendations(
            avg_score, executor_scores, drift_reports
        )
        
        report = EvalReport(
            report_id=f"eval_{uuid.uuid4().hex[:12]}",
            period_start=period_start,
            period_end=now,
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            average_quality_score=round(avg_score, 4),
            executor_scores=executor_scores,
            drift_reports=drift_reports,
            recommendations=recommendations,
        )
        
        logger.info(
            f"Eval report generated: {total_tasks} tasks, "
            f"avg score: {avg_score:.2f}, drifts: {len(drift_reports)}"
        )
        
        return report

    def _generate_recommendations(
        self,
        avg_score: float,
        executor_scores: Dict[str, float],
        drift_reports: List[DriftReport],
    ) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于平均分
        if avg_score < 0.6:
            recommendations.append(
                "整体质量分数偏低，建议检查任务拆解策略和执行者能力匹配。"
            )
        
        # 基于执行者分数
        for executor, score in executor_scores.items():
            if score < 0.5:
                recommendations.append(
                    f"执行者 '{executor}' 评分过低（{score:.2f}），"
                    f"建议降低路由权重或重新评估任务分配。"
                )
        
        # 基于漂移报告
        for drift in drift_reports:
            if drift.severity == Severity.CRITICAL:
                recommendations.append(
                    f"[严重] {drift.drift_type.value}: {drift.recommendation}"
                )
        
        if not recommendations:
            recommendations.append("系统运行正常，无需调整。")
        
        return recommendations

    def trigger_strategy_update(self):
        """触发策略更新"""
        for callback in self._update_callbacks:
            try:
                callback(self._executor_weights, self._baselines)
            except Exception as e:
                logger.error(f"Strategy update callback error: {e}")
        
        logger.info("Strategy update triggered")

    def get_executor_weights(self) -> Dict[str, float]:
        """获取所有执行者权重"""
        return dict(self._executor_weights)

    def get_statistics(self) -> Dict[str, Any]:
        """获取评估统计"""
        if not self._scores:
            return {'total_scores': 0}
        
        all_scores = [s.score for s in self._scores]
        
        return {
            'total_scores': len(self._scores),
            'average_score': round(sum(all_scores) / len(all_scores), 4),
            'min_score': round(min(all_scores), 4),
            'max_score': round(max(all_scores), 4),
            'executor_weights': self._executor_weights,
            'drift_reports_count': len(self._drift_reports),
            'baselines': self._baselines,
        }


# 全局单例
_eval_loop: Optional[EvalLoop] = None


def get_eval_loop() -> EvalLoop:
    """获取全局评估闭环实例"""
    global _eval_loop
    if _eval_loop is None:
        _eval_loop = EvalLoop()
    return _eval_loop
