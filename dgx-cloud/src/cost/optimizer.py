"""
Cost Optimization and Budget Tracking for Portalis DGX Cloud
Real-time cost monitoring, budget alerts, and optimization strategies
"""

import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

from loguru import logger
from pydantic import BaseModel, Field
import redis


class InstanceType(str, Enum):
    """DGX instance types"""
    DGXA100_80G = "dgxa100.80g"  # 8x A100 80GB
    DGXA100_40G = "dgxa100.40g"  # 4x A100 40GB
    DGXH100_80G = "dgxh100.80g"  # 4x H100 80GB
    T4 = "t4.xlarge"  # Single T4 (dev/test)


class PricingModel(str, Enum):
    """Pricing models"""
    ON_DEMAND = "on_demand"
    SPOT = "spot"
    RESERVED = "reserved"
    SAVINGS_PLAN = "savings_plan"


@dataclass
class InstancePricing:
    """Pricing information for instance types"""
    instance_type: InstanceType
    pricing_model: PricingModel
    price_per_hour: float
    spot_discount: float = 0.7  # 70% discount
    reserved_discount: float = 0.4  # 40% discount for 1-year

    # Performance metrics
    gpus: int = 8
    gpu_memory_gb: int = 640  # Total across all GPUs
    cpu_cores: int = 64
    system_memory_gb: int = 512


# Pricing database (example prices)
PRICING_TABLE = {
    InstanceType.DGXA100_80G: InstancePricing(
        instance_type=InstanceType.DGXA100_80G,
        pricing_model=PricingModel.ON_DEMAND,
        price_per_hour=32.77,  # AWS p4d.24xlarge equivalent
        gpus=8,
        gpu_memory_gb=640,
        cpu_cores=64,
        system_memory_gb=512
    ),
    InstanceType.DGXA100_40G: InstancePricing(
        instance_type=InstanceType.DGXA100_40G,
        pricing_model=PricingModel.ON_DEMAND,
        price_per_hour=16.00,  # Estimated
        gpus=4,
        gpu_memory_gb=160,
        cpu_cores=32,
        system_memory_gb=256
    ),
    InstanceType.DGXH100_80G: InstancePricing(
        instance_type=InstanceType.DGXH100_80G,
        pricing_model=PricingModel.ON_DEMAND,
        price_per_hour=45.00,  # Estimated
        gpus=4,
        gpu_memory_gb=320,
        cpu_cores=32,
        system_memory_gb=256
    ),
    InstanceType.T4: InstancePricing(
        instance_type=InstanceType.T4,
        pricing_model=PricingModel.ON_DEMAND,
        price_per_hour=1.37,  # g4dn.xlarge
        gpus=1,
        gpu_memory_gb=16,
        cpu_cores=4,
        system_memory_gb=16
    )
}


@dataclass
class CostEvent:
    """Individual cost event"""
    event_id: str
    timestamp: datetime
    instance_type: InstanceType
    pricing_model: PricingModel
    duration_hours: float
    cost: float
    job_id: Optional[str] = None
    tenant_id: str = "default"
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class BudgetConfig:
    """Budget configuration"""
    tenant_id: str
    daily_limit: float
    weekly_limit: float
    monthly_limit: float

    # Alerts
    alert_threshold_pct: float = 0.8  # Alert at 80%
    hard_stop_enabled: bool = False

    # Notifications
    notification_emails: List[str] = field(default_factory=list)
    notification_webhook: Optional[str] = None


@dataclass
class CostMetrics:
    """Cost metrics for a time period"""
    period_start: datetime
    period_end: datetime
    total_cost: float
    cost_by_instance: Dict[InstanceType, float]
    cost_by_tenant: Dict[str, float]
    cost_by_job: Dict[str, float]

    # Efficiency metrics
    avg_gpu_utilization: float
    cost_per_translation: float
    cost_per_gpu_hour: float

    # Optimization potential
    potential_savings: float
    spot_savings: float
    idle_cost: float


class CostOptimizer:
    """
    Cost optimization engine
    Tracks spending, enforces budgets, and recommends optimizations
    """

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        """
        Initialize cost optimizer

        Args:
            redis_host: Redis host for state persistence
            redis_port: Redis port
        """
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )

        self.budgets: Dict[str, BudgetConfig] = {}
        self.events: List[CostEvent] = []

        logger.info("CostOptimizer initialized")

    def register_budget(self, config: BudgetConfig):
        """
        Register budget configuration for a tenant

        Args:
            config: Budget configuration
        """
        self.budgets[config.tenant_id] = config

        # Persist to Redis
        key = f"budget:{config.tenant_id}"
        self.redis_client.set(key, json.dumps({
            "tenant_id": config.tenant_id,
            "daily_limit": config.daily_limit,
            "weekly_limit": config.weekly_limit,
            "monthly_limit": config.monthly_limit,
            "alert_threshold_pct": config.alert_threshold_pct,
            "hard_stop_enabled": config.hard_stop_enabled
        }))

        logger.info(f"Budget registered for tenant {config.tenant_id}")

    def record_usage(
        self,
        instance_type: InstanceType,
        pricing_model: PricingModel,
        duration_hours: float,
        job_id: Optional[str] = None,
        tenant_id: str = "default",
        tags: Optional[Dict[str, str]] = None
    ) -> CostEvent:
        """
        Record resource usage and calculate cost

        Args:
            instance_type: Type of instance used
            pricing_model: Pricing model (on-demand, spot, etc.)
            duration_hours: Duration of usage in hours
            job_id: Associated job ID
            tenant_id: Tenant identifier
            tags: Additional tags for categorization

        Returns:
            Cost event
        """
        # Get base pricing
        pricing = PRICING_TABLE[instance_type]
        base_price = pricing.price_per_hour

        # Apply pricing model discount
        if pricing_model == PricingModel.SPOT:
            price = base_price * (1 - pricing.spot_discount)
        elif pricing_model == PricingModel.RESERVED:
            price = base_price * (1 - pricing.reserved_discount)
        else:
            price = base_price

        # Calculate cost
        cost = price * duration_hours

        # Create event
        event = CostEvent(
            event_id=f"cost-{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            instance_type=instance_type,
            pricing_model=pricing_model,
            duration_hours=duration_hours,
            cost=cost,
            job_id=job_id,
            tenant_id=tenant_id,
            tags=tags or {}
        )

        # Store event
        self.events.append(event)
        self._persist_event(event)

        # Check budget
        self._check_budget(tenant_id, cost)

        logger.debug(f"Recorded usage: {cost:.2f} USD for {duration_hours:.3f}h on {instance_type.value}")

        return event

    def _persist_event(self, event: CostEvent):
        """Persist cost event to Redis"""
        key = f"cost_event:{event.event_id}"
        self.redis_client.set(key, json.dumps({
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "instance_type": event.instance_type.value,
            "pricing_model": event.pricing_model.value,
            "duration_hours": event.duration_hours,
            "cost": event.cost,
            "job_id": event.job_id,
            "tenant_id": event.tenant_id,
            "tags": event.tags
        }), ex=2592000)  # 30 day expiry

        # Add to time-series index
        ts_key = f"cost_ts:{event.tenant_id}:{event.timestamp.date()}"
        self.redis_client.incrbyfloat(ts_key, event.cost)
        self.redis_client.expire(ts_key, 2592000)

    def _check_budget(self, tenant_id: str, new_cost: float):
        """
        Check if budget limits are exceeded

        Args:
            tenant_id: Tenant identifier
            new_cost: New cost to add
        """
        if tenant_id not in self.budgets:
            return

        budget = self.budgets[tenant_id]

        # Get current spending
        daily_spend = self._get_spend(tenant_id, timedelta(days=1))
        weekly_spend = self._get_spend(tenant_id, timedelta(days=7))
        monthly_spend = self._get_spend(tenant_id, timedelta(days=30))

        # Check limits
        checks = [
            ("daily", daily_spend + new_cost, budget.daily_limit),
            ("weekly", weekly_spend + new_cost, budget.weekly_limit),
            ("monthly", monthly_spend + new_cost, budget.monthly_limit)
        ]

        for period, spend, limit in checks:
            if limit > 0:  # Only check if limit is set
                pct = spend / limit

                # Alert threshold
                if pct >= budget.alert_threshold_pct:
                    self._send_budget_alert(
                        tenant_id,
                        period,
                        spend,
                        limit,
                        pct
                    )

                # Hard stop
                if budget.hard_stop_enabled and pct >= 1.0:
                    raise BudgetExceededError(
                        f"Budget exceeded for {tenant_id}: "
                        f"{period} spend ${spend:.2f} exceeds limit ${limit:.2f}"
                    )

    def _get_spend(self, tenant_id: str, period: timedelta) -> float:
        """
        Get total spending for a time period

        Args:
            tenant_id: Tenant identifier
            period: Time period to check

        Returns:
            Total spending in USD
        """
        cutoff = datetime.utcnow() - period

        total = 0.0
        for event in self.events:
            if event.tenant_id == tenant_id and event.timestamp >= cutoff:
                total += event.cost

        return total

    def _send_budget_alert(
        self,
        tenant_id: str,
        period: str,
        spend: float,
        limit: float,
        percentage: float
    ):
        """Send budget alert notification"""
        logger.warning(
            f"Budget alert for {tenant_id}: "
            f"{period} spend ${spend:.2f} is {percentage*100:.1f}% of limit ${limit:.2f}"
        )

        # Send notifications (webhook, email, etc.)
        budget = self.budgets[tenant_id]

        if budget.notification_webhook:
            # Send webhook notification
            import requests
            requests.post(budget.notification_webhook, json={
                "tenant_id": tenant_id,
                "period": period,
                "spend": spend,
                "limit": limit,
                "percentage": percentage,
                "timestamp": datetime.utcnow().isoformat()
            })

    def get_metrics(
        self,
        tenant_id: str = None,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> CostMetrics:
        """
        Get cost metrics for a time period

        Args:
            tenant_id: Optional tenant filter
            start_time: Start of period
            end_time: End of period

        Returns:
            Cost metrics
        """
        if not start_time:
            start_time = datetime.utcnow() - timedelta(days=1)
        if not end_time:
            end_time = datetime.utcnow()

        # Filter events
        filtered = [
            e for e in self.events
            if start_time <= e.timestamp <= end_time
            and (tenant_id is None or e.tenant_id == tenant_id)
        ]

        # Calculate totals
        total_cost = sum(e.cost for e in filtered)

        # Group by instance type
        cost_by_instance = {}
        for event in filtered:
            cost_by_instance[event.instance_type] = (
                cost_by_instance.get(event.instance_type, 0.0) + event.cost
            )

        # Group by tenant
        cost_by_tenant = {}
        for event in filtered:
            cost_by_tenant[event.tenant_id] = (
                cost_by_tenant.get(event.tenant_id, 0.0) + event.cost
            )

        # Group by job
        cost_by_job = {}
        for event in filtered:
            if event.job_id:
                cost_by_job[event.job_id] = (
                    cost_by_job.get(event.job_id, 0.0) + event.cost
                )

        # Calculate efficiency metrics (simplified)
        total_gpu_hours = sum(
            e.duration_hours * PRICING_TABLE[e.instance_type].gpus
            for e in filtered
        )

        avg_gpu_utilization = 0.7  # Would be fetched from monitoring
        cost_per_gpu_hour = total_cost / total_gpu_hours if total_gpu_hours > 0 else 0.0

        # Calculate potential savings
        spot_savings = self._calculate_spot_savings(filtered)
        idle_cost = self._calculate_idle_cost(filtered)

        return CostMetrics(
            period_start=start_time,
            period_end=end_time,
            total_cost=total_cost,
            cost_by_instance=cost_by_instance,
            cost_by_tenant=cost_by_tenant,
            cost_by_job=cost_by_job,
            avg_gpu_utilization=avg_gpu_utilization,
            cost_per_translation=0.0,  # Would be calculated from job data
            cost_per_gpu_hour=cost_per_gpu_hour,
            potential_savings=spot_savings + idle_cost,
            spot_savings=spot_savings,
            idle_cost=idle_cost
        )

    def _calculate_spot_savings(self, events: List[CostEvent]) -> float:
        """Calculate potential savings from using spot instances"""
        on_demand_events = [
            e for e in events
            if e.pricing_model == PricingModel.ON_DEMAND
        ]

        savings = 0.0
        for event in on_demand_events:
            pricing = PRICING_TABLE[event.instance_type]
            spot_cost = event.cost * (1 - pricing.spot_discount)
            savings += event.cost - spot_cost

        return savings

    def _calculate_idle_cost(self, events: List[CostEvent]) -> float:
        """Estimate cost from idle resources"""
        # Simplified: assume 30% idle time
        idle_percentage = 0.3
        total_cost = sum(e.cost for e in events)
        return total_cost * idle_percentage

    def recommend_optimizations(
        self,
        tenant_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        Generate cost optimization recommendations

        Args:
            tenant_id: Optional tenant filter

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # Get recent metrics
        metrics = self.get_metrics(
            tenant_id=tenant_id,
            start_time=datetime.utcnow() - timedelta(days=7)
        )

        # Recommendation 1: Use spot instances
        if metrics.spot_savings > 10.0:
            recommendations.append({
                "type": "spot_instances",
                "priority": "high",
                "potential_savings": metrics.spot_savings,
                "description": (
                    f"Switch to spot instances to save ${metrics.spot_savings:.2f}/week. "
                    "Spot instances are 70% cheaper and suitable for batch workloads."
                ),
                "action": "Enable spot instances in resource allocation policy"
            })

        # Recommendation 2: Right-size instances
        if metrics.avg_gpu_utilization < 0.5:
            recommendations.append({
                "type": "downsize_instances",
                "priority": "medium",
                "potential_savings": metrics.total_cost * 0.3,
                "description": (
                    f"GPU utilization is {metrics.avg_gpu_utilization*100:.1f}%. "
                    "Consider using smaller instances or increasing batch sizes."
                ),
                "action": "Review instance type selection and workload batching"
            })

        # Recommendation 3: Reduce idle time
        if metrics.idle_cost > 50.0:
            recommendations.append({
                "type": "reduce_idle",
                "priority": "high",
                "potential_savings": metrics.idle_cost,
                "description": (
                    f"Estimated ${metrics.idle_cost:.2f}/week wasted on idle resources. "
                    "Enable aggressive auto-scaling down policies."
                ),
                "action": "Reduce idle timeout from 5 minutes to 2 minutes"
            })

        # Recommendation 4: Reserved instances
        if metrics.total_cost > 1000.0:
            savings = metrics.total_cost * 0.4
            recommendations.append({
                "type": "reserved_instances",
                "priority": "medium",
                "potential_savings": savings,
                "description": (
                    f"High sustained usage detected. Save ${savings:.2f}/week "
                    "with 1-year reserved instances (40% discount)."
                ),
                "action": "Evaluate reserved instance commitment"
            })

        # Sort by potential savings
        recommendations.sort(key=lambda x: x["potential_savings"], reverse=True)

        return recommendations

    def export_report(self, tenant_id: str = None, format: str = "json") -> str:
        """
        Export cost report

        Args:
            tenant_id: Optional tenant filter
            format: Export format (json, csv)

        Returns:
            Formatted report
        """
        metrics = self.get_metrics(tenant_id=tenant_id)
        recommendations = self.recommend_optimizations(tenant_id=tenant_id)

        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "tenant_id": tenant_id or "all",
            "period": {
                "start": metrics.period_start.isoformat(),
                "end": metrics.period_end.isoformat()
            },
            "summary": {
                "total_cost": metrics.total_cost,
                "cost_per_gpu_hour": metrics.cost_per_gpu_hour,
                "avg_gpu_utilization": metrics.avg_gpu_utilization,
                "potential_savings": metrics.potential_savings
            },
            "breakdown": {
                "by_instance": {k.value: v for k, v in metrics.cost_by_instance.items()},
                "by_tenant": metrics.cost_by_tenant,
                "by_job": metrics.cost_by_job
            },
            "recommendations": recommendations
        }

        if format == "json":
            return json.dumps(report, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")


class BudgetExceededError(Exception):
    """Raised when budget limit is exceeded"""
    pass
