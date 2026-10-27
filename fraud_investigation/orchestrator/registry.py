"""Wire adapters and steps. Add a new investigation = register + YAML."""

from __future__ import annotations

from fraud_investigation.adapters.base import AdapterRegistry
from fraud_investigation.adapters.memory_oat import (
    MemoryDeviceAdapter,
    MemoryIpAdapter,
    MemoryNotesAdapter,
    MemoryOatAdapter,
)
from fraud_investigation.steps.aggregate import AggregateStep
from fraud_investigation.steps.base import StepRegistry
from fraud_investigation.steps.oat_review import OatReviewStep
from fraud_investigation.steps.previous_interaction import PreviousInteractionStep
from fraud_investigation.steps.stubs import CustomerProfileReviewStep, TransactionPayeeReviewStep


def build_adapter_registry() -> AdapterRegistry:
    registry = AdapterRegistry()
    registry.register("notes_memory", MemoryNotesAdapter())
    registry.register("oat_memory", MemoryOatAdapter())
    registry.register("device_memory", MemoryDeviceAdapter())
    registry.register("ip_memory", MemoryIpAdapter())
    return registry


def build_step_registry() -> StepRegistry:
    registry = StepRegistry()
    registry.register("previous_interaction", PreviousInteractionStep())
    registry.register("oat_review", OatReviewStep())
    registry.register("transaction_payee_review", TransactionPayeeReviewStep())
    registry.register("customer_profile_review", CustomerProfileReviewStep())
    registry.register("aggregate", AggregateStep())
    return registry
