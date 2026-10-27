"""Optional LLM interpreter hook for unstructured data.

Sample mode returns a deterministic heuristic result so the pipeline runs
without an API key. Swap `interpret_location` / `interpret_notes` with a
real provider (OpenAI/Azure/Anthropic) using structured JSON outputs.
"""

from __future__ import annotations

from typing import Any


class LLMInterpreter:
    enabled: bool = False

    def interpret_location(
        self,
        ip_evidence: dict[str, Any],
        payer_profile: dict[str, Any] | None,
        travel_notes: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {
                "used_llm": False,
                "message": "LLM disabled; deterministic geo rules used upstream.",
            }
        # Placeholder for future GenAI call
        return {
            "used_llm": True,
            "flag": "AMBER",
            "rationale": "LLM location interpretation not configured in sample.",
        }

    def interpret_notes(self, notes: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.enabled:
            return {"used_llm": False, "verified": False}
        return {
            "used_llm": True,
            "verified": False,
            "rationale": "LLM notes interpretation not configured in sample.",
        }
