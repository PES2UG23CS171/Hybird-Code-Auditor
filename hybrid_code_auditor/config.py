from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AuditorConfig:
    model_name: str = "llama3.1:latest"
    ollama_host: str = "http://localhost:11434"
    complexity_threshold: int = 10
    max_rounds: int = 5
    top_k_guidelines: int = 3
    max_loc_change_ratio: float = 0.2
    # Absolute LOC slack so small files can still gain a helper function
    # without tripping the percentage limit.
    loc_change_allowance: int = 8
