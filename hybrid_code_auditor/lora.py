from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LoRAConfig:
    enabled: bool = False
    base_model: str = "meta-llama/Meta-Llama-3.2-3B-Instruct"
    output_dir: Path = Path("artifacts/lora")


def describe_optional_lora_workflow() -> str:
    return (
        "Optional LoRA fine-tuning can be added offline with a curated dataset of Python refactor pairs, "
        "using PEFT + Transformers on a local base model. The current implementation keeps this as a stub "
        "so the refactoring pipeline remains dependency-light and fully local by default."
    )
