"""Educational decoder-only Transformer language model."""
from .config import Config
from .model import GPTLanguageModel

__all__ = ["Config", "GPTLanguageModel"]
