from typing import Optional

from pydantic.dataclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass


@dataclass
class GITModel:
    branch: Optional[str] = None
    tag: Optional[str] = None
    commit: Optional[str] = None
    dirty: Optional[bool] = None
    origin: Optional[str] = None
