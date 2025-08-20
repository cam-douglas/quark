from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass


__all__ = ["ScriptObjectMeta"]


@dataclass
class ScriptObjectMeta:
    """
    Metadata which is stored on nodes representing ScriptObjects.
    """

    # Key into constants table to retrieve the real ScriptObject.
    constant_name: str

    class_fqn: str
