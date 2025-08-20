from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass
from subprocess import CompletedProcess


@dataclass
class ToolResult:
    process: CompletedProcess
    duration_ms: int
