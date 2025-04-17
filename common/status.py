from enum import Enum

class TaskStatus(Enum):
    """Status of a task in the timeline."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WARNING = "warning" 