from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

# ======================
# Enum Definitions
# ======================

class ObjectiveType(Enum):
    RESEARCH_QUESTION = "research_question"
    GOAL = "goal"
    REQUIREMENT = "requirement"
    MILESTONE = "milestone"

class HypothesisState(Enum):
    PROPOSED = "proposed"    # Initial state when created
    TESTING = "testing"      # Currently being tested
    UNTESTED = "untested"    # Ready for testing but not yet tested
    VALIDATED = "validated"  # Confirmed through evidence
    REJECTED = "rejected"    # Rejected through evidence
    INCONCLUSIVE = "inconclusive"  # Evidence is mixed/unclear
    MODIFIED = "modified"    # Hypothesis has been changed
    CONFIRMED = "confirmed"  # Alias for VALIDATED, for backward compatibility

class ConnectionType(Enum):
    CONTRIBUTES_TO = "contributes_to"
    DEPENDS_ON = "depends_on"
    ALTERNATIVE_TO = "alternative_to"
    CONTRASTS_WITH = "contrasts_with"

class EvidenceSourceType(Enum):
    DATA = "data"            # Statistical results
    LITERATURE = "literature"
    EXPERIMENT = "experiment"
    OBSERVATION = "observation"


# ======================
# Data Classes
# ======================


@dataclass
class Evidence:
    """Evidence for or against a hypothesis"""
    id: str
    type: EvidenceSourceType
    description: str
    supports: bool = True  # True if supporting, False if contradicting
    confidence: float = 0.5  # 0-1 confidence scale
    source: str = ""  # Source or citation
    notes: str = ""
    status: str = "validated"  # validated, rejected, inconclusive

@dataclass
class HypothesisConfig:
    """Configuration for hypothesis nodes"""
    id: str
    text: str
    state: HypothesisState = HypothesisState.PROPOSED
    confidence: float = 0.5  # 0-1 confidence scale
    supporting_evidence: List[Evidence] = None
    contradicting_evidence: List[Evidence] = None
    variables: List[str] = None
    literature_evidence: dict = None  # Literature evidence from HypothesesManager
    test_results: dict = None  # Model-based evidence from statistical tests
    alpha_level: float = 0.05  # Alpha level for significance testing
    
    def __post_init__(self):
        if self.supporting_evidence is None:
            self.supporting_evidence = []
        if self.contradicting_evidence is None:
            self.contradicting_evidence = []
        if self.variables is None:
            self.variables = []
        if self.literature_evidence is None:
            self.literature_evidence = {}

@dataclass
class ObjectiveConfig:
    """Configuration for objective nodes"""
    id: str
    text: str
    type: ObjectiveType = ObjectiveType.RESEARCH_QUESTION
    description: str = ""
    progress: float = 0.0  # 0-1 progress scale
    parent_id: Optional[str] = None  # For sub-objectives
    auto_generate: bool = False  # Whether to auto-generate hypotheses
    
# ======================
# Node Configuration
# ======================

@dataclass
class NodeConfig:
    """Configuration for node appearance and behavior"""
    width: int
    height: int
    color: str
    description: str = ""

# Define node appearance configurations
NODE_CONFIGS = {
    "objective": NodeConfig(
        width=200,
        height=120,
        color="#4527A0",  # Deep purple
        description="Research objective"
    ),
    "hypothesis": NodeConfig(
        width=180,
        height=100,
        color="#7986CB",  # Indigo
        description="Research hypothesis"
    )
}