from enum import Enum
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
import random
import uuid
from datetime import datetime

# =============================================================================
# Data Type and Enumerations
# =============================================================================

class CFDataType(Enum):
    """Data type definitions for clinical features."""
    CONTINUOUS = 'continuous'
    CATEGORICAL = 'categorical'
    BINARY = 'binary'
    ORDINAL = 'ordinal'
    DATETIME = 'datetime'
    COUNT = 'count'
    TIME_TO_EVENT = 'time_to_event'

class StudyType(Enum):
    """Enumerates the various types of study designs."""
    BETWEEN_SUBJECTS = "between_subjects"
    WITHIN_SUBJECTS = "within_subjects"
    MIXED = "mixed"
    SINGLE_SUBJECT = "single_subject"
    CROSS_OVER = "cross_over"
    FACTORIAL = "factorial"
    REPEATED_CROSS_SECTIONAL = "repeated_cross_sectional"
    NESTED = "nested"
    LATIN_SQUARE = "latin_square"
    ONE_SAMPLE = "one_sample"
    SURVIVAL_ANALYSIS = "survival_analysis"
    
    @property
    def display_name(self) -> str:
        """Return a formatted display name for this study type."""
        return self.value.replace("_", " ").title()
    
    @property
    def description(self):
        """Return a description of the study type."""
        descriptions = {
            self.BETWEEN_SUBJECTS: "Different participants in each group",
            self.WITHIN_SUBJECTS: "Same participants measured at different time points",
            self.MIXED: "Combination of between and within factors",
            self.SINGLE_SUBJECT: "Intensive study of one or few subjects with repeated measurements",
            self.CROSS_OVER: "Participants receive all treatments in sequence with washout periods",
            self.FACTORIAL: "Examines effects of two or more independent variables simultaneously",
            self.REPEATED_CROSS_SECTIONAL: "Different participants measured at each time point",
            self.NESTED: "Hierarchical design with participants nested within groups",
            self.LATIN_SQUARE: "Balanced design controlling for order effects",
            self.ONE_SAMPLE: "Design with a single sample and a population mean to test against",
            self.SURVIVAL_ANALYSIS: "Analysis of survival data"
        }
        return descriptions.get(self, "No description available")

class TimePoint(Enum):
    BASELINE = "baseline"      # Initial measurement
    INTERIM = "interim"        # During the intervention/study period
    COMPLETION = "completion"  # Final measurement
    FOLLOW_UP = "follow_up"    # After the main study period
    CUSTOM = "custom"          # User-defined timepoint

# =============================================================================
# Study Timepoints, Interventions, and Arms
# =============================================================================

@dataclass
class StudyTimepoint:
    """Represents a timepoint in the study."""
    name: str
    point_type: TimePoint
    order: int  # Order to maintain the intended sequence of timepoints
    offset_days: Optional[int] = None  # Days from baseline
    description: Optional[str] = None
    window_days: Optional[int] = None
    visit_number: Optional[int] = None

class InterventionType(Enum):
    PHARMACOLOGICAL = "pharmacological"
    BEHAVIORAL = "behavioral"
    SURGICAL = "surgical"
    DEVICE = "device"
    PROGRAM = "program"
    OTHER = "other"

@dataclass
class Intervention:
    """Represents an intervention in the study."""
    name: str
    description: Optional[str] = None  # e.g., Drug X, Procedure Y, Policy Z
    type: InterventionType = InterventionType.OTHER

@dataclass
class Arm:
    """Represents a study arm (group of participants)."""
    name: str
    interventions: List[Intervention]
    description: Optional[str] = None
    start_date: Optional[str] = None  # ISO-formatted date string
    end_date: Optional[str] = None    # ISO-formatted date string
    cohort_size: Optional[int] = None

# =============================================================================
# Outcome Measures and Composite Outcomes
# =============================================================================

class OutcomeCategory(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    EXPLORATORY = "exploratory"
    SAFETY = "safety"

class DataCollectionMethod(Enum):
    SELF_REPORT = "self_report"
    OBSERVATION = "observation"
    QUESTIONNAIRE = "questionnaire"
    INTERVIEW = "interview"
    DATABASE = "database"
    LAB_TEST = "lab_test"
    IMAGING = "imaging"
    CLINICAL_ASSESSMENT = "clinical_assessment"
    OTHER = "other"

@dataclass
class OutcomeMeasure:
    """Represents an outcome measure in the study."""
    name: str
    timepoints: List[str]  # References to StudyTimepoint.name(s)
    data_type: CFDataType = CFDataType.CONTINUOUS
    category: OutcomeCategory = OutcomeCategory.PRIMARY
    description: Optional[str] = None
    data_collection_method: Optional[DataCollectionMethod] = None
    applicable_arms: Optional[List[str]] = None  # Which arms this outcome applies to
    units: Optional[str] = None  # e.g., "mm Hg", "points", etc.

class CombineMethod(Enum):
    SUM = "sum"
    AVERAGE = "average"
    WEIGHTED_AVERAGE = "weighted_average"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"

@dataclass
class CompositeOutcomeMeasure:
    """
    A composite outcome measure defined by combining multiple outcome measures.
    """
    name: str
    component_outcomes: List[OutcomeMeasure]
    timepoints: List[str]  # References to StudyTimepoint.name(s)
    combine_method: CombineMethod = CombineMethod.SUM
    category: OutcomeCategory = OutcomeCategory.PRIMARY
    data_type: CFDataType = CFDataType.CONTINUOUS
    description: Optional[str] = None
    data_collection_method: Optional[DataCollectionMethod] = None
    applicable_arms: Optional[List[str]] = None
    weights: Optional[List[float]] = None

    def __post_init__(self):
        if self.combine_method == CombineMethod.WEIGHTED_AVERAGE:
            if not self.weights:
                raise ValueError("Weights must be provided for weighted average.")
            if len(self.weights) != len(self.component_outcomes):
                raise ValueError("Number of weights must match number of component outcomes.")
        self.validate_components()

    def validate_components(self):
        """Validates that component outcomes have consistent units (if defined)."""
        units = [comp.units for comp in self.component_outcomes]
        defined_units = [u for u in units if u is not None]
        if defined_units and len(set(defined_units)) > 1:
            raise ValueError("Component outcome units must be consistent for sum/average.")

# =============================================================================
# Covariates and Statistical Analysis
# =============================================================================

@dataclass
class CovariateDefinition:
    """Represents a covariate in the study."""
    name: str
    description: Optional[str] = None
    data_type: CFDataType = CFDataType.CONTINUOUS

class BlindingType(Enum):
    NONE = "none"
    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"

class StatisticalTest(Enum):
    """Enumeration of supported statistical tests."""
    ONE_SAMPLE_T_TEST = "one_sample_t_test"
    INDEPENDENT_T_TEST = "independent_t_test"
    PAIRED_T_TEST = "paired_t_test"
    MANN_WHITNEY_U_TEST = "mann_whitney_u_test"
    WILCOXON_SIGNED_RANK_TEST = "wilcoxon_signed_rank_test"
    ONE_WAY_ANOVA = "one_way_anova"
    KRUSKAL_WALLIS_TEST = "kruskal_wallis_test"
    CHI_SQUARE_TEST = "chi_square_test"
    FISHERS_EXACT_TEST = "fishers_exact_test"
    PEARSON_CORRELATION = "pearson_correlation"
    SPEARMAN_CORRELATION = "spearman_correlation"
    KENDALL_TAU_CORRELATION = "kendall_tau_correlation"
    POINT_BISERIAL_CORRELATION = "point_biserial_correlation"
    REPEATED_MEASURES_ANOVA = "repeated_measures_anova"
    MIXED_ANOVA = "mixed_anova"
    LINEAR_MIXED_EFFECTS_MODEL = "linear_mixed_effects_model"
    ORDINAL_REGRESSION = "ordinal_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    POISSON_REGRESSION = "poisson_regression"
    NEGATIVE_BINOMIAL_REGRESSION = "negative_binomial_regression"
    MULTINOMIAL_LOGISTIC_REGRESSION = "multinomial_logistic_regression"
    LINEAR_REGRESSION = "linear_regression"
    ANCOVA = "ancova"
    SURVIVAL_ANALYSIS = "survival_analysis"

class MultipleComparisonCorrection(Enum):
    BONFERRONI = "bonferroni"
    BENJAMINI_HOCHBERG = "benjamini-hochberg"
    NONE = "none"

@dataclass
class MissingDataHandling:
    """Represents the approach for handling missing data in analysis."""
    method: str  # e.g., "multiple imputation", "last observation carried forward"
    description: Optional[str] = None

@dataclass
class AnalysisPlan:
    """Represents a statistical analysis plan for an outcome measure."""
    outcome_measure: str  # Reference to OutcomeMeasure.name
    statistical_test: StatisticalTest
    covariates: List[str]  # Names of covariate variables
    description: Optional[str] = None
    alpha: float = 0.05  # Significance level
    power: Optional[float] = 0.8  # Desired statistical power
    statistical_test_parameters: Optional[Dict[str, Any]] = None
    multiple_comparison_correction: MultipleComparisonCorrection = MultipleComparisonCorrection.NONE
    interaction_terms: Optional[List[List[str]]] = None
    analysis_population: Optional[str] = None  # e.g., ITT, per-protocol
    missing_data_handling: Optional[MissingDataHandling] = None

# =============================================================================
# Randomization and Recruitment
# =============================================================================

class RandomizationMethod(Enum):
    SIMPLE = "simple"
    BLOCK = "block"
    STRATIFIED = "stratified"
    MINIMIZATION = "minimization"

@dataclass
class RandomizationScheme:
    """Defines the randomization scheme for the study."""
    method: RandomizationMethod
    block_size: Optional[int] = None
    stratification_factors: Optional[List[str]] = None
    description: Optional[str] = None
    random_seed: Optional[int] = None
    randomization_ratio: Optional[List[int]] = None  # Ratio for assignment

@dataclass
class RecruitmentPlan:
    """Describes the plan for recruiting participants."""
    recruitment_start_date: Optional[str] = None  # ISO-formatted date
    recruitment_end_date: Optional[str] = None
    recruitment_methods: Optional[List[str]] = None  # e.g., "advertising", "referral"
    target_sample_size: Optional[int] = None
    description: Optional[str] = None

# =============================================================================
# Eligibility Criteria and Participant Data
# =============================================================================

class EligibilityOperator(Enum):
    EQUALS = "="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    IN = "in"
    NOT_IN = "not in"

@dataclass
class EligibilityCriterion:
    """Represents a single eligibility criterion."""
    criterion: str  # e.g., "age", "diagnosis"
    operator: EligibilityOperator
    value: Union[str, int, float, List[Any]]  # Allow multiple values

@dataclass
class EligibilityCriteria:
    """Defines the inclusion and exclusion criteria for the study."""
    inclusion_criteria: List[EligibilityCriterion]
    exclusion_criteria: List[EligibilityCriterion]

@dataclass
class InformedConsent:
    """Represents details of an informed consent document for a participant."""
    consent_id: str
    consent_date: str  # ISO-formatted date
    version: str
    document_url: Optional[str] = None
    notes: Optional[str] = None

@dataclass
class Participant:
    """Represents a participant in the study."""
    participant_id: str
    demographics: Dict[str, Union[str, int, float]]
    assigned_arm: Optional[str] = None
    baseline_data: Dict[str, Union[str, int, float]] = field(default_factory=dict)
    status: str = "active"
    withdrawal_reason: Optional[str] = None
    enrollment_date: Optional[str] = None  # ISO date
    date_of_birth: Optional[str] = None
    consents: List[InformedConsent] = field(default_factory=list)

# =============================================================================
# Adverse Events
# =============================================================================

class AdverseEventSeverity(Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    LIFE_THREATENING = "life-threatening"
    DEATH = "death"

class AdverseEventCausality(Enum):
    UNRELATED = "unrelated"
    POSSIBLE = "possible"
    PROBABLE = "probable"
    DEFINITE = "definite"

@dataclass
class AdverseEvent:
    """Represents an adverse event experienced by a participant."""
    participant_id: str
    description: str
    severity: AdverseEventSeverity
    causality: Optional[AdverseEventCausality] = None
    intervention: Optional[str] = None  # Reference to Intervention.name if applicable
    onset_date: Optional[str] = None  # ISO date
    resolution_date: Optional[str] = None  # ISO date
    action_taken: Optional[str] = None

# =============================================================================
# Additional Research Components
# =============================================================================

@dataclass
class SampleSizeCalculation:
    """Represents sample size calculation details for the study."""
    outcome_measure: str  # Reference to OutcomeMeasure.name
    effect_size: float
    alpha: float = 0.05
    power: float = 0.8
    calculated_sample_size: Optional[int] = None
    calculation_method: Optional[str] = None
    description: Optional[str] = None

@dataclass
class DataManagementPlan:
    """Represents a plan for data management in the study."""
    data_collection_tools: List[str] = field(default_factory=list)
    data_storage_location: Optional[str] = None
    data_security_measures: Optional[str] = None
    data_sharing_policy: Optional[str] = None
    quality_control_measures: Optional[str] = None
    backup_procedures: Optional[str] = None

@dataclass
class EthicsApproval:
    """Represents details of ethics committee approval for the study."""
    committee_name: str
    approval_id: str
    approval_date: str  # ISO date
    expiration_date: Optional[str] = None
    notes: Optional[str] = None

@dataclass
class StudyRegistration:
    """Represents study registration details in a public registry."""
    registry_name: str
    registration_id: str
    registration_date: str  # ISO date
    url: Optional[str] = None

@dataclass
class DataCollectionPlan:
    """Represents the plan for data collection in the study."""
    tools: List[str] = field(default_factory=list)
    procedures: Optional[str] = None
    quality_control: Optional[str] = None

@dataclass
class SafetyMonitoringPlan:
    """Represents the safety monitoring plan including DSMB details."""
    dsmb_members: List[str] = field(default_factory=list)
    meeting_schedule: Optional[str] = None
    reporting_procedures: Optional[str] = None

@dataclass
class RetentionPlan:
    """Represents the plan to retain participants and minimize dropout."""
    follow_up_frequency: Optional[str] = None
    retention_strategies: Optional[str] = None

# =============================================================================
# Overall Study Design
# =============================================================================

@dataclass
class StudyDesign:
    """Represents the overall design of the study."""
    study_id: Optional[str] = None
    title: Optional[str] = None
    study_type: Optional[StudyType] = StudyType.BETWEEN_SUBJECTS
    description: Optional[str] = None
    timepoints: Optional[List[StudyTimepoint]] = field(default_factory=list)
    arms: Optional[List[Arm]] = field(default_factory=list)
    outcome_measures: Optional[List[Union[OutcomeMeasure, CompositeOutcomeMeasure]]] = field(default_factory=list)
    covariates: Optional[List[CovariateDefinition]] = field(default_factory=list)
    analysis_plans: Optional[List[AnalysisPlan]] = field(default_factory=list)
    randomization_scheme: Optional[RandomizationScheme] = None
    recruitment_plan: Optional[RecruitmentPlan] = None
    eligibility_criteria: Optional[EligibilityCriteria] = None
    blinding: Optional[BlindingType] = BlindingType.NONE
    adverse_events: Optional[List[AdverseEvent]] = field(default_factory=list)
    sample_size_calculations: Optional[List[SampleSizeCalculation]] = field(default_factory=list)
    data_management_plan: Optional[DataManagementPlan] = None
    ethics_approval: Optional[EthicsApproval] = None
    registration: Optional[StudyRegistration] = None
    data_collection_plan: Optional[DataCollectionPlan] = None
    safety_monitoring_plan: Optional[SafetyMonitoringPlan] = None
    retention_plan: Optional[RetentionPlan] = None
    participants: Optional[List[Participant]] = field(default_factory=list)
    hypotheses: Optional[List[Dict]] = field(default_factory=list)  # Store hypothesis data

    def validate_design(self):
        """
        Validates that the study design is consistent.
        """
        timepoint_names = {tp.name for tp in self.timepoints}
        arm_names = {arm.name for arm in self.arms}
        covariate_names = {cov.name for cov in self.covariates}
        outcome_measure_names = {om.name for om in self.outcome_measures}

        # Check for duplicate timepoint names
        if len(timepoint_names) != len(self.timepoints):
            raise ValueError("Duplicate timepoint names found.")

        # Check for duplicate arm names
        if len(arm_names) != len(self.arms):
            raise ValueError("Duplicate arm names found.")

        # Check for duplicate outcome measure names
        if len(outcome_measure_names) != len(self.outcome_measures):
            raise ValueError("Duplicate outcome measure names found.")

        for outcome in self.outcome_measures:
            for tp in outcome.timepoints:
                if tp not in timepoint_names:
                    raise ValueError(f"Outcome measure '{outcome.name}' references undefined timepoint '{tp}'.")
            if outcome.applicable_arms:
                for arm_name in outcome.applicable_arms:
                    if arm_name not in arm_names:
                        raise ValueError(f"Outcome measure '{outcome.name}' references undefined arm '{arm_name}'.")
            if isinstance(outcome, CompositeOutcomeMeasure):
                for component in outcome.component_outcomes:
                    if component.name not in outcome_measure_names:
                        raise ValueError(f"Composite outcome measure '{outcome.name}' references undefined component outcome '{component.name}'.")

        # Validate randomization scheme (if provided)
        if self.randomization_scheme:
            if self.randomization_scheme.block_size is not None and self.randomization_scheme.block_size <= 0:
                raise ValueError(f"Block size must be greater than zero. Current value: {self.randomization_scheme.block_size}")
            if self.randomization_scheme.randomization_ratio:
                if len(self.randomization_scheme.randomization_ratio) != len(self.arms):
                    raise ValueError("Randomization ratio must have the same length as the number of arms.")
                if self.randomization_scheme.method in [RandomizationMethod.BLOCK, RandomizationMethod.STRATIFIED]:
                    if self.randomization_scheme.block_size is not None and (self.randomization_scheme.block_size % sum(self.randomization_scheme.randomization_ratio) != 0):
                        raise ValueError("Block size must be a multiple of the sum of the randomization ratio for block randomization.")

        # Validate that covariates used in analysis plans are defined
        for plan in self.analysis_plans:
            if plan.outcome_measure not in outcome_measure_names:
                raise ValueError(f"Analysis plan references undefined outcome measure '{plan.outcome_measure}'.")
            for covariate in plan.covariates:
                if covariate not in covariate_names:
                    raise ValueError(f"Analysis plan for '{plan.outcome_measure}' references undefined covariate '{covariate}'.")

        print("Study design validation passed.")

    def get_outcome_measure_by_name(self, name: str) -> Optional[Union[OutcomeMeasure, CompositeOutcomeMeasure]]:
        """Retrieves an OutcomeMeasure object by its name."""
        for om in self.outcome_measures:
            if om.name == name:
                return om
        return None

    def add_outcome_measure(self, outcome_measure: Union[OutcomeMeasure, CompositeOutcomeMeasure]):
        """Adds an outcome measure to the study design, performing validation."""
        if outcome_measure.name in [om.name for om in self.outcome_measures]:
            raise ValueError(f"Outcome measure with name '{outcome_measure.name}' already exists.")
        self.outcome_measures.append(outcome_measure)

    def randomize_participants(self, participants: List[Participant]) -> None:
        """
        Assigns participants to arms using the defined randomization scheme.
        """
        if not self.randomization_scheme:
            raise ValueError("Randomization scheme not defined in study design.")
        if not self.arms:
            raise ValueError("No arms defined in study design.")

        method = self.randomization_scheme.method
        num_arms = len(self.arms)
        arm_names = [arm.name for arm in self.arms]

        # Determine randomization ratio
        if self.randomization_scheme.randomization_ratio:
            if len(self.randomization_scheme.randomization_ratio) != num_arms:
                raise ValueError("Randomization ratio must have the same length as the number of arms.")
            ratio = self.randomization_scheme.randomization_ratio
        else:
            ratio = [1] * num_arms

        # Use the random seed for reproducibility
        if self.randomization_scheme.random_seed is not None:
            random.seed(self.randomization_scheme.random_seed)

        if method == RandomizationMethod.SIMPLE:
            for participant in participants:
                assigned_arm_index = random.choices(range(num_arms), weights=ratio)[0]
                participant.assigned_arm = arm_names[assigned_arm_index]

        elif method == RandomizationMethod.BLOCK:
            block_size = self.randomization_scheme.block_size
            if block_size is None:
                raise ValueError("Block size must be defined for block randomization.")
            if block_size % sum(ratio) != 0:
                raise ValueError("Block size must be a multiple of the sum of the randomization ratio for block randomization.")

            # Create a block respecting the ratio
            block = []
            for i in range(num_arms):
                block.extend([self.arms[i]] * ratio[i])
            block = block * (block_size // sum(ratio))

            assignments = []
            num_blocks = (len(participants) + block_size - 1) // block_size
            for _ in range(num_blocks):
                shuffled_block = block.copy()
                random.shuffle(shuffled_block)
                assignments.extend(shuffled_block)

            for participant, arm in zip(participants, assignments):
                participant.assigned_arm = arm.name

        elif method == RandomizationMethod.STRATIFIED:
            strat_factors = self.randomization_scheme.stratification_factors
            if not strat_factors:
                raise ValueError("Stratification factors must be defined for stratified randomization.")

            for participant in participants:
                for factor in strat_factors:
                    if factor not in participant.demographics:
                        raise ValueError(f"Stratification factor '{factor}' not found in demographics of participant '{participant.participant_id}'.")

            strata: Dict[tuple, List[Participant]] = {}
            for participant in participants:
                key = tuple(participant.demographics.get(factor) for factor in strat_factors)
                strata.setdefault(key, []).append(participant)

            for group in strata.values():
                if len(group) < num_arms:
                    print(f"Warning: Stratum {group[0].demographics} has fewer participants than arms. Using simple randomization with ratio.")
                    for participant in group:
                        assigned_arm_index = random.choices(range(num_arms), weights=ratio)[0]
                        participant.assigned_arm = arm_names[assigned_arm_index]
                else:
                    block_size = self.randomization_scheme.block_size or sum(ratio)
                    if block_size % sum(ratio) != 0:
                        raise ValueError("Block size must be a multiple of the sum of the randomization ratio for stratified block randomization.")

                    block = []
                    for i in range(num_arms):
                        block.extend([self.arms[i]] * ratio[i])
                    block = block * (block_size // sum(ratio))

                    assignments = []
                    num_blocks = (len(group) + block_size - 1) // block_size
                    for _ in range(num_blocks):
                        shuffled_block = block.copy()
                        random.shuffle(shuffled_block)
                        assignments.extend(shuffled_block)
                    for participant, arm in zip(group, assignments):
                        participant.assigned_arm = arm.name

        elif method == RandomizationMethod.MINIMIZATION:
            # Placeholder: using simple randomization with ratio as default.
            for participant in participants:
                assigned_arm_index = random.choices(range(num_arms), weights=ratio)[0]
                participant.assigned_arm = arm_names[assigned_arm_index]

        else:
            raise ValueError(f"Unsupported randomization method: {method}")
    
    def get_analysis_plan(self, outcome_measure_name: str) -> Optional[AnalysisPlan]:
        """Retrieves the AnalysisPlan for a given outcome measure."""
        for plan in self.analysis_plans:
            if plan.outcome_measure == outcome_measure_name:
                return plan
        return None

    def add_hypothesis(self, hypothesis_data: Dict) -> str:
        """
        Add a new hypothesis to the study design.
        
        Args:
            hypothesis_data: Dictionary containing hypothesis information
            
        Returns:
            The ID of the newly added hypothesis
            
        Raises:
            ValueError: If the hypothesis data is invalid
        """
        if not hypothesis_data.get('title'):
            raise ValueError("Hypothesis must have a title")
            
        # Ensure we have a hypothesis list
        if not hasattr(self, 'hypotheses'):
            self.hypotheses = []
            
        # Ensure the hypothesis has required fields
        if 'id' not in hypothesis_data:
            hypothesis_data['id'] = str(uuid.uuid4())
        if 'created_at' not in hypothesis_data:
            hypothesis_data['created_at'] = datetime.now().isoformat()
        if 'updated_at' not in hypothesis_data:
            hypothesis_data['updated_at'] = datetime.now().isoformat()
        if 'status' not in hypothesis_data:
            hypothesis_data['status'] = 'untested'
            
        # Add the hypothesis
        self.hypotheses.append(hypothesis_data)
        
        return hypothesis_data['id']
        
    def update_hypothesis(self, hypothesis_id: str, updated_data: Dict) -> bool:
        """
        Update an existing hypothesis in the study design.
        
        Args:
            hypothesis_id: ID of the hypothesis to update
            updated_data: Dictionary containing updated hypothesis information
            
        Returns:
            True if the hypothesis was found and updated, False otherwise
        """
        if not hasattr(self, 'hypotheses'):
            return False
            
        for i, hyp in enumerate(self.hypotheses):
            if hyp.get('id') == hypothesis_id:
                # Update the timestamp
                updated_data['updated_at'] = datetime.now().isoformat()
                
                # Update the hypothesis
                self.hypotheses[i] = updated_data
                return True
                
        return False
        
    def delete_hypothesis(self, hypothesis_id: str) -> bool:
        """
        Delete a hypothesis from the study design.
        
        Args:
            hypothesis_id: ID of the hypothesis to delete
            
        Returns:
            True if the hypothesis was found and deleted, False otherwise
        """
        if not hasattr(self, 'hypotheses'):
            return False
            
        for i, hyp in enumerate(self.hypotheses):
            if hyp.get('id') == hypothesis_id:
                del self.hypotheses[i]
                return True
                
        return False
        
    def get_hypothesis(self, hypothesis_id: str) -> Optional[Dict]:
        """
        Get a hypothesis by ID.
        
        Args:
            hypothesis_id: ID of the hypothesis to retrieve
            
        Returns:
            The hypothesis data as a dictionary, or None if not found
        """
        if not hasattr(self, 'hypotheses'):
            return None
            
        for hyp in self.hypotheses:
            if hyp.get('id') == hypothesis_id:
                return hyp
                
        return None
        
    def update_hypothesis_with_results(self, hypothesis_id: str, test_results: Dict) -> bool:
        """
        Update a hypothesis with test results.
        
        Args:
            hypothesis_id: ID of the hypothesis to update
            test_results: Dictionary containing test results
            
        Returns:
            True if the hypothesis was found and updated, False otherwise
        """
        if not hasattr(self, 'hypotheses'):
            return False
            
        for i, hyp in enumerate(self.hypotheses):
            if hyp.get('id') == hypothesis_id:
                # Add test results
                self.hypotheses[i]['test_results'] = test_results
                
                # Update status based on results
                p_value = test_results.get('p_value')
                if p_value is not None:
                    alpha = self.hypotheses[i].get('alpha_level', 0.05)
                    if p_value < alpha:
                        # Significant result - update based on directionality
                        direction = self.hypotheses[i].get('directionality', 'non-directional')
                        
                        # For directional tests, need to check effect direction too
                        if direction == 'non-directional':
                            self.hypotheses[i]['status'] = 'confirmed'
                        else:
                            # Check if effect direction matches hypothesis
                            effect_direction = test_results.get('effect_direction', '')
                            if (direction == 'greater' and effect_direction == 'positive') or \
                               (direction == 'less' and effect_direction == 'negative'):
                                self.hypotheses[i]['status'] = 'confirmed'
                            else:
                                self.hypotheses[i]['status'] = 'rejected'
                    else:
                        self.hypotheses[i]['status'] = 'rejected'
                
                # Update test date
                self.hypotheses[i]['test_date'] = datetime.now().isoformat()
                
                # Update the timestamp
                self.hypotheses[i]['updated_at'] = datetime.now().isoformat()
                
                return True
                
        return False

