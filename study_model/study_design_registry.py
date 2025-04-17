from typing import Dict, List, Set, Optional, Type, Callable
from enum import Enum
from study_model.study_model import StudyType, StatisticalTest
from dataclasses import dataclass

class VariableRequirement(Enum):
    """Specifies whether a variable is required for a study design."""
    REQUIRED = "required"
    OPTIONAL = "optional"
    NOT_USED = "not_used"

@dataclass
class StudyDesignSpecification:
    """Specification for a study design including required variables and compatible tests."""
    study_type: StudyType
    description: str
    variable_requirements: Dict[str, VariableRequirement]  # Maps variable role to requirement
    compatible_tests: List[StatisticalTest]
    example_research_question: str
    
    def get_required_variables(self) -> List[str]:
        """Returns a list of variable roles that are required for this design."""
        return [role for role, req in self.variable_requirements.items() 
                if req == VariableRequirement.REQUIRED]
        
    def get_optional_variables(self) -> List[str]:
        """Returns a list of variable roles that are optional for this design."""
        return [role for role, req in self.variable_requirements.items() 
                if req == VariableRequirement.OPTIONAL]

STUDY_DESIGN_REGISTRY: Dict[StudyType, StudyDesignSpecification] = {
    StudyType.BETWEEN_SUBJECTS: StudyDesignSpecification(
        study_type=StudyType.BETWEEN_SUBJECTS,
        description="Compares different groups of participants, each exposed to a different condition.",
        variable_requirements={
            "outcome": VariableRequirement.REQUIRED,
            "group": VariableRequirement.REQUIRED,
            "subject_id": VariableRequirement.OPTIONAL,
            "time": VariableRequirement.NOT_USED,
            "pair_id": VariableRequirement.NOT_USED,
            "covariate": VariableRequirement.OPTIONAL,
            "event": VariableRequirement.OPTIONAL
        },
        compatible_tests=[
            StatisticalTest.INDEPENDENT_T_TEST,
            StatisticalTest.ONE_WAY_ANOVA,
            StatisticalTest.CHI_SQUARE_TEST,
            StatisticalTest.FISHERS_EXACT_TEST,  # Added
            StatisticalTest.MANN_WHITNEY_U_TEST,
            StatisticalTest.KRUSKAL_WALLIS_TEST,
            StatisticalTest.LOGISTIC_REGRESSION,
            StatisticalTest.MULTINOMIAL_LOGISTIC_REGRESSION,  # Added
            StatisticalTest.LINEAR_REGRESSION,
            StatisticalTest.ANCOVA,
            StatisticalTest.POISSON_REGRESSION,  # Added
            StatisticalTest.NEGATIVE_BINOMIAL_REGRESSION,
            StatisticalTest.ORDINAL_REGRESSION,
            StatisticalTest.POINT_BISERIAL_CORRELATION,
            StatisticalTest.SPEARMAN_CORRELATION,
            StatisticalTest.KENDALL_TAU_CORRELATION,
            StatisticalTest.PEARSON_CORRELATION,
            StatisticalTest.SURVIVAL_ANALYSIS  # Added
        ],
        example_research_question="Does medication A lead to better outcomes than medication B?"
    ),
    
    StudyType.WITHIN_SUBJECTS: StudyDesignSpecification(
        study_type=StudyType.WITHIN_SUBJECTS,
        description="Each participant experiences all conditions, serving as their own control.",
        variable_requirements={
            "outcome": VariableRequirement.REQUIRED,
            "group": VariableRequirement.NOT_USED,  # Not typically used in within-subjects
            "subject_id": VariableRequirement.REQUIRED,
            "time": VariableRequirement.REQUIRED,
            "pair_id": VariableRequirement.NOT_USED,
            "covariate": VariableRequirement.OPTIONAL,
            "event": VariableRequirement.NOT_USED
        },
        compatible_tests=[
            StatisticalTest.PAIRED_T_TEST,
            StatisticalTest.WILCOXON_SIGNED_RANK_TEST,
            StatisticalTest.REPEATED_MEASURES_ANOVA,
            StatisticalTest.LINEAR_MIXED_EFFECTS_MODEL,  # Added
            StatisticalTest.ONE_SAMPLE_T_TEST,  # Added - can be used for change scores
            StatisticalTest.PEARSON_CORRELATION,  # Added - for correlating measures across time
            StatisticalTest.SPEARMAN_CORRELATION  # Added
        ],
        example_research_question="Does an intervention improve outcomes from baseline to follow-up?"
    ),
    
    StudyType.MIXED: StudyDesignSpecification(
        study_type=StudyType.MIXED,
        description="Combines between-subjects and within-subjects factors.",
        variable_requirements={
            "outcome": VariableRequirement.REQUIRED,
            "group": VariableRequirement.REQUIRED,
            "subject_id": VariableRequirement.REQUIRED,
            "time": VariableRequirement.REQUIRED,
            "pair_id": VariableRequirement.OPTIONAL,
            "covariate": VariableRequirement.OPTIONAL,
            "event": VariableRequirement.OPTIONAL
        },
        compatible_tests=[
            StatisticalTest.MIXED_ANOVA,
            StatisticalTest.LINEAR_MIXED_EFFECTS_MODEL,
            StatisticalTest.ANCOVA,  # Added - can use time 1 as covariate
            StatisticalTest.REPEATED_MEASURES_ANOVA,  # Added - with between group factor
            StatisticalTest.SURVIVAL_ANALYSIS  # Added - for time-to-event with group factors
        ],
        example_research_question="Do different treatments affect outcomes differently over time?"
    ),
    
    StudyType.SINGLE_SUBJECT: StudyDesignSpecification(
        study_type=StudyType.SINGLE_SUBJECT,
        description="Intensive study of one or few subjects with repeated measurements.",
        variable_requirements={
            "outcome": VariableRequirement.REQUIRED,
            "group": VariableRequirement.NOT_USED,
            "subject_id": VariableRequirement.REQUIRED,
            "time": VariableRequirement.REQUIRED,
            "pair_id": VariableRequirement.NOT_USED,
            "covariate": VariableRequirement.OPTIONAL,
            "event": VariableRequirement.NOT_USED
        },
        compatible_tests=[
            StatisticalTest.LINEAR_MIXED_EFFECTS_MODEL,
            StatisticalTest.REPEATED_MEASURES_ANOVA,  # Added
            StatisticalTest.WILCOXON_SIGNED_RANK_TEST,  # Added
            StatisticalTest.PAIRED_T_TEST,  # Added
            StatisticalTest.PEARSON_CORRELATION,  # Added
            StatisticalTest.SPEARMAN_CORRELATION  # Added
        ],
        example_research_question="How does a participant's symptoms change in response to treatment phases?"
    ),
    
    StudyType.CROSS_OVER: StudyDesignSpecification(
        study_type=StudyType.CROSS_OVER,
        description="Participants receive all treatments in sequence with washout periods.",
        variable_requirements={
            "outcome": VariableRequirement.REQUIRED,
            "group": VariableRequirement.OPTIONAL,  # For treatment sequence
            "subject_id": VariableRequirement.REQUIRED,
            "time": VariableRequirement.REQUIRED,
            "pair_id": VariableRequirement.NOT_USED,
            "covariate": VariableRequirement.OPTIONAL,
            "event": VariableRequirement.NOT_USED
        },
        compatible_tests=[
            StatisticalTest.PAIRED_T_TEST,
            StatisticalTest.WILCOXON_SIGNED_RANK_TEST,
            StatisticalTest.REPEATED_MEASURES_ANOVA,
            StatisticalTest.LINEAR_MIXED_EFFECTS_MODEL,
            StatisticalTest.MIXED_ANOVA,  # Added
            StatisticalTest.ANCOVA,  # Added
            StatisticalTest.ONE_WAY_ANOVA,  # Added
            StatisticalTest.KRUSKAL_WALLIS_TEST,  # Added
            StatisticalTest.CHI_SQUARE_TEST,  # Added - for categorical outcomes
            StatisticalTest.LOGISTIC_REGRESSION  # Added - for binary outcomes
        ],
        example_research_question="How do outcomes compare when participants receive treatment A then B versus B then A?"
    ),
    
    StudyType.FACTORIAL: StudyDesignSpecification(
        study_type=StudyType.FACTORIAL,
        description="Examines effects of two or more independent variables simultaneously.",
        variable_requirements={
            "outcome": VariableRequirement.REQUIRED,
            "group": VariableRequirement.REQUIRED,  # Multiple group variables needed
            "subject_id": VariableRequirement.OPTIONAL,
            "time": VariableRequirement.OPTIONAL,
            "pair_id": VariableRequirement.NOT_USED,
            "covariate": VariableRequirement.OPTIONAL,
            "event": VariableRequirement.OPTIONAL
        },
        compatible_tests=[
            StatisticalTest.ONE_WAY_ANOVA,  # Would actually use two-way+ ANOVA
            StatisticalTest.LINEAR_REGRESSION,
            StatisticalTest.LOGISTIC_REGRESSION,
            StatisticalTest.LINEAR_MIXED_EFFECTS_MODEL,
            StatisticalTest.MIXED_ANOVA,  # Added
            StatisticalTest.REPEATED_MEASURES_ANOVA,  # Added
            StatisticalTest.ANCOVA,  # Added
            StatisticalTest.CHI_SQUARE_TEST,  # Added
            StatisticalTest.MULTINOMIAL_LOGISTIC_REGRESSION,  # Added
            StatisticalTest.POISSON_REGRESSION,  # Added
            StatisticalTest.NEGATIVE_BINOMIAL_REGRESSION,  # Added
            StatisticalTest.ORDINAL_REGRESSION,  # Added
            StatisticalTest.SURVIVAL_ANALYSIS,  # Added
            StatisticalTest.POINT_BISERIAL_CORRELATION,  # Added - for binary group and continuous outcome
            StatisticalTest.PEARSON_CORRELATION,  # Added for consistency with other correlation tests
            StatisticalTest.SPEARMAN_CORRELATION,  # Already present 
            StatisticalTest.KENDALL_TAU_CORRELATION  # Already present
        ],
        example_research_question="How do drug type and dosage interact to affect patient outcomes?"
    ),
    
    StudyType.REPEATED_CROSS_SECTIONAL: StudyDesignSpecification(
        study_type=StudyType.REPEATED_CROSS_SECTIONAL,
        description="Different participants measured at each time point.",
        variable_requirements={
            "outcome": VariableRequirement.REQUIRED,
            "group": VariableRequirement.OPTIONAL,
            "subject_id": VariableRequirement.OPTIONAL,
            "time": VariableRequirement.REQUIRED,
            "pair_id": VariableRequirement.NOT_USED,
            "covariate": VariableRequirement.OPTIONAL,
            "event": VariableRequirement.OPTIONAL
        },
        compatible_tests=[
            StatisticalTest.INDEPENDENT_T_TEST,
            StatisticalTest.ONE_WAY_ANOVA,
            StatisticalTest.CHI_SQUARE_TEST,
            StatisticalTest.LINEAR_REGRESSION,
            StatisticalTest.LOGISTIC_REGRESSION,  # Added
            StatisticalTest.MULTINOMIAL_LOGISTIC_REGRESSION,  # Added
            StatisticalTest.POISSON_REGRESSION,  # Added
            StatisticalTest.NEGATIVE_BINOMIAL_REGRESSION,  # Added
            StatisticalTest.MANN_WHITNEY_U_TEST,  # Added
            StatisticalTest.KRUSKAL_WALLIS_TEST,  # Added
            StatisticalTest.ANCOVA,  # Added
            StatisticalTest.FISHERS_EXACT_TEST,  # Added
            StatisticalTest.ORDINAL_REGRESSION,  # Added
            StatisticalTest.SURVIVAL_ANALYSIS,  # Added
            StatisticalTest.POINT_BISERIAL_CORRELATION,  # Added - for binary group and continuous outcome
            StatisticalTest.PEARSON_CORRELATION,  # Added for consistency
            StatisticalTest.SPEARMAN_CORRELATION,  # Added for consistency
            StatisticalTest.KENDALL_TAU_CORRELATION  # Added for consistency
        ],
        example_research_question="Has the prevalence of a condition changed over time in the population?"
    ),
    
    StudyType.NESTED: StudyDesignSpecification(
        study_type=StudyType.NESTED,
        description="Hierarchical design with participants nested within groups.",
        variable_requirements={
            "outcome": VariableRequirement.REQUIRED,
            "group": VariableRequirement.REQUIRED,
            "subject_id": VariableRequirement.REQUIRED,
            "time": VariableRequirement.OPTIONAL,
            "pair_id": VariableRequirement.NOT_USED,
            "covariate": VariableRequirement.OPTIONAL,
            "event": VariableRequirement.OPTIONAL
        },
        compatible_tests=[
            StatisticalTest.LINEAR_MIXED_EFFECTS_MODEL,
            StatisticalTest.ONE_WAY_ANOVA,  # Added
            StatisticalTest.LINEAR_REGRESSION,  # Added
            StatisticalTest.LOGISTIC_REGRESSION,  # Added
            StatisticalTest.CHI_SQUARE_TEST,  # Added
            StatisticalTest.SURVIVAL_ANALYSIS,  # Added
            StatisticalTest.POINT_BISERIAL_CORRELATION,  # Added - for binary group and continuous outcome
            StatisticalTest.PEARSON_CORRELATION,  # Added for consistency
            StatisticalTest.SPEARMAN_CORRELATION,  # Added for consistency
            StatisticalTest.KENDALL_TAU_CORRELATION  # Added for consistency
        ],
        example_research_question="How do outcomes differ between schools while accounting for clustering of students within schools?"
    ),
    
    StudyType.LATIN_SQUARE: StudyDesignSpecification(
        study_type=StudyType.LATIN_SQUARE,
        description="Balanced design controlling for order effects.",
        variable_requirements={
            "outcome": VariableRequirement.REQUIRED,
            "group": VariableRequirement.REQUIRED,  # Treatment group
            "subject_id": VariableRequirement.REQUIRED,
            "time": VariableRequirement.REQUIRED,  # Period/sequence
            "pair_id": VariableRequirement.NOT_USED,
            "covariate": VariableRequirement.OPTIONAL,
            "event": VariableRequirement.NOT_USED
        },
        compatible_tests=[
            StatisticalTest.LINEAR_MIXED_EFFECTS_MODEL,
            StatisticalTest.REPEATED_MEASURES_ANOVA,
            StatisticalTest.ANCOVA,  # Added
            StatisticalTest.ONE_WAY_ANOVA,  # Added
            StatisticalTest.MIXED_ANOVA,  # Added
            StatisticalTest.LOGISTIC_REGRESSION,  # Added
            StatisticalTest.POISSON_REGRESSION,  # Added
            StatisticalTest.NEGATIVE_BINOMIAL_REGRESSION,  # Added
            StatisticalTest.ORDINAL_REGRESSION  # Added
        ],
        example_research_question="How do treatments compare when controlling for sequence and period effects?"
    ),

    StudyType.ONE_SAMPLE: StudyDesignSpecification(
        study_type=StudyType.ONE_SAMPLE,
        description="Design with a single sample and a population mean to test against",
        variable_requirements={
            "outcome": VariableRequirement.REQUIRED,
            "subject_id": VariableRequirement.OPTIONAL,
            "time": VariableRequirement.NOT_USED,
            "pair_id": VariableRequirement.NOT_USED,
            "covariate": VariableRequirement.NOT_USED,
            "event": VariableRequirement.NOT_USED
        },
        compatible_tests=[
            StatisticalTest.ONE_SAMPLE_T_TEST
        ],
        example_research_question="How does the sample mean compare to the population mean?"
    ),
    
    StudyType.SURVIVAL_ANALYSIS: StudyDesignSpecification(
        study_type=StudyType.SURVIVAL_ANALYSIS,
        description="Design for time-to-event outcomes with potential censoring",
        variable_requirements={
            "outcome": VariableRequirement.REQUIRED,  # Time-to-event variable
            "group": VariableRequirement.REQUIRED,    # Comparison groups
            "event": VariableRequirement.REQUIRED,    # Event indicator (0=censored, 1=event occurred)
            "subject_id": VariableRequirement.OPTIONAL,
            "time": VariableRequirement.NOT_USED,
            "pair_id": VariableRequirement.NOT_USED,
            "covariate": VariableRequirement.OPTIONAL
        },
        compatible_tests=[
            StatisticalTest.SURVIVAL_ANALYSIS
        ],
        example_research_question="How does the time to event differ between treatment groups?"
    )
}