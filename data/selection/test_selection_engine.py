"""
Module for stepwise determination of appropriate statistical tests.
This module uses an exclusion-based approach to ensure comprehensive test selection.
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import List, Dict, Set, Tuple, Optional, Union, Any, NamedTuple
import logging
import scipy.stats as stats
from scipy.stats import shapiro, levene, bartlett, anderson, jarque_bera, normaltest
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

from study_model.study_model import StatisticalTest, StudyType, CFDataType
from study_model.study_design_registry import STUDY_DESIGN_REGISTRY, VariableRequirement
from study_model.data_type_registry import get_compatible_tests_for_data_type, infer_data_type
from data.selection.select import VariableRole
from data.selection.stat_tests import TEST_REGISTRY

class SelectionCheck(Enum):
    """Types of checks that can be performed during test selection."""
    STUDY_DESIGN_COMPATIBLE = "study_design_compatible"
    REQUIRED_VARIABLES_PRESENT = "required_variables_present"
    DATA_TYPE_COMPATIBLE = "data_type_compatible"
    SAMPLE_SIZE_SUFFICIENT = "sample_size_sufficient"
    ASSUMPTIONS_MET = "assumptions_met"
    DATA_STRUCTURE_VALID = "data_structure_valid"
    MISSING_DATA_ACCEPTABLE = "missing_data_acceptable"
    DATA_CHARACTERISTICS = "data_characteristics"

class AssumptionResult(NamedTuple):
    """Result of an assumption check."""
    test_name: str
    passed: bool
    p_value: Optional[float] = None
    statistic: Optional[float] = None
    details: Optional[str] = None
    severity: str = "warning"  # "warning" or "violation"

class SelectionStepResult:
    """Result of a selection step, including which tests were filtered out and why."""
    
    def __init__(self):
        self.passed_tests: Set[StatisticalTest] = set()
        self.failed_tests: Dict[StatisticalTest, str] = {}
        self.warning_tests: Dict[StatisticalTest, str] = {}
        self.check_type: Optional[SelectionCheck] = None
        self.additional_info: Dict[str, Any] = {}
    
    def add_passed_test(self, test: StatisticalTest):
        """Add a test that passed this selection step."""
        self.passed_tests.add(test)
    
    def add_failed_test(self, test: StatisticalTest, reason: str):
        """Add a test that failed this selection step with a reason."""
        self.failed_tests[test] = reason
    
    def add_warning_test(self, test: StatisticalTest, warning: str):
        """Add a test that has a warning but wasn't excluded."""
        self.warning_tests[test] = warning

class TestSelectionEngine:
    """
    Engine for step-by-step selection of appropriate statistical tests.
    Uses an exclusion-based approach to provide a thorough and robust test selection process.
    """
    
    def __init__(self, dataframe: pd.DataFrame, column_roles: Dict[str, VariableRole]):
        """
        Initialize the test selection engine.
        
        Args:
            dataframe: The dataset to analyze
            column_roles: Dictionary mapping column names to their roles
        """
        self.df = dataframe
        self.column_roles = column_roles
        
        # Extract variable roles for easier access
        self.outcome = next((col for col, role in column_roles.items() 
                    if role == VariableRole.OUTCOME), None)
        self.group = next((col for col, role in column_roles.items() 
                    if role == VariableRole.GROUP), None)
        self.subject_id = next((col for col, role in column_roles.items() 
                       if role == VariableRole.SUBJECT_ID), None)
        self.time = next((col for col, role in column_roles.items() 
                 if role == VariableRole.TIME), None)
        self.pair_id = next((col for col, role in column_roles.items() 
                    if role == VariableRole.PAIR_ID), None)
        self.event = next((col for col, role in column_roles.items() 
                    if role == VariableRole.EVENT), None)
        self.covariates = [col for col, role in column_roles.items() 
                          if role == VariableRole.COVARIATE]
        
        # Initialize tracking
        self.all_tests = list(StatisticalTest)
        self.remaining_tests = set(self.all_tests)
        self.excluded_tests: Dict[StatisticalTest, str] = {}
        self.warnings: Dict[StatisticalTest, List[str]] = {test: [] for test in self.all_tests}
        self.step_results: List[SelectionStepResult] = []
        
        # Storage for statistical checks
        self.normality_results: Dict[str, AssumptionResult] = {}
        self.homogeneity_results: Optional[AssumptionResult] = None
        self.outlier_results: Dict[str, AssumptionResult] = {}
        self.collinearity_results: Dict[str, float] = {}  # VIF values
        self.single_sample_test_appropriate = False
        self.paired_test_appropriate = False
        self.parametric_appropriate = True
        self.regression_appropriate = False
        
        # Default confidence level
        self.confidence_level = "high"
        
        # Analyze the dataset
        self._analyze_dataset()
    
    def _analyze_dataset(self):
        """Analyze the dataset for key characteristics."""
        if self.df is None or self.df.empty:
            return
        
        self.sample_size = len(self.df)
        
        # Initialize time_levels to avoid AttributeError
        self.time_levels = 0
        
        # Initialize zero_variance_outcome to avoid AttributeError
        self.zero_variance_outcome = False
        
        # Analyze outcome variable if available
        if self.outcome:
            self.outcome_data_type = infer_data_type(self.df, self.outcome)
            self.outcome_unique_values = self.df[self.outcome].nunique()
            self.outcome_missing_pct = self.df[self.outcome].isna().mean() * 100
            
            # Determine if data is numeric for additional tests
            outcome_series = self.df[self.outcome]
            self.outcome_is_numeric = pd.api.types.is_numeric_dtype(outcome_series)
            
            # Enhanced categorical data detection
            # Check if the values are strings, objects, or categorical type
            if (pd.api.types.is_string_dtype(outcome_series) or 
                pd.api.types.is_object_dtype(outcome_series) or 
                pd.api.types.is_categorical_dtype(outcome_series)):
                # Override data type to categorical if detected as such
                if self.outcome_data_type != CFDataType.CATEGORICAL:
                    logging.info(f"Overriding detected data type for '{self.outcome}' to CATEGORICAL based on string/object/category type")
                    self.outcome_data_type = CFDataType.CATEGORICAL
                
            # Check if numeric values are actually categories (medium, high, low encoded as 1,2,3)
            elif self.outcome_is_numeric and 2 < self.outcome_unique_values <= 10:
                # Check if values are whole numbers suggesting categorical encoding
                if outcome_series.dropna().apply(lambda x: float(x).is_integer()).all():
                    logging.info(f"Outcome '{self.outcome}' has {self.outcome_unique_values} unique integer values - might be categorical")
                    if self.outcome_data_type != CFDataType.CATEGORICAL and self.outcome_data_type != CFDataType.ORDINAL:
                        logging.info(f"Considering '{self.outcome}' as potentially categorical/ordinal")
            
            if self.outcome_is_numeric:
                # Store basic statistics
                self.outcome_mean = outcome_series.mean()
                self.outcome_median = outcome_series.median()
                self.outcome_std = outcome_series.std()
                self.outcome_min = outcome_series.min()
                self.outcome_max = outcome_series.max()
                
                # Check for zero/near-zero variance - problematic for many tests
                if self.outcome_std < 1e-10:
                    self.zero_variance_outcome = True
                else:
                    self.zero_variance_outcome = False
                
                # Check normality if enough data and numeric
                if len(outcome_series.dropna()) > 3:
                    try:
                        # Shapiro-Wilk test (best for smaller samples)
                        if len(outcome_series.dropna()) <= 5000:
                            stat, p = shapiro(outcome_series.dropna())
                            self.normality_results[self.outcome] = AssumptionResult(
                                test_name="Shapiro-Wilk",
                                passed=p > 0.05,
                                p_value=p,
                                statistic=stat,
                                details=f"p={p:.4f}; {'normal' if p > 0.05 else 'non-normal'} distribution"
                            )
                        # D'Agostino-Pearson for larger samples
                        else:
                            stat, p = normaltest(outcome_series.dropna())
                            self.normality_results[self.outcome] = AssumptionResult(
                                test_name="D'Agostino-Pearson",
                                passed=p > 0.05,
                                p_value=p,
                                statistic=stat,
                                details=f"p={p:.4f}; {'normal' if p > 0.05 else 'non-normal'} distribution"
                            )
                    except Exception as e:
                        self.normality_results[self.outcome] = AssumptionResult(
                            test_name="Normality test",
                            passed=False,
                            details=f"Error testing normality: {str(e)}"
                        )
                
                # Check for extreme outliers using IQR method
                q1 = outcome_series.quantile(0.25)
                q3 = outcome_series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                outliers = outcome_series[(outcome_series < lower_bound) | (outcome_series > upper_bound)]
                
                if len(outliers) > 0:
                    pct_outliers = (len(outliers) / len(outcome_series)) * 100
                    self.outlier_results[self.outcome] = AssumptionResult(
                        test_name="IQR Outlier Detection",
                        passed=pct_outliers < 5,  # Less than 5% outliers is often acceptable
                        details=f"{len(outliers)} outliers ({pct_outliers:.1f}% of data)"
                    )
                else:
                    self.outlier_results[self.outcome] = AssumptionResult(
                        test_name="IQR Outlier Detection",
                        passed=True,
                        details="No extreme outliers detected"
                    )
                
                # Check if single sample test might be appropriate
                # For single sample test, we typically only have outcome variable with no grouping
                if not self.group and not self.time and not self.subject_id:
                    self.single_sample_test_appropriate = True
                    
                # Use normality results to determine if parametric tests are appropriate
                if self.outcome in self.normality_results and not self.normality_results[self.outcome].passed:
                    # Only flag non-parametric for smaller samples
                    if self.sample_size < 30:
                        self.parametric_appropriate = False
        else:
            self.outcome_data_type = None
            self.outcome_unique_values = 0
            self.outcome_missing_pct = 0
            self.outcome_is_numeric = False
            self.zero_variance_outcome = False
        
        # Analyze group variable if available
        if self.group:
            self.group_levels = self.df[self.group].nunique()
            self.group_missing_pct = self.df[self.group].isna().mean() * 100
            
            # Get group balance information
            if self.group_levels > 0:
                group_counts = self.df[self.group].value_counts()
                self.smallest_group_size = group_counts.min()
                self.largest_group_size = group_counts.max()
                if self.smallest_group_size > 0:
                    self.group_imbalance_ratio = self.largest_group_size / self.smallest_group_size
                else:
                    self.group_imbalance_ratio = float('inf')
                    
                # Check if binary (for logistic regression)
                if self.outcome_is_numeric and self.outcome_unique_values == 2:
                    self.regression_appropriate = True
                
                # Check homogeneity of variance if we have groups and numeric outcome
                if self.outcome and self.outcome_is_numeric and self.group_levels >= 2:
                    # Create groups for variance testing
                    groups = []
                    unique_groups = self.df[self.group].dropna().unique()
                    
                    for group_val in unique_groups:
                        group_data = self.df[self.df[self.group] == group_val][self.outcome].dropna()
                        if len(group_data) > 0:
                            groups.append(group_data)
                    
                    # Test homogeneity of variance if we have enough groups with data
                    if len(groups) >= 2 and all(len(g) > 1 for g in groups):
                        try:
                            # Levene's test is more robust to non-normality
                            stat, p = levene(*groups)
                            self.homogeneity_results = AssumptionResult(
                                test_name="Levene's Test",
                                passed=p > 0.05,
                                p_value=p,
                                statistic=stat,
                                details=f"p={p:.4f}; {'equal' if p > 0.05 else 'unequal'} variances"
                            )
                            
                            # If Levene's test fails and we have unequal variances, parametric tests may be inappropriate
                            if p < 0.05 and self.sample_size < 50:
                                self.parametric_appropriate = False
                        except Exception as e:
                            self.homogeneity_results = AssumptionResult(
                                test_name="Levene's Test",
                                passed=False,
                                details=f"Error testing homogeneity: {str(e)}"
                            )
            else:
                self.smallest_group_size = 0
                self.largest_group_size = 0
                self.group_imbalance_ratio = 0
        else:
            self.group_levels = 0
            self.group_missing_pct = 0
            self.smallest_group_size = 0
            self.largest_group_size = 0
            self.group_imbalance_ratio = 0
        
        # Analyze subject_id and time variables for repeated measures
        if self.subject_id:
            # Check if it's truly a subject ID by looking at repeats
            if self.df[self.subject_id].duplicated().any():
                self.has_repeated_measures = True
                self.measurements_per_subject = self.df.groupby(self.subject_id).size()
                self.min_measurements_per_subject = self.measurements_per_subject.min()
                self.max_measurements_per_subject = self.measurements_per_subject.max()
                self.avg_measurements_per_subject = self.measurements_per_subject.mean()
                
                # Check if paired test is appropriate (exactly 2 measurements per subject)
                if self.time and self.time_levels == 2 and self.outcome_is_numeric:
                    valid_subjects = self.measurements_per_subject[self.measurements_per_subject == 2].index
                    if len(valid_subjects) > len(self.measurements_per_subject) * 0.8:  # At least 80% have exactly 2 measures
                        self.paired_test_appropriate = True
            else:
                # No duplicated subject IDs - not repeated measures
                self.has_repeated_measures = False
                self.min_measurements_per_subject = 1
                self.max_measurements_per_subject = 1
                self.avg_measurements_per_subject = 1
        else:
            self.has_repeated_measures = False
            self.min_measurements_per_subject = 0
            self.max_measurements_per_subject = 0
            self.avg_measurements_per_subject = 0
            
        # Time levels if time variable exists
        if self.time:
            self.time_levels = self.df[self.time].nunique()
        else:
            self.time_levels = 0
            
        # Check for multicollinearity if we have multiple predictors
        if len(self.covariates) > 1 and all(pd.api.types.is_numeric_dtype(self.df[cov]) for cov in self.covariates):
            try:
                # Create a smaller df with just the covariates, handling missing values
                cov_df = self.df[self.covariates].dropna()
                
                # Add a constant term for the VIF calculation
                cov_df = sm.add_constant(cov_df)
                
                # Calculate VIF for each covariate
                for i, cov in enumerate(cov_df.columns[1:], 1):  # Skip the constant
                    self.collinearity_results[cov] = variance_inflation_factor(cov_df.values, i)
                    
                # If any VIF > 10, we have strong multicollinearity
                self.has_multicollinearity = any(vif > 10 for vif in self.collinearity_results.values())
            except Exception:
                self.has_multicollinearity = False
        else:
            self.has_multicollinearity = False
            
        # Determine if regression tests are appropriate based on data characteristics
        if self.outcome_is_numeric:
            if (self.group and pd.api.types.is_numeric_dtype(self.df[self.group])) or len(self.covariates) > 0:
                self.regression_appropriate = True
    
    def filter_by_study_design(self, study_type: StudyType) -> SelectionStepResult:
        """
        Filter tests based on compatibility with the study design.
        
        Args:
            study_type: The study design type
            
        Returns:
            SelectionStepResult with the filtered tests
        """
        result = SelectionStepResult()
        result.check_type = SelectionCheck.STUDY_DESIGN_COMPATIBLE
        
        # Get the study design specification from the registry
        design_spec = STUDY_DESIGN_REGISTRY.get(study_type)
        if not design_spec:
            # If no specification is found, don't exclude any tests
            result.passed_tests = set(self.remaining_tests)
            return result
        
        # Get the list of compatible tests for this design
        compatible_tests = set(test.value for test in design_spec.compatible_tests)
        
        # Process each remaining test
        for test in list(self.remaining_tests):
            if test.value in compatible_tests:
                result.add_passed_test(test)
            else:
                result.add_failed_test(test, f"Not compatible with {study_type.display_name} design")
                self.remaining_tests.remove(test)
                self.excluded_tests[test] = f"Not compatible with {study_type.display_name} design"
        
        # Add additional info about the study design
        result.additional_info["study_type"] = study_type
        result.additional_info["design_description"] = design_spec.description
        result.additional_info["required_variables"] = design_spec.get_required_variables()
        result.additional_info["optional_variables"] = design_spec.get_optional_variables()
        
        self.step_results.append(result)
        return result
    
    def filter_by_required_variables(self) -> SelectionStepResult:
        """
        Filter tests based on having the required variables assigned.
        
        Returns:
            SelectionStepResult with the filtered tests
        """
        result = SelectionStepResult()
        result.check_type = SelectionCheck.REQUIRED_VARIABLES_PRESENT
        
        # Check each test for variable requirements
        for test in list(self.remaining_tests):
            missing_vars = self._get_missing_variables(test)
            
            if not missing_vars:
                result.add_passed_test(test)
            else:
                reason = f"Missing required variables: {', '.join(missing_vars)}"
                result.add_failed_test(test, reason)
                self.remaining_tests.remove(test)
                self.excluded_tests[test] = reason
        
        # Add variable assignments to the result for reference
        result.additional_info["assigned_variables"] = {
            "outcome": self.outcome,
            "group": self.group,
            "subject_id": self.subject_id,
            "time": self.time,
            "pair_id": self.pair_id,
            "event": self.event,
            "covariates": self.covariates
        }
        
        self.step_results.append(result)
        return result
    
    def _get_missing_variables(self, test: StatisticalTest) -> List[str]:
        """Get a list of missing required variables for a specific test."""
        missing = []
        
        # One-sample t-test only requires outcome variable
"""
Module for stepwise determination of appropriate statistical tests.
This module uses an exclusion-based approach to ensure comprehensive test selection.
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import List, Dict, Set, Tuple, Optional, Union, Any, NamedTuple
import logging
import scipy.stats as stats
from scipy.stats import shapiro, levene, bartlett, anderson, jarque_bera, normaltest
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

from study_model.study_model import StatisticalTest, StudyType, CFDataType
from study_model.study_design_registry import STUDY_DESIGN_REGISTRY, VariableRequirement
from study_model.data_type_registry import get_compatible_tests_for_data_type, infer_data_type
from data.selection.select import VariableRole
from data.selection.stat_tests import TEST_REGISTRY

class SelectionCheck(Enum):
    """Types of checks that can be performed during test selection."""
    STUDY_DESIGN_COMPATIBLE = "study_design_compatible"
    REQUIRED_VARIABLES_PRESENT = "required_variables_present"
    DATA_TYPE_COMPATIBLE = "data_type_compatible"
    SAMPLE_SIZE_SUFFICIENT = "sample_size_sufficient"
    ASSUMPTIONS_MET = "assumptions_met"
    DATA_STRUCTURE_VALID = "data_structure_valid"
    MISSING_DATA_ACCEPTABLE = "missing_data_acceptable"
    DATA_CHARACTERISTICS = "data_characteristics"

class AssumptionResult(NamedTuple):
    """Result of an assumption check."""
    test_name: str
    passed: bool
    p_value: Optional[float] = None
    statistic: Optional[float] = None
    details: Optional[str] = None
    severity: str = "warning"  # "warning" or "violation"

class SelectionStepResult:
    """Result of a selection step, including which tests were filtered out and why."""
    
    def __init__(self):
        self.passed_tests: Set[StatisticalTest] = set()
        self.failed_tests: Dict[StatisticalTest, str] = {}
        self.warning_tests: Dict[StatisticalTest, str] = {}
        self.check_type: Optional[SelectionCheck] = None
        self.additional_info: Dict[str, Any] = {}
    
    def add_passed_test(self, test: StatisticalTest):
        """Add a test that passed this selection step."""
        self.passed_tests.add(test)
    
    def add_failed_test(self, test: StatisticalTest, reason: str):
        """Add a test that failed this selection step with a reason."""
        self.failed_tests[test] = reason
    
    def add_warning_test(self, test: StatisticalTest, warning: str):
        """Add a test that has a warning but wasn't excluded."""
        self.warning_tests[test] = warning

class TestSelectionEngine:
    """
    Engine for step-by-step selection of appropriate statistical tests.
    Uses an exclusion-based approach to provide a thorough and robust test selection process.
    """
    
    def __init__(self, dataframe: pd.DataFrame, column_roles: Dict[str, VariableRole]):
        """
        Initialize the test selection engine.
        
        Args:
            dataframe: The dataset to analyze
            column_roles: Dictionary mapping column names to their roles
        """
        self.df = dataframe
        self.column_roles = column_roles
        
        # Extract variable roles for easier access
        self.outcome = next((col for col, role in column_roles.items() 
                    if role == VariableRole.OUTCOME), None)
        self.group = next((col for col, role in column_roles.items() 
                    if role == VariableRole.GROUP), None)
        self.subject_id = next((col for col, role in column_roles.items() 
                       if role == VariableRole.SUBJECT_ID), None)
        self.time = next((col for col, role in column_roles.items() 
                 if role == VariableRole.TIME), None)
        self.pair_id = next((col for col, role in column_roles.items() 
                    if role == VariableRole.PAIR_ID), None)
        self.event = next((col for col, role in column_roles.items() 
                    if role == VariableRole.EVENT), None)
        self.covariates = [col for col, role in column_roles.items() 
                          if role == VariableRole.COVARIATE]
        
        # Initialize tracking
        self.all_tests = list(StatisticalTest)
        self.remaining_tests = set(self.all_tests)
        self.excluded_tests: Dict[StatisticalTest, str] = {}
        self.warnings: Dict[StatisticalTest, List[str]] = {test: [] for test in self.all_tests}
        self.step_results: List[SelectionStepResult] = []
        
        # Storage for statistical checks
        self.normality_results: Dict[str, AssumptionResult] = {}
        self.homogeneity_results: Optional[AssumptionResult] = None
        self.outlier_results: Dict[str, AssumptionResult] = {}
        self.collinearity_results: Dict[str, float] = {}  # VIF values
        self.single_sample_test_appropriate = False
        self.paired_test_appropriate = False
        self.parametric_appropriate = True
        self.regression_appropriate = False
        
        # Default confidence level
        self.confidence_level = "high"
        
        # Analyze the dataset
        self._analyze_dataset()
    
    def _analyze_dataset(self):
        """Analyze the dataset for key characteristics."""
        if self.df is None or self.df.empty:
            return
        
        self.sample_size = len(self.df)
        
        # Initialize time_levels to avoid AttributeError
        self.time_levels = 0
        
        # Initialize zero_variance_outcome to avoid AttributeError
        self.zero_variance_outcome = False
        
        # Analyze outcome variable if available
        if self.outcome:
            self.outcome_data_type = infer_data_type(self.df, self.outcome)
            self.outcome_unique_values = self.df[self.outcome].nunique()
            self.outcome_missing_pct = self.df[self.outcome].isna().mean() * 100
            
            # Determine if data is numeric for additional tests
            outcome_series = self.df[self.outcome]
            self.outcome_is_numeric = pd.api.types.is_numeric_dtype(outcome_series)
            
            # Enhanced categorical data detection
            # Check if the values are strings, objects, or categorical type
            if (pd.api.types.is_string_dtype(outcome_series) or 
                pd.api.types.is_object_dtype(outcome_series) or 
                pd.api.types.is_categorical_dtype(outcome_series)):
                # Override data type to categorical if detected as such
                if self.outcome_data_type != CFDataType.CATEGORICAL:
                    logging.info(f"Overriding detected data type for '{self.outcome}' to CATEGORICAL based on string/object/category type")
                    self.outcome_data_type = CFDataType.CATEGORICAL
                
            # Check if numeric values are actually categories (medium, high, low encoded as 1,2,3)
            elif self.outcome_is_numeric and 2 < self.outcome_unique_values <= 10:
                # Check if values are whole numbers suggesting categorical encoding
                if outcome_series.dropna().apply(lambda x: float(x).is_integer()).all():
                    logging.info(f"Outcome '{self.outcome}' has {self.outcome_unique_values} unique integer values - might be categorical")
                    if self.outcome_data_type != CFDataType.CATEGORICAL and self.outcome_data_type != CFDataType.ORDINAL:
                        logging.info(f"Considering '{self.outcome}' as potentially categorical/ordinal")
            
            if self.outcome_is_numeric:
                # Store basic statistics
                self.outcome_mean = outcome_series.mean()
                self.outcome_median = outcome_series.median()
                self.outcome_std = outcome_series.std()
                self.outcome_min = outcome_series.min()
                self.outcome_max = outcome_series.max()
                
                # Check for zero/near-zero variance - problematic for many tests
                if self.outcome_std < 1e-10:
                    self.zero_variance_outcome = True
                else:
                    self.zero_variance_outcome = False
                
                # Check normality if enough data and numeric
                if len(outcome_series.dropna()) > 3:
                    try:
                        # Shapiro-Wilk test (best for smaller samples)
                        if len(outcome_series.dropna()) <= 5000:
                            stat, p = shapiro(outcome_series.dropna())
                            self.normality_results[self.outcome] = AssumptionResult(
                                test_name="Shapiro-Wilk",
                                passed=p > 0.05,
                                p_value=p,
                                statistic=stat,
                                details=f"p={p:.4f}; {'normal' if p > 0.05 else 'non-normal'} distribution"
                            )
                        # D'Agostino-Pearson for larger samples
                        else:
                            stat, p = normaltest(outcome_series.dropna())
                            self.normality_results[self.outcome] = AssumptionResult(
                                test_name="D'Agostino-Pearson",
                                passed=p > 0.05,
                                p_value=p,
                                statistic=stat,
                                details=f"p={p:.4f}; {'normal' if p > 0.05 else 'non-normal'} distribution"
                            )
                    except Exception as e:
                        self.normality_results[self.outcome] = AssumptionResult(
                            test_name="Normality test",
                            passed=False,
                            details=f"Error testing normality: {str(e)}"
                        )
                
                # Check for extreme outliers using IQR method
                q1 = outcome_series.quantile(0.25)
                q3 = outcome_series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                outliers = outcome_series[(outcome_series < lower_bound) | (outcome_series > upper_bound)]
                
                if len(outliers) > 0:
                    pct_outliers = (len(outliers) / len(outcome_series)) * 100
                    self.outlier_results[self.outcome] = AssumptionResult(
                        test_name="IQR Outlier Detection",
                        passed=pct_outliers < 5,  # Less than 5% outliers is often acceptable
                        details=f"{len(outliers)} outliers ({pct_outliers:.1f}% of data)"
                    )
                else:
                    self.outlier_results[self.outcome] = AssumptionResult(
                        test_name="IQR Outlier Detection",
                        passed=True,
                        details="No extreme outliers detected"
                    )
                
                # Check if single sample test might be appropriate
                # For single sample test, we typically only have outcome variable with no grouping
                if not self.group and not self.time and not self.subject_id:
                    self.single_sample_test_appropriate = True
                    
                # Use normality results to determine if parametric tests are appropriate
                if self.outcome in self.normality_results and not self.normality_results[self.outcome].passed:
                    # Only flag non-parametric for smaller samples
                    if self.sample_size < 30:
                        self.parametric_appropriate = False
        else:
            self.outcome_data_type = None
            self.outcome_unique_values = 0
            self.outcome_missing_pct = 0
            self.outcome_is_numeric = False
            self.zero_variance_outcome = False
        
        # Analyze group variable if available
        if self.group:
            self.group_levels = self.df[self.group].nunique()
            self.group_missing_pct = self.df[self.group].isna().mean() * 100
            
            # Get group balance information
            if self.group_levels > 0:
                group_counts = self.df[self.group].value_counts()
                self.smallest_group_size = group_counts.min()
                self.largest_group_size = group_counts.max()
                if self.smallest_group_size > 0:
                    self.group_imbalance_ratio = self.largest_group_size / self.smallest_group_size
                else:
                    self.group_imbalance_ratio = float('inf')
                    
                # Check if binary (for logistic regression)
                if self.outcome_is_numeric and self.outcome_unique_values == 2:
                    self.regression_appropriate = True
                
                # Check homogeneity of variance if we have groups and numeric outcome
                if self.outcome and self.outcome_is_numeric and self.group_levels >= 2:
                    # Create groups for variance testing
                    groups = []
                    unique_groups = self.df[self.group].dropna().unique()
                    
                    for group_val in unique_groups:
                        group_data = self.df[self.df[self.group] == group_val][self.outcome].dropna()
                        if len(group_data) > 0:
                            groups.append(group_data)
                    
                    # Test homogeneity of variance if we have enough groups with data
                    if len(groups) >= 2 and all(len(g) > 1 for g in groups):
                        try:
                            # Levene's test is more robust to non-normality
                            stat, p = levene(*groups)
                            self.homogeneity_results = AssumptionResult(
                                test_name="Levene's Test",
                                passed=p > 0.05,
                                p_value=p,
                                statistic=stat,
                                details=f"p={p:.4f}; {'equal' if p > 0.05 else 'unequal'} variances"
                            )
                            
                            # If Levene's test fails and we have unequal variances, parametric tests may be inappropriate
                            if p < 0.05 and self.sample_size < 50:
                                self.parametric_appropriate = False
                        except Exception as e:
                            self.homogeneity_results = AssumptionResult(
                                test_name="Levene's Test",
                                passed=False,
                                details=f"Error testing homogeneity: {str(e)}"
                            )
            else:
                self.smallest_group_size = 0
                self.largest_group_size = 0
                self.group_imbalance_ratio = 0
        else:
            self.group_levels = 0
            self.group_missing_pct = 0
            self.smallest_group_size = 0
            self.largest_group_size = 0
            self.group_imbalance_ratio = 0
        
        # Analyze subject_id and time variables for repeated measures
        if self.subject_id:
            # Check if it's truly a subject ID by looking at repeats
            if self.df[self.subject_id].duplicated().any():
                self.has_repeated_measures = True
                self.measurements_per_subject = self.df.groupby(self.subject_id).size()
                self.min_measurements_per_subject = self.measurements_per_subject.min()
                self.max_measurements_per_subject = self.measurements_per_subject.max()
                self.avg_measurements_per_subject = self.measurements_per_subject.mean()
                
                # Check if paired test is appropriate (exactly 2 measurements per subject)
                if self.time and self.time_levels == 2 and self.outcome_is_numeric:
                    valid_subjects = self.measurements_per_subject[self.measurements_per_subject == 2].index
                    if len(valid_subjects) > len(self.measurements_per_subject) * 0.8:  # At least 80% have exactly 2 measures
                        self.paired_test_appropriate = True
            else:
                # No duplicated subject IDs - not repeated measures
                self.has_repeated_measures = False
                self.min_measurements_per_subject = 1
                self.max_measurements_per_subject = 1
                self.avg_measurements_per_subject = 1
        else:
            self.has_repeated_measures = False
            self.min_measurements_per_subject = 0
            self.max_measurements_per_subject = 0
            self.avg_measurements_per_subject = 0
            
        # Time levels if time variable exists
        if self.time:
            self.time_levels = self.df[self.time].nunique()
        else:
            self.time_levels = 0
            
        # Check for multicollinearity if we have multiple predictors
        if len(self.covariates) > 1 and all(pd.api.types.is_numeric_dtype(self.df[cov]) for cov in self.covariates):
            try:
                # Create a smaller df with just the covariates, handling missing values
                cov_df = self.df[self.covariates].dropna()
                
                # Add a constant term for the VIF calculation
                cov_df = sm.add_constant(cov_df)
                
                # Calculate VIF for each covariate
                for i, cov in enumerate(cov_df.columns[1:], 1):  # Skip the constant
                    self.collinearity_results[cov] = variance_inflation_factor(cov_df.values, i)
                    
                # If any VIF > 10, we have strong multicollinearity
                self.has_multicollinearity = any(vif > 10 for vif in self.collinearity_results.values())
            except Exception:
                self.has_multicollinearity = False
        else:
            self.has_multicollinearity = False
            
        # Determine if regression tests are appropriate based on data characteristics
        if self.outcome_is_numeric:
            if (self.group and pd.api.types.is_numeric_dtype(self.df[self.group])) or len(self.covariates) > 0:
                self.regression_appropriate = True
    
    def filter_by_study_design(self, study_type: StudyType) -> SelectionStepResult:
        """
        Filter tests based on compatibility with the study design.
        
        Args:
            study_type: The study design type
            
        Returns:
            SelectionStepResult with the filtered tests
        """
        result = SelectionStepResult()
        result.check_type = SelectionCheck.STUDY_DESIGN_COMPATIBLE
        
        # Get the study design specification from the registry
        design_spec = STUDY_DESIGN_REGISTRY.get(study_type)
        if not design_spec:
            # If no specification is found, don't exclude any tests
            result.passed_tests = set(self.remaining_tests)
            return result
        
        # Get the list of compatible tests for this design
        compatible_tests = set(test.value for test in design_spec.compatible_tests)
        
        # Process each remaining test
        for test in list(self.remaining_tests):
            if test.value in compatible_tests:
                result.add_passed_test(test)
            else:
                result.add_failed_test(test, f"Not compatible with {study_type.display_name} design")
                self.remaining_tests.remove(test)
                self.excluded_tests[test] = f"Not compatible with {study_type.display_name} design"
        
        # Add additional info about the study design
        result.additional_info["study_type"] = study_type
        result.additional_info["design_description"] = design_spec.description
        result.additional_info["required_variables"] = design_spec.get_required_variables()
        result.additional_info["optional_variables"] = design_spec.get_optional_variables()
        
        self.step_results.append(result)
        return result
    
    def filter_by_required_variables(self) -> SelectionStepResult:
        """
        Filter tests based on having the required variables assigned.
        
        Returns:
            SelectionStepResult with the filtered tests
        """
        result = SelectionStepResult()
        result.check_type = SelectionCheck.REQUIRED_VARIABLES_PRESENT
        
        # Check each test for variable requirements
        for test in list(self.remaining_tests):
            missing_vars = self._get_missing_variables(test)
            
            if not missing_vars:
                result.add_passed_test(test)
            else:
                reason = f"Missing required variables: {', '.join(missing_vars)}"
                result.add_failed_test(test, reason)
                self.remaining_tests.remove(test)
                self.excluded_tests[test] = reason
        
        # Add variable assignments to the result for reference
        result.additional_info["assigned_variables"] = {
            "outcome": self.outcome,
            "group": self.group,
            "subject_id": self.subject_id,
            "time": self.time,
            "pair_id": self.pair_id,
            "covariates": self.covariates
        }
        
        self.step_results.append(result)
        return result
    
    def _get_missing_variables(self, test: StatisticalTest) -> List[str]:
        """Get a list of missing required variables for a specific test."""
        missing = []
        
        # One-sample t-test only requires outcome variable
        if test == StatisticalTest.ONE_SAMPLE_T_TEST:
            if not self.outcome:
                missing.append("outcome")
            return missing
            
        # All other tests require outcome
        if not self.outcome:
            missing.append("outcome")
        
        if test in [StatisticalTest.INDEPENDENT_T_TEST, StatisticalTest.MANN_WHITNEY_U_TEST,
                   StatisticalTest.CHI_SQUARE_TEST, StatisticalTest.FISHERS_EXACT_TEST]:
            if not self.group:
                missing.append("group")
            elif self.df is not None and self.group and self.df[self.group].nunique() != 2:
                missing.append("binary group variable (must have exactly 2 unique values)")
        
        elif test in [StatisticalTest.ONE_WAY_ANOVA, StatisticalTest.KRUSKAL_WALLIS_TEST]:
            if not self.group:
                missing.append("group")
            elif self.df is not None and self.group and self.df[self.group].nunique() < 3:
                missing.append("group variable with at least 3 categories")
        
        elif test in [StatisticalTest.PAIRED_T_TEST, StatisticalTest.WILCOXON_SIGNED_RANK_TEST]:
            if not self.subject_id:
                missing.append("subject_id")
            if not self.time:
                missing.append("time")
        
        elif test == StatisticalTest.REPEATED_MEASURES_ANOVA:
            if not self.subject_id:
                missing.append("subject_id")
            if not self.time:
                missing.append("time")
        
        elif test == StatisticalTest.MIXED_ANOVA:
            if not self.group:
                missing.append("group")
            if not self.subject_id:
                missing.append("subject_id")
            if not self.time:
                missing.append("time")
        
        elif test in [StatisticalTest.LINEAR_REGRESSION, StatisticalTest.LOGISTIC_REGRESSION, 
                    StatisticalTest.MULTINOMIAL_LOGISTIC_REGRESSION, StatisticalTest.POISSON_REGRESSION,
                    StatisticalTest.NEGATIVE_BINOMIAL_REGRESSION, StatisticalTest.ORDINAL_REGRESSION]:
            if not (self.group or self.covariates):
                missing.append("group or covariates (at least one predictor variable)")
            
            # Additional check specifically for multinomial logistic regression
            if test == StatisticalTest.MULTINOMIAL_LOGISTIC_REGRESSION:
                # Ensure outcome has more than 2 categories for multinomial
                if self.outcome and self.df is not None and self.outcome in self.df.columns:
                    num_categories = self.df[self.outcome].nunique()
                    if num_categories < 3:
                        missing.append(f"outcome with at least 3 categories (current: {num_categories})")
        
        elif test == StatisticalTest.ANCOVA:
            if not self.group:
                missing.append("group")
            if not self.covariates:
                missing.append("covariates")
        
        elif test == StatisticalTest.LINEAR_MIXED_EFFECTS_MODEL:
            if not self.subject_id:
                missing.append("subject_id")
        
        elif test in [StatisticalTest.PEARSON_CORRELATION, StatisticalTest.SPEARMAN_CORRELATION, 
                    StatisticalTest.KENDALL_TAU_CORRELATION]:
            if not (self.group or self.covariates):
                missing.append("second variable (group or covariate)")

        elif test == StatisticalTest.POINT_BISERIAL_CORRELATION:
            if not self.group:
                missing.append("binary group variable")
            elif self.df is not None and self.group and self.df[self.group].nunique() != 2:
                missing.append("binary group variable (must have exactly 2 unique values)")
        
        elif test == StatisticalTest.SURVIVAL_ANALYSIS:
            if not self.outcome:
                missing.append("time-to-event variable")
            if not self.group:
                missing.append("group variable")
            
            # Check for event indicator
            event_var = next((col for col, role in self.column_roles.items() 
                            if role == VariableRole.EVENT), None) if self.column_roles else None
            if not event_var:
                missing.append("event indicator variable")
        
        return missing
    
    def filter_by_data_type(self) -> SelectionStepResult:
        """
        Filter tests based on compatibility with the outcome data type.
        
        Returns:
            SelectionStepResult with the filtered tests
        """
        result = SelectionStepResult()
        result.check_type = SelectionCheck.DATA_TYPE_COMPATIBLE
        
        # Skip if outcome variable or type is not available
        if not self.outcome or not self.outcome_data_type:
            # If no outcome data type, don't filter
            result.passed_tests = set(self.remaining_tests)
            result.additional_info["outcome_data_type"] = None
            self.step_results.append(result)
            return result
        
        # Get compatible tests for the outcome data type
        compatible_tests = set(test.value for test in get_compatible_tests_for_data_type(self.outcome_data_type))
        
        # Log data type detection information
        logging.info(f"Outcome '{self.outcome}' detected as {self.outcome_data_type.value}")
        logging.info(f"Unique values in outcome: {self.outcome_unique_values}")
        if self.outcome_data_type == CFDataType.CATEGORICAL:
            # Show the first few values for debugging
            sample_values = self.df[self.outcome].dropna().unique()[:5]
            logging.info(f"Sample outcome values: {sample_values}")
            logging.info(f"Compatible tests for categorical data: {compatible_tests}")
        
        # Process each remaining test
        for test in list(self.remaining_tests):
            if test.value in compatible_tests:
                result.add_passed_test(test)
            else:
                reason = f"Not compatible with {self.outcome_data_type.value} outcome data type"
                result.add_failed_test(test, reason)
                self.remaining_tests.remove(test)
                self.excluded_tests[test] = reason
                logging.info(f"Excluded test {test.value}: {reason}")
        
        # Add data type info to the result
        result.additional_info["outcome_data_type"] = self.outcome_data_type
        
        self.step_results.append(result)
        return result
    
    def check_sample_size(self) -> SelectionStepResult:
        """
        Check if the sample size is sufficient for each test.
        This step adds warnings but doesn't exclude tests.
        
        Returns:
            SelectionStepResult with sample size warnings
        """
        result = SelectionStepResult()
        result.check_type = SelectionCheck.SAMPLE_SIZE_SUFFICIENT
        
        # Add all tests as passing (we only add warnings)
        result.passed_tests = set(self.remaining_tests)
        
        for test in self.remaining_tests:
            test_data = TEST_REGISTRY.get(test.value, None)
            if not test_data:
                continue
                
            # Get recommended sample size for this test
            recommended_size = test_data.sample_size
            
            # Compare with actual sample size
            if self.sample_size < recommended_size:
                warning = f"Sample size ({self.sample_size}) below recommended minimum ({recommended_size})"
                result.add_warning_test(test, warning)
                self.warnings[test].append(warning)
                
        # Add sample size info to the result
        result.additional_info["sample_size"] = self.sample_size
        
        self.step_results.append(result)
        return result
    
    def check_data_structure(self) -> SelectionStepResult:
        """
        Check if data structure is appropriate for each test.
        For example, if repeated measures tests require multiple observations per subject.
        
        Returns:
            SelectionStepResult with data structure checks
        """
        result = SelectionStepResult()
        result.check_type = SelectionCheck.DATA_STRUCTURE_VALID
        
        # Specific data structure checks for different test types
        
        # Repeated measures checks
        repeated_measures_tests = [
            StatisticalTest.PAIRED_T_TEST,
            StatisticalTest.WILCOXON_SIGNED_RANK_TEST,
            StatisticalTest.REPEATED_MEASURES_ANOVA,
            StatisticalTest.MIXED_ANOVA,
            StatisticalTest.LINEAR_MIXED_EFFECTS_MODEL
        ]
        
        for test in list(self.remaining_tests):
            # For tests that require repeated measures
            if test in repeated_measures_tests:
                if not self.has_repeated_measures:
                    reason = "Data does not have repeated measurements per subject"
                    result.add_failed_test(test, reason)
                    self.remaining_tests.remove(test)
                    self.excluded_tests[test] = reason
                elif self.min_measurements_per_subject < 2:
                    reason = "Some subjects have fewer than 2 measurements"
                    result.add_failed_test(test, reason)
                    self.remaining_tests.remove(test)
                    self.excluded_tests[test] = reason
                else:
                    result.add_passed_test(test)
            
            # For point-biserial correlation
            elif test == StatisticalTest.POINT_BISERIAL_CORRELATION:
                if self.group and self.df[self.group].nunique() != 2:
                    reason = f"Group variable must have exactly 2 levels, but has {self.df[self.group].nunique()}"
                    result.add_failed_test(test, reason)
                    self.remaining_tests.remove(test)
                    self.excluded_tests[test] = reason
                else:
                    result.add_passed_test(test)
            
            # For chi-square test
            elif test == StatisticalTest.CHI_SQUARE_TEST:
                # Chi-square needs sufficient expected counts
                if self.group and self.outcome and self.smallest_group_size < 5:
                    warning = "Some groups have fewer than 5 observations, which may violate chi-square assumptions"
                    result.add_warning_test(test, warning)
                    self.warnings[test].append(warning)
                    result.add_passed_test(test)
                else:
                    result.add_passed_test(test)
            
            # For ANOVA tests
            elif test in [StatisticalTest.ONE_WAY_ANOVA, StatisticalTest.KRUSKAL_WALLIS_TEST]:
                if self.group and self.group_levels < 2:
                    reason = f"Group variable must have at least 2 levels, but has {self.group_levels}"
                    result.add_failed_test(test, reason)
                    self.remaining_tests.remove(test)
                    self.excluded_tests[test] = reason
                else:
                    result.add_passed_test(test)
            
"""
Module for stepwise determination of appropriate statistical tests.
This module uses an exclusion-based approach to ensure comprehensive test selection.
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import List, Dict, Set, Tuple, Optional, Union, Any, NamedTuple
import logging
import scipy.stats as stats
from scipy.stats import shapiro, levene, bartlett, anderson, jarque_bera, normaltest
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

from study_model.study_model import StatisticalTest, StudyType, CFDataType
from study_model.study_design_registry import STUDY_DESIGN_REGISTRY, VariableRequirement
from study_model.data_type_registry import get_compatible_tests_for_data_type, infer_data_type
from data.selection.select import VariableRole
from data.selection.stat_tests import TEST_REGISTRY

class SelectionCheck(Enum):
    """Types of checks that can be performed during test selection."""
    STUDY_DESIGN_COMPATIBLE = "study_design_compatible"
    REQUIRED_VARIABLES_PRESENT = "required_variables_present"
    DATA_TYPE_COMPATIBLE = "data_type_compatible"
    SAMPLE_SIZE_SUFFICIENT = "sample_size_sufficient"
    ASSUMPTIONS_MET = "assumptions_met"
    DATA_STRUCTURE_VALID = "data_structure_valid"
    MISSING_DATA_ACCEPTABLE = "missing_data_acceptable"
    DATA_CHARACTERISTICS = "data_characteristics"

class AssumptionResult(NamedTuple):
    """Result of an assumption check."""
    test_name: str
    passed: bool
    p_value: Optional[float] = None
    statistic: Optional[float] = None
    details: Optional[str] = None
    severity: str = "warning"  # "warning" or "violation"

class SelectionStepResult:
    """Result of a selection step, including which tests were filtered out and why."""
    
    def __init__(self):
        self.passed_tests: Set[StatisticalTest] = set()
        self.failed_tests: Dict[StatisticalTest, str] = {}
        self.warning_tests: Dict[StatisticalTest, str] = {}
        self.check_type: Optional[SelectionCheck] = None
        self.additional_info: Dict[str, Any] = {}
    
    def add_passed_test(self, test: StatisticalTest):
        """Add a test that passed this selection step."""
        self.passed_tests.add(test)
    
    def add_failed_test(self, test: StatisticalTest, reason: str):
        """Add a test that failed this selection step with a reason."""
        self.failed_tests[test] = reason
    
    def add_warning_test(self, test: StatisticalTest, warning: str):
        """Add a test that has a warning but wasn't excluded."""
        self.warning_tests[test] = warning

class TestSelectionEngine:
    """
    Engine for step-by-step selection of appropriate statistical tests.
    Uses an exclusion-based approach to provide a thorough and robust test selection process.
    """
    
    def __init__(self, dataframe: pd.DataFrame, column_roles: Dict[str, VariableRole]):
        """
        Initialize the test selection engine.
        
        Args:
            dataframe: The dataset to analyze
            column_roles: Dictionary mapping column names to their roles
        """
        self.df = dataframe
        self.column_roles = column_roles
        
        # Extract variable roles for easier access
        self.outcome = next((col for col, role in column_roles.items() 
                    if role == VariableRole.OUTCOME), None)
        self.group = next((col for col, role in column_roles.items() 
                    if role == VariableRole.GROUP), None)
        self.subject_id = next((col for col, role in column_roles.items() 
                       if role == VariableRole.SUBJECT_ID), None)
        self.time = next((col for col, role in column_roles.items() 
                 if role == VariableRole.TIME), None)
        self.pair_id = next((col for col, role in column_roles.items() 
                    if role == VariableRole.PAIR_ID), None)
        self.covariates = [col for col, role in column_roles.items() 
                          if role == VariableRole.COVARIATE]
        
        # Initialize tracking
        self.all_tests = list(StatisticalTest)
        self.remaining_tests = set(self.all_tests)
        self.excluded_tests: Dict[StatisticalTest, str] = {}
        self.warnings: Dict[StatisticalTest, List[str]] = {test: [] for test in self.all_tests}
        self.step_results: List[SelectionStepResult] = []
        
        # Storage for statistical checks
        self.normality_results: Dict[str, AssumptionResult] = {}
        self.homogeneity_results: Optional[AssumptionResult] = None
        self.outlier_results: Dict[str, AssumptionResult] = {}
        self.collinearity_results: Dict[str, float] = {}  # VIF values
        self.single_sample_test_appropriate = False
        self.paired_test_appropriate = False
        self.parametric_appropriate = True
        self.regression_appropriate = False
        
        # Default confidence level
        self.confidence_level = "high"
        
        # Analyze the dataset
        self._analyze_dataset()
    
    def _analyze_dataset(self):
        """Analyze the dataset for key characteristics."""
        if self.df is None or self.df.empty:
            return
        
        self.sample_size = len(self.df)
        
        # Initialize time_levels to avoid AttributeError
        self.time_levels = 0
        
        # Initialize zero_variance_outcome to avoid AttributeError
        self.zero_variance_outcome = False
        
        # Analyze outcome variable if available
        if self.outcome:
            self.outcome_data_type = infer_data_type(self.df, self.outcome)
            self.outcome_unique_values = self.df[self.outcome].nunique()
            self.outcome_missing_pct = self.df[self.outcome].isna().mean() * 100
            
            # Determine if data is numeric for additional tests
            outcome_series = self.df[self.outcome]
            self.outcome_is_numeric = pd.api.types.is_numeric_dtype(outcome_series)
            
            # Enhanced categorical data detection
            # Check if the values are strings, objects, or categorical type
            if (pd.api.types.is_string_dtype(outcome_series) or 
                pd.api.types.is_object_dtype(outcome_series) or 
                pd.api.types.is_categorical_dtype(outcome_series)):
                # Override data type to categorical if detected as such
                if self.outcome_data_type != CFDataType.CATEGORICAL:
                    logging.info(f"Overriding detected data type for '{self.outcome}' to CATEGORICAL based on string/object/category type")
                    self.outcome_data_type = CFDataType.CATEGORICAL
                
            # Check if numeric values are actually categories (medium, high, low encoded as 1,2,3)
            elif self.outcome_is_numeric and 2 < self.outcome_unique_values <= 10:
                # Check if values are whole numbers suggesting categorical encoding
                if outcome_series.dropna().apply(lambda x: float(x).is_integer()).all():
                    logging.info(f"Outcome '{self.outcome}' has {self.outcome_unique_values} unique integer values - might be categorical")
                    if self.outcome_data_type != CFDataType.CATEGORICAL and self.outcome_data_type != CFDataType.ORDINAL:
                        logging.info(f"Considering '{self.outcome}' as potentially categorical/ordinal")
            
            if self.outcome_is_numeric:
                # Store basic statistics
                self.outcome_mean = outcome_series.mean()
                self.outcome_median = outcome_series.median()
                self.outcome_std = outcome_series.std()
                self.outcome_min = outcome_series.min()
                self.outcome_max = outcome_series.max()
                
                # Check for zero/near-zero variance - problematic for many tests
                if self.outcome_std < 1e-10:
                    self.zero_variance_outcome = True
                else:
                    self.zero_variance_outcome = False
                
                # Check normality if enough data and numeric
                if len(outcome_series.dropna()) > 3:
                    try:
                        # Shapiro-Wilk test (best for smaller samples)
                        if len(outcome_series.dropna()) <= 5000:
                            stat, p = shapiro(outcome_series.dropna())
                            self.normality_results[self.outcome] = AssumptionResult(
                                test_name="Shapiro-Wilk",
                                passed=p > 0.05,
                                p_value=p,
                                statistic=stat,
                                details=f"p={p:.4f}; {'normal' if p > 0.05 else 'non-normal'} distribution"
                            )
                        # D'Agostino-Pearson for larger samples
                        else:
                            stat, p = normaltest(outcome_series.dropna())
                            self.normality_results[self.outcome] = AssumptionResult(
                                test_name="D'Agostino-Pearson",
                                passed=p > 0.05,
                                p_value=p,
                                statistic=stat,
                                details=f"p={p:.4f}; {'normal' if p > 0.05 else 'non-normal'} distribution"
                            )
                    except Exception as e:
                        self.normality_results[self.outcome] = AssumptionResult(
                            test_name="Normality test",
                            passed=False,
                            details=f"Error testing normality: {str(e)}"
                        )
                
                # Check for extreme outliers using IQR method
                q1 = outcome_series.quantile(0.25)
                q3 = outcome_series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                outliers = outcome_series[(outcome_series < lower_bound) | (outcome_series > upper_bound)]
                
                if len(outliers) > 0:
                    pct_outliers = (len(outliers) / len(outcome_series)) * 100
                    self.outlier_results[self.outcome] = AssumptionResult(
                        test_name="IQR Outlier Detection",
                        passed=pct_outliers < 5,  # Less than 5% outliers is often acceptable
                        details=f"{len(outliers)} outliers ({pct_outliers:.1f}% of data)"
                    )
                else:
                    self.outlier_results[self.outcome] = AssumptionResult(
                        test_name="IQR Outlier Detection",
                        passed=True,
                        details="No extreme outliers detected"
                    )
                
                # Check if single sample test might be appropriate
                # For single sample test, we typically only have outcome variable with no grouping
                if not self.group and not self.time and not self.subject_id:
                    self.single_sample_test_appropriate = True
                    
                # Use normality results to determine if parametric tests are appropriate
                if self.outcome in self.normality_results and not self.normality_results[self.outcome].passed:
                    # Only flag non-parametric for smaller samples
                    if self.sample_size < 30:
                        self.parametric_appropriate = False
        else:
            self.outcome_data_type = None
            self.outcome_unique_values = 0
            self.outcome_missing_pct = 0
            self.outcome_is_numeric = False
            self.zero_variance_outcome = False
        
        # Analyze group variable if available
        if self.group:
            self.group_levels = self.df[self.group].nunique()
            self.group_missing_pct = self.df[self.group].isna().mean() * 100
            
            # Get group balance information
            if self.group_levels > 0:
                group_counts = self.df[self.group].value_counts()
                self.smallest_group_size = group_counts.min()
                self.largest_group_size = group_counts.max()
                if self.smallest_group_size > 0:
                    self.group_imbalance_ratio = self.largest_group_size / self.smallest_group_size
                else:
                    self.group_imbalance_ratio = float('inf')
                    
                # Check if binary (for logistic regression)
                if self.outcome_is_numeric and self.outcome_unique_values == 2:
                    self.regression_appropriate = True
                
                # Check homogeneity of variance if we have groups and numeric outcome
                if self.outcome and self.outcome_is_numeric and self.group_levels >= 2:
                    # Create groups for variance testing
                    groups = []
                    unique_groups = self.df[self.group].dropna().unique()
                    
                    for group_val in unique_groups:
                        group_data = self.df[self.df[self.group] == group_val][self.outcome].dropna()
                        if len(group_data) > 0:
                            groups.append(group_data)
                    
                    # Test homogeneity of variance if we have enough groups with data
                    if len(groups) >= 2 and all(len(g) > 1 for g in groups):
                        try:
                            # Levene's test is more robust to non-normality
                            stat, p = levene(*groups)
                            self.homogeneity_results = AssumptionResult(
                                test_name="Levene's Test",
                                passed=p > 0.05,
                                p_value=p,
                                statistic=stat,
                                details=f"p={p:.4f}; {'equal' if p > 0.05 else 'unequal'} variances"
                            )
                            
                            # If Levene's test fails and we have unequal variances, parametric tests may be inappropriate
                            if p < 0.05 and self.sample_size < 50:
                                self.parametric_appropriate = False
                        except Exception as e:
                            self.homogeneity_results = AssumptionResult(
                                test_name="Levene's Test",
                                passed=False,
                                details=f"Error testing homogeneity: {str(e)}"
                            )
            else:
                self.smallest_group_size = 0
                self.largest_group_size = 0
                self.group_imbalance_ratio = 0
        else:
            self.group_levels = 0
            self.group_missing_pct = 0
            self.smallest_group_size = 0
            self.largest_group_size = 0
            self.group_imbalance_ratio = 0
        
        # Analyze subject_id and time variables for repeated measures
        if self.subject_id:
            # Check if it's truly a subject ID by looking at repeats
            if self.df[self.subject_id].duplicated().any():
                self.has_repeated_measures = True
                self.measurements_per_subject = self.df.groupby(self.subject_id).size()
                self.min_measurements_per_subject = self.measurements_per_subject.min()
                self.max_measurements_per_subject = self.measurements_per_subject.max()
                self.avg_measurements_per_subject = self.measurements_per_subject.mean()
                
                # Check if paired test is appropriate (exactly 2 measurements per subject)
                if self.time and self.time_levels == 2 and self.outcome_is_numeric:
                    valid_subjects = self.measurements_per_subject[self.measurements_per_subject == 2].index
                    if len(valid_subjects) > len(self.measurements_per_subject) * 0.8:  # At least 80% have exactly 2 measures
                        self.paired_test_appropriate = True
            else:
                # No duplicated subject IDs - not repeated measures
                self.has_repeated_measures = False
                self.min_measurements_per_subject = 1
                self.max_measurements_per_subject = 1
                self.avg_measurements_per_subject = 1
        else:
            self.has_repeated_measures = False
            self.min_measurements_per_subject = 0
            self.max_measurements_per_subject = 0
            self.avg_measurements_per_subject = 0
            
        # Time levels if time variable exists
        if self.time:
            self.time_levels = self.df[self.time].nunique()
        else:
            self.time_levels = 0
            
        # Check for multicollinearity if we have multiple predictors
        if len(self.covariates) > 1 and all(pd.api.types.is_numeric_dtype(self.df[cov]) for cov in self.covariates):
            try:
                # Create a smaller df with just the covariates, handling missing values
                cov_df = self.df[self.covariates].dropna()
                
                # Add a constant term for the VIF calculation
                cov_df = sm.add_constant(cov_df)
                
                # Calculate VIF for each covariate
                for i, cov in enumerate(cov_df.columns[1:], 1):  # Skip the constant
                    self.collinearity_results[cov] = variance_inflation_factor(cov_df.values, i)
                    
                # If any VIF > 10, we have strong multicollinearity
                self.has_multicollinearity = any(vif > 10 for vif in self.collinearity_results.values())
            except Exception:
                self.has_multicollinearity = False
        else:
            self.has_multicollinearity = False
            
        # Determine if regression tests are appropriate based on data characteristics
        if self.outcome_is_numeric:
            if (self.group and pd.api.types.is_numeric_dtype(self.df[self.group])) or len(self.covariates) > 0:
                self.regression_appropriate = True
    
    def filter_by_study_design(self, study_type: StudyType) -> SelectionStepResult:
        """
        Filter tests based on compatibility with the study design.
        
        Args:
            study_type: The study design type
            
        Returns:
            SelectionStepResult with the filtered tests
        """
        result = SelectionStepResult()
        result.check_type = SelectionCheck.STUDY_DESIGN_COMPATIBLE
        
        # Get the study design specification from the registry
        design_spec = STUDY_DESIGN_REGISTRY.get(study_type)
        if not design_spec:
            # If no specification is found, don't exclude any tests
            result.passed_tests = set(self.remaining_tests)
            return result
        
        # Get the list of compatible tests for this design
        compatible_tests = set(test.value for test in design_spec.compatible_tests)
        
        # Process each remaining test
        for test in list(self.remaining_tests):
            if test.value in compatible_tests:
                result.add_passed_test(test)
            else:
                result.add_failed_test(test, f"Not compatible with {study_type.display_name} design")
                self.remaining_tests.remove(test)
                self.excluded_tests[test] = f"Not compatible with {study_type.display_name} design"
        
        # Add additional info about the study design
        result.additional_info["study_type"] = study_type
        result.additional_info["design_description"] = design_spec.description
        result.additional_info["required_variables"] = design_spec.get_required_variables()
        result.additional_info["optional_variables"] = design_spec.get_optional_variables()
        
        self.step_results.append(result)
        return result
    
    def filter_by_required_variables(self) -> SelectionStepResult:
        """
        Filter tests based on having the required variables assigned.
        
        Returns:
            SelectionStepResult with the filtered tests
        """
        result = SelectionStepResult()
        result.check_type = SelectionCheck.REQUIRED_VARIABLES_PRESENT
        
        # Check each test for variable requirements
        for test in list(self.remaining_tests):
            missing_vars = self._get_missing_variables(test)
            
            if not missing_vars:
                result.add_passed_test(test)
            else:
                reason = f"Missing required variables: {', '.join(missing_vars)}"
                result.add_failed_test(test, reason)
                self.remaining_tests.remove(test)
                self.excluded_tests[test] = reason
        
        # Add variable assignments to the result for reference
        result.additional_info["assigned_variables"] = {
            "outcome": self.outcome,
            "group": self.group,
            "subject_id": self.subject_id,
            "time": self.time,
            "pair_id": self.pair_id,
            "covariates": self.covariates
        }
        
        self.step_results.append(result)
        return result
    
    def _get_missing_variables(self, test: StatisticalTest) -> List[str]:
        """Get a list of missing required variables for a specific test."""
        missing = []
        
        # One-sample t-test only requires outcome variable
        if test == StatisticalTest.ONE_SAMPLE_T_TEST:
            if not self.outcome:
                missing.append("outcome")
            return missing
            
        # All other tests require outcome
        if not self.outcome:
            missing.append("outcome")
        
        if test in [StatisticalTest.INDEPENDENT_T_TEST, StatisticalTest.MANN_WHITNEY_U_TEST,
                   StatisticalTest.CHI_SQUARE_TEST, StatisticalTest.FISHERS_EXACT_TEST]:
            if not self.group:
                missing.append("group")
            elif self.df is not None and self.group and self.df[self.group].nunique() != 2:
                missing.append("binary group variable (must have exactly 2 unique values)")
        
        elif test in [StatisticalTest.ONE_WAY_ANOVA, StatisticalTest.KRUSKAL_WALLIS_TEST]:
            if not self.group:
                missing.append("group")
            elif self.df is not None and self.group and self.df[self.group].nunique() < 3:
                missing.append("group variable with at least 3 categories")
        
        elif test in [StatisticalTest.PAIRED_T_TEST, StatisticalTest.WILCOXON_SIGNED_RANK_TEST]:
            if not self.subject_id:
                missing.append("subject_id")
            if not self.time:
                missing.append("time")
        
        elif test == StatisticalTest.REPEATED_MEASURES_ANOVA:
            if not self.subject_id:
                missing.append("subject_id")
            if not self.time:
                missing.append("time")
        
        elif test == StatisticalTest.MIXED_ANOVA:
            if not self.group:
                missing.append("group")
            if not self.subject_id:
                missing.append("subject_id")
            if not self.time:
                missing.append("time")
        
        elif test in [StatisticalTest.LINEAR_REGRESSION, StatisticalTest.LOGISTIC_REGRESSION, 
                    StatisticalTest.MULTINOMIAL_LOGISTIC_REGRESSION, StatisticalTest.POISSON_REGRESSION,
                    StatisticalTest.NEGATIVE_BINOMIAL_REGRESSION, StatisticalTest.ORDINAL_REGRESSION]:
            if not (self.group or self.covariates):
                missing.append("group or covariates (at least one predictor variable)")
            
            # Additional check specifically for multinomial logistic regression
            if test == StatisticalTest.MULTINOMIAL_LOGISTIC_REGRESSION:
                # Ensure outcome has more than 2 categories for multinomial
                if self.outcome and self.df is not None and self.outcome in self.df.columns:
                    num_categories = self.df[self.outcome].nunique()
                    if num_categories < 3:
                        missing.append(f"outcome with at least 3 categories (current: {num_categories})")
        
        elif test == StatisticalTest.ANCOVA:
            if not self.group:
                missing.append("group")
            if not self.covariates:
                missing.append("covariates")
        
        elif test == StatisticalTest.LINEAR_MIXED_EFFECTS_MODEL:
            if not self.subject_id:
                missing.append("subject_id")
        
        elif test in [StatisticalTest.PEARSON_CORRELATION, StatisticalTest.SPEARMAN_CORRELATION, 
                    StatisticalTest.KENDALL_TAU_CORRELATION]:
            if not (self.group or self.covariates):
                missing.append("second variable (group or covariate)")

        elif test == StatisticalTest.POINT_BISERIAL_CORRELATION:
            if not self.group:
                missing.append("binary group variable")
            elif self.df is not None and self.group and self.df[self.group].nunique() != 2:
                missing.append("binary group variable (must have exactly 2 unique values)")
        
        elif test == StatisticalTest.SURVIVAL_ANALYSIS:
            if not self.outcome:
                missing.append("time-to-event variable")
            if not self.group:
                missing.append("group variable")
            
            # Check for event indicator
            event_var = next((col for col, role in self.column_roles.items() 
                            if role == VariableRole.EVENT), None) if self.column_roles else None
            if not event_var:
                missing.append("event indicator variable")
        
        return missing
    
    def filter_by_data_type(self) -> SelectionStepResult:
        """
        Filter tests based on compatibility with the outcome data type.
        
        Returns:
            SelectionStepResult with the filtered tests
        """
        result = SelectionStepResult()
        result.check_type = SelectionCheck.DATA_TYPE_COMPATIBLE
        
        # Skip if outcome variable or type is not available
        if not self.outcome or not self.outcome_data_type:
            # If no outcome data type, don't filter
            result.passed_tests = set(self.remaining_tests)
            result.additional_info["outcome_data_type"] = None
            self.step_results.append(result)
            return result
        
        # Get compatible tests for the outcome data type
        compatible_tests = set(test.value for test in get_compatible_tests_for_data_type(self.outcome_data_type))
        
        # Log data type detection information
        logging.info(f"Outcome '{self.outcome}' detected as {self.outcome_data_type.value}")
        logging.info(f"Unique values in outcome: {self.outcome_unique_values}")
        if self.outcome_data_type == CFDataType.CATEGORICAL:
            # Show the first few values for debugging
            sample_values = self.df[self.outcome].dropna().unique()[:5]
            logging.info(f"Sample outcome values: {sample_values}")
            logging.info(f"Compatible tests for categorical data: {compatible_tests}")
        
        # Process each remaining test
        for test in list(self.remaining_tests):
            if test.value in compatible_tests:
                result.add_passed_test(test)
            else:
                reason = f"Not compatible with {self.outcome_data_type.value} outcome data type"
                result.add_failed_test(test, reason)
                self.remaining_tests.remove(test)
                self.excluded_tests[test] = reason
                logging.info(f"Excluded test {test.value}: {reason}")
        
        # Add data type info to the result
        result.additional_info["outcome_data_type"] = self.outcome_data_type
        
        self.step_results.append(result)
        return result
    
    def check_sample_size(self) -> SelectionStepResult:
        """
        Check if the sample size is sufficient for each test.
        This step adds warnings but doesn't exclude tests.
        
        Returns:
            SelectionStepResult with sample size warnings
        """
        result = SelectionStepResult()
        result.check_type = SelectionCheck.SAMPLE_SIZE_SUFFICIENT
        
        # Add all tests as passing (we only add warnings)
        result.passed_tests = set(self.remaining_tests)
        
        for test in self.remaining_tests:
            test_data = TEST_REGISTRY.get(test.value, None)
            if not test_data:
                continue
                
            # Get recommended sample size for this test
            recommended_size = test_data.sample_size
            
            # Compare with actual sample size
            if self.sample_size < recommended_size:
                warning = f"Sample size ({self.sample_size}) below recommended minimum ({recommended_size})"
                result.add_warning_test(test, warning)
                self.warnings[test].append(warning)
                
        # Add sample size info to the result
        result.additional_info["sample_size"] = self.sample_size
        
        self.step_results.append(result)
        return result
    
    def check_data_structure(self) -> SelectionStepResult:
        """
        Check if data structure is appropriate for each test.
        For example, if repeated measures tests require multiple observations per subject.
        
        Returns:
            SelectionStepResult with data structure checks
        """
        result = SelectionStepResult()
        result.check_type = SelectionCheck.DATA_STRUCTURE_VALID
        
        # Specific data structure checks for different test types
        
        # Repeated measures checks
        repeated_measures_tests = [
            StatisticalTest.PAIRED_T_TEST,
            StatisticalTest.WILCOXON_SIGNED_RANK_TEST,
            StatisticalTest.REPEATED_MEASURES_ANOVA,
            StatisticalTest.MIXED_ANOVA,
            StatisticalTest.LINEAR_MIXED_EFFECTS_MODEL
        ]
        
        for test in list(self.remaining_tests):
            # For tests that require repeated measures
            if test in repeated_measures_tests:
                if not self.has_repeated_measures:
                    reason = "Data does not have repeated measurements per subject"
                    result.add_failed_test(test, reason)
                    self.remaining_tests.remove(test)
                    self.excluded_tests[test] = reason
                elif self.min_measurements_per_subject < 2:
                    reason = "Some subjects have fewer than 2 measurements"
                    result.add_failed_test(test, reason)
                    self.remaining_tests.remove(test)
                    self.excluded_tests[test] = reason
                else:
                    result.add_passed_test(test)
            
            # For point-biserial correlation
            elif test == StatisticalTest.POINT_BISERIAL_CORRELATION:
                if self.group and self.df[self.group].nunique() != 2:
                    reason = f"Group variable must have exactly 2 levels, but has {self.df[self.group].nunique()}"
                    result.add_failed_test(test, reason)
                    self.remaining_tests.remove(test)
                    self.excluded_tests[test] = reason
                else:
                    result.add_passed_test(test)
            
            # For chi-square test
            elif test == StatisticalTest.CHI_SQUARE_TEST:
                # Chi-square needs sufficient expected counts
                if self.group and self.outcome and self.smallest_group_size < 5:
                    warning = "Some groups have fewer than 5 observations, which may violate chi-square assumptions"
                    result.add_warning_test(test, warning)
                    self.warnings[test].append(warning)
                    result.add_passed_test(test)
                else:
                    result.add_passed_test(test)
            
            # For ANOVA tests
            elif test in [StatisticalTest.ONE_WAY_ANOVA, StatisticalTest.KRUSKAL_WALLIS_TEST]:
                if self.group and self.group_levels < 2:
                    reason = f"Group variable must have at least 2 levels, but has {self.group_levels}"
                    result.add_failed_test(test, reason)
                    self.remaining_tests.remove(test)
                    self.excluded_tests[test] = reason
                else:
                    result.add_passed_test(test)
            
            # For other tests, mark as passed for this check
            else:
                result.add_passed_test(test)
        
        # Add data structure info to the result
        result.additional_info["has_repeated_measures"] = self.has_repeated_measures
        if self.has_repeated_measures:
            result.additional_info["min_measurements_per_subject"] = self.min_measurements_per_subject
            result.additional_info["max_measurements_per_subject"] = self.max_measurements_per_subject
        
        result.additional_info["group_levels"] = self.group_levels
        result.additional_info["time_levels"] = self.time_levels
        
        self.step_results.append(result)
        return result
    
    def check_data_quality(self) -> SelectionStepResult:
        """
        Check for missing data and other data quality issues.
        This step adds warnings but doesn't exclude tests.
        
        Returns:
            SelectionStepResult with data quality warnings
        """
        result = SelectionStepResult()
        result.check_type = SelectionCheck.MISSING_DATA_ACCEPTABLE
        
        # Add all tests as passing (we only add warnings)
        result.passed_tests = set(self.remaining_tests)
        
        # Check for missing data in key variables
        missing_data_warnings = []
        
        if self.outcome and self.outcome_missing_pct > 0:
            missing_data_warnings.append(f"Outcome variable has {self.outcome_missing_pct:.1f}% missing values")
            
        if self.group and self.group_missing_pct > 0:
            missing_data_warnings.append(f"Group variable has {self.group_missing_pct:.1f}% missing values")
            
        if self.subject_id and self.df[self.subject_id].isna().mean() * 100 > 0:
            missing_data_warnings.append(f"Subject ID variable has {self.df[self.subject_id].isna().mean() * 100:.1f}% missing values")
            
        if self.time and self.df[self.time].isna().mean() * 100 > 0:
            missing_data_warnings.append(f"Time variable has {self.df[self.time].isna().mean() * 100:.1f}% missing values")
        
        # Add warnings to all tests if we found missing data
        if missing_data_warnings:
            for test in self.remaining_tests:
                for warning in missing_data_warnings:
                    result.add_warning_test(test, warning)
                    self.warnings[test].append(warning)
        
        # Check group balance for relevant tests
        group_balance_tests = [
            StatisticalTest.INDEPENDENT_T_TEST, 
            StatisticalTest.ONE_WAY_ANOVA,
            StatisticalTest.MANN_WHITNEY_U_TEST, 
            StatisticalTest.KRUSKAL_WALLIS_TEST,
            StatisticalTest.CHI_SQUARE_TEST,
            StatisticalTest.MIXED_ANOVA
        ]
        
        if self.group and self.group_imbalance_ratio > 3.0:
            for test in self.remaining_tests:
                if test in group_balance_tests:
                    warning = f"Groups are imbalanced (ratio {self.group_imbalance_ratio:.1f}:1), which may affect test power"
                    result.add_warning_test(test, warning)
                    self.warnings[test].append(warning)
        
        # Add data quality info to the result
        result.additional_info["missing_data_warnings"] = missing_data_warnings
        result.additional_info["group_imbalance_ratio"] = self.group_imbalance_ratio
        
        self.step_results.append(result)
        return result
    
    def check_assumptions(self) -> SelectionStepResult:
        """
        Check if statistical assumptions for each test are met.
        Adds warnings but doesn't exclude tests unless assumptions are severely violated.
        
        Returns:
            SelectionStepResult with assumption check results
        """
        result = SelectionStepResult()
        result.check_type = SelectionCheck.ASSUMPTIONS_MET
        
        # List tests that require normality
        normality_dependent_tests = [
            StatisticalTest.ONE_SAMPLE_T_TEST,
            StatisticalTest.INDEPENDENT_T_TEST,
            StatisticalTest.PAIRED_T_TEST,
            StatisticalTest.ONE_WAY_ANOVA,
            StatisticalTest.REPEATED_MEASURES_ANOVA,
            StatisticalTest.MIXED_ANOVA,
            StatisticalTest.ANCOVA,
            StatisticalTest.PEARSON_CORRELATION
        ]
        
        # Homogeneity of variance dependent tests
        homogeneity_dependent_tests = [
            StatisticalTest.INDEPENDENT_T_TEST,
            StatisticalTest.ONE_WAY_ANOVA,
            StatisticalTest.MIXED_ANOVA,
            StatisticalTest.ANCOVA
        ]
        
        # Check each test's assumptions
        for test in list(self.remaining_tests):
            warnings_for_test = []
            should_exclude = False
            exclusion_reason = ""
            
            # Check normality assumption
            if test in normality_dependent_tests:
                if self.outcome in self.normality_results and not self.normality_results[self.outcome].passed:
                    # For smaller samples, normality is more critical
                    if self.sample_size < 30:
                        warnings_for_test.append(
                            f"Normality assumption violated: {self.normality_results[self.outcome].details}. "
                            "Consider non-parametric alternative."
                        )
                    else:
                        warnings_for_test.append(
                            f"Normality assumption violated: {self.normality_results[self.outcome].details}. "
                            f"However, with n={self.sample_size} the central limit theorem may apply."
                        )
            
            # Check homogeneity of variance
            if test in homogeneity_dependent_tests and self.homogeneity_results and not self.homogeneity_results.passed:
                warnings_for_test.append(
                    f"Homogeneity of variance assumption violated: {self.homogeneity_results.details}. "
                    "Consider non-parametric alternative or Welch's correction."
                )
            
            # Check outliers
            if test in normality_dependent_tests and self.outcome in self.outlier_results and not self.outlier_results[self.outcome].passed:
                warnings_for_test.append(
                    f"Outliers detected: {self.outlier_results[self.outcome].details}. "
                    "Consider data transformation or non-parametric tests."
                )
            
            # Check multicollinearity for regression tests
            if test in [StatisticalTest.LINEAR_REGRESSION, StatisticalTest.LOGISTIC_REGRESSION, 
                     StatisticalTest.ANCOVA] and self.has_multicollinearity:
                high_vif_covs = [f"{cov} (VIF={vif:.1f})" for cov, vif in self.collinearity_results.items() 
                               if vif > 10]
                warnings_for_test.append(
                    f"Multicollinearity detected between covariates: {', '.join(high_vif_covs)}. "
                    "May affect coefficient interpretation."
                )
            
            # Special handling for zero variance outcome
            if self.zero_variance_outcome and test in normality_dependent_tests:
                should_exclude = True
                exclusion_reason = "Outcome variable has zero or near-zero variance, making this test invalid"
            
            # Handle test-specific requirements
            if test == StatisticalTest.CHI_SQUARE_TEST and self.group:
                # Check for small expected counts in chi-square
                if self.smallest_group_size < 5:
                    warnings_for_test.append(
                        f"Small expected counts: smallest group has only {self.smallest_group_size} observations. "
                        "Consider Fisher's exact test instead."
                    )
                    
            # Based on the checks, decide whether to pass, warn, or exclude the test
            if should_exclude:
                result.add_failed_test(test, exclusion_reason)
                self.remaining_tests.remove(test)
                self.excluded_tests[test] = exclusion_reason
            else:
                result.add_passed_test(test)
                # Add any warnings that were found
                for warning in warnings_for_test:
                    result.add_warning_test(test, warning)
                    self.warnings[test].append(warning)
        
        # Add assumption check results to additional info
        result.additional_info["normality_results"] = {
            col: {
                "passed": res.passed,
                "details": res.details
            } for col, res in self.normality_results.items()
        }
        
        if self.homogeneity_results:
            result.additional_info["homogeneity_result"] = {
                "passed": self.homogeneity_results.passed,
                "details": self.homogeneity_results.details
            }
        
        result.additional_info["parametric_appropriate"] = self.parametric_appropriate
        
        self.step_results.append(result)
        return result
    
    def run_all_checks(self, study_type: StudyType) -> List[StatisticalTest]:
        """
        Run all selection steps in sequence.
        
        Args:
            study_type: The study design type to check against
            
        Returns:
            List of recommended statistical tests in order of suitability
        """
        # Start with all tests
        self.remaining_tests = set(self.all_tests)
        self.excluded_tests = {}
        self.warnings = {test: [] for test in self.all_tests}
        self.step_results = []
        
        # Run each filter step in sequence
        self.filter_by_study_design(study_type)
        self.filter_by_required_variables()
        self.filter_by_data_type()
        
        # Run check steps
        self.check_sample_size()
        self.check_data_structure()
        self.check_assumptions()  # New step for checking statistical assumptions
        self.check_data_quality()
        
        # Return the remaining tests, ranked by suitability
        return self.rank_remaining_tests()
    
    def rank_remaining_tests(self) -> List[StatisticalTest]:
        """
        Rank the remaining tests by suitability.
        
        Returns:
            List of tests ordered by suitability (most suitable first)
        """
        tests_with_scores = []
        
        # List tests that require normality (duplicated from check_assumptions)
        normality_dependent_tests = [
            StatisticalTest.ONE_SAMPLE_T_TEST,
            StatisticalTest.INDEPENDENT_T_TEST,
            StatisticalTest.PAIRED_T_TEST,
            StatisticalTest.ONE_WAY_ANOVA,
            StatisticalTest.REPEATED_MEASURES_ANOVA,
            StatisticalTest.MIXED_ANOVA,
            StatisticalTest.ANCOVA,
            StatisticalTest.PEARSON_CORRELATION
        ]
        
        for test in self.remaining_tests:
            # Start with a perfect score
            score = 100
            
            # Deduct points for each warning
            score -= len(self.warnings[test]) * 5  # Reduced penalty per warning
            
            # Special handling for single sample designs
            if self.single_sample_test_appropriate and test == StatisticalTest.ONE_SAMPLE_T_TEST:
                score += 30
            
            # Special handling for paired/repeated measures designs
            if self.paired_test_appropriate:
                if test == StatisticalTest.PAIRED_T_TEST and self.parametric_appropriate:
                    score += 30
                elif test == StatisticalTest.WILCOXON_SIGNED_RANK_TEST and not self.parametric_appropriate:
                    score += 30
            
            # Special handling for repeated measures with >2 time points
            if self.has_repeated_measures and self.time_levels > 2:
                if test == StatisticalTest.REPEATED_MEASURES_ANOVA and self.parametric_appropriate:
                    score += 25
                elif test == StatisticalTest.LINEAR_MIXED_EFFECTS_MODEL:
                    score += 20  # Slightly less boost as it's more complex
            
            # Special handling for between-subjects designs
            if self.group and not self.has_repeated_measures:
                if self.group_levels == 2:
                    # Binary group
                    if test == StatisticalTest.INDEPENDENT_T_TEST and self.parametric_appropriate:
                        score += 25
                    elif test == StatisticalTest.MANN_WHITNEY_U_TEST and not self.parametric_appropriate:
                        score += 25
                elif self.group_levels > 2:
                    # Multiple groups
                    if test == StatisticalTest.ONE_WAY_ANOVA and self.parametric_appropriate:
                        score += 25
                    elif test == StatisticalTest.KRUSKAL_WALLIS_TEST and not self.parametric_appropriate:
                        score += 25
            
            # Boost for categorical outcome variables
            if self.outcome_data_type == CFDataType.CATEGORICAL:
                if test in [StatisticalTest.CHI_SQUARE_TEST, StatisticalTest.FISHERS_EXACT_TEST]:
                    score += 20
                if test == StatisticalTest.LOGISTIC_REGRESSION and len(self.covariates) > 0:
                    score += 15
                # Add boost for multinomial logistic regression when outcome has multiple categories
                if test == StatisticalTest.MULTINOMIAL_LOGISTIC_REGRESSION:
                    # Check if outcome has more than 2 categories
                    if self.outcome and self.outcome_unique_values > 2:
                        score += 30  # Significant boost for multinomial with multi-category outcome
                    # General boost for categorical data
                    score += 20
            
            # Boost for count data
            if self.outcome_data_type == CFDataType.COUNT:
                if test in [StatisticalTest.POISSON_REGRESSION, StatisticalTest.NEGATIVE_BINOMIAL_REGRESSION]:
                    score += 20
            
            # Boost for ordinal data
            if self.outcome_data_type == CFDataType.ORDINAL:
                if test == StatisticalTest.ORDINAL_REGRESSION:
                    score += 20
                elif test in [StatisticalTest.SPEARMAN_CORRELATION, StatisticalTest.KENDALL_TAU_CORRELATION]:
                    score += 15
            
            # Boost regressions when we have covariates
            if len(self.covariates) > 0:
                if test in [StatisticalTest.LINEAR_REGRESSION, StatisticalTest.LOGISTIC_REGRESSION, 
                           StatisticalTest.POISSON_REGRESSION]:
                    score += 15
                if test == StatisticalTest.ANCOVA and self.group:
                    score += 20
            
            # Mixed design boost
            if self.has_repeated_measures and self.group:
                if test == StatisticalTest.MIXED_ANOVA and self.parametric_appropriate:
                    score += 25
                elif test == StatisticalTest.LINEAR_MIXED_EFFECTS_MODEL:
                    score += 20
            
            # Adjust for complexity (with less penalty than before)
            if test in [StatisticalTest.LINEAR_MIXED_EFFECTS_MODEL, 
                       StatisticalTest.MULTINOMIAL_LOGISTIC_REGRESSION,
                       StatisticalTest.ORDINAL_REGRESSION]:
                # More complex tests
                score -= 10
            
            if test in [StatisticalTest.ONE_WAY_ANOVA, StatisticalTest.ANCOVA, 
                       StatisticalTest.REPEATED_MEASURES_ANOVA, StatisticalTest.MIXED_ANOVA]:
                # Moderate complexity
                score -= 5
            
            # When outcome is not continuous and parametric is not appropriate,
            # boost non-parametric tests and reduce parametric ones
            if not self.parametric_appropriate:
                if test in [StatisticalTest.MANN_WHITNEY_U_TEST, StatisticalTest.WILCOXON_SIGNED_RANK_TEST,
                           StatisticalTest.KRUSKAL_WALLIS_TEST, StatisticalTest.SPEARMAN_CORRELATION,
                           StatisticalTest.KENDALL_TAU_CORRELATION]:
                    # Boost non-parametric tests for non-parametric situations
                    score += 15
                elif test in normality_dependent_tests:
                    # Reduce parametric tests when parametric not appropriate
                    score -= 15
            
            tests_with_scores.append((test, score))
        
        # Sort by score (descending)
        tests_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the tests in sorted order
        return [test for test, _ in tests_with_scores]
    
    def get_test_recommendations(self, top_n: int = 3) -> Dict:
        """
        Get detailed recommendations for the top N tests.
        
        Args:
            top_n: Number of top tests to include
            
        Returns:
            Dictionary with detailed recommendation information
        """
        ranked_tests = self.rank_remaining_tests()
        top_tests = ranked_tests[:min(top_n, len(ranked_tests))]
        
        recommendations = {
            "top_recommendations": [],
            "alternative_tests": [],
            "excluded_tests": [],
            "selection_process": [],
            "data_characteristics": {},
            "assumption_summary": {}
        }
        
        # Add data characteristics for LLM use
        recommendations["data_characteristics"] = {
            "sample_size": self.sample_size,
            "outcome_variable": self.outcome,
            "outcome_data_type": str(self.outcome_data_type.value) if self.outcome_data_type else None,
            "group_variable": self.group,
            "group_levels": self.group_levels if self.group else 0,
            "subject_id_variable": self.subject_id,
            "time_variable": self.time,
            "time_levels": self.time_levels if self.time else 0,
            "covariates": self.covariates,
            "has_repeated_measures": self.has_repeated_measures,
            "parametric_appropriate": self.parametric_appropriate,
            "single_sample_appropriate": self.single_sample_test_appropriate,
            "paired_test_appropriate": self.paired_test_appropriate,
            "regression_appropriate": self.regression_appropriate
        }
        
        # Add assumption summary
        normality_summary = {}
        for var, result in self.normality_results.items():
            normality_summary[var] = {
                "passed": result.passed,
                "test_name": result.test_name,
                "p_value": result.p_value,
                "details": result.details
            }
            
        homogeneity_summary = None
        if self.homogeneity_results:
            homogeneity_summary = {
                "passed": self.homogeneity_results.passed,
                "test_name": self.homogeneity_results.test_name,
                "p_value": self.homogeneity_results.p_value,
                "details": self.homogeneity_results.details
            }
            
        recommendations["assumption_summary"] = {
            "normality": normality_summary,
            "homogeneity_of_variance": homogeneity_summary,
            "multicollinearity": {
                "has_multicollinearity": self.has_multicollinearity if hasattr(self, 'has_multicollinearity') else False,
                "vif_values": self.collinearity_results if hasattr(self, 'collinearity_results') else {}
            }
        }
        
        # Add top recommendations with details
        for test in top_tests:
            test_data = TEST_REGISTRY.get(test.value)
            if not test_data:
                continue
                
            # Generate LLM-friendly explanation of why this test was selected
            explanation = self._generate_test_explanation(test)
            
            recommendations["top_recommendations"].append({
                "test": test.value,
                "name": test_data.name,
                "description": test_data.description,
                "warnings": self.warnings[test],
                "assumptions": test_data.assumptions,
                "explanation": explanation,
                "example_hypothesis": test_data.example_hypothesis if hasattr(test_data, 'example_hypothesis') else None
            })
        
        # Add remaining tests as alternatives
        for test in ranked_tests[len(top_tests):]:
            test_data = TEST_REGISTRY.get(test.value)
            if not test_data:
                continue
                
            recommendations["alternative_tests"].append({
                "test": test.value,
                "name": test_data.name,
                "warnings": self.warnings[test]
            })
        
        # Add excluded tests with reasons
        for test, reason in self.excluded_tests.items():
            test_data = TEST_REGISTRY.get(test.value)
            if not test_data:
                continue
                
            recommendations["excluded_tests"].append({
                "test": test.value,
                "name": test_data.name,
                "reason": reason
            })
        
        # Add step-by-step selection process
        for i, step in enumerate(self.step_results):
            step_info = {
                "step": i + 1,
                "check_type": step.check_type.value if step.check_type else "unknown",
                "tests_passed": len(step.passed_tests),
                "tests_failed": len(step.failed_tests),
                "additional_info": step.additional_info
            }
            recommendations["selection_process"].append(step_info)
        
        return recommendations
    
    def _generate_test_explanation(self, test: StatisticalTest) -> str:
        """Generate an explanation of why this test was selected."""
        explanation_parts = []
        
        # Basic test descriptions based on test type
        if test == StatisticalTest.ONE_SAMPLE_T_TEST:
            explanation_parts.append(
                f"One-sample t-test is appropriate because you have a single outcome variable ('{self.outcome}')"
                " without comparison groups."
            )
        
        elif test == StatisticalTest.INDEPENDENT_T_TEST:
            explanation_parts.append(
                f"Independent samples t-test is appropriate because you're comparing a continuous outcome ('{self.outcome}')"
                f" between two independent groups ('{self.group}')."
            )
        
        elif test == StatisticalTest.PAIRED_T_TEST:
            explanation_parts.append(
                f"Paired samples t-test is appropriate because you're comparing measurements of '{self.outcome}'"
                f" at two time points within the same subjects (identified by '{self.subject_id}')."
            )
        
        elif test == StatisticalTest.ONE_WAY_ANOVA:
            explanation_parts.append(
                f"One-way ANOVA is appropriate because you're comparing a continuous outcome ('{self.outcome}')"
                f" across {self.group_levels} groups ('{self.group}')."
            )
        
        elif test == StatisticalTest.REPEATED_MEASURES_ANOVA:
            explanation_parts.append(
                f"Repeated measures ANOVA is appropriate because you're comparing '{self.outcome}'"
                f" across {self.time_levels} time points within the same subjects ('{self.subject_id}')."
            )
        
        elif test == StatisticalTest.MIXED_ANOVA:
            explanation_parts.append(
                f"Mixed ANOVA is appropriate because you're comparing '{self.outcome}'"
                f" across {self.time_levels} time points within subjects ('{self.subject_id}')"
                f" and between {self.group_levels} groups ('{self.group}')."
            )
        
        elif test in [StatisticalTest.MANN_WHITNEY_U_TEST, StatisticalTest.WILCOXON_SIGNED_RANK_TEST, 
                     StatisticalTest.KRUSKAL_WALLIS_TEST]:
            explanation_parts.append(
                f"Non-parametric test recommended because {'the data appears to violate normality assumptions' if not self.parametric_appropriate else 'of the data characteristics'}."
            )
            
        elif test == StatisticalTest.LINEAR_REGRESSION:
            if self.group and len(self.covariates) == 0:
                explanation_parts.append(
                    f"Linear regression is appropriate for modeling the relationship between '{self.outcome}'"
                    f" and '{self.group}' as a predictor."
                )
            elif len(self.covariates) > 0:
                explanation_parts.append(
                    f"Linear regression is appropriate for modeling '{self.outcome}' using"
                    f" {len(self.covariates)} predictor variables."
                )
        
        elif test == StatisticalTest.LOGISTIC_REGRESSION:
            explanation_parts.append(
                f"Logistic regression is appropriate because '{self.outcome}' is a binary outcome"
                f" with {len(self.covariates) + (1 if self.group else 0)} predictor variables."
            )
            
        elif test == StatisticalTest.MULTINOMIAL_LOGISTIC_REGRESSION:
            num_categories = self.df[self.outcome].nunique() if self.outcome in self.df.columns else "multiple"
            predictors_count = len(self.covariates) + (1 if self.group else 0)
            explanation_parts.append(
                f"Multinomial logistic regression is appropriate because '{self.outcome}' has {num_categories} categories"
                f" (such as 'high', 'medium', 'low') and you have {predictors_count} predictor variable(s)."
            )
            
        elif test == StatisticalTest.ANCOVA:
            explanation_parts.append(
                f"ANCOVA is appropriate because you're comparing '{self.outcome}' across groups ('{self.group}')"
                f" while controlling for {len(self.covariates)} covariates."
            )
        
        elif test == StatisticalTest.CHI_SQUARE_TEST:
            explanation_parts.append(
                f"Chi-square test is appropriate because both '{self.outcome}' and '{self.group}'"
                " are categorical variables."
            )
        
        elif test == StatisticalTest.LINEAR_MIXED_EFFECTS_MODEL:
            explanation_parts.append(
                f"Linear mixed effects model is appropriate because your data has a hierarchical structure"
                f" with repeated measurements within subjects ('{self.subject_id}')."
            )
            
        # Add information about sample size
        if self.sample_size < 30:
            explanation_parts.append(f"Note that your sample size (n={self.sample_size}) is relatively small.")
        
        # Add assumption information if relevant
        if test in [StatisticalTest.INDEPENDENT_T_TEST, StatisticalTest.PAIRED_T_TEST, 
                   StatisticalTest.ONE_WAY_ANOVA, StatisticalTest.REPEATED_MEASURES_ANOVA]:
            if self.outcome in self.normality_results:
                if self.normality_results[self.outcome].passed:
                    explanation_parts.append(
                        f"Normality tests indicate that '{self.outcome}' appears to be normally distributed."
                    )
                else:
                    if self.sample_size >= 30:
                        explanation_parts.append(
                            f"While '{self.outcome}' may not be perfectly normally distributed,"
                            f" your sample size (n={self.sample_size}) is sufficient for the central limit theorem to apply."
                        )
        
        return " ".join(explanation_parts)