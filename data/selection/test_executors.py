from abc import ABC, abstractmethod
from typing import Dict, Type, Callable, List, Optional, Any
import pandas as pd
import numpy as np

from study_model.study_model import StatisticalTest

class TestExecutor(ABC):
    """Base class for test executors that handle different statistical tests."""
    
    @abstractmethod
    def execute(self, df, outcome, group, covariates, subject_id, time, test_function, **kwargs):
        """Execute the statistical test and return results."""
        pass

class IndependentTTestExecutor(TestExecutor):
    """Executor for independent t-test."""
    
    def execute(self, df, outcome, group, covariates, subject_id, time, test_function, **kwargs):
        # Validate group has exactly 2 levels for t-test
        group_values = df[group].unique()
        if len(group_values) != 2:
            raise ValueError("Independent t-test requires exactly 2 groups")
        
        # Extract the data for each group
        group1_data = df[df[group] == group_values[0]][outcome]
        group2_data = df[df[group] == group_values[1]][outcome]
        
        # Run the test
        return test_function(group1_data, group2_data, alpha=0.05)

class OneSampleTTestExecutor(TestExecutor):
    """Executor for one-sample t-test."""
    
    def execute(self, df, outcome, group, covariates, subject_id, time, test_function, **kwargs):
        # Extract the sample data
        sample_data = df[outcome]
        
        print("One sample t-test executor")
        print(f"\n\nkwargs: {kwargs}\n\n")

        # Get population mean to test against (required parameter)
        if 'mu' not in kwargs:
            raise ValueError("One-sample t-test requires 'mu' parameter (population mean to test against)")
        
        # Get the mu value explicitly to avoid any reference issues
        mu = float(kwargs.get('mu'))
        
        # Get optional parameters
        alpha = kwargs.get('alpha', 0.05)
        alternative = kwargs.get('alternative', 'two-sided')
        
        # Call the test function with explicit parameters
        result = test_function(
            data=sample_data, 
            mu=mu,  # Pass mu explicitly
            alpha=alpha,
            alternative=alternative
        )
        
        # Add mu to the result for verification
        if result and isinstance(result, dict):
            result['mu_used'] = mu
            
        return result

class OneWayAnovaExecutor(TestExecutor):
    """Executor for one-way ANOVA."""
    
    def execute(self, df, outcome, group, covariates, subject_id, time, test_function, **kwargs):
        # Group data by the group variable
        groups_list = [df[df[group] == val][outcome] for val in df[group].unique()]
        
        # Run the test
        return test_function(groups_list, alpha=0.05)

class PairedTTestExecutor(TestExecutor):
    """Executor for paired t-test."""
    
    def execute(self, df, outcome, group, covariates, subject_id, time, test_function, **kwargs):
        # Validate time has exactly 2 levels for paired t-test
        time_values = df[time].unique()
        if len(time_values) != 2:
            raise ValueError("Paired t-test requires exactly 2 time points")
        
        # Reshape the data to wide format for paired test
        df_wide = df.pivot(index=subject_id, columns=time, values=outcome)
        
        # Extract the paired data
        time1_data = df_wide[time_values[0]]
        time2_data = df_wide[time_values[1]]
        
        # Run the test
        return test_function(time1_data, time2_data, alpha=0.05)

class WilcoxonSignedRankTestExecutor(TestExecutor):
    """Executor for Wilcoxon signed-rank test."""
    
    def execute(self, df, outcome, group, covariates, subject_id, time, test_function, **kwargs):
        # Validate time has exactly 2 levels for Wilcoxon test
        time_values = df[time].unique()
        if len(time_values) != 2:
            raise ValueError("Wilcoxon signed-rank test requires exactly 2 time points")
        
        # Reshape the data to wide format for paired test
        df_wide = df.pivot(index=subject_id, columns=time, values=outcome)
        
        # Extract the paired data
        time1_data = df_wide[time_values[0]]
        time2_data = df_wide[time_values[1]]
        
        # Run the test
        return test_function(time1_data, time2_data, alpha=0.05)

class MannWhitneyUTestExecutor(TestExecutor):
    """Executor for Mann-Whitney U test."""
    
    def execute(self, df, outcome, group, covariates, subject_id, time, test_function, **kwargs):
        # Validate group has exactly 2 levels for Mann-Whitney U
        group_values = df[group].unique()
        if len(group_values) != 2:
            raise ValueError("Mann-Whitney U test requires exactly 2 groups")
        
        # Extract the data for each group
        group1_data = df[df[group] == group_values[0]][outcome]
        group2_data = df[df[group] == group_values[1]][outcome]
        
        # Run the test
        return test_function(group1_data, group2_data, alpha=0.05)

class KruskalWallisTestExecutor(TestExecutor):
    """Executor for Kruskal-Wallis test."""
    
    def execute(self, df, outcome, group, covariates, subject_id, time, test_function, **kwargs):
        # Group data by the group variable
        groups_list = [df[df[group] == val][outcome] for val in df[group].unique()]
        
        # Run the test
        return test_function(groups_list, alpha=0.05)

class ChiSquareTestExecutor(TestExecutor):
    """Executor for Chi-square test."""
    
    def execute(self, df, outcome, group, covariates, subject_id, time, test_function, **kwargs):
        # Create contingency table
        contingency_table = pd.crosstab(df[group], df[outcome])
        
        # Run the test
        return test_function(contingency_table)

class FishersExactTestExecutor(TestExecutor):
    """Executor for Fisher's exact test."""
    
    def execute(self, df, outcome, group, covariates, subject_id, time, test_function, **kwargs):
        # Create contingency table
        contingency_table = pd.crosstab(df[group], df[outcome])
        
        # Run the test
        return test_function(contingency_table, alpha=0.05)

class RepeatedMeasuresAnovaExecutor(TestExecutor):
    """Executor for repeated measures ANOVA."""
    
    def execute(self, df, outcome, group, covariates, subject_id, time, test_function, **kwargs):
        # Run the test
        return test_function(df, subject_id=subject_id, within_factors=[time], alpha=0.05, outcome=outcome)

class MixedAnovaExecutor(TestExecutor):
    """Executor for mixed ANOVA."""
    
    def execute(self, df, outcome, group, covariates, subject_id, time, test_function, **kwargs):
        # Run the test
        return test_function(df, between_factors=[group], within_factors=[time], 
                           subject_id=subject_id, alpha=0.05, outcome=outcome)

class LinearMixedEffectsModelExecutor(TestExecutor):
    """Executor for linear mixed effects model."""
    
    def execute(self, df, outcome, group, covariates, subject_id, time, test_function, **kwargs):
        # Build fixed effects list (group, time, and covariates)
        fixed_effects = [var for var in [group, time] if var is not None]
        fixed_effects.extend(covariates)
        
        # Run the test
        return test_function(df, fixed_effects=fixed_effects, 
                           random_effects=[subject_id], alpha=0.05, outcome=outcome)

class LinearRegressionExecutor(TestExecutor):
    """Executor for linear regression."""
    
    def execute(self, df, outcome, group, covariates, subject_id, time, test_function, **kwargs):
        # Build predictors list
        predictors = [group] if group is not None else []
        if time is not None:
            predictors.append(time)
        predictors.extend(covariates)
        
        # Run the test
        return test_function(df, outcome, predictors, alpha=0.05)

class LogisticRegressionExecutor(TestExecutor):
    """Executor for logistic regression."""
    
    def execute(self, df, outcome, group, covariates, subject_id, time, test_function, **kwargs):
        # Build predictors list
        predictors = [group] if group is not None else []
        if time is not None:
            predictors.append(time)
        predictors.extend(covariates)
        
        # Run the test
        return test_function(df, outcome, predictors, alpha=0.05)

class AncovaExecutor(TestExecutor):
    """Executor for ANCOVA."""
    
    def execute(self, df, outcome, group, covariates, subject_id, time, test_function, **kwargs):
        # Validate we have covariates for ANCOVA
        if not covariates:
            raise ValueError("ANCOVA requires at least one covariate")
        
        # Run the test
        return test_function(df, outcome, group, covariates, alpha=0.05)

class PearsonCorrelationExecutor(TestExecutor):
    """Executor for Pearson correlation."""
    
    def execute(self, df, outcome, group, covariates, subject_id, time, test_function, **kwargs):
        # We need two variables to correlate - using the first covariate if available
        if not covariates:
            raise ValueError("Correlation requires two variables (outcome and at least one covariate)")
        
        # Run the test
        return test_function(df[outcome], df[covariates[0]], alpha=0.05)

class SpearmanCorrelationExecutor(TestExecutor):
    """Executor for Spearman correlation."""
    
    def execute(self, df, outcome, group, covariates, subject_id, time, test_function, **kwargs):
        # We need two variables to correlate - using the first covariate if available
        if not covariates:
            raise ValueError("Correlation requires two variables (outcome and at least one covariate)")
        
        # Run the test
        return test_function(df[outcome], df[covariates[0]], alpha=0.05)

class KendallTauCorrelationExecutor(TestExecutor):
    """Executor for Kendall Tau correlation."""
    
    def execute(self, df, outcome, group, covariates, subject_id, time, test_function, **kwargs):
        # We need two variables to correlate - using the first covariate if available
        if not covariates:
            raise ValueError("Correlation requires two variables (outcome and at least one covariate)")
        
        # Run the test
        return test_function(df[outcome], df[covariates[0]], alpha=0.05)

class PointBiserialCorrelationExecutor(TestExecutor):
    """Executor for Point-Biserial correlation."""
    
    def execute(self, df, outcome, group, covariates, subject_id, time, test_function, **kwargs):
        # We need a binary variable and a continuous variable
        binary_var = group
        
        # Validate the binary variable
        if binary_var is None:
            raise ValueError("Point-Biserial correlation requires a binary group variable")
            
        # Verify that the group variable is actually binary
        if binary_var in df.columns:
            if df[binary_var].nunique() != 2:
                raise ValueError(f"Group variable must have exactly 2 unique values, found {df[binary_var].nunique()}")
        
        # Run the test - Group (binary) should be first parameter, outcome (continuous) should be second
        return test_function(df[binary_var], df[outcome], alpha=0.05)

class OrdinalRegressionExecutor(TestExecutor):
    """Executor for ordinal regression."""
    
    def execute(self, df, outcome, group, covariates, subject_id, time, test_function, **kwargs):
        # Build predictors list
        predictors = [group] if group is not None else []
        if time is not None:
            predictors.append(time)
        predictors.extend(covariates)
        
        # Run the test
        return test_function(df, outcome, predictors, alpha=0.05)

class PoissonRegressionExecutor(TestExecutor):
    """Executor for Poisson regression."""
    
    def execute(self, df, outcome, group, covariates, subject_id, time, test_function, **kwargs):
        # Build predictors list
        predictors = [group] if group is not None else []
        if time is not None:
            predictors.append(time)
        predictors.extend(covariates)
        
        # Run the test
        return test_function(df, outcome, predictors, alpha=0.05)

class NegativeBinomialRegressionExecutor(TestExecutor):
    """Executor for negative binomial regression."""
    
    def execute(self, df, outcome, group, covariates, subject_id, time, test_function, **kwargs):
        # Build predictors list
        predictors = [group] if group is not None else []
        if time is not None:
            predictors.append(time)
        predictors.extend(covariates)
        
        # Run the test
        return test_function(df, outcome, predictors, alpha=0.05)

class MultinomialLogisticRegressionExecutor(TestExecutor):
    """Executor for multinomial logistic regression."""
    
    def execute(self, df, outcome, group, covariates, subject_id, time, test_function, **kwargs):
        # Build predictors list
        predictors = [group] if group is not None else []
        if time is not None:
            predictors.append(time)
        predictors.extend(covariates)
        
        # Run the test
        return test_function(df, outcome, predictors, alpha=0.05)

class SurvivalAnalysisExecutor(TestExecutor):
    """Executor for survival analysis."""
    
    def execute(self, df, outcome, group, covariates, subject_id, time, test_function, **kwargs):
        # Find event indicator - use designated event role if available
        event_indicator = kwargs.get('event', None)
        
        # If no event variable was provided through the event role, fall back to the previous logic
        if not event_indicator:
            # First try covariates
            if covariates:
                event_indicator = covariates[0]
            else:
                # Then look for a column with event or status in the name
                for col in df.columns:
                    if col != outcome and (col.lower().find('event') >= 0 or col.lower().find('status') >= 0):
                        if df[col].nunique() == 2:
                            event_indicator = col
                            break
        
        if not event_indicator:
            raise ValueError("Survival analysis requires an event indicator variable")
            
        if not group:
            raise ValueError("Survival analysis requires a group variable for comparison")
        
        # Run the test
        return test_function(df, time_variable=outcome, event_variable=event_indicator, 
                             group_variable=group, alpha=0.05)

class TestExecutorFactory:
    """Factory class to create appropriate test executors."""
    
    # Registry mapping test keys to executor classes
    _registry: Dict[str, Type[TestExecutor]] = {
        "independent_t_test": IndependentTTestExecutor,
        "one_way_anova": OneWayAnovaExecutor,
        "one_sample_t_test": OneSampleTTestExecutor,
        "paired_t_test": PairedTTestExecutor,
        "wilcoxon_signed_rank_test": WilcoxonSignedRankTestExecutor,
        "mann_whitney_u_test": MannWhitneyUTestExecutor,
        "kruskal_wallis_test": KruskalWallisTestExecutor,
        "chi_square_test": ChiSquareTestExecutor,
        "fishers_exact_test": FishersExactTestExecutor,
        "repeated_measures_anova": RepeatedMeasuresAnovaExecutor,
        "mixed_anova": MixedAnovaExecutor,
        "linear_mixed_effects_model": LinearMixedEffectsModelExecutor,
        "linear_regression": LinearRegressionExecutor,
        "logistic_regression": LogisticRegressionExecutor,
        "ancova": AncovaExecutor,
        "pearson_correlation": PearsonCorrelationExecutor,
        "spearman_correlation": SpearmanCorrelationExecutor,
        "kendall_tau_correlation": KendallTauCorrelationExecutor,
        "point_biserial_correlation": PointBiserialCorrelationExecutor,
        "ordinal_regression": OrdinalRegressionExecutor,
        "poisson_regression": PoissonRegressionExecutor,
        "negative_binomial_regression": NegativeBinomialRegressionExecutor,
        "multinomial_logistic_regression": MultinomialLogisticRegressionExecutor,
        "survival_analysis": SurvivalAnalysisExecutor
    }
    
    @classmethod
    def get_executor(cls, test_key: str) -> Optional[TestExecutor]:
        """Get the appropriate executor for a test key."""
        executor_class = cls._registry.get(test_key)
        if executor_class:
            return executor_class()
        return None
    
    @classmethod
    def register_executor(cls, test_key: str, executor_class: Type[TestExecutor]):
        """Register a new executor class for a test key."""
        cls._registry[test_key] = executor_class
        
    @classmethod
    def get_required_parameters(cls, test_key: str) -> Dict[str, str]:
        """Get the required parameters for a test."""
        if test_key == "one_sample_t_test":
            return {"mu": "Population mean to test against"}
        # Add other tests with required parameters here
        return {}
