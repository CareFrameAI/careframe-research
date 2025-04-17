import traceback
import pandas as pd
from scipy import stats
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from lifelines import KaplanMeierFitter, CoxPHFitter
from data.selection.detailed_tests.fishers_exact_test import fishers_exact_test
from data.selection.detailed_tests.independent_t_test import independent_t_test
from data.selection.detailed_tests.kendall_tau_correlation import kendall_tau_correlation
from data.selection.detailed_tests.ancova import ancova
from data.selection.detailed_tests.chi_square_test import chi_square_test
from data.selection.detailed_tests.kruskal_wallis_test import kruskal_wallis_test
from data.selection.detailed_tests.linear_mixed_effects_model import linear_mixed_effects_model
from data.selection.detailed_tests.linear_regression import linear_regression
from data.selection.detailed_tests.mann_whitney_u_test import mann_whitney_u_test
from data.selection.detailed_tests.mixed_anova import mixed_anova
from data.selection.detailed_tests.multinomial_logistic_regression import multinomial_logistic_regression
from data.selection.detailed_tests.negative_binomial_regression import negative_binomial_regression
from data.selection.detailed_tests.logistic_regression import logistic_regression
from data.selection.detailed_tests.one_sample_t_test import one_sample_t_test
from data.selection.detailed_tests.one_way_anova import one_way_anova
from data.selection.detailed_tests.ordinal_regression import ordinal_regression
from data.selection.detailed_tests.paired_t_test import paired_t_test
from data.selection.detailed_tests.pearson_correlation import pearson_correlation
from data.selection.detailed_tests.point_biserial_correlation import point_biserial_correlation
from data.selection.detailed_tests.poisson_regression import poisson_regression
from data.selection.detailed_tests.repeated_measures_anova import repeated_measures_anova
from data.selection.detailed_tests.spearman_correlation import spearman_correlation
from data.selection.detailed_tests.survival_analysis import survival_analysis
from data.selection.detailed_tests.wilcoxon_signed_rank_test import wilcoxon_signed_rank_test
from study_model.study_model import StatisticalTest
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
from data.assumptions.tests import (
    OutlierTest,
    ResidualNormalityTest, 
    HomoscedasticityTest,
    MulticollinearityTest, 
    LinearityTest,
    AutocorrelationTest,
    NormalityTest,
    HomogeneityOfVarianceTest,
    HomogeneityOfRegressionSlopesTest,
    OverdispersionTest,
    IndependenceTest,
    GoodnessOfFitTest,
    ProportionalOddsTest,
    SampleSizeTest
)

def _interpret_cohens_d(d: float) -> str:
    """Helper function to interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

def shapiro_wilk_test(data: pd.Series, alpha: float) -> Dict[str, Any]:
    """Performs the Shapiro-Wilk test for normality."""
    try:
        statistic, p_value = stats.shapiro(data)
        return {
            'test': 'Shapiro-Wilk',
            'statistic': statistic,
            'p_value': p_value,  # Consistent key
            'satisfied': p_value > alpha
        }
    except ValueError as e:
        return {
            'test': 'Shapiro-Wilk',
            'statistic': None,
            'p_value': None,  # Consistent key
            'satisfied': False,
            'reason': str(e)
        }

def levene_test(groups: List[pd.Series], alpha: float) -> Dict[str, Any]:
    """Performs Levene's test for homogeneity of variances."""
    try:
        # Levene's test requires at least two groups.
        if len(groups) < 2:
            return {
                'test': "Levene's",
                'statistic': None,
                'p_value': None,  # Consistent key
                'satisfied': False,
                'reason': 'Less than two groups provided.'
            }
        # Convert list of Series to a list of arrays for scipy.stats.levene
        groups_arrays = [group.values for group in groups]
        statistic, p_value = stats.levene(*groups_arrays)
        return {
            'test': "Levene's",
            'statistic': statistic,
            'p_value': p_value,  # Consistent key
            'satisfied': p_value > alpha
        }
    except ValueError as e:
        return {
            'test': "Levene's",
            'statistic': None,
            'p_value': None,  # Consistent key
            'satisfied': False,
            'reason': str(e)
        }


import pandas as pd
from scipy import stats
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from lifelines import KaplanMeierFitter, CoxPHFitter
from study_model.study_model import StatisticalTest

# Define a TestData class to store metadata about each test
@dataclass
class TestData:
    """Class for storing metadata about statistical tests."""
    name: str
    test_function: Callable
    description: str
    assumptions: List[str]
    required_data_type: str
    sample_size: int = 30
    
TEST_REGISTRY = {
    StatisticalTest.ONE_SAMPLE_T_TEST.value: TestData(
        name='One Sample T-Test',
        test_function=one_sample_t_test,
        sample_size=30,
        description='Tests if a sample mean differs from a population mean',
        assumptions=['normality', 'independence'],
        required_data_type='continuous'
    ),
    StatisticalTest.INDEPENDENT_T_TEST.value: TestData(
        name='Independent T-Test', 
        test_function=independent_t_test,
        sample_size=30,
        description='Compares means between two independent groups',
        assumptions=['normality', 'homogeneity_of_variance', 'independence'],
        required_data_type='continuous'
    ),
    StatisticalTest.PAIRED_T_TEST.value: TestData(
        name='Paired T-Test',
        test_function=paired_t_test,
        sample_size=30,
        description='Compares means between paired measurements',
        assumptions=['normality', 'independence'],
        required_data_type='continuous'
    ),
    StatisticalTest.MANN_WHITNEY_U_TEST.value: TestData(
        name='Mann-Whitney U Test',
        test_function=mann_whitney_u_test,
        sample_size=30,
        description='Non-parametric alternative to independent t-test',
        assumptions=['independence'],
        required_data_type='ordinal_or_continuous'
    ),
    StatisticalTest.WILCOXON_SIGNED_RANK_TEST.value: TestData(
        name='Wilcoxon Signed-Rank Test',
        test_function=wilcoxon_signed_rank_test,
        sample_size=30,
        description='Non-parametric alternative to paired t-test',
        assumptions=['independence'],
        required_data_type='ordinal_or_continuous'
    ),
    StatisticalTest.PEARSON_CORRELATION.value: TestData(
        name='Pearson Correlation',
        test_function=pearson_correlation,
        sample_size=30,
        description='Tests linear correlation between two variables',
        assumptions=['normality', 'linearity'],
        required_data_type='continuous'
    ),
    StatisticalTest.SPEARMAN_CORRELATION.value: TestData(
        name='Spearman Correlation',
        test_function=spearman_correlation,
        sample_size=30,
        description='Non-parametric correlation for monotonic relationships',
        assumptions=['independence'],
        required_data_type='ordinal_or_continuous'
    ),
    StatisticalTest.KENDALL_TAU_CORRELATION.value: TestData(
        name='Kendall Tau Correlation',
        test_function=kendall_tau_correlation,
        sample_size=30,
        description='Non-parametric correlation for ordinal data',
        assumptions=['independence'],
        required_data_type='ordinal'
    ),
    StatisticalTest.POINT_BISERIAL_CORRELATION.value: TestData(
        name='Point-Biserial Correlation',
        test_function=point_biserial_correlation,
        sample_size=30,
        description='Correlation between continuous and binary variables',
        assumptions=['normality'],
        required_data_type='continuous'
    ),
    StatisticalTest.CHI_SQUARE_TEST.value: TestData(
        name='Chi-Square Test',
        test_function=chi_square_test,
        sample_size=50,
        description='Tests independence between categorical variables',
        assumptions=['independence', 'sample_size'],
        required_data_type='categorical'
    ),
    StatisticalTest.FISHERS_EXACT_TEST.value: TestData(
        name='Fisher\'s Exact Test',
        test_function=fishers_exact_test,
        sample_size=20,
        description='Alternative to chi-square for small samples',
        assumptions=['independence'],
        required_data_type='categorical'
    ),
    StatisticalTest.ONE_WAY_ANOVA.value: TestData(
        name='One-Way ANOVA',
        test_function=one_way_anova,
        sample_size=45,
        description='Compares means across multiple groups',
        assumptions=['normality', 'homogeneity_of_variance', 'independence'],
        required_data_type='continuous'
    ),
    StatisticalTest.KRUSKAL_WALLIS_TEST.value: TestData(
        name='Kruskal-Wallis Test',
        test_function=kruskal_wallis_test,
        sample_size=45,
        description='Non-parametric alternative to one-way ANOVA',
        assumptions=['independence'],
        required_data_type='ordinal_or_continuous'
    ),
    StatisticalTest.REPEATED_MEASURES_ANOVA.value: TestData(
        name='Repeated Measures ANOVA',
        test_function=repeated_measures_anova,
        sample_size=20,
        description='Compares means across multiple timepoints',
        assumptions=['normality', 'sphericity', 'outliers'],
        required_data_type='continuous'
    ),
    StatisticalTest.MIXED_ANOVA.value: TestData(
        name='Mixed ANOVA',
        test_function=mixed_anova,
        sample_size=40,
        description='Combines between and within-subjects factors',
        assumptions=['normality', 'homogeneity_of_variance', 'sphericity'],
        required_data_type='continuous'
    ),
    StatisticalTest.LINEAR_MIXED_EFFECTS_MODEL.value: TestData(
        name='Linear Mixed Effects Model',
        test_function=linear_mixed_effects_model,
        sample_size=50,
        description='Model with both fixed and random effects',
        assumptions=['residual_normality', 'random_effects_normality'],
        required_data_type='continuous'
    ),
    StatisticalTest.ORDINAL_REGRESSION.value: TestData(
        name='Ordinal Regression',
        test_function=ordinal_regression,
        sample_size=50,
        description='Regression for ordinal outcomes',
        assumptions=['proportional_odds'],
        required_data_type='ordinal'
    ),
    StatisticalTest.LOGISTIC_REGRESSION.value: TestData(
        name='Logistic Regression',
        test_function=logistic_regression,
        sample_size=100,
        description='Regression for binary outcomes',
        assumptions=['linearity', 'multicollinearity'],
        required_data_type='binary'
    ),
    StatisticalTest.POISSON_REGRESSION.value: TestData(
        name='Poisson Regression',
        test_function=poisson_regression,
        sample_size=100,
        description='Regression for count outcomes',
        assumptions=['goodness_of_fit', 'overdispersion'],
        required_data_type='count'
    ),
    StatisticalTest.NEGATIVE_BINOMIAL_REGRESSION.value: TestData(
        name='Negative Binomial Regression',
        test_function=negative_binomial_regression,
        sample_size=100,
        description='Regression for count outcomes with overdispersion',
        assumptions=['goodness_of_fit', 'overdispersion'],
        required_data_type='count'
    ),
    StatisticalTest.MULTINOMIAL_LOGISTIC_REGRESSION.value: TestData(
        name='Multinomial Logistic Regression',
        test_function=multinomial_logistic_regression,
        sample_size=150,
        description='Regression for categorical outcomes with 3+ levels',
        assumptions=['linearity', 'multicollinearity'],
        required_data_type='categorical'
    ),
    StatisticalTest.ANCOVA.value: TestData(
        name='ANCOVA',
        test_function=ancova,
        sample_size=60,
        description='Compares means while controlling for covariates',
        assumptions=['normality', 'homogeneity_of_variance', 'homogeneity_of_regression_slopes'],
        required_data_type='continuous'
    ),
    StatisticalTest.SURVIVAL_ANALYSIS.value: TestData(
        name='Survival Analysis',
        test_function=survival_analysis,
        sample_size=100,
        description='Analyzes time-to-event data',
        assumptions=['proportional_hazards'],
        required_data_type='time_to_event'
    ),
    StatisticalTest.LINEAR_REGRESSION.value: TestData(
        name='Linear Regression',
        test_function=linear_regression,
        sample_size=100,
        description='Regression for continuous outcomes',
        assumptions=['normality', 'linearity', 'multicollinearity', 'homoscedasticity', 'autocorrelation'],
        required_data_type='continuous'
    ),
}
