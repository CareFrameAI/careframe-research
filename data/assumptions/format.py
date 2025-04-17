from data.assumptions.tests import (
    NormalityTest, 
    HomogeneityOfVarianceTest, 
    MulticollinearityTest, 
    LinearityTest, 
    AutocorrelationTest, 
    OutlierTest, 
    ProportionalHazardsTest, 
    GoodnessOfFitTest, 
    SphericityTest, 
    IndependenceTest, 
    ResidualNormalityTest, 
    OverdispersionTest, 
    ProportionalOddsTest, 
    RandomEffectsNormalityTest, 
    HomoscedasticityTest, 
    SampleSizeTest, 
    HomogeneityOfRegressionSlopesTest, 
    EndogeneityTest, 
    ParameterStabilityTest, 
    SerialCorrelationTest, 
    InformationCriteriaTest, 
    PowerAnalysisTest, 
    InfluentialPointsTest, 
    ModelSpecificationTest, 
    DistributionFitTest, 
    ZeroInflationTest, 
    StationarityTest, 
    MissingDataRandomnessTest, 
    MeasurementErrorTest, 
    MonotonicityTest, 
    BalanceTest, 
    ConditionalIndependenceTest, 
    SelectionBiasTest, 
    SeparabilityTest, 
    JointDistributionTest, 
    ExogeneityTest
)
from enum import Enum

class AssumptionTestKeys(Enum):
    """Enum representing the keys used in the assumption test results."""
    NORMALITY = {
        "input_variables": ["data"],
        "function": NormalityTest,
        "input_types": ["numeric"],
        "output_variables": ["result", "statistic", "p_value", "details", "warnings", "figures"],
        "output_types": ["enum", "float", "float", "str", "list", "dict"]
    }
    HOMOGENEITY = {
        "input_variables": ["data", "groups"],
        "function": HomogeneityOfVarianceTest,
        "input_types": ["numeric", "categorical"],
        "output_variables": ["result", "statistic", "p_value", "details", "test_used", "group_variances", "warnings", "figures"],
        "output_types": ["enum", "float", "float", "str", "str", "dict", "list", "dict"]
    }
    MULTICOLLINEARITY = {
        "input_variables": ["df", "covariates"],
        "function": MulticollinearityTest,
        "input_types": ["dataframe", "list"],
        "output_variables": ["result", "details", "vif_values", "correlation_matrix", "warnings", "figures"],
        "output_types": ["enum", "str", "dict", "dataframe", "list", "dict"]
    }
    LINEARITY = {
        "input_variables": ["x", "y"],
        "function": LinearityTest,
        "input_types": ["numeric", "numeric"],
        "output_variables": ["result", "statistic", "p_value", "details", "test_used", "warnings", "figures"],
        "output_types": ["enum", "float", "float", "str", "str", "list", "dict"]
    }
    AUTOCORRELATION = {
        "input_variables": ["residuals"],
        "function": AutocorrelationTest,
        "input_types": ["numeric"],
        "output_variables": ["result", "statistic", "p_value", "details", "test_used", "warnings", "figures"],
        "output_types": ["enum", "float", "float", "str", "str", "list", "dict"]
    }
    OUTLIERS = {
        "input_variables": ["data"],
        "function": OutlierTest,
        "input_types": ["numeric"],
        "output_variables": ["result", "outliers", "details", "test_used", "warnings", "figures"],
        "output_types": ["enum", "list", "str", "str", "list", "dict"]
    }
    PROPORTIONAL_HAZARDS = {
        "input_variables": ["time", "event", "covariates"],
        "function": ProportionalHazardsTest,
        "input_types": ["numeric", "categorical", "dataframe"],
        "output_variables": ["result", "statistic", "p_value", "details", "test_used", "warnings", "figures"],
        "output_types": ["enum", "float", "float", "str", "str", "list", "dict"]
    }
    GOODNESS_OF_FIT = {
        "input_variables": ["observed", "expected"],
        "function": GoodnessOfFitTest,
        "input_types": ["numeric", "numeric"],
        "output_variables": ["result", "statistic", "p_value", "details", "test_used", "warnings", "figures"],
        "output_types": ["enum", "float", "float", "str", "str", "list", "dict"]
    }
    SPHERICITY = {
        "input_variables": ["data", "subject_id", "within_factor", "outcome"],
        "function": SphericityTest,
        "input_types": ["dataframe", "str", "str", "str"],
        "output_variables": ["result", "statistic", "p_value", "details", "test_used", "warnings", "figures"],
        "output_types": ["enum", "float", "float", "str", "str", "list", "dict"]
    }
    INDEPENDENCE = {
        "input_variables": ["data"],
        "function": IndependenceTest,
        "input_types": ["numeric"],
        "output_variables": ["result", "message", "statistic", "details"],
        "output_types": ["enum", "str", "float", "dict"]
    }
    RESIDUAL_NORMALITY = {
        "input_variables": ["residuals"],
        "function": ResidualNormalityTest,
        "input_types": ["numeric"],
        "output_variables": ["result", "message", "details"],
        "output_types": ["enum", "str", "dict"]
    }
    OVERDISPERSION = {
        "input_variables": ["observed", "predicted"],
        "function": OverdispersionTest,
        "input_types": ["numeric", "numeric"],
        "output_variables": ["result", "message", "details"],
        "output_types": ["enum", "str", "dict"]
    }
    PROPORTIONAL_ODDS = {
        "input_variables": ["outcome", "covariates"],
        "function": ProportionalOddsTest,
        "input_types": ["categorical", "dataframe"],
        "output_variables": ["result", "message", "details"],
        "output_types": ["enum", "str", "dict"]
    }
    RANDOM_EFFECTS_NORMALITY = {
        "input_variables": ["random_effects"],
        "function": RandomEffectsNormalityTest,
        "input_types": ["dataframe"],
        "output_variables": ["result", "statistic", "p_value", "details", "test_used", "random_effects_summary", "warnings", "figures"],
        "output_types": ["enum", "float", "float", "str", "str", "dataframe", "list", "dict"]
    }
    HOMOSCEDASTICITY = {
        "input_variables": ["residuals", "predicted"],
        "function": HomoscedasticityTest,
        "input_types": ["numeric", "numeric"],
        "output_variables": ["result", "statistic", "p_value", "details", "test_used", "warnings", "figures"],
        "output_types": ["enum", "float", "float", "str", "str", "list", "dict"]
    }
    SAMPLE_SIZE = {
        "input_variables": ["data", "min_recommended"],
        "function": SampleSizeTest,
        "input_types": ["numeric", "int"],
        "output_variables": ["result", "details", "sample_size", "minimum_required", "power", "warnings"],
        "output_types": ["enum", "str", "int", "int", "float", "list"]
    }
    HOMOGENEITY_OF_REGRESSION_SLOPES = {
        "input_variables": ["df", "outcome", "group", "covariates"],
        "function": HomogeneityOfRegressionSlopesTest,
        "input_types": ["dataframe", "str", "str", "list"],
        "output_variables": ["result", "statistic", "p_value", "details", "test_used", "warnings", "figures"],
        "output_types": ["enum", "float", "float", "str", "str", "list", "dict"]
    }
    ENDOGENEITY = {
        "input_variables": ["residuals", "predictors", "instruments"],
        "function": EndogeneityTest,
        "input_types": ["numeric", "dataframe", "dataframe"],
        "output_variables": ["result", "statistic", "p_value", "details", "test_used", "warnings", "figures"],
        "output_types": ["enum", "float", "float", "str", "str", "list", "dict"]
    }
    PARAMETER_STABILITY = {
        "input_variables": ["x", "y", "split_point"],
        "function": ParameterStabilityTest,
        "input_types": ["numeric", "numeric", "float"],
        "output_variables": ["result", "statistic", "p_value", "details", "test_used", "warnings", "figures"],
        "output_types": ["enum", "float", "float", "str", "str", "list", "dict"]
    }
    SERIAL_CORRELATION = {
        "input_variables": ["data", "lags"],
        "function": SerialCorrelationTest,
        "input_types": ["numeric", "int"],
        "output_variables": ["result", "statistic", "p_value", "details", "test_used", "warnings", "figures"],
        "output_types": ["enum", "float", "float", "str", "str", "list", "dict"]
    }
    INFORMATION_CRITERIA = {
        "input_variables": ["models"],
        "function": InformationCriteriaTest,
        "input_types": ["dict"],
        "output_variables": ["result", "details", "models_comparison", "best_model", "warnings", "figures"],
        "output_types": ["enum", "str", "dataframe", "str", "list", "dict"]
    }
    POWER_ANALYSIS = {
        "input_variables": ["effect_size", "sample_size", "alpha", "power"],
        "function": PowerAnalysisTest,
        "input_types": ["float", "int", "float", "float"],
        "output_variables": ["result", "calculated_value", "details", "warnings", "figures"],
        "output_types": ["enum", "float", "str", "list", "dict"]
    }
    INFLUENTIAL_POINTS = {
        "input_variables": ["residuals", "leverage", "fitted", "X"],
        "function": InfluentialPointsTest,
        "input_types": ["numeric", "numeric", "numeric", "dataframe"],
        "output_variables": ["result", "details", "influential_points", "warnings", "figures"],
        "output_types": ["enum", "str", "dict", "list", "dict"]
    }
    MODEL_SPECIFICATION = {
        "input_variables": ["residuals", "fitted", "X"],
        "function": ModelSpecificationTest,
        "input_types": ["numeric", "numeric", "dataframe"],
        "output_variables": ["result", "statistic", "p_value", "details", "test_used", "warnings", "figures"],
        "output_types": ["enum", "float", "float", "str", "str", "list", "dict"]
    }
    DISTRIBUTION_FIT = {
        "input_variables": ["data", "distribution"],
        "function": DistributionFitTest,
        "input_types": ["numeric", "str"],
        "output_variables": ["result", "statistic", "p_value", "details", "test_used", "parameters", "warnings", "figures"],
        "output_types": ["enum", "float", "float", "str", "str", "dict", "list", "dict"]
    }
    ZERO_INFLATION = {
        "input_variables": ["data"],
        "function": ZeroInflationTest,
        "input_types": ["numeric"],
        "output_variables": ["result", "statistic", "p_value", "details", "test_used", "warnings", "figures"],
        "output_types": ["enum", "float", "float", "str", "str", "list", "dict"]
    }
    STATIONARITY = {
        "input_variables": ["data"],
        "function": StationarityTest,
        "input_types": ["numeric"],
        "output_variables": ["result", "statistic", "p_value", "details", "test_used", "warnings", "figures"],
        "output_types": ["enum", "float", "float", "str", "str", "list", "dict"]
    }
    MISSING_DATA_RANDOMNESS = {
        "input_variables": ["df"],
        "function": MissingDataRandomnessTest,
        "input_types": ["dataframe"],
        "output_variables": ["result", "details", "test_results", "warnings", "figures"],
        "output_types": ["enum", "str", "dict", "list", "dict"]
    }
    MEASUREMENT_ERROR = {
        "input_variables": ["observed_data", "replicate_data", "reliability"],
        "function": MeasurementErrorTest,
        "input_types": ["numeric", "numeric", "float"],
        "output_variables": ["result", "details", "error_estimates", "warnings", "figures"],
        "output_types": ["enum", "str", "dict", "list", "dict"]
    }
    MONOTONICITY = {
        "input_variables": ["x", "y"],
        "function": MonotonicityTest,
        "input_types": ["numeric", "numeric"],
        "output_variables": ["result", "statistic", "p_value", "details", "test_used", "warnings", "figures"],
        "output_types": ["enum", "float", "float", "str", "str", "list", "dict"]
    }
    BALANCE = {
        "input_variables": ["df", "treatment_var", "covariates"],
        "function": BalanceTest,
        "input_types": ["dataframe", "str", "list"],
        "output_variables": ["result", "details", "balance_statistics", "warnings", "figures"],
        "output_types": ["enum", "str", "dataframe", "list", "dict"]
    }
    CONDITIONAL_INDEPENDENCE = {
        "input_variables": ["df", "var1", "var2", "conditioning_vars"],
        "function": ConditionalIndependenceTest,
        "input_types": ["dataframe", "str", "str", "list"],
        "output_variables": ["result", "statistic", "p_value", "details", "test_used", "warnings", "figures"],
        "output_types": ["enum", "float", "float", "str", "str", "list", "dict"]
    }
    SELECTION_BIAS = {
        "input_variables": ["sample_data", "population_data", "selection_variable"],
        "function": SelectionBiasTest,
        "input_types": ["dataframe", "dataframe", "str"],
        "output_variables": ["result", "details", "bias_statistics", "warnings", "figures"],
        "output_types": ["enum", "str", "dict", "list", "dict"]
    }
    SEPARABILITY = {
        "input_variables": ["df", "outcome", "factors"],
        "function": SeparabilityTest,
        "input_types": ["dataframe", "str", "list"],
        "output_variables": ["result", "details", "separability_statistics", "warnings", "figures"],
        "output_types": ["enum", "str", "dict", "list", "dict"]
    }
    JOINT_DISTRIBUTION = {
        "input_variables": ["df", "variables"],
        "function": JointDistributionTest,
        "input_types": ["dataframe", "list"],
        "output_variables": ["result", "details", "distribution_tests", "warnings", "figures"],
        "output_types": ["enum", "str", "dict", "list", "dict"]
    }
    EXOGENEITY = {
        "input_variables": ["df", "treatment_var", "outcome_var", "covariates"],
        "function": ExogeneityTest,
        "input_types": ["dataframe", "str", "str", "list"],
        "output_variables": ["result", "statistic", "p_value", "details", "test_used", "warnings", "figures"],
        "output_types": ["enum", "float", "float", "str", "str", "list", "dict"]
    }
