from typing import Dict, List
from study_model.study_model import CFDataType, StatisticalTest

DATA_TYPE_TEST_REGISTRY: Dict[CFDataType, List[StatisticalTest]] = {
    CFDataType.CONTINUOUS: [
        StatisticalTest.ONE_SAMPLE_T_TEST,  
        StatisticalTest.INDEPENDENT_T_TEST,
        StatisticalTest.PAIRED_T_TEST,
        StatisticalTest.ONE_WAY_ANOVA,
        StatisticalTest.REPEATED_MEASURES_ANOVA,
        StatisticalTest.MIXED_ANOVA,
        StatisticalTest.LINEAR_REGRESSION,
        StatisticalTest.ANCOVA,
        StatisticalTest.LINEAR_MIXED_EFFECTS_MODEL,
        StatisticalTest.MANN_WHITNEY_U_TEST,
        StatisticalTest.WILCOXON_SIGNED_RANK_TEST,
        StatisticalTest.KRUSKAL_WALLIS_TEST,
        StatisticalTest.PEARSON_CORRELATION,
        StatisticalTest.SPEARMAN_CORRELATION,
        StatisticalTest.KENDALL_TAU_CORRELATION,
        StatisticalTest.POINT_BISERIAL_CORRELATION,
        StatisticalTest.SURVIVAL_ANALYSIS  # Can be used with continuous predictors
    ],
    
    CFDataType.BINARY: [
        StatisticalTest.CHI_SQUARE_TEST,
        StatisticalTest.FISHERS_EXACT_TEST,
        StatisticalTest.LOGISTIC_REGRESSION,
        StatisticalTest.MANN_WHITNEY_U_TEST,
        StatisticalTest.POINT_BISERIAL_CORRELATION,
        StatisticalTest.INDEPENDENT_T_TEST,  # Added - can be used with binary as 0/1
        StatisticalTest.ONE_WAY_ANOVA,  # Added - can be used with binary as factor
        StatisticalTest.LINEAR_REGRESSION,  # Added - can be used with binary predictors
        StatisticalTest.LINEAR_MIXED_EFFECTS_MODEL,  # Added
        StatisticalTest.SURVIVAL_ANALYSIS  # Added - binary predictors in survival models
    ],
    
    CFDataType.CATEGORICAL: [
        StatisticalTest.CHI_SQUARE_TEST,
        StatisticalTest.FISHERS_EXACT_TEST,
        StatisticalTest.MULTINOMIAL_LOGISTIC_REGRESSION,
        StatisticalTest.ONE_WAY_ANOVA,  # Added - categorical as factor
        StatisticalTest.KRUSKAL_WALLIS_TEST,  # Added
        StatisticalTest.LINEAR_MIXED_EFFECTS_MODEL,  # Added
        StatisticalTest.ANCOVA  # Added - categorical factors in ANCOVA
    ],
    
    CFDataType.ORDINAL: [
        StatisticalTest.ORDINAL_REGRESSION,
        StatisticalTest.MANN_WHITNEY_U_TEST,
        StatisticalTest.WILCOXON_SIGNED_RANK_TEST,
        StatisticalTest.KRUSKAL_WALLIS_TEST,
        StatisticalTest.SPEARMAN_CORRELATION,
        StatisticalTest.KENDALL_TAU_CORRELATION,
        StatisticalTest.LINEAR_MIXED_EFFECTS_MODEL,  # Added
        StatisticalTest.ONE_WAY_ANOVA,  # Added - can treat ordinal as factor
        StatisticalTest.CHI_SQUARE_TEST  # Added - can treat ordinal as categorical
    ],
    
    CFDataType.COUNT: [
        StatisticalTest.POISSON_REGRESSION,
        StatisticalTest.NEGATIVE_BINOMIAL_REGRESSION,
        StatisticalTest.MANN_WHITNEY_U_TEST,
        StatisticalTest.KRUSKAL_WALLIS_TEST,
        StatisticalTest.LINEAR_REGRESSION,  # Added - for transformed count data
        StatisticalTest.LINEAR_MIXED_EFFECTS_MODEL,  # Added
        StatisticalTest.SPEARMAN_CORRELATION,  # Added
        StatisticalTest.WILCOXON_SIGNED_RANK_TEST  # Added
    ],
    
    CFDataType.TIME_TO_EVENT: [
        StatisticalTest.SURVIVAL_ANALYSIS,
        StatisticalTest.MANN_WHITNEY_U_TEST,  # Added - can compare distributions
        StatisticalTest.KRUSKAL_WALLIS_TEST,  # Added - for comparing multiple groups
        StatisticalTest.SPEARMAN_CORRELATION,  # Added - for correlation with continuous variables
        StatisticalTest.LINEAR_MIXED_EFFECTS_MODEL  # Added - for longitudinal analysis
    ]
}

def get_compatible_tests_for_data_type(data_type: CFDataType) -> List[StatisticalTest]:
    """Returns a list of statistical tests compatible with the given data type."""
    return DATA_TYPE_TEST_REGISTRY.get(data_type, [])

def infer_data_type(df, column_name):
    """Infer the data type of a column in a dataframe."""
    import pandas as pd
    import numpy as np
    
    if column_name not in df.columns:
        return None
    
    series = df[column_name]
    
    # Check for binary data (0/1 or True/False)
    if pd.api.types.is_bool_dtype(series) or (pd.api.types.is_numeric_dtype(series) and series.nunique() == 2):
        return CFDataType.BINARY
    
    # Check for categorical/nominal data
    if pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
        if series.nunique() <= 10:  # Arbitrary threshold
            return CFDataType.CATEGORICAL
    
    # Check for ordinal data (integer or integer-like float with few unique values)
    if pd.api.types.is_integer_dtype(series) and series.nunique() <= 10:
        return CFDataType.ORDINAL
    
    # Also check for float columns that are actually ordinal (contain whole numbers)
    if pd.api.types.is_float_dtype(series) and series.nunique() <= 10:
        # Check if all values are whole numbers (integers stored as floats)
        if np.isclose(series, series.round()).all():
            return CFDataType.ORDINAL
    
    # Check for count data (non-negative integers)
    if pd.api.types.is_integer_dtype(series) and (series >= 0).all():
        return CFDataType.COUNT
    
    # Also check for float columns that are actually counts
    if pd.api.types.is_float_dtype(series) and (series >= 0).all():
        if np.isclose(series, series.round()).all():  # Check if whole numbers
            return CFDataType.COUNT
    
    # Check for time-to-event data (typically positive, right-skewed with possible censoring)
    if pd.api.types.is_numeric_dtype(series):
        # Time-to-event data is typically positive
        if (series >= 0).all():
            # Look for a potential censoring indicator in the dataset
            for col in df.columns:
                if col != column_name and col.lower().find('status') >= 0 or col.lower().find('event') >= 0:
                    # If we have a binary column that could be a censoring indicator
                    if df[col].nunique() == 2:
                        return CFDataType.TIME_TO_EVENT
            
            # If no obvious censoring indicator but has characteristics of survival data
            if series.skew() > 1.0 and not np.isclose(series % 1, 0).all():
                # High skew, non-integer values are common in survival data
                return CFDataType.TIME_TO_EVENT
    
    # Default to continuous for numeric data
    if pd.api.types.is_numeric_dtype(series):
        return CFDataType.CONTINUOUS
    
    # Default to categorical for anything else
    return CFDataType.CATEGORICAL