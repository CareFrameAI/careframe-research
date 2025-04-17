from enum import Enum
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from data.selection.detailed_tests.formatting import fig_to_svg, PASTEL_COLORS, PASTEL_CMAP

from enum import Enum

class AssumptionType(Enum):
    """Defines the types of statistical assumptions that can be tested."""
    NORMALITY = "normality"  # Tests if data follows a normal distribution, needed for many parametric tests
    HOMOGENEITY = "homogeneity_of_variance"  # Tests if variance is equal across groups, important for ANOVA and t-tests
    MULTICOLLINEARITY = "multicollinearity"  # Tests if predictors are correlated, impacts regression coefficient reliability
    LINEARITY = "linearity"  # Tests if relationships between variables are linear, critical for linear regression
    AUTOCORRELATION = "autocorrelation"  # Tests if residuals are correlated over time, important for time series
    OUTLIERS = "outliers"  # Tests for extreme values that may distort statistical results
    PROPORTIONAL_HAZARDS = "proportional_hazards"  # Tests assumption in survival analysis that hazard functions are proportional
    GOODNESS_OF_FIT = "goodness_of_fit"  # Tests how well a model fits observed data
    SPHERICITY = "sphericity"  # Tests if variances of differences between all pairs of groups are equal, important for repeated measures ANOVA
    INDEPENDENCE = "independence"  # Tests if observations are independent of each other
    RESIDUAL_NORMALITY = "residual_normality"  # Tests if model residuals follow normal distribution
    OVERDISPERSION = "overdispersion"  # Tests if variance exceeds mean in count data models
    PROPORTIONAL_ODDS = "proportional_odds"  # Tests assumption in ordinal logistic regression
    RANDOM_EFFECTS_NORMALITY = "random_effects_normality"  # Tests if random effects in mixed models follow normal distribution
    HOMOSCEDASTICITY = "homoscedasticity"  # Tests if residuals have constant variance across predicted values
    SAMPLE_SIZE = "sample_size"  # Tests if sample size is adequate for reliable statistical inference
    HOMOGENEITY_OF_REGRESSION_SLOPES = "homogeneity_of_regression_slopes"  # Tests if regression slopes are equal across groups
    ENDOGENEITY = "endogeneity"  # Tests if predictors are correlated with error terms, causing biased estimates
    PARAMETER_STABILITY = "parameter_stability"  # Tests if model parameters remain stable across different subsamples
    SERIAL_CORRELATION = "serial_correlation"  # Tests for correlation between error terms in sequential observations
    INFORMATION_CRITERIA = "information_criteria"  # Compares model fit using metrics like AIC and BIC
    POWER_ANALYSIS = "power_analysis"  # Tests if study has sufficient power to detect effects of interest
    INFLUENTIAL_POINTS = "influential_points"  # Tests for observations that disproportionately influence model fit
    MODEL_SPECIFICATION = "model_specification"  # Tests if model is correctly specified (e.g., includes all relevant variables)
    DISTRIBUTION_FIT = "distribution_fit"  # Tests how well data fits a specific probability distribution
    ZERO_INFLATION = "zero_inflation"  # Tests for excess zeros in count data beyond what's expected
    STATIONARITY = "stationarity"  # Tests if time series data's statistical properties remain constant over time
    MISSING_DATA_RANDOMNESS = "missing_data_randomness"  # Tests if missing data follows MCAR/MAR/MNAR patterns
    MEASUREMENT_ERROR = "measurement_error"  # Tests the impact of measurement errors on model estimates
    MONOTONICITY = "monotonicity"  # Tests if relationships follow monotonic patterns (important for non-parametric methods)
    BALANCE = "balance"  # Tests if treatment/control groups are balanced on covariates
    CONDITIONAL_INDEPENDENCE = "conditional_independence"  # Tests if variables are independent conditional on other variables (for causal inference)
    SELECTION_BIAS = "selection_bias"  # Tests if sample selection is biased
    SEPARABILITY = "separability"  # Tests for separability in multifactorial designs
    JOINT_DISTRIBUTION = "joint_distribution"  # Tests multivariate distributions beyond normality
    EXOGENEITY = "exogeneity"  # Tests if treatment assignment is independent of potential outcomes

class AssumptionResult(Enum):
    """Enum representing the result of an assumption check."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"

class AssumptionTest:
    """Base class for statistical assumption tests."""
    
    def __init__(self, name, description, applicable_roles, applicable_types):
        """
        Initialize an assumption test.
        
        Args:
            name (str): Name of the test
            description (str): Description of what the test checks
            applicable_roles (list): Variable roles this test applies to
            applicable_types (list): Data types this test applies to
        """
        self.name = name
        self.description = description
        self.applicable_roles = applicable_roles
        self.applicable_types = applicable_types
        
    def is_applicable(self, variable_role, data_type):
        """Check if this test is applicable to a variable with given role and type."""
        return variable_role in self.applicable_roles and data_type in self.applicable_types
        
    def run_test(self, data, **kwargs):
        """
        Run the assumption test.
        
        Args:
            data: Data to test
            **kwargs: Additional parameters for the test
            
        Returns:
            dict: Test results with at minimum 'passed' (bool) and 'details' (str) keys
        """
        raise NotImplementedError("Subclasses must implement run_test method")

# Set default color palette
sns.set_palette(PASTEL_COLORS)

class NormalityTest(AssumptionTest):
    """Test for normality assumption using Shapiro-Wilk or Kolmogorov-Smirnov tests."""
    
    def __init__(self):
        super().__init__(
            name="Normality Test",
            description="Tests whether the data follows a normal distribution",
            applicable_roles=["outcome", "covariate"],
            applicable_types=["numeric", "continuous"]
        )
        
    def run_test(self, data, **kwargs):
        """
        Run normality test on the data.
        
        Args:
            data: Pandas Series or NumPy array of values to test
            method (str, optional): 'shapiro' or 'ks'. Defaults to 'shapiro'.
            
        Returns:
            dict: Test results following the NormalityTest format
        """
        import matplotlib.pyplot as plt
        from scipy import stats
        import numpy as np
        
        method = kwargs.get('method', 'shapiro')
        alpha = kwargs.get('alpha', 0.05)
        data = pd.Series(data).dropna()
        warnings = []
        
        # Create figures with transparent backgrounds
        fig_qq = plt.figure(figsize=(8, 6))
        fig_qq.patch.set_alpha(0.0)
        ax_qq = fig_qq.add_subplot(111)
        ax_qq.patch.set_alpha(0.0)
        
        fig_hist = plt.figure(figsize=(8, 6))
        fig_hist.patch.set_alpha(0.0)
        ax_hist = fig_hist.add_subplot(111)
        ax_hist.patch.set_alpha(0.0)
        
        if len(data) < 3:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'statistic': None,
                'p_value': None,
                'details': "Not enough data points for normality test",
                'test_used': method,
                'skewness': None,
                'kurtosis': None,
                'warnings': ["Sample size too small for normality testing"],
                'figures': {
                    'qq_plot': fig_to_svg(fig_qq),
                    'histogram': fig_to_svg(fig_hist)
                }
            }
            
        # Calculate skewness and kurtosis
        try:
            skewness = float(stats.skew(data))
            # Calculate excess kurtosis (normal = 0)
            kurtosis = float(stats.kurtosis(data))
            # Convert to total kurtosis (normal = 3)
            total_kurtosis = kurtosis + 3
        except Exception as e:
            warnings.append(f"Could not calculate skewness and kurtosis: {str(e)}")
            skewness = 0.0
            kurtosis = 0.0
            total_kurtosis = 3.0
            
        if method == 'shapiro':
            # Shapiro-Wilk test (better for n < 50)
            if len(data) < 3 or len(data) > 5000:
                warnings.append(f"Sample size ({len(data)}) not ideal for Shapiro-Wilk test")
            
            statistic, p_value = stats.shapiro(data)
            test_used = 'Shapiro-Wilk'
            
        elif method == 'ks':
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
            test_used = 'Kolmogorov-Smirnov'
        else:
            raise ValueError(f"Unknown normality test method: {method}")
            
        # Determine result
        if p_value > alpha:
            result = AssumptionResult.PASSED
            details = f"The data appears to be normally distributed (p={p_value:.4f})."
        else:
            result = AssumptionResult.FAILED
            details = f"The data does not appear to be normally distributed (p={p_value:.4f})."
            
            # Add skewness and kurtosis information to details
            if abs(skewness) > 1:
                skew_direction = "positively" if skewness > 0 else "negatively"
                details += f" Data is {skew_direction} skewed (skewness={skewness:.2f})."
                
            if abs(kurtosis) > 1:
                kurt_type = "leptokurtic (heavy tails)" if kurtosis > 0 else "platykurtic (light tails)"
                details += f" Distribution is {kurt_type} (excess kurtosis={kurtosis:.2f})."
                
            warnings.append("Non-normal data may affect the validity of parametric tests")
        
        # Create QQ plot
        stats.probplot(data, plot=ax_qq)
        ax_qq.set_title('Q-Q Plot')
        
        # Create histogram with normal curve overlay
        ax_hist.hist(data, bins='auto', density=True, alpha=0.7, color=PASTEL_COLORS[0])
        x = np.linspace(min(data), max(data), 100)
        ax_hist.plot(x, stats.norm.pdf(x, data.mean(), data.std()), 'r-', lw=2)
        ax_hist.set_title('Histogram with Normal Curve')
        
        # Add skewness and kurtosis to the histogram title
        ax_hist.set_title(f'Histogram with Normal Curve\nSkewness: {skewness:.2f}, Kurtosis: {total_kurtosis:.2f}')
        
        return {
            'result': result,
            'statistic': statistic,
            'p_value': p_value,
            'details': details,
            'test_used': test_used,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'warnings': warnings,
            'figures': {
                'qq_plot': fig_to_svg(fig_qq),
                'histogram': fig_to_svg(fig_hist)
            }
        }

class HomogeneityOfVarianceTest(AssumptionTest):
    """Test for homogeneity of variance using Levene's or Bartlett's test."""
    
    def __init__(self):
        super().__init__(
            name="Homogeneity of Variance",
            description="Tests whether variances are equal across groups",
            applicable_roles=["outcome"],
            applicable_types=["numeric", "continuous"]
        )
        
    def run_test(self, data, groups, **kwargs):
        """
        Run homogeneity of variance test.
        
        Args:
            data: Series containing the outcome variable
            groups: Series containing group assignments
            method (str, optional): 'levene' or 'bartlett'. Defaults to 'levene'.
            
        Returns:
            dict: Test results following the HomogeneityOfVarianceTest format
        """
        import matplotlib.pyplot as plt
        from scipy import stats
        
        method = kwargs.get('method', 'levene')
        alpha = kwargs.get('alpha', 0.05)
        warnings = []
        
        # Create figure for boxplot
        fig_boxplot = plt.figure(figsize=(10, 6))
        fig_boxplot.patch.set_alpha(0.0)
        ax_boxplot = fig_boxplot.add_subplot(111)
        ax_boxplot.patch.set_alpha(0.0)
        
        # Create list of samples, one for each group
        grouped_data = []
        group_names = []
        group_variances = {}
        
        for group in sorted(pd.Series(groups).unique()):
            group_data = data[groups == group].dropna()
            if len(group_data) > 0:
                grouped_data.append(group_data)
                group_names.append(str(group))
                group_variances[str(group)] = group_data.var()
        
        if len(grouped_data) < 2:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'statistic': None,
                'p_value': None,
                'details': "Need at least two groups for homogeneity test",
                'test_used': method,
                'group_variances': group_variances,
                'warnings': ["Insufficient groups for variance comparison"],
                'figures': {
                    'boxplot': fig_to_svg(fig_boxplot)
                }
            }
        
        # Check if any group has less than 3 observations
        for i, group_data in enumerate(grouped_data):
            if len(group_data) < 3:
                warnings.append(f"Group {group_names[i]} has fewer than 3 observations")
        
        # Run the appropriate test
        if method == 'levene':
            statistic, p_value = stats.levene(*grouped_data)
            test_name = "Levene's test"
        elif method == 'bartlett':
            statistic, p_value = stats.bartlett(*grouped_data)
            test_name = "Bartlett's test"
        else:
            raise ValueError(f"Unknown homogeneity test method: {method}")
        
        # Determine result
        if p_value > alpha:
            result = AssumptionResult.PASSED
            details = f"The variances appear to be homogeneous across groups ({test_name}: p={p_value:.4f})."
        else:
            result = AssumptionResult.FAILED
            details = f"The variances appear to be heterogeneous across groups ({test_name}: p={p_value:.4f})."
            warnings.append("Heterogeneous variances may affect the validity of parametric tests")
        
        # Create boxplot
        ax_boxplot.boxplot(grouped_data, labels=group_names)
        ax_boxplot.set_title('Boxplot by Group')
        ax_boxplot.set_ylabel('Value')
        ax_boxplot.set_xlabel('Group')
        
        return {
            'result': result,
            'statistic': statistic,
            'p_value': p_value,
            'details': details,
            'test_used': method,
            'group_variances': group_variances,
            'warnings': warnings,
            'figures': {
                'boxplot': fig_to_svg(fig_boxplot)
            }
        }

class MulticollinearityTest(AssumptionTest):
    """Test for multicollinearity using Variance Inflation Factor (VIF)."""
    
    def __init__(self):
        super().__init__(
            name="Multicollinearity Test",
            description="Tests whether predictor variables are highly correlated",
            applicable_roles=["covariate"],
            applicable_types=["numeric", "continuous", "categorical"]
        )
        
    def run_test(self, df, covariates, **kwargs):
        """
        Calculate Variance Inflation Factor for each covariate.
        
        Args:
            df: Pandas DataFrame with the data
            covariates: List of covariate column names
            
        Returns:
            dict: Test results following the MulticollinearityTest format
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        import pandas as pd
        
        warnings = []
        
        # Create figure for correlation heatmap
        fig_heatmap = plt.figure(figsize=(10, 8))
        fig_heatmap.patch.set_alpha(0.0)
        ax_heatmap = fig_heatmap.add_subplot(111)
        ax_heatmap.patch.set_alpha(0.0)
        
        if len(covariates) < 2:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': "Need at least two covariates to test multicollinearity",
                'vif_values': {},
                'correlation_matrix': pd.DataFrame(),
                'warnings': ["Insufficient covariates for multicollinearity testing"],
                'figures': {
                    'correlation_heatmap': fig_to_svg(fig_heatmap)
                }
            }
        
        # Create a dataframe with just the covariates
        X = df[covariates].copy()
        
        # Handle categorical variables before calculating correlation
        X_numeric = X.copy()
        categorical_columns = []
        
        # Identify categorical columns
        for col in X.columns:
            if X[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X[col]):
                categorical_columns.append(col)
        
        # Create dummy variables for correlation calculation
        if categorical_columns:
            X_numeric = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
            warnings.append(f"Categorical variables were dummy-encoded: {', '.join(categorical_columns)}")
        
        # Calculate correlation matrix on the numeric data
        try:
            corr_matrix = X_numeric.corr()
        except Exception as e:
            corr_matrix = pd.DataFrame()
            warnings.append(f"Could not calculate correlation matrix: {str(e)}")
        
        # Check for high correlations in the correlation matrix
        high_corr_pairs = []
        if not corr_matrix.empty:
            # Get upper triangle of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            # Find index of feature columns with correlation greater than 0.8
            high_corr = [(upper.columns[i], upper.columns[j], upper.iloc[i, j]) 
                         for i in range(len(upper.columns)) 
                         for j in range(i+1, len(upper.columns)) 
                         if abs(upper.iloc[i, j]) > 0.8]
            
            if high_corr:
                high_corr_pairs = [f"{pair[0]}-{pair[1]} ({pair[2]:.2f})" for pair in high_corr]
                warnings.append(f"High correlations (>0.8) detected between: {', '.join(high_corr_pairs)}")
        
        # Handle categorical variables for VIF calculation
        X_vif = pd.get_dummies(X, drop_first=True)
        
        # Try a different approach for VIF calculation
        try:
            # Force all columns to be float64
            X_vif_clean = X_vif.copy()
            
            # First check for constant columns and remove them
            constant_columns = [col for col in X_vif_clean.columns 
                               if X_vif_clean[col].nunique() <= 1]
            if constant_columns:
                X_vif_clean = X_vif_clean.drop(columns=constant_columns)
                warnings.append(f"Removed constant columns for VIF calculation: {', '.join(constant_columns)}")
            
            # Convert columns to numeric
            for col in X_vif_clean.columns:
                try:
                    X_vif_clean[col] = X_vif_clean[col].astype('float64')
                except Exception as e:
                    warnings.append(f"Could not convert {col} to float64: {str(e)}")
                    try:
                        X_vif_clean[col] = pd.to_numeric(X_vif_clean[col], errors='coerce')
                    except Exception as e2:
                        warnings.append(f"Could not convert {col} using pd.to_numeric: {str(e2)}")
            
            # Now select only numeric columns
            X_vif_numeric = X_vif_clean.select_dtypes(include=[np.number])
            
            # If we lost any columns during the numeric selection, warn about it
            if len(X_vif_numeric.columns) < len(X_vif_clean.columns):
                dropped_cols = set(X_vif_clean.columns) - set(X_vif_numeric.columns)
                warnings.append(f"Dropped non-numeric columns for VIF calculation: {', '.join(dropped_cols)}")
            
            # Handle missing or infinite values
            X_vif_numeric = X_vif_numeric.replace([np.inf, -np.inf], np.nan)
            
            # Fill missing values with column means or zeros if all are missing
            for col in X_vif_numeric.columns:
                if X_vif_numeric[col].isna().all():
                    X_vif_numeric[col] = 0
                    warnings.append(f"Column {col} had all missing values, filled with zeros")
                else:
                    X_vif_numeric[col] = X_vif_numeric[col].fillna(X_vif_numeric[col].mean())
            
            # Check for columns with near-zero variance
            low_var_cols = []
            for col in X_vif_numeric.columns:
                if X_vif_numeric[col].var() < 1e-10:
                    low_var_cols.append(col)
            
            if low_var_cols:
                warnings.append(f"Columns with near-zero variance detected: {', '.join(low_var_cols)}")
                # Add a tiny bit more variance to these columns
                for col in low_var_cols:
                    X_vif_numeric[col] = X_vif_numeric[col] + np.random.normal(0, 1e-5, len(X_vif_numeric))
            
            # Check if we have enough columns for VIF calculation
            if len(X_vif_numeric.columns) < 2:
                warnings.append("Not enough numeric columns for VIF calculation after data cleaning")
                vif_data = {col: None for col in X_vif.columns}
            else:
                # Check for perfect multicollinearity by examining the condition number
                X_array = X_vif_numeric.values
                from numpy.linalg import svd
                _, s, _ = svd(X_array)
                condition_number = max(s) / min(s) if min(s) > 0 else float('inf')
                
                if condition_number > 1000:
                    warnings.append(f"High condition number detected: {condition_number:.2f}. This indicates potential multicollinearity issues.")
                
                # Add a constant term for the VIF calculation
                from statsmodels.tools.tools import add_constant
                X_with_const = add_constant(X_vif_numeric)
                
                # Calculate VIF for each variable (skip the constant term)
                vif_data = {}
                for i, column in enumerate(X_vif_numeric.columns):
                    try:
                        # Use the index i+1 to skip the constant column
                        vif_value = variance_inflation_factor(X_with_const.values, i+1)
                        
                        # Check for unreasonably high VIF values (which might indicate numerical issues)
                        if vif_value > 100:
                            warnings.append(f"Extremely high VIF detected for {column}: {vif_value:.2f}. This may indicate numerical issues.")
                            # Cap the VIF at 100 for reporting purposes
                            vif_data[column] = 100.0
                        else:
                            vif_data[column] = vif_value
                    except Exception as e:
                        vif_data[column] = None
                        warnings.append(f"Could not calculate VIF for {column}: {str(e)}")
                
                # For any columns that were in X_vif but not in X_vif_numeric, set VIF to None
                for col in X_vif.columns:
                    if col not in X_vif_numeric.columns:
                        vif_data[col] = None
        
        except Exception as e:
            # If the entire VIF calculation fails, set all VIFs to None
            vif_data = {col: None for col in X_vif.columns}
            warnings.append(f"VIF calculation failed: {str(e)}")
        
        # Determine if there's multicollinearity
        high_vif = {k: v for k, v in vif_data.items() if v is not None and v > 5}
        very_high_vif = {k: v for k, v in vif_data.items() if v is not None and v > 10}
        
        # Cross-check VIF results with correlation matrix
        if very_high_vif and not high_corr_pairs:
            warnings.append("High VIF values detected but correlation matrix doesn't show strong correlations. This may indicate multicollinearity involving multiple variables or numerical issues with synthetic data.")
        
        if very_high_vif:
            result = AssumptionResult.FAILED
            details = f"Severe multicollinearity detected (VIF > 10) for: {', '.join(very_high_vif.keys())}"
            warnings.append("Severe multicollinearity may cause unstable coefficient estimates")
        elif high_vif:
            result = AssumptionResult.WARNING
            details = f"Potential multicollinearity detected (VIF > 5) for: {', '.join(high_vif.keys())}"
            warnings.append("Moderate multicollinearity may affect interpretation of coefficients")
        else:
            result = AssumptionResult.PASSED
            details = "No significant multicollinearity detected among covariates."
        
        # Create correlation heatmap if we have a correlation matrix
        if not corr_matrix.empty:
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax_heatmap)
            ax_heatmap.set_title('Correlation Matrix of Covariates')
        else:
            ax_heatmap.text(0.5, 0.5, "Could not generate correlation heatmap", 
                           horizontalalignment='center', verticalalignment='center')
        
        return {
            'result': result,
            'details': details,
            'vif_values': vif_data,
            'correlation_matrix': corr_matrix,
            'warnings': warnings,
            'figures': {
                'correlation_heatmap': fig_to_svg(fig_heatmap)
            }
        }

class OutlierTest(AssumptionTest):
    """Test for outliers using various methods."""
    
    def __init__(self):
        super().__init__(
            name="Outlier Detection",
            description="Identifies potential outliers in the data",
            applicable_roles=["outcome", "covariate"],
            applicable_types=["numeric", "continuous"]
        )
        
    def run_test(self, data, **kwargs):
        """
        Detect outliers in the data.
        
        Args:
            data: Pandas Series or NumPy array
            method (str, optional): 'zscore', 'iqr', Default is 'iqr'
            
        Returns:
            dict: Test results following the OutlierTest format
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats
        
        method = kwargs.get('method', 'iqr')
        threshold = kwargs.get('threshold', 1.5 if method == 'iqr' else 3)
        warnings = []
        
        # Create figures
        fig_boxplot = plt.figure(figsize=(10, 6))
        fig_boxplot.patch.set_alpha(0.0)
        ax_boxplot = fig_boxplot.add_subplot(111)
        ax_boxplot.patch.set_alpha(0.0)
        
        fig_zscore = plt.figure(figsize=(10, 6))
        fig_zscore.patch.set_alpha(0.0)
        ax_zscore = fig_zscore.add_subplot(111)
        ax_zscore.patch.set_alpha(0.0)
        
        data = pd.Series(data).dropna()
        
        if len(data) < 5:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': "Not enough data points for outlier detection",
                'outliers': {
                    'indices': [],
                    'values': []
                },
                'method': method,
                'threshold': threshold,
                'z_scores': [],
                'warnings': ["Sample size too small for outlier detection"],
                'figures': {
                    'boxplot': fig_to_svg(fig_boxplot),
                    'z_score_plot': fig_to_svg(fig_zscore)
                }
            }
        
        outlier_indices = []
        z_scores = []
        
        if method == 'zscore':
            # Z-score method
            z_scores = stats.zscore(data)
            outlier_indices = np.where(np.abs(z_scores) > threshold)[0].tolist()
            threshold_description = f"Z-score > {threshold}"
            
        elif method == 'iqr':
            # IQR method
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outlier_indices = data[(data < lower_bound) | (data > upper_bound)].index.tolist()
            threshold_description = f"Outside {threshold} × IQR"
            
            # Calculate z-scores anyway for the plot
            z_scores = stats.zscore(data)
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        # Get the outlier values
        outlier_values = data.iloc[outlier_indices].tolist() if isinstance(data.index, pd.RangeIndex) else data.loc[outlier_indices].tolist()
        
        # Determine result
        if len(outlier_indices) > 0:
            result = AssumptionResult.WARNING
            details = f"Detected {len(outlier_indices)} potential outliers using {method} method ({threshold_description})."
            warnings.append("Outliers may affect statistical tests and model performance")
        else:
            result = AssumptionResult.PASSED
            details = f"No outliers detected using {method} method ({threshold_description})."
        
        # Create boxplot
        ax_boxplot.boxplot(data, patch_artist=True, boxprops=dict(facecolor=PASTEL_COLORS[0]))
        ax_boxplot.set_title('Boxplot with Potential Outliers')
        ax_boxplot.set_ylabel('Value')
        if outlier_values:
            ax_boxplot.scatter(np.ones(len(outlier_values)), outlier_values, color='red', s=30)
        
        # Create z-score plot
        ax_zscore.scatter(range(len(data)), np.abs(z_scores), color=PASTEL_COLORS[1])
        ax_zscore.axhline(y=threshold if method == 'zscore' else 0, color='r', linestyle='-')
        ax_zscore.set_title('Absolute Z-scores')
        ax_zscore.set_xlabel('Data Point Index')
        ax_zscore.set_ylabel('Absolute Z-score')
        
        return {
            'result': result,
            'details': details,
            'outliers': {
                'indices': outlier_indices,
                'values': outlier_values
            },
            'method': method,
            'threshold': threshold,
            'z_scores': z_scores.tolist() if hasattr(z_scores, 'tolist') else list(z_scores),
            'warnings': warnings,
            'figures': {
                'boxplot': fig_to_svg(fig_boxplot),
                'z_score_plot': fig_to_svg(fig_zscore)
            }
        }

class LinearityTest(AssumptionTest):
    """Test for linearity between predictor and outcome variables."""
    
    def __init__(self):
        super().__init__(
            name="Linearity Test",
            description="Tests whether the relationship between variables is linear",
            applicable_roles=["covariate", "outcome"],
            applicable_types=["numeric", "continuous"]
        )
        
    def run_test(self, x, y, **kwargs):
        """
        Test linearity by comparing linear model to LOWESS.
        
        Args:
            x: Predictor variable (Series or array)
            y: Outcome variable (Series or array)
            
        Returns:
            dict: Test results following the LinearityTest format
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats
        import statsmodels.api as sm
        import pandas as pd
        
        warnings = []
        
        # Create figures
        fig_scatter = plt.figure(figsize=(10, 6))
        fig_scatter.patch.set_alpha(0.0)
        ax_scatter = fig_scatter.add_subplot(111)
        ax_scatter.patch.set_alpha(0.0)
        
        fig_residual = plt.figure(figsize=(10, 6))
        fig_residual.patch.set_alpha(0.0)
        ax_residual = fig_residual.add_subplot(111)
        ax_residual.patch.set_alpha(0.0)
        
        # Convert inputs to pandas Series if they aren't already
        if not isinstance(x, pd.Series):
            x = pd.Series(x)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
            
        # Drop NaN values
        data = pd.DataFrame({'x': x, 'y': y}).dropna()
        if len(data) < 10:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': "Too few samples to test linearity (minimum 10 required)",
                'pearson_r': None,
                'pearson_p': None,
                'spearman_rho': None,
                'spearman_p': None,
                'warnings': ["Sample size too small for linearity testing"],
                'figures': {
                    'scatter_plot': fig_to_svg(fig_scatter),
                    'residual_plot': fig_to_svg(fig_residual)
                }
            }
        
        x_clean = data['x']
        y_clean = data['y']
        
        # Check for constant values
        if x_clean.nunique() <= 1:
            ax_scatter.scatter(x_clean, y_clean, alpha=0.6)
            ax_scatter.set_title('Scatter Plot - Cannot Test Linearity')
            ax_scatter.set_xlabel('X (constant)')
            ax_scatter.set_ylabel('Y')
            
            ax_residual.text(0.5, 0.5, "Cannot calculate residuals with constant X", 
                           horizontalalignment='center', verticalalignment='center')
            
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': "X variable is constant, cannot test linearity",
                'pearson_r': None,
                'pearson_p': None,
                'spearman_rho': None,
                'spearman_p': None,
                'warnings': ["X variable has no variance, cannot assess relationship with Y"],
                'figures': {
                    'scatter_plot': fig_to_svg(fig_scatter),
                    'residual_plot': fig_to_svg(fig_residual)
                }
            }
        
        if y_clean.nunique() <= 1:
            ax_scatter.scatter(x_clean, y_clean, alpha=0.6)
            ax_scatter.set_title('Scatter Plot - Cannot Test Linearity')
            ax_scatter.set_xlabel('X')
            ax_scatter.set_ylabel('Y (constant)')
            
            ax_residual.text(0.5, 0.5, "Cannot calculate residuals with constant Y", 
                           horizontalalignment='center', verticalalignment='center')
            
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': "Y variable is constant, cannot test linearity",
                'pearson_r': None,
                'pearson_p': None,
                'spearman_rho': None,
                'spearman_p': None,
                'warnings': ["Y variable has no variance, cannot assess relationship with X"],
                'figures': {
                    'scatter_plot': fig_to_svg(fig_scatter),
                    'residual_plot': fig_to_svg(fig_residual)
                }
            }
        
        # Check for near-zero variance
        x_var = np.var(x_clean)
        y_var = np.var(y_clean)
        
        if x_var < 1e-10 or y_var < 1e-10:
            warnings.append(f"Near-zero variance detected: X variance = {x_var:.10f}, Y variance = {y_var:.10f}")
            warnings.append("Results may be unreliable due to extremely low variance")
        
        # Calculate correlation coefficients with error handling
        try:
            pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)
        except Exception as e:
            warnings.append(f"Could not calculate Pearson correlation: {str(e)}")
            pearson_r, pearson_p = None, None
            
        try:
            spearman_rho, spearman_p = stats.spearmanr(x_clean, y_clean)
        except Exception as e:
            warnings.append(f"Could not calculate Spearman correlation: {str(e)}")
            spearman_rho, spearman_p = None, None
        
        # Check for suspicious p-values
        if pearson_p is not None and (pearson_p == 0 or pearson_p == 1):
            warnings.append(f"Suspicious Pearson p-value: {pearson_p}. This may indicate numerical issues.")
            
        if spearman_p is not None and (spearman_p == 0 or spearman_p == 1):
            warnings.append(f"Suspicious Spearman p-value: {spearman_p}. This may indicate numerical issues.")
        
        # Fit linear model with error handling
        try:
            X = sm.add_constant(x_clean)
            model = sm.OLS(y_clean, X).fit()
            r_squared = model.rsquared
            
            # Get residuals
            linear_pred = model.predict(X)
            residuals = y_clean - linear_pred
        except Exception as e:
            warnings.append(f"Could not fit linear model: {str(e)}")
            r_squared = None
            linear_pred = None
            residuals = None
        
        # Fit LOWESS with error handling
        try:
            # Use a smaller fraction for larger datasets
            frac = min(0.6, max(0.2, 30 / len(x_clean)))
            lowess = sm.nonparametric.lowess(y_clean, x_clean, frac=frac)
            
            # Compare linear prediction to LOWESS if both are available
            if linear_pred is not None:
                lowess_pred = np.interp(x_clean, lowess[:, 0], lowess[:, 1])
                
                # Calculate mean squared difference
                msd = np.mean((linear_pred - lowess_pred) ** 2)
                variance = np.var(y_clean)
                relative_diff = msd / variance if variance > 0 else float('inf')
            else:
                relative_diff = None
        except Exception as e:
            warnings.append(f"Could not fit LOWESS model: {str(e)}")
            lowess = None
            relative_diff = None
        
        # Determine result
        if r_squared is None or relative_diff is None:
            result = AssumptionResult.NOT_APPLICABLE
            details = "Could not assess linearity due to computational issues"
        elif r_squared > 0.7 and relative_diff < 0.1:
            result = AssumptionResult.PASSED
            details = f"Relationship appears linear (R²={r_squared:.2f}, Pearson r={pearson_r:.2f}, p={pearson_p:.4f})"
        elif r_squared > 0.3 and relative_diff < 0.2:
            result = AssumptionResult.WARNING
            details = f"Relationship may not be strictly linear (R²={r_squared:.2f}, Pearson r={pearson_r:.2f}, p={pearson_p:.4f})"
            warnings.append("Consider transformations or non-linear terms")
        else:
            result = AssumptionResult.FAILED
            details = f"Relationship is not linear (R²={r_squared:.2f}, Pearson r={pearson_r:.2f}, p={pearson_p:.4f})"
            warnings.append("Consider non-linear models or transformations")
        
        # Create scatter plot with regression line and LOWESS curve
        ax_scatter.scatter(x_clean, y_clean, alpha=0.6, color=PASTEL_COLORS[0])
        
        if linear_pred is not None:
            ax_scatter.plot(x_clean, linear_pred, 'r-', label='Linear fit')
            
        if lowess is not None:
            ax_scatter.plot(lowess[:, 0], lowess[:, 1], 'g-', label='LOWESS')
            
        ax_scatter.set_title('Scatter Plot with Regression Line')
        ax_scatter.set_xlabel('X')
        ax_scatter.set_ylabel('Y')
        ax_scatter.legend()
        
        # Create residual plot
        if residuals is not None and linear_pred is not None:
            ax_residual.scatter(linear_pred, residuals, alpha=0.6, color=PASTEL_COLORS[1])
            ax_residual.axhline(y=0, color='r', linestyle='-')
            ax_residual.set_title('Residuals vs. Fitted Values')
            ax_residual.set_xlabel('Fitted Values')
            ax_residual.set_ylabel('Residuals')
        else:
            ax_residual.text(0.5, 0.5, "Could not calculate residuals", 
                           horizontalalignment='center', verticalalignment='center')
        
        return {
            'result': result,
            'details': details,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_rho': spearman_rho,
            'spearman_p': spearman_p,
            'r_squared': r_squared,
            'relative_diff': relative_diff,
            'warnings': warnings,
            'figures': {
                'scatter_plot': fig_to_svg(fig_scatter),
                'residual_plot': fig_to_svg(fig_residual)
            }
        }

class SampleSizeTest(AssumptionTest):
    """Test for adequate sample size."""
    
    def __init__(self):
        super().__init__(
            name="Sample Size Check",
            description="Checks if sample size is adequate for the analysis",
            applicable_roles=["outcome", "group"],
            applicable_types=["numeric", "categorical"]
        )
        
    def run_test(self, data, min_recommended, **kwargs):
        """
        Check if sample size is adequate.
        
        Args:
            data: Data to check (used only for length)
            min_recommended: Minimum recommended sample size
            
        Returns:
            dict: Test results following the SampleSizeTest format
        """
        warnings = []
        n = len(pd.Series(data).dropna())
        
        if n >= min_recommended:
            result = AssumptionResult.PASSED
            details = f"Sample size is adequate ({n} ≥ {min_recommended})"
        elif n >= min_recommended / 2:
            result = AssumptionResult.WARNING
            details = f"Sample size may be too small ({n} < {min_recommended})"
            warnings.append("Results may lack statistical power")
        else:
            result = AssumptionResult.FAILED
            details = f"Sample size is too small ({n} < {min_recommended/2})"
            warnings.append("Consider collecting more data for reliable results")
        
        return {
            'result': result,
            'details': details,
            'sample_size': n,
            'min_recommended': min_recommended,
            'warnings': warnings
        }

class HomogeneityOfRegressionSlopesTest(AssumptionTest):
    """Test for homogeneity of regression slopes for ANCOVA."""
    
    def __init__(self):
        super().__init__(
            name="Homogeneity of Regression Slopes",
            description="Tests whether covariates have the same relationship with the outcome across groups",
            applicable_roles=["outcome", "covariate", "group"],
            applicable_types=["numeric", "categorical"]
        )
        
    def run_test(self, df, outcome, group, covariates, **kwargs):
        """
        Check homogeneity of regression slopes assumption for ANCOVA.
        
        Args:
            df: DataFrame with the data
            outcome: Name of outcome variable
            group: Name of grouping variable
            covariates: List of names of covariates
            alpha: Significance level (optional, default 0.05)
            
        Returns:
            dict: Test results following the HomogeneityOfRegressionSlopesTest format
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import statsmodels.api as sm
        
        alpha = kwargs.get('alpha', 0.05)
        warnings = []
        
        # Create figure for slopes plot
        fig_slopes = plt.figure(figsize=(12, 8))
        fig_slopes.patch.set_alpha(0.0)
        ax_slopes = fig_slopes.add_subplot(111)
        ax_slopes.patch.set_alpha(0.0)
        
        # For each covariate, test the interaction with the group
        interaction_terms = []
        p_values = {}
        f_statistics = {}
        interaction_df = pd.DataFrame()
        
        # If there are no covariates or groups, return not applicable
        if not covariates or len(df[group].unique()) < 2:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'statistic': None,
                'p_value': None,
                'details': "Need at least one covariate and two groups for homogeneity test",
                'interaction_df': pd.DataFrame(),
                'warnings': ["Insufficient data for homogeneity of regression slopes test"],
                'figures': {
                    'slopes_plot': fig_to_svg(fig_slopes)
                }
            }
        
        # Choose the first covariate for visualization
        primary_covariate = covariates[0]
        
        # Plot regression lines for each group
        for i, group_val in enumerate(df[group].unique()):
            group_data = df[df[group] == group_val]
            if len(group_data) > 2:  # Need at least 3 points for regression
                try:
                    x = group_data[primary_covariate]
                    y = group_data[outcome]
                    
                    # Skip if x or y has non-numeric values
                    if not pd.api.types.is_numeric_dtype(x) or not pd.api.types.is_numeric_dtype(y):
                        continue
                        
                    ax_slopes.scatter(x, y, label=f"{group}={group_val}", alpha=0.6, 
                                    color=PASTEL_COLORS[i % len(PASTEL_COLORS)])
                    
                    # Fit and plot regression line
                    if len(x) > 1:  # Need at least 2 points for line
                        X = sm.add_constant(x)
                        model = sm.OLS(y, X).fit()
                        x_range = np.linspace(x.min(), x.max(), 100)
                        X_range = sm.add_constant(x_range)
                        y_pred = model.predict(X_range)
                        ax_slopes.plot(x_range, y_pred, '-')
                except Exception as e:
                    warnings.append(f"Error plotting regression line for {group}={group_val}: {str(e)}")
        
        for covariate in covariates:
            try:
                # Skip non-numeric covariates
                if not pd.api.types.is_numeric_dtype(df[covariate]):
                    warnings.append(f"Skipping non-numeric covariate: {covariate}")
                    continue
                
                # Check for missing values
                if df[covariate].isna().any() or df[outcome].isna().any():
                    # Drop missing values for this analysis
                    clean_df = df.dropna(subset=[covariate, outcome])
                    if len(clean_df) < 10:  # Need enough data after dropping NAs
                        warnings.append(f"Too few complete cases for {covariate} after removing missing values")
                        continue
                else:
                    clean_df = df
                
                # Create interaction term formula
                formula = f"{outcome} ~ C({group}) + {covariate} + C({group}):{covariate}"
                
                # Fit the model
                model = sm.formula.ols(formula, data=clean_df).fit()
                
                # Extract p-value and F-statistic for the interaction term
                interaction_term = None
                for term in model.pvalues.index:
                    if f"C({group}):{covariate}" in term:
                        interaction_term = term
                        p_values[f"{group}:{covariate}"] = float(model.pvalues[term])
                        interaction_terms.append(term)
                        break
                
                # If we didn't find the interaction term, try a different approach
                if interaction_term is None:
                    warnings.append(f"Could not find interaction term for {covariate}, trying alternative approach")
                    
                    # Try ANOVA approach
                    try:
                        # Create separate models with and without interaction
                        formula_with_int = f"{outcome} ~ C({group}) * {covariate}"
                        formula_no_int = f"{outcome} ~ C({group}) + {covariate}"
                        
                        model_with_int = sm.formula.ols(formula_with_int, data=clean_df).fit()
                        model_no_int = sm.formula.ols(formula_no_int, data=clean_df).fit()
                        
                        # Compare models with anova_lm
                        from statsmodels.stats.anova import anova_lm
                        anova_table = anova_lm(model_no_int, model_with_int)
                        
                        # Extract p-value from the last row (the interaction effect)
                        p_value = float(anova_table.iloc[-1, -1])
                        f_stat = float(anova_table.iloc[-1, 0])
                        
                        p_values[f"{group}:{covariate}"] = p_value
                        f_statistics[f"{group}:{covariate}"] = f_stat
                    except Exception as e:
                        warnings.append(f"Alternative approach failed for {covariate}: {str(e)}")
                        continue
                else:
                    # Get F-statistic from ANOVA table
                    try:
                        anova_table = sm.stats.anova_lm(model, typ=2)
                        for anova_term in anova_table.index:
                            if f"C({group}):{covariate}" in anova_term:
                                f_statistics[f"{group}:{covariate}"] = float(anova_table.loc[anova_term, 'F'])
                                break
                    except Exception as e:
                        warnings.append(f"Could not extract F-statistic for {covariate}: {str(e)}")
                        # Use a placeholder value
                        f_statistics[f"{group}:{covariate}"] = 0.0
                
                # Add to interaction DataFrame
                temp_df = pd.DataFrame({
                    'covariate': [covariate],
                    'p_value': [p_values.get(f"{group}:{covariate}", None)],
                    'f_statistic': [f_statistics.get(f"{group}:{covariate}", None)]
                })
                interaction_df = pd.concat([interaction_df, temp_df], ignore_index=True)
                
            except Exception as e:
                warnings.append(f"Error testing interaction for {covariate}: {str(e)}")
        
        # Check if any interaction terms are significant
        significant_interactions = [term for term, p in p_values.items() if p < alpha]
        
        # Get overall F-statistic and p-value (use the first one if multiple)
        overall_statistic = next(iter(f_statistics.values())) if f_statistics else None
        overall_p_value = next(iter(p_values.values())) if p_values else None
        
        if not p_values:
            result = AssumptionResult.NOT_APPLICABLE
            details = "Could not test interaction terms. Check model specification."
        elif significant_interactions:
            result = AssumptionResult.FAILED
            details = f"Homogeneity of regression slopes violated for: {', '.join(significant_interactions)}"
            warnings.append("Different slopes across groups may invalidate ANCOVA results")
        else:
            result = AssumptionResult.PASSED
            details = "Homogeneity of regression slopes assumption satisfied"
        
        # Finalize the plot
        ax_slopes.set_title(f'Regression Slopes by Group for {primary_covariate}')
        ax_slopes.set_xlabel(primary_covariate)
        ax_slopes.set_ylabel(outcome)
        ax_slopes.legend()
        
        return {
            'result': result,
            'statistic': overall_statistic,
            'p_value': overall_p_value,
            'details': details,
            'interaction_df': interaction_df,
            'warnings': warnings,
            'figures': {
                'slopes_plot': fig_to_svg(fig_slopes)
            }
        }

class MeasurementErrorTest(AssumptionTest):
    """Test the impact of measurement errors on model estimates."""
    
    def __init__(self):
        super().__init__(
            name="Measurement Error Test",
            description="Tests the impact of measurement errors on model estimates",
            applicable_roles=["covariate", "outcome"],
            applicable_types=["numeric", "continuous"]
        )
        
    def run_test(self, observed_data, replicate_data=None, reliability=None, **kwargs):
        """
        Assess the impact of measurement error.
        
        Args:
            observed_data: Primary measured data (with potential measurement error)
            replicate_data: Optional repeat measurements of the same variable
            reliability: Known reliability coefficient (between 0-1) if available
            
        Returns:
            dict: Test results with measurement error assessment
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats
        
        warnings = []
        
        # Create figures
        fig_reliability = plt.figure(figsize=(10, 6))
        fig_reliability.patch.set_alpha(0.0)
        ax_reliability = fig_reliability.add_subplot(111)
        ax_reliability.patch.set_alpha(0.0)
        
        # Check if we have enough information to assess measurement error
        if replicate_data is None and reliability is None:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': "Cannot assess measurement error without repeat measurements or reliability coefficient",
                'reliability_estimate': None,
                'attenuation_factor': None,
                'warnings': ["No information provided to quantify measurement error"],
                'figures': {
                    'reliability_plot': fig_to_svg(fig_reliability)
                }
            }
        
        # Calculate reliability from replicate measurements if available
        if replicate_data is not None:
            observed = pd.Series(observed_data).dropna()
            replicate = pd.Series(replicate_data).dropna()
            
            # Match observations between the two series
            if len(observed) != len(replicate):
                # Find common indices if Series have different lengths
                common_indices = observed.index.intersection(replicate.index)
                observed = observed.loc[common_indices]
                replicate = replicate.loc[common_indices]
                
                if len(observed) < 3:
                    return {
                        'result': AssumptionResult.NOT_APPLICABLE,
                        'details': "Not enough matched measurements to assess reliability",
                        'reliability_estimate': None,
                        'attenuation_factor': None,
                        'warnings': ["Insufficient paired measurements"],
                        'figures': {
                            'reliability_plot': fig_to_svg(fig_reliability)
                        }
                    }
            
            # Calculate correlation between measurements (reliability estimate)
            try:
                reliability_estimate, p_value = stats.pearsonr(observed, replicate)
            except Exception as e:
                warnings.append(f"Error calculating reliability: {str(e)}")
                reliability_estimate = None
            
            # Plot the two measurements against each other
            ax_reliability.scatter(observed, replicate, alpha=0.6, color=PASTEL_COLORS[0])
            ax_reliability.plot([observed.min(), observed.max()], [observed.min(), observed.max()], 'r--')
            ax_reliability.set_title('Test-Retest Reliability Plot')
            ax_reliability.set_xlabel('First Measurement')
            ax_reliability.set_ylabel('Second Measurement')
            
            # Add correlation text to plot
            if reliability_estimate is not None:
                ax_reliability.text(0.05, 0.95, f"Reliability: {reliability_estimate:.3f}", 
                                  transform=ax_reliability.transAxes,
                                  verticalalignment='top', horizontalalignment='left')
                
        # Use provided reliability if replicate data not available
        elif reliability is not None:
            reliability_estimate = reliability
            ax_reliability.text(0.5, 0.5, f"Reliability coefficient: {reliability_estimate:.3f}", 
                              ha='center', va='center')
            ax_reliability.set_title('Provided Reliability Coefficient')
            ax_reliability.set_xticks([])
            ax_reliability.set_yticks([])
        
        # Calculate attenuation factor (impact on regression coefficients)
        if reliability_estimate is not None:
            attenuation_factor = np.sqrt(reliability_estimate)
        else:
            attenuation_factor = None
        
        # Determine result
        if reliability_estimate is None:
            result = AssumptionResult.NOT_APPLICABLE
            details = "Could not determine reliability"
        elif reliability_estimate > 0.9:
            result = AssumptionResult.PASSED
            details = f"Excellent reliability ({reliability_estimate:.3f}), minimal measurement error impact"
        elif reliability_estimate > 0.7:
            result = AssumptionResult.WARNING
            details = f"Acceptable reliability ({reliability_estimate:.3f}), but some measurement error present"
            warnings.append(f"Regression coefficients may be attenuated by a factor of ~{attenuation_factor:.3f}")
        else:
            result = AssumptionResult.FAILED
            details = f"Poor reliability ({reliability_estimate:.3f}), substantial measurement error"
            warnings.append("High measurement error may severely bias estimates")
            warnings.append(f"Regression coefficients may be attenuated by a factor of ~{attenuation_factor:.3f}")
            
        return {
            'result': result,
            'details': details,
            'reliability_estimate': reliability_estimate,
            'attenuation_factor': attenuation_factor,
            'warnings': warnings,
            'figures': {
                'reliability_plot': fig_to_svg(fig_reliability)
            }
        }

class SelectionBiasTest(AssumptionTest):
    """Test if sample selection is biased."""
    
    def __init__(self):
        super().__init__(
            name="Selection Bias Test",
            description="Tests if sample selection is biased",
            applicable_roles=["covariate", "outcome", "group"],
            applicable_types=["numeric", "categorical"]
        )
        
    def run_test(self, sample_data, population_data=None, selection_variable=None, **kwargs):
        """
        Test for sample selection bias.
        
        Args:
            sample_data: DataFrame containing the sample data
            population_data: Optional DataFrame or summary statistics from population
            selection_variable: Optional variable that might affect selection
            
        Returns:
            dict: Test results following the SelectionBiasTest format
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats
        
        warnings = []
        
        # Create figures
        fig_distribution = plt.figure(figsize=(10, 6))
        fig_distribution.patch.set_alpha(0.0)
        ax_distribution = fig_distribution.add_subplot(111)
        ax_distribution.patch.set_alpha(0.0)
        
        # Check if we have enough information to assess selection bias
        if population_data is None and selection_variable is None:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': "Cannot assess selection bias without population data or selection variable",
                'comparison_statistics': {},
                'warnings': ["Insufficient information to test for selection bias"],
                'figures': {
                    'distribution_plot': fig_to_svg(fig_distribution)
                }
            }
        
        comparison_statistics = {}
        
        # Approach 1: Compare sample to population on key variables
        if population_data is not None:
            # If population_data is a DataFrame, compare distributions
            if isinstance(population_data, pd.DataFrame):
                # Find common columns for comparison
                common_cols = [col for col in sample_data.columns if col in population_data.columns]
                
                if not common_cols:
                    warnings.append("No common variables between sample and population data")
                
                # Compare means and variances for each common column
                for col in common_cols:
                    if pd.api.types.is_numeric_dtype(sample_data[col]) and pd.api.types.is_numeric_dtype(population_data[col]):
                        sample_mean = sample_data[col].mean()
                        pop_mean = population_data[col].mean()
                        sample_std = sample_data[col].std()
                        pop_std = population_data[col].std()
                        
                        # Calculate standardized difference
                        std_diff = (sample_mean - pop_mean) / pop_std if pop_std > 0 else 0
                        
                        comparison_statistics[col] = {
                            'sample_mean': sample_mean,
                            'population_mean': pop_mean,
                            'standardized_difference': std_diff
                        }
                        
                        # Plot histogram of sample vs population for the first numeric column
                        if col == common_cols[0]:
                            ax_distribution.hist(sample_data[col].dropna(), bins=20, alpha=0.5, 
                                               label='Sample', color=PASTEL_COLORS[0])
                            ax_distribution.hist(population_data[col].dropna(), bins=20, alpha=0.5, 
                                               label='Population', color=PASTEL_COLORS[1])
                            ax_distribution.set_title(f'Distribution Comparison: {col}')
                            ax_distribution.set_xlabel(col)
                            ax_distribution.set_ylabel('Frequency')
                            ax_distribution.legend()
                    
                    elif pd.api.types.is_categorical_dtype(sample_data[col]) or sample_data[col].dtype == 'object':
                        # For categorical variables, compare proportions
                        sample_counts = sample_data[col].value_counts(normalize=True)
                        pop_counts = population_data[col].value_counts(normalize=True)
                        
                        # Calculate chi-square for distribution comparison
                        # This requires aligning the categories first
                        all_categories = set(sample_counts.index) | set(pop_counts.index)
                        aligned_sample = pd.Series({cat: sample_counts.get(cat, 0) for cat in all_categories})
                        aligned_pop = pd.Series({cat: pop_counts.get(cat, 0) for cat in all_categories})
                        
                        # Calculate chi-square
                        chi2, p = stats.chisquare(
                            f_obs=(aligned_sample * len(sample_data)).values,
                            f_exp=(aligned_pop * len(sample_data)).values
                        )
                        
                        comparison_statistics[col] = {
                            'chi_square': chi2,
                            'p_value': p,
                            'sample_distribution': sample_counts.to_dict(),
                            'population_distribution': pop_counts.to_dict()
                        }
            
            # If population_data is a dict of summary statistics
            elif isinstance(population_data, dict):
                for var, stats_dict in population_data.items():
                    if var in sample_data.columns:
                        if 'mean' in stats_dict and pd.api.types.is_numeric_dtype(sample_data[var]):
                            sample_mean = sample_data[var].mean()
                            pop_mean = stats_dict['mean']
                            
                            # Calculate standardized difference if std is available
                            if 'std' in stats_dict and stats_dict['std'] > 0:
                                std_diff = (sample_mean - pop_mean) / stats_dict['std']
                            else:
                                std_diff = None
                                
                            comparison_statistics[var] = {
                                'sample_mean': sample_mean,
                                'population_mean': pop_mean,
                                'standardized_difference': std_diff
                            }
        
        # Approach 2: Heckman-type selection model using selection variable
        if selection_variable is not None and selection_variable in sample_data.columns:
            try:
                # Simplified version: check correlation between selection variable and outcomes
                # In a full implementation, this would fit a selection model (e.g., Heckman)
                outcome_cols = [col for col in sample_data.columns 
                               if col != selection_variable and pd.api.types.is_numeric_dtype(sample_data[col])]
                
                for col in outcome_cols:
                    corr, p = stats.pearsonr(sample_data[selection_variable], sample_data[col])
                    comparison_statistics[f'selection_corr_{col}'] = {
                        'correlation': corr,
                        'p_value': p
                    }
                    
                    if abs(corr) > 0.3 and p < 0.05:
                        warnings.append(f"Significant correlation between selection variable and {col}: r={corr:.3f}, p={p:.4f}")
            except Exception as e:
                warnings.append(f"Error in selection model analysis: {str(e)}")
        
        # Evaluate results
        # Look for standardized differences > 0.25 (a common threshold)
        biased_variables = []
        for var, stats_dict in comparison_statistics.items():
            if 'standardized_difference' in stats_dict and stats_dict['standardized_difference'] is not None:
                if abs(stats_dict['standardized_difference']) > 0.25:
                    biased_variables.append(f"{var} (diff={stats_dict['standardized_difference']:.2f})")
            elif 'p_value' in stats_dict and stats_dict['p_value'] < 0.05:
                biased_variables.append(f"{var} (p={stats_dict['p_value']:.4f})")
        
        if not comparison_statistics:
            result = AssumptionResult.NOT_APPLICABLE
            details = "Could not assess selection bias with provided information"
        elif biased_variables:
            result = AssumptionResult.FAILED
            details = f"Selection bias detected for variables: {', '.join(biased_variables)}"
            warnings.append("Sample may not be representative of the target population")
        else:
            result = AssumptionResult.PASSED
            details = "No significant selection bias detected in analyzed variables"
        
        return {
            'result': result,
            'details': details,
            'comparison_statistics': comparison_statistics,
            'warnings': warnings,
            'figures': {
                'distribution_plot': fig_to_svg(fig_distribution)
            }
        }
class SeparabilityTest(AssumptionTest):
    """Test for separability in multifactorial designs."""
    
    def __init__(self):
        super().__init__(
            name="Separability Test",
            description="Tests for separability in multifactorial designs",
            applicable_roles=["covariate", "group"],
            applicable_types=["categorical"]
        )
        
    def run_test(self, df, outcome, factors, **kwargs):
        """
        Test for separability in multifactorial designs.
        
        Args:
            df: DataFrame with the data
            outcome: Name of outcome variable
            factors: List of factor variable names
            
        Returns:
            dict: Test results following the SeparabilityTest format
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        
        warnings = []
        alpha = kwargs.get('alpha', 0.05)
        
        # Create figure for interaction plot
        fig_interaction = plt.figure(figsize=(12, 8))
        fig_interaction.patch.set_alpha(0.0)
        ax_interaction = fig_interaction.add_subplot(111)
        ax_interaction.patch.set_alpha(0.0)
        
        # Need at least two factors
        if len(factors) < 2:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': "Need at least two factors to test separability",
                'interaction_summary': None,
                'warnings': ["Insufficient factors for separability test"],
                'figures': {
                    'interaction_plot': fig_to_svg(fig_interaction)
                }
            }
        
        # Check if outcome is numeric or can be converted to numeric
        outcome_data = df[outcome].copy()
        df_copy = df.copy()
        
        if not pd.api.types.is_numeric_dtype(outcome_data):
            # Try to convert to numeric if it contains numeric-like strings
            try:
                outcome_data = pd.to_numeric(outcome_data)
                warnings.append(f"Outcome variable '{outcome}' was converted from non-numeric to numeric format")
                # Update the dataframe with the converted outcome
                df_copy[outcome] = outcome_data
            except:
                # Check if it's categorical and encode it
                if pd.api.types.is_categorical_dtype(outcome_data) or pd.api.types.is_object_dtype(outcome_data):
                    unique_values = outcome_data.nunique()
                    if unique_values <= 10:  # Reasonable number of categories
                        # Encode categorical variable
                        from sklearn.preprocessing import LabelEncoder
                        encoder = LabelEncoder()
                        encoded_outcome = encoder.fit_transform(outcome_data)
                        df_copy[outcome] = encoded_outcome
                        warnings.append(f"Outcome variable '{outcome}' was encoded from categorical to numeric (mapping: {dict(zip(encoder.classes_, range(len(encoder.classes_))))})")
                    else:
                        return {
                            'result': AssumptionResult.NOT_APPLICABLE,
                            'details': f"Outcome variable has too many categories ({unique_values}) for encoding",
                            'interaction_summary': None,
                            'warnings': warnings,
                            'figures': {
                                'interaction_plot': fig_to_svg(fig_interaction)
                            }
                        }
                else:
                    return {
                        'result': AssumptionResult.NOT_APPLICABLE,
                        'details': "Outcome variable must be numeric or convertible to numeric",
                        'interaction_summary': None,
                        'warnings': warnings,
                        'figures': {
                            'interaction_plot': fig_to_svg(fig_interaction)
                        }
                    }
        
        # Check for sufficient data in each cell
        cross_tab = pd.crosstab(*[df_copy[f] for f in factors[:2]])
        
        min_cell_size = cross_tab.min().min()
        if min_cell_size < 3:
            warnings.append(f"Some cells have fewer than 3 observations (min={min_cell_size})")
            if min_cell_size == 0:
                warnings.append("Empty cells detected. Results may be unreliable.")
        
        # Create formula for full model with all interactions
        main_effects = " + ".join([f"C({f})" for f in factors])
        two_way_interactions = []
        
        for i in range(len(factors)):
            for j in range(i+1, len(factors)):
                two_way_interactions.append(f"C({factors[i]}):C({factors[j]})")
        
        interaction_terms = " + ".join(two_way_interactions)
        formula_with_interactions = f"{outcome} ~ {main_effects} + {interaction_terms}"
        formula_main_effects = f"{outcome} ~ {main_effects}"
        
        try:
            # Fit models with and without interactions
            model_with_int = ols(formula_with_interactions, data=df_copy).fit()
            model_main = ols(formula_main_effects, data=df_copy).fit()
            
            # Compare models
            from statsmodels.stats.anova import anova_lm
            anova_table = anova_lm(model_main, model_with_int)
            
            # Extract p-value from the last row (the interaction effect)
            p_value = float(anova_table.iloc[-1, -1])
            f_stat = float(anova_table.iloc[-1, 0])
            
            # Test individual interaction terms
            interaction_summary = {}
            for term in model_with_int.pvalues.index:
                if ":" in term:  # This is an interaction term
                    interaction_summary[term] = {
                        'p_value': float(model_with_int.pvalues[term])
                    }
            
            # Create interaction plot for first two factors
            factor1, factor2 = factors[0], factors[1]
            
            # Calculate mean of outcome for each combination of factors
            grouped = df_copy.groupby([factor1, factor2])[outcome].mean().reset_index()
            
            # Pivot for interaction plot
            pivot_table = grouped.pivot(index=factor1, columns=factor2, values=outcome)
            
            # Plot each line (one line per level of factor2)
            for col in pivot_table.columns:
                ax_interaction.plot(pivot_table.index, pivot_table[col], marker='o', label=f"{factor2}={col}")
            
            ax_interaction.set_title(f'Interaction Plot: {factor1} x {factor2}')
            ax_interaction.set_xlabel(factor1)
            ax_interaction.set_ylabel(outcome)
            ax_interaction.legend()
            
            # Determine result
            significant_interactions = [term for term, info in interaction_summary.items() 
                                      if info['p_value'] < alpha]
            
            if p_value < alpha:
                result = AssumptionResult.FAILED
                details = f"Significant interactions detected (p={p_value:.4f}). Factors are not separable."
                if significant_interactions:
                    details += f" Significant terms: {', '.join(significant_interactions)}"
                warnings.append("Non-separable factors suggest effects cannot be interpreted independently")
            else:
                result = AssumptionResult.PASSED
                details = f"No significant interactions detected (p={p_value:.4f}). Factors appear to be separable."
                
        except Exception as e:
            result = AssumptionResult.NOT_APPLICABLE
            details = f"Could not test separability: {str(e)}"
            interaction_summary = None
            warnings.append("Error in model fitting. Check data structure and factor levels.")
        
        return {
            'result': result,
            'details': details,
            'interaction_summary': interaction_summary,
            'anova_p_value': p_value if 'p_value' in locals() else None,
            'f_statistic': f_stat if 'f_stat' in locals() else None,
            'warnings': warnings,
            'figures': {
                'interaction_plot': fig_to_svg(fig_interaction)
            }
        }

class JointDistributionTest(AssumptionTest):
    """Test multivariate distributions beyond normality."""
    
    def __init__(self):
        super().__init__(
            name="Joint Distribution Test",
            description="Tests multivariate distributions beyond normality",
            applicable_roles=["covariate", "outcome"],
            applicable_types=["numeric", "continuous"]
        )
        
    def run_test(self, df, variables, **kwargs):
        """
        Test the joint distribution of variables.
        
        Args:
            df: DataFrame with the data
            variables: List of column names to test jointly
            test: Type of test to perform ('normality', 'correlation_structure')
            comparison_matrix: Optional correlation matrix to compare against (default: None, which tests against zero)
            
        Returns:
            dict: Test results following the JointDistributionTest format
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        from scipy import stats
        from scipy.stats import chi2
        import pandas as pd
        
        test_type = kwargs.get('test', 'normality')
        comparison_matrix = kwargs.get('comparison_matrix', None)
        warnings = []
        
        # Create figures
        fig_scatter = plt.figure(figsize=(10, 8))
        fig_scatter.patch.set_alpha(0.0)
        gs = fig_scatter.add_gridspec(len(variables), len(variables))
        scatter_axes = np.empty((len(variables), len(variables)), dtype=object)
        
        fig_qq = plt.figure(figsize=(10, 6))
        fig_qq.patch.set_alpha(0.0)
        ax_qq = fig_qq.add_subplot(111)
        ax_qq.patch.set_alpha(0.0)
        
        # Check if we have enough variables
        if len(variables) < 2:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': "Need at least two variables for joint distribution test",
                'test_statistic': None,
                'p_value': None,
                'warnings': ["Insufficient variables for multivariate analysis"],
                'figures': {
                    'scatter_matrix': fig_to_svg(fig_scatter),
                    'qq_plot': fig_to_svg(fig_qq)
                }
            }
        
        # Check if all variables are numeric
        if not all(pd.api.types.is_numeric_dtype(df[var]) for var in variables):
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': "All variables must be numeric for joint distribution test",
                'test_statistic': None,
                'p_value': None,
                'warnings': ["Non-numeric variables in the selected set"],
                'figures': {
                    'scatter_matrix': fig_to_svg(fig_scatter),
                    'qq_plot': fig_to_svg(fig_qq)
                }
            }
        
        # Extract the subset of variables and drop rows with missing values
        X = df[variables].dropna()
        
        # Check if we have enough observations
        if len(X) < 10:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': "Too few complete observations for joint distribution test",
                'test_statistic': None,
                'p_value': None,
                'warnings': ["Insufficient observations after removing missing values"],
                'figures': {
                    'scatter_matrix': fig_to_svg(fig_scatter),
                    'qq_plot': fig_to_svg(fig_qq)
                }
            }
        
        # Create scatter matrix
        for i in range(len(variables)):
            for j in range(len(variables)):
                scatter_axes[i, j] = fig_scatter.add_subplot(gs[i, j])
                scatter_axes[i, j].patch.set_alpha(0.0)
                
                if i == j:  # Diagonal: histograms
                    scatter_axes[i, j].hist(X[variables[i]], bins=20, color=PASTEL_COLORS[i % len(PASTEL_COLORS)])
                    if i == 0:  # First diagonal plot
                        scatter_axes[i, j].set_title(variables[i])
                else:  # Off-diagonal: scatter plots
                    scatter_axes[i, j].scatter(X[variables[j]], X[variables[i]], 
                                             color=PASTEL_COLORS[j % len(PASTEL_COLORS)], alpha=0.5)
                    
                # Turn off y-ticks for all but the leftmost column
                if j > 0:
                    scatter_axes[i, j].set_yticks([])
                # Turn off x-ticks for all but the bottom row
                if i < len(variables) - 1:
                    scatter_axes[i, j].set_xticks([])
        
        # Add variable names as labels
        for i, var in enumerate(variables):
            scatter_axes[len(variables)-1, i].set_xlabel(var)
            scatter_axes[i, 0].set_ylabel(var)
        
        fig_scatter.suptitle('Scatter Matrix')
        fig_scatter.tight_layout()
        
        # Set up test results placeholder
        test_results = {
            'test_type': test_type
        }
        
        # Perform the selected test
        if test_type == 'normality':
            # Test multivariate normality using Mardia's test
            # This is a simplified version - a full implementation would use a package like statsmodels or pingouin
            try:
                # Standardize the data
                X_std = (X - X.mean()) / X.std()
                
                # Calculate Mahalanobis distances
                cov_matrix = X_std.cov()
                try:
                    # Try to calculate inverse of covariance matrix
                    from numpy.linalg import inv
                    cov_inv = inv(cov_matrix)
                except np.linalg.LinAlgError:
                    # If singular, use pseudoinverse
                    from numpy.linalg import pinv
                    cov_inv = pinv(cov_matrix)
                    warnings.append("Covariance matrix is singular, using pseudoinverse")
                
                # Calculate squared Mahalanobis distances
                n = len(X_std)
                p = len(variables)
                
                # For each observation, calculate its Mahalanobis distance
                d_squared = np.zeros(n)
                for i in range(n):
                    x_diff = X_std.iloc[i] - X_std.mean()
                    d_squared[i] = np.dot(np.dot(x_diff, cov_inv), x_diff)
                
                # Create Q-Q plot of squared Mahalanobis distances vs. chi-square quantiles
                chi2_quantiles = chi2.ppf(np.arange(1, n+1) / (n+1), p)
                d_squared_sorted = np.sort(d_squared)
                
                ax_qq.scatter(chi2_quantiles, d_squared_sorted, color=PASTEL_COLORS[0])
                ax_qq.plot([0, max(chi2_quantiles)], [0, max(chi2_quantiles)], 'r--')
                ax_qq.set_title("Chi-Square Q-Q Plot for Multivariate Normality")
                ax_qq.set_xlabel("Chi-Square Quantiles")
                ax_qq.set_ylabel("Squared Mahalanobis Distances")
                
                # Calculate Mardia's multivariate skewness and kurtosis
                # This is a simplification - a full implementation would calculate the test statistics properly
                skewness_statistic = np.mean(d_squared**3)
                kurtosis_statistic = np.mean(d_squared**2)
                
                # Calculate p-value using chi-square approximation
                # This is a rough approximation
                skewness_p = 1 - chi2.cdf(skewness_statistic/6, p*(p+1)*(p+2)/6)
                kurtosis_p = 2 * (1 - stats.norm.cdf(abs((kurtosis_statistic - p*(p+2)) / np.sqrt(8*p*(p+2)/n))))
                
                test_results.update({
                    'skewness': {
                        'statistic': skewness_statistic,
                        'p_value': skewness_p
                    },
                    'kurtosis': {
                        'statistic': kurtosis_statistic,
                        'p_value': kurtosis_p
                    }
                })
                
                # Check if data deviates from multivariate normality
                if skewness_p < 0.05 or kurtosis_p < 0.05:
                    result = AssumptionResult.FAILED
                    details = "Data deviates significantly from multivariate normality"
                    if skewness_p < 0.05:
                        details += f" (significant skewness, p={skewness_p:.4f})"
                    if kurtosis_p < 0.05:
                        details += f" (significant kurtosis, p={kurtosis_p:.4f})"
                    warnings.append("Non-normal multivariate distribution may affect multivariate analyses")
                else:
                    result = AssumptionResult.PASSED
                    details = "Data appears to follow a multivariate normal distribution"
                
            except Exception as e:
                result = AssumptionResult.NOT_APPLICABLE
                details = f"Could not test multivariate normality: {str(e)}"
                warnings.append("Error in multivariate normality test calculation")
        
        elif test_type == 'correlation_structure':
            # Test if correlation structure is as expected
            try:
                # Calculate correlation matrix
                corr_matrix = X.corr()
                
                # Fisher's z-transformation to test correlations
                n = len(X)
                z_critical = stats.norm.ppf(0.975)  # Two-tailed 5% significance level
                
                # If comparison matrix is provided, use it; otherwise test against zero
                if comparison_matrix is not None:
                    # Validate comparison matrix
                    if not isinstance(comparison_matrix, pd.DataFrame):
                        try:
                            comparison_matrix = pd.DataFrame(comparison_matrix, 
                                                           index=variables, 
                                                           columns=variables)
                        except Exception as e:
                            warnings.append(f"Could not convert comparison matrix to DataFrame: {str(e)}")
                            comparison_matrix = None
                    
                    if comparison_matrix is not None:
                        # Test if observed correlations differ from comparison matrix
                        significant_corrs = 0
                        total_corrs = 0
                        
                        for i in range(len(variables)):
                            for j in range(i+1, len(variables)):
                                total_corrs += 1
                                observed_r = corr_matrix.iloc[i, j]
                                expected_r = comparison_matrix.loc[variables[i], variables[j]]
                                
                                # Fisher's z-transformation for the difference
                                z_observed = np.arctanh(observed_r)
                                z_expected = np.arctanh(expected_r)
                                z_diff = (z_observed - z_expected) * np.sqrt((n-3)/2)
                                
                                if abs(z_diff) > z_critical:
                                    significant_corrs += 1
                        
                        test_results.update({
                            'correlation_matrix': corr_matrix.to_dict(),
                            'comparison_matrix': comparison_matrix.to_dict(),
                            'significant_differences': significant_corrs,
                            'total_correlations': total_corrs
                        })
                        
                        if significant_corrs / max(1, total_corrs) > 0.3:
                            result = AssumptionResult.FAILED
                            details = f"{significant_corrs} out of {total_corrs} correlations differ significantly from expected values"
                            warnings.append("Correlation structure deviates from expected pattern")
                        else:
                            result = AssumptionResult.PASSED
                            details = f"Only {significant_corrs} out of {total_corrs} correlations differ from expected values"
                else:
                    # Test against zero correlation (default behavior)
                    critical_r = np.tanh(z_critical / np.sqrt(n - 3))
                    
                    # Count significant correlations
                    significant_corrs = 0
                    total_corrs = 0
                    
                    for i in range(len(variables)):
                        for j in range(i+1, len(variables)):
                            total_corrs += 1
                            if abs(corr_matrix.iloc[i, j]) > critical_r:
                                significant_corrs += 1
                    
                    test_results.update({
                        'correlation_matrix': corr_matrix.to_dict(),
                        'significant_correlations': significant_corrs,
                        'total_correlations': total_corrs,
                        'critical_r': critical_r,
                        'comparison': 'zero correlation'
                    })
                    
                    # Determine result based on proportion of significant correlations
                    if significant_corrs / max(1, total_corrs) > 0.3:  # If more than 30% are significant
                        result = AssumptionResult.WARNING
                        details = f"{significant_corrs} out of {total_corrs} correlations are significantly different from zero"
                        warnings.append("Strong correlation structure detected, consider multivariate methods")
                    else:
                        result = AssumptionResult.PASSED
                        details = f"Only {significant_corrs} out of {total_corrs} correlations are significantly different from zero"
                
            except Exception as e:
                result = AssumptionResult.NOT_APPLICABLE
                details = f"Could not test correlation structure: {str(e)}"
                warnings.append("Error in correlation structure test calculation")
                
        else:
            result = AssumptionResult.NOT_APPLICABLE
            details = f"Unknown test type: {test_type}"
            warnings.append(f"Valid test types are 'normality' and 'correlation_structure'")
        
        return {
            'result': result,
            'details': details,
            'test_results': test_results,
            'warnings': warnings,
            'figures': {
                'scatter_matrix': fig_to_svg(fig_scatter),
                'qq_plot': fig_to_svg(fig_qq)
            }
        }
    

class AutocorrelationTest(AssumptionTest):
    """Tests for autocorrelation in time series or regression residuals."""

    def __init__(self):
        super().__init__(
            name="Autocorrelation Test",
            description="Tests whether residuals are autocorrelated (dependent on previous values).",
            applicable_roles=["residuals"],
            applicable_types=["continuous"]
        )

    def run_test(self, residuals, **kwargs):
        """
        Run the Durbin-Watson test for autocorrelation of residuals.
        
        Parameters:
        -----------
        residuals : array-like
            The residuals from a regression model
        
        Returns:
        --------
        dict
            Dictionary containing test results following the AutocorrelationTest format
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from statsmodels.stats.stattools import durbin_watson
        from statsmodels.graphics.tsaplots import plot_acf
        
        alpha = kwargs.get('alpha', 0.05)
        warnings = []
        
        # Create figures
        fig_acf = plt.figure(figsize=(10, 6))
        fig_acf.patch.set_alpha(0.0)
        ax_acf = fig_acf.add_subplot(111)
        ax_acf.patch.set_alpha(0.0)
        
        fig_residual = plt.figure(figsize=(10, 6))
        fig_residual.patch.set_alpha(0.0)
        ax_residual = fig_residual.add_subplot(111)
        ax_residual.patch.set_alpha(0.0)
        
        try:
            residuals = np.array(residuals).flatten()
            dw_stat = durbin_watson(residuals)
            
            # Calculate ACF values
            from statsmodels.tsa.stattools import acf
            acf_values = acf(residuals, nlags=min(20, len(residuals)//2), fft=True)
            
            # Critical values for Durbin-Watson test
            critical_values = {
                'lower_bound': 1.5,
                'upper_bound': 2.5,
                'severe_lower': 1.0,
                'severe_upper': 3.0
            }
            
            # Interpret Durbin-Watson statistic
            # DW ≈ 2: No autocorrelation
            # DW < 1.5: Positive autocorrelation
            # DW > 2.5: Negative autocorrelation
            if 1.5 <= dw_stat <= 2.5:
                result = AssumptionResult.PASSED
                details = f"No significant autocorrelation detected (Durbin-Watson = {dw_stat:.3f})."
            elif dw_stat < 1.0 or dw_stat > 3.0:
                result = AssumptionResult.FAILED
                details = f"Strong autocorrelation detected (Durbin-Watson = {dw_stat:.3f})."
                warnings.append("Autocorrelation may invalidate standard errors and hypothesis tests")
            else:
                result = AssumptionResult.WARNING
                details = f"Possible autocorrelation detected (Durbin-Watson = {dw_stat:.3f})."
                warnings.append("Consider using robust standard errors or time series models")
            
            # Plot ACF
            plot_acf(residuals, ax=ax_acf, lags=min(20, len(residuals)//2), alpha=0.05)
            ax_acf.set_title('Autocorrelation Function')
            
            # Plot residuals over time/sequence
            ax_residual.plot(range(len(residuals)), residuals, color=PASTEL_COLORS[0])
            ax_residual.set_title('Residuals Over Time/Sequence')
            ax_residual.set_xlabel('Sequence')
            ax_residual.set_ylabel('Residual')
            ax_residual.axhline(y=0, color='r', linestyle='-')
            
            return {
                "result": result,
                "statistic": dw_stat,
                "details": details,
                "acf_values": acf_values.tolist(),
                "critical_values": critical_values,
                "warnings": warnings,
                "figures": {
                    "acf_plot": fig_to_svg(fig_acf),
                    "residual_plot": fig_to_svg(fig_residual)
                }
            }
            
        except Exception as e:
            return {
                "result": AssumptionResult.NOT_APPLICABLE,
                "statistic": None,
                "details": f"Could not perform autocorrelation test: {str(e)}",
                "acf_values": [],
                "critical_values": {},
                "warnings": [f"Error in autocorrelation test: {str(e)}"],
                "figures": {
                    "acf_plot": fig_to_svg(fig_acf),
                    "residual_plot": fig_to_svg(fig_residual)
                }
            }
        
class ProportionalHazardsTest(AssumptionTest):
    """Tests the proportional hazards assumption in Cox regression models."""

    def __init__(self):
        super().__init__(
            name="Proportional Hazards Test",
            description="Tests the proportional hazards assumption in survival analysis.",
            applicable_roles=["time", "event", "covariates"],
            applicable_types=["continuous", "categorical"]
        )

    def run_test(self, time, event, covariates, **kwargs):
        """
        Test the proportional hazards assumption using Schoenfeld residuals.
        
        Parameters:
        -----------
        time : array-like
            Survival times or time-to-event data
        event : array-like
            Event indicator (1=event, 0=censored)
        covariates : DataFrame
            Covariate data for the Cox model
        
        Returns:
        --------
        dict
            Dictionary containing test results following the ProportionalHazardsTest format
        """
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            from lifelines import CoxPHFitter
            from lifelines.statistics import proportional_hazard_test
            
            alpha = kwargs.get('alpha', 0.05)
            warnings = []
            
            # Create figure for Schoenfeld residuals plot
            fig_schoenfeld = plt.figure(figsize=(12, 8))
            fig_schoenfeld.patch.set_alpha(0.0)
            
            # Prepare data for Cox model
            if isinstance(covariates, pd.DataFrame):
                df = covariates.copy()
            else:
                df = pd.DataFrame(covariates)
            
            df['time'] = time
            df['event'] = event
            
            # Fit Cox model
            cph = CoxPHFitter()
            cph.fit(df, duration_col='time', event_col='event')
            
            # Test proportional hazards assumption
            ph_results = proportional_hazard_test(cph, df, time_transform='rank')
            
            # Extract results
            global_p_value = ph_results.p
            global_statistic = ph_results.test_statistic
            
            # Create dictionary for per-covariate results
            covariate_results = {}
            for i, covariate in enumerate(ph_results.names):
                covariate_results[covariate] = {
                    'p_value': ph_results.p_values[i],
                    'statistic': ph_results.test_statistics[i]
                }
            
            # Determine overall result
            if global_p_value > alpha:
                result = AssumptionResult.PASSED
                details = f"Proportional hazards assumption satisfied (p={global_p_value:.4f})"
            else:
                result = AssumptionResult.FAILED
                details = f"Proportional hazards assumption violated (p={global_p_value:.4f})"
                
                # Add warnings for specific covariates that violate the assumption
                for covariate, res in covariate_results.items():
                    if res['p_value'] < alpha:
                        warnings.append(f"Covariate '{covariate}' violates proportional hazards (p={res['p_value']:.4f})")
            
            # Plot Schoenfeld residuals
            try:
                ax = fig_schoenfeld.add_subplot(111)
                cph.check_assumptions(df, show_plots=True, ax=ax)
                ax.set_title('Schoenfeld Residuals')
            except Exception as e:
                warnings.append(f"Could not create Schoenfeld residuals plot: {str(e)}")
            
            return {
                'result': result,
                'statistic': global_statistic,
                'p_value': global_p_value,
                'details': details,
                'covariate_results': covariate_results,
                'warnings': warnings,
                'figures': {
                    'schoenfeld_plot': fig_to_svg(fig_schoenfeld)
                }
            }
            
        except Exception as e:
            fig_schoenfeld = plt.figure(figsize=(12, 8))
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'statistic': None,
                'p_value': None,
                'details': f"Could not perform proportional hazards test: {str(e)}",
                'covariate_results': {},
                'warnings': [f"Error in proportional hazards test: {str(e)}"],
                'figures': {
                    'schoenfeld_plot': fig_to_svg(fig_schoenfeld)
                }
            }
        
class GoodnessOfFitTest(AssumptionTest):
    """Tests the goodness of fit for statistical models."""

    def __init__(self):
        super().__init__(
            name="Goodness of Fit Test",
            description="Tests how well a model fits the data.",
            applicable_roles=["observed", "expected"],
            applicable_types=["continuous", "categorical"]
        )

    def run_test(self, observed, expected=None, **kwargs):
        """
        Test goodness of fit using appropriate test based on data type.
        
        Parameters:
        -----------
        observed : array-like
            Observed values or categorical data
        expected : array-like, optional
            Expected values (for chi-square test)
        **kwargs : dict
            Additional parameters including:
            - model_type: str, type of model ('logistic', 'categorical', etc.)
            - predicted: array-like, predicted probabilities for Hosmer-Lemeshow
        
        Returns:
        --------
        dict
            Dictionary containing test results following the GoodnessOfFitTest format
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        model_type = kwargs.get('model_type', 'categorical')
        alpha = kwargs.get('alpha', 0.05)
        warnings = []
        
        # Create figure for observed vs expected plot
        fig_obs_exp = plt.figure(figsize=(10, 6))
        fig_obs_exp.patch.set_alpha(0.0)
        ax_obs_exp = fig_obs_exp.add_subplot(111)
        ax_obs_exp.patch.set_alpha(0.0)
        
        try:
            if model_type == 'logistic':
                result = self._run_hosmer_lemeshow_test(observed, kwargs.get('predicted'), **kwargs)
                
                # Add plot for Hosmer-Lemeshow test
                predicted = kwargs.get('predicted')
                if predicted is not None:
                    # Create decile groups
                    groups = 10
                    indices = np.argsort(predicted)
                    sorted_observed = np.array(observed)[indices]
                    sorted_predicted = np.array(predicted)[indices]
                    
                    # Calculate observed and expected frequencies by group
                    group_size = len(sorted_observed) // groups
                    observed_freqs = []
                    expected_freqs = []
                    
                    for i in range(groups):
                        start = i * group_size
                        end = (i + 1) * group_size if i < groups - 1 else len(sorted_observed)
                        group_observed = np.mean(sorted_observed[start:end])
                        group_expected = np.mean(sorted_predicted[start:end])
                        observed_freqs.append(group_observed)
                        expected_freqs.append(group_expected)
                    
                    # Plot observed vs expected
                    ax_obs_exp.scatter(expected_freqs, observed_freqs, color=PASTEL_COLORS[0])
                    ax_obs_exp.plot([0, 1], [0, 1], 'r--')
                    ax_obs_exp.set_xlabel('Expected Proportion')
                    ax_obs_exp.set_ylabel('Observed Proportion')
                    ax_obs_exp.set_title('Hosmer-Lemeshow Goodness of Fit')
                    
                    # Add to result
                    result['figures'] = {'observed_expected_plot': fig_to_svg(fig_obs_exp)}
                    result['test_type'] = 'hosmer_lemeshow'
                    
                return result
            else:
                result = self._run_chi_square_test(observed, expected, **kwargs)
                
                # Add plot for chi-square test
                if expected is not None:
                    # Plot observed vs expected
                    x = np.arange(len(observed))
                    width = 0.35
                    ax_obs_exp.bar(x - width/2, observed, width, label='Observed', color=PASTEL_COLORS[0])
                    ax_obs_exp.bar(x + width/2, expected, width, label='Expected', color=PASTEL_COLORS[1])
                    ax_obs_exp.set_xlabel('Category')
                    ax_obs_exp.set_ylabel('Frequency')
                    ax_obs_exp.set_title('Chi-Square Goodness of Fit')
                    ax_obs_exp.set_xticks(x)
                    ax_obs_exp.legend()
                    
                    # Add to result
                    result['figures'] = {'observed_expected_plot': fig_to_svg(fig_obs_exp)}
                    result['test_type'] = 'chi_square'
                    
                return result
        
        except Exception as e:
            return {
                "result": AssumptionResult.NOT_APPLICABLE,
                "statistic": None,
                "p_value": None,
                "details": f"Error in goodness of fit test: {str(e)}",
                "test_type": model_type,
                "df": None,
                "residuals": [],
                "warnings": [f"Error in goodness of fit test: {str(e)}"],
                "figures": {
                    "observed_expected_plot": fig_to_svg(fig_obs_exp)
                }
            }
    
    def _run_chi_square_test(self, observed, expected=None, **kwargs):
        """Run a chi-square goodness of fit test."""
        import numpy as np
        from scipy import stats
        
        alpha = kwargs.get('alpha', 0.05)
        warnings = []
        
        # Convert inputs to numpy arrays
        observed = np.array(observed)
        
        # If expected is not provided, assume uniform distribution
        if expected is None:
            expected = np.ones_like(observed) * np.sum(observed) / len(observed)
        else:
            expected = np.array(expected)
        
        # Calculate chi-square statistic and p-value
        chi2_stat, p_value = stats.chisquare(observed, expected)
        df = len(observed) - 1
        
        # Calculate standardized residuals
        residuals = (observed - expected) / np.sqrt(expected)
        
        # Determine result
        if p_value > alpha:
            result = AssumptionResult.PASSED
            details = f"Model fits the data well (Chi-square = {chi2_stat:.3f}, p = {p_value:.4f}, df = {df})"
        else:
            result = AssumptionResult.FAILED
            details = f"Model does not fit the data well (Chi-square = {chi2_stat:.3f}, p = {p_value:.4f}, df = {df})"
            warnings.append("Consider revising the model or exploring alternative models")
            
            # Check for categories with large residuals
            large_residuals = np.where(np.abs(residuals) > 2)[0]
            if len(large_residuals) > 0:
                categories = ", ".join([str(i) for i in large_residuals])
                warnings.append(f"Large discrepancies in categories: {categories}")
        
        return {
            "result": result,
            "statistic": chi2_stat,
            "p_value": p_value,
            "details": details,
            "test_type": "chi_square",
            "df": df,
            "residuals": residuals.tolist(),
            "warnings": warnings
        }
    
    def _run_hosmer_lemeshow_test(self, observed, predicted, **kwargs):
        """Run Hosmer-Lemeshow goodness of fit test for logistic regression."""
        import numpy as np
        from scipy import stats
        
        alpha = kwargs.get('alpha', 0.05)
        warnings = []
        
        if predicted is None:
            return {
                "result": AssumptionResult.NOT_APPLICABLE,
                "statistic": None,
                "p_value": None,
                "details": "Predicted probabilities are required for Hosmer-Lemeshow test",
                "test_type": "hosmer_lemeshow",
                "df": None,
                "residuals": [],
                "warnings": ["Missing predicted probabilities for Hosmer-Lemeshow test"]
            }
        
        # Convert inputs to numpy arrays
        observed = np.array(observed)
        predicted = np.array(predicted)
        
        # Create groups (typically 10)
        groups = 10
        indices = np.argsort(predicted)
        sorted_observed = observed[indices]
        sorted_predicted = predicted[indices]
        
        # Calculate observed and expected frequencies by group
        group_size = len(sorted_observed) // groups
        observed_freqs = []
        expected_freqs = []
        
        for i in range(groups):
            start = i * group_size
            end = (i + 1) * group_size if i < groups - 1 else len(sorted_observed)
            group_observed = np.sum(sorted_observed[start:end])
            group_expected = np.sum(sorted_predicted[start:end])
            observed_freqs.append(group_observed)
            expected_freqs.append(group_expected)
        
        # Calculate Hosmer-Lemeshow statistic
        hl_stat = np.sum((np.array(observed_freqs) - np.array(expected_freqs))**2 / 
                         (np.array(expected_freqs) * (1 - np.array(expected_freqs)/group_size)))
        df = groups - 2
        p_value = 1 - stats.chi2.cdf(hl_stat, df)
        
        # Calculate residuals
        residuals = (np.array(observed_freqs) - np.array(expected_freqs)) / np.sqrt(np.array(expected_freqs))
        
        # Determine result
        if p_value > alpha:
            result = AssumptionResult.PASSED
            details = f"Logistic model fits the data well (H-L = {hl_stat:.3f}, p = {p_value:.4f}, df = {df})"
        else:
            result = AssumptionResult.FAILED
            details = f"Logistic model does not fit the data well (H-L = {hl_stat:.3f}, p = {p_value:.4f}, df = {df})"
            warnings.append("Consider revising the logistic regression model")
        
        return {
            "result": result,
            "statistic": hl_stat,
            "p_value": p_value,
            "details": details,
            "test_type": "hosmer_lemeshow",
            "df": df,
            "residuals": residuals.tolist(),
            "warnings": warnings
        }

class SphericityTest(AssumptionTest):
    """Tests for sphericity in repeated measures designs."""

    def __init__(self):
        super().__init__(
            name="Sphericity Test",
            description="Tests whether variances of differences between all combinations of levels are equal.",
            applicable_roles=["within_factor", "subject_id", "outcome"],
            applicable_types=["continuous", "categorical"]
        )

    def run_test(self, data, subject_id, within_factor, outcome, **kwargs):
        """
        Run Mauchly's test of sphericity for repeated measures ANOVA.
        
        Parameters:
        -----------
        data : DataFrame
            Data containing the measurements
        subject_id : str
            Column name for subject identifier
        within_factor : str
            Column name for within-subjects factor
        outcome : str
            Column name for outcome variable
        
        Returns:
        --------
        dict
            Dictionary containing test results
        """
        import pandas as pd
        import numpy as np
        from scipy import stats
        
        alpha = kwargs.get('alpha', 0.05)
        
        try:
            # Reshape data to wide format for sphericity testing
            wide_data = data.pivot(index=subject_id, columns=within_factor, values=outcome)
            
            # Get the levels of the within-subject factor
            levels = wide_data.columns.tolist()
            n_levels = len(levels)
            
            if n_levels < 3:
                return {
                    "result": AssumptionResult.NOT_APPLICABLE,
                    "message": "Sphericity test requires at least 3 levels of the within-subjects factor.",
                    "statistic": None,
                    "details": {"error": "Not enough levels for sphericity test"}
                }
            
            # Calculate all pairwise difference variances
            difference_vars = []
            for i in range(n_levels):
                for j in range(i+1, n_levels):
                    diff = wide_data[levels[i]] - wide_data[levels[j]]
                    difference_vars.append(np.var(diff, ddof=1))
            
            # Calculate coefficient of variation of these variances
            cv = np.std(difference_vars, ddof=1) / np.mean(difference_vars)
            
            # Approximate Mauchly's test using Greenhouse-Geisser epsilon
            n_subjects = len(wide_data)
            X = wide_data.to_numpy()
            
            # Center the data
            X = X - np.mean(X, axis=0)
            
            # Calculate covariance matrix
            S = np.cov(X, rowvar=False)
            
            # Calculate sphericity statistic (Greenhouse-Geisser epsilon)
            # This is a simplified approximation of the GG correction
            epsilon = np.sum(S) ** 2 / ((n_levels ** 2) * np.sum(S ** 2))
            
            # Mauchly's W approximation
            # For proper Mauchly's test, we'd need to calculate the determinant
            # of the sphericity matrix, but this is a simplified version
            W = epsilon * n_levels / (n_levels - 1)
            
            # Approximate chi-squared statistic
            df = (n_levels * (n_levels + 1)) / 2 - 1
            chi2 = -(n_subjects - 1) * np.log(W)
            p_value = 1 - stats.chi2.cdf(chi2, df)
            
            if p_value >= alpha:
                result = AssumptionResult.PASSED
                message = f"Sphericity assumption is met (p={p_value:.4f})."
            else:
                if epsilon > 0.75:
                    result = AssumptionResult.WARNING
                    message = f"Sphericity violated (p={p_value:.4f}), but Greenhouse-Geisser correction is mild (ε={epsilon:.3f})."
                else:
                    result = AssumptionResult.FAILED
                    message = f"Sphericity violated (p={p_value:.4f}), consider using Greenhouse-Geisser correction (ε={epsilon:.3f})."
            
            return {
                "result": result,
                "message": message,
                "statistic": chi2,
                "details": {
                    "mauchly_w": W,
                    "chi_square": chi2,
                    "df": df,
                    "p_value": p_value,
                    "greenhouse_geisser_epsilon": epsilon,
                    "levels": levels,
                    "method": "Mauchly's test (approximated)"
                }
            }
            
        except Exception as e:
            return {
                "result": AssumptionResult.FAILED,
                "message": f"Error in sphericity test: {str(e)}",
                "statistic": None,
                "details": {"error": str(e)}
            }

class ParameterStabilityTest(AssumptionTest):
    """Tests parameter stability across subsamples or over time."""

    def __init__(self):
        super().__init__(
            name="Parameter Stability Test",
            description="Tests whether regression coefficients are stable across subsamples or time",
            applicable_roles=["covariates", "outcome", "time"],
            applicable_types=["continuous", "categorical"]
        )

    def run_test(self, x, y, split_point=None, **kwargs):
        """
        Test parameter stability using Chow test or CUSUM test.
        
        Parameters:
        -----------
        x : DataFrame or array-like
            Predictor variables (design matrix)
        y : array-like
            Response variable
        split_point : int or float, optional
            Point at which to split the sample. If float between 0-1, interpreted as proportion.
            If None, automatically find the optimal breakpoint.
        method : str, optional
            'chow' for Chow test or 'cusum' for CUSUM test. Default is 'chow'.
            
        Returns:
        --------
        dict
            Dictionary containing test results
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import recursive_olsresiduals
        
        method = kwargs.get('method', 'chow')
        alpha = kwargs.get('alpha', 0.05)
        warnings = []
        
        # Create figures
        fig_stability = plt.figure(figsize=(10, 6))
        fig_stability.patch.set_alpha(0.0)
        ax_stability = fig_stability.add_subplot(111)
        ax_stability.patch.set_alpha(0.0)
        
        # Convert inputs to appropriate format
        if isinstance(x, pd.DataFrame):
            X = x.copy()
        else:
            X = np.asarray(x)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
        
        y = np.asarray(y)
        n_obs = len(y)
        
        # Determine split point if not provided
        if split_point is None:
            # Default to middle of the sample
            split_point = n_obs // 2
        elif 0 < split_point < 1:
            # Interpret as proportion
            split_point = int(n_obs * split_point)
        
        # Ensure split point is valid
        if split_point <= 0 or split_point >= n_obs:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'statistic': None,
                'p_value': None,
                'details': "Invalid split point specified",
                'warnings': ["Split point must be between 0 and the number of observations"],
                'figures': {
                    'stability_plot': fig_to_svg(fig_stability)
                }
            }
        
        try:
            if method == 'chow':
                # Add constant term
                X_with_const = sm.add_constant(X)
                
                # Fit full model
                full_model = sm.OLS(y, X_with_const).fit()
                
                # Split data
                X1 = X_with_const[:split_point]
                y1 = y[:split_point]
                X2 = X_with_const[split_point:]
                y2 = y[split_point:]
                
                # Fit models on each subsample
                model1 = sm.OLS(y1, X1).fit()
                model2 = sm.OLS(y2, X2).fit()
                
                # Calculate RSS for each model
                rss_full = full_model.ssr
                rss_1 = model1.ssr
                rss_2 = model2.ssr
                
                # Calculate Chow statistic
                k = X_with_const.shape[1]  # Number of parameters
                chow_stat = ((rss_full - (rss_1 + rss_2)) / k) / ((rss_1 + rss_2) / (n_obs - 2 * k))
                p_value = 1 - stats.f.cdf(chow_stat, k, n_obs - 2 * k)
                # Check for significance
                if p_value > alpha:
                    result = AssumptionResult.PASSED
                    details = f"Parameters appear stable across subsamples (Chow test: F={chow_stat:.3f}, p={p_value:.4f})"
                else:
                    result = AssumptionResult.FAILED
                    details = f"Evidence of parameter instability (Chow test: F={chow_stat:.3f}, p={p_value:.4f})"
                    warnings.append("Consider using time-varying parameters or separate models for subsamples")
                
                # Plot coefficient comparison
                coef_names = ["Constant"] + [f"X{i+1}" for i in range(X.shape[1])]
                coef1 = model1.params
                coef2 = model2.params
                
                x_pos = np.arange(len(coef_names))
                width = 0.35
                
                ax_stability.bar(x_pos - width/2, coef1, width, label='Subsample 1', color=PASTEL_COLORS[0])
                ax_stability.bar(x_pos + width/2, coef2, width, label='Subsample 2', color=PASTEL_COLORS[1])
                
                ax_stability.set_xlabel('Coefficient')
                ax_stability.set_ylabel('Value')
                ax_stability.set_title('Coefficient Comparison Between Subsamples')
                ax_stability.set_xticks(x_pos)
                ax_stability.set_xticklabels(coef_names)
                ax_stability.legend()
                
                return {
                    'result': result,
                    'statistic': chow_stat,
                    'p_value': p_value,
                    'details': details,
                    'method': 'Chow test',
                    'split_point': split_point,
                    'subsample1_coef': coef1.tolist(),
                    'subsample2_coef': coef2.tolist(),
                    'warnings': warnings,
                    'figures': {
                        'stability_plot': fig_to_svg(fig_stability)
                    }
                }
                
            elif method == 'cusum':
                # Add constant term
                X_with_const = sm.add_constant(X)
                
                # Calculate recursive residuals and CUSUM test
                rresid, rparams, rstderr = recursive_olsresiduals(sm.OLS(y, X_with_const).fit(), return_constituents=True)
                
                # Normalize recursive residuals
                nobs = rresid.shape[0]
                sigma = np.std(rresid, ddof=1)
                cusumols = np.cumsum(rresid/sigma) / np.sqrt(nobs)
                
                # Calculate CUSUM bounds
                k = X_with_const.shape[1]
                crit = 0.948  # 5% critical value for CUSUM test
                bound = crit * np.sqrt(np.arange(k, nobs + 1) / nobs)
                
                # Check if CUSUM crosses bounds
                exceeds_bounds = np.any(np.abs(cusumols) > bound)
                
                if exceeds_bounds:
                    result = AssumptionResult.FAILED
                    details = "CUSUM test indicates parameter instability"
                    warnings.append("Consider using time-varying parameters or structural break models")
                else:
                    result = AssumptionResult.PASSED
                    details = "CUSUM test indicates parameters are stable"
                
                # Plot CUSUM
                ax_stability.plot(range(k, nobs + 1), cusumols, label='CUSUM', color=PASTEL_COLORS[0])
                ax_stability.plot(range(k, nobs + 1), bound, 'r--', label='5% Significance Bound')
                ax_stability.plot(range(k, nobs + 1), -bound, 'r--')
                ax_stability.set_title('CUSUM Test for Parameter Stability')
                ax_stability.set_xlabel('Observation')
                ax_stability.set_ylabel('CUSUM')
                ax_stability.legend()
                
                return {
                    'result': result,
                    'statistic': np.max(np.abs(cusumols)),
                    'p_value': None,  # CUSUM test doesn't have a direct p-value
                    'details': details,
                    'method': 'CUSUM test',
                    'warnings': warnings,
                    'figures': {
                        'stability_plot': fig_to_svg(fig_stability)
                    }
                }
                
            else:
                raise ValueError(f"Unknown method: {method}. Use 'chow' or 'cusum'.")
                
        except Exception as e:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'statistic': None,
                'p_value': None,
                'details': f"Error in parameter stability test: {str(e)}",
                'warnings': [f"Test failed with error: {str(e)}"],
                'figures': {
                    'stability_plot': fig_to_svg(fig_stability)
                }
            }

class SerialCorrelationTest(AssumptionTest):
    """Advanced tests for serial correlation in time series or regression residuals."""

    def __init__(self):
        super().__init__(
            name="Serial Correlation Test",
            description="Comprehensive tests for serial correlation in time series data or model residuals",
            applicable_roles=["residuals", "time_series"],
            applicable_types=["continuous"]
        )

    def run_test(self, data, lags=None, **kwargs):
        """
        Run comprehensive serial correlation tests.
        
        Parameters:
        -----------
        data : array-like
            Time series data or residuals to test for serial correlation
        lags : int, optional
            Number of lags to test. Default is min(10, n/5)
        method : str, optional
            Test method: 'breusch_godfrey', 'ljung_box', or 'all'. Default is 'all'
            
        Returns:
        --------
        dict
            Dictionary containing test results
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import acorr_breusch_godfrey
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        method = kwargs.get('method', 'all')
        alpha = kwargs.get('alpha', 0.05)
        warnings = []
        
        # Create figures
        fig_acf = plt.figure(figsize=(10, 6))
        fig_acf.patch.set_alpha(0.0)
        ax_acf = fig_acf.add_subplot(111)
        ax_acf.patch.set_alpha(0.0)
        
        fig_pacf = plt.figure(figsize=(10, 6))
        fig_pacf.patch.set_alpha(0.0)
        ax_pacf = fig_pacf.add_subplot(111)
        ax_pacf.patch.set_alpha(0.0)
        
        # Convert data to array
        data = np.asarray(data).flatten()
        n = len(data)
        
        # Determine number of lags
        if lags is None:
            lags = min(10, int(n/5))
        
        # Ensure lags is not too large
        if lags >= n:
            lags = n // 2
            warnings.append(f"Reduced lags to {lags} due to sample size constraints")
        
        try:
            # Create a DataFrame for results
            results_df = pd.DataFrame(columns=['Test', 'Lag', 'Statistic', 'P-Value', 'Result'])
            
            # Run Breusch-Godfrey test if requested
            if method in ['breusch_godfrey', 'all']:
                try:
                    # Need to run a regression model and get residuals for B-G test
                    # Using an AR(1) model as a simple example
                    X = sm.add_constant(np.zeros_like(data))  # Dummy regressors
                    model = sm.OLS(data, X).fit()
                    
                    # Run B-G test
                    bg_result = acorr_breusch_godfrey(model, nlags=lags)
                    
                    # Add to results DataFrame
                    results_df = pd.concat([results_df, pd.DataFrame({
                        'Test': ['Breusch-Godfrey'],
                        'Lag': [lags],
                        'Statistic': [bg_result[0]],
                        'P-Value': [bg_result[1]],
                        'Result': ['Reject H₀' if bg_result[1] < alpha else 'Fail to reject H₀']
                    })], ignore_index=True)
                    
                    if bg_result[1] < alpha:
                        warnings.append(f"Breusch-Godfrey test indicates serial correlation (p={bg_result[1]:.4f})")
                except Exception as e:
                    warnings.append(f"Error in Breusch-Godfrey test: {str(e)}")
            
            # Run Ljung-Box test if requested
            if method in ['ljung_box', 'all']:
                try:
                    lb_result = acorr_ljungbox(data, lags=[lags])
                    
                    # Add to results DataFrame
                    results_df = pd.concat([results_df, pd.DataFrame({
                        'Test': ['Ljung-Box'],
                        'Lag': [lags],
                        'Statistic': [lb_result.iloc[0, 0]],
                        'P-Value': [lb_result.iloc[0, 1]],
                        'Result': ['Reject H₀' if lb_result.iloc[0, 1] < alpha else 'Fail to reject H₀']
                    })], ignore_index=True)
                    
                    if lb_result.iloc[0, 1] < alpha:
                        warnings.append(f"Ljung-Box test indicates serial correlation (p={lb_result.iloc[0, 1]:.4f})")
                except Exception as e:
                    warnings.append(f"Error in Ljung-Box test: {str(e)}")
            
            # Plot ACF and PACF
            try:
                from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
                
                plot_acf(data, lags=lags, alpha=0.05, ax=ax_acf)
                ax_acf.set_title('Autocorrelation Function (ACF)')
                
                plot_pacf(data, lags=lags, alpha=0.05, ax=ax_pacf)
                ax_pacf.set_title('Partial Autocorrelation Function (PACF)')
            except Exception as e:
                warnings.append(f"Error plotting ACF/PACF: {str(e)}")
                ax_acf.text(0.5, 0.5, "Error plotting ACF", horizontalalignment='center')
                ax_pacf.text(0.5, 0.5, "Error plotting PACF", horizontalalignment='center')
            
            # Determine overall result
            if results_df.empty:
                result = AssumptionResult.NOT_APPLICABLE
                details = "Could not perform serial correlation tests"
            elif (results_df['P-Value'] < alpha).any():
                result = AssumptionResult.FAILED
                details = "Evidence of serial correlation detected"
                warnings.append("Serial correlation may invalidate standard errors and hypothesis tests")
            else:
                result = AssumptionResult.PASSED
                details = "No significant serial correlation detected"
            
            return {
                'result': result,
                'details': details,
                'results_table': results_df,
                'lags': lags,
                'warnings': warnings,
                'figures': {
                    'acf_plot': fig_to_svg(fig_acf),
                    'pacf_plot': fig_to_svg(fig_pacf)
                }
            }
            
        except Exception as e:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': f"Error in serial correlation test: {str(e)}",
                'warnings': [f"Test failed with error: {str(e)}"],
                'figures': {
                    'acf_plot': fig_to_svg(fig_acf),
                    'pacf_plot': fig_to_svg(fig_pacf)
                }
            }

class InformationCriteriaTest(AssumptionTest):
    """Calculates and compares information criteria for model selection."""

    def __init__(self):
        super().__init__(
            name="Information Criteria Comparison",
            description="Calculates AIC, BIC, and other information criteria for model selection",
            applicable_roles=["model"],
            applicable_types=["model"]
        )

    def run_test(self, models, **kwargs):
        """
        Calculate and compare information criteria for model selection.
        
        Parameters:
        -----------
        models : list or dict
            List of fitted statsmodels models or dictionary of models with names as keys
            
        Returns:
        --------
        dict
            Dictionary containing test results
        """
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        
        warnings = []
        
        # Create figure for comparison plot
        fig_ic = plt.figure(figsize=(10, 6))
        fig_ic.patch.set_alpha(0.0)
        ax_ic = fig_ic.add_subplot(111)
        ax_ic.patch.set_alpha(0.0)
        
        try:
            # Create empty DataFrame for results
            columns = ['Model', 'AIC', 'BIC', 'Log-Likelihood', 'Parameters', 'N']
            results_df = pd.DataFrame(columns=columns)
            
            # Process models
            if isinstance(models, dict):
                model_items = models.items()
            else:
                model_items = [(f"Model {i+1}", model) for i, model in enumerate(models)]
            
            # Extract information criteria for each model
            for name, model in model_items:
                try:
                    model_row = {
                        'Model': name,
                        'AIC': getattr(model, 'aic', None),
                        'BIC': getattr(model, 'bic', None),
                        'Log-Likelihood': getattr(model, 'llf', None),
                        'Parameters': getattr(model, 'df_model', None),
                        'N': getattr(model, 'nobs', None)
                    }
                    
                    # Handle alternative attribute names
                    if model_row['AIC'] is None and hasattr(model, 'AIC'):
                        model_row['AIC'] = model.AIC
                    if model_row['BIC'] is None and hasattr(model, 'BIC'):
                        model_row['BIC'] = model.BIC
                    if model_row['Log-Likelihood'] is None and hasattr(model, 'loglike'):
                        model_row['Log-Likelihood'] = model.loglike
                    if model_row['Parameters'] is None and hasattr(model, 'k_params'):
                        model_row['Parameters'] = model.k_params
                    if model_row['N'] is None and hasattr(model, 'n'):
                        model_row['N'] = model.n
                    
                    # Add to results DataFrame
                    results_df = pd.concat([results_df, pd.DataFrame([model_row])], ignore_index=True)
                    
                except Exception as e:
                    warnings.append(f"Error processing model '{name}': {str(e)}")
            
            if results_df.empty:
                return {
                    'result': AssumptionResult.NOT_APPLICABLE,
                    'details': "Could not extract information criteria from any models",
                    'warnings': warnings,
                    'figures': {
                        'ic_plot': fig_to_svg(fig_ic)
                    }
                }
            
            # Find the best model according to AIC and BIC
            aic_cols = [col for col in results_df.columns if 'AIC' in col]
            bic_cols = [col for col in results_df.columns if 'BIC' in col]
            
            if aic_cols:
                best_aic_idx = results_df[aic_cols[0]].idxmin()
                best_aic_model = results_df.loc[best_aic_idx, 'Model']
            else:
                best_aic_model = None
                warnings.append("AIC not available for model comparison")
            
            if bic_cols:
                best_bic_idx = results_df[bic_cols[0]].idxmin()
                best_bic_model = results_df.loc[best_bic_idx, 'Model']
            else:
                best_bic_model = None
                warnings.append("BIC not available for model comparison")
            
            # Create bar plot of information criteria
            if 'AIC' in results_df.columns and 'BIC' in results_df.columns:
                # Set up positions for bars
                models_count = len(results_df)
                x_pos = np.arange(models_count)
                width = 0.35
                
                # Plot AIC and BIC values
                ax_ic.bar(x_pos - width/2, results_df['AIC'], width, label='AIC', color=PASTEL_COLORS[0])
                ax_ic.bar(x_pos + width/2, results_df['BIC'], width, label='BIC', color=PASTEL_COLORS[1])
                
                # Set labels and title
                ax_ic.set_xlabel('Model')
                ax_ic.set_ylabel('Value')
                ax_ic.set_title('AIC and BIC Comparison')
                ax_ic.set_xticks(x_pos)
                ax_ic.set_xticklabels(results_df['Model'])
                ax_ic.legend()
                
            elif 'AIC' in results_df.columns:
                ax_ic.bar(results_df['Model'], results_df['AIC'], color=PASTEL_COLORS[0])
                ax_ic.set_title('AIC Comparison')
                ax_ic.set_ylabel('AIC')
                
            elif 'BIC' in results_df.columns:
                ax_ic.bar(results_df['Model'], results_df['BIC'], color=PASTEL_COLORS[1])
                ax_ic.set_title('BIC Comparison')
                ax_ic.set_ylabel('BIC')
                
            else:
                ax_ic.text(0.5, 0.5, "No information criteria available for plotting",
                          horizontalalignment='center', verticalalignment='center')
            
            # Rotate x-tick labels if there are many models
            if models_count > 3:
                plt.xticks(rotation=45, ha='right')
                fig_ic.tight_layout()
            
            # Format details message
            details = []
            if best_aic_model:
                details.append(f"Best model by AIC: {best_aic_model}")
            if best_bic_model:
                details.append(f"Best model by BIC: {best_bic_model}")
            
            if best_aic_model == best_bic_model:
                result = AssumptionResult.PASSED
                details.append(f"Both AIC and BIC agree on the best model: {best_aic_model}")
            elif best_aic_model and best_bic_model:
                result = AssumptionResult.WARNING
                details.append("AIC and BIC suggest different models, consider the trade-off between fit and complexity")
                warnings.append("BIC tends to prefer simpler models than AIC")
            else:
                result = AssumptionResult.NOT_APPLICABLE
                details.append("Could not determine best model")
            
            return {
                'result': result,
                'details': ". ".join(details),
                'criteria_table': results_df,
                'best_aic_model': best_aic_model,
                'best_bic_model': best_bic_model,
                'warnings': warnings,
                'figures': {
                    'ic_plot': fig_to_svg(fig_ic)
                }
            }
            
        except Exception as e:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': f"Error in information criteria comparison: {str(e)}",
                'warnings': [f"Test failed with error: {str(e)}"],
                'figures': {
                    'ic_plot': fig_to_svg(fig_ic)
                }
            }

class PowerAnalysisTest(AssumptionTest):
    """Tests whether a study has adequate statistical power."""

    def __init__(self):
        super().__init__(
            name="Statistical Power Analysis",
            description="Calculates statistical power or required sample size for hypothesis tests",
            applicable_roles=["effect_size", "sample_size"],
            applicable_types=["numeric"]
        )

    def run_test(self, effect_size=None, sample_size=None, alpha=0.05, power=None, **kwargs):
        """
        Calculate statistical power or required sample size.
        
        Parameters:
        -----------
        effect_size : float, optional
            Standardized effect size (Cohen's d, f, r, etc.)
        sample_size : int, optional
            Total sample size
        alpha : float, optional
            Significance level. Default is 0.05
        power : float, optional
            Desired statistical power. Default is None
        test_type : str, optional
            Type of test: 't_test', 'f_test', 'correlation', 'chi_square', etc.
            
        Returns:
        --------
        dict
            Dictionary containing test results
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import stats
        
        test_type = kwargs.get('test_type', 't_test')
        alt_hypothesis = kwargs.get('alternative', 'two-sided')
        groups = kwargs.get('groups', 2)
        warnings = []
        
        # Create figure for power curve
        fig_power = plt.figure(figsize=(10, 6))
        fig_power.patch.set_alpha(0.0)
        ax_power = fig_power.add_subplot(111)
        ax_power.patch.set_alpha(0.0)
        
        try:
            # Determine what to calculate
            if power is None and sample_size is not None and effect_size is not None:
                # Calculate power
                calculation_type = 'power'
            elif power is not None and sample_size is None and effect_size is not None:
                # Calculate required sample size
                calculation_type = 'sample_size'
            elif power is not None and sample_size is not None and effect_size is None:
                # Calculate detectable effect size
                calculation_type = 'effect_size'
            else:
                return {
                    'result': AssumptionResult.NOT_APPLICABLE,
                    'details': "Must specify exactly two of: power, sample_size, effect_size",
                    'warnings': ["Insufficient parameters for power analysis"],
                    'figures': {
                        'power_plot': fig_to_svg(fig_power)
                    }
                }
            
            # Perform calculation based on test type
            if test_type == 't_test':
                result = self._t_test_power(effect_size, sample_size, alpha, power, 
                                         alt_hypothesis, groups, calculation_type)
            elif test_type == 'f_test':
                result = self._f_test_power(effect_size, sample_size, alpha, power, 
                                         groups, calculation_type)
            elif test_type == 'correlation':
                result = self._correlation_power(effect_size, sample_size, alpha, power, 
                                              alt_hypothesis, calculation_type)
            elif test_type == 'chi_square':
                result = self._chi_square_power(effect_size, sample_size, alpha, power, 
                                             groups, calculation_type)
            else:
                return {
                    'result': AssumptionResult.NOT_APPLICABLE,
                    'details': f"Unsupported test type: {test_type}",
                    'warnings': [f"Test type {test_type} not implemented for power analysis"],
                    'figures': {
                        'power_plot': fig_to_svg(fig_power)
                    }
                }
            
            # Create power curve plot
            if calculation_type == 'power':
                # Plot power vs. effect size
                effect_sizes = np.linspace(effect_size * 0.5, effect_size * 1.5, 100)
                powers = []
                
                for es in effect_sizes:
                    if test_type == 't_test':
                        pwr = self._t_test_power(es, sample_size, alpha, None, 
                                              alt_hypothesis, groups, 'power')['calculated_value']
                    elif test_type == 'f_test':
                        pwr = self._f_test_power(es, sample_size, alpha, None, 
                                              groups, 'power')['calculated_value']
                    elif test_type == 'correlation':
                        pwr = self._correlation_power(es, sample_size, alpha, None, 
                                                   alt_hypothesis, 'power')['calculated_value']
                    elif test_type == 'chi_square':
                        pwr = self._chi_square_power(es, sample_size, alpha, None, 
                                                  groups, 'power')['calculated_value']
                    powers.append(pwr)
                
                ax_power.plot(effect_sizes, powers, color=PASTEL_COLORS[0])
                ax_power.axhline(y=0.8, color='r', linestyle='--', label='0.8 (recommended)')
                ax_power.axvline(x=effect_size, color='g', linestyle='--', label='Current effect size')
                ax_power.set_xlabel('Effect Size')
                ax_power.set_ylabel('Power')
                ax_power.set_title(f'Power Curve for {test_type.replace("_", " ").title()}')
                ax_power.legend()
                
            elif calculation_type == 'sample_size':
                # Plot sample size vs. effect size
                effect_sizes = np.linspace(effect_size * 0.5, effect_size * 1.5, 100)
                sample_sizes = []
                
                for es in effect_sizes:
                    if test_type == 't_test':
                        ss = self._t_test_power(es, None, alpha, power, 
                                             alt_hypothesis, groups, 'sample_size')['calculated_value']
                    elif test_type == 'f_test':
                        ss = self._f_test_power(es, None, alpha, power, 
                                             groups, 'sample_size')['calculated_value']
                    elif test_type == 'correlation':
                        ss = self._correlation_power(es, None, alpha, power, 
                                                  alt_hypothesis, 'sample_size')['calculated_value']
                    elif test_type == 'chi_square':
                        ss = self._chi_square_power(es, None, alpha, power, 
                                                 groups, 'sample_size')['calculated_value']
                    sample_sizes.append(ss)
                
                ax_power.plot(effect_sizes, sample_sizes, color=PASTEL_COLORS[0])
                ax_power.axvline(x=effect_size, color='g', linestyle='--', label='Current effect size')
                ax_power.set_xlabel('Effect Size')
                ax_power.set_ylabel('Required Sample Size')
                ax_power.set_title(f'Required Sample Size for {test_type.replace("_", " ").title()}')
                ax_power.legend()
                
            elif calculation_type == 'effect_size':
                # Plot power vs. sample size
                sample_sizes = np.linspace(sample_size * 0.5, sample_size * 1.5, 100)
                powers = []
                
                # Calculate effect size for each sample size, keeping power constant
                for ss in sample_sizes:
                    if test_type == 't_test':
                        es = self._t_test_power(None, ss, alpha, power, 
                                             alt_hypothesis, groups, 'effect_size')['calculated_value']
                    elif test_type == 'f_test':
                        es = self._f_test_power(None, ss, alpha, power, 
                                             groups, 'effect_size')['calculated_value']
                    elif test_type == 'correlation':
                        es = self._correlation_power(None, ss, alpha, power, 
                                                  alt_hypothesis, 'effect_size')['calculated_value']
                    elif test_type == 'chi_square':
                        es = self._chi_square_power(None, ss, alpha, power, 
                                                 groups, 'effect_size')['calculated_value']
                    powers.append(es)
                
                ax_power.plot(sample_sizes, powers, color=PASTEL_COLORS[0])
                ax_power.axvline(x=sample_size, color='g', linestyle='--', label='Current sample size')
                ax_power.set_xlabel('Sample Size')
                ax_power.set_ylabel('Minimum Detectable Effect Size')
                ax_power.set_title(f'Minimum Detectable Effect Size for {test_type.replace("_", " ").title()}')
                ax_power.legend()
            
            # Add result figure
            result['figures'] = {
                'power_plot': fig_to_svg(fig_power)
            }
            
            return result
            
        except Exception as e:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': f"Error in power analysis: {str(e)}",
                'warnings': [f"Test failed with error: {str(e)}"],
                'figures': {
                    'power_plot': fig_to_svg(fig_power)
                }
            }
    
    def _t_test_power(self, effect_size, sample_size, alpha, power, alternative, groups, calculation_type):
        """Calculate power, sample size, or effect size for t-test."""
        from scipy import stats
        
        # Convert alternative hypothesis string to appropriate parameter
        if alternative == 'two-sided':
            alternative_param = 'two-sided'
        elif alternative in ['less', 'greater']:
            alternative_param = 'one-sided'
        else:
            alternative_param = 'two-sided'
        
        # Calculate degrees of freedom based on test type
        if groups == 1:
            # One-sample or paired t-test
            if calculation_type == 'power':
                calculated_power = stats.t.sf(
                    stats.t.ppf(1 - alpha/2, df=sample_size-1),
                    df=sample_size-1,
                    nc=effect_size * np.sqrt(sample_size)
                )
                if alternative_param == 'two-sided':
                    calculated_power = 2 * calculated_power
                
                # Determine if power is adequate
                if calculated_power >= 0.8:
                    result = AssumptionResult.PASSED
                    details = f"Study has adequate power ({calculated_power:.3f}) to detect effect size of {effect_size:.3f}"
                elif calculated_power >= 0.5:
                    result = AssumptionResult.WARNING
                    details = f"Study may be underpowered ({calculated_power:.3f}) to detect effect size of {effect_size:.3f}"
                else:
                    result = AssumptionResult.FAILED
                    details = f"Study is underpowered ({calculated_power:.3f}) to detect effect size of {effect_size:.3f}"
                
                return {
                    'result': result,
                    'details': details,
                    'test_type': 'one-sample t-test',
                    'alternative': alternative,
                    'effect_size': effect_size,
                    'sample_size': sample_size,
                    'alpha': alpha,
                    'calculated_value': calculated_power,
                    'calculation_type': 'power'
                }
                
            elif calculation_type == 'sample_size':
                # Approximate using normal distribution first
                from math import ceil
                from scipy.optimize import minimize_scalar
                
                # Function to minimize: difference between target power and achieved power
                def objective(n):
                    n = int(n)
                    if n <= 1:
                        return float('inf')
                    actual_power = stats.t.sf(
                        stats.t.ppf(1 - alpha/2, df=n-1),
                        df=n-1,
                        nc=effect_size * np.sqrt(n)
                    )
                    if alternative_param == 'two-sided':
                        actual_power = 2 * actual_power
                    return abs(actual_power - power)
                
                # Initial guess using normal approximation
                z_alpha = stats.norm.ppf(1 - alpha/2)
                z_beta = stats.norm.ppf(power)
                initial_guess = ((z_alpha + z_beta) / effect_size) ** 2
                
                # Optimize to find exact sample size
                result = minimize_scalar(objective, bounds=(2, 10000), method='bounded')
                calculated_sample_size = ceil(result.x)
                
                # Determine if sample size is reasonable
                if calculated_sample_size <= 1000:
                    result = AssumptionResult.PASSED
                    details = f"Required sample size ({calculated_sample_size}) is feasible for detecting effect size of {effect_size:.3f}"
                elif calculated_sample_size <= 5000:
                    result = AssumptionResult.WARNING
                    details = f"Required sample size ({calculated_sample_size}) may be large for detecting effect size of {effect_size:.3f}"
                else:
                    result = AssumptionResult.FAILED
                    details = f"Required sample size ({calculated_sample_size}) is very large for detecting effect size of {effect_size:.3f}"
                
                return {
                    'result': result,
                    'details': details,
                    'test_type': 'one-sample t-test',
                    'alternative': alternative,
                    'effect_size': effect_size,
                    'alpha': alpha,
                    'power': power,
                    'calculated_value': calculated_sample_size,
                    'calculation_type': 'sample_size'
                }
                
            elif calculation_type == 'effect_size':
                from scipy.optimize import minimize_scalar
                
                # Function to minimize: difference between target power and achieved power
                def objective(d):
                    actual_power = stats.t.sf(
                        stats.t.ppf(1 - alpha/2, df=sample_size-1),
                        df=sample_size-1,
                        nc=d * np.sqrt(sample_size)
                    )
                    if alternative_param == 'two-sided':
                        actual_power = 2 * actual_power
                    return abs(actual_power - power)
                
                # Optimize to find exact effect size
                result = minimize_scalar(objective, bounds=(0.01, 5.0), method='bounded')
                calculated_effect_size = result.x
                
                # Interpret effect size (Cohen's guidelines)
                if calculated_effect_size <= 0.2:
                    effect_size_interpretation = "small"
                elif calculated_effect_size <= 0.5:
                    effect_size_interpretation = "medium"
                else:
                    effect_size_interpretation = "large"
                
                # Determine if effect size is detectable
                if calculated_effect_size <= 0.5:
                    result = AssumptionResult.PASSED
                    details = f"Study can detect {effect_size_interpretation} effects ({calculated_effect_size:.3f}) with {power:.1f} power"
                elif calculated_effect_size <= 0.8:
                    result = AssumptionResult.WARNING
                    details = f"Study can only detect {effect_size_interpretation} effects ({calculated_effect_size:.3f}) with {power:.1f} power"
                else:
                    result = AssumptionResult.FAILED
                    details = f"Study can only detect very {effect_size_interpretation} effects ({calculated_effect_size:.3f}) with {power:.1f} power"
                
                return {
                    'result': result,
                    'details': details,
                    'test_type': 'one-sample t-test',
                    'alternative': alternative,
                    'sample_size': sample_size,
                    'alpha': alpha,
                    'power': power,
                    'calculated_value': calculated_effect_size,
                    'calculation_type': 'effect_size'
                }
                
        else:  # groups == 2 (independent t-test)
            # Assume equal allocation to groups
            n1 = sample_size // 2
            n2 = sample_size - n1
            
            if calculation_type == 'power':
                df = n1 + n2 - 2
                # Calculate non-centrality parameter
                nc = effect_size * np.sqrt((n1 * n2) / (n1 + n2))
                
                calculated_power = stats.t.sf(
                    stats.t.ppf(1 - alpha/2, df=df),
                    df=df,
                    nc=nc
                )
                if alternative_param == 'two-sided':
                    calculated_power = 2 * calculated_power
                
                # Determine if power is adequate
                if calculated_power >= 0.8:
                    result = AssumptionResult.PASSED
                    details = f"Study has adequate power ({calculated_power:.3f}) to detect effect size of {effect_size:.3f}"
                elif calculated_power >= 0.5:
                    result = AssumptionResult.WARNING
                    details = f"Study may be underpowered ({calculated_power:.3f}) to detect effect size of {effect_size:.3f}"
                else:
                    result = AssumptionResult.FAILED
                    details = f"Study is underpowered ({calculated_power:.3f}) to detect effect size of {effect_size:.3f}"
                
                return {
                    'result': result,
                    'details': details,
                    'test_type': 'two-sample t-test',
                    'alternative': alternative,
                    'effect_size': effect_size,
                    'sample_size': sample_size,
                    'alpha': alpha,
                    'calculated_value': calculated_power,
                    'calculation_type': 'power'
                }
                
            elif calculation_type == 'sample_size':
                from math import ceil
                from scipy.optimize import minimize_scalar
                
                # Function to minimize: difference between target power and achieved power
                def objective(n):
                    n = int(n)
                    if n <= 2:
                        return float('inf')
                    
                    n1 = n // 2
                    n2 = n - n1
                    df = n1 + n2 - 2
                    nc = effect_size * np.sqrt((n1 * n2) / (n1 + n2))
                    
                    actual_power = stats.t.sf(
                        stats.t.ppf(1 - alpha/2, df=df),
                        df=df,
                        nc=nc
                    )
                    if alternative_param == 'two-sided':
                        actual_power = 2 * actual_power
                    
                    return abs(actual_power - power)
                
                # Optimize to find exact sample size
                result = minimize_scalar(objective, bounds=(4, 10000), method='bounded')
                calculated_sample_size = ceil(result.x)
                
                # Determine if sample size is reasonable
                if calculated_sample_size <= 200:
                    result = AssumptionResult.PASSED
                    details = f"Required sample size ({calculated_sample_size}) is feasible for detecting effect size of {effect_size:.3f}"
                elif calculated_sample_size <= 1000:
                    result = AssumptionResult.WARNING
                    details = f"Required sample size ({calculated_sample_size}) may be large for detecting effect size of {effect_size:.3f}"
                else:
                    result = AssumptionResult.FAILED
                    details = f"Required sample size ({calculated_sample_size}) is very large for detecting effect size of {effect_size:.3f}"
                
                return {
                    'result': result,
                    'details': details,
                    'test_type': 'two-sample t-test',
                    'alternative': alternative,
                    'effect_size': effect_size,
                    'alpha': alpha,
                    'power': power,
                    'calculated_value': calculated_sample_size,
                    'calculation_type': 'sample_size'
                }
                
            elif calculation_type == 'effect_size':
                from scipy.optimize import minimize_scalar
                
                # Function to minimize: difference between target power and achieved power
                def objective(d):
                    n1 = sample_size // 2
                    n2 = sample_size - n1
                    df = n1 + n2 - 2
                    nc = d * np.sqrt((n1 * n2) / (n1 + n2))
                    
                    actual_power = stats.t.sf(
                        stats.t.ppf(1 - alpha/2, df=df),
                        df=df,
                        nc=nc
                    )
                    if alternative_param == 'two-sided':
                        actual_power = 2 * actual_power
                    
                    return abs(actual_power - power)
                
                # Optimize to find exact effect size
                result = minimize_scalar(objective, bounds=(0.01, 5.0), method='bounded')
                calculated_effect_size = result.x
                
                # Interpret effect size (Cohen's guidelines)
                if calculated_effect_size <= 0.2:
                    effect_size_interpretation = "small"
                elif calculated_effect_size <= 0.5:
                    effect_size_interpretation = "medium"
                else:
                    effect_size_interpretation = "large"
                
                # Determine if effect size is detectable
                if calculated_effect_size <= 0.5:
                    result = AssumptionResult.PASSED
                    details = f"Study can detect {effect_size_interpretation} effects ({calculated_effect_size:.3f}) with {power:.1f} power"
                elif calculated_effect_size <= 0.8:
                    result = AssumptionResult.WARNING
                    details = f"Study can only detect {effect_size_interpretation} effects ({calculated_effect_size:.3f}) with {power:.1f} power"
                else:
                    result = AssumptionResult.FAILED
                    details = f"Study can only detect very {effect_size_interpretation} effects ({calculated_effect_size:.3f}) with {power:.1f} power"
                
                return {
                    'result': result,
                    'details': details,
                    'test_type': 'two-sample t-test',
                    'alternative': alternative,
                    'sample_size': sample_size,
                    'alpha': alpha,
                    'power': power,
                    'calculated_value': calculated_effect_size,
                    'calculation_type': 'effect_size'
                }
    
    def _f_test_power(self, effect_size, sample_size, alpha, power, groups, calculation_type):
        """Calculate power, sample size, or effect size for F-test (ANOVA)."""
        from scipy import stats
        import numpy as np
        from scipy.optimize import minimize_scalar
        from math import ceil
        
        # Calculate degrees of freedom
        df1 = groups - 1  # Between groups df
        
        if calculation_type == 'power':
            # Calculate within groups df
            df2 = sample_size - groups
            
            # Calculate non-centrality parameter
            nc = sample_size * effect_size**2
            
            # Calculate power
            calculated_power = 1 - stats.ncf.cdf(
                stats.f.ppf(1-alpha, df1, df2),
                df1, df2, nc
            )
            
            # Determine if power is adequate
            if calculated_power >= 0.8:
                result = AssumptionResult.PASSED
                details = f"Study has adequate power ({calculated_power:.3f}) to detect effect size of {effect_size:.3f}"
            elif calculated_power >= 0.5:
                result = AssumptionResult.WARNING
                details = f"Study may be underpowered ({calculated_power:.3f}) to detect effect size of {effect_size:.3f}"
            else:
                result = AssumptionResult.FAILED
                details = f"Study is underpowered ({calculated_power:.3f}) to detect effect size of {effect_size:.3f}"
            
            return {
                'result': result,
                'details': details,
                'test_type': 'ANOVA (F-test)',
                'effect_size': effect_size,
                'sample_size': sample_size,
                'alpha': alpha,
                'calculated_value': calculated_power,
                'calculation_type': 'power',
                'groups': groups
            }
            
        elif calculation_type == 'sample_size':
            # Function to minimize: difference between target power and achieved power
            def objective(n):
                n = int(n)
                if n <= groups:
                    return float('inf')
                
                df2 = n - groups
                nc = n * effect_size**2
                
                actual_power = 1 - stats.ncf.cdf(
                    stats.f.ppf(1-alpha, df1, df2),
                    df1, df2, nc
                )
                
                return abs(actual_power - power)
            
            # Optimize to find exact sample size
            result = minimize_scalar(objective, bounds=(groups+1, 10000), method='bounded')
            calculated_sample_size = ceil(result.x)
            
            # Determine if sample size is reasonable
            if calculated_sample_size <= 200:
                result = AssumptionResult.PASSED
                details = f"Required sample size ({calculated_sample_size}) is feasible for detecting effect size of {effect_size:.3f}"
            elif calculated_sample_size <= 1000:
                result = AssumptionResult.WARNING
                details = f"Required sample size ({calculated_sample_size}) may be large for detecting effect size of {effect_size:.3f}"
            else:
                result = AssumptionResult.FAILED
                details = f"Required sample size ({calculated_sample_size}) is very large for detecting effect size of {effect_size:.3f}"
            
            return {
                'result': result,
                'details': details,
                'test_type': 'ANOVA (F-test)',
                'effect_size': effect_size,
                'alpha': alpha,
                'power': power,
                'calculated_value': calculated_sample_size,
                'calculation_type': 'sample_size',
                'groups': groups
            }
            
        elif calculation_type == 'effect_size':
            # Calculate within groups df
            df2 = sample_size - groups
            
            # Function to minimize: difference between target power and achieved power
            def objective(f):
                nc = sample_size * f**2
                
                actual_power = 1 - stats.ncf.cdf(
                    stats.f.ppf(1-alpha, df1, df2),
                    df1, df2, nc
                )
                
                return abs(actual_power - power)
            
            # Optimize to find exact effect size
            result = minimize_scalar(objective, bounds=(0.01, 2.0), method='bounded')
            calculated_effect_size = result.x
            
            # Interpret effect size (Cohen's guidelines for f)
            if calculated_effect_size <= 0.1:
                effect_size_interpretation = "small"
            elif calculated_effect_size <= 0.25:
                effect_size_interpretation = "medium"
            else:
                effect_size_interpretation = "large"
            
            # Determine if effect size is detectable
            if calculated_effect_size <= 0.25:
                result = AssumptionResult.PASSED
                details = f"Study can detect {effect_size_interpretation} effects ({calculated_effect_size:.3f}) with {power:.1f} power"
            elif calculated_effect_size <= 0.4:
                result = AssumptionResult.WARNING
                details = f"Study can only detect {effect_size_interpretation} effects ({calculated_effect_size:.3f}) with {power:.1f} power"
            else:
                result = AssumptionResult.FAILED
                details = f"Study can only detect very {effect_size_interpretation} effects ({calculated_effect_size:.3f}) with {power:.1f} power"
            
            return {
                'result': result,
                'details': details,
                'test_type': 'ANOVA (F-test)',
                'sample_size': sample_size,
                'alpha': alpha,
                'power': power,
                'calculated_value': calculated_effect_size,
                'calculation_type': 'effect_size',
                'groups': groups
            }
    
    def _correlation_power(self, effect_size, sample_size, alpha, power, alternative, calculation_type):
        """Calculate power, sample size, or effect size for correlation test."""
        from scipy import stats
        import numpy as np
        from scipy.optimize import minimize_scalar
        from math import ceil
        
        # Convert alternative hypothesis string to appropriate parameter
        if alternative == 'two-sided':
            alternative_param = 'two-sided'
        elif alternative in ['less', 'greater']:
            alternative_param = 'one-sided'
        else:
            alternative_param = 'two-sided'
        
        if calculation_type == 'power':
            # Convert correlation to Fisher's z
            z = 0.5 * np.log((1 + effect_size) / (1 - effect_size))
            
            # Calculate standard error
            se = 1 / np.sqrt(sample_size - 3)
            
            # Calculate critical value
            if alternative_param == 'two-sided':
                z_crit = stats.norm.ppf(1 - alpha/2)
            else:
                z_crit = stats.norm.ppf(1 - alpha)
            
            # Calculate power
            if alternative_param == 'two-sided':
                calculated_power = (1 - stats.norm.cdf(z_crit - z/se)) + stats.norm.cdf(-z_crit - z/se)
            else:
                calculated_power = 1 - stats.norm.cdf(z_crit - z/se)
            
            # Determine if power is adequate
            if calculated_power >= 0.8:
                result = AssumptionResult.PASSED
                details = f"Study has adequate power ({calculated_power:.3f}) to detect correlation of {effect_size:.3f}"
            elif calculated_power >= 0.5:
                result = AssumptionResult.WARNING
                details = f"Study may be underpowered ({calculated_power:.3f}) to detect correlation of {effect_size:.3f}"
            else:
                result = AssumptionResult.FAILED
                details = f"Study is underpowered ({calculated_power:.3f}) to detect correlation of {effect_size:.3f}"
            
            return {
                'result': result,
                'details': details,
                'test_type': 'correlation test',
                'alternative': alternative,
                'effect_size': effect_size,
                'sample_size': sample_size,
                'alpha': alpha,
                'calculated_value': calculated_power,
                'calculation_type': 'power'
            }
            
        elif calculation_type == 'sample_size':
            # Function to minimize: difference between target power and achieved power
            def objective(n):
                n = int(n)
                if n <= 3:
                    return float('inf')
                
                # Convert correlation to Fisher's z
                z = 0.5 * np.log((1 + effect_size) / (1 - effect_size))
                
                # Calculate standard error
                se = 1 / np.sqrt(n - 3)
                
                # Calculate critical value
                if alternative_param == 'two-sided':
                    z_crit = stats.norm.ppf(1 - alpha/2)
                else:
                    z_crit = stats.norm.ppf(1 - alpha)
                
                # Calculate power
                if alternative_param == 'two-sided':
                    actual_power = (1 - stats.norm.cdf(z_crit - z/se)) + stats.norm.cdf(-z_crit - z/se)
                else:
                    actual_power = 1 - stats.norm.cdf(z_crit - z/se)
                
                return abs(actual_power - power)
            
            # Optimize to find exact sample size
            result = minimize_scalar(objective, bounds=(4, 10000), method='bounded')
            calculated_sample_size = ceil(result.x)
            
            # Determine if sample size is reasonable
            if calculated_sample_size <= 200:
                result = AssumptionResult.PASSED
                details = f"Required sample size ({calculated_sample_size}) is feasible for detecting correlation of {effect_size:.3f}"
            elif calculated_sample_size <= 1000:
                result = AssumptionResult.WARNING
                details = f"Required sample size ({calculated_sample_size}) may be large for detecting correlation of {effect_size:.3f}"
            else:
                result = AssumptionResult.FAILED
                details = f"Required sample size ({calculated_sample_size}) is very large for detecting correlation of {effect_size:.3f}"
            
            return {
                'result': result,
                'details': details,
                'test_type': 'correlation test',
                'alternative': alternative,
                'effect_size': effect_size,
                'alpha': alpha,
                'power': power,
                'calculated_value': calculated_sample_size,
                'calculation_type': 'sample_size'
            }
            
        elif calculation_type == 'effect_size':
            # Function to minimize: difference between target power and achieved power
            def objective(r):
                if r >= 1.0 or r <= -1.0:
                    return float('inf')
                
                # Convert correlation to Fisher's z
                z = 0.5 * np.log((1 + r) / (1 - r))
                
                # Calculate standard error
                se = 1 / np.sqrt(sample_size - 3)
                
                # Calculate critical value
                if alternative_param == 'two-sided':
                    z_crit = stats.norm.ppf(1 - alpha/2)
                else:
                    z_crit = stats.norm.ppf(1 - alpha)
                
                # Calculate power
                if alternative_param == 'two-sided':
                    actual_power = (1 - stats.norm.cdf(z_crit - z/se)) + stats.norm.cdf(-z_crit - z/se)
                else:
                    actual_power = 1 - stats.norm.cdf(z_crit - z/se)
                
                return abs(actual_power - power)
            
            # Optimize to find exact effect size
            result = minimize_scalar(objective, bounds=(-0.99, 0.99), method='bounded')
            calculated_effect_size = result.x
            
            # Interpret effect size (Cohen's guidelines for r)
            if abs(calculated_effect_size) <= 0.1:
                effect_size_interpretation = "small"
            elif abs(calculated_effect_size) <= 0.3:
                effect_size_interpretation = "medium"
            else:
                effect_size_interpretation = "large"
            
            # Determine if effect size is detectable
            if abs(calculated_effect_size) <= 0.3:
                result = AssumptionResult.PASSED
                details = f"Study can detect {effect_size_interpretation} correlations ({calculated_effect_size:.3f}) with {power:.1f} power"
            elif abs(calculated_effect_size) <= 0.5:
                result = AssumptionResult.WARNING
                details = f"Study can only detect {effect_size_interpretation} correlations ({calculated_effect_size:.3f}) with {power:.1f} power"
            else:
                result = AssumptionResult.FAILED
                details = f"Study can only detect very {effect_size_interpretation} correlations ({calculated_effect_size:.3f}) with {power:.1f} power"
            
            return {
                'result': result,
                'details': details,
                'test_type': 'correlation test',
                'alternative': alternative,
                'sample_size': sample_size,
                'alpha': alpha,
                'power': power,
                'calculated_value': calculated_effect_size,
                'calculation_type': 'effect_size'
            }
    
    def _chi_square_power(self, effect_size, sample_size, alpha, power, df, calculation_type):
        """Calculate power, sample size, or effect size for chi-square test."""
        from scipy import stats
        import numpy as np
        from scipy.optimize import minimize_scalar
        from math import ceil
        
        if calculation_type == 'power':
            # Calculate non-centrality parameter
            nc = sample_size * effect_size**2
            
            # Calculate power
            calculated_power = 1 - stats.ncx2.cdf(
                stats.chi2.ppf(1-alpha, df),
                df, nc
            )
            
            # Determine if power is adequate
            if calculated_power >= 0.8:
                result = AssumptionResult.PASSED
                details = f"Study has adequate power ({calculated_power:.3f}) to detect effect size of {effect_size:.3f}"
            elif calculated_power >= 0.5:
                result = AssumptionResult.WARNING
                details = f"Study may be underpowered ({calculated_power:.3f}) to detect effect size of {effect_size:.3f}"
            else:
                result = AssumptionResult.FAILED
                details = f"Study is underpowered ({calculated_power:.3f}) to detect effect size of {effect_size:.3f}"
            
            return {
                'result': result,
                'details': details,
                'test_type': 'chi-square test',
                'effect_size': effect_size,
                'sample_size': sample_size,
                'alpha': alpha,
                'df': df,
                'calculated_value': calculated_power,
                'calculation_type': 'power'
            }
            
        elif calculation_type == 'sample_size':
            # Function to minimize: difference between target power and achieved power
            def objective(n):
                n = int(n)
                if n <= 0:
                    return float('inf')
                
                # Calculate non-centrality parameter
                nc = n * effect_size**2
                
                # Calculate power
                actual_power = 1 - stats.ncx2.cdf(
                    stats.chi2.ppf(1-alpha, df),
                    df, nc
                )
                
                return abs(actual_power - power)
            
            # Optimize to find exact sample size
            result = minimize_scalar(objective, bounds=(1, 10000), method='bounded')
            calculated_sample_size = ceil(result.x)
            
            # Determine if sample size is reasonable
            if calculated_sample_size <= 200:
                result = AssumptionResult.PASSED
                details = f"Required sample size ({calculated_sample_size}) is feasible for detecting effect size of {effect_size:.3f}"
            elif calculated_sample_size <= 1000:
                result = AssumptionResult.WARNING
                details = f"Required sample size ({calculated_sample_size}) may be large for detecting effect size of {effect_size:.3f}"
            else:
                result = AssumptionResult.FAILED
                details = f"Required sample size ({calculated_sample_size}) is very large for detecting effect size of {effect_size:.3f}"
            
            return {
                'result': result,
                'details': details,
                'test_type': 'chi-square test',
                'effect_size': effect_size,
                'alpha': alpha,
                'power': power,
                'df': df,
                'calculated_value': calculated_sample_size,
                'calculation_type': 'sample_size'
            }
            
        elif calculation_type == 'effect_size':
            # Function to minimize: difference between target power and achieved power
            def objective(w):
                # Calculate non-centrality parameter
                nc = sample_size * w**2
                
                # Calculate power
                actual_power = 1 - stats.ncx2.cdf(
                    stats.chi2.ppf(1-alpha, df),
                    df, nc
                )
                
                return abs(actual_power - power)
            
            # Optimize to find exact effect size
            result = minimize_scalar(objective, bounds=(0.01, 2.0), method='bounded')
            calculated_effect_size = result.x
            
            # Interpret effect size (Cohen's guidelines for w)
            if calculated_effect_size <= 0.1:
                effect_size_interpretation = "small"
            elif calculated_effect_size <= 0.3:
                effect_size_interpretation = "medium"
            else:
                effect_size_interpretation = "large"
            
            # Determine if effect size is detectable
            if calculated_effect_size <= 0.3:
                result = AssumptionResult.PASSED
                details = f"Study can detect {effect_size_interpretation} effects ({calculated_effect_size:.3f}) with {power:.1f} power"
            elif calculated_effect_size <= 0.5:
                result = AssumptionResult.WARNING
                details = f"Study can only detect {effect_size_interpretation} effects ({calculated_effect_size:.3f}) with {power:.1f} power"
            else:
                result = AssumptionResult.FAILED
                details = f"Study can only detect very {effect_size_interpretation} effects ({calculated_effect_size:.3f}) with {power:.1f} power"
            
            return {
                'result': result,
                'details': details,
                'test_type': 'chi-square test',
                'sample_size': sample_size,
                'alpha': alpha,
                'power': power,
                'df': df,
                'calculated_value': calculated_effect_size,
                'calculation_type': 'effect_size'
            }
        
class IndependenceTest(AssumptionTest):
    """Tests for independence of observations."""

    def __init__(self):
        super().__init__(
            name="Independence Test",
            description="Tests whether observations are independent of each other.",
            applicable_roles=["residuals", "outcome"],
            applicable_types=["continuous", "categorical"]
        )

    def run_test(self, data, **kwargs):
        """
        Test for independence of observations using Runs test or correlation-based approach.
        
        Parameters:
        -----------
        data : array-like
            The data to test for independence
        method : str, optional
            Method to use: 'runs_test' or 'correlation' (default: 'runs_test')
        lag : int, optional
            Lag to use for correlation test (default: 1)
            
        Returns:
        --------
        dict
            Dictionary containing test results
        """
        method = kwargs.get('method', 'runs_test')
        alpha = kwargs.get('alpha', 0.05)
        
        try:
            # Handle multi-dimensional data (e.g., from multinomial logistic regression)
            import numpy as np
            data = np.array(data)
            if len(data.shape) > 1 and data.shape[1] > 1:
                # For multi-column data, take the mean across columns
                data = np.mean(data, axis=1)
            
            if method == 'runs_test':
                return self._run_runs_test(data)
            else:
                return self._run_correlation_test(data, **kwargs)
        except Exception as e:
            return {
                "result": AssumptionResult.FAILED,
                "message": f"Error in independence test: {str(e)}",
                "statistic": None,
                "details": {"error": str(e)}
            }
    
    def _run_runs_test(self, data):
        """Run the Wald-Wolfowitz runs test for randomness."""
        import numpy as np
        from scipy import stats
        
        alpha = 0.05
        
        # Convert data to numpy array and remove NAs
        data = np.array(data)
        data = data[~np.isnan(data)]
        
        # Get median
        median = np.median(data)
        
        # Create binary sequence
        binary = np.where(data > median, 1, 0)
        
        # Count runs
        n = len(binary)
        n1 = np.sum(binary)
        n2 = n - n1
        
        # Count runs
        runs = 1
        for i in range(1, n):
            if binary[i] != binary[i-1]:
                runs += 1
        
        # Expected number of runs and standard deviation
        expected_runs = (2 * n1 * n2) / n + 1
        std_runs = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n)) / (n**2 * (n - 1)))
        
        # Z-statistic
        z = (runs - expected_runs) / std_runs
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))  # Two-sided test
        
        if p_value >= alpha:
            result = AssumptionResult.PASSED
            message = f"No significant dependence detected (p={p_value:.4f})."
        else:
            result = AssumptionResult.FAILED
            message = f"Observations appear to be dependent (p={p_value:.4f})."
        
        return {
            "result": result,
            "message": message,
            "statistic": z,
            "details": {
                "runs": runs,
                "expected_runs": expected_runs,
                "z_statistic": z,
                "p_value": p_value,
                "method": "Runs test",
                "n1": n1,
                "n2": n2
            }
        }
    
    def _run_correlation_test(self, data, **kwargs):
        """Test for autocorrelation at given lag."""
        import numpy as np
        from scipy import stats
        
        alpha = kwargs.get('alpha', 0.05)
        lag = kwargs.get('lag', 1)
        
        # Convert to numpy array and remove NAs
        data = np.array(data)
        data = data[~np.isnan(data)]
        
        if len(data) <= lag:
            return {
                "result": AssumptionResult.NOT_APPLICABLE,
                "message": f"Data length ({len(data)}) is not sufficient for lag {lag}",
                "statistic": None,
                "details": {"error": "Insufficient data length"}
            }
        
        # Calculate autocorrelation
        x1 = data[:-lag]
        x2 = data[lag:]
        
        correlation, p_value = stats.pearsonr(x1, x2)
        
        if p_value >= alpha:
            result = AssumptionResult.PASSED
            message = f"No significant autocorrelation at lag {lag} (p={p_value:.4f})."
        else:
            if abs(correlation) < 0.3:
                result = AssumptionResult.WARNING
                message = f"Statistically significant but weak autocorrelation detected (r={correlation:.3f}, p={p_value:.4f})."
            else:
                result = AssumptionResult.FAILED
                message = f"Significant autocorrelation detected at lag {lag} (r={correlation:.3f}, p={p_value:.4f})."
        
        return {
            "result": result,
            "message": message,
            "statistic": correlation,
            "details": {
                "autocorrelation": correlation,
                "p_value": p_value,
                "lag": lag,
                "method": "Autocorrelation test"
            }
        }
    
class ResidualNormalityTest(AssumptionTest):
    """Tests for normality of residuals in regression models."""

    def __init__(self):
        super().__init__(
            name="Residual Normality Test",
            description="Tests whether model residuals follow a normal distribution.",
            applicable_roles=["residuals"],
            applicable_types=["continuous"]
        )

    def run_test(self, residuals, **kwargs):
        """
        Test for normality of residuals.
        
        Parameters:
        -----------
        residuals : array-like
            Residuals from a regression model
        method : str, optional
            Method to use: 'shapiro' (default), 'jarque_bera', or 'anderson'
            
        Returns:
        --------
        dict
            Dictionary containing test results
        """
        import numpy as np
        from scipy import stats
        
        method = kwargs.get('method', 'shapiro')
        alpha = kwargs.get('alpha', 0.05)
        
        try:
            # Convert to numpy array and remove NAs
            residuals = np.array(residuals)
            residuals = residuals[~np.isnan(residuals)]
            
            if len(residuals) < 3:
                return {
                    "result": AssumptionResult.NOT_APPLICABLE,
                    "message": "Not enough data points for normality test",
                    "statistic": None,
                    "details": {"error": "Insufficient data"}
                }
            
            if method == 'shapiro':
                # Shapiro-Wilk test
                if len(residuals) > 5000:
                    return {
                        "result": AssumptionResult.NOT_APPLICABLE,
                        "message": "Shapiro-Wilk test not applicable for sample size > 5000",
                        "statistic": None,
                        "details": {"error": "Sample size too large for Shapiro-Wilk"}
                    }
                
                statistic, p_value = stats.shapiro(residuals)
                test_name = "Shapiro-Wilk"
                
            elif method == 'jarque_bera':
                # Jarque-Bera test
                statistic, p_value = stats.jarque_bera(residuals)
                test_name = "Jarque-Bera"
                
            elif method == 'anderson':
                # Anderson-Darling test
                result = stats.anderson(residuals, dist='norm')
                statistic = result.statistic
                critical_values = result.critical_values
                significance_levels = result.significance_level
                
                # Find the appropriate p-value approximation
                p_value = None
                for sig_level, crit_val in zip(significance_levels, critical_values):
                    if statistic < crit_val:
                        p_value = sig_level / 100  # Convert percentage to proportion
                        break
                
                if p_value is None:
                    p_value = 0.01  # Smaller than the smallest significance level
                
                test_name = "Anderson-Darling"
            
            else:
                return {
                    "result": AssumptionResult.FAILED,
                    "message": f"Unknown method: {method}",
                    "statistic": None,
                    "details": {"error": f"Unknown method: {method}"}
                }
            
            # Interpret result
            if p_value >= alpha:
                result = AssumptionResult.PASSED
                message = f"Residuals appear to be normally distributed ({test_name} p={p_value:.4f})."
            else:
                # Check if the deviation is severe using skewness and kurtosis
                skewness = stats.skew(residuals)
                kurtosis = stats.kurtosis(residuals)
                
                if abs(skewness) < 0.5 and abs(kurtosis) < 1:
                    result = AssumptionResult.WARNING
                    message = f"Minor deviation from normality detected ({test_name} p={p_value:.4f})."
                else:
                    result = AssumptionResult.FAILED
                    message = f"Residuals are not normally distributed ({test_name} p={p_value:.4f})."
            
            return {
                "result": result,
                "message": message,
                "details": {
                    "test": test_name,
                    "statistic": statistic,
                    "p_value": p_value,
                    "skewness": stats.skew(residuals),
                    "kurtosis": stats.kurtosis(residuals),
                    "sample_size": len(residuals)
                }
            }
            
        except Exception as e:
            return {
                "result": AssumptionResult.FAILED,
                "message": f"Error in residual normality test: {str(e)}",
                "statistic": None,
                "details": {"error": str(e)}
            }
        
class OverdispersionTest(AssumptionTest):
    """Tests for overdispersion in count data models."""

    def __init__(self):
        super().__init__(
            name="Overdispersion Test",
            description="Tests for overdispersion in Poisson or binomial regression models.",
            applicable_roles=["outcome", "predicted"],
            applicable_types=["count", "binary"]
        )

    def run_test(self, observed, predicted, **kwargs):
        """
        Test for overdispersion in count or binary data.
        
        Parameters:
        -----------
        observed : array-like
            Observed values (counts or binary outcomes)
        predicted : array-like
            Model-predicted values or probabilities
        model_type : str, optional
            Type of model: 'poisson' or 'binomial' (default: 'poisson')
            
        Returns:
        --------
        dict
            Dictionary containing test results
        """
        import numpy as np
        from scipy import stats
        
        model_type = kwargs.get('model_type', 'poisson')
        alpha = kwargs.get('alpha', 0.05)
        
        try:
            # Convert to numpy arrays
            observed = np.array(observed)
            predicted = np.array(predicted)
            
            # Remove NAs
            mask = ~(np.isnan(observed) | np.isnan(predicted))
            observed = observed[mask]
            predicted = predicted[mask]
            
            if len(observed) == 0:
                return {
                    "result": AssumptionResult.NOT_APPLICABLE,
                    "message": "No valid data for overdispersion test",
                    "statistic": None,
                    "details": {"error": "No valid data"}
                }
            
            if model_type == 'poisson':
                # Pearson chi-square for Poisson regression
                # Calculate Pearson residuals
                pearson_residuals = (observed - predicted) / np.sqrt(predicted)
                
                # Sum of squared Pearson residuals
                chi_square = np.sum(pearson_residuals ** 2)
                
                # Dispersion parameter
                n = len(observed)
                p = kwargs.get('num_params', 1)  # Number of parameters in the model
                df = n - p
                dispersion = chi_square / df
                
                # Test if dispersion is significantly > 1 using chi-square distribution
                p_value = 1 - stats.chi2.cdf(chi_square, df)
                
                if dispersion <= 1.1:
                    result = AssumptionResult.PASSED
                    message = f"No overdispersion detected (dispersion={dispersion:.3f})."
                elif dispersion <= 2:
                    result = AssumptionResult.WARNING
                    message = f"Mild overdispersion detected (dispersion={dispersion:.3f})."
                else:
                    result = AssumptionResult.FAILED
                    message = f"Significant overdispersion detected (dispersion={dispersion:.3f})."
                
                return {
                    "result": result,
                    "message": message,
                    "statistic": dispersion,
                    "details": {
                        "dispersion": dispersion,
                        "chi_square": chi_square,
                        "df": df,
                        "p_value": p_value,
                        "method": "Pearson chi-square dispersion test"
                    }
                }
                
            elif model_type == 'binomial':
                # For binomial models, check for extra-binomial variation
                # Calculate Pearson residuals
                var_pred = predicted * (1 - predicted)
                pearson_residuals = (observed - predicted) / np.sqrt(var_pred)
                
                # Sum of squared Pearson residuals
                chi_square = np.sum(pearson_residuals ** 2)
                
                # Dispersion parameter
                n = len(observed)
                p = kwargs.get('num_params', 1)  # Number of parameters in the model
                df = n - p
                dispersion = chi_square / df
                
                # Test if dispersion is significantly > 1
                p_value = 1 - stats.chi2.cdf(chi_square, df)
                
                if dispersion <= 1.1:
                    result = AssumptionResult.PASSED
                    message = f"No overdispersion detected (dispersion={dispersion:.3f})."
                elif dispersion <= 1.5:
                    result = AssumptionResult.WARNING
                    message = f"Mild overdispersion detected (dispersion={dispersion:.3f})."
                else:
                    result = AssumptionResult.FAILED
                    message = f"Significant overdispersion detected (dispersion={dispersion:.3f})."
                
                return {
                    "result": result,
                    "message": message,
                    "details": {
                        "dispersion": dispersion,
                        "chi_square": chi_square,
                        "df": df,
                        "p_value": p_value,
                        "method": "Pearson chi-square dispersion test"
                    }
                }
            
            else:
                return {
                    "result": AssumptionResult.NOT_APPLICABLE,
                    "message": f"Unknown model type: {model_type}",
                    "details": {"error": f"Unknown model type: {model_type}"}
                }
                
        except Exception as e:
            return {
                "result": AssumptionResult.FAILED,
                "message": f"Error in overdispersion test: {str(e)}",
                "details": {"error": str(e)}
            }
        
class ProportionalOddsTest(AssumptionTest):
    """Tests the proportional odds assumption in ordinal logistic regression."""

    def __init__(self):
        super().__init__(
            name="Proportional Odds Test",
            description="Tests the proportional odds assumption in ordinal logistic regression.",
            applicable_roles=["outcome", "covariates"],
            applicable_types=["ordinal", "continuous", "categorical"]
        )

    def run_test(self, outcome, covariates, **kwargs):
        """
        Test proportional odds assumption using the Brant test.
        
        Parameters:
        -----------
        outcome : array-like
            Ordinal outcome variable
        covariates : DataFrame
            Covariate data
        
        Returns:
        --------
        dict
            Dictionary containing test results
        """
        import numpy as np
        import pandas as pd
        from scipy import stats
        
        alpha = kwargs.get('alpha', 0.05)
        
        try:
            # Convert inputs to numpy arrays or pandas objects
            if not isinstance(covariates, pd.DataFrame):
                raise ValueError("Covariates must be provided as a pandas DataFrame")
            
            outcome = np.array(outcome)
            
            # Check if outcome is ordinal (has ordered categories)
            unique_values = np.unique(outcome)
            if len(unique_values) < 3:
                return {
                    "result": AssumptionResult.NOT_APPLICABLE,
                    "message": "Outcome variable must have at least 3 ordinal categories",
                    "statistic": None,
                    "details": {"error": "Insufficient categories"}
                }
            
            # Simplified Brant test - we'll fit binary logistic regressions for each cut-point
            # and compare coefficients
            
            # First, we'll get the ordered categories
            categories = sorted(unique_values)
            
            # For each threshold between categories, fit a binary logistic regression
            coefficients = []
            standard_errors = []
            
            for i in range(len(categories) - 1):
                threshold = categories[i]
                
                # Create binary outcome for this threshold
                binary_outcome = (outcome > threshold).astype(int)
                
                # Fit logistic regression (using statsmodels would be better but we'll simplify)
                # We'll run a simple logistic regression for each covariate separately
                threshold_coeffs = {}
                threshold_ses = {}
                
                for col in covariates.columns:
                    X = covariates[col].values
                    X = sm.add_constant(X)  # Add intercept
                    
                    try:
                        # Fit logistic regression
                        model = sm.Logit(binary_outcome, X).fit(disp=0)
                        
                        # Store coefficient and SE (excluding intercept)
                        threshold_coeffs[col] = model.params[1]
                        threshold_ses[col] = model.bse[1]
                    except:
                        # If model fails to converge or other issues
                        threshold_coeffs[col] = np.nan
                        threshold_ses[col] = np.nan
                
                coefficients.append(threshold_coeffs)
                standard_errors.append(threshold_ses)
            
            # Now test if coefficients are similar across thresholds
            # For each covariate, compute chi-square test of coefficient differences
            chi2_results = {}
            p_values = {}
            
            for col in covariates.columns:
                # Extract coefficients and SEs for this covariate
                coeff_values = [coeff[col] for coeff in coefficients if col in coeff]
                se_values = [se[col] for se in standard_errors if col in se]
                
                # Remove any NaN values
                valid_indices = [i for i, (c, s) in enumerate(zip(coeff_values, se_values)) 
                               if not (np.isnan(c) or np.isnan(s))]
                
                if len(valid_indices) < 2:
                    chi2_results[col] = np.nan
                    p_values[col] = np.nan
                    continue
                
                coeff_values = [coeff_values[i] for i in valid_indices]
                se_values = [se_values[i] for i in valid_indices]
                
                # Calculate mean coefficient
                weights = [1 / (se ** 2) for se in se_values]
                mean_coeff = sum(c * w for c, w in zip(coeff_values, weights)) / sum(weights)
                
                # Calculate chi-square statistic
                chi2 = sum(((c - mean_coeff) / s) ** 2 for c, s in zip(coeff_values, se_values))
                df = len(coeff_values) - 1
                
                # Calculate p-value
                p_value = 1 - stats.chi2.cdf(chi2, df)
                
                chi2_results[col] = chi2
                p_values[col] = p_value
            
            # Find variables that violate proportional odds
            violating_vars = [col for col, p in p_values.items() if not np.isnan(p) and p < alpha]
            
            # Combine into a single overall test
            all_chi2 = sum(chi2 for chi2 in chi2_results.values() if not np.isnan(chi2))
            all_df = sum(1 for p in p_values.values() if not np.isnan(p))
            if all_df > 0:
                overall_p = 1 - stats.chi2.cdf(all_chi2, all_df)
            else:
                overall_p = np.nan
            
            if np.isnan(overall_p):
                result = AssumptionResult.NOT_APPLICABLE
                message = "Could not test proportional odds assumption due to convergence issues."
            elif overall_p >= alpha:
                result = AssumptionResult.PASSED
                message = f"Proportional odds assumption appears to be satisfied (p={overall_p:.4f})."
            elif len(violating_vars) <= 1:
                result = AssumptionResult.WARNING
                message = f"Proportional odds assumption may be violated for: {', '.join(violating_vars)}"
            else:
                result = AssumptionResult.FAILED
                message = f"Proportional odds assumption violated for: {', '.join(violating_vars)}"
            
            return {
                "result": result,
                "message": message,
                "details": {
                    "overall_chi2": all_chi2,
                    "overall_df": all_df,
                    "overall_p_value": overall_p,
                    "individual_chi2": chi2_results,
                    "individual_p_values": p_values,
                    "violating_variables": violating_vars,
                    "method": "Approximate Brant test"
                }
            }
            
        except Exception as e:
            return {
                "result": AssumptionResult.FAILED,
                "message": f"Error in proportional odds test: {str(e)}",
                "details": {"error": str(e)}
            }

class RandomEffectsNormalityTest(AssumptionTest):
    """Test for normality of random effects in mixed models."""
    
    def __init__(self):
        super().__init__(
            name="Random Effects Normality",
            description="Tests whether random effects follow a normal distribution",
            applicable_roles=["outcome", "covariates", "subject_id"],
            applicable_types=["random_effects"]
        )
    
    def run_test(self, random_effects, **kwargs):
        """
        Test whether random effects follow a normal distribution.
        
        Parameters
        ----------
        random_effects : pandas.DataFrame or dict
            DataFrame or dictionary containing random effects
        
        Returns
        -------
        dict
            Dictionary containing test results following the RandomEffectsNormalityTest format
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats
        
        alpha = kwargs.get('alpha', 0.05)
        warnings = []
        
        # Create figures
        fig_qq = plt.figure(figsize=(10, 6))
        fig_qq.patch.set_alpha(0.0)
        fig_hist = plt.figure(figsize=(10, 6))
        fig_hist.patch.set_alpha(0.0)
        
        # Create summary dictionary for random effects
        random_effects_summary = {}
        
        # Initialize results dictionary
        results = {}
        overall_result = AssumptionResult.PASSED
        overall_p_value = None
        overall_statistic = None
        
        if isinstance(random_effects, dict):
            # Handle case where random_effects is a dictionary of effects
            for i, (effect_name, effect_values) in enumerate(random_effects.items()):
                # Create subplots for each effect
                if i == 0:  # Use first effect for main plots
                    ax_qq = fig_qq.add_subplot(111)
                    ax_qq.patch.set_alpha(0.0)
                    ax_hist = fig_hist.add_subplot(111)
                    ax_hist.patch.set_alpha(0.0)
                    
                    # Store summary statistics
                    random_effects_summary[effect_name] = {
                        'mean': np.mean(effect_values),
                        'std': np.std(effect_values),
                        'min': np.min(effect_values),
                        'max': np.max(effect_values),
                        'n': len(effect_values)
                    }
                    
                    # Create QQ plot and histogram for first effect
                    if len(effect_values) >= 8:
                        stats.probplot(effect_values, plot=ax_qq)
                        ax_qq.set_title(f'Q-Q Plot for {effect_name}')
                        
                        ax_hist.hist(effect_values, bins='auto', density=True, alpha=0.7, color=PASTEL_COLORS[0])
                        x = np.linspace(min(effect_values), max(effect_values), 100)
                        ax_hist.plot(x, stats.norm.pdf(x, np.mean(effect_values), np.std(effect_values)), 'r-', lw=2)
                        ax_hist.set_title(f'Histogram for {effect_name}')
                
                if len(effect_values) >= 8:  # Minimum sample size for reliable tests
                    stat, p_value = stats.shapiro(effect_values)
                    
                    if p_value <= alpha:
                        effect_result = AssumptionResult.FAILED
                        warnings.append(f"Random effect '{effect_name}' is not normally distributed (p={p_value:.4f})")
                        overall_result = AssumptionResult.FAILED
                    else:
                        effect_result = AssumptionResult.PASSED
                    
                    message = f"Shapiro-Wilk test: p={p_value:.4f}"
                    
                    # Store first effect's p-value and statistic as overall
                    if i == 0:
                        overall_p_value = p_value
                        overall_statistic = stat
                    
                    results[effect_name] = {
                        'result': effect_result,
                        'p_value': p_value,
                        'statistic': stat,
                        'message': message
                    }
                else:
                    results[effect_name] = {
                        'result': AssumptionResult.NOT_APPLICABLE,
                        'message': "Not enough data for normality test (n<8)"
                    }
                    warnings.append(f"Insufficient data for testing normality of '{effect_name}' (n<8)")
        
        elif isinstance(random_effects, pd.DataFrame) or isinstance(random_effects, pd.Series):
            # Handle case where random_effects is a DataFrame or Series
            for i, col in enumerate(random_effects.columns if isinstance(random_effects, pd.DataFrame) else [random_effects.name]):
                values = random_effects[col] if isinstance(random_effects, pd.DataFrame) else random_effects
                
                # Create subplots for each column
                if i == 0:  # Use first column for main plots
                    ax_qq = fig_qq.add_subplot(111)
                    ax_qq.patch.set_alpha(0.0)
                    ax_hist = fig_hist.add_subplot(111)
                    ax_hist.patch.set_alpha(0.0)
                    
                    # Store summary statistics
                    random_effects_summary[col] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max(),
                        'n': len(values)
                    }
                    
                    # Create QQ plot and histogram for first column
                    if len(values) >= 8:
                        stats.probplot(values, plot=ax_qq)
                        ax_qq.set_title(f'Q-Q Plot for {col}')
                        
                        ax_hist.hist(values, bins='auto', density=True, alpha=0.7, color=PASTEL_COLORS[0])
                        x = np.linspace(min(values), max(values), 100)
                        ax_hist.plot(x, stats.norm.pdf(x, values.mean(), values.std()), 'r-', lw=2)
                        ax_hist.set_title(f'Histogram for {col}')
                
                if len(values) >= 8:
                    stat, p_value = stats.shapiro(values)
                    
                    if p_value <= alpha:
                        col_result = AssumptionResult.FAILED
                        warnings.append(f"Random effect '{col}' is not normally distributed (p={p_value:.4f})")
                        overall_result = AssumptionResult.FAILED
                    else:
                        col_result = AssumptionResult.PASSED
                    
                    message = f"Shapiro-Wilk test: p={p_value:.4f}"
                    
                    # Store first column's p-value and statistic as overall
                    if i == 0:
                        overall_p_value = p_value
                        overall_statistic = stat
                    
                    results[col] = {
                        'result': col_result,
                        'p_value': p_value,
                        'statistic': stat,
                        'message': message
                    }
                else:
                    results[col] = {
                        'result': AssumptionResult.NOT_APPLICABLE,
                        'message': "Not enough data for normality test (n<8)"
                    }
                    warnings.append(f"Insufficient data for testing normality of '{col}' (n<8)")
        
        else:
            overall_result = AssumptionResult.NOT_APPLICABLE
            warnings.append("Invalid input type for random effects")
        
        # Create overall details message
        if overall_result == AssumptionResult.PASSED:
            details = "Random effects appear to be normally distributed"
        elif overall_result == AssumptionResult.FAILED:
            details = "One or more random effects are not normally distributed"
        else:
            details = "Could not test normality of random effects"
        
        return {
            'result': overall_result,
            'statistic': overall_statistic,
            'p_value': overall_p_value,
            'details': details,
            'test_used': 'shapiro',
            'random_effects_summary': random_effects_summary,
            'warnings': warnings,
            'figures': {
                'qq_plot': fig_to_svg(fig_qq),
                'histogram': fig_to_svg(fig_hist)
            }
        }

class HomoscedasticityTest(AssumptionTest):
    """Test for homoscedasticity in regression models."""
    
    def __init__(self):
        super().__init__(
            name="Homoscedasticity",
            description="Tests whether the variance of residuals is constant across the range of predictors",
            applicable_roles=["outcome", "covariates"],
            applicable_types=["continuous"]
        )
    
    def run_test(self, residuals, predicted=None, **kwargs):
        """
        Test for homoscedasticity using Breusch-Pagan or White test.
        
        Parameters
        ----------
        residuals : array-like
            Model residuals
        predicted : array-like, optional
            Predicted values from the model
            
        Returns
        -------
        dict
            Dictionary containing test results following the HomoscedasticityTest format
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from scipy import stats
        from statsmodels.stats.diagnostic import het_breuschpagan, het_white
        
        alpha = kwargs.get('alpha', 0.05)
        exog = kwargs.get('exog', None)
        warnings = []
        
        # Create figures
        fig_residual = plt.figure(figsize=(10, 6))
        fig_residual.patch.set_alpha(0.0)
        ax_residual = fig_residual.add_subplot(111)
        ax_residual.patch.set_alpha(0.0)
        
        fig_scale_location = plt.figure(figsize=(10, 6))
        fig_scale_location.patch.set_alpha(0.0)
        ax_scale_location = fig_scale_location.add_subplot(111)
        ax_scale_location.patch.set_alpha(0.0)
        
        if predicted is None and exog is None:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'statistic': None,
                'p_value': None,
                'details': "Either predicted values or exogenous variables are required",
                'test_used': None,
                'warnings': ["Missing predicted values or exogenous variables"],
                'figures': {
                    'residual_plot': fig_to_svg(fig_residual),
                    'scale_location_plot': fig_to_svg(fig_scale_location)
                }
            }
        
        try:
            # Convert inputs to numpy arrays
            residuals = np.array(residuals)
            if predicted is not None:
                predicted = np.array(predicted)
            
            # Check if we have enough data points
            if len(residuals) < 10:
                return {
                    'result': AssumptionResult.NOT_APPLICABLE,
                    'statistic': None,
                    'p_value': None,
                    'details': "Not enough data points for homoscedasticity test (minimum 10 required)",
                    'test_used': None,
                    'warnings': ["Insufficient data for homoscedasticity testing"],
                    'figures': {
                        'residual_plot': fig_to_svg(fig_residual),
                        'scale_location_plot': fig_to_svg(fig_scale_location)
                    }
                }
            
            # Check for constant residuals
            if np.std(residuals) < 1e-10:
                warnings.append("Residuals have near-zero variance, test results may be unreliable")
            
            # Breusch-Pagan test if we have exogenous variables
            if exog is not None:
                try:
                    # Make sure exog has a constant term
                    from statsmodels.tools.tools import add_constant
                    if isinstance(exog, pd.DataFrame):
                        # Check if there's already a constant column
                        has_const = False
                        for col in exog.columns:
                            if np.all(exog[col] == 1):
                                has_const = True
                                break
                        if not has_const:
                            exog = add_constant(exog)
                    else:
                        # For numpy arrays, check if first column is constant
                        if not np.all(exog[:, 0] == exog[0, 0]):
                            exog = add_constant(exog)
                    
                    bp_test = het_breuschpagan(residuals, exog)
                    stat = bp_test[0]
                    p_value = bp_test[1]
                    test_name = "Breusch-Pagan"
                except Exception as e:
                    warnings.append(f"Breusch-Pagan test failed: {str(e)}")
                    # Fall back to visual inspection
                    stat = None
                    p_value = None
                    test_name = "Visual inspection only"
            # White test using squared predicted values if no exog is provided
            elif predicted is not None:
                try:
                    # Create exog matrix with constant, predicted and predicted^2
                    from statsmodels.tools.tools import add_constant
                    pred_exog = np.column_stack((predicted, predicted**2))
                    pred_exog = add_constant(pred_exog)  # Add constant term
                    
                    # Check for collinearity in pred_exog
                    if np.linalg.matrix_rank(pred_exog) < pred_exog.shape[1]:
                        warnings.append("Collinearity detected in predictors, using Breusch-Pagan test instead")
                        # Use simpler test
                        pred_exog = add_constant(predicted.reshape(-1, 1))
                        bp_test = het_breuschpagan(residuals, pred_exog)
                        test_name = "Breusch-Pagan"
                    else:
                        # Use White test
                        bp_test = het_white(residuals, pred_exog)
                        test_name = "White"
                    
                    stat = bp_test[0]
                    p_value = bp_test[1]
                except Exception as e:
                    warnings.append(f"Heteroscedasticity test failed: {str(e)}")
                    # Fall back to Goldfeld-Quandt test
                    try:
                        # Sort by predicted values
                        sorted_indices = np.argsort(predicted)
                        sorted_residuals = residuals[sorted_indices]
                        
                        # Split into two groups
                        n = len(sorted_residuals)
                        middle = n // 2
                        omit = n // 10  # Omit middle 10%
                        
                        group1 = sorted_residuals[:middle - omit//2]
                        group2 = sorted_residuals[middle + omit//2:]
                        
                        # Calculate variances
                        var1 = np.var(group1)
                        var2 = np.var(group2)
                        
                        # Calculate F statistic
                        if var1 > var2:
                            f_stat = var1 / var2
                        else:
                            f_stat = var2 / var1
                        
                        # Calculate p-value
                        df1 = len(group1) - 1
                        df2 = len(group2) - 1
                        p_value = 2 * (1 - stats.f.cdf(f_stat, df1, df2))
                        
                        stat = f_stat
                        test_name = "Goldfeld-Quandt"
                    except Exception as e2:
                        warnings.append(f"Goldfeld-Quandt test failed: {str(e2)}")
                        # Fall back to visual inspection
                        stat = None
                        p_value = None
                        test_name = "Visual inspection only"
            
            # Determine result
            if p_value is None:
                result = AssumptionResult.WARNING
                details = "Could not perform formal test, check plots for visual assessment"
                warnings.append("Formal test could not be performed, results based on visual inspection")
            elif p_value > alpha:
                result = AssumptionResult.PASSED
                details = f"Residuals appear to have constant variance ({test_name} test: p={p_value:.4f})"
            else:
                result = AssumptionResult.FAILED
                details = f"Residuals show heteroscedasticity ({test_name} test: p={p_value:.4f})"
                warnings.append("Heteroscedasticity may affect standard errors and hypothesis tests")
            
            # Create residual plot
            if predicted is not None:
                ax_residual.scatter(predicted, residuals, alpha=0.6, color=PASTEL_COLORS[0])
                ax_residual.axhline(y=0, color='r', linestyle='-')
                ax_residual.set_title('Residuals vs. Fitted Values')
                ax_residual.set_xlabel('Fitted Values')
                ax_residual.set_ylabel('Residuals')
                
                # Create scale-location plot (sqrt of abs standardized residuals vs fitted)
                std_resid = residuals / np.std(residuals) if np.std(residuals) > 0 else residuals
                sqrt_abs_resid = np.sqrt(np.abs(std_resid))
                ax_scale_location.scatter(predicted, sqrt_abs_resid, alpha=0.6, color=PASTEL_COLORS[1])
                ax_scale_location.set_title('Scale-Location Plot')
                ax_scale_location.set_xlabel('Fitted Values')
                ax_scale_location.set_ylabel('√|Standardized Residuals|')
                
                # Add lowess line to scale-location plot
                try:
                    import statsmodels.api as sm
                    # Use a smaller fraction for larger datasets
                    frac = min(0.6, max(0.2, 30 / len(predicted)))
                    lowess = sm.nonparametric.lowess(sqrt_abs_resid, predicted, frac=frac)
                    ax_scale_location.plot(lowess[:, 0], lowess[:, 1], 'r-', lw=1)
                except Exception as e:
                    warnings.append(f"Could not add LOWESS line to scale-location plot: {str(e)}")
            else:
                # If no predicted values, just plot residuals in sequence
                ax_residual.plot(residuals, 'o-', color=PASTEL_COLORS[0])
                ax_residual.axhline(y=0, color='r', linestyle='-')
                ax_residual.set_title('Residual Plot')
                ax_residual.set_xlabel('Index')
                ax_residual.set_ylabel('Residuals')
                
                # Simple scale-location plot
                std_dev = np.std(residuals)
                if std_dev > 0:
                    sqrt_abs_resid = np.sqrt(np.abs(residuals / std_dev))
                else:
                    sqrt_abs_resid = np.sqrt(np.abs(residuals))
                    warnings.append("Zero standard deviation in residuals, could not standardize")
                
                ax_scale_location.plot(sqrt_abs_resid, 'o-', color=PASTEL_COLORS[1])
                ax_scale_location.set_title('Scale-Location Plot')
                ax_scale_location.set_xlabel('Index')
                ax_scale_location.set_ylabel('√|Standardized Residuals|')
            
            return {
                'result': result,
                'statistic': stat,
                'p_value': p_value,
                'details': details,
                'test_used': test_name,
                'warnings': warnings,
                'figures': {
                    'residual_plot': fig_to_svg(fig_residual),
                    'scale_location_plot': fig_to_svg(fig_scale_location)
                }
            }
            
        except Exception as e:
            # Create basic plots even if test fails
            try:
                if predicted is not None:
                    ax_residual.scatter(predicted, residuals, alpha=0.6, color=PASTEL_COLORS[0])
                else:
                    ax_residual.plot(residuals, 'o-', color=PASTEL_COLORS[0])
                ax_residual.axhline(y=0, color='r', linestyle='-')
                ax_residual.set_title('Residual Plot (Test Failed)')
                
                ax_scale_location.text(0.5, 0.5, f"Test failed: {str(e)}", 
                                     horizontalalignment='center', verticalalignment='center')
            except:
                pass
                
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'statistic': None,
                'p_value': None,
                'details': f"Error in homoscedasticity test: {str(e)}",
                'test_used': None,
                'warnings': [f"Error in homoscedasticity test: {str(e)}"],
                'figures': {
                    'residual_plot': fig_to_svg(fig_residual),
                    'scale_location_plot': fig_to_svg(fig_scale_location)
                }
            }
class InfluentialPointsTest(AssumptionTest):
    """Test for influential points in regression models using Cook's Distance and other measures."""
    
    def __init__(self):
        super().__init__(
            name="Influential Points Test",
            description="Identifies influential points that may disproportionately affect regression results",
            applicable_roles=["residuals", "leverage", "fitted"],
            applicable_types=["continuous"]
        )
        
    def run_test(self, residuals, leverage=None, fitted=None, X=None, **kwargs):
        """
        Detect influential points using Cook's Distance and other diagnostics.
        
        Parameters
        ----------
        residuals : array-like
            Residuals from the regression model
        leverage : array-like, optional
            Leverage values (diagonal elements of the hat matrix)
        fitted : array-like, optional
            Fitted values from the model
        X : array-like or DataFrame, optional
            Design matrix, used to calculate leverage if not provided
        **kwargs : dict
            Additional parameters including:
            - n_params: int, number of parameters in the model (including intercept)
            - method: str, 'cooks_distance', 'dffits', or 'all' (default)
            - threshold: float, threshold for flagging influential points (default depends on method)
            
        Returns
        -------
        dict
            Dictionary containing test results following the InfluentialPointsTest format
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from scipy import stats
        import statsmodels.api as sm
        
        method = kwargs.get('method', 'all')
        alpha = kwargs.get('alpha', 0.05)
        n_params = kwargs.get('n_params', 2)  # Default assumes intercept + 1 predictor
        warnings = []
        
        # Create figures
        fig_cooks = plt.figure(figsize=(10, 6))
        fig_cooks.patch.set_alpha(0.0)
        ax_cooks = fig_cooks.add_subplot(111)
        ax_cooks.patch.set_alpha(0.0)
        
        fig_leverage = plt.figure(figsize=(10, 6))
        fig_leverage.patch.set_alpha(0.0)
        ax_leverage = fig_leverage.add_subplot(111)
        ax_leverage.patch.set_alpha(0.0)
        
        # Convert inputs to numpy arrays
        residuals = np.array(residuals)
        n = len(residuals)
        
        if n < n_params + 2:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': f"Not enough data points ({n}) for influential points detection with {n_params} parameters",
                'warnings': ["Sample size too small for influence diagnostics"],
                'figures': {
                    'cooks_distance': fig_to_svg(fig_cooks),
                    'leverage_plot': fig_to_svg(fig_leverage)
                }
            }
        
        # Calculate leverage if not provided
        if leverage is None:
            if X is not None:
                try:
                    # Convert X to numpy array if it's a DataFrame
                    if isinstance(X, pd.DataFrame):
                        X_array = X.values
                    else:
                        X_array = np.array(X)
                    
                    # Check if X already has a constant term
                    has_const = False
                    if X_array.shape[1] > 0:
                        const_col = np.ones(X_array.shape[0])
                        for i in range(X_array.shape[1]):
                            if np.allclose(X_array[:, i], const_col):
                                has_const = True
                                break
                    
                    # Add constant if needed
                    if not has_const:
                        X_with_const = sm.add_constant(X_array)
                    else:
                        X_with_const = X_array
                    
                    # Calculate hat matrix diagonal (leverage)
                    hat_matrix = X_with_const.dot(np.linalg.pinv(X_with_const.T.dot(X_with_const))).dot(X_with_const.T)
                    leverage = np.diag(hat_matrix)
                except Exception as e:
                    warnings.append(f"Could not calculate leverage from X: {str(e)}")
                    # Use a placeholder for leverage
                    leverage = np.ones(n) / n
            else:
                warnings.append("Neither leverage nor design matrix X provided, using uniform leverage")
                leverage = np.ones(n) / n
        else:
            leverage = np.array(leverage)
            
        if fitted is not None:
            fitted = np.array(fitted)
        
        # Calculate standardized residuals
        try:
            mse = np.sum(residuals**2) / (n - n_params)
            std_resid = residuals / np.sqrt(mse * (1 - leverage))
        except Exception as e:
            warnings.append(f"Could not calculate standardized residuals: {str(e)}")
            # Use regular residuals divided by their standard deviation
            std_resid = residuals / np.std(residuals) if np.std(residuals) > 0 else residuals
        
        # Calculate Cook's Distance
        try:
            cooks_d = (std_resid**2 * leverage) / (n_params * (1 - leverage))
        except Exception as e:
            warnings.append(f"Could not calculate Cook's Distance: {str(e)}")
            cooks_d = np.zeros(n)
        
        # Calculate DFFITS (scaled difference in fits)
        try:
            dffits = std_resid * np.sqrt(leverage / (1 - leverage))
        except Exception as e:
            warnings.append(f"Could not calculate DFFITS: {str(e)}")
            dffits = np.zeros(n)
        
        # Set thresholds for influence measures
        # Common rule for Cook's D: 4/n or 4/(n-k-1)
        cooks_threshold = kwargs.get('cooks_threshold', 4 / (n - n_params))
        # Common rule for leverage: 2*p/n or 3*p/n
        leverage_threshold = kwargs.get('leverage_threshold', 3 * n_params / n)
        # Common rule for DFFITS: 2*sqrt(p/n)
        dffits_threshold = kwargs.get('dffits_threshold', 2 * np.sqrt(n_params / n))
        
        # Identify influential points
        influential_points = {
            'cooks_d': {
                'indices': np.where(cooks_d > cooks_threshold)[0].tolist(),
                'values': cooks_d[cooks_d > cooks_threshold].tolist(),
                'threshold': cooks_threshold
            },
            'leverage': {
                'indices': np.where(leverage > leverage_threshold)[0].tolist(),
                'values': leverage[leverage > leverage_threshold].tolist(),
                'threshold': leverage_threshold
            },
            'dffits': {
                'indices': np.where(np.abs(dffits) > dffits_threshold)[0].tolist(),
                'values': dffits[np.abs(dffits) > dffits_threshold].tolist(),
                'threshold': dffits_threshold
            }
        }
        
        # Combine indices from different methods
        all_influential = set()
        for measure in influential_points:
            all_influential.update(influential_points[measure]['indices'])
        
        all_influential = sorted(list(all_influential))
        
        # Determine result
        if len(all_influential) == 0:
            result = AssumptionResult.PASSED
            details = "No influential points detected"
        elif len(all_influential) <= n * 0.05:  # Less than 5% of points are influential
            result = AssumptionResult.WARNING
            details = f"Detected {len(all_influential)} potentially influential points"
            warnings.append("Influential points may affect regression results")
        else:
            result = AssumptionResult.FAILED
            details = f"Detected {len(all_influential)} influential points ({len(all_influential)/n:.1%} of data)"
            warnings.append("Large number of influential points may indicate model issues")
        
        # Create Cook's distance plot
        markerline, stemlines, baseline = ax_cooks.stem(np.arange(len(cooks_d)), cooks_d, markerfmt='.', linefmt='-', basefmt=' ')
        ax_cooks.axhline(y=cooks_threshold, color='r', linestyle='--', label=f'Threshold ({cooks_threshold:.4f})')
        ax_cooks.set_title("Cook's Distance")
        ax_cooks.set_xlabel('Observation Index')
        ax_cooks.set_ylabel("Cook's Distance")
        ax_cooks.legend()
        
        # Highlight influential points
        if influential_points['cooks_d']['indices']:
            ax_cooks.scatter(influential_points['cooks_d']['indices'], 
                           influential_points['cooks_d']['values'], 
                           color='red', s=50, label='Influential')
            # Add labels for top 3 most influential points
            if len(influential_points['cooks_d']['indices']) > 0:
                top_indices = np.argsort(np.array(influential_points['cooks_d']['values']))[-3:]
                for i in top_indices:
                    idx = influential_points['cooks_d']['indices'][i]
                    val = influential_points['cooks_d']['values'][i]
                    ax_cooks.annotate(f'{idx}', (idx, val), textcoords="offset points", 
                                     xytext=(0,10), ha='center')
        
        # Create leverage vs. standardized residuals plot (with contours for Cook's distance)
        ax_leverage.scatter(leverage, std_resid, alpha=0.6, color=PASTEL_COLORS[0])
        ax_leverage.axhline(y=0, color='k', linestyle='-')
        ax_leverage.axvline(x=leverage_threshold, color='r', linestyle='--', 
                          label=f'Leverage threshold ({leverage_threshold:.4f})')
        
        # Add threshold lines for standardized residuals at ±3
        ax_leverage.axhline(y=3, color='r', linestyle='--')
        ax_leverage.axhline(y=-3, color='r', linestyle='--')
        
        # Highlight points with high leverage or standardized residuals
        high_leverage = leverage > leverage_threshold
        large_residual = np.abs(std_resid) > 3
        
        if np.any(high_leverage):
            ax_leverage.scatter(leverage[high_leverage], std_resid[high_leverage], 
                              color='orange', s=50, label='High Leverage')
        
        if np.any(large_residual):
            ax_leverage.scatter(leverage[large_residual], std_resid[large_residual], 
                              color='purple', s=50, label='Large Residual')
        
        # Highlight influential points (both high leverage and large residual)
        influential = high_leverage & large_residual
        if np.any(influential):
            ax_leverage.scatter(leverage[influential], std_resid[influential], 
                              color='red', s=100, label='Influential')
        
        # Try to add Cook's distance contours
        try:
            # Create a grid of points
            max_leverage = max(leverage.max() * 1.1, leverage_threshold * 1.5)
            max_resid = max(abs(std_resid).max() * 1.1, 3.5)
            
            h = np.linspace(0, max_leverage, 100)
            r = np.linspace(-max_resid, max_resid, 100)
            H, R = np.meshgrid(h, r)
            
            # Calculate Cook's distance for each grid point
            C = (R**2 * H) / (n_params * (1 - H))
            
            # Plot contours
            contour_levels = [cooks_threshold, 0.5, 1.0]
            contour_levels = [level for level in contour_levels if level <= C.max()]
            
            if contour_levels:
                CS = ax_leverage.contour(H, R, C, levels=contour_levels, 
                                     colors=['green', 'blue', 'red'], linestyles=['--', '-', '-'])
                ax_leverage.clabel(CS, inline=1, fontsize=8, fmt='%.2f')
        except Exception as e:
            warnings.append(f"Could not create Cook's distance contours: {str(e)}")
        
        ax_leverage.set_title('Leverage vs. Standardized Residuals')
        ax_leverage.set_xlabel('Leverage')
        ax_leverage.set_ylabel('Standardized Residuals')
        ax_leverage.legend(loc='best')
        
        return {
            'result': result,
            'details': details,
            'influential_points': {
                'combined': all_influential,
                'by_measure': influential_points
            },
            'statistics': {
                'cooks_d': cooks_d.tolist(),
                'leverage': leverage.tolist(),
                'std_residuals': std_resid.tolist(),
                'dffits': dffits.tolist()
            },
            'thresholds': {
                'cooks_d': cooks_threshold,
                'leverage': leverage_threshold,
                'dffits': dffits_threshold
            },
            'warnings': warnings,
            'figures': {
                'cooks_distance': fig_to_svg(fig_cooks),
                'leverage_plot': fig_to_svg(fig_leverage)
            }
        }

class ModelSpecificationTest(AssumptionTest):
    """Tests for model specification errors in regression models."""
    
    def __init__(self):
        super().__init__(
            name="Model Specification Test",
            description="Tests for model specification errors such as omitted variables or incorrect functional form",
            applicable_roles=["residuals", "fitted", "X"],
            applicable_types=["continuous"]
        )
    
    def run_test(self, residuals, fitted, X=None, **kwargs):
        """
        Test for model specification errors using Ramsey's RESET test or similar methods.
        
        Parameters
        ----------
        residuals : array-like
            Residuals from the regression model
        fitted : array-like
            Fitted values from the model
        X : array-like or DataFrame, optional
            Design matrix, used for more advanced specification tests
        **kwargs : dict
            Additional parameters including:
            - method: str, 'reset' (default), 'link_test'
            - power: int, highest power of fitted values to include in RESET test (default: 3)
            
        Returns
        -------
        dict
            Dictionary containing test results
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from scipy import stats
        import statsmodels.api as sm
        
        method = kwargs.get('method', 'reset')
        power = kwargs.get('power', 3)
        alpha = kwargs.get('alpha', 0.05)
        warnings = []
        
        # Create figures
        fig_nonlinearity = plt.figure(figsize=(10, 6))
        fig_nonlinearity.patch.set_alpha(0.0)
        ax_nonlinearity = fig_nonlinearity.add_subplot(111)
        ax_nonlinearity.patch.set_alpha(0.0)
        
        # Convert inputs to numpy arrays
        residuals = np.array(residuals)
        fitted = np.array(fitted)
        
        if len(residuals) < 10:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': "Not enough data points for specification test",
                'warnings': ["Sample size too small for specification testing"],
                'figures': {
                    'nonlinearity_plot': fig_to_svg(fig_nonlinearity)
                }
            }
        
        # Run appropriate specification test
        if method == 'reset':
            try:
                # Add powers of fitted values to the regression
                X_powers = np.column_stack([fitted**(i+2) for i in range(power-1)])
                X_reset = sm.add_constant(X_powers)
                
                # Regress residuals on powers of fitted values
                model = sm.OLS(residuals, X_reset).fit()
                
                # Calculate F-statistic for joint significance
                f_stat = model.fvalue
                f_pvalue = model.f_pvalue
                
                # Extract R-squared
                r_squared = model.rsquared
                
                test_name = f"Ramsey RESET (powers 2 to {power})"
                
                if f_pvalue < alpha:
                    result = AssumptionResult.FAILED
                    details = f"Model may be misspecified ({test_name}: F={f_stat:.3f}, p={f_pvalue:.4f})"
                    warnings.append("Consider adding non-linear terms or omitted variables")
                else:
                    result = AssumptionResult.PASSED
                    details = f"No evidence of model misspecification ({test_name}: F={f_stat:.3f}, p={f_pvalue:.4f})"
                
                test_result = {
                    'f_statistic': f_stat,
                    'p_value': f_pvalue,
                    'r_squared': r_squared,
                    'method': test_name,
                    'df_numerator': power - 1,
                    'df_denominator': len(residuals) - power
                }
                
            except Exception as e:
                warnings.append(f"Could not perform RESET test: {str(e)}")
                test_result = {
                    'error': str(e)
                }
                result = AssumptionResult.WARNING
                details = f"Could not perform specification test: {str(e)}"
        
        elif method == 'link_test':
            try:
                # Calculate predicted values and their square
                X_link = sm.add_constant(np.column_stack((fitted, fitted**2)))
                
                # Regress residuals on predicted values and their square
                model = sm.OLS(residuals, X_link).fit()
                
                # Extract p-value for squared term
                p_value_squared = model.pvalues[-1]
                t_stat_squared = model.tvalues[-1]
                
                test_name = "Link Test"
                
                if p_value_squared < alpha:
                    result = AssumptionResult.FAILED
                    details = f"Model may be misspecified ({test_name}: t={t_stat_squared:.3f}, p={p_value_squared:.4f})"
                    warnings.append("Consider adding non-linear terms or omitted variables")
                else:
                    result = AssumptionResult.PASSED
                    details = f"No evidence of model misspecification ({test_name}: t={t_stat_squared:.3f}, p={p_value_squared:.4f})"
                
                test_result = {
                    't_statistic': t_stat_squared,
                    'p_value': p_value_squared,
                    'method': test_name
                }
                
            except Exception as e:
                warnings.append(f"Could not perform Link Test: {str(e)}")
                test_result = {
                    'error': str(e)
                }
                result = AssumptionResult.WARNING
                details = f"Could not perform specification test: {str(e)}"
        
        else:
            result = AssumptionResult.NOT_APPLICABLE
            details = f"Unknown specification test method: {method}"
            test_result = {
                'error': f"Unknown method: {method}"
            }
            warnings.append(f"Unknown specification test method: {method}")
        
        # Plot residuals vs fitted values^2 (to visualize non-linearity)
        ax_nonlinearity.scatter(fitted**2, residuals, alpha=0.6, color=PASTEL_COLORS[0])
        ax_nonlinearity.axhline(y=0, color='r', linestyle='-')
        ax_nonlinearity.set_title('Residuals vs. Fitted Values Squared')
        ax_nonlinearity.set_xlabel('Fitted Values Squared')
        ax_nonlinearity.set_ylabel('Residuals')
        
        # Add lowess line to help identify patterns
        try:
            # Use a smaller fraction for larger datasets
            frac = min(0.6, max(0.2, 30 / len(fitted)))
            lowess = sm.nonparametric.lowess(residuals, fitted**2, frac=frac)
            ax_nonlinearity.plot(lowess[:, 0], lowess[:, 1], 'r-', lw=1)
        except Exception as e:
            warnings.append(f"Could not add LOWESS line to nonlinearity plot: {str(e)}")
        
        return {
            'result': result,
            'details': details,
            'test_result': test_result,
            'warnings': warnings,
            'figures': {
                'nonlinearity_plot': fig_to_svg(fig_nonlinearity)
            }
        }

class DistributionFitTest(AssumptionTest):
    """Tests whether data follows a specific distribution like Poisson, exponential, etc."""
    
    def __init__(self):
        super().__init__(
            name="Distribution Fit Test",
            description="Tests whether data follows a specific probability distribution",
            applicable_roles=["outcome", "residuals", "data"],
            applicable_types=["continuous", "count", "positive"]
        )
    
    def run_test(self, data, distribution='normal', **kwargs):
        """
        Test whether data follows a specific probability distribution.
        
        Parameters
        ----------
        data : array-like
            Data to test for distribution fit
        distribution : str
            Distribution to test against: 'normal', 'poisson', 'exponential', 
            'gamma', 'lognormal', 'weibull', 'negative_binomial', 'uniform'
        **kwargs : dict
            Additional parameters including:
            - est_method: str, parameter estimation method ('mle', 'mm')
            - discrete: bool, whether the distribution is discrete
            
        Returns
        -------
        dict
            Dictionary containing test results
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from scipy import stats
        import statsmodels.api as sm
        import warnings as py_warnings
        
        # Suppress scipy warnings during fitting
        py_warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        alpha = kwargs.get('alpha', 0.05)
        est_method = kwargs.get('est_method', 'mle')
        discrete = kwargs.get('discrete', distribution in ['poisson', 'negative_binomial'])
        my_warnings = []  # Use different name to avoid conflict with python warnings
        
        # Create figures
        fig_hist = plt.figure(figsize=(10, 6))
        fig_hist.patch.set_alpha(0.0)
        ax_hist = fig_hist.add_subplot(111)
        ax_hist.patch.set_alpha(0.0)
        
        fig_pp = plt.figure(figsize=(10, 6))
        fig_pp.patch.set_alpha(0.0)
        ax_pp = fig_pp.add_subplot(111)
        ax_pp.patch.set_alpha(0.0)
        
        # Convert input to numpy array and remove NAs
        data = np.array(pd.Series(data).dropna())
        
        if len(data) < 8:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': "Not enough data points for distribution fit test",
                'warnings': ["Sample size too small for distribution fitting"],
                'figures': {
                    'histogram': fig_to_svg(fig_hist),
                    'pp_plot': fig_to_svg(fig_pp)
                }
            }
        
        # Check for negative values for distributions that require positive data
        positive_only = ['exponential', 'gamma', 'lognormal', 'weibull']
        if distribution in positive_only and np.any(data <= 0):
            my_warnings.append(f"Data contains non-positive values, which are invalid for {distribution} distribution")
            
            # Filter out non-positive values for fitting
            original_length = len(data)
            data = data[data > 0]
            
            if len(data) < 8:
                return {
                    'result': AssumptionResult.FAILED,
                    'details': f"Not enough positive data points for {distribution} distribution fit test",
                    'warnings': [
                        f"Removed {original_length - len(data)} non-positive values", 
                        f"{distribution} distribution requires positive values"
                    ],
                    'figures': {
                        'histogram': fig_to_svg(fig_hist),
                        'pp_plot': fig_to_svg(fig_pp)
                    }
                }
            
            my_warnings.append(f"Removed {original_length - len(data)} non-positive values for fitting")
        
        # For count data, check if values are integers
        if distribution in ['poisson', 'negative_binomial'] and not np.all(np.equal(np.mod(data, 1), 0)):
            my_warnings.append(f"Data contains non-integer values, which are invalid for {distribution} distribution")
            
            # Round values for fitting
            data = np.round(data)
            my_warnings.append("Non-integer values have been rounded for fitting")
        
        # Fit distribution and perform goodness-of-fit test
        try:
            if distribution == 'normal':
                # Fit normal distribution
                loc, scale = stats.norm.fit(data)
                dist = stats.norm(loc=loc, scale=scale)
                dist_name = "Normal"
                params = {'mean': loc, 'std': scale}
                
                # Perform Shapiro-Wilk test for normality
                if len(data) <= 5000:
                    stat, p_value = stats.shapiro(data)
                    test_name = "Shapiro-Wilk"
                else:
                    # For larger samples, use Anderson-Darling
                    result = stats.anderson(data, dist='norm')
                    stat = result.statistic
                    # Find critical value for alpha = 0.05
                    cv_idx = np.where(result.significance_level == 5)[0][0]
                    critical_value = result.critical_values[cv_idx]
                    p_value = 0.05 if stat < critical_value else 0.01
                    test_name = "Anderson-Darling"
                
            elif distribution == 'poisson':
                # Fit Poisson distribution
                lambda_param = np.mean(data)
                dist = stats.poisson(mu=lambda_param)
                dist_name = "Poisson"
                params = {'lambda': lambda_param}
                
                # Chi-square goodness of fit for discrete distribution
                observed = np.bincount(data.astype(int))
                k = len(observed)
                expected = np.zeros(k)
                for i in range(k):
                    expected[i] = dist.pmf(i) * len(data)
                
                # Combine bins with expected frequency < 5
                observed_adj, expected_adj = [], []
                i, current_obs, current_exp = 0, 0, 0
                
                while i < k:
                    current_obs += observed[i] if i < len(observed) else 0
                    current_exp += expected[i]
                    
                    if current_exp >= 5 or i == k - 1:
                        observed_adj.append(current_obs)
                        expected_adj.append(current_exp)
                        current_obs, current_exp = 0, 0
                    
                    i += 1
                
                # Calculate chi-square statistic if we have enough bins
                if len(expected_adj) > 1:
                    stat, p_value = stats.chisquare(f_obs=observed_adj, f_exp=expected_adj)
                    test_name = "Chi-square"
                else:
                    # Not enough bins for chi-square, use Kolmogorov-Smirnov as alternative
                    stat, p_value = stats.kstest(data, dist.cdf)
                    test_name = "Kolmogorov-Smirnov"
                    my_warnings.append("Too few distinct values for chi-square test, using K-S test instead")
                
            elif distribution == 'exponential':
                # Fit exponential distribution
                scale = np.mean(data)
                dist = stats.expon(scale=scale)
                dist_name = "Exponential"
                params = {'scale': scale}
                
                # Kolmogorov-Smirnov test
                stat, p_value = stats.kstest(data, dist.cdf)
                test_name = "Kolmogorov-Smirnov"
                
            elif distribution == 'gamma':
                # Fit gamma distribution
                shape, loc, scale = stats.gamma.fit(data)
                dist = stats.gamma(shape, loc=loc, scale=scale)
                dist_name = "Gamma"
                params = {'shape': shape, 'loc': loc, 'scale': scale}
                
                # Kolmogorov-Smirnov test
                stat, p_value = stats.kstest(data, dist.cdf)
                test_name = "Kolmogorov-Smirnov"
                
            elif distribution == 'lognormal':
                # Fit lognormal distribution
                shape, loc, scale = stats.lognorm.fit(data)
                dist = stats.lognorm(s=shape, loc=loc, scale=scale)
                dist_name = "Log-normal"
                params = {'shape': shape, 'loc': loc, 'scale': scale}
                
                # Kolmogorov-Smirnov test
                stat, p_value = stats.kstest(data, dist.cdf)
                test_name = "Kolmogorov-Smirnov"
                
            elif distribution == 'weibull':
                # Fit Weibull distribution
                shape, loc, scale = stats.weibull_min.fit(data)
                dist = stats.weibull_min(c=shape, loc=loc, scale=scale)
                dist_name = "Weibull"
                params = {'shape': shape, 'loc': loc, 'scale': scale}
                
                # Kolmogorov-Smirnov test
                stat, p_value = stats.kstest(data, dist.cdf)
                test_name = "Kolmogorov-Smirnov"
                
            elif distribution == 'negative_binomial':
                # Fit negative binomial distribution
                # Need to estimate n (number of successes) and p (probability of success)
                mean = np.mean(data)
                var = np.var(data)
                
                # Check if variance > mean (overdispersion)
                if var <= mean:
                    my_warnings.append("Variance ≤ mean, data may be better modeled by Poisson distribution")
                    
                # For negative binomial: mean = n(1-p)/p, var = n(1-p)/p²
                # Solving for n and p: p = mean/var, n = mean²/(var-mean)
                if var > mean:
                    p = mean / var
                    n = mean**2 / (var - mean)
                else:
                    # Default to reasonable values if underdispersed
                    p = 0.5
                    n = mean * 2
                
                dist = stats.nbinom(n=n, p=p)
                dist_name = "Negative Binomial"
                params = {'n': n, 'p': p}
                
                # Chi-square goodness of fit for discrete distribution
                observed = np.bincount(data.astype(int))
                k = len(observed)
                expected = np.zeros(k)
                for i in range(k):
                    expected[i] = dist.pmf(i) * len(data)
                
                # Combine bins with expected frequency < 5
                observed_adj, expected_adj = [], []
                i, current_obs, current_exp = 0, 0, 0
                
                while i < k:
                    current_obs += observed[i] if i < len(observed) else 0
                    current_exp += expected[i]
                    
                    if current_exp >= 5 or i == k - 1:
                        observed_adj.append(current_obs)
                        expected_adj.append(current_exp)
                        current_obs, current_exp = 0, 0
                    
                    i += 1
                
                # Calculate chi-square statistic if we have enough bins
                if len(expected_adj) > 1:
                    stat, p_value = stats.chisquare(f_obs=observed_adj, f_exp=expected_adj)
                    test_name = "Chi-square"
                else:
                    # Not enough bins for chi-square, use Kolmogorov-Smirnov as alternative
                    stat, p_value = stats.kstest(data, dist.cdf)
                    test_name = "Kolmogorov-Smirnov"
                    my_warnings.append("Too few distinct values for chi-square test, using K-S test instead")
                
            elif distribution == 'uniform':
                # Fit uniform distribution
                loc, scale = stats.uniform.fit(data)
                dist = stats.uniform(loc=loc, scale=scale)
                dist_name = "Uniform"
                params = {'loc': loc, 'scale': scale}
                
                # Kolmogorov-Smirnov test
                stat, p_value = stats.kstest(data, dist.cdf)
                test_name = "Kolmogorov-Smirnov"
                
            else:
                return {
                    'result': AssumptionResult.NOT_APPLICABLE,
                    'details': f"Unknown distribution: {distribution}",
                    'warnings': [f"Unsupported distribution type: {distribution}"],
                    'figures': {
                        'histogram': fig_to_svg(fig_hist),
                        'pp_plot': fig_to_svg(fig_pp)
                    }
                }
                
        except Exception as e:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': f"Error fitting {distribution} distribution: {str(e)}",
                'warnings': [f"Distribution fitting failed: {str(e)}"],
                'figures': {
                    'histogram': fig_to_svg(fig_hist),
                    'pp_plot': fig_to_svg(fig_pp)
                }
            }
        
        # Determine result
        if p_value > alpha:
            result = AssumptionResult.PASSED
            details = f"Data appears to follow {dist_name} distribution ({test_name} test: p={p_value:.4f})"
        else:
            result = AssumptionResult.FAILED
            details = f"Data does not appear to follow {dist_name} distribution ({test_name} test: p={p_value:.4f})"
            my_warnings.append(f"Consider using a different distribution for modeling this data")
        
        # Create histogram with distribution overlay
        if discrete:
            # For discrete distributions (e.g., Poisson), use bars
            max_val = min(int(max(data)) + 1, 50)  # Limit to reasonable range
            x = np.arange(0, max_val)
            pmf = dist.pmf(x)
            
            # Plot histogram with frequency counts
            counts, edges, _ = ax_hist.hist(data, bins=np.arange(-0.5, max_val + 0.5), 
                                        density=True, alpha=0.6, color=PASTEL_COLORS[0])
            
            # Overlay PMF
            markerline, stemlines, baseline = ax_hist.stem(x, pmf, linefmt='r-', markerfmt='ro', basefmt=' ')
            plt.setp(stemlines, 'linewidth', 1, 'alpha', 0.7)
            plt.setp(markerline, 'markersize', 3)
            
            ax_hist.set_title(f'Histogram with {dist_name} PMF Overlay')
            ax_hist.set_xlabel('Value')
            ax_hist.set_ylabel('Probability')
            
        else:
            # For continuous distributions, use density curve
            counts, edges, _ = ax_hist.hist(data, bins='auto', density=True, 
                                        alpha=0.6, color=PASTEL_COLORS[0])
            
            # Create x values for plotting the distribution
            x = np.linspace(min(data), max(data), 1000)
            ax_hist.plot(x, dist.pdf(x), 'r-', lw=2)
            
            ax_hist.set_title(f'Histogram with {dist_name} PDF Overlay')
            ax_hist.set_xlabel('Value')
            ax_hist.set_ylabel('Density')
        
        # Create P-P plot (probability plot)
        # Sort the data
        sorted_data = np.sort(data)
        
        # Calculate the empirical CDF (proportion of data points <= x)
        ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # Calculate the theoretical CDF for these points
        tcdf = dist.cdf(sorted_data)
        
        # Plot the P-P plot
        ax_pp.scatter(tcdf, ecdf, alpha=0.6, color=PASTEL_COLORS[1])
        ax_pp.plot([0, 1], [0, 1], 'r-', lw=1)  # Reference line
        ax_pp.set_title(f'P-P Plot for {dist_name} Distribution')
        ax_pp.set_xlabel('Theoretical Probability')
        ax_pp.set_ylabel('Empirical Probability')
        
        # Add distribution parameters to plot as text
        param_text = ""
        for param_name, param_value in params.items():
            param_text += f"{param_name} = {param_value:.4g}, "
        param_text = param_text[:-2]  # Remove trailing comma and space
        
        ax_hist.text(0.95, 0.95, param_text, transform=ax_hist.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return {
            'result': result,
            'details': details,
            'statistic': stat,
            'p_value': p_value,
            'distribution': dist_name,
            'distribution_params': params,
            'test_used': test_name,
            'warnings': my_warnings,
            'figures': {
                'histogram': fig_to_svg(fig_hist),
                'pp_plot': fig_to_svg(fig_pp)
            }
        }

class ZeroInflationTest(AssumptionTest):
    """Test for zero-inflation in count data."""
    
    def __init__(self):
        super().__init__(
            name="Zero Inflation Test",
            description="Tests whether count data has excessive zeros compared to standard distributions",
            applicable_roles=["outcome", "response"],
            applicable_types=["count"]
        )
    
    def run_test(self, data, **kwargs):
        """
        Test whether count data exhibits zero-inflation.
        
        Parameters
        ----------
        data : array-like
            Count data to test for zero-inflation
        **kwargs : dict
            Additional parameters including:
            - distribution: str, base distribution to compare against ('poisson' or 'negative_binomial')
            - model_regressors: array-like or DataFrame, optional covariates for model fitting
            
        Returns
        -------
        dict
            Dictionary containing test results
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from scipy import stats
        import statsmodels.api as sm
        import warnings as py_warnings
        
        # Suppress warnings during fitting
        py_warnings.filterwarnings('ignore')
        
        alpha = kwargs.get('alpha', 0.05)
        distribution = kwargs.get('distribution', 'poisson')
        regressors = kwargs.get('model_regressors', None)
        my_warnings = []
        
        # Create figures
        fig_observed_expected = plt.figure(figsize=(10, 6))
        fig_observed_expected.patch.set_alpha(0.0)
        ax_obs_exp = fig_observed_expected.add_subplot(111)
        ax_obs_exp.patch.set_alpha(0.0)
        
        fig_model_comparison = plt.figure(figsize=(10, 6))
        fig_model_comparison.patch.set_alpha(0.0)
        ax_model_comp = fig_model_comparison.add_subplot(111)
        ax_model_comp.patch.set_alpha(0.0)
        
        # Convert input to numpy array and remove NAs
        data = np.array(pd.Series(data).dropna())
        
        # Check if this is count data
        if not np.all(np.equal(np.mod(data, 1), 0)) or np.any(data < 0):
            # Round and convert to non-negative integers
            original_data = data.copy()
            data = np.round(data)
            data = np.maximum(data, 0)
            
            if not np.array_equal(original_data, data):
                my_warnings.append("Data contains non-integer or negative values, which have been rounded and converted")
        
        # Need sufficient data for testing
        if len(data) < 20:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': "Not enough data points for zero-inflation test (minimum 20 required)",
                'warnings': ["Sample size too small for zero-inflation testing"],
                'figures': {
                    'observed_vs_expected': fig_to_svg(fig_observed_expected),
                    'model_comparison': fig_to_svg(fig_model_comparison)
                }
            }
            
        # Check frequency of zeros
        zero_count = np.sum(data == 0)
        zero_proportion = zero_count / len(data)
        
        # If no zeros, zero-inflation is not an issue
        if zero_count == 0:
            ax_obs_exp.text(0.5, 0.5, "No zeros in the data", 
                         horizontalalignment='center', verticalalignment='center',
                         transform=ax_obs_exp.transAxes)
            
            ax_model_comp.text(0.5, 0.5, "Zero-inflation not applicable\n(no zeros in the data)", 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax_model_comp.transAxes)
            
            return {
                'result': AssumptionResult.PASSED,
                'details': "No zeros in the data, zero-inflation is not a concern",
                'zero_count': 0,
                'zero_proportion': 0,
                'warnings': [],
                'figures': {
                    'observed_vs_expected': fig_to_svg(fig_observed_expected),
                    'model_comparison': fig_to_svg(fig_model_comparison)
                }
            }
        
        # Calculate mean and variance
        mean = np.mean(data)
        variance = np.var(data)
        dispersion = variance / mean if mean > 0 else float('inf')
        
        # Get observed frequencies
        max_value = int(max(data))
        observed_freq = np.bincount(data.astype(int), minlength=max_value+1)
        
        # Calculate expected frequencies under Poisson
        poisson_dist = stats.poisson(mu=mean)
        poisson_expected = np.array([poisson_dist.pmf(i) * len(data) for i in range(max_value+1)])
        
        # For Negative Binomial, estimate parameters (need to handle failure modes)
        try:
            if dispersion > 1:  # Overdispersed compared to Poisson
                # For negative binomial: mean = n(1-p)/p, var = n(1-p)/p²
                # Solving for n and p: p = mean/var, n = mean²/(var-mean)
                p = mean / variance
                n = mean**2 / (variance - mean)
                
                # Ensure valid parameters
                if n <= 0 or p <= 0 or p >= 1:
                    raise ValueError("Invalid parameters for negative binomial")
                
                # Calculate expected frequencies
                nbinom_dist = stats.nbinom(n=n, p=p)
                nbinom_expected = np.array([nbinom_dist.pmf(i) * len(data) for i in range(max_value+1)])
                nbinom_params = {'n': n, 'p': p}
            else:
                # If not overdispersed, just use Poisson
                nbinom_expected = poisson_expected
                nbinom_params = {'n': float('nan'), 'p': float('nan')}
                my_warnings.append("Data is not overdispersed, using Poisson instead of Negative Binomial")
        except Exception as e:
            # Fall back to Poisson in case of errors
            nbinom_expected = poisson_expected
            nbinom_params = {'n': float('nan'), 'p': float('nan')}
            my_warnings.append(f"Could not fit Negative Binomial: {str(e)}, using Poisson instead")
        
        # Select the appropriate baseline distribution based on the input parameter
        if distribution == 'negative_binomial':
            base_dist_name = "Negative Binomial"
            expected_freq = nbinom_expected
            dist_params = nbinom_params
        else:
            base_dist_name = "Poisson"
            expected_freq = poisson_expected
            dist_params = {'lambda': mean}
        
        # Calculate expected zeros
        expected_zeros = expected_freq[0]
        zero_excess = zero_count - expected_zeros
        zero_ratio = zero_count / expected_zeros if expected_zeros > 0 else float('inf')
        
        # Use statistical test to assess if the observed count of zeros is significantly higher than expected
        # For this, we'll use a one-sided test (Z-test for proportions or binomial test)
        
        # Method 1: Z-test for the difference in proportions
        # H0: p_observed <= p_expected, H1: p_observed > p_expected
        p_observed = zero_proportion
        p_expected = expected_zeros / len(data)
        pooled_p = (zero_count + expected_zeros) / (2 * len(data))
        se = np.sqrt(pooled_p * (1 - pooled_p) * (1/len(data) + 1/len(data)))
        
        if se > 0:
            z_score = (p_observed - p_expected) / se
            p_value = 1 - stats.norm.cdf(z_score)
        else:
            # If standard error is 0, use an exact binomial test
            p_value = stats.binom_test(zero_count, len(data), p_expected, alternative='greater')
            z_score = None
        
        # For regression models, try to fit standard and zero-inflated models for comparison
        model_comparison_results = {}
        vuong_test_result = None
        
        if regressors is not None:
            try:
                import statsmodels.formula.api as smf
                from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP
                
                # Convert regressors to DataFrame if not already
                if not isinstance(regressors, pd.DataFrame):
                    regressors = pd.DataFrame(regressors)
                
                # Create dataset with outcome and regressors
                dataset = pd.DataFrame({
                    'outcome': data
                })
                
                for i, col in enumerate(regressors.columns):
                    dataset[f'x{i+1}'] = regressors[col]
                
                # Construct formula
                x_vars = [f'x{i+1}' for i in range(regressors.shape[1])]
                formula = 'outcome ~ ' + ' + '.join(x_vars)
                
                # Fit models
                if distribution == 'negative_binomial':
                    # Standard Negative Binomial
                    nb_model = smf.negativebinomial(formula, dataset).fit(disp=0)
                    
                    # Zero-Inflated Negative Binomial
                    zinb_model = ZeroInflatedNegativeBinomialP(
                        dataset['outcome'], 
                        sm.add_constant(regressors),
                        exog_infl=sm.add_constant(regressors)
                    ).fit(disp=0)
                    
                    # Vuong test for non-nested models
                    try:
                        from scipy import stats
                        
                        # Calculate log-likelihoods for each observation
                        ll_nb = nb_model.llf
                        ll_zinb = zinb_model.llf
                        
                        # Store results
                        model_comparison_results = {
                            'nb_aic': nb_model.aic,
                            'nb_bic': nb_model.bic,
                            'zinb_aic': zinb_model.aic,
                            'zinb_bic': zinb_model.bic,
                            'nb_llf': ll_nb,
                            'zinb_llf': ll_zinb,
                            'preferred_model': 'ZINB' if ll_zinb > ll_nb else 'NB'
                        }
                    except Exception as e:
                        my_warnings.append(f"Could not perform Vuong test: {str(e)}")
                        model_comparison_results = {
                            'nb_aic': nb_model.aic,
                            'nb_bic': nb_model.bic,
                            'zinb_aic': zinb_model.aic,
                            'zinb_bic': zinb_model.bic,
                            'preferred_model': 'ZINB' if zinb_model.aic < nb_model.aic else 'NB'
                        }
                else:
                    # Standard Poisson
                    poisson_model = smf.poisson(formula, dataset).fit(disp=0)
                    
                    # Zero-Inflated Poisson
                    zip_model = ZeroInflatedPoisson(
                        dataset['outcome'], 
                        sm.add_constant(regressors),
                        exog_infl=sm.add_constant(regressors)
                    ).fit(disp=0)
                    
                    # Vuong test for non-nested models
                    try:
                        from scipy import stats
                        
                        # Calculate log-likelihoods for each observation
                        ll_poisson = poisson_model.llf
                        ll_zip = zip_model.llf
                        
                        # Store results
                        model_comparison_results = {
                            'poisson_aic': poisson_model.aic,
                            'poisson_bic': poisson_model.bic,
                            'zip_aic': zip_model.aic,
                            'zip_bic': zip_model.bic,
                            'poisson_llf': ll_poisson,
                            'zip_llf': ll_zip,
                            'preferred_model': 'ZIP' if ll_zip > ll_poisson else 'Poisson'
                        }
                    except Exception as e:
                        my_warnings.append(f"Could not perform model comparison test: {str(e)}")
                        model_comparison_results = {
                            'poisson_aic': poisson_model.aic,
                            'poisson_bic': poisson_model.bic,
                            'zip_aic': zip_model.aic,
                            'zip_bic': zip_model.bic,
                            'preferred_model': 'ZIP' if zip_model.aic < poisson_model.aic else 'Poisson'
                        }
            except Exception as e:
                my_warnings.append(f"Could not fit regression models: {str(e)}")
        
        # Create plot of observed vs expected frequencies
        # Plot for first 10 values (0 to 9) or up to max value, whichever is smaller
        plot_max = min(10, max_value + 1)
        x = np.arange(plot_max)
        
        # Truncate arrays for plotting
        obs_for_plot = observed_freq[:plot_max]
        exp_for_plot = expected_freq[:plot_max]
        
        width = 0.35
        ax_obs_exp.bar(x - width/2, obs_for_plot, width, label='Observed', color=PASTEL_COLORS[0])
        ax_obs_exp.bar(x + width/2, exp_for_plot, width, label=f'Expected ({base_dist_name})', color=PASTEL_COLORS[1])
        
        # Add data labels to bars
        for i, v in enumerate(obs_for_plot):
            ax_obs_exp.text(i - width/2, v + 0.1, str(int(v)), ha='center')
        
        for i, v in enumerate(exp_for_plot):
            ax_obs_exp.text(i + width/2, v + 0.1, f'{v:.1f}', ha='center')
            
        ax_obs_exp.set_xticks(x)
        ax_obs_exp.set_xlabel('Count Value')
        ax_obs_exp.set_ylabel('Frequency')
        ax_obs_exp.set_title(f'Observed vs Expected Frequencies ({base_dist_name})')
        ax_obs_exp.legend()
        
        # Highlight the zero count
        ax_obs_exp.axvline(x=-0.5, color='red', linestyle='--', alpha=0.3)
        ax_obs_exp.text(0 - width/2, obs_for_plot[0] + max(obs_for_plot) * 0.1, 
                     f'Excess: {zero_excess:.1f}', color='red', fontweight='bold')
        
        # Create AIC/BIC comparison plot if model comparison was performed
        if model_comparison_results:
            if distribution == 'negative_binomial':
                # Plot AIC and BIC for NB and ZINB
                models = ['NB', 'ZINB']
                aic_values = [model_comparison_results['nb_aic'], model_comparison_results['zinb_aic']]
                bic_values = [model_comparison_results['nb_bic'], model_comparison_results['zinb_bic']]
                
                x = np.arange(len(models))
                width = 0.35
                
                ax_model_comp.bar(x - width/2, aic_values, width, label='AIC', color=PASTEL_COLORS[0])
                ax_model_comp.bar(x + width/2, bic_values, width, label='BIC', color=PASTEL_COLORS[1])
                
                ax_model_comp.set_xticks(x)
                ax_model_comp.set_xticklabels(models)
                ax_model_comp.set_ylabel('Information Criterion')
                ax_model_comp.set_title('Model Comparison (lower is better)')
                ax_model_comp.legend()
                
                # Add preferred model indicator
                preferred_model = model_comparison_results['preferred_model']
                preferred_idx = 0 if preferred_model == 'NB' else 1
                ax_model_comp.annotate('Preferred', 
                                    xy=(preferred_idx, min(aic_values[preferred_idx], bic_values[preferred_idx])),
                                    xytext=(preferred_idx, min(aic_values[preferred_idx], bic_values[preferred_idx]) - 
                                          max(aic_values+bic_values) * 0.1),
                                    arrowprops=dict(facecolor='black', shrink=0.05),
                                    horizontalalignment='center')
            else:
                # Plot AIC and BIC for Poisson and ZIP
                models = ['Poisson', 'ZIP']
                aic_values = [model_comparison_results['poisson_aic'], model_comparison_results['zip_aic']]
                bic_values = [model_comparison_results['poisson_bic'], model_comparison_results['zip_bic']]
                
                x = np.arange(len(models))
                width = 0.35
                
                ax_model_comp.bar(x - width/2, aic_values, width, label='AIC', color=PASTEL_COLORS[0])
                ax_model_comp.bar(x + width/2, bic_values, width, label='BIC', color=PASTEL_COLORS[1])
                
                ax_model_comp.set_xticks(x)
                ax_model_comp.set_xticklabels(models)
                ax_model_comp.set_ylabel('Information Criterion')
                ax_model_comp.set_title('Model Comparison (lower is better)')
                ax_model_comp.legend()
                
                # Add preferred model indicator
                preferred_model = model_comparison_results['preferred_model']
                preferred_idx = 0 if preferred_model == 'Poisson' else 1
                ax_model_comp.annotate('Preferred', 
                                    xy=(preferred_idx, min(aic_values[preferred_idx], bic_values[preferred_idx])),
                                    xytext=(preferred_idx, min(aic_values[preferred_idx], bic_values[preferred_idx]) - 
                                          max(aic_values+bic_values) * 0.1),
                                    arrowprops=dict(facecolor='black', shrink=0.05),
                                    horizontalalignment='center')
        else:
            # No model comparison was performed
            ax_model_comp.text(0.5, 0.5, "Model comparison not performed\n(no regressors provided)", 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax_model_comp.transAxes)
        
        # Determine if there's zero-inflation based on statistical test
        if p_value < alpha:
            # Statistically significant zero-inflation
            result = AssumptionResult.FAILED
            if zero_ratio > 2:
                details = f"Strong zero-inflation detected (p={p_value:.4f}, {zero_ratio:.1f}x expected zeros)"
                my_warnings.append("Consider using a zero-inflated model")
            else:
                details = f"Moderate zero-inflation detected (p={p_value:.4f}, {zero_ratio:.1f}x expected zeros)"
                my_warnings.append("Consider using a zero-inflated or hurdle model")
        else:
            # No statistically significant zero-inflation
            if zero_ratio > 1.5:
                # Still might have some zero-inflation, just not statistically significant
                result = AssumptionResult.WARNING
                details = f"Possible zero-inflation (p={p_value:.4f}, {zero_ratio:.1f}x expected zeros, not statistically significant)"
                my_warnings.append("Consider checking model fit with and without zero-inflation")
            else:
                result = AssumptionResult.PASSED
                details = f"No significant zero-inflation detected (p={p_value:.4f}, {zero_ratio:.1f}x expected zeros)"
        
        # If model comparison was done and contradicts our statistical test
        if model_comparison_results:
            preferred_model = model_comparison_results['preferred_model']
            if (result == AssumptionResult.PASSED and ('ZIP' in preferred_model or 'ZINB' in preferred_model)):
                result = AssumptionResult.WARNING
                details += f". However, model comparison favors {preferred_model}."
                my_warnings.append(f"Statistical test indicates no zero-inflation, but model comparison prefers {preferred_model}")
            elif (result != AssumptionResult.PASSED and ('ZIP' not in preferred_model and 'ZINB' not in preferred_model)):
                details += f". However, model comparison favors {preferred_model}."
                my_warnings.append(f"Statistical test indicates zero-inflation, but model comparison prefers {preferred_model}")
        
        return {
            'result': result,
            'details': details,
            'statistics': {
                'zero_count': int(zero_count),
                'zero_proportion': float(zero_proportion),
                'expected_zeros': float(expected_zeros),
                'zero_excess': float(zero_excess),
                'zero_ratio': float(zero_ratio),
                'mean': float(mean),
                'variance': float(variance),
                'dispersion': float(dispersion),
                'test_statistic': float(z_score) if z_score is not None else None,
                'p_value': float(p_value),
                'distribution': base_dist_name,
                'dist_parameters': dist_params
            },
            'model_comparison': model_comparison_results,
            'warnings': my_warnings,
            'figures': {
                'observed_vs_expected': fig_to_svg(fig_observed_expected),
                'model_comparison': fig_to_svg(fig_model_comparison)
            }
        }

class EndogeneityTest(AssumptionTest):
    """Test for endogeneity in regression models."""
    
    def __init__(self):
        super().__init__(
            name="Endogeneity Test",
            description="Tests whether predictors are correlated with error terms, which can lead to biased estimates",
            applicable_roles=["residuals", "predictors", "instruments"],
            applicable_types=["continuous", "categorical"]
        )
    
    def run_test(self, residuals, predictors, instruments=None, **kwargs):
        """
        Test for endogeneity in regression models using various methods.
        
        Parameters
        ----------
        residuals : array-like
            Residuals from the regression model
        predictors : array-like or DataFrame
            Predictor variables to test for endogeneity
        instruments : array-like or DataFrame, optional
            Instrumental variables for Hausman test
        **kwargs : dict
            Additional parameters including:
            - method: str, 'correlation', 'durbin', or 'hausman'
            - suspected_cols: list, columns suspected of endogeneity
            
        Returns
        -------
        dict
            Dictionary containing test results
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from scipy import stats
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        import statsmodels.formula.api as smf
        import warnings as py_warnings
        
        # Suppress warnings
        py_warnings.filterwarnings('ignore')
        
        method = kwargs.get('method', 'correlation')
        alpha = kwargs.get('alpha', 0.05)
        suspected_cols = kwargs.get('suspected_cols', None)
        outcome = kwargs.get('outcome', None)
        my_warnings = []
        
        # Create figures
        fig_correlation = plt.figure(figsize=(10, 6))
        fig_correlation.patch.set_alpha(0.0)
        ax_corr = fig_correlation.add_subplot(111)
        ax_corr.patch.set_alpha(0.0)
        
        fig_residuals = plt.figure(figsize=(10, 6))
        fig_residuals.patch.set_alpha(0.0)
        ax_resid = fig_residuals.add_subplot(111)
        ax_resid.patch.set_alpha(0.0)
        
        # Convert inputs to numpy arrays or DataFrames
        residuals = np.array(residuals).flatten()
        
        if isinstance(predictors, pd.DataFrame):
            predictors_df = predictors.copy()
        else:
            # If not a DataFrame, convert to one
            predictors_df = pd.DataFrame(predictors)
            predictors_df.columns = [f'X{i+1}' for i in range(predictors_df.shape[1])]
        
        # If suspected_cols is None, test all variables
        if suspected_cols is None:
            suspected_cols = predictors_df.columns
        elif isinstance(suspected_cols, str):
            suspected_cols = [suspected_cols]
        
        # Filter to just the suspected columns
        suspected_predictors = predictors_df[suspected_cols]
        
        # Need sufficient data for testing
        if len(residuals) < 20 or len(residuals) != len(predictors_df):
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': "Insufficient data or dimension mismatch between residuals and predictors",
                'warnings': ["Sample size too small or dimension mismatch"],
                'figures': {
                    'correlation_plot': fig_to_svg(fig_correlation),
                    'residual_plot': fig_to_svg(fig_residuals)
                }
            }
        
        # Simple Correlation Test (Baseline approach)
        correlations = {}
        p_values = {}
        significant_correlations = []
        
        for col in suspected_cols:
            # Check if column exists
            if col not in predictors_df.columns:
                my_warnings.append(f"Column {col} not found in predictors")
                continue
                
            # Calculate correlation between predictor and residuals
            r, p = stats.pearsonr(predictors_df[col], residuals)
            correlations[col] = r
            p_values[col] = p
            
            if p < alpha:
                significant_correlations.append(col)
        
        # Create correlation plot for suspected variables
        if len(suspected_cols) > 0:
            try:
                # Create a bar plot of correlations
                cols = list(correlations.keys())
                corrs = [correlations[col] for col in cols]
                
                # Create x positions
                x_pos = np.arange(len(cols))
                
                # Create bar plot
                bars = ax_corr.bar(x_pos, corrs, color=PASTEL_COLORS)
                
                # Add significance markers
                for i, col in enumerate(cols):
                    if p_values[col] < alpha:
                        bars[i].set_color('red')
                        ax_corr.text(i, corrs[i] + 0.02, '*', 
                                   ha='center', va='center', 
                                   fontweight='bold', fontsize=12)
                
                # Add value labels on top of bars
                for i, v in enumerate(corrs):
                    ax_corr.text(i, v + 0.02 * (1 if v >= 0 else -1), 
                               f'{v:.2f}', ha='center', va='center')
                
                # Add horizontal line at y=0
                ax_corr.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # Set labels and title
                ax_corr.set_ylabel('Correlation with Residuals')
                ax_corr.set_title('Correlation between Predictors and Residuals')
                ax_corr.set_xticks(x_pos)
                ax_corr.set_xticklabels(cols, rotation=45, ha='right')
                
                # Add p-value text
                for i, col in enumerate(cols):
                    ax_corr.text(i, -0.1, f'p={p_values[col]:.3f}', 
                              ha='center', va='center', rotation=90, 
                              fontsize=8, alpha=0.7)
            except Exception as e:
                my_warnings.append(f"Could not create correlation plot: {str(e)}")
        else:
            ax_corr.text(0.5, 0.5, "No suspected columns provided", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax_corr.transAxes)
        
        # Method-specific tests
        if method == 'durbin':
            # Durbin-Wu-Hausman test
            # Requires outcome variable and instruments
            if outcome is None or instruments is None:
                my_warnings.append("Durbin-Wu-Hausman test requires outcome variable and instruments")
                durbin_results = None
            else:
                try:
                    if isinstance(instruments, pd.DataFrame):
                        instruments_df = instruments.copy()
                    else:
                        # If not a DataFrame, convert to one
                        instruments_df = pd.DataFrame(instruments)
                        instruments_df.columns = [f'Z{i+1}' for i in range(instruments_df.shape[1])]
                    
                    # Step 1: First stage regression
                    first_stage_results = {}
                    for col in suspected_cols:
                        # Regress endogenous variable on instruments
                        X = sm.add_constant(instruments_df)
                        model = sm.OLS(predictors_df[col], X).fit()
                        first_stage_results[col] = model
                    
                    # Step 2: Get predicted values and residuals
                    for col in suspected_cols:
                        predicted = first_stage_results[col].predict()
                        predictors_df[f'{col}_hat'] = predicted
                        predictors_df[f'{col}_resid'] = predictors_df[col] - predicted
                    
                    # Step 3: Second stage regression (augmented with first-stage residuals)
                    X = sm.add_constant(predictors_df)
                    model = sm.OLS(outcome, X).fit()
                    
                    # Step 4: Test if residuals are significant
                    durbin_results = {
                        'model': model,
                        'test_stats': {},
                        'p_values': {},
                        'significant': []
                    }
                    
                    for col in suspected_cols:
                        # Get t-test result for residual term
                        t_stat = model.tvalues[f'{col}_resid']
                        p_val = model.pvalues[f'{col}_resid']
                        
                        durbin_results['test_stats'][col] = t_stat
                        durbin_results['p_values'][col] = p_val
                        
                        if p_val < alpha:
                            durbin_results['significant'].append(col)
                except Exception as e:
                    my_warnings.append(f"Could not perform Durbin-Wu-Hausman test: {str(e)}")
                    durbin_results = None
        else:
            durbin_results = None
        
        if method == 'hausman':
            # Hausman test
            # Requires both OLS and IV estimates
            if outcome is None or instruments is None:
                my_warnings.append("Hausman test requires outcome variable and instruments")
                hausman_results = None
            else:
                try:
                    # Step 1: Fit OLS model
                    X_ols = sm.add_constant(predictors_df)
                    ols_model = sm.OLS(outcome, X_ols).fit()
                    
                    # Step 2: Fit IV model
                    # If not a DataFrame, convert to one
                    if isinstance(instruments, pd.DataFrame):
                        instruments_df = instruments.copy()
                    else:
                        instruments_df = pd.DataFrame(instruments)
                        instruments_df.columns = [f'Z{i+1}' for i in range(instruments_df.shape[1])]
                    
                    # Prepare data for IV regression
                    endog = outcome
                    exog = sm.add_constant(predictors_df)
                    instr = sm.add_constant(instruments_df)
                    
                    # Fit IV model
                    iv_model = sm.IV2SLS(endog, exog, instr).fit()
                    
                    # Step 3: Calculate Hausman statistic
                    b_ols = ols_model.params
                    b_iv = iv_model.params
                    
                    # Get common parameter indices
                    common_params = list(set(b_ols.index) & set(b_iv.index))
                    
                    # Calculate difference in coefficients
                    diff = b_iv[common_params] - b_ols[common_params]
                    
                    # Covariance of the difference
                    cov_diff = iv_model.cov_params().loc[common_params, common_params] - ols_model.cov_params().loc[common_params, common_params]
                    
                    # Calculate Hausman statistic
                    # Note: In some cases, cov_diff might not be positive definite due to sampling variation
                    # In such cases, we'll use a generalized inverse
                    try:
                        import numpy.linalg as la
                        # Try standard inverse
                        cov_diff_inv = la.inv(cov_diff)
                    except:
                        # If that fails, use pseudoinverse
                        cov_diff_inv = la.pinv(cov_diff)
                    
                    h_stat = diff.dot(cov_diff_inv).dot(diff)
                    df = len(common_params)
                    p_value = 1 - stats.chi2.cdf(h_stat, df)
                    
                    # Store results
                    hausman_results = {
                        'statistic': h_stat,
                        'df': df,
                        'p_value': p_value,
                        'ols_coef': b_ols[common_params],
                        'iv_coef': b_iv[common_params],
                        'difference': diff
                    }
                except Exception as e:
                    my_warnings.append(f"Could not perform Hausman test: {str(e)}")
                    hausman_results = None
        else:
            hausman_results = None
        
        # Create residual plots for suspected variables
        if len(suspected_cols) > 0:
            try:
                # If just one column, plot single scatter plot
                if len(suspected_cols) == 1:
                    col = suspected_cols[0]
                    ax_resid.scatter(predictors_df[col], residuals, alpha=0.6, color=PASTEL_COLORS[0])
                    ax_resid.axhline(y=0, color='r', linestyle='-')
                    ax_resid.set_title(f'Residuals vs. {col}')
                    ax_resid.set_xlabel(col)
                    ax_resid.set_ylabel('Residuals')
                    
                    # Add lowess line to help identify patterns
                    try:
                        frac = min(0.6, max(0.2, 30 / len(residuals)))
                        lowess = sm.nonparametric.lowess(residuals, predictors_df[col], frac=frac)
                        ax_resid.plot(lowess[:, 0], lowess[:, 1], 'r-', lw=1)
                    except Exception as e:
                        my_warnings.append(f"Could not add LOWESS line: {str(e)}")
                
                # If 2+ columns, create grid of small plots
                else:
                    # Maximum number of columns to plot (avoid too many plots)
                    max_cols = min(6, len(suspected_cols))
                    
                    # Clear the current axis and create a grid
                    ax_resid.clear()
                    num_rows = (max_cols + 1) // 2
                    fig_residuals.clear()
                    
                    for i, col in enumerate(suspected_cols[:max_cols]):
                        ax = fig_residuals.add_subplot(num_rows, 2, i+1)
                        ax.scatter(predictors_df[col], residuals, alpha=0.6, 
                                  color=PASTEL_COLORS[i % len(PASTEL_COLORS)])
                        ax.axhline(y=0, color='r', linestyle='-')
                        ax.set_title(f'{col} (r={correlations[col]:.2f}, p={p_values[col]:.3f})')
                        ax.set_xlabel(col)
                        ax.set_ylabel('Residuals')
                        
                        # Add lowess line
                        try:
                            frac = min(0.6, max(0.2, 30 / len(residuals)))
                            lowess = sm.nonparametric.lowess(residuals, predictors_df[col], frac=frac)
                            ax.plot(lowess[:, 0], lowess[:, 1], 'r-', lw=1)
                        except:
                            pass
            
            except Exception as e:
                my_warnings.append(f"Could not create residual plots: {str(e)}")
        
        # Determine the overall result
        if method == 'correlation':
            # Based on correlation test
            if len(significant_correlations) > 0:
                result = AssumptionResult.FAILED
                details = f"Endogeneity detected: {len(significant_correlations)} variables show significant correlation with residuals"
                my_warnings.append(f"Potentially endogenous variables: {', '.join(significant_correlations)}")
            else:
                result = AssumptionResult.PASSED
                details = "No evidence of endogeneity based on correlation test"
        
        elif method == 'durbin' and durbin_results is not None:
            # Based on Durbin-Wu-Hausman test
            if len(durbin_results['significant']) > 0:
                result = AssumptionResult.FAILED
                details = f"Endogeneity detected: {len(durbin_results['significant'])} variables are endogenous (Durbin-Wu-Hausman test)"
                my_warnings.append(f"Endogenous variables: {', '.join(durbin_results['significant'])}")
            else:
                result = AssumptionResult.PASSED
                details = "No evidence of endogeneity based on Durbin-Wu-Hausman test"
                
        elif method == 'hausman' and hausman_results is not None:
            # Based on Hausman test
            if hausman_results['p_value'] < alpha:
                result = AssumptionResult.FAILED
                details = f"Endogeneity detected: Hausman test significant (p={hausman_results['p_value']:.4f})"
                
                # Add specific variables with large differences
                largest_diffs = hausman_results['difference'].abs().nlargest(3)
                vars_with_large_diffs = largest_diffs.index.tolist()
                
                if vars_with_large_diffs:
                    my_warnings.append(f"Variables with largest differences between OLS and IV: {', '.join(vars_with_large_diffs)}")
            else:
                result = AssumptionResult.PASSED
                details = f"No evidence of endogeneity based on Hausman test (p={hausman_results['p_value']:.4f})"
        
        else:
            # Fallback to correlation test if specified method fails
            if len(significant_correlations) > 0:
                result = AssumptionResult.FAILED
                details = f"Endogeneity detected: {len(significant_correlations)} variables correlated with residuals"
                my_warnings.append(f"Potentially endogenous variables: {', '.join(significant_correlations)}")
                
                if method != 'correlation':
                    my_warnings.append(f"Note: Falling back to correlation test as {method} test could not be performed")
            else:
                result = AssumptionResult.PASSED
                details = "No evidence of endogeneity based on correlation test"
                
                if method != 'correlation':
                    my_warnings.append(f"Note: Falling back to correlation test as {method} test could not be performed")
        
        # Add recommendations if endogeneity is detected
        if result == AssumptionResult.FAILED:
            my_warnings.append("Consider using instrumental variables (2SLS), control function approach, or other methods to address endogeneity")
            
            # If no instruments were provided but needed
            if (method in ['durbin', 'hausman']) and instruments is None:
                my_warnings.append("Instrumental variables are required for formal Hausman or Durbin-Wu-Hausman tests")
        
        return {
            'result': result,
            'details': details,
            'correlations': correlations,
            'p_values': p_values,
            'significant_correlations': significant_correlations,
            'durbin_results': durbin_results,
            'hausman_results': hausman_results,
            'method': method,
            'warnings': my_warnings,
            'figures': {
                'correlation_plot': fig_to_svg(fig_correlation),
                'residual_plot': fig_to_svg(fig_residuals)
            }
        }
    

class StationarityTest(AssumptionTest):
    """Test for stationarity in time series data."""
    
    def __init__(self):
        super().__init__(
            name="Stationarity Test",
            description="Tests if time series data's statistical properties remain constant over time",
            applicable_roles=["time_series"],
            applicable_types=["numeric", "continuous"]
        )
        
    def run_test(self, data, **kwargs):
        """
        Run stationarity test on time series data.
        
        Args:
            data: Pandas Series or NumPy array of time series values
            method (str, optional): 'adf' (Augmented Dickey-Fuller) or 'kpss'. Defaults to 'adf'.
            
        Returns:
            dict: Test results following the StationarityTest format
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from statsmodels.tsa.stattools import adfuller, kpss
        
        method = kwargs.get('method', 'adf')
        alpha = kwargs.get('alpha', 0.05)
        warnings = []
        
        # Create figures for visualization
        fig_ts = plt.figure(figsize=(10, 6))
        fig_ts.patch.set_alpha(0.0)
        ax_ts = fig_ts.add_subplot(111)
        ax_ts.patch.set_alpha(0.0)
        
        fig_acf = plt.figure(figsize=(10, 6))
        fig_acf.patch.set_alpha(0.0)
        ax_acf = fig_acf.add_subplot(111)
        ax_acf.patch.set_alpha(0.0)
        
        # Convert to pandas Series if not already
        data = pd.Series(data).dropna()
        
        if len(data) < 10:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'statistic': None,
                'p_value': None,
                'details': "Too few observations for stationarity testing (minimum 10 required)",
                'test_used': method,
                'warnings': ["Sample size too small for stationarity testing"],
                'figures': {
                    'time_series': fig_to_svg(fig_ts),
                    'autocorrelation': fig_to_svg(fig_acf)
                }
            }
        
        # Plot the time series
        ax_ts.plot(data.values, marker='o', markersize=3, linestyle='-', color=PASTEL_COLORS[0])
        ax_ts.set_title('Time Series Plot')
        ax_ts.set_xlabel('Time')
        ax_ts.set_ylabel('Value')
        
        # Plot the autocorrelation function
        try:
            from statsmodels.graphics.tsaplots import plot_acf
            plot_acf(data, ax=ax_acf, alpha=0.05, color=PASTEL_COLORS[1])
            ax_acf.set_title('Autocorrelation Function')
        except Exception as e:
            warnings.append(f"Could not plot autocorrelation function: {str(e)}")
            ax_acf.text(0.5, 0.5, "Could not generate autocorrelation plot", 
                       horizontalalignment='center', verticalalignment='center')
        
        # Run the appropriate test
        if method == 'adf':
            # Augmented Dickey-Fuller test
            # Null hypothesis: The series has a unit root (non-stationary)
            # Alternative hypothesis: The series is stationary
            try:
                result = adfuller(data, autolag='AIC')
                statistic = result[0]
                p_value = result[1]
                critical_values = result[4]
                
                # In ADF test, we reject the null hypothesis if p-value is small
                if p_value < alpha:
                    test_result = AssumptionResult.PASSED
                    details = f"Time series appears to be stationary (ADF statistic: {statistic:.4f}, p-value: {p_value:.4f})"
                else:
                    test_result = AssumptionResult.FAILED
                    details = f"Time series appears to be non-stationary (ADF statistic: {statistic:.4f}, p-value: {p_value:.4f})"
                    warnings.append("Non-stationary time series may lead to spurious correlations and invalid inference")
            except Exception as e:
                warnings.append(f"ADF test failed: {str(e)}")
                test_result = AssumptionResult.NOT_APPLICABLE
                statistic = None
                p_value = None
                details = f"Could not perform ADF test: {str(e)}"
                critical_values = {}
        
        elif method == 'kpss':
            # KPSS test
            # Null hypothesis: The series is stationary
            # Alternative hypothesis: The series has a unit root (non-stationary)
            try:
                result = kpss(data, regression='c', nlags="auto")
                statistic = result[0]
                p_value = result[1]
                critical_values = result[3]
                
                # In KPSS test, we reject the null hypothesis if p-value is small
                # So we pass if p-value is large
                if p_value > alpha:
                    test_result = AssumptionResult.PASSED
                    details = f"Time series appears to be stationary (KPSS statistic: {statistic:.4f}, p-value: {p_value:.4f})"
                else:
                    test_result = AssumptionResult.FAILED
                    details = f"Time series appears to be non-stationary (KPSS statistic: {statistic:.4f}, p-value: {p_value:.4f})"
                    warnings.append("Non-stationary time series may lead to spurious correlations and invalid inference")
            except Exception as e:
                warnings.append(f"KPSS test failed: {str(e)}")
                test_result = AssumptionResult.NOT_APPLICABLE
                statistic = None
                p_value = None
                details = f"Could not perform KPSS test: {str(e)}"
                critical_values = {}
        
        else:
            raise ValueError(f"Unknown stationarity test method: {method}")
        
        # Check for trends visually
        try:
            # Fit a linear trend
            x = np.arange(len(data))
            X = sm.add_constant(x)
            model = sm.OLS(data, X).fit()
            
            # Plot the trend line
            trend = model.params[0] + model.params[1] * x
            ax_ts.plot(trend, 'r--', label='Linear Trend')
            ax_ts.legend()
            
            # Check if trend is significant
            trend_p_value = model.pvalues[1]
            if trend_p_value < alpha and test_result == AssumptionResult.PASSED:
                warnings.append(f"While the stationarity test passed, there appears to be a significant linear trend (p={trend_p_value:.4f})")
                warnings.append("Consider detrending the data before analysis")
        except Exception as e:
            warnings.append(f"Could not check for trend: {str(e)}")
        
        return {
            'result': test_result,
            'statistic': statistic,
            'p_value': p_value,
            'details': details,
            'test_used': method,
            'critical_values': critical_values,
            'warnings': warnings,
            'figures': {
                'time_series': fig_to_svg(fig_ts),
                'autocorrelation': fig_to_svg(fig_acf)
            }
        }

class MissingDataRandomnessTest(AssumptionTest):
    """Test for randomness in missing data patterns."""
    
    def __init__(self):
        super().__init__(
            name="Missing Data Randomness Test",
            description="Tests if missing data follows MCAR/MAR/MNAR patterns",
            applicable_roles=["count", "numeric", "continuous", "categorical", "ordinal", "boolean"],
            applicable_types=["count", "numeric", "continuous", "categorical", "ordinal", "boolean"]
        )
        
    def run_test(self, df, **kwargs):
        """
        Test randomness of missing data patterns.
        
        Args:
            df: DataFrame with potentially missing data
            test_var (str, optional): Variable to test for missingness mechanism
            
        Returns:
            dict: Test results following the MissingDataRandomnessTest format
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from scipy import stats
        
        test_var = kwargs.get('test_var', None)
        alpha = kwargs.get('alpha', 0.05)
        warnings = []
        
        # Create figures for visualization
        fig_heatmap = plt.figure(figsize=(10, 8))
        fig_heatmap.patch.set_alpha(0.0)
        ax_heatmap = fig_heatmap.add_subplot(111)
        ax_heatmap.patch.set_alpha(0.0)
        
        fig_bars = plt.figure(figsize=(10, 6))
        fig_bars.patch.set_alpha(0.0)
        ax_bars = fig_bars.add_subplot(111)
        ax_bars.patch.set_alpha(0.0)
        
        # Check if there's any missing data
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        total_cells = df.shape[0] * df.shape[1]
        missing_percent = (total_missing / total_cells) * 100
        
        if total_missing == 0:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': "No missing data detected in the dataset",
                'missing_percent': 0.0,
                'missing_by_column': missing_counts.to_dict(),
                'warnings': [],
                'figures': {
                    'missing_heatmap': fig_to_svg(fig_heatmap),
                    'missing_by_var': fig_to_svg(fig_bars)
                }
            }
        
        # Create a missing data indicator matrix
        missing_matrix = df.isnull().astype(int)
        
        # Plot missing data heatmap
        try:
            import seaborn as sns
            sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax_heatmap)
            ax_heatmap.set_title('Missing Data Heatmap')
            ax_heatmap.set_xlabel('Variables')
            ax_heatmap.set_ylabel('Observations')
        except Exception as e:
            warnings.append(f"Could not create missing data heatmap: {str(e)}")
            ax_heatmap.text(0.5, 0.5, "Could not generate missing data heatmap", 
                           horizontalalignment='center', verticalalignment='center')
        
        # Plot missing data counts by variable
        try:
            missing_counts.plot(kind='bar', ax=ax_bars, color=PASTEL_COLORS)
            ax_bars.set_title('Missing Values by Variable')
            ax_bars.set_xlabel('Variables')
            ax_bars.set_ylabel('Count of Missing Values')
        except Exception as e:
            warnings.append(f"Could not create missing data bar chart: {str(e)}")
            ax_bars.text(0.5, 0.5, "Could not generate missing data bar chart", 
                        horizontalalignment='center', verticalalignment='center')
        
        # If no specific variable is specified for testing, use the one with moderate missingness
        if test_var is None:
            # Choose a variable with a moderate amount of missing data
            moderate_missing = missing_counts[(missing_counts > 0) & (missing_counts < df.shape[0] * 0.8)]
            if not moderate_missing.empty:
                test_var = moderate_missing.idxmax()
                warnings.append(f"No test variable specified, using '{test_var}' which has {missing_counts[test_var]} missing values")
        
        # Initialize test results
        little_mcar_p = None
        mcar_likely = None
        mar_tests = {}
        detailed_pattern = {}
        
        # Test for Missing Completely At Random (MCAR) using Little's MCAR test
        # Since Little's test is complex, we'll approximate with a simplified approach
        try:
            # Create indicators for missing values
            indicators = df.isnull().astype(int)
            
            # Select indicators with sufficient missingness
            valid_indicators = [col for col in indicators.columns 
                               if indicators[col].sum() > 0 and indicators[col].sum() < len(indicators)]
            
            if len(valid_indicators) >= 2:
                # Calculate correlations between missing indicators
                corr_matrix = indicators[valid_indicators].corr()
                
                # Extract correlations from upper triangle of matrix
                upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                significant_corrs = []
                
                # Check for significant correlations
                for i, row in enumerate(valid_indicators):
                    for j, col in enumerate(valid_indicators):
                        if j <= i:  # Skip diagonal and lower triangle
                            continue
                        
                        # Calculate chi-square test for association between missingness patterns
                        contingency = pd.crosstab(indicators[row], indicators[col])
                        if contingency.shape == (2, 2):  # Only if we have a proper 2x2 table
                            _, p, _, _ = stats.chi2_contingency(contingency)
                            if p < alpha:
                                significant_corrs.append((row, col, p))
                
                # If no significant correlations, likely MCAR
                if not significant_corrs:
                    mcar_likely = True
                    little_mcar_p = 0.8  # Approximation, not a real p-value
                else:
                    mcar_likely = False
                    little_mcar_p = 0.01  # Approximation, not a real p-value
                    
                    # Store details of significant correlations
                    for row, col, p in significant_corrs:
                        detailed_pattern[f"{row}-{col}"] = {
                            'correlation': corr_matrix.loc[row, col],
                            'p_value': p
                        }
            else:
                warnings.append("Could not test MCAR pattern: Need at least two columns with partial missingness")
                
        except Exception as e:
            warnings.append(f"Error in MCAR testing: {str(e)}")
        
        # Test for Missing At Random (MAR) by checking if missingness of test_var
        # is related to values of other variables
        if test_var is not None and test_var in df.columns:
            missing_indicator = df[test_var].isnull().astype(int)
            
            # Test relationship between missingness in test_var and other complete variables
            for col in df.columns:
                if col == test_var or df[col].isnull().any():
                    continue
                
                try:
                    # For categorical predictors
                    if pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
                        contingency = pd.crosstab(missing_indicator, df[col])
                        if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                            _, p, _, _ = stats.chi2_contingency(contingency)
                            mar_tests[col] = {'method': 'chi2', 'p_value': p}
                    
                    # For numeric predictors
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        try:
                            # Use t-test to compare values where test_var is missing vs. not missing
                            group1 = df.loc[df[test_var].isnull(), col].dropna()
                            group2 = df.loc[~df[test_var].isnull(), col].dropna()
                            
                            if len(group1) > 0 and len(group2) > 0:
                                _, p = stats.ttest_ind(group1, group2, equal_var=False)
                                mar_tests[col] = {'method': 't-test', 'p_value': p}
                        except Exception as e:
                            warnings.append(f"Error in t-test for {col}: {str(e)}")
                except Exception as e:
                    warnings.append(f"Error testing MAR for {col}: {str(e)}")
        
            # Check if any significant relationships were found
            significant_mars = {k: v for k, v in mar_tests.items() if v['p_value'] < alpha}
        else:
            significant_mars = {}
            if test_var is not None:
                warnings.append(f"Specified test variable '{test_var}' not found in dataset")
        
        # Determine missing data mechanism
        if mcar_likely is True:
            result = AssumptionResult.PASSED
            details = "Data appears to be Missing Completely At Random (MCAR)"
        elif significant_mars:
            result = AssumptionResult.WARNING
            mar_vars = ", ".join(significant_mars.keys())
            details = f"Data appears to be Missing At Random (MAR). Missingness of '{test_var}' is related to values of: {mar_vars}"
            warnings.append("MAR data requires appropriate missing data handling like multiple imputation")
        else:
            result = AssumptionResult.WARNING
            details = "Missing data pattern is inconclusive - could be MAR or MNAR"
            warnings.append("Consider sensitivity analysis to assess impact of missing data mechanism")
        
        return {
            'result': result,
            'details': details,
            'missing_percent': missing_percent,
            'missing_by_column': missing_counts.to_dict(),
            'mcar_test': {
                'p_value': little_mcar_p,
                'is_mcar': mcar_likely
            },
            'mar_tests': mar_tests,
            'pattern_details': detailed_pattern,
            'warnings': warnings,
            'figures': {
                'missing_heatmap': fig_to_svg(fig_heatmap),
                'missing_by_var': fig_to_svg(fig_bars)
            }
        }

class MonotonicityTest(AssumptionTest):
    """Test for monotonic relationships between variables."""
    
    def __init__(self):
        super().__init__(
            name="Monotonicity Test",
            description="Tests if relationships follow monotonic patterns (important for non-parametric methods)",
            applicable_roles=["outcome", "covariate"],
            applicable_types=["numeric", "ordinal"]
        )
        
    def run_test(self, x, y, **kwargs):
        """
        Test for monotonicity in the relationship between two variables.
        
        Args:
            x: Predictor variable (Series or array)
            y: Outcome variable (Series or array)
            
        Returns:
            dict: Test results following the MonotonicityTest format
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats
        
        alpha = kwargs.get('alpha', 0.05)
        warnings = []
        
        # Create figure for visualization
        fig_scatter = plt.figure(figsize=(10, 6))
        fig_scatter.patch.set_alpha(0.0)
        ax_scatter = fig_scatter.add_subplot(111)
        ax_scatter.patch.set_alpha(0.0)
        
        # Convert inputs to pandas Series if they aren't already
        if not isinstance(x, pd.Series):
            x = pd.Series(x)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
            
        # Drop NaN values
        data = pd.DataFrame({'x': x, 'y': y}).dropna()
        
        if len(data) < 5:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': "Too few samples to test monotonicity (minimum 5 required)",
                'spearman_rho': None,
                'spearman_p': None,
                'kendall_tau': None,
                'kendall_p': None,
                'warnings': ["Sample size too small for monotonicity testing"],
                'figures': {
                    'scatter_plot': fig_to_svg(fig_scatter)
                }
            }
        
        x_clean = data['x']
        y_clean = data['y']
        
        # Check for constant values
        if x_clean.nunique() <= 1 or y_clean.nunique() <= 1:
            ax_scatter.scatter(x_clean, y_clean, alpha=0.6)
            ax_scatter.set_title('Scatter Plot - Cannot Test Monotonicity')
            ax_scatter.set_xlabel('X')
            ax_scatter.set_ylabel('Y')
            
            if x_clean.nunique() <= 1 and y_clean.nunique() <= 1:
                details = "Both X and Y variables are constant, cannot test monotonicity"
            elif x_clean.nunique() <= 1:
                details = "X variable is constant, cannot test monotonicity"
            else:
                details = "Y variable is constant, cannot test monotonicity"
                
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': details,
                'spearman_rho': None,
                'spearman_p': None,
                'kendall_tau': None,
                'kendall_p': None,
                'warnings': ["Cannot assess monotonicity with constant variable(s)"],
                'figures': {
                    'scatter_plot': fig_to_svg(fig_scatter)
                }
            }
        
        # Calculate Spearman's rank correlation
        try:
            spearman_rho, spearman_p = stats.spearmanr(x_clean, y_clean)
        except Exception as e:
            warnings.append(f"Could not calculate Spearman correlation: {str(e)}")
            spearman_rho, spearman_p = None, None
            
        # Calculate Kendall's tau
        try:
            kendall_tau, kendall_p = stats.kendalltau(x_clean, y_clean)
        except Exception as e:
            warnings.append(f"Could not calculate Kendall's tau: {str(e)}")
            kendall_tau, kendall_p = None, None
        
        # Plot the scatter plot with LOWESS curve to visualize potential monotonicity
        ax_scatter.scatter(x_clean, y_clean, alpha=0.6, color=PASTEL_COLORS[0])
        
        # Add LOWESS smoothing curve
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            # Use a smaller fraction for larger datasets
            frac = min(0.8, max(0.3, 30 / len(x_clean)))
            lowess_result = lowess(y_clean, x_clean, frac=frac)
            ax_scatter.plot(lowess_result[:, 0], lowess_result[:, 1], 'r-', linewidth=2, label='LOWESS')
            
            # Determine if the smoothed curve is monotonic
            # We'll check if the derivative of the smoothed curve changes sign
            smoothed_x = lowess_result[:, 0]
            smoothed_y = lowess_result[:, 1]
            
            # Calculate differences between consecutive points
            diff_y = np.diff(smoothed_y)
            sign_changes = np.sum(np.diff(np.signbit(diff_y)))
            
            is_strictly_monotonic = (sign_changes == 0)
            # Allow for some noise in the smoothed curve
            is_relatively_monotonic = (sign_changes <= len(diff_y) * 0.1)
            
        except Exception as e:
            warnings.append(f"Could not calculate LOWESS curve: {str(e)}")
            is_strictly_monotonic = None
            is_relatively_monotonic = None
            sign_changes = None
        
        # Add regression line for reference
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
            line_x = np.array([min(x_clean), max(x_clean)])
            line_y = intercept + slope * line_x
            ax_scatter.plot(line_x, line_y, 'b--', label='Linear fit')
            ax_scatter.legend()
        except Exception as e:
            warnings.append(f"Could not calculate regression line: {str(e)}")
        
        # Add correlation coefficients to the plot
        if spearman_rho is not None and kendall_tau is not None:
            ax_scatter.set_title(f'Scatter Plot\nSpearman ρ: {spearman_rho:.3f} (p={spearman_p:.3f}), Kendall τ: {kendall_tau:.3f} (p={kendall_p:.3f})')
        
        ax_scatter.set_xlabel('X')
        ax_scatter.set_ylabel('Y')
        
        # Determine monotonicity
        # Use both correlation coefficients and visual inspection of the LOWESS curve
        if spearman_p is None or kendall_p is None:
            if is_strictly_monotonic is not None:
                # Fall back to visual inspection only
                if is_strictly_monotonic:
                    result = AssumptionResult.PASSED
                    details = "Relationship appears to be strictly monotonic based on visual inspection"
                elif is_relatively_monotonic:
                    result = AssumptionResult.WARNING
                    details = "Relationship appears to be approximately monotonic with some deviations"
                    warnings.append("Mild violations of monotonicity detected, interpret results with caution")
                else:
                    result = AssumptionResult.FAILED
                    details = "Relationship is not monotonic based on visual inspection"
                    warnings.append("Non-monotonic relationship may affect the validity of rank-based methods")
            else:
                result = AssumptionResult.NOT_APPLICABLE
                details = "Could not assess monotonicity due to computational issues"
        elif (spearman_p < alpha or kendall_p < alpha) and (abs(spearman_rho) > 0.3 or abs(kendall_tau) > 0.3):
            # Strong and significant correlation
            if is_strictly_monotonic or is_strictly_monotonic is None:
                # If LOWESS confirms or we couldn't calculate LOWESS
                result = AssumptionResult.PASSED
                if spearman_rho > 0:
                    details = f"Variables exhibit significant monotonic increasing relationship (Spearman ρ={spearman_rho:.3f}, p={spearman_p:.3f})"
                else:
                    details = f"Variables exhibit significant monotonic decreasing relationship (Spearman ρ={spearman_rho:.3f}, p={spearman_p:.3f})"
            else:
                # LOWESS suggests non-monotonicity despite significant correlation
                result = AssumptionResult.WARNING
                details = f"Relationship has significant rank correlation (Spearman ρ={spearman_rho:.3f}) but may not be strictly monotonic"
                warnings.append("Visual inspection suggests potential non-monotonic patterns despite significant correlation")
        else:
            # Weak or non-significant correlation
            result = AssumptionResult.FAILED
            details = f"No significant monotonic relationship detected (Spearman ρ={spearman_rho:.3f}, p={spearman_p:.3f})"
            warnings.append("Non-monotonic relationship may affect the validity of rank-based methods")
        
        return {
            'result': result,
            'details': details,
            'spearman_rho': spearman_rho,
            'spearman_p': spearman_p,
            'kendall_tau': kendall_tau,
            'kendall_p': kendall_p,
            'is_strictly_monotonic': is_strictly_monotonic,
            'is_relatively_monotonic': is_relatively_monotonic,
            'sign_changes': sign_changes,
            'warnings': warnings,
            'figures': {
                'scatter_plot': fig_to_svg(fig_scatter)
            }
        }

class BalanceTest(AssumptionTest):
    """Test for balance between treatment/control groups on covariates."""
    
    def __init__(self):
        super().__init__(
            name="Balance Test",
            description="Tests if treatment/control groups are balanced on covariates",
            applicable_roles=["treatment", "covariate"],
            applicable_types=["categorical", "numeric", "continuous"]
        )
        
    def run_test(self, df, treatment_var, covariates, **kwargs):
        """
        Test balance of covariates across treatment groups.
        
        Args:
            df: DataFrame with the data
            treatment_var: Name of the treatment variable column
            covariates: List of covariate column names to test for balance
            
        Returns:
            dict: Test results following the BalanceTest format
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from scipy import stats
        
        alpha = kwargs.get('alpha', 0.05)
        warnings = []
        
        # Create figures for visualization
        fig_std_diff = plt.figure(figsize=(10, 8))
        fig_std_diff.patch.set_alpha(0.0)
        ax_std_diff = fig_std_diff.add_subplot(111)
        ax_std_diff.patch.set_alpha(0.0)
        
        fig_dist = plt.figure(figsize=(12, 10))
        fig_dist.patch.set_alpha(0.0)
        
        # Check if treatment variable exists in dataframe
        if treatment_var not in df.columns:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': f"Treatment variable '{treatment_var}' not found in data",
                'balance_statistics': {},
                'warnings': [f"Could not find treatment variable: {treatment_var}"],
                'figures': {
                    'std_diff_plot': fig_to_svg(fig_std_diff),
                    'distribution_plot': fig_to_svg(fig_dist)
                }
            }
        
        # Check if we have at least two groups in the treatment variable
        treatment_values = df[treatment_var].dropna().unique()
        if len(treatment_values) < 2:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': f"Treatment variable has fewer than 2 groups: {list(treatment_values)}",
                'balance_statistics': {},
                'warnings': ["Need at least two treatment groups for balance testing"],
                'figures': {
                    'std_diff_plot': fig_to_svg(fig_std_diff),
                    'distribution_plot': fig_to_svg(fig_dist)
                }
            }
        
        # Check that treatment is binary or categorical with few levels
        if len(treatment_values) > 10:
            warnings.append(f"Treatment variable has many levels ({len(treatment_values)}), which may make balance interpretation difficult")
        
        # If binary treatment, identify the treatment and control labels
        if len(treatment_values) == 2:
            # Try to intelligently determine which is treatment (1) and which is control (0)
            if 1 in treatment_values and 0 in treatment_values:
                treatment_label = 1
                control_label = 0
            elif "treatment" in [str(v).lower() for v in treatment_values]:
                treatment_label = [v for v in treatment_values if str(v).lower() == "treatment"][0]
                control_label = [v for v in treatment_values if v != treatment_label][0]
            elif "treated" in [str(v).lower() for v in treatment_values]:
                treatment_label = [v for v in treatment_values if str(v).lower() == "treated"][0]
                control_label = [v for v in treatment_values if v != treatment_label][0]
            else:
                # Just take the larger value as treatment
                treatment_label = max(treatment_values)
                control_label = min(treatment_values)
                
            binary_treatment = True
        else:
            binary_treatment = False
            treatment_label = None
            control_label = None
        
        # Check that covariates exist in the dataframe
        valid_covariates = [cov for cov in covariates if cov in df.columns]
        if not valid_covariates:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': "None of the specified covariates found in data",
                'balance_statistics': {},
                'warnings': ["Could not find any of the specified covariates"],
                'figures': {
                    'std_diff_plot': fig_to_svg(fig_std_diff),
                    'distribution_plot': fig_to_svg(fig_dist)
                }
            }
        
        if len(valid_covariates) < len(covariates):
            warnings.append(f"Some covariates not found in data: {set(covariates) - set(valid_covariates)}")
        
        # Initialize results containers
        balance_stats = {}
        std_mean_diffs = {}
        significant_imbalances = []
        
        # For distribution plots, create a grid based on number of covariates
        n_covariates = len(valid_covariates)
        nrows = int(np.ceil(np.sqrt(n_covariates)))
        ncols = int(np.ceil(n_covariates / nrows))
        
        # Create axes for distribution plots
        axes_dist = fig_dist.subplots(nrows, ncols)
        if n_covariates == 1:
            axes_dist = np.array([axes_dist])
        axes_dist = axes_dist.flatten()
        
        # Set remaining plots as invisible if we have too many subplots
        for i in range(n_covariates, len(axes_dist)):
            axes_dist[i].set_visible(False)
        
        # Test balance for each covariate
        for i, covariate in enumerate(valid_covariates):
            # Check covariate type
            if pd.api.types.is_numeric_dtype(df[covariate]):
                try:
                    # For numeric variables, compare means with t-test or non-parametric alternative
                    
                    if binary_treatment:
                        # Binary treatment case
                        group1 = df[df[treatment_var] == treatment_label][covariate].dropna()
                        group2 = df[df[treatment_var] == control_label][covariate].dropna()
                        
                        n1 = len(group1)
                        n2 = len(group2)
                        
                        if n1 < 3 or n2 < 3:
                            warnings.append(f"Too few observations in groups for covariate '{covariate}'")
                            balance_stats[covariate] = {
                                'test_type': 'none',
                                'p_value': None,
                                'statistic': None,
                                'group_stats': {
                                    str(treatment_label): {
                                        'n': n1,
                                        'mean': group1.mean() if n1 > 0 else None,
                                        'std': group1.std() if n1 > 1 else None
                                    },
                                    str(control_label): {
                                        'n': n2,
                                        'mean': group2.mean() if n2 > 0 else None,
                                        'std': group2.std() if n2 > 1 else None
                                    }
                                },
                                'std_mean_diff': None
                            }
                            continue
                        
                        # Try t-test first
                        try:
                            t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
                            test_type = "t-test"
                        except Exception as e:
                            # Fall back to non-parametric test
                            try:
                                u_stat, p_value = stats.mannwhitneyu(group1, group2)
                                t_stat = u_stat  # Just for storing
                                test_type = "Mann-Whitney U"
                            except Exception as e2:
                                warnings.append(f"Could not perform balance test for '{covariate}': {str(e2)}")
                                t_stat, p_value = None, None
                                test_type = "none"
                        
                        # Calculate standardized mean difference
                        mean1 = group1.mean()
                        mean2 = group2.mean()
                        std1 = group1.std()
                        std2 = group2.std()
                        
                        # Pooled standard deviation formula
                        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
                        
                        # Calculate standardized mean difference
                        if pooled_std > 0:
                            std_mean_diff = (mean1 - mean2) / pooled_std
                        else:
                            std_mean_diff = np.nan
                            warnings.append(f"Zero pooled standard deviation for '{covariate}'")
                        
                        # Store standardized mean difference for plotting
                        std_mean_diffs[covariate] = std_mean_diff
                        
                        # Check for imbalance
                        if p_value is not None and p_value < alpha:
                            significant_imbalances.append(covariate)
                        
                        # Store results
                        balance_stats[covariate] = {
                            'test_type': test_type,
                            'p_value': p_value,
                            'statistic': t_stat,
                            'group_stats': {
                                str(treatment_label): {
                                    'n': n1,
                                    'mean': mean1,
                                    'std': std1
                                },
                                str(control_label): {
                                    'n': n2,
                                    'mean': mean2,
                                    'std': std2
                                }
                            },
                            'std_mean_diff': std_mean_diff
                        }
                        
                        # Plot distribution comparison
                        ax = axes_dist[i]
                        bins = min(20, max(5, int(np.sqrt(n1 + n2))))
                        ax.hist(group1, bins=bins, alpha=0.5, label=f"{treatment_label}", color=PASTEL_COLORS[0])
                        ax.hist(group2, bins=bins, alpha=0.5, label=f"{control_label}", color=PASTEL_COLORS[1])
                        ax.set_title(f"{covariate}\nSMD: {std_mean_diff:.3f}")
                        ax.legend()
                    else:
                        # Multi-group treatment case
                        group_stats = {}
                        overall_p_value = None
                        test_statistic = None
                        
                        # Use ANOVA for multiple groups
                        groups = []
                        group_names = []
                        
                        for group_val in treatment_values:
                            group_data = df[df[treatment_var] == group_val][covariate].dropna()
                            if len(group_data) > 0:
                                groups.append(group_data)
                                group_names.append(str(group_val))
                                
                                # Store group statistics
                                group_stats[str(group_val)] = {
                                    'n': len(group_data),
                                    'mean': group_data.mean(),
                                    'std': group_data.std() if len(group_data) > 1 else None
                                }
                        
                        # Perform ANOVA if we have enough groups
                        if len(groups) >= 2:
                            try:
                                f_stat, p_value = stats.f_oneway(*groups)
                                test_type = "ANOVA"
                                overall_p_value = p_value
                                test_statistic = f_stat
                            except Exception as e:
                                # Fall back to Kruskal-Wallis test
                                try:
                                    h_stat, p_value = stats.kruskal(*groups)
                                    test_type = "Kruskal-Wallis"
                                    overall_p_value = p_value
                                    test_statistic = h_stat
                                except Exception as e2:
                                    warnings.append(f"Could not perform balance test for '{covariate}': {str(e2)}")
                                    test_type = "none"
                        else:
                            test_type = "none"
                            warnings.append(f"Insufficient groups for '{covariate}'")
                        
                        # Check for imbalance
                        if overall_p_value is not None and overall_p_value < alpha:
                            significant_imbalances.append(covariate)
                        
                        # Store results
                        balance_stats[covariate] = {
                            'test_type': test_type,
                            'p_value': overall_p_value,
                            'statistic': test_statistic,
                            'group_stats': group_stats
                        }
                        
                        # Plot distribution comparison
                        ax = axes_dist[i]
                        bins = min(20, max(5, int(np.sqrt(sum(len(g) for g in groups)))))
                
                except Exception as e:
                    warnings.append(f"Could not perform balance test for '{covariate}': {str(e)}")
                    balance_stats[covariate] = {
                        'test_type': 'none',
                        'p_value': None,
                        'statistic': None,
                        'group_stats': {}
                    }
        
        # Plot standardized mean differences if we have binary treatment
        if binary_treatment and std_mean_diffs:
            covs = list(std_mean_diffs.keys())
            smds = list(std_mean_diffs.values())
            
            # Sort by absolute SMD value
            sorted_indices = np.argsort(np.abs(smds))[::-1]
            covs = [covs[i] for i in sorted_indices]
            smds = [smds[i] for i in sorted_indices]
            
            # Plot SMDs
            ax_std_diff.barh(covs, smds, color=[PASTEL_COLORS[0] if smd >= 0 else PASTEL_COLORS[1] for smd in smds])
            ax_std_diff.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax_std_diff.axvline(x=0.1, color='red', linestyle='--', alpha=0.5)
            ax_std_diff.axvline(x=-0.1, color='red', linestyle='--', alpha=0.5)
            ax_std_diff.set_title('Standardized Mean Differences')
            ax_std_diff.set_xlabel('Standardized Mean Difference')
        
        # Determine overall result
        if significant_imbalances:
            result = AssumptionResult.FAILED
            details = f"Imbalance detected in {len(significant_imbalances)} covariates: {', '.join(significant_imbalances[:5])}"
            if len(significant_imbalances) > 5:
                details += f" and {len(significant_imbalances) - 5} more"
            warnings.append("Covariate imbalance may lead to biased treatment effect estimates")
            warnings.append("Consider using matching, weighting, or regression adjustment to address imbalance")
        else:
            result = AssumptionResult.PASSED
            details = "No significant imbalance detected in covariates"
        
        return {
            'result': result,
            'details': details,
            'balance_statistics': balance_stats,
            'significant_imbalances': significant_imbalances,
            'std_mean_diffs': std_mean_diffs,
            'warnings': warnings,
            'figures': {
                'std_diff_plot': fig_to_svg(fig_std_diff),
                'distribution_plot': fig_to_svg(fig_dist)
            }
        }
    
class ConditionalIndependenceTest(AssumptionTest):
    """Test for conditional independence between variables."""
    
    def __init__(self):
        super().__init__(
            name="Conditional Independence Test",
            description="Tests if variables are independent conditional on other variables (crucial for causal inference)",
            applicable_roles=["covariate", "treatment", "outcome"],
            applicable_types=["numeric", "categorical"]
        )
        
    def run_test(self, df, var1, var2, conditioning_vars, **kwargs):
        """
        Test conditional independence between two variables given conditioning variables.
        
        Args:
            df: DataFrame with the data
            var1: First variable name
            var2: Second variable name
            conditioning_vars: List of conditioning variable names
            method (str, optional): Method to use ('partial_correlation', 'regression', 'discrete'). Default is 'regression'.
            
        Returns:
            dict: Test results following the ConditionalIndependenceTest format
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from scipy import stats
        import statsmodels.api as sm
        
        method = kwargs.get('method', 'regression')
        alpha = kwargs.get('alpha', 0.05)
        warnings = []
        
        # Create figure
        fig = plt.figure(figsize=(10, 6))
        fig.patch.set_alpha(0.0)
        ax = fig.add_subplot(111)
        ax.patch.set_alpha(0.0)
        
        # Check if variables exist in the DataFrame
        for var in [var1, var2] + conditioning_vars:
            if var not in df.columns:
                return {
                    'result': AssumptionResult.NOT_APPLICABLE,
                    'details': f"Variable '{var}' not found in data",
                    'p_value': None,
                    'statistic': None,
                    'warnings': [f"Could not find variable: {var}"],
                    'figures': {
                        'plot': fig_to_svg(fig)
                    }
                }
        
        # Handle empty conditioning set
        if not conditioning_vars:
            warnings.append("No conditioning variables provided, testing marginal independence instead")
        
        # Clean data by dropping rows with missing values in any of the variables
        clean_df = df[[var1, var2] + conditioning_vars].dropna()
        
        if len(clean_df) < 10:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': f"Too few complete cases ({len(clean_df)}) after removing missing values",
                'p_value': None,
                'statistic': None,
                'warnings': ["Sample size too small for conditional independence testing"],
                'figures': {
                    'plot': fig_to_svg(fig)
                }
            }
        
        # Check variable types
        var1_is_numeric = pd.api.types.is_numeric_dtype(clean_df[var1])
        var2_is_numeric = pd.api.types.is_numeric_dtype(clean_df[var2])
        
        # Choose appropriate method based on variable types
        if method == 'auto':
            if var1_is_numeric and var2_is_numeric:
                method = 'partial_correlation'
            elif not var1_is_numeric or not var2_is_numeric:
                method = 'regression'
            else:
                method = 'discrete'
        
        # Perform the test
        if method == 'partial_correlation' and var1_is_numeric and var2_is_numeric:
            # Partial correlation for numeric variables
            try:
                # Create design matrix of conditioning variables
                X = clean_df[conditioning_vars]
                
                # Convert categorical variables to dummies
                X_dummies = pd.get_dummies(X, drop_first=True)
                
                # Add constant
                X_with_const = sm.add_constant(X_dummies)
                
                # Regress var1 on conditioning variables
                model1 = sm.OLS(clean_df[var1], X_with_const).fit()
                residuals1 = model1.resid
                
                # Regress var2 on conditioning variables
                model2 = sm.OLS(clean_df[var2], X_with_const).fit()
                residuals2 = model2.resid
                
                # Calculate correlation between residuals
                r, p_value = stats.pearsonr(residuals1, residuals2)
                
                # Plot residuals
                ax.scatter(residuals1, residuals2, alpha=0.6, color=PASTEL_COLORS[0])
                ax.set_title(f'Partial Correlation: r={r:.3f}, p={p_value:.4f}')
                ax.set_xlabel(f'{var1} residuals')
                ax.set_ylabel(f'{var2} residuals')
                
                # Add regression line
                slope, intercept, _, _, _ = stats.linregress(residuals1, residuals2)
                x_line = np.array([min(residuals1), max(residuals1)])
                y_line = intercept + slope * x_line
                ax.plot(x_line, y_line, 'r-')
                
                statistic = r
                test_name = "Partial Correlation"
                
            except Exception as e:
                warnings.append(f"Partial correlation failed: {str(e)}")
                r, p_value, statistic = None, None, None
                test_name = "None"
                ax.text(0.5, 0.5, "Could not calculate partial correlation", 
                       horizontalalignment='center', verticalalignment='center')
        
        elif method == 'regression':
            # Regression-based approach
            try:
                # Determine which variable to use as outcome
                if var1_is_numeric:
                    outcome_var = var1
                    predictor_var = var2
                else:
                    outcome_var = var2
                    predictor_var = var1
                
                # Create formula with conditioning variables
                conditioning_str = ' + '.join(conditioning_vars)
                formula1 = f"{outcome_var} ~ {conditioning_str}"
                formula2 = f"{outcome_var} ~ {conditioning_str} + {predictor_var}"
                
                # Fit models
                model1 = sm.formula.ols(formula1, data=clean_df).fit()
                model2 = sm.formula.ols(formula2, data=clean_df).fit()
                
                # Extract p-value for the predictor of interest
                if predictor_var in model2.pvalues:
                    p_value = model2.pvalues[predictor_var]
                    statistic = model2.tvalues[predictor_var]
                else:
                    # Try to find the predictor with categorical variables
                    p_value = None
                    statistic = None
                    for param in model2.pvalues.index:
                        if predictor_var in param:
                            p_value = model2.pvalues[param]
                            statistic = model2.tvalues[param]
                            break
                
                # Compare models with likelihood ratio test or F-test
                from statsmodels.stats.anova import anova_lm
                anova_table = anova_lm(model1, model2)
                
                # Get F-statistic and p-value from comparison
                if len(anova_table) > 1:
                    f_stat = anova_table.iloc[-1, 0]
                    model_p_value = anova_table.iloc[-1, -1]
                    
                    # If we didn't find a p-value above, use the model comparison p-value
                    if p_value is None:
                        p_value = model_p_value
                        statistic = f_stat
                
                # Plot residuals or partial regression plot
                if var1_is_numeric and var2_is_numeric:
                    # Partial regression plot (added variable plot)
                    # Get residuals from regressing var2 on conditioning variables
                    formula_partial = f"{var2} ~ {conditioning_str}"
                    model_partial = sm.formula.ols(formula_partial, data=clean_df).fit()
                    residuals_x = model_partial.resid
                    
                    # Get residuals from regressing var1 on conditioning variables
                    formula_partial2 = f"{var1} ~ {conditioning_str}"
                    model_partial2 = sm.formula.ols(formula_partial2, data=clean_df).fit()
                    residuals_y = model_partial2.resid
                    
                    # Plot residuals against each other
                    ax.scatter(residuals_x, residuals_y, alpha=0.6, color=PASTEL_COLORS[0])
                    ax.set_title(f'Added Variable Plot: p={p_value:.4f}')
                    ax.set_xlabel(f'{var2} residuals')
                    ax.set_ylabel(f'{var1} residuals')
                    
                    # Add regression line
                    slope, intercept, _, _, _ = stats.linregress(residuals_x, residuals_y)
                    x_line = np.array([min(residuals_x), max(residuals_x)])
                    y_line = intercept + slope * x_line
                    ax.plot(x_line, y_line, 'r-')
                else:
                    # For categorical variables, create boxplot or similar
                    if not var1_is_numeric and var2_is_numeric:
                        # Group numeric by categorical
                        cat_var = var1
                        num_var = var2
                    else:
                        cat_var = var2
                        num_var = var1
                    
                    # Calculate adjusted values
                    from statsmodels.graphics.factorplots import interaction_plot
                    
                    try:
                        # Just show means by category
                        cats = clean_df[cat_var].unique()
                        means = [clean_df[clean_df[cat_var] == cat][num_var].mean() for cat in cats]
                        ax.bar(range(len(cats)), means, tick_label=cats, color=PASTEL_COLORS)
                        ax.set_title(f'Mean {num_var} by {cat_var}: p={p_value:.4f}')
                        ax.set_xlabel(cat_var)
                        ax.set_ylabel(f'Mean {num_var}')
                    except Exception as e:
                        warnings.append(f"Could not create category plot: {str(e)}")
                        ax.text(0.5, 0.5, "Could not create visualization", 
                               horizontalalignment='center', verticalalignment='center')
                
                test_name = "Regression F-test"
                
            except Exception as e:
                warnings.append(f"Regression test failed: {str(e)}")
                p_value, statistic = None, None
                test_name = "None"
                ax.text(0.5, 0.5, "Could not perform regression test", 
                       horizontalalignment='center', verticalalignment='center')
        
        elif method == 'discrete':
            # For discrete variables, use chi-square test of conditional independence
            try:
                # Create contingency table for each stratum of conditioning variables
                overall_chi2 = 0
                overall_dof = 0
                n_strata = 0
                
                # Implement stratification by conditioning variables
                if conditioning_vars:
                    # Create stratification variables
                    strata_df = clean_df[conditioning_vars].copy()
                    
                    # For numeric conditioning variables, bin them
                    for var in conditioning_vars:
                        if pd.api.types.is_numeric_dtype(strata_df[var]):
                            # Create 3 bins (low, medium, high)
                            strata_df[var] = pd.qcut(strata_df[var], 3, labels=['low', 'medium', 'high'])
                    
                    # Create a combined stratum identifier
                    strata_df['stratum'] = strata_df.astype(str).agg('_'.join, axis=1)
                    strata = strata_df['stratum'].unique()
                    
                    # Create a subplot grid for visualizing each stratum
                    if len(strata) > 1:
                        plt.close(fig)  # Close the original figure
                        n_cols = min(3, len(strata))
                        n_rows = (len(strata) + n_cols - 1) // n_cols
                        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
                        fig.patch.set_alpha(0.0)
                        axes = np.array(axes).flatten() if len(strata) > 1 else [axes]
                    
                    # Calculate chi-square for each stratum
                    p_values = []
                    chi2_values = []
                    dof_values = []
                    
                    for i, s in enumerate(strata):
                        # Get data for this stratum
                        stratum_data = clean_df[strata_df['stratum'] == s]
                        
                        if len(stratum_data) < 5:
                            warnings.append(f"Stratum {s} has fewer than 5 observations, skipping")
                            continue
                        
                        # Create contingency table
                        contingency = pd.crosstab(stratum_data[var1], stratum_data[var2])
                        
                        # Skip if contingency table has dimensions less than 2x2
                        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                            warnings.append(f"Stratum {s} has insufficient variation, skipping")
                            continue
                        
                        # Calculate chi-square
                        chi2, p, dof, expected = stats.chi2_contingency(contingency)
                        
                        # Add to overall statistics using Cochran-Mantel-Haenszel approach
                        overall_chi2 += chi2
                        overall_dof += dof
                        n_strata += 1
                        
                        p_values.append(p)
                        chi2_values.append(chi2)
                        dof_values.append(dof)
                        
                        # Plot contingency table for this stratum if we have multiple strata
                        if len(strata) > 1:
                            ax = axes[i]
                            import seaborn as sns
                            sns.heatmap(contingency, annot=True, cmap='YlGnBu', ax=ax, fmt='d')
                            ax.set_title(f'Stratum: {s}\nChi2={chi2:.2f}, p={p:.4f}')
                    
                    # Calculate combined p-value using Fisher's method
                    if p_values:
                        from scipy.stats import combine_pvalues
                        combined_chi2, combined_p = combine_pvalues(p_values, method='fisher')
                        
                        # Use the combined p-value
                        p_value = combined_p
                        statistic = combined_chi2
                        
                        # Add summary information
                        if len(strata) > 1:
                            summary_ax = fig.add_subplot(n_rows+1, 1, n_rows+1)
                            summary_ax.axis('off')
                            summary_text = f"Combined p-value (Fisher's method): {p_value:.4f}\n"
                            summary_text += f"Number of strata: {n_strata}\n"
                            summary_text += f"Individual p-values: {', '.join([f'{p:.4f}' for p in p_values])}"
                            summary_ax.text(0.5, 0.5, summary_text, 
                                          horizontalalignment='center', verticalalignment='center')
                    else:
                        warnings.append("No valid strata for chi-square test")
                        p_value = None
                        statistic = None
                    
                    test_name = "Stratified Chi-square Test"
                else:
                    # Just test marginal independence with chi-square
                    contingency = pd.crosstab(clean_df[var1], clean_df[var2])
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                    
                    # Plot the contingency table as a heatmap
                    import seaborn as sns
                    sns.heatmap(contingency, annot=True, cmap='YlGnBu', ax=ax, fmt='d')
                    ax.set_title(f'Contingency Table: Chi2={chi2:.2f}, p={p_value:.4f}')
                    
                    statistic = chi2
                    test_name = "Chi-square Test"
                
            except Exception as e:
                warnings.append(f"Discrete test failed: {str(e)}")
                p_value, statistic = None, None
                test_name = "None"
                ax.text(0.5, 0.5, "Could not perform discrete test", 
                       horizontalalignment='center', verticalalignment='center')
        
        else:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': f"Unknown test method: {method}",
                'p_value': None,
                'statistic': None,
                'warnings': [f"Unrecognized test method: {method}"],
                'figures': {
                    'plot': fig_to_svg(fig)
                }
            }
        
        # Determine result
        if p_value is None:
            result = AssumptionResult.NOT_APPLICABLE
            details = f"Could not test conditional independence between '{var1}' and '{var2}'"
        elif p_value < alpha:
            result = AssumptionResult.FAILED
            details = f"Variables '{var1}' and '{var2}' are not conditionally independent given {conditioning_vars} ({test_name}: p={p_value:.4f})"
            warnings.append("Lack of conditional independence may violate assumptions of causal inference methods")
        else:
            result = AssumptionResult.PASSED
            details = f"Variables '{var1}' and '{var2}' appear to be conditionally independent given {conditioning_vars} ({test_name}: p={p_value:.4f})"
        
        return {
            'result': result,
            'details': details,
            'p_value': p_value,
            'statistic': statistic,
            'test_used': test_name,
            'conditioning_vars': conditioning_vars,
            'warnings': warnings,
            'figures': {
                'plot': fig_to_svg(fig)
            }
        }

class ExogeneityTest(AssumptionTest):
    """Test for exogeneity of treatment assignment."""
    
    def __init__(self):
        super().__init__(
            name="Exogeneity Test",
            description="Tests if treatment assignment is independent of potential outcomes",
            applicable_roles=["treatment", "outcome", "covariate"],
            applicable_types=["numeric", "categorical"]
        )
        
    def run_test(self, df, treatment_var, outcome_var, covariates, **kwargs):
        """
        Test for exogeneity of treatment assignment.
        
        Args:
            df: DataFrame with the data
            treatment_var: Name of the treatment variable column
            outcome_var: Name of the outcome variable column
            covariates: List of covariate column names to control for
            method (str, optional): Method to use ('hausman', 'placebo'). Default is 'hausman'.
            
        Returns:
            dict: Test results following the ExogeneityTest format
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import statsmodels.api as sm
        from scipy import stats
        
        method = kwargs.get('method', 'hausman')
        placebo_var = kwargs.get('placebo_var', None)
        alpha = kwargs.get('alpha', 0.05)
        warnings = []
        
        # Create figure
        fig = plt.figure(figsize=(10, 6))
        fig.patch.set_alpha(0.0)
        ax = fig.add_subplot(111)
        ax.patch.set_alpha(0.0)
        
        # Check if required variables exist in the DataFrame
        required_vars = [treatment_var, outcome_var] + covariates
        if method == 'placebo' and placebo_var:
            required_vars.append(placebo_var)
            
        for var in required_vars:
            if var not in df.columns:
                return {
                    'result': AssumptionResult.NOT_APPLICABLE,
                    'details': f"Variable '{var}' not found in data",
                    'p_value': None,
                    'statistic': None,
                    'warnings': [f"Could not find variable: {var}"],
                    'figures': {
                        'plot': fig_to_svg(fig)
                    }
                }
        
        # Clean data by dropping rows with missing values in required variables
        clean_df = df[required_vars].dropna()
        
        if len(clean_df) < 20:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': f"Too few complete cases ({len(clean_df)}) after removing missing values",
                'p_value': None,
                'statistic': None,
                'warnings': ["Sample size too small for exogeneity testing"],
                'figures': {
                    'plot': fig_to_svg(fig)
                }
            }
        
        # Check treatment variable type
        if not pd.api.types.is_numeric_dtype(clean_df[treatment_var]):
            # Try to convert categorical to numeric
            if clean_df[treatment_var].nunique() <= 2:
                warnings.append(f"Converting categorical treatment variable to numeric")
                unique_values = clean_df[treatment_var].unique()
                treatment_map = {val: i for i, val in enumerate(unique_values)}
                clean_df['treatment_numeric'] = clean_df[treatment_var].map(treatment_map)
                treatment_var_numeric = 'treatment_numeric'
            else:
                return {
                    'result': AssumptionResult.NOT_APPLICABLE,
                    'details': f"Treatment variable is categorical with more than 2 levels, not supported",
                    'p_value': None,
                    'statistic': None,
                    'warnings': ["Categorical treatment with >2 levels not supported for exogeneity test"],
                    'figures': {
                        'plot': fig_to_svg(fig)
                    }
                }
        else:
            treatment_var_numeric = treatment_var
        
        # Check outcome variable type
        if not pd.api.types.is_numeric_dtype(clean_df[outcome_var]):
            # Try to convert categorical to numeric
            if clean_df[outcome_var].nunique() <= 2:
                warnings.append(f"Converting categorical outcome variable to numeric")
                unique_values = clean_df[outcome_var].unique()
                outcome_map = {val: i for i, val in enumerate(unique_values)}
                clean_df['outcome_numeric'] = clean_df[outcome_var].map(outcome_map)
                outcome_var_numeric = 'outcome_numeric'
            else:
                return {
                    'result': AssumptionResult.NOT_APPLICABLE,
                    'details': f"Outcome variable is categorical with more than 2 levels, not supported",
                    'p_value': None,
                    'statistic': None,
                    'warnings': ["Categorical outcome with >2 levels not supported for exogeneity test"],
                    'figures': {
                        'plot': fig_to_svg(fig)
                    }
                }
        else:
            outcome_var_numeric = outcome_var
        
        # Perform the test based on the specified method
        if method == 'hausman':
            # Hausman test approach using control function
            try:
                # Step 1: Create formula string for covariates
                if covariates:
                    covariates_str = ' + '.join(covariates)
                    formula_first_stage = f"{treatment_var_numeric} ~ {covariates_str}"
                else:
                    # If no covariates, use an intercept-only model
                    formula_first_stage = f"{treatment_var_numeric} ~ 1"
                
                # Step 2: First stage regression to get residuals
                model_first_stage = sm.formula.ols(formula_first_stage, data=clean_df).fit()
                clean_df['treatment_residuals'] = model_first_stage.resid
                
                # Step 3: Second stage regression including residuals
                if covariates:
                    formula_second_stage = f"{outcome_var_numeric} ~ {treatment_var_numeric} + {covariates_str} + treatment_residuals"
                else:
                    formula_second_stage = f"{outcome_var_numeric} ~ {treatment_var_numeric} + treatment_residuals"
                
                model_second_stage = sm.formula.ols(formula_second_stage, data=clean_df).fit()
                
                # Step 4: Check significance of the residuals coefficient
                p_value = model_second_stage.pvalues['treatment_residuals']
                statistic = model_second_stage.tvalues['treatment_residuals']
                
                # Plot the relationship between treatment residuals and outcome
                ax.scatter(clean_df['treatment_residuals'], clean_df[outcome_var_numeric], alpha=0.6, color=PASTEL_COLORS[0])
                ax.set_title(f'Treatment Residuals vs. Outcome: p={p_value:.4f}')
                ax.set_xlabel('Treatment Residuals')
                ax.set_ylabel(f'Outcome: {outcome_var}')
                
                # Add regression line
                slope, intercept, _, _, _ = stats.linregress(clean_df['treatment_residuals'], clean_df[outcome_var_numeric])
                x_line = np.array([min(clean_df['treatment_residuals']), max(clean_df['treatment_residuals'])])
                y_line = intercept + slope * x_line
                ax.plot(x_line, y_line, 'r-')
                
                test_name = "Hausman/Control Function Test"
                details_appendix = "residual coefficient in augmented regression"
                
            except Exception as e:
                warnings.append(f"Hausman test failed: {str(e)}")
                p_value, statistic = None, None
                test_name = "None"
                details_appendix = "test calculation failed"
                ax.text(0.5, 0.5, "Could not perform Hausman test", 
                       horizontalalignment='center', verticalalignment='center')
        
        elif method == 'placebo':
            # Placebo outcome test
            if not placebo_var:
                return {
                    'result': AssumptionResult.NOT_APPLICABLE,
                    'details': "Placebo variable not specified for placebo test",
                    'p_value': None,
                    'statistic': None,
                    'warnings': ["Must specify placebo_var for placebo test method"],
                    'figures': {
                        'plot': fig_to_svg(fig)
                    }
                }
            
            # Check placebo variable type
            if not pd.api.types.is_numeric_dtype(clean_df[placebo_var]):
                # Try to convert categorical to numeric
                if clean_df[placebo_var].nunique() <= 2:
                    warnings.append(f"Converting categorical placebo variable to numeric")
                    unique_values = clean_df[placebo_var].unique()
                    placebo_map = {val: i for i, val in enumerate(unique_values)}
                    clean_df['placebo_numeric'] = clean_df[placebo_var].map(placebo_map)
                    placebo_var_numeric = 'placebo_numeric'
                else:
                    return {
                        'result': AssumptionResult.NOT_APPLICABLE,
                        'details': f"Placebo variable is categorical with more than 2 levels, not supported",
                        'p_value': None,
                        'statistic': None,
                        'warnings': ["Categorical placebo with >2 levels not supported for exogeneity test"],
                        'figures': {
                            'plot': fig_to_svg(fig)
                        }
                    }
            else:
                placebo_var_numeric = placebo_var
            
            try:
                # Create regression formula
                if covariates:
                    covariates_str = ' + '.join(covariates)
                    formula = f"{placebo_var_numeric} ~ {treatment_var_numeric} + {covariates_str}"
                else:
                    formula = f"{placebo_var_numeric} ~ {treatment_var_numeric}"
                
                # Fit the model
                model = sm.formula.ols(formula, data=clean_df).fit()
                
                # Extract treatment effect on placebo outcome
                p_value = model.pvalues[treatment_var_numeric]
                statistic = model.tvalues[treatment_var_numeric]
                
                # Plot the relationship
                if pd.api.types.is_numeric_dtype(clean_df[treatment_var]):
                    # Scatter plot for continuous treatment
                    ax.scatter(clean_df[treatment_var], clean_df[placebo_var], alpha=0.6, color=PASTEL_COLORS[0])
                    
                    # Add regression line
                    slope, intercept, _, _, _ = stats.linregress(clean_df[treatment_var], clean_df[placebo_var])
                    x_line = np.array([min(clean_df[treatment_var]), max(clean_df[treatment_var])])
                    y_line = intercept + slope * x_line
                    ax.plot(x_line, y_line, 'r-')
                else:
                    # Box plot for categorical treatment
                    clean_df.boxplot(column=placebo_var, by=treatment_var, ax=ax)
                
                ax.set_title(f'Treatment vs. Placebo Outcome: p={p_value:.4f}')
                ax.set_xlabel(f'Treatment: {treatment_var}')
                ax.set_ylabel(f'Placebo: {placebo_var}')
                
                test_name = "Placebo Outcome Test"
                details_appendix = "treatment effect on placebo outcome"
                
            except Exception as e:
                warnings.append(f"Placebo test failed: {str(e)}")
                p_value, statistic = None, None
                test_name = "None"
                details_appendix = "test calculation failed"
                ax.text(0.5, 0.5, "Could not perform placebo test", 
                       horizontalalignment='center', verticalalignment='center')
        
        else:
            return {
                'result': AssumptionResult.NOT_APPLICABLE,
                'details': f"Unknown test method: {method}",
                'p_value': None,
                'statistic': None,
                'warnings': [f"Unrecognized test method: {method}"],
                'figures': {
                    'plot': fig_to_svg(fig)
                }
            }
        
        # Determine result
        if p_value is None:
            result = AssumptionResult.NOT_APPLICABLE
            details = f"Could not test exogeneity of {treatment_var}"
        elif p_value < alpha:
            result = AssumptionResult.FAILED
            details = f"Treatment variable '{treatment_var}' appears to be endogenous (p={p_value:.4f} for {details_appendix})"
            warnings.append("Endogenous treatment may lead to biased treatment effect estimates")
            warnings.append("Consider using instrumental variables or other methods to address endogeneity")
        else:
            result = AssumptionResult.PASSED
            details = f"No evidence of endogeneity for treatment variable '{treatment_var}' (p={p_value:.4f} for {details_appendix})"
        
        return {
            'result': result,
            'details': details,
            'p_value': p_value,
            'statistic': statistic,
            'test_used': test_name,
            'method': method,
            'covariates': covariates,
            'warnings': warnings,
            'figures': {
                'plot': fig_to_svg(fig)
            }
        }
