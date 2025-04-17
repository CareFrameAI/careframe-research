import matplotlib.pyplot as plt
import seaborn as sns
from data.selection.detailed_tests.formatting import fig_to_svg, PASTEL_COLORS, PASTEL_CMAP
import traceback
import pandas as pd
from scipy import stats
import numpy as np
from typing import Dict, Any
from data.assumptions.tests import NormalityTest, OutlierTest, SampleSizeTest, HomogeneityOfVarianceTest, LinearityTest, IndependenceTest

def point_biserial_correlation(binary_var: pd.Series, continuous_var: pd.Series, alpha: float) -> Dict[str, Any]:
    """Calculates Point-Biserial correlation with comprehensive statistics and assumption checks."""
    try:
        # Remove rows with NaN values
        valid_data = pd.DataFrame({'binary': binary_var, 'continuous': continuous_var}).dropna()
        binary = valid_data['binary']
        continuous = valid_data['continuous']
        
        # Check if continuous variable is numeric
        if not pd.api.types.is_numeric_dtype(continuous):
            # Try to convert to numeric if possible
            try:
                continuous = pd.to_numeric(continuous)
            except:
                raise ValueError("Continuous variable must contain numeric values only")
        
        # Run assumption tests
        assumptions = {}
        assumption_violations = []
        
        # 1. Check if binary variable is actually binary
        unique_vals = binary.unique()
        if len(unique_vals) != 2:
            raise ValueError(f"Binary variable must have exactly 2 unique values, found {len(unique_vals)}")
        
        # Convert binary variable to 0/1 if needed
        # Store original values for labeling
        original_vals = unique_vals.copy()
        
        if not set(unique_vals).issubset({0, 1}):
            # Convert to 0/1 format
            mapping = {val: i for i, val in enumerate(unique_vals)}
            binary = binary.map(mapping)
            unique_vals = [0, 1]
            
        # Get the two groups
        group0 = continuous[binary == 0]
        group1 = continuous[binary == 1]
        group0_name = f"Group {unique_vals[0]}"
        group1_name = f"Group {unique_vals[1]}"
        
        # 2. Check normality of continuous variable within each group
        normality_test = NormalityTest()
        group0_normality = normality_test.run_test(data=group0)
        group1_normality = normality_test.run_test(data=group1)
        
        assumptions[f'normality_{group0_name}'] = group0_normality
        assumptions[f'normality_{group1_name}'] = group1_normality
        
        if group0_normality.get('result') == 'failed' or group1_normality.get('result') == 'failed':
            assumption_violations.append("normality")
        
        # 3. Check for outliers in continuous variable
        outlier_test = OutlierTest()
        group0_outliers = outlier_test.run_test(data=group0)
        group1_outliers = outlier_test.run_test(data=group1)
        
        assumptions[f'outliers_{group0_name}'] = group0_outliers
        assumptions[f'outliers_{group1_name}'] = group1_outliers
        
        if group0_outliers.get('result') == 'failed' or group1_outliers.get('result') == 'failed':
            assumption_violations.append("outliers")
        
        # 4. Check sample size
        sample_size_test = SampleSizeTest()
        n0, n1 = len(group0), len(group1)
        
        group0_size_result = sample_size_test.run_test(data=group0, min_recommended=20)
        group1_size_result = sample_size_test.run_test(data=group1, min_recommended=20)
        
        assumptions[f'sample_size_{group0_name}'] = group0_size_result
        assumptions[f'sample_size_{group1_name}'] = group1_size_result
        
        if group0_size_result.get('result') == 'failed' or group1_size_result.get('result') == 'failed':
            assumption_violations.append("sample_size")
        
        # 5. Check for homogeneity of variance
        homogeneity_test = HomogeneityOfVarianceTest()
        homogeneity_result = homogeneity_test.run_test(data=continuous, groups=binary)
        assumptions['homogeneity_of_variance'] = homogeneity_result
            
        if homogeneity_result.get('result') == 'failed':
            assumption_violations.append("homogeneity_of_variance")
        
        # 6. Add linearity test (important for correlation analysis)
        linearity_test = LinearityTest()
        linearity_result = linearity_test.run_test(x=binary, y=continuous)
        assumptions['linearity'] = linearity_result
        
        if linearity_result.get('result') == 'failed':
            assumption_violations.append("linearity")
            
        # 7. Add independence test
        independence_test = IndependenceTest()
        independence_result = independence_test.run_test(data=continuous)
        assumptions['independence'] = independence_result
        
        if independence_result.get('result') == 'failed':
            assumption_violations.append("independence")
        
        # Calculate Point-Biserial correlation (equivalent to Pearson with binary)
        r_pb, p_value = stats.pointbiserialr(binary, continuous)
        
        # Sample size
        n = len(binary)
        
        # Get group statistics
        mean0, mean1 = group0.mean(), group1.mean()
        std0, std1 = group0.std(), group1.std()
        
        # Calculate Cohen's d effect size
        # Pooled standard deviation
        s_pooled = np.sqrt(((n0-1)*(std0**2) + (n1-1)*(std1**2)) / (n0+n1-2))
        cohen_d = (mean1 - mean0) / s_pooled
        
        # Calculate confidence interval for r_pb
        # Fisher's z-transformation
        z = 0.5 * np.log((1 + r_pb) / (1 - r_pb))
        se = 1 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf(0.975)
        
        ci_lower_z = z - z_crit * se
        ci_upper_z = z + z_crit * se
        
        # Back-transform to r
        ci_lower = (np.exp(2 * ci_lower_z) - 1) / (np.exp(2 * ci_lower_z) + 1)
        ci_upper = (np.exp(2 * ci_upper_z) - 1) / (np.exp(2 * ci_upper_z) + 1)
        
        # Calculate statistical power
        power_analysis = calculate_correlation_power(n, r_pb, alpha)
        
        # Interpret effect size
        if abs(r_pb) < 0.1:
            effect_magnitude = "Negligible"
        elif abs(r_pb) < 0.3:
            effect_magnitude = "Small"
        elif abs(r_pb) < 0.5:
            effect_magnitude = "Medium"
        else:
            effect_magnitude = "Large"
            
        # Coefficient of determination (r²)
        r_squared = r_pb ** 2
        
        # Independent t-test (alternative view of point-biserial)
        t_stat = r_pb * np.sqrt((n - 2) / (1 - r_pb**2))
        df = n - 2
        
        # Determine significance
        significant = p_value < alpha
        
        # Create interpretation
        direction = "positive" if r_pb > 0 else "negative"
        
        # Create figures for visualization
        figures = {}
        
        # 1. Box plot comparing the groups
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x=binary, y=continuous, ax=ax, palette=[PASTEL_COLORS[0], PASTEL_COLORS[1]])
        # Set alpha for the boxes after creating the plot
        for patch in ax.artists:
            patch.set_alpha(0.6)
        ax.set_xlabel("Group")
        ax.set_ylabel(continuous.name if hasattr(continuous, 'name') else 'Continuous Variable')
        ax.set_title(f"Point-Biserial Correlation: r_pb = {r_pb:.3f}, p = {p_value:.5f}")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([group0_name, group1_name])
        figures['boxplot'] = fig_to_svg(fig)
        
        # 2. Violin plot for more detailed distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.violinplot(x=binary, y=continuous, ax=ax, palette=[PASTEL_COLORS[0], PASTEL_COLORS[1]])
        # Set alpha for the violin plot elements
        for collection in ax.collections:
            collection.set_alpha(0.6)
        ax.set_xlabel("Group")
        ax.set_ylabel(continuous.name if hasattr(continuous, 'name') else 'Continuous Variable')
        ax.set_title(f"Distribution by Group")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([group0_name, group1_name])
        figures['violinplot'] = fig_to_svg(fig)
        
        # 3. Group means with error bars
        fig, ax = plt.subplots(figsize=(8, 6))
        means = [mean0, mean1]
        errors = [std0/np.sqrt(n0), std1/np.sqrt(n1)]  # Standard error
        ax.bar([0, 1], means, yerr=errors, capsize=10, color=[PASTEL_COLORS[0], PASTEL_COLORS[1]], alpha=0.6)
        ax.set_xlabel("Group")
        ax.set_ylabel("Mean Value")
        ax.set_title(f"Group Means with Standard Error")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([group0_name, group1_name])
        figures['means_plot'] = fig_to_svg(fig)
        
        # 4. Effect size visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        # Create distribution curves for both groups
        x_range = np.linspace(
            min(continuous.min(), continuous.mean() - 4*continuous.std()),
            max(continuous.max(), continuous.mean() + 4*continuous.std()),
            1000
        )
        
        y0 = stats.norm.pdf(x_range, mean0, std0)
        y1 = stats.norm.pdf(x_range, mean1, std1)
        
        ax.plot(x_range, y0, color=PASTEL_COLORS[0], label=group0_name)
        ax.plot(x_range, y1, color=PASTEL_COLORS[1], label=group1_name)
        ax.axvline(mean0, color=PASTEL_COLORS[0], linestyle='--')
        ax.axvline(mean1, color=PASTEL_COLORS[1], linestyle='--')
        ax.fill_between(x_range, y0, color=PASTEL_COLORS[0], alpha=0.3)
        ax.fill_between(x_range, y1, color=PASTEL_COLORS[1], alpha=0.3)
        
        # Add Cohen's d indicator
        ax.annotate(f"Cohen's d = {cohen_d:.3f}", 
                   xy=(0.5, 0.9), 
                   xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        # Add r_pb indicator
        ax.annotate(f"r_pb = {r_pb:.3f}", 
                   xy=(0.5, 0.8), 
                   xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        ax.set_title("Effect Size Visualization")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        figures['effect_size_plot'] = fig_to_svg(fig)
        
        # Create main interpretation text
        interpretation = f"Point-biserial correlation: r_pb = {r_pb:.3f}, p = {p_value:.5f}.\n"
        
        if significant:
            interpretation += f"There is a statistically significant {effect_magnitude.lower()} {direction} relationship "
            interpretation += f"between the binary and continuous variables.\n"
            interpretation += f"The coefficient of determination (r² = {r_squared:.3f}) indicates that {r_squared*100:.1f}% "
            interpretation += f"of the variance in the continuous variable can be explained by group membership.\n"
            
            if direction == "positive":
                interpretation += f"{group1_name} (Mean = {mean1:.3f}) has higher values than {group0_name} (Mean = {mean0:.3f}).\n"
            else:
                interpretation += f"{group0_name} (Mean = {mean0:.3f}) has higher values than {group1_name} (Mean = {mean1:.3f}).\n"

            from data.selection.stat_tests import _interpret_cohens_d
            interpretation += f"The standardized mean difference (Cohen's d = {cohen_d:.3f}) indicates a {_interpret_cohens_d(cohen_d)} effect.\n"
            interpretation += f"The {(1-alpha)*100:.1f}% confidence interval for r_pb is [{ci_lower:.3f}, {ci_upper:.3f}]."
        else:
            interpretation += "There is no statistically significant relationship between the binary and continuous variables."
        
        # Add power analysis to interpretation
        if power_analysis['power'] < 0.8 and p_value >= alpha:
            interpretation += f"\n\nStatistical power is low ({power_analysis['power']:.2f}). The test may not have enough "
            interpretation += f"power to detect a true effect of this size. Consider increasing the sample size to at least {power_analysis['recommended_sample_size']}."
        elif power_analysis['power'] < 0.8 and p_value < alpha:
            interpretation += f"\n\nNote: Despite achieving statistical significance, the statistical power is relatively low ({power_analysis['power']:.2f})."
        
        # Add assumption check results to interpretation
        interpretation += "\n\nAssumption checks:"
        
        # Normality
        group0_normality_satisfied = assumptions[f'normality_{group0_name}'].get('result') == 'passed'
        group1_normality_satisfied = assumptions[f'normality_{group1_name}'].get('result') == 'passed'
        
        if group0_normality_satisfied and group1_normality_satisfied:
            interpretation += "\n- Normality: The continuous variable appears to be normally distributed within each group."
        else:
            interpretation += "\n- Normality: The continuous variable may not be normally distributed within one or both groups. "
            interpretation += "This may affect the validity of the point-biserial correlation."
            
        # Outliers
        group0_outliers_satisfied = assumptions[f'outliers_{group0_name}'].get('result') == 'passed'
        group1_outliers_satisfied = assumptions[f'outliers_{group1_name}'].get('result') == 'passed'
        
        if group0_outliers_satisfied and group1_outliers_satisfied:
            interpretation += "\n- Outliers: No significant outliers detected in either group."
        else:
            interpretation += "\n- Outliers: Potential outliers detected in one or both groups. "
            interpretation += "Consider removing outliers or using a robust method."
            
        # Sample size
        group0_size_satisfied = assumptions[f'sample_size_{group0_name}'].get('result') == 'passed'
        group1_size_satisfied = assumptions[f'sample_size_{group1_name}'].get('result') == 'passed'
        
        if group0_size_satisfied and group1_size_satisfied:
            interpretation += "\n- Sample size: Both groups have adequate sample sizes."
        else:
            interpretation += "\n- Sample size: One or both groups may have insufficient sample sizes. "
            interpretation += f"Group sizes: {group0_name}: {n0}, {group1_name}: {n1}."
            
        # Homogeneity of variance
        homogeneity_satisfied = assumptions['homogeneity_of_variance'].get('result') == 'passed'
        if homogeneity_satisfied:
            interpretation += "\n- Homogeneity of variance: The variances appear to be similar across groups."
        else:
            interpretation += "\n- Homogeneity of variance: The variances may differ between groups. "
            interpretation += "This may affect the interpretation of the point-biserial correlation."
        
        # Linearity
        linearity_satisfied = assumptions['linearity'].get('result') == 'passed'
        if linearity_satisfied:
            interpretation += "\n- Linearity: The relationship between variables appears to be linear."
        else:
            interpretation += "\n- Linearity: The relationship between variables may not be linear. "
            interpretation += "This could affect the interpretation of the correlation coefficient."
            
        # Independence
        independence_satisfied = assumptions['independence'].get('result') == 'passed'
        if independence_satisfied:
            interpretation += "\n- Independence: The observations appear to be independent."
        else:
            interpretation += "\n- Independence: The observations may not be independent. "
            interpretation += "This affects the validity of the point-biserial correlation."
        
        return {
            'test': "Point-Biserial Correlation",
            'statistic': float(r_pb),
            'p_value': float(p_value),
            'significant': significant,
            'interpretation': interpretation,
            'assumptions': assumptions,
            'assumption_violations': assumption_violations,
            
            # Group statistics
            'n0': int(n0),
            'n1': int(n1),
            'mean0': float(mean0),
            'mean1': float(mean1),
            'std0': float(std0),
            'std1': float(std1),
            'mean_diff': float(mean1 - mean0),
            
            # Effect sizes
            'effect_size': {
                'r_pb': float(r_pb),
                'r_squared': float(r_squared),
                'cohen_d': float(cohen_d),
                'interpretation': effect_magnitude,
                'direction': direction
            },
            
            # Confidence interval
            'confidence_interval': [float(ci_lower), float(ci_upper)],
            
            # T-test equivalent
            't_statistic': float(t_stat),
            'df': int(df),
            
            # Power analysis
            'power_analysis': power_analysis,
            
            # Sample size
            'sample_size': int(n),
            
            # Visualizations
            'figures': figures
        }
    except Exception as e:
        return {
            'test': 'Point-Biserial Correlation',
            'statistic': None,
            'p_value': None,
            'significant': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def calculate_correlation_power(n: int, r: float, alpha: float = 0.05) -> Dict[str, float]:
    """Calculate the statistical power for a correlation test."""
    # Fisher's Z transformation of r
    if r == 1.0:  # Avoid division by zero
        r = 0.9999
    elif r == -1.0:
        r = -0.9999
        
    z = 0.5 * np.log((1 + r) / (1 - r))
    
    # Standard error
    se = 1 / np.sqrt(n - 3)
    
    # Critical value
    z_crit = stats.norm.ppf(1 - alpha/2)
    
    # Non-centrality parameter
    ncp = abs(z) / se
    
    # Calculate power (two-tailed test)
    power = 1 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)
    
    # Calculate required sample size for power of 0.80
    if power < 0.8:
        # Solve for n: power = 1 - Φ(z_crit - |z|/sqrt(1/(n-3))) + Φ(-z_crit - |z|/sqrt(1/(n-3)))
        # This is an approximation using iteration
        target_power = 0.8
        test_n = n
        max_n = 5000  # Set a reasonable upper limit
        
        while test_n < max_n:
            test_n += 5
            test_se = 1 / np.sqrt(test_n - 3)
            test_ncp = abs(z) / test_se
            test_power = 1 - stats.norm.cdf(z_crit - test_ncp) + stats.norm.cdf(-z_crit - test_ncp)
            
            if test_power >= target_power:
                break
                
        recommended_n = test_n
    else:
        recommended_n = n  # Already sufficient
    
    return {
        'power': float(power),
        'recommended_sample_size': int(recommended_n)
    }