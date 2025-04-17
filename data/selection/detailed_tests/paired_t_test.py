import traceback
import pandas as pd
from scipy import stats
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import io
from typing import Dict, Any

# Import the fig_to_svg function and color palette from formatting
from data.selection.detailed_tests.formatting import fig_to_svg, PASTEL_COLORS, PASTEL_CMAP

# Import the necessary classes from format.py
from data.assumptions.format import AssumptionTestKeys
from data.assumptions.tests import (
    NormalityTest, OutlierTest, SampleSizeTest, HomoscedasticityTest, 
    LinearityTest, DistributionFitTest
)

def paired_t_test(group1: pd.Series, group2: pd.Series, alpha: float, alternative: str = 'two-sided') -> Dict[str, Any]:
    """Performs a paired t-test with comprehensive statistics and assumption checks."""
    try:
        # Drop missing values (only keep pairs where both values are present)
        valid_pairs = pd.DataFrame({'group1': group1, 'group2': group2}).dropna()
        
        if len(valid_pairs) < 2:
            return {
                'test': 'Paired T-Test',
                'statistic': None,
                'p_value': None,
                'significant': False,
                'reason': 'Less than two valid pairs provided.'
            }
            
        # Extract the paired data
        group1_clean = valid_pairs['group1']
        group2_clean = valid_pairs['group2']
        
        # Calculate the differences
        differences = group1_clean - group2_clean
        
        # Calculate basic statistics
        n = len(differences)
        mean_diff = float(differences.mean())
        std_diff = float(differences.std(ddof=1))
        se_diff = std_diff / np.sqrt(n)
        
        # Perform the paired t-test
        statistic, p_value = stats.ttest_rel(a=group1_clean, b=group2_clean, alternative=alternative)
        
        # Determine significance based on alpha and alternative
        if alternative == 'two-sided':
            significant = p_value < alpha
        else:
            significant = p_value < alpha
            
        # Calculate degrees of freedom
        df = n - 1
        
        # Calculate confidence interval for mean difference
        if alternative == 'two-sided':
            ci_lower = mean_diff - stats.t.ppf(1 - alpha/2, df) * se_diff
            ci_upper = mean_diff + stats.t.ppf(1 - alpha/2, df) * se_diff
        elif alternative == 'less':
            ci_lower = -np.inf
            ci_upper = mean_diff + stats.t.ppf(1 - alpha, df) * se_diff
        else:  # 'greater'
            ci_lower = mean_diff - stats.t.ppf(1 - alpha, df) * se_diff
            ci_upper = np.inf
            
        # Calculate effect size (Cohen's d for paired samples)
        # For paired samples, d = mean_diff / std_diff
        cohens_d = mean_diff / std_diff
        
        # Adjust for small sample bias if needed
        if n < 50:
            # Correction factor for small samples
            correction = 1 - (3 / (4 * df - 1))
            hedges_g = cohens_d * correction
        else:
            hedges_g = cohens_d
            
        # Determine magnitude of effect size
        if abs(hedges_g) < 0.2:
            effect_magnitude = "Negligible"
        elif abs(hedges_g) < 0.5:
            effect_magnitude = "Small"
        elif abs(hedges_g) < 0.8:
            effect_magnitude = "Medium"
        else:
            effect_magnitude = "Large"
            
        # Test assumptions
        assumptions = {}
        
        # 1. Test normality of differences
        try:
            normality_test = NormalityTest()
            assumptions['normality_differences'] = normality_test.run_test(data=differences)
        except Exception as e:
            assumptions['normality_differences'] = {
                'result': "failed",
                'error': f'Could not test normality: {str(e)}'
            }
            
        # 2. Test for outliers in differences
        try:
            outlier_test = OutlierTest()
            assumptions['outliers_differences'] = outlier_test.run_test(data=differences)
        except Exception as e:
            assumptions['outliers_differences'] = {
                'result': "failed",
                'error': f'Could not check for outliers: {str(e)}'
            }
            
        # 3. Sample size check
        try:
            # Group 1
            sample_size_test = SampleSizeTest()
            assumptions['sample_size_group1'] = sample_size_test.run_test(data=group1_clean, min_recommended=30)
            
            # Group 2
            assumptions['sample_size_group2'] = sample_size_test.run_test(data=group2_clean, min_recommended=30)
            
            # Differences
            assumptions['sample_size_differences'] = sample_size_test.run_test(data=differences, min_recommended=30)
        except Exception as e:
            assumptions['sample_size_differences'] = {
                'result': "failed",
                'error': f'Could not check sample size: {str(e)}'
            }
            
        # 4. Homoscedasticity check (comparing differences against their mean)
        try:
            # Create a proxy for X values (could be index or mean values)
            x_values = pd.Series(np.ones(len(differences)) * differences.mean())
            
            homoscedasticity_test = HomoscedasticityTest()
            assumptions['homoscedasticity_differences'] = homoscedasticity_test.run_test(
                residuals=differences - differences.mean(), 
                predicted=x_values
            )
        except Exception as e:
            assumptions['homoscedasticity_differences'] = {
                'result': "failed",
                'error': f'Could not check homoscedasticity: {str(e)}'
            }
            
        # 5. Test distribution fit of differences (specifically normal distribution)
        try:
            distribution_fit_test = DistributionFitTest()
            assumptions['distribution_fit_differences'] = distribution_fit_test.run_test(
                data=differences, 
                distribution="normal"
            )
        except Exception as e:
            assumptions['distribution_fit_differences'] = {
                'result': "failed",
                'error': f'Could not check distribution fit: {str(e)}'
            }
            
        # 6. Test linearity between paired values (if appropriate)
        try:
            linearity_test = LinearityTest()
            assumptions['linearity_between_groups'] = linearity_test.run_test(
                x=group1_clean, 
                y=group2_clean
            )
        except Exception as e:
            assumptions['linearity_between_groups'] = {
                'result': "failed",
                'error': f'Could not check linearity: {str(e)}'
            }
        
        # Create interpretation
        interpretation = f"Paired t-test comparing two related groups.\n\n"
        
        # Basic results
        interpretation += f"t({df}) = {statistic:.3f}, p = {p_value:.5f}\n"
        interpretation += f"Mean difference = {mean_diff:.3f}, SE = {se_diff:.3f}\n"
        interpretation += f"95% CI for difference: [{ci_lower:.3f}, {ci_upper:.3f}]\n"
        interpretation += f"Effect size: d = {cohens_d:.3f}, g = {hedges_g:.3f} ({effect_magnitude.lower()} effect)\n\n"
        
        # Group statistics
        interpretation += "Group statistics:\n"
        interpretation += f"- Group 1: mean = {float(group1_clean.mean()):.3f}, SD = {float(group1_clean.std(ddof=1)):.3f}\n"
        interpretation += f"- Group 2: mean = {float(group2_clean.mean()):.3f}, SD = {float(group2_clean.std(ddof=1)):.3f}\n"
        interpretation += f"- Number of pairs: {n}\n\n"
        
        # Assumptions
        interpretation += "Assumption tests:\n"
        
        # Normality of differences
        if 'normality_differences' in assumptions and 'result' in assumptions['normality_differences']:
            norm_result = assumptions['normality_differences']
            interpretation += f"- Normality of differences: "
            
            if norm_result['result'].value == 'passed':
                interpretation += "Assumption satisfied. The differences appear to be normally distributed.\n"
            elif norm_result['result'].value == 'warning':
                interpretation += f"Potential violation of normality ({norm_result.get('test_used', 'Shapiro-Wilk')} test, p = {norm_result.get('p_value', 0):.5f}). "
                
                if n >= 30:
                    interpretation += "However, due to the sample size (n ≥ 30), the t-test should be robust to violations of normality (Central Limit Theorem).\n"
                else:
                    interpretation += "Consider using a non-parametric alternative like the Wilcoxon signed-rank test.\n"
            else:  # failed
                interpretation += f"Violation of normality ({norm_result.get('test_used', 'Shapiro-Wilk')} test, p = {norm_result.get('p_value', 0):.5f}). "
                
                if n >= 30:
                    interpretation += "However, due to the sample size (n ≥ 30), the t-test should be robust to violations of normality (Central Limit Theorem).\n"
                else:
                    interpretation += "Consider using a non-parametric alternative like the Wilcoxon signed-rank test.\n"
        
        # Distribution fit of differences
        if 'distribution_fit_differences' in assumptions and 'result' in assumptions['distribution_fit_differences']:
            dist_result = assumptions['distribution_fit_differences']
            interpretation += f"- Distribution fit (normality): "
            
            if dist_result['result'].value == 'passed':
                interpretation += f"The differences follow a normal distribution (p = {dist_result.get('p_value', 0):.5f}). Parametric tests are appropriate.\n"
            elif dist_result['result'].value == 'warning':
                interpretation += f"The differences may not perfectly follow a normal distribution (p = {dist_result.get('p_value', 0):.5f}). "
                interpretation += "Consider checking a Q-Q plot for further assessment.\n"
            else:
                interpretation += f"The differences do not follow a normal distribution (p = {dist_result.get('p_value', 0):.5f}). "
                interpretation += "Consider a non-parametric alternative like the Wilcoxon signed-rank test.\n"
        
        # Outliers
        if 'outliers_differences' in assumptions and 'result' in assumptions['outliers_differences']:
            outlier_result = assumptions['outliers_differences']
            interpretation += f"- Outliers: "
            
            if outlier_result['result'].value == 'passed':
                interpretation += "No significant outliers detected in the differences.\n"
            elif outlier_result['result'].value == 'warning':
                interpretation += f"Some potential outliers detected in the differences. Consider examining these cases.\n"
            else:  # failed
                interpretation += f"Significant outliers detected in the differences. These may be influencing the results. "
                if 'outliers' in outlier_result:
                    interpretation += f"Outliers found at indices: {outlier_result['outliers']}. "
                interpretation += "Consider removing outliers if they represent measurement errors or using robust methods.\n"
        
        # Sample size
        if 'sample_size_differences' in assumptions and 'result' in assumptions['sample_size_differences']:
            size_result = assumptions['sample_size_differences']
            interpretation += f"- Sample size: "
            
            if size_result['result'].value == 'passed':
                interpretation += f"Sample size (n = {n}) is adequate for the paired t-test.\n"
            elif size_result['result'].value == 'warning':
                interpretation += f"Sample size (n = {n}) is smaller than recommended. Results should be interpreted with caution.\n"
            else:  # failed
                interpretation += f"Sample size (n = {n}) is very small. Results may not be reliable.\n"
                
                if 'minimum_required' in size_result:
                    interpretation += f"Recommended minimum sample size: {size_result['minimum_required']}.\n"
        
        # Homoscedasticity
        if 'homoscedasticity_differences' in assumptions and 'result' in assumptions['homoscedasticity_differences']:
            homo_result = assumptions['homoscedasticity_differences']
            interpretation += f"- Homoscedasticity of differences: "
            
            if homo_result['result'].value == 'passed':
                interpretation += "The variance of differences appears to be constant.\n"
            elif homo_result['result'].value == 'warning':
                interpretation += f"The variance of differences may not be perfectly constant. "
                interpretation += "This may affect the reliability of the test but is not a critical violation for paired t-tests.\n"
            else:  # failed
                interpretation += f"The variance of differences is not constant. "
                interpretation += "This may affect the reliability of the standard error estimation.\n"
        
        # Linearity between groups
        if 'linearity_between_groups' in assumptions and 'result' in assumptions['linearity_between_groups']:
            lin_result = assumptions['linearity_between_groups']
            interpretation += f"- Linearity between groups: "
            
            if lin_result['result'].value == 'passed':
                interpretation += "There appears to be a linear relationship between the paired measurements.\n"
            elif lin_result['result'].value == 'warning':
                interpretation += f"The relationship between paired measurements may not be perfectly linear. "
                interpretation += "This is not a critical assumption for paired t-tests but may be relevant for interpretation.\n"
            else:  # failed
                interpretation += f"There does not appear to be a linear relationship between paired measurements. "
                interpretation += "While not a critical assumption for paired t-tests, this may be relevant for interpretation.\n"
        
        # Conclusion
        interpretation += f"\nConclusion: There is {'a statistically significant' if significant else 'no statistically significant'} "
        
        if alternative == 'two-sided':
            interpretation += f"difference between the paired groups (p = {p_value:.5f}). "
        elif alternative == 'less':
            interpretation += f"evidence that Group 1 is less than Group 2 (p = {p_value:.5f}). "
        else:  # 'greater'
            interpretation += f"evidence that Group 1 is greater than Group 2 (p = {p_value:.5f}). "
        
        if significant:
            interpretation += f"The {effect_magnitude.lower()} effect size (g = {hedges_g:.3f}) suggests that "
            
            if mean_diff > 0:
                interpretation += f"Group 1 scores are higher than Group 2 scores by approximately {mean_diff:.3f} units on average."
            else:
                interpretation += f"Group 2 scores are higher than Group 1 scores by approximately {abs(mean_diff):.3f} units on average."
        
        # Additional statistics
        additional_stats = {}
        
        # 1. Non-parametric alternative: Wilcoxon signed-rank test
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(group1_clean, group2_clean, alternative=alternative)
            wilcoxon_significant = wilcoxon_p < alpha
            
            additional_stats['wilcoxon_test'] = {
                'statistic': float(wilcoxon_stat),
                'p_value': float(wilcoxon_p),
                'significant': wilcoxon_significant,
                'agreement_with_ttest': wilcoxon_significant == significant
            }
        except Exception as e:
            additional_stats['wilcoxon_test'] = {
                'error': str(e)
            }
        
        # 2. Bootstrap confidence interval for the mean difference
        try:
            n_bootstrap = 2000
            bootstrap_samples = []
            
            # Generate bootstrap samples of difference
            for _ in range(n_bootstrap):
                # Sample with replacement
                indices = np.random.choice(range(n), size=n, replace=True)
                bootstrap_diff = differences.iloc[indices].mean()
                bootstrap_samples.append(bootstrap_diff)
                
            # Calculate bootstrap CI
            boot_ci_lower = np.percentile(bootstrap_samples, 100 * alpha/2)
            boot_ci_upper = np.percentile(bootstrap_samples, 100 * (1 - alpha/2))
            
            additional_stats['bootstrap'] = {
                'n_samples': n_bootstrap,
                'ci_lower': float(boot_ci_lower),
                'ci_upper': float(boot_ci_upper),
                'mean': float(np.mean(bootstrap_samples)),
                'std_error': float(np.std(bootstrap_samples))
            }
        except Exception as e:
            additional_stats['bootstrap'] = {
                'error': str(e)
            }
        
        # 3. Power analysis
        try:
            from statsmodels.stats.power import TTestPower
            
            # Calculate power for the observed effect size
            power_analysis = TTestPower()
            observed_power = power_analysis.power(effect_size=abs(cohens_d), nobs=n, alpha=alpha, alternative=alternative)
            
            # Calculate required sample size for standard power levels
            n_for_80_power = int(np.ceil(power_analysis.solve_power(effect_size=abs(cohens_d), power=0.8, alpha=alpha, alternative=alternative)))
            n_for_90_power = int(np.ceil(power_analysis.solve_power(effect_size=abs(cohens_d), power=0.9, alpha=alpha, alternative=alternative)))
            n_for_95_power = int(np.ceil(power_analysis.solve_power(effect_size=abs(cohens_d), power=0.95, alpha=alpha, alternative=alternative)))
            
            additional_stats['power_analysis'] = {
                'observed_power': float(observed_power),
                'sample_size_for_80_power': n_for_80_power,
                'sample_size_for_90_power': n_for_90_power,
                'sample_size_for_95_power': n_for_95_power
            }
        except Exception as e:
            additional_stats['power_analysis'] = {
                'error': str(e)
            }
        
        # 4. Robust statistics - median difference and trimmed mean difference
        try:
            from scipy.stats import trim_mean
            
            # Median difference
            median_diff = np.median(differences)
            
            # Trimmed mean (removing 10% from each end)
            trimmed_mean_diff = trim_mean(differences, 0.1)
            
            additional_stats['robust_statistics'] = {
                'median_difference': float(median_diff),
                'trimmed_mean_difference': float(trimmed_mean_diff),
                'mad': float(stats.median_abs_deviation(differences, scale=1.4826))  # Median Absolute Deviation
            }
        except Exception as e:
            additional_stats['robust_statistics'] = {
                'error': str(e)
            }
        
        # 5. Descriptive statistics of differences
        try:
            from scipy.stats import skew, kurtosis
            
            # Calculate skewness and kurtosis of differences
            skewness = skew(differences)
            kurt = kurtosis(differences)  # Excess kurtosis
            
            additional_stats['difference_descriptives'] = {
                'min': float(differences.min()),
                'q1': float(np.percentile(differences, 25)),
                'median': float(np.median(differences)),
                'q3': float(np.percentile(differences, 75)),
                'max': float(differences.max()),
                'skewness': float(skewness),
                'kurtosis': float(kurt),
                'iqr': float(np.percentile(differences, 75) - np.percentile(differences, 25))
            }
        except Exception as e:
            additional_stats['difference_descriptives'] = {
                'error': str(e)
            }
            
        # Generate figures
        figures = {}
        
        # Figure 1: Histogram of differences with normal curve overlay
        try:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            
            # Plot histogram with pastel color
            sns.histplot(differences, kde=True, ax=ax1, color=PASTEL_COLORS[0], edgecolor='black')
            
            # Add normal curve
            x = np.linspace(differences.min(), differences.max(), 100)
            y = stats.norm.pdf(x, mean_diff, std_diff)
            y = y * len(differences) * (differences.max() - differences.min()) / 10  # Scale to match histogram height
            ax1.plot(x, y, 'r--', linewidth=2, label='Normal Curve')
            
            # Add vertical line at mean
            ax1.axvline(x=mean_diff, color='red', linestyle='-', linewidth=2, label=f'Mean = {mean_diff:.3f}')
            
            # Add zero line
            ax1.axvline(x=0, color='gray', linestyle='--', linewidth=1, label='Zero Difference')
            
            # Add confidence interval
            if not np.isinf(ci_lower) and not np.isinf(ci_upper):
                ax1.axvspan(ci_lower, ci_upper, alpha=0.2, color='green', label=f'{(1-alpha)*100:.0f}% CI')
            
            # Add labels and title
            ax1.set_xlabel('Difference (Group 1 - Group 2)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Distribution of Paired Differences')
            ax1.legend()
            
            # Add normality test result
            if 'normality_differences' in assumptions and 'p_value' in assumptions['normality_differences']:
                normality_p = assumptions['normality_differences']['p_value']
                normality_test_used = assumptions['normality_differences'].get('test_used', 'Normality test')
                
                normality_text = f"{normality_test_used}: p = {normality_p:.4f}"
                normality_text += f"\nNormality {'not rejected' if normality_p >= alpha else 'rejected'}"
                ax1.text(0.05, 0.95, normality_text, transform=ax1.transAxes, 
                        verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
            
            fig1.tight_layout()
            figures['difference_histogram'] = fig_to_svg(fig1)
        except Exception as e:
            figures['difference_histogram_error'] = str(e)
        
        # Figure 2: Box plots with individual data points
        try:
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            
            # Create DataFrame for seaborn
            plot_data = pd.DataFrame({
                'Group 1': group1_clean,
                'Group 2': group2_clean
            })
            plot_data_melted = pd.melt(plot_data, var_name='Group', value_name='Value')
            
            # Draw boxplot with pastel colors
            sns.boxplot(x='Group', y='Value', data=plot_data_melted, ax=ax2, width=0.5, 
                       palette=[PASTEL_COLORS[1], PASTEL_COLORS[2]])
            sns.stripplot(x='Group', y='Value', data=plot_data_melted, ax=ax2, 
                         color='black', alpha=0.5, jitter=True, size=4)
            
            # Add mean markers
            means = [group1_clean.mean(), group2_clean.mean()]
            ax2.plot(['Group 1', 'Group 2'], means, 'ro', markersize=10, label='Mean')
            
            # Connect paired observations with lines
            for i in range(len(group1_clean)):
                ax2.plot(['Group 1', 'Group 2'], [group1_clean.iloc[i], group2_clean.iloc[i]], 
                         'gray', alpha=0.2, linestyle='-', linewidth=1)
            
            # Add legend and labels
            ax2.legend()
            ax2.set_title('Paired Observations')
            ax2.set_ylabel('Value')
            
            # Add statistical results
            stats_text = f"t({df}) = {statistic:.3f}, p = {p_value:.4f}"
            if significant:
                stats_text += " *"
            ax2.text(0.5, 0.01, stats_text, transform=ax2.transAxes, 
                    horizontalalignment='center', bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
            
            fig2.tight_layout()
            figures['paired_boxplot'] = fig_to_svg(fig2)
        except Exception as e:
            figures['paired_boxplot_error'] = str(e)
        
        # Figure 3: QQ plot of differences
        try:
            fig3, ax3 = plt.subplots(figsize=(8, 8))
            
            # Create QQ plot
            stats.probplot(differences, dist="norm", plot=ax3)
            
            # Add title
            ax3.set_title('Q-Q Plot of Paired Differences')
            
            # Add line connecting first and third quartiles
            ax3.set_xlabel('Theoretical Quantiles')
            ax3.set_ylabel('Sample Quantiles')
            
            fig3.tight_layout()
            figures['qq_plot'] = fig_to_svg(fig3)
        except Exception as e:
            figures['qq_plot_error'] = str(e)
        
        # Figure 4: Paired differences plot
        try:
            fig4, ax4 = plt.subplots(figsize=(12, 6))
            
            # Create indices
            indices = range(len(differences))
            
            # Plot differences with pastel color
            ax4.scatter(indices, differences, color=PASTEL_COLORS[3], alpha=0.7)
            
            # Add horizontal line for mean difference
            ax4.axhline(y=mean_diff, color='red', linestyle='-', linewidth=2, label=f'Mean Difference = {mean_diff:.3f}')
            
            # Add horizontal line at zero
            ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, label='No Difference')
            
            # Add confidence interval
            if not np.isinf(ci_lower) and not np.isinf(ci_upper):
                ax4.axhspan(ci_lower, ci_upper, alpha=0.2, color='green', 
                           label=f'{(1-alpha)*100:.0f}% CI [{ci_lower:.3f}, {ci_upper:.3f}]')
            
            # Add labels and title
            ax4.set_xlabel('Observation')
            ax4.set_ylabel('Difference (Group 1 - Group 2)')
            ax4.set_title('Individual Paired Differences')
            ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
            
            # Add statistical results
            stats_text = f"t({df}) = {statistic:.3f}, p = {p_value:.4f}"
            if significant:
                stats_text += " *"
            ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, 
                    verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
            
            fig4.tight_layout()
            figures['differences_plot'] = fig_to_svg(fig4)
        except Exception as e:
            figures['differences_plot_error'] = str(e)
        
        # Figure 5: Power analysis curve
        try:
            if 'power_analysis' in additional_stats and 'error' not in additional_stats['power_analysis']:
                fig5, ax5 = plt.subplots(figsize=(10, 6))
                
                # Create range of sample sizes
                sample_sizes = np.arange(3, max(100, n*2), 1)
                
                # Calculate power for each sample size
                powers = [power_analysis.power(effect_size=abs(cohens_d), nobs=n_sample, 
                                             alpha=alpha, alternative=alternative) 
                          for n_sample in sample_sizes]
                
                # Plot power curve with pastel color
                ax5.plot(sample_sizes, powers, color=PASTEL_COLORS[4], linewidth=2)
                
                # Add horizontal lines at common power thresholds
                ax5.axhline(y=0.8, color='red', linestyle='--', label='0.8 (conventional minimum)')
                ax5.axhline(y=0.9, color='orange', linestyle='--', label='0.9')
                ax5.axhline(y=0.95, color='green', linestyle='--', label='0.95')
                
                # Add vertical line at current sample size
                ax5.axvline(x=n, color='purple', linestyle='-', 
                           label=f'Current n = {n} (power = {additional_stats["power_analysis"]["observed_power"]:.3f})')
                
                # Add labels and title
                ax5.set_xlabel('Sample Size (n)')
                ax5.set_ylabel('Statistical Power')
                ax5.set_title(f'Power Analysis for Effect Size d = {abs(cohens_d):.3f}')
                ax5.legend(loc='lower right')
                ax5.grid(True, linestyle='--', alpha=0.7)
                
                # Set y-axis limits
                ax5.set_ylim(0, 1.05)
                
                # Add text with key sample size recommendations
                text = "Recommended sample sizes:\n"
                text += f"For 80% power: n = {additional_stats['power_analysis']['sample_size_for_80_power']}\n"
                text += f"For 90% power: n = {additional_stats['power_analysis']['sample_size_for_90_power']}\n"
                text += f"For 95% power: n = {additional_stats['power_analysis']['sample_size_for_95_power']}"
                
                ax5.text(0.02, 0.02, text, transform=ax5.transAxes, 
                        verticalalignment='bottom', bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
                
                fig5.tight_layout()
                figures['power_analysis'] = fig_to_svg(fig5)
        except Exception as e:
            figures['power_analysis_error'] = str(e)
        
        # Figure 6: Bootstrap distribution
        try:
            if 'bootstrap' in additional_stats and 'error' not in additional_stats['bootstrap']:
                fig6, ax6 = plt.subplots(figsize=(10, 6))
                
                # Plot bootstrap distribution with pastel color
                sns.histplot(bootstrap_samples, kde=True, ax=ax6, color=PASTEL_COLORS[5], edgecolor='black')
                
                # Add vertical line for observed mean difference
                ax6.axvline(x=mean_diff, color='red', linestyle='-', linewidth=2, 
                           label=f'Observed Mean Difference = {mean_diff:.3f}')
                
                # Add vertical lines for bootstrap CI
                ax6.axvline(x=boot_ci_lower, color='green', linestyle='--', linewidth=2, 
                           label=f'{(1-alpha)*100:.0f}% CI Lower = {boot_ci_lower:.3f}')
                ax6.axvline(x=boot_ci_upper, color='green', linestyle='--', linewidth=2, 
                           label=f'{(1-alpha)*100:.0f}% CI Upper = {boot_ci_upper:.3f}')
                
                # Add vertical line at zero
                ax6.axvline(x=0, color='black', linestyle='--', linewidth=1, label='No Difference')
                
                # Add labels and title
                ax6.set_xlabel('Mean Difference (Group 1 - Group 2)')
                ax6.set_ylabel('Frequency')
                ax6.set_title('Bootstrap Distribution of Mean Difference')
                ax6.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
                
                # Add statistical results
                boot_text = f"Bootstrap SE = {additional_stats['bootstrap']['std_error']:.4f}"
                boot_text += f"\n{(1-alpha)*100:.0f}% CI: [{boot_ci_lower:.3f}, {boot_ci_upper:.3f}]"
                ax6.text(0.02, 0.98, boot_text, transform=ax6.transAxes, 
                        verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
                
                fig6.tight_layout()
                figures['bootstrap_distribution'] = fig_to_svg(fig6)
        except Exception as e:
            figures['bootstrap_distribution_error'] = str(e)
        
        return {
            'test': 'Paired T-Test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': significant,
            'df': int(df),
            'mean_difference': float(mean_diff),
            'std_error': float(se_diff),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'cohens_d': float(cohens_d),
            'hedges_g': float(hedges_g),
            'effect_magnitude': effect_magnitude,
            'n_pairs': int(n),
            'assumptions': assumptions,
            'interpretation': interpretation,
            'additional_statistics': additional_stats,
            'figures': figures
        }
    except Exception as e:
        return {
            'test': 'Paired T-Test',
            'statistic': None,
            'p_value': None,
            'significant': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }