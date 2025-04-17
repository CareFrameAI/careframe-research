import traceback
import pandas as pd
from scipy import stats
import numpy as np
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from lifelines import KaplanMeierFitter, CoxPHFitter
from study_model.study_model import StatisticalTest
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
from data.assumptions.tests import (
    AssumptionResult,
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
    SampleSizeTest,
    MonotonicityTest
)
from data.selection.detailed_tests.formatting import fig_to_svg, PASTEL_COLORS, PASTEL_CMAP

def wilcoxon_signed_rank_test(group1: pd.Series, group2: pd.Series, alpha: float, alternative: str = 'two-sided') -> Dict[str, Any]:
    """Performs a Wilcoxon signed-rank test with comprehensive statistics and assumption checks."""
    try:
        # Drop missing values (only keep pairs where both values are present)
        valid_pairs = pd.DataFrame({'group1': group1, 'group2': group2}).dropna()
        
        if len(valid_pairs) < 2:
            return {
                'test': 'Wilcoxon Signed-Rank Test',
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
        
        # Remove zero differences (they are excluded from the Wilcoxon test)
        non_zero_diff = differences[differences != 0]
        n_zero_diff = len(differences) - len(non_zero_diff)
        
        if len(non_zero_diff) < 2:
            return {
                'test': 'Wilcoxon Signed-Rank Test',
                'statistic': None,
                'p_value': None,
                'significant': False,
                'reason': 'Less than two non-zero differences available for analysis.'
            }
        
        # Calculate basic statistics
        n = len(non_zero_diff)
        median_diff = float(non_zero_diff.median())
        
        # Perform the Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxon(x=group1_clean, y=group2_clean, alternative=alternative)
        
        # Determine significance based on alpha and alternative
        if alternative == 'two-sided':
            significant = p_value < alpha
        else:
            significant = p_value < alpha
            
        # Calculate effect size (r = Z / sqrt(N))
        # For Wilcoxon, we need to convert the statistic to a Z-score
        # Z = (W - 0.5) / sqrt(n(n+1)(2n+1)/6)
        # where W is the test statistic and n is the number of pairs
        if n > 0:
            w_mean = n * (n + 1) / 4
            w_std = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
            z_score = (statistic - w_mean) / w_std
            effect_size_r = abs(z_score) / np.sqrt(n)
        else:
            z_score = 0
            effect_size_r = 0
            
        # Determine magnitude of effect size
        if effect_size_r < 0.1:
            effect_magnitude = "Negligible"
        elif effect_size_r < 0.3:
            effect_magnitude = "Small"
        elif effect_size_r < 0.5:
            effect_magnitude = "Medium"
        else:
            effect_magnitude = "Large"
            
        # Test assumptions
        assumptions = {}
        
        # 1. Normality of differences
        try:
            # Use the correct NormalityTest from format.py
            normality_test = NormalityTest()
            normality_result = normality_test.run_test(data=non_zero_diff)
            assumptions['normality_differences'] = normality_result
        except Exception as e:
            assumptions['normality_differences'] = {
                'error': f'Could not test normality: {str(e)}'
            }
        
        # 2. Sample size check
        try:
            sample_size_test = SampleSizeTest()
            # For Wilcoxon, minimum recommended is typically 5-10 pairs
            sample_size_result = sample_size_test.run_test(data=non_zero_diff, min_recommended=10)
            assumptions['sample_size'] = sample_size_result
        except Exception as e:
            assumptions['sample_size'] = {
                'error': f'Could not check sample size: {str(e)}'
            }
        
        # 3. Outliers check
        try:
            outlier_test = OutlierTest()
            outlier_result = outlier_test.run_test(data=non_zero_diff)
            assumptions['outliers'] = outlier_result
        except Exception as e:
            assumptions['outliers'] = {
                'error': f'Could not check for outliers: {str(e)}'
            }
        
        # 4. Symmetry of differences (using a different approach instead of custom implementation)
        try:
            # We can't directly use a test from format.py for symmetry
            # But we can check skewness and formalize the result
            skewness = float(stats.skew(non_zero_diff))
            _, p_skew = stats.normaltest(non_zero_diff)
            
            if abs(skewness) < 0.5:
                symmetry_result = AssumptionResult.PASSED
                symmetry_message = "The distribution of differences appears to be symmetric."
            elif abs(skewness) < 1.0:
                symmetry_result = AssumptionResult.WARNING
                symmetry_message = f"The distribution of differences shows some asymmetry (skewness = {skewness:.3f})."
            else:
                symmetry_result = AssumptionResult.FAILED
                symmetry_message = f"The distribution of differences is asymmetric (skewness = {skewness:.3f})."
                
            assumptions['symmetry'] = {
                'result': symmetry_result,
                'message': symmetry_message,
                'skewness': skewness,
                'p_value': float(p_skew),
                'details': "The Wilcoxon signed-rank test assumes that the differences are symmetrically distributed around the median."
            }
        except Exception as e:
            assumptions['symmetry'] = {
                'error': f'Could not test symmetry: {str(e)}'
            }
        
        # 5. Independence check (for paired data)
        try:
            independence_test = IndependenceTest()
            # Apply to the differences
            independence_result = independence_test.run_test(data=non_zero_diff)
            assumptions['independence'] = independence_result
        except Exception as e:
            assumptions['independence'] = {
                'error': f'Could not test independence: {str(e)}'
            }
        
        # 6. Test for Group 1
        try:
            normality_test = NormalityTest()
            normality_result = normality_test.run_test(data=group1_clean)
            assumptions['normality_group1'] = normality_result
        except Exception as e:
            assumptions['normality_group1'] = {
                'error': f'Could not test normality for group 1: {str(e)}'
            }
        
        # 7. Test for Group 2
        try:
            normality_test = NormalityTest()
            normality_result = normality_test.run_test(data=group2_clean)
            assumptions['normality_group2'] = normality_result
        except Exception as e:
            assumptions['normality_group2'] = {
                'error': f'Could not test normality for group 2: {str(e)}'
            }
        
        # 8. Check for outliers in original groups
        try:
            outlier_test = OutlierTest()
            outlier_result = outlier_test.run_test(data=group1_clean)
            assumptions['outliers_group1'] = outlier_result
        except Exception as e:
            assumptions['outliers_group1'] = {
                'error': f'Could not check for outliers in group 1: {str(e)}'
            }
        
        try:
            outlier_test = OutlierTest()
            outlier_result = outlier_test.run_test(data=group2_clean)
            assumptions['outliers_group2'] = outlier_result
        except Exception as e:
            assumptions['outliers_group2'] = {
                'error': f'Could not check for outliers in group 2: {str(e)}'
            }
        
        # 9. Monotonicity between paired groups
        try:
            monotonicity_test = MonotonicityTest()
            monotonicity_result = monotonicity_test.run_test(x=group1_clean, y=group2_clean)
            assumptions['monotonicity'] = monotonicity_result
        except Exception as e:
            assumptions['monotonicity'] = {
                'error': f'Could not test monotonicity between groups: {str(e)}'
            }
        
        # Create interpretation
        interpretation = f"Wilcoxon signed-rank test comparing two related groups.\n\n"
        
        # Basic results
        interpretation += f"W = {statistic:.3f}, p = {p_value:.5f}\n"
        interpretation += f"Median difference = {median_diff:.3f}\n"
        if n_zero_diff > 0:
            interpretation += f"Note: {n_zero_diff} pair(s) with zero differences were excluded from the analysis.\n"
        interpretation += f"Effect size: r = {effect_size_r:.3f} ({effect_magnitude.lower()} effect)\n\n"
        
        # Group statistics
        interpretation += "Group statistics:\n"
        interpretation += f"- Group 1: median = {float(group1_clean.median()):.3f}, mean = {float(group1_clean.mean()):.3f}\n"
        interpretation += f"- Group 2: median = {float(group2_clean.median()):.3f}, mean = {float(group2_clean.mean()):.3f}\n"
        interpretation += f"- Number of pairs: {len(differences)} (with {n} non-zero differences)\n\n"
        
        # Assumptions section
        interpretation += "Assumption checks:\n"
        
        # 1. Symmetry assumption
        if 'symmetry' in assumptions and 'result' in assumptions['symmetry']:
            result = assumptions['symmetry']['result']
            message = assumptions['symmetry']['message']
            skewness = assumptions['symmetry'].get('skewness', 'unknown')
            
            interpretation += f"- Symmetry of differences: {message}\n"
            if isinstance(skewness, (int, float)):
                interpretation += f"  (skewness = {skewness:.3f})\n"
            
            if result == AssumptionResult.FAILED:
                interpretation += "  NOTE: The Wilcoxon test assumes symmetric distribution of differences. Results should be interpreted with caution.\n"
        
        # 2. Sample size
        if 'sample_size' in assumptions and 'result' in assumptions['sample_size']:
            result = assumptions['sample_size']['result']
            sample_size = assumptions['sample_size'].get('sample_size', n)
            min_required = assumptions['sample_size'].get('minimum_required', 10)
            
            if result == AssumptionResult.PASSED:
                interpretation += f"- Sample size: Adequate ({sample_size} pairs, minimum recommended: {min_required}).\n"
            elif result == AssumptionResult.WARNING:
                interpretation += f"- Sample size: Borderline ({sample_size} pairs, minimum recommended: {min_required}). Results should be interpreted with caution.\n"
            elif result == AssumptionResult.FAILED:
                interpretation += f"- Sample size: Insufficient ({sample_size} pairs, minimum recommended: {min_required}). Results may not be reliable.\n"
        
        # 3. Outliers in differences
        if 'outliers' in assumptions and 'result' in assumptions['outliers']:
            result = assumptions['outliers']['result']
            outliers = assumptions['outliers'].get('outliers', [])
            
            if result == AssumptionResult.PASSED:
                interpretation += f"- Outliers: No significant outliers detected in the differences.\n"
            elif result == AssumptionResult.WARNING:
                interpretation += f"- Outliers: Some potential outliers detected ({len(outliers)} found). These may influence the results.\n"
            elif result == AssumptionResult.FAILED:
                interpretation += f"- Outliers: Significant outliers detected ({len(outliers)} found). The Wilcoxon test may not be appropriate.\n"
        
        # 4. Independence
        if 'independence' in assumptions and 'result' in assumptions['independence']:
            result = assumptions['independence']['result']
            message = assumptions['independence'].get('message', '')
            
            if result == AssumptionResult.PASSED:
                interpretation += f"- Independence: The differences appear to be independent.\n"
            elif result == AssumptionResult.WARNING:
                interpretation += f"- Independence: Possible dependency detected in the differences. {message}\n"
            elif result == AssumptionResult.FAILED:
                interpretation += f"- Independence: The differences show dependency. {message} The Wilcoxon test assumes independent observations.\n"
        
        # 5. Normality (differences) - not strictly required but useful for comparison
        if 'normality_differences' in assumptions and 'result' in assumptions['normality_differences']:
            result = assumptions['normality_differences']['result']
            test_used = assumptions['normality_differences'].get('test_used', '')
            p_value_norm = assumptions['normality_differences'].get('p_value', None)
            
            if result == AssumptionResult.PASSED:
                interpretation += f"- Normality of differences: The differences appear normally distributed"
            elif result == AssumptionResult.WARNING:
                interpretation += f"- Normality of differences: The differences show some deviation from normality"
            elif result == AssumptionResult.FAILED:
                interpretation += f"- Normality of differences: The differences are not normally distributed"
            
            if p_value_norm is not None:
                interpretation += f" (p = {p_value_norm:.3f}).\n"
            else:
                interpretation += ".\n"
            
            if result == AssumptionResult.PASSED:
                interpretation += "  NOTE: With normally distributed differences, a paired t-test might be more appropriate.\n"
        
        # 6. Group normality checks (optional but informative)
        group_normality = []
        if 'normality_group1' in assumptions and 'result' in assumptions['normality_group1']:
            result = assumptions['normality_group1']['result']
            if result != AssumptionResult.PASSED:
                group_normality.append("Group 1")
        
        if 'normality_group2' in assumptions and 'result' in assumptions['normality_group2']:
            result = assumptions['normality_group2']['result']
            if result != AssumptionResult.PASSED:
                group_normality.append("Group 2")
        
        if group_normality:
            interpretation += f"- Non-normality detected in: {', '.join(group_normality)}. "
            interpretation += "The Wilcoxon test is appropriate for non-normal data.\n"
        
        # 7. Monotonicity (if available)
        if 'monotonicity' in assumptions and 'result' in assumptions['monotonicity']:
            result = assumptions['monotonicity']['result']
            p_value_mono = assumptions['monotonicity'].get('p_value', None)
            
            if result == AssumptionResult.PASSED:
                interpretation += f"- Monotonic relationship: There appears to be a monotonic relationship between the groups"
            elif result == AssumptionResult.WARNING:
                interpretation += f"- Monotonic relationship: The relationship between groups may not be strictly monotonic"
            elif result == AssumptionResult.FAILED:
                interpretation += f"- Monotonic relationship: No monotonic relationship detected between groups"
            
            if p_value_mono is not None:
                interpretation += f" (p = {p_value_mono:.3f}).\n"
            else:
                interpretation += ".\n"
        
        # Summary of assumption checks
        violated_assumptions = []
        for key, value in assumptions.items():
            if 'result' in value and value['result'] == AssumptionResult.FAILED:
                violated_assumptions.append(key)
        
        if not violated_assumptions:
            interpretation += "\nAll key assumptions for the Wilcoxon signed-rank test appear to be satisfied.\n"
        else:
            formatted_violations = [violation.replace('_', ' ') for violation in violated_assumptions]
            interpretation += f"\nCAUTION: The following assumptions may be violated: {', '.join(formatted_violations)}.\n"
            interpretation += "This may affect the validity of the test results.\n"
        
        # Conclusion
        interpretation += f"\nConclusion: There is {'a statistically significant' if significant else 'no statistically significant'} "
        
        if alternative == 'two-sided':
            interpretation += f"difference between the paired groups (p = {p_value:.5f}). "
        elif alternative == 'less':
            interpretation += f"evidence that values in Group 1 are less than values in Group 2 (p = {p_value:.5f}). "
        else:  # 'greater'
            interpretation += f"evidence that values in Group 1 are greater than values in Group 2 (p = {p_value:.5f}). "
        
        if significant:
            interpretation += f"The {effect_magnitude.lower()} effect size (r = {effect_size_r:.3f}) suggests that "
            
            if median_diff > 0:
                interpretation += f"Group 1 values tend to be higher than Group 2 values."
            else:
                interpretation += f"Group 2 values tend to be higher than Group 1 values."
        
        # Calculate additional statistics
        additional_stats = {}
        
        # 1. Comparison with parametric test (Paired t-test)
        try:
            t_stat, t_p = stats.ttest_rel(a=group1_clean, b=group2_clean, alternative=alternative)
            t_sig = t_p < alpha
            
            # Calculate effect size for t-test (Cohen's d)
            t_mean_diff = float(differences.mean())
            t_std_diff = float(differences.std(ddof=1))
            cohens_d = t_mean_diff / t_std_diff
            
            additional_stats['paired_t_test'] = {
                'statistic': float(t_stat),
                'p_value': float(t_p),
                'significant': t_sig,
                'cohens_d': float(cohens_d),
                'agreement_with_wilcoxon': t_sig == significant
            }
        except Exception as e:
            additional_stats['paired_t_test'] = {
                'error': str(e)
            }
        
        # 2. Sign test (simpler non-parametric alternative)
        try:
            # Count positive and negative differences
            n_pos = sum(differences > 0)
            n_neg = sum(differences < 0)
            
            # Perform sign test
            sign_stat, sign_p = stats.binom_test(x=min(n_pos, n_neg), n=n_pos + n_neg, p=0.5, alternative='two-sided')
            sign_sig = sign_p < alpha
            
            additional_stats['sign_test'] = {
                'statistic': float(sign_stat),
                'p_value': float(sign_p),
                'significant': sign_sig,
                'positive_differences': int(n_pos),
                'negative_differences': int(n_neg),
                'agreement_with_wilcoxon': sign_sig == significant
            }
        except Exception as e:
            additional_stats['sign_test'] = {
                'error': str(e)
            }
        
        # 3. Calculate Hodges-Lehmann estimator (confidence interval for median difference)
        try:
            # Compute all pairwise averages of differences
            diff_array = non_zero_diff.to_numpy()
            n_diff = len(diff_array)
            
            # For small samples, compute all pairwise averages
            if n_diff <= 50:
                all_pairs = []
                for i in range(n_diff):
                    for j in range(i, n_diff):
                        all_pairs.append((diff_array[i] + diff_array[j]) / 2)
                
                hodges_lehmann = np.median(all_pairs)
            else:
                # For larger samples, just use the median of the differences
                hodges_lehmann = np.median(diff_array)
            
            # Calculate confidence interval based on signed-rank distribution
            # This is an approximation using the normal approximation
            alpha_level = alpha
            z_critical = stats.norm.ppf(1 - alpha_level/2)
            se = np.sqrt(n * (n+1) * (2*n+1) / 24)
            margin = z_critical * se
            
            # Calculate rank of critical values
            critical_rank = int(np.ceil((n * (n+1) / 4) - margin))
            critical_rank = max(0, min(critical_rank, (n * (n+1)) // 2))
            
            # Sort differences
            sorted_diffs = np.sort(diff_array)
            
            # Select values at critical ranks
            if critical_rank == 0:
                lower_bound = sorted_diffs[0]
                upper_bound = sorted_diffs[-1]
            else:
                lower_bound = sorted_diffs[critical_rank-1]
                upper_bound = sorted_diffs[-(critical_rank)]
            
            additional_stats['hodges_lehmann'] = {
                'estimator': float(hodges_lehmann),
                'ci_lower': float(lower_bound),
                'ci_upper': float(upper_bound),
                'confidence_level': float(1 - alpha)
            }
        except Exception as e:
            additional_stats['hodges_lehmann'] = {
                'error': str(e)
            }
        
        # 4. Bootstrap confidence interval for median difference
        try:
            n_bootstrap = 2000
            bootstrap_medians = []
            
            # Generate bootstrap samples
            for _ in range(n_bootstrap):
                # Sample with replacement
                indices = np.random.choice(range(len(differences)), size=len(differences), replace=True)
                bootstrap_diff = differences.iloc[indices]
                bootstrap_medians.append(np.median(bootstrap_diff))
                
            # Calculate bootstrap CI
            boot_ci_lower = np.percentile(bootstrap_medians, 100 * alpha/2)
            boot_ci_upper = np.percentile(bootstrap_medians, 100 * (1 - alpha/2))
            
            additional_stats['bootstrap'] = {
                'n_samples': n_bootstrap,
                'ci_lower': float(boot_ci_lower),
                'ci_upper': float(boot_ci_upper),
                'median': float(np.median(bootstrap_medians))
            }
        except Exception as e:
            additional_stats['bootstrap'] = {
                'error': str(e)
            }
        
        # 5. Calculate ranks and signed ranks for visualization
        try:
            # Get absolute differences
            abs_diffs = np.abs(non_zero_diff.values)
            
            # Create a dataframe with differences and their ranks
            diff_df = pd.DataFrame({
                'difference': non_zero_diff.values,
                'abs_difference': abs_diffs
            })
            
            # Rank the absolute differences
            diff_df['rank'] = diff_df['abs_difference'].rank()
            
            # Calculate signed rank
            diff_df['signed_rank'] = diff_df['rank'] * np.sign(diff_df['difference'])
            
            # Count positive and negative ranks
            pos_rank_sum = diff_df.loc[diff_df['difference'] > 0, 'rank'].sum()
            neg_rank_sum = diff_df.loc[diff_df['difference'] < 0, 'rank'].sum()
            
            additional_stats['rank_analysis'] = {
                'positive_rank_sum': float(pos_rank_sum),
                'negative_rank_sum': float(neg_rank_sum),
                'min_rank_sum': float(min(pos_rank_sum, neg_rank_sum)),
                'ranks': diff_df['rank'].tolist(),
                'signed_ranks': diff_df['signed_rank'].tolist(),
                'differences': diff_df['difference'].tolist()
            }
        except Exception as e:
            additional_stats['rank_analysis'] = {
                'error': str(e)
            }
        
        # 6. Descriptive statistics of differences
        try:
            from scipy.stats import skew, kurtosis
            
            # Calculate skewness and kurtosis of differences
            skewness = skew(non_zero_diff)
            kurt = kurtosis(non_zero_diff)  # Excess kurtosis
            
            additional_stats['difference_descriptives'] = {
                'min': float(non_zero_diff.min()),
                'q1': float(np.percentile(non_zero_diff, 25)),
                'median': float(np.median(non_zero_diff)),
                'q3': float(np.percentile(non_zero_diff, 75)),
                'max': float(non_zero_diff.max()),
                'mean': float(np.mean(non_zero_diff)),
                'std': float(np.std(non_zero_diff, ddof=1)),
                'skewness': float(skewness),
                'kurtosis': float(kurt),
                'iqr': float(np.percentile(non_zero_diff, 75) - np.percentile(non_zero_diff, 25))
            }
        except Exception as e:
            additional_stats['difference_descriptives'] = {
                'error': str(e)
            }
        
        # Generate figures
        figures = {}
        
        # Figure 1: Distribution of differences with median and symmetry
        try:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            
            # Plot histogram of differences
            sns.histplot(non_zero_diff, kde=True, ax=ax1, color='skyblue', edgecolor='black')
            
            # Add vertical line at median
            ax1.axvline(x=median_diff, color='red', linestyle='-', linewidth=2, 
                      label=f'Median = {median_diff:.3f}')
            
            # Add vertical line at zero
            ax1.axvline(x=0, color='black', linestyle='--', linewidth=1, label='Zero Difference')
            
            # Add mirror of the left side of distribution to visualize symmetry
            if 'symmetry' in assumptions and 'result' in assumptions['symmetry']:
                # Only try to visualize symmetry if we have symmetry data
                try:
                    # Get bin edges and frequencies
                    hist, bin_edges = np.histogram(non_zero_diff, bins='auto')
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    # Find the median bin
                    median_bin_idx = np.searchsorted(bin_centers, median_diff)
                    
                    # Create mirrored distribution for left side
                    left_bins = bin_centers[:median_bin_idx]
                    left_hist = hist[:median_bin_idx]
                    
                    if len(left_bins) > 0:
                        # Create mirror points
                        mirror_bins = 2 * median_diff - left_bins
                        mirror_hist = left_hist
                        
                        # Only include mirror points within the range of original data
                        valid_idx = (mirror_bins <= non_zero_diff.max())
                        mirror_bins = mirror_bins[valid_idx]
                        mirror_hist = mirror_hist[valid_idx]
                        
                        # Plot mirrored distribution with lower opacity
                        if len(mirror_bins) > 0:
                            ax1.plot(mirror_bins, mirror_hist, 'r--', linewidth=2, alpha=0.6, 
                                   label='Mirror (Perfect Symmetry)')
                except:
                    # If symmetry visualization fails, just skip it
                    pass
            
            # Add labels and title
            ax1.set_xlabel('Difference (Group 1 - Group 2)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Distribution of Paired Differences')
            ax1.legend()
            
            # Add symmetry information
            if 'symmetry' in assumptions and 'result' in assumptions['symmetry']:
                symmetry_result = assumptions['symmetry']
                symmetry_text = f"Skewness = {symmetry_result['skewness']:.3f}"
                symmetry_text += f"\nSymmetry {'appears adequate' if symmetry_result['result'] == 'passed' else 'may be violated'}"
                ax1.text(0.05, 0.95, symmetry_text, transform=ax1.transAxes, 
                        verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
            
            fig1.tight_layout()
            figures['difference_histogram'] = fig_to_svg(fig1)
        except Exception as e:
            figures['difference_histogram_error'] = str(e)
        
        # Figure 2: Box plots with individual data points and connecting lines
        try:
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            
            # Create DataFrame for seaborn
            plot_data = pd.DataFrame({
                'Group 1': group1_clean,
                'Group 2': group2_clean
            })
            plot_data_melted = pd.melt(plot_data, var_name='Group', value_name='Value')
            
            # Draw boxplot with jittered data points
            sns.boxplot(x='Group', y='Value', data=plot_data_melted, ax=ax2, width=0.5)
            sns.stripplot(x='Group', y='Value', data=plot_data_melted, ax=ax2, 
                        color='black', alpha=0.5, jitter=True, size=4)
            
            # Add median markers
            medians = [group1_clean.median(), group2_clean.median()]
            ax2.plot(['Group 1', 'Group 2'], medians, 'ro', markersize=10, label='Median')
            
            # Connect paired observations with lines
            for i in range(len(group1_clean)):
                ax2.plot(['Group 1', 'Group 2'], [group1_clean.iloc[i], group2_clean.iloc[i]], 
                       'gray', alpha=0.2, linestyle='-', linewidth=1)
            
            # Add statistical results
            stats_text = f"Wilcoxon W = {statistic:.3f}, p = {p_value:.4f}"
            if significant:
                stats_text += " *"
            if n_zero_diff > 0:
                stats_text += f"\n{n_zero_diff} zero difference(s) excluded"
            ax2.text(0.5, 0.01, stats_text, transform=ax2.transAxes, 
                    horizontalalignment='center', bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
            
            # Add labels and title
            ax2.set_title('Paired Observations')
            ax2.set_ylabel('Value')
            ax2.legend()
            
            fig2.tight_layout()
            figures['paired_boxplot'] = fig_to_svg(fig2)
        except Exception as e:
            figures['paired_boxplot_error'] = str(e)
        
        # Figure 3: Rank visualization
        if 'rank_analysis' in additional_stats and 'error' not in additional_stats['rank_analysis']:
            try:
                fig3, ax3 = plt.subplots(figsize=(12, 6))
                
                # Extract data from rank analysis
                ranks = np.array(additional_stats['rank_analysis']['ranks'])
                signed_ranks = np.array(additional_stats['rank_analysis']['signed_ranks'])
                differences = np.array(additional_stats['rank_analysis']['differences'])
                
                # Create indices for plotting
                indices = np.arange(len(ranks))
                
                # Sort by difference for better visualization
                sort_idx = np.argsort(differences)
                ranks = ranks[sort_idx]
                signed_ranks = signed_ranks[sort_idx]
                differences = differences[sort_idx]
                
                # Plot signed ranks
                ax3.bar(indices, signed_ranks, color=['red' if sr < 0 else 'blue' for sr in signed_ranks])
                
                # Add horizontal line at zero
                ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
                
                # Add labels for positive and negative sums
                pos_sum = additional_stats['rank_analysis']['positive_rank_sum']
                neg_sum = additional_stats['rank_analysis']['negative_rank_sum']
                
                # Add annotations for rank sums
                ax3.text(len(indices) * 0.05, max(signed_ranks) * 0.8, 
                        f"Positive Rank Sum = {pos_sum:.1f}", 
                        color='blue', fontweight='bold')
                
                ax3.text(len(indices) * 0.05, min(signed_ranks) * 0.8, 
                        f"Negative Rank Sum = {neg_sum:.1f}", 
                        color='red', fontweight='bold')
                
                # Add test statistic (W is the minimum of positive and negative rank sums)
                ax3.text(len(indices) * 0.7, max(signed_ranks) * 0.8, 
                        f"W = {min(pos_sum, neg_sum):.1f}", 
                        fontweight='bold', 
                        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
                
                # Add labels and title
                ax3.set_xlabel('Observation (sorted by difference)')
                ax3.set_ylabel('Signed Rank')
                ax3.set_title('Signed Ranks of Differences')
                
                # Set x-ticks with differences
                n_ticks = min(20, len(indices))  # Limit number of ticks for readability
                tick_step = len(indices) // n_ticks
                tick_idx = indices[::tick_step]
                tick_differences = [f"{differences[i]:.1f}" for i in tick_idx]
                ax3.set_xticks(tick_idx)
                ax3.set_xticklabels(tick_differences, rotation=45)
                
                # Add secondary x-axis with ranks
                ax_secondary = ax3.twiny()
                ax_secondary.set_xticks(tick_idx)
                ax_secondary.set_xticklabels([f"{ranks[i]:.0f}" for i in tick_idx], rotation=45)
                ax_secondary.set_xlabel('Rank')
                
                fig3.tight_layout()
                figures['signed_ranks'] = fig_to_svg(fig3)
            except Exception as e:
                figures['signed_ranks_error'] = str(e)
        
        # Figure 4: Comparison of parametric vs non-parametric tests
        if 'paired_t_test' in additional_stats and 'error' not in additional_stats['paired_t_test'] and \
           'sign_test' in additional_stats and 'error' not in additional_stats['sign_test']:
            try:
                fig4, ax4 = plt.subplots(figsize=(10, 6))
                
                # Extract p-values
                tests = ['Wilcoxon', 'Paired t-test', 'Sign test']
                p_values = [
                    p_value, 
                    additional_stats['paired_t_test']['p_value'],
                    additional_stats['sign_test']['p_value']
                ]
                
                # Create bar plot of p-values
                bars = ax4.bar(tests, p_values, color=['blue', 'green', 'orange'])
                
                # Add horizontal line at alpha level
                ax4.axhline(y=alpha, color='red', linestyle='--', label=f'Alpha = {alpha}')
                
                # Add significance markers
                for i, p in enumerate(p_values):
                    if p < alpha:
                        ax4.text(i, p + 0.01, '*', ha='center', va='bottom', fontsize=20)
                
                # Add labels and title
                ax4.set_ylabel('p-value')
                ax4.set_title('Comparison of Test Results')
                ax4.legend()
                
                # Set y-axis limit
                ax4.set_ylim(0, max(p_values) * 1.2 + 0.05)
                
                # Add test statistics
                test_stats = [
                    f"W = {statistic:.2f}",
                    f"t = {additional_stats['paired_t_test']['statistic']:.2f}",
                    f"p = {additional_stats['sign_test']['statistic']:.2f}"
                ]
                
                for i, (bar, stat) in enumerate(zip(bars, test_stats)):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                            stat, ha='center', va='bottom')
                
                fig4.tight_layout()
                figures['test_comparison'] = fig_to_svg(fig4)
            except Exception as e:
                figures['test_comparison_error'] = str(e)
        
        # Figure 5: Bootstrap distribution of median difference
        if 'bootstrap' in additional_stats and 'error' not in additional_stats['bootstrap']:
            try:
                fig5, ax5 = plt.subplots(figsize=(10, 6))
                
                # Extract bootstrap data
                boot_ci_lower = additional_stats['bootstrap']['ci_lower']
                boot_ci_upper = additional_stats['bootstrap']['ci_upper']
                bootstrap_medians = np.linspace(boot_ci_lower - (boot_ci_upper - boot_ci_lower) * 0.5,
                                             boot_ci_upper + (boot_ci_upper - boot_ci_lower) * 0.5,
                                             2000)
                
                # Simulate bootstrap distribution
                bootstrap_density = stats.norm.pdf(bootstrap_medians, 
                                                median_diff, 
                                                (boot_ci_upper - boot_ci_lower) / (2 * stats.norm.ppf(1 - alpha/2)))
                
                # Plot bootstrap distribution
                ax5.plot(bootstrap_medians, bootstrap_density, 'b-', linewidth=2)
                
                # Add vertical line for observed median difference
                ax5.axvline(x=median_diff, color='red', linestyle='-', linewidth=2, 
                          label=f'Median Difference = {median_diff:.3f}')
                
                # Add vertical lines for bootstrap CI
                ax5.axvline(x=boot_ci_lower, color='green', linestyle='--', linewidth=2, 
                          label=f'{(1-alpha)*100:.0f}% CI Lower = {boot_ci_lower:.3f}')
                ax5.axvline(x=boot_ci_upper, color='green', linestyle='--', linewidth=2, 
                          label=f'{(1-alpha)*100:.0f}% CI Upper = {boot_ci_upper:.3f}')
                
                # Add vertical line at zero
                ax5.axvline(x=0, color='black', linestyle='--', linewidth=1, label='No Difference')
                
                # Add labels and title
                ax5.set_xlabel('Median Difference')
                ax5.set_ylabel('Density')
                ax5.set_title('Bootstrap Distribution of Median Difference')
                ax5.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
                
                fig5.tight_layout()
                figures['bootstrap_distribution'] = fig_to_svg(fig5)
            except Exception as e:
                figures['bootstrap_distribution_error'] = str(e)
        
        # Figure 6: Scatter plot of differences vs. averages (Bland-Altman plot)
        try:
            fig6, ax6 = plt.subplots(figsize=(10, 6))
            
            # Calculate averages
            averages = (group1_clean + group2_clean) / 2
            
            # Create scatter plot
            ax6.scatter(averages, differences, alpha=0.6)
            
            # Add horizontal line at mean difference
            mean_diff = differences.mean()
            ax6.axhline(y=mean_diff, color='red', linestyle='-', linewidth=2,
                      label=f'Mean Difference = {mean_diff:.3f}')
            
            # Add horizontal line at median difference
            ax6.axhline(y=median_diff, color='blue', linestyle='-', linewidth=2,
                      label=f'Median Difference = {median_diff:.3f}')
            
            # Add horizontal line at zero
            ax6.axhline(y=0, color='black', linestyle='--', linewidth=1, label='No Difference')
            
            # Add limits of agreement (mean ± 1.96 SD)
            sd_diff = differences.std()
            upper_loa = mean_diff + 1.96 * sd_diff
            lower_loa = mean_diff - 1.96 * sd_diff
            
            ax6.axhline(y=upper_loa, color='green', linestyle='--', linewidth=1,
                      label=f'Upper LoA = {upper_loa:.3f}')
            ax6.axhline(y=lower_loa, color='green', linestyle='--', linewidth=1,
                      label=f'Lower LoA = {lower_loa:.3f}')
            
            # Add labels and title
            ax6.set_xlabel('Average of Group 1 and Group 2')
            ax6.set_ylabel('Difference (Group 1 - Group 2)')
            ax6.set_title('Difference vs. Average (Bland-Altman Plot)')
            ax6.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
            
            # Add test results
            test_text = f"Wilcoxon W = {statistic:.2f}, p = {p_value:.4f}"
            if significant:
                test_text += " *"
            ax6.text(0.02, 0.98, test_text, transform=ax6.transAxes, 
                    verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
            
            fig6.tight_layout()
            figures['bland_altman'] = fig_to_svg(fig6)
        except Exception as e:
            figures['bland_altman_error'] = str(e)
        
        return {
            'test': 'Wilcoxon Signed-Rank Test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': significant,
            'n_pairs': int(len(differences)),
            'n_non_zero': int(n),
            'n_zero_diff': int(n_zero_diff),
            'median_difference': float(median_diff),
            'effect_size_r': float(effect_size_r),
            'effect_magnitude': effect_magnitude,
            'assumptions': assumptions,
            'interpretation': interpretation,
            'additional_statistics': additional_stats,
            'figures': figures
        }
    except Exception as e:
        return {
            'test': 'Wilcoxon Signed-Rank Test',
            'statistic': None,
            'p_value': None,
            'significant': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }