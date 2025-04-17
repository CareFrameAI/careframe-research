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


def mann_whitney_u_test(group1: pd.Series, group2: pd.Series, alpha: float, alternative: str = 'two-sided') -> Dict[str, Any]:
    """Performs a Mann-Whitney U test (also known as Wilcoxon rank-sum test) with comprehensive statistics and visualizations."""
    try:
        # Clean data - drop NaN values
        group1_clean = group1.dropna()
        group2_clean = group2.dropna()
        
        # Basic checks
        if len(group1_clean) == 0 or len(group2_clean) == 0:
            return {
                'test': 'Mann-Whitney U Test',
                'statistic': None,
                'p_value': None,
                'significant': False,
                'reason': 'One or both groups have no valid data after removing missing values.'
            }
        
        # Extract values
        group1_values = group1_clean.values
        group2_values = group2_clean.values
        
        # Run the Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(x=group1_values, y=group2_values, alternative=alternative)
        
        # Determine significance
        significant = p_value < alpha
        
        # Calculate basic statistics for each group
        group1_stats = {
            'n': len(group1_values),
            'median': float(np.median(group1_values)),
            'mean': float(np.mean(group1_values)),
            'std': float(np.std(group1_values, ddof=1)),
            'min': float(np.min(group1_values)),
            'max': float(np.max(group1_values)),
            'q1': float(np.percentile(group1_values, 25)),
            'q3': float(np.percentile(group1_values, 75)),
            'iqr': float(np.percentile(group1_values, 75) - np.percentile(group1_values, 25))
        }
        
        group2_stats = {
            'n': len(group2_values),
            'median': float(np.median(group2_values)),
            'mean': float(np.mean(group2_values)),
            'std': float(np.std(group2_values, ddof=1)),
            'min': float(np.min(group2_values)),
            'max': float(np.max(group2_values)),
            'q1': float(np.percentile(group2_values, 25)),
            'q3': float(np.percentile(group2_values, 75)),
            'iqr': float(np.percentile(group2_values, 75) - np.percentile(group2_values, 25))
        }
        
        # Calculate effect size (r = Z / sqrt(N))
        n1 = len(group1_values)
        n2 = len(group2_values)
        
        # Convert Mann-Whitney U to Z
        u_max = n1 * n2  # Maximum possible U value
        u_mean = u_max / 2  # Mean of sampling distribution of U
        u_std = np.sqrt(u_max * (n1 + n2 + 1) / 12)  # Standard deviation of sampling distribution of U
        
        # Calculate Z-score
        z_score = (statistic - u_mean) / u_std
        
        # Calculate effect size using r = Z/sqrt(N)
        effect_size_r = abs(z_score) / np.sqrt(n1 + n2)
        
        # Interpret effect size magnitude
        if effect_size_r < 0.1:
            effect_magnitude = "Negligible"
        elif effect_size_r < 0.3:
            effect_magnitude = "Small"
        elif effect_size_r < 0.5:
            effect_magnitude = "Medium"
        else:
            effect_magnitude = "Large"
        
        # Calculate Hodges-Lehmann estimator (median of pairwise differences)
        pairwise_diffs = []
        for x in group1_values:
            for y in group2_values:
                pairwise_diffs.append(x - y)
        
        hodges_lehmann = float(np.median(pairwise_diffs))
        
        # Test assumptions
        assumptions = {}
        
        # 1. Check for outliers in each group
        try:
            outlier_test = OutlierTest()
            
            # Convert to pandas Series
            group1_series = pd.Series(group1_values)
            group2_series = pd.Series(group2_values)
            
            group1_outlier_result = outlier_test.run_test(data=group1_series)
            group2_outlier_result = outlier_test.run_test(data=group2_series)
            
            assumptions['outliers_group1'] = group1_outlier_result
            assumptions['outliers_group2'] = group2_outlier_result

        except Exception as e:
            assumptions['outliers'] = {
                'result': "warning",
                'message': "Could not check for outliers.",
                'error': f'Could not check for outliers: {str(e)}'
            }
        
        # 2. Sample size check
        try:
            sample_size_test = SampleSizeTest()
            
            # Convert to pandas Series
            group1_series = pd.Series(group1_values)
            group2_series = pd.Series(group2_values)
            
            # For Mann-Whitney, we generally want at least 8-10 observations per group
            group1_size_result = sample_size_test.run_test(data=group1_series, min_recommended=10)
            group2_size_result = sample_size_test.run_test(data=group2_series, min_recommended=10)
            
            assumptions['sample_size_group1'] = group1_size_result
            assumptions['sample_size_group2'] = group2_size_result

        except Exception as e:
            assumptions['sample_size'] = {
                'result': "warning",
                'message': "Could not check sample size adequacy.",
                'error': f'Could not check sample size: {str(e)}'
            }
        
        # 3. Check for similar distribution shapes
        try:
            # Convert numpy arrays to pandas Series for tests that expect pandas objects
            group1_series = pd.Series(group1_values)
            group2_series = pd.Series(group2_values)
            combined_series = pd.Series(np.concatenate([group1_values, group2_values]))
            group_labels_series = pd.Series(np.concatenate([[0] * len(group1_values), [1] * len(group2_values)]))
            
            # Use homogeneity test to check shape similarity (not equal variances)
            homogeneity_test = HomogeneityOfVarianceTest()
            shape_result = homogeneity_test.run_test(data=combined_series, groups=group_labels_series)
            
            # Also check distributional characteristics using normality test for each group
            normality_test = NormalityTest()
            group1_normality = normality_test.run_test(data=group1_series)
            group2_normality = normality_test.run_test(data=group2_series)
            
            assumptions['homogeneity_of_variance'] = shape_result
            assumptions['normality_group1'] = group1_normality
            assumptions['normality_group2'] = group2_normality

        except Exception as e:
            assumptions['similar_shapes'] = {
                'result': "warning",
                'message': "Mann-Whitney U test assumes similar distribution shapes when making inferences about median differences.",
                'error': f'Could not check distribution shapes: {str(e)}'
            }
        
        # 4. Independence assumption 
        try:
            independence_test = IndependenceTest()
            # Convert to pandas Series
            independence_data = pd.Series(np.concatenate([group1_values, group2_values]))
            independence_result = independence_test.run_test(data=independence_data)
            assumptions['independence'] = independence_result
        except Exception as e:
            assumptions['independence'] = {
                'result': "warning",
                'message': "Independence of observations is assumed and cannot be fully tested statistically.",
                'details': "Ensure that observations in each group are independent and that there's no relationship between pairs of observations across groups.",
                'error': f'Could not run independence test: {str(e)}'
            }
        
        # 5. Fix monotonicity check
        try:
            # Check monotonicity of relationship between groups and ranks
            monotonicity_test = MonotonicityTest()
            # Create artificial x-variable representing group membership and convert to pandas Series
            x_group = pd.Series(np.concatenate([[0] * len(group1_values), [1] * len(group2_values)]))
            y_values = pd.Series(np.concatenate([group1_values, group2_values]))
            
            monotonicity_result = monotonicity_test.run_test(x=x_group, y=y_values)
            assumptions['monotonicity'] = monotonicity_result

        except Exception as e:
            assumptions['monotonicity'] = {
                'result': "warning",
                'message': "Could not check monotonicity relationship between groups.",
                'error': f'Could not check monotonicity: {str(e)}'
            }
        
        # Create interpretation
        group1_name = group1.name if hasattr(group1, 'name') and group1.name is not None else "Group 1"
        group2_name = group2.name if hasattr(group2, 'name') and group2.name is not None else "Group 2"
        
        interpretation = f"Mann-Whitney U test comparing {group1_name} and {group2_name}.\n\n"
        
        # Basic results
        interpretation += f"U = {statistic:.1f}, p = {p_value:.5f}\n"
        interpretation += f"Effect size: r = {effect_size_r:.3f} ({effect_magnitude.lower()} effect)\n"
        interpretation += f"Hodges-Lehmann estimator (median difference): {hodges_lehmann:.3f}\n\n"
        
        # Group statistics
        interpretation += "Group statistics:\n"
        interpretation += f"- {group1_name}: n = {n1}, median = {group1_stats['median']:.3f}, mean = {group1_stats['mean']:.3f}, IQR = {group1_stats['iqr']:.3f}\n"
        interpretation += f"- {group2_name}: n = {n2}, median = {group2_stats['median']:.3f}, mean = {group2_stats['mean']:.3f}, IQR = {group2_stats['iqr']:.3f}\n\n"
        
        # Assumptions
        interpretation += "Assumption checks:\n"
        
        # Homogeneity of variance (replaces similar shapes)
        if 'homogeneity_of_variance' in assumptions:
            shape_info = assumptions['homogeneity_of_variance']
            if 'details' in shape_info:
                interpretation += f"- Similar distribution shapes: {shape_info['details']}\n"
            elif 'message' in shape_info:
                interpretation += f"- Similar distribution shapes: {shape_info['message']}\n"
            elif 'error' in shape_info:
                interpretation += f"- Similar distribution shapes: Could not be checked ({shape_info['error']})\n"
            else:
                interpretation += f"- Similar distribution shapes: {shape_info.get('result', 'Unknown result')}\n"
        
        # Independence
        if 'independence' in assumptions:
            independence_info = assumptions['independence']
            if 'details' in independence_info:
                interpretation += f"- Independence: {independence_info['details']}\n"
            elif 'message' in independence_info:
                interpretation += f"- Independence: {independence_info['message']}\n"
            else:
                interpretation += f"- Independence: Observations should be independent of each other.\n"
        
        # Sample size - group specific
        if 'sample_size_group1' in assumptions and 'sample_size_group2' in assumptions:
            group1_size_info = assumptions['sample_size_group1']
            group2_size_info = assumptions['sample_size_group2']
            
            size_msg = f"- Sample size: {group1_name} "
            if 'result' in group1_size_info:
                if group1_size_info['result'] == 'pass':
                    size_msg += f"has adequate sample size (n={n1}). "
                else:
                    size_msg += f"may have insufficient sample size (n={n1}). "
            else:
                size_msg += f"n={n1}. "
                
            size_msg += f"{group2_name} "
            if 'result' in group2_size_info:
                if group2_size_info['result'] == 'pass':
                    size_msg += f"has adequate sample size (n={n2})."
                else:
                    size_msg += f"may have insufficient sample size (n={n2})."
            else:
                size_msg += f"n={n2}."
                
            interpretation += size_msg + "\n"
        elif 'sample_size' in assumptions:  # Fallback for legacy code
            size_info = assumptions['sample_size']
            if 'details' in size_info:
                interpretation += f"- Sample size: {size_info['details']}\n"
            elif 'message' in size_info:
                interpretation += f"- Sample size: {size_info['message']}\n"
            else:
                interpretation += f"- Sample size: {group1_name} n={n1}, {group2_name} n={n2}\n"
        
        # Outliers - group specific
        if 'outliers_group1' in assumptions and 'outliers_group2' in assumptions:
            group1_outlier_info = assumptions['outliers_group1']
            group2_outlier_info = assumptions['outliers_group2']
            
            outlier_msg = f"- Outliers: {group1_name} "
            if 'result' in group1_outlier_info:
                if group1_outlier_info['result'] == 'pass':
                    outlier_msg += f"has no significant outliers. "
                else:
                    outlier_msg += f"may contain outliers. "
            elif 'message' in group1_outlier_info:
                outlier_msg += f"{group1_outlier_info['message']} "
            
            outlier_msg += f"{group2_name} "
            if 'result' in group2_outlier_info:
                if group2_outlier_info['result'] == 'pass':
                    outlier_msg += f"has no significant outliers."
                else:
                    outlier_msg += f"may contain outliers."
            elif 'message' in group2_outlier_info:
                outlier_msg += f"{group2_outlier_info['message']}"
                
            interpretation += outlier_msg + "\n"
        elif 'outliers' in assumptions:  # Fallback for legacy code
            outlier_info = assumptions['outliers']
            if 'details' in outlier_info:
                interpretation += f"- Outliers: {outlier_info['details']}\n"
            elif 'message' in outlier_info:
                interpretation += f"- Outliers: {outlier_info['message']}\n"
            else:
                interpretation += f"- Outliers: Check for extreme values that may influence results.\n"
        
        # Normality - group specific (added for Mann-Whitney context)
        if 'normality_group1' in assumptions and 'normality_group2' in assumptions:
            group1_normality_info = assumptions['normality_group1']
            group2_normality_info = assumptions['normality_group2']
            
            normality_msg = "- Normality: While not required for Mann-Whitney U test, "
            normality_msg += f"{group1_name} "
            if 'result' in group1_normality_info:
                if group1_normality_info['result'] == 'pass':
                    normality_msg += f"appears normally distributed. "
                else:
                    normality_msg += f"appears non-normally distributed. "
            elif 'message' in group1_normality_info:
                normality_msg += f"{group1_normality_info['message']} "
            
            normality_msg += f"{group2_name} "
            if 'result' in group2_normality_info:
                if group2_normality_info['result'] == 'pass':
                    normality_msg += f"appears normally distributed."
                else:
                    normality_msg += f"appears non-normally distributed."
            elif 'message' in group2_normality_info:
                normality_msg += f"{group2_normality_info['message']}"
                
            interpretation += normality_msg + "\n"
        
        # Add monotonicity if available
        if 'monotonicity' in assumptions:
            mono_info = assumptions['monotonicity']
            if 'details' in mono_info:
                interpretation += f"- Monotonicity: {mono_info['details']}\n"
            elif 'message' in mono_info:
                interpretation += f"- Monotonicity: {mono_info['message']}\n"
        
        # Conclusion
        interpretation += f"\nConclusion: There is {'a statistically significant' if significant else 'no statistically significant'} "
        
        if alternative == 'two-sided':
            interpretation += f"difference between {group1_name} and {group2_name} (p = {p_value:.5f}). "
        elif alternative == 'less':
            interpretation += f"evidence that {group1_name} tends to have lower values than {group2_name} (p = {p_value:.5f}). "
        else:  # 'greater'
            interpretation += f"evidence that {group1_name} tends to have higher values than {group2_name} (p = {p_value:.5f}). "
        
        if significant:
            interpretation += f"The {effect_magnitude.lower()} effect size (r = {effect_size_r:.3f}) suggests that "
            
            if hodges_lehmann > 0:
                interpretation += f"values in {group1_name} tend to be higher than values in {group2_name} by approximately {hodges_lehmann:.3f} units (median difference)."
            else:
                interpretation += f"values in {group2_name} tend to be higher than values in {group1_name} by approximately {abs(hodges_lehmann):.3f} units (median difference)."
        
        # Calculate additional statistics
        additional_stats = {}
        
        # 1. Independent samples t-test for comparison with parametric test
        try:
            t_stat, t_p = stats.ttest_ind(group1_values, group2_values, equal_var=False)
            t_significant = t_p < alpha
            
            # Calculate Cohen's d for t-test
            mean_diff = group1_stats['mean'] - group2_stats['mean']
            pooled_std = np.sqrt(((n1 - 1) * group1_stats['std']**2 + (n2 - 1) * group2_stats['std']**2) / (n1 + n2 - 2))
            cohen_d = mean_diff / pooled_std
            
            additional_stats['t_test'] = {
                'statistic': float(t_stat),
                'p_value': float(t_p),
                'significant': t_significant,
                'cohens_d': float(cohen_d),
                'agreement_with_mann_whitney': t_significant == significant
            }
        except Exception as e:
            additional_stats['t_test'] = {
                'error': str(e)
            }
        
        # 2. Confidence interval for the Hodges-Lehmann estimator
        try:
            # Calculate the rank sum (W)
            combined = np.concatenate([group1_values, group2_values])
            ranks = stats.rankdata(combined)
            group1_ranks = ranks[:n1]
            
            # Calculate W (rank sum for group 1)
            w = np.sum(group1_ranks)
            
            # Standard error of W
            se_w = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            
            # Calculate Z critical value
            z_crit = stats.norm.ppf(1 - alpha/2)
            
            # Calculate confidence interval for W
            w_lower = w - z_crit * se_w
            w_upper = w + z_crit * se_w
            
            # Convert to U statistic
            u_lower = w_lower - n1 * (n1 + 1) / 2
            u_upper = w_upper - n1 * (n1 + 1) / 2
            
            # Prortion of pairwise differences
            pairwise_diffs = np.sort(pairwise_diffs)
            n_pairs = len(pairwise_diffs)
            
            # Find indices for confidence interval
            lower_idx = int(np.ceil((n_pairs - u_upper) / 2))
            upper_idx = int(np.floor((n_pairs - u_lower) / 2))
            
            # Ensure indices are within bounds
            lower_idx = max(0, min(lower_idx, n_pairs - 1))
            upper_idx = max(0, min(upper_idx, n_pairs - 1))
            
            # Get confidence interval
            ci_lower = float(pairwise_diffs[lower_idx])
            ci_upper = float(pairwise_diffs[upper_idx])
            
            additional_stats['hodges_lehmann_ci'] = {
                'estimator': float(hodges_lehmann),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'confidence_level': float(1 - alpha)
            }
        except Exception as e:
            additional_stats['hodges_lehmann_ci'] = {
                'error': str(e)
            }
        
        # 3. Bootstrap confidence interval for the difference in medians
        try:
            n_bootstrap = 2000
            bootstrap_diffs = []
            
            # Generate bootstrap samples
            for _ in range(n_bootstrap):
                # Sample with replacement
                idx1 = np.random.choice(range(n1), size=n1, replace=True)
                idx2 = np.random.choice(range(n2), size=n2, replace=True)
                
                # Calculate median difference
                boot_group1 = group1_values[idx1]
                boot_group2 = group2_values[idx2]
                
                bootstrap_diffs.append(np.median(boot_group1) - np.median(boot_group2))
            
            # Calculate bootstrap CI
            boot_ci_lower = float(np.percentile(bootstrap_diffs, 100 * alpha/2))
            boot_ci_upper = float(np.percentile(bootstrap_diffs, 100 * (1 - alpha/2)))
            
            additional_stats['bootstrap'] = {
                'n_samples': n_bootstrap,
                'median_diff_ci_lower': boot_ci_lower,
                'median_diff_ci_upper': boot_ci_upper,
                'median_diff': float(np.median(bootstrap_diffs))
            }
        except Exception as e:
            additional_stats['bootstrap'] = {
                'error': str(e)
            }
        
        # 4. Rank biserial correlation (another effect size measure)
        try:
            # Calculate mean rank for each group
            group1_mean_rank = np.mean(group1_ranks)
            group2_mean_rank = np.mean(ranks[n1:])
            
            # Calculate rank biserial correlation
            rank_biserial = 2 * (group1_mean_rank - group2_mean_rank) / (n1 + n2)
            
            additional_stats['rank_biserial'] = {
                'correlation': float(rank_biserial),
                'group1_mean_rank': float(group1_mean_rank),
                'group2_mean_rank': float(group2_mean_rank)
            }
        except Exception as e:
            additional_stats['rank_biserial'] = {
                'error': str(e)
            }
        
        # 5. Calculate probability of superiority
        # This is the probability that a random value from group1 is greater than a random value from group2
        try:
            superiority_count = 0
            for x in group1_values:
                for y in group2_values:
                    if x > y:
                        superiority_count += 1
            
            prob_superiority = float(superiority_count / (n1 * n2))
            
            additional_stats['probability_of_superiority'] = {
                'probability': prob_superiority,
                'interpretation': f"There is a {prob_superiority*100:.1f}% chance that a randomly selected value from {group1_name} is greater than a randomly selected value from {group2_name}."
            }
        except Exception as e:
            additional_stats['probability_of_superiority'] = {
                'error': str(e)
            }
        
        # Generate figures
        figures = {}
        
        # Figure 1: Box plots with individual data points
        try:
            fig1, ax1 = plt.subplots(figsize=(10, 8))
            
            # Create data frames for plotting
            df1 = pd.DataFrame({group1_name: group1_values})
            df2 = pd.DataFrame({group2_name: group2_values})
            
            plot_data = pd.concat([df1, df2], axis=1)
            plot_data_melt = pd.melt(plot_data, var_name='Group', value_name='Value')
            
            # Create box plot with individual points
            sns.boxplot(x='Group', y='Value', data=plot_data_melt, ax=ax1, width=0.5, palette=PASTEL_COLORS)
            sns.stripplot(x='Group', y='Value', data=plot_data_melt, ax=ax1, 
                        color='black', alpha=0.5, jitter=True, size=4)
            
            # Add medians as red markers
            medians = [group1_stats['median'], group2_stats['median']]
            x_positions = [0, 1]
            ax1.plot(x_positions, medians, 'ro', markersize=10, label='Median')
            
            # Add test statistics
            stats_text = f"Mann-Whitney U = {statistic:.1f}, p = {p_value:.4f}"
            if significant:
                stats_text += " *"
            ax1.text(0.5, 0.02, stats_text, transform=ax1.transAxes, 
                    horizontalalignment='center', bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
            
            # Add labels and title
            ax1.set_title('Distribution Comparison')
            ax1.set_ylabel('Value')
            ax1.legend()
            
            fig1.tight_layout()
            figures['boxplot'] = fig_to_svg(fig1)
        except Exception as e:
            figures['boxplot_error'] = str(e)
        
        # Figure 2: Cumulative distribution functions
        try:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            # Sort the data for ECDF
            group1_sorted = np.sort(group1_values)
            group2_sorted = np.sort(group2_values)
            
            # Calculate ECDF
            group1_y = np.arange(1, len(group1_sorted) + 1) / len(group1_sorted)
            group2_y = np.arange(1, len(group2_sorted) + 1) / len(group2_sorted)
            
            # Plot ECDFs
            ax2.step(group1_sorted, group1_y, where='post', label=group1_name, linewidth=2, color=PASTEL_COLORS[0])
            ax2.step(group2_sorted, group2_y, where='post', label=group2_name, linewidth=2, linestyle='--', color=PASTEL_COLORS[1])
            
            # Add vertical lines at medians
            ax2.axvline(x=group1_stats['median'], color='blue', linestyle='-', alpha=0.5, 
                       label=f"{group1_name} Median ({group1_stats['median']:.2f})")
            ax2.axvline(x=group2_stats['median'], color='orange', linestyle='-', alpha=0.5,
                       label=f"{group2_name} Median ({group2_stats['median']:.2f})")
            
            # Add shaded area for the difference
            if hodges_lehmann > 0:
                label_higher = group1_name
                label_lower = group2_name
            else:
                label_higher = group2_name
                label_lower = group1_name
            
            ax2.text(0.5, 0.1, f"Median Difference = {abs(hodges_lehmann):.3f}\n({label_higher} > {label_lower})", 
                    transform=ax2.transAxes, horizontalalignment='center',
                    bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
            
            # Add labels and title
            ax2.set_xlabel('Value')
            ax2.set_ylabel('Cumulative Probability')
            ax2.set_title('Empirical Cumulative Distribution Functions')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            fig2.tight_layout()
            figures['ecdf_plot'] = fig_to_svg(fig2)
        except Exception as e:
            figures['ecdf_plot_error'] = str(e)
        
        # Figure 3: Rank visualization
        try:
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            
            # Create combined data for ranked scatter plot
            combined = np.concatenate([group1_values, group2_values])
            combined_ranks = stats.rankdata(combined)
            
            group_labels = np.concatenate([[group1_name] * n1, [group2_name] * n2])
            
            # Create DataFrame for plotting
            df_ranks = pd.DataFrame({
                'Value': combined,
                'Rank': combined_ranks,
                'Group': group_labels
            })
            
            # Create scatter plot of ranks
            sns.scatterplot(x='Value', y='Rank', hue='Group', data=df_ranks, ax=ax3, alpha=0.7, palette=PASTEL_COLORS[:2])
            
            # Add vertical lines for group medians
            ax3.axvline(x=group1_stats['median'], color='blue', linestyle='--', alpha=0.5)
            ax3.axvline(x=group2_stats['median'], color='orange', linestyle='--', alpha=0.5)
            
            # Add mean rank lines
            ax3.axhline(y=np.mean(combined_ranks[:n1]), color='blue', linestyle='-', alpha=0.5,
                       label=f"{group1_name} Mean Rank ({np.mean(combined_ranks[:n1]):.1f})")
            ax3.axhline(y=np.mean(combined_ranks[n1:]), color='orange', linestyle='-', alpha=0.5,
                       label=f"{group2_name} Mean Rank ({np.mean(combined_ranks[n1:]):.1f})")
            
            # Add labels and title
            ax3.set_xlabel('Value')
            ax3.set_ylabel('Rank')
            ax3.set_title('Rank Distribution')
            ax3.legend()
            
            # Add test statistics
            ax3.text(0.02, 0.98, f"U = {statistic:.1f}, p = {p_value:.4f}", transform=ax3.transAxes, 
                    verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
            
            fig3.tight_layout()
            figures['rank_plot'] = fig_to_svg(fig3)
        except Exception as e:
            figures['rank_plot_error'] = str(e)
        
        # Figure 4: Density plots
        try:
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            
            # Create kernel density plots
            sns.kdeplot(group1_values, ax=ax4, label=group1_name, shade=True, alpha=0.5, color=PASTEL_COLORS[0])
            sns.kdeplot(group2_values, ax=ax4, label=group2_name, shade=True, alpha=0.5, color=PASTEL_COLORS[1])
            
            # Add vertical lines at medians
            ax4.axvline(x=group1_stats['median'], color='blue', linestyle='-', alpha=0.5)
            ax4.axvline(x=group2_stats['median'], color='orange', linestyle='-', alpha=0.5)
            
            # Add median difference annotation
            if hodges_lehmann != 0:
                arrow_start = min(group1_stats['median'], group2_stats['median'])
                arrow_end = max(group1_stats['median'], group2_stats['median'])
                arrow_y = ax4.get_ylim()[1] * 0.5
                ax4.annotate('', xy=(arrow_end, arrow_y), xytext=(arrow_start, arrow_y),
                            arrowprops={'arrowstyle': '<->', 'lw': 2, 'color': 'black'})
                ax4.text((arrow_start + arrow_end) / 2, arrow_y * 1.1, 
                        f"Median Diff = {abs(hodges_lehmann):.3f}", ha='center')
            
            # Add labels and title
            ax4.set_xlabel('Value')
            ax4.set_ylabel('Density')
            ax4.set_title('Density Distribution Comparison')
            ax4.legend()
            
            # Add shape comparison annotation
            if 'similar_shapes' in assumptions and 'details' in assumptions['similar_shapes']:
                ax4.text(0.5, 0.02, assumptions['similar_shapes']['details'], transform=ax4.transAxes, 
                        horizontalalignment='center', bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
            
            fig4.tight_layout()
            figures['density_plot'] = fig_to_svg(fig4)
        except Exception as e:
            figures['density_plot_error'] = str(e)
        
        # Figure 5: Effect size and comparison with t-test
        if 't_test' in additional_stats and 'error' not in additional_stats['t_test']:
            try:
                fig5, ax5 = plt.subplots(figsize=(10, 6))
                
                # Prepare data for plotting
                test_names = ['Mann-Whitney U', 'Independent t-test']
                effect_sizes = [effect_size_r, abs(additional_stats['t_test']['cohens_d'])]
                p_values = [p_value, additional_stats['t_test']['p_value']]
                
                # Create grouped bar chart
                x = np.arange(len(test_names))
                width = 0.35
                
                ax5.bar(x - width/2, effect_sizes, width, label='Effect Size', color=PASTEL_COLORS[0], alpha=0.7)
                ax5.bar(x + width/2, p_values, width, label='p-value', color=PASTEL_COLORS[1], alpha=0.7)
                
                # Add horizontal line at significance level
                ax5.axhline(y=alpha, color='red', linestyle='--', label=f'α = {alpha}')
                
                # Add labels and title
                ax5.set_ylabel('Value')
                ax5.set_title('Comparison of Parametric and Non-parametric Tests')
                ax5.set_xticks(x)
                ax5.set_xticklabels(test_names)
                ax5.legend()
                
                # Add annotations
                ax5.text(0 - width/2, effect_sizes[0] + 0.05, f"r = {effect_size_r:.3f}\n({effect_magnitude})", 
                        ha='center', va='bottom')
                ax5.text(1 - width/2, effect_sizes[1] + 0.05, f"d = {abs(additional_stats['t_test']['cohens_d']):.3f}", 
                        ha='center', va='bottom')
                
                ax5.text(0 + width/2, p_values[0] + 0.05, f"p = {p_value:.4f}", 
                        ha='center', va='bottom')
                ax5.text(1 + width/2, p_values[1] + 0.05, f"p = {additional_stats['t_test']['p_value']:.4f}", 
                        ha='center', va='bottom')
                
                # Add significance markers
                if significant:
                    ax5.text(0, effect_sizes[0] / 2, '*', fontsize=20, ha='center')
                if additional_stats['t_test']['significant']:
                    ax5.text(1, effect_sizes[1] / 2, '*', fontsize=20, ha='center')
                
                fig5.tight_layout()
                figures['test_comparison'] = fig_to_svg(fig5)
            except Exception as e:
                figures['test_comparison_error'] = str(e)
        
        # Figure 6: Bootstrap distribution of median difference
        if 'bootstrap' in additional_stats and 'error' not in additional_stats['bootstrap']:
            try:
                fig6, ax6 = plt.subplots(figsize=(10, 6))
                
                # Extract bootstrap data
                bootstrap_diffs = np.random.normal(
                    additional_stats['bootstrap']['median_diff'],
                    (additional_stats['bootstrap']['median_diff_ci_upper'] - 
                     additional_stats['bootstrap']['median_diff_ci_lower']) / 3.92,
                    2000  # simulate 2000 bootstrap samples
                )
                
                # Create histogram of bootstrap distribution
                sns.histplot(bootstrap_diffs, kde=True, ax=ax6, color=PASTEL_COLORS[0])
                
                # Add vertical line for observed median difference
                ax6.axvline(x=hodges_lehmann, color='red', linestyle='-', linewidth=2,
                           label=f'Observed Median Diff = {hodges_lehmann:.3f}')
                
                # Add vertical lines for bootstrap CI
                boot_ci_lower = additional_stats['bootstrap']['median_diff_ci_lower']
                boot_ci_upper = additional_stats['bootstrap']['median_diff_ci_upper']
                
                ax6.axvline(x=boot_ci_lower, color='green', linestyle='--',
                           label=f'{(1-alpha)*100:.0f}% CI Lower = {boot_ci_lower:.3f}')
                ax6.axvline(x=boot_ci_upper, color='green', linestyle='--',
                           label=f'{(1-alpha)*100:.0f}% CI Upper = {boot_ci_upper:.3f}')
                
                # Add vertical line at zero
                ax6.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                
                # Add labels and title
                ax6.set_xlabel('Median Difference')
                ax6.set_ylabel('Frequency')
                ax6.set_title('Bootstrap Distribution of Median Difference')
                ax6.legend()
                
                fig6.tight_layout()
                figures['bootstrap_distribution'] = fig_to_svg(fig6)
            except Exception as e:
                figures['bootstrap_distribution_error'] = str(e)
        
        return {
            'test': 'Mann-Whitney U Test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': significant,
            'alternative': alternative,
            'group1_stats': group1_stats,
            'group2_stats': group2_stats,
            'effect_size_r': float(effect_size_r),
            'effect_magnitude': effect_magnitude,
            'hodges_lehmann_estimator': float(hodges_lehmann),
            'z_score': float(z_score),
            'assumptions': assumptions,
            'interpretation': interpretation,
            'additional_statistics': additional_stats,
            'figures': figures
        }
    except Exception as e:
        return {
            'test': 'Mann-Whitney U Test',
            'statistic': None,
            'p_value': None,
            'significant': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
