import traceback
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import io
from matplotlib.patches import Patch
from typing import Dict, List, Any
from data.selection.detailed_tests.formatting import fig_to_svg, PASTEL_COLORS, PASTEL_CMAP
from data.assumptions.tests import (
    AssumptionResult,
    SampleSizeTest, 
    OutlierTest, 
    IndependenceTest,
    MonotonicityTest
)



def kruskal_wallis_test(groups: List[pd.Series], alpha: float) -> Dict[str, Any]:
    """Performs a Kruskal-Wallis test with comprehensive statistics and assumption checks."""
    try:
        if len(groups) < 2:
            return {
                'test': "Kruskal-Wallis Test",
                'statistic': None,
                'p_value': None,
                'significant': False,
                'reason': 'Less than two groups provided.'
            }
            
        # Drop NaN values from each group
        groups = [group.dropna() for group in groups]
        
        # Calculate basic statistics for each group
        group_stats = []
        for i, group in enumerate(groups):
            group_stats.append({
                'n': len(group),
                'median': float(group.median()),
                'mean': float(group.mean()),
                'std': float(group.std(ddof=1)),
                'min': float(group.min()),
                'max': float(group.max()),
                'q1': float(np.percentile(group, 25)),
                'q3': float(np.percentile(group, 75)),
                'iqr': float(np.percentile(group, 75) - np.percentile(group, 25)),
                'name': f'Group {i+1}'
            })
            
        # Set group names based on Series names if available
        for i, group in enumerate(groups):
            if hasattr(group, 'name') and group.name is not None:
                group_stats[i]['name'] = str(group.name)
             
        from scipy import stats
        # Perform Kruskal-Wallis test
        groups_arrays = [group.values for group in groups]
        statistic, p_value = stats.kruskal(*groups_arrays)
        
        # Calculate degrees of freedom
        df = len(groups) - 1
        
        # Calculate effect size (eta-squared)
        # For Kruskal-Wallis, eta-squared can be calculated as:
        # H / (n - 1), where H is the test statistic and n is the total sample size
        total_n = sum(len(group) for group in groups)
        eta_squared = statistic / (total_n - 1)
        
        # Calculate epsilon-squared (less biased than eta-squared)
        epsilon_squared = statistic - len(groups) + 1
        epsilon_squared = max(0, epsilon_squared) / (total_n**2 - 1) / (total_n + 1)
        
        # Determine magnitude of effect size
        if eta_squared < 0.01:
            effect_magnitude = "Negligible"
        elif eta_squared < 0.06:
            effect_magnitude = "Small"
        elif eta_squared < 0.14:
            effect_magnitude = "Medium"
        else:
            effect_magnitude = "Large"
            
        # Perform post-hoc tests if Kruskal-Wallis is significant
        post_hoc = {}
        if p_value < alpha:
            # Dunn's test
            try:
                from scikit_posthocs import posthoc_dunn
                
                # Prepare data for Dunn's test
                all_data = np.concatenate(groups_arrays)
                group_labels = np.concatenate([[i] * len(group) for i, group in enumerate(groups)])
                
                # Create a DataFrame for posthoc_dunn
                data_df = pd.DataFrame({'value': all_data, 'group': group_labels})
                
                # Perform Dunn's test
                dunn_result = posthoc_dunn(data_df, val_col='value', group_col='group', p_adjust='bonferroni')
                
                # Extract results
                dunn_data = []
                for i in range(len(groups)):
                    for j in range(i+1, len(groups)):
                        p_adj = dunn_result.iloc[i, j]
                        dunn_data.append({
                            'group1': group_stats[i]['name'],
                            'group2': group_stats[j]['name'],
                            'p_value': float(p_adj),
                            'significant': p_adj < alpha
                        })
                    
                post_hoc['dunn'] = dunn_data
            except Exception as e:
                post_hoc['dunn'] = {'error': f'Could not perform Dunn\'s test: {str(e)}'}
                
            # Mann-Whitney U tests with Bonferroni correction
            try:
                mw_results = []
                
                for i in range(len(groups)):
                    for j in range(i+1, len(groups)):
                        # Run Mann-Whitney U test
                        u_stat, p_val = stats.mannwhitneyu(
                            groups[i], 
                            groups[j], 
                            alternative='two-sided'
                        )
                        
                        # Apply Bonferroni correction
                        n_comparisons = len(groups) * (len(groups) - 1) / 2
                        p_adj = min(p_val * n_comparisons, 1.0)
                        
                        # Calculate effect size (r)
                        n1 = len(groups[i])
                        n2 = len(groups[j])
                        
                        # Calculate z-score from U statistic
                        u_mean = n1 * n2 / 2
                        u_std = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
                        z_score = (u_stat - u_mean) / u_std
                        
                        # Calculate effect size r = Z/sqrt(N)
                        effect_size_r = abs(z_score) / np.sqrt(n1 + n2)
                        
                        # Calculate median difference
                        median_diff = np.median(groups[i]) - np.median(groups[j])
                        
                        mw_results.append({
                            'group1': group_stats[i]['name'],
                            'group2': group_stats[j]['name'],
                            'u_statistic': float(u_stat),
                            'z_score': float(z_score),
                            'effect_size_r': float(effect_size_r),
                            'median_diff': float(median_diff),
                            'p_value': float(p_val),
                            'p_adjusted': float(p_adj),
                            'significant': p_adj < alpha
                        })
                
                post_hoc['mann_whitney'] = mw_results
            except Exception as e:
                post_hoc['mann_whitney'] = {'error': f'Could not perform Mann-Whitney U tests: {str(e)}'}
                
            # Conover-Iman test
            try:
                from scikit_posthocs import posthoc_conover
                
                # Perform Conover-Iman test
                conover_result = posthoc_conover(data_df, val_col='value', group_col='group', p_adjust='bonferroni')
                
                # Extract results
                conover_data = []
                for i in range(len(groups)):
                    for j in range(i+1, len(groups)):
                        p_adj = conover_result.iloc[i, j]
                        conover_data.append({
                            'group1': group_stats[i]['name'],
                            'group2': group_stats[j]['name'],
                            'p_value': float(p_adj),
                            'significant': p_adj < alpha
                        })
                    
                post_hoc['conover'] = conover_data
            except Exception as e:
                post_hoc['conover'] = {'error': f'Could not perform Conover-Iman test: {str(e)}'}
                
        # Run formal assumption tests
        assumptions = {}
        
        # 1. Sample size check - using formal test
        sample_size_test = SampleSizeTest()
        for i, group in enumerate(groups):
            # For Kruskal-Wallis, minimum recommended sample size is typically 5 per group
            sample_size_result = sample_size_test.run_test(data=group, min_recommended=5)
            assumptions['sample_size_' + group_stats[i]['name']] = sample_size_result
                
        # 2. Outliers check - using formal test
        outlier_test = OutlierTest()
        
        for i, group in enumerate(groups):
            outlier_result = outlier_test.run_test(data=group)
            assumptions['outliers_' + group_stats[i]['name']] = outlier_result
            
        # 3. Similar distribution shapes check (using skewness and kurtosis)
        # This is a key assumption for Kruskal-Wallis when interpreting it as a test of medians
        try:
            # Check if distributions have similar shapes using skewness and kurtosis
            distribution_results = []
            
            skewness_values = []
            kurtosis_values = []
            
            for i, group in enumerate(groups):
                skewness = float(stats.skew(group))
                kurtosis = float(stats.kurtosis(group))
                
                skewness_values.append(skewness)
                kurtosis_values.append(kurtosis)
                
                distribution_results.append({
                    'group': group_stats[i]['name'],
                    'skewness': skewness,
                    'kurtosis': kurtosis
                })
            
            # Check if skewness and kurtosis are reasonably similar across groups
            skewness_range = max(skewness_values) - min(skewness_values)
            kurtosis_range = max(kurtosis_values) - min(kurtosis_values)
            
            similar_shapes = skewness_range < 2.0 and kurtosis_range < 4.0
            
            assumptions['similar_distributions'] = {
                'results': distribution_results,
                'skewness_range': float(skewness_range),
                'kurtosis_range': float(kurtosis_range),
                'satisfied': similar_shapes,
                'overall_result': AssumptionResult.PASSED if similar_shapes else AssumptionResult.WARNING
            }
        except Exception as e:
            assumptions['similar_distributions'] = {
                'error': str(e),
                'overall_result': AssumptionResult.NOT_APPLICABLE
            }
        
        # 4. Independence check - using formal test
        try:
            independence_test = IndependenceTest()
            for i, group in enumerate(groups):
                independence_result = independence_test.run_test(data=group)
                assumptions['independence_' + group_stats[i]['name']] = independence_result

        except Exception as e:
            assumptions['independence'] = {
                'error': str(e),
                'overall_result': AssumptionResult.NOT_APPLICABLE
            }
            
        # 5. Monotonicity check - beneficial for checking relationships between groups
        try:
            # Only add if we have exactly two groups
            if len(groups) == 2:
                monotonicity_test = MonotonicityTest()
                monotonicity_result = monotonicity_test.run_test(x=groups[0], y=groups[1])
                
                assumptions['monotonicity'] = {
                    'result': monotonicity_result['result'],
                    'details': monotonicity_result.get('details', ''),
                    'statistics': monotonicity_result.get('statistic', None),
                    'p_value': monotonicity_result.get('p_value', None),
                    'raw_result': monotonicity_result,
                    'figures': monotonicity_result.get('figures', {})
                }
        except Exception as e:
            assumptions['monotonicity'] = {
                'error': str(e),
                'overall_result': AssumptionResult.NOT_APPLICABLE
            }
        
        # Create interpretation
        interpretation = f"Kruskal-Wallis test comparing {len(groups)} groups.\n\n"
        
        # Basic results
        interpretation += f"H({df}) = {statistic:.3f}, p = {p_value:.5f}\n"
        interpretation += f"Effect size: η² = {eta_squared:.3f}, ε² = {epsilon_squared:.3f} ({effect_magnitude.lower()} effect)\n\n"
        
        # Group statistics
        interpretation += "Group statistics:\n"
        for i, stats in enumerate(group_stats):
            interpretation += f"- {stats['name']}: n = {stats['n']}, median = {stats['median']:.3f}, mean = {stats['mean']:.3f}, IQR = {stats['iqr']:.3f}\n"
            
        # Assumptions
        interpretation += "\nAssumption checks:\n"
        
        # Sample size
        sample_size_keys = [key for key in assumptions if key.startswith('sample_size_')]
        if sample_size_keys:
            interpretation += f"- Sample size: "
            small_groups = []
            all_adequate = True
            
            for key in sample_size_keys:
                size_result = assumptions[key]
                group_name = key.replace('sample_size_', '')
                if not size_result.get('satisfied', False):
                    small_groups.append(group_name)
                    all_adequate = False
            
            if all_adequate:
                interpretation += "Sample sizes are adequate for all groups.\n"
            else:
                interpretation += f"Sample sizes may be too small for {', '.join(small_groups)}. Results should be interpreted with caution.\n"
        
        # Outliers
        outlier_keys = [key for key in assumptions if key.startswith('outliers_')]
        if outlier_keys:
            interpretation += f"- Outliers: "
            groups_with_outliers = []
            no_outliers = True
            
            for key in outlier_keys:
                outlier_result = assumptions[key]
                group_name = key.replace('outliers_', '')
                if not outlier_result.get('satisfied', True):
                    groups_with_outliers.append(group_name)
                    no_outliers = False
            
            if no_outliers:
                interpretation += "No significant outliers detected in any group.\n"
            else:
                interpretation += f"Potential outliers detected in {', '.join(groups_with_outliers)}. "
                interpretation += "Consider examining these outliers and potentially removing them if they represent measurement errors.\n"
        
        # Similar distributions
        if 'similar_distributions' in assumptions and 'satisfied' in assumptions['similar_distributions']:
            dist_result = assumptions['similar_distributions']
            interpretation += f"- Similar distributions: "
            
            if dist_result['satisfied']:
                interpretation += "Groups appear to have similar distribution shapes, which is appropriate for interpreting Kruskal-Wallis as a test of medians.\n"
            else:
                interpretation += "Groups may have different distribution shapes. "
                interpretation += "Kruskal-Wallis should be interpreted as testing stochastic dominance rather than specifically testing for differences in medians.\n"
        
        # Independence 
        independence_keys = [key for key in assumptions if key.startswith('independence_')]
        if independence_keys:
            interpretation += f"- Independence: "
            dependent_groups = []
            all_independent = True
            
            for key in independence_keys:
                indep_result = assumptions[key]
                group_name = key.replace('independence_', '')
                if not indep_result.get('satisfied', True):
                    dependent_groups.append(group_name)
                    all_independent = False
            
            if all_independent:
                interpretation += "Data within all groups appears to be independent.\n"
            else:
                interpretation += f"There may be dependence within these groups: {', '.join(dependent_groups)}. "
                interpretation += "This could affect the validity of the Kruskal-Wallis test results.\n"
        
        # Conclusion
        interpretation += f"\nConclusion: There is {'a statistically significant' if p_value < alpha else 'no statistically significant'} "
        interpretation += f"difference in the distributions of the groups (p = {p_value:.5f}). "
        
        if p_value < alpha:
            interpretation += f"The {effect_magnitude.lower()} effect size (η² = {eta_squared:.3f}) suggests that "
            interpretation += f"group membership explains approximately {eta_squared*100:.1f}% of the variance in ranks.\n\n"
            
            # Post-hoc results
            if 'dunn' in post_hoc and isinstance(post_hoc['dunn'], list):
                interpretation += "Post-hoc comparisons (Dunn's test with Bonferroni correction):\n"
                
                significant_pairs = [pair for pair in post_hoc['dunn'] if pair['significant']]
                if significant_pairs:
                    for pair in significant_pairs:
                        interpretation += f"- {pair['group1']} vs {pair['group2']}: p = {pair['p_value']:.5f}\n"
                else:
                    interpretation += "- No significant pairwise differences were found despite the significant overall test.\n"
        
        # Calculate additional statistics
        additional_stats = {}
        
        # 1. Jonckheere-Terpstra test for ordered alternatives
        try:
            # Fix: Direct import for norm
            from scipy.stats import mannwhitneyu, norm
            
            # Simple implementation of Jonckheere-Terpstra test
            j_stat = 0
            for i in range(len(groups)):
                for j in range(i+1, len(groups)):
                    # Count number of times values in later groups exceed values in earlier groups
                    for val_i in groups[i]:
                        for val_j in groups[j]:
                            if val_j > val_i:
                                j_stat += 1
            
            # Calculate expected value and standard deviation under null hypothesis
            n_values = [len(group) for group in groups]
            N = sum(n_values)
            E_J = sum(n_i * n_j for i, n_i in enumerate(n_values) for j, n_j in enumerate(n_values) if i < j) / 2
            var_J = sum(n_i * n_j * (n_i + n_j + 1) for i, n_i in enumerate(n_values) for j, n_j in enumerate(n_values) if i < j) / 72
            
            # Calculate z-statistic and p-value
            z_stat = (j_stat - E_J) / np.sqrt(var_J)
            j_p_value = 2 * (1 - norm.cdf(abs(z_stat)))
            
            additional_stats['jonckheere_terpstra'] = {
                'statistic': float(j_stat),
                'z_statistic': float(z_stat),
                'p_value': float(j_p_value),
                'significant': j_p_value < alpha,
                'interpretation': "Tests for an ordered pattern across groups (i.e., whether medians increase or decrease across groups in the order provided)."
            }
        except Exception as e:
            additional_stats['jonckheere_terpstra'] = {
                'error': str(e)
            }
        
        # 2. Compare with one-way ANOVA
        try:
            # Fix: Use direct import for f_oneway
            from scipy.stats import f_oneway
            
            # Perform one-way ANOVA
            anova_stat, anova_p = f_oneway(*groups_arrays)
            
            # Calculate ANOVA effect size (eta-squared)
            anova_eta_squared = (anova_stat * df) / ((anova_stat * df) + (total_n - len(groups)))
            
            additional_stats['anova_comparison'] = {
                'f_statistic': float(anova_stat),
                'p_value': float(anova_p),
                'significant': anova_p < alpha,
                'eta_squared': float(anova_eta_squared),
                'agreement_with_kw': (anova_p < alpha) == (p_value < alpha)
            }
        except Exception as e:
            additional_stats['anova_comparison'] = {
                'error': str(e)
            }
        
        # 3. Calculate mean ranks and differences
        try:
            # Fix: Direct import for rankdata
            from scipy.stats import rankdata
            
            # Calculate mean ranks for each group
            all_data = np.concatenate(groups_arrays)
            all_ranks = rankdata(all_data)
            
            start_idx = 0
            mean_ranks = []
            
            for i, group in enumerate(groups):
                group_size = len(group)
                end_idx = start_idx + group_size
                
                group_ranks = all_ranks[start_idx:end_idx]
                mean_rank = np.mean(group_ranks)
                
                mean_ranks.append({
                    'group': group_stats[i]['name'],
                    'mean_rank': float(mean_rank),
                    'sum_rank': float(np.sum(group_ranks))
                })
                
                start_idx = end_idx
            
            # Calculate mean rank differences between groups
            rank_diffs = []
            for i in range(len(groups)):
                for j in range(i+1, len(groups)):
                    rank_diff = mean_ranks[i]['mean_rank'] - mean_ranks[j]['mean_rank']
                    
                    rank_diffs.append({
                        'group1': group_stats[i]['name'],
                        'group2': group_stats[j]['name'],
                        'rank_diff': float(rank_diff),
                        'abs_rank_diff': float(abs(rank_diff))
                    })
            
            additional_stats['rank_analysis'] = {
                'mean_ranks': mean_ranks,
                'rank_differences': rank_diffs
            }
        except Exception as e:
            additional_stats['rank_analysis'] = {
                'error': str(e)
            }
        
        # 4. Bootstrap confidence intervals for median differences
        try:
            n_bootstrap = 2000
            bootstrap_results = []
            
            for i in range(len(groups)):
                for j in range(i+1, len(groups)):
                    bootstrap_diffs = []
                    
                    # Generate bootstrap samples
                    for _ in range(n_bootstrap):
                        # Sample with replacement
                        idx1 = np.random.choice(range(len(groups[i])), size=len(groups[i]), replace=True)
                        idx2 = np.random.choice(range(len(groups[j])), size=len(groups[j]), replace=True)
                        
                        # Calculate median difference
                        boot_group1 = groups[i].iloc[idx1]
                        boot_group2 = groups[j].iloc[idx2]
                        
                        bootstrap_diffs.append(np.median(boot_group1) - np.median(boot_group2))
                    
                    # Calculate bootstrap CI
                    boot_ci_lower = float(np.percentile(bootstrap_diffs, 100 * alpha/2))
                    boot_ci_upper = float(np.percentile(bootstrap_diffs, 100 * (1 - alpha/2)))
                    
                    bootstrap_results.append({
                        'group1': group_stats[i]['name'],
                        'group2': group_stats[j]['name'],
                        'median_diff': float(np.median(bootstrap_diffs)),
                        'ci_lower': boot_ci_lower,
                        'ci_upper': boot_ci_upper,
                        'includes_zero': boot_ci_lower <= 0 <= boot_ci_upper
                    })
            
            additional_stats['bootstrap'] = {
                'n_samples': n_bootstrap,
                'results': bootstrap_results
            }
        except Exception as e:
            additional_stats['bootstrap'] = {
                'error': str(e)
            }
        
        # 5. Power analysis for Kruskal-Wallis test
        try:
            # Fix: Use a different approach for non-central chi-square
            from scipy.stats import chi2, ncx2
            
            # Observed effect size
            observed_effect = eta_squared
            
            # Non-centrality parameter
            ncp = statistic
            
            # Power of current test - using non-central chi-square distribution
            critical_value = chi2.ppf(1-alpha, df=df)
            observed_power = 1 - ncx2.cdf(critical_value, df=df, nc=ncp)
            
            # Calculate required sample size for standard power levels
            def kw_power(total_n, effect, groups_count, alpha_level):
                # Calculate approximated non-centrality parameter
                approx_ncp = effect * (total_n - 1)
                # Get critical value
                critical = chi2.ppf(1-alpha_level, df=groups_count-1)
                # Return power using non-central chi-square
                return 1 - ncx2.cdf(critical, df=groups_count-1, nc=approx_ncp)
            
            # Find sample size for desired power
            def find_n_for_power(target_power, effect, groups_count, alpha_level):
                n_start = len(groups) * 5  # Start with minimal viable sample size
                while True:
                    power = kw_power(n_start, effect, groups_count, alpha_level)
                    if power >= target_power:
                        return n_start
                    n_start += 5
                    if n_start > 10000:  # Avoid infinite loops
                        return -1
            
            # Calculate required sample sizes
            n_for_80_power = find_n_for_power(0.8, observed_effect, len(groups), alpha)
            n_for_90_power = find_n_for_power(0.9, observed_effect, len(groups), alpha)
            n_for_95_power = find_n_for_power(0.95, observed_effect, len(groups), alpha)
            
            additional_stats['power_analysis'] = {
                'observed_power': float(observed_power),
                'current_total_n': total_n,
                'total_n_for_80_power': int(n_for_80_power),
                'total_n_for_90_power': int(n_for_90_power),
                'total_n_for_95_power': int(n_for_95_power)
            }
        except Exception as e:
            additional_stats['power_analysis'] = {
                'error': str(e)
            }
        
        # Generate figures
        figures = {}
        
        # Figure 1: Box plots with individual data points
        try:
            fig1, ax1 = plt.subplots(figsize=(12, 8))
            
            # Create data for plotting
            all_data = []
            labels = []
            
            for i, group in enumerate(groups):
                all_data.extend(group.values)
                labels.extend([group_stats[i]['name']] * len(group))
            
            plot_data = pd.DataFrame({
                'Value': all_data,
                'Group': labels
            })
            
            # Create box plot with individual points
            sns.boxplot(x='Group', y='Value', data=plot_data, ax=ax1, width=0.5, palette=PASTEL_COLORS)
            sns.stripplot(x='Group', y='Value', data=plot_data, ax=ax1, 
                        color='black', alpha=0.5, jitter=True, size=4)
            
            # Add medians as red markers
            medians = [stats['median'] for stats in group_stats]
            x_positions = range(len(groups))
            ax1.plot(x_positions, medians, 'ro', markersize=10, label='Median')
            
            # Add test statistics
            stats_text = f"Kruskal-Wallis H({df}) = {statistic:.3f}, p = {p_value:.4f}"
            significant = p_value < alpha
            if significant:
                stats_text += " *"
            ax1.text(0.5, 0.02, stats_text, transform=ax1.transAxes, 
                    horizontalalignment='center', bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
            
            # Add labels and title
            ax1.set_title('Distribution Comparison Across Groups')
            ax1.set_ylabel('Value')
            ax1.legend()
            
            fig1.tight_layout()
            figures['boxplot'] = fig_to_svg(fig1)
        except Exception as e:
            figures['boxplot_error'] = str(e)
        
        # Figure 2: Distribution density plots
        try:
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            
            # Create density plots for each group
            for i, group in enumerate(groups):
                # Create kernel density estimate
                sns.kdeplot(group, ax=ax2, label=group_stats[i]['name'], color=PASTEL_COLORS[i % len(PASTEL_COLORS)])
                
                # Add vertical line at median
                ax2.axvline(x=group_stats[i]['median'], linestyle='--', color=PASTEL_COLORS[i % len(PASTEL_COLORS)],
                           alpha=0.7, label=f"{group_stats[i]['name']} median")
            
            # Add labels and title
            ax2.set_xlabel('Value')
            ax2.set_ylabel('Density')
            ax2.set_title('Distribution Density Comparison')
            
            # Customize legend to avoid duplicates
            handles, labels = ax2.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax2.legend(by_label.values(), by_label.keys())
            
            # Add shape comparison annotation
            if 'similar_distributions' in assumptions and 'satisfied' in assumptions['similar_distributions']:
                dist_result = assumptions['similar_distributions']
                shape_text = "Distributions have similar shapes" if dist_result['satisfied'] else "Distributions may have different shapes"
                ax2.text(0.5, 0.02, shape_text, transform=ax2.transAxes, 
                        horizontalalignment='center', bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
            
            fig2.tight_layout()
            figures['density_plot'] = fig_to_svg(fig2)
        except Exception as e:
            figures['density_plot_error'] = str(e)
        
        # Figure 3: Mean ranks visualization
        if 'rank_analysis' in additional_stats and 'error' not in additional_stats['rank_analysis']:
            try:
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                
                # Extract mean ranks
                mean_ranks_data = additional_stats['rank_analysis']['mean_ranks']
                group_names = [rank['group'] for rank in mean_ranks_data]
                rank_values = [rank['mean_rank'] for rank in mean_ranks_data]
                
                # Create bar chart of mean ranks
                bars = ax3.bar(group_names, rank_values, color='skyblue')
                
                # Add overall mean rank line
                overall_mean_rank = (total_n + 1) / 2
                ax3.axhline(y=overall_mean_rank, color='red', linestyle='--', 
                           label=f'Overall Mean Rank ({overall_mean_rank:.1f})')
                
                # Add labels and title
                ax3.set_xlabel('Group')
                ax3.set_ylabel('Mean Rank')
                ax3.set_title('Mean Ranks by Group')
                ax3.legend()
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{height:.1f}', ha='center', va='bottom')
                
                # Add test statistics
                ax3.text(0.02, 0.98, f"H({df}) = {statistic:.3f}, p = {p_value:.4f}", transform=ax3.transAxes, 
                        verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
                
                fig3.tight_layout()
                figures['mean_ranks'] = fig_to_svg(fig3)
            except Exception as e:
                figures['mean_ranks_error'] = str(e)
        
        # Figure 4: Post-hoc comparison heatmap
        if p_value < alpha and 'dunn' in post_hoc and isinstance(post_hoc['dunn'], list):
            try:
                fig4, ax4 = plt.subplots(figsize=(10, 8))
                
                # Create matrix for heatmap
                n_groups = len(groups)
                p_values_matrix = np.ones((n_groups, n_groups))
                
                # Fill in p-values
                for pair in post_hoc['dunn']:
                    idx1 = next(i for i, stat in enumerate(group_stats) if stat['name'] == pair['group1'])
                    idx2 = next(i for i, stat in enumerate(group_stats) if stat['name'] == pair['group2'])
                    
                    p_values_matrix[idx1, idx2] = pair['p_value']
                    p_values_matrix[idx2, idx1] = pair['p_value']  # Symmetric
                
                # Create mask for diagonal
                mask = np.eye(n_groups, dtype=bool)
                
                # Create annotations with significance indicators
                annot = np.empty_like(p_values_matrix, dtype=object)
                for i in range(n_groups):
                    for j in range(n_groups):
                        if i != j:
                            p_val = p_values_matrix[i, j]
                            if p_val < alpha:
                                stars = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else '*')
                                annot[i, j] = f"{p_val:.3f}{stars}"
                            else:
                                annot[i, j] = f"{p_val:.3f}"
                        else:
                            annot[i, j] = ""
                
                # Create heatmap
                sns.heatmap(p_values_matrix, mask=mask, annot=annot, fmt='', 
                          cmap='YlOrRd_r', vmin=0, vmax=1, ax=ax4)
                
                # Set labels
                ax4.set_xlabel('Group')
                ax4.set_ylabel('Group')
                ax4.set_title("Dunn's Test p-values (Bonferroni adjusted)")
                
                # Set tick labels
                ax4.set_xticklabels([stat['name'] for stat in group_stats])
                ax4.set_yticklabels([stat['name'] for stat in group_stats])
                
                # Add significance legend
                legend_text = "* p < 0.05\n** p < 0.01\n*** p < 0.001"
                ax4.text(1.05, 0.5, legend_text, transform=ax4.transAxes,
                        verticalalignment='center')
                
                fig4.tight_layout()
                figures['posthoc_heatmap'] = fig_to_svg(fig4)
            except Exception as e:
                figures['posthoc_heatmap_error'] = str(e)
        
        # Figure 5: Comparison with ANOVA
        if 'anova_comparison' in additional_stats and 'error' not in additional_stats['anova_comparison']:
            try:
                fig5, ax5 = plt.subplots(figsize=(8, 6))
                
                # Prepare data for plotting
                tests = ['Kruskal-Wallis', 'One-way ANOVA']
                test_stats = [
                    statistic / df,  # Adjusted to be comparable to F
                    additional_stats['anova_comparison']['f_statistic']
                ]
                p_values = [
                    p_value,
                    additional_stats['anova_comparison']['p_value']
                ]
                effect_sizes = [
                    eta_squared,
                    additional_stats['anova_comparison']['eta_squared']
                ]
                
                # Create grouped bar chart
                x = np.arange(len(tests))
                width = 0.25
                
                ax5.bar(x - width, test_stats, width, label='Test Statistic', color=PASTEL_COLORS[0], alpha=0.7)
                ax5.bar(x, p_values, width, label='p-value', color=PASTEL_COLORS[1], alpha=0.7)
                ax5.bar(x + width, effect_sizes, width, label='Effect Size (η²)', color=PASTEL_COLORS[2], alpha=0.7)
                
                # Add horizontal line at significance level
                ax5.axhline(y=alpha, color='red', linestyle='--', label=f'α = {alpha}')
                
                # Add labels and title
                ax5.set_ylabel('Value')
                ax5.set_title('Comparison of Parametric and Non-parametric Tests')
                ax5.set_xticks(x)
                ax5.set_xticklabels(tests)
                ax5.legend()
                
                # Add annotations
                for i, (stat, p, effect) in enumerate(zip(test_stats, p_values, effect_sizes)):
                    # Test statistic
                    ax5.text(i - width, stat + 0.05, f"{stat:.2f}", ha='center', va='bottom')
                    
                    # p-value
                    p_text = f"p={p:.4f}"
                    if p < alpha:
                        p_text += "*"
                    ax5.text(i, p + 0.05, p_text, ha='center', va='bottom')
                    
                    # Effect size
                    ax5.text(i + width, effect + 0.05, f"η²={effect:.3f}", ha='center', va='bottom')
                
                fig5.tight_layout()
                figures['test_comparison'] = fig_to_svg(fig5)
            except Exception as e:
                figures['test_comparison_error'] = str(e)
        
        # Figure 6: Bootstrap confidence intervals for median differences
        if 'bootstrap' in additional_stats and 'error' not in additional_stats['bootstrap']:
            try:
                # Get number of comparisons
                bootstrap_results = additional_stats['bootstrap']['results']
                n_comparisons = len(bootstrap_results)
                
                # Create figure with appropriate height
                fig6, ax6 = plt.subplots(figsize=(10, max(6, n_comparisons * 0.5)))
                # Prepare data for forest plot
                comparisons = []
                medians = []
                lower_cis = []
                upper_cis = []
                colors = []
                
                for result in bootstrap_results:
                    comparisons.append(f"{result['group1']} vs {result['group2']}")
                    medians.append(result['median_diff'])
                    lower_cis.append(result['ci_lower'])
                    upper_cis.append(result['ci_upper'])
                    colors.append('red' if not result['includes_zero'] else 'gray')
                
                # Calculate error bars
                errors_minus = [m - l for m, l in zip(medians, lower_cis)]
                errors_plus = [u - m for m, u in zip(medians, upper_cis)]
                
                # Reverse lists for better visualization (bottom to top)
                comparisons = comparisons[::-1]
                medians = medians[::-1]
                errors_minus = errors_minus[::-1]
                errors_plus = errors_plus[::-1]
                colors = colors[::-1]
                
                # Create forest plot
                y_pos = np.arange(len(comparisons))
                ax6.errorbar(medians, y_pos, xerr=[errors_minus, errors_plus], fmt='o', 
                           capsize=5, color=colors, markersize=8)
                
                # Add vertical line at zero
                ax6.axvline(x=0, color='black', linestyle='--', alpha=0.7)
                
                # Add group labels
                ax6.set_yticks(y_pos)
                ax6.set_yticklabels(comparisons)
                
                # Add labels and title
                ax6.set_xlabel('Median Difference')
                ax6.set_title(f'Bootstrap {(1-alpha)*100:.0f}% Confidence Intervals for Median Differences')
                
                # Add legend
                legend_elements = [
                    Patch(facecolor='red', edgecolor='black', label='Significant Difference'),
                    Patch(facecolor='gray', edgecolor='black', label='Non-significant Difference')
                ]
                ax6.legend(handles=legend_elements, loc='lower right')
                
                # Add explanation
                bootstrap_text = f"Based on {additional_stats['bootstrap']['n_samples']} bootstrap samples"
                ax6.text(0.98, 0.02, bootstrap_text, transform=ax6.transAxes,
                        horizontalalignment='right', verticalalignment='bottom',
                        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
                
                fig6.tight_layout()
                figures['bootstrap_intervals'] = fig_to_svg(fig6)
            except Exception as e:
                figures['bootstrap_intervals_error'] = str(e)
        
        # Figure 7: Violin plots for distributional comparison
        try:
            fig7, ax7 = plt.subplots(figsize=(12, 8))
            
            # Create data for plotting
            all_data = []
            labels = []
            
            for i, group in enumerate(groups):
                all_data.extend(group.values)
                labels.extend([group_stats[i]['name']] * len(group))
            
            plot_data = pd.DataFrame({
                'Value': all_data,
                'Group': labels
            })
            
            # Create violin plot
            sns.violinplot(x='Group', y='Value', data=plot_data, ax=ax7, inner='quartile', palette=PASTEL_COLORS)
            
            # Add individual points with slight jitter
            sns.stripplot(x='Group', y='Value', data=plot_data, ax=ax7,
                        size=3, color='black', alpha=0.3, jitter=True)
            
            # Add group medians
            medians = [stats['median'] for stats in group_stats]
            x_positions = range(len(groups))
            ax7.plot(x_positions, medians, 'ro', markersize=8, label='Median')
            
            # Add test statistics
            stats_text = f"Kruskal-Wallis: H({df}) = {statistic:.3f}, p = {p_value:.4f}"
            if p_value < alpha:
                stats_text += " *"
            ax7.text(0.5, 0.02, stats_text, transform=ax7.transAxes, 
                    horizontalalignment='center', 
                    bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
            
            # Add labels and title
            ax7.set_title('Distribution Comparison (Violin Plots)')
            ax7.set_ylabel('Value')
            ax7.legend()
            
            fig7.tight_layout()
            figures['violin_plot'] = fig_to_svg(fig7)
        except Exception as e:
            figures['violin_plot_error'] = str(e)
        
        # Figure 8: Power analysis curve
        if 'power_analysis' in additional_stats and 'error' not in additional_stats['power_analysis']:
            try:
                fig8, ax8 = plt.subplots(figsize=(10, 6))
                
                # Create range of sample sizes
                sample_sizes = np.arange(len(groups) * 5, max(150, total_n * 2), 5)
                
                # Calculate power for each sample size
                powers = []
                
                for n_sample in sample_sizes:
                    # Calculate approximated non-centrality parameter
                    approx_ncp = eta_squared * (n_sample - 1)
                    # Calculate power
                    power = 1 - stats.chi2.cdf(
                        stats.chi2.ppf(1-alpha, df=df), 
                        df=df, 
                        nc=approx_ncp
                    )
                    powers.append(power)
                
                # Plot power curve
                ax8.plot(sample_sizes, powers, 'b-', linewidth=2)
                
                # Add horizontal lines at common power thresholds
                ax8.axhline(y=0.8, color='red', linestyle='--', label='0.8 (conventional minimum)')
                ax8.axhline(y=0.9, color='orange', linestyle='--', label='0.9')
                ax8.axhline(y=0.95, color='green', linestyle='--', label='0.95')
                
                # Add vertical line at current sample size
                ax8.axvline(x=total_n, color='purple', linestyle='-', 
                           label=f'Current n = {total_n} (power ≈ {additional_stats["power_analysis"]["observed_power"]:.3f})')
                
                # Add labels and title
                ax8.set_xlabel('Total Sample Size')
                ax8.set_ylabel('Statistical Power')
                ax8.set_title(f'Power Analysis for Effect Size η² = {eta_squared:.3f}')
                ax8.legend(loc='lower right')
                ax8.grid(True, linestyle='--', alpha=0.7)
                
                # Set y-axis limits
                ax8.set_ylim(0, 1.05)
                
                # Add text with key sample size recommendations
                if 'total_n_for_80_power' in additional_stats['power_analysis']:
                    text = "Recommended total sample sizes:\n"
                    text += f"For 80% power: n ≈ {additional_stats['power_analysis']['total_n_for_80_power']}\n"
                    text += f"For 90% power: n ≈ {additional_stats['power_analysis']['total_n_for_90_power']}\n"
                    text += f"For 95% power: n ≈ {additional_stats['power_analysis']['total_n_for_95_power']}"
                    
                    ax8.text(0.02, 0.02, text, transform=ax8.transAxes, 
                            verticalalignment='bottom', 
                            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
                
                fig8.tight_layout()
                figures['power_analysis'] = fig_to_svg(fig8)
            except Exception as e:
                figures['power_analysis_error'] = str(e)
        
        # Figure 9: Cumulative distribution functions
        try:
            fig9, ax9 = plt.subplots(figsize=(10, 6))
            
            # Plot ECDF for each group
            for i, group in enumerate(groups):
                # Sort values
                sorted_values = np.sort(group.values)
                # Calculate ECDF
                ecdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
                
                # Plot ECDF
                ax9.step(sorted_values, ecdf, where='post', label=group_stats[i]['name'])
                
                # Add vertical line at median
                ax9.axvline(x=group_stats[i]['median'], color=PASTEL_COLORS[i % len(PASTEL_COLORS)], linestyle='--', alpha=0.5)
            
            # Add labels and title
            ax9.set_xlabel('Value')
            ax9.set_ylabel('Cumulative Probability')
            ax9.set_title('Empirical Cumulative Distribution Functions')
            ax9.legend()
            ax9.grid(True, linestyle='--', alpha=0.5)
            
            # Add test statistics
            ax9.text(0.02, 0.98, f"H({df}) = {statistic:.3f}, p = {p_value:.4f}", transform=ax9.transAxes, 
                    verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
            
            fig9.tight_layout()
            figures['ecdf_plot'] = fig_to_svg(fig9)
        except Exception as e:
            figures['ecdf_plot_error'] = str(e)
        
        return {
            'test': 'Kruskal-Wallis Test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'df': int(df),
            'eta_squared': float(eta_squared),
            'epsilon_squared': float(epsilon_squared),
            'effect_magnitude': effect_magnitude,
            'group_stats': group_stats,
            'post_hoc': post_hoc,
            'assumptions': assumptions,
            'interpretation': interpretation,
            'additional_statistics': additional_stats,
            'figures': figures
        }
    except Exception as e:
        return {
            'test': 'Kruskal-Wallis Test',
            'statistic': None,
            'p_value': None,
            'significant': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }