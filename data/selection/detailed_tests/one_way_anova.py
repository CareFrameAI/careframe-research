import traceback
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import io
from typing import Dict, List, Any
from data.selection.detailed_tests.formatting import fig_to_svg, PASTEL_COLORS, PASTEL_CMAP


def one_way_anova(groups: List[pd.Series], alpha: float) -> Dict[str, Any]:
    """Performs a one-way ANOVA with comprehensive statistics and assumption checks."""
    try:
        if len(groups) < 2:
            return {
                'test': "One-Way ANOVA",
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
                'mean': float(group.mean()),
                'std': float(group.std(ddof=1)),
                'min': float(group.min()),
                'max': float(group.max()),
                'name': f'Group {i+1}'
            })
            
        # Set group names based on Series names if available
        for i, group in enumerate(groups):
            if hasattr(group, 'name') and group.name is not None:
                group_stats[i]['name'] = str(group.name)
        
        from scipy import stats

        # Perform ANOVA
        groups_arrays = [group.values for group in groups]
        statistic, p_value = stats.f_oneway(*groups_arrays)
        
        # Calculate degrees of freedom
        df_between = len(groups) - 1
        df_within = sum(len(group) - 1 for group in groups)
        df_total = df_between + df_within
        
        # Calculate sum of squares
        grand_mean = np.mean([g.mean() for g in groups])
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
        ss_within = sum(sum((x - g.mean())**2 for x in g) for g in groups)
        ss_total = ss_between + ss_within
        
        # Calculate mean squares
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        
        # Calculate effect size (eta-squared)
        eta_squared = ss_between / ss_total
        
        # Calculate partial eta-squared
        partial_eta_squared = ss_between / (ss_between + ss_within)
        
        # Calculate omega-squared (less biased than eta-squared)
        if statistic > 1:
            omega_squared = (df_between * (statistic - 1)) / (df_between * (statistic - 1) + sum(len(g) for g in groups))
        else:
            omega_squared = 0
        
        # Determine magnitude of effect size
        if partial_eta_squared < 0.01:
            effect_magnitude = "Negligible"
        elif partial_eta_squared < 0.06:
            effect_magnitude = "Small"
        elif partial_eta_squared < 0.14:
            effect_magnitude = "Medium"
        else:
            effect_magnitude = "Large"
            
        # Perform post-hoc tests if ANOVA is significant
        post_hoc = {}
        if p_value < alpha:
            # Tukey's HSD test
            try:
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                
                # Prepare data for Tukey's test
                all_data = np.concatenate(groups_arrays)
                group_labels = np.concatenate([[i] * len(group) for i, group in enumerate(groups)])
                
                # Perform Tukey's test
                tukey_result = pairwise_tukeyhsd(all_data, group_labels, alpha=alpha)
                
                # Extract results
                tukey_data = []
                for i, row in enumerate(tukey_result.summary().data[1:]):
                    group1 = int(row[0])
                    group2 = int(row[1])
                    mean_diff = float(row[2])
                    p_adj = float(row[3])
                    ci_lower = float(row[4])
                    ci_upper = float(row[5])
                    reject = row[6] == 'True'
                    
                    tukey_data.append({
                        'group1': group_stats[group1]['name'],
                        'group2': group_stats[group2]['name'],
                        'mean_diff': mean_diff,
                        'p_value': p_adj,
                        'significant': reject,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper
                    })
                    
                post_hoc['tukey_hsd'] = tukey_data
            except Exception as e:
                post_hoc['tukey_hsd'] = {'error': f'Could not perform Tukey\'s HSD test: {str(e)}'}
            
            # Add Bonferroni correction
            try:
                from statsmodels.stats.multicomp import pairwise_ttests
                
                # Perform pairwise t-tests with Bonferroni correction
                bonferroni_result = []
                n_comparisons = len(groups) * (len(groups) - 1) // 2
                
                for i in range(len(groups)):
                    for j in range(i+1, len(groups)):
                        t_stat, p_val = stats.ttest_ind(groups[i], groups[j], equal_var=True)
                        p_adj = min(p_val * n_comparisons, 1.0)  # Bonferroni correction
                        mean_diff = groups[i].mean() - groups[j].mean()
                        
                        # Calculate confidence interval
                        pooled_sd = np.sqrt(((len(groups[i])-1) * groups[i].var() + 
                                        (len(groups[j])-1) * groups[j].var()) / 
                                       (len(groups[i]) + len(groups[j]) - 2))
                        
                        se = pooled_sd * np.sqrt(1/len(groups[i]) + 1/len(groups[j]))
                        df = len(groups[i]) + len(groups[j]) - 2
                        t_crit = stats.t.ppf(1 - alpha/2, df)
                        ci_lower = mean_diff - t_crit * se
                        ci_upper = mean_diff + t_crit * se
                        
                        bonferroni_result.append({
                            'group1': group_stats[i]['name'],
                            'group2': group_stats[j]['name'],
                            'mean_diff': float(mean_diff),
                            'p_value': float(p_val),
                            'p_adjusted': float(p_adj),
                            'significant': p_adj < alpha,
                            'ci_lower': float(ci_lower),
                            'ci_upper': float(ci_upper)
                        })
                
                post_hoc['bonferroni'] = bonferroni_result
            except Exception as e:
                post_hoc['bonferroni'] = {'error': f'Could not perform Bonferroni tests: {str(e)}'}
                
            # Add Scheffe test
            try:
                scheffe_result = []
                
                for i in range(len(groups)):
                    for j in range(i+1, len(groups)):
                        mean_diff = groups[i].mean() - groups[j].mean()
                        
                        # Calculate Scheffe critical value
                        pooled_sd = np.sqrt(((len(groups[i])-1) * groups[i].var() + 
                                        (len(groups[j])-1) * groups[j].var()) / 
                                       (len(groups[i]) + len(groups[j]) - 2))
                        
                        se = pooled_sd * np.sqrt(1/len(groups[i]) + 1/len(groups[j]))
                        
                        # Scheffe's critical value = sqrt((k-1) * F_critical) where k is number of groups
                        f_crit = stats.f.ppf(1-alpha, df_between, df_within)
                        scheffe_crit = np.sqrt((len(groups)-1) * f_crit)
                        
                        # Calculate test statistic
                        scheffe_stat = abs(mean_diff) / se
                        
                        # Calculate p-value based on F distribution
                        # F = t^2 for this comparison
                        f_val = scheffe_stat**2
                        p_val = 1 - stats.f.cdf(f_val/(len(groups)-1), len(groups)-1, df_within)
                        
                        # Calculate CI
                        ci_lower = mean_diff - scheffe_crit * se
                        ci_upper = mean_diff + scheffe_crit * se
                        
                        scheffe_result.append({
                            'group1': group_stats[i]['name'],
                            'group2': group_stats[j]['name'],
                            'mean_diff': float(mean_diff),
                            'test_statistic': float(scheffe_stat),
                            'critical_value': float(scheffe_crit),
                            'p_value': float(p_val),
                            'significant': scheffe_stat > scheffe_crit,
                            'ci_lower': float(ci_lower),
                            'ci_upper': float(ci_upper)
                        })
                
                post_hoc['scheffe'] = scheffe_result
            except Exception as e:
                post_hoc['scheffe'] = {'error': f'Could not perform Scheffe tests: {str(e)}'}
                
        # Test assumptions
        assumptions = {}
        
        # Test normality for each group
        from data.assumptions.tests import NormalityTest
        normality_test = NormalityTest()
        
        for i, group in enumerate(groups):
            group_name = group_stats[i]['name']
            normality_result = normality_test.run_test(data=group)
            assumptions[f'normality_{group_name}'] = normality_result
        
        # Test homogeneity of variance with improved error handling
        try:
            # Extra check on group counts
            non_empty_groups = [g for g in groups if len(g) > 0]
            if len(non_empty_groups) < 2:
                # Not enough non-empty groups for comparison
                from data.assumptions.tests import AssumptionResult
                assumptions['homogeneity'] = {
                    'result': AssumptionResult.NOT_APPLICABLE,
                    'details': f"Need at least two non-empty groups for homogeneity test. Found {len(non_empty_groups)} non-empty groups.",
                    'warnings': [f"Homogeneity test skipped - only {len(non_empty_groups)} non-empty group(s) after filtering"]
                }
            else:
                # We have 2+ groups, DIRECTLY RUN Levene's test - NEED TO REVIEW THIS - DO NOT DELETE
                from data.assumptions.tests import AssumptionResult
                from scipy import stats
                import matplotlib.pyplot as plt
                from data.selection.detailed_tests.formatting import fig_to_svg
                
                # Get group names
                group_names = [stat['name'] for stat in group_stats]
                print(f"DEBUG: Group names: {group_names}")
                
                # DIRECT IMPLEMENTATION - Skip HomogeneityOfVarianceTest
                # Run Levene's test directly with group arrays
                group_arrays = [g.values for g in non_empty_groups]
                
                # Directly run Levene's test
                lev_stat, lev_p_value = stats.levene(*group_arrays)
                
                # Create results dictionary manually
                group_variances = {group_names[i]: float(g.var()) for i, g in enumerate(non_empty_groups) if i < len(group_names)}
                
                # Determine result
                result = AssumptionResult.PASSED if lev_p_value > alpha else AssumptionResult.FAILED
                
                details = (f"Levene's test for homogeneity of variance: p={lev_p_value:.4f}. "
                          f"Testing across {len(non_empty_groups)} groups: {', '.join(group_names[:len(non_empty_groups)])}")
                
                warnings = []
                if result == AssumptionResult.FAILED:
                    warnings.append("Heterogeneous variances may affect the validity of parametric tests")
                
                # Create simple boxplot for visualization
                fig = plt.figure(figsize=(10, 6))
                fig.patch.set_alpha(0.0)
                ax = fig.add_subplot(111)
                ax.patch.set_alpha(0.0)
                
                ax.boxplot(group_arrays, labels=group_names[:len(non_empty_groups)])
                ax.set_title('Boxplot by Group')
                ax.set_ylabel('Value')
                ax.set_xlabel('Group')
                
                # Create our own homogeneity result
                homogeneity_result = {
                    'result': result,
                    'statistic': float(lev_stat),
                    'p_value': float(lev_p_value),
                    'details': details,
                    'test_used': "Levene's test",
                    'group_variances': group_variances,
                    'warnings': warnings,
                    'figures': {
                        'boxplot': fig_to_svg(fig)
                    }
                }
                
                # Add to assumptions
                assumptions['homogeneity'] = homogeneity_result
                
                # Add extra info for 2-group case
                if len(non_empty_groups) == 2:
                    # Calculate variance ratio for context
                    var_ratio = max(non_empty_groups[0].var(), non_empty_groups[1].var()) / min(non_empty_groups[0].var(), non_empty_groups[1].var())
                    homogeneity_result['variance_ratio'] = float(var_ratio)
                    
                    # Add information about 2-group variance testing
                    homogeneity_result['details'] = (f"Variance ratio = {var_ratio:.2f}. " +
                                                     f"For 2 groups, this test checks equal variances " +
                                                     f"(important for t-tests and ANOVA). " +
                                                     f"{homogeneity_result['details']}")
                    
                    # Add helpful context about the result
                    if result == AssumptionResult.FAILED:
                        homogeneity_result['warnings'].append(
                            "With unequal variances, consider Welch's t-test/ANOVA instead of the standard version."
                        )
        except Exception as e:
            # Emergency fallback - manually calculate variance ratio for 2 groups
            from data.assumptions.tests import AssumptionResult
            
            if len(groups) >= 2 and all(len(g) > 0 for g in groups[:2]):
                var_ratio = max(groups[0].var(), groups[1].var()) / min(groups[0].var(), groups[1].var())
                is_homogeneous = var_ratio < 4  # Rule of thumb: variance ratio < 4 is acceptable
                
                assumptions['homogeneity'] = {
                    'result': AssumptionResult.PASSED if is_homogeneous else AssumptionResult.WARNING,
                    'details': f"Homogeneity test implementation failed, but manual variance ratio check = {var_ratio:.2f}",
                    'warnings': [f"Variance ratio = {var_ratio:.2f}. Original error: {str(e)}"]
                }
            else:
                # Cannot calculate variance ratio, provide useful error
                assumptions['homogeneity'] = {
                    'result': AssumptionResult.WARNING,
                    'details': f"Homogeneity test failed. Groups: {len(groups)}. Error: {str(e)}",
                    'warnings': [f"Unable to test homogeneity of variance. Check if both groups have sufficient data."]
                }
        
        # Test for sample size adequacy for each group
        from data.assumptions.tests import SampleSizeTest
        sample_size_test = SampleSizeTest()
        
        for i, group in enumerate(groups):
            group_name = group_stats[i]['name']
            sample_size_result = sample_size_test.run_test(data=group, min_recommended=30)
            assumptions[f'sample_size_{group_name}'] = sample_size_result
        
        # Test for outliers in each group
        from data.assumptions.tests import OutlierTest
        outlier_test = OutlierTest()
        
        for i, group in enumerate(groups):
            group_name = group_stats[i]['name']
            outlier_result = outlier_test.run_test(data=group)
            assumptions[f'outliers_{group_name}'] = outlier_result
        
        # Test for independence
        from data.assumptions.tests import IndependenceTest
        independence_test = IndependenceTest()
        
        # For ANOVA, we could test residuals for independence
        all_values = np.concatenate([group.values for group in groups])
        all_predicted = np.concatenate([[group.mean()] * len(group) for group in groups])
        residuals = all_values - all_predicted
        independence_result = independence_test.run_test(data=residuals)
        assumptions['independence'] = independence_result
        
        # Add sphericity test (relevant for repeated measures ANOVA)
        # Commented out as standard one-way ANOVA doesn't assume sphericity
        # Only include if your data might have repeated measures
        # from data.assumptions.tests import SphericityTest
        # if repeated_measures_data_available:
        #     sphericity_test = SphericityTest()
        #     sphericity_result = sphericity_test.run_test(
        #         data=df_repeated_measures, 
        #         subject_id='subject_id', 
        #         within_factor='time_point', 
        #         outcome='value'
        #     )
        #     assumptions['sphericity'] = sphericity_result
        
        # Test for linearity between DV and group means
        # Conceptually this is checking if the "effect" of group is linear
        from data.assumptions.tests import LinearityTest
        linearity_test = LinearityTest()
        
        # Create indices for group membership and test against values
        group_indices = np.concatenate([[i] * len(group) for i, group in enumerate(groups)])
        all_values = np.concatenate([group.values for group in groups])
        linearity_result = linearity_test.run_test(x=group_indices, y=all_values)
        assumptions['linearity'] = linearity_result
        
        # Test for influential points
        from data.assumptions.tests import InfluentialPointsTest
        influential_test = InfluentialPointsTest()
        
        # Create a simple design matrix for the ANOVA
        X = pd.get_dummies(np.concatenate([[i] * len(group) for i, group in enumerate(groups)]))
        X = X.iloc[:, :-1]  # Remove one column to avoid perfect collinearity
        X['intercept'] = 1  # Add intercept
        
        all_values = np.concatenate([group.values for group in groups])
        all_predicted = np.concatenate([[group.mean()] * len(group) for group in groups])
        all_residuals = all_values - all_predicted
        
        # Calculate leverage (hat matrix diagonal)
        leverage = np.zeros(len(all_values))
        try:
            leverage = np.diagonal(X.dot(np.linalg.inv(X.T.dot(X))).dot(X.T))
        except:
            # If matrix inversion fails, use a simpler approach
            leverage = np.array([1/len(group) for group in groups for _ in range(len(group))])
        
        influential_result = influential_test.run_test(
            residuals=all_residuals,
            leverage=leverage,
            fitted=all_predicted,
            X=X
        )
        assumptions['influential_points'] = influential_result
        
        # Test distribution fit (are the residuals normally distributed?)
        from data.assumptions.tests import DistributionFitTest
        distribution_test = DistributionFitTest()
        
        distribution_result = distribution_test.run_test(data=all_residuals, distribution='normal')
        assumptions['residuals_distribution'] = distribution_result
        
        # Import AssumptionResult at the point of use
        from data.assumptions.tests import AssumptionResult
        
        # Simple power analysis using direct statsmodels implementation
        try:
            from statsmodels.stats.power import FTestAnovaPower
            
            # Convert partial eta squared to Cohen's f effect size
            f_squared = partial_eta_squared / (1 - partial_eta_squared)
            cohen_f = np.sqrt(f_squared)
            
            # Total sample size
            total_n = sum(len(group) for group in groups)
            
            # Calculate power directly using statsmodels
            power_calculator = FTestAnovaPower()
            observed_power = power_calculator.power(
                effect_size=cohen_f,
                nobs=total_n,
                alpha=alpha,
                k_groups=len(groups)
            )
            
            # Create our own power analysis result dictionary
            assumptions['power_analysis'] = {
                'result': AssumptionResult.PASSED if observed_power >= 0.8 else AssumptionResult.WARNING,
                'statistic': float(observed_power),
                'details': f"Power analysis for effect size f = {cohen_f:.3f} with n = {total_n}",
                'warnings': [] if observed_power >= 0.8 else ["Study may be underpowered."],
                'observed_power': float(observed_power),
                'effect_size_f': float(cohen_f),
                'total_n': int(total_n)
            }
            
            # Include required sample sizes without using the PowerAnalysisTest
            try:
                # Manually calculate required sample size for 80% power
                n_80 = power_calculator.solve_power(
                    effect_size=cohen_f,
                    power=0.8,
                    alpha=alpha,
                    k_groups=len(groups)
                )
                
                if not np.isnan(n_80):
                    assumptions['power_analysis']['n_for_80_power'] = int(np.ceil(n_80))
                    
                    # Add sample sizes for 90% and 95% power if 80% worked
                    n_90 = power_calculator.solve_power(
                        effect_size=cohen_f,
                        power=0.9,
                        alpha=alpha,
                        k_groups=len(groups)
                    )
                    
                    n_95 = power_calculator.solve_power(
                        effect_size=cohen_f,
                        power=0.95,
                        alpha=alpha,
                        k_groups=len(groups)
                    )
                    
                    if not np.isnan(n_90):
                        assumptions['power_analysis']['n_for_90_power'] = int(np.ceil(n_90))
                    
                    if not np.isnan(n_95):
                        assumptions['power_analysis']['n_for_95_power'] = int(np.ceil(n_95))
            except Exception as e:
                # Just log the error in calculating required sample sizes
                assumptions['power_analysis']['sample_size_error'] = str(e)
                
        except Exception as e:
            # If entire power analysis fails, add a placeholder with error info
            assumptions['power_analysis'] = {
                'result': AssumptionResult.WARNING,
                'error': str(e),
                'details': "Power analysis failed but continuing with other tests.",
                'warnings': ["Power analysis could not be completed: " + str(e)]
            }
        
        # Test for model specification
        from data.assumptions.tests import ModelSpecificationTest
        model_spec_test = ModelSpecificationTest()
        
        # Create a design matrix for the ANOVA model
        X = pd.get_dummies(np.concatenate([[i] * len(group) for i, group in enumerate(groups)]))
        
        # For model specification test, we need residuals and fitted values
        model_spec_result = model_spec_test.run_test(
            residuals=all_residuals, 
            fitted=all_predicted,
            X=X
        )
        
        assumptions['model_specification'] = model_spec_result
        
        # Initialize additional_stats dictionary BEFORE the interpretation section
        additional_stats = {}
        
        # Create interpretation
        interpretation = f"One-way ANOVA comparing {len(groups)} groups.\n\n"
        
        # Basic results
        interpretation += f"F({df_between}, {df_within}) = {statistic:.3f}, p = {p_value:.5f}\n"
        interpretation += f"Effect size: η² = {eta_squared:.3f}, partial η² = {partial_eta_squared:.3f} ({effect_magnitude.lower()} effect)\n\n"
        
        # Group statistics
        interpretation += "Group statistics:\n"
        for i, stats in enumerate(group_stats):
            interpretation += f"- {stats['name']}: n = {stats['n']}, mean = {stats['mean']:.3f}, SD = {stats['std']:.3f}\n"
            
        # Assumptions
        interpretation += "\nAssumption tests:\n"
        
        # Normality
        interpretation += "- Normality: "
        normality_keys = [key for key in assumptions.keys() if key.startswith('normality_')]
        if normality_keys:
            # Check if any group violates normality
            violated_groups = []
            for key in normality_keys:
                group_name = key.replace('normality_', '')
                if assumptions[key]['result'].value != 'passed':
                    violated_groups.append(group_name)
            
            if not violated_groups:
                interpretation += "Assumption satisfied for all groups.\n"
            else:
                interpretation += f"Assumption violated for {', '.join(violated_groups)}. "
                
                # Check if all groups have large sample sizes
                all_large = all(len(group) >= 30 for group in groups)
                if all_large:
                    interpretation += "However, due to large sample sizes, ANOVA should be robust to violations of normality (Central Limit Theorem).\n"
                else:
                    interpretation += "Consider using a non-parametric alternative like the Kruskal-Wallis test.\n"
        else:
            interpretation += "Not tested.\n"
        
        # Homogeneity of variance
        interpretation += "- Homogeneity of variance: "
        if 'homogeneity' in assumptions and 'test_used' in assumptions['homogeneity']:
            hov_result = assumptions['homogeneity']
            test_used = hov_result.get('test_used', 'Homogeneity test')
            
            # Safely access p_value and handle None case
            p_value_hov = hov_result.get('p_value')
            if p_value_hov is not None:
                interpretation += f"{test_used} test, p = {p_value_hov:.5f}, "
            else:
                interpretation += f"{test_used} test, p = N/A, "
            
            if hov_result['result'].value == 'passed':
                interpretation += "assumption satisfied.\n"
            else:
                interpretation += f"assumption potentially violated. Consider using Welch's ANOVA which does not assume equal variances.\n"
        else:
            interpretation += "Not tested.\n"
        
        # Sample size
        interpretation += "- Sample size: "
        sample_size_keys = [key for key in assumptions.keys() if key.startswith('sample_size_')]
        if sample_size_keys:
            # Check if any group has inadequate sample size
            small_groups = []
            for key in sample_size_keys:
                group_name = key.replace('sample_size_', '')
                if assumptions[key]['result'].value != 'passed':
                    small_groups.append(group_name)
            
            if not small_groups:
                interpretation += "Sample sizes are adequate for all groups.\n"
            else:
                interpretation += f"Sample sizes may be too small for {', '.join(small_groups)}. Results should be interpreted with caution.\n"
        else:
            interpretation += "Not tested.\n"
        
        # Outliers
        interpretation += "- Outliers: "
        outlier_keys = [key for key in assumptions.keys() if key.startswith('outliers_')]
        if outlier_keys:
            # Check if any group has outliers
            groups_with_outliers = []
            for key in outlier_keys:
                group_name = key.replace('outliers_', '')
                if assumptions[key]['result'].value != 'passed':
                    groups_with_outliers.append(group_name)
            
            if not groups_with_outliers:
                interpretation += "No significant outliers detected in any group.\n"
            else:
                interpretation += f"Potential outliers detected in {', '.join(groups_with_outliers)}. "
                interpretation += "Consider examining these outliers and potentially removing them if they represent measurement errors.\n"
        else:
            interpretation += "Not tested.\n"
        
        # Independence
        interpretation += "- Independence: "
        if 'independence' in assumptions:
            independence_result = assumptions['independence']
            if independence_result['result'].value == 'passed':
                interpretation += "Residuals appear to be independent.\n"
            else:
                interpretation += f"Potential issues with independence detected. {independence_result.get('message', '')}\n"
        else:
            interpretation += "Not tested.\n"
        
        # Linearity
        interpretation += "- Linearity: "
        if 'linearity' in assumptions:
            linearity_result = assumptions['linearity']
            if linearity_result['result'].value == 'passed':
                interpretation += "Relationship between group membership and outcome appears to be adequately linear.\n"
            else:
                interpretation += f"Potential issues with linearity detected. {linearity_result.get('details', '')}\n"
        else:
            interpretation += "Not tested.\n"
        
        # Influential Points
        interpretation += "- Influential Points: "
        if 'influential_points' in assumptions:
            influential_result = assumptions['influential_points']
            if influential_result['result'].value == 'passed':
                interpretation += "No highly influential points detected.\n"
            else:
                interpretation += f"Potential influential points detected that may be affecting the results. Consider examining these points further.\n"
        else:
            interpretation += "Not tested.\n"
        
        # Residuals distribution
        interpretation += "- Residual Distribution: "
        if 'residuals_distribution' in assumptions:
            dist_result = assumptions['residuals_distribution']
            if dist_result['result'].value == 'passed':
                interpretation += "Residuals appear to be normally distributed.\n"
            else:
                interpretation += f"Non-normal distribution of residuals detected, which may affect inference. {dist_result.get('details', '')}\n"
        else:
            interpretation += "Not tested.\n"
        
        # Conclusion
        interpretation += f"\nConclusion: There is {'a statistically significant' if p_value < alpha else 'no statistically significant'} "
        interpretation += f"difference between the groups (p = {p_value:.5f}). "
        
        if p_value < alpha:
            interpretation += f"The {effect_magnitude.lower()} effect size (partial η² = {partial_eta_squared:.3f}) suggests that "
            interpretation += f"group membership explains approximately {partial_eta_squared*100:.1f}% of the variance in the outcome.\n\n"
            
            # Post-hoc results
            if 'tukey_hsd' in post_hoc and isinstance(post_hoc['tukey_hsd'], list):
                interpretation += "Post-hoc comparisons (Tukey's HSD):\n"
                
                significant_pairs = [pair for pair in post_hoc['tukey_hsd'] if pair['significant']]
                if significant_pairs:
                    for pair in significant_pairs:
                        interpretation += f"- {pair['group1']} vs {pair['group2']}: Mean difference = {pair['mean_diff']:.3f}, "
                        interpretation += f"p = {pair['p_value']:.5f}, 95% CI [{pair['ci_lower']:.3f}, {pair['ci_upper']:.3f}]\n"
                else:
                    interpretation += "- No significant pairwise differences were found despite the significant overall F-test.\n"
                    interpretation += "  This can occur with complex patterns of differences or limited statistical power.\n"
        
        # Add recommendations based on assumption violations
        interpretation += "\nRecommendations based on assumption tests:\n"
        
        # Check normality and homogeneity to recommend alternative tests
        normality_violated = any(assumptions[key]['result'].value != 'passed' for key in normality_keys) if normality_keys else False
        homogeneity_violated = (assumptions.get('homogeneity', {}).get('result', {}).value != 'passed') if 'homogeneity' in assumptions else False
        
        if normality_violated and not homogeneity_violated:
            interpretation += "- Consider using Kruskal-Wallis test (non-parametric) due to normality violations.\n"
            if 'kruskal_wallis' in additional_stats and 'error' not in additional_stats['kruskal_wallis']:
                kw_result = additional_stats['kruskal_wallis']
                interpretation += f"  Kruskal-Wallis result: H({len(groups)-1}) = {kw_result['statistic']:.3f}, p = {kw_result['p_value']:.5f}\n"
        
        if homogeneity_violated:
            interpretation += "- Consider using Welch's ANOVA due to unequal variances.\n"
            if 'welch_anova' in additional_stats and 'error' not in additional_stats['welch_anova']:
                welch_result = additional_stats['welch_anova']
                interpretation += f"  Welch's ANOVA result: F({welch_result['df_numerator']:.1f}, {welch_result['df_denominator']:.1f}) = {welch_result['statistic']:.3f}, p = {welch_result['p_value']:.5f}\n"
        
        # Power analysis recommendation
        if 'power_analysis' in additional_stats and 'error' not in additional_stats['power_analysis']:
            power_result = additional_stats['power_analysis']
            if power_result['observed_power'] < 0.8:
                interpretation += f"- Current study has limited power ({power_result['observed_power']:.2f}). For detecting this effect size, a total sample size of {power_result['n_for_80_power']} would be recommended for 80% power.\n"
        
        # Outlier recommendation
        outlier_violated = any(assumptions[key]['result'].value != 'passed' for key in outlier_keys) if outlier_keys else False
        if outlier_violated:
            interpretation += "- Examine outliers and consider analyses both with and without outliers to assess their impact.\n"
        
        # Add a final note about robust alternatives if multiple assumptions are violated
        if (normality_violated and homogeneity_violated) or outlier_violated:
            interpretation += "- With multiple assumption violations, consider robust methods like bootstrapping or permutation tests.\n"
            if 'bootstrap' in additional_stats and 'error' not in additional_stats['bootstrap']:
                interpretation += "  Bootstrap confidence intervals for means are provided in the additional statistics section.\n"
        
        # Calculate additional statistics
        additional_stats = {}
        
        # 1. Welch's ANOVA (robust to unequal variances)
        try:
            # Calculate Welch's ANOVA
            numerator = 0
            denominator = 0
            
            weights = [len(group) / group.var() for group in groups]
            weighted_means = [group.mean() * (len(group) / group.var()) for group in groups]
            
            grand_mean_welch = sum(weighted_means) / sum(weights)
            
            numerator = sum(w * (m - grand_mean_welch)**2 for w, m in zip(weights, [g.mean() for g in groups]))
            numerator /= (len(groups) - 1)
            
            denominator = sum((1 - (len(group) / sum(len(g) for g in groups))) * (group.var() / len(group)) for group in groups)
            denominator /= (len(groups)**2 - 1)
            
            welch_statistic = numerator / denominator
            
            # Calculate degrees of freedom
            df_num = len(groups) - 1
            
            sum_weights = 0
            sum_squared_weights = 0
            for group in groups:
                weight = (1 - len(group) / sum(len(g) for g in groups)) * (group.var() / len(group))
                sum_weights += weight
                sum_squared_weights += weight**2
            
            df_denom = (df_num**2 - 1) * (3 * sum_squared_weights / sum_weights**2)
            
            # Calculate p-value
            welch_p_value = 1 - stats.f.cdf(welch_statistic, df_num, df_denom)
            
            additional_stats['welch_anova'] = {
                'statistic': float(welch_statistic),
                'p_value': float(welch_p_value),
                'df_numerator': float(df_num),
                'df_denominator': float(df_denom),
                'significant': welch_p_value < alpha,
                'agrees_with_anova': (welch_p_value < alpha) == (p_value < alpha)
            }
        except Exception as e:
            additional_stats['welch_anova'] = {
                'error': str(e)
            }
        
        # 2. Kruskal-Wallis test (non-parametric alternative)
        try:
            kw_statistic, kw_p_value = stats.kruskal(*groups_arrays)
            
            additional_stats['kruskal_wallis'] = {
                'statistic': float(kw_statistic),
                'p_value': float(kw_p_value),
                'df': len(groups) - 1,
                'significant': kw_p_value < alpha,
                'agrees_with_anova': (kw_p_value < alpha) == (p_value < alpha)
            }
            
            # If Kruskal-Wallis is significant, add pairwise Mann-Whitney tests
            if kw_p_value < alpha:
                mw_results = []
                
                for i in range(len(groups)):
                    for j in range(i+1, len(groups)):
                        mw_stat, mw_p = stats.mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                        
                        # Apply Bonferroni correction
                        n_comparisons = len(groups) * (len(groups) - 1) // 2
                        mw_p_adj = min(mw_p * n_comparisons, 1.0)
                        
                        mw_results.append({
                            'group1': group_stats[i]['name'],
                            'group2': group_stats[j]['name'],
                            'statistic': float(mw_stat),
                            'p_value': float(mw_p),
                            'p_adjusted': float(mw_p_adj),
                            'significant': mw_p_adj < alpha
                        })
                
                additional_stats['mann_whitney_tests'] = mw_results
        except Exception as e:
            additional_stats['kruskal_wallis'] = {
                'error': str(e)
            }
        
        # 3. Robust statistics (trimmed means)
        try:
            from scipy.stats import trim_mean
            
            # Calculate 10% trimmed means
            trimmed_stats = []
            
            for i, group in enumerate(groups):
                trimmed_mean = trim_mean(group, 0.1)
                
                # Calculate trimmed standard deviation manually
                sorted_data = np.sort(group)
                trim_amount = int(0.1 * len(group))
                if trim_amount > 0:
                    trimmed_data = sorted_data[trim_amount:-trim_amount]
                else:
                    trimmed_data = sorted_data
                
                trimmed_sd = np.std(trimmed_data, ddof=1) if len(trimmed_data) > 1 else 0
                
                trimmed_stats.append({
                    'group': group_stats[i]['name'],
                    'trimmed_mean': float(trimmed_mean),
                    'trimmed_sd': float(trimmed_sd)
                })
            
            additional_stats['trimmed_means'] = trimmed_stats
        except Exception as e:
            additional_stats['trimmed_means'] = {
                'error': str(e)
            }
        
        # 4. Power analysis
        try:
            from statsmodels.stats.power import FTestAnovaPower
            
            # Calculate power for the observed effect size
            power_analysis = FTestAnovaPower()
            
            # Convert partial eta squared to Cohen's f effect size
            # f^2 = η²/(1-η²)
            f_squared = partial_eta_squared / (1 - partial_eta_squared)
            cohen_f = np.sqrt(f_squared)
            
            # Calculate observed power
            total_n = sum(len(group) for group in groups)
            observed_power = power_analysis.power(
                effect_size=cohen_f,
                nobs=total_n,
                alpha=alpha,
                k_groups=len(groups)
            )
            
            # Calculate required sample size for different power levels
            n_for_80_power = power_analysis.solve_power(
                effect_size=cohen_f,
                power=0.8,
                alpha=alpha,
                k_groups=len(groups)
            )
            
            n_for_90_power = power_analysis.solve_power(
                effect_size=cohen_f,
                power=0.9,
                alpha=alpha,
                k_groups=len(groups)
            )
            
            n_for_95_power = power_analysis.solve_power(
                effect_size=cohen_f,
                power=0.95,
                alpha=alpha,
                k_groups=len(groups)
            )
            
            additional_stats['power_analysis'] = {
                'effect_size_f': float(cohen_f),
                'observed_power': float(observed_power),
                'total_n': int(total_n),
                'n_for_80_power': int(np.ceil(n_for_80_power)),
                'n_for_90_power': int(np.ceil(n_for_90_power)),
                'n_for_95_power': int(np.ceil(n_for_95_power))
            }
        except Exception as e:
            additional_stats['power_analysis'] = {
                'error': str(e)
            }
        
        # 5. Bootstrapped confidence intervals for group means
        try:
            n_bootstrap = 2000
            bootstrap_results = []
            
            for i, group in enumerate(groups):
                bootstrap_means = []
                
                # Generate bootstrap samples
                for _ in range(n_bootstrap):
                    # Sample with replacement
                    indices = np.random.choice(range(len(group)), size=len(group), replace=True)
                    bootstrap_means.append(np.mean(group.iloc[indices]))
                
                # Calculate bootstrap CI
                boot_ci_lower = np.percentile(bootstrap_means, 100 * alpha/2)
                boot_ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha/2))
                
                bootstrap_results.append({
                    'group': group_stats[i]['name'],
                    'mean': float(group.mean()),
                    'bootstrap_mean': float(np.mean(bootstrap_means)),
                    'bootstrap_se': float(np.std(bootstrap_means)),
                    'ci_lower': float(boot_ci_lower),
                    'ci_upper': float(boot_ci_upper)
                })
            
            additional_stats['bootstrap'] = {
                'n_samples': n_bootstrap,
                'results': bootstrap_results
            }
        except Exception as e:
            additional_stats['bootstrap'] = {
                'error': str(e)
            }
        
        # Generate figures
        figures = {}
        
        # Figure 1: Box plots with individual data points
        try:
            fig1, ax1 = plt.subplots(figsize=(12, 8))
            fig1.patch.set_alpha(0.0)
            ax1.patch.set_alpha(0.0)
            
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
            sns.boxplot(x='Group', y='Value', data=plot_data, ax=ax1, width=0.5, 
                      palette=PASTEL_COLORS, saturation=0.8)
            sns.stripplot(x='Group', y='Value', data=plot_data, ax=ax1, 
                        color='black', alpha=0.5, jitter=True, size=4)
            
            # Add group means as red markers
            means = [stat['mean'] for stat in group_stats]
            x_positions = range(len(groups))
            ax1.plot(x_positions, means, 'ro', markersize=10, label='Mean')
            
            # Add test statistics
            stats_text = f"ANOVA: F({df_between}, {df_within}) = {statistic:.3f}, p = {p_value:.4f}"
            if p_value < alpha:
                stats_text += " *"
            ax1.text(0.5, 0.02, stats_text, transform=ax1.transAxes, 
                    horizontalalignment='center', bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
            
            # Add labels and title
            ax1.set_title('Distribution of Values by Group')
            ax1.set_ylabel('Value')
            ax1.legend()
            
            fig1.tight_layout()
            figures['boxplot'] = fig_to_svg(fig1)
        except Exception as e:
            figures['boxplot_error'] = str(e)
        
        # Figure 2: Means plot with error bars and significance indicators
        try:
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            fig2.patch.set_alpha(0.0)
            ax2.patch.set_alpha(0.0)
            
            # Plot means with error bars
            x_positions = range(len(groups))
            means = [stat['mean'] for stat in group_stats]
            errors = [stat['std'] / np.sqrt(stat['n']) for stat in group_stats]  # Standard error
            
            bars = ax2.bar(x_positions, means, yerr=errors, capsize=10, 
                         color='lightblue', edgecolor='black', alpha=0.7)
            
            # Add group labels
            ax2.set_xticks(x_positions)
            ax2.set_xticklabels([stat['name'] for stat in group_stats])
            
            # Add overall mean as a horizontal line
            ax2.axhline(y=grand_mean, color='red', linestyle='--', linewidth=2, label='Grand Mean')
            
            # Add post-hoc test results if significant
            if p_value < alpha and 'tukey_hsd' in post_hoc and isinstance(post_hoc['tukey_hsd'], list):
                significant_pairs = [pair for pair in post_hoc['tukey_hsd'] if pair['significant']]
                
                # Calculate the maximum height for the significance bars
                max_height = max(means) + max(errors) * 2
                bar_height = max_height * 0.05  # For spacing between bars
                
                # Plot significance bars
                for i, pair in enumerate(significant_pairs):
                    # Find the indices for this pair
                    idx1 = next(j for j, s in enumerate(group_stats) if s['name'] == pair['group1'])
                    idx2 = next(j for j, s in enumerate(group_stats) if s['name'] == pair['group2'])
                    
                    # Ensure idx1 < idx2 for consistency
                    if idx1 > idx2:
                        idx1, idx2 = idx2, idx1
                    
                    # Calculate y position for this comparison line
                    y_pos = max_height + i * bar_height
                    
                    # Draw the significance line
                    ax2.plot([idx1, idx2], [y_pos, y_pos], 'k-', linewidth=1.5)
                    
                    # Add small vertical lines at the ends
                    ax2.plot([idx1, idx1], [y_pos-bar_height/4, y_pos], 'k-', linewidth=1.5)
                    ax2.plot([idx2, idx2], [y_pos-bar_height/4, y_pos], 'k-', linewidth=1.5)
                    
                    # Add p-value or significance stars
                    if pair['p_value'] < 0.001:
                        sig_text = '***'
                    elif pair['p_value'] < 0.01:
                        sig_text = '**'
                    elif pair['p_value'] < 0.05:
                        sig_text = '*'
                    else:
                        sig_text = f"p={pair['p_value']:.3f}"
                    
                    ax2.text((idx1 + idx2) / 2, y_pos + bar_height/4, sig_text, 
                           ha='center', va='bottom', fontweight='bold')
                
                # Adjust y-axis limit to accommodate the significance bars
                ax2.set_ylim(top=max_height + len(significant_pairs) * bar_height + bar_height * 2)
            
            # Add labels and title
            ax2.set_title('Group Means with Standard Error')
            ax2.set_ylabel('Mean Value')
            ax2.set_xlabel('Group')
            ax2.legend()
            
            # Add ANOVA result
            result_text = f"F({df_between}, {df_within}) = {statistic:.3f}, p = {p_value:.4f}"
            if p_value < alpha:
                result_text += " *"
            result_text += f"\nη² = {eta_squared:.3f}, partial η² = {partial_eta_squared:.3f}"
            
            ax2.text(0.02, 0.98, result_text, transform=ax2.transAxes, 
                    verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
            
            fig2.tight_layout()
            figures['means_plot'] = fig_to_svg(fig2)
        except Exception as e:
            figures['means_plot_error'] = str(e)

        # Figure 3: QQ plots for each group to check normality
        try:
            fig3, axes = plt.subplots(1, len(groups), figsize=(4*len(groups), 5))
            fig3.patch.set_alpha(0.0)
            
            # Handle the case with only one group
            if len(groups) == 1:
                axes = [axes]
            
            # Make each axis background transparent
            for ax in axes:
                ax.patch.set_alpha(0.0)
            
            # Create QQ plot for each group
            for i, group in enumerate(groups):
                # Get group data
                data = group.values
                group_name = group_stats[i]['name']
                
                # Create QQ plot
                stats.probplot(data, dist="norm", plot=axes[i])
                
                # Add title
                axes[i].set_title(f'Q-Q Plot: {group_name}')
                
                # Add Shapiro-Wilk test results if available
                norm_key = f'normality_{group_name}'
                if norm_key in assumptions:
                    normality_result = assumptions[norm_key]
                    shapiro_p = normality_result.get('p_value', 0)
                    test_used = normality_result.get('test_used', 'Normality test')
                    
                    test_text = f"{test_used}: p = {shapiro_p:.4f}"
                    
                    if normality_result['result'].value != 'passed':
                        test_text += "\nNormality rejected"
                    else:
                        test_text += "\nNormality not rejected"
                    
                    axes[i].text(0.05, 0.95, test_text, transform=axes[i].transAxes, 
                               verticalalignment='top', 
                               bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
            
            fig3.tight_layout()
            figures['qq_plots'] = fig_to_svg(fig3)
        except Exception as e:
            figures['qq_plots_error'] = str(e)
        
        # Figure 4: Residuals vs. predicted values
        try:
            # Prepare data for plotting residuals
            all_data = []
            group_nums = []
            
            for i, group in enumerate(groups):
                all_data.extend(group.values)
                group_nums.extend([i] * len(group))
            
            # Calculate residuals
            predicted = [group_stats[i]['mean'] for i in group_nums]
            residuals = [all_data[i] - predicted[i] for i in range(len(all_data))]
            
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            fig4.patch.set_alpha(0.0)
            ax4.patch.set_alpha(0.0)
            
            # Scatter plot of residuals vs. predicted values
            ax4.scatter(predicted, residuals, alpha=0.6)
            
            # Add horizontal line at zero
            ax4.axhline(y=0, color='red', linestyle='-')
            
            # Add labels and title
            ax4.set_xlabel('Predicted Values (Group Means)')
            ax4.set_ylabel('Residuals')
            ax4.set_title('Residuals vs. Predicted Values')
            
            # Add homogeneity of variance test result if available
            if 'homogeneity' in assumptions:
                hov_result = assumptions['homogeneity']
                hov_p = hov_result.get('p_value', 1)
                test_used = hov_result.get('test_used', 'Homogeneity test')
                
                test_text = f"{test_used}: p = {hov_p:.4f}"
                if hov_result['result'].value != 'passed':
                    test_text += "\nEqual variance rejected"
                else:
                    test_text += "\nEqual variance not rejected"
                
                ax4.text(0.05, 0.95, test_text, transform=ax4.transAxes, 
                        verticalalignment='top', 
                        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
            
            fig4.tight_layout()
            figures['residuals_plot'] = fig_to_svg(fig4)
        except Exception as e:
            figures['residuals_plot_error'] = str(e)
        
        # Figure 5: Violin plots for distributional comparison
        try:
            fig5, ax5 = plt.subplots(figsize=(12, 8))
            fig5.patch.set_alpha(0.0)
            ax5.patch.set_alpha(0.0)
            
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
            sns.violinplot(x='Group', y='Value', data=plot_data, ax=ax5, 
                         inner='quartile', palette=PASTEL_COLORS)
            
            # Add group means as points
            means = [stat['mean'] for stat in group_stats]
            x_positions = range(len(groups))
            ax5.plot(x_positions, means, 'ro', markersize=8, label='Mean')
            
            # Add labels and title
            ax5.set_title('Distribution Comparison (Violin Plots)')
            ax5.set_ylabel('Value')
            ax5.legend()
            
            # Add ANOVA result
            result_text = f"F({df_between}, {df_within}) = {statistic:.3f}, p = {p_value:.4f}"
            if p_value < alpha:
                result_text += " *"
            
            ax5.text(0.02, 0.98, result_text, transform=ax5.transAxes, 
                    verticalalignment='top', 
                    bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
            
            fig5.tight_layout()
            figures['violin_plot'] = fig_to_svg(fig5)
        except Exception as e:
            figures['violin_plot_error'] = str(e)
        
        # Figure 6: Power analysis curve
        if 'power_analysis' in additional_stats and 'error' not in additional_stats['power_analysis']:
            try:
                fig6, ax6 = plt.subplots(figsize=(10, 6))
                fig6.patch.set_alpha(0.0)
                ax6.patch.set_alpha(0.0)
                
                # Create range of sample sizes
                total_n = sum(len(group) for group in groups)
                sample_sizes = np.arange(len(groups) * 2, max(150, total_n * 2), 5)
                
                # Calculate power for each sample size
                from statsmodels.stats.power import FTestAnovaPower
                power_analysis = FTestAnovaPower()
                powers = [power_analysis.power(
                    effect_size=additional_stats['power_analysis']['effect_size_f'],
                    nobs=n_sample,
                    alpha=alpha,
                    k_groups=len(groups)
                ) for n_sample in sample_sizes]
                
                # Plot power curve
                ax6.plot(sample_sizes, powers, 'b-', linewidth=2)
                
                # Add horizontal lines at common power thresholds
                ax6.axhline(y=0.8, color='red', linestyle='--', label='0.8 (conventional minimum)')
                ax6.axhline(y=0.9, color='orange', linestyle='--', label='0.9')
                ax6.axhline(y=0.95, color='green', linestyle='--', label='0.95')
                
                # Add vertical line at current sample size
                ax6.axvline(x=total_n, color='purple', linestyle='-', 
                           label=f'Current n = {total_n} (power = {additional_stats["power_analysis"]["observed_power"]:.3f})')
                
                # Add vertical lines at recommended sample sizes
                if 'n_for_80_power' in additional_stats['power_analysis']:
                    n_80 = additional_stats['power_analysis']['n_for_80_power']
                    ax6.axvline(x=n_80, color='red', linestyle=':', 
                              label=f'n for 80% power = {n_80}')
                
                # Add labels and title
                ax6.set_xlabel('Total Sample Size')
                ax6.set_ylabel('Statistical Power')
                ax6.set_title(f'Power Analysis for Effect Size f = {additional_stats["power_analysis"]["effect_size_f"]:.3f}')
                ax6.legend(loc='lower right')
                ax6.grid(True, linestyle='--', alpha=0.7)
                
                # Set y-axis limits
                ax6.set_ylim(0, 1.05)
                
                # Add text with key sample size recommendations
                text = "Recommended total sample sizes:\n"
                text += f"For 80% power: n = {additional_stats['power_analysis']['n_for_80_power']}\n"
                text += f"For 90% power: n = {additional_stats['power_analysis']['n_for_90_power']}\n"
                text += f"For 95% power: n = {additional_stats['power_analysis']['n_for_95_power']}"
                
                ax6.text(0.02, 0.02, text, transform=ax6.transAxes, 
                        verticalalignment='bottom', 
                        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
                
                fig6.tight_layout()
                figures['power_analysis'] = fig_to_svg(fig6)
            except Exception as e:
                figures['power_analysis_error'] = str(e)
        
        # Figure 7: Post-hoc test visualization
        if p_value < alpha and 'tukey_hsd' in post_hoc and isinstance(post_hoc['tukey_hsd'], list):
            try:
                # Count the number of comparisons
                n_comparisons = len(post_hoc['tukey_hsd'])
                
                # Create heatmap data
                n_groups = len(groups)
                heatmap_data = np.zeros((n_groups, n_groups))
                p_values_matrix = np.ones((n_groups, n_groups))
                
                for pair in post_hoc['tukey_hsd']:
                    idx1 = next(j for j, s in enumerate(group_stats) if s['name'] == pair['group1'])
                    idx2 = next(j for j, s in enumerate(group_stats) if s['name'] == pair['group2'])
                    
                    # Fill in the mean difference
                    heatmap_data[idx1, idx2] = pair['mean_diff']
                    heatmap_data[idx2, idx1] = -pair['mean_diff']
                    
                    # Fill in the p-values
                    p_values_matrix[idx1, idx2] = pair['p_value']
                    p_values_matrix[idx2, idx1] = pair['p_value']
                
                fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(14, 7))
                fig7.patch.set_alpha(0.0)
                ax7a.patch.set_alpha(0.0)
                ax7b.patch.set_alpha(0.0)
                
                # Create mean differences heatmap
                mask = np.zeros_like(heatmap_data, dtype=bool)
                mask[np.tril_indices_from(mask)] = True  # Mask lower triangle
                
                # Custom colormap centered on zero
                cmap = sns.diverging_palette(240, 10, as_cmap=True)
                
                sns.heatmap(heatmap_data, mask=mask, cmap=cmap, center=0, 
                          annot=True, fmt='.2f', linewidths=1, ax=ax7a)
                
                # Add group names as labels
                ax7a.set_xticks(np.arange(n_groups) + 0.5)
                ax7a.set_yticks(np.arange(n_groups) + 0.5)
                ax7a.set_xticklabels([s['name'] for s in group_stats])
                ax7a.set_yticklabels([s['name'] for s in group_stats])
                
                ax7a.set_title('Mean Differences')
                
                # Create p-values heatmap
                # Highlight significant comparisons
                annot = np.empty_like(p_values_matrix, dtype=object)
                for i in range(n_groups):
                    for j in range(n_groups):
                        if i != j:
                            if p_values_matrix[i, j] < alpha:
                                stars = '***' if p_values_matrix[i, j] < 0.001 else ('**' if p_values_matrix[i, j] < 0.01 else '*')
                                annot[i, j] = f"{p_values_matrix[i, j]:.3f}{stars}"
                            else:
                                annot[i, j] = f"{p_values_matrix[i, j]:.3f}"
                        else:
                            annot[i, j] = ""
                
                # Custom colormap for p-values
                cmap_p = plt.cm.YlOrRd_r
                
                sns.heatmap(p_values_matrix, mask=mask, cmap=cmap_p, vmin=0, vmax=1,
                          annot=annot, fmt='', linewidths=1, ax=ax7b)
                
                # Add group names as labels
                ax7b.set_xticks(np.arange(n_groups) + 0.5)
                ax7b.set_yticks(np.arange(n_groups) + 0.5)
                ax7b.set_xticklabels([s['name'] for s in group_stats])
                ax7b.set_yticklabels([s['name'] for s in group_stats])
                
                ax7b.set_title('P-values (Tukey\'s HSD)')
                
                # Add legend for significance stars
                legend_text = "* p < 0.05\n** p < 0.01\n*** p < 0.001"
                ax7b.text(1.05, 0.5, legend_text, transform=ax7b.transAxes,
                        verticalalignment='center')
                
                fig7.tight_layout()
                figures['post_hoc_heatmap'] = fig_to_svg(fig7)
            except Exception as e:
                figures['post_hoc_heatmap_error'] = str(e)
        
        # Figure 8: Bootstrap confidence intervals
        if 'bootstrap' in additional_stats and 'error' not in additional_stats['bootstrap']:
            try:
                fig8, ax8 = plt.subplots(figsize=(12, 6))
                fig8.patch.set_alpha(0.0)
                ax8.patch.set_alpha(0.0)
                
                # Extract bootstrap data
                bootstrap_results = additional_stats['bootstrap']['results']
                
                # Create data for plotting
                group_names = [res['group'] for res in bootstrap_results]
                means = [res['mean'] for res in bootstrap_results]
                lower_ci = [res['ci_lower'] for res in bootstrap_results]
                upper_ci = [res['ci_upper'] for res in bootstrap_results]
                
                # Calculate error bar positions
                lower_error = [means[i] - lower_ci[i] for i in range(len(means))]
                upper_error = [upper_ci[i] - means[i] for i in range(len(means))]
                
                # Create error bar plot
                x_pos = np.arange(len(group_names))
                ax8.errorbar(x_pos, means, yerr=[lower_error, upper_error], fmt='o', capsize=10,
                           color='blue', ecolor='black', elinewidth=2, capthick=2)
                
                # Add horizontal line at grand mean
                ax8.axhline(y=grand_mean, color='red', linestyle='--', linewidth=2, label='Grand Mean')
                
                # Add labels and title
                ax8.set_xticks(x_pos)
                ax8.set_xticklabels(group_names)
                ax8.set_ylabel('Mean Value')
                ax8.set_title(f'Bootstrap {(1-alpha)*100:.0f}% Confidence Intervals')
                ax8.legend()
                
                # Add explanation
                text = f"Bootstrap results based on {additional_stats['bootstrap']['n_samples']} resamples"
                ax8.text(0.5, 0.02, text, transform=ax8.transAxes,
                        horizontalalignment='center',
                        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
                
                fig8.tight_layout()
                figures['bootstrap_intervals'] = fig_to_svg(fig8)
            except Exception as e:
                figures['bootstrap_intervals_error'] = str(e)
        
        return {
            'test': 'One-Way ANOVA',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'df_between': int(df_between),
            'df_within': int(df_within),
            'df_total': int(df_total),
            'ss_between': float(ss_between),
            'ss_within': float(ss_within),
            'ss_total': float(ss_total),
            'ms_between': float(ms_between),
            'ms_within': float(ms_within),
            'eta_squared': float(eta_squared),
            'partial_eta_squared': float(partial_eta_squared),
            'omega_squared': float(omega_squared),
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
            'test': 'One-Way ANOVA',
            'statistic': None,
            'p_value': None,
            'significant': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }