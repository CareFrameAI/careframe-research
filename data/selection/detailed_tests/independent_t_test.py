import traceback
import pandas as pd
from scipy import stats
import numpy as np
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from matplotlib.patches import Patch
import matplotlib.patheffects as path_effects
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from lifelines import KaplanMeierFitter, CoxPHFitter
from study_model.study_model import StatisticalTest
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
from scipy.stats import shapiro, normaltest, jarque_bera
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
    PowerAnalysisTest
)
from data.selection.detailed_tests.formatting import fig_to_svg, PASTEL_COLORS, PASTEL_CMAP



def independent_t_test(group1: pd.Series, group2: pd.Series, alpha: float, alternative: str = 'two-sided') -> Dict[str, Any]:
    """Performs an independent t-test with comprehensive statistics and assumption checks."""
    try:
        from scipy import stats
        import numpy as np
        # Handle missing values separately for each group
        group1_clean = group1.dropna()
        group2_clean = group2.dropna()
        
        if len(group1_clean) < 2 or len(group2_clean) < 2:
            return {
                'test': 'Independent T-Test',
                'statistic': None,
                'p_value': None,
                'significant': False,
                'reason': 'One or both groups have less than two valid values.'
            }
        
        # Calculate the differences
        differences = group1_clean - group2_clean
        
        # Calculate basic statistics
        n = len(differences)
        mean_diff = float(differences.mean())
        std_diff = float(differences.std(ddof=1))
        se_diff = std_diff / np.sqrt(n)
        
        # Perform the independent t-test
        statistic, p_value = stats.ttest_ind(a=group1_clean, b=group2_clean, equal_var=False, alternative=alternative)
        
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
            
        # Calculate effect size (Cohen's d for independent samples)
        # For independent samples, d = mean_diff / sqrt((n1-1)*s1^2 + (n2-1)*s2^2) / sqrt(n1+n2-2)
        # where s1 and s2 are the sample standard deviations
        s1 = group1_clean.std(ddof=1)
        s2 = group2_clean.std(ddof=1)
        pooled_std = np.sqrt(((n - 1) * s1**2 + (n - 1) * s2**2) / (n + n - 2))
        cohens_d = mean_diff / pooled_std
        
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
        
        # 1. Normality of each group
        try:
            # Test normality for each group
            normality_test = NormalityTest()
            
            for i, group in enumerate([group1_clean, group2_clean]):
                group_key = f'normality_group_{i+1}'
                # Using exact input variables as defined in format.py
                assumptions[group_key] = normality_test.run_test(data=group)
                
        except Exception as e:
            assumptions['normality_error'] = str(e)
            
        # 2. Homogeneity of variances
        try:
            # Create a combined dataframe with all values and group indicators
            combined_values = pd.concat([group1_clean, group2_clean])
            combined_groups = pd.Series(['Group 1'] * len(group1_clean) + ['Group 2'] * len(group2_clean))
            
            # Use our test with proper parameters as defined in format.py
            homogeneity_test = HomogeneityOfVarianceTest()
            assumptions['homogeneity_of_variance'] = homogeneity_test.run_test(data=combined_values, groups=combined_groups)
            
        except Exception as e:
            assumptions['homogeneity_of_variance_error'] = str(e)
            
        # 3. Independence test
        try:
            # For t-test, independence is usually assumed based on study design,
            # but we can test for certain patterns that might indicate dependency
            independence_test = IndependenceTest()
            
            # Create a combined dataset for testing independence
            all_data = pd.concat([group1_clean, group2_clean])
            assumptions['independence'] = independence_test.run_test(data=all_data)
            
        except Exception as e:
            assumptions['independence_error'] = str(e)
            
        # 4. Sample size check
        try:
            # Check sample size for each group
            sample_size_test = SampleSizeTest()
            
            for i, group in enumerate([group1_clean, group2_clean]):
                group_key = f'sample_size_group_{i+1}'
                assumptions[group_key] = sample_size_test.run_test(data=group, min_recommended=30)
                
        except Exception as e:
            assumptions['sample_size_error'] = str(e)
            
        # 5. Outliers check
        try:
            # Check for outliers in each group
            outlier_test = OutlierTest()
            
            for i, group in enumerate([group1_clean, group2_clean]):
                group_key = f'outliers_group_{i+1}'
                assumptions[group_key] = outlier_test.run_test(data=group)
                
        except Exception as e:
            assumptions['outliers_error'] = str(e)

        # 6. Equal group sizes (not a formal test, but often considered)
        try:
            group_size_ratio = max(len(group1_clean)/len(group2_clean), len(group2_clean)/len(group1_clean))
            equal_size_warning = group_size_ratio > 1.5
            
            assumptions['equal_group_sizes'] = {
                'result': AssumptionResult.WARNING if equal_size_warning else AssumptionResult.PASSED,
                'details': f"Group size ratio is {group_size_ratio:.2f}. Groups should ideally be of similar size.",
                'warnings': ["Groups have substantially different sizes. This may reduce the robustness of the t-test."] if equal_size_warning else []
            }
        except Exception as e:
            assumptions['equal_group_sizes_error'] = str(e)
        
        # Power Analysis with graph using transparent background and pastel colors
        try:
            from statsmodels.stats.power import TTestIndPower
            
            # Initialize power analysis
            power_analysis = TTestIndPower()
            
            # Calculate group sizes
            n1 = len(group1_clean)
            n2 = len(group2_clean)
            total_n = n1 + n2
            
            # Recalculate Cohen's d to ensure it's valid
            # Using pooled standard deviation
            pooled_sd = np.sqrt(((n1 - 1) * np.var(group1_clean, ddof=1) + 
                                (n2 - 1) * np.var(group2_clean, ddof=1)) / 
                               (n1 + n2 - 2))
            
            # Calculate effect size (Cohen's d)
            mean_diff = np.mean(group1_clean) - np.mean(group2_clean)
            
            # Ensure we have a valid pooled_sd to avoid division by zero
            if pooled_sd > 0:
                actual_effect_size = mean_diff / pooled_sd
            else:
                actual_effect_size = 0.01
            
            # Ensure effect size is not NaN or infinite
            if np.isnan(actual_effect_size) or np.isinf(actual_effect_size):
                actual_effect_size = 0.01
            
            # Ensure effect size is not zero to avoid division issues
            if abs(actual_effect_size) < 0.01:
                actual_effect_size = 0.01 if actual_effect_size >= 0 else -0.01
            
            # Calculate ratio of sample sizes
            ratio = n2/n1 if n1 > 0 else 1.0
            
            # Calculate achieved power
            achieved_power = power_analysis.power(
                effect_size=abs(actual_effect_size), 
                nobs1=n1, 
                ratio=ratio, 
                alpha=alpha, 
                alternative=alternative
            )
            
            # Check if power calculation worked
            if np.isnan(achieved_power) or np.isinf(achieved_power):
                # Fall back to a simpler approximation
                from scipy import stats
                n_per_group = (n1 + n2) / 2
                z_crit = stats.norm.ppf(1 - alpha/2)
                achieved_power = stats.norm.cdf(abs(actual_effect_size) * np.sqrt(n_per_group/2) - z_crit)
            
            # Calculate sample size needed for 80% power
            try:
                recommended_n = power_analysis.solve_power(
                    effect_size=abs(actual_effect_size), 
                    power=0.8, 
                    ratio=ratio, 
                    alpha=alpha, 
                    alternative=alternative
                )
                if np.isnan(recommended_n) or np.isinf(recommended_n):
                    z_crit = stats.norm.ppf(1 - alpha/2)
                    z_power = stats.norm.ppf(0.8)
                    recommended_n = 2 * ((z_crit + z_power) / abs(actual_effect_size))**2
            except:
                z_crit = stats.norm.ppf(1 - alpha/2)
                z_power = stats.norm.ppf(0.8)
                recommended_n = 2 * ((z_crit + z_power) / abs(actual_effect_size))**2
            
            # Create power curve graph with transparent background
            fig_power = plt.figure(figsize=(10, 6))
            fig_power.patch.set_alpha(0.0)  # Set figure background transparent
            ax_power = fig_power.add_subplot(111)
            ax_power.patch.set_alpha(0.0)  # Set axes background transparent
            
            # Generate effect sizes for plotting
            effect_sizes = np.linspace(0.1, max(1.0, abs(actual_effect_size)*1.5), 100)
            
            # Calculate power for different effect sizes
            power_curve = []
            for es in effect_sizes:
                pwr = power_analysis.power(
                    effect_size=es, 
                    nobs1=n1, 
                    ratio=ratio, 
                    alpha=alpha, 
                    alternative=alternative
                )
                power_curve.append(pwr)
            
            # Plot power curve using pastel colors
            ax_power.plot(effect_sizes, power_curve, '-', linewidth=2, color=PASTEL_COLORS[0], label='Power curve')
            
            # Mark the observed effect size
            ax_power.plot([abs(actual_effect_size)], [achieved_power], 'o', markersize=8, 
                         color=PASTEL_COLORS[1], label='Observed effect')
            
            # Add reference lines for 80% and 90% power using pastel colors
            ax_power.axhline(y=0.8, color=PASTEL_COLORS[2], linestyle='--', alpha=0.7, 
                            label='80% Power (recommended)')
            ax_power.axhline(y=0.9, color=PASTEL_COLORS[3], linestyle='--', alpha=0.7, 
                            label='90% Power')
            ax_power.axhline(y=0.5, color=PASTEL_COLORS[4], linestyle='--', alpha=0.7, 
                            label='50% Power (coin flip)')
            
            # Shade the area below 80% power as a warning zone with pastel color
            ax_power.fill_between(effect_sizes, 0, 0.8, alpha=0.1, color=PASTEL_COLORS[5], 
                                 label='Underpowered region')
            
            # Add annotation for observed effect
            ax_power.annotate(f"Observed effect: d = {abs(actual_effect_size):.3f}\nPower = {achieved_power:.3f}",
                             xy=(abs(actual_effect_size), achieved_power), 
                             xytext=(abs(actual_effect_size)+0.1, achieved_power-0.1),
                             arrowprops=dict(arrowstyle='->', color=PASTEL_COLORS[1]),
                             bbox=dict(boxstyle='round', fc='white', alpha=0.8))
            
            # Add annotation for recommended sample size
            ax_power.annotate(f"Sample size for 80% power: {int(np.ceil(recommended_n))}",
                             xy=(0.95, 0.05), xycoords='axes fraction', 
                             ha='right', va='bottom',
                             bbox=dict(boxstyle='round', fc='white', alpha=0.8))
            
            # Add title and labels
            ax_power.set_title("Power Analysis for Independent T-Test", fontsize=14)
            ax_power.set_xlabel("Effect Size (Cohen's d)", fontsize=12)
            ax_power.set_ylabel("Statistical Power (1-β)", fontsize=12)
            
            # Add legend with transparent background
            legend = ax_power.legend(loc='upper left')
            legend.get_frame().set_alpha(0.8)
            legend.get_frame().set_facecolor('white')
            
            # Set axis limits
            ax_power.set_xlim(0, max(1.0, abs(actual_effect_size)*1.5))
            ax_power.set_ylim(0, 1.05)
            
            # Add grid with light color
            ax_power.grid(True, alpha=0.2, color='gray')
            
            # Format the result to match our assumption test format
            if achieved_power >= 0.8:
                result = AssumptionResult.PASSED
                details = f"Study has adequate power ({achieved_power:.3f}) to detect effect size of {actual_effect_size:.3f}"
            elif achieved_power >= 0.5:
                result = AssumptionResult.WARNING
                details = f"Study may be underpowered ({achieved_power:.3f}) to detect effect size of {actual_effect_size:.3f}"
            else:
                result = AssumptionResult.FAILED
                details = f"Study is underpowered ({achieved_power:.3f}) to detect effect size of {actual_effect_size:.3f}"
            
            # Store the result in the same format as other assumption tests
            assumptions['power_analysis'] = {
                'result': result,
                'details': details,
                'test_used': 'TTestIndPower',
                'effect_size': float(actual_effect_size),
                'sample_size': int(total_n),
                'alpha': float(alpha),
                'calculated_value': float(achieved_power),
                'recommended_sample_size': int(np.ceil(recommended_n)),
                'warnings': [] if achieved_power >= 0.8 else ["Study may be underpowered to detect the effect of interest."],
                'figures': {
                    'power_curve': fig_to_svg(fig_power)
                }
            }
            
        except Exception as e:
            assumptions['power_analysis_error'] = f"Error in power analysis: {str(e)}"
            import traceback
            assumptions['power_analysis_traceback'] = traceback.format_exc()
            
        # Create interpretation
        interpretation = f"Independent T-Test comparing two independent groups.\n\n"
        
        # Basic results
        interpretation += f"t({df}) = {statistic:.3f}, p = {p_value:.5f}\n"
        interpretation += f"Mean difference = {mean_diff:.3f}, SE = {se_diff:.3f}\n"
        interpretation += f"95% CI for difference: [{ci_lower:.3f}, {ci_upper:.3f}]\n"
        interpretation += f"Effect size: d = {cohens_d:.3f}, g = {hedges_g:.3f} ({effect_magnitude.lower()} effect)\n\n"
        
        # Group statistics
        interpretation += "Group statistics:\n"
        for i, group in enumerate([group1_clean, group2_clean]):
            interpretation += f"- Group {i+1}: mean = {float(group.mean()):.3f}, SD = {float(group.std(ddof=1)):.3f}\n"
        
        # Assumptions
        interpretation += "\nAssumption tests:\n"
        
        # Normality
        normality_group1 = assumptions.get('normality_group_1', {})
        normality_group2 = assumptions.get('normality_group_2', {})

        if 'result' in normality_group1 and 'result' in normality_group2:
            g1_passed = normality_group1['result'].value == 'passed'
            g2_passed = normality_group2['result'].value == 'passed'
            
            interpretation += f"- Normality: "
            
            if g1_passed and g2_passed:
                interpretation += "Assumption satisfied for both groups.\n"
            else:
                violated_groups = []
                if not g1_passed:
                    violated_groups.append("Group 1")
                if not g2_passed:
                    violated_groups.append("Group 2")
                    
                interpretation += f"Assumption violated for {', '.join(violated_groups)}. "
                
                # Check if all groups have large sample sizes
                all_large = all(len(group) >= 30 for group in [group1_clean, group2_clean])
                if all_large:
                    interpretation += "However, due to large sample sizes, t-test should be robust to violations of normality (Central Limit Theorem).\n"
                else:
                    interpretation += "Consider using a non-parametric alternative like the Mann-Whitney U test.\n"
        
        # Homogeneity of variance
        homogeneity = assumptions.get('homogeneity_of_variance', {})
        if 'result' in homogeneity and 'p_value' in homogeneity:
            hov_passed = homogeneity['result'].value == 'passed'
            interpretation += f"- Homogeneity of variance: {homogeneity.get('test_used', 'Levene')} test, p = {homogeneity['p_value']:.5f}, "
            
            if hov_passed:
                interpretation += "assumption satisfied.\n"
            else:
                interpretation += f"assumption potentially violated. However, we're using Welch's t-test which does not assume equal variances.\n"
        
        # Independence
        independence = assumptions.get('independence', {})
        if 'result' in independence:
            ind_passed = independence['result'].value == 'passed'
            interpretation += f"- Independence: "
            
            if ind_passed:
                interpretation += "No significant dependency patterns detected.\n"
            else:
                interpretation += f"Potential dependency detected. {independence.get('message', '')} Independent observations are required for t-test validity.\n"

        # Power Analysis
        power_analysis = assumptions.get('power_analysis', {})
        if 'result' in power_analysis:
            power_passed = power_analysis['result'].value == 'passed'
            achieved_power = power_analysis.get('calculated_value', 0)
            
            interpretation += f"- Statistical Power: "
            
            if power_passed:
                interpretation += f"Adequate power ({achieved_power:.2f}) to detect the observed effect size.\n"
            else:
                interpretation += f"Insufficient power ({achieved_power:.2f}). "
                interpretation += f"{power_analysis.get('details', 'Study may be underpowered to detect the effect of interest.')}\n"

        # Continue with other assumptions...
        
        # Conclusion
        interpretation += f"\nConclusion: There is {'a statistically significant' if significant else 'no statistically significant'} "
        
        if alternative == 'two-sided':
            interpretation += f"difference between the two groups (p = {p_value:.5f}). "
        elif alternative == 'less':
            interpretation += f"evidence that Group 1 is less than Group 2 (p = {p_value:.5f}). "
        else:  # 'greater'
            interpretation += f"evidence that Group 1 is greater than Group 2 (p = {p_value:.5f}). "
        
        if significant:
            interpretation += f"The {effect_magnitude.lower()} effect size (d = {cohens_d:.3f}) suggests that "
            
            if mean_diff > 0:
                interpretation += f"Group 1 scores are higher than Group 2 scores by approximately {mean_diff:.3f} units on average."
            else:
                interpretation += f"Group 2 scores are higher than Group 1 scores by approximately {abs(mean_diff):.3f} units on average."
        
        # Additional statistics
        additional_stats = {}
        
        # 1. Mann-Whitney U test (non-parametric alternative)
        try:
            u_stat, u_p = stats.mannwhitneyu(group1_clean, group2_clean, alternative=alternative)
            
            additional_stats['mann_whitney_u'] = {
                'statistic': float(u_stat),
                'p_value': float(u_p),
                'significant': u_p < alpha,
                'agrees_with_t_test': (u_p < alpha) == significant
            }
        except Exception as e:
            additional_stats['mann_whitney_u'] = {
                'error': str(e)
            }
        
        # 2. Bayes Factor calculation (approximation)
        try:
            # Calculate Bayes Factor using BIC approximation
            # BF10 = exp((BIC_null - BIC_alt) / 2)
            
            # Fit models
            # Combined data for null model
            combined_data = np.concatenate([group1_clean, group2_clean])
            n_total = len(combined_data)
            
            # Calculate log-likelihoods
            # For null model: one mean and variance
            null_mean = np.mean(combined_data)
            null_var = np.var(combined_data, ddof=1)
            null_ll = -n_total/2 * np.log(2 * np.pi * null_var) - np.sum((combined_data - null_mean)**2) / (2 * null_var)
            
            # For alternative model: two means and pooled variance
            g1_mean = np.mean(group1_clean)
            g2_mean = np.mean(group2_clean)
            alt_ll = (-len(group1_clean)/2 * np.log(2 * np.pi * pooled_std**2) 
                     - np.sum((group1_clean - g1_mean)**2) / (2 * pooled_std**2)
                     - len(group2_clean)/2 * np.log(2 * np.pi * pooled_std**2)
                     - np.sum((group2_clean - g2_mean)**2) / (2 * pooled_std**2))
            
            # Calculate BIC
            null_bic = -2 * null_ll + 2 * np.log(n_total)  # 2 parameters: mean and variance
            alt_bic = -2 * alt_ll + 3 * np.log(n_total)   # 3 parameters: two means and pooled variance
            
            # Calculate Bayes Factor
            bf10 = np.exp((null_bic - alt_bic) / 2)
            
            # Interpret Bayes Factor
            if bf10 < 1/30:
                bf_interpretation = "Very strong evidence for null"
            elif bf10 < 1/10:
                bf_interpretation = "Strong evidence for null"
            elif bf10 < 1/3:
                bf_interpretation = "Moderate evidence for null"
            elif bf10 < 1:
                bf_interpretation = "Anecdotal evidence for null"
            elif bf10 == 1:
                bf_interpretation = "No evidence for either hypothesis"
            elif bf10 < 3:
                bf_interpretation = "Anecdotal evidence for alternative"
            elif bf10 < 10:
                bf_interpretation = "Moderate evidence for alternative"
            elif bf10 < 30:
                bf_interpretation = "Strong evidence for alternative"
            else:
                bf_interpretation = "Very strong evidence for alternative"
            
            additional_stats['bayes_factor'] = {
                'bf10': float(bf10),
                'interpretation': bf_interpretation
            }
        except Exception as e:
            additional_stats['bayes_factor'] = {
                'error': str(e)
            }
        
        # 3. Bootstrap confidence intervals
        try:
            # Number of bootstrap samples
            n_bootstrap = 5000
            bootstrap_diffs = []
            
            # Generate bootstrap samples
            for _ in range(n_bootstrap):
                # Sample with replacement from each group
                boot1 = np.random.choice(group1_clean, size=len(group1_clean), replace=True)
                boot2 = np.random.choice(group2_clean, size=len(group2_clean), replace=True)
                
                # Calculate mean difference
                boot_diff = np.mean(boot1) - np.mean(boot2)
                bootstrap_diffs.append(boot_diff)
            
            # Calculate percentile confidence intervals
            bootstrap_ci_lower = np.percentile(bootstrap_diffs, 2.5)
            bootstrap_ci_upper = np.percentile(bootstrap_diffs, 97.5)
            
            additional_stats['bootstrap_ci'] = {
                'ci_lower': float(bootstrap_ci_lower),
                'ci_upper': float(bootstrap_ci_upper),
                'n_bootstrap': n_bootstrap
            }
        except Exception as e:
            additional_stats['bootstrap_ci'] = {
                'error': str(e)
            }
        
        # 4. Equivalence test (TOST - Two One-Sided Tests)
        try:
            # Define equivalence bounds (e.g., +/- 0.5 * pooled_std)
            # This approach defines "practically equivalent" as a difference less than
            # 0.5 standard deviations in either direction
            bound = 0.5 * pooled_std
            
            # Perform two one-sided tests
            t_lower = (mean_diff - (-bound)) / se_diff
            p_lower = 1 - stats.t.cdf(t_lower, df)
            
            t_upper = (bound - mean_diff) / se_diff
            p_upper = 1 - stats.t.cdf(t_upper, df)
            
            # The overall p-value is the maximum of the two p-values
            p_tost = max(p_lower, p_upper)
            
            # The means are equivalent if both one-sided tests are significant
            equivalent = p_tost < alpha
            
            additional_stats['equivalence_test'] = {
                'bound': float(bound),
                'p_value': float(p_tost),
                'equivalent': equivalent,
                'interpretation': "The means are statistically equivalent (within bounds of ±{:.3f})".format(bound) if equivalent 
                                else "Cannot conclude that the means are equivalent"
            }
        except Exception as e:
            additional_stats['equivalence_test'] = {
                'error': str(e)
            }
        
        # 5. False discovery rate and false negative rate
        try:
            # This is a demonstration of multiple testing considerations
            # Assume a prior probability for alternative hypothesis
            prior_h1 = 0.5  # Equal prior probabilities
            
            # Type I error rate (false positive) = alpha
            alpha_val = alpha
            
            # Type II error rate (false negative) = 1 - power
            if 'power_analysis' in additional_stats and 'achieved_power' in additional_stats['power_analysis']:
                beta_val = 1 - additional_stats['power_analysis']['achieved_power']
            else:
                beta_val = 0.2  # Default assuming 80% power
            
            # Bayes' theorem for false discovery rate
            # P(H0|sig) = P(sig|H0) * P(H0) / P(sig)
            # = α * (1-prior_h1) / [α * (1-prior_h1) + (1-β) * prior_h1]
            
            if significant:
                false_discovery_rate = (alpha_val * (1 - prior_h1)) / (alpha_val * (1 - prior_h1) + (1 - beta_val) * prior_h1)
            else:
                # False negative rate
                # P(H1|non-sig) = P(non-sig|H1) * P(H1) / P(non-sig)
                # = β * prior_h1 / [β * prior_h1 + (1-α) * (1-prior_h1)]
                false_negative_rate = (beta_val * prior_h1) / (beta_val * prior_h1 + (1 - alpha_val) * (1 - prior_h1))
                false_discovery_rate = None
            
            additional_stats['error_rates'] = {
                'prior_h1': prior_h1,
                'false_discovery_rate': float(false_discovery_rate) if false_discovery_rate is not None else None,
                'false_negative_rate': float(false_negative_rate) if 'false_negative_rate' in locals() else None,
                'warning': "These rates assume a prior probability of {:.0%} for the alternative hypothesis".format(prior_h1)
            }
        except Exception as e:
            additional_stats['error_rates'] = {
                'error': str(e)
            }
        
        # Create figures
        figures = {}
        
        # Figure 1: Box plots with individual data points
        try:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            
            # Convert data to long format for seaborn
            data_long = pd.DataFrame({
                'Group': ['Group 1'] * len(group1_clean) + ['Group 2'] * len(group2_clean),
                'Value': pd.concat([group1_clean, group2_clean])
            })
            
            # Create box plot with individual points
            sns.boxplot(x='Group', y='Value', data=data_long, ax=ax1, width=0.5, palette=PASTEL_COLORS[:2])
            sns.stripplot(x='Group', y='Value', data=data_long, ax=ax1, size=5, alpha=0.6, color=PASTEL_COLORS[2])
            
            # Add mean lines
            group_means = [float(group1_clean.mean()), float(group2_clean.mean())]
            for i, mean_val in enumerate(group_means):
                ax1.axhline(y=mean_val, xmin=i/2, xmax=(i+1)/2, color='red', linestyle='--', linewidth=2)
            
            # Add mean difference annotation
            ax1.annotate(f"Mean diff: {mean_diff:.3f}\np = {p_value:.4f}{' *' if significant else ''}",
                       xy=(0.5, 0.95), xycoords='axes fraction', ha='center', va='top',
                       bbox=dict(boxstyle='round', fc='white', alpha=0.8))
            
            # Add a title
            ax1.set_title(f"Distribution Comparison for Independent Groups", fontsize=14)
            
            # Label axes
            ax1.set_ylabel("Value", fontsize=12)
            
            fig1.tight_layout()
            figures['box_plot'] = fig_to_svg(fig1)
        except Exception as e:
            figures['box_plot_error'] = str(e)
        
        # Figure 2: Overlapping distribution density plot
        try:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            # Create density plots
            sns.kdeplot(group1_clean, ax=ax2, label='Group 1', fill=True, alpha=0.3, color=PASTEL_COLORS[0])
            sns.kdeplot(group2_clean, ax=ax2, label='Group 2', fill=True, alpha=0.3, color=PASTEL_COLORS[1])
            
            # Add mean lines
            group_means = [float(group1_clean.mean()), float(group2_clean.mean())]
            for i, mean_val in enumerate(group_means):
                ax2.axvline(x=mean_val, color=PASTEL_COLORS[i], linestyle='--', 
                          label=f"Mean Group {i+1}: {mean_val:.3f}")
            
            # Add mean difference annotation
            ax2.annotate(f"Mean diff: {mean_diff:.3f}\np = {p_value:.4f}{' *' if significant else ''}",
                       xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top',
                       bbox=dict(boxstyle='round', fc='white', alpha=0.8))
            
            # Add a title
            ax2.set_title(f"Probability Density Functions", fontsize=14)
            
            # Label axes
            ax2.set_xlabel("Value", fontsize=12)
            ax2.set_ylabel("Density", fontsize=12)
            
            # Add legend
            ax2.legend()
            
            fig2.tight_layout()
            figures['density_plot'] = fig_to_svg(fig2)
        except Exception as e:
            figures['density_plot_error'] = str(e)
        
        # Figure 3: Means with 95% CIs
        try:
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            
            # Calculate means and CIs
            means = [float(group1_clean.mean()), float(group2_clean.mean())]
            ses = [float(group1_clean.std(ddof=1) / np.sqrt(len(group1_clean))), 
                 float(group2_clean.std(ddof=1) / np.sqrt(len(group2_clean)))]
            
            ci_factors = stats.t.ppf(1 - alpha/2, [len(group1_clean)-1, len(group2_clean)-1])
            ci_widths = [ses[0] * ci_factors[0], ses[1] * ci_factors[1]]
            
            # Create point plot
            groups = ['Group 1', 'Group 2']
            ax3.errorbar(groups, means, yerr=ci_widths, fmt='o', capsize=10, 
                       capthick=2, elinewidth=2, markersize=10, color='blue')
            
            # Add mean difference region
            rect = plt.Rectangle((0, min(means) - 0.05 * (max(means) - min(means))), 
                               2, abs(means[0] - means[1]), alpha=0.2, color='gray')
            ax3.add_patch(rect)
            
            # Add annotation for mean difference
            mid_y = (means[0] + means[1]) / 2
            ax3.annotate(f"Mean difference: {mean_diff:.3f}\n95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]",
                       xy=(0.5, mid_y), xytext=(1.2, mid_y),
                       arrowprops=dict(arrowstyle='->'))
            
            # Add significance annotation
            ax3.annotate(f"t({df}) = {statistic:.3f}, p = {p_value:.4f}{' *' if significant else ''}",
                       xy=(0.5, 0.95), xycoords='axes fraction', ha='center', va='top',
                       bbox=dict(boxstyle='round', fc='white', alpha=0.8))
            
            # Add effect size annotation
            ax3.annotate(f"Effect size: d = {cohens_d:.3f} ({effect_magnitude.lower()})",
                       xy=(0.5, 0.05), xycoords='axes fraction', ha='center', va='bottom',
                       bbox=dict(boxstyle='round', fc='white', alpha=0.8))
            
            # Set title and labels
            ax3.set_title("Group Means with 95% Confidence Intervals", fontsize=14)
            ax3.set_ylabel("Value", fontsize=12)
            
            # Set appropriate y-axis limits
            all_vals = np.concatenate([means, 
                                      [means[0] - ci_widths[0], means[0] + ci_widths[0]],
                                      [means[1] - ci_widths[1], means[1] + ci_widths[1]]])
            y_range = max(all_vals) - min(all_vals)
            ax3.set_ylim(min(all_vals) - 0.2 * y_range, max(all_vals) + 0.2 * y_range)
            
            # Set x-axis limits to make room for annotations
            ax3.set_xlim(-0.5, 2.5)
            
            fig3.tight_layout()
            figures['means_with_ci'] = fig_to_svg(fig3)
        except Exception as e:
            figures['means_with_ci_error'] = str(e)
        
        # Figure 4: QQ plots for normality assessment
        try:
            fig4, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            for i, (group, ax) in enumerate(zip([group1_clean, group2_clean], axes)):
                # Calculate quantiles
                data_sorted = np.sort(group)
                n = len(data_sorted)
                p = np.arange(1, n + 1) / (n + 1)  # Quantile probabilities
                
                # Calculate theoretical quantiles
                theoretical_quantiles = stats.norm.ppf(p, loc=np.mean(group), scale=np.std(group, ddof=1))
                
                # Create QQ plot
                ax.scatter(theoretical_quantiles, data_sorted, alpha=0.7)
                
                # Add reference line
                min_val = min(np.min(theoretical_quantiles), np.min(data_sorted))
                max_val = max(np.max(theoretical_quantiles), np.max(data_sorted))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                
                # Add title and labels
                ax.set_title(f"Group {i+1} QQ Plot", fontsize=12)
                ax.set_xlabel("Theoretical Quantiles", fontsize=10)
                ax.set_ylabel("Sample Quantiles", fontsize=10)
                
                # Add normality test result
                if 'normality' in assumptions and 'results' in assumptions['normality']:
                    p_val = assumptions['normality']['results'][i]['p_value']
                    test_name = assumptions['normality']['results'][i]['test']
                    
                    result_text = f"{test_name}: p = {p_val:.4f}"
                    if p_val < alpha:
                        result_text += " *"
                        result_text += "\nNon-normal"
                    else:
                        result_text += "\nNormal"
                        
                    ax.text(0.05, 0.95, result_text, transform=ax.transAxes, 
                          va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
            
            fig4.suptitle("Normal Q-Q Plots for Assessing Normality", fontsize=14)
            fig4.tight_layout()
            figures['qq_plots'] = fig_to_svg(fig4)
        except Exception as e:
            figures['qq_plots_error'] = str(e)
        
            # Figure 5: Power analysis curve
            try:
                from statsmodels.stats.power import TTestIndPower
                
                fig5, ax5 = plt.subplots(figsize=(10, 6))
                
                # Setup power analysis
                power_analysis = TTestIndPower()
                
                # Generate effect sizes for plotting
                effect_sizes = np.linspace(0.1, 1.0, 100)
                
                # Calculate power for different effect sizes
                power = power_analysis.power(effect_sizes, 
                                           nobs1=len(group1_clean), 
                                           ratio=len(group2_clean)/len(group1_clean), 
                                           alpha=alpha, 
                                           alternative=alternative)
                
                # Plot power curve
                ax5.plot(effect_sizes, power, 'b-', linewidth=2)
                
                # Mark the observed effect size
                effect_size_abs = abs(hedges_g)
                observed_power = power_analysis.power(effect_size_abs, 
                                                    nobs1=len(group1_clean), 
                                                    ratio=len(group2_clean)/len(group1_clean), 
                                                    alpha=alpha, 
                                                    alternative=alternative)
                ax5.plot([effect_size_abs], [observed_power], 'ro', markersize=8)
                
                # Add reference lines for 80% and 90% power
                ax5.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80% Power')
                ax5.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='90% Power')
                
                # Add annotation for observed effect
                ax5.annotate(f"Observed effect: d = {effect_size_abs:.3f}\nPower = {observed_power:.3f}",
                           xy=(effect_size_abs, observed_power), xytext=(effect_size_abs+0.1, observed_power-0.1),
                           arrowprops=dict(arrowstyle='->'))
                
                # Add title and labels
                ax5.set_title("Power Analysis Curve", fontsize=14)
                ax5.set_xlabel("Effect Size (Cohen's d)", fontsize=12)
                ax5.set_ylabel("Statistical Power (1-β)", fontsize=12)
                
                # Add legend
                ax5.legend()
                
                # Set axis limits
                ax5.set_xlim(0, max(1.0, effect_size_abs*1.2))
                ax5.set_ylim(0, 1.05)
                
                # Add grid
                ax5.grid(True, alpha=0.3)
                
                fig5.tight_layout()
                figures['power_curve'] = fig_to_svg(fig5)
            except Exception as e:
                figures['power_curve_error'] = str(e)
            
            # Figure 6: Bootstrap distribution of mean differences
            try:
                if 'bootstrap_ci' in additional_stats and 'error' not in additional_stats['bootstrap_ci']:
                    fig6, ax6 = plt.subplots(figsize=(10, 6))
                    
                    # Generate bootstrap samples for the plot (simplified version)
                    n_bootstrap_display = 1000
                    bootstrap_diffs_display = []
                    
                    for _ in range(n_bootstrap_display):
                        # Sample with replacement from each group
                        boot1 = np.random.choice(group1_clean, size=len(group1_clean), replace=True)
                        boot2 = np.random.choice(group2_clean, size=len(group2_clean), replace=True)
                        
                        # Calculate mean difference
                        boot_diff = np.mean(boot1) - np.mean(boot2)
                        bootstrap_diffs_display.append(boot_diff)
                    
                    # Plot bootstrap distribution
                    sns.histplot(bootstrap_diffs_display, kde=True, ax=ax6, color='blue', alpha=0.6)
                    
                    # Add vertical line for observed difference
                    ax6.axvline(x=mean_diff, color='red', linestyle='-', linewidth=2, 
                              label=f'Observed diff: {mean_diff:.3f}')
                    
                    # Add vertical lines for bootstrap CI
                    ci_lower = additional_stats['bootstrap_ci']['ci_lower']
                    ci_upper = additional_stats['bootstrap_ci']['ci_upper']
                    
                    ax6.axvline(x=ci_lower, color='green', linestyle='--', linewidth=2, 
                              label=f'95% CI Lower: {ci_lower:.3f}')
                    ax6.axvline(x=ci_upper, color='green', linestyle='--', linewidth=2, 
                              label=f'95% CI Upper: {ci_upper:.3f}')
                    
                    # Add vertical line for zero
                    ax6.axvline(x=0, color='black', linestyle='--', alpha=0.5, 
                              label='No difference')
                    
                    # Add title and labels
                    ax6.set_title("Bootstrap Distribution of Mean Differences", fontsize=14)
                    ax6.set_xlabel("Mean Difference (Group 1 - Group 2)", fontsize=12)
                    ax6.set_ylabel("Frequency", fontsize=12)
                    
                    # Add legend
                    ax6.legend()
                    
                    # Add significance annotation
                    if ci_lower > 0 or ci_upper < 0:  # CI doesn't include zero
                        sig_text = "Significant difference (95% CI excludes 0)"
                    else:
                        sig_text = "Non-significant difference (95% CI includes 0)"
                    
                    ax6.annotate(sig_text, xy=(0.5, 0.05), xycoords='axes fraction', 
                               ha='center', va='bottom', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
                    
                    fig6.tight_layout()
                    figures['bootstrap_distribution'] = fig_to_svg(fig6)
            except Exception as e:
                figures['bootstrap_distribution_error'] = str(e)
            
            # Figure 7: Effect size visualization
            try:
                fig7, ax7 = plt.subplots(figsize=(10, 6))
                
                # Set up Cohen's d visualization
                # This visualizes two normal distributions separated by d standard deviations
                
                # Create a range for x-axis
                x = np.linspace(-4, 4, 1000)
                
                # Group 1 distribution (standardized to mean=0, sd=1)
                y1 = stats.norm.pdf(x, loc=0, scale=1)
                
                # Group 2 distribution (shifted by Cohen's d)
                y2 = stats.norm.pdf(x, loc=-cohens_d, scale=1)  # Negative because we're looking at it from Group 1's perspective
                
                # Plot the distributions
                ax7.plot(x, y1, 'b-', linewidth=2, label='Group 1 (standardized)')
                ax7.plot(x, y2, 'r-', linewidth=2, label='Group 2 (standardized)')
                
                # Add vertical lines for means
                ax7.axvline(x=0, color='blue', linestyle='--', alpha=0.7)
                ax7.axvline(x=-cohens_d, color='red', linestyle='--', alpha=0.7)
                
                # Fill the area representing overlap
                overlap_x = np.linspace(-4, 4, 1000)
                overlap_y = np.minimum(stats.norm.pdf(overlap_x, loc=0, scale=1), 
                                     stats.norm.pdf(overlap_x, loc=-cohens_d, scale=1))
                ax7.fill_between(overlap_x, overlap_y, alpha=0.3, color='purple', 
                               label='Distribution Overlap')
                
                # Calculate overlap percentage
                # For normal distributions, the overlap can be calculated using Cohen's d
                overlap_pct = 2 * stats.norm.cdf(-abs(cohens_d)/2)
                
                # Add annotations
                ax7.annotate(f"Cohen's d = {cohens_d:.3f} ({effect_magnitude.lower()})",
                           xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',
                           bbox=dict(boxstyle='round', fc='white', alpha=0.8))
                
                ax7.annotate(f"Distribution overlap: {overlap_pct:.1%}",
                           xy=(0.05, 0.87), xycoords='axes fraction', ha='left', va='top',
                           bbox=dict(boxstyle='round', fc='white', alpha=0.8))
                
                # Add title and labels
                ax7.set_title("Effect Size Visualization (Cohen's d)", fontsize=14)
                ax7.set_xlabel("Standardized Mean Difference", fontsize=12)
                ax7.set_ylabel("Probability Density", fontsize=12)
                
                # Add legend
                ax7.legend()
                
                # Set reasonable axis limits
                ax7.set_xlim(-3, 3)
                
                fig7.tight_layout()
                figures['effect_size_visualization'] = fig_to_svg(fig7)
            except Exception as e:
                figures['effect_size_visualization_error'] = str(e)
            
            # Figure 8: Assumption summary
            try:
                fig8, ax8 = plt.subplots(figsize=(10, 6))
                
                # Collect assumption results
                assumption_names = []
                assumption_satisfied = []
                assumption_colors = []
                
                # Normality
                if 'normality' in assumptions and 'results' in assumptions['normality']:
                    assumption_names.append('Normality')
                    assumption_satisfied.append(all(result['result'].value == 'passed' for result in assumptions['normality']['results']))
                    assumption_colors.append(PASTEL_COLORS[0] if assumption_satisfied[-1] else PASTEL_COLORS[3])
                
                # Homogeneity of variance
                if 'homogeneity_of_variance' in assumptions and 'result' in assumptions['homogeneity_of_variance']:
                    assumption_names.append('Equal Variances')
                    assumption_satisfied.append(assumptions['homogeneity_of_variance']['result'].value == 'passed')
                    assumption_colors.append(PASTEL_COLORS[0] if assumption_satisfied[-1] else PASTEL_COLORS[3])
                
                # Sample size
                if 'sample_size' in assumptions and 'results' in assumptions['sample_size']:
                    assumption_names.append('Sample Size')
                    assumption_satisfied.append(all(result['result'].value == 'passed' for result in assumptions['sample_size']['results']))
                    assumption_colors.append(PASTEL_COLORS[0] if assumption_satisfied[-1] else PASTEL_COLORS[3])
                
                # Outliers
                if 'outliers' in assumptions and 'results' in assumptions['outliers']:
                    assumption_names.append('No Outliers')
                    assumption_satisfied.append(all(result['result'].value == 'passed' for result in assumptions['outliers']['results']))
                    assumption_colors.append(PASTEL_COLORS[0] if assumption_satisfied[-1] else PASTEL_COLORS[3])
                
                # Create a visual summary
                y_pos = np.arange(len(assumption_names))
                bars = ax8.barh(y_pos, [1] * len(assumption_names), color=assumption_colors, alpha=0.7)
                
                # Add labels and annotations
                ax8.set_yticks(y_pos)
                ax8.set_yticklabels(assumption_names)
                
                # Add text annotations on the bars
                for i, bar in enumerate(bars):
                    text = 'Satisfied' if assumption_satisfied[i] else 'Violated'
                    text_color = 'white' if assumption_colors[i] == PASTEL_COLORS[3] else 'black'
                    
                    ax8.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2, 
                           text, ha='center', va='center', color=text_color, fontweight='bold')
                
                # Add title
                ax8.set_title('T-Test Assumption Check Summary', fontsize=14)
                
                # Hide x-axis ticks and labels
                ax8.set_xticks([])
                ax8.set_xticklabels([])
                
                # Add overall assessment
                all_assumptions = all(assumption_satisfied)
                most_assumptions = sum(assumption_satisfied) >= len(assumption_satisfied) - 1
                
                if all_assumptions:
                    assessment = "All assumptions are satisfied."
                elif most_assumptions:
                    assessment = "Most assumptions are satisfied. Results should be valid."
                else:
                    assessment = "Multiple assumptions are violated. Consider a non-parametric alternative."
                
                ax8.annotate(assessment, xy=(0.5, -0.1), xycoords='axes fraction', 
                           ha='center', va='center', fontweight='bold',
                           bbox=dict(boxstyle='round', fc='yellow', alpha=0.2))
                
                fig8.tight_layout()
                figures['assumption_summary'] = fig_to_svg(fig8)
            except Exception as e:
                figures['assumption_summary_error'] = str(e)
                
            # Figure 9: Residuals against sequence (homogeneity of variance visualization)
            try:
                fig9, ax9 = plt.subplots(figsize=(10, 6))
                
                # Create a data frame with group indicator
                data_df = pd.DataFrame({
                    'value': pd.concat([group1_clean, group2_clean]),
                    'group': ['Group 1'] * len(group1_clean) + ['Group 2'] * len(group2_clean)
                })
                
                # Calculate residuals (deviation from group means)
                group_means = data_df.groupby('group')['value'].transform('mean')
                residuals = data_df['value'] - group_means
                sequence = np.arange(len(residuals))
                
                # Create scatter plot
                scatter = ax9.scatter(sequence, residuals, c=data_df['group'] == 'Group 1', 
                                    cmap='coolwarm', alpha=0.7)
                
                # Add reference line at y=0
                ax9.axhline(y=0, color='black', linestyle='--', alpha=0.7)
                
                # Add title and labels
                ax9.set_title('Residuals by Sequence', fontsize=14)
                ax9.set_xlabel('Observation Sequence', fontsize=12)
                ax9.set_ylabel('Residual', fontsize=12)
                
                # Add legend
                ax9.legend(*scatter.legend_elements(), title="Group")
                
                # Add homogeneity of variance test result
                if 'homogeneity_of_variance' in assumptions and 'result' in assumptions['homogeneity_of_variance']:
                    hov_passed = assumptions['homogeneity_of_variance']['result'].value == 'passed'
                    hov_test = assumptions['homogeneity_of_variance']['test']
                    
                    result_text = f"{hov_test}: "
                    if hov_passed:
                        result_text += "Equal variances assumption satisfied"
                    else:
                        result_text += "Unequal variances detected"
                        
                    ax9.text(0.05, 0.95, result_text, transform=ax9.transAxes, 
                           va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
                
                fig9.tight_layout()
                figures['residuals_plot'] = fig_to_svg(fig9)
            except Exception as e:
                figures['residuals_plot_error'] = str(e)
                
            # Figure 10: Parametric vs Non-parametric test comparison
            try:
                if 'mann_whitney_u' in additional_stats and 'error' not in additional_stats['mann_whitney_u']:
                    fig10, ax10 = plt.subplots(figsize=(8, 6))
                    
                    # Test results
                    tests = ['t-test', 'Mann-Whitney U']
                    p_values = [p_value, additional_stats['mann_whitney_u']['p_value']]
                    
                    # Create bar chart
                    bars = ax10.bar(tests, p_values, color=[PASTEL_COLORS[0], PASTEL_COLORS[1]], alpha=0.7)
                    
                    # Add horizontal line at alpha
                    ax10.axhline(y=alpha, color='red', linestyle='--', label=f'α = {alpha}')
                    
                    # Add value labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax10.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{height:.4f}', ha='center', va='bottom')
                    
                    # Mark significant tests
                    for i, p in enumerate(p_values):
                        if p < alpha:
                            # Add a marker for significance
                            ax10.text(i, p/2, '*', fontsize=24, ha='center', va='center')
                    
                    # Add title and labels
                    ax10.set_title('Parametric vs. Non-parametric Test Comparison', fontsize=14)
                    ax10.set_ylabel('p-value', fontsize=12)
                    
                    # Add agreement indicator
                    agrees = (p_value < alpha) == (additional_stats['mann_whitney_u']['p_value'] < alpha)
                    agreement_text = "Both tests agree" if agrees else "Tests disagree on significance"
                    
                    ax10.annotate(agreement_text, xy=(0.5, -0.1), xycoords='axes fraction', 
                                ha='center', va='center', fontweight='bold',
                                bbox=dict(boxstyle='round', fc='yellow' if not agrees else 'lightgreen', alpha=0.2))
                    
                    # Add legend
                    ax10.legend()
                    
                    fig10.tight_layout()
                    figures['test_comparison'] = fig_to_svg(fig10)
            except Exception as e:
                figures['test_comparison_error'] = str(e)
        
        return {
            'test': 'Independent T-Test',
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
            'n_samples': int(n),
            'assumptions': assumptions,
            'interpretation': interpretation,
            'additional_statistics': additional_stats,
            'figures': figures
        }
    except Exception as e:
        return {
            'test': 'Independent T-Test',
            'statistic': None,
            'p_value': None,
            'significant': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }