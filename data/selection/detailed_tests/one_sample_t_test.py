import traceback
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional
from data.assumptions.tests import NormalityTest, OutlierTest, SampleSizeTest, DistributionFitTest, IndependenceTest
from data.selection.detailed_tests.formatting import fig_to_svg, PASTEL_COLORS, PASTEL_CMAP
from data.assumptions.tests import AssumptionResult

def one_sample_t_test(data: pd.Series, mu: float, alpha: float, alternative: str = 'two-sided') -> Dict[str, Any]:
    """
    Performs a one-sample t-test with comprehensive statistics, assumption checks, and visualizations.
    
    Parameters:
    -----------
    data : pd.Series
        The sample data to test
    mu : float
        The population mean to test against
    alpha : float
        Significance level
    alternative : str
        'two-sided', 'less', or 'greater'
        
    Returns:
    --------
    Dict[str, Any]
        Comprehensive results including test statistics, effect sizes, assumption checks, and visualizations
    """
    try:
        # Ensure mu is properly converted to float
        mu = float(mu)
        
        # Handle NaN values
        has_nan = False
        original_n = len(data)
        if data.isnull().any():
            has_nan = True
            data = data.dropna()
        
        # Calculate basic statistics
        n = len(data)
        mean = float(data.mean())
        std = float(data.std(ddof=1))
        se = std / np.sqrt(n)
        
        # Perform the t-test
        statistic, p_value = stats.ttest_1samp(a=data, popmean=mu, alternative=alternative)
        
        # Calculate degrees of freedom
        df = n - 1
        
        # Determine significance
        significant = p_value < alpha
        
        # Calculate confidence interval
        if alternative == 'two-sided':
            ci_lower = mean - stats.t.ppf(1 - alpha/2, df) * se
            ci_upper = mean + stats.t.ppf(1 - alpha/2, df) * se
        elif alternative == 'less':
            ci_lower = -np.inf
            ci_upper = mean + stats.t.ppf(1 - alpha, df) * se
        else:  # 'greater'
            ci_lower = mean - stats.t.ppf(1 - alpha, df) * se
            ci_upper = np.inf
        
        # Calculate effect size (Cohen's d)
        cohens_d = (mean - mu) / std
        
        # Determine magnitude of effect size
        if abs(cohens_d) < 0.2:
            effect_magnitude = "Negligible"
        elif abs(cohens_d) < 0.5:
            effect_magnitude = "Small"
        elif abs(cohens_d) < 0.8:
            effect_magnitude = "Medium"
        else:
            effect_magnitude = "Large"
        
        # Test assumptions
        assumptions = {}
        
        # 1. Normality test using the standardized format
        try:
            normality_test = NormalityTest()
            normality_result = normality_test.run_test(data=data)
            assumptions['normality'] = normality_result
        except Exception as e:
            assumptions['normality'] = {
                'result': AssumptionResult.FAILED,
                'details': f'Could not test normality: {str(e)}',
                'error': str(e)
            }
        
        # 2. Sample size check using the standardized format
        try:
            sample_size_test = SampleSizeTest()
            sample_size_result = sample_size_test.run_test(data=data, min_recommended=30)
            assumptions['sample_size'] = sample_size_result
        except Exception as e:
            assumptions['sample_size'] = {
                'result': AssumptionResult.FAILED,
                'details': f'Could not check sample size: {str(e)}',
                'error': str(e)
            }
        
        # 3. Outliers check using the standardized format
        try:
            outlier_test = OutlierTest()
            outlier_result = outlier_test.run_test(data=data)
            assumptions['outliers'] = outlier_result
        except Exception as e:
            assumptions['outliers'] = {
                'result': AssumptionResult.FAILED,
                'details': f'Could not check for outliers: {str(e)}',
                'error': str(e)
            }
        
        # 4. Distribution fit test (new)
        try:
            distribution_test = DistributionFitTest()
            # For t-test, checking for normal distribution
            distribution_result = distribution_test.run_test(data=data, distribution="normal")
            assumptions['distribution_fit'] = distribution_result
        except Exception as e:
            assumptions['distribution_fit'] = {
                'result': AssumptionResult.FAILED,
                'details': f'Could not check distribution fit: {str(e)}',
                'error': str(e)
            }
        
        # 5. Independence test (new)
        try:
            independence_test = IndependenceTest()
            independence_result = independence_test.run_test(data=data)
            assumptions['independence'] = independence_result
        except Exception as e:
            assumptions['independence'] = {
                'result': AssumptionResult.FAILED,
                'details': f'Could not check independence: {str(e)}',
                'error': str(e)
            }
        
        # Calculate additional statistics
        additional_stats = {}
        
        # 1. Bootstrap confidence interval
        try:
            # Number of bootstrap samples
            n_bootstrap = 5000
            bootstrap_means = []
            
            # Generate bootstrap samples
            for _ in range(n_bootstrap):
                # Sample with replacement
                boot_sample = np.random.choice(data, size=n, replace=True)
                
                # Calculate mean
                boot_mean = np.mean(boot_sample)
                bootstrap_means.append(boot_mean)
            
            # Calculate percentile confidence intervals
            bootstrap_ci_lower = np.percentile(bootstrap_means, 2.5)
            bootstrap_ci_upper = np.percentile(bootstrap_means, 97.5)
            
            additional_stats['bootstrap_ci'] = {
                'ci_lower': float(bootstrap_ci_lower),
                'ci_upper': float(bootstrap_ci_upper),
                'n_bootstrap': n_bootstrap
            }
        except Exception as e:
            additional_stats['bootstrap_ci'] = {
                'error': str(e)
            }
        
        # 2. Power analysis
        try:
            from statsmodels.stats.power import TTestPower
            
            # Initialize power analysis
            power_analysis = TTestPower()
            
            # Calculate achieved power
            # If the effect size is 0, set a minimal value to avoid division by zero
            actual_effect_size = cohens_d if cohens_d != 0 else 0.01
            
            achieved_power = power_analysis.power(effect_size=abs(actual_effect_size), 
                                               nobs=n, 
                                               alpha=alpha, 
                                               alternative=alternative)
            
            # Recommend sample size for 80% power
            recommended_n = power_analysis.solve_power(effect_size=abs(actual_effect_size), 
                                                    power=0.8, 
                                                    alpha=alpha, 
                                                    alternative=alternative)
            
            additional_stats['power_analysis'] = {
                'achieved_power': float(achieved_power),
                'recommended_n': int(np.ceil(recommended_n)),
                'actual_effect_size': float(actual_effect_size),
                'is_powered': achieved_power >= 0.8
            }
        except Exception as e:
            additional_stats['power_analysis'] = {
                'error': str(e)
            }
        
        # 3. Bayesian estimation
        try:
            # Approximate Bayesian approach using t-distribution
            from scipy.stats import t
            
            # Prior: Using a weakly informative prior
            prior_mean = mu
            prior_std = std * 3  # Wide prior
            
            # Likelihood: Based on data
            likelihood_mean = mean
            likelihood_std = se
            
            # Posterior mean (weighted average)
            posterior_precision = 1/prior_std**2 + n/std**2
            posterior_mean = (prior_mean/prior_std**2 + n*mean/std**2) / posterior_precision
            posterior_std = np.sqrt(1 / posterior_precision)
            
            # Calculate 95% credible interval
            credible_lower = posterior_mean - 1.96 * posterior_std
            credible_upper = posterior_mean + 1.96 * posterior_std
            
            # Calculate Bayes Factor using BIC approximation
            # BF10 = exp((BIC_null - BIC_alt) / 2)
            # For t-test, this approximation is reasonable
            log_bf10 = n/2 * np.log(1 + (statistic**2)/n)
            bayes_factor = np.exp(log_bf10)
            
            # Interpret Bayes Factor
            if bayes_factor < 1/30:
                bf_interpretation = "Very strong evidence for null"
            elif bayes_factor < 1/10:
                bf_interpretation = "Strong evidence for null"
            elif bayes_factor < 1/3:
                bf_interpretation = "Moderate evidence for null"
            elif bayes_factor < 1:
                bf_interpretation = "Anecdotal evidence for null"
            elif bayes_factor < 3:
                bf_interpretation = "Anecdotal evidence for alternative"
            elif bayes_factor < 10:
                bf_interpretation = "Moderate evidence for alternative"
            elif bayes_factor < 30:
                bf_interpretation = "Strong evidence for alternative"
            else:
                bf_interpretation = "Very strong evidence for alternative"
            
            additional_stats['bayesian_analysis'] = {
                'posterior_mean': float(posterior_mean),
                'posterior_std': float(posterior_std),
                'credible_lower': float(credible_lower),
                'credible_upper': float(credible_upper),
                'bayes_factor': float(bayes_factor),
                'bf_interpretation': bf_interpretation
            }
        except Exception as e:
            additional_stats['bayesian_analysis'] = {
                'error': str(e)
            }
        
        # 4. Equivalence testing (TOST - Two One-Sided Tests)
        try:
            # Define equivalence bounds (e.g., +/- 0.5 * std)
            # This approach defines "practically equivalent" as a difference less than
            # 0.5 standard deviations in either direction
            bound = 0.5 * std
            
            # Perform two one-sided tests
            t_lower = (mean - (mu - bound)) / se
            p_lower = 1 - stats.t.cdf(t_lower, df)
            
            t_upper = ((mu + bound) - mean) / se
            p_upper = 1 - stats.t.cdf(t_upper, df)
            
            # The overall p-value is the maximum of the two p-values
            p_tost = max(p_lower, p_upper)
            
            # The means are equivalent if both one-sided tests are significant
            equivalent = p_tost < alpha
            
            additional_stats['equivalence_test'] = {
                'bound': float(bound),
                'p_value': float(p_tost),
                'equivalent': equivalent,
                'interpretation': "The sample mean is statistically equivalent to the reference value (within bounds of ±{:.3f})".format(bound) if equivalent 
                                else "Cannot conclude that the sample mean is equivalent to the reference value"
            }
        except Exception as e:
            additional_stats['equivalence_test'] = {
                'error': str(e)
            }
        
        # Create interpretation
        interpretation = f"One-Sample T-Test comparing sample mean to μ = {mu}\n\n"
        
        # Basic results
        interpretation += f"t({df}) = {statistic:.3f}, p = {p_value:.5f}\n"
        interpretation += f"Sample mean = {mean:.3f}, SD = {std:.3f}, SE = {se:.3f}\n"
        interpretation += f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]\n"
        interpretation += f"Effect size: Cohen's d = {cohens_d:.3f} ({effect_magnitude.lower()} effect)\n\n"
        
        # Sample information
        if has_nan:
            interpretation += f"Note: {original_n - n} missing values were removed from the analysis.\n"
        interpretation += f"Sample size: n = {n}\n\n"
        
        # Assumptions
        interpretation += "Assumption tests:\n"
        
        # Normality
        if 'normality' in assumptions and 'result' in assumptions['normality']:
            norm_result = assumptions['normality']
            norm_test = norm_result.get('test_used', 'Normality')
            norm_pvalue = norm_result.get('p_value', None)
            norm_passed = norm_result['result'].value == 'passed'
            
            interpretation += f"- Normality: {norm_test} test, "
            if norm_pvalue is not None:
                interpretation += f"p = {norm_pvalue:.5f}, "
            
            if norm_passed:
                interpretation += "assumption satisfied.\n"
            else:
                interpretation += "assumption potentially violated. "
                
                # Check if sample size is large enough for CLT
                if n >= 30:
                    interpretation += "However, with n >= 30, the t-test is robust to violations of normality due to the Central Limit Theorem.\n"
                else:
                    interpretation += "Consider using a non-parametric alternative like the Wilcoxon signed-rank test.\n"
        
        # Sample size
        if 'sample_size' in assumptions and 'result' in assumptions['sample_size']:
            size_result = assumptions['sample_size']
            size_passed = size_result['result'].value == 'passed'
            
            interpretation += f"- Sample size: n = {n}, "
            
            if size_passed:
                interpretation += "sample size is adequate.\n"
            else:
                interpretation += "sample size may be too small; interpret results with caution.\n"
        
        # Outliers
        if 'outliers' in assumptions and 'result' in assumptions['outliers']:
            outlier_result = assumptions['outliers']
            outlier_passed = outlier_result['result'].value == 'passed'
            outlier_count = outlier_result.get('num_outliers', 0)
            
            interpretation += f"- Outliers: "
            
            if outlier_passed:
                interpretation += "No significant outliers detected.\n"
            else:
                interpretation += f"Detected {outlier_count} potential outliers. "
                interpretation += "Consider removing outliers or using a robust method.\n"
        
        # Distribution fit (new)
        if 'distribution_fit' in assumptions and 'result' in assumptions['distribution_fit']:
            dist_result = assumptions['distribution_fit']
            dist_passed = dist_result['result'].value == 'passed'
            
            interpretation += f"- Distribution fit: "
            
            if dist_passed:
                interpretation += "Data appears to follow the normal distribution.\n"
            else:
                interpretation += "Data may not follow the normal distribution. "
                interpretation += "Consider transformations or non-parametric methods.\n"
        
        # Independence (new)
        if 'independence' in assumptions and 'result' in assumptions['independence']:
            indep_result = assumptions['independence']
            indep_passed = indep_result['result'].value == 'passed'
            
            interpretation += f"- Independence: "
            
            if indep_passed:
                interpretation += "Data appears to be independent.\n"
            else:
                interpretation += "Data may not be independent. "
                interpretation += "Consider methods that account for dependence structure.\n"
        
        # Power analysis
        if 'power_analysis' in additional_stats and 'achieved_power' in additional_stats['power_analysis']:
            power_result = additional_stats['power_analysis']
            interpretation += f"- Statistical Power: {power_result['achieved_power']:.3f}, "
            
            if power_result['is_powered']:
                interpretation += "the test has adequate power.\n"
            else:
                interpretation += f"the test is underpowered. A sample size of approximately {power_result['recommended_n']} would be needed for 80% power.\n"
        
        # Conclusion
        interpretation += f"\nConclusion: There is {'a statistically significant' if significant else 'no statistically significant'} "
        
        if alternative == 'two-sided':
            interpretation += f"difference between the sample mean and the reference value μ = {mu} (p = {p_value:.5f}). "
        elif alternative == 'less':
            interpretation += f"evidence that the sample mean is less than the reference value μ = {mu} (p = {p_value:.5f}). "
        else:  # 'greater'
            interpretation += f"evidence that the sample mean is greater than the reference value μ = {mu} (p = {p_value:.5f}). "
        
        if significant:
            interpretation += f"The {effect_magnitude.lower()} effect size (d = {cohens_d:.3f}) suggests that "
            
            if mean > mu:
                interpretation += f"the sample mean is {abs(mean - mu):.3f} units higher than the reference value μ = {mu}."
            else:
                interpretation += f"the sample mean is {abs(mean - mu):.3f} units lower than the reference value μ = {mu}."
                
        # Bayesian perspective if available
        if 'bayesian_analysis' in additional_stats and 'bf_interpretation' in additional_stats['bayesian_analysis']:
            bayes_result = additional_stats['bayesian_analysis']
            interpretation += f"\n\nFrom a Bayesian perspective, the data show {bayes_result['bf_interpretation'].lower()} "
            interpretation += f"(Bayes Factor = {bayes_result['bayes_factor']:.3f})."
        
        # Create visualizations
        figures = {}
        
        # Figure 1: Histogram with normal curve and reference mean
        try:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            
            # Create histogram
            sns.histplot(data, kde=True, ax=ax1, alpha=0.6, color=PASTEL_COLORS[0])
            
            # Add vertical lines for sample mean and reference mean
            ax1.axvline(x=mean, color=PASTEL_COLORS[2], linestyle='-', linewidth=2, label=f'Sample Mean = {mean:.3f}')
            ax1.axvline(x=mu, color=PASTEL_COLORS[4], linestyle='--', linewidth=2, label=f'Reference Mean (μ) = {mu:.3f}')
            
            # Add confidence interval
            if alternative == 'two-sided':
                ax1.axvspan(ci_lower, ci_upper, alpha=0.2, color=PASTEL_COLORS[2], label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
            
            # Add title and labels
            ax1.set_title(f'Distribution of Data with Sample Mean and Reference Value', fontsize=14)
            ax1.set_xlabel('Value', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            
            # Add legend
            ax1.legend()
            
            # Add annotation with test results
            result_text = f"t({df}) = {statistic:.3f}, p = {p_value:.5f}"
            if significant:
                result_text += " *"
            ax1.annotate(result_text, xy=(0.5, 0.96), xycoords='axes fraction', 
                        ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            fig1.tight_layout()
            figures['histogram'] = fig_to_svg(fig1)
        except Exception as e:
            figures['histogram_error'] = str(e)
        
        # Figure 2: QQ Plot for normality assessment
        try:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            # Calculate quantiles
            sorted_data = np.sort(data)
            n_data = len(sorted_data)
            p = np.arange(1, n_data + 1) / (n_data + 1)  # Quantile probabilities
            
            # Calculate theoretical quantiles
            theoretical_quantiles = stats.norm.ppf(p, loc=mean, scale=std)
            
            # Create QQ plot
            ax2.scatter(theoretical_quantiles, sorted_data, alpha=0.7)
            
            # Add reference line
            min_val = min(np.min(theoretical_quantiles), np.min(sorted_data))
            max_val = max(np.max(theoretical_quantiles), np.max(sorted_data))
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            # Add title and labels
            ax2.set_title('Normal Q-Q Plot', fontsize=14)
            ax2.set_xlabel('Theoretical Quantiles', fontsize=12)
            ax2.set_ylabel('Sample Quantiles', fontsize=12)
            
            # Add normality test result
            if 'normality' in assumptions and 'result' in assumptions['normality']:
                p_val = assumptions['normality']['p_value']
                test_name = assumptions['normality']['test']
                
                result_text = f"{test_name}: p = {p_val:.4f}"
                if p_val < alpha:
                    result_text += " *\nNon-normal"
                else:
                    result_text += "\nNormal"
                    
                ax2.text(0.05, 0.95, result_text, transform=ax2.transAxes, 
                        va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
            
            fig2.tight_layout()
            figures['qq_plot'] = fig_to_svg(fig2)
        except Exception as e:
            figures['qq_plot_error'] = str(e)
        
        # Figure 3: Bootstrap Distribution of Sample Mean
        try:
            if 'bootstrap_ci' in additional_stats and 'error' not in additional_stats['bootstrap_ci']:
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                
                # Create histogram of bootstrap means
                bootstrap_means = [np.random.choice(data, size=n, replace=True).mean() for _ in range(1000)]
                sns.histplot(bootstrap_means, kde=True, ax=ax3, color=PASTEL_COLORS[0], alpha=0.6)
                
                # Add vertical lines for sample mean, reference mean, and CI
                ax3.axvline(x=mean, color=PASTEL_COLORS[2], linestyle='-', linewidth=2, label=f'Sample Mean = {mean:.3f}')
                ax3.axvline(x=mu, color=PASTEL_COLORS[4], linestyle='--', linewidth=2, label=f'Reference Mean (μ) = {mu:.3f}')
                
                # Add bootstrap CI
                ci_lower = additional_stats['bootstrap_ci']['ci_lower']
                ci_upper = additional_stats['bootstrap_ci']['ci_upper']
                ax3.axvspan(ci_lower, ci_upper, alpha=0.2, color=PASTEL_COLORS[1], label=f'95% Bootstrap CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
                
                # Add title and labels
                ax3.set_title('Bootstrap Distribution of Sample Mean', fontsize=14)
                ax3.set_xlabel('Mean Value', fontsize=12)
                ax3.set_ylabel('Frequency', fontsize=12)
                
                # Add legend
                ax3.legend()
                
                # Add significance indicator
                if significant:
                    if (mu < ci_lower) or (mu > ci_upper):
                        significance_text = "Significant difference (μ outside CI)"
                    else:
                        significance_text = "Inconsistent results between t-test and bootstrap CI"
                else:
                    if (mu >= ci_lower) and (mu <= ci_upper):
                        significance_text = "Non-significant difference (μ inside CI)"
                    else:
                        significance_text = "Inconsistent results between t-test and bootstrap CI"
                
                ax3.annotate(significance_text, xy=(0.5, 0.05), xycoords='axes fraction', 
                            ha='center', va='bottom',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                fig3.tight_layout()
                figures['bootstrap_distribution'] = fig_to_svg(fig3)
        except Exception as e:
            figures['bootstrap_distribution_error'] = str(e)
        
        # Figure 4: Power Analysis Curve
        try:
            if 'power_analysis' in additional_stats and 'error' not in additional_stats['power_analysis']:
                fig4, ax4 = plt.subplots(figsize=(10, 6))
                
                # Generate sample sizes for plotting
                sample_sizes = np.linspace(5, max(100, n*2), 100)
                
                # Calculate power for each sample size
                power_levels = []
                for size in sample_sizes:
                    power = TTestPower().power(effect_size=abs(actual_effect_size), 
                                              nobs=size,
                                              alpha=alpha,
                                              alternative=alternative)
                    power_levels.append(power)
                
                # Plot power curve
                ax4.plot(sample_sizes, power_levels, 'b-', linewidth=2)
                
                # Mark the current sample size and power
                ax4.plot([n], [achieved_power], 'ro', markersize=8)
                
                # Add reference lines for 80% and 90% power
                ax4.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80% Power')
                ax4.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='90% Power')
                
                # Add vertical line for current sample size
                ax4.axvline(x=n, color='r', linestyle='-', alpha=0.7, label=f'Current n = {n}')
                
                # Add title and labels
                ax4.set_title(f'Power Analysis for d = {actual_effect_size:.3f}', fontsize=14)
                ax4.set_xlabel('Sample Size (n)', fontsize=12)
                ax4.set_ylabel('Statistical Power (1-β)', fontsize=12)
                
                # Set axis limits
                ax4.set_xlim(0, max(sample_sizes))
                ax4.set_ylim(0, 1.05)
                
                # Add legend
                ax4.legend()
                
                # Add annotation for required sample sizes
                required_n_80 = additional_stats['power_analysis']['recommended_n']
                required_n_90 = int(TTestPower().solve_power(effect_size=abs(actual_effect_size), 
                                                          power=0.9, 
                                                          alpha=alpha,
                                                          alternative=alternative))
                
                ax4.annotate(f"Required n for 80% power: {required_n_80}\nRequired n for 90% power: {required_n_90}",
                           xy=(0.05, 0.95), xycoords='axes fraction', 
                           ha='left', va='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                fig4.tight_layout()
                figures['power_curve'] = fig_to_svg(fig4)
        except Exception as e:
            figures['power_curve_error'] = str(e)
        
        # Figure 5: Bayesian Analysis
        try:
            if 'bayesian_analysis' in additional_stats and 'error' not in additional_stats['bayesian_analysis']:
                fig5, ax5 = plt.subplots(figsize=(10, 6))
                
                # Extract parameters
                posterior_mean = additional_stats['bayesian_analysis']['posterior_mean']
                posterior_std = additional_stats['bayesian_analysis']['posterior_std']
                bf10 = additional_stats['bayesian_analysis']['bayes_factor']
                
                # Create a range of values for plotting
                x = np.linspace(mu - 4*std, mu + 4*std, 1000)
                
                # Calculate prior, likelihood, and posterior densities
                prior = stats.norm.pdf(x, loc=mu, scale=std*3)
                likelihood = stats.norm.pdf(x, loc=mean, scale=se)
                posterior = stats.norm.pdf(x, loc=posterior_mean, scale=posterior_std)
                
                # Scale for better visualization
                prior = prior / prior.max() * likelihood.max()
                
                # Plot densities
                ax5.plot(x, prior, color=PASTEL_COLORS[5], linestyle='--', alpha=0.7, label='Prior')
                ax5.plot(x, likelihood, color=PASTEL_COLORS[2], alpha=0.7, label='Likelihood')
                ax5.plot(x, posterior, color=PASTEL_COLORS[0], alpha=0.9, label='Posterior')
                
                # Add vertical lines for means
                ax5.axvline(x=mu, color=PASTEL_COLORS[5], linestyle='--', alpha=0.7, label=f'Reference (μ) = {mu:.3f}')
                ax5.axvline(x=mean, color=PASTEL_COLORS[2], linestyle='-', alpha=0.7, label=f'Sample Mean = {mean:.3f}')
                ax5.axvline(x=posterior_mean, color=PASTEL_COLORS[0], linestyle='-', alpha=0.7, label=f'Posterior Mean = {posterior_mean:.3f}')
                
                # Add title and labels
                ax5.set_title('Bayesian Analysis', fontsize=14)
                ax5.set_xlabel('Value', fontsize=12)
                ax5.set_ylabel('Density (scaled)', fontsize=12)
                
                # Add legend
                ax5.legend()
                
                # Add Bayes Factor information
                bf_text = f"Bayes Factor (BF₁₀) = {bf10:.3f}\n{additional_stats['bayesian_analysis']['bf_interpretation']}"
                ax5.annotate(bf_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                           ha='left', va='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                fig5.tight_layout()
                figures['bayesian_analysis'] = fig_to_svg(fig5)
        except Exception as e:
            figures['bayesian_analysis_error'] = str(e)
        
        # Figure 6: Assumption Check Summary
        try:
            fig6, ax6 = plt.subplots(figsize=(10, 6))
            
            # Collect assumption results
            assumption_names = []
            assumption_satisfied = []
            assumption_colors = []
            
            # Iterate through all assumption results
            for assumption_name, assumption_data in assumptions.items():
                if 'result' in assumption_data:
                    display_name = assumption_name.replace('_', ' ').title()
                    assumption_names.append(display_name)
                    # Check if the assumption passed
                    assumption_passed = assumption_data['result'].value == 'passed'
                    assumption_satisfied.append(assumption_passed)
                    # Assign colors based on result
                    if assumption_data['result'].value == 'passed':
                        assumption_colors.append(PASTEL_COLORS[1])  # green
                    elif assumption_data['result'].value == 'warning':
                        assumption_colors.append(PASTEL_COLORS[3])  # yellow/orange
                    else:
                        assumption_colors.append(PASTEL_COLORS[0])  # red
            
            # Create a visual summary
            y_pos = np.arange(len(assumption_names))
            bars = ax6.barh(y_pos, [1] * len(assumption_names), color=assumption_colors, alpha=0.7)
            
            # Add labels and annotations
            ax6.set_yticks(y_pos)
            ax6.set_yticklabels(assumption_names)
            
            # Add text annotations on the bars
            for i, bar in enumerate(bars):
                text = 'Satisfied' if assumption_satisfied[i] else 'Violated'
                text_color = 'white' if assumption_colors[i] == PASTEL_COLORS[0] else 'black'
                
                ax6.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2, 
                       text, ha='center', va='center', color=text_color, fontweight='bold')
            
            # Add title
            ax6.set_title('Assumption Check Summary', fontsize=14)
            
            # Hide x-axis ticks and labels
            ax6.set_xticks([])
            ax6.set_xticklabels([])
            
            # No overall assessment as per requirements
            
            fig6.tight_layout()
            figures['assumption_summary'] = fig_to_svg(fig6)
        except Exception as e:
            figures['assumption_summary_error'] = str(e)
        
        return {
            'test': 'One-Sample T-Test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': significant,
            'df': int(df),
            'mean': float(mean),
            'std': float(std),
            'se': float(se),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'cohens_d': float(cohens_d),
            'effect_magnitude': effect_magnitude,
            'n_samples': int(n),
            'mu': float(mu),
            'alternative': alternative,
            'assumptions': assumptions,
            'additional_statistics': additional_stats,
            'interpretation': interpretation,
            'figures': figures,
            'summary': f"Warning: {original_n - n} NaN values removed." if has_nan else ""
        }
    except Exception as e:
        return {
            'test': 'One-Sample T-Test',
            'statistic': None,
            'p_value': None,
            'satisfied': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'reason': str(e)
        }