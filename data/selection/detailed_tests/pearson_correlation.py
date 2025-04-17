import traceback
import pandas as pd
from scipy import stats
import numpy as np
from typing import Dict, Any
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Dict, Any
from data.assumptions.tests import (
    OutlierTest,
    LinearityTest,
    NormalityTest
)
from data.assumptions.format import AssumptionTestKeys
from data.assumptions.tests import AssumptionResult
from data.selection.detailed_tests.formatting import fig_to_svg, PASTEL_COLORS, PASTEL_CMAP

def pearson_correlation(x: pd.Series, y: pd.Series, alpha: float) -> Dict[str, Any]:
    """Calculates Pearson correlation and checks significance with comprehensive assumption checks."""
    try:
        # Validate inputs
        if not isinstance(x, pd.Series) or not isinstance(y, pd.Series):
            raise ValueError("Both x and y must be pandas Series objects")
            
        # Ensure we have paired data (drop rows where either x or y is missing)
        valid_data = pd.DataFrame({'x': x, 'y': y}).dropna()
        
        if len(valid_data) < 3:
            return {
                'test': 'Pearson Correlation',
                'statistic': None,
                'p_value': None,
                'significant': False,
                'reason': 'Insufficient data points after removing missing values (need at least 3)'
            }
            
        x_clean = valid_data['x']
        y_clean = valid_data['y']
        n = len(x_clean)
        
        # Run assumption tests using standardized framework
        assumptions = {}
        assumption_violations = []
        
        # 1. Sample Size Test
        sample_size_test = AssumptionTestKeys.SAMPLE_SIZE.value["function"]()
        sample_size_result = sample_size_test.run_test(data=x_clean, min_recommended=10)
        assumptions["sample_size"] = sample_size_result
        
        if sample_size_result.get("result", AssumptionResult.PASSED) != AssumptionResult.PASSED:
            assumption_violations.append("sample_size")
        
        # 2. Normality test for both variables
        normality_test = AssumptionTestKeys.NORMALITY.value["function"]()
        x_normality_result = normality_test.run_test(data=x_clean)
        y_normality_result = normality_test.run_test(data=y_clean)
        
        assumptions['normality'] = {
            'x_variable': x_normality_result,
            'y_variable': y_normality_result
        }
        
        # Check if normality assumption is violated
        if (x_normality_result.get("result", AssumptionResult.PASSED) != AssumptionResult.PASSED or 
            y_normality_result.get("result", AssumptionResult.PASSED) != AssumptionResult.PASSED):
            assumption_violations.append("normality")
        
        # 3. Linearity test
        linearity_test = AssumptionTestKeys.LINEARITY.value["function"]()
        linearity_result = linearity_test.run_test(x=x_clean, y=y_clean)
        assumptions['linearity'] = linearity_result
        
        # Check if linearity assumption is violated
        if linearity_result.get("result", AssumptionResult.PASSED) != AssumptionResult.PASSED:
            assumption_violations.append("linearity")
        
        # 4. Check for outliers
        outlier_test = AssumptionTestKeys.OUTLIERS.value["function"]()
        x_outliers_result = outlier_test.run_test(data=x_clean)
        y_outliers_result = outlier_test.run_test(data=y_clean)
        
        assumptions['outliers'] = {
            'x_variable': x_outliers_result,
            'y_variable': y_outliers_result
        }
        
        # Check if outliers are present
        if (x_outliers_result.get("result", AssumptionResult.PASSED) != AssumptionResult.PASSED or 
            y_outliers_result.get("result", AssumptionResult.PASSED) != AssumptionResult.PASSED):
            assumption_violations.append("outliers")
        
        # 5. Independence Test
        independence_test = AssumptionTestKeys.INDEPENDENCE.value["function"]()
        independence_result = independence_test.run_test(data=pd.DataFrame({'x': x_clean, 'y': y_clean}))
        assumptions["independence"] = independence_result
        
        if independence_result.get("result", AssumptionResult.PASSED) != AssumptionResult.PASSED:
            assumption_violations.append("independence")
        
        # 6. Check for homoscedasticity (constant variance)
        try:
            model = sm.OLS(y_clean, sm.add_constant(x_clean)).fit()
            residuals = model.resid
            
            homoscedasticity_test = AssumptionTestKeys.HOMOSCEDASTICITY.value["function"]()
            homoscedasticity_result = homoscedasticity_test.run_test(residuals=residuals, predicted=model.fittedvalues)
            
            assumptions['homoscedasticity'] = homoscedasticity_result
            
            if homoscedasticity_result.get("result", AssumptionResult.PASSED) != AssumptionResult.PASSED:
                assumption_violations.append("homoscedasticity")
        except Exception as e:
            assumptions['homoscedasticity'] = {
                'result': AssumptionResult.FAILED,
                'details': f"Error testing homoscedasticity: {str(e)}",
                'warnings': ["Homoscedasticity could not be tested due to an error"]
            }
            assumption_violations.append("homoscedasticity")
        
        # 7. Distribution Fit Test (additional check that might be useful)
        distribution_test = AssumptionTestKeys.DISTRIBUTION_FIT.value["function"]()
        x_distribution = distribution_test.run_test(data=x_clean, distribution="normal")
        y_distribution = distribution_test.run_test(data=y_clean, distribution="normal")
        assumptions["distribution_fit_x"] = x_distribution
        assumptions["distribution_fit_y"] = y_distribution
        
        # Calculate correlation
        cor, p_corr = stats.pearsonr(x_clean, y_clean)
        
        # Calculate confidence interval for correlation
        # Fisher's z-transformation
        z = 0.5 * np.log((1 + cor) / (1 - cor))
        se = 1 / np.sqrt(len(x_clean) - 3)
        z_crit = stats.norm.ppf(1 - alpha/2)
        
        # Confidence interval in z-space
        lo_z = z - z_crit * se
        hi_z = z + z_crit * se
        
        # Transform back to correlation
        lo_r = (np.exp(2 * lo_z) - 1) / (np.exp(2 * lo_z) + 1)
        hi_r = (np.exp(2 * hi_z) - 1) / (np.exp(2 * hi_z) + 1)
        
        # Interpret the correlation strength
        strength = "weak"
        if abs(cor) >= 0.7:
            strength = "strong"
        elif abs(cor) >= 0.3:
            strength = "moderate"
            
        # Create interpretation
        interpretation = f"Pearson correlation: r = {cor:.3f}, p = {p_corr:.5f}.\n"
        
        if p_corr < alpha:
            interpretation += f"There is a statistically significant {strength} "
            interpretation += f"{'positive' if cor > 0 else 'negative'} correlation between the variables."
        else:
            interpretation += "There is no statistically significant correlation between the variables."
            
        # Add assumption violation warnings
        if assumption_violations:
            interpretation += f"\n\nWarning: The following assumptions may be violated: {', '.join(assumption_violations)}."
            if "normality" in assumption_violations or "outliers" in assumption_violations:
                interpretation += "\nConsider using Spearman's rank correlation instead."
            if "independence" in assumption_violations:
                interpretation += "\nLack of independence between observations may affect the validity of the results."
            if "sample_size" in assumption_violations:
                interpretation += "\nSmall sample size may limit the reliability of the correlation estimate."
        
        # Calculate additional statistics
        additional_stats = {}
        
        # Calculate alternative correlation coefficients for comparison
        spearman_r, spearman_p = stats.spearmanr(x_clean, y_clean)
        kendall_tau, kendall_p = stats.kendalltau(x_clean, y_clean)
        
        additional_stats['alternative_correlations'] = {
            'spearman': {
                'coefficient': float(spearman_r),
                'p_value': float(spearman_p),
                'significant': spearman_p < alpha
            },
            'kendall': {
                'coefficient': float(kendall_tau),
                'p_value': float(kendall_p),
                'significant': kendall_p < alpha
            }
        }
        
        # Calculate effect sizes
        r_squared = cor ** 2
        # Cohen's standard for r²: small = 0.01, medium = 0.09, large = 0.25
        effect_size_interpretation = "small"
        if r_squared >= 0.25:
            effect_size_interpretation = "large"
        elif r_squared >= 0.09:
            effect_size_interpretation = "medium"
        
        additional_stats['effect_size'] = {
            'r_squared': float(r_squared),
            'cohen_interpretation': effect_size_interpretation,
            'explained_variance_percent': float(r_squared * 100)
        }
        
        # Calculate power for correlation test like in Kendall Tau implementation
        try:
            # Fisher's Z transformation of r
            if cor == 1.0:  # Avoid division by zero
                cor_for_power = 0.9999
            elif cor == -1.0:
                cor_for_power = -0.9999
            else:
                cor_for_power = cor
                
            z = 0.5 * np.log((1 + cor_for_power) / (1 - cor_for_power))
            
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
            
            power_analysis = {
                'power': float(power),
                'recommended_sample_size': int(recommended_n)
            }
            
            additional_stats['power_analysis'] = power_analysis
            
            # Add power analysis to interpretation
            if power < 0.8 and p_corr >= alpha:
                interpretation += f"\n\nStatistical power is low ({power:.2f}). The test may not have enough "
                interpretation += f"power to detect a true effect of this size. Consider increasing the sample size to at least {recommended_n}."
            elif power < 0.8 and p_corr < alpha:
                interpretation += f"\n\nNote: Despite achieving statistical significance, the statistical power is relatively low ({power:.2f})."
        except Exception as e:
            additional_stats['power_analysis'] = {
                'error': str(e)
            }
        
        # Bootstrap confidence intervals (more robust than parametric)
        try:
            n_bootstrap = 2000
            bootstrap_samples = np.zeros(n_bootstrap)
            
            # Generate bootstrap samples
            indices = np.arange(len(x_clean))
            for i in range(n_bootstrap):
                # Sample with replacement
                idx = np.random.choice(indices, size=len(indices), replace=True)
                x_boot = x_clean.iloc[idx]
                y_boot = y_clean.iloc[idx]
                bootstrap_samples[i], _ = stats.pearsonr(x_boot, y_boot)
            
            # Calculate percentile bootstrap CI
            boot_ci_lower = np.percentile(bootstrap_samples, 100 * alpha/2)
            boot_ci_upper = np.percentile(bootstrap_samples, 100 * (1 - alpha/2))
            
            additional_stats['bootstrap'] = {
                'n_samples': n_bootstrap,
                'confidence_interval': [float(boot_ci_lower), float(boot_ci_upper)],
                'standard_error': float(np.std(bootstrap_samples))
            }
        except Exception as e:
            additional_stats['bootstrap'] = {
                'error': str(e)
            }
        
        # Detect influential points using Cook's distance
        try:
            influence = model.get_influence()
            cooks = influence.cooks_distance[0]
            
            # Identify potential influential points (Cook's distance > 4/n)
            threshold = 4 / len(x_clean)
            influential_points = []
            
            for i, (x_val, y_val, cook_d) in enumerate(zip(x_clean, y_clean, cooks)):
                if cook_d > threshold:
                    influential_points.append({
                        'index': x_clean.index[i] if hasattr(x_clean, 'index') else i,
                        'x_value': float(x_val),
                        'y_value': float(y_val),
                        'cooks_distance': float(cook_d)
                    })
            
            additional_stats['influential_points'] = {
                'points': influential_points,
                'threshold': float(threshold),
                'max_cooks_distance': float(np.max(cooks)) if len(cooks) > 0 else 0
            }
        except Exception as e:
            additional_stats['influential_points'] = {
                'error': str(e)
            }
        
        # Generate figures
        figures = {}
        
        # Figure 1: Scatter plot with regression line
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        
        # Plot the scatter points
        ax1.scatter(x_clean, y_clean, alpha=0.6, color=PASTEL_COLORS[0])
        
        # Add regression line
        slope, intercept = np.polyfit(x_clean, y_clean, 1)
        x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
        ax1.plot(x_range, slope * x_range + intercept, color=PASTEL_COLORS[1])
        
        # Highlight influential points if any
        if 'influential_points' in additional_stats and 'points' in additional_stats['influential_points']:
            for point in additional_stats['influential_points']['points']:
                ax1.plot(point['x_value'], point['y_value'], 'ro', markersize=10, fillstyle='none')
        
        # Add correlation information
        text_info = f"r = {cor:.3f}\np = {p_corr:.5f}\nr² = {r_squared:.3f}"
        ax1.text(0.05, 0.95, text_info, transform=ax1.transAxes, 
                verticalalignment='top', horizontalalignment='left',
                bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
        
        # Add confidence interval
        if len(x_clean) > 5:  # Only add confidence band if enough data points
            try:
                from scipy.stats import t
                # Calculate confidence interval for regression
                n = len(x_clean)
                m = 2  # Number of parameters (slope and intercept)
                dof = n - m
                t_val = t.ppf(1 - alpha/2, dof)
                
                # Fitted values for the original x data
                y_fit = slope * x_clean + intercept
                
                # Mean squared error
                mse = np.sum((y_clean - y_fit) ** 2) / dof
                
                # Mean of x
                x_mean = np.mean(x_clean)
                
                # Sum of squares of x
                x_ss = np.sum((x_clean - x_mean) ** 2)
                
                # Prediction interval
                x_pred = np.linspace(x_clean.min(), x_clean.max(), 100)
                
                # Standard error of prediction for new x values
                se_pred = np.sqrt(mse * (1 + 1/n + (x_pred - x_mean)**2 / x_ss))
                
                # Confidence interval
                y_pred = slope * x_pred + intercept
                ci_upper = y_pred + t_val * se_pred
                ci_lower = y_pred - t_val * se_pred
                
                ax1.fill_between(x_pred, ci_lower, ci_upper, color='blue', alpha=0.1)
                
                # Add prediction interval
                pi_factor = np.sqrt(1 + 1/n + (x_pred - x_mean)**2 / x_ss)
                pi_upper = y_pred + t_val * np.sqrt(mse) * pi_factor
                pi_lower = y_pred - t_val * np.sqrt(mse) * pi_factor
                
                ax1.fill_between(x_pred, pi_lower, pi_upper, color='gray', alpha=0.1)
                
                # Add legend for the shaded areas
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='blue', alpha=0.1, label=f'{(1-alpha)*100:.0f}% Confidence Interval'),
                    Patch(facecolor='gray', alpha=0.1, label=f'{(1-alpha)*100:.0f}% Prediction Interval')
                ]
                ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
            
            except Exception as e:
                # Just proceed without confidence bands if calculation fails
                pass
        
        # Labels and title
        x_label = x.name if hasattr(x, 'name') else "X Variable"
        y_label = y.name if hasattr(y, 'name') else "Y Variable"
        
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)
        ax1.set_title(f'Scatter Plot with Regression Line (r = {cor:.3f})')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        fig1.tight_layout()
        figures['scatter_plot'] = fig_to_svg(fig1)
        
        # Figure 2: Distribution plots for X and Y variables
        fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 10))
        
        # X distribution
        sns.histplot(x_clean, kde=True, ax=ax2a, color=PASTEL_COLORS[0])
        ax2a.set_title(f'Distribution of {x_label}')
        
        # Add normal curve for comparison
        x_mean = np.mean(x_clean)
        x_std = np.std(x_clean)
        x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
        x_norm_pdf = stats.norm.pdf(x_range, x_mean, x_std)
        x_norm_pdf = x_norm_pdf * (len(x_clean) * (x_clean.max() - x_clean.min()) / 10)  # Scale to match histogram
        ax2a.plot(x_range, x_norm_pdf, color=PASTEL_COLORS[1], label='Normal Curve')
        ax2a.legend()
        
        # Add shapiro test results
        shapiro_stat, shapiro_p = stats.shapiro(x_clean)
        normal_text = f"Shapiro-Wilk test: p = {shapiro_p:.5f}"
        normal_text += f"\nNormality {'not rejected' if shapiro_p >= alpha else 'rejected'}"
        ax2a.text(0.95, 0.95, normal_text, transform=ax2a.transAxes, 
                 horizontalalignment='right', verticalalignment='top',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
        
        # Y distribution
        sns.histplot(y_clean, kde=True, ax=ax2b, color=PASTEL_COLORS[2])
        ax2b.set_title(f'Distribution of {y_label}')
        
        # Add normal curve for comparison
        y_mean = np.mean(y_clean)
        y_std = np.std(y_clean)
        y_range = np.linspace(y_clean.min(), y_clean.max(), 100)
        y_norm_pdf = stats.norm.pdf(y_range, y_mean, y_std)
        y_norm_pdf = y_norm_pdf * (len(y_clean) * (y_clean.max() - y_clean.min()) / 10)  # Scale to match histogram
        ax2b.plot(y_range, y_norm_pdf, color=PASTEL_COLORS[1], label='Normal Curve')
        ax2b.legend()
        
        # Add shapiro test results
        shapiro_stat, shapiro_p = stats.shapiro(y_clean)
        normal_text = f"Shapiro-Wilk test: p = {shapiro_p:.5f}"
        normal_text += f"\nNormality {'not rejected' if shapiro_p >= alpha else 'rejected'}"
        ax2b.text(0.95, 0.95, normal_text, transform=ax2b.transAxes, 
                 horizontalalignment='right', verticalalignment='top',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
        
        fig2.tight_layout()
        figures['distribution_plots'] = fig_to_svg(fig2)
        
        # Figure 3: Residual plots for checking homoscedasticity and linearity
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Residuals vs fitted values
        fitted = model.fittedvalues
        ax3a.scatter(fitted, residuals, alpha=0.6, color=PASTEL_COLORS[0])
        ax3a.axhline(y=0, color=PASTEL_COLORS[1], linestyle='-')
        
        # Add loess/lowess smooth line to check for patterns
        try:
            lowess = sm.nonparametric.lowess(residuals, fitted, frac=0.6)
            ax3a.plot(lowess[:, 0], lowess[:, 1], 'r-', linewidth=2)
        except:
            # If lowess fails, just skip it
            pass
        
        ax3a.set_xlabel('Fitted values')
        ax3a.set_ylabel('Residuals')
        ax3a.set_title('Residuals vs Fitted')
        ax3a.grid(True, linestyle='--', alpha=0.7)
        
        # Add homoscedasticity test results if available
        if 'homoscedasticity' in assumptions and 'p_value' in assumptions['homoscedasticity']:
            bp_p = assumptions['homoscedasticity']['p_value']
            bp_text = f"Homoscedasticity test: p = {bp_p:.5f}"
            bp_text += f"\nConstant variance {'not rejected' if bp_p >= alpha else 'rejected'}"
            ax3a.text(0.05, 0.05, bp_text, transform=ax3a.transAxes, 
                     bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
        
        # QQ plot to check normality of residuals
        from scipy.stats import probplot
        probplot(residuals, dist="norm", plot=ax3b)
        ax3b.set_title('Normal Q-Q Plot of Residuals')
        ax3b.grid(True, linestyle='--', alpha=0.7)
        
        # Add Shapiro-Wilk test results for residuals
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        qq_text = f"Shapiro-Wilk test: p = {shapiro_p:.5f}"
        qq_text += f"\nNormality {'not rejected' if shapiro_p >= alpha else 'rejected'}"
        ax3b.text(0.05, 0.95, qq_text, transform=ax3b.transAxes, 
                 verticalalignment='top',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
        
        fig3.tight_layout()
        figures['residual_plots'] = fig_to_svg(fig3)
        
        # Figure 4: Bootstrap distribution of correlation coefficient
        if 'bootstrap' in additional_stats and 'n_samples' in additional_stats['bootstrap']:
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            
            # Plot bootstrap distribution
            sns.histplot(bootstrap_samples, kde=True, ax=ax4, color=PASTEL_COLORS[0])
            ax4.axvline(x=cor, color=PASTEL_COLORS[1], linestyle='-', label=f'Sample r = {cor:.3f}')
            
            # Add confidence interval lines
            boot_ci_lower = additional_stats['bootstrap']['confidence_interval'][0]
            boot_ci_upper = additional_stats['bootstrap']['confidence_interval'][1]
            
            ax4.axvline(x=boot_ci_lower, color=PASTEL_COLORS[2], linestyle='--', 
                       label=f'{(1-alpha)*100:.0f}% CI Lower = {boot_ci_lower:.3f}')
            ax4.axvline(x=boot_ci_upper, color=PASTEL_COLORS[2], linestyle='--',
                       label=f'{(1-alpha)*100:.0f}% CI Upper = {boot_ci_upper:.3f}')
            
            ax4.set_xlabel('Correlation Coefficient (r)')
            ax4.set_title('Bootstrap Distribution of Correlation Coefficient')
            ax4.legend()
            ax4.grid(True, linestyle='--', alpha=0.7)
            
            # Add bootstrap statistics
            boot_text = f"Bootstrap SE = {additional_stats['bootstrap']['standard_error']:.4f}"
            boot_text += f"\n{(1-alpha)*100:.0f}% CI: [{boot_ci_lower:.3f}, {boot_ci_upper:.3f}]"
            ax4.text(0.05, 0.95, boot_text, transform=ax4.transAxes, 
                    verticalalignment='top',
                    bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
            
            fig4.tight_layout()
            figures['bootstrap_distribution'] = fig_to_svg(fig4)
        
        # Figure 5: Comparison of correlation coefficients (Pearson, Spearman, Kendall)
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        
        corr_labels = ['Pearson', 'Spearman', 'Kendall']
        corr_values = [cor, spearman_r, kendall_tau]
        
        bars = ax5.bar(corr_labels, corr_values, color=PASTEL_COLORS[:3], alpha=0.7)
        
        # Add line at zero
        ax5.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Add significance markers
        corr_p_values = [p_corr, spearman_p, kendall_p]
        
        for i, (p_val, bar) in enumerate(zip(corr_p_values, bars)):
            height = bar.get_height()
            sign_marker = '*' if p_val < alpha else 'ns'
            y_pos = height + 0.02 if height >= 0 else height - 0.08
            ax5.text(i, y_pos, sign_marker, ha='center', fontsize=12, 
                    fontweight='bold' if p_val < alpha else 'normal')
        
        ax5.set_ylabel('Correlation Coefficient')
        ax5.set_title('Comparison of Correlation Coefficients')
        
        # Add p-values to the figure
        p_text = "p-values:\n"
        p_text += f"Pearson: {p_corr:.5f}\n"
        p_text += f"Spearman: {spearman_p:.5f}\n"
        p_text += f"Kendall: {kendall_p:.5f}"
        
        ax5.text(0.02, 0.02, p_text, transform=ax5.transAxes,
                verticalalignment='bottom',
                bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
        
        # Add legend for significance markers
        ax5.text(0.98, 0.02, "* p < {}\nns not significant".format(alpha), transform=ax5.transAxes,
                horizontalalignment='right', verticalalignment='bottom',
                bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
        
        fig5.tight_layout()
        figures['correlation_comparison'] = fig_to_svg(fig5)
        
        return {
            'test': 'Pearson Correlation',
            'statistic': float(cor),
            'p_value': float(p_corr),
            'significant': p_corr < alpha,
            'interpretation': interpretation,
            'assumptions': assumptions,
            'assumption_violations': assumption_violations,
            'confidence_interval': [float(lo_r), float(hi_r)],
            'effect_size': {
                'r': float(cor),
                'r_squared': float(cor**2),
                'interpretation': strength
            },
            'sample_size': len(x_clean),
            'additional_statistics': additional_stats,
            'figures': figures
        }
    except Exception as e:
        return {
            'test': 'Pearson Correlation',
            'statistic': None,
            'p_value': None,
            'significant': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }