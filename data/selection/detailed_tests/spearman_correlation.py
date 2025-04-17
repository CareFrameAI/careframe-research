import traceback
import pandas as pd
from scipy import stats
import numpy as np
from typing import Dict, Any, List, Tuple
from data.assumptions.tests import (
    NormalityTest, 
    OutlierTest, 
    SampleSizeTest,
    LinearityTest,
    MonotonicityTest,
    IndependenceTest
)
from data.selection.detailed_tests.formatting import fig_to_svg, PASTEL_COLORS, PASTEL_CMAP
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_rank_transformation(x: pd.Series, y: pd.Series) -> Dict[str, Any]:
    """Analyze the impact of rank transformation on the data."""
    # Calculate Pearson correlation on original data
    pearson_r, pearson_p = stats.pearsonr(x, y)
    
    # Calculate Spearman correlation (which is Pearson on ranked data)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    
    # Calculate ranks
    x_ranks = pd.Series(x).rank()
    y_ranks = pd.Series(y).rank()
    
    # Calculate the difference between Pearson and Spearman
    difference = abs(pearson_r - spearman_r)
    
    # Analyze the distribution of the original data vs. ranks
    x_skew = stats.skew(x)
    y_skew = stats.skew(y)
    x_rank_skew = stats.skew(x_ranks)
    y_rank_skew = stats.skew(y_ranks)
    
    # Assess linearity of the original relationship
    linearity_assessment = "linear" if difference < 0.1 else "non-linear"
    
    # Create interpretation
    if difference < 0.1:
        interpretation = "The rank transformation had minimal impact, suggesting the original relationship was already relatively linear and free of extreme values."
    elif difference < 0.3:
        interpretation = "The rank transformation had a moderate impact, suggesting some non-linearity or influence from outliers in the original data."
    else:
        interpretation = "The rank transformation had a substantial impact, indicating a strongly non-linear relationship or significant influence from outliers in the original data."
        
    # Add information about distribution normalization
    if (abs(x_skew) > 1 or abs(y_skew) > 1) and (abs(x_rank_skew) < 0.5 and abs(y_rank_skew) < 0.5):
        interpretation += " The rank transformation effectively normalized the distribution of the data."
    
    return {
        'pearson_r': float(pearson_r),
        'spearman_r': float(spearman_r),
        'difference': float(difference),
        'linearity': linearity_assessment,
        'original_skewness': {
            'x': float(x_skew),
            'y': float(y_skew)
        },
        'rank_skewness': {
            'x': float(x_rank_skew),
            'y': float(y_rank_skew)
        },
        'interpretation': interpretation
    }

def interpret_spearman_rho(rho: float) -> Tuple[str, str]:
    """Provide a detailed interpretation of Spearman's rho value."""
    abs_rho = abs(rho)
    
    if abs_rho < 0.1:
        strength = "negligible"
        practical = "The variables show almost no monotonic relationship; knowing the rank of one variable provides virtually no information about the rank of the other."
    elif abs_rho < 0.3:
        strength = "weak"
        practical = "The variables show a weak monotonic relationship; knowing the rank of one variable provides limited information about the rank of the other."
    elif abs_rho < 0.5:
        strength = "moderate"
        practical = "The variables show a moderate monotonic relationship; knowing the rank of one variable provides some useful information about the rank of the other."
    elif abs_rho < 0.7:
        strength = "moderately strong"
        practical = "The variables show a substantial monotonic relationship; knowing the rank of one variable provides considerable information about the rank of the other."
    elif abs_rho < 0.9:
        strength = "strong"
        practical = "The variables show a strong monotonic relationship; knowing the rank of one variable provides highly reliable information about the rank of the other."
    else:
        strength = "very strong"
        practical = "The variables show an extremely strong monotonic relationship; the ranks of the variables are almost perfectly aligned."
    
    # Add predictive power interpretation
    rho_squared = rho**2
    if rho > 0:
        practical += f" Approximately {rho_squared:.0%} of the variance in ranks can be predicted from the relationship, suggesting that {rho_squared:.0%} of the variation in one variable's ranks can be explained by the other variable's ranks."
    elif rho < 0:
        practical += f" Approximately {rho_squared:.0%} of the variance in ranks can be predicted from the inverse relationship, suggesting that {rho_squared:.0%} of the variation in one variable's ranks can be explained by the opposite of the other variable's ranks."
    
    return strength, practical

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

def compare_correlation_methods(cor: float, ties_present: bool, outliers_present: bool, sample_size: int) -> str:
    """Provide a comparison of when Spearman might be preferred over other correlation methods."""
    insights = []
    
    # Compare with Pearson
    insights.append("Spearman's rank correlation is more appropriate than Pearson when the relationship is non-linear but monotonic, or when data contains outliers.")
    
    # Compare with Kendall
    if ties_present:
        insights.append("With tied ranks present, Kendall's tau might provide a more robust measure of association.")
    
    if sample_size > 100:
        insights.append("For large datasets, Spearman is computationally more efficient than Kendall's tau.")
    
    if outliers_present:
        insights.append("While more robust than Pearson, Spearman can still be affected by extreme outliers. Consider Kendall's tau for even greater robustness.")
    
    if abs(cor) > 0.7:
        insights.append("The strong correlation suggests a consistent monotonic relationship, making Spearman a good choice for measuring association strength.")
    
    # Choose most relevant insights (max 2)
    if len(insights) > 2:
        if ties_present and outliers_present:
            selected_insights = [insights[1], insights[4]]  # Focus on ties and outliers
        elif ties_present:
            selected_insights = [insights[1], insights[0]]  # Focus on ties and Pearson comparison
        elif outliers_present:
            selected_insights = [insights[4], insights[0]]  # Focus on outliers and Pearson comparison
        else:
            selected_insights = [insights[0], insights[3] if abs(cor) > 0.7 else insights[2]]  # General insights
    else:
        selected_insights = insights
    
    return " ".join(selected_insights)

def spearman_correlation(x: pd.Series, y: pd.Series, alpha: float) -> Dict[str, Any]:
    """Calculates Spearman's rank correlation and checks significance with comprehensive assumption checks."""
    try:
        # Validate inputs
        if not isinstance(x, pd.Series) or not isinstance(y, pd.Series):
            raise ValueError("Both x and y must be pandas Series objects")
            
        # Ensure we have paired data (drop rows where either x or y is missing)
        valid_data = pd.DataFrame({'x': x, 'y': y}).dropna()
        
        if len(valid_data) < 3:
            return {
                'test': 'Spearman Correlation',
                'statistic': None,
                'p_value': None,
                'significant': False,
                'reason': 'Insufficient data points after removing missing values (need at least 3)'
            }
            
        x_clean = valid_data['x']
        y_clean = valid_data['y']
        n = len(x_clean)
        
        # Run assumption tests using format.py specifications
        assumptions = {}
        assumption_violations = []
        
        # 1. Normality test for x and y (though not strictly required for Spearman's)
        normality_test = NormalityTest()
        assumptions['normality_x'] = normality_test.run_test(data=x_clean)
        assumptions['normality_y'] = normality_test.run_test(data=y_clean)
        
        # 2. Outlier test for x and y
        outlier_test = OutlierTest()
        assumptions['outliers_x'] = outlier_test.run_test(data=x_clean)
        assumptions['outliers_y'] = outlier_test.run_test(data=y_clean)
        
        # 3. Sample size test
        sample_size_test = SampleSizeTest()
        assumptions['sample_size'] = sample_size_test.run_test(data=x_clean, min_recommended=10)
        
        # 4. Monotonicity test
        monotonicity_test = MonotonicityTest()
        assumptions['monotonicity'] = monotonicity_test.run_test(x=x_clean, y=y_clean)
        
        # 5. Independence test
        independence_test = IndependenceTest()
        assumptions['independence'] = independence_test.run_test(data=valid_data)
        
        # 6. Check for tied ranks (custom test)
        x_ties = len(x_clean) - len(x_clean.unique())
        y_ties = len(y_clean) - len(y_clean.unique())
        
        tie_check_result = {
            'result': 'passed' if (x_ties / len(x_clean) < 0.2) and (y_ties / len(y_clean) < 0.2) else 'warning',
            'message': 'Minimal tied ranks' if (x_ties / len(x_clean) < 0.2) and (y_ties / len(y_clean) < 0.2) else 'Substantial tied ranks present',
            'details': {
                'x_tie_percentage': float(x_ties / len(x_clean) * 100),
                'y_tie_percentage': float(y_ties / len(y_clean) * 100),
            }
        }
        assumptions['tied_ranks'] = tie_check_result
        
        # Check which assumption violations exist
        if assumptions['outliers_x'].get('result') == 'failed' or assumptions['outliers_y'].get('result') == 'failed':
            assumption_violations.append("outliers")
            
        if assumptions['sample_size'].get('result') == 'failed':
            assumption_violations.append("sample_size")
            
        if assumptions['monotonicity'].get('result') == 'failed':
            assumption_violations.append("monotonicity")
            
        if assumptions['independence'].get('result') == 'failed':
            assumption_violations.append("independence")
            
        if tie_check_result['result'] == 'warning':
            assumption_violations.append("tied_ranks")
        
        # Perform rank transformation analysis
        rank_transformation_analysis = analyze_rank_transformation(x_clean, y_clean)
        
        # Calculate correlation
        cor, p_corr = stats.spearmanr(x_clean, y_clean)
        
        # Calculate statistical power for the correlation test
        power_analysis = calculate_correlation_power(n, cor, alpha)
        
        # Create figures for visualization
        figures = {}
        
        # Generate scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=x_clean, y=y_clean, ax=ax, alpha=0.7, palette=PASTEL_COLORS)
        ax.set_xlabel(x.name if hasattr(x, 'name') else 'X Variable')
        ax.set_ylabel(y.name if hasattr(y, 'name') else 'Y Variable')
        ax.set_title(f"Spearman's rho: {cor:.3f} (p={p_corr:.5f})")

        # Add both linear and LOWESS regression lines
        sns.regplot(x=x_clean, y=y_clean, scatter=False, ci=95, 
                    line_kws={'color': PASTEL_COLORS[2], 'label': 'Linear fit'}, ax=ax)
        sns.regplot(x=x_clean, y=y_clean, scatter=False, ci=None, lowess=True,
                    line_kws={'color': PASTEL_COLORS[3], 'linestyle': '--', 'label': 'LOWESS fit'}, ax=ax)

        # Highlight potential outliers
        outlier_test_x = OutlierTest()
        outlier_test_y = OutlierTest()
        x_outlier_results = outlier_test_x.run_test(data=x_clean)
        y_outlier_results = outlier_test_y.run_test(data=y_clean)

        # Extract outlier indices from the test results
        x_outlier_indices = x_outlier_results['outliers']['indices']
        y_outlier_indices = y_outlier_results['outliers']['indices']
        outlier_indices = list(set(x_outlier_indices + y_outlier_indices))

        if outlier_indices:
            ax.scatter(x_clean.iloc[outlier_indices], y_clean.iloc[outlier_indices], 
                      s=100, facecolors='none', edgecolors='red', linewidth=2, label='Potential outliers')

        ax.legend(loc='best')
        
        # Convert to SVG
        scatter_svg = fig_to_svg(fig)
        
        scatter_data = {
            'x': x_clean.values.tolist(),
            'y': y_clean.values.tolist(),
            'x_label': x.name if hasattr(x, 'name') else 'X Variable',
            'y_label': y.name if hasattr(y, 'name') else 'Y Variable',
            'title': f"Spearman's rho: {cor:.3f} (p={p_corr:.5f})",
            'svg': scatter_svg
        }
        figures['scatter_plot'] = scatter_svg
        
        # Generate rank scatter plot
        x_ranks = pd.Series(x_clean).rank()
        y_ranks = pd.Series(y_clean).rank()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        # Add jitter for tied ranks
        sns.scatterplot(x=x_ranks, y=y_ranks, ax=ax, alpha=0.7, palette=PASTEL_COLORS)

        # Add perfect correlation reference line
        min_val = min(x_ranks.min(), y_ranks.min())
        max_val = max(x_ranks.max(), y_ranks.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Perfect correlation')

        # Add regression line for ranks
        sns.regplot(x=x_ranks, y=y_ranks, scatter=False, ci=95,
                    line_kws={'color': PASTEL_COLORS[3], 'label': 'Best fit'}, ax=ax)

        # Highlight tied ranks
        x_tied_values = [i for i, count in x_ranks.value_counts().items() if count > 1]
        y_tied_values = [i for i, count in y_ranks.value_counts().items() if count > 1]
        
        if x_tied_values or y_tied_values:
            # Find indices where either x or y has tied values
            x_tied_mask = x_ranks.isin(x_tied_values)
            y_tied_mask = y_ranks.isin(y_tied_values)
            tied_mask = x_tied_mask | y_tied_mask
            
            # Use the same indices for both x and y to ensure equal length
            if any(tied_mask):
                tied_x_points = x_ranks[tied_mask]
                tied_y_points = y_ranks[tied_mask]
                ax.scatter(tied_x_points, tied_y_points, s=80, facecolors='none', edgecolors='orange', 
                          alpha=0.7, linewidth=1.5, label='Points with tied ranks')

        ax.set_xlabel(f'Rank of {x.name if hasattr(x, "name") else "X Variable"}')
        ax.set_ylabel(f'Rank of {y.name if hasattr(y, "name") else "Y Variable"}')
        ax.set_title(f"Ranked Data Scatter Plot (rho={cor:.3f})")
        ax.legend(loc='best')
        
        # Convert to SVG
        rank_scatter_svg = fig_to_svg(fig)
        
        rank_scatter_data = {
            'x': x_ranks.values.tolist(),
            'y': y_ranks.values.tolist(),
            'x_label': f'Rank of {x.name if hasattr(x, "name") else "X Variable"}',
            'y_label': f'Rank of {y.name if hasattr(y, "name") else "Y Variable"}',
            'title': f"Ranked Data Scatter Plot (rho={cor:.3f})",
            'svg': rank_scatter_svg
        }
        figures['rank_scatter_plot'] = rank_scatter_svg
        
        # Add rank transformation visualization
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original data
        sns.scatterplot(x=x_clean, y=y_clean, ax=axs[0], alpha=0.7, palette=PASTEL_COLORS)
        pearson_r, _ = stats.pearsonr(x_clean, y_clean)
        axs[0].set_title(f"Original Data (Pearson r={pearson_r:.3f})")
        axs[0].set_xlabel(x.name if hasattr(x, 'name') else 'X Variable')
        axs[0].set_ylabel(y.name if hasattr(y, 'name') else 'Y Variable')
        sns.regplot(x=x_clean, y=y_clean, scatter=False, ci=None, ax=axs[0], 
                    line_kws={'color': PASTEL_COLORS[2]})

        # Ranked data
        sns.scatterplot(x=x_ranks, y=y_ranks, ax=axs[1], alpha=0.7, palette=PASTEL_COLORS)
        axs[1].set_title(f"Ranked Data (Spearman rho={cor:.3f})")
        axs[1].set_xlabel(f'Rank of {x.name if hasattr(x, "name") else "X Variable"}')
        axs[1].set_ylabel(f'Rank of {y.name if hasattr(y, "name") else "Y Variable"}')
        sns.regplot(x=x_ranks, y=y_ranks, scatter=False, ci=None, ax=axs[1], 
                    line_kws={'color': PASTEL_COLORS[3]})

        # Ensure equal aspect ratios for better comparison
        try:
            axs[0].set_box_aspect(1)
            axs[1].set_box_aspect(1)
        except AttributeError:
            # set_box_aspect is only available in newer matplotlib versions
            pass

        plt.tight_layout()
        rank_transform_svg = fig_to_svg(fig)
        
        rank_transformation_data = {
            'original_values': {
                'x': x_clean.values.tolist(),
                'y': y_clean.values.tolist()
            },
            'rank_values': {
                'x': x_ranks.values.tolist(),
                'y': y_ranks.values.tolist()
            },
            'title': 'Rank Transformation Analysis',
            'svg': rank_transform_svg
        }
        figures['rank_transformation'] = rank_transform_svg
        
        # Calculate confidence interval for Spearman's rho using bootstrap
        try:
            n_bootstrap = 2000  # Increased for better stability
            bootstrap_rhos = []
            
            for _ in range(n_bootstrap):
                indices = np.random.choice(len(x_clean), size=len(x_clean), replace=True)
                x_boot = x_clean.iloc[indices]
                y_boot = y_clean.iloc[indices]
                rho, _ = stats.spearmanr(x_boot, y_boot)
                bootstrap_rhos.append(rho)
                
            # Calculate percentile-based CI
            lo_rho = np.percentile(bootstrap_rhos, alpha/2 * 100)
            hi_rho = np.percentile(bootstrap_rhos, (1-alpha/2) * 100)
            
            # Calculate bias-corrected CI (more accurate)
            z0 = stats.norm.ppf(sum(r < cor for r in bootstrap_rhos) / n_bootstrap)
            z_alpha = stats.norm.ppf(alpha/2)
            z_1_alpha = stats.norm.ppf(1 - alpha/2)
            
            # Percentiles for bias-corrected CI
            p_low = stats.norm.cdf(2 * z0 + z_alpha)
            p_high = stats.norm.cdf(2 * z0 + z_1_alpha)
            
            lo_rho_bc = np.percentile(bootstrap_rhos, p_low * 100)
            hi_rho_bc = np.percentile(bootstrap_rhos, p_high * 100)
            
            ci = [float(lo_rho_bc), float(hi_rho_bc)]
            ci_method = "bias-corrected bootstrap"
            
            # Add bootstrap distribution to figures
            bootstrap_data = {
                'distribution': bootstrap_rhos,
                'observed_value': cor,
                'ci_lower': ci[0],
                'ci_upper': ci[1],
                'title': "Bootstrap Distribution of Spearman's Rho",
            }
            
        except Exception:
            # Fallback to Fisher's z-transformation (approximate)
            z = 0.5 * np.log((1 + cor) / (1 - cor))
            se = 1 / np.sqrt(len(x_clean) - 3)
            z_crit = stats.norm.ppf(1 - alpha/2)
            
            # Confidence interval in z-space
            lo_z = z - z_crit * se
            hi_z = z + z_crit * se
            
            # Transform back to correlation
            lo_r = (np.exp(2 * lo_z) - 1) / (np.exp(2 * lo_z) + 1)
            hi_r = (np.exp(2 * hi_z) - 1) / (np.exp(2 * hi_z) + 1)
            
            ci = [float(lo_r), float(hi_r)]
            ci_method = "Fisher's z-transformation"
        
        # Interpret the correlation strength and practical significance
        strength, practical_interpretation = interpret_spearman_rho(cor)
        
        # Perform methodological comparison
        method_comparison = compare_correlation_methods(
            cor=cor, 
            ties_present=(x_ties > 0 or y_ties > 0),
            outliers_present=("outliers" in assumption_violations),
            sample_size=n
        )
            
        # Create interpretation
        interpretation = f"Spearman's rank correlation: rho = {cor:.3f}, p = {p_corr:.5f}.\n"
        
        if p_corr < alpha:
            interpretation += f"There is a statistically significant {strength} "
            interpretation += f"{'positive' if cor > 0 else 'negative'} monotonic relationship between the variables.\n"
            interpretation += f"Practical significance: {practical_interpretation}"
        else:
            interpretation += "There is no statistically significant monotonic relationship between the variables."
            
        # Add rank transformation insight
        interpretation += f"\n\nRank transformation insight: {rank_transformation_analysis['interpretation']}"
        
        # Add confidence interval to interpretation
        interpretation += f"\n\nThe {(1-alpha)*100:.1f}% confidence interval for rho is [{ci[0]:.3f}, {ci[1]:.3f}] (calculated using {ci_method})."
        
        # Add coefficient of determination
        interpretation += f"\n\nCoefficient of determination (rho²) = {cor**2:.3f}, suggesting that approximately "
        interpretation += f"{cor**2*100:.1f}% of the variance in ranks can be explained by the relationship."
        
        # Add power analysis to interpretation
        if power_analysis['power'] < 0.8 and p_corr >= alpha:
            interpretation += f"\n\nStatistical power is low ({power_analysis['power']:.2f}). The test may not have enough "
            interpretation += f"power to detect a true effect of this size. Consider increasing the sample size to at least {power_analysis['recommended_sample_size']}."
        elif power_analysis['power'] < 0.8 and p_corr < alpha:
            interpretation += f"\n\nNote: Despite achieving statistical significance, the statistical power is relatively low ({power_analysis['power']:.2f})."
        
        # Add methodological comparison
        interpretation += f"\n\nMethodological insight: {method_comparison}"
        
        # Add assumption violation warnings
        if assumption_violations:
            interpretation += f"\n\nWarning: The following assumptions may be violated: {', '.join(assumption_violations)}."
            if "tied_ranks" in assumption_violations and "tied_ranks" in assumption_violations:
                interpretation += "\nConsider using Kendall's tau correlation instead due to the presence of many tied ranks."
            if "outliers" in assumption_violations:
                interpretation += "\nOutliers can affect Spearman's rho, but less severely than they would affect Pearson correlation."
            if "sample_size" in assumption_violations:
                interpretation += "\nResults should be interpreted with caution due to small sample size."
        
        # Add bootstrap distribution visualization if bootstrap analysis was successful
        if 'bootstrap_rhos' in locals() and len(bootstrap_rhos) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Plot bootstrap distribution with better bin settings
            n_bins = min(50, int(np.sqrt(len(bootstrap_rhos))))
            sns.histplot(bootstrap_rhos, kde=True, ax=ax, color=PASTEL_COLORS[0], bins=n_bins)
            
            # Add vertical line for observed correlation
            ax.axvline(cor, color=PASTEL_COLORS[1], linestyle='--', 
                       linewidth=2, label=f'Observed ρ = {cor:.3f}')
            
            # Add zero reference line
            ax.axvline(0, color='gray', linestyle='-', alpha=0.3, label='Zero correlation')
            
            # Add confidence interval as shaded area
            ax.axvspan(ci[0], ci[1], alpha=0.2, color=PASTEL_COLORS[4], 
                      label=f'{(1-alpha)*100:.0f}% CI: [{ci[0]:.3f}, {ci[1]:.3f}]')
            
            # Add rug plot for more detailed distribution
            sns.rugplot(bootstrap_rhos, ax=ax, color=PASTEL_COLORS[0], alpha=0.3)
            
            ax.set_xlabel("Spearman's ρ")
            ax.set_ylabel("Frequency")
            ax.set_title("Bootstrap Distribution of Spearman's ρ")
            ax.legend(loc='best', framealpha=0.9)  # Add background to legend for better visibility
            
            bootstrap_svg = fig_to_svg(fig)
            figures['bootstrap_distribution'] = bootstrap_svg
        
        return {
            'test': 'Spearman Correlation',
            'statistic': float(cor),
            'p_value': float(p_corr),
            'significant': p_corr < alpha,
            'interpretation': interpretation,
            'assumptions': assumptions,
            'assumption_violations': assumption_violations,
            'confidence_interval': ci,
            'effect_size': {
                'rho': float(cor),
                'rho_squared': float(cor**2),
                'interpretation': strength,
                'practical_significance': practical_interpretation
            },
            'rank_transformation_analysis': rank_transformation_analysis,
            'power_analysis': power_analysis,
            'method_comparison': method_comparison,
            'sample_size': n,
            'figures': figures
        }
    except Exception as e:
        return {
            'test': 'Spearman Correlation',
            'statistic': None,
            'p_value': None,
            'significant': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


