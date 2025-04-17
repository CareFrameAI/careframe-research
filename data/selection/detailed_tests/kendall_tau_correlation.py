import traceback
import pandas as pd
from scipy import stats
import numpy as np
from typing import Dict, Any
from typing import Dict, Any
import traceback
import pandas as pd
from scipy import stats
import numpy as np
from typing import Dict, Any
import statsmodels.api as sm
from typing import Dict, Any
import io
import matplotlib.pyplot as plt
import seaborn as sns
from data.selection.detailed_tests.formatting import fig_to_svg, PASTEL_COLORS, PASTEL_CMAP

from data.assumptions.tests import SampleSizeTest
from data.assumptions.format import AssumptionTestKeys
from data.assumptions.tests import AssumptionResult

import traceback
import pandas as pd
from scipy import stats
import numpy as np
from typing import Dict, Any, List, Tuple
from data.assumptions.tests import SampleSizeTest, OutlierTest


def calculate_concordant_discordant_pairs(x: pd.Series, y: pd.Series) -> Tuple[int, int, int]:
    """Calculate the number of concordant, discordant, and tied pairs for Kendall's tau."""
    n = len(x)
    concordant = 0
    discordant = 0
    tied_x = 0
    tied_y = 0
    tied_both = 0
    
    for i in range(n):
        for j in range(i+1, n):
            x_diff = x.iloc[i] - x.iloc[j]
            y_diff = y.iloc[i] - y.iloc[j]
            
            if x_diff == 0 and y_diff == 0:
                tied_both += 1
            elif x_diff == 0:
                tied_x += 1
            elif y_diff == 0:
                tied_y += 1
            elif (x_diff > 0 and y_diff > 0) or (x_diff < 0 and y_diff < 0):
                concordant += 1
            else:
                discordant += 1
    
    tied_pairs = tied_x + tied_y + tied_both
    
    return concordant, discordant, tied_pairs

def interpret_kendall_tau(tau: float) -> Tuple[str, str]:
    """Provide a detailed interpretation of Kendall's tau value."""
    abs_tau = abs(tau)
    
    if abs_tau < 0.1:
        strength = "negligible"
        practical = "The variables show almost no association; knowing one variable provides virtually no information about the other."
    elif abs_tau < 0.3:
        strength = "weak"
        practical = "The variables show minimal association; knowing one variable provides limited information about the other."
    elif abs_tau < 0.5:
        strength = "moderate"
        practical = "The variables show a moderate association; knowing one variable provides some useful information about the other."
    elif abs_tau < 0.7:
        strength = "moderately strong"
        practical = "The variables show substantial association; knowing one variable provides considerable information about the other."
    elif abs_tau < 0.9:
        strength = "strong"
        practical = "The variables show a strong association; knowing one variable provides highly reliable information about the other."
    else:
        strength = "very strong"
        practical = "The variables show an extremely strong association; they are almost completely predictable from each other."
    
    # Add prediction accuracy interpretation based on probability of concordance
    prob_concordance = (tau + 1) / 2
    if tau > 0:
        practical += f" The probability of concordance is {prob_concordance:.2f}, meaning that for a randomly selected pair of observations, there's a {prob_concordance:.0%} chance that they rank in the same order on both variables."
    elif tau < 0:
        practical += f" The probability of discordance is {1-prob_concordance:.2f}, meaning that for a randomly selected pair of observations, there's a {1-prob_concordance:.0%} chance that they rank in the opposite order on both variables."
    
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

def kendall_tau_correlation(x: pd.Series, y: pd.Series, alpha: float) -> Dict[str, Any]:
    """Calculates Kendall's Tau correlation and checks significance with comprehensive assumption checks."""
    try:
        # Validate inputs
        if not isinstance(x, pd.Series) or not isinstance(y, pd.Series):
            raise ValueError("Both x and y must be pandas Series objects")
            
        # Ensure we have paired data (drop rows where either x or y is missing)
        valid_data = pd.DataFrame({'x': x, 'y': y}).dropna()
        
        if len(valid_data) < 3:
            return {
                'test': 'Kendall Tau Correlation',
                'statistic': None,
                'p_value': None,
                'significant': False,
                'reason': 'Insufficient data points after removing missing values (need at least 3)'
            }
            
        x_clean = valid_data['x']
        y_clean = valid_data['y']
        n = len(x_clean)
        
        # Assumptions testing using standardized framework
        assumptions = {}
        assumption_violations = []
        
        # 1. Sample Size Test
        sample_size_test = AssumptionTestKeys.SAMPLE_SIZE.value["function"]()
        sample_size_result = sample_size_test.run_test(data=x_clean, min_recommended=10)
        assumptions["sample_size"] = sample_size_result
        
        if not sample_size_result.get("result", AssumptionResult.PASSED) == AssumptionResult.PASSED:
            assumption_violations.append("sample_size")
        
        # 2. Outlier Test
        outlier_test = AssumptionTestKeys.OUTLIERS.value["function"]()
        x_outliers_result = outlier_test.run_test(data=x_clean)
        y_outliers_result = outlier_test.run_test(data=y_clean)
        
        assumptions["outliers"] = {
            "x_variable": x_outliers_result,
            "y_variable": y_outliers_result
        }
        
        if (x_outliers_result.get("result", AssumptionResult.PASSED) != AssumptionResult.PASSED or 
            y_outliers_result.get("result", AssumptionResult.PASSED) != AssumptionResult.PASSED):
            assumption_violations.append("outliers")
            
        # 3. Monotonicity Test
        monotonicity_test = AssumptionTestKeys.MONOTONICITY.value["function"]()
        monotonicity_result = monotonicity_test.run_test(x=x_clean, y=y_clean)
        assumptions["monotonicity"] = monotonicity_result
        
        if monotonicity_result.get("result", AssumptionResult.PASSED) != AssumptionResult.PASSED:
            assumption_violations.append("monotonicity")
        
        # 4. Independence Test
        independence_test = AssumptionTestKeys.INDEPENDENCE.value["function"]()
        independence_result = independence_test.run_test(data=pd.DataFrame({'x': x_clean, 'y': y_clean}))
        assumptions["independence"] = independence_result
        
        if independence_result.get("result", AssumptionResult.PASSED) != AssumptionResult.PASSED:
            assumption_violations.append("independence")
            
        # 5. Tie analysis (custom test as no specific tie test exists in format.py)
        x_ties = len(x_clean) - len(x_clean.unique())
        y_ties = len(y_clean) - len(y_clean.unique())
        
        ties_result = {
            "result": AssumptionResult.PASSED if (x_ties / len(x_clean) < 0.2 and y_ties / len(y_clean) < 0.2) 
                      else AssumptionResult.WARNING,
            "details": f"X has {x_ties} ties ({x_ties/len(x_clean)*100:.1f}%), Y has {y_ties} ties ({y_ties/len(y_clean)*100:.1f}%)",
            "warnings": [] if (x_ties / len(x_clean) < 0.2 and y_ties / len(y_clean) < 0.2) 
                         else ["High percentage of ties may affect interpretation"]
        }
        assumptions["ties"] = ties_result
        
        if (x_ties / len(x_clean) > 0.5) or (y_ties / len(y_clean) > 0.5):
            assumption_violations.append("excessive_ties")
        
        # 6. Distribution Fit Test
        distribution_test = AssumptionTestKeys.DISTRIBUTION_FIT.value["function"]()
        x_distribution = distribution_test.run_test(data=x_clean, distribution="normal")
        y_distribution = distribution_test.run_test(data=y_clean, distribution="normal")
        assumptions["distribution_fit_x"] = x_distribution
        assumptions["distribution_fit_y"] = y_distribution
        
        
        # Calculate correlation - use tau-b which corrects for ties
        cor, p_corr = stats.kendalltau(x_clean, y_clean)
        
        # Calculate concordant and discordant pairs (key insight for Kendall tau)
        concordant_pairs, discordant_pairs, tied_pairs = calculate_concordant_discordant_pairs(x_clean, y_clean)
        total_comparisons = (n * (n - 1)) / 2
        
        concordance_analysis = {
            'concordant_pairs': concordant_pairs,
            'discordant_pairs': discordant_pairs, 
            'tied_pairs': tied_pairs,
            'total_comparisons': total_comparisons,
            'concordant_percentage': float(concordant_pairs / total_comparisons * 100),
            'discordant_percentage': float(discordant_pairs / total_comparisons * 100),
            'tied_percentage': float(tied_pairs / total_comparisons * 100)
        }
        
        # Calculate statistical power for the correlation test
        power_analysis = calculate_correlation_power(n, cor, alpha)
        
        # Create figures for visualization
        figures = {}
        
        # Generate scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x_clean, y_clean, color=PASTEL_COLORS[0], alpha=0.6)
        ax.set_xlabel(x.name if hasattr(x, 'name') else 'X Variable')
        ax.set_ylabel(y.name if hasattr(y, 'name') else 'Y Variable')
        ax.set_title(f"Kendall's Tau: {cor:.3f} (p={p_corr:.5f})")
        figures['scatter_plot'] = fig_to_svg(fig)

        # Add concordance visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(['Concordant', 'Discordant', 'Tied'], 
                     [concordant_pairs, discordant_pairs, tied_pairs],
                     color=PASTEL_COLORS[:3])
        ax.set_title('Concordant vs Discordant Pairs Analysis')
        ax.set_ylabel('Number of Pairs')
        figures['concordance_plot'] = fig_to_svg(fig)

        # Calculate approximate confidence interval for Kendall's tau
        # Using bootstrap method (preferred)
        try:
            n_bootstrap = 2000  # Increased from 1000 for better stability
            bootstrap_taus = []
            
            for _ in range(n_bootstrap):
                indices = np.random.choice(len(x_clean), size=len(x_clean), replace=True)
                x_boot = x_clean.iloc[indices]
                y_boot = y_clean.iloc[indices]
                tau, _ = stats.kendalltau(x_boot, y_boot)
                bootstrap_taus.append(tau)
                
            # Calculate percentile-based CI
            lo_tau = np.percentile(bootstrap_taus, alpha/2 * 100)
            hi_tau = np.percentile(bootstrap_taus, (1-alpha/2) * 100)
            
            # Calculate bias-corrected CI (more accurate)
            z0 = stats.norm.ppf(sum(t < cor for t in bootstrap_taus) / n_bootstrap)
            z_alpha = stats.norm.ppf(alpha/2)
            z_1_alpha = stats.norm.ppf(1 - alpha/2)
            
            # Percentiles for bias-corrected CI
            p_low = stats.norm.cdf(2 * z0 + z_alpha)
            p_high = stats.norm.cdf(2 * z0 + z_1_alpha)
            
            lo_tau_bc = np.percentile(bootstrap_taus, p_low * 100)
            hi_tau_bc = np.percentile(bootstrap_taus, p_high * 100)
            
            ci = [float(lo_tau_bc), float(hi_tau_bc)]
            ci_method = "bias-corrected bootstrap"
            
            # Add bootstrap distribution if available
            if 'bootstrap_taus' in locals():
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(bootstrap_taus, color=PASTEL_COLORS[2], alpha=0.6, ax=ax)
                ax.axvline(cor, color=PASTEL_COLORS[4], linestyle='--', label='Observed Tau')
                ax.axvline(ci[0], color=PASTEL_COLORS[5], linestyle=':', label='CI Bounds')
                ax.axvline(ci[1], color=PASTEL_COLORS[5], linestyle=':')
                ax.set_title("Bootstrap Distribution of Kendall's Tau")
                ax.legend()
                figures['bootstrap_distribution'] = fig_to_svg(fig)
            
        except Exception:
            # Fallback to approximate CI using normal approximation
            # Improved formula for standard error accounting for ties
            tie_correction = 1.0
            
            if x_ties > 0 or y_ties > 0:
                # More accurate SE calculation with tie correction
                # Based on Kendall's tau-b formula
                x_ranks = pd.Series(x_clean).rank()
                y_ranks = pd.Series(y_clean).rank()
                
                tie_x = sum([(count * (count - 1) / 2) for count in x_ranks.value_counts()])
                tie_y = sum([(count * (count - 1) / 2) for count in y_ranks.value_counts()])
                
                n0 = n * (n - 1) / 2
                n1 = tie_x
                n2 = tie_y
                
                tie_correction = np.sqrt(((n0 - n1) * (n0 - n2)) / (n0 * (n0 - 1)))
            
            se = np.sqrt((2 * (2 * n + 5)) / (9 * n * (n - 1))) / tie_correction
            z_crit = stats.norm.ppf(1 - alpha/2)
            ci = [float(max(-1, cor - z_crit * se)), float(min(1, cor + z_crit * se))]
            ci_method = "normal approximation with tie correction"
        
        # Interpret the correlation strength with more detailed guidelines
        strength, practical_interpretation = interpret_kendall_tau(cor)
            
        # Create interpretation
        interpretation = f"Kendall's tau correlation: tau = {cor:.3f}, p = {p_corr:.5f}.\n"
        
        if p_corr < alpha:
            interpretation += f"There is a statistically significant {strength} "
            interpretation += f"{'positive' if cor > 0 else 'negative'} monotonic relationship between the variables.\n"
            interpretation += f"Practical significance: {practical_interpretation}"
        else:
            interpretation += "There is no statistically significant monotonic relationship between the variables."
        
        # Add concordance analysis to interpretation
        interpretation += f"\n\nKendall's tau measures the difference between the proportion of concordant pairs "
        interpretation += f"({concordance_analysis['concordant_percentage']:.1f}%) and discordant pairs "
        interpretation += f"({concordance_analysis['discordant_percentage']:.1f}%)."
            
        # Add confidence interval to interpretation
        interpretation += f"\n\nThe {(1-alpha)*100:.1f}% confidence interval for tau is [{ci[0]:.3f}, {ci[1]:.3f}] (calculated using {ci_method})."
        
        # Add power analysis to interpretation
        if power_analysis['power'] < 0.8 and p_corr >= alpha:
            interpretation += f"\n\nStatistical power is low ({power_analysis['power']:.2f}). The test may not have enough "
            interpretation += f"power to detect a true effect of this size. Consider increasing the sample size to at least {power_analysis['recommended_sample_size']}."
        elif power_analysis['power'] < 0.8 and p_corr < alpha:
            interpretation += f"\n\nNote: Despite achieving statistical significance, the statistical power is relatively low ({power_analysis['power']:.2f})."
        
        # Add comparison with other correlation methods
        interpretation += "\n\nMethodological insight: Kendall's tau is more robust to outliers than Pearson correlation "
        interpretation += "and has a more direct interpretation than Spearman's rho in terms of probability of concordance."
        
        # Add assumption violation warnings
        if assumption_violations:
            interpretation += f"\n\nWarning: The following assumptions may be violated: {', '.join(assumption_violations)}."
            if "sample_size" in assumption_violations:
                interpretation += "\nResults should be interpreted with caution due to small sample size."
            if "excessive_ties" in assumption_violations:
                interpretation += "\nResults account for ties but with many ties present, interpretation may be affected."
            if "outliers" in assumption_violations:
                interpretation += "\nWhile Kendall's tau is more robust to outliers than Pearson correlation, extreme outliers should still be examined."
        
        return {
            'test': 'Kendall Tau Correlation',
            'statistic': float(cor),
            'p_value': float(p_corr),
            'significant': p_corr < alpha,
            'interpretation': interpretation,
            'assumptions': assumptions,
            'assumption_violations': assumption_violations,
            'confidence_interval': ci,
            'effect_size': {
                'tau': float(cor),
                'tau_squared': float(cor**2),
                'interpretation': strength,
                'practical_significance': practical_interpretation
            },
            'concordance_analysis': concordance_analysis,
            'power_analysis': power_analysis,
            'sample_size': n,
            'figures': figures
        }
    except Exception as e:
        return {
            'test': 'Kendall Tau Correlation',
            'statistic': None,
            'p_value': None,
            'significant': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
