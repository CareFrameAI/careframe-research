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
    SampleSizeTest
)
from data.selection.detailed_tests.formatting import fig_to_svg, PASTEL_COLORS, PASTEL_CMAP

def fishers_exact_test(contingency_table: pd.DataFrame, alpha: float) -> Dict[str, Any]:
    """Performs Fisher's exact test for 2x2 contingency tables with comprehensive statistics and assumption checks."""
    try:
        # Check if the contingency table is 2x2
        if contingency_table.shape != (2, 2):
            return {
                'test': "Fisher's Exact Test",
                'odds_ratio': None,
                'p_value': None,
                'significant': False,
                'reason': 'Contingency table must be 2x2.'
            }

        # Extract the values from the contingency table
        table_values = contingency_table.values
        
        # Extract row and column names
        row_names = list(contingency_table.index.astype(str))
        col_names = list(contingency_table.columns.astype(str))
        
        # Calculate row and column totals
        row_totals = np.sum(table_values, axis=1)
        col_totals = np.sum(table_values, axis=0)
        total_n = np.sum(table_values)
        
        # Perform Fisher's exact test
        odds_ratio, p_value = stats.fisher_exact(table_values)
        
        # Calculate effect size (Cramer's V)
        # For a 2x2 table, Cramer's V is equivalent to the phi coefficient
        chi2 = stats.chi2_contingency(table_values, correction=False)[0]
        cramers_v = np.sqrt(chi2 / (total_n * min(2-1, 2-1)))
        
        # Determine magnitude of effect size
        if cramers_v < 0.1:
            effect_magnitude = "Negligible"
        elif cramers_v < 0.3:
            effect_magnitude = "Small"
        elif cramers_v < 0.5:
            effect_magnitude = "Medium"
        else:
            effect_magnitude = "Large"
            
        # Calculate relative risk if possible (for 2x2 tables)
        # Assuming the table is organized as:
        #   [exposed_cases, exposed_non_cases]
        #   [unexposed_cases, unexposed_non_cases]
        try:
            exposed_risk = table_values[0, 0] / row_totals[0]
            unexposed_risk = table_values[1, 0] / row_totals[1]
            relative_risk = exposed_risk / unexposed_risk
            
            # Calculate 95% CI for relative risk
            se_log_rr = np.sqrt((1 - exposed_risk) / (exposed_risk * row_totals[0]) + 
                               (1 - unexposed_risk) / (unexposed_risk * row_totals[1]))
            log_rr = np.log(relative_risk)
            ci_lower_rr = np.exp(log_rr - 1.96 * se_log_rr)
            ci_upper_rr = np.exp(log_rr + 1.96 * se_log_rr)
            
            relative_risk_data = {
                'relative_risk': float(relative_risk),
                'ci_lower': float(ci_lower_rr),
                'ci_upper': float(ci_upper_rr)
            }
        except:
            relative_risk_data = {
                'error': 'Could not calculate relative risk'
            }
            
        # Calculate 95% CI for odds ratio
        # Using the formula for the standard error of the log odds ratio
        try:
            se_log_or = np.sqrt(1/table_values[0, 0] + 1/table_values[0, 1] + 
                              1/table_values[1, 0] + 1/table_values[1, 1])
            log_or = np.log(odds_ratio)
            ci_lower_or = np.exp(log_or - 1.96 * se_log_or)
            ci_upper_or = np.exp(log_or + 1.96 * se_log_or)
            
            odds_ratio_data = {
                'odds_ratio': float(odds_ratio),
                'ci_lower': float(ci_lower_or),
                'ci_upper': float(ci_upper_or)
            }
        except:
            odds_ratio_data = {
                'odds_ratio': float(odds_ratio),
                'error': 'Could not calculate confidence intervals'
            }
            
        # Test assumptions using the tests imported in format.py
        assumptions = {}
        
        # 1. Sample size check using SampleSizeTest
        # Create a proper sample array that represents the actual total_n observations
        # instead of just the 4 cells of the contingency table
        sample_data = np.ones(int(total_n))  # Create an array with the correct sample size
        sample_size_test = SampleSizeTest()
        assumptions['sample_size'] = sample_size_test.run_test(data=sample_data, min_recommended=20)

        # 2. Independence test - check if observations are independent
        independence_test = IndependenceTest()
        # Create data that represents the structure of the contingency table
        independence_data = np.concatenate([np.ones(int(table_values[0,0])), 
                                          np.zeros(int(table_values[0,1])), 
                                          np.ones(int(table_values[1,0])), 
                                          np.zeros(int(table_values[1,1]))])
        assumptions['independence'] = independence_test.run_test(data=independence_data)
        
        # 3. Goodness of fit test - to check how well observed data fits expected values
        # Expected values under the null hypothesis of independence
        expected = np.outer(row_totals, col_totals) / total_n
        goodness_of_fit_test = GoodnessOfFitTest()
        assumptions['goodness_of_fit'] = goodness_of_fit_test.run_test(
            observed=table_values.flatten(), 
            expected=expected.flatten()
        )
        
        # 4. Check for small expected frequencies - a specific concern for contingency tables
        # While Fisher's Exact Test is designed for small samples, very small expected
        # frequencies can still impact reliability
        min_expected = np.min(expected)
        if min_expected < 1:
            assumptions['expected_frequencies'] = {
                'result': AssumptionResult.WARNING,
                'details': f"Minimum expected cell frequency is {min_expected:.2f}, which is < 1. "
                           f"Results should be interpreted with caution."
            }
        elif min_expected < 5:
            assumptions['expected_frequencies'] = {
                'result': AssumptionResult.WARNING,
                'details': f"Minimum expected cell frequency is {min_expected:.2f}, which is < 5. "
                           f"Fisher's Exact Test is appropriate for this situation."
            }
        else:
            assumptions['expected_frequencies'] = {
                'result': AssumptionResult.PASSED,
                'details': f"All expected cell frequencies are ≥ 5 ({min_expected:.2f})."
            }
        
        # Create interpretation
        interpretation = f"Fisher's Exact Test for 2x2 contingency table.\n\n"
        
        # Basic results
        interpretation += f"p = {p_value:.5f}, Odds Ratio = {odds_ratio:.3f}\n"
        if 'ci_lower' in odds_ratio_data:
            interpretation += f"95% CI for Odds Ratio: [{odds_ratio_data['ci_lower']:.3f}, {odds_ratio_data['ci_upper']:.3f}]\n"
        
        if 'relative_risk' in relative_risk_data:
            interpretation += f"Relative Risk = {relative_risk_data['relative_risk']:.3f}\n"
            interpretation += f"95% CI for Relative Risk: [{relative_risk_data['ci_lower']:.3f}, {relative_risk_data['ci_upper']:.3f}]\n"
            
        interpretation += f"Effect size (Cramer's V) = {cramers_v:.3f} ({effect_magnitude.lower()} effect)\n\n"
        
        # Table information
        interpretation += "Contingency table:\n"
        interpretation += f"  {row_names[0]}: {row_totals[0]} observations\n"
        interpretation += f"  {row_names[1]}: {row_totals[1]} observations\n"
        interpretation += f"  {col_names[0]}: {col_totals[0]} observations\n"
        interpretation += f"  {col_names[1]}: {col_totals[1]} observations\n"
        interpretation += f"  Total: {total_n} observations\n\n"
        
        # Assumptions - reference the results directly from the assumption tests
        interpretation += "Assumption checks:\n"
        
        # Add assumption information without trying to directly access specific keys
        # that might not be present in the returned dictionaries
        for assumption_name, assumption_result in assumptions.items():
            interpretation += f"- {assumption_name.replace('_', ' ').title()}: "
            
            if 'details' in assumption_result:
                interpretation += f"{assumption_result['details']}\n"
            elif 'message' in assumption_result:
                interpretation += f"{assumption_result['message']}\n"
            else:
                interpretation += f"Check performed. Result: {assumption_result.get('result', 'Unknown')}\n"
        
        # Conclusion
        interpretation += f"\nConclusion: There is {'a statistically significant' if p_value < alpha else 'no statistically significant'} "
        interpretation += f"association between the row and column variables (p = {p_value:.5f}). "
        
        if p_value < alpha:
            if 'odds_ratio' in odds_ratio_data:
                if odds_ratio > 1:
                    interpretation += f"The odds ratio of {odds_ratio:.3f} indicates that the odds of the outcome are {odds_ratio:.3f} times higher "
                    interpretation += f"in the first group compared to the second group.\n"
                else:
                    interpretation += f"The odds ratio of {odds_ratio:.3f} indicates that the odds of the outcome are {1/odds_ratio:.3f} times lower "
                    interpretation += f"in the first group compared to the second group.\n"
                    
            if 'relative_risk' in relative_risk_data:
                rr = relative_risk_data['relative_risk']
                if rr > 1:
                    interpretation += f"The relative risk of {rr:.3f} indicates that the risk of the outcome is {rr:.3f} times higher "
                    interpretation += f"in the first group compared to the second group.\n"
                else:
                    interpretation += f"The relative risk of {rr:.3f} indicates that the risk of the outcome is {1/rr:.3f} times lower "
                    interpretation += f"in the first group compared to the second group.\n"
        
        # Calculate additional statistics
        additional_stats = {}
        
        # 1. Calculate Number Needed to Treat (NNT) or Number Needed to Harm (NNH)
        try:
            # Absolute risk difference
            risk_difference = exposed_risk - unexposed_risk
            
            if risk_difference == 0:
                nnt = float('inf')
                nnt_type = "NNT"
            else:
                nnt = 1 / abs(risk_difference)
                nnt_type = "NNT" if risk_difference > 0 else "NNH"
            
            # Calculate 95% CI for risk difference
            se_rd = np.sqrt(exposed_risk * (1 - exposed_risk) / row_totals[0] + 
                          unexposed_risk * (1 - unexposed_risk) / row_totals[1])
            rd_ci_lower = risk_difference - 1.96 * se_rd
            rd_ci_upper = risk_difference + 1.96 * se_rd
            
            additional_stats['risk_measures'] = {
                'absolute_risk_difference': float(risk_difference),
                'rd_ci_lower': float(rd_ci_lower),
                'rd_ci_upper': float(rd_ci_upper),
                'number_needed_to_treat': float(nnt),
                'nnt_type': nnt_type
            }
        except Exception as e:
            additional_stats['risk_measures'] = {
                'error': str(e)
            }
        
        # 2. Calculate exact confidence intervals for odds ratio
        try:
            from scipy.stats import hypergeom
            
            a = table_values[0, 0]
            b = table_values[0, 1]
            c = table_values[1, 0]
            d = table_values[1, 1]
            
            # This is a simple approximation for the exact confidence interval
            # For a true exact method, more complex algorithms are needed
            
            # Let's estimate the CI using profile likelihood
            def profile_likelihood(or_test):
                # Calculate probability of observed table given odds ratio
                prob = 0
                for i in range(min(a+c, a+b) + 1):
                    j = a + b - i
                    k = a + c - i
                    l = total_n - i - j - k
                    
                    if min(j, k, l) < 0:
                        continue
                    
                    # Check if the odds ratio matches
                    if i * l == 0 or j * k == 0:
                        test_or = np.inf if i * l > 0 else 0
                    else:
                        test_or = (i * l) / (j * k)
                    
                    if abs(test_or - or_test) < 0.01:
                        # Calculate probability using hypergeometric distribution
                        prob += hypergeom.pmf(i, total_n, a+b, a+c)
                
                return prob
            
            # Finding exact bounds is complex, let's use the approximation from above
            # but label it correctly
            additional_stats['exact_confidence_intervals'] = {
                'note': 'Using normal approximation as true exact confidence intervals require complex computation',
                'ci_lower': float(ci_lower_or),
                'ci_upper': float(ci_upper_or)
            }
        except Exception as e:
            additional_stats['exact_confidence_intervals'] = {
                'error': str(e)
            }
        
        # 3. Calculate diagnostic test statistics if applicable
        try:
            # Assuming this is a diagnostic test contingency table:
            # [true_positive, false_positive]
            # [false_negative, true_negative]
            
            # Sensitivity = TP / (TP + FN)
            sensitivity = table_values[0, 0] / (table_values[0, 0] + table_values[1, 0])
            
            # Specificity = TN / (TN + FP)
            specificity = table_values[1, 1] / (table_values[0, 1] + table_values[1, 1])
            
            # Positive predictive value = TP / (TP + FP)
            ppv = table_values[0, 0] / (table_values[0, 0] + table_values[0, 1])
            
            # Negative predictive value = TN / (TN + FN)
            npv = table_values[1, 1] / (table_values[1, 0] + table_values[1, 1])
            
            # Positive likelihood ratio = Sensitivity / (1 - Specificity)
            pos_lr = sensitivity / (1 - specificity) if specificity < 1 else float('inf')
            
            # Negative likelihood ratio = (1 - Sensitivity) / Specificity
            neg_lr = (1 - sensitivity) / specificity if sensitivity < 1 else 0
            
            # Youden's J statistic = Sensitivity + Specificity - 1
            youdens_j = sensitivity + specificity - 1
            
            additional_stats['diagnostic_measures'] = {
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'positive_predictive_value': float(ppv),
                'negative_predictive_value': float(npv),
                'positive_likelihood_ratio': float(pos_lr),
                'negative_likelihood_ratio': float(neg_lr),
                'youdens_j': float(youdens_j)
            }
        except Exception as e:
            additional_stats['diagnostic_measures'] = {
                'error': str(e)
            }
        
        # 4. Chi-square with continuity correction for comparison
        try:
            chi2_stat, chi2_p, _, _ = stats.chi2_contingency(table_values, correction=True)
            additional_stats['chi_square_comparison'] = {
                'statistic': float(chi2_stat),
                'p_value': float(chi2_p),
                'significant': chi2_p < alpha,
                'agrees_with_fisher': (chi2_p < alpha) == (p_value < alpha)
            }
        except Exception as e:
            additional_stats['chi_square_comparison'] = {
                'error': str(e)
            }
            
        # 5. Barnard's exact test (has more power but less commonly used)
        try:
            # This is an approximation since Barnard's test is not built into scipy
            # We'll use a simple Monte Carlo approximation

            # Function to compute the difference in proportions
            def prop_diff(table):
                p1 = table[0, 0] / (table[0, 0] + table[0, 1])
                p2 = table[1, 0] / (table[1, 0] + table[1, 1])
                return abs(p1 - p2)
            
            # Observed difference
            observed_diff = prop_diff(table_values)
            
            # Perform Monte Carlo simulation
            n_simulations = 10000
            count_more_extreme = 0
            
            for _ in range(n_simulations):
                # Generate random table with same marginals
                r1 = np.random.hypergeometric(col_totals[0], col_totals[1], row_totals[0])
                sim_table = np.array([[r1, row_totals[0] - r1], 
                                    [col_totals[0] - r1, row_totals[1] - (col_totals[0] - r1)]])
                
                # Check if difference is more extreme
                if prop_diff(sim_table) >= observed_diff:
                    count_more_extreme += 1
            
            barnard_p = count_more_extreme / n_simulations
            
            additional_stats['barnard_test_approximation'] = {
                'p_value': float(barnard_p),
                'n_simulations': n_simulations,
                'significant': barnard_p < alpha,
                'agrees_with_fisher': (barnard_p < alpha) == (p_value < alpha)
            }
        except Exception as e:
            additional_stats['barnard_test_approximation'] = {
                'error': str(e)
            }
            
        # Generate figures
        figures = {}
        
        # Figure 1: Heatmap of the contingency table
        try:
            fig1, ax1 = plt.subplots(figsize=(10, 8))
            
            # Create mask for text color (black on light cells, white on dark cells)
            text_color_threshold = (table_values.max() + table_values.min()) / 2
            
            # Create heatmap
            sns.heatmap(contingency_table, annot=True, fmt="d", cmap="Blues", ax=ax1,
                      annot_kws={"size": 16, "fontweight": "bold"})
            
            # Add row and column totals
            for i, total in enumerate(row_totals):
                ax1.text(contingency_table.shape[1] + 0.5, i + 0.5, f"{total}", 
                        ha="center", va="center", fontsize=14, fontweight="bold",
                        bbox=dict(facecolor='lightgray', alpha=0.6))
            
            for j, total in enumerate(col_totals):
                ax1.text(j + 0.5, contingency_table.shape[0] + 0.5, f"{total}", 
                        ha="center", va="center", fontsize=14, fontweight="bold",
                        bbox=dict(facecolor='lightgray', alpha=0.6))
            
            # Add total n
            ax1.text(contingency_table.shape[1] + 0.5, contingency_table.shape[0] + 0.5, f"{total_n}", 
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    bbox=dict(facecolor='lightgray', alpha=0.6))
            
            # Add a title
            ax1.set_title("Contingency Table", fontsize=16)
            
            # Add Fisher's exact test results
            result_text = f"Fisher's exact test: p = {p_value:.4f}"
            if p_value < alpha:
                result_text += " *"
            result_text += f"\nOdds Ratio = {odds_ratio:.3f}"
            
            ax1.text(0.5, -0.15, result_text, transform=ax1.transAxes, 
                    ha="center", va="center", fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.8))
            
            fig1.tight_layout()
            figures['contingency_table'] = fig_to_svg(fig1)
        except Exception as e:
            figures['contingency_table_error'] = str(e)
        
        # Figure 2: Mosaic plot for visualizing contingency table
        try:
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            
            # Set the plotting area
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.set_axis_off()
            
            # Calculate positions for the mosaic
            # First, split horizontally by row groups
            row_widths = row_totals / total_n
            row_lefts = [0, row_widths[0]]
            
            # For each row, split vertically by column groups
            for i in range(2):
                col_heights = table_values[i, :] / row_totals[i]
                col_bottoms = [0, col_heights[0]]
                
                # Draw the rectangles for this row
                for j in range(2):
                    # Determine color based on whether this is a significant cell
                    color = PASTEL_COLORS[0]  # Default light color
                    if p_value < alpha:
                        # Use standardized residuals to determine color intensity
                        expected = row_totals[i] * col_totals[j] / total_n
                        residual = (table_values[i, j] - expected) / np.sqrt(expected)
                        if residual > 1.96:  # Significantly more than expected
                            color = PASTEL_COLORS[1]
                        elif residual < -1.96:  # Significantly less than expected
                            color = PASTEL_COLORS[4]
                    
                    # Draw rectangle
                    rect = plt.Rectangle((row_lefts[i], col_bottoms[j]), 
                                       row_widths[i], col_heights[j], 
                                       facecolor=color, edgecolor='black')
                    ax2.add_patch(rect)
                    
                    # Add text
                    ax2.text(row_lefts[i] + row_widths[i]/2, col_bottoms[j] + col_heights[j]/2, 
                            f"{table_values[i, j]}", ha='center', va='center', fontsize=12)
            
            # Add row labels on the left
            ax2.text(-0.05, 0.25, row_names[1], ha='right', va='center', fontsize=12)
            ax2.text(-0.05, 0.75, row_names[0], ha='right', va='center', fontsize=12)
            
            # Add column labels on top
            ax2.text(0.25, 1.05, col_names[0], ha='center', va='bottom', fontsize=12)
            ax2.text(0.75, 1.05, col_names[1], ha='center', va='bottom', fontsize=12)
            
            # Add title
            ax2.set_title("Mosaic Plot", fontsize=16, pad=20)
            
            # Add result text
            result_text = f"Fisher's exact test: p = {p_value:.4f}"
            if p_value < alpha:
                result_text += " *"
            ax2.text(0.5, -0.1, result_text, ha='center', va='center', fontsize=12,
                    transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8))
            
            fig2.tight_layout()
            figures['mosaic_plot'] = fig_to_svg(fig2)
        except Exception as e:
            figures['mosaic_plot_error'] = str(e)
        
        # Figure 3: Forest plot for odds ratio and relative risk
        try:
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            
            # Set up y-positions and labels
            measures = []
            values = []
            ci_lowers = []
            ci_uppers = []
            
            # Odds ratio
            if 'ci_lower' in odds_ratio_data:
                measures.append("Odds Ratio")
                values.append(odds_ratio)
                ci_lowers.append(odds_ratio_data['ci_lower'])
                ci_uppers.append(odds_ratio_data['ci_upper'])
            
            # Relative risk
            if 'relative_risk' in relative_risk_data:
                measures.append("Relative Risk")
                values.append(relative_risk_data['relative_risk'])
                ci_lowers.append(relative_risk_data['ci_lower'])
                ci_uppers.append(relative_risk_data['ci_upper'])
            
            # Risk difference
            if 'risk_measures' in additional_stats and 'error' not in additional_stats['risk_measures']:
                measures.append("Risk Difference")
                values.append(additional_stats['risk_measures']['absolute_risk_difference'])
                ci_lowers.append(additional_stats['risk_measures']['rd_ci_lower'])
                ci_uppers.append(additional_stats['risk_measures']['rd_ci_upper'])
            
            # If we have measures to plot
            if measures:
                # Calculate error bar sizes
                errors_minus = [val - lower for val, lower in zip(values, ci_lowers)]
                errors_plus = [upper - val for val, upper in zip(values, ci_uppers)]
                
                # Create y-positions
                y_pos = np.arange(len(measures))
                
                # Plot forest plot
                ax3.errorbar(values, y_pos, xerr=[errors_minus, errors_plus], fmt='o', 
                           capsize=5, color='black', markersize=8)
                
                # Add reference line at 1.0 for ratios, 0.0 for differences
                if "Risk Difference" in measures:
                    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                else:
                    ax3.axvline(x=1, color='red', linestyle='--', alpha=0.7)
                
                # Add measure labels
                ax3.set_yticks(y_pos)
                ax3.set_yticklabels(measures)
                
                # Add value annotations
                for i, (val, lower, upper) in enumerate(zip(values, ci_lowers, ci_uppers)):
                    ax3.text(upper + 0.1, i, f"{val:.3f} [{lower:.3f}, {upper:.3f}]", 
                            va='center', fontsize=10)
                
                # Add labels and title
                ax3.set_xlabel('Value (with 95% CI)')
                ax3.set_title('Forest Plot of Effect Measures')
                
                # Set appropriate x-axis limits
                all_values = values + ci_lowers + ci_uppers
                min_val = min(0, min(all_values)) if "Risk Difference" in measures else min(0.1, min(all_values))
                max_val = max(all_values) * 1.2
                ax3.set_xlim(min_val, max_val)
                
                # Add a grid
                ax3.grid(axis='x', linestyle='--', alpha=0.7)
                
                # Add interpretation
                if p_value < alpha:
                    result_text = "Statistically significant (p < 0.05)"
                else:
                    result_text = "Not statistically significant (p ≥ 0.05)"
                ax3.text(0.02, -0.15, result_text, transform=ax3.transAxes, 
                        fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            
            fig3.tight_layout()
            figures['forest_plot'] = fig_to_svg(fig3)
        except Exception as e:
            figures['forest_plot_error'] = str(e)
        
        # Figure 4: Bar chart comparing observed vs expected
        try:
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            
            # Flatten the contingency table for plotting
            cell_labels = []
            for i in range(2):
                for j in range(2):
                    cell_labels.append(f"{row_names[i]}, {col_names[j]}")
            
            observed = table_values.flatten()
            expected = expected.flatten()
            
            # Set positions for bars
            x = np.arange(len(cell_labels))
            width = 0.35
            
            # Plot bars
            ax4.bar(x - width/2, observed, width, label='Observed', color=PASTEL_COLORS[0], alpha=0.7)
            ax4.bar(x + width/2, expected, width, label='Expected', color=PASTEL_COLORS[1], alpha=0.7)
            
            # Add labels and title
            ax4.set_xlabel('Contingency Table Cell')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Observed vs Expected Frequencies')
            ax4.set_xticks(x)
            ax4.set_xticklabels(cell_labels, rotation=45, ha='right')
            ax4.legend()
            
            # Add Fisher's exact test result
            result_text = f"Fisher's exact test: p = {p_value:.4f}"
            if p_value < alpha:
                result_text += " *"
            ax4.text(0.5, 0.02, result_text, transform=ax4.transAxes, 
                    ha='center', bbox=dict(facecolor='white', alpha=0.8))
            
            fig4.tight_layout()
            figures['observed_vs_expected'] = fig_to_svg(fig4)
        except Exception as e:
            figures['observed_vs_expected_error'] = str(e)
        
        # Figure 5: Diagnostic measures visualization (if applicable)
        if 'diagnostic_measures' in additional_stats and 'error' not in additional_stats['diagnostic_measures']:
            try:
                fig5, ax5 = plt.subplots(figsize=(10, 6))
                
                # Extract measures
                measures = ['Sensitivity', 'Specificity', 'PPV', 'NPV']
                values = [
                    additional_stats['diagnostic_measures']['sensitivity'],
                    additional_stats['diagnostic_measures']['specificity'],
                    additional_stats['diagnostic_measures']['positive_predictive_value'],
                    additional_stats['diagnostic_measures']['negative_predictive_value']
                ]
                
                # Create bar chart
                bars = ax5.bar(measures, values, 
                             color=[PASTEL_COLORS[i] for i in range(4)], 
                             alpha=0.7)
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{height:.3f}', ha='center', va='bottom')
                
                # Add labels and title
                ax5.set_ylabel('Value')
                ax5.set_title('Diagnostic Test Measures')
                ax5.set_ylim(0, 1.1)
                
                # Add reference line at 0.5
                ax5.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
                
                # Add LR+ and LR- as text annotation
                lr_text = f"LR+ = {additional_stats['diagnostic_measures']['positive_likelihood_ratio']:.2f}\n"
                lr_text += f"LR- = {additional_stats['diagnostic_measures']['negative_likelihood_ratio']:.2f}"
                ax5.text(0.02, 0.02, lr_text, transform=ax5.transAxes, 
                        bbox=dict(facecolor='white', alpha=0.8))
                
                fig5.tight_layout()
                figures['diagnostic_measures'] = fig_to_svg(fig5)
            except Exception as e:
                figures['diagnostic_measures_error'] = str(e)
        
        # Figure 6: p-value comparison for different tests
        if 'chi_square_comparison' in additional_stats and 'error' not in additional_stats['chi_square_comparison']:
            try:
                fig6, ax6 = plt.subplots(figsize=(8, 6))
                
                # Prepare test names and p-values
                tests = ["Fisher's Exact", "Chi-Square w/ Yates"]
                p_values = [p_value, additional_stats['chi_square_comparison']['p_value']]
                
                # Add Barnard's test if available
                if 'barnard_test_approximation' in additional_stats and 'error' not in additional_stats['barnard_test_approximation']:
                    tests.append("Barnard's (Approx)")
                    p_values.append(additional_stats['barnard_test_approximation']['p_value'])
                
                # Create bar chart
                bars = ax6.bar(tests, p_values, color=PASTEL_COLORS[2])
                
                # Add horizontal line at alpha level
                ax6.axhline(y=alpha, color='red', linestyle='--', label=f'Alpha = {alpha}')
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.4f}', ha='center', va='bottom')
                
                # Add labels and title
                ax6.set_ylabel('p-value')
                ax6.set_title('p-value Comparison Across Different Tests')
                ax6.legend()
                
                # Add significance markers
                for i, p in enumerate(p_values):
                    if p < alpha:
                        ax6.text(i, p/2, '*', fontsize=24, ha='center', va='center')
                
                fig6.tight_layout()
                figures['p_value_comparison'] = fig_to_svg(fig6)
            except Exception as e:
                figures['p_value_comparison_error'] = str(e)
        
        return {
            'test': "Fisher's Exact Test",
            'odds_ratio': odds_ratio_data,
            'relative_risk': relative_risk_data if 'relative_risk' in relative_risk_data else None,
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'cramers_v': float(cramers_v),
            'effect_magnitude': effect_magnitude,
            'table_summary': {
                'row_totals': row_totals.tolist(),
                'col_totals': col_totals.tolist(),
                'total_n': int(total_n)
            },
            'assumptions': assumptions,
            'interpretation': interpretation,
            'additional_statistics': additional_stats,
            'figures': figures
        }
    except Exception as e:
        return {
            'test': "Fisher's Exact Test",
            'odds_ratio': None,
            'p_value': None,
            'significant': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }