import traceback
import pandas as pd
from scipy import stats
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
import io
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from data.selection.detailed_tests.formatting import fig_to_svg, PASTEL_COLORS, PASTEL_CMAP
from data.assumptions.format import AssumptionTestKeys

def chi_square_test(contingency_table: pd.DataFrame, alpha: float = 0.05) -> Dict[str, Any]:
    """Performs the Chi-Square test of independence with comprehensive statistics and assumption checks."""
    try:
        # Ensure the input is numeric by converting if necessary
        if contingency_table.dtypes.any() == object:
            # If we have string data, we need to ensure it's properly formatted as a contingency table
            if not isinstance(contingency_table.values[0][0], (int, float)):
                return {
                    'test': 'Chi-Square Test of Independence',
                    'chi2': None,
                    'p_value': None,
                    'dof': None,
                    'expected': None,
                    'min_expected': None,
                    'significant': False,
                    'reason': 'Input must be a contingency table with numeric values. Please create a cross-tabulation first.'
                }

        # Extract the values from the contingency table
        table_values = contingency_table.values
        
        # Calculate row and column totals
        row_totals = np.sum(table_values, axis=1)
        col_totals = np.sum(table_values, axis=0)
        total_n = np.sum(table_values)
        
        # Perform Chi-Square test
        chi2, p_value, dof, expected = stats.chi2_contingency(table_values)
        
        # Determine significance
        significant = p_value < alpha
        
        # Calculate effect size (Cramer's V)
        # Cramer's V = sqrt(chi2 / (n * min(r-1, c-1)))
        # where r is the number of rows and c is the number of columns
        n_rows, n_cols = table_values.shape
        min_dim = min(n_rows - 1, n_cols - 1)
        if min_dim > 0 and total_n > 0:
            cramers_v = np.sqrt(chi2 / (total_n * min_dim))
        else:
            cramers_v = 0
            
        # Determine magnitude of effect size
        if cramers_v < 0.1:
            effect_magnitude = "Negligible"
        elif cramers_v < 0.3:
            effect_magnitude = "Small"
        elif cramers_v < 0.5:
            effect_magnitude = "Medium"
        else:
            effect_magnitude = "Large"
            
        # Prepare data for assumption tests
        min_expected = float(expected.min())
        n_cells = expected.size
        
        # Run assumption tests using imported test functions
        assumptions = {}
        
        # 1. Independence test
        independence_test_obj = AssumptionTestKeys.INDEPENDENCE.value["function"]()
        independence_test = independence_test_obj.run_test(data=table_values)
        assumptions['independence'] = independence_test
        
        # 2. Sample size test
        sample_size_test_obj = AssumptionTestKeys.SAMPLE_SIZE.value["function"]()
        sample_size_test = sample_size_test_obj.run_test(
            data=np.array([total_n]),  # Passing as array since test expects numeric data
            min_recommended=5 * n_cells
        )
        assumptions['sample_size'] = sample_size_test
        
        # 3. Expected frequencies check - using goodness of fit
        goodness_of_fit_test_obj = AssumptionTestKeys.GOODNESS_OF_FIT.value["function"]()
        goodness_of_fit_test = goodness_of_fit_test_obj.run_test(
            observed=table_values.flatten(),
            expected=expected.flatten()
        )
        assumptions['expected_frequencies'] = goodness_of_fit_test
        
        # Also add an outlier check
        outlier_test_obj = AssumptionTestKeys.OUTLIERS.value["function"]()
        outlier_test = outlier_test_obj.run_test(data=table_values.flatten())
        assumptions['outliers'] = outlier_test
        
        # Calculate standardized residuals
        # (observed - expected) / sqrt(expected)
        observed = table_values
        std_residuals = (observed - expected) / np.sqrt(expected)
        
        # Find cells with large residuals (> 1.96 for alpha = 0.05)
        critical_value = stats.norm.ppf(1 - alpha/2)  # Two-tailed
        significant_cells = []
        
        for i in range(n_rows):
            for j in range(n_cols):
                if abs(std_residuals[i, j]) > critical_value:
                    significant_cells.append({
                        'row': i,
                        'col': j,
                        'observed': float(observed[i, j]),
                        'expected': float(expected[i, j]),
                        'std_residual': float(std_residuals[i, j]),
                        'direction': 'higher' if std_residuals[i, j] > 0 else 'lower'
                    })
        
        # Create interpretation
        interpretation = f"Chi-Square Test of Independence for a {n_rows}x{n_cols} contingency table.\n\n"
        
        # Basic results
        interpretation += f"χ²({dof}) = {chi2:.3f}, p = {p_value:.5f}\n"
        interpretation += f"Effect size (Cramer's V) = {cramers_v:.3f} ({effect_magnitude.lower()} effect)\n\n"
        
        # Table information
        interpretation += "Contingency table summary:\n"
        interpretation += f"- Rows: {n_rows}, Columns: {n_cols}\n"
        interpretation += f"- Total observations: {total_n}\n"
        
        # Expected frequencies
        interpretation += f"- Minimum expected frequency: {min_expected:.2f}\n"
        
        # Assumptions summary in interpretation
        interpretation += "\nAssumption checks:\n"
        
        # Check expected frequencies
        if 'expected_frequencies' in assumptions:
            freq_result = assumptions['expected_frequencies']
            interpretation += f"- Expected frequencies: {freq_result.get('details', '')}\n"
        
        # Check sample size
        if 'sample_size' in assumptions:
            size_result = assumptions['sample_size']
            interpretation += f"- Sample size: {size_result.get('details', '')}\n"
        
        # Check independence
        if 'independence' in assumptions:
            independence_result = assumptions['independence']
            interpretation += f"- Independence: {independence_result.get('message', '')}\n"
        
        # Conclusion
        interpretation += f"\nConclusion: There is {'a statistically significant' if significant else 'no statistically significant'} "
        interpretation += f"association between the row and column variables (p = {p_value:.5f}). "
        
        if significant:
            interpretation += f"The {effect_magnitude.lower()} effect size (Cramer's V = {cramers_v:.3f}) suggests that "
            
            if effect_magnitude == "Negligible":
                interpretation += "the association, while statistically significant, may not be practically meaningful."
            elif effect_magnitude == "Small":
                interpretation += "there is a weak association between the variables."
            elif effect_magnitude == "Medium":
                interpretation += "there is a moderate association between the variables."
            else:  # Large
                interpretation += "there is a strong association between the variables."
        
        # Standardized residuals analysis if significant
        if significant and significant_cells:
            interpretation += "\n\nPost-hoc analysis of standardized residuals:\n"
            interpretation += "The following cells contribute significantly to the chi-square value:\n"
            
            for cell in significant_cells:
                row_name = contingency_table.index[cell['row']] if hasattr(contingency_table, 'index') else f"Row {cell['row']+1}"
                col_name = contingency_table.columns[cell['col']] if hasattr(contingency_table, 'columns') else f"Column {cell['col']+1}"
                
                interpretation += f"- {row_name}, {col_name}: Observed = {cell['observed']:.0f}, Expected = {cell['expected']:.1f}, "
                interpretation += f"Std. Residual = {cell['std_residual']:.2f} "
                interpretation += f"(significantly {cell['direction']} than expected)\n"
        
        # Calculate additional statistics
        row_proportions = observed / row_totals[:, np.newaxis]
        col_proportions = observed / col_totals
        
        # Calculate contribution of each cell to total chi-square
        cell_contributions = ((observed - expected) ** 2) / expected
        cell_contributions_percent = cell_contributions / chi2 * 100
        
        # Calculate adjusted residuals
        row_factors = 1 - (row_totals / total_n)[:, np.newaxis]
        col_factors = 1 - (col_totals / total_n)
        denom = np.sqrt(expected * row_factors * col_factors)
        adjusted_residuals = (observed - expected) / denom
        
        # Generate figures
        figures = {}
        
        # Figure 1: Observed values heatmap
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        im = ax1.imshow(observed, cmap="Blues")
        
        # Set labels
        if hasattr(contingency_table, 'index') and hasattr(contingency_table, 'columns'):
            ax1.set_xticks(np.arange(len(contingency_table.columns)))
            ax1.set_yticks(np.arange(len(contingency_table.index)))
            ax1.set_xticklabels(contingency_table.columns)
            ax1.set_yticklabels(contingency_table.index)
        else:
            ax1.set_xticks(np.arange(n_cols))
            ax1.set_yticks(np.arange(n_rows))
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations in each cell
        for i in range(n_rows):
            for j in range(n_cols):
                text = ax1.text(j, i, f"{observed[i, j]:.0f}", 
                                ha="center", va="center", color="black" if observed[i, j] < np.max(observed)/2 else "white")
        
        ax1.set_title("Observed Frequencies")
        fig1.colorbar(im, ax=ax1)
        fig1.tight_layout()
        figures['observed_heatmap'] = fig_to_svg(fig1)
        
        # Figure 2: Standardized residuals heatmap
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        # Create a diverging colormap (red for negative, blue for positive)
        cmap = LinearSegmentedColormap.from_list('custom_diverging', 
                                                [PASTEL_COLORS[0], '#fdfdfd', PASTEL_COLORS[2]], 
                                                N=256)
        
        # Find the maximum absolute value for symmetric color scaling
        vmax = np.max(np.abs(std_residuals))
        im = ax2.imshow(std_residuals, cmap=cmap, vmin=-vmax, vmax=vmax)
        
        # Set labels
        if hasattr(contingency_table, 'index') and hasattr(contingency_table, 'columns'):
            ax2.set_xticks(np.arange(len(contingency_table.columns)))
            ax2.set_yticks(np.arange(len(contingency_table.index)))
            ax2.set_xticklabels(contingency_table.columns)
            ax2.set_yticklabels(contingency_table.index)
        else:
            ax2.set_xticks(np.arange(n_cols))
            ax2.set_yticks(np.arange(n_rows))
        
        # Rotate the tick labels
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations and highlight significant cells
        for i in range(n_rows):
            for j in range(n_cols):
                # Add cell borders for significant cells
                if abs(std_residuals[i, j]) > critical_value:
                    ax2.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, 
                                               edgecolor='black', lw=2))
                
                text = ax2.text(j, i, f"{std_residuals[i, j]:.2f}", 
                               ha="center", va="center", 
                               color="black" if abs(std_residuals[i, j]) < vmax/2 else "white",
                               fontweight='bold' if abs(std_residuals[i, j]) > critical_value else 'normal')
        
        ax2.set_title(f"Standardized Residuals (|value| > {critical_value:.2f} is significant)")
        fig2.colorbar(im, ax=ax2)
        fig2.tight_layout()
        figures['std_residuals_heatmap'] = fig_to_svg(fig2)
        
        # Figure 3: Mosaic plot (visualization of contingency table)
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        
        # Create mosaic plot using matplotlib
        # First, calculate positions
        widths = col_totals / total_n
        lefts = np.zeros(n_cols)
        for i in range(1, n_cols):
            lefts[i] = lefts[i-1] + widths[i-1]
            
        # Calculate heights for each segment within a column
        heights = []
        for j in range(n_cols):
            # Get column probabilities
            col_probs = observed[:, j] / col_totals[j]
            heights.append(col_probs)
            
        # Plot rectangles for each cell
        for j in range(n_cols):
            bottoms = np.zeros(n_rows)
            for i in range(1, n_rows):
                bottoms[i] = bottoms[i-1] + heights[j][i-1]
                
            for i in range(n_rows):
                # Determine color based on standardized residual
                if std_residuals[i, j] > critical_value:
                    color = PASTEL_COLORS[2]  # blue for positive significant
                elif std_residuals[i, j] < -critical_value:
                    color = PASTEL_COLORS[0]  # red for negative significant
                else:
                    color = '#f7f7f7'  # neutral for non-significant
                
                # Draw rectangle
                rect = plt.Rectangle((lefts[j], bottoms[i]), widths[j], heights[j][i], 
                                    facecolor=color, edgecolor='black', alpha=0.7)
                ax3.add_patch(rect)
                
                # Add text label if cell is big enough
                if widths[j] * heights[j][i] > 0.03:  # Only add text if rectangle is large enough
                    ax3.text(lefts[j] + widths[j]/2, bottoms[i] + heights[j][i]/2, 
                            f"{observed[i, j]:.0f}\n({expected[i, j]:.1f})", 
                            ha='center', va='center', fontsize=9)
                
        # Set axis limits and remove axes
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        
        # Add column labels at the top
        for j in range(n_cols):
            col_name = contingency_table.columns[j] if hasattr(contingency_table, 'columns') else f"Column {j+1}"
            ax3.text(lefts[j] + widths[j]/2, 1.01, str(col_name), ha='center', va='bottom', rotation=45)
            
        # Add row labels on the right
        for i in range(n_rows):
            # Calculate approximate vertical position for the row label
            # This is a simplification and may need adjustment
            avg_pos = 0
            total_weight = 0
            for j in range(n_cols):
                bottoms = np.zeros(n_rows)
                for k in range(1, n_rows):
                    bottoms[k] = bottoms[k-1] + heights[j][k-1]
                avg_pos += (bottoms[i] + heights[j][i]/2) * col_totals[j]
                total_weight += col_totals[j]
            avg_pos /= total_weight
            
            row_name = contingency_table.index[i] if hasattr(contingency_table, 'index') else f"Row {i+1}"
            ax3.text(1.01, avg_pos, str(row_name), ha='left', va='center')
            
        ax3.set_title('Mosaic Plot (Observed counts with Expected counts in parentheses)')
        
        # Add legend for color meaning
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=PASTEL_COLORS[2], edgecolor='black', alpha=0.7, 
                 label=f'Positive significant residual (> {critical_value:.2f})'),
            Patch(facecolor='#f7f7f7', edgecolor='black', alpha=0.7, 
                 label='Non-significant residual'),
            Patch(facecolor=PASTEL_COLORS[0], edgecolor='black', alpha=0.7, 
                 label=f'Negative significant residual (< -{critical_value:.2f})')
        ]
        ax3.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=1)
        
        fig3.tight_layout()
        figures['mosaic_plot'] = fig_to_svg(fig3)
        
        # Figure 4: Bar chart comparison of observed vs expected frequencies
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        
        # Reshape data for plotting
        cells = []
        obs_vals = []
        exp_vals = []
        
        for i in range(n_rows):
            for j in range(n_cols):
                row_name = contingency_table.index[i] if hasattr(contingency_table, 'index') else f"Row {i+1}"
                col_name = contingency_table.columns[j] if hasattr(contingency_table, 'columns') else f"Column {j+1}"
                cells.append(f"{row_name}, {col_name}")
                obs_vals.append(observed[i, j])
                exp_vals.append(expected[i, j])
        
        x = np.arange(len(cells))
        width = 0.35
        
        ax4.bar(x - width/2, obs_vals, width, label='Observed', color=PASTEL_COLORS[0])
        ax4.bar(x + width/2, exp_vals, width, label='Expected', color=PASTEL_COLORS[1], alpha=0.7)
        
        ax4.set_xticks(x)
        ax4.set_xticklabels(cells, rotation=90)
        ax4.legend()
        
        ax4.set_title('Observed vs Expected Frequencies')
        ax4.set_ylabel('Frequency')
        ax4.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add significant markers
        for i, (obs, exp) in enumerate(zip(obs_vals, exp_vals)):
            cell_idx = i
            row_idx = cell_idx // n_cols
            col_idx = cell_idx % n_cols
            if abs(std_residuals[row_idx, col_idx]) > critical_value:
                marker = '*' if std_residuals[row_idx, col_idx] > 0 else '†'
                ax4.text(i, max(obs, exp) + 1, marker, ha='center', fontsize=15)
        
        # Add legend for markers
        if any(abs(r) > critical_value for r in std_residuals.flatten()):
            marker_text = "* Significantly higher than expected\n† Significantly lower than expected"
            ax4.text(1.02, 0.02, marker_text, transform=ax4.transAxes, 
                    ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.7))
        
        fig4.tight_layout()
        figures['observed_vs_expected_bar'] = fig_to_svg(fig4)
        
        return {
            'test': 'Chi-Square Test of Independence',
            'chi2': float(chi2),
            'p_value': float(p_value),
            'dof': int(dof),
            'expected': expected.tolist(),
            'min_expected': float(min_expected),
            'significant': significant,
            'cramers_v': float(cramers_v),
            'effect_magnitude': effect_magnitude,
            'table_summary': {
                'n_rows': int(n_rows),
                'n_cols': int(n_cols),
                'row_totals': row_totals.tolist(),
                'col_totals': col_totals.tolist(),
                'total_n': int(total_n)
            },
            'additional_statistics': {
                'standardized_residuals': std_residuals.tolist(),
                'adjusted_residuals': adjusted_residuals.tolist(),
                'cell_contributions': cell_contributions.tolist(),
                'cell_contributions_percent': cell_contributions_percent.tolist(),
                'row_proportions': row_proportions.tolist(),
                'col_proportions': (observed / col_totals).tolist(),
                'significant_cells': significant_cells,
                'critical_value': float(critical_value)
            },
            'assumptions': assumptions,
            'interpretation': interpretation,
            'figures': figures
        }
    except Exception as e:
        return {
            'test': 'Chi-Square Test of Independence',
            'chi2': None,
            'p_value': None,
            'dof': None,
            'expected': None,
            'min_expected': None,
            'significant': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }