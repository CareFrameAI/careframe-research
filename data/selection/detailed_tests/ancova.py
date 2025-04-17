import traceback
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import statsmodels.formula.api as smf
import statsmodels.api as sm
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib
import base64
import io
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import OLSInfluence
# Update imports to use the format module
from data.assumptions.format import AssumptionTestKeys
import seaborn as sns
from data.selection.detailed_tests.formatting import fig_to_svg, PASTEL_COLORS, PASTEL_CMAP

def ancova(data: pd.DataFrame, outcome: str, group_var: str, covariates: List[str], alpha: float) -> Dict[str, Any]:
    """Performs Analysis of Covariance (ANCOVA) with comprehensive statistics."""
    try:
        # Set matplotlib to use a non-interactive backend
        matplotlib.use('Agg')
        figures = {}
        
        # Helper function to safely handle fig_to_svg results which might be tuples
        def safe_fig_to_svg(fig):
            result = fig_to_svg(fig)
            if isinstance(result, tuple):
                if len(result) > 0:
                    # Make sure we're returning a string, not another tuple
                    if isinstance(result[0], tuple):
                        return result[0][0] if result[0] and len(result[0]) > 0 else ""
                    return result[0]  # Return first item if it's a tuple
                return ""
            return result
        
        # Validate inputs
        if outcome not in data.columns:
            raise ValueError(f"Outcome variable '{outcome}' not found in data")
            
        if group_var not in data.columns:
            raise ValueError(f"Group variable '{group_var}' not found in data")
            
        for cov in covariates:
            if cov not in data.columns:
                raise ValueError(f"Covariate '{cov}' not found in data")
                
        # Convert group variable to categorical if not already
        if not pd.api.types.is_categorical_dtype(data[group_var]):
            data = data.copy()
            data[group_var] = pd.Categorical(data[group_var])
            
        # Construct formula
        covariates_str = ' + '.join(covariates)
        formula = f"{outcome} ~ C({group_var}) + {covariates_str}"
        
        # Fit the model
        model = smf.ols(formula, data)
        results = model.fit()
        
        # Extract model parameters
        r_squared = float(results.rsquared)
        adj_r_squared = float(results.rsquared_adj)
        f_statistic = float(results.fvalue)
        f_pvalue = float(results.f_pvalue)
        aic = float(results.aic)
        bic = float(results.bic)
        mse = float(results.mse_resid)
        rmse = float(np.sqrt(results.mse_resid))
        
        # Run ANOVA type III to get SS and p-values for each effect
        # Type III accounts for other effects in the model
        anova_results = sm.stats.anova_lm(results, typ=3)
        
        # Extract ANOVA results
        anova_table = {}
        
        for term in anova_results.index:
            if term == 'Residual':
                continue
                
            anova_table[term] = {
                'sum_sq': float(anova_results.loc[term, 'sum_sq']),
                'df': int(anova_results.loc[term, 'df']),
                'mean_sq': float(anova_results.loc[term, 'sum_sq'] / anova_results.loc[term, 'df']),
                'f_value': float(anova_results.loc[term, 'F']),
                'p_value': float(anova_results.loc[term, 'PR(>F)']),
                'significant': anova_results.loc[term, 'PR(>F)'] < alpha
            }
            
        # Calculate effect sizes (partial eta-squared)
        effect_sizes = {}
        
        for term in anova_table:
            # Partial eta-squared
            ss_term = anova_table[term]['sum_sq']
            ss_error = float(anova_results.loc['Residual', 'sum_sq'])
            partial_eta_sq = ss_term / (ss_term + ss_error)
            
            # Cohen's f (can be derived from partial eta-squared)
            cohen_f = np.sqrt(partial_eta_sq / (1 - partial_eta_sq))
            
            # Determine magnitude of effect size
            if partial_eta_sq < 0.01:
                magnitude = "Negligible"
            elif partial_eta_sq < 0.06:
                magnitude = "Small"
            elif partial_eta_sq < 0.14:
                magnitude = "Medium"
            else:
                magnitude = "Large"
                
            effect_sizes[term] = {
                'partial_eta_squared': float(partial_eta_sq),
                'cohen_f': float(cohen_f),
                'magnitude': magnitude
            }
            
        # Calculate adjusted means for each group
        adjusted_means = {}
        adjusted_std_errors = {}
        groups = data[group_var].unique()
        
        try:
            # Get mean values of covariates
            covariate_means = {cov: data[cov].mean() for cov in covariates}
            
            # For each group, predict outcome at mean covariate values
            for group in groups:
                # Create a test dataset with this group and mean covariate values
                test_data = pd.DataFrame({group_var: [group]})
                for cov, mean_val in covariate_means.items():
                    test_data[cov] = mean_val
                    
                # Predict
                adj_mean = float(results.predict(test_data)[0])
                adjusted_means[str(group)] = adj_mean
                
                # Calculate standard error for this prediction
                try:
                    # Get prediction with standard error
                    pred_with_se = results.get_prediction(test_data)
                    se = float(pred_with_se.se_mean[0])
                    adjusted_std_errors[str(group)] = se
                except:
                    adjusted_std_errors[str(group)] = None
                
            # Create a bar plot of adjusted means with error bars
            plt.figure(figsize=(10, 6))
            group_labels = list(adjusted_means.keys())
            means = [adjusted_means[g] for g in group_labels]
            
            # Add error bars if standard errors are available
            if all(se is not None for se in adjusted_std_errors.values()):
                errors = [adjusted_std_errors[g] * 1.96 for g in group_labels]  # 95% CI
                plt.bar(group_labels, means, yerr=errors, capsize=10, alpha=0.7)
                plt.title(f'Adjusted Means with 95% Confidence Intervals\n(Controlling for Covariates)')
            else:
                plt.bar(group_labels, means, alpha=0.7)
                plt.title(f'Adjusted Means (Controlling for Covariates)')
                
            plt.xlabel(group_var)
            plt.ylabel(f'Adjusted {outcome}')
            plt.grid(axis='y', alpha=0.3)
            
            # Save figure
            figures['adjusted_means'] = safe_fig_to_svg(plt.gcf())
            
        except Exception as e:
            # If there's any error, don't include adjusted means
            print(f"Error calculating adjusted means: {str(e)}")
            
        # Create a covariate-adjusted group comparison table
        try:
            # Perform pairwise comparisons with covariate adjustment
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            from statsmodels.sandbox.stats.multicomp import MultiComparison
            
            # Get residuals and add back group means (adjusting for covariates)
            resid = results.resid
            fitted = results.fittedvalues
            
            # Extract group effects by removing covariate effects
            # This is a simplification - ideally we'd use proper contrasts
            group_effects = {}
            
            # Create dummy variables for the group
            group_dummies = pd.get_dummies(data[group_var], prefix=group_var, drop_first=False)
            
            # Get coefficients for group dummies
            coef_df = results.summary2().tables[1]
            for col in coef_df.index:
                if f"C({group_var})" in col:
                    group_name = col.split('[')[1].split(']')[0]
                    group_effects[group_name] = float(coef_df.loc[col, 'Coef.'])
            
            # If we have group effects, create a dataset for post-hoc tests
            if group_effects:
                # Create a dataframe with adjusted values
                posthoc_data = pd.DataFrame({
                    'group': data[group_var],
                    'adjusted_outcome': fitted
                })
                
                # Perform Tukey HSD on adjusted values
                mc = MultiComparison(posthoc_data['adjusted_outcome'], posthoc_data['group'])
                tukey_result = mc.tukeyhsd(alpha=alpha)
                
                # Extract results
                pairwise_comparisons = []
                for i, (group1, group2, mean_diff, p_adj, lower, upper, reject) in enumerate(
                    zip(tukey_result.groups[0], tukey_result.groups[1], 
                        tukey_result.meandiffs, tukey_result.pvalues, 
                        tukey_result.confint[:, 0], tukey_result.confint[:, 1], 
                        tukey_result.reject)):
                    
                    pairwise_comparisons.append({
                        'group1': str(group1),
                        'group2': str(group2),
                        'mean_difference': float(mean_diff),
                        'std_error': float((upper - lower) / (2 * 1.96)),
                        'p_value': float(p_adj),
                        'ci_lower': float(lower),
                        'ci_upper': float(upper),
                        'significant': bool(reject)
                    })
                
                # Create a forest plot for pairwise comparisons
                plt.figure(figsize=(12, len(pairwise_comparisons) * 0.8 + 2))
                
                labels = [f"{comp['group1']} vs {comp['group2']}" for comp in pairwise_comparisons]
                mean_diffs = [comp['mean_difference'] for comp in pairwise_comparisons]
                ci_lowers = [comp['ci_lower'] for comp in pairwise_comparisons]
                ci_uppers = [comp['ci_upper'] for comp in pairwise_comparisons]
                
                # Calculate error bar widths
                err_min = [md - cil for md, cil in zip(mean_diffs, ci_lowers)]
                err_max = [ciu - md for md, ciu in zip(mean_diffs, ci_uppers)]
                
                # Plot
                y_pos = np.arange(len(labels))
                plt.errorbar(mean_diffs, y_pos, xerr=[err_min, err_max], fmt='o', capsize=5)
                plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                plt.yticks(y_pos, labels)
                plt.xlabel('Mean Difference')
                plt.title('Pairwise Comparisons with 95% Confidence Intervals\n(Tukey HSD, Adjusted for Covariates)')
                plt.grid(axis='x', alpha=0.3)
                
                # Save figure
                figures['pairwise_comparisons'] = safe_fig_to_svg(plt.gcf())
                
            else:
                pairwise_comparisons = {"error": "Could not extract group effects from model"}
        except Exception as e:
            pairwise_comparisons = {"error": f"Could not perform pairwise comparisons: {str(e)}"}
            
        # Extract residuals and fitted values for diagnostic plots
        residuals = results.resid
        fitted_values = results.fittedvalues
        
        # Create residual diagnostic plots
        try:
            # Residual vs Fitted plot
            plt.figure(figsize=(10, 6))
            plt.scatter(fitted_values, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.xlabel('Fitted Values')
            plt.ylabel('Residuals')
            plt.title('Residuals vs Fitted Values')
            plt.grid(True, alpha=0.3)
            
            # Add a lowess smoother
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                smooth = lowess(residuals, fitted_values)
                plt.plot(smooth[:, 0], smooth[:, 1], 'r-', linewidth=2)
            except:
                pass
                
            # Save figure
            figures['residuals_vs_fitted'] = safe_fig_to_svg(plt.gcf())
            
            # Q-Q plot
            plt.figure(figsize=(10, 6))
            qq = ProbPlot(residuals)
            qq.qqplot(line='45', alpha=0.5, ax=plt.gca())
            plt.title('Q-Q Plot of Residuals')
            plt.grid(True, alpha=0.3)
            
            # Save figure
            figures['qq_plot'] = safe_fig_to_svg(plt.gcf())
            
            # Scale-Location plot (sqrt of abs residuals vs fitted)
            plt.figure(figsize=(10, 6))
            plt.scatter(fitted_values, np.sqrt(np.abs(residuals)), alpha=0.5)
            plt.xlabel('Fitted Values')
            plt.ylabel('√|Residuals|')
            plt.title('Scale-Location Plot')
            plt.grid(True, alpha=0.3)
            
            # Add a lowess smoother
            try:
                smooth = lowess(np.sqrt(np.abs(residuals)), fitted_values)
                plt.plot(smooth[:, 0], smooth[:, 1], 'r-', linewidth=2)
            except:
                pass
                
            # Save figure
            figures['scale_location'] = safe_fig_to_svg(plt.gcf())
            
            # Cook's distance plot
            influence = OLSInfluence(results)
            cooks_d = influence.cooks_distance[0]
            
            plt.figure(figsize=(10, 6))
            plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt='ro', basefmt=' ')
            plt.axhline(y=4/len(data), color='r', linestyle='--', 
                       label=f'Threshold (4/n = {4/len(data):.5f})')
            plt.xlabel('Observation Index')
            plt.ylabel("Cook's Distance")
            plt.title("Cook's Distance for Influential Observations")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save figure
            figures['cooks_distance'] = safe_fig_to_svg(plt.gcf())
            
            # Residuals by group
            plt.figure(figsize=(10, 6))
            residuals_by_group = {str(group): residuals[data[group_var] == group] for group in groups}
            
            # Create boxplot
            boxplot_data = [residuals_by_group[str(group)] for group in groups]
            plt.boxplot(boxplot_data, labels=[str(group) for group in groups])
            plt.axhline(y=0, color='r', linestyle='-')
            plt.xlabel(group_var)
            plt.ylabel('Residuals')
            plt.title(f'Residuals by {group_var}')
            plt.grid(axis='y', alpha=0.3)
            
            # Save figure
            figures['residuals_by_group'] = safe_fig_to_svg(plt.gcf())
            
            # Partial regression plots for each covariate
            n_covariates = len(covariates)
            if n_covariates > 0:
                # Calculate number of rows and columns for subplots
                n_cols = min(2, n_covariates)
                n_rows = (n_covariates + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
                if n_covariates == 1:
                    axes = np.array([axes])
                
                axes = axes.flatten()
                
                for i, cov in enumerate(covariates):
                    # Skip categorical covariates
                    if pd.api.types.is_categorical_dtype(data[cov]) or pd.api.types.is_object_dtype(data[cov]):
                        axes[i].text(0.5, 0.5, f"{cov} is categorical", 
                                   horizontalalignment='center', verticalalignment='center')
                        axes[i].set_title(f"Partial Regression Plot: {cov}")
                        continue
                    
                    # For numeric covariates, create partial regression plot
                    sm.graphics.plot_partregress(outcome, cov, [(group_var, data[group_var])] + 
                                               [(c, data[c]) for c in covariates if c != cov],
                                               data=data, ax=axes[i], obs_labels=False)
                    axes[i].set_title(f"Partial Regression Plot: {cov}")
                    axes[i].grid(True, alpha=0.3)
                
                # Hide any unused subplots
                for j in range(i+1, len(axes)):
                    axes[j].set_visible(False)
                
                plt.tight_layout()
                
                # Save figure
                figures['partial_regression_plots'] = safe_fig_to_svg(plt.gcf())
            
            # Interaction plots for homogeneity of regression slopes
            try:
                # Only create for numeric covariates
                numeric_covariates = [cov for cov in covariates 
                                    if pd.api.types.is_numeric_dtype(data[cov])]
                
                if numeric_covariates:
                    n_covs = len(numeric_covariates)
                    n_cols = min(2, n_covs)
                    n_rows = (n_covs + n_cols - 1) // n_cols
                    
                    # If we have enough unique groups, use colors; otherwise use markers
                    n_groups = len(groups)
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
                    if n_covs == 1:
                        axes = np.array([axes])
                    
                    axes = axes.flatten()
                    
                    for i, cov in enumerate(numeric_covariates):
                        ax = axes[i]
                        
                        for j, group in enumerate(groups):
                            group_data = data[data[group_var] == group]
                            
                            # Plot the scatter points
                            ax.scatter(group_data[cov], group_data[outcome], 
                                     label=str(group), alpha=0.5)
                            
                            # Add regression line
                            try:
                                slope, intercept = np.polyfit(group_data[cov], group_data[outcome], 1)
                                x_range = np.linspace(group_data[cov].min(), group_data[cov].max(), 100)
                                ax.plot(x_range, intercept + slope * x_range, '-', 
                                      label=f"{group} (slope={slope:.3f})")
                            except:
                                pass
                                
                        ax.set_xlabel(cov)
                        ax.set_ylabel(outcome)
                        ax.set_title(f"Interaction Plot: {cov} × {group_var}")
                        ax.grid(True, alpha=0.3)
                        
                        # Only add legend for the first subplot to avoid clutter
                        if i == 0:
                            ax.legend()
                    
                    # Hide any unused subplots
                    for j in range(i+1, len(axes)):
                        axes[j].set_visible(False)
                    
                    plt.tight_layout()
                    
                    # Save figure
                    figures['interaction_plots'] = safe_fig_to_svg(plt.gcf())
            except Exception as e:
                print(f"Error creating interaction plots: {str(e)}")
                
            # Effect size visualization
            plt.figure(figsize=(10, 6))
            terms = list(effect_sizes.keys())
            eta_sq_values = [effect_sizes[term]['partial_eta_squared'] for term in terms]
            
            # Clean up term names for display
            display_terms = [term.replace('C(', '').replace(')', '') for term in terms]
            
            # Sort by effect size
            sorted_indices = np.argsort(eta_sq_values)
            sorted_terms = [display_terms[i] for i in sorted_indices]
            sorted_values = [eta_sq_values[i] for i in sorted_indices]
            
            # Define colors based on effect size magnitude
            colors = []
            for term in terms:
                magnitude = effect_sizes[term]['magnitude']
                if magnitude == 'Large':
                    colors.append('darkred')
                elif magnitude == 'Medium':
                    colors.append('orangered')
                elif magnitude == 'Small':
                    colors.append('orange')
                else:
                    colors.append('lightgrey')
            
            sorted_colors = [colors[i] for i in sorted_indices]
            
            # Create horizontal bar chart
            bars = plt.barh(sorted_terms, sorted_values, color=sorted_colors, alpha=0.7)
            
            # Add value annotations
            for i, bar in enumerate(bars):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f"{sorted_values[i]:.3f}", va='center')
            
            # Add vertical lines for effect size benchmarks
            plt.axvline(x=0.01, color='grey', linestyle='--', alpha=0.7, label='Small (0.01)')
            plt.axvline(x=0.06, color='orange', linestyle='--', alpha=0.7, label='Medium (0.06)')
            plt.axvline(x=0.14, color='red', linestyle='--', alpha=0.7, label='Large (0.14)')
            
            plt.xlabel('Partial η²')
            plt.title('Effect Sizes (Partial η²)')
            plt.legend(loc='lower right')
            plt.xlim(0, max(eta_sq_values) * 1.2)
            plt.grid(axis='x', alpha=0.3)
            
            # Save figure
            figures['effect_sizes'] = safe_fig_to_svg(plt.gcf())
            
        except Exception as e:
            print(f"Error creating diagnostic plots: {str(e)}")
            
        # Test assumptions using the AssumptionTestKeys from format.py
        assumptions = {}
        
        # 1. Normality of residuals
        try:
            # Get the test function
            normality_test_info = AssumptionTestKeys.NORMALITY.value
            normality_test_func = normality_test_info["function"]()
            
            # Run the test
            normality_result = normality_test_func.run_test(data=residuals)
            
            # Extract results
            assumptions['normality'] = {
                'test': normality_result.get('test_used', 'Shapiro-Wilk'),
                'statistic': float(normality_result.get('statistic', 0)),
                'p_value': float(normality_result.get('p_value', 1)),
                'satisfied': normality_result['result'].value == 'passed',
                'details': normality_result.get('details', ''),
                'skewness': float(normality_result.get('skewness', 0)),
                'kurtosis': float(normality_result.get('kurtosis', 0)),
                'warnings': normality_result.get('warnings', []),
                'figures': normality_result.get('figures', {})
            }
        except Exception as e:
            assumptions['normality'] = {
                'error': f'Could not test normality: {str(e)}'
            }
            
        # 2. Homogeneity of variances
        try:
            # Group residuals by group variable
            grouped_residuals = []
            group_names = []
            for group in groups:
                group_mask = data[group_var] == group
                group_resid = residuals[group_mask]
                grouped_residuals.append(group_resid)
                group_names.append(str(group))
                
            # Get the test function
            homogeneity_test_info = AssumptionTestKeys.HOMOGENEITY.value
            homogeneity_test_func = homogeneity_test_info["function"]()
            
            # Run the test
            homogeneity_result = homogeneity_test_func.run_test(data=grouped_residuals, groups=group_names)
            
            # Extract results
            assumptions['homogeneity_of_variance'] = {
                'test': homogeneity_result.get('test_used', 'Levene'),
                'statistic': float(homogeneity_result.get('statistic', 0)),
                'p_value': float(homogeneity_result.get('p_value', 1)),
                'satisfied': homogeneity_result['result'].value == 'passed',
                'details': homogeneity_result.get('details', ''),
                'group_variances': homogeneity_result.get('group_variances', {}),
                'warnings': homogeneity_result.get('warnings', []),
                'figures': homogeneity_result.get('figures', {})
            }
        except Exception as e:
            assumptions['homogeneity_of_variance'] = {
                'error': f'Could not test homogeneity of variance: {str(e)}'
            }
            
        # 3. Homogeneity of regression slopes
        try:
            # Get the test function
            slopes_test_info = AssumptionTestKeys.HOMOGENEITY_OF_REGRESSION_SLOPES.value
            slopes_test_func = slopes_test_info["function"]()
            
            # Test each covariate
            interaction_results = {}
            
            # Filter to only numeric covariates for the test
            numeric_covariates = [cov for cov in covariates 
                                 if pd.api.types.is_numeric_dtype(data[cov])]
            
            # For categorical covariates, use a different approach
            categorical_covariates = [cov for cov in covariates 
                                     if not pd.api.types.is_numeric_dtype(data[cov])]
            
            # First handle numeric covariates with our dedicated test
            if numeric_covariates:
                try:
                    # Run the test for all numeric covariates at once
                    slopes_result = slopes_test_func.run_test(
                        df=data,
                        outcome=outcome,
                        group=group_var,
                        covariates=numeric_covariates
                    )
                    
                    # Extract results from the interaction_df
                    if 'interaction_df' in slopes_result and not slopes_result['interaction_df'].empty:
                        for _, row in slopes_result['interaction_df'].iterrows():
                            covariate = row['covariate']
                            p_value = row['p_value']
                            
                            if p_value is not None:
                                interaction_results[f"{group_var}*{covariate}"] = {
                                    'p_value': float(p_value),
                                    'satisfied': p_value > alpha,
                                    'details': f"Interaction p-value: {p_value:.4f}"
                                }
                except Exception as e:
                    for cov in numeric_covariates:
                        interaction_results[f"{group_var}*{cov}"] = {
                            'error': f'Could not test interaction with {cov}: {str(e)}'
                        }
            
            # Now handle categorical covariates with the direct approach
            for cov in categorical_covariates:
                try:
                    # Create interaction formula
                    int_formula = f"{outcome} ~ C({group_var}) * C({cov})"
                    
                    # Fit model with interaction
                    int_model = smf.ols(int_formula, data).fit()
                    
                    # Extract interaction term p-value
                    # Find the interaction term
                    int_term = None
                    int_pval = 1.0  # Default to non-significant
                    
                    for term in int_model.pvalues.index:
                        if f"C({group_var})" in term and f"C({cov})" in term:
                            int_term = term
                            int_pval = float(int_model.pvalues[term])
                            break
                    
                    interaction_results[f"{group_var}*{cov}"] = {
                        'p_value': int_pval,
                        'satisfied': int_pval > alpha,
                        'details': f"Interaction p-value: {int_pval:.4f}"
                    }
                except Exception as e:
                    interaction_results[f"{group_var}*{cov}"] = {
                        'error': f'Could not test interaction with {cov}: {str(e)}'
                    }
                    
            assumptions['homogeneity_of_regression_slopes'] = interaction_results
        except Exception as e:
            assumptions['homogeneity_of_regression_slopes'] = {
                'error': f'Could not test homogeneity of regression slopes: {str(e)}'
            }
            
        # 4. Linearity
        try:
            # Get the test function
            linearity_test_info = AssumptionTestKeys.LINEARITY.value
            linearity_test_func = linearity_test_info["function"]()
            
            linearity_results = {}
            
            for cov in covariates:
                try:
                    # Skip categorical variables
                    if pd.api.types.is_object_dtype(data[cov]) or pd.api.types.is_categorical_dtype(data[cov]):
                        linearity_results[cov] = {
                            'r_squared': None,
                            'pearson_r': None,
                            'pearson_p': None,
                            'satisfied': True,  # Assume satisfied for categorical variables
                            'details': f"Linearity not applicable for categorical variable {cov}"
                        }
                        continue
                    
                    # Run the test for numeric variables
                    linearity_result = linearity_test_func.run_test(
                        x=data[cov],
                        y=data[outcome]
                    )
                    
                    # Extract results
                    linearity_results[cov] = {
                        'r_squared': float(linearity_result.get('r_squared', 0)),
                        'pearson_r': float(linearity_result.get('pearson_r', 0)),
                        'pearson_p': float(linearity_result.get('pearson_p', 1)),
                        'satisfied': linearity_result['result'].value == 'passed',
                        'details': linearity_result.get('details', ''),
                        'figures': linearity_result.get('figures', {})
                    }
                except Exception as e:
                    linearity_results[cov] = {
                        'error': f'Could not test linearity for {cov}: {str(e)}'
                    }
                    
            assumptions['linearity'] = linearity_results
        except Exception as e:
            assumptions['linearity'] = {
                'error': f'Could not test linearity: {str(e)}'
            }
            
        # 5. Multicollinearity
        try:
            # Get the test function
            multicollinearity_test_info = AssumptionTestKeys.MULTICOLLINEARITY.value
            multicollinearity_test_func = multicollinearity_test_info["function"]()
            
            # Run the test
            multicollinearity_result = multicollinearity_test_func.run_test(
                df=data,
                covariates=covariates
            )
            
            # Extract results
            assumptions['multicollinearity'] = {
                'vif_values': multicollinearity_result['vif_values'],
                'satisfied': multicollinearity_result['result'].value == 'passed',
                'details': multicollinearity_result['details'],
                'correlation_matrix': multicollinearity_result['correlation_matrix'],
                'warnings': multicollinearity_result['warnings'],
                'figures': multicollinearity_result['figures']
            }
        except Exception as e:
            assumptions['multicollinearity'] = {
                'error': f'Could not test multicollinearity: {str(e)}'
            }
        
        # 6. Check for influential observations (outliers)
        try:
            # Get the test function
            outlier_test_info = AssumptionTestKeys.OUTLIERS.value
            outlier_test_func = outlier_test_info["function"]()
            
            # Run the test
            outlier_result = outlier_test_func.run_test(data=residuals, method='zscore', threshold=3)
            
            # Extract results
            assumptions['influential_observations'] = {
                'n_influential': len(outlier_result['outliers']['indices']),
                'outlier_indices': outlier_result['outliers']['indices'],
                'outlier_values': outlier_result['outliers']['values'],
                'threshold': outlier_result.get('threshold', 3),
                'satisfied': outlier_result['result'].value != 'failed',
                'details': outlier_result.get('details', ''),
                'warnings': outlier_result.get('warnings', []),
                'figures': outlier_result.get('figures', {})
            }
        except Exception as e:
            # Fallback to Cook's distance approach if OutlierTest fails
            try:
                # Calculate Cook's distance
                influence = results.get_influence()
                cooks_d = influence.cooks_distance[0]
                
                # Identify influential observations (Cook's D > 4/n)
                threshold = 4 / len(data)
                influential_indices = np.where(cooks_d > threshold)[0]
                
                # Extract results
                assumptions['influential_observations'] = {
                    'n_influential': int(len(influential_indices)),
                    'threshold': float(threshold),
                    'max_cooks_d': float(np.max(cooks_d)),
                    'outlier_indices': influential_indices.tolist() if len(influential_indices) < 20 else influential_indices[:20].tolist(),
                    'satisfied': len(influential_indices) == 0,
                    'details': f"Found {len(influential_indices)} influential observations with Cook's distance > {threshold:.5f}"
                }
            except Exception as e2:
                assumptions['influential_observations'] = {
                    'error': f'Could not check for influential observations: {str(e)} / {str(e2)}'
                }
                
        # 7. Independence - Add this as it's important for ANCOVA
        try:
            # Get the test function
            independence_test_info = AssumptionTestKeys.INDEPENDENCE.value
            independence_test_func = independence_test_info["function"]()
            
            # Run the test on residuals
            independence_result = independence_test_func.run_test(data=residuals)
            
            # Extract results
            assumptions['independence'] = {
                'statistic': float(independence_result.get('statistic', 0)),
                'satisfied': independence_result['result'].value == 'passed',
                'details': independence_result.get('message', ''),
                'test_details': independence_result.get('details', {})
            }
        except Exception as e:
            assumptions['independence'] = {
                'error': f'Could not test independence: {str(e)}'
            }
            
        # 8. Sample Size Adequacy - Add this as it's important for statistical power
        try:
            # Get the test function
            sample_size_test_info = AssumptionTestKeys.SAMPLE_SIZE.value
            sample_size_test_func = sample_size_test_info["function"]()
            
            # Run the test
            # For ANCOVA, a good rule of thumb is 20 observations per group
            min_recommended = 20 * len(groups) 
            sample_size_result = sample_size_test_func.run_test(
                data=residuals,  # Just using this to get the sample size
                min_recommended=min_recommended
            )
            
            # Extract results
            assumptions['sample_size'] = {
                'sample_size': sample_size_result.get('sample_size', len(residuals)),
                'minimum_required': sample_size_result.get('minimum_required', min_recommended),
                'satisfied': sample_size_result['result'].value == 'passed',
                'details': sample_size_result.get('details', ''),
                'warnings': sample_size_result.get('warnings', [])
            }
        except Exception as e:
            assumptions['sample_size'] = {
                'error': f'Could not test sample size adequacy: {str(e)}'
            }
        
        
        # Create interpretation
        interpretation = f"Analysis of Covariance (ANCOVA) with {outcome} as outcome, {group_var} as grouping variable, "
        interpretation += f"and covariates ({', '.join(covariates)}).\n\n"
        
        # Overall model
        interpretation += f"Overall model: F({results.df_model:.0f}, {results.df_resid:.0f}) = {f_statistic:.3f}, p = {f_pvalue:.5f}, "
        interpretation += f"R² = {r_squared:.3f}, Adjusted R² = {adj_r_squared:.3f}.\n"
        interpretation += f"Root Mean Square Error (RMSE): {rmse:.3f}\n\n"
        
        # Main effects
        interpretation += "Main effects:\n"
        for term, stats in anova_table.items():
            term_name = term.replace("C(", "").replace(")", "") if "C(" in term else term
            
            interpretation += f"- {term_name}: F({stats['df']}, {anova_results.loc['Residual', 'df']:.0f}) = {stats['f_value']:.3f}, "
            interpretation += f"p = {stats['p_value']:.5f}, partial η² = {effect_sizes[term]['partial_eta_squared']:.3f} "
            interpretation += f"({effect_sizes[term]['magnitude'].lower()} effect).\n"
            
            if stats['significant']:
                interpretation += f"  {term_name} has a significant effect on {outcome} after controlling for covariates.\n"
            else:
                interpretation += f"  {term_name} does not have a significant effect on {outcome} after controlling for covariates.\n"
                
        # Adjusted means if available
        if adjusted_means:
            interpretation += "\nAdjusted means (controlling for covariates):\n"
            for group, mean in adjusted_means.items():
                if group in adjusted_std_errors and adjusted_std_errors[group] is not None:
                    ci_lower = mean - 1.96 * adjusted_std_errors[group]
                    ci_upper = mean + 1.96 * adjusted_std_errors[group]
                    interpretation += f"- {group_var}={group}: {mean:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})\n"
                else:
                    interpretation += f"- {group_var}={group}: {mean:.3f}\n"
                
        # Pairwise comparisons if available
        if isinstance(pairwise_comparisons, list) and pairwise_comparisons:
            interpretation += "\nPairwise comparisons (Tukey HSD, adjusted for covariates):\n"
            
            for comp in pairwise_comparisons:
                interpretation += f"- {comp['group1']} vs {comp['group2']}: "
                interpretation += f"Mean difference = {comp['mean_difference']:.3f} "
                interpretation += f"(95% CI: {comp['ci_lower']:.3f} to {comp['ci_upper']:.3f}), "
                interpretation += f"p = {comp['p_value']:.5f}"
                
                if comp['significant']:
                    interpretation += ", significant\n"
                else:
                    interpretation += ", not significant\n"
                
        # Assumptions
        interpretation += "\nAssumption tests:\n"
        
        # Normality
        if 'normality' in assumptions and 'p_value' in assumptions['normality']:
            norm_result = assumptions['normality']
            interpretation += f"- Normality of residuals: {norm_result['test']} test, p = {norm_result['p_value']:.5f}, "
            
            if norm_result['satisfied']:
                interpretation += "assumption satisfied.\n"
            else:
                interpretation += f"assumption potentially violated. {norm_result.get('details', '')}\n"
                interpretation += "  Note: ANCOVA is generally robust to moderate violations of normality with sufficient sample size.\n"
                
        # Homogeneity of variance
        if 'homogeneity_of_variance' in assumptions and 'p_value' in assumptions['homogeneity_of_variance']:
            hov_result = assumptions['homogeneity_of_variance']
            interpretation += f"- Homogeneity of variance: {hov_result['test']} test, p = {hov_result['p_value']:.5f}, "
            
            if hov_result['satisfied']:
                interpretation += "assumption satisfied.\n"
            else:
                interpretation += f"assumption potentially violated. {hov_result.get('details', '')}\n"
                interpretation += "  Consider using robust standard errors or non-parametric alternatives if this violation is severe.\n"
                
        # Homogeneity of regression slopes
        if 'homogeneity_of_regression_slopes' in assumptions and isinstance(assumptions['homogeneity_of_regression_slopes'], dict):
            # Check if any interactions have errors
            has_errors = any('error' in result for result in assumptions['homogeneity_of_regression_slopes'].values())
            error_covs = [cov.split('*')[1] for cov, result in assumptions['homogeneity_of_regression_slopes'].items() 
                         if 'error' in result]
            
            # Get covariates with valid results
            valid_covs = [cov for cov, result in assumptions['homogeneity_of_regression_slopes'].items() 
                         if 'satisfied' in result]
            
            if valid_covs:
                homogeneity_violated = any(not assumptions['homogeneity_of_regression_slopes'][cov]['satisfied'] 
                                          for cov in valid_covs)
                
                if homogeneity_violated:
                    violated_covs = [cov.split('*')[1] for cov in valid_covs 
                                    if not assumptions['homogeneity_of_regression_slopes'][cov]['satisfied']]
                    interpretation += f"- Homogeneity of regression slopes: Assumption violated for covariates: {', '.join(violated_covs)}. "
                    interpretation += "ANCOVA results should be interpreted with caution, as significant interactions suggest that "
                    interpretation += "the effect of the group variable depends on the value of the covariate(s).\n"
                    interpretation += "  Consider analyzing these interactions directly or using a different analytical approach.\n"
                else:
                    interpretation += "- Homogeneity of regression slopes: Assumption satisfied for tested covariates.\n"
                
                if error_covs:
                    interpretation += f"  Note: Could not test homogeneity for: {', '.join(error_covs)}.\n"
            else:
                interpretation += "- Homogeneity of regression slopes: Could not test for any covariates.\n"
        
        # Linearity
        if 'linearity' in assumptions and isinstance(assumptions['linearity'], dict):
            # Check if any linearity tests have errors
            has_errors = any('error' in result for result in assumptions['linearity'].values() 
                            if isinstance(result, dict))
            
            # Filter out categorical variables and those with errors
            numeric_covs = [cov for cov, result in assumptions['linearity'].items() 
                           if isinstance(result, dict) and 'r_squared' in result and result['r_squared'] is not None]
            
            if numeric_covs:
                linearity_violated = any(not assumptions['linearity'][cov]['satisfied'] 
                                        for cov in numeric_covs)
                
                if linearity_violated:
                    nonlinear_covs = [cov for cov in numeric_covs 
                                     if not assumptions['linearity'][cov]['satisfied']]
                    interpretation += f"- Linearity: Assumption violated for covariates: {', '.join(nonlinear_covs)}. "
                    interpretation += "Consider transformations or including polynomial terms for these covariates.\n"
                else:
                    interpretation += "- Linearity: Assumption satisfied for all numeric covariates.\n"
            else:
                interpretation += "- Linearity: No numeric covariates to test for linearity.\n"
        
        # Multicollinearity
        if 'multicollinearity' in assumptions and 'satisfied' in assumptions['multicollinearity']:
            if assumptions['multicollinearity']['satisfied']:
                interpretation += "- Multicollinearity: No significant multicollinearity detected among covariates.\n"
            else:
                interpretation += f"- Multicollinearity: {assumptions['multicollinearity']['details']}\n"
                interpretation += "  Consider removing or combining highly correlated covariates.\n"
        
        # Influential observations
        if 'influential_observations' in assumptions and 'satisfied' in assumptions['influential_observations']:
            influential_result = assumptions['influential_observations']
            interpretation += "- Influential observations: "
            
            if influential_result['satisfied']:
                interpretation += "No influential observations detected.\n"
            else:
                interpretation += f"Found {influential_result['n_influential']} potentially influential observations "
                if 'threshold' in influential_result:
                    interpretation += f"(threshold > {influential_result['threshold']:.5f}). "
                else:
                    interpretation += ". "
                interpretation += "Consider examining these cases and potentially refitting the model without extreme outliers if they represent errors.\n"
        
        # Independence
        if 'independence' in assumptions and 'satisfied' in assumptions['independence']:
            independence_result = assumptions['independence']
            interpretation += "- Independence of observations: "
            
            if independence_result['satisfied']:
                interpretation += "Assumption satisfied.\n"
            else:
                interpretation += "Assumption potentially violated. "
                interpretation += f"{independence_result.get('details', '')}\n"
                interpretation += "  Consider using methods that account for dependence in the data, such as mixed-effects models.\n"
        
        # Sample Size
        if 'sample_size' in assumptions and 'satisfied' in assumptions['sample_size']:
            sample_size_result = assumptions['sample_size']
            interpretation += "- Sample size adequacy: "
            
            if sample_size_result['satisfied']:
                interpretation += f"Sample size ({sample_size_result['sample_size']}) is adequate.\n"
            else:
                interpretation += f"Sample size ({sample_size_result['sample_size']}) may be insufficient "
                interpretation += f"(recommended minimum: {sample_size_result['minimum_required']}). "
                interpretation += "Results should be interpreted with caution, as small sample sizes can reduce statistical power.\n"
        
        # Additional guidance
        interpretation += "\nAdditional guidance:\n"
        
        # Based on model fit
        if r_squared < 0.1:
            interpretation += "- The model explains a very small proportion of the variance in the outcome. "
            interpretation += "Consider including additional relevant covariates or examining if the linear model is appropriate for these data.\n"
        elif r_squared > 0.9:
            interpretation += "- The model explains a very high proportion of the variance. "
            interpretation += "Ensure this isn't due to overfitting or collinearity issues.\n"
            
        # Based on violations
        has_violations = False
        violation_text = ""
        
        if 'normality' in assumptions and 'satisfied' in assumptions['normality'] and not assumptions['normality']['satisfied']:
            has_violations = True
            violation_text += "normality of residuals, "
            
        if 'homogeneity_of_variance' in assumptions and 'satisfied' in assumptions['homogeneity_of_variance'] and not assumptions['homogeneity_of_variance']['satisfied']:
            has_violations = True
            violation_text += "homogeneity of variance, "
            
        if 'homogeneity_of_regression_slopes' in assumptions and isinstance(assumptions['homogeneity_of_regression_slopes'], dict):
            valid_covs = [cov for cov, result in assumptions['homogeneity_of_regression_slopes'].items() 
                         if 'satisfied' in result]
            if valid_covs and any(not assumptions['homogeneity_of_regression_slopes'][cov]['satisfied'] for cov in valid_covs):
                has_violations = True
                violation_text += "homogeneity of regression slopes, "
        
        # Add checks for new assumptions
        if 'independence' in assumptions and 'satisfied' in assumptions['independence'] and not assumptions['independence']['satisfied']:
            has_violations = True
            violation_text += "independence, "
            
        if 'sample_size' in assumptions and 'satisfied' in assumptions['sample_size'] and not assumptions['sample_size']['satisfied']:
            has_violations = True
            violation_text += "sample size adequacy, "
        
        if has_violations:
            violation_text = violation_text.rstrip(", ")
            interpretation += f"- The model violates the assumption(s) of {violation_text}. "
            interpretation += "Consider these alternative approaches:\n"
            interpretation += "  1. Data transformations for the outcome or covariates\n"
            interpretation += "  2. Robust regression methods\n"
            interpretation += "  3. Non-parametric alternatives such as rank ANCOVA\n"
            interpretation += "  4. If homogeneity of regression slopes is violated, consider a moderation analysis instead\n"
            interpretation += "  5. If independence is violated, consider mixed-effects models\n"
        
        # Confidence in results
        confidence_level = "high"
        confidence_issues = []
        
        if has_violations:
            confidence_level = "moderate"
            confidence_issues.append("assumption violations")
            
        if 'influential_observations' in assumptions and 'satisfied' in assumptions['influential_observations'] and not assumptions['influential_observations']['satisfied']:
            confidence_level = "moderate" if confidence_level == "high" else "low"
            confidence_issues.append("influential observations")
            
        if 'multicollinearity' in assumptions and 'satisfied' in assumptions['multicollinearity'] and not assumptions['multicollinearity']['satisfied']:
            confidence_level = "moderate" if confidence_level == "high" else "low"
            confidence_issues.append("multicollinearity")
            
        if 'sample_size' in assumptions and 'satisfied' in assumptions['sample_size'] and not assumptions['sample_size']['satisfied']:
            confidence_level = "moderate" if confidence_level == "high" else "low"
            confidence_issues.append("insufficient sample size")
            
        issues_text = " and ".join(confidence_issues)
        if issues_text:
            interpretation += f"- Overall confidence in results: {confidence_level} (due to {issues_text}).\n"
        else:
            interpretation += f"- Overall confidence in results: {confidence_level}.\n"
        
        # Diagnostic plots information
        interpretation += "\nDiagnostic plots:\n"
        interpretation += "- Residuals vs. Fitted: Check for patterns or non-linearity\n"
        interpretation += "- QQ Plot: Check if residuals follow a normal distribution\n"
        interpretation += "- Scale-Location: Check for homoscedasticity (constant variance)\n"
        interpretation += "- Cook's Distance: Identify influential observations\n"
        interpretation += "- Residuals by Group: Check if residuals are similarly distributed across groups\n"
        interpretation += "- Partial Regression Plots: Examine relationship between each covariate and outcome\n"
        interpretation += "- Interaction Plots: Visualize whether the effect of covariates differs across groups\n"
        
        return {
            'test': 'ANCOVA',
            'overall': {
                'r_squared': r_squared,
                'adj_r_squared': adj_r_squared,
                'f_statistic': f_statistic,
                'f_pvalue': f_pvalue,
                'df_model': int(results.df_model),
                'df_residual': int(results.df_resid),
                'mse': mse,
                'rmse': rmse,
                'aic': aic,
                'bic': bic
            },
            'anova_table': anova_table,
            'effect_sizes': effect_sizes,
            'adjusted_means': adjusted_means,
            'adjusted_std_errors': adjusted_std_errors,
            'pairwise_comparisons': pairwise_comparisons,
            'assumptions': assumptions,
            'formula': formula,
            'summary': str(results.summary()),
            'interpretation': interpretation,
            'figures': figures
        }
    except Exception as e:
        return {
            'test': 'ANCOVA',
            'error': str(e),
            'traceback': traceback.format_exc()
        }