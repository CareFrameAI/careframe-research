import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Dict, List, Any, Tuple, Optional
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
import scipy.stats as stats
from itertools import combinations
import warnings
import traceback
from statsmodels.graphics.factorplots import interaction_plot
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import pingouin as pg

from data.assumptions.tests import AssumptionResult
from data.assumptions.tests import ResidualNormalityTest, SphericityTest, HomogeneityOfVarianceTest, OutlierTest, SampleSizeTest, IndependenceTest, HomoscedasticityTest, NormalityTest
from data.selection.detailed_tests.formatting import fig_to_svg, PASTEL_COLORS, PASTEL_CMAP

def repeated_measures_anova(data: pd.DataFrame, subject_id: str, within_factors: List[str], alpha: float, outcome: str) -> Dict[str, Any]:
    """
    Performs Repeated Measures ANOVA with comprehensive statistics, assumption checks, and visualizations.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataset in long format with observations for each subject
    subject_id : str
        Column name identifying the subject
    within_factors : List[str]
        List of column names for within-subject factors
    alpha : float
        Significance level for statistical tests
    outcome : str
        Name of the dependent variable
    
    Returns:
    --------
    Dict[str, Any]
        Comprehensive results including model statistics, effect sizes, post-hoc tests,
        assumption checks, and visualizations
    """
    try:
        # Check if data is in proper long format
        if len(data[subject_id].unique()) * np.prod([len(data[factor].unique()) for factor in within_factors]) != len(data):
            # Check if the data is balanced
            counts = data.groupby([subject_id] + within_factors).size().reset_index(name='count')
            if counts['count'].nunique() > 1:
                warnings.warn("Data appears to be unbalanced. Some subjects may have missing combinations of factor levels.")
        
        # Create figures dictionary to store visualizations
        figures = {}
        
        # Calculate descriptive statistics for all combinations of factor levels
        descriptives = data.groupby(within_factors)[outcome].agg(['mean', 'std', 'count', 'sem']).reset_index()
        descriptives['ci95_lower'] = descriptives['mean'] - 1.96 * descriptives['sem']
        descriptives['ci95_upper'] = descriptives['mean'] + 1.96 * descriptives['sem']
        
        # Calculate overall mean and SD
        overall_mean = data[outcome].mean()
        overall_sd = data[outcome].std()
        
        # Run the repeated measures ANOVA using pingouin
        try:
            rm_anova_results = pg.rm_anova(
                data=data,
                dv=outcome,
                within=within_factors,
                subject=subject_id,
                correction=True  # Apply sphericity correction if needed
            )
            
            # Store the main ANOVA results
            anova_results = rm_anova_results.to_dict('records')
            
            # Extract effect sizes and add to results
            effect_sizes = {}
            for result in anova_results:
                effect_name = result['Source']
                effect_sizes[effect_name] = {
                    'partial_eta_sq': result.get('np2', None),
                    'partial_omega_sq': None,  # Calculate this below if not available
                    'f_value': result.get('F', None),
                    'p_value': result.get('p-unc', None),
                    'significant': result.get('p-unc', 1.0) < alpha
                }
                
                # Calculate partial omega squared if not provided by pingouin
                if 'np2' in result and 'F' in result and 'DF' in result and 'num Obs' in result:
                    F = result['F']
                    df1 = result['DF']
                    num_obs = result.get('num Obs', len(data[subject_id].unique()))
                    df2 = num_obs - 1  # Approximate
                    
                    # Formula for partial omega squared
                    partial_omega_sq = (df1 * (F - 1)) / (df1 * (F - 1) + df2)
                    effect_sizes[effect_name]['partial_omega_sq'] = partial_omega_sq
            
            # Create sphericity test results
            sphericity_tests = {}
            for source in rm_anova_results['Source'].unique():
                if source in within_factors or ':' in source:  # Main effect or interaction
                    # Extract from rm_anova_results if available
                    source_row = rm_anova_results[rm_anova_results['Source'] == source]
                    if 'sphericity' in source_row.columns:
                        sphericity_tests[source] = {
                            'W': source_row['sphericity'].values[0] if not pd.isna(source_row['sphericity'].values[0]) else None,
                            'p_value': source_row['p-GG-corr'].values[0] if 'p-GG-corr' in source_row.columns else None,
                            'eps_gg': source_row['eps-GG'].values[0] if 'eps-GG' in source_row.columns else None,
                            'eps_hf': source_row['eps-HF'].values[0] if 'eps-HF' in source_row.columns else None,
                            'significant': source_row['p-unc'].values[0] < alpha if 'p-unc' in source_row.columns else None
                        }
                    else:
                        # If sphericity info not in results, indicate it was not tested
                        sphericity_tests[source] = {
                            'note': "Sphericity not tested or not applicable"
                        }
            
            # Create post-hoc test results for significant effects
            posthoc_results = {}
            
            # Conduct post-hoc tests for main effects
            for factor in within_factors:
                effect_name = factor
                if effect_name in effect_sizes and effect_sizes[effect_name]['significant']:
                    # Run pairwise t-tests with Bonferroni correction
                    posthoc = pg.pairwise_ttests(
                        data=data,
                        dv=outcome,
                        within=factor,
                        subject=subject_id,
                        padjust='bonf'
                    )
                    
                    # Convert to more readable format
                    factor_posthoc = []
                    for _, row in posthoc.iterrows():
                        factor_posthoc.append({
                            'A': row['A'],
                            'B': row['B'],
                            'mean_diff': row['mean(A)'] - row['mean(B)'],
                            't': row['T'],
                            'p_uncorrected': row['p-unc'],
                            'p_adjusted': row['p-corr'],
                            'significant': row['p-corr'] < alpha,
                            'hedges_g': row.get('hedges', None),
                            'cohen_d': row.get('cohen-d', None)
                        })
                    
                    posthoc_results[effect_name] = factor_posthoc
            
            # Conduct post-hoc tests for interactions
            for i in range(len(within_factors)):
                for j in range(i + 1, len(within_factors)):
                    interaction_name = f"{within_factors[i]}:{within_factors[j]}"
                    
                    # Check if this interaction exists and is significant
                    if interaction_name in effect_sizes and effect_sizes[interaction_name]['significant']:
                        # For each level of the first factor, compare levels of the second factor
                        for level_i in data[within_factors[i]].unique():
                            # Create a subset of data for this level
                            subset = data[data[within_factors[i]] == level_i]
                            
                            # Run pairwise t-tests for the second factor within this level
                            posthoc = pg.pairwise_ttests(
                                data=subset,
                                dv=outcome,
                                within=within_factors[j],
                                subject=subject_id,
                                padjust='bonf'
                            )
                            
                            # Store results
                            interaction_key = f"{interaction_name} (at {within_factors[i]}={level_i})"
                            
                            # Convert to more readable format
                            interaction_posthoc = []
                            for _, row in posthoc.iterrows():
                                interaction_posthoc.append({
                                    'A': row['A'],
                                    'B': row['B'],
                                    'mean_diff': row['mean(A)'] - row['mean(B)'],
                                    't': row['T'],
                                    'p_uncorrected': row['p-unc'],
                                    'p_adjusted': row['p-corr'],
                                    'significant': row['p-corr'] < alpha,
                                    'hedges_g': row.get('hedges', None),
                                    'cohen_d': row.get('cohen-d', None)
                                })
                            
                            posthoc_results[interaction_key] = interaction_posthoc
            
            # Handle higher-order interactions (for 3+ factors)
            if len(within_factors) > 2:
                # This would need a more complex approach, potentially breaking down by combinations of levels
                pass
            
        except Exception as e:
            # Fallback to using statsmodels mixedlm if pingouin fails
            warnings.warn(f"Error using pingouin for RM ANOVA: {str(e)}. Falling back to mixed linear model.")
            
            # Construct the formula
            formula = f"{outcome} ~ C({') * C('.join(within_factors)})"
            model = smf.mixedlm(formula, data, groups=data[subject_id])
            results = model.fit()
            
            # Get the names of the within-subject factors and their interactions
            within_terms = [f"C({factor})" for factor in within_factors]
            interaction_terms = []
            for i in range(len(within_factors)):
                for j in range(i + 1, len(within_factors)):
                    interaction_terms.append(f"C({within_factors[i]}):C({within_factors[j]})")
            
            # Add higher-order interactions if there are more than two within-subject factors
            if len(within_factors) > 2:
                interaction_terms.append(":".join([f"C({factor})" for factor in within_factors]))
            
            # Extract p-values and F-values for within-subject factors and interactions
            anova_results = []
            effect_sizes = {}
            
            for term in within_terms + interaction_terms:
                term_p_values = {}
                term_params = {}
                term_t_values = {}
                
                for key, p_value in results.pvalues.items():
                    if term in key:  # Check if the term is part of the key
                        term_p_values[key] = p_value
                        term_params[key] = results.params[key]
                        term_t_values[key] = results.tvalues[key]
                
                if term_p_values:
                    # Calculate mean p-value and t-value for the term
                    mean_p = np.mean(list(term_p_values.values()))
                    mean_t = np.mean([abs(val) for val in term_t_values.values()])
                    
                    # Calculate approximate F-value from t-values
                    approx_F = mean_t ** 2
                    
                    # Calculate approximate partial eta-squared
                    df1 = len(term_p_values)
                    df2 = results.df_resid
                    approx_eta_sq = (df1 * approx_F) / (df1 * approx_F + df2)
                    
                    # Calculate approximate partial omega-squared
                    approx_omega_sq = (df1 * (approx_F - 1)) / (df1 * (approx_F - 1) + df2)
                    
                    # Clean up term name for display
                    clean_term = term.replace('C(', '').replace(')', '')
                    
                    anova_results.append({
                        'Source': clean_term,
                        'F': approx_F,
                        'p-unc': mean_p,
                        'DF': df1,
                        'DF_error': df2
                    })
                    
                    # Store effect sizes
                    effect_sizes[clean_term] = {
                        'partial_eta_sq': approx_eta_sq,
                        'partial_omega_sq': approx_omega_sq,
                        'f_value': approx_F,
                        'p_value': mean_p,
                        'significant': mean_p < alpha
                    }
            
            # We don't have sphericity tests in this fallback method
            sphericity_tests = {factor: {'note': "Sphericity not tested (using mixed model fallback)"} for factor in within_factors}
            
            # Post-hoc tests - simplified approach for fallback
            posthoc_results = {}
            
            # Try basic pairwise comparisons for main effects
            for factor in within_factors:
                if factor in effect_sizes and effect_sizes[factor]['significant']:
                    try:
                        posthoc = []
                        factor_levels = data[factor].unique()
                        
                        for level_a, level_b in combinations(factor_levels, 2):
                            # Filter data for the two levels
                            data_a = data[data[factor] == level_a][outcome]
                            data_b = data[data[factor] == level_b][outcome]
                            
                            # Perform t-test
                            t_stat, p_val = stats.ttest_ind(data_a, data_b)
                            
                            # Calculate effect size (Cohen's d)
                            d = (data_a.mean() - data_b.mean()) / np.sqrt((data_a.std()**2 + data_b.std()**2) / 2)
                            
                            posthoc.append({
                                'A': level_a,
                                'B': level_b,
                                'mean_diff': data_a.mean() - data_b.mean(),
                                't': t_stat,
                                'p_uncorrected': p_val,
                                'p_adjusted': min(p_val * len(list(combinations(factor_levels, 2))), 1),  # Bonferroni
                                'significant': p_val * len(list(combinations(factor_levels, 2))) < alpha,
                                'cohen_d': d
                            })
                        
                        posthoc_results[factor] = posthoc
                    except Exception as e:
                        posthoc_results[factor] = [{'error': str(e)}]
        
        # Replace current assumptions implementation with standardized tests
        # Initialize assumptions dictionary
        assumptions = {}
        
        # 1. Normality test on the residuals
        try:
            # Calculate residuals similar to current approach
            # Different approaches depending on if we used pingouin or the fallback
            if 'rm_anova_results' in locals():
                # Calculate residuals from observed minus predicted
                # Reconstruct predicted values based on group means
                observed = data[outcome].values
                
                # Create a mapping to get predicted values
                data['residuals'] = np.nan
                for i, row in data.iterrows():
                    # Find the corresponding row in descriptives
                    filter_condition = True
                    for factor in within_factors:
                        filter_condition = filter_condition & (descriptives[factor] == row[factor])
                    
                    if filter_condition.any():
                        predicted_value = descriptives.loc[filter_condition, 'mean'].values[0]
                        data.loc[i, 'residuals'] = row[outcome] - predicted_value
                
                residuals = data['residuals'].dropna().values
            else:
                # For the fallback method, use model residuals
                residuals = results.resid
            
            # Create the test object first, then call run_test
            residual_normality_test = ResidualNormalityTest()
            normality_result = residual_normality_test.run_test(residuals=residuals)
            
            assumptions['residual_normality'] = normality_result
        except Exception as e:
            from data.assumptions.tests import AssumptionResult
            assumptions['residual_normality'] = {
                'result': AssumptionResult.FAILED,
                'details': str(e),
                'warnings': []
            }
        
        # 2. Sphericity test for each factor and interaction
        try:
            for source in rm_anova_results['Source'].unique():
                if source in within_factors or ':' in source:  # Main effect or interaction
                    source_row = rm_anova_results[rm_anova_results['Source'] == source]
                    
                    # Create the test object first, then call run_test
                    sphericity_test = SphericityTest()
                    within_factor = source if source in within_factors else source.split(':')
                    sphericity_result = sphericity_test.run_test(
                        data=data, 
                        subject_id=subject_id, 
                        within_factor=within_factor,
                        outcome=outcome
                    )
                    
                    key_prefix = f"sphericity_{source.replace(':', '_')}"
                    assumptions[key_prefix] = sphericity_result
                    
        except Exception as e:
            from data.assumptions.tests import AssumptionResult
            assumptions['sphericity'] = {
                'result': AssumptionResult.FAILED,
                'details': str(e),
                'warnings': []
            }
        
        # 3. Homogeneity of variance
        try:
            # For each combination of within-factors
            factor_combinations = data.groupby(within_factors).groups.keys()
            
            for combo in factor_combinations:
                # Create a condition name
                if len(within_factors) == 1:
                    condition_name = str(combo)
                else:
                    condition_name = "_".join([str(c) for c in combo])
                
                # Filter data for this condition
                condition_filter = True
                for i, factor in enumerate(within_factors):
                    if len(within_factors) == 1:
                        condition_filter = condition_filter & (data[factor] == combo)
                    else:
                        condition_filter = condition_filter & (data[factor] == combo[i])
                
                condition_data = data[condition_filter]
                
                # Create the test object first, then call run_test
                homogeneity_test = HomogeneityOfVarianceTest()
                
                # Convert numpy arrays to pandas Series to avoid the 'dropna' error
                numeric_data = pd.Series(condition_data[outcome].values)
                group_values = pd.Series(condition_data[subject_id].values)
                
                homogeneity_result = homogeneity_test.run_test(
                    data=numeric_data,
                    groups=group_values
                )
                
                key_prefix = f"homogeneity_condition_{condition_name}"
                assumptions[key_prefix] = homogeneity_result
        except Exception as e:
            from data.assumptions.tests import AssumptionResult
            assumptions['homogeneity'] = {
                'result': AssumptionResult.FAILED,
                'details': str(e),
                'warnings': []
            }
        
        # 4. Outlier detection for the outcome variable
        try:
            # Create the test object first, then call run_test
            outlier_test = OutlierTest()
            outlier_result = outlier_test.run_test(data=data[outcome].values)
            
            assumptions['outliers'] = outlier_result
        except Exception as e:
            from data.assumptions.tests import AssumptionResult
            assumptions['outliers'] = {
                'result': AssumptionResult.FAILED,
                'details': str(e),
                'warnings': []
            }
        
        # 5. Sample size adequacy
        try:
            # Create the test object first, then call run_test
            sample_size_test = SampleSizeTest()
            # For repeated measures, minimum recommended depends on various factors
            # A conservative rule of thumb might be 10 subjects per condition
            num_conditions = np.prod([len(data[factor].unique()) for factor in within_factors])
            min_recommended = num_conditions * 10
            
            sample_size_result = sample_size_test.run_test(
                data=data[subject_id].unique(),
                min_recommended=min_recommended
            )
            
            assumptions['sample_size'] = sample_size_result
        except Exception as e:
            from data.assumptions.tests import AssumptionResult
            assumptions['sample_size'] = {
                'result': AssumptionResult.FAILED,
                'details': str(e),
                'warnings': []
            }
        
        # 6. Independence test
        try:
            if 'residuals' in locals():
                # Create the test object first, then call run_test
                independence_test = IndependenceTest()
                independence_result = independence_test.run_test(data=residuals)
                
                assumptions['independence'] = independence_result
        except Exception as e:
            from data.assumptions.tests import AssumptionResult
            assumptions['independence'] = {
                'result': AssumptionResult.FAILED,
                'details': str(e),
                'warnings': []
            }
        
        # 7. Balance test for the design
        try:
            counts = data.groupby(within_factors + [subject_id]).size().reset_index(name='count')
            count_summary = counts.groupby(within_factors)['count'].agg(['mean', 'min', 'max', 'std']).reset_index()
            
            # Check if all counts are equal
            is_balanced = count_summary['std'].sum() == 0
            
            # Import here to ensure it's available
            from data.assumptions.tests import AssumptionResult
            assumptions['balanced_design'] = {
                'result': AssumptionResult.PASSED if is_balanced else AssumptionResult.WARNING,
                'satisfied': is_balanced,
                'count_summary': count_summary.to_dict('records'),
                'message': ("Design is balanced with equal observations across conditions" if is_balanced else 
                                 "Design is not perfectly balanced; some conditions may have different numbers of observations")
            }
        except Exception as e:
            from data.assumptions.tests import AssumptionResult
            assumptions['balanced_design'] = {
                'result': AssumptionResult.FAILED,
                'details': str(e),
                'warnings': []
            }
            
        # 8. Homoscedasticity test
        try:
            if 'residuals' in locals():
                # Calculate predicted values
                predicted = data[outcome].values - residuals
                
                # Create the test object first, then call run_test
                homoscedasticity_test = HomoscedasticityTest()
                homoscedasticity_result = homoscedasticity_test.run_test(
                    residuals=residuals,
                    predicted=predicted
                )
                
                assumptions['homoscedasticity'] = homoscedasticity_result
        except Exception as e:
            from data.assumptions.tests import AssumptionResult
            assumptions['homoscedasticity'] = {
                'result': AssumptionResult.FAILED,
                'details': str(e),
                'warnings': []
            }
            
        # 9. Normality test for each group separately
        try:
            # For each combination of within-factors
            factor_combinations = data.groupby(within_factors).groups.keys()
            
            for combo in factor_combinations:
                # Create a condition name
                if len(within_factors) == 1:
                    condition_name = str(combo)
                else:
                    condition_name = "_".join([str(c) for c in combo])
                
                # Filter data for this condition
                condition_filter = True
                for i, factor in enumerate(within_factors):
                    if len(within_factors) == 1:
                        condition_filter = condition_filter & (data[factor] == combo)
                    else:
                        condition_filter = condition_filter & (data[factor] == combo[i])
                
                condition_data = data[condition_filter]
                
                # Create the test object first, then call run_test
                normality_test = NormalityTest()
                normality_result = normality_test.run_test(data=condition_data[outcome].values)
                
                key_prefix = f"normality_condition_{condition_name}"
                assumptions[key_prefix] = normality_result
                
        except Exception as e:
            from data.assumptions.tests import AssumptionResult
            assumptions['normality'] = {
                'result': AssumptionResult.FAILED,
                'details': str(e),
                'warnings': []
            }
        
        # Create main effects plots
        for i, factor in enumerate(within_factors):
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Get factor data directly without additional aggregation
                if len(within_factors) == 1:
                    factor_data = descriptives  # If only one factor, use descriptives directly
                else:
                    # For multiple factors, calculate the mean across other factors
                    factor_data = data.groupby(factor)[outcome].agg(['mean', 'std', 'count']).reset_index()
                    factor_data['sem'] = factor_data['std'] / np.sqrt(factor_data['count'])
                
                # Create the bar plot with error bars
                sns.barplot(x=factor, y='mean', data=factor_data, ax=ax, color=PASTEL_COLORS[0], alpha=0.7)
                
                # Add error bars
                ax.errorbar(
                    x=np.arange(len(factor_data)), 
                    y=factor_data['mean'],
                    yerr=factor_data['sem'] * 1.96,  # 95% CI
                    fmt='none', 
                    color='black', 
                    capsize=5
                )
                
                # Add individual data points if there aren't too many
                if len(data[factor].unique()) * len(data[subject_id].unique()) <= 100:
                    sns.stripplot(
                        x=factor, 
                        y=outcome, 
                        data=data, 
                        ax=ax, 
                        color='black', 
                        alpha=0.5,
                        size=3,
                        jitter=True
                    )
                
                # Add significance indicators if we have effect sizes for this factor
                if factor in effect_sizes and 'p_value' in effect_sizes[factor]:
                    p_val = effect_sizes[factor]['p_value']
                    eta_sq = effect_sizes[factor]['partial_eta_sq']
                    
                    # Add annotation - with check for None values
                    significance = ""
                    if p_val is not None:
                        if p_val < 0.001:
                            significance = "***"
                        elif p_val < 0.01:
                            significance = "**"
                        elif p_val < 0.05:
                            significance = "*"
                        
                        if eta_sq is not None:
                            ax.set_title(f"Main Effect of {factor} {significance}\np={p_val:.4f}, η²={eta_sq:.3f}")
                        else:
                            ax.set_title(f"Main Effect of {factor} {significance}\np={p_val:.4f}")
                    else:
                        if eta_sq is not None:
                            ax.set_title(f"Main Effect of {factor}\nη²={eta_sq:.3f}")
                        else:
                            ax.set_title(f"Main Effect of {factor}")
                else:
                    ax.set_title(f"Main Effect of {factor}")
                
                ax.set_xlabel(factor)
                ax.set_ylabel(outcome)
                
                # Add horizontal line at overall mean
                ax.axhline(y=overall_mean, color='red', linestyle='--', alpha=0.7, label=f'Overall Mean = {overall_mean:.2f}')
                ax.legend()
                
                fig.tight_layout()
                figures[f'main_effect_{factor}'] = fig_to_svg(fig)
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                figures[f'main_effect_{factor}_error'] = f"{str(e)}\n{error_details}"
        
        # Create interaction plots for all two-way interactions
        for i in range(len(within_factors)):
            for j in range(i + 1, len(within_factors)):
                factor1 = within_factors[i]
                factor2 = within_factors[j]
                interaction_name = f"{factor1}:{factor2}"
                
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Get the means for each combination of factors
                    interaction_data = descriptives[descriptives[within_factors].isin(data[within_factors].drop_duplicates())]
                    
                    # Create the interaction plot
                    interaction_plot(
                        x=interaction_data[factor1],
                        trace=interaction_data[factor2],
                        response=interaction_data['mean'],
                        ax=ax,
                        legendtitle=factor2
                    )
                    
                    # Enhance the plot
                    ax.set_xlabel(factor1)
                    ax.set_ylabel(f'Mean {outcome}')
                    
                    # Add significance information if available
                    if interaction_name in effect_sizes and 'p_value' in effect_sizes[interaction_name]:
                        p_val = effect_sizes[interaction_name]['p_value']
                        eta_sq = effect_sizes[interaction_name]['partial_eta_sq']
                        
                        # Add annotation
                        significance = ""
                        if p_val < 0.001:
                            significance = "***"
                        elif p_val < 0.01:
                            significance = "**"
                        elif p_val < 0.05:
                            significance = "*"
                        
                        ax.set_title(f"Interaction Effect: {factor1} × {factor2} {significance}\np={p_val:.4f}, η²={eta_sq:.3f}")
                    else:
                        ax.set_title(f"Interaction Effect: {factor1} × {factor2}")
                    
                    # Add horizontal line at overall mean
                    ax.axhline(y=overall_mean, color='red', linestyle='--', alpha=0.7, label=f'Overall Mean = {overall_mean:.2f}')
                    
                    # Make sure the legend includes all elements
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles=handles, labels=labels)
                    
                    fig.tight_layout()
                    figures[f'interaction_{factor1}_{factor2}'] = fig_to_svg(fig)
                except Exception as e:
                    figures[f'interaction_{factor1}_{factor2}_error'] = str(e)
        
        # Create forest plot of effect sizes
        try:
            # Extract effect names and values
            names = []
            eta_sqs = []
            significant = []
            
            for name, effect in effect_sizes.items():
                if 'partial_eta_sq' in effect and effect['partial_eta_sq'] is not None:
                    names.append(name)
                    eta_sqs.append(effect['partial_eta_sq'])
                    significant.append(effect['significant'])
            
            # Check if we have any valid effect sizes before proceeding
            if not names:
                # Try to calculate effect sizes from F values if available
                for name, effect in effect_sizes.items():
                    if ('f_value' in effect and effect['f_value'] is not None and 
                        'p_value' in effect and effect['p_value'] is not None):
                        # Simple estimate for partial eta squared
                        f_val = effect['f_value']
                        df1 = 1  # Approximate
                        df2 = len(data[subject_id].unique()) - 1  # n-1
                        
                        # Calculate eta squared
                        eta_sq = (f_val * df1) / (f_val * df1 + df2)
                        
                        # Update the effect
                        effect['partial_eta_sq'] = eta_sq
                        
                        # Add to our plotting lists
                        names.append(name)
                        eta_sqs.append(eta_sq)
                        significant.append(effect['significant'])
            
            # Final check if we have valid effect sizes to plot
            if not names:
                figures['effect_sizes_error'] = "No valid effect sizes found with partial_eta_sq values"
            else:
                fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.5)))
                
                # Sort by effect size (largest first)
                sorted_indices = np.argsort(eta_sqs)[::-1]
                names = [names[i] for i in sorted_indices]
                eta_sqs = [eta_sqs[i] for i in sorted_indices]
                significant = [significant[i] for i in sorted_indices]
                
                # Create the plot
                y_pos = np.arange(len(names))
                colors = [PASTEL_COLORS[0] if sig else PASTEL_COLORS[3] for sig in significant]
                
                ax.barh(y_pos, eta_sqs, color=colors, alpha=0.7)
                
                # Add effect size labels
                for i, v in enumerate(eta_sqs):
                    ax.text(v + 0.01, i, f"{v:.3f}", va='center')
                
                # Add effect size interpretation
                for i, v in enumerate(eta_sqs):
                    effect_size_text = ""
                    if v >= 0.14:
                        effect_size_text = "Large"
                    elif v >= 0.06:
                        effect_size_text = "Medium"
                    elif v >= 0.01:
                        effect_size_text = "Small"
                    else:
                        effect_size_text = "Negligible"
                    
                    ax.text(max(eta_sqs) + 0.05, i, effect_size_text, va='center')
                
                # Add vertical reference lines
                ax.axvline(x=0.01, color='gray', linestyle='--', alpha=0.5, label='Small (η²=0.01)')
                ax.axvline(x=0.06, color='gray', linestyle='--', alpha=0.5, label='Medium (η²=0.06)')
                ax.axvline(x=0.14, color='gray', linestyle='--', alpha=0.5, label='Large (η²=0.14)')
                
                # Customize plot
                ax.set_yticks(y_pos)
                ax.set_yticklabels(names)
                ax.set_xlabel('Partial η² (Effect Size)')
                ax.set_title('Effect Sizes for All Effects')
                ax.set_xlim(0, max(eta_sqs) * 1.3)
                
                # Add legend for significance
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='s', color='w', markerfacecolor='tab:blue', markersize=10, label='Significant'),
                    Line2D([0], [0], marker='s', color='w', markerfacecolor='tab:gray', markersize=10, label='Non-significant')
                ]
                ax.legend(handles=legend_elements, loc='lower right')
                
                fig.tight_layout()
                figures['effect_sizes'] = fig_to_svg(fig)
        except Exception as e:
            figures['effect_sizes_error'] = str(e)
        
        # Create post-hoc comparison plot for significant effects
        for effect_name, posthoc in posthoc_results.items():
            try:
                # Check if we have valid post-hoc results
                if posthoc and isinstance(posthoc, list) and 'error' not in posthoc[0]:
                    # Create a clean version of effect name for display
                    clean_effect = effect_name.replace(':', ' × ').replace(' (at ', '\n(').replace('=', ' = ')
                    
                    fig, ax = plt.subplots(figsize=(10, max(6, len(posthoc) * 0.4)))
                    
                    # Extract comparison information
                    comparisons = []
                    p_values = []
                    mean_diffs = []
                    significant = []
                    
                    for comp in posthoc:
                        comparisons.append(f"{comp['A']} vs {comp['B']}")
                        p_values.append(comp['p_adjusted'])
                        mean_diffs.append(comp['mean_diff'])
                        significant.append(comp['significant'])
                    
                    # Create the plot
                    y_pos = np.arange(len(comparisons))
                    colors = [PASTEL_COLORS[0] if sig else PASTEL_COLORS[3] for sig in significant]
                    
                    ax.barh(y_pos, mean_diffs, color=colors, alpha=0.7)
                    
                    # Add zero reference line
                    ax.axvline(x=0, color='black', linestyle='-', alpha=0.7)
                    
                    # Add comparison labels and p-values
                    for i, (comp, p, sig) in enumerate(zip(comparisons, p_values, significant)):
                        # Add significance markers
                        sig_marker = ""
                        if p < 0.001:
                            sig_marker = "***"
                        elif p < 0.01:
                            sig_marker = "**"
                        elif p < 0.05:
                            sig_marker = "*"
                        
                        # Add p-value
                        p_text = f"p = {p:.4f} {sig_marker}"
                        ax.text(mean_diffs[i] + (0.05 * max(abs(min(mean_diffs)), abs(max(mean_diffs)))),
                              i, p_text, va='center')
                    
                    # Customize plot
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(comparisons)
                    ax.set_xlabel(f'Mean Difference in {outcome}')
                    ax.set_title(f'Post-hoc Comparisons for {clean_effect}')
                    
                    # Set x-axis limits to be symmetric if there are both positive and negative diffs
                    if min(mean_diffs) < 0 and max(mean_diffs) > 0:
                        max_abs = max(abs(min(mean_diffs)), abs(max(mean_diffs)))
                        ax.set_xlim(-max_abs * 1.5, max_abs * 1.5)
                    
                    # Add legend for significance
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], marker='s', color='w', markerfacecolor='tab:blue', markersize=10, label='Significant'),
                        Line2D([0], [0], marker='s', color='w', markerfacecolor='tab:gray', markersize=10, label='Non-significant')
                    ]
                    ax.legend(handles=legend_elements, loc='lower right')
                    
                    fig.tight_layout()
                    figures[f'posthoc_{effect_name.replace(":", "_").replace(" ", "_").replace("=", "_")}'] = fig_to_svg(fig)
            except Exception as e:
                figures[f'posthoc_{effect_name}_error'] = str(e)
       
        # Create profile plots to show individual subject responses across conditions
        try:
            # For simpler cases with one within-subject factor
            if len(within_factors) == 1:
                factor = within_factors[0]
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Get unique subjects and factor levels
                subjects = data[subject_id].unique()
                levels = data[factor].unique()
                
                # Limit to 20 random subjects if there are too many
                if len(subjects) > 20:
                    np.random.seed(42)  # For reproducibility
                    subjects = np.random.choice(subjects, 20, replace=False)
                
                # Create profile plots for each subject
                for subj in subjects:
                    subj_data = data[data[subject_id] == subj]
                    
                    # Make sure we have all levels for this subject
                    if len(subj_data) == len(levels):
                        # Sort by factor level for proper line connections
                        subj_data = subj_data.sort_values(by=factor)
                        
                        # Convert factor levels to numeric position for line plot
                        x_positions = np.arange(len(levels))
                        
                        # Plot subject's profile
                        ax.plot(x_positions, subj_data[outcome].values, marker='o', 
                               color=PASTEL_COLORS[i % len(PASTEL_COLORS)], alpha=0.3, linewidth=0.8)
                
                # Add the mean profile (bold line)
                mean_data = data.groupby(factor)[outcome].mean().reset_index()
                mean_data = mean_data.sort_values(by=factor)
                
                # Get x positions matching the factor levels
                x_positions = np.arange(len(levels))
                
                # Plot the mean profile
                ax.plot(x_positions, mean_data[outcome].values, marker='o', color='orange', 
                      linewidth=3, markersize=8, label='Group Mean')
                
                # Set x-axis labels to the factor levels
                ax.set_xticks(x_positions)
                ax.set_xticklabels(mean_data[factor])
                
                # Add labels and title
                ax.set_xlabel(factor)
                ax.set_ylabel(outcome)
                ax.set_title(f'Individual Subject Profiles Across {factor} Conditions')
                
                # Add legend
                ax.legend()
                
                fig.tight_layout()
                figures['subject_profiles'] = fig_to_svg(fig)
            elif len(within_factors) == 2:
                # For two within-subject factors, show profiles across first factor for each level of second factor
                factor1 = within_factors[0]
                factor2 = within_factors[1]
                
                # Get unique levels for both factors
                levels1 = sorted(data[factor1].unique())
                levels2 = sorted(data[factor2].unique())
                
                # Create a figure with subplots for each level of factor2
                fig, axes = plt.subplots(1, len(levels2), figsize=(len(levels2) * 5, 6), sharey=True)
                
                # Ensure axes is always a list
                if len(levels2) == 1:
                    axes = [axes]
                
                # Get random subjects (limit to 20 if there are too many)
                subjects = data[subject_id].unique()
                if len(subjects) > 20:
                    np.random.seed(42)  # For reproducibility
                    subjects = np.random.choice(subjects, 20, replace=False)
                
                for i, level2 in enumerate(levels2):
                    ax = axes[i]
                    
                    # Filter data for this level of factor2
                    level_data = data[data[factor2] == level2]
                    
                    # Create profile plots for each subject
                    for subj in subjects:
                        subj_data = level_data[level_data[subject_id] == subj]
                        
                        # Make sure we have all levels of factor1 for this subject
                        if len(subj_data) == len(levels1):
                            # Sort by factor1 level
                            subj_data = subj_data.sort_values(by=factor1)
                            
                            # Convert factor levels to numeric position for line plot
                            x_positions = np.arange(len(levels1))
                            
                            # Plot subject's profile
                            ax.plot(x_positions, subj_data[outcome].values, marker='o', 
                                   color=PASTEL_COLORS[i % len(PASTEL_COLORS)], alpha=0.3, linewidth=0.8)
                    
                    # Add the mean profile (bold line)
                    mean_data = level_data.groupby(factor1)[outcome].mean().reset_index()
                    mean_data = mean_data.sort_values(by=factor1)
                    
                    # Get x positions matching the factor levels
                    x_positions = np.arange(len(levels1))
                    
                    # Plot the mean profile
                    ax.plot(x_positions, mean_data[outcome].values, marker='o', color='orange', 
                          linewidth=3, markersize=8, label='Group Mean')
                    
                    # Set x-axis labels to the factor1 levels
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels(mean_data[factor1])
                    
                    # Add labels and title
                    ax.set_xlabel(factor1)
                    if i == 0:
                        ax.set_ylabel(outcome)
                    ax.set_title(f'{factor2} = {level2}')
                
                # Add common title
                fig.suptitle(f'Individual Subject Profiles Across Conditions', fontsize=16)
                
                # Add legend to the first subplot
                axes[0].legend()
                
                fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
                figures['subject_profiles'] = fig_to_svg(fig)
        except Exception as e:
            figures['subject_profiles_error'] = str(e)
        
        # Create an interpretation
        try:
            interpretation = f"**Repeated Measures ANOVA Results for {outcome}**\n\n"
            
            # Main effects
            interpretation += "**Main Effects:**\n"
            for factor in within_factors:
                if factor in effect_sizes:
                    effect = effect_sizes[factor]
                    interpretation += f"- {factor}: "
                    
                    if effect['significant']:
                        interpretation += f"Significant effect (F = {effect['f_value']:.2f}, p = {effect['p_value']:.4f}, partial η² = {effect['partial_eta_sq']:.3f}). "
                        
                        # Interpret effect size
                        eta_sq = effect['partial_eta_sq']
                        if eta_sq >= 0.14:
                            effect_text = "large"
                        elif eta_sq >= 0.06:
                            effect_text = "medium"
                        else:
                            effect_text = "small"
                        
                        interpretation += f"This represents a {effect_text} effect size. "
                        
                        # Add information about post-hoc tests if available
                        if factor in posthoc_results:
                            sig_comparisons = [f"{comp['A']} vs {comp['B']}" for comp in posthoc_results[factor] if comp['significant']]
                            
                            if sig_comparisons:
                                interpretation += f"Post-hoc tests revealed significant differences between: {', '.join(sig_comparisons)}."
                            else:
                                interpretation += "Post-hoc tests did not reveal any significant pairwise differences, suggesting the overall effect may be driven by the cumulative differences across all conditions."
                    else:
                        interpretation += f"No significant effect (F = {effect['f_value']:.2f}, p = {effect['p_value']:.4f}, partial η² = {effect['partial_eta_sq']:.3f})."
                    
                    interpretation += "\n"
            
            # Interactions
            interaction_found = False
            if len(within_factors) > 1:
                interpretation += "\n**Interactions:**\n"
                
                for i in range(len(within_factors)):
                    for j in range(i + 1, len(within_factors)):
                        interaction_name = f"{within_factors[i]}:{within_factors[j]}"
                        
                        if interaction_name in effect_sizes:
                            interaction_found = True
                            effect = effect_sizes[interaction_name]
                            interpretation += f"- {within_factors[i]} × {within_factors[j]}: "
                            
                            if effect['significant']:
                                interpretation += f"Significant interaction (F = {effect['f_value']:.2f}, p = {effect['p_value']:.4f}, partial η² = {effect['partial_eta_sq']:.3f}). "
                                
                                # Interpret effect size
                                eta_sq = effect['partial_eta_sq']
                                if eta_sq >= 0.14:
                                    effect_text = "large"
                                elif eta_sq >= 0.06:
                                    effect_text = "medium"
                                else:
                                    effect_text = "small"
                                
                                interpretation += f"This represents a {effect_text} effect size. "
                                
                                # Add information about simple effects if available
                                simple_effects_keys = [k for k in posthoc_results.keys() if k.startswith(interaction_name)]
                                
                                if simple_effects_keys:
                                    interpretation += "Simple effects analysis shows that:\n"
                                    
                                    for key in simple_effects_keys:
                                        condition = key.replace(interaction_name + " (at ", "").replace(")", "")
                                        sig_comparisons = [f"{comp['A']} vs {comp['B']}" for comp in posthoc_results[key] if comp['significant']]
                                        
                                        if sig_comparisons:
                                            interpretation += f"  - When {condition}, significant differences were found between: {', '.join(sig_comparisons)}.\n"
                                        else:
                                            interpretation += f"  - When {condition}, no significant differences were found between conditions.\n"
                            else:
                                interpretation += f"No significant interaction (F = {effect['f_value']:.2f}, p = {effect['p_value']:.4f}, partial η² = {effect['partial_eta_sq']:.3f})."
                            
                            interpretation += "\n"
                
                if not interaction_found:
                    interpretation += "No interactions were examined.\n"
            
            # Update the assumption checks section
            interpretation += "\n**Assumption Checks:**\n"
            
            # Residual Normality
            if 'residual_normality' in assumptions:
                result = assumptions['residual_normality']['result']
                message = assumptions['residual_normality'].get('message', '')
                
                if result == AssumptionResult.PASSED:
                    interpretation += f"- Normality of Residuals: The assumption was met. {message}\n"
                elif result == AssumptionResult.WARNING:
                    interpretation += f"- Normality of Residuals: The assumption has minor violations. {message} "
                    interpretation += "This may slightly affect the Type I error rate.\n"
                elif result == AssumptionResult.FAILED:
                    interpretation += f"- Normality of Residuals: The assumption was violated. {message} "
                    interpretation += "Consider using non-parametric alternatives like Friedman's test if this violation is severe.\n"
                elif result == AssumptionResult.NOT_APPLICABLE:
                    interpretation += "- Normality of Residuals: This assumption could not be tested.\n"
            
            # Sphericity
            sphericity_violations = False
            sphericity_keys = [key for key in assumptions.keys() if key.startswith('sphericity_')]
            
            for key in sphericity_keys:
                factor_name = key.replace('sphericity_', '')
                result = assumptions[key]['result']
                
                if result == AssumptionResult.PASSED:
                    interpretation += f"- Sphericity for {factor_name}: Assumption was met.\n"
                elif result == AssumptionResult.WARNING:
                    sphericity_violations = True
                    p_value = assumptions[key].get('p_value', "unknown")
                    interpretation += f"- Sphericity for {factor_name}: Minor violation detected. "
                    interpretation += f"Greenhouse-Geisser correction is recommended.\n"
                elif result == AssumptionResult.FAILED:
                    sphericity_violations = True
                    p_value = assumptions[key].get('p_value', "unknown")
                    interpretation += f"- Sphericity for {factor_name}: Significant violation detected. "
                    interpretation += f"Greenhouse-Geisser correction has been applied.\n"
                elif result == AssumptionResult.NOT_APPLICABLE:
                    interpretation += f"- Sphericity for {factor_name}: This assumption could not be tested or is not applicable.\n"
            
            if sphericity_violations:
                interpretation += "  Note: When sphericity is violated, the reported p-values should be adjusted accordingly.\n"
            
            # Homogeneity of Variance
            homogeneity_keys = [key for key in assumptions.keys() if key.startswith('homogeneity_condition_')]
            
            homogeneity_violations = 0
            for key in homogeneity_keys:
                condition_name = key.replace('homogeneity_condition_', '')
                if assumptions[key]['result'] == AssumptionResult.FAILED:
                    homogeneity_violations += 1
            
            if homogeneity_keys:
                if homogeneity_violations == 0:
                    interpretation += f"- Homogeneity of Variance: The assumption was met for all {len(homogeneity_keys)} conditions.\n"
                elif homogeneity_violations < len(homogeneity_keys) / 2:
                    interpretation += f"- Homogeneity of Variance: The assumption was violated for {homogeneity_violations} out of {len(homogeneity_keys)} conditions. "
                    interpretation += "This may not be problematic for balanced designs but could affect results with unbalanced data.\n"
                else:
                    interpretation += f"- Homogeneity of Variance: The assumption was violated for {homogeneity_violations} out of {len(homogeneity_keys)} conditions. "
                    interpretation += "This could affect the validity of the results, especially with unbalanced data.\n"
            
            # Outliers
            if 'outliers' in assumptions:
                result = assumptions['outliers']['result']
                
                if result == AssumptionResult.PASSED:
                    interpretation += "- Outliers: No extreme outliers were detected in the outcome variable.\n"
                elif result == AssumptionResult.WARNING:
                    outlier_count = len(assumptions['outliers'].get('outliers_list', []))
                    interpretation += f"- Outliers: {outlier_count} potential outliers were detected. "
                    interpretation += "These may have a minor influence on the results and should be examined.\n"
                elif result == AssumptionResult.FAILED:
                    outlier_count = len(assumptions['outliers'].get('outliers_list', []))
                    interpretation += f"- Outliers: {outlier_count} significant outliers were detected. "
                    interpretation += "These may substantially influence the results and should be carefully examined.\n"
                elif result == AssumptionResult.NOT_APPLICABLE:
                    interpretation += "- Outliers: This assumption could not be tested.\n"
            
            # Balanced design
            if 'balanced_design' in assumptions:
                result = assumptions['balanced_design']['result']
                
                if result == AssumptionResult.PASSED:
                    interpretation += "- Balanced Design: The design is balanced with equal observations across all conditions.\n"
                elif result == AssumptionResult.WARNING:
                    interpretation += "- Balanced Design: The design is not perfectly balanced. Some conditions may have different numbers of observations, which could affect the validity of the results.\n"
                elif result == AssumptionResult.FAILED:
                    interpretation += "- Balanced Design: The design has significant imbalances. The unequal observations across conditions may substantially affect the analysis results.\n"
                elif result == AssumptionResult.NOT_APPLICABLE:
                    interpretation += "- Balanced Design: This could not be assessed.\n"
            
            # Sample Size
            if 'sample_size' in assumptions:
                result = assumptions['sample_size']['result']
                
                if result == AssumptionResult.PASSED:
                    sample_size = assumptions['sample_size'].get('actual', 'unknown')
                    minimum = assumptions['sample_size'].get('minimum_required', 'unknown')
                    interpretation += f"- Sample Size: The sample size ({sample_size}) is adequate (minimum recommended: {minimum}).\n"
                elif result == AssumptionResult.WARNING:
                    sample_size = assumptions['sample_size'].get('actual', 'unknown')
                    minimum = assumptions['sample_size'].get('minimum_required', 'unknown')
                    interpretation += f"- Sample Size: The sample size ({sample_size}) is smaller than recommended ({minimum}). "
                    interpretation += "This may limit statistical power and the ability to detect small effects.\n"
                elif result == AssumptionResult.FAILED:
                    sample_size = assumptions['sample_size'].get('actual', 'unknown')
                    minimum = assumptions['sample_size'].get('minimum_required', 'unknown')
                    interpretation += f"- Sample Size: The sample size ({sample_size}) is considerably smaller than recommended ({minimum}). "
                    interpretation += "This significantly limits statistical power and increases the risk of Type II errors.\n"
                elif result == AssumptionResult.NOT_APPLICABLE:
                    interpretation += "- Sample Size: This could not be assessed.\n"
            
            # Independence
            if 'independence' in assumptions:
                result = assumptions['independence']['result']
                message = assumptions['independence'].get('message', '')
                
                if result == AssumptionResult.PASSED:
                    interpretation += f"- Independence: The assumption was met. {message}\n"
                elif result == AssumptionResult.WARNING:
                    interpretation += f"- Independence: Minor violations detected. {message} "
                    interpretation += "This may slightly affect the validity of the results.\n"
                elif result == AssumptionResult.FAILED:
                    interpretation += f"- Independence: Significant violations detected. {message} "
                    interpretation += "This could substantially affect the validity of the results. Consider using models that account for dependency in the data.\n"
                elif result == AssumptionResult.NOT_APPLICABLE:
                    interpretation += "- Independence: This assumption could not be tested.\n"
            
            # Homoscedasticity
            if 'homoscedasticity' in assumptions:
                result = assumptions['homoscedasticity']['result']
                
                if result == AssumptionResult.PASSED:
                    interpretation += "- Homoscedasticity: The assumption of constant variance of residuals was met.\n"
                elif result == AssumptionResult.WARNING:
                    interpretation += "- Homoscedasticity: Minor heteroscedasticity detected in the residuals. "
                    interpretation += "This may slightly affect standard errors and inference.\n"
                elif result == AssumptionResult.FAILED:
                    interpretation += "- Homoscedasticity: Significant heteroscedasticity detected in the residuals. "
                    interpretation += "This could substantially affect standard errors and inference. Consider transformations or robust standard errors.\n"
                elif result == AssumptionResult.NOT_APPLICABLE:
                    interpretation += "- Homoscedasticity: This assumption could not be tested.\n"
            
            # Normality by condition
            normality_condition_keys = [key for key in assumptions.keys() if key.startswith('normality_condition_')]
            
            normality_violations = 0
            for key in normality_condition_keys:
                if assumptions[key]['result'] == AssumptionResult.FAILED:
                    normality_violations += 1
            
            if normality_condition_keys:
                if normality_violations == 0:
                    interpretation += f"- Normality by Condition: All {len(normality_condition_keys)} conditions meet the normality assumption.\n"
                elif normality_violations < len(normality_condition_keys) / 2:
                    interpretation += f"- Normality by Condition: {normality_violations} out of {len(normality_condition_keys)} conditions violate the normality assumption. "
                    interpretation += "This may have a minor impact on results due to the robustness of ANOVA to moderate normality violations.\n"
                else:
                    interpretation += f"- Normality by Condition: {normality_violations} out of {len(normality_condition_keys)} conditions violate the normality assumption. "
                    interpretation += "This could affect the validity of the results. Consider transformations or non-parametric alternatives.\n"
            
            # Update the Summary and Recommendations section
            interpretation += "\n**Summary and Recommendations:**\n"
            
            # Count significant effects
            significant_effects = [name for name, effect in effect_sizes.items() if effect['significant']]
            
            if significant_effects:
                interpretation += f"The analysis revealed significant effects for: {', '.join(significant_effects)}. "
                
                # Get the strongest effect
                strongest_effect = max(effect_sizes.items(), key=lambda x: x[1].get('partial_eta_sq', 0))
                interpretation += f"The strongest effect was for {strongest_effect[0]} (partial η² = {strongest_effect[1]['partial_eta_sq']:.3f}).\n\n"
            else:
                interpretation += "The analysis did not reveal any significant effects.\n\n"
            
            # Add specific recommendations based on assumption violations
            assumption_violations = []
            
            # Check for violations in the main assumptions
            if 'residual_normality' in assumptions and assumptions['residual_normality']['result'] == AssumptionResult.FAILED:
                assumption_violations.append("normality of residuals")
            
            if sphericity_violations:
                assumption_violations.append("sphericity")
            
            if homogeneity_keys and homogeneity_violations > len(homogeneity_keys) / 2:
                assumption_violations.append("homogeneity of variance")
            
            if 'outliers' in assumptions and assumptions['outliers']['result'] == AssumptionResult.FAILED:
                assumption_violations.append("presence of outliers")
            
            if 'independence' in assumptions and assumptions['independence']['result'] == AssumptionResult.FAILED:
                assumption_violations.append("independence")
            
            if 'homoscedasticity' in assumptions and assumptions['homoscedasticity']['result'] == AssumptionResult.FAILED:
                assumption_violations.append("homoscedasticity")
            
            if normality_condition_keys and normality_violations > len(normality_condition_keys) / 2:
                assumption_violations.append("normality by condition")
            
            if assumption_violations:
                interpretation += f"Due to violations of {', '.join(assumption_violations)}, "
                
                if any(v in ["normality of residuals", "normality by condition", "outliers"] for v in assumption_violations):
                    interpretation += "consider using robust methods or non-parametric alternatives like Friedman's test. "
                
                if "sphericity" in assumption_violations:
                    interpretation += "ensure you use corrected p-values (Greenhouse-Geisser or Huynh-Feldt). "
                
                if "homogeneity of variance" in assumption_violations or "homoscedasticity" in assumption_violations:
                    interpretation += "consider transformations of the outcome variable or using mixed models that can handle heteroscedasticity. "
                
                if "independence" in assumption_violations:
                    interpretation += "consider models that account for dependencies in the data. "
                
                interpretation += "These violations may affect the Type I error rate or statistical power of the analysis.\n"
            else:
                interpretation += "All assumptions for repeated measures ANOVA appear to be reasonably satisfied, suggesting the results can be interpreted with confidence.\n"
        except Exception as e:
            interpretation = f"Error generating interpretation: {str(e)}"
            
        # Prepare return dictionary with comprehensive results
        return {
            'test': 'Repeated Measures ANOVA',
            'outcome': outcome,
            'within_factors': within_factors,
            'descriptives': descriptives.to_dict('records'),
            'anova_results': anova_results,
            'effect_sizes': effect_sizes,
            'sphericity_tests': sphericity_tests if 'sphericity_tests' in locals() else {},
            'posthoc_results': posthoc_results,
            'assumptions': assumptions,
            'figures': figures,
            'interpretation': interpretation,
            'alpha': alpha,
            'subject_id': subject_id,
            'n_subjects': len(data[subject_id].unique()),
            'balanced': assumptions.get('balanced_design', None) == AssumptionResult.PASSED
        }
    except Exception as e:
        return {
            'test': 'Repeated Measures ANOVA',
            'error': str(e),
            'traceback': traceback.format_exc()
        }