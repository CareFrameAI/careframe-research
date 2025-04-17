import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Dict, List, Any, Optional, Tuple
import traceback
from lifelines import KaplanMeierFitter, CoxPHFitter, NelsonAalenFitter
from lifelines.statistics import logrank_test, proportional_hazard_test, multivariate_logrank_test
from lifelines.utils import concordance_index
from lifelines.plotting import plot_lifetimes
from scipy import stats
from matplotlib.patches import Patch
from data.selection.detailed_tests.formatting import fig_to_svg, PASTEL_COLORS, PASTEL_CMAP
from data.assumptions.tests import (
    NormalityTest, 
    HomogeneityOfVarianceTest, 
    OutlierTest, 
    ProportionalHazardsTest, 
    SampleSizeTest, 
    InfluentialPointsTest,
    AutocorrelationTest,
    LinearityTest,
    DistributionFitTest,
    ParameterStabilityTest,
    AssumptionResult
)
from data.assumptions.format import AssumptionTestKeys

def survival_analysis(data: pd.DataFrame, time_variable: str, event_variable: str, group_variable: str, alpha: float) -> Dict[str, Any]:
    """
    Performs survival analysis with comprehensive statistics, assumption checks, and visualizations.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataset containing all variables
    time_variable : str
        Name of the variable representing time to event
    event_variable : str
        Name of the binary variable indicating whether event occurred (1) or was censored (0)
    group_variable : str
        Name of variable to group by (optional, use empty string if none)
    alpha : float
        Significance level for statistical tests
        
    Returns:
    --------
    Dict[str, Any]
        Comprehensive results including survival curves, test statistics, visualizations,
        Cox regression results, assumption checks, and interpretation
    """
    try:
            
        # Validate inputs
        if time_variable not in data.columns:
            raise ValueError(f"Time variable '{time_variable}' not found in data")
            
        if event_variable not in data.columns:
            raise ValueError(f"Event variable '{event_variable}' not found in data")
            
        if group_variable and group_variable not in data.columns:
            raise ValueError(f"Group variable '{group_variable}' not found in data")
            
        # Ensure time is numeric
        if not pd.api.types.is_numeric_dtype(data[time_variable]):
            raise ValueError(f"Time variable '{time_variable}' must be numeric")
            
        # Ensure event indicator is binary (0/1)
        if not set(data[event_variable].unique()).issubset({0, 1}):
            raise ValueError(f"Event variable '{event_variable}' must be binary (0/1)")
            
        # Create copy of data to avoid modifying original
        df = data.copy()
        
        # Store figures
        figures = {}
        
        # 1. Kaplan-Meier Analysis
        kmf = KaplanMeierFitter()
        kmf.fit(df[time_variable], df[event_variable], label='Overall')
        
        # Extract survival curve data
        survival_curve = {
            'times': kmf.survival_function_.index.tolist(),
            'survival_probs': kmf.survival_function_['Overall'].tolist(),
            'confidence_lower': kmf.confidence_interval_['Overall_lower'].tolist() if 'Overall_lower' in kmf.confidence_interval_ else kmf.confidence_interval_.iloc[:, 0].tolist(),
            'confidence_upper': kmf.confidence_interval_['Overall_upper'].tolist() if 'Overall_upper' in kmf.confidence_interval_ else kmf.confidence_interval_.iloc[:, 1].tolist(),
            'num_at_risk': kmf.event_table['at_risk'].tolist() if hasattr(kmf, 'event_table') else None
        }
        
        # Calculate median survival time and other statistics
        try:
            median_survival = float(kmf.median_survival_time_)
        except:
            median_survival = None
            
        # Get survival rates at specific timepoints
        survival_times = [1, 3, 5, 10]  # 1, 3, 5, 10 year/month survival rates
        survival_rates = {}
        
        for t in survival_times:
            try:
                if t <= max(survival_curve['times']):
                    rate = float(kmf.predict(t))
                    survival_rates[t] = rate
            except:
                # Skip timepoints beyond the data
                pass
        
        # Calculate restricted mean survival time (RMST)
        try:
            # Use the maximum follow-up time or a specified clinical timepoint
            max_time = max(survival_curve['times'])
            
            # Calculate RMST up to different time horizons
            rmst_values = {}
            for t in [max_time/4, max_time/2, max_time]:
                if t > 0:
                    rmst = kmf.restricted_mean_survival_time(t)
                    rmst_values[float(t)] = float(rmst)
        except:
            rmst_values = {"error": "Could not calculate RMST"}
        
        # Calculate cumulative hazard
        cumulative_hazard = {
            'times': kmf.survival_function_.index.tolist(),
            'hazard': (-np.log(kmf.survival_function_['Overall'])).tolist()
        }
        
        # Store overall KM results
        overall_km = {
            'survival_curve': survival_curve,
            'median_survival': median_survival,
            'survival_rates': survival_rates,
            'rmst': rmst_values,
            'cumulative_hazard': cumulative_hazard,
            'num_subjects': int(df.shape[0]),
            'num_events': int(df[event_variable].sum()),
            'event_rate': float(df[event_variable].mean()),
            'max_follow_up': float(max(df[time_variable]))
        }
        
        # Create KM plot
        try:
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
            
            # Plot the KM curve with confidence intervals
            kmf.plot_survival_function(ax=ax, ci_show=True)
            
            # Add censored markers
            kmf.plot_survival_function(ax=ax, ci_show=False, show_censors=True, 
                                     censor_styles={'ms': 6, 'marker': '+', 'color': 'black'})
            
            # Add median survival line if available
            if median_survival is not None:
                ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
                ax.axvline(x=median_survival, color='gray', linestyle='--', alpha=0.5)
                ax.text(median_survival, 0.5, f' Median: {median_survival:.2f}', 
                      verticalalignment='center')
            
            # Add number of subjects at risk
            if hasattr(kmf, 'event_table'):
                # Select timepoints for at-risk table (e.g., 0, max/4, max/2, 3*max/4, max)
                time_breaks = np.linspace(0, max(survival_curve['times']), 5)
                
                # Add table of numbers at risk
                for i, t in enumerate(time_breaks):
                    n_at_risk = kmf.event_table.at_risk[kmf.event_table.index <= t]
                    if len(n_at_risk) > 0:
                        n = n_at_risk.iloc[-1]
                        ax.text(t, -0.05, str(int(n)), transform=ax.get_xaxis_transform(), 
                              ha='center', va='top')
            
            # Add event rate in the plot
            ax.text(0.05, 0.05, f'Events: {overall_km["num_events"]}/{overall_km["num_subjects"]} ({overall_km["event_rate"]*100:.1f}%)', 
                  transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
            
            # Customize the plot
            ax.set_xlabel('Time')
            ax.set_ylabel('Survival Probability')
            ax.set_title('Kaplan-Meier Survival Curve - Overall')
            ax.grid(True, alpha=0.3)
            fig.patch.set_alpha(0)
            
            figures['km_overall'] = fig_to_svg(fig)
        except Exception as e:
            figures['km_overall_error'] = str(e)
        
        # Create cumulative hazard plot (Overall)
        try:
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
            
            # Use Nelson-Aalen Fitter to estimate cumulative hazard
            naf = NelsonAalenFitter()
            naf.fit(df[time_variable], event_observed=df[event_variable])
            naf.plot(ax=ax, ci_show=True)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Cumulative Hazard')
            ax.set_title('Nelson-Aalen Cumulative Hazard - Overall')
            ax.grid(True, alpha=0.3)
            fig.patch.set_alpha(0)
            
            figures['hazard_overall'] = fig_to_svg(fig)
        except Exception as e:
            figures['hazard_overall_error'] = str(e)
        
        # Group-specific analysis if group variable provided
        group_km = {}
        group_colors = {}
        
        if group_variable:
            groups = df[group_variable].unique()
            
            # Create a colormap for the groups using PASTEL_COLORS from imports
            group_colors = {str(group): PASTEL_COLORS[i % len(PASTEL_COLORS)] for i, group in enumerate(groups)}
            
            # For KM curve by group
            fig_group, ax_group = plt.subplots(figsize=(10, 6), facecolor='none')
            
            # For cumulative hazard by group
            fig_hazard, ax_hazard = plt.subplots(figsize=(10, 6), facecolor='none')
            
            # Store survival curves for each group
            for group in groups:
                mask = df[group_variable] == group
                group_data = df[mask]
                
                # Skip if group is empty
                if group_data.empty:
                    continue
                    
                kmf = KaplanMeierFitter()
                group_label = str(group)
                kmf.fit(group_data[time_variable], group_data[event_variable], label=group_label)
                
                # Extract survival curve data
                group_survival_curve = {
                    'times': kmf.survival_function_.index.tolist(),
                    'survival_probs': kmf.survival_function_[group_label].tolist(),
                    'confidence_lower': kmf.confidence_interval_[f'{group_label}_lower'].tolist() if f'{group_label}_lower' in kmf.confidence_interval_ else kmf.confidence_interval_.iloc[:, 0].tolist(),
                    'confidence_upper': kmf.confidence_interval_[f'{group_label}_upper'].tolist() if f'{group_label}_upper' in kmf.confidence_interval_ else kmf.confidence_interval_.iloc[:, 1].tolist(),
                    'num_at_risk': kmf.event_table['at_risk'].tolist() if hasattr(kmf, 'event_table') else None
                }
                
                # Calculate median survival time
                try:
                    group_median = float(kmf.median_survival_time_)
                except:
                    group_median = None
                    
                # Get survival rates at specific timepoints
                group_rates = {}
                for t in survival_times:
                    try:
                        if t <= max(group_survival_curve['times']):
                            rate = float(kmf.predict(t))
                            group_rates[t] = rate
                    except:
                        # Skip timepoints beyond the data
                        pass
                
                # Calculate restricted mean survival time (RMST) for the group
                try:
                    # Use the maximum follow-up time or a specified clinical timepoint
                    group_max_time = max(group_survival_curve['times'])
                    
                    # Calculate RMST up to different time horizons
                    group_rmst_values = {}
                    for t in [group_max_time/4, group_max_time/2, group_max_time]:
                        if t > 0:
                            group_rmst = kmf.restricted_mean_survival_time(t)
                            group_rmst_values[float(t)] = float(group_rmst)
                except:
                    group_rmst_values = {"error": "Could not calculate RMST"}
                
                # Calculate cumulative hazard for the group
                group_cumulative_hazard = {
                    'times': kmf.survival_function_.index.tolist(),
                    'hazard': (-np.log(kmf.survival_function_[group_label])).tolist()
                }
                
                group_km[group_label] = {
                    'survival_curve': group_survival_curve,
                    'median_survival': group_median,
                    'survival_rates': group_rates,
                    'rmst': group_rmst_values,
                    'cumulative_hazard': group_cumulative_hazard,
                    'num_subjects': int(group_data.shape[0]),
                    'num_events': int(group_data[event_variable].sum()),
                    'event_rate': float(group_data[event_variable].mean()),
                    'max_follow_up': float(max(group_data[time_variable]))
                }
                
                # Plot this group's KM curve
                color = group_colors[group_label]
                kmf.plot_survival_function(ax=ax_group, ci_show=True, color=color)
                
                # Add censored markers with matching color
                kmf.plot_survival_function(ax=ax_group, ci_show=False, show_censors=True,
                                         color=color, censor_styles={'ms': 6, 'marker': '+'})
                
                # Use Nelson-Aalen Fitter for cumulative hazard
                naf_group = NelsonAalenFitter()
                naf_group.fit(group_data[time_variable], event_observed=group_data[event_variable])
                naf_group.plot(ax=ax_hazard, ci_show=True, color=color)
            
            # Finalize group KM plot
            ax_group.set_xlabel('Time')
            ax_group.set_ylabel('Survival Probability')
            ax_group.set_title(f'Kaplan-Meier Survival Curves by {group_variable}')
            ax_group.grid(True, alpha=0.3)
            ax_group.legend(title=group_variable)
            fig_group.patch.set_alpha(0)
            
            # Add group event rates in the plot
            txt_pos = 0.05
            for group, stats in group_km.items():
                txt = f'{group}: {stats["num_events"]}/{stats["num_subjects"]} ({stats["event_rate"]*100:.1f}%)'
                ax_group.text(0.05, txt_pos, txt, transform=ax_group.transAxes, 
                            bbox=dict(facecolor='white', alpha=0.8), color=group_colors[group])
                txt_pos += 0.05
            
            # Add median survival
            for group, stats in group_km.items():
                if stats['median_survival'] is not None:
                    ax_group.axhline(y=0.5, color=group_colors[group], linestyle='--', alpha=0.3)
                    ax_group.axvline(x=stats['median_survival'], color=group_colors[group], linestyle='--', alpha=0.3)
            
            figures['km_by_group'] = fig_to_svg(fig_group)
            
            # Finalize group hazard plot
            ax_hazard.set_xlabel('Time')
            ax_hazard.set_ylabel('Cumulative Hazard')
            ax_hazard.set_title(f'Nelson-Aalen Cumulative Hazard by {group_variable}')
            ax_hazard.grid(True, alpha=0.3)
            ax_hazard.legend(title=group_variable)
            fig_hazard.patch.set_alpha(0)
            
            figures['hazard_by_group'] = fig_to_svg(fig_hazard)
            
            # 2. Log-rank test for comparing groups
            # Only do test if we have 2+ groups
            if len(groups) >= 2:
                # For pairwise comparisons if more than 2 groups
                pairwise_results = []
                
                if len(groups) > 2:
                    # Perform pairwise log-rank tests
                    for i, group1 in enumerate(groups):
                        for group2 in groups[i+1:]:
                            # Get data for the two groups
                            mask1 = df[group_variable] == group1
                            mask2 = df[group_variable] == group2
                            
                            # Skip if either group is empty
                            if not mask1.any() or not mask2.any():
                                continue
                                
                            times1 = df.loc[mask1, time_variable]
                            events1 = df.loc[mask1, event_variable]
                            times2 = df.loc[mask2, time_variable]
                            events2 = df.loc[mask2, event_variable]
                            
                            # Perform log-rank test
                            results = logrank_test(times1, times2, events1, events2)
                            
                            pairwise_results.append({
                                'group1': str(group1),
                                'group2': str(group2),
                                'test_statistic': float(results.test_statistic),
                                'p_value': float(results.p_value),
                                'significant': results.p_value < alpha
                            })
                
                # Perform multivariate log-rank test
                try:
                    # Create groups indicator
                    data_with_group = df[[time_variable, event_variable, group_variable]].dropna()
                    
                    # Check if we have valid data
                    if len(data_with_group) > 0 and len(groups) > 1:
                        # Perform multivariate log-rank test
                        mv_results = multivariate_logrank_test(
                            data_with_group[time_variable],
                            data_with_group[group_variable],
                            data_with_group[event_variable]
                        )
                        
                        logrank_result = {
                            'test_statistic': float(mv_results.test_statistic),
                            'p_value': float(mv_results.p_value),
                            'significant': mv_results.p_value < alpha,
                            'pairwise_comparisons': pairwise_results if pairwise_results else None
                        }
                    else:
                        logrank_result = {
                            'error': 'Insufficient data for multivariate log-rank test'
                        }
                except Exception as e:
                    # Fall back to regular log-rank test for two groups
                    if len(groups) == 2:
                        # Get data for the two groups
                        mask1 = df[group_variable] == groups[0]
                        mask2 = df[group_variable] == groups[1]
                        
                        # Skip if either group is empty
                        if not mask1.any() or not mask2.any():
                            logrank_result = {
                                'error': 'Empty groups for log-rank test'
                            }
                        else:
                            times1 = df.loc[mask1, time_variable]
                            events1 = df.loc[mask1, event_variable]
                            times2 = df.loc[mask2, time_variable]
                            events2 = df.loc[mask2, event_variable]
                            
                            # Perform log-rank test
                            results = logrank_test(times1, times2, events1, events2)
                            
                            logrank_result = {
                                'test_statistic': float(results.test_statistic),
                                'p_value': float(results.p_value),
                                'significant': results.p_value < alpha,
                                'pairwise_comparisons': None
                            }
                    else:
                        logrank_result = {
                            'error': f'Error in multivariate log-rank test: {str(e)}'
                        }
                
                # Create pairwise comparison matrix visualization if we have pairwise results
                if pairwise_results:
                    try:
                        # Create a matrix of p-values
                        group_list = sorted(list(groups))
                        n_groups = len(group_list)
                        p_matrix = np.ones((n_groups, n_groups))
                        
                        # Fill the matrix with p-values
                        for result in pairwise_results:
                            idx1 = group_list.index(result['group1'])
                            idx2 = group_list.index(result['group2'])
                            p_matrix[idx1, idx2] = result['p_value']
                            p_matrix[idx2, idx1] = result['p_value']  # Mirror for heatmap
                        
                        # Create the heatmap
                        fig, ax = plt.subplots(figsize=(8, 6), facecolor='none')
                        mask = np.triu(np.ones_like(p_matrix, dtype=bool))  # Mask upper triangle
                        
                        # Create heatmap
                        sns.heatmap(p_matrix, mask=mask, annot=True, fmt=".4f", cmap=PASTEL_CMAP,
                                  vmin=0, vmax=alpha*2, cbar_kws={'label': 'p-value'},
                                  linewidths=1, ax=ax)
                        
                        # Add significance indicators
                        for i in range(n_groups):
                            for j in range(n_groups):
                                if i > j:  # Lower triangle only
                                    sig = p_matrix[i, j] < alpha
                                    if sig:
                                        ax.text(j+0.5, i+0.85, "*", ha="center", va="center", 
                                              color="white", fontweight="bold", fontsize=16)
                        
                        ax.set_title('Pairwise Log-Rank Test P-Values')
                        ax.set_xlabel(group_variable)
                        ax.set_ylabel(group_variable)
                        
                        # Adjust tick labels
                        ax.set_xticks(np.arange(n_groups) + 0.5)
                        ax.set_yticks(np.arange(n_groups) + 0.5)
                        ax.set_xticklabels(group_list)
                        ax.set_yticklabels(group_list)
                        
                        # Add note about significance
                        plt.figtext(0.5, 0.01, f"* indicates p < {alpha}", ha="center", fontsize=10)
                        
                        fig.patch.set_alpha(0)
                        
                        figures['logrank_pairwise'] = fig_to_svg(fig)
                    except Exception as e:
                        figures['logrank_pairwise_error'] = str(e)
            else:
                logrank_result = {
                    'error': 'Need at least 2 groups for log-rank test'
                }
        else:
            logrank_result = None
        
        # 3. Cox Proportional Hazards Model
        cox_result = {}
        
        if group_variable:
            # Create formula for Cox model
            try:
                cph = CoxPHFitter()
                
                # If group variable is categorical, create dummy variables
                if not pd.api.types.is_numeric_dtype(df[group_variable]):
                    # One-hot encode the group variable
                    group_dummies = pd.get_dummies(df[group_variable], prefix=group_variable, drop_first=True)
                    model_df = pd.concat([df, group_dummies], axis=1)
                    
                    # Get dummy column names
                    dummy_cols = group_dummies.columns.tolist()
                    
                    # Fit the model with dummy variables
                    cph.fit(model_df, duration_col=time_variable, event_col=event_variable, formula=dummy_cols)
                else:
                    # If numeric, use directly
                    cph.fit(df, duration_col=time_variable, event_col=event_variable, formula=group_variable)
                
                # Extract coefficients
                coefficients = {}
                for term in cph.params_.index:
                    coefficients[term] = {
                        'estimate': float(cph.params_[term]),
                        'hazard_ratio': float(np.exp(cph.params_[term])),
                        'std_error': float(cph.standard_errors_[term]),
                        'p_value': float(cph.pvalues[term]),
                        'significant': cph.pvalues[term] < alpha,
                        'ci_lower': float(np.exp(cph.params_[term] - 1.96 * cph.standard_errors_[term])),
                        'ci_upper': float(np.exp(cph.params_[term] + 1.96 * cph.standard_errors_[term]))
                    }
                
                # Model fit statistics
                model_fit = {
                    'log_likelihood': float(cph.log_likelihood_),
                    'concordance': float(cph.concordance_index_),
                    'aic': float(cph.AIC_),
                    'bic': float(cph.BIC_) if hasattr(cph, 'BIC_') else None
                }
                
                # Proportional hazards assumption test
                try:
                    # Perform proportional hazards test
                    ph_test = proportional_hazard_test(cph, df, time_col=time_variable, event_col=event_variable)
                    
                    # Extract results
                    ph_test_result = {
                        'global_p_value': float(ph_test.p_value_for_omnibus_test),
                        'global_satisfied': ph_test.p_value_for_omnibus_test > alpha,
                        'individual_p_values': {}
                    }
                    
                    # Extract individual variable test results
                    for var in ph_test.table.index:
                        p_value = float(ph_test.table.loc[var, 'p'])
                        ph_test_result['individual_p_values'][var] = {
                            'p_value': p_value,
                            'satisfied': p_value > alpha
                        }
                except Exception as ph_error:
                    ph_test_result = {
                        'error': f'Could not perform proportional hazards test: {str(ph_error)}'
                    }
                
                # Create Schoenfeld residual plots for PH assumption
                try:
                    # Get Schoenfeld residuals
                    schoenfeld_resids = cph.compute_residuals(df, 'schoenfeld')
                    
                    # Create plots for each covariate
                    for term in schoenfeld_resids.columns:
                        fig, ax = plt.subplots(figsize=(8, 5), facecolor='none')
                        
                        # Get the times for events (not censored observations)
                        event_times = df.loc[df[event_variable] == 1, time_variable].values
                        
                        # Get corresponding Schoenfeld residuals
                        # This assumes residuals are in same order as original data for events
                        schoen_resids = schoenfeld_resids[term].dropna().values
                        
                        # Plot the residuals
                        ax.scatter(event_times, schoen_resids, alpha=0.6)
                        
                        # Add smoothed line
                        try:
                            from statsmodels.nonparametric.smoothers_lowess import lowess
                            smooth = lowess(schoen_resids, event_times, frac=0.6)
                            ax.plot(smooth[:, 0], smooth[:, 1], 'r-', linewidth=2)
                        except:
                            # If lowess fails, use basic moving average
                            from scipy.ndimage import gaussian_filter1d
                            # Sort by time
                            sort_idx = np.argsort(event_times)
                            sorted_times = event_times[sort_idx]
                            sorted_resids = schoen_resids[sort_idx]
                            # Apply smoothing
                            smoothed = gaussian_filter1d(sorted_resids, sigma=3)
                            ax.plot(sorted_times, smoothed, 'r-', linewidth=2)
                        
                        # Add horizontal reference line at y=0
                        ax.axhline(y=0, color='blue', linestyle='--')
                        
                        # Add p-value from PH test if available
                        if term in ph_test_result['individual_p_values']:
                            p_val = ph_test_result['individual_p_values'][term]['p_value']
                            ax.text(0.05, 0.95, f"PH test p-value: {p_val:.4f}", transform=ax.transAxes,
                                  va='top', ha='left', bbox=dict(facecolor='white', alpha=0.8))
                        
                        ax.set_xlabel('Time')
                        ax.set_ylabel(f'Schoenfeld Residual for {term}')
                        ax.set_title(f'Schoenfeld Residuals for {term}')
                        ax.grid(True, alpha=0.3)
                        
                        fig.patch.set_alpha(0)
                        
                        figures[f'schoenfeld_{term}'] = fig_to_svg(fig)
                except Exception as e:
                    figures['schoenfeld_error'] = str(e)
                
                # Create forest plot of hazard ratios
                try:
                    # Extract hazard ratios and confidence intervals
                    terms = []
                    hrs = []
                    lower_cis = []
                    upper_cis = []
                    p_values = []
                    
                    for term, coef in coefficients.items():
                        terms.append(term)
                        hrs.append(coef['hazard_ratio'])
                        lower_cis.append(coef['ci_lower'])
                        upper_cis.append(coef['ci_upper'])
                        p_values.append(coef['p_value'])
                    
                    # Convert to numpy arrays for calculations
                    hrs = np.array(hrs)
                    lower_cis = np.array(lower_cis)
                    upper_cis = np.array(upper_cis)
                    p_values = np.array(p_values)
                    
                    # Create the plot
                    fig, ax = plt.subplots(figsize=(10, max(6, len(terms) * 0.5)), facecolor='none')
                    
                    # Plot the hazard ratios and confidence intervals
                    y_pos = np.arange(len(terms))
                    
                    # Calculate error bar sizes
                    err_minus = hrs - lower_cis
                    err_plus = upper_cis - hrs
                    
                    # Set colors based on significance
                    colors = ['tab:blue' if p < alpha else 'tab:gray' for p in p_values]
                    
                    # Plot error bars
                    ax.errorbar(hrs, y_pos, xerr=[err_minus, err_plus], fmt='none', ecolor='black')
                    
                    # Plot points with colors from PASTEL_COLORS
                    for i, (hr, y, p) in enumerate(zip(hrs, y_pos, p_values)):
                        color = PASTEL_COLORS[i % len(PASTEL_COLORS)] if p < alpha else 'lightgray'
                        ax.scatter(hr, y, color=color, s=100, zorder=10)
                    
                    # Add vertical line at HR=1 (no effect)
                    ax.axvline(x=1, color='red', linestyle='--', alpha=0.7)
                    
                    # Set axis labels
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(terms)
                    ax.set_xlabel('Hazard Ratio (log scale)')
                    ax.set_title('Cox PH Model: Hazard Ratios with 95% CI')
                    
                    # Add HR values as text
                    for i, (hr, p) in enumerate(zip(hrs, p_values)):
                        sig_str = '*' if p < alpha else ''
                        ax.text(hr * 1.05, i, f" {hr:.3f} {sig_str}", va='center')
                    
                    # Set logarithmic scale for x-axis
                    ax.set_xscale('log')
                    
                    # Set reasonable limits based on the data
                    min_ci = min(lower_cis)
                    max_ci = max(upper_cis)
                    margin = 0.5  # Log scale margin
                    ax.set_xlim(min_ci / margin, max_ci * margin)
                    
                    # Add legend for significance
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], marker='o', color='w', markerfacecolor=PASTEL_COLORS[0], markersize=10, label='Significant'),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=10, label='Non-significant')
                    ]
                    ax.legend(handles=legend_elements, loc='upper right')
                    
                    # Add grid
                    ax.grid(True, alpha=0.3)
                    
                    fig.patch.set_alpha(0)
                    
                    figures['cox_forest_plot'] = fig_to_svg(fig)
                except Exception as e:
                    figures['cox_forest_plot_error'] = str(e)
                
                # Create predicted survival curves based on model
                try:
                    if not pd.api.types.is_numeric_dtype(df[group_variable]):
                        fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
                        
                        # Plot baseline survival curve
                        times = np.linspace(0, df[time_variable].max(), 100)
                        ax.plot(times, cph.baseline_survival_at_times(times), 
                              label='Baseline', linestyle='--', color='black')
                        
                        # Plot survival curves for each group
                        for group in groups:
                            # Create a representative individual for this group
                            if group == groups[0]:  # Reference group (drop_first=True)
                                x = pd.DataFrame({col: [0] for col in dummy_cols})
                            else:
                                # Set the dummy variable for this group to 1
                                x = pd.DataFrame({col: [1 if group_variable in col and str(group) in col else 0] 
                                               for col in dummy_cols})
                            
                            # Predict survival curve
                            survival_curve = cph.predict_survival_function(x).T
                            
                            # Plot the predicted curve using PASTEL_COLORS
                            color = group_colors.get(str(group), PASTEL_COLORS[groups.tolist().index(group) % len(PASTEL_COLORS)])
                            ax.plot(survival_curve.index, survival_curve.iloc[:, 0], 
                                  label=f'{group_variable}={group}', color=color)
                        
                        ax.set_xlabel('Time')
                        ax.set_ylabel('Predicted Survival Probability')
                        ax.set_title('Predicted Survival Curves from Cox PH Model')
                        ax.grid(True, alpha=0.3)
                        ax.legend(title=group_variable)
                        
                        fig.patch.set_alpha(0)
                        
                        figures['cox_predicted_survival'] = fig_to_svg(fig)
                except Exception as e:
                    figures['cox_predicted_survival_error'] = str(e)
                
                cox_result = {
                    'coefficients': coefficients,
                    'model_fit': model_fit,
                    'ph_test': ph_test_result,
                    'summary': str(cph.summary),
                    'formula': str(cph.formula) if hasattr(cph, 'formula') else None
                }
            except Exception as e:
                cox_result = {
                    'error': f'Error in Cox regression: {str(e)}'
                }
        
        # 4. Additional visualizations
        
        # Create lifetimes plot
        try:
            fig, ax = plt.subplots(figsize=(10, 8), facecolor='none')
            
            # Determine number of subjects to plot (limit to avoid overcrowding)
            max_subjects = min(100, df.shape[0])
            
            # If we have groups, stratify the selection
            if group_variable:
                # Select a proportional number of subjects from each group
                subjects_to_plot = pd.DataFrame()
                for group in groups:
                    group_data = df[df[group_variable] == group]
                    n_to_select = max(1, int(max_subjects * len(group_data) / len(df)))
                    
                    if len(group_data) > n_to_select:
                        # Select random subjects from this group
                        selected = group_data.sample(n=n_to_select, random_state=42)
                    else:
                        # Use all subjects from this group
                        selected = group_data
                    
                    subjects_to_plot = pd.concat([subjects_to_plot, selected])
                
                # Plot lifetimes with group coloring
                plot_lifetimes(subjects_to_plot[time_variable], 
                             event_observed=subjects_to_plot[event_variable],
                             groups=subjects_to_plot[group_variable], ax=ax)
            else:
                # Select random subjects if needed
                if len(df) > max_subjects:
                    subjects_to_plot = df.sample(n=max_subjects, random_state=42)
                else:
                    subjects_to_plot = df
                
                # Plot lifetimes without grouping
                plot_lifetimes(subjects_to_plot[time_variable], 
                             event_observed=subjects_to_plot[event_variable], ax=ax)
            
            # Customize plot
            ax.set_xlabel('Time')
            ax.set_title('Subject Lifetimes')
            
            # Add legend if groups are used
            if group_variable:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, title=group_variable)
            
            fig.patch.set_alpha(0)
            
            figures['lifetimes_plot'] = fig_to_svg(fig)
        except Exception as e:
            figures['lifetimes_plot_error'] = str(e)
        
        # Create survival rates comparison plot
        if group_variable and group_km:
            try:
                # Get standardized time points to compare
                common_times = [1, 3, 5]
                # Filter to times that exist in the data
                common_times = [t for t in common_times if t <= max(df[time_variable])]
                
                if common_times:
                    fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
                    
                    # Set up data for bar chart
                    group_names = list(group_km.keys())
                    x = np.arange(len(group_names))
                    width = 0.8 / len(common_times)  # Width of each bar group
                    
                    # Plot bars for each time point
                    for i, t in enumerate(common_times):
                        rates = []
                        for group in group_names:
                            # Get rate at this time point if available
                            if t in group_km[group]['survival_rates']:
                                rates.append(group_km[group]['survival_rates'][t])
                            else:
                                # Try to interpolate
                                try:
                                    times = group_km[group]['survival_curve']['times']
                                    probs = group_km[group]['survival_curve']['survival_probs']
                                    
                                    if t <= max(times):
                                        # Find closest time point
                                        idx = np.searchsorted(times, t)
                                        if idx == 0:
                                            rate = probs[0]
                                        elif idx == len(times):
                                            rate = probs[-1]
                                        else:
                                            # Linear interpolation
                                            t1, t2 = times[idx-1], times[idx]
                                            p1, p2 = probs[idx-1], probs[idx]
                                            rate = p1 + (p2 - p1) * (t - t1) / (t2 - t1)
                                        rates.append(rate)
                                    else:
                                        rates.append(None)
                                except:
                                    rates.append(None)
                        
                        # Filter out None values
                        valid_rates = [(j, r) for j, r in enumerate(rates) if r is not None]
                        if valid_rates:
                            valid_idx, valid_rates = zip(*valid_rates)
                            
                            # Plot bars with pastel colors
                            offset = width * (i - len(common_times)/2 + 0.5)
                            ax.bar(np.array(valid_idx) + offset, valid_rates, width, 
                                 label=f't={t}', alpha=0.7, 
                                 color=[PASTEL_COLORS[j % len(PASTEL_COLORS)] for j in valid_idx])
                    
                    # Customize plot
                    ax.set_ylabel('Survival Probability')
                    ax.set_title(f'Survival Rates by {group_variable} at Different Time Points')
                    ax.set_xticks(x)
                    ax.set_xticklabels(group_names)
                    ax.set_ylim(0, 1.1)
                    ax.legend(title='Time')
                    
                    # Add value labels on bars
                    for c in ax.containers:
                        ax.bar_label(c, fmt='%.2f', padding=3)
                    
                    fig.patch.set_alpha(0)
                    
                    figures['survival_rates_comparison'] = fig_to_svg(fig)
            except Exception as e:
                figures['survival_rates_comparison_error'] = str(e)
        
        # Create stratified risk table plot with KM curve
        if group_variable and group_km:
            try:
                fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, facecolor='none')
                ax_km = axes[0]
                ax_risk = axes[1]
                
                # Plot KM curves for each group
                for group, stats in group_km.items():
                    color = group_colors.get(group, PASTEL_COLORS[list(group_km.keys()).index(group) % len(PASTEL_COLORS)])
                    times = stats['survival_curve']['times']
                    probs = stats['survival_curve']['survival_probs']
                    
                    # Plot the KM curve
                    ax_km.step(times, probs, where='post', label=group, color=color)
                    
                    # Add confidence intervals (with lighter color)
                    if 'confidence_lower' in stats['survival_curve'] and 'confidence_upper' in stats['survival_curve']:
                        ci_lower = stats['survival_curve']['confidence_lower']
                        ci_upper = stats['survival_curve']['confidence_upper']
                        ax_km.fill_between(times, ci_lower, ci_upper, alpha=0.2, step='post', color=color)
                
                # Customize the KM plot
                ax_km.set_xlabel('')  # No label, as it's shared with risk table
                ax_km.set_ylabel('Survival Probability')
                ax_km.set_title('Kaplan-Meier Survival Curves with Number at Risk')
                ax_km.grid(True, alpha=0.3)
                ax_km.legend(title=group_variable)
                
                # Create risk table
                # Set up time points for risk table (e.g., quartiles of max follow-up)
                max_time = max(df[time_variable])
                time_points = np.linspace(0, max_time, 5)
                
                # For each group, calculate number at risk at each time point
                for i, group in enumerate(group_km.keys()):
                    # Get event table for this group if available
                    if 'num_at_risk' in group_km[group]['survival_curve'] and group_km[group]['survival_curve']['num_at_risk'] is not None:
                        times = group_km[group]['survival_curve']['times']
                        at_risk = group_km[group]['survival_curve']['num_at_risk']
                        
                        # For each time point, find closest time in KM table
                        risk_counts = []
                        for t in time_points:
                            if t <= max(times):
                                idx = np.searchsorted(times, t)
                                if idx == 0:
                                    risk_counts.append(at_risk[0])
                                elif idx >= len(at_risk):
                                    risk_counts.append(at_risk[-1])
                                else:
                                    risk_counts.append(at_risk[idx-1])  # Use previous time point's count
                            else:
                                risk_counts.append(0)
                        
                        # Plot risk counts
                        color = group_colors.get(group, PASTEL_COLORS[list(group_km.keys()).index(group) % len(PASTEL_COLORS)])
                        ax_risk.plot(time_points, risk_counts, 'o-', color=color, label=group)
                        
                        # Add text labels for counts
                        for j, (t, count) in enumerate(zip(time_points, risk_counts)):
                            ax_risk.text(t, i+0.1, str(int(count)), ha='center', color=color)
                
                # Customize risk table
                ax_risk.set_xlabel('Time')
                ax_risk.set_ylabel('At Risk')
                ax_risk.set_yticks([])  # Hide y-ticks
                ax_risk.grid(True, alpha=0.3, axis='x')
                
                # Share x-axis limits between plots
                ax_risk.set_xlim(ax_km.get_xlim())
                
                # Adjust layout
                fig.tight_layout()
                fig.patch.set_alpha(0)
                
                figures['km_with_risk_table'] = fig_to_svg(fig)
            except Exception as e:
                figures['km_with_risk_table_error'] = str(e)
        
        # Create martingale residuals plot for Cox model
        if 'coefficients' in cox_result:
            try:
                # Calculate martingale residuals
                if group_variable and not pd.api.types.is_numeric_dtype(df[group_variable]):
                    # For categorical group variables, we need the dummy variables
                    group_dummies = pd.get_dummies(df[group_variable], prefix=group_variable, drop_first=True)
                    model_df = pd.concat([df, group_dummies], axis=1)
                    
                    # Get martingale residuals
                    martingale_residuals = cph.compute_residuals(model_df, 'martingale')
                else:
                    # For numeric group variables
                    martingale_residuals = cph.compute_residuals(df, 'martingale')
                
                # Create scatter plot of residuals vs. predicted values
                fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
                
                # Get the linear predictor values
                if group_variable and not pd.api.types.is_numeric_dtype(df[group_variable]):
                    linear_pred = cph.predict_partial_hazard(model_df)
                else:
                    linear_pred = cph.predict_partial_hazard(df)
                
                # Create scatter plot with PASTEL_COLORS
                ax.scatter(linear_pred, martingale_residuals, alpha=0.6, edgecolor='k', 
                         color=PASTEL_COLORS[0])
                
                # Add reference line at y=0
                ax.axhline(y=0, color='red', linestyle='--')
                
                # Add smoother to show trend
                try:
                    from statsmodels.nonparametric.smoothers_lowess import lowess
                    smooth = lowess(martingale_residuals, linear_pred, frac=0.6)
                    ax.plot(smooth[:, 0], smooth[:, 1], 'r-', linewidth=2)
                except:
                    # Skip smoother if it fails
                    pass
                
                # Customize plot
                ax.set_xlabel('Linear Predictor (log hazard)')
                ax.set_ylabel('Martingale Residual')
                ax.set_title('Martingale Residuals vs. Linear Predictor')
                ax.grid(True, alpha=0.3)
                
                fig.patch.set_alpha(0)
                
                figures['martingale_residuals'] = fig_to_svg(fig)
            except Exception as e:
                figures['martingale_residuals_error'] = str(e)
        
        # Initialize the assumptions dictionary
        assumptions = {}
        
        # 1. Sample size test for each group
        if group_variable:
            for group in groups:
                mask = df[group_variable] == group
                group_data = df[mask]
                
                group_label = str(group)
                # Create test instance first, then run the test
                sample_size_test = SampleSizeTest()
                sample_size_result = sample_size_test.run_test(
                    # Pass as pandas Series, not numpy array
                    data=group_data[time_variable],
                    min_recommended=30  # common minimum for survival analysis
                )
                
                assumptions[f'sample_size_{group_label}'] = sample_size_result
        else:
            # Overall sample size test
            sample_size_test = SampleSizeTest()
            sample_size_result = sample_size_test.run_test(
                # Pass as pandas Series, not numpy array
                data=df[time_variable],
                min_recommended=30
            )
            assumptions['sample_size'] = sample_size_result
        
        # 2. Proportional hazards test (only if Cox model was fitted)
        if 'coefficients' in cox_result:
            try:
                ph_test = ProportionalHazardsTest()
                
                # Prepare covariates dataframe
                if group_variable and pd.api.types.is_numeric_dtype(df[group_variable]):
                    covariates_df = df[[group_variable]]
                else:
                    # For categorical, we need the dummy variables
                    if group_variable:
                        covariates_df = pd.get_dummies(df[group_variable], prefix=group_variable, drop_first=True)
                    else:
                        covariates_df = None
                
                if covariates_df is not None:
                    ph_test_result = ph_test.run_test(
                        # Pass pandas Series/DataFrame, not numpy arrays
                        time=df[time_variable],
                        event=df[event_variable],
                        covariates=covariates_df
                    )
                    
                    assumptions['proportional_hazards'] = ph_test_result
                    
                    # For each covariate if applicable
                    if 'ph_test' in cox_result and 'individual_p_values' in cox_result['ph_test']:
                        for var, test in cox_result['ph_test']['individual_p_values'].items():
                            var_satisfied = test.get('satisfied', True)
                            assumptions[f'proportional_hazards_{var}'] = (
                                AssumptionResult.PASSED if var_satisfied 
                                else AssumptionResult.FAILED
                            )
            except Exception as e:
                assumptions['proportional_hazards']['result'] = AssumptionResult.NOT_APPLICABLE
        
        # 3. Outlier test for each group's survival times
        if group_variable:
            for group in groups:
                mask = df[group_variable] == group
                group_data = df[mask]
                
                group_label = str(group)
                outlier_test = OutlierTest()
                outlier_result = outlier_test.run_test(
                    # Pass pandas Series, not numpy array
                    data=group_data[time_variable]
                )
                
                assumptions[f'outliers_{group_label}'] = outlier_result
        else:
            # Overall outlier test
            outlier_test = OutlierTest()
            outlier_result = outlier_test.run_test(
                # Pass pandas Series, not numpy array
                data=df[time_variable]
            )
            assumptions['outliers'] = outlier_result
        
        # 4. Influential points test (if Cox model was fitted)
        if 'coefficients' in cox_result:
            try:
                # Calculate residuals and leverage if not done yet
                if group_variable and not pd.api.types.is_numeric_dtype(df[group_variable]):
                    # For categorical variables, create dummy variables
                    group_dummies = pd.get_dummies(df[group_variable], prefix=group_variable, drop_first=True)
                    model_df = pd.concat([df, group_dummies], axis=1)
                    
                    # Get residuals
                    martingale_residuals = cph.compute_residuals(model_df, 'martingale')
                    # Compute leverage metrics (approximation for Cox model)
                    X = group_dummies
                    leverage = pd.Series(np.zeros(len(df)), index=df.index)  # As pandas Series
                    fitted = cph.predict_partial_hazard(model_df)
                else:
                    # For numeric group variables
                    martingale_residuals = cph.compute_residuals(df, 'martingale')
                    X = df[[group_variable]] if group_variable else pd.DataFrame(index=df.index)
                    leverage = pd.Series(np.zeros(len(df)), index=df.index)  # As pandas Series
                    fitted = cph.predict_partial_hazard(df)
                
                influential_test = InfluentialPointsTest()
                influential_result = influential_test.run_test(
                    # Keep all as pandas objects
                    residuals=martingale_residuals,
                    leverage=leverage,
                    fitted=fitted,
                    X=X
                )
                
                assumptions['influential_points'] = influential_result
            except Exception as e:
                assumptions['influential_points']['result'] = AssumptionResult.NOT_APPLICABLE
        
        # 5. Autocorrelation in residuals (for time-ordered data)
        if 'coefficients' in cox_result:
            try:
                autocorr_test = AutocorrelationTest()
                autocorr_result = autocorr_test.run_test(
                    # Pass pandas Series, not numpy array
                    residuals=martingale_residuals
                )
                
                assumptions['autocorrelation'] = autocorr_result
            except Exception as e:
                assumptions['autocorrelation']['result'] = AssumptionResult.NOT_APPLICABLE
        
        # 6. Linearity test (if numeric predictors)
        if 'coefficients' in cox_result and group_variable and pd.api.types.is_numeric_dtype(df[group_variable]):
            try:
                linearity_test = LinearityTest()
                linearity_result = linearity_test.run_test(
                    # Pass pandas Series, not numpy arrays
                    x=df[group_variable],
                    y=martingale_residuals
                )
                
                assumptions['linearity'] = linearity_result
            except Exception as e:
                assumptions['linearity']['result'] = AssumptionResult.NOT_APPLICABLE
        
        # 7. Normality test on residuals if Cox model was fitted
        if 'coefficients' in cox_result:
            try:
                normality_test = NormalityTest()
                normality_result = normality_test.run_test(
                    # Pass pandas Series, not numpy array
                    data=martingale_residuals
                )
                
                assumptions['residual_normality'] = normality_result
            except Exception as e:
                assumptions['residual_normality']['result'] = AssumptionResult.NOT_APPLICABLE
        
        # 8. Homogeneity of variance across groups
        if group_variable:
            homogeneity_test = HomogeneityOfVarianceTest()
            homogeneity_result = homogeneity_test.run_test(
                # Pass pandas Series, not numpy arrays
                data=df[time_variable],
                groups=df[group_variable]
            )
            
            assumptions['homogeneity'] = homogeneity_result
        
        # Create assumption check summary visualization
        try:
            # Collect all assumption checks
            assumption_names = []
            assumption_satisfied = []
            assumption_details = []
            
            # PH Assumption
            if 'proportional_hazards' in assumptions and 'global_satisfied' in assumptions['proportional_hazards']:
                assumption_names.append('Proportional Hazards')
                assumption_satisfied.append(assumptions['proportional_hazards']['global_satisfied'])
                p_val = assumptions['proportional_hazards']['global_p_value']
                assumption_details.append(f"p = {p_val:.4f}")
            
            # Influential Observations
            if 'influential_points' in assumptions and 'result' in assumptions['influential_points']:
                assumption_names.append('No Influential Obs')
                assumption_satisfied.append(assumptions['influential_points']['result'] == AssumptionResult.PASSED)
                n_extreme = assumptions['influential_points'].get('num_extreme', 0)
                assumption_details.append(f"{n_extreme} extreme")
            
            # Non-linearity (if applicable)
            if 'linearity' in assumptions and 'result' in assumptions['linearity']:
                assumption_names.append('Linearity')
                assumption_satisfied.append(assumptions['linearity']['result'] == AssumptionResult.PASSED)
                r2 = assumptions['linearity'].get('r_squared', 0)
                assumption_details.append(f"R² = {r2:.3f}")
            
            # Only create plot if we have assumptions to check
            if assumption_names:
                fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
                
                # Create horizontal bar chart with colors from PASTEL_COLORS
                y_pos = np.arange(len(assumption_names))
                colors = [PASTEL_COLORS[0] if sat else PASTEL_COLORS[1] for sat in assumption_satisfied]
                
                ax.barh(y_pos, [1] * len(assumption_names), color=colors, alpha=0.6)
                
                # Add assumption names and details
                for i, (name, detail, sat) in enumerate(zip(assumption_names, assumption_details, assumption_satisfied)):
                    status = "✓" if sat else "✗"
                    ax.text(0.05, i, f"{status} {name}", va='center', ha='left', fontsize=12, 
                          color='black', fontweight='bold')
                    if detail:
                        ax.text(0.7, i, detail, va='center', ha='left', fontsize=10, color='black')
                
                # Set up axes
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.5, len(assumption_names) - 0.5)
                
                # Add title
                ax.set_title('Assumption Checks Summary')
                
                fig.tight_layout()
                fig.patch.set_alpha(0)
                
                figures['assumption_summary'] = fig_to_svg(fig)
        except Exception as e:
            figures['assumption_summary_error'] = str(e)
        
        # Create interpretation
        interpretation = f"Survival analysis with time variable '{time_variable}' and event indicator '{event_variable}'.\n\n"
        
        # Overall survival
        interpretation += "**Overall Survival Statistics:**\n"
        interpretation += f"- Number of subjects: {overall_km['num_subjects']}\n"
        interpretation += f"- Number of events: {overall_km['num_events']} ({overall_km['event_rate']*100:.1f}%)\n"
        interpretation += f"- Maximum follow-up time: {overall_km['max_follow_up']:.2f}\n"
        
        if overall_km['median_survival'] is not None:
            interpretation += f"- Median survival time: {overall_km['median_survival']:.2f}\n"
        else:
            interpretation += "- Median survival time: Not reached\n"
        
        # Add survival rates at specific timepoints
        if overall_km['survival_rates']:
            interpretation += "- Survival rates:\n"
            for time, rate in overall_km['survival_rates'].items():
                interpretation += f"  - {time}-unit survival: {rate*100:.1f}%\n"
        
        # Add RMST information if available
        if isinstance(overall_km['rmst'], dict) and not 'error' in overall_km['rmst']:
            interpretation += "- Restricted mean survival time (RMST):\n"
            for time, rmst_val in overall_km['rmst'].items():
                interpretation += f"  - Up to t={time:.1f}: {rmst_val:.2f} units\n"
        
        # Group-specific results if available
        if group_variable:
            interpretation += f"\n**Group-specific Analysis by '{group_variable}':**\n"
            
            for group, stats in group_km.items():
                interpretation += f"\nGroup '{group}':\n"
                interpretation += f"- Number of subjects: {stats['num_subjects']}\n"
                interpretation += f"- Number of events: {stats['num_events']} ({stats['event_rate']*100:.1f}%)\n"
                
                if stats['median_survival'] is not None:
                    interpretation += f"- Median survival time: {stats['median_survival']:.2f}\n"
                else:
                    interpretation += "- Median survival time: Not reached\n"
                
                # Add survival rates at specific timepoints
                if stats['survival_rates']:
                    interpretation += "- Survival rates:\n"
                    for time, rate in stats['survival_rates'].items():
                        interpretation += f"  - {time}-unit survival: {rate*100:.1f}%\n"
            
            # Log-rank test results
            if logrank_result:
                if 'test_statistic' in logrank_result:
                    interpretation += f"\n**Log-rank Test Comparing Groups:** "
                    if logrank_result['significant']:
                        interpretation += f"Significant difference found (χ² = {logrank_result['test_statistic']:.3f}, p = {logrank_result['p_value']:.5f}).\n"
                    else:
                        interpretation += f"No significant difference found (χ² = {logrank_result['test_statistic']:.3f}, p = {logrank_result['p_value']:.5f}).\n"
                    
                    # Add information about pairwise comparisons if available
                    if 'pairwise_comparisons' in logrank_result and logrank_result['pairwise_comparisons']:
                        interpretation += "\nPairwise comparisons:\n"
                        for comp in logrank_result['pairwise_comparisons']:
                            interpretation += f"- {comp['group1']} vs {comp['group2']}: "
                            if comp['significant']:
                                interpretation += f"Significant difference (p = {comp['p_value']:.5f})\n"
                            else:
                                interpretation += f"No significant difference (p = {comp['p_value']:.5f})\n"
            
            # Cox proportional hazards results
            if 'coefficients' in cox_result:
                interpretation += "\n**Cox Proportional Hazards Model Results:**\n"
                
                # Interpret coefficients
                significant_terms = [term for term, coef in cox_result['coefficients'].items() if coef.get('significant', False)]
                
                if significant_terms:
                    interpretation += "Significant predictors:\n"
                    for term in significant_terms:
                        coef = cox_result['coefficients'][term]
                        interpretation += f"- {term}: HR = {coef['hazard_ratio']:.3f} "
                        interpretation += f"(95% CI: {coef['ci_lower']:.3f}-{coef['ci_upper']:.3f}), p = {coef['p_value']:.5f}\n"
                        
                        # Interpret hazard ratio
                        if coef['hazard_ratio'] > 1:
                            interpretation += f"  {term} increases the hazard by {(coef['hazard_ratio']-1)*100:.1f}%.\n"
                        else:
                            interpretation += f"  {term} decreases the hazard by {(1-coef['hazard_ratio'])*100:.1f}%.\n"
                else:
                    interpretation += "No significant predictors found in the Cox model.\n"
                
                # Model fit
                if 'model_fit' in cox_result and 'concordance' in cox_result['model_fit']:
                    concordance = cox_result['model_fit']['concordance']
                    interpretation += f"\nModel concordance index (C-index): {concordance:.3f}. "
                    
                    if concordance > 0.7:
                        interpretation += "This indicates good discrimination.\n"
                    elif concordance > 0.6:
                        interpretation += "This indicates moderate discrimination.\n"
                    else:
                        interpretation += "This indicates poor discrimination.\n"
        
        # Assumption testing results
        interpretation += "\n**Assumption Testing Results:**\n"
        
        # Sample size assumption
        if group_variable:
            interpretation += "\nSample size adequacy by group:\n"
            for group in groups:
                group_label = str(group)
                key = f'sample_size_{group_label}'
                if key in assumptions:
                    result = assumptions[key]['result'] 
                    n = assumptions[key].get('n', 0)
                    interpretation += f"- Group '{group_label}': {n} subjects - "
                    if result == AssumptionResult.PASSED:
                        interpretation += "Sufficient sample size.\n"
                    elif result == AssumptionResult.WARNING:
                        interpretation += "Borderline sample size, interpret with caution.\n"
                    else:
                        interpretation += "Insufficient sample size, results may be unreliable.\n"
        else:
            if 'sample_size' in assumptions:
                result = assumptions['sample_size']['result']
                n = assumptions['sample_size'].get('n', 0)
                interpretation += f"Sample size: {n} subjects - "
                if result == AssumptionResult.PASSED:
                    interpretation += "Sufficient for reliable analysis.\n"
                elif result == AssumptionResult.WARNING:
                    interpretation += "Borderline, consider results preliminary.\n"
                else:
                    interpretation += "Insufficient, results should be interpreted with extreme caution.\n"
        
        # Proportional hazards assumption
        if 'proportional_hazards' in assumptions:
            ph_result = assumptions['proportional_hazards']['result']
            interpretation += "\nProportional hazards assumption: "
            
            if ph_result == AssumptionResult.PASSED:
                interpretation += "Satisfied - model is appropriate for this data.\n"
            elif ph_result == AssumptionResult.WARNING:
                interpretation += "Partially satisfied - some concerns present.\n"
            elif ph_result == AssumptionResult.FAILED:
                interpretation += "Violated - consider time-dependent coefficients or stratified models.\n"
            else:
                interpretation += "Could not be tested.\n"
            
            # Add details for individual variables if available
            individual_ph_keys = [k for k in assumptions.keys() if k.startswith('proportional_hazards_') and k != 'proportional_hazards']
            if individual_ph_keys:
                interpretation += "Individual variable PH test results:\n"
                for key in individual_ph_keys:
                    var_name = key.replace('proportional_hazards_', '')
                    if assumptions[key] == AssumptionResult.PASSED:
                        interpretation += f"- {var_name}: Satisfied\n"
                    else:
                        interpretation += f"- {var_name}: Violated\n"
        
        # Outlier assessment
        if group_variable:
            interpretation += "\nOutlier assessment by group:\n"
            for group in groups:
                group_label = str(group)
                key = f'outliers_{group_label}'
                if key in assumptions:
                    result = assumptions[key]['result']
                    num_outliers = assumptions[key].get('num_outliers', 0)
                    interpretation += f"- Group '{group_label}': "
                    if result == AssumptionResult.PASSED:
                        interpretation += f"No significant outliers detected.\n"
                    elif result == AssumptionResult.WARNING:
                        interpretation += f"{num_outliers} potential outliers - verify data quality.\n"
                    else:
                        interpretation += f"Severe outliers present ({num_outliers}) - consider robust methods.\n"
        else:
            if 'outliers' in assumptions:
                result = assumptions['outliers']['result']
                num_outliers = assumptions['outliers'].get('num_outliers', 0)
                interpretation += "\nOutlier assessment: "
                if result == AssumptionResult.PASSED:
                    interpretation += "No significant outliers detected.\n"
                elif result == AssumptionResult.WARNING:
                    interpretation += f"{num_outliers} potential outliers - verify data quality.\n"
                else:
                    interpretation += f"Severe outliers present ({num_outliers}) - consider robust methods.\n"
        
        # Influential points 
        if 'influential_points' in assumptions:
            result = assumptions['influential_points']['result']
            num_influential = assumptions['influential_points'].get('num_extreme', 0)
            interpretation += "\nInfluential observations: "
            if result == AssumptionResult.PASSED:
                interpretation += "No highly influential points detected.\n"
            elif result == AssumptionResult.WARNING:
                interpretation += f"{num_influential} potentially influential points - verify model stability.\n"
            elif result == AssumptionResult.FAILED:
                interpretation += f"{num_influential} highly influential points detected - model may be unstable.\n"
            else:
                interpretation += "Could not be assessed.\n"
        
        # Autocorrelation 
        if 'autocorrelation' in assumptions:
            result = assumptions['autocorrelation']['result']
            interpretation += "\nAutocorrelation in residuals: "
            if result == AssumptionResult.PASSED:
                interpretation += "No significant autocorrelation detected.\n"
            elif result == AssumptionResult.WARNING:
                interpretation += "Mild autocorrelation present - consider time-dependent effects.\n"
            elif result == AssumptionResult.FAILED:
                interpretation += "Significant autocorrelation detected - model may not account for time-related structure.\n"
            else:
                interpretation += "Could not be assessed.\n"
        
        # Linearity 
        if 'linearity' in assumptions:
            result = assumptions['linearity']['result']
            interpretation += "\nLinearity of continuous predictors: "
            if result == AssumptionResult.PASSED:
                interpretation += "Linear relationship confirmed.\n"
            elif result == AssumptionResult.WARNING:
                interpretation += "Potential non-linearity - consider transformation or splines.\n"
            elif result == AssumptionResult.FAILED:
                interpretation += "Non-linear relationship detected - use non-linear terms or stratification.\n"
            else:
                interpretation += "Could not be assessed.\n"
        
        # Normality of residuals 
        if 'residual_normality' in assumptions:
            result = assumptions['residual_normality']['result']
            interpretation += "\nNormality of residuals: "
            if result == AssumptionResult.PASSED:
                interpretation += "Residuals appear normally distributed.\n"
            elif result == AssumptionResult.WARNING:
                interpretation += "Slight deviation from normality - generally acceptable for Cox model.\n"
            elif result == AssumptionResult.FAILED:
                interpretation += "Non-normal residuals - check for model misspecification.\n"
            else:
                interpretation += "Could not be assessed.\n"
        
        # Homogeneity of variance
        if 'homogeneity' in assumptions:
            result = assumptions['homogeneity']['result']
            interpretation += "\nHomogeneity of variance across groups: "
            if result == AssumptionResult.PASSED:
                interpretation += "Variances appear homogeneous.\n"
            elif result == AssumptionResult.WARNING:
                interpretation += "Minor heterogeneity present - generally acceptable.\n"
            elif result == AssumptionResult.FAILED:
                interpretation += "Significant heterogeneity detected - consider stratified analysis.\n"
            else:
                interpretation += "Could not be assessed.\n"
        
        # Overall summary and recommendations
        interpretation += "\n**Summary and Recommendations:**\n"
        
        # Overall survival and group differences
        if group_variable and logrank_result and 'significant' in logrank_result:
            if logrank_result['significant']:
                interpretation += f"There are significant differences in survival between the {group_variable} groups. "
                interpretation += "Further analysis of specific group differences and hazard ratios is warranted.\n"
            else:
                interpretation += f"No significant differences in survival were detected between the {group_variable} groups. "
                interpretation += "Consider whether the study had sufficient power or whether other factors might influence survival.\n"
        
        # Assess model validity based on assumptions
        failed_assumptions = [k for k, v in assumptions.items() if isinstance(v, dict) and v.get('result') == AssumptionResult.FAILED]
        warning_assumptions = [k for k, v in assumptions.items() if isinstance(v, dict) and v.get('result') == AssumptionResult.WARNING]
        
        if failed_assumptions:
            interpretation += "\nModel validity concerns: Some key assumptions were violated:\n"
            for assumption in failed_assumptions:
                readable_name = assumption.replace('_', ' ').title().replace('Ph', 'PH')
                interpretation += f"- {readable_name}\n"
            
            interpretation += "\nRecommended actions:\n"
            
            # Specific recommendations based on which assumptions failed
            if 'proportional_hazards' in failed_assumptions or any(k.startswith('proportional_hazards_') for k in failed_assumptions):
                interpretation += "- Consider stratified Cox model or time-dependent coefficients\n"
                interpretation += "- Accelerated failure time models may be more appropriate\n"
            
            if 'linearity' in failed_assumptions:
                interpretation += "- Include non-linear transformations of continuous predictors\n"
                interpretation += "- Consider using splines or polynomial terms\n"
            
            if 'influential_points' in failed_assumptions or 'outliers' in failed_assumptions:
                interpretation += "- Investigate extreme observations and verify data quality\n"
                interpretation += "- Consider robust regression methods\n"
            
            if 'sample_size' in failed_assumptions or any(k.startswith('sample_size_') for k in failed_assumptions):
                interpretation += "- Results should be considered exploratory only\n"
                interpretation += "- A larger sample size is strongly recommended\n"
            
            if 'homogeneity' in failed_assumptions:
                interpretation += "- Consider separate models for each group\n"
                
        elif warning_assumptions:
            interpretation += "\nModel validity concerns: Some assumptions showed borderline results:\n"
            for assumption in warning_assumptions:
                readable_name = assumption.replace('_', ' ').title().replace('Ph', 'PH')
                interpretation += f"- {readable_name}\n"
            
            interpretation += "Results should be interpreted with some caution.\n"
        else:
            interpretation += "\nModel validity: All tested assumptions appear to be reasonably satisfied, "
            interpretation += "suggesting the model is appropriate for this data.\n"
        
        # Event rate
        if overall_km['event_rate'] < 0.1:
            interpretation += "\nThe overall event rate is low, which may limit the statistical power. "
            interpretation += "Consider longer follow-up or larger sample size for future studies.\n"
        
        # Final assessment of results reliability
        interpretation += "\n**Overall Assessment of Results Reliability:**\n"
        
        if failed_assumptions:
            reliability = "Low to Moderate"
            confidence = "Results should be interpreted with substantial caution due to violated assumptions."
        elif warning_assumptions:
            reliability = "Moderate to High"
            confidence = "Results are generally trustworthy but some caution is warranted."
        else:
            reliability = "High"
            confidence = "Results appear reliable based on all tested assumptions."
        
        interpretation += f"Reliability: {reliability}\n"
        interpretation += f"{confidence}\n"
        
        return {
            'test': 'Survival Analysis',
            'overall_km': overall_km,
            'group_km': group_km,
            'logrank_test': logrank_result,
            'cox_ph': cox_result,
            'assumptions': assumptions,
            'figures': figures,
            'interpretation': interpretation
        }
    except Exception as e:
        return {
            'test': 'Survival Analysis',
            'error': str(e),
            'traceback': traceback.format_exc()
        }