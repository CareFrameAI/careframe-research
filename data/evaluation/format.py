# ancova_keys = {
#     'test': str,  # Always 'ANCOVA'
    
#     'overall': {
#         'r_squared': float,
#         'adj_r_squared': float,
#         'f_statistic': float,
#         'f_pvalue': float,
#         'df_model': int,
#         'df_residual': int,
#         'mse': float,
#         'rmse': float,
#         'aic': float,
#         'bic': float
#     },
    
#     'anova_table': {
#         str: {  # Term name as key
#             'sum_sq': float,
#             'df': int,
#             'mean_sq': float,
#             'f_value': float,
#             'p_value': float,
#             'significant': bool
#         }
#     },
    
#     'effect_sizes': {
#         str: {  # Term name as key
#             'partial_eta_squared': float,
#             'cohen_f': float,
#             'magnitude': str  # One of: "Negligible", "Small", "Medium", "Large"
#         }
#     },
    
#     'adjusted_means': {
#         str: float  # Group name as key, adjusted mean as value
#     },
    
#     'adjusted_std_errors': {
#         str: Optional[float]  # Group name as key, standard error as value
#     },
    
#     'pairwise_comparisons': List[{
#         'group1': str,
#         'group2': str,
#         'mean_difference': float,
#         'std_error': float,
#         'p_value': float,
#         'ci_lower': float,
#         'ci_upper': float,
#         'significant': bool
#     }],
    
#     'assumptions': {
#         'normality': {
#             'test': str,
#             'statistic': float,
#             'p_value': float,
#             'satisfied': bool,
#             'details': str,
#             'skewness': float,
#             'kurtosis': float,
#             'warnings': List[str],
#             'figures': Dict[str, str]
#         },
#         'homogeneity_of_variance': {
#             'test': str,
#             'statistic': float,
#             'p_value': float,
#             'satisfied': bool,
#             'details': str,
#             'group_variances': Dict[str, float],
#             'warnings': List[str],
#             'figures': Dict[str, str]
#         },
#         'homogeneity_of_regression_slopes': Dict[str, {
#             'p_value': float,
#             'satisfied': bool,
#             'details': str
#         }],
#         'linearity': Dict[str, {  # Covariate name as key
#             'r_squared': Optional[float],
#             'pearson_r': Optional[float],
#             'pearson_p': Optional[float],
#             'satisfied': bool,
#             'details': str
#         }],
#         'multicollinearity': {
#             'vif_values': Dict[str, float],
#             'satisfied': bool,
#             'details': str,
#             'correlation_matrix': pd.DataFrame,
#             'warnings': List[str],
#             'figures': Dict[str, str]
#         },
#         'influential_observations': {
#             'n_influential': int,
#             'threshold': float,
#             'max_cooks_d': float,
#             'influential_indices': List[int],
#             'satisfied': bool,
#             'details': str
#         }
#     },
    
#     'formula': str,
#     'summary': str,
#     'interpretation': str,
#     'figures': {
#         'adjusted_means': Optional[str],  # Base64 encoded SVG
#         'pairwise_comparisons': Optional[str],  # Base64 encoded SVG
#         'residuals_vs_fitted': Optional[str],  # Base64 encoded SVG
#         'qq_plot': Optional[str],  # Base64 encoded SVG
#         'scale_location': Optional[str],  # Base64 encoded SVG
#         'cooks_distance': Optional[str],  # Base64 encoded SVG
#         'residuals_by_group': Optional[str],  # Base64 encoded SVG
#         'partial_regression_plots': Optional[str],  # Base64 encoded SVG
#         'interaction_plots': Optional[str],  # Base64 encoded SVG
#         'effect_sizes': Optional[str]  # Base64 encoded SVG
#     }
# }

# chi_square_keys = {
#     'test': str,  # Always 'Chi-Square Test of Independence'
#     'chi2': float,
#     'p_value': float,
#     'dof': int,
#     'expected': List[List[float]],
#     'min_expected': float,
#     'significant': bool,
#     'cramers_v': float,
#     'effect_magnitude': str,  # One of: "Negligible", "Small", "Medium", "Large"
    
#     'table_summary': {
#         'n_rows': int,
#         'n_cols': int,
#         'row_totals': List[float],
#         'col_totals': List[float],
#         'total_n': int
#     },
    
#     'additional_statistics': {
#         'standardized_residuals': List[List[float]],
#         'adjusted_residuals': List[List[float]],
#         'cell_contributions': List[List[float]],
#         'cell_contributions_percent': List[List[float]],
#         'row_proportions': List[List[float]],
#         'col_proportions': List[List[float]],
#         'significant_cells': List[{
#             'row': int,
#             'col': int,
#             'observed': float,
#             'expected': float,
#             'std_residual': float,
#             'direction': str  # 'higher' or 'lower'
#         }],
#         'critical_value': float
#     },
    
#     'assumptions': {
#         'expected_frequencies': {
#             'result': str,  # One of: "passed", "warning", "failed"
#             'message': str,
#             'min_expected': float,
#             'percent_below_5': float,
#             'any_below_1': bool,
#             'details': str
#         },
#         'sample_size': {
#             'result': str,  # One of: "passed", "warning", "failed"
#             'message': str,
#             'total_n': int,
#             'min_recommended': int,
#             'details': str
#         },
#         'independence': {
#             'result': None,  # Always None as it can't be tested statistically
#             'message': str,
#             'details': str
#         }
#     },
    
#     'interpretation': str,
#     'figures': {
#         'observed_heatmap': Optional[str],  # Base64 encoded SVG
#         'std_residuals_heatmap': Optional[str],  # Base64 encoded SVG
#         'mosaic_plot': Optional[str],  # Base64 encoded SVG
#         'observed_vs_expected_bar': Optional[str]  # Base64 encoded SVG
#     }
# }

# fishers_exact_keys = {
#     'test': str,  # Always "Fisher's Exact Test"
#     'odds_ratio': {
#         'odds_ratio': float,
#         'ci_lower': Optional[float],
#         'ci_upper': Optional[float]
#     },
#     'relative_risk': Optional[{
#         'relative_risk': float,
#         'ci_lower': float,
#         'ci_upper': float
#     }],
#     'p_value': float,
#     'significant': bool,
#     'cramers_v': float,
#     'effect_magnitude': str,  # One of: "Negligible", "Small", "Medium", "Large"
    
#     'table_summary': {
#         'row_totals': List[float],
#         'col_totals': List[float],
#         'total_n': int
#     },
    
#     'assumptions': {
#         'sample_size': {
#             'total_n': int,
#             'min_expected_frequency': float,
#             'satisfied': bool,
#             'details': str
#         },
#         'independence': {
#             'satisfied': None,  # Always None as it can't be tested statistically
#             'details': str
#         }
#     },
    
#     'additional_statistics': {
#         'risk_measures': Optional[{
#             'absolute_risk_difference': float,
#             'rd_ci_lower': float,
#             'rd_ci_upper': float,
#             'number_needed_to_treat': float,
#             'nnt_type': str  # "NNT" or "NNH"
#         }],
#         'exact_confidence_intervals': Optional[{
#             'note': str,
#             'ci_lower': float,
#             'ci_upper': float
#         }],
#         'diagnostic_measures': Optional[{
#             'sensitivity': float,
#             'specificity': float,
#             'positive_predictive_value': float,
#             'negative_predictive_value': float,
#             'positive_likelihood_ratio': float,
#             'negative_likelihood_ratio': float,
#             'youdens_j': float
#         }],
#         'chi_square_comparison': Optional[{
#             'statistic': float,
#             'p_value': float,
#             'significant': bool,
#             'agrees_with_fisher': bool
#         }],
#         'barnard_test_approximation': Optional[{
#             'p_value': float,
#             'n_simulations': int,
#             'significant': bool,
#             'agrees_with_fisher': bool
#         }]
#     },
    
#     'interpretation': str,
#     'figures': {
#         'contingency_table': Optional[str],  # Base64 encoded SVG
#         'mosaic_plot': Optional[str],  # Base64 encoded SVG
#         'forest_plot': Optional[str],  # Base64 encoded SVG
#         'observed_vs_expected': Optional[str],  # Base64 encoded SVG
#         'diagnostic_measures': Optional[str],  # Base64 encoded SVG
#         'p_value_comparison': Optional[str]  # Base64 encoded SVG
#     }
# }

# independent_t_test_keys = {
#     'test': str,  # Always 'Independent T-Test'
#     'statistic': float,
#     'p_value': float,
#     'significant': bool,
#     'df': int,
#     'mean_difference': float,
#     'std_error': float,
#     'ci_lower': float,
#     'ci_upper': float,
#     'cohens_d': float,
#     'hedges_g': float,
#     'effect_magnitude': str,  # One of: "Negligible", "Small", "Medium", "Large"
#     'n_samples': int,
    
#     'assumptions': {
#         'normality': {
#             'results': List[{
#                 'group': str,
#                 'test': str,
#                 'statistic': float,
#                 'p_value': float,
#                 'satisfied': bool,
#                 'details': str,
#                 'skewness': float,
#                 'kurtosis': float,
#                 'warnings': List[str],
#                 'figures': Dict[str, str]
#             }],
#             'overall_satisfied': bool
#         },
#         'homogeneity_of_variance': {
#             'test': str,
#             'statistic': float,
#             'p_value': float,
#             'satisfied': bool,
#             'details': str,
#             'group_variances': Dict[str, float],
#             'warnings': List[str],
#             'figures': Dict[str, str]
#         },
#         'sample_size': {
#             'results': List[{
#                 'group': str,
#                 'n': int,
#                 'satisfied': bool,
#                 'details': str
#             }],
#             'overall_satisfied': bool
#         },
#         'outliers': {
#             'results': List[{
#                 'group': str,
#                 'has_outliers': bool,
#                 'num_outliers': int,
#                 'details': str
#             }],
#             'overall_satisfied': bool
#         }
#     },
    
#     'additional_statistics': {
#         'mann_whitney_u': Optional[{
#             'statistic': float,
#             'p_value': float,
#             'significant': bool,
#             'agrees_with_t_test': bool
#         }],
#         'bayes_factor': Optional[{
#             'bf10': float,
#             'interpretation': str
#         }],
#         'bootstrap_ci': Optional[{
#             'ci_lower': float,
#             'ci_upper': float,
#             'n_bootstrap': int
#         }],
#         'equivalence_test': Optional[{
#             'bound': float,
#             'p_value': float,
#             'equivalent': bool,
#             'interpretation': str
#         }],
#         'power_analysis': Optional[{
#             'achieved_power': float,
#             'recommended_n': int,
#             'actual_effect_size': float,
#             'is_powered': bool
#         }],
#         'sensitivity': Optional[{
#             'min_detectable_effect': float,
#             'interpretation': str
#         }],
#         'error_rates': Optional[{
#             'prior_h1': float,
#             'false_discovery_rate': Optional[float],
#             'false_negative_rate': Optional[float],
#             'warning': str
#         }]
#     },
    
#     'interpretation': str,
#     'figures': {
#         'box_plot': Optional[str],  # Base64 encoded SVG
#         'density_plot': Optional[str],  # Base64 encoded SVG
#         'means_with_ci': Optional[str],  # Base64 encoded SVG
#         'qq_plots': Optional[str],  # Base64 encoded SVG
#         'power_curve': Optional[str],  # Base64 encoded SVG
#         'bootstrap_distribution': Optional[str],  # Base64 encoded SVG
#         'effect_size_visualization': Optional[str],  # Base64 encoded SVG
#         'assumption_summary': Optional[str],  # Base64 encoded SVG
#         'residuals_plot': Optional[str],  # Base64 encoded SVG
#         'test_comparison': Optional[str]  # Base64 encoded SVG
#     }
# }

# kendall_tau_keys = {
#     'test': str,  # Always 'Kendall Tau Correlation'
#     'statistic': float,
#     'p_value': float,
#     'significant': bool,
#     'interpretation': str,
    
#     'assumptions': {
#         'monotonicity': {
#             'test': str,
#             'statistic': Optional[float],
#             'satisfied': bool,
#             'details': {
#                 'monotonicity_correlation': Optional[float],
#                 'interpretation': str
#             }
#         },
#         'outliers': {
#             'x_variable': Dict,  # OutlierTest result
#             'y_variable': Dict   # OutlierTest result
#         },
#         'sample_size': Dict,  # SampleSizeTest result
#         'ties': {
#             'test': str,
#             'x_ties': int,
#             'y_ties': int,
#             'satisfied': bool,
#             'details': {
#                 'x_tie_percentage': float,
#                 'y_tie_percentage': float,
#                 'interpretation': str
#             }
#         }
#     },
    
#     'assumption_violations': List[str],
#     'confidence_interval': List[float],  # [lower, upper]
    
#     'effect_size': {
#         'tau': float,
#         'tau_squared': float,
#         'interpretation': str,
#         'practical_significance': str
#     },
    
#     'concordance_analysis': {
#         'concordant_pairs': int,
#         'discordant_pairs': int,
#         'tied_pairs': int,
#         'total_comparisons': float,
#         'concordant_percentage': float,
#         'discordant_percentage': float,
#         'tied_percentage': float
#     },
    
#     'power_analysis': {
#         'power': float,
#         'recommended_sample_size': int
#     },
    
#     'sample_size': int,
    
#     'figures': {
#         'scatter_plot': {
#             'x': List[float],
#             'y': List[float],
#             'x_label': str,
#             'y_label': str,
#             'title': str
#         },
#         'concordance_plot': {
#             'labels': List[str],
#             'values': List[int],
#             'title': str
#         },
#         'bootstrap_distribution': Optional[{
#             'distribution': List[float],
#             'observed_value': float,
#             'ci_lower': float,
#             'ci_upper': float,
#             'title': str
#         }]
#     }
# }

# kruskal_wallis_keys = {
#     'test': str,  # Always 'Kruskal-Wallis Test'
#     'statistic': float,
#     'p_value': float,
#     'significant': bool,
#     'df': int,
#     'eta_squared': float,
#     'epsilon_squared': float,
#     'effect_magnitude': str,  # One of: "Negligible", "Small", "Medium", "Large"
    
#     'group_stats': List[{
#         'n': int,
#         'median': float,
#         'mean': float,
#         'std': float,
#         'min': float,
#         'max': float,
#         'q1': float,
#         'q3': float,
#         'iqr': float,
#         'name': str
#     }],
    
#     'post_hoc': {
#         'dunn': Optional[List[{
#             'group1': str,
#             'group2': str,
#             'p_value': float,
#             'significant': bool
#         }]],
#         'mann_whitney': Optional[List[{
#             'group1': str,
#             'group2': str,
#             'u_statistic': float,
#             'z_score': float,
#             'effect_size_r': float,
#             'median_diff': float,
#             'p_value': float,
#             'p_adjusted': float,
#             'significant': bool
#         }]],
#         'conover': Optional[List[{
#             'group1': str,
#             'group2': str,
#             'p_value': float,
#             'significant': bool
#         }]]
#     },
    
#     'assumptions': {
#         'sample_size': {
#             'results': List[{
#                 'group': str,
#                 'n': int,
#                 'satisfied': bool,
#                 'details': str
#             }],
#             'overall_satisfied': bool
#         },
#         'outliers': {
#             'results': List[{
#                 'group': str,
#                 'has_outliers': bool,
#                 'num_outliers': int,
#                 'details': str
#             }],
#             'overall_satisfied': bool
#         },
#         'similar_distributions': {
#             'results': List[{
#                 'group': str,
#                 'skewness': float,
#                 'kurtosis': float
#             }],
#             'skewness_range': float,
#             'kurtosis_range': float,
#             'satisfied': bool
#         }
#     },
    
#     'additional_statistics': {
#         'jonckheere_terpstra': Optional[{
#             'statistic': float,
#             'z_statistic': float,
#             'p_value': float,
#             'significant': bool,
#             'interpretation': str
#         }],
#         'anova_comparison': Optional[{
#             'f_statistic': float,
#             'p_value': float,
#             'significant': bool,
#             'eta_squared': float,
#             'agreement_with_kw': bool
#         }],
#         'rank_analysis': Optional[{
#             'mean_ranks': List[{
#                 'group': str,
#                 'mean_rank': float,
#                 'sum_rank': float
#             }],
#             'rank_differences': List[{
#                 'group1': str,
#                 'group2': str,
#                 'rank_diff': float,
#                 'abs_rank_diff': float
#             }]
#         }],
#         'bootstrap': Optional[{
#             'n_samples': int,
#             'results': List[{
#                 'group1': str,
#                 'group2': str,
#                 'median_diff': float,
#                 'ci_lower': float,
#                 'ci_upper': float,
#                 'includes_zero': bool
#             }]
#         }],
#         'power_analysis': Optional[{
#             'observed_power': float,
#             'current_total_n': int,
#             'total_n_for_80_power': int,
#             'total_n_for_90_power': int,
#             'total_n_for_95_power': int
#         }]
#     },
    
#     'interpretation': str,
#     'figures': {
#         'boxplot': Optional[str],  # Base64 encoded SVGs
#         'density_plot': Optional[str],  # Base64 encoded SVG
#         'mean_ranks': Optional[str],  # Base64 encoded SVG
#         'posthoc_heatmap': Optional[str],  # Base64 encoded SVG
#         'test_comparison': Optional[str],  # Base64 encoded SVG
#         'bootstrap_intervals': Optional[str],  # Base64 encoded SVG
#         'violin_plot': Optional[str],  # Base64 encoded SVG
#         'power_analysis': Optional[str],  # Base64 encoded SVG
#         'ecdf_plot': Optional[str]  # Base64 encoded SVG
#     }
# }

# linear_mixed_effects_keys = {
#     'test': str,  # Always 'Linear Mixed Effects Model'
#     'model_summary': str,
#     'fixed_effects': List[{
#         'parameter': str,
#         'coefficient': float,
#         'std_error': float,
#         't_value': float,
#         'p_value': float,
#         'ci_lower': float,
#         'ci_upper': float,
#         'significant': bool
#     }],
#     'random_effects_variance': Dict[str, float],
#     'residual_variance': float,
    
#     'model_statistics': {
#         'aic': float,
#         'bic': float,
#         'marginal_r2': float,
#         'conditional_r2': float,
#         'rmse': float,
#         'mae': float,
#         'icc': Optional[float]
#     },
    
#     'assumptions': {
#         'residual_normality': {
#             'test': str,
#             'statistic': float,
#             'p_value': float,
#             'satisfied': bool,
#             'details': str,
#             'skewness': float,
#             'kurtosis': float,
#             'warnings': List[str]
#         },
#         'homoscedasticity': {
#             'test': str,
#             'p_value': float,
#             'satisfied': bool,
#             'details': str
#         },
#         'multicollinearity': {
#             'vif_values': Dict[str, float],
#             'high_vif_predictors': Dict[str, float],
#             'very_high_vif_predictors': Dict[str, float],
#             'satisfied': bool,
#             'details': str
#         },
#         'linearity': {
#             'correlations': Dict[str, {
#                 'correlation': float,
#                 'p_value': float
#             }],
#             'nonlinear_predictors': Dict[str, {
#                 'correlation': float,
#                 'p_value': float
#             }],
#             'satisfied': bool,
#             'details': str
#         },
#         'independence': {
#             'test': str,
#             'statistic': float,
#             'satisfied': bool,
#             'details': str
#         },
#         'influential_observations': {
#             'method': str,
#             'high_influence_indices': Optional[List[int]],
#             'outlier_indices': Optional[List[int]],
#             'satisfied': bool,
#             'details': str
#         }
#     },
    
#     'interpretation': str,
#     'figures': {
#         'residuals_vs_fitted': Optional[str],  # Base64 encoded SVG
#         'qq_plot': Optional[str],  # Base64 encoded SVG
#         'residuals_histogram': Optional[str],  # Base64 encoded SVG
#         'random_effects_boxplot': Optional[str],  # Base64 encoded SVG
#         'coefficients_plot': Optional[str],  # Base64 encoded SVG
#         'predicted_vs_actual': Optional[str],  # Base64 encoded SVG
#         'assumption_summary': Optional[str],  # Base64 encoded SVG
#         'variance_decomposition': Optional[str],  # Base64 encoded SVG
#         'residuals_by_group': Optional[str],  # Base64 encoded SVG
#         'vif_plot': Optional[str]  # Base64 encoded SVG
#     }
# }

# linear_regression_keys = {
#     'test': str,  # Always 'Linear Regression'
#     'coefficients': List[{
#         'name': str,
#         'coef': float,
#         'std_err': float,
#         't_value': float,
#         'p_value': float,
#         'significant': bool,
#         'ci_lower': float,
#         'ci_upper': float,
#         'standardized_coef': Optional[float]
#     }],
#     'overall': {
#         'r_squared': float,
#         'adj_r_squared': float,
#         'f_statistic': float,
#         'f_pvalue': float,
#         'df_model': int,
#         'df_residual': int,
#         'mse': float,
#         'rmse': float,
#         'aic': float,
#         'bic': float
#     },
#     'assumptions': {
#         'multicollinearity': {
#             'vif_values': Dict[str, float],
#             'satisfied': bool,
#             'details': str,
#             'correlation_matrix': Dict,
#             'warnings': List[str],
#             'figures': Dict[str, str]
#         },
#         'normality': {
#             'test': str,
#             'statistic': float,
#             'p_value': float,
#             'satisfied': bool,
#             'details': str,
#             'skewness': float,
#             'kurtosis': float,
#             'warnings': List[str],
#             'figures': Dict[str, str]
#         },
#         'homoscedasticity': {
#             'test': str,
#             'statistic': float,
#             'p_value': float,
#             'satisfied': bool,
#             'details': str,
#             'warnings': List[str],
#             'figures': Dict[str, str]
#         },
#         'linearity': {
#             'results': Dict[str, {
#                 'r_squared': float,
#                 'pearson_r': float,
#                 'pearson_p': float,
#                 'satisfied': bool,
#                 'details': str
#             }],
#             'overall_satisfied': Optional[bool]
#         },
#         'autocorrelation': {
#             'test': str,
#             'statistic': float,
#             'p_value': Optional[float],
#             'satisfied': bool,
#             'details': str,
#             'warnings': List[str],
#             'figures': Dict[str, str]
#         },
#         'influential_observations': {
#             'n_influential': int,
#             'threshold': float,
#             'max_cooks_d': float,
#             'influential_indices': List[int],
#             'satisfied': bool,
#             'details': str
#         }
#     },
#     'prediction_stats': {
#         'mae': float,
#         'mse': float,
#         'rmse': float,
#         'cross_validation': {
#             'cv_mae': float,
#             'cv_mse': float,
#             'cv_rmse': float,
#             'cv_r2': float,
#             'optimism_r2': float,
#             'optimism_rmse': float
#         }
#     },
#     'formula': str,
#     'summary': str,
#     'interpretation': str,
#     'figures': {
#         'coefficient_plot': Optional[str],  # Base64 encoded SVG
#         'standardized_coefficient_plot': Optional[str],  # Base64 encoded SVG
#         'residuals_vs_fitted': Optional[str],  # Base64 encoded SVG
#         'qq_plot': Optional[str],  # Base64 encoded SVG
#         'scale_location': Optional[str],  # Base64 encoded SVG
#         'cooks_distance': Optional[str],  # Base64 encoded SVG
#         'partial_regression_plots': Optional[str],  # Base64 encoded SVG
#         'correlation_heatmap': Optional[str],  # Base64 encoded SVG
#         'actual_vs_predicted': Optional[str],  # Base64 encoded SVG
#         'residual_histogram': Optional[str]  # Base64 encoded SVG
#     }
# }

# logistic_regression_keys = {
#     'test': str,  # Always 'Logistic Regression'
#     'coefficients': Dict[str, {
#         'estimate': float,
#         'std_error': float,
#         'z_value': float,
#         'p_value': float,
#         'significant': bool,
#         'odds_ratio': float,
#         'ci_lower': float,
#         'ci_upper': float
#     }],
#     'model_fit': {
#         'log_likelihood': float,
#         'null_log_likelihood': float,
#         'lr_statistic': float,
#         'lr_df': int,
#         'lr_p_value': float,
#         'mcfadden_r2': float,
#         'cox_snell_r2': float,
#         'nagelkerke_r2': float,
#         'aic': float,
#         'bic': float,
#         'significant': bool
#     },
#     'prediction_stats': {
#         'confusion_matrix': {
#             'tp': int,
#             'tn': int,
#             'fp': int,
#             'fn': int
#         },
#         'accuracy': float,
#         'sensitivity': float,
#         'specificity': float,
#         'ppv': float,
#         'npv': float,
#         'f1': float,
#         'auc': float,
#         'brier_score': float,
#         'probability_distribution': {
#             'mean': float,
#             'std': float,
#             'min': float,
#             'max': float
#         },
#         'threshold_analysis': List[{
#             'threshold': float,
#             'accuracy': float,
#             'sensitivity': float,
#             'specificity': float,
#             'ppv': float,
#             'npv': float,
#             'f1': float
#         }],
#         'cross_validation': {
#             'cv_accuracy_mean': float,
#             'cv_accuracy_std': float,
#             'cv_sensitivity_mean': float,
#             'cv_sensitivity_std': float,
#             'cv_specificity_mean': float,
#             'cv_specificity_std': float,
#             'cv_auc_mean': float,
#             'cv_auc_std': float
#         }
#     },
#     'assumptions': {
#         'multicollinearity': {
#             'vif_values': Dict[str, float],
#             'satisfied': bool,
#             'details': str,
#             'correlation_matrix': Dict,
#             'warnings': List[str],
#             'figures': Dict[str, str]
#         },
#         'linearity_of_logit': {
#             'results': Dict[str, {
#                 'r_squared': float,
#                 'pearson_r': float,
#                 'pearson_p': float,
#                 'satisfied': bool,
#                 'details': str
#             }],
#             'overall_satisfied': bool
#         },
#         'sample_size': {
#             'n_total': int,
#             'n_events': int,
#             'n_non_events': int,
#             'n_minority_class': int,
#             'min_recommended': int,
#             'events_per_predictor': float,
#             'satisfied': bool,
#             'details': str
#         },
#         'influential_observations': {
#             'n_influential': int,
#             'threshold': float,
#             'max_cooks_d': float,
#             'influential_indices': List[int],
#             'satisfied': bool,
#             'details': str
#         },
#         'goodness_of_fit': {
#             'test': str,
#             'statistic': float,
#             'p_value': float,
#             'satisfied': bool,
#             'details': str
#         }
#     },
#     'formula': str,
#     'summary': str,
#     'interpretation': str,
#     'figures': {
#         'odds_ratio_plot': Optional[str],  # Base64 encoded SVG
#         'roc_curve': Optional[str],  # Base64 encoded SVG
#         'confusion_matrix': Optional[str],  # Base64 encoded SVG
#         'calibration_curve': Optional[str],  # Base64 encoded SVG
#         'probability_distribution': Optional[str],  # Base64 encoded SVG
#         'threshold_analysis': Optional[str],  # Base64 encoded SVG
#         'linearity_plots': Optional[str],  # Base64 encoded SVG
#         'cooks_distance': Optional[str]  # Base64 encoded SVG
#     }
# }

# mann_whitney_keys = {
#     'test': str,  # Always 'Mann-Whitney U Test'
#     'statistic': float,
#     'p_value': float,
#     'significant': bool,
#     'alternative': str,
#     'group1_stats': {
#         'n': int,
#         'median': float,
#         'mean': float,
#         'std': float,
#         'min': float,
#         'max': float,
#         'q1': float,
#         'q3': float,
#         'iqr': float
#     },
#     'group2_stats': {
#         'n': int,
#         'median': float,
#         'mean': float,
#         'std': float,
#         'min': float,
#         'max': float,
#         'q1': float,
#         'q3': float,
#         'iqr': float
#     },
#     'effect_size_r': float,
#     'effect_magnitude': str,  # One of: "Negligible", "Small", "Medium", "Large"
#     'hodges_lehmann_estimator': float,
#     'z_score': float,
    
#     'assumptions': {
#         'similar_shapes': {
#             'result': str,  # One of: "passed", "warning", "failed"
#             'message': str,
#             'group1_skewness': float,
#             'group2_skewness': float,
#             'group1_kurtosis': float,
#             'group2_kurtosis': float,
#             'details': str
#         },
#         'independence': {
#             'result': None,  # Always None as it can't be tested statistically
#             'message': str,
#             'details': str
#         },
#         'sample_size': {
#             'result': str,  # One of: "passed", "warning", "failed"
#             'message': str,
#             'group1_size': {
#                 'n': int,
#                 'satisfied': bool
#             },
#             'group2_size': {
#                 'n': int,
#                 'satisfied': bool
#             },
#             'details': str
#         },
#         'outliers': {
#             'result': str,  # One of: "passed", "warning", "failed"
#             'message': str,
#             'group1_outliers': int,
#             'group2_outliers': int,
#             'details': str
#         }
#     },
    
#     'additional_statistics': {
#         't_test': Optional[{
#             'statistic': float,
#             'p_value': float,
#             'significant': bool,
#             'cohens_d': float,
#             'agreement_with_mann_whitney': bool
#         }],
#         'hodges_lehmann_ci': Optional[{
#             'estimator': float,
#             'ci_lower': float,
#             'ci_upper': float,
#             'confidence_level': float
#         }],
#         'bootstrap': Optional[{
#             'n_samples': int,
#             'median_diff_ci_lower': float,
#             'median_diff_ci_upper': float,
#             'median_diff': float
#         }],
#         'rank_biserial': Optional[{
#             'correlation': float,
#             'group1_mean_rank': float,
#             'group2_mean_rank': float
#         }],
#         'probability_of_superiority': Optional[{
#             'probability': float,
#             'interpretation': str
#         }]
#     },
    
#     'interpretation': str,
#     'figures': {
#         'boxplot': Optional[str],  # Base64 encoded SVG
#         'ecdf_plot': Optional[str],  # Base64 encoded SVG
#         'rank_plot': Optional[str],  # Base64 encoded SVG
#         'density_plot': Optional[str],  # Base64 encoded SVG
#         'test_comparison': Optional[str],  # Base64 encoded SVG
#         'bootstrap_distribution': Optional[str]  # Base64 encoded SVG
#     }
# }

# mixed_anova_keys = {
#     'test': str,  # Always 'Mixed ANOVA'
#     'design': {
#         'between_factors': List[str],
#         'within_factors': List[str],
#         'n_subjects': int,
#         'cells_per_subject': float,
#         'balanced': bool
#     },
#     'anova_results': Dict[str, {
#         'term': str,
#         'df': Optional[float],
#         'df_res': Optional[float],
#         'F': Optional[float],
#         'p_value': float,
#         'significant': bool,
#         'partial_eta_squared': Optional[float],
#         'sphericity': bool,
#         'p_gg': Optional[float],  # Greenhouse-Geisser corrected p-value
#         'p_hf': Optional[float],  # Huynh-Feldt corrected p-value
#         't_value': Optional[float],
#         'coefficient': Optional[float],
#         'std_error': Optional[float]
#     }],
#     'descriptive_statistics': List[{
#         'cell': str,
#         'n': int,
#         'mean': float,
#         'std': float,
#         'se': float,
#         'ci95_lower': float,
#         'ci95_upper': float,
#         'min': float,
#         'max': float,
#         'median': float,
#         'skewness': float,
#         'kurtosis': float
#     }],
#     'posthoc_tests': Dict[str, {  # Factor name as key
#         'method': str,  # Usually 'Tukey HSD'
#         'comparisons': List[{
#             'group1': str,
#             'group2': str,
#             'mean_difference': float,
#             'ci_lower': float,
#             'ci_upper': float,
#             'p_adjust': Optional[float],
#             'significant': bool,
#             'cohens_d': float
#         }]
#     }],
#     'assumptions': {
#         'normality': {
#             'test': str,
#             'results': List[{
#                 'cell': str,
#                 'n': int,
#                 'shapiro_p': float,
#                 'satisfied': bool,
#                 'skewness': float,
#                 'kurtosis': float
#             }],
#             'overall_satisfied': bool,
#             'details': str
#         },
#         'homogeneity_of_variance': Optional[{
#             'levene_test': {
#                 'statistic': float,
#                 'p_value': float,
#                 'satisfied': bool
#             },
#             'bartlett_test': {
#                 'statistic': float,
#                 'p_value': float,
#                 'satisfied': bool
#             },
#             'overall_satisfied': bool,
#             'details': str
#         }],
#         'sphericity': Optional[{
#             'results': Dict[str, {  # Term as key
#                 'satisfied': bool,
#                 'details': str
#             }],
#             'overall_satisfied': bool,
#             'details': str
#         }],
#         'sample_size': {
#             'n_subjects': int,
#             'min_cell_count': int,
#             'balanced': bool,
#             'adequate': bool,
#             'details': str
#         },
#         'outliers': {
#             'method': str,
#             'num_outliers': int,
#             'outlier_indices': List[int],
#             'satisfied': bool,
#             'details': str
#         }
#     },
#     'interpretation': str,
#     'figures': {
#         'main_effect_*': Optional[str],  # Base64 encoded SVG, one per factor
#         'interaction_*_*': Optional[str],  # Base64 encoded SVG, one per interaction
#         'boxplot_*': Optional[str],  # Base64 encoded SVG, one per between-subjects factor
#         'profile_*': Optional[str],  # Base64 encoded SVG, one per within-subjects factor
#         'assumption_checks': Optional[str],  # Base64 encoded SVG
#         'posthoc_*': Optional[str],  # Base64 encoded SVG, one per factor with post-hoc tests
#         'significant_effects_summary': Optional[str]  # Base64 encoded SVG
#     }
# }

# multinomial_logistic_regression_keys = {
#     'test': str,  # Always 'Multinomial Logistic Regression'
#     'coefficients': Dict[str, {  # Category name as key
#         str: {  # Predictor name as key
#             'estimate': float,
#             'std_error': float,
#             'z_value': float,
#             'p_value': float,
#             'significant': bool,
#             'odds_ratio': float,
#             'ci_lower': float,
#             'ci_upper': float
#         }
#     }],
#     'model_fit': {
#         'log_likelihood': float,
#         'null_log_likelihood': Optional[float],
#         'lr_statistic': Optional[float],
#         'lr_df': Optional[int],
#         'lr_p_value': Optional[float],
#         'mcfadden_r2': Optional[float],
#         'cox_snell_r2': Optional[float],
#         'nagelkerke_r2': Optional[float],
#         'aic': float,
#         'bic': float,
#         'significant': Optional[bool]
#     },
#     'prediction_stats': {
#         'accuracy': float,
#         'confusion_matrix': List[List[int]],
#         'per_class': Dict[str, {  # Category name as key
#             'precision': float,
#             'recall': float,
#             'f1_score': float,
#             'support': int
#         }],
#         'macro_avg': {
#             'precision': float,
#             'recall': float,
#             'f1_score': float
#         },
#         'weighted_avg': {
#             'precision': float,
#             'recall': float,
#             'f1_score': float
#         },
#         'roc_curves': Optional[Dict[str, {  # Category name as key
#             'false_positive_rate': List[float],
#             'true_positive_rate': List[float],
#             'auc': float
#         }]],
#         'macro_avg_auc': Optional[float]
#     },
#     'assumptions': {
#         'multicollinearity': {
#             'vif_values': Dict[str, float],  # Predictor name as key
#             'high_vif_predictors': Dict[str, float],
#             'very_high_vif_predictors': Dict[str, float],
#             'satisfied': bool,
#             'details': str
#         },
#         'influential_observations': {
#             'by_category': Dict[str, {  # Category name as key
#                 'threshold': float,
#                 'max_cooks_d': float,
#                 'influential_indices': List[int]
#             }],
#             'has_influential': bool,
#             'satisfied': bool,
#             'details': str
#         },
#         'linearity_of_logit': Optional[{
#             'by_category': Dict[str, {  # Category name as key
#                 str: {  # Predictor name as key
#                     'correlation': float,
#                     'correlation_p': float,
#                     'nonlinear': Optional[bool],
#                     'satisfied': Optional[bool]
#                 }
#             }],
#             'all_linear': bool,
#             'satisfied': bool,
#             'details': str
#         }],
#         'sample_size': {
#             'n_observations': int,
#             'n_predictors': int,
#             'category_counts': Dict[str, int],  # Category name as key
#             'min_count': int,
#             'min_required': int,
#             'satisfied': bool,
#             'details': str
#         }
#     },
#     'formula': str,
#     'summary': str,
#     'interpretation': str,
#     'figures': {
#         'odds_ratio_plot': Optional[str],  # Base64 encoded SVG
#         'confusion_matrix': Optional[str],  # Base64 encoded SVG
#         'roc_curves': Optional[str],  # Base64 encoded SVG
#         'prediction_distribution': Optional[str],  # Base64 encoded SVG
#         'assumption_summary': Optional[str],  # Base64 encoded SVG
#         'vif_plot': Optional[str],  # Base64 encoded SVG
#         'performance_metrics': Optional[str],  # Base64 encoded SVG
#         'effect_plot_*': Optional[str],  # Base64 encoded SVG, one per categorical predictor
#         'boxplot_*': Optional[str]  # Base64 encoded SVG, one per continuous predictor
#     }
# }

# negative_binomial_regression_keys = {
#     'test': str,  # Always 'Negative Binomial Regression'
#     'coefficients': Dict[str, {  # Term name as key
#         'estimate': float,
#         'std_error': float,
#         'z_value': float,
#         'p_value': float,
#         'significant': bool,
#         'irr': float,  # Incidence Rate Ratio
#         'ci_lower': float,
#         'ci_upper': float,
#         'percent_change': float,
#         'standardized_coef': float
#     }],
#     'model_fit': {
#         'log_likelihood': float,
#         'null_log_likelihood': Optional[float],
#         'lr_statistic': Optional[float],
#         'lr_df': Optional[int],
#         'lr_p_value': Optional[float],
#         'mcfadden_r2': Optional[float],
#         'cox_snell_r2': Optional[float],
#         'nagelkerke_r2': Optional[float],
#         'aic': float,
#         'bic': float,
#         'alpha_parameter': float,
#         'significant': Optional[bool],
#         'deviance': float,
#         'pearson_chi2': float,
#         'df_resid': int,
#         'df_model': int
#     },
#     'overdispersion': {
#         'lr_statistic': Optional[float],
#         'p_value': Optional[float],
#         'significant': Optional[bool],
#         'poisson_aic': Optional[float],
#         'nb_aic': Optional[float],
#         'poisson_bic': Optional[float],
#         'nb_bic': Optional[float],
#         'pearson_dispersion': Optional[float],
#         'alpha_parameter': Optional[float],
#         'alpha_p_value': Optional[float]
#     },
#     'prediction_stats': {
#         'mean_observed': float,
#         'mean_predicted': float,
#         'min_predicted': float,
#         'max_predicted': float,
#         'correlation_obs_pred': float,
#         'mean_absolute_error': float,
#         'root_mean_squared_error': float,
#         'mean_squared_error': float,
#         'median_absolute_error': float,
#         'r_squared_raw': float,
#         'sum_squared_error': float,
#         'total_sum_squares': float,
#         'mean_pearson_residual': float,
#         'max_pearson_residual': float,
#         'mean_deviance_residual': float,
#         'max_deviance_residual': float,
#         'zero_count': int,
#         'zero_percent': float
#     },
#     'assumptions': {
#         'multicollinearity': Dict,  # MulticollinearityTest result
#         'overdispersion': Dict,  # OverdispersionTest result
#         'outliers': Dict,  # OutlierTest result
#         'independence': Dict,  # IndependenceTest result
#         'goodness_of_fit': Dict,  # GoodnessOfFitTest result
#         'sample_size': Dict  # SampleSizeTest result
#     },
#     'formula': str,
#     'summary': str,
#     'interpretation': str,
#     'figures': {
#         'residuals_vs_predicted': Optional[str],  # Base64 encoded SVG
#         'qq_plot': Optional[str],  # Base64 encoded SVG
#         'observed_vs_predicted': Optional[str],  # Base64 encoded SVG
#         'plot_generation_error': Optional[str]
#     },
#     'alpha_parameter': float,
#     'n_observations': int,
#     'n_predictors': int,
#     'zero_count': int,
#     'zero_percent': float
# }

# one_sample_t_test_keys = {
#     'test': str,  # Always 'One-Sample T-Test'
#     'statistic': float,
#     'p_value': float,
#     'significant': bool,
#     'df': int,
#     'mean': float,
#     'std': float,
#     'se': float,
#     'ci_lower': float,
#     'ci_upper': float,
#     'cohens_d': float,
#     'effect_magnitude': str,  # One of: "Negligible", "Small", "Medium", "Large"
#     'n_samples': int,
#     'mu': float,
#     'alternative': str,  # One of: 'two-sided', 'less', 'greater'
    
#     'assumptions': {
#         'normality': {
#             'test': str,
#             'statistic': float,
#             'p_value': float,
#             'satisfied': bool,
#             'details': str,
#             'skewness': float,
#             'kurtosis': float,
#             'warnings': List[str],
#             'figures': Dict[str, str]
#         },
#         'sample_size': {
#             'n': int,
#             'satisfied': bool,
#             'details': str
#         },
#         'outliers': {
#             'has_outliers': bool,
#             'num_outliers': int,
#             'details': str,
#             'figures': Dict[str, str]
#         }
#     },
    
#     'additional_statistics': {
#         'bootstrap_ci': Optional[{
#             'ci_lower': float,
#             'ci_upper': float,
#             'n_bootstrap': int
#         }],
#         'power_analysis': Optional[{
#             'achieved_power': float,
#             'recommended_n': int,
#             'actual_effect_size': float,
#             'is_powered': bool
#         }],
#         'bayesian_analysis': Optional[{
#             'posterior_mean': float,
#             'posterior_std': float,
#             'credible_lower': float,
#             'credible_upper': float,
#             'bayes_factor': float,
#             'bf_interpretation': str
#         }],
#         'equivalence_test': Optional[{
#             'bound': float,
#             'p_value': float,
#             'equivalent': bool,
#             'interpretation': str
#         }]
#     },
    
#     'interpretation': str,
#     'figures': {
#         'histogram': Optional[str],  # Base64 encoded SVG
#         'qq_plot': Optional[str],  # Base64 encoded SVG
#         'bootstrap_distribution': Optional[str],  # Base64 encoded SVG
#         'power_curve': Optional[str],  # Base64 encoded SVG
#         'bayesian_analysis': Optional[str],  # Base64 encoded SVG
#         'assumption_summary': Optional[str]  # Base64 encoded SVG
#     },
#     'summary': str  # Contains NaN warning if applicable
# }

# one_way_anova_keys = {
#     'test': str,  # Always 'One-Way ANOVA'
#     'statistic': float,  # F-statistic
#     'p_value': float,
#     'significant': bool,
#     'df_between': int,
#     'df_within': int,
#     'df_total': int,
#     'ss_between': float,
#     'ss_within': float,
#     'ss_total': float,
#     'ms_between': float,
#     'ms_within': float,
#     'eta_squared': float,
#     'partial_eta_squared': float,
#     'omega_squared': float,
#     'effect_magnitude': str,  # One of: "Negligible", "Small", "Medium", "Large"
    
#     'group_stats': List[{
#         'n': int,
#         'mean': float,
#         'std': float,
#         'min': float,
#         'max': float,
#         'name': str
#     }],
    
#     'post_hoc': {
#         'tukey_hsd': Optional[List[{
#             'group1': str,
#             'group2': str,
#             'mean_diff': float,
#             'p_value': float,
#             'significant': bool,
#             'ci_lower': float,
#             'ci_upper': float
#         }]],
#         'bonferroni': Optional[List[{
#             'group1': str,
#             'group2': str,
#             'mean_diff': float,
#             'p_value': float,
#             'p_adjusted': float,
#             'significant': bool,
#             'ci_lower': float,
#             'ci_upper': float
#         }]],
#         'scheffe': Optional[List[{
#             'group1': str,
#             'group2': str,
#             'mean_diff': float,
#             'test_statistic': float,
#             'critical_value': float,
#             'p_value': float,
#             'significant': bool,
#             'ci_lower': float,
#             'ci_upper': float
#         }]]
#     },
    
#     'assumptions': {
#         'normality': {
#             'results': List[{
#                 'group': str,
#                 'test': str,
#                 'statistic': float,
#                 'p_value': float,
#                 'satisfied': bool,
#                 'details': str,
#                 'skewness': float,
#                 'kurtosis': float,
#                 'warnings': List[str],
#                 'figures': Dict[str, str]
#             }],
#             'overall_satisfied': bool
#         },
#         'homogeneity_of_variance': {
#             'test': str,
#             'statistic': float,
#             'p_value': float,
#             'satisfied': bool,
#             'details': str,
#             'group_variances': Dict[str, float],
#             'warnings': List[str],
#             'figures': Dict[str, str]
#         },
#         'sample_size': {
#             'results': List[{
#                 'group': str,
#                 'n': int,
#                 'satisfied': bool,
#                 'details': str
#             }],
#             'overall_satisfied': bool
#         },
#         'outliers': {
#             'results': List[{
#                 'group': str,
#                 'has_outliers': bool,
#                 'num_outliers': int,
#                 'details': str
#             }],
#             'overall_satisfied': bool
#         }
#     },
    
#     'additional_statistics': {
#         'welch_anova': Optional[{
#             'statistic': float,
#             'p_value': float,
#             'df_numerator': float,
#             'df_denominator': float,
#             'significant': bool,
#             'agrees_with_anova': bool
#         }],
#         'kruskal_wallis': Optional[{
#             'statistic': float,
#             'p_value': float,
#             'df': int,
#             'significant': bool,
#             'agrees_with_anova': bool
#         }],
#         'mann_whitney_tests': Optional[List[{
#             'group1': str,
#             'group2': str,
#             'statistic': float,
#             'p_value': float,
#             'p_adjusted': float,
#             'significant': bool
#         }]],
#         'trimmed_means': Optional[List[{
#             'group': str,
#             'trimmed_mean': float,
#             'trimmed_sd': float
#         }]],
#         'power_analysis': Optional[{
#             'effect_size_f': float,
#             'observed_power': float,
#             'total_n': int,
#             'n_for_80_power': int,
#             'n_for_90_power': int,
#             'n_for_95_power': int
#         }],
#         'bootstrap': Optional[{
#             'n_samples': int,
#             'results': List[{
#                 'group': str,
#                 'mean': float,
#                 'bootstrap_mean': float,
#                 'bootstrap_se': float,
#                 'ci_lower': float,
#                 'ci_upper': float
#             }]
#         }]
#     },
    
#     'interpretation': str,
#     'figures': {
#         'boxplot': Optional[str],  # Base64 encoded SVG
#         'means_plot': Optional[str],  # Base64 encoded SVG
#         'qq_plots': Optional[str],  # Base64 encoded SVG
#         'residuals_plot': Optional[str],  # Base64 encoded SVG
#         'violin_plot': Optional[str],  # Base64 encoded SVG
#         'power_analysis': Optional[str],  # Base64 encoded SVG
#         'post_hoc_heatmap': Optional[str],  # Base64 encoded SVG
#         'bootstrap_intervals': Optional[str]  # Base64 encoded SVG
#     }
# }

# ordinal_regression_keys = {
#     'test': str,  # Always 'Ordinal Regression'
#     'coefficients': Dict[str, {  # Term name as key
#         'estimate': float,
#         'std_error': Optional[float],
#         'z_value': Optional[float],
#         'p_value': Optional[float],
#         'significant': Optional[bool],
#         'odds_ratio': float,
#         'ci_lower': Optional[float],
#         'ci_upper': Optional[float]
#     }],
#     'thresholds': Dict[str, {  # Threshold name as key
#         'estimate': float,
#         'std_error': Optional[float],
#         'z_value': Optional[float],
#         'p_value': Optional[float]
#     }],
#     'model_fit': {
#         'log_likelihood': float,
#         'null_log_likelihood': Optional[float],
#         'lr_statistic': Optional[float],
#         'lr_df': Optional[int],
#         'lr_p_value': Optional[float],
#         'mcfadden_r2': Optional[float],
#         'cox_snell_r2': Optional[float],
#         'nagelkerke_r2': Optional[float],
#         'aic': float,
#         'bic': float,
#         'significant': Optional[bool]
#     },
#     'prediction_stats': {
#         'avg_predicted_probabilities': Dict[str, float],  # Category as key
#         'accuracy': float,
#         'class_metrics': Dict[str, {  # Category as key
#             'precision': float,
#             'recall': float,
#             'f1': float
#         }],
#         'cross_validation': {
#             'cv_accuracy_mean': float,
#             'cv_accuracy_std': float
#         }
#     },
#     'assumptions': {
#         'outcome_check': {
#             'result': str,  # 'passed' or 'warning'
#             'details': {
#                 'num_categories': int,
#                 'categories': List[str],
#                 'interpretation': str
#             }
#         },
#         'multicollinearity': Dict,  # MulticollinearityTest result
#         'sample_size': Dict,  # SampleSizeTest result
#         'cell_sparsity': Optional[Dict[str, {  # Predictor name as key
#             'result': str,  # 'passed' or 'warning'
#             'details': {
#                 'min_cell_count': int,
#                 'sparse_cells': int,
#                 'interpretation': str
#             }
#         }]],
#         'proportional_odds': Dict  # ProportionalOddsTest result
#     },
#     'outcome_categories': List[str],
#     'formula': str,
#     'summary': str,
#     'interpretation': str,
#     'figures': {
#         'outcome_distribution': Optional[str],  # Base64 encoded SVG
#         'odds_ratio_plot': Optional[str],  # Base64 encoded SVG
#         'predicted_probabilities': Optional[str],  # Base64 encoded SVG
#         'confusion_matrix': Optional[str],  # Base64 encoded SVG
#         'predicted_vs_actual': Optional[str],  # Base64 encoded SVG
#         'proportional_odds': Optional[str],  # Base64 encoded SVG
#         'thresholds': Optional[str],  # Base64 encoded SVG
#         'cumulative_probability': Optional[str]  # Base64 encoded SVG
#     }
# }

# paired_t_test_keys = {
#     'test': str,  # Always 'Paired T-Test'
#     'statistic': float,
#     'p_value': float,
#     'significant': bool,
#     'df': int,
#     'mean_difference': float,
#     'std_error': float,
#     'ci_lower': float,
#     'ci_upper': float,
#     'cohens_d': float,
#     'hedges_g': float,
#     'effect_magnitude': str,  # One of: "Negligible", "Small", "Medium", "Large"
#     'n_pairs': int,
    
#     'assumptions': {
#         'normality': Dict,  # NormalityTest result
#         'outliers': Dict,  # OutlierTest result
#         'sample_size': Dict  # SampleSizeTest result
#     },
    
#     'additional_statistics': {
#         'wilcoxon_test': Optional[{
#             'statistic': float,
#             'p_value': float,
#             'significant': bool,
#             'agreement_with_ttest': bool
#         }],
#         'bootstrap': Optional[{
#             'n_samples': int,
#             'ci_lower': float,
#             'ci_upper': float,
#             'mean': float,
#             'std_error': float
#         }],
#         'power_analysis': Optional[{
#             'observed_power': float,
#             'sample_size_for_80_power': int,
#             'sample_size_for_90_power': int,
#             'sample_size_for_95_power': int
#         }],
#         'robust_statistics': Optional[{
#             'median_difference': float,
#             'trimmed_mean_difference': float,
#             'mad': float  # Median Absolute Deviation
#         }],
#         'difference_descriptives': Optional[{
#             'min': float,
#             'q1': float,
#             'median': float,
#             'q3': float,
#             'max': float,
#             'skewness': float,
#             'kurtosis': float,
#             'iqr': float
#         }]
#     },
    
#     'interpretation': str,
#     'figures': {
#         'difference_histogram': Optional[str],  # Base64 encoded SVG
#         'paired_boxplot': Optional[str],  # Base64 encoded SVG
#         'qq_plot': Optional[str],  # Base64 encoded SVG
#         'differences_plot': Optional[str],  # Base64 encoded SVG
#         'power_analysis': Optional[str],  # Base64 encoded SVG
#         'bootstrap_distribution': Optional[str]  # Base64 encoded SVG
#     }
# }

# pearson_correlation_keys = {
#     'test': str,  # Always 'Pearson Correlation'
#     'statistic': float,  # Correlation coefficient (r)
#     'p_value': float,
#     'significant': bool,
#     'interpretation': str,
#     'assumptions': {
#         'normality': {
#             'x_variable': Dict,  # NormalityTest result
#             'y_variable': Dict   # NormalityTest result
#         },
#         'linearity': Dict,  # LinearityTest result
#         'outliers': {
#             'x_variable': Dict,  # OutlierTest result
#             'y_variable': Dict   # OutlierTest result
#         },
#         'homoscedasticity': {
#             'test': str,  # 'Breusch-Pagan'
#             'statistic': float,
#             'p_value': float,
#             'satisfied': bool
#         }
#     },
#     'assumption_violations': List[str],  # List of violated assumptions
#     'confidence_interval': List[float],  # [lower, upper]
#     'effect_size': {
#         'r': float,
#         'r_squared': float,
#         'interpretation': str  # One of: "weak", "moderate", "strong"
#     },
#     'sample_size': int,
#     'additional_statistics': {
#         'alternative_correlations': {
#             'spearman': {
#                 'coefficient': float,
#                 'p_value': float,
#                 'significant': bool
#             },
#             'kendall': {
#                 'coefficient': float,
#                 'p_value': float,
#                 'significant': bool
#             }
#         },
#         'effect_size': {
#             'r_squared': float,
#             'cohen_interpretation': str,  # One of: "small", "medium", "large"
#             'explained_variance_percent': float
#         },
#         'bootstrap': Optional[{
#             'n_samples': int,
#             'confidence_interval': List[float],
#             'standard_error': float
#         }],
#         'influential_points': Optional[{
#             'points': List[{
#                 'index': int,
#                 'x_value': float,
#                 'y_value': float,
#                 'cooks_distance': float
#             }],
#             'threshold': float,
#             'max_cooks_distance': float
#         }]
#     },
#     'figures': {
#         'scatter_plot': Optional[str],  # Base64 encoded SVG
#         'distribution_plots': Optional[str],  # Base64 encoded SVG
#         'residual_plots': Optional[str],  # Base64 encoded SVG
#         'bootstrap_distribution': Optional[str],  # Base64 encoded SVG
#         'correlation_comparison': Optional[str]  # Base64 encoded SVG
#     }
# }

# point_biserial_correlation_keys = {
#     'test': str,  # Always 'Point-Biserial Correlation'
#     'correlation': float,
#     'p_value': float,
#     'significant': bool,
    
#     # Group statistics
#     'n0': int,
#     'n1': int,
#     'mean0': float,
#     'mean1': float,
#     'std0': float,
#     'std1': float,
#     'mean_diff': float,
    
#     # Effect sizes
#     'r_squared': float,
#     'cohen_d': float,
#     'effect_magnitude': str,  # One of: "Negligible", "Small", "Medium", "Large"
#     'direction': str,  # One of: "positive", "negative"
    
#     # Confidence interval
#     'ci_lower': float,
#     'ci_upper': float,
    
#     # T-test equivalent
#     't_statistic': float,
#     'df': int,
    
#     # Sample size
#     'n': int,
    
#     'assumptions': {
#         'binary_check': {
#             'result': str,  # One of: "passed", "failed"
#             'details': {
#                 'unique_values': List[Union[float, str]],
#                 'is_binary': bool,
#                 'interpretation': str
#             }
#         },
#         'normality': {
#             'group0': Dict,  # NormalityTest result
#             'group1': Dict   # NormalityTest result
#         },
#         'outliers': {
#             'group0': Dict,  # OutlierTest result
#             'group1': Dict   # OutlierTest result
#         },
#         'sample_size': {
#             'group0': Dict,  # SampleSizeTest result
#             'group1': Dict   # SampleSizeTest result
#         },
#         'homogeneity_of_variance': Dict  # HomogeneityOfVarianceTest result
#     },
    
#     'interpretation': str
# }

# poisson_regression_keys = {
#     'test': str,  # Always 'Poisson Regression'
#     'coefficients': Dict[str, {  # Term name as key
#         'estimate': float,
#         'std_error': float,
#         'z_value': float,
#         'p_value': float,
#         'significant': bool,
#         'irr': float,  # Incidence Rate Ratio
#         'ci_lower': float,
#         'ci_upper': float
#     }],
    
#     'model_fit': {
#         'log_likelihood': float,
#         'null_log_likelihood': Optional[float],
#         'lr_statistic': Optional[float],
#         'lr_df': Optional[int],
#         'lr_p_value': Optional[float],
#         'mcfadden_r2': Optional[float],
#         'cox_snell_r2': Optional[float],
#         'nagelkerke_r2': Optional[float],
#         'aic': float,
#         'bic': float,
#         'significant': Optional[bool]
#     },
    
#     'overdispersion': {
#         'method': str,  # "Dean's test" or "Dispersion parameter"
#         't_statistic': Optional[float],
#         'p_value': Optional[float],
#         'significant': Optional[bool],
#         'dispersion_parameter': float,
#         'has_overdispersion': bool,
#         'nb_comparison': Optional[{
#             'lr_statistic': float,
#             'p_value': float,
#             'significant': bool,
#             'poisson_aic': float,
#             'nb_aic': float,
#             'prefer_nb': bool
#         }]
#     },
    
#     'zero_inflation': {
#         'observed_zeros': int,
#         'predicted_zeros': int,
#         'zero_ratio': float,
#         'possible_zero_inflation': bool
#     },
    
#     'prediction_stats': {
#         'mean_observed': float,
#         'mean_predicted': float,
#         'min_predicted': float,
#         'max_predicted': float,
#         'correlation_obs_pred': float,
#         'mean_absolute_error': float,
#         'root_mean_squared_error': float,
#         'mean_abs_percentage_error': float,
#         'goodness_of_fit': Optional[{
#             'method': str,
#             'statistic': float,
#             'df': int,
#             'p_value': float,
#             'satisfied': bool,
#             'details': str
#         }],
#         'cross_validation': Optional[{
#             'k_folds': int,
#             'cv_mae': float,
#             'cv_rmse': float,
#             'cv_correlation': float
#         }]
#     },
    
#     'assumptions': {
#         'multicollinearity': {
#             'vif_values': Dict[str, float],
#             'high_vif_predictors': Dict[str, float],
#             'very_high_vif_predictors': Dict[str, float],
#             'satisfied': bool,
#             'details': str
#         },
#         'linearity': {
#             'predictors': Dict[str, {
#                 'correlation': float,
#                 'p_value': float,
#                 'satisfied': bool,
#                 'details': str
#             }],
#             'overall_satisfied': bool,
#             'details': str
#         },
#         'influential_observations': {
#             'method': str,
#             'threshold': Optional[float],
#             'max_cooks_d': Optional[float],
#             'high_influence_indices': Optional[List[int]],
#             'satisfied': bool,
#             'details': str
#         },
#         'independence': {
#             'details': str
#         }
#     },
    
#     'formula': str,
#     'summary': str,
#     'interpretation': str,
#     'figures': {
#         'observed_vs_predicted': Optional[str],  # Base64 encoded SVG
#         'pearson_residuals': Optional[str],  # Base64 encoded SVG
#         'irr_forest_plot': Optional[str],  # Base64 encoded SVG
#         'count_distribution': Optional[str],  # Base64 encoded SVG
#         'assumption_summary': Optional[str],  # Base64 encoded SVG
#         'vif_plot': Optional[str],  # Base64 encoded SVG
#         'predictor_effects': Optional[str],  # Base64 encoded SVG
#         'cooks_distance': Optional[str],  # Base64 encoded SVG
#         'pearson_residual_outliers': Optional[str],  # Base64 encoded SVG
#         'linearity_check': Optional[str]  # Base64 encoded SVG
#     }
# }

# repeated_measures_anova_keys = {
#     'test': str,  # Always 'Repeated Measures ANOVA'
#     'outcome': str,
#     'within_factors': List[str],
#     'descriptives': List[{
#         'mean': float,
#         'std': float,
#         'count': int,
#         'sem': float,
#         'ci95_lower': float,
#         'ci95_upper': float
#     }],
#     'anova_results': List[{
#         'Source': str,
#         'F': float,
#         'p-unc': float,
#         'DF': int,
#         'num Obs': int
#     }],
#     'effect_sizes': Dict[str, {  # Effect name as key
#         'partial_eta_sq': Optional[float],
#         'partial_omega_sq': Optional[float],
#         'f_value': Optional[float],
#         'p_value': Optional[float],
#         'significant': bool
#     }],
#     'sphericity_tests': Dict[str, {  # Source name as key
#         'W': Optional[float],
#         'p_value': Optional[float],
#         'eps_gg': Optional[float],
#         'eps_hf': Optional[float],
#         'significant': Optional[bool],
#         'note': Optional[str]
#     }],
#     'posthoc_results': Dict[str, List[{  # Effect name as key
#         'A': str,
#         'B': str,
#         'mean_diff': float,
#         't': float,
#         'p_uncorrected': float,
#         'p_adjusted': float,
#         'significant': bool,
#         'hedges_g': Optional[float],
#         'cohen_d': Optional[float]
#     }]],
#     'assumptions': {
#         'normality': {
#             'test': str,
#             'statistic': float,
#             'p_value': float,
#             'significant': bool,
#             'satisfied': bool,
#             'interpretation': str
#         },
#         'sphericity': Dict[str, Dict],  # Same as sphericity_tests
#         'homogeneity_of_variance': {
#             'min_variance': float,
#             'max_variance': float,
#             'variance_ratio': float,
#             'satisfied': bool,
#             'interpretation': str
#         },
#         'outliers': {
#             'method': str,
#             'n_outliers': int,
#             'outliers': List[{
#                 'index': int,
#                 'subject': Any,
#                 'value': float,
#                 'z_score': float,
#                 'factors': Dict[str, Any]
#             }],
#             'satisfied': bool,
#             'interpretation': str
#         },
#         'balanced_design': {
#             'is_balanced': bool,
#             'count_summary': List[Dict],
#             'interpretation': str
#         }
#     },
#     'figures': {
#         'qq_plot': Optional[str],  # Base64 encoded SVG
#         'residuals_histogram': Optional[str],  # Base64 encoded SVG
#         'variance_boxplot': Optional[str],  # Base64 encoded SVG
#         'boxplot': Optional[str],  # Base64 encoded SVG
#         'assumption_summary': Optional[str],  # Base64 encoded SVG
#         'effect_sizes': Optional[str],  # Base64 encoded SVG
#         'subject_profiles': Optional[str],  # Base64 encoded SVG
#         'main_effect_*': Optional[str],  # Base64 encoded SVG, one per factor
#         'interaction_*_*': Optional[str],  # Base64 encoded SVG, one per interaction
#         'posthoc_*': Optional[str]  # Base64 encoded SVG, one per effect with post-hoc tests
#     },
#     'interpretation': str,
#     'alpha': float,
#     'subject_id': str,
#     'n_subjects': int,
#     'balanced': Optional[bool]
# }

# spearman_correlation_keys = {
#     'test': str,  # Always 'Spearman Correlation'
#     'statistic': float,  # Correlation coefficient (rho)
#     'p_value': float,
#     'significant': bool,
#     'interpretation': str,
    
#     'assumptions': {
#         'monotonicity': {
#             'test': str,
#             'statistic': Optional[float],
#             'satisfied': bool,
#             'details': {
#                 'monotonicity_correlation': Optional[float],
#                 'interpretation': str
#             }
#         },
#         'outliers': {
#             'x_variable': Dict,  # OutlierTest result
#             'y_variable': Dict   # OutlierTest result
#         },
#         'sample_size': Dict,  # SampleSizeTest result
#         'tied_ranks': {
#             'test': str,
#             'x_ties': int,
#             'y_ties': int,
#             'satisfied': bool,
#             'details': {
#                 'x_tie_percentage': float,
#                 'y_tie_percentage': float,
#                 'interpretation': str
#             }
#         }
#     },
    
#     'assumption_violations': List[str],
#     'confidence_interval': List[float],  # [lower, upper]
    
#     'effect_size': {
#         'rho': float,
#         'rho_squared': float,
#         'interpretation': str,
#         'practical_significance': str
#     },
    
#     'rank_transformation_analysis': {
#         'pearson_r': float,
#         'spearman_r': float,
#         'difference': float,
#         'linearity': str,
#         'original_skewness': {
#             'x': float,
#             'y': float
#         },
#         'rank_skewness': {
#             'x': float,
#             'y': float
#         },
#         'interpretation': str
#     },
    
#     'power_analysis': {
#         'power': float,
#         'recommended_sample_size': int
#     },
    
#     'method_comparison': str,
#     'sample_size': int,
    
#     'figures': {
#         'scatter_plot': {
#             'x': List[float],
#             'y': List[float],
#             'x_label': str,
#             'y_label': str,
#             'title': str
#         },
#         'rank_scatter_plot': {
#             'x': List[float],
#             'y': List[float],
#             'x_label': str,
#             'y_label': str,
#             'title': str
#         },
#         'rank_transformation': {
#             'original_values': {
#                 'x': List[float],
#                 'y': List[float]
#             },
#             'rank_values': {
#                 'x': List[float],
#                 'y': List[float]
#             },
#             'title': str
#         },
#         'bootstrap_distribution': Optional[{
#             'distribution': List[float],
#             'observed_value': float,
#             'ci_lower': float,
#             'ci_upper': float,
#             'title': str
#         }]
#     }
# }

# survival_analysis_keys = {
#     'test': str,  # Always 'Survival Analysis'
    
#     'overall_km': {
#         'survival_curve': {
#             'times': List[float],
#             'survival_probs': List[float],
#             'confidence_lower': List[float],
#             'confidence_upper': List[float],
#             'num_at_risk': Optional[List[int]]
#         },
#         'median_survival': Optional[float],
#         'survival_rates': Dict[float, float],  # time point -> survival rate
#         'rmst': Dict[float, float],  # time point -> restricted mean survival time
#         'cumulative_hazard': {
#             'times': List[float],
#             'hazard': List[float]
#         },
#         'num_subjects': int,
#         'num_events': int,
#         'event_rate': float,
#         'max_follow_up': float
#     },
    
#     'group_km': Dict[str, {  # Group name as key
#         'survival_curve': {
#             'times': List[float],
#             'survival_probs': List[float],
#             'confidence_lower': List[float],
#             'confidence_upper': List[float],
#             'num_at_risk': Optional[List[int]]
#         },
#         'median_survival': Optional[float],
#         'survival_rates': Dict[float, float],
#         'rmst': Dict[float, float],
#         'cumulative_hazard': {
#             'times': List[float],
#             'hazard': List[float]
#         },
#         'num_subjects': int,
#         'num_events': int,
#         'event_rate': float,
#         'max_follow_up': float
#     }],
    
#     'logrank_test': Optional[{
#         'test_statistic': float,
#         'p_value': float,
#         'significant': bool,
#         'pairwise_comparisons': Optional[List[{
#             'group1': str,
#             'group2': str,
#             'test_statistic': float,
#             'p_value': float,
#             'significant': bool
#         }]]
#     }],
    
#     'cox_ph': {
#         'coefficients': Dict[str, {  # Term name as key
#             'estimate': float,
#             'hazard_ratio': float,
#             'std_error': float,
#             'p_value': float,
#             'significant': bool,
#             'ci_lower': float,
#             'ci_upper': float
#         }],
#         'model_fit': {
#             'log_likelihood': float,
#             'concordance': float,
#             'aic': float,
#             'bic': Optional[float]
#         },
#         'ph_test': {
#             'global_p_value': float,
#             'global_satisfied': bool,
#             'individual_p_values': Dict[str, {  # Variable name as key
#                 'p_value': float,
#                 'satisfied': bool
#             }]
#         },
#         'summary': str,
#         'formula': Optional[str]
#     },
    
#     'assumption_tests': {
#         'influential_observations': {
#             'test': str,
#             'num_extreme': int,
#             'threshold': float,
#             'proportion': float,
#             'satisfied': bool,
#             'interpretation': str
#         },
#         'proportional_hazards': Dict,  # Same as cox_ph['ph_test']
#         'non_linearity': Optional[{
#             'test': str,
#             'predictor': str,
#             'r_squared': float,
#             'mean_squared_deviation': float,
#             'satisfied': bool,
#             'interpretation': str
#         }]
#     },
    
#     'figures': {
#         'km_overall': Optional[str],  # Base64 encoded SVG
#         'hazard_overall': Optional[str],  # Base64 encoded SVG
#         'km_by_group': Optional[str],  # Base64 encoded SVG
#         'hazard_by_group': Optional[str],  # Base64 encoded SVG
#         'logrank_pairwise': Optional[str],  # Base64 encoded SVG
#         'cox_forest_plot': Optional[str],  # Base64 encoded SVG
#         'cox_predicted_survival': Optional[str],  # Base64 encoded SVG
#         'schoenfeld_*': Optional[str],  # Base64 encoded SVG, one per predictor
#         'martingale_residuals': Optional[str],  # Base64 encoded SVG
#         'deviance_residuals': Optional[str],  # Base64 encoded SVG
#         'functional_form': Optional[str],  # Base64 encoded SVG
#         'assumption_summary': Optional[str],  # Base64 encoded SVG
#         'lifetimes_plot': Optional[str],  # Base64 encoded SVG
#         'survival_rates_comparison': Optional[str],  # Base64 encoded SVG
#         'km_with_risk_table': Optional[str]  # Base64 encoded SVG
#     },
    
#     'interpretation': str
# }

# wilcoxon_signed_rank_test_keys = {
#     'test': str,  # Always 'Wilcoxon Signed-Rank Test'
#     'statistic': float,
#     'p_value': float,
#     'significant': bool,
#     'n_pairs': int,
#     'n_non_zero': int,
#     'n_zero_diff': int,
#     'median_difference': float,
#     'effect_size_r': float,
#     'effect_magnitude': str,  # One of: "Negligible", "Small", "Medium", "Large"
    
#     'assumptions': {
#         'symmetry': {
#             'result': str,  # One of: "passed", "warning", "failed"
#             'message': str,
#             'skewness': float,
#             'p_value': float,
#             'details': str
#         },
#         'sample_size': Dict,  # SampleSizeTest result
#         'outliers': Dict  # OutlierTest result
#     },
    
#     'additional_statistics': {
#         'paired_t_test': Optional[{
#             'statistic': float,
#             'p_value': float,
#             'significant': bool,
#             'cohens_d': float,
#             'agreement_with_wilcoxon': bool
#         }],
#         'sign_test': Optional[{
#             'statistic': float,
#             'p_value': float,
#             'significant': bool,
#             'positive_differences': int,
#             'negative_differences': int,
#             'agreement_with_wilcoxon': bool
#         }],
#         'hodges_lehmann': Optional[{
#             'estimator': float,
#             'ci_lower': float,
#             'ci_upper': float,
#             'confidence_level': float
#         }],
#         'bootstrap': Optional[{
#             'n_samples': int,
#             'ci_lower': float,
#             'ci_upper': float,
#             'median': float
#         }],
#         'rank_analysis': Optional[{
#             'positive_rank_sum': float,
#             'negative_rank_sum': float,
#             'min_rank_sum': float,
#             'ranks': List[float],
#             'signed_ranks': List[float],
#             'differences': List[float]
#         }],
#         'difference_descriptives': Optional[{
#             'min': float,
#             'q1': float,
#             'median': float,
#             'q3': float,
#             'max': float,
#             'mean': float,
#             'std': float,
#             'skewness': float,
#             'kurtosis': float,
#             'iqr': float
#         }]
#     },
    
#     'figures': {
#         'difference_histogram': Optional[str],  # Base64 encoded SVG
#         'paired_boxplot': Optional[str],  # Base64 encoded SVG
#         'signed_ranks': Optional[str],  # Base64 encoded SVG
#         'test_comparison': Optional[str],  # Base64 encoded SVG
#         'bootstrap_distribution': Optional[str],  # Base64 encoded SVG
#         'bland_altman': Optional[str]  # Base64 encoded SVG
#     },
    
#     'interpretation': str
# }
