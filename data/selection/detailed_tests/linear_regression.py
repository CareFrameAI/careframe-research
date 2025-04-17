import traceback
import pandas as pd
from scipy import stats
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict, KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from lifelines import KaplanMeierFitter, CoxPHFitter
from data.selection.detailed_tests.ancova import ancova
from data.selection.detailed_tests.negative_binomial_regression import negative_binomial_regression
from data.selection.detailed_tests.logistic_regression import logistic_regression
from study_model.study_model import StatisticalTest
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import OLSInfluence
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import base64
import io
from data.assumptions.tests import (
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
from data.assumptions.format import AssumptionTestKeys

def linear_regression(data: pd.DataFrame, outcome: str, predictors: List[str], alpha: float) -> Dict[str, Any]:
    """
    Performs comprehensive linear regression analysis with detailed diagnostics and visualizations.
    
    Args:
        data: DataFrame containing the data
        outcome: Name of the outcome variable
        predictors: List of predictor variable names
        alpha: Significance level
        
    Returns:
        Dictionary with regression results, diagnostics, and visualizations
    """
    try:
        # Set matplotlib to use a non-interactive backend
        matplotlib.use('Agg')
        figures = {}
        
        # Create formula for statsmodels
        formula = f"{outcome} ~ {' + '.join(predictors)}"
        
        # Fit the model
        model = smf.ols(formula=formula, data=data).fit()
        results = model
        
        # Extract results
        r_squared = float(results.rsquared)
        adj_r_squared = float(results.rsquared_adj)
        f_statistic = float(results.fvalue)
        f_pvalue = float(results.f_pvalue)
        mse = float(results.mse_resid)
        rmse = float(np.sqrt(results.mse_resid))
        aic = float(results.aic)
        bic = float(results.bic)
        
        # Get coefficients info
        coefficients = []
        for name, coef, std_err, t_val, p_val, ci_low, ci_high in zip(
            results.model.exog_names,
            results.params,
            results.bse,
            results.tvalues,
            results.pvalues,
            results.conf_int()[0],
            results.conf_int()[1]
        ):
            is_significant = p_val < alpha
            standardized_coef = None
            
            # Calculate standardized coefficient (beta)
            try:
                if name != "Intercept":
                    x_std = data[name].std()
                    y_std = data[outcome].std()
                    standardized_coef = coef * (x_std / y_std)
            except:
                pass
                
            coefficient = {
                'name': name,
                'coef': float(coef),
                'std_err': float(std_err),
                't_value': float(t_val),
                'p_value': float(p_val),
                'significant': bool(is_significant),
                'ci_lower': float(ci_low),
                'ci_upper': float(ci_high),
                'standardized_coef': standardized_coef
            }
            coefficients.append(coefficient)
        
        # Get residuals for assumption testing
        residuals = results.resid
        predicted = results.fittedvalues
        
        # Create coefficient plot
        plt.figure(figsize=(10, max(6, len(coefficients) * 0.5)))
        coef_names = [c['name'] for c in coefficients if c['name'] != 'Intercept']
        coef_values = [c['coef'] for c in coefficients if c['name'] != 'Intercept']
        ci_errors = [
            [c['coef'] - c['ci_lower'] for c in coefficients if c['name'] != 'Intercept'],
            [c['ci_upper'] - c['coef'] for c in coefficients if c['name'] != 'Intercept']
        ]
        
        # Sort by coefficient magnitude for better visualization
        if coef_names:
            coef_abs = [abs(c) for c in coef_values]
            sorted_idx = np.argsort(coef_abs)
            coef_names = [coef_names[i] for i in sorted_idx]
            coef_values = [coef_values[i] for i in sorted_idx]
            ci_errors[0] = [ci_errors[0][i] for i in sorted_idx]
            ci_errors[1] = [ci_errors[1][i] for i in sorted_idx]
            
            plt.errorbar(coef_values, coef_names, xerr=ci_errors, fmt='o', capsize=5, color=PASTEL_COLORS[0])
            plt.axvline(x=0, color=PASTEL_COLORS[4], linestyle='--')
            plt.xlabel('Coefficient Value')
            plt.title('Regression Coefficients with 95% Confidence Intervals')
            plt.grid(True, alpha=0.3)
            
            figures['coefficient_plot'] = fig_to_svg(plt.gcf())
            plt.close()
            
        # Create a standardized coefficient plot (if available)
        std_coef_values = [c.get('standardized_coef') for c in coefficients if c['name'] != 'Intercept' and c.get('standardized_coef') is not None]
        if std_coef_values and len(std_coef_values) == len(coef_names):
            plt.figure(figsize=(10, max(6, len(std_coef_values) * 0.5)))
            
            # Sort by standardized coefficient magnitude
            std_coef_abs = [abs(c) for c in std_coef_values]
            sorted_idx = np.argsort(std_coef_abs)
            std_coef_names = [coef_names[i] for i in sorted_idx]
            std_coef_values = [std_coef_values[i] for i in sorted_idx]
            
            plt.barh(std_coef_names, std_coef_values, color=PASTEL_COLORS[1])
            plt.axvline(x=0, color=PASTEL_COLORS[4], linestyle='--')
            plt.xlabel('Standardized Coefficient (Beta)')
            plt.title('Standardized Regression Coefficients')
            plt.grid(True, alpha=0.3)
            
            figures['standardized_coefficient_plot'] = fig_to_svg(plt.gcf())
            plt.close()
        
        # Generate diagnostic plots
        
        # 1. Residuals vs Fitted plot
        plt.figure(figsize=(10, 6))
        plt.scatter(predicted, residuals, alpha=0.5, color=PASTEL_COLORS[2])
        plt.axhline(y=0, color=PASTEL_COLORS[4], linestyle='-')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Fitted Values')
        plt.grid(True, alpha=0.3)
        
        # Add a lowess smoother
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smooth = lowess(residuals, predicted)
            plt.plot(smooth[:, 0], smooth[:, 1], color=PASTEL_COLORS[5], linewidth=2)
        except:
            pass
            
        figures['residuals_vs_fitted'] = fig_to_svg(plt.gcf())
        plt.close()
        
        # 2. Q-Q plot for residuals
        plt.figure(figsize=(10, 6))
        qq = ProbPlot(residuals)
        qq.qqplot(line='45', alpha=0.5, ax=plt.gca(), color=PASTEL_COLORS[0])
        plt.title('Q-Q Plot of Residuals')
        plt.grid(True, alpha=0.3)
        
        figures['qq_plot'] = fig_to_svg(plt.gcf())
        plt.close()
        
        # 3. Scale-Location plot (sqrt of abs residuals vs fitted)
        plt.figure(figsize=(10, 6))
        plt.scatter(predicted, np.sqrt(np.abs(residuals)), alpha=0.5, color=PASTEL_COLORS[3])
        plt.xlabel('Fitted Values')
        plt.ylabel('√|Residuals|')
        plt.title('Scale-Location Plot')
        plt.grid(True, alpha=0.3)
        
        # Add a lowess smoother
        try:
            smooth = lowess(np.sqrt(np.abs(residuals)), predicted)
            plt.plot(smooth[:, 0], smooth[:, 1], color=PASTEL_COLORS[5], linewidth=2)
        except:
            pass
            
        figures['scale_location'] = fig_to_svg(plt.gcf())
        plt.close()
        
        # 4. Cook's distance plot
        influence = OLSInfluence(results)
        cooks_d = influence.cooks_distance[0]
        
        plt.figure(figsize=(10, 6))
        markerline, stemlines, baseline = plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt='o', 
                linefmt='-', basefmt=' ')
        plt.setp(markerline, color=PASTEL_COLORS[0])
        plt.setp(stemlines, color=PASTEL_COLORS[0])
        plt.axhline(y=4/len(data), color=PASTEL_COLORS[4], linestyle='--', 
                   label=f'Threshold (4/n = {4/len(data):.5f})')
        plt.xlabel('Observation Index')
        plt.ylabel("Cook's Distance")
        plt.title("Cook's Distance for Influential Observations")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        figures['cooks_distance'] = fig_to_svg(plt.gcf())
        plt.close()
        
        # 5. Partial regression plots
        fig, axes = plt.subplots(1, len(predictors), figsize=(5*len(predictors), 5))
        if len(predictors) == 1:
            axes = [axes]
            
        for i, predictor in enumerate(predictors):
            if predictor in data.columns:  # Skip intercept or variables not in data
                try:
                    # Skip categorical variables
                    if pd.api.types.is_categorical_dtype(data[predictor]) or pd.api.types.is_object_dtype(data[predictor]):
                        axes[i].text(0.5, 0.5, f"{predictor} is categorical", 
                                   horizontalalignment='center', verticalalignment='center')
                        axes[i].set_title(f"Partial Regression Plot: {predictor}")
                    else:
                        sm.graphics.plot_partregress(outcome, predictor, 
                                                   [p for p in predictors if p != predictor], 
                                                   data=data, ax=axes[i])
                except:
                    axes[i].text(0.5, 0.5, f"Could not plot {predictor}", 
                               horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        
        figures['partial_regression_plots'] = fig_to_svg(plt.gcf())
        plt.close()
        
        # 6. Correlation heatmap
        # Only include numeric predictors
        numeric_predictors = [p for p in predictors if p in data.columns and pd.api.types.is_numeric_dtype(data[p])]
        
        if len(numeric_predictors) > 1:
            plt.figure(figsize=(10, 8))
            corr_matrix = data[numeric_predictors].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", 
                       cmap=sns.color_palette(PASTEL_COLORS), 
                       square=True, linewidths=.5)
            plt.title('Correlation Heatmap of Predictors')
            
            figures['correlation_heatmap'] = fig_to_svg(plt.gcf())
            plt.close()
        
        # 7. Actual vs Predicted
        plt.figure(figsize=(10, 8))
        plt.scatter(data[outcome], predicted, alpha=0.5, color=PASTEL_COLORS[0])
        
        # Add perfect prediction line
        min_val = min(data[outcome].min(), predicted.min())
        max_val = max(data[outcome].max(), predicted.max())
        plt.plot([min_val, max_val], [min_val, max_val], '--', color=PASTEL_COLORS[4])
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs. Predicted Values')
        plt.grid(True, alpha=0.3)
        
        # Add r-squared annotation
        plt.annotate(f'R² = {r_squared:.3f}', 
                   xy=(0.05, 0.95), xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        figures['actual_vs_predicted'] = fig_to_svg(plt.gcf())
        plt.close()
        
        # 8. Residual histogram
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black', color=PASTEL_COLORS[2])
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Histogram of Residuals')
        
        # Add normal curve
        from scipy import stats as spstats
        x = np.linspace(residuals.min(), residuals.max(), 100)
        y = spstats.norm.pdf(x, residuals.mean(), residuals.std()) * len(residuals) * (residuals.max() - residuals.min()) / 30
        plt.plot(x, y, '-', color=PASTEL_COLORS[4], linewidth=2)
        
        figures['residual_histogram'] = fig_to_svg(plt.gcf())
        plt.close()
        
        # Run assumption tests
        assumptions = {}
        
        # Use tests from AssumptionTestKeys enum
        # Normality test
        try:
            normality_test = AssumptionTestKeys.NORMALITY.value["function"]()
            normality_result = normality_test.run_test(data=residuals)
            
            assumptions['normality'] = {
                'test': normality_result.get('test_used', 'Shapiro-Wilk'),
                'statistic': float(normality_result.get('statistic', 0)),
                'p_value': float(normality_result.get('p_value', 1)),
                'satisfied': normality_result['result'].value == 'passed',
                'details': normality_result.get('details', ''),
                'skewness': float(normality_result.get('skewness', 0)) if 'skewness' in normality_result else None,
                'kurtosis': float(normality_result.get('kurtosis', 0)) if 'kurtosis' in normality_result else None,
                'warnings': normality_result.get('warnings', []),
                'figures': normality_result.get('figures', {})
            }
        except Exception as e:
            assumptions['normality'] = {
                'error': f'Could not test normality: {str(e)}'
            }
        
        # Homoscedasticity test
        try:
            homoscedasticity_test = AssumptionTestKeys.HOMOSCEDASTICITY.value["function"]()
            homoscedasticity_result = homoscedasticity_test.run_test(residuals=residuals, predicted=predicted)
            
            assumptions['homoscedasticity'] = {
                'test': homoscedasticity_result.get('test_used', 'Breusch-Pagan'),
                'statistic': float(homoscedasticity_result.get('statistic', 0)),
                'p_value': float(homoscedasticity_result.get('p_value', 1)),
                'satisfied': homoscedasticity_result['result'].value == 'passed',
                'details': homoscedasticity_result.get('details', ''),
                'warnings': homoscedasticity_result.get('warnings', []),
                'figures': homoscedasticity_result.get('figures', {})
            }
        except Exception as e:
            assumptions['homoscedasticity'] = {
                'error': f'Could not test homoscedasticity: {str(e)}'
            }
        
        # Multicollinearity test
        try:
            multicollinearity_test = AssumptionTestKeys.MULTICOLLINEARITY.value["function"]()
            multicollinearity_result = multicollinearity_test.run_test(df=data, covariates=predictors)
            
            # Extract VIF values for easier access
            vif_values = multicollinearity_result.get('vif_values', {})
            
            assumptions['multicollinearity'] = {
                'vif_values': vif_values,
                'satisfied': multicollinearity_result['result'].value == 'passed',
                'details': multicollinearity_result.get('details', ''),
                'correlation_matrix': multicollinearity_result.get('correlation_matrix', {}),
                'warnings': multicollinearity_result.get('warnings', []),
                'figures': multicollinearity_result.get('figures', {})
            }
        except Exception as e:
            assumptions['multicollinearity'] = {
                'error': f'Could not test multicollinearity: {str(e)}'
            }
        
        # Linearity test
        try:
            linearity_results = {}
            linearity_test = AssumptionTestKeys.LINEARITY.value["function"]()
            
            for predictor in predictors:
                if predictor != 'Intercept' and predictor in data.columns:
                    # Skip categorical variables
                    if pd.api.types.is_object_dtype(data[predictor]) or pd.api.types.is_categorical_dtype(data[predictor]):
                        continue
                        
                    try:
                        linearity_result = linearity_test.run_test(x=data[predictor], y=data[outcome])
                        
                        linearity_results[predictor] = {
                            'r_squared': float(linearity_result.get('r_squared', 0)),
                            'pearson_r': float(linearity_result.get('pearson_r', 0)),
                            'pearson_p': float(linearity_result.get('pearson_p', 1)),
                            'satisfied': linearity_result['result'].value == 'passed',
                            'details': linearity_result.get('details', '')
                        }
                    except Exception as inner_e:
                        linearity_results[predictor] = {
                            'error': f'Could not test linearity for {predictor}: {str(inner_e)}'
                        }
            
            # Overall linearity assessment
            valid_results = [result for result in linearity_results.values() if 'satisfied' in result]
            if valid_results:
                overall_satisfied = all(result['satisfied'] for result in valid_results)
            else:
                overall_satisfied = None
                
            assumptions['linearity'] = {
                'results': linearity_results,
                'overall_satisfied': overall_satisfied
            }
        except Exception as e:
            assumptions['linearity'] = {
                'error': f'Could not test linearity: {str(e)}'
            }
        
        # Autocorrelation test
        try:
            autocorrelation_test = AssumptionTestKeys.AUTOCORRELATION.value["function"]()
            autocorrelation_result = autocorrelation_test.run_test(residuals=residuals)
            
            assumptions['autocorrelation'] = {
                'test': autocorrelation_result.get('test_used', 'Durbin-Watson'),
                'statistic': float(autocorrelation_result.get('statistic', 0)),
                'p_value': float(autocorrelation_result.get('p_value', 1)) if 'p_value' in autocorrelation_result else None,
                'satisfied': autocorrelation_result['result'].value == 'passed',
                'details': autocorrelation_result.get('details', ''),
                'warnings': autocorrelation_result.get('warnings', []),
                'figures': autocorrelation_result.get('figures', {})
            }
        except Exception as e:
            assumptions['autocorrelation'] = {
                'error': f'Could not test autocorrelation: {str(e)}'
            }
        
        # Outliers test
        try:
            outlier_test = AssumptionTestKeys.OUTLIERS.value["function"]()
            outlier_result = outlier_test.run_test(data=residuals)
            
            assumptions['outliers'] = {
                'test': outlier_result.get('test_used', 'Z-Score'),
                'outliers': outlier_result.get('outliers', []),
                'satisfied': outlier_result['result'].value == 'passed',
                'details': outlier_result.get('details', ''),
                'warnings': outlier_result.get('warnings', []),
                'figures': outlier_result.get('figures', {})
            }
        except Exception as e:
            assumptions['outliers'] = {
                'error': f'Could not test for outliers: {str(e)}'
            }
        
        # Independence test
        try:
            independence_test = AssumptionTestKeys.INDEPENDENCE.value["function"]()
            independence_result = independence_test.run_test(data=residuals)
            
            assumptions['independence'] = {
                'message': independence_result.get('message', ''),
                'satisfied': independence_result['result'].value == 'passed',
                'details': independence_result.get('details', {})
            }
        except Exception as e:
            assumptions['independence'] = {
                'error': f'Could not test independence: {str(e)}'
            }
        
        # Influential points test
        try:
            # Check if the required parameters are available
            if hasattr(influence, 'hat_matrix_diag') and hasattr(results, 'resid') and hasattr(results, 'fittedvalues'):
                X_matrix = sm.add_constant(data[predictors]) if 'Intercept' not in predictors else data[predictors]
                influential_test = AssumptionTestKeys.INFLUENTIAL_POINTS.value["function"]()
                influential_result = influential_test.run_test(
                    residuals=residuals, 
                    leverage=influence.hat_matrix_diag,
                    fitted=predicted,
                    X=X_matrix
                )
                
                assumptions['influential_points'] = {
                    'details': influential_result.get('details', ''),
                    'influential_points': influential_result.get('influential_points', {}),
                    'satisfied': influential_result['result'].value == 'passed',
                    'warnings': influential_result.get('warnings', []),
                    'figures': influential_result.get('figures', {})
                }
            else:
                # Calculate Cook's distance as fallback
                cooks_d = influence.cooks_distance[0]
                
                # Identify influential observations (Cook's D > 4/n)
                threshold = 4 / len(data)
                influential_indices = np.where(cooks_d > threshold)[0]
                
                # Extract results
                assumptions['influential_observations'] = {
                    'n_influential': int(len(influential_indices)),
                    'threshold': float(threshold),
                    'max_cooks_d': float(np.max(cooks_d)),
                    'influential_indices': influential_indices.tolist() if len(influential_indices) < 20 else influential_indices[:20].tolist(),
                    'satisfied': len(influential_indices) == 0,
                    'details': f"Found {len(influential_indices)} influential observations with Cook's distance > {threshold:.5f}"
                }
        except Exception as e:
            assumptions['influential_points'] = {
                'error': f'Could not check for influential points: {str(e)}'
            }
        
        # Model specification test
        try:
            X_matrix = sm.add_constant(data[predictors]) if 'Intercept' not in predictors else data[predictors]
            model_spec_test = AssumptionTestKeys.MODEL_SPECIFICATION.value["function"]()
            model_spec_result = model_spec_test.run_test(
                residuals=residuals,
                fitted=predicted,
                X=X_matrix
            )
            
            assumptions['model_specification'] = {
                'test': model_spec_result.get('test_used', 'RESET'),
                'statistic': float(model_spec_result.get('statistic', 0)),
                'p_value': float(model_spec_result.get('p_value', 1)),
                'satisfied': model_spec_result['result'].value == 'passed',
                'details': model_spec_result.get('details', ''),
                'warnings': model_spec_result.get('warnings', []),
                'figures': model_spec_result.get('figures', {})
            }
        except Exception as e:
            assumptions['model_specification'] = {
                'error': f'Could not test model specification: {str(e)}'
            }
        
        # Sample size test
        try:
            sample_size_test = AssumptionTestKeys.SAMPLE_SIZE.value["function"]()
            # Use a common rule of thumb: at least 10-15 observations per predictor
            min_recommended = len(predictors) * 15
            sample_size_result = sample_size_test.run_test(
                data=data[outcome],
                min_recommended=min_recommended
            )
            
            assumptions['sample_size'] = {
                'sample_size': int(sample_size_result.get('sample_size', 0)),
                'minimum_required': int(sample_size_result.get('minimum_required', 0)),
                'power': float(sample_size_result.get('power', 0)) if 'power' in sample_size_result else None,
                'satisfied': sample_size_result['result'].value == 'passed',
                'details': sample_size_result.get('details', ''),
                'warnings': sample_size_result.get('warnings', [])
            }
        except Exception as e:
            assumptions['sample_size'] = {
                'error': f'Could not test sample size requirements: {str(e)}'
            }
        
        # Goodness of fit test
        try:
            gof_test = AssumptionTestKeys.GOODNESS_OF_FIT.value["function"]()
            gof_result = gof_test.run_test(
                observed=data[outcome],
                expected=predicted
            )
            
            assumptions['goodness_of_fit'] = {
                'test': gof_result.get('test_used', 'Chi-Square'),
                'statistic': float(gof_result.get('statistic', 0)),
                'p_value': float(gof_result.get('p_value', 1)),
                'satisfied': gof_result['result'].value == 'passed',
                'details': gof_result.get('details', ''),
                'warnings': gof_result.get('warnings', []),
                'figures': gof_result.get('figures', {})
            }
        except Exception as e:
            assumptions['goodness_of_fit'] = {
                'error': f'Could not test goodness of fit: {str(e)}'
            }
        
        # Generate prediction statistics
        # Calculate prediction metrics with actual vs predicted data
        y_true = data[outcome]
        y_pred = predicted
        
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        
        # Calculate cross-validated prediction metrics
        try:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            
            # Prepare X and y
            X = data[predictors].copy() if all(p in data.columns for p in predictors) else None
            y = data[outcome]
            
            if X is not None and 'Intercept' not in X.columns:
                X['Intercept'] = 1
                
            # Get cross-validated predictions
            if X is not None:
                cv_pred = cross_val_predict(sm.OLS(y, X).fit(), X, y, cv=cv)
                
                cv_mae = float(mean_absolute_error(y, cv_pred))
                cv_mse = float(mean_squared_error(y, cv_pred))
                cv_rmse = float(np.sqrt(cv_mse))
                cv_r2 = float(r2_score(y, cv_pred))
                
                # Calculate model optimism (difference between training and CV performance)
                optimism_r2 = r_squared - cv_r2
                optimism_rmse = cv_rmse - rmse
                
                cv_results = {
                    'cv_mae': cv_mae,
                    'cv_mse': cv_mse,
                    'cv_rmse': cv_rmse,
                    'cv_r2': cv_r2,
                    'optimism_r2': optimism_r2,
                    'optimism_rmse': optimism_rmse
                }
            else:
                cv_results = {
                    'error': 'Could not perform cross-validation. Some predictors not found in data.'
                }
        except Exception as e:
            cv_results = {
                'error': f'Could not perform cross-validation: {str(e)}'
            }
            
        # Prepare prediction statistics
        prediction_stats = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'cross_validation': cv_results
        }
        
        # Create an interpretation of the results
        interpretation = f"Linear Regression Analysis with {outcome} as the outcome and {len(predictors)} predictors.\n\n"
        
        # Overall model performance
        interpretation += f"Model Performance:\n"
        interpretation += f"- R² = {r_squared:.3f}, Adjusted R² = {adj_r_squared:.3f}\n"
        interpretation += f"- F({results.df_model:.0f}, {results.df_resid:.0f}) = {f_statistic:.3f}, p = {f_pvalue:.5f}\n"
        interpretation += f"- Root Mean Square Error (RMSE) = {rmse:.3f}\n"
        
        # Is the model significant?
        if f_pvalue < alpha:
            interpretation += "- The model is statistically significant.\n"
        else:
            interpretation += "- The model is not statistically significant.\n"
            
        # How much variance is explained?
        if r_squared < 0.3:
            interpretation += "- The model explains a relatively small proportion of the variance in the outcome.\n"
        elif r_squared < 0.6:
            interpretation += "- The model explains a moderate proportion of the variance in the outcome.\n"
        else:
            interpretation += "- The model explains a large proportion of the variance in the outcome.\n"
            
        # Significant predictors
        significant_predictors = [c for c in coefficients if c['significant'] and c['name'] != 'Intercept']
        if significant_predictors:
            interpretation += f"\nSignificant Predictors:\n"
            for coef in significant_predictors:
                interpretation += f"- {coef['name']}: b = {coef['coef']:.3f}, t = {coef['t_value']:.3f}, p = {coef['p_value']:.5f}\n"
                
                # Direction of effect
                if coef['coef'] > 0:
                    interpretation += f"  For each unit increase in {coef['name']}, {outcome} increases by {coef['coef']:.3f} units.\n"
                else:
                    interpretation += f"  For each unit increase in {coef['name']}, {outcome} decreases by {abs(coef['coef']):.3f} units.\n"
                    
                # Add standardized coefficient if available
                if coef.get('standardized_coef') is not None:
                    interpretation += f"  Standardized coefficient (beta) = {coef['standardized_coef']:.3f}\n"
        else:
            interpretation += "\nNo significant predictors were found in the model.\n"
            
        # Cross-validation results
        if 'cross_validation' in prediction_stats and 'cv_r2' in prediction_stats['cross_validation']:
            cv = prediction_stats['cross_validation']
            interpretation += f"\nCross-Validation Results (5-fold):\n"
            interpretation += f"- CV R² = {cv['cv_r2']:.3f}\n"
            interpretation += f"- CV RMSE = {cv['cv_rmse']:.3f}\n"
            
            # Assess optimism
            if cv['optimism_r2'] > 0.1:
                interpretation += "- There is evidence of overfitting as the model performs substantially worse in cross-validation.\n"
            elif cv['optimism_r2'] < 0.05:
                interpretation += "- The model generalizes well with minimal overfitting.\n"
                
        # Assumption tests
        interpretation += "\nAssumption Tests:\n"
        
        # Normality
        if 'normality' in assumptions and 'satisfied' in assumptions['normality']:
            norm_result = assumptions['normality']
            interpretation += f"- Normality of residuals: {norm_result['test']} test, p = {norm_result['p_value']:.5f}, "
            
            if norm_result['satisfied']:
                interpretation += "assumption satisfied.\n"
            else:
                interpretation += f"assumption potentially violated. {norm_result.get('details', '')}\n"
                interpretation += "  Consider using robust standard errors or transforming the outcome variable.\n"
                
        # Homoscedasticity
        if 'homoscedasticity' in assumptions and 'satisfied' in assumptions['homoscedasticity']:
            homo_result = assumptions['homoscedasticity']
            interpretation += f"- Homoscedasticity: {homo_result['test']} test, p = {homo_result['p_value']:.5f}, "
            
            if homo_result['satisfied']:
                interpretation += "assumption satisfied.\n"
            else:
                interpretation += f"assumption potentially violated. {homo_result.get('details', '')}\n"
                interpretation += "  Consider using heteroscedasticity-robust standard errors or transforming variables.\n"
                
        # Linearity
        if 'linearity' in assumptions and 'overall_satisfied' in assumptions['linearity']:
            lin_result = assumptions['linearity']
            interpretation += "- Linearity: "
            
            if lin_result['overall_satisfied'] is True:
                interpretation += "The relationships between predictors and outcome appear to be linear.\n"
            elif lin_result['overall_satisfied'] is False:
                # Find non-linear predictors
                nonlinear_predictors = [pred for pred, res in lin_result['results'].items() 
                                      if 'satisfied' in res and not res['satisfied']]
                
                if nonlinear_predictors:
                    interpretation += f"Potential non-linear relationships detected for: {', '.join(nonlinear_predictors)}. "
                    interpretation += "Consider transformations or polynomial terms.\n"
                else:
                    interpretation += "Some relationships may be non-linear. Examine partial regression plots.\n"
            else:
                interpretation += "Could not fully assess linearity.\n"
                
        # Autocorrelation
        if 'autocorrelation' in assumptions and 'satisfied' in assumptions['autocorrelation']:
            auto_result = assumptions['autocorrelation']
            interpretation += f"- Autocorrelation: {auto_result['test']} test, statistic = {auto_result['statistic']:.3f}, "
            
            if auto_result['satisfied']:
                interpretation += "no significant autocorrelation detected.\n"
            else:
                interpretation += f"potential autocorrelation detected. {auto_result.get('details', '')}\n"
                interpretation += "  Consider time series methods or adding lagged variables if data is time-dependent.\n"
                
        # Multicollinearity
        if 'multicollinearity' in assumptions and 'satisfied' in assumptions['multicollinearity']:
            multi_result = assumptions['multicollinearity']
            interpretation += "- Multicollinearity: "
            
            if multi_result['satisfied']:
                interpretation += "No significant multicollinearity detected among predictors.\n"
            else:
                # Find high VIF predictors
                high_vif_predictors = [f"{pred} (VIF={vif:.1f})" for pred, vif in multi_result['vif_values'].items() 
                                     if vif > 5]
                
                if high_vif_predictors:
                    interpretation += f"Potential multicollinearity issues with: {', '.join(high_vif_predictors)}. "
                    interpretation += "Consider removing or combining highly correlated predictors.\n"
                else:
                    interpretation += f"{multi_result.get('details', '')}\n"
                    
        # Influential observations
        if 'influential_observations' in assumptions and 'satisfied' in assumptions['influential_observations']:
            infl_result = assumptions['influential_observations']
            interpretation += "- Influential observations: "
            
            if infl_result['satisfied']:
                interpretation += "No influential observations detected.\n"
            else:
                interpretation += f"Found {infl_result['n_influential']} potentially influential observations "
                interpretation += f"(Cook's distance > {infl_result['threshold']:.5f}). "
                interpretation += "Consider examining these cases and potentially refitting the model without extreme outliers.\n"
        
        # Additional guidance
        interpretation += "\nAdditional Guidance:\n"
        
        # Guidance based on model performance
        if r_squared < 0.2:
            interpretation += "- The model explains relatively little variance. Consider including additional relevant predictors.\n"
        
        # Guidance based on assumption violations
        has_violations = False
        if ('normality' in assumptions and 'satisfied' in assumptions['normality'] and not assumptions['normality']['satisfied']) or \
           ('homoscedasticity' in assumptions and 'satisfied' in assumptions['homoscedasticity'] and not assumptions['homoscedasticity']['satisfied']) or \
           ('linearity' in assumptions and 'overall_satisfied' in assumptions['linearity'] and assumptions['linearity']['overall_satisfied'] is False) or \
           ('autocorrelation' in assumptions and 'satisfied' in assumptions['autocorrelation'] and not assumptions['autocorrelation']['satisfied']):
            has_violations = True
        
        if has_violations:
            interpretation += "- Due to assumption violations, consider these options:\n"
            interpretation += "  1. Use robust regression techniques\n"
            interpretation += "  2. Transform variables to better meet assumptions\n"
            interpretation += "  3. Use bootstrapping to obtain robust confidence intervals\n"
            interpretation += "  4. Consider non-parametric regression alternatives\n"
        
        # Guidance based on influential observations
        if 'influential_observations' in assumptions and 'satisfied' in assumptions['influential_observations'] and not assumptions['influential_observations']['satisfied']:
            interpretation += "- Examine the identified influential observations to determine if they represent errors or valid extreme cases.\n"
            
        # Diagnostic plot interpretation guidance
        interpretation += "\nDiagnostic Plot Interpretation Guide:\n"
        interpretation += "- Residuals vs Fitted: Points should be randomly scattered around the horizontal line at y=0. Patterns suggest non-linearity.\n"
        interpretation += "- Q-Q Plot: Points following the line suggest normally distributed residuals. Deviations suggest non-normality.\n"
        interpretation += "- Scale-Location: Horizontal band with points equally spread suggests homoscedasticity. Patterns suggest heteroscedasticity.\n"
        interpretation += "- Cook's Distance: Points above the threshold line identify influential observations that may distort the results.\n"
        interpretation += "- Partial Regression Plots: Show the relationship between each predictor and the outcome after controlling for other predictors.\n"
        interpretation += "- Actual vs Predicted: Points close to the diagonal line indicate good prediction accuracy.\n"
        
        # Final assessment
        confidence_level = "high"
        if has_violations:
            confidence_level = "moderate"
            
        if 'influential_observations' in assumptions and 'satisfied' in assumptions['influential_observations'] and not assumptions['influential_observations']['satisfied']:
            confidence_level = "moderate" if confidence_level == "high" else "low"
            
        if 'multicollinearity' in assumptions and 'satisfied' in assumptions['multicollinearity'] and not assumptions['multicollinearity']['satisfied']:
            confidence_level = "moderate" if confidence_level == "high" else "low"
            
        interpretation += f"\nOverall confidence in model results: {confidence_level}\n"
        
        # Create full results dictionary
        results_dict = {
            'test': 'Linear Regression',
            'coefficients': coefficients,
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
            'assumptions': assumptions,
            'prediction_stats': prediction_stats,
            'formula': formula,
            'summary': str(results.summary()),
            'interpretation': interpretation,
            'figures': figures
        }
        
        return results_dict
    except Exception as e:
        return {
            'test': 'Linear Regression',
            'error': str(e),
            'traceback': traceback.format_exc()
        }