import traceback
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Dict, List, Any, Optional
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy import dmatrices
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data.selection.detailed_tests.formatting import fig_to_svg, PASTEL_COLORS, PASTEL_CMAP
from data.assumptions.format import AssumptionTestKeys  
from data.assumptions.tests import (
    NormalityTest, HomogeneityOfVarianceTest, MulticollinearityTest, 
    LinearityTest, AutocorrelationTest, OutlierTest, OverdispersionTest,
    HomoscedasticityTest, SampleSizeTest, InfluentialPointsTest, 
    ModelSpecificationTest, ZeroInflationTest, IndependenceTest
)
from data.assumptions.tests import AssumptionResult

def poisson_regression(data: pd.DataFrame, outcome: str, predictors: List[str], alpha: float) -> Dict[str, Any]:
    """
    Performs Poisson Regression with comprehensive statistics, assumption checks, and visualizations.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataset containing all variables
    outcome : str
        Name of the count outcome variable
    predictors : List[str]
        List of predictor variable names
    alpha : float
        Significance level
        
    Returns:
    --------
    Dict[str, Any]
        Comprehensive results including model statistics, diagnostics, and visualizations
    """
    try:
        # Validate that outcome contains count data
        if not np.issubdtype(data[outcome].dtype, np.integer):
            # Try to convert to integer if possible
            try:
                data[outcome] = data[outcome].astype(int)
            except:
                raise ValueError(f"Outcome variable {outcome} must contain count data (integers)")
                
        # Check if data contains negative values
        if data[outcome].min() < 0:
            raise ValueError(f"Outcome variable {outcome} contains negative values, which are not valid for count data")
            
        # Construct formula
        formula = f"{outcome} ~ {' + '.join(predictors)}"
        
        # Fit the Poisson model
        model = smf.poisson(formula, data)
        results = model.fit()
        
        # Also fit a negative binomial model if statsmodels supports it
        try:
            nb_model = smf.negativebinomial(formula, data)
            nb_results = nb_model.fit(disp=0)
            has_nb_model = True
        except:
            has_nb_model = False
        
        # Extract model parameters
        aic = float(results.aic)
        bic = float(results.bic)
        log_likelihood = float(results.llf)
        
        # Extract coefficients
        coefficients = {}
        for term in results.params.index:
            # Get confidence intervals
            conf_int = results.conf_int()
            
            coefficients[term] = {
                'estimate': float(results.params[term]),
                'std_error': float(results.bse[term]),
                'z_value': float(results.tvalues[term]),
                'p_value': float(results.pvalues[term]),
                'significant': results.pvalues[term] < alpha,
                'irr': float(np.exp(results.params[term])),
                'ci_lower': float(np.exp(conf_int.loc[term, 0])),
                'ci_upper': float(np.exp(conf_int.loc[term, 1]))
            }
            
        # Calculate model fit statistics
        # Null model (intercept only)
        try:
            null_formula = f"{outcome} ~ 1"
            null_model = smf.poisson(null_formula, data)
            null_results = null_model.fit()
            
            # Likelihood ratio test
            lr_stat = 2 * (results.llf - null_results.llf)
            df = len(predictors)
            lr_pval = stats.chi2.sf(lr_stat, df)
            
            # McFadden's pseudo R-squared
            mcfadden_r2 = 1 - (results.llf / null_results.llf)
            
            # Additional pseudo R² measures
            # Cox & Snell R²
            cox_snell_r2 = 1 - np.exp((2 * null_results.llf - 2 * results.llf) / len(data))
            
            # Nagelkerke/Cragg & Uhler R²
            nagelkerke_r2 = cox_snell_r2 / (1 - np.exp(2 * null_results.llf / len(data)))
            
            model_fit = {
                'log_likelihood': float(results.llf),
                'null_log_likelihood': float(null_results.llf),
                'lr_statistic': float(lr_stat),
                'lr_df': int(df),
                'lr_p_value': float(lr_pval),
                'mcfadden_r2': float(mcfadden_r2),
                'cox_snell_r2': float(cox_snell_r2),
                'nagelkerke_r2': float(nagelkerke_r2),
                'aic': aic,
                'bic': bic,
                'significant': lr_pval < alpha
            }
        except Exception as e:
            model_fit = {
                'log_likelihood': float(results.llf),
                'aic': aic,
                'bic': bic,
                'error': f'Could not fit null model for comparison: {str(e)}'
            }
            
        # Calculate dispersion
        y_pred = results.predict()
        pearson_chi2 = np.sum((data[outcome] - y_pred)**2 / y_pred)
        dispersion = pearson_chi2 / results.df_resid
        
        # Test for overdispersion 
        # Dean's test for overdispersion
        try:
            residuals = data[outcome] - y_pred
            pearson_residuals = residuals / np.sqrt(y_pred)
            auxiliary_model = sm.OLS(pearson_residuals**2 - 1, y_pred).fit()
            t_statistic = auxiliary_model.params.iloc[0] / auxiliary_model.bse.iloc[0]
            p_value = 2 * (1 - stats.norm.cdf(abs(t_statistic)))
            
            overdispersion_test = {
                'method': "Dean's test",
                't_statistic': float(t_statistic),
                'p_value': float(p_value),
                'significant': p_value < alpha,
                'dispersion_parameter': float(dispersion),
                'has_overdispersion': dispersion > 1.5  # Common rule of thumb
            }
        except Exception as e:
            overdispersion_test = {
                'method': 'Dispersion parameter',
                'dispersion_parameter': float(dispersion),
                'has_overdispersion': dispersion > 1.5,
                'error': f'Could not perform Dean\'s test: {str(e)}'
            }
            
        # If we have fit a negative binomial model, compare it to Poisson
        if has_nb_model:
            try:
                # Check if the nb_results has valid parameters before proceeding
                if hasattr(nb_results, 'params') and hasattr(nb_results, 'bse') and nb_results.params is not None and nb_results.bse is not None:
                    # Likelihood ratio test for dispersion
                    lr_stat_disp = 2 * (nb_results.llf - results.llf)
                    df_disp = 1  # Testing one parameter (alpha)
                    lr_pval_disp = stats.chi2.sf(lr_stat_disp, df_disp)
                    
                    # Add to overdispersion test results
                    overdispersion_test.update({
                        'nb_comparison': {
                            'lr_statistic': float(lr_stat_disp),
                            'p_value': float(lr_pval_disp),
                            'significant': lr_pval_disp < alpha,
                            'poisson_aic': float(results.aic),
                            'nb_aic': float(nb_results.aic),
                            'prefer_nb': nb_results.aic < results.aic
                        }
                    })
                else:
                    overdispersion_test['nb_comparison'] = {
                        'error': 'Negative Binomial model failed to converge properly'
                    }
            except Exception as e:
                overdispersion_test['nb_comparison'] = {
                    'error': f'Could not compare with Negative Binomial model: {str(e)}'
                }
        
        # Test for zero-inflation
        # Simple test: compare observed zeros to predicted zeros
        observed_zeros = (data[outcome] == 0).sum()
        predicted_zeros = (y_pred < 0.5).sum()  # Approximation
        zero_ratio = observed_zeros / predicted_zeros if predicted_zeros > 0 else float('inf')
        
        zero_inflation_test = {
            'observed_zeros': int(observed_zeros),
            'predicted_zeros': int(predicted_zeros),
            'zero_ratio': float(zero_ratio),
            'possible_zero_inflation': zero_ratio > 1.5  # Rule of thumb
        }
        
        # Extract model parameters and prepare data
        residuals = data[outcome] - y_pred
        pearson_residuals = residuals / np.sqrt(y_pred)
        
        # Initialize assumptions dictionary to store all test results
        assumptions = {}
        
        # Test for overdispersion
        try:
            overdispersion_result = OverdispersionTest().run_test(
                observed=data[outcome], 
                predicted=y_pred
            )
            assumptions["overdispersion"] = overdispersion_result
        except Exception as e:
            assumptions["overdispersion"] = {
                "result": AssumptionResult.NOT_APPLICABLE,
                "message": f"Could not test overdispersion: {str(e)}",
                "details": {}
            }
        
        # Test for multicollinearity among predictors
        try:
            predictor_df = data[predictors].copy()
            multicollinearity_result = MulticollinearityTest().run_test(
                df=predictor_df,
                covariates=predictors
            )
            assumptions["multicollinearity"] = multicollinearity_result
        except Exception as e:
            assumptions["multicollinearity"] = {
                "result": AssumptionResult.NOT_APPLICABLE,
                "details": f"Could not test multicollinearity: {str(e)}",
                "vif_values": {},
                "correlation_matrix": None,
                "warnings": [str(e)],
                "figures": {}
            }
        
        # Test for linearity (log-linearity) between predictors and log of outcome
            for pred in predictors:
                if pd.api.types.is_numeric_dtype(data[pred]):
                    try:
                        # For Poisson, we check linearity with log(outcome)
                        # Add a small constant to handle zeros
                        log_outcome = np.log(data[outcome] + 0.01)
                        linearity_result = LinearityTest().run_test(
                            x=data[pred].values,  
                            y=log_outcome.values
                        )
                        assumptions[f"linearity_{pred}"] = linearity_result
                    except Exception as e:
                        assumptions[f"linearity_{pred}"] = {
                            "result": AssumptionResult.NOT_APPLICABLE,
                            "statistic": None,
                            "p_value": None,
                            "details": f"Could not test linearity for {pred}: {str(e)}",
                            "test_used": None,
                            "warnings": [str(e)],
                            "figures": {}
                        }
            
        # Test for influential points
        try:
            # Calculate leverage and Cook's distance
            influence = results.get_influence()
            cooks_d = influence.cooks_distance[0]
            leverage = influence.hat_matrix_diag
            
            # Create X matrix
            X_df = pd.DataFrame(data[predictors])
            
            influential_result = InfluentialPointsTest().run_test(
                residuals=residuals,
                leverage=leverage,
                fitted=y_pred,
                X=X_df
            )
            assumptions["influential_points"] = influential_result
        except Exception as e:
            assumptions["influential_points"] = {
                "result": AssumptionResult.NOT_APPLICABLE,
                "details": f"Could not test for influential points: {str(e)}",
                "influential_points": {},
                "warnings": [str(e)],
                "figures": {}
            }
        
        # Test for zero inflation
        try:
            zero_inflation_result = ZeroInflationTest().run_test(
                data=data[outcome].values
            )
            assumptions["zero_inflation"] = zero_inflation_result
        except Exception as e:
            # Fall back to simple test
            observed_zeros = (data[outcome] == 0).sum()
            predicted_zeros = (y_pred < 0.5).sum()
            zero_ratio = observed_zeros / predicted_zeros if predicted_zeros > 0 else float('inf')
            
            assumptions["zero_inflation"] = {
                "result": AssumptionResult.PASSED if zero_ratio <= 1.5 else AssumptionResult.FAILED,
                "statistic": float(zero_ratio),
                "p_value": None,
                "details": f"Simple zero ratio test: {zero_ratio:.2f}",
                "test_used": "Zero ratio",
                "warnings": ["Used simple zero ratio test due to error in standard test"],
                "figures": {}
            }
        
        # Test for homoscedasticity of Pearson residuals
        try:
            homoscedasticity_result = HomoscedasticityTest().run_test(
                residuals=pearson_residuals,
                predicted=y_pred
            )
            assumptions["homoscedasticity"] = homoscedasticity_result
        except Exception as e:
            assumptions["homoscedasticity"] = {
                "result": AssumptionResult.NOT_APPLICABLE,
                "statistic": None,
                "p_value": None,
                "details": f"Could not test homoscedasticity: {str(e)}",
                "test_used": None,
                "warnings": [str(e)],
                "figures": {}
            }
            
        # Test for residual independence (autocorrelation)
        try:
            independence_result = IndependenceTest().run_test(
                data=residuals
            )
            assumptions["independence"] = independence_result
        except Exception as e:
            assumptions["independence"] = {
                "result": AssumptionResult.NOT_APPLICABLE,
                "message": f"Could not test independence: {str(e)}",
                "statistic": None,
                "details": {}
            }
            
        # Test for sample size adequacy
        try:
            # Rule of thumb: 10 events per predictor for Poisson regression
            min_recommended = 10 * len(predictors)
            sample_size_result = SampleSizeTest().run_test(
                data=data[outcome].values,
                min_recommended=min_recommended
            )
            assumptions["sample_size"] = sample_size_result
        except Exception as e:
            assumptions["sample_size"] = {
                "result": AssumptionResult.NOT_APPLICABLE,
                "details": f"Could not test sample size: {str(e)}",
                "sample_size": len(data),
                "minimum_required": min_recommended,
                "power": None,
                "warnings": [str(e)]
            }
            
        # Test for model specification
        try:
            X_df = pd.DataFrame(data[predictors])
            specification_result = ModelSpecificationTest().run_test(
                residuals=residuals,
                fitted=y_pred,
                X=X_df
            )
            assumptions["model_specification"] = specification_result
        except Exception as e:
            assumptions["model_specification"] = {
                "result": AssumptionResult.NOT_APPLICABLE,
                "statistic": None,
                "p_value": None,
                "details": f"Could not test model specification: {str(e)}",
                "test_used": None,
                "warnings": [str(e)],
                "figures": {}
            }

        # No overall assumption summary - per requirements

        # Calculate prediction statistics
        prediction_stats = {
            'mean_observed': float(data[outcome].mean()),
            'mean_predicted': float(y_pred.mean()),
            'min_predicted': float(y_pred.min()),
            'max_predicted': float(y_pred.max()),
            'correlation_obs_pred': float(np.corrcoef(data[outcome], y_pred)[0, 1]),
            'mean_absolute_error': float(np.mean(np.abs(residuals))),
            'root_mean_squared_error': float(np.sqrt(np.mean(residuals**2))),
            'mean_abs_percentage_error': float(np.mean(np.abs(residuals / np.maximum(data[outcome], 1)))) * 100
        }
        
        # Create K-fold cross-validation if feasible
        try:
            # Simplified cross-validation for demonstration
            X = data[predictors]
            y = data[outcome]
            
            # Create K-fold cross-validation object
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            
            # Collect cross-validation results
            cv_maes = []
            cv_rmses = []
            cv_correlations = []
            
            for train_idx, test_idx in kf.split(X):
                # Split data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Create training dataset with outcome
                train_data = X_train.copy()
                train_data[outcome] = y_train
                
                # Fit model
                cv_model = smf.poisson(formula, train_data).fit()
                
                # Make predictions on test data
                test_data = X_test.copy()
                test_predictions = cv_model.predict(test_data)
                
                # Calculate metrics
                cv_mae = mean_absolute_error(y_test, test_predictions)
                cv_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
                cv_corr = np.corrcoef(y_test, test_predictions)[0, 1]
                
                cv_maes.append(cv_mae)
                cv_rmses.append(cv_rmse)
                cv_correlations.append(cv_corr)
            
            # Calculate average metrics
            avg_cv_mae = np.mean(cv_maes)
            avg_cv_rmse = np.mean(cv_rmses)
            avg_cv_corr = np.mean(cv_correlations)
            
            prediction_stats['cross_validation'] = {
                'k_folds': 5,
                'cv_mae': float(avg_cv_mae),
                'cv_rmse': float(avg_cv_rmse),
                'cv_correlation': float(avg_cv_corr)
            }
        except Exception as e:
            prediction_stats['cross_validation_error'] = str(e)
        
        # Create interpretation section that uses our assumption test results
        significant_predictors = [term for term, coef in coefficients.items() if coef.get('significant', False) and term != 'Intercept']
        
        interpretation = f"Poisson Regression with {outcome} as outcome and predictors ({', '.join(predictors)}).\n\n"
        
        # Interpret model fit
        if 'lr_p_value' in model_fit:
            interpretation += f"The model is {'statistically significant' if model_fit['significant'] else 'not statistically significant'} "
            interpretation += f"compared to the null model (χ²({model_fit['lr_df']}) = {model_fit['lr_statistic']:.3f}, p = {model_fit['lr_p_value']:.5f}).\n"
            
            # Interpret pseudo R²
            if 'mcfadden_r2' in model_fit and 'nagelkerke_r2' in model_fit:
                interpretation += f"Model fit: McFadden's R² = {model_fit['mcfadden_r2']:.3f}, "
                interpretation += f"Nagelkerke's R² = {model_fit['nagelkerke_r2']:.3f}.\n"
                
                if model_fit['mcfadden_r2'] < 0.2:
                    interpretation += "These values suggest a modest fit.\n\n"
                elif model_fit['mcfadden_r2'] < 0.4:
                    interpretation += "These values suggest a reasonable fit.\n\n"
                else:
                    interpretation += "These values suggest a good fit.\n\n"
        
        # Interpret coefficients
        if significant_predictors:
            interpretation += "Significant predictors:\n"
            for term in significant_predictors:
                coef = coefficients[term]
                interpretation += f"- {term}: β = {coef['estimate']:.3f}, IRR = {coef['irr']:.3f} "
                interpretation += f"(95% CI: {coef['ci_lower']:.3f}-{coef['ci_upper']:.3f}), p = {coef['p_value']:.5f}\n"
                
                # Interpret direction of effect
                if coef['irr'] > 1:
                    interpretation += f"  For each unit increase in {term}, the expected count of {outcome} "
                    interpretation += f"increases by {(coef['irr']-1)*100:.1f}%, holding other variables constant.\n"
                else:
                    interpretation += f"  For each unit increase in {term}, the expected count of {outcome} "
                    interpretation += f"decreases by {(1-coef['irr'])*100:.1f}%, holding other variables constant.\n"
        else:
            interpretation += "No significant predictors were found.\n"
            
        # Interpret model performance
        if 'correlation_obs_pred' in prediction_stats:
            interpretation += f"\nModel performance: The correlation between observed and predicted counts is {prediction_stats['correlation_obs_pred']:.3f}.\n"
            interpretation += f"Root mean squared error (RMSE): {prediction_stats['root_mean_squared_error']:.3f}, "
            interpretation += f"Mean absolute error (MAE): {prediction_stats['mean_absolute_error']:.3f}.\n"
            
            # Cross-validation results
            if 'cross_validation' in prediction_stats:
                cv = prediction_stats['cross_validation']
                interpretation += f"5-fold cross-validation results: CV-RMSE = {cv['cv_rmse']:.3f}, "
                interpretation += f"CV-MAE = {cv['cv_mae']:.3f}, CV-Correlation = {cv['cv_correlation']:.3f}.\n"
        
        # Interpret assumptions based on standardized tests
        interpretation += "\nAssumption checks:\n"
        
        # Overdispersion
        if "overdispersion" in assumptions:
            overdispersion = assumptions["overdispersion"]
            if overdispersion.get("result") == AssumptionResult.PASSED:
                interpretation += "- Overdispersion: No significant overdispersion detected. The Poisson model is appropriate.\n"
            elif overdispersion.get("result") == AssumptionResult.WARNING:
                interpretation += "- Overdispersion: Some overdispersion detected. Consider a negative binomial model as an alternative.\n"
            elif overdispersion.get("result") == AssumptionResult.FAILED:
                interpretation += "- Overdispersion: Significant overdispersion detected. A negative binomial model would be more appropriate.\n"
            else:
                interpretation += "- Overdispersion: Could not be reliably tested.\n"
        
        # Zero-inflation
        if "zero_inflation" in assumptions:
            zero_inflation = assumptions["zero_inflation"]
            if zero_inflation.get("result") == AssumptionResult.PASSED:
                interpretation += "- Zero-inflation: No evidence of excess zeros. The Poisson distribution fits the zero counts adequately.\n"
            elif zero_inflation.get("result") == AssumptionResult.WARNING:
                interpretation += "- Zero-inflation: Some evidence of excess zeros. Consider exploring zero-inflated models.\n"
            elif zero_inflation.get("result") == AssumptionResult.FAILED:
                interpretation += "- Zero-inflation: Strong evidence of excess zeros. A zero-inflated Poisson or zero-inflated negative binomial model is recommended.\n"
            else:
                interpretation += "- Zero-inflation: Could not be reliably tested.\n"
        
        # Independence
        if "independence" in assumptions:
            independence = assumptions["independence"]
            if independence.get("result") == AssumptionResult.PASSED:
                interpretation += "- Independence: No significant autocorrelation detected in the residuals.\n"
            elif independence.get("result") == AssumptionResult.WARNING:
                interpretation += "- Independence: Mild autocorrelation detected in the residuals. Results should be interpreted with caution.\n"
            elif independence.get("result") == AssumptionResult.FAILED:
                interpretation += "- Independence: Significant autocorrelation detected in the residuals. Consider time-series models or GEE approaches.\n"
            else:
                interpretation += "- Independence: Could not be reliably tested. The assumption of independent observations is critical for valid inference.\n"
        
        # Multicollinearity
        if "multicollinearity" in assumptions:
            multicollinearity = assumptions["multicollinearity"]
            if multicollinearity.get("result") == AssumptionResult.PASSED:
                interpretation += "- Multicollinearity: No problematic collinearity detected among predictor variables.\n"
            elif multicollinearity.get("result") == AssumptionResult.WARNING:
                interpretation += "- Multicollinearity: Some collinearity detected among predictors. Standard errors may be inflated.\n"
            elif multicollinearity.get("result") == AssumptionResult.FAILED:
                high_vif = []
                if "vif_values" in multicollinearity:
                    high_vif = [k for k, v in multicollinearity["vif_values"].items() if v > 10]
                high_vif_str = ", ".join(high_vif) if high_vif else "some predictors"
                interpretation += f"- Multicollinearity: Severe collinearity detected in {high_vif_str}. Coefficient estimates may be unstable.\n"
            else:
                interpretation += "- Multicollinearity: Could not be reliably tested.\n"
        
        # Linearity tests for individual predictors
        linearity_keys = [k for k in assumptions.keys() if k.startswith("linearity_")]
        if linearity_keys:
            linearity_issues = []
            for key in linearity_keys:
                pred_name = key.replace("linearity_", "")
                linearity_test = assumptions[key]
                if linearity_test.get("result") == AssumptionResult.FAILED:
                    linearity_issues.append(pred_name)
            
            if not linearity_issues:
                interpretation += "- Log-linearity: The relationship between predictors and log(outcome) appears linear.\n"
            else:
                issues_str = ", ".join(linearity_issues)
                interpretation += f"- Log-linearity: Non-linear relationships detected for: {issues_str}. Consider transformations or more flexible models.\n"
        
        # Influential points
        if "influential_points" in assumptions:
            influential = assumptions["influential_points"]
            if influential.get("result") == AssumptionResult.PASSED:
                interpretation += "- Influential observations: No influential points detected that substantially impact the model.\n"
            elif influential.get("result") == AssumptionResult.WARNING:
                interpretation += "- Influential observations: Some potentially influential points detected. Consider robust estimation methods.\n"
            elif influential.get("result") == AssumptionResult.FAILED:
                influential_count = 0
                if "influential_points" in influential and isinstance(influential["influential_points"], dict):
                    influential_count = len(influential["influential_points"])
                interpretation += f"- Influential observations: {influential_count} influential points detected that may substantially alter model results.\n"
            else:
                interpretation += "- Influential observations: Could not be reliably tested.\n"
        
        # Homoscedasticity
        if "homoscedasticity" in assumptions:
            homoscedasticity = assumptions["homoscedasticity"]
            if homoscedasticity.get("result") == AssumptionResult.PASSED:
                interpretation += "- Homoscedasticity: Residuals show approximately constant variance across predicted values.\n"
            elif homoscedasticity.get("result") == AssumptionResult.WARNING:
                interpretation += "- Homoscedasticity: Some heteroscedasticity detected in residuals. Standard errors may be affected.\n"
            elif homoscedasticity.get("result") == AssumptionResult.FAILED:
                interpretation += "- Homoscedasticity: Significant heteroscedasticity detected. Consider robust standard errors or a different link function.\n"
            else:
                interpretation += "- Homoscedasticity: Could not be reliably tested.\n"
        
        # Sample size
        if "sample_size" in assumptions:
            sample_size = assumptions["sample_size"]
            if sample_size.get("result") == AssumptionResult.PASSED:
                interpretation += "- Sample size: The sample size is adequate for reliable parameter estimation.\n"
            elif sample_size.get("result") == AssumptionResult.WARNING:
                interpretation += "- Sample size: The sample size is smaller than ideal. Results should be interpreted with caution.\n"
            elif sample_size.get("result") == AssumptionResult.FAILED:
                sample_size_val = sample_size.get("sample_size", "unknown")
                min_required = sample_size.get("minimum_required", "unknown")
                interpretation += f"- Sample size: Insufficient sample size ({sample_size_val}, minimum recommended: {min_required}). Results may be unreliable.\n"
            else:
                interpretation += "- Sample size: Could not be reliably tested.\n"
        
        # Model specification
        if "model_specification" in assumptions:
            specification = assumptions["model_specification"]
            if specification.get("result") == AssumptionResult.PASSED:
                interpretation += "- Model specification: No evidence of model misspecification detected.\n"
            elif specification.get("result") == AssumptionResult.WARNING:
                interpretation += "- Model specification: Some evidence of model misspecification. Consider adding additional predictors or interactions.\n"
            elif specification.get("result") == AssumptionResult.FAILED:
                interpretation += "- Model specification: Significant evidence of model misspecification. Important variables may be missing or relationships may be misspecified.\n"
            else:
                interpretation += "- Model specification: Could not be reliably tested.\n"
                
        # Overall recommendation based on assumption test results
        failed_assumptions = [k for k, v in assumptions.items() if v.get("result") == AssumptionResult.FAILED]
        warning_assumptions = [k for k, v in assumptions.items() if v.get("result") == AssumptionResult.WARNING]
        
        interpretation += "\nSummary and Recommendations:\n"
        
        if not failed_assumptions and not warning_assumptions:
            interpretation += "All tested assumptions appear to be satisfied. The Poisson regression model is appropriate for this data.\n"
        elif "overdispersion" in failed_assumptions:
            interpretation += "Due to significant overdispersion, a negative binomial regression model would be more appropriate than Poisson regression.\n"
        elif "zero_inflation" in failed_assumptions:
            interpretation += "Due to excess zeros, a zero-inflated Poisson or zero-inflated negative binomial model would be more appropriate.\n"
        elif failed_assumptions:
            failed_str = ', '.join([a.replace('_', ' ') for a in failed_assumptions])
            interpretation += f"The following assumption(s) are violated: {failed_str}. Consider alternative modeling approaches or data transformations.\n"
        elif warning_assumptions:
            interpretation += "Some assumptions show minor violations. Results should be interpreted with caution, but the model may still be useful.\n"
        
        # Create visualizations
        figures = {}
        
        # Figure 1: Observed vs. Predicted counts
        try:
            fig1, ax1 = plt.subplots(figsize=(10, 7))
            
            # Create scatter plot of observed vs predicted
            ax1.scatter(y_pred, data[outcome], alpha=0.6, color=PASTEL_COLORS[0])
            
            # Add reference line (y=x)
            min_val = min(y_pred.min(), data[outcome].min())
            max_val = max(y_pred.max(), data[outcome].max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            # Add labels and title
            ax1.set_xlabel('Predicted Count', fontsize=12)
            ax1.set_ylabel('Observed Count', fontsize=12)
            ax1.set_title('Observed vs. Predicted Counts', fontsize=14)
            
            # Add correlation annotation
            corr = prediction_stats['correlation_obs_pred']
            ax1.annotate(f'Correlation: {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                       ha='left', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add RMSE and MAE
            rmse = prediction_stats['root_mean_squared_error']
            mae = prediction_stats['mean_absolute_error']
            ax1.annotate(f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}', xy=(0.05, 0.85), xycoords='axes fraction',
                       ha='left', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            fig1.tight_layout()
            figures['observed_vs_predicted'] = fig_to_svg(fig1)
        except Exception as e:
            figures['observed_vs_predicted_error'] = str(e)
        
        # Figure 2: Pearson Residuals Plot
        try:
            fig2, ax2 = plt.subplots(figsize=(10, 7))
            
            # Create scatter plot of Pearson residuals vs predicted
            ax2.scatter(y_pred, pearson_residuals, alpha=0.6, color=PASTEL_COLORS[1])
            
            # Add horizontal reference line at y=0
            ax2.axhline(y=0, color='r', linestyle='--')
            
            # Add horizontal reference lines at +/-2 (approximate 95% bounds for Pearson residuals)
            ax2.axhline(y=2, color='r', linestyle=':', alpha=0.5)
            ax2.axhline(y=-2, color='r', linestyle=':', alpha=0.5)
            
            # Add labels and title
            ax2.set_xlabel('Predicted Count', fontsize=12)
            ax2.set_ylabel('Pearson Residuals', fontsize=12)
            ax2.set_title('Pearson Residuals vs. Predicted Counts', fontsize=14)
            
            # Add dispersion annotation
            ax2.annotate(f'Dispersion parameter: {dispersion:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                       ha='left', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add overdispersion assessment
            if overdispersion_test['has_overdispersion']:
                overdispersion_text = 'Evidence of overdispersion'
            else:
                overdispersion_text = 'No strong evidence of overdispersion'
                
            ax2.annotate(overdispersion_text, xy=(0.05, 0.85), xycoords='axes fraction',
                       ha='left', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            fig2.tight_layout()
            figures['pearson_residuals'] = fig_to_svg(fig2)
        except Exception as e:
            figures['pearson_residuals_error'] = str(e)
        
        # Figure 3: Forest plot of IRRs
        try:
            # Extract predictor names (excluding intercept)
            predictor_names = [name for name in coefficients.keys() if name != 'Intercept']
            
            # If we have predictors, create a forest plot
            if predictor_names:
                fig3, ax3 = plt.subplots(figsize=(10, max(6, len(predictor_names) * 0.5)))
                
                # Extract IRRs and CIs
                irrs = [coefficients[name]['irr'] for name in predictor_names]
                ci_lowers = [coefficients[name]['ci_lower'] for name in predictor_names]
                ci_uppers = [coefficients[name]['ci_upper'] for name in predictor_names]
                
                # Calculate error bar sizes
                errors_minus = [irr - ci_l for irr, ci_l in zip(irrs, ci_lowers)]
                errors_plus = [ci_u - irr for irr, ci_u in zip(irrs, ci_uppers)]
                
                # Set colors based on significance
                significant = [coefficients[name]['significant'] for name in predictor_names]
                colors = [PASTEL_COLORS[0] if sig else PASTEL_COLORS[3] for sig in significant]
                
                # Create forest plot
                y_pos = np.arange(len(predictor_names))
                ax3.errorbar(irrs, y_pos, xerr=[errors_minus, errors_plus], fmt='none', capsize=5, ecolor='black')
                
                # Add scatter points with colors
                for i, (x, y, color) in enumerate(zip(irrs, y_pos, colors)):
                    ax3.scatter(x, y, color=color, s=100, zorder=10)
                
                # Add predictor labels
                ax3.set_yticks(y_pos)
                ax3.set_yticklabels(predictor_names)
                
                # Add vertical line at IRR=1 (no effect)
                ax3.axvline(x=1, color='red', linestyle='--', alpha=0.7)
                
                # Add title and labels
                ax3.set_title('Incidence Rate Ratios (IRR) with 95% CI', fontsize=14)
                ax3.set_xlabel('IRR (log scale)', fontsize=12)
                
                # Add IRR values as text
                for i, (irr, name) in enumerate(zip(irrs, predictor_names)):
                    p_val = coefficients[name]['p_value']
                    sig_str = '*' if p_val < alpha else ''
                    ax3.text(irr * 1.1, i, f" {irr:.3f} {sig_str}", va='center')
                
                # Set logarithmic scale for x-axis if range is large
                min_ci = min(ci_lowers)
                max_ci = max(ci_uppers)
                
                if max_ci / min_ci > 5:
                    ax3.set_xscale('log')
                    ax3.set_xlim(min_ci / 2, max_ci * 2)
                else:
                    ax3.set_xlim(min(0.5, min_ci / 2), max(1.5, max_ci * 1.5))
                
                # Add legend for significance
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=PASTEL_COLORS[0], markersize=10, label='Significant'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=PASTEL_COLORS[3], markersize=10, label='Non-significant')
                ]
                ax3.legend(handles=legend_elements, loc='upper right')
                
                # Add grid
                ax3.grid(True, alpha=0.3)
                
                fig3.tight_layout()
                figures['irr_forest_plot'] = fig_to_svg(fig3)
        except Exception as e:
            figures['irr_forest_plot_error'] = str(e)
        
        # Figure 4: Distribution of observed vs. predicted counts
        try:
            fig4, ax4 = plt.subplots(figsize=(10, 7))
            
            # Create histograms
            bins = min(30, int(data[outcome].max() - data[outcome].min() + 1))
            ax4.hist(data[outcome], bins=bins, alpha=0.5, label='Observed', color=PASTEL_COLORS[0])
            ax4.hist(y_pred, bins=bins, alpha=0.5, label='Predicted', color=PASTEL_COLORS[1])
            
            # Add vertical lines for means
            ax4.axvline(x=data[outcome].mean(), color='blue', linestyle='--', 
                      label=f'Observed Mean = {data[outcome].mean():.3f}')
            ax4.axvline(x=y_pred.mean(), color='orange', linestyle='--', 
                      label=f'Predicted Mean = {y_pred.mean():.3f}')
            
            # Add labels and title
            ax4.set_xlabel('Count', fontsize=12)
            ax4.set_ylabel('Frequency', fontsize=12)
            ax4.set_title('Distribution of Observed and Predicted Counts', fontsize=14)
            
            # Add legend
            ax4.legend()
            
            # Add zero-inflation annotation
            if zero_inflation_test['possible_zero_inflation']:
                zero_text = f"Possible zero-inflation detected.\nObserved zeros: {zero_inflation_test['observed_zeros']}\nPredicted zeros: {zero_inflation_test['predicted_zeros']}"
            else:
                zero_text = f"No strong evidence of zero-inflation.\nObserved zeros: {zero_inflation_test['observed_zeros']}\nPredicted zeros: {zero_inflation_test['predicted_zeros']}"
                
            ax4.annotate(zero_text, xy=(0.05, 0.95), xycoords='axes fraction',
                       ha='left', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            fig4.tight_layout()
            figures['count_distribution'] = fig_to_svg(fig4)
        except Exception as e:
            figures['count_distribution_error'] = str(e)
        
        # Figure 5: VIF Plot for multicollinearity assessment
        try:
            if 'multicollinearity' in assumptions and 'result' in assumptions['multicollinearity']:
                multicollinearity_test = assumptions['multicollinearity']
                
                if multicollinearity_test['result'] == AssumptionResult.FAILED:
                    fig5, ax5 = plt.subplots(figsize=(10, max(6, len(multicollinearity_test['details'].items()) * 0.4)))
                    
                    # Create horizontal bar chart
                    for i, (k, v) in enumerate(multicollinearity_test['details'].items()):
                        ax5.barh(i, v, color=PASTEL_COLORS[4], alpha=0.7)
                    
                    # Add labels
                    ax5.set_yticks(range(len(multicollinearity_test['details'])))
                    ax5.set_yticklabels(multicollinearity_test['details'].keys())
                    ax5.set_xlabel('Variance Inflation Factor (VIF)', fontsize=12)
                    ax5.set_title('Multicollinearity Assessment', fontsize=14)
                    
                    # Add grid
                    ax5.grid(True, axis='x', alpha=0.3)
                    
                    fig5.tight_layout()
                    figures['vif_plot'] = fig_to_svg(fig5)
        except Exception as e:
            figures['vif_plot_error'] = str(e)
        
        # Figure 6: Predictor Effects Plot
        try:
            # Only create this plot if we have continuous predictors
            continuous_predictors = [pred for pred in predictors 
                                  if pd.api.types.is_numeric_dtype(data[pred])]
            
            if continuous_predictors:
                # Create subplots, one for each continuous predictor (limit to first 9)
                n_predictors = min(9, len(continuous_predictors))
                fig6, axes = plt.subplots(1, n_predictors, figsize=(n_predictors * 4, 4), 
                                         sharey=True, squeeze=False)
                axes = axes.flatten()
                
                for i, pred in enumerate(continuous_predictors[:n_predictors]):
                    ax = axes[i]
                    
                    # Calculate predicted values holding other predictors constant
                    # at their means or modes
                    pred_values = np.linspace(data[pred].min(), data[pred].max(), 100)
                    
                    # Create prediction data
                    predict_data = pd.DataFrame()
                    
                    # Set other predictors to mean/mode
                    for other_pred in predictors:
                        if other_pred != pred:
                            if pd.api.types.is_numeric_dtype(data[other_pred]):
                                predict_data[other_pred] = [data[other_pred].mean()] * len(pred_values)
                            else:
                                # For categorical predictors, use the mode
                                mode_value = data[other_pred].mode()[0]
                                predict_data[other_pred] = [mode_value] * len(pred_values)
                    
                    # Set the main predictor to range of values
                    predict_data[pred] = pred_values
                    
                    # Make predictions
                    y_preds = results.predict(predict_data)
                    
                    # Plot the effect curve
                    ax.plot(pred_values, y_preds, '-', color=PASTEL_COLORS[0], linewidth=2)
                    
                    # Add reference bands for prediction uncertainty if available
                    try:
                        # Calculate prediction standard errors
                        y_std_errs = results.get_prediction(predict_data).summary_frame()['mean_se']
                        
                        # Add 95% confidence bands
                        ax.fill_between(pred_values, 
                                      y_preds - 1.96 * y_std_errs, 
                                      y_preds + 1.96 * y_std_errs, 
                                      color=PASTEL_COLORS[0], alpha=0.2)
                    except:
                        pass  # Skip confidence bands if they can't be calculated
                    
                    # Add labels
                    ax.set_xlabel(pred, fontsize=10)
                    if i == 0:
                        ax.set_ylabel(f'Predicted {outcome}', fontsize=10)
                    
                    # Add gridlines
                    ax.grid(True, alpha=0.3)
                
                # Add overall title
                fig6.suptitle('Predictor Effects on Outcome', fontsize=14)
                fig6.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
                
                figures['predictor_effects'] = fig_to_svg(fig6)
        except Exception as e:
            figures['predictor_effects_error'] = str(e)
        
        # Figure 7: Cook's Distance Plot for influential observations
        try:
            if 'influential_points' in assumptions and 'result' in assumptions['influential_points']:
                if assumptions['influential_points']['result'] == AssumptionResult.FAILED:
                    # Extract Cook's distances from results
                    influence = results.get_influence()
                    cooks_d = influence.cooks_distance[0]
                    
                    fig7, ax7 = plt.subplots(figsize=(10, 6))
                    
                    # Create index values
                    indices = np.arange(len(cooks_d))
                    
                    # Plot Cook's distances
                    ax7.scatter(indices, cooks_d, alpha=0.7, edgecolor='k', color=PASTEL_COLORS[0])
                    
                    # Add threshold line
                    threshold = assumptions['influential_points']['threshold']
                    ax7.axhline(y=threshold, color='red', linestyle='--', 
                              label=f'Threshold ({threshold:.3f})')
                    
                    # Identify points with high influence
                    high_influence = np.where(cooks_d > threshold)[0]
                    
                    if len(high_influence) > 0:
                        # Highlight influential points
                        ax7.scatter(high_influence, cooks_d[high_influence], color=PASTEL_COLORS[4], s=80, 
                                  zorder=10, label='Influential Observations')
                        
                        # Label some of the influential points (limit to avoid cluttering)
                        for idx in high_influence[:min(5, len(high_influence))]:
                            ax7.annotate(f'{idx}', (idx, cooks_d[idx]), 
                                      xytext=(5, 5), textcoords='offset points')
                    
                    # Add labels and title
                    ax7.set_xlabel('Observation Index', fontsize=12)
                    ax7.set_ylabel('Cook\'s Distance', fontsize=12)
                    ax7.set_title('Influential Observations (Cook\'s Distance)', fontsize=14)
                    
                    # Add legend
                    ax7.legend()
                    
                    # Add grid
                    ax7.grid(True, alpha=0.3)
                    
                    fig7.tight_layout()
                    figures['cooks_distance'] = fig_to_svg(fig7)
                else:
                    # Alternative influential observations plot using Pearson residuals
                    fig7, ax7 = plt.subplots(figsize=(10, 6))
                    
                    # Create index values
                    indices = np.arange(len(pearson_residuals))
                    
                    # Plot Pearson residuals
                    ax7.scatter(indices, pearson_residuals, alpha=0.7, edgecolor='k')
                    
                    # Add threshold lines
                    ax7.axhline(y=0, color='grey', linestyle='-', alpha=0.5)
                    ax7.axhline(y=3, color='red', linestyle='--', label='Threshold (±3)')
                    ax7.axhline(y=-3, color='red', linestyle='--')
                    
                    # Identify potential outliers
                    outliers = np.where(np.abs(pearson_residuals) > 3)[0]
                    
                    if len(outliers) > 0:
                        # Highlight outliers
                        ax7.scatter(outliers, pearson_residuals[outliers], color='red', s=80, 
                                  zorder=10, label='Potential Outliers')
                        
                        # Label some of the outliers (limit to avoid cluttering)
                        for idx in outliers[:min(5, len(outliers))]:
                            ax7.annotate(f'{idx}', (idx, pearson_residuals[idx]), 
                                      xytext=(5, 5), textcoords='offset points')
                    
                    # Add labels and title
                    ax7.set_xlabel('Observation Index', fontsize=12)
                    ax7.set_ylabel('Pearson Residuals', fontsize=12)
                    ax7.set_title('Potential Outliers (Pearson Residuals)', fontsize=14)
                    
                    # Add legend
                    ax7.legend()
                    
                    # Add grid
                    ax7.grid(True, alpha=0.3)
                    
                    fig7.tight_layout()
                    figures['pearson_residual_outliers'] = fig_to_svg(fig7)
        except Exception as e:
            figures['influential_observations_plot_error'] = str(e)
            
        # Figure 8: Linearity Check Plot
        try:
            if 'linearity' in assumptions and 'result' in assumptions['linearity']:
                linearity_test = assumptions['linearity']
                
                # Get the continuous predictors that were checked for linearity
                checked_predictors = list(linearity_test['details'].keys())
                
                # Only create this plot if we have predictors to check
                if checked_predictors:
                    # Create subplots, one for each predictor (limit to first 9)
                    n_predictors = min(9, len(checked_predictors))
                    fig8, axes = plt.subplots(1, n_predictors, figsize=(n_predictors * 4, 4), 
                                             sharey=True, squeeze=False)
                    axes = axes.flatten()
                    
                    for i, pred in enumerate(checked_predictors[:n_predictors]):
                        ax = axes[i]
                        
                        # Plot predictor vs. residuals
                        ax.scatter(data[pred], residuals, alpha=0.7, edgecolor='k', color=PASTEL_COLORS[0])
                        
                        # Add a horizontal line at y=0
                        ax.axhline(y=0, color='grey', linestyle='-', alpha=0.5)
                        
                        # Add a LOWESS smoother if available
                        try:
                            from statsmodels.nonparametric.smoothers_lowess import lowess
                            
                            # Compute LOWESS
                            z = lowess(residuals, data[pred], frac=0.6)
                            
                            # Plot the smooth curve
                            ax.plot(z[:, 0], z[:, 1], color=PASTEL_COLORS[4], linewidth=2)
                        except:
                            # If LOWESS fails or isn't available, just add a best fit line
                            from scipy.stats import linregress
                            
                            # Only calculate regression if there are enough valid points
                            valid = ~np.isnan(data[pred]) & ~np.isnan(residuals)
                            if sum(valid) > 2:
                                slope, intercept, r_value, p_value, std_err = linregress(
                                    data[pred][valid], residuals[valid])
                                x_vals = np.array([data[pred].min(), data[pred].max()])
                                y_vals = intercept + slope * x_vals
                                ax.plot(x_vals, y_vals, 'r-', linewidth=2)
                        
                        # Add correlation information
                        corr = linearity_test['details'][pred]['correlation']
                        p_val = linearity_test['details'][pred]['p_value']
                        sig_str = '*' if p_val < alpha else ''
                        ax.annotate(f"r = {corr:.2f}{sig_str}", xy=(0.05, 0.95), xycoords='axes fraction',
                                  ha='left', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                        
                        # Add labels
                        ax.set_xlabel(pred, fontsize=10)
                        if i == 0:
                            ax.set_ylabel('Residuals', fontsize=10)
                        
                        # Add gridlines
                        ax.grid(True, alpha=0.3)
                    
                    # Add overall title
                    fig8.suptitle('Linearity Check: Residuals vs. Predictors', fontsize=14)
                    fig8.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
                    
                    figures['linearity_check'] = fig_to_svg(fig8)
        except Exception as e:
            figures['linearity_check_error'] = str(e)

        return {
            'test': 'Poisson Regression',
            'coefficients': coefficients,
            'model_fit': model_fit,
            'overdispersion': overdispersion_test,
            'zero_inflation': zero_inflation_test,
            'prediction_stats': prediction_stats,
            'assumptions': assumptions,
            'formula': formula,
            'summary': str(results.summary()),
            'interpretation': interpretation,
            'figures': figures
        }
    except Exception as e:
        return {
            'test': 'Poisson Regression',
            'error': str(e),
            'traceback': traceback.format_exc()
        }