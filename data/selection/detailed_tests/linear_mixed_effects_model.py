import traceback
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from matplotlib.patches import Patch
from typing import Dict, List, Any, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
from data.assumptions.tests import (
    AssumptionResult,
    HomogeneityOfRegressionSlopesTest,
    InfluentialPointsTest,
    ModelSpecificationTest,
    OutlierTest,
    RandomEffectsNormalityTest,
    ResidualNormalityTest, 
    HomoscedasticityTest,
    MulticollinearityTest, 
    LinearityTest,
    AutocorrelationTest,
    SampleSizeTest
)
from data.selection.detailed_tests.formatting import fig_to_svg, PASTEL_COLORS, PASTEL_CMAP

def linear_mixed_effects_model(data: pd.DataFrame, fixed_effects: List[str], random_effects: List[str], alpha: float, outcome: str) -> Dict[str, Any]:
    """
    Fits a Linear Mixed Effects Model (LME) using statsmodels with comprehensive statistics and diagnostics.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataset containing the variables
    fixed_effects : List[str]
        List of column names to use as fixed effects
    random_effects : List[str]
        List of column names to use as random effects (grouping variables)
    alpha : float
        Significance level
    outcome : str
        Name of the outcome/dependent variable
        
    Returns:
    --------
    Dict[str, Any]
        Comprehensive results including model statistics, diagnostics, and visualizations
    """
    try:
        import numpy as np
        from scipy import stats
        
        # Filter out any rows with missing values
        data_clean = data.dropna(subset=[outcome] + fixed_effects + random_effects)
        
        if len(data_clean) == 0:
            return {
                'test': 'Linear Mixed Effects Model',
                'error': 'No valid data after removing missing values',
                'satisfied': False
            }
        
        # Construct the formula
        fixed_effects_str = ' + '.join(fixed_effects)
        formula = f"{outcome} ~ {fixed_effects_str}"
        
        # Check if we need to create a combined grouping variable for multiple random effects
        if len(random_effects) > 1:
            # Create a combined grouping variable
            data_clean['_combined_groups'] = data_clean[random_effects].astype(str).agg('_'.join, axis=1)
            groups = data_clean['_combined_groups']
            re_formula = ' + '.join(['1'] + random_effects[1:])  # Include other random effects as needed
        else:
            groups = data_clean[random_effects[0]]
            re_formula = '1'  # Just intercept as random effect
        
        # Fit the model
        try:
            model = smf.mixedlm(formula, data_clean, groups=groups, re_formula=re_formula)
            results = model.fit(reml=True)
        except Exception as model_error:
            # Try with a simpler model if the original fails
            try:
                # Only random intercept
                model = smf.mixedlm(formula, data_clean, groups=data_clean[random_effects[0]])
                results = model.fit(reml=True)
                warnings.warn(f"Used simplified model due to: {str(model_error)}")
            except Exception as e:
                return {
                    'test': 'Linear Mixed Effects Model',
                    'error': f"Model fitting failed: {str(e)}",
                    'original_error': str(model_error),
                    'satisfied': False
                }
        
        # Extract model information
        # Fixed effects summary
        fixed_effects_summary = results.summary().tables[1]
        fixed_effects_results = []
        
        for i in range(len(fixed_effects_summary)):
            # Skip header row
            if i == 0:
                continue
                
            try:
                # Extract parameter information
                row = fixed_effects_summary.iloc[i]
                param_name = str(row[0]).strip()
                
                # Handle potentially missing or empty values with safe conversion
                try:
                    coef = float(row[1]) if row[1] and row[1].strip() else 0.0
                except (ValueError, TypeError, AttributeError):
                    coef = 0.0
                    
                try:
                    std_err = float(row[2]) if row[2] and row[2].strip() else None
                except (ValueError, TypeError, AttributeError):
                    std_err = None
                    
                try:
                    t_value = float(row[3]) if row[3] and row[3].strip() else None
                except (ValueError, TypeError, AttributeError):
                    t_value = None
                    
                try:
                    p_value = float(row[4]) if row[4] and row[4].strip() else 1.0
                except (ValueError, TypeError, AttributeError):
                    p_value = 1.0
                
                # Calculate confidence intervals
                if std_err is not None and not np.isnan(std_err):
                    ci_lower = coef - 1.96 * std_err
                    ci_upper = coef + 1.96 * std_err
                else:
                    # If std_err is missing, set CI to None or some default
                    ci_lower = coef
                    ci_upper = coef
                
                fixed_effects_results.append({
                    'parameter': param_name,
                    'coefficient': coef,
                    'std_error': std_err if std_err is not None else float('nan'),
                    't_value': t_value if t_value is not None else float('nan'),
                    'p_value': p_value,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'significant': p_value < alpha if p_value is not None else False
                })
            except Exception as param_error:
                # If we can't process this parameter, log it and continue
                warnings.warn(f"Could not process parameter at row {i}: {str(param_error)}")
                continue
        
        # Random effects summary
        random_effects_summary = results.random_effects
        random_effects_variance = results.cov_re
        
        # Convert variance components to a standard format
        random_variance_components = {}
        try:
            for i, component in enumerate(random_effects_variance):
                random_variance_components[f"random_var_{i}"] = float(component)
        except:
            # Handle different structure of cov_re
            if isinstance(random_effects_variance, pd.DataFrame):
                for col in random_effects_variance.columns:
                    random_variance_components[col] = float(random_effects_variance.loc[col, col])
            elif isinstance(random_effects_variance, dict):
                for key, val in random_effects_variance.items():
                    if isinstance(val, (int, float)):
                        random_variance_components[str(key)] = float(val)
                    else:
                        try:
                            random_variance_components[str(key)] = float(val[0, 0])
                        except:
                            random_variance_components[str(key)] = "Unable to extract"
        
        # Calculate model performance metrics
        y_true = data_clean[outcome]
        y_pred = results.fittedvalues
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Calculate R-squared for mixed models (fixed effects only and conditional R²)
        # This is an approximation based on the marginal and conditional R² approach
        
        # Total variance
        total_var = np.var(y_true, ddof=1)
        
        # Residual variance
        residual_var = np.var(results.resid, ddof=1)
        
        # Random effects variance (approximate)
        random_var = sum(random_variance_components.values())
        
        # Marginal R² (fixed effects only)
        marginal_r2 = (total_var - residual_var - random_var) / total_var
        
        # Conditional R² (fixed + random effects)
        conditional_r2 = (total_var - residual_var) / total_var
        
        # Ensure R² values are within [0, 1]
        marginal_r2 = max(0, min(1, marginal_r2))
        conditional_r2 = max(0, min(1, conditional_r2))
        
        # Calculate AIC and BIC
        aic = results.aic
        bic = results.bic
        
        # Calculate ICC (Intraclass Correlation Coefficient)
        # ICC = random intercept variance / (random intercept variance + residual variance)
        try:
            random_intercept_var = random_variance_components.get("random_var_0", 0)
            icc = random_intercept_var / (random_intercept_var + results.scale)
        except:
            icc = "Unable to calculate"
        
        # Test assumptions
        assumptions = {}
        
        # 1. Residual normality
        try:
            residual_normality_test = ResidualNormalityTest()
            normality_result = residual_normality_test.run_test(results.resid)
            assumptions['RESIDUAL_NORMALITY'] = normality_result
        except Exception as e:
            assumptions['RESIDUAL_NORMALITY'] = {
                'error': f'Could not test residual normality: {str(e)}',
                'result': AssumptionResult.FAILED
            }

        # 2. Homoscedasticity
        try:
            homoscedasticity_test = HomoscedasticityTest()
            homoscedasticity_result = homoscedasticity_test.run_test(
                residuals=results.resid, 
                predicted=results.fittedvalues
            )
            assumptions['HOMOSCEDASTICITY'] = homoscedasticity_result
        except Exception as e:
            assumptions['HOMOSCEDASTICITY'] = {
                'error': f'Could not test homoscedasticity: {str(e)}',
                'result': AssumptionResult.FAILED
            }
        
        # 3. Multicollinearity
        try:
            # Create a dataframe with just the fixed effects
            X = data_clean[fixed_effects].copy()
            multicollinearity_test = MulticollinearityTest()
            multicollinearity_result = multicollinearity_test.run_test(
                df=X,
                covariates=fixed_effects
            )
            
            # Ensure all dictionary keys are strings
            if 'vif_values' in multicollinearity_result and isinstance(multicollinearity_result['vif_values'], dict):
                multicollinearity_result['vif_values'] = {
                    str(k): v for k, v in multicollinearity_result['vif_values'].items()
                }
            
            assumptions['MULTICOLLINEARITY'] = multicollinearity_result
        except Exception as e:
            assumptions['MULTICOLLINEARITY'] = {
                'error': f'Could not test multicollinearity: {str(e)}',
                'result': AssumptionResult.FAILED
            }

        # 4. Linearity
        try:
            # For each numerical predictor vs outcome
            linearity_results = {}
            for predictor in fixed_effects:
                if pd.api.types.is_numeric_dtype(data_clean[predictor]):
                    linearity_test = LinearityTest()
                    linearity_result = linearity_test.run_test(
                        x=data_clean[predictor],
                        y=data_clean[outcome]
                    )
                    linearity_results[predictor] = linearity_result
            
            # Combine results or use the worst case
            if linearity_results:
                # Simplistic approach: if any test fails, the overall test fails
                failed_predictors = [pred for pred, result in linearity_results.items() 
                                    if result['result'].value != 'passed']
                
                if failed_predictors:
                    combined_result = AssumptionResult.FAILED
                    details = f"Non-linear relationships detected for: {', '.join(failed_predictors)}"
                else:
                    combined_result = AssumptionResult.PASSED
                    details = "Linearity assumption appears to be satisfied for all predictors."
                
                assumptions['LINEARITY'] = {
                    'result': combined_result,
                    'details': details,
                    'individual_tests': linearity_results
                }
            else:
                assumptions['LINEARITY'] = {
                    'result': AssumptionResult.NOT_APPLICABLE,
                    'details': "No numerical predictors to test for linearity."
            }
        except Exception as e:
            assumptions['LINEARITY'] = {
                'error': f'Could not test linearity: {str(e)}',
                'result': AssumptionResult.FAILED
            }

        # 5. Autocorrelation (independence of residuals)
        try:
            autocorrelation_test = AutocorrelationTest()
            autocorrelation_result = autocorrelation_test.run_test(
                residuals=results.resid
            )
            assumptions['AUTOCORRELATION'] = autocorrelation_result
        except Exception as e:
            assumptions['AUTOCORRELATION'] = {
                'error': f'Could not test autocorrelation: {str(e)}',
                'result': AssumptionResult.FAILED
            }

        # 6. Outliers
        try:
            outlier_test = OutlierTest()
            outlier_result = outlier_test.run_test(
                data=results.resid
            )
            
            # Convert any numeric keys to strings in the result
            if 'outliers' in outlier_result and isinstance(outlier_result['outliers'], dict):
                outlier_result['outliers'] = {str(k): v for k, v in outlier_result['outliers'].items()}
            
            assumptions['OUTLIERS'] = outlier_result
        except Exception as e:
            assumptions['OUTLIERS'] = {
                'error': f'Could not test for outliers: {str(e)}',
                'result': AssumptionResult.FAILED
            }

        # Additional tests that should be implemented:

        # 7. Random Effects Normality
        try:
            # Extract random effects for testing
            random_effects_data = pd.DataFrame(results.random_effects)
            
            # If there are multiple columns, we need to test each
            if isinstance(random_effects_data, pd.DataFrame) and not random_effects_data.empty:
                random_effects_normality_test = RandomEffectsNormalityTest()
                random_effects_result = random_effects_normality_test.run_test(
                    random_effects=random_effects_data
                )
                assumptions['RANDOM_EFFECTS_NORMALITY'] = random_effects_result
            else:
                assumptions['RANDOM_EFFECTS_NORMALITY'] = {
                    'result': AssumptionResult.NOT_APPLICABLE,
                    'details': "Random effects structure not suitable for normality testing."
                }
        except Exception as e:
            assumptions['RANDOM_EFFECTS_NORMALITY'] = {
                'error': f'Could not test random effects normality: {str(e)}',
                'result': AssumptionResult.FAILED
            }

        # 8. Influential Points
        try:
            # We need model matrix X, residuals, and leverage
            try:
                X = add_constant(results.model.exog)
                h_ii = np.diag(X.dot(np.linalg.inv(X.T.dot(X))).dot(X.T))  # leverage
                
                influential_points_test = InfluentialPointsTest()
                influential_points_result = influential_points_test.run_test(
                    residuals=results.resid,
                    leverage=h_ii,
                    fitted=results.fittedvalues,
                    X=pd.DataFrame(X)
                )
                
                # Convert any numeric keys to strings
                if 'influential_points' in influential_points_result and isinstance(influential_points_result['influential_points'], dict):
                    influential_points_result['influential_points'] = {
                        str(k): v for k, v in influential_points_result['influential_points'].items()
                    }
                
                assumptions['INFLUENTIAL_POINTS'] = influential_points_result
            except:
                # Model matrix approach failed, use a different method
                assumptions['INFLUENTIAL_POINTS'] = {
                    'result': AssumptionResult.NOT_APPLICABLE,
                    'details': "Could not calculate leverage values for mixed model."
            }
        except Exception as e:
            assumptions['INFLUENTIAL_POINTS'] = {
                'error': f'Could not test for influential points: {str(e)}',
                'result': AssumptionResult.FAILED
            }

        # 9. Sample Size Test
        try:
            sample_size_test = SampleSizeTest()
            # Assuming we need at least 10 observations per parameter
            min_recommended = len(fixed_effects) * 10
            
            sample_size_result = sample_size_test.run_test(
                data=data_clean[outcome],
                min_recommended=min_recommended
            )
            assumptions['SAMPLE_SIZE'] = sample_size_result
        except Exception as e:
            assumptions['SAMPLE_SIZE'] = {
                'error': f'Could not test sample size adequacy: {str(e)}',
                'result': AssumptionResult.FAILED
            }

        # 10. Model Specification Test (if applicable)
        try:
            model_specification_test = ModelSpecificationTest()
            # For this test, we need residuals, fitted values, and predictors
            X = pd.DataFrame(results.model.exog, columns=[f"X{i}" for i in range(results.model.exog.shape[1])])
            
            model_specification_result = model_specification_test.run_test(
                residuals=results.resid,
                fitted=results.fittedvalues,
                X=X
            )
            assumptions['MODEL_SPECIFICATION'] = model_specification_result
        except Exception as e:
            assumptions['MODEL_SPECIFICATION'] = {
                'error': f'Could not test model specification: {str(e)}',
                'result': AssumptionResult.FAILED
            }

        # 11. Homogeneity of Regression Slopes (if applicable for mixed models with interaction)
        if len(fixed_effects) > 1 and len(random_effects) > 0:
            try:
                homogeneity_test = HomogeneityOfRegressionSlopesTest()
                # This test requires outcome, group, and covariates
                homogeneity_result = homogeneity_test.run_test(
                    df=data_clean,
                    outcome=outcome,
                    group=random_effects[0],  # Using the first random effect as the grouping variable
                    covariates=fixed_effects
                )
                assumptions['HOMOGENEITY_OF_REGRESSION_SLOPES'] = homogeneity_result
            except Exception as e:
                assumptions['HOMOGENEITY_OF_REGRESSION_SLOPES'] = {
                    'error': f'Could not test homogeneity of regression slopes: {str(e)}',
                    'result': AssumptionResult.FAILED
                }
        
        # Create an interpretation of the results
        interpretation = f"Linear Mixed Effects Model Analysis\n\n"
        
        # Model formula
        interpretation += f"Model Formula: {formula}\n"
        if len(random_effects) > 1:
            interpretation += f"Random Effects: {', '.join(random_effects)}\n"
        else:
            interpretation += f"Random Effect: {random_effects[0]}\n"
        
        # Model fit
        interpretation += f"\nModel Fit:\n"
        interpretation += f"- Marginal R² (fixed effects only): {marginal_r2:.3f}\n"
        interpretation += f"- Conditional R² (fixed + random effects): {conditional_r2:.3f}\n"
        interpretation += f"- AIC: {aic:.2f}, BIC: {bic:.2f}\n"
        interpretation += f"- RMSE: {rmse:.3f}, MAE: {mae:.3f}\n"
        
        if isinstance(icc, (int, float)):
            interpretation += f"- Intraclass Correlation Coefficient (ICC): {icc:.3f}\n"
        
        # Fixed effects
        interpretation += f"\nFixed Effects:\n"
        sig_effects = []
        non_sig_effects = []
        
        for effect in fixed_effects_results:
            # Check if the intercept or interaction term
            name = effect['parameter']
            
            # Format the p-value with asterisks for significance
            p_val = effect['p_value']
            p_str = f"{p_val:.5f}"
            if p_val < 0.001:
                p_str += " ***"
            elif p_val < 0.01:
                p_str += " **"
            elif p_val < 0.05:
                p_str += " *"
            
            effect_str = f"- {name}: β = {effect['coefficient']:.4f}, SE = {effect['std_error']:.4f}, "
            effect_str += f"t = {effect['t_value']:.3f}, p = {p_str}, "
            effect_str += f"95% CI: [{effect['ci_lower']:.4f}, {effect['ci_upper']:.4f}]"
            
            if effect['significant']:
                sig_effects.append(effect_str)
            else:
                non_sig_effects.append(effect_str)
        
        # List significant effects first
        if sig_effects:
            interpretation += "Significant effects:\n"
            interpretation += '\n'.join(sig_effects) + '\n'
            
        if non_sig_effects:
            interpretation += "Non-significant effects:\n"
            interpretation += '\n'.join(non_sig_effects) + '\n'
        
        # Random effects
        interpretation += f"\nRandom Effects Variance Components:\n"
        for component, value in random_variance_components.items():
            if isinstance(value, (int, float)):
                interpretation += f"- {component}: {value:.6f}\n"
            else:
                interpretation += f"- {component}: {value}\n"
        
        interpretation += f"- Residual variance: {results.scale:.6f}\n"
        
        # Assumptions
        interpretation += f"\nAssumption Checks:\n"
        
        # Normality
        if 'RESIDUAL_NORMALITY' in assumptions and 'result' in assumptions['RESIDUAL_NORMALITY']:
            norm_result = assumptions['RESIDUAL_NORMALITY']
            
            # Extract the test_used field if it exists, otherwise use a default
            test_name = norm_result.get('test_used', 'Normality')
            
            # Extract p_value if it exists AND is not None
            interpretation += f"- Residual Normality: {test_name} test"
            if 'p_value' in norm_result and norm_result['p_value'] is not None:
                p_value = norm_result['p_value']
                interpretation += f", p = {p_value:.5f}"
            
            interpretation += ", "
            
            if norm_result['result'].value == 'passed':
                interpretation += "assumption satisfied.\n"
            else:
                interpretation += "assumption potentially violated. "
                interpretation += "Consider transforming the outcome variable or using robust standard errors.\n"
        
        # Homoscedasticity
        if 'HOMOSCEDASTICITY' in assumptions and 'result' in assumptions['HOMOSCEDASTICITY']:
            homo_result = assumptions['HOMOSCEDASTICITY']
            
            # Extract the test_used field if it exists
            test_name = homo_result.get('test_used', 'Homoscedasticity')
            
            # Extract p_value if it exists AND is not None
            interpretation += f"- Homoscedasticity: {test_name} test"
            if 'p_value' in homo_result and homo_result['p_value'] is not None:
                p_value = homo_result['p_value']
                interpretation += f", p = {p_value:.5f}"
            
            interpretation += ", "
            
            if homo_result['result'].value == 'passed':
                interpretation += "assumption satisfied.\n"
            else:
                interpretation += "assumption potentially violated. "
                interpretation += "Consider using robust standard errors or transforming the outcome variable.\n"
        
        # Multicollinearity
        if 'MULTICOLLINEARITY' in assumptions and 'result' in assumptions['MULTICOLLINEARITY']:
            multi_result = assumptions['MULTICOLLINEARITY']
            interpretation += f"- Multicollinearity: "
            
            if multi_result['result'].value == 'passed':
                interpretation += "No severe multicollinearity detected.\n"
            else:
                # Get high VIF predictors if available
                if 'vif_values' in multi_result:
                    vif_values = multi_result['vif_values']
                    high_vif = {k: v for k, v in vif_values.items() if v is not None and v > 10}
                    if high_vif:
                        interpretation += f"Severe multicollinearity detected in: {', '.join(high_vif.keys())}. "
                    else:
                        interpretation += "Some multicollinearity detected. "
                else:
                    interpretation += "Multicollinearity may be present. "
                    
                interpretation += "This may affect the stability and interpretation of coefficients.\n"
        
        # Linearity
        if 'LINEARITY' in assumptions and 'result' in assumptions['LINEARITY']:
            linear_result = assumptions['LINEARITY']
            interpretation += f"- Linearity: "
            
            if linear_result['result'].value == 'passed':
                interpretation += "Assumption appears to be satisfied.\n"
            else:
                # Check if we have details on non-linear predictors
                if 'details' in linear_result and linear_result['details']:
                    interpretation += f"{linear_result['details']}\n"
                else:
                    interpretation += "Potential non-linear relationships detected. "
                interpretation += "Consider transformations or adding polynomial terms.\n"
        
        # Independence (Autocorrelation)
        if 'AUTOCORRELATION' in assumptions and 'result' in assumptions['AUTOCORRELATION']:
            indep_result = assumptions['AUTOCORRELATION']
            
            # Extract test_used and statistic if available
            test_name = indep_result.get('test_used', 'Autocorrelation')
            
            interpretation += f"- Independence: {test_name} test"
            if 'statistic' in indep_result and indep_result['statistic'] is not None:
                statistic = indep_result['statistic']
                interpretation += f", statistic = {statistic:.3f}"
            
            interpretation += ", "
            
            if indep_result['result'].value == 'passed':
                interpretation += "no significant autocorrelation detected.\n"
            else:
                interpretation += "potential autocorrelation detected. "
                interpretation += "If your data has a time component, this could affect inference.\n"
        
        # Influential observations/Outliers
        if 'INFLUENTIAL_POINTS' in assumptions and 'result' in assumptions['INFLUENTIAL_POINTS']:
            infl_result = assumptions['INFLUENTIAL_POINTS']
            interpretation += f"- Influential Observations: "
            
            if infl_result['result'].value == 'passed':
                interpretation += "No influential observations detected.\n"
            else:
                # Check if we have details on influential points
                if 'influential_points' in infl_result and infl_result['influential_points']:
                    count = len(infl_result['influential_points'])
                    interpretation += f"Detected {count} influential observations. "
                elif 'details' in infl_result:
                    interpretation += f"{infl_result['details']} "
                else:
                    interpretation += "Potential influential observations detected. "
                    
                interpretation += "Consider checking these observations for data entry errors or rerunning the model without them.\n"
        
        # Outliers (if separate from influential points)
        if 'OUTLIERS' in assumptions and 'result' in assumptions['OUTLIERS']:
            out_result = assumptions['OUTLIERS']
            interpretation += f"- Outliers: "
            
            if out_result['result'].value == 'passed':
                interpretation += "No significant outliers detected.\n"
            else:
                # Check if we have details on outliers
                if 'outliers' in out_result and out_result['outliers']:
                    count = len(out_result['outliers'])
                    interpretation += f"Detected {count} potential outliers. "
                elif 'details' in out_result:
                    interpretation += f"{out_result['details']} "
                else:
                    interpretation += "Potential outliers detected. "
                    
                interpretation += "Consider examining these data points or using robust methods.\n"

        # Random Effects Normality (additional test)
        if 'RANDOM_EFFECTS_NORMALITY' in assumptions and 'result' in assumptions['RANDOM_EFFECTS_NORMALITY']:
            re_norm_result = assumptions['RANDOM_EFFECTS_NORMALITY']
            
            # Extract test_used if available
            test_name = re_norm_result.get('test_used', 'Normality')
            
            interpretation += f"- Random Effects Normality: {test_name} test"
            
            # Add p-value if available AND not None
            if 'p_value' in re_norm_result and re_norm_result['p_value'] is not None:
                p_value = re_norm_result['p_value']
                interpretation += f", p = {p_value:.5f}"
            
            interpretation += ", "
            
            if re_norm_result['result'].value == 'passed':
                interpretation += "assumption satisfied.\n"
            else:
                interpretation += "assumption potentially violated. "
                if 'details' in re_norm_result:
                    interpretation += f"{re_norm_result['details']}\n"
                else:
                    interpretation += "This may affect the reliability of random effects estimates.\n"

        # Sample Size
        if 'SAMPLE_SIZE' in assumptions and 'result' in assumptions['SAMPLE_SIZE']:
            size_result = assumptions['SAMPLE_SIZE']
            interpretation += f"- Sample Size: "
            
            if size_result['result'].value == 'passed':
                interpretation += "Sample size is adequate for the analysis.\n"
            else:
                # Extract sample size info if available
                if 'sample_size' in size_result and 'minimum_required' in size_result:
                    sample_size = size_result.get('sample_size')
                    min_required = size_result.get('minimum_required')
                    if sample_size is not None and min_required is not None:
                        interpretation += f"Sample size ({sample_size}) is below the recommended minimum "
                        interpretation += f"({min_required}) for this model. "
                    else:
                        interpretation += "Sample size may be insufficient. "
                elif 'details' in size_result:
                    interpretation += f"{size_result['details']} "
                else:
                    interpretation += "Sample size may be insufficient. "
                    
                interpretation += "Results should be interpreted with caution.\n"

        # Model Specification
        if 'MODEL_SPECIFICATION' in assumptions and 'result' in assumptions['MODEL_SPECIFICATION']:
            spec_result = assumptions['MODEL_SPECIFICATION']
            
            # Extract test_used if available
            test_name = spec_result.get('test_used', 'Specification')
            
            interpretation += f"- Model Specification: {test_name} test"
            
            # Add p-value if available AND not None
            if 'p_value' in spec_result and spec_result['p_value'] is not None:
                p_value = spec_result['p_value']
                interpretation += f", p = {p_value:.5f}"
            
            interpretation += ", "
            
            if spec_result['result'].value == 'passed':
                interpretation += "model appears to be correctly specified.\n"
            else:
                interpretation += "potential model misspecification detected. "
                if 'details' in spec_result:
                    interpretation += f"{spec_result['details']}\n"
                else:
                    interpretation += "Consider adding transformed predictors or interaction terms.\n"

        # Homogeneity of Regression Slopes
        if 'HOMOGENEITY_OF_REGRESSION_SLOPES' in assumptions and 'result' in assumptions['HOMOGENEITY_OF_REGRESSION_SLOPES']:
            homo_slopes_result = assumptions['HOMOGENEITY_OF_REGRESSION_SLOPES']
            
            # Extract test_used if available
            test_name = homo_slopes_result.get('test_used', 'Homogeneity')
            
            interpretation += f"- Homogeneity of Regression Slopes: {test_name} test"
            
            # Add p-value if available AND not None
            if 'p_value' in homo_slopes_result and homo_slopes_result['p_value'] is not None:
                p_value = homo_slopes_result['p_value']
                interpretation += f", p = {p_value:.5f}"
            
            interpretation += ", "
            
            if homo_slopes_result['result'].value == 'passed':
                interpretation += "assumption satisfied.\n"
            else:
                interpretation += "assumption potentially violated. "
                if 'details' in homo_slopes_result:
                    interpretation += f"{homo_slopes_result['details']}\n"
                else:
                    interpretation += "Consider using a more complex model with interaction terms.\n"

        # Overall conclusion
        interpretation += f"\nConclusion:\n"
        sig_fixed_effects = [e['parameter'] for e in fixed_effects_results if e['significant']]
        
        if sig_fixed_effects:
            interpretation += f"The analysis identified significant effects for the following predictors: {', '.join(sig_fixed_effects)}. "
            interpretation += f"The model explains approximately {conditional_r2:.1%} of the variance in {outcome}, "
            interpretation += f"with {marginal_r2:.1%} attributable to the fixed effects alone.\n"
        else:
            interpretation += f"The analysis did not identify any significant fixed effects. "
            interpretation += f"The model explains approximately {conditional_r2:.1%} of the variance in {outcome}, "
            interpretation += f"with most of this being attributable to the random effects structure.\n"
        
        # Comment on random effects
        if isinstance(icc, (int, float)) and icc > 0.1:
            interpretation += f"The intraclass correlation coefficient (ICC) of {icc:.3f} suggests substantial clustering in the data, "
            interpretation += f"confirming that a mixed-effects approach was appropriate for this analysis.\n"
        
        # Create visualizations
        figures = {}
        
        # Figure 1: Residuals vs. Fitted Values
        try:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.scatter(results.fittedvalues, results.resid, alpha=0.6, color=PASTEL_COLORS[0])
            ax1.axhline(y=0, color=PASTEL_COLORS[5], linestyle='--')
            
            # Draw a lowess smoothed line to check for non-linear patterns - improved error handling
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                if len(results.fittedvalues) > 10:  # Only try lowess with sufficient data points
                    smoothed = lowess(results.resid, results.fittedvalues, frac=0.6)
                    ax1.plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2)
            except Exception as lowess_error:
                # Log the error but continue with the plot
                print(f"Lowess smoothing failed: {str(lowess_error)}")
            
            # Add labels
            ax1.set_xlabel('Fitted Values')
            ax1.set_ylabel('Residuals')
            ax1.set_title('Residuals vs. Fitted Values')
            
            # Add reference text for homoscedasticity test
            if 'HOMOSCEDASTICITY' in assumptions and 'p_value' in assumptions['HOMOSCEDASTICITY']:
                homo_p = assumptions['HOMOSCEDASTICITY']['p_value']
                homo_test = assumptions['HOMOSCEDASTICITY'].get('test_used', 'Homoscedasticity test')
                
                result_text = f"{homo_test}: p = {homo_p:.4f}"
                if homo_p < alpha:
                    result_text += " *\nHeteroscedasticity detected"
                else:
                    result_text += "\nHomoscedasticity satisfied"
                    
                ax1.text(0.05, 0.95, result_text, transform=ax1.transAxes, 
                       va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
            
            fig1.tight_layout()
            figures['residuals_vs_fitted'] = fig_to_svg(fig1)
        except Exception as e:
            figures['residuals_vs_fitted_error'] = str(e)
        
        # Figure 2: QQ Plot of Residuals
        try:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            # Calculate quantiles
            residuals = results.resid
            sorted_residuals = np.sort(residuals)
            n = len(sorted_residuals)
            quantiles = np.arange(1, n + 1) / (n + 1)  # Use (n + 1) to avoid boundary issues
            
            # Calculate theoretical quantiles
            theoretical_quantiles = stats.norm.ppf(quantiles, loc=0, scale=1)
            
            # Create QQ plot
            ax2.scatter(theoretical_quantiles, sorted_residuals, alpha=0.6, color=PASTEL_COLORS[1])
            
            # Add reference line
            min_val = min(np.min(theoretical_quantiles), np.min(sorted_residuals))
            max_val = max(np.max(theoretical_quantiles), np.max(sorted_residuals))
            ax2.plot([min_val, max_val], [min_val, max_val], '--', color=PASTEL_COLORS[5])
            
            # Add labels
            ax2.set_xlabel('Theoretical Quantiles')
            ax2.set_ylabel('Sample Quantiles')
            ax2.set_title('Normal Q-Q Plot of Residuals')
            
            # Add reference text for normality test
            if 'RESIDUAL_NORMALITY' in assumptions and 'p_value' in assumptions['RESIDUAL_NORMALITY']:
                norm_p = assumptions['RESIDUAL_NORMALITY']['p_value']
                norm_test = assumptions['RESIDUAL_NORMALITY']['test']
                
                result_text = f"{norm_test}: p = {norm_p:.4f}"
                if norm_p < alpha:
                    result_text += " *\nNon-normality detected"
                else:
                    result_text += "\nNormality satisfied"
                    
                ax2.text(0.05, 0.95, result_text, transform=ax2.transAxes, 
                       va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
            
            fig2.tight_layout()
            figures['qq_plot'] = fig_to_svg(fig2)
        except Exception as e:
            figures['qq_plot_error'] = str(e)
        
        # Figure 3: Histogram of Residuals
        try:
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            
            # Create histogram with KDE
            sns.histplot(results.resid, kde=True, ax=ax3, color=PASTEL_COLORS[2])
            
            # Add a vertical line at the mean
            ax3.axvline(x=np.mean(results.resid), color=PASTEL_COLORS[5], linestyle='--')
            
            # Add labels
            ax3.set_xlabel('Residuals')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of Residuals')
            
            # Add skewness and kurtosis information
            if 'RESIDUAL_NORMALITY' in assumptions and 'skewness' in assumptions['RESIDUAL_NORMALITY']:
                skew = assumptions['RESIDUAL_NORMALITY']['skewness']
                kurt = assumptions['RESIDUAL_NORMALITY']['kurtosis']
                
                stats_text = f"Skewness: {skew:.3f}\nKurtosis: {kurt:.3f}"
                ax3.text(0.95, 0.95, stats_text, transform=ax3.transAxes, 
                       va='top', ha='right', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
            
            fig3.tight_layout()
            figures['residuals_histogram'] = fig_to_svg(fig3)
        except Exception as e:
            figures['residuals_histogram_error'] = str(e)
        
        # Figure 4: Boxplot of Random Effects
        try:
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            
            # Extract random effects
            re_data = []
            re_labels = []
            
            for group, effect in random_effects_summary.items():
                if isinstance(effect, dict):
                    for key, val in effect.items():
                        re_data.append(val)
                        re_labels.append(f"{group} - {key}")
                else:
                    re_data.append(effect)
                    re_labels.append(str(group))
            
            # Create dataframe for plotting
            re_df = pd.DataFrame({
                'Group': re_labels[:20] if len(re_labels) > 20 else re_labels,  # Limit to 20 for readability
                'Random Effect': re_data[:20] if len(re_data) > 20 else re_data
            })
            
            # Create boxplot - fixing to include x parameter for proper grouping
            if len(re_df) > 0:  # Only create plot if we have data
                sns.boxplot(x='Group', y='Random Effect', data=re_df, ax=ax4, color=PASTEL_COLORS[3])
                
                # Rotate x labels for readability
                plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
                
                # Add reference line at 0
                ax4.axhline(y=0, color=PASTEL_COLORS[5], linestyle='--')
                
                # Add labels
                ax4.set_title('Distribution of Random Effects')
                
                # Add note if we limited the display
                if len(re_labels) > 20:
                    ax4.text(0.5, -0.1, f"Note: Showing only 20 out of {len(re_labels)} random effects", 
                           transform=ax4.transAxes, ha='center', va='center', 
                           bbox=dict(boxstyle='round', fc='yellow', alpha=0.2))
            else:
                # If no data, create empty plot with message
                ax4.text(0.5, 0.5, "No random effects data available to plot", 
                       transform=ax4.transAxes, ha='center', va='center')
            
            fig4.tight_layout()
            figures['random_effects_boxplot'] = fig_to_svg(fig4)
        except Exception as e:
            figures['random_effects_boxplot_error'] = str(e)
        
        # Figure 5: Coefficients Plot for Fixed Effects
        try:
            fig5, ax5 = plt.subplots(figsize=(10, 8))
            
            # Extract coefficient data
            coef_names = []
            coef_values = []
            ci_lowers = []
            ci_uppers = []
            is_significant = []
            
            for effect in fixed_effects_results:
                coef_names.append(effect['parameter'])
                coef_values.append(effect['coefficient'])
                ci_lowers.append(effect['ci_lower'])
                ci_uppers.append(effect['ci_upper'])
                is_significant.append(effect['significant'])
            
            # Create y-positions
            y_pos = np.arange(len(coef_names))
            
            # Calculate error bar sizes
            errors_minus = np.array([val - lower for val, lower in zip(coef_values, ci_lowers)])
            errors_plus = np.array([upper - val for val, upper in zip(coef_values, ci_uppers)])
            
            # Set colors based on significance
            colors = [PASTEL_COLORS[0] if sig else PASTEL_COLORS[4] for sig in is_significant]
            
            # Create coefficient plot with safer error bar handling
            ax5.errorbar(coef_values, y_pos, xerr=[errors_minus, errors_plus], fmt='o', 
                       capsize=5, color='black', markersize=8)
            
            # Add coefficient markers with colors
            for i, (x, y, color) in enumerate(zip(coef_values, y_pos, colors)):
                ax5.scatter(x, y, color=color, s=100, zorder=10)
            
            # Add reference line at 0
            ax5.axvline(x=0, color=PASTEL_COLORS[5], linestyle='--', alpha=0.7)
            
            # Add coefficient labels
            ax5.set_yticks(y_pos)
            ax5.set_yticklabels(coef_names)
            
            # Add value annotations with proper spacing
            x_range = max(max(coef_values) - min(coef_values), 0.1)  # Prevent division by zero
            for i, (val, lower, upper, sig) in enumerate(zip(coef_values, ci_lowers, ci_uppers, is_significant)):
                text = f"{val:.3f} [{lower:.3f}, {upper:.3f}]"
                if sig:
                    text += " *"
                ax5.text(upper + 0.05 * x_range, i, text, va='center', fontsize=9)
            
            # Add labels and title
            ax5.set_xlabel('Coefficient Value (with 95% CI)')
            ax5.set_title('Fixed Effects Coefficients')
            
            # Add a legend for significance
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=PASTEL_COLORS[0], markersize=10, label='Significant'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=PASTEL_COLORS[4], markersize=10, label='Non-significant')
            ]
            ax5.legend(handles=legend_elements, loc='best')
            
            # Set reasonable x-axis limits
            all_vals = coef_values + ci_lowers + ci_uppers
            xmin = min(all_vals) - 0.1 * x_range
            xmax = max(all_vals) + 0.3 * x_range  # Extra space for text
            ax5.set_xlim(xmin, xmax)
            
            # Add a grid
            ax5.grid(axis='x', linestyle='--', alpha=0.7)
            
            fig5.tight_layout()
            figures['coefficients_plot'] = fig_to_svg(fig5)
        except Exception as e:
            figures['coefficients_plot_error'] = str(e)
        
        # Figure 6: Predicted vs. Actual Values
        try:
            fig6, ax6 = plt.subplots(figsize=(10, 6))
            
            # Plot predicted vs. actual values
            ax6.scatter(results.fittedvalues, y_true, alpha=0.6, color=PASTEL_COLORS[0])
            
            # Add a perfect prediction line
            min_val = min(min(results.fittedvalues), min(y_true))
            max_val = max(max(results.fittedvalues), max(y_true))
            ax6.plot([min_val, max_val], [min_val, max_val], '--', color=PASTEL_COLORS[5])
            
            # Add labels
            ax6.set_xlabel('Predicted Values')
            ax6.set_ylabel('Actual Values')
            ax6.set_title('Predicted vs. Actual Values')
            
            # Add R² information
            r2_text = f"Marginal R² = {marginal_r2:.3f}\nConditional R² = {conditional_r2:.3f}"
            ax6.text(0.05, 0.95, r2_text, transform=ax6.transAxes, 
                   va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
            
            fig6.tight_layout()
            figures['predicted_vs_actual'] = fig_to_svg(fig6)
        except Exception as e:
            figures['predicted_vs_actual_error'] = str(e)
        
        # Figure 8: Random Effects Visualization
        try:
            # For this visualization, we'll show the variance explained by different components
            fig8, ax8 = plt.subplots(figsize=(10, 6))
            
            # Calculate variance components
            random_var_total = sum(random_variance_components.values())
            residual_var = results.scale
            total_var = random_var_total + residual_var
            
            # Prepare data for pie chart
            labels = ['Residual']
            sizes = [residual_var]
            
            # Add random effects components
            for key, val in random_variance_components.items():
                if isinstance(val, (int, float)):
                    labels.append(key.replace('random_var_', 'Random '))
                    sizes.append(val)
            
            # Calculate percentages
            percentages = [100 * s / total_var for s in sizes]
            
            # Create pie chart
            wedges, texts, autotexts = ax8.pie(
                sizes, 
                labels=None, 
                autopct='%1.1f%%',
                textprops={'color': "w", 'fontweight': 'bold'},
                startangle=90,
                colors=PASTEL_COLORS[:len(sizes)]  # Use pastel colors for pie slices
            )
            
            # Change text color for better readability
            for i, autotext in enumerate(autotexts):
                if percentages[i] < 10:  # If slice is small, make text dark
                    autotext.set_color('black')
            
            # Create a legend
            ax8.legend(wedges, labels, title="Variance Components", 
                     loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            
            # Add title
            ax8.set_title('Variance Decomposition', fontsize=14)
            
            # Add ICC information if available
            if isinstance(icc, (int, float)):
                icc_text = f"Intraclass Correlation (ICC): {icc:.3f}"
                ax8.annotate(icc_text, xy=(0.5, -0.1), xycoords='axes fraction', 
                           ha='center', va='center', fontweight='bold',
                           bbox=dict(boxstyle='round', fc='white', alpha=0.8))
            
            fig8.tight_layout()
            figures['variance_decomposition'] = fig_to_svg(fig8)
        except Exception as e:
            figures['variance_decomposition_error'] = str(e)
        
        # Figure 9: Residuals by Group
        try:
            # For this visualization, we'll show boxplots of residuals by the primary grouping variable
            fig9, ax9 = plt.subplots(figsize=(12, 6))
            
            # Create dataframe with residuals and group
            resid_df = pd.DataFrame({
                'residuals': results.resid,
                'group': data_clean[random_effects[0]]
            })
            
            # If there are too many groups, limit to the first 15
            unique_groups = resid_df['group'].unique()
            if len(unique_groups) > 15:
                selected_groups = unique_groups[:15]
                resid_df = resid_df[resid_df['group'].isin(selected_groups)]
                ax9.text(0.5, -0.1, f"Note: Showing only 15 out of {len(unique_groups)} groups", 
                       transform=ax9.transAxes, ha='center', va='center', 
                       bbox=dict(boxstyle='round', fc='yellow', alpha=0.2))
            
            # Create boxplot
            sns.boxplot(x='group', y='residuals', data=resid_df, ax=ax9, 
                       palette=PASTEL_CMAP[:len(resid_df['group'].unique())])
            
            # Add reference line at 0
            ax9.axhline(y=0, color=PASTEL_COLORS[5], linestyle='--')
            
            # Add title and labels
            ax9.set_title(f'Residuals by {random_effects[0]}', fontsize=14)
            ax9.set_xlabel(random_effects[0])
            ax9.set_ylabel('Residuals')
            
            # Rotate x-axis labels if needed
            plt.xticks(rotation=45 if len(unique_groups) > 5 else 0, ha='right' if len(unique_groups) > 5 else 'center')
            
            fig9.tight_layout()
            figures['residuals_by_group'] = fig_to_svg(fig9)
        except Exception as e:
            figures['residuals_by_group_error'] = str(e)
        
        # Figure 10: VIF (Variance Inflation Factor) for Multicollinearity
        try:
            if 'MULTICOLLINEARITY' in assumptions and 'vif_values' in assumptions['MULTICOLLINEARITY']:
                vif_data = assumptions['MULTICOLLINEARITY']['vif_values']
                
                if vif_data and all(v is not None for v in vif_data.values()):
                    fig10, ax10 = plt.subplots(figsize=(10, 6))
                    
                    # Sort VIF values for better visualization
                    sorted_vif = {k: v for k, v in sorted(vif_data.items(), key=lambda item: item[1], reverse=True)}
                    
                    # Create bar chart
                    names = list(sorted_vif.keys())
                    values = list(sorted_vif.values())
                    
                    # Set colors based on VIF thresholds
                    colors = [PASTEL_COLORS[1] if v <= 5 else PASTEL_COLORS[2] if v <= 10 else PASTEL_COLORS[0] 
                             for v in values]
                    
                    # Create bar chart
                    bars = ax10.bar(names, values, color=colors, alpha=0.7)
                    
                    # Add reference lines for VIF thresholds
                    ax10.axhline(y=5, color=PASTEL_COLORS[3], linestyle='--', alpha=0.7, label='VIF = 5')
                    ax10.axhline(y=10, color=PASTEL_COLORS[0], linestyle='--', alpha=0.7, label='VIF = 10')
                    
                    # Add value labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax10.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{height:.2f}', ha='center', va='bottom')
                    
                    # Add title and labels
                    ax10.set_title('Variance Inflation Factors (VIF)', fontsize=14)
                    ax10.set_ylabel('VIF Value')
                    ax10.set_xlabel('Predictor')
                    
                    # Rotate x-axis labels if there are many predictors
                    plt.xticks(rotation=45 if len(names) > 5 else 0, ha='right' if len(names) > 5 else 'center')
                    
                    # Add legend
                    ax10.legend()
                    
                    # Add interpretation note
                    note = (
                        "VIF < 5: Low multicollinearity\n"
                        "5 < VIF < 10: Moderate multicollinearity\n"
                        "VIF > 10: High multicollinearity"
                    )
                    ax10.text(0.95, 0.95, note, transform=ax10.transAxes, 
                           va='top', ha='right', bbox=dict(boxstyle='round', fc='white', alpha=0.8),
                           fontsize=9)
                    
                    fig10.tight_layout()
                    figures['vif_plot'] = fig_to_svg(fig10)
        except Exception as e:
            figures['vif_plot_error'] = str(e)
        
        # Compile the full results
        def ensure_string_keys(d):
            """Recursively convert all dictionary keys to strings."""
            if not isinstance(d, dict):
                return d
            
            return {
                str(k): ensure_string_keys(v) for k, v in d.items()
            }

        assumptions = ensure_string_keys(assumptions)

        return {
            'test': 'Linear Mixed Effects Model',
            'model_summary': str(results.summary()),
            'fixed_effects': fixed_effects_results,
            'random_effects_variance': random_variance_components,
            'residual_variance': float(results.scale),
            'model_statistics': {
                'aic': float(aic),
                'bic': float(bic),
                'marginal_r2': float(marginal_r2),
                'conditional_r2': float(conditional_r2),
                'rmse': float(rmse),
                'mae': float(mae),
                'icc': float(icc) if isinstance(icc, (int, float)) else None
            },
            'assumptions': assumptions,
            'interpretation': interpretation,
            'figures': figures
        }
    except Exception as e:
        return {
            'test': 'Linear Mixed Effects Model',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'satisfied': False
        }