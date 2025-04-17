import traceback
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Dict, List, Any, Optional
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import label_binarize
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy import dmatrices
from data.assumptions.tests import ModelSpecificationTest
from data.assumptions.tests import MulticollinearityTest, InfluentialPointsTest, LinearityTest, SampleSizeTest, IndependenceTest, SeparabilityTest, AssumptionResult


def fig_to_svg(fig):
    # Set figure and axes backgrounds to transparent
    fig.patch.set_alpha(0.0)
    for ax in fig.get_axes():
        ax.patch.set_alpha(0.0)
        # Make legend background transparent if it exists
        if ax.get_legend() is not None:
            ax.get_legend().get_frame().set_alpha(0.0)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight', transparent=True)
    buf.seek(0)
    svg_string = buf.getvalue().decode('utf-8')
    buf.close()
    plt.close(fig)
    return svg_string

# Define pastel color palette
PASTEL_COLORS = ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#FFB3F7', '#B3FFF7']
PASTEL_CMAP = sns.color_palette(PASTEL_COLORS)

def multinomial_logistic_regression(data: pd.DataFrame, outcome: str, predictors: List[str], alpha: float) -> Dict[str, Any]:
    """Performs Multinomial Logistic Regression with comprehensive statistics, visualizations, and assumption checks."""
    try:
        # Make a copy of the data to avoid modifying the original
        data_copy = data.copy()
        
        # Ensure the outcome variable is correctly formatted as a categorical variable
        if pd.api.types.is_numeric_dtype(data_copy[outcome]):
            data_copy[outcome] = data_copy[outcome].astype(str)
                
        # Check if outcome has more than 2 categories
        n_categories = len(data_copy[outcome].unique())
        if n_categories < 3:
            raise ValueError(f"Multinomial logistic regression requires 3+ categories in outcome. Found {n_categories}.")
            
        # Construct formula for patsy
        formula = f"{outcome} ~ {' + '.join(predictors)}"
        
        # Use patsy directly to create design matrices
        y, X = dmatrices(formula, data_copy, return_type='dataframe')
        
        # Convert y to a 1D array of category indices
        y_1d = y.iloc[:, 0].astype(str)
        
        # Fit the model using MNLogit instead of formula
        model = sm.MNLogit(y_1d, X)
        results = model.fit()
        
        # Extract model parameters
        aic = float(results.aic)
        bic = float(results.bic)
        log_likelihood = float(results.llf)
        
        # Get categories and reference category
        categories = sorted(y_1d.unique())
        ref_category = categories[0]
        
        # Save all categories from the original data for prediction and other calculations
        all_categories = sorted(data_copy[outcome].astype(str).unique())
        
        # Ensure consistent ordering of categories
        if set(categories) != set(all_categories):
            print(f"Warning: Categories from model ({categories}) differ from data ({all_categories})")
            
        # Extract coefficients - these are per category compared to reference
        coefficients = {}
        for i, category in enumerate(categories[1:]):  # Skip reference category
            category_coeffs = {}
            
            # Get the parameters for this category (MNLogit format)
            params = results.params[i]
            bse = results.bse[i] if hasattr(results, 'bse') else None
            tvalues = results.tvalues[i] if hasattr(results, 'tvalues') else None
            pvalues = results.pvalues[i] if hasattr(results, 'pvalues') else None
            
            # Process each predictor
            for j, predictor in enumerate(X.columns):
                # Get coefficient values
                estimate = float(params[j])
                std_error = float(bse[j]) if bse is not None else None
                z_value = float(tvalues[j]) if tvalues is not None else None
                p_value = float(pvalues[j]) if pvalues is not None else None
                
                # Calculate odds ratio and CI
                odds_ratio = np.exp(estimate)
                ci_lower = np.exp(estimate - 1.96 * std_error) if std_error else None
                ci_upper = np.exp(estimate + 1.96 * std_error) if std_error else None
                
                category_coeffs[predictor] = {
                    'estimate': estimate,
                    'std_error': std_error,
                    'z_value': z_value,
                    'p_value': p_value,
                    'significant': p_value < alpha if p_value is not None else None,
                    'odds_ratio': float(odds_ratio),
                    'ci_lower': float(ci_lower) if ci_lower is not None else None,
                    'ci_upper': float(ci_upper) if ci_upper is not None else None
                }
                
            coefficients[category] = category_coeffs
            
        # Calculate model fit statistics
        # Null model (intercept only)
        try:
            # Create null model with only intercept
            null_formula = f"{outcome} ~ 1"
            y_null, X_null = dmatrices(null_formula, data_copy, return_type='dataframe')
            
            # Convert y to a 1D array
            y_null_1d = y_null.iloc[:, 0].astype(str)
            
            # Fit the null model
            null_model = sm.MNLogit(y_null_1d, X_null)
            null_results = null_model.fit(disp=0)  # Suppress convergence messages
            
            # Likelihood ratio test
            lr_stat = 2 * (results.llf - null_results.llf)
            df = len(predictors) * (n_categories - 1)  # Parameters per category * (categories - 1)
            lr_pval = stats.chi2.sf(lr_stat, df)
            
            # McFadden's pseudo R-squared
            mcfadden_r2 = 1 - (results.llf / null_results.llf)
            
            # Cox-Snell R-squared
            cox_snell_r2 = 1 - np.exp(-lr_stat / len(data_copy))
            
            # Nagelkerke R-squared
            nagelkerke_r2 = cox_snell_r2 / (1 - np.exp((2 * null_results.llf) / len(data_copy)))
            
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
            
        # Calculate predictions and classification accuracy
        try:
            # Get predicted probabilities
            predicted_probs = results.predict()
            
            # First get the exact string categories from the data, ensuring they match the outcome variable
            all_categories = sorted(data_copy[outcome].astype(str).unique())
            
            # Get true categories directly from data
            true_categories = np.array(data_copy[outcome].astype(str))
            
            # Get predicted category (highest probability)
            predicted_categories = []
            for i, probs in enumerate(predicted_probs):
                # Get the index of the highest probability
                idx = np.argmax(probs)
                # Map back to the corresponding category
                if idx < len(all_categories):
                    category = all_categories[idx]
                    predicted_categories.append(category)
                else:
                    # Fallback in case of index error
                    predicted_categories.append(all_categories[0])
            
            # Convert to numpy array
            pred_categories = np.array(predicted_categories)
            
            # Overall accuracy
            accuracy = np.mean(pred_categories == true_categories)
            
            # Create confusion matrix with the correct labels
            conf_matrix = confusion_matrix(true_categories, pred_categories, labels=all_categories)
            
            # Calculate per-class metrics
            per_class_metrics = {}
            for i, category in enumerate(all_categories):
                true_pos = conf_matrix[i, i]
                false_pos = conf_matrix[:, i].sum() - true_pos
                false_neg = conf_matrix[i, :].sum() - true_pos
                true_neg = conf_matrix.sum() - true_pos - false_pos - false_neg
                
                # Calculate metrics
                precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
                recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                per_class_metrics[category] = {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'support': int(conf_matrix[i, :].sum())
                }
            
            # Calculate macro and weighted averages
            macro_precision = np.mean([m['precision'] for m in per_class_metrics.values()])
            macro_recall = np.mean([m['recall'] for m in per_class_metrics.values()])
            macro_f1 = np.mean([m['f1_score'] for m in per_class_metrics.values()])
            
            # Weighted averages
            total_support = sum(m['support'] for m in per_class_metrics.values())
            weighted_precision = sum(m['precision'] * m['support'] for m in per_class_metrics.values()) / total_support
            weighted_recall = sum(m['recall'] * m['support'] for m in per_class_metrics.values()) / total_support
            weighted_f1 = sum(m['f1_score'] * m['support'] for m in per_class_metrics.values()) / total_support
            
            prediction_stats = {
                'accuracy': float(accuracy),
                'confusion_matrix': conf_matrix.tolist(),
                'per_class': per_class_metrics,
                'macro_avg': {
                    'precision': float(macro_precision),
                    'recall': float(macro_recall),
                    'f1_score': float(macro_f1)
                },
                'weighted_avg': {
                    'precision': float(weighted_precision),
                    'recall': float(weighted_recall),
                    'f1_score': float(weighted_f1)
                }
            }
        except Exception as e:
            prediction_stats = {
                'error': f'Could not calculate predictions: {str(e)}'
            }
            
        # Calculate ROC curves and AUC (one-vs-rest approach)
        try:
            # Use the same string categories from outcome column directly
            all_categories = sorted(data_copy[outcome].astype(str).unique())
            
            # Explicitly set the categories for binarization to match exactly what's in the data
            y_true_bin = label_binarize(true_categories, classes=all_categories)
            
            # Calculate ROC curves for each class
            roc_data = {}
            
            for i, category in enumerate(all_categories):
                # Ensure index is within bounds of predicted_probs
                if i < predicted_probs.shape[1]:
                    category_probs = predicted_probs[:, i]
                    
                    # Calculate ROC curve and AUC with explicit error handling
                    try:
                        # Verify the binary labels for this category are valid
                        if len(np.unique(y_true_bin[:, i])) > 1:  # Must have both 0 and 1 values
                            fpr, tpr, _ = roc_curve(y_true_bin[:, i], category_probs)
                            roc_auc = auc(fpr, tpr)
                        else:
                            # Can't calculate ROC for a single class
                            fpr, tpr = [0, 1], [0, 1]
                            roc_auc = 0.5
                            print(f"Cannot calculate ROC curve for category {category}: only one class present")
                    except Exception as e:
                        # Handle any other errors
                        fpr, tpr = [0, 1], [0, 1]
                        roc_auc = 0.5
                        print(f"ROC calculation error for category {category}: {str(e)}")
                    
                    roc_data[category] = {
                        'false_positive_rate': fpr.tolist(),
                        'true_positive_rate': tpr.tolist(),
                        'auc': float(roc_auc)
                    }
            
            # Calculate macro-average AUC only for valid values
            valid_aucs = [data['auc'] for data in roc_data.values() 
                         if not np.isnan(data['auc']) and data['auc'] > 0 and data['auc'] <= 1]
            
            macro_auc = np.mean(valid_aucs) if valid_aucs else np.nan
                
            prediction_stats['roc_curves'] = roc_data
            prediction_stats['macro_avg_auc'] = float(macro_auc)
        except Exception as e:
            prediction_stats['roc_error'] = f'Could not calculate ROC curves: {str(e)}'
            
        # Test assumptions
        assumptions = {}
        
        # 1. Check for multicollinearity among predictors
        try:
            # First instantiate the test object
            multicollinearity_test = MulticollinearityTest()
            
            # Then run the test with the correct parameters
            multicollinearity_result = multicollinearity_test.run_test(df=data_copy, covariates=predictors)
            
            # Store results
            assumptions['multicollinearity'] = multicollinearity_result
        except Exception as e:
            assumptions['multicollinearity'] = {
                'error': str(e),
            }
        
        # 2. Check for influential observations
        try:
            # For multinomial models, check each category separately
            for i, category in enumerate(all_categories[1:]):  # Skip reference category
                category_idx = i
                try:
                    if category_idx < predicted_probs.shape[1]:
                        # Get actual binary outcome for this category
                        y_true_category = (true_categories == category).astype(int)
                        y_pred_category = predicted_probs[:, category_idx]
                        
                        # Calculate actual residuals
                        residuals_category = y_true_category - y_pred_category
                        
                        # Use proper design matrix
                        X_design = sm.add_constant(data_copy[predictors])
                        
                        # Calculate leverage with proper error handling
                        try:
                            # Calculate proper hat values if possible
                            hat_matrix = X_design @ np.linalg.inv(X_design.T @ X_design) @ X_design.T
                            leverage = np.diag(hat_matrix)
                        except:
                            # Fallback to a simple approximation if matrix is singular
                            leverage = np.ones(len(residuals_category)) * (len(predictors) + 1) / len(residuals_category)
                        
                        # Instantiate test object first
                        influential_test = InfluentialPointsTest()
                        
                        # Then run the test
                        influential_result = influential_test.run_test(
                            residuals=residuals_category,
                            leverage=leverage, 
                            fitted=y_pred_category,
                            X=X_design
                        )
                        
                        cat_key = f"influential_points_{category}"
                        assumptions[cat_key] = influential_result
                except Exception as e:
                    assumptions[f"influential_points_{category}"] = {
                        'error': str(e),
                    }
        except Exception as e:
            assumptions['influential_points'] = {
                'error': str(e),
            }
        
        # 3. Check for linearity of logit for continuous predictors
        try:
            # Only use actual continuous predictors
            continuous_predictors = []
            for pred in predictors:
                if pd.api.types.is_numeric_dtype(data_copy[pred]) and not pd.api.types.is_bool_dtype(data_copy[pred]):
                    continuous_predictors.append(pred)
            
            if continuous_predictors:
                for i, category in enumerate(all_categories[1:]):  # Skip reference category
                    category_idx = i
                    try:
                        if category_idx < predicted_probs.shape[1]:
                            # Get actual probabilities
                            y_pred_category = predicted_probs[:, category_idx]
                            
                            for pred in continuous_predictors:
                                # Calculate actual log odds for this category
                                # Add small epsilon to avoid log(0) and log(1)
                                epsilon = 1e-10
                                y_pred_safe = np.clip(y_pred_category, epsilon, 1-epsilon)
                                log_odds = np.log(y_pred_safe / (1 - y_pred_safe))
                                
                                # Instantiate test object first
                                linearity_test = LinearityTest()
                                
                                # Then run the test
                                linearity_result = linearity_test.run_test(x=data_copy[pred], y=log_odds)
                                
                                key = f"linearity_{category}_{pred}"
                                assumptions[key] = linearity_result
                    except Exception as e:
                        assumptions[f"linearity_{category}"] = {
                            'error': str(e),
                        }
        except Exception as e:
            assumptions['linearity'] = {
                'error': str(e),
            }
        
        # 4. Check for sample size adequacy
        try:
            # For multinomial logistic regression, check sample size for each category
            category_counts = data_copy[outcome].value_counts()
            min_required = 10 * len(predictors)  # Standard rule of thumb
            
            # Check each category with actual data
            for category, count in category_counts.items():
                try:
                    # Get actual data for this category
                    category_data = data_copy[data_copy[outcome] == category]
                    
                    # Instantiate test object first
                    sample_size_test = SampleSizeTest()
                    
                    # Then run the test - extract a representative numeric column for testing
                    sample_data = None
                    for pred in predictors:
                        if pd.api.types.is_numeric_dtype(category_data[pred]):
                            sample_data = category_data[pred].values
                            break
                    
                    # If no numeric predictor is found, use a dummy array of appropriate length
                    if sample_data is None:
                        sample_data = np.ones(len(category_data))
                        
                    sample_size_result = sample_size_test.run_test(data=sample_data, min_recommended=min_required)
                    
                    cat_key = f"sample_size_{category}"
                    assumptions[cat_key] = sample_size_result
                except Exception as e:
                    assumptions[f"sample_size_{category}"] = {
                        'error': str(e),
                    }
        except Exception as e:
            assumptions['sample_size'] = {
                'error': str(e),
            }
        
        # 5. Check for independence of observations
        try:
            # Don't use resid_response since it doesn't work with multinomial models
            # Instead, check independence for each category separately
            independence_results = {}
            
            # Loop through categories and test independence for each one
            for i, category in enumerate(all_categories[1:]):  # Skip reference category
                try:
                    # Get binary outcome for this category (1 if belongs to category, 0 otherwise)
                    y_true_category = (true_categories == category).astype(int)
                    
                    # Get predicted probability for this category
                    if i < predicted_probs.shape[1]:
                        y_pred_category = predicted_probs[:, i]
                        
                        # Calculate residuals manually for this category
                        residuals_category = y_true_category - y_pred_category
                        
                        # Pass these residuals to the independence test
                        independence_test = IndependenceTest()
                        cat_result = independence_test.run_test(data=residuals_category)
                        
                        independence_results[category] = cat_result
                except Exception as cat_e:
                    independence_results[category] = {
                        "result": AssumptionResult.NOT_APPLICABLE,
                        "message": f"Independence test failed for category {category}",
                        "details": {"error": str(cat_e)}
                    }
            
            # Create a summary result
            if independence_results:
                # Count how many categories passed
                passed = sum(1 for r in independence_results.values() 
                            if isinstance(r, dict) and r.get("result") == AssumptionResult.PASSED)
                total = len(independence_results)
                
                # Create overall assessment
                if passed == total:
                    result = AssumptionResult.PASSED
                    message = "Residuals for all categories appear to be independent."
                elif passed >= total / 2:
                    result = AssumptionResult.WARNING
                    message = f"Residuals for {passed} out of {total} categories appear independent."
                else:
                    result = AssumptionResult.FAILED
                    message = f"Residuals for only {passed} out of {total} categories appear independent."
                
                assumptions['independence'] = {
                    "result": result,
                    "message": message,
                    "details": {
                        "category_results": independence_results,
                        "note": "Independence was tested separately for each category."
                    }
                }
            else:
                # Fallback if no categories could be tested
                assumptions['independence'] = {
                    "result": AssumptionResult.NOT_APPLICABLE,
                    "message": "Independence test could not be performed for any category.",
                    "details": {"error": "No valid residuals could be calculated"}
                }
        except Exception as e:
            # Catch-all error handler with detailed information
            assumptions['independence'] = {
                "result": AssumptionResult.NOT_APPLICABLE,
                "message": "Independence test error",
                "details": {
                    "error": str(e),
                    "note": "Independence testing for multinomial models requires category-specific residuals."
                }
            }
        
        # 6. Check for separability issues
        try:
            # Instantiate test object first
            separability_test = SeparabilityTest()
            
            # Then run the test
            separability_result = separability_test.run_test(df=data_copy, outcome=outcome, factors=predictors)
            
            assumptions['separability'] = separability_result
        except Exception as e:
            assumptions['separability'] = {
                'error': str(e),
            }
        
        # 7. Check for model specification
        try:
            # Use actual design matrix
            X_spec = sm.add_constant(data_copy[predictors])
            
            # For multinomial models, check each category
            for i, category in enumerate(all_categories[1:]):  # Skip reference category
                category_idx = i
                try:
                    if category_idx < predicted_probs.shape[1]:
                        # Get actual data for this category
                        y_true_category = (true_categories == category).astype(int)
                        y_pred_category = predicted_probs[:, category_idx]
                        residuals_category = y_true_category - y_pred_category
                        
                        # Instantiate test object first
                        specification_test = ModelSpecificationTest()
                        
                        # Then run the test
                        specification_result = specification_test.run_test(
                            residuals=residuals_category,
                            fitted=y_pred_category,
                            X=X_spec
                        )
                        
                        cat_key = f"model_specification_{category}"
                        assumptions[cat_key] = specification_result
                except Exception as e:
                    assumptions[f"model_specification_{category}"] = {
                        'error': str(e),
                    }
        except Exception as e:
            assumptions['model_specification'] = {
                'error': str(e),
            }
        
        # Find significant predictors across all categories
        significant_predictors = []
        for category, category_coeffs in coefficients.items():
            for predictor, coef in category_coeffs.items():
                if coef.get('significant', False) and predictor != 'Intercept':
                    significant_predictors.append((category, predictor))
        # Create interpretation
        interpretation = f"Multinomial Logistic Regression with {outcome} as outcome (reference: {ref_category}) "
        interpretation += f"and predictors ({', '.join(predictors)}).\n\n"
        
        # Interpret model fit
        if 'lr_p_value' in model_fit:
            interpretation += f"The model is {'statistically significant' if model_fit['significant'] else 'not statistically significant'} "
            interpretation += f"compared to the null model (χ²({model_fit['lr_df']}) = {model_fit['lr_statistic']:.3f}, p = {model_fit['lr_p_value']:.5f}).\n"
            
            if 'mcfadden_r2' in model_fit:
                interpretation += f"McFadden's pseudo R² = {model_fit['mcfadden_r2']:.3f}, "
                
                # Add interpretation of McFadden's R2
                if model_fit['mcfadden_r2'] < 0.2:
                    interpretation += "indicating a modest fit. "
                elif model_fit['mcfadden_r2'] < 0.4:
                    interpretation += "indicating a reasonable fit. "
                else:
                    interpretation += "indicating a good fit. "
                    
            if 'nagelkerke_r2' in model_fit:
                interpretation += f"Nagelkerke's R² = {model_fit['nagelkerke_r2']:.3f}.\n\n"
        
        # Interpret coefficients
        if significant_predictors:
            interpretation += "Significant predictors:\n"
            for category, predictor in significant_predictors:
                coef = coefficients[category][predictor]
                interpretation += f"- {predictor} (for category '{category}'): β = {coef['estimate']:.3f}, OR = {coef['odds_ratio']:.3f} "
                interpretation += f"(95% CI: {coef['ci_lower']:.3f}-{coef['ci_upper']:.3f}), p = {coef['p_value']:.5f}\n"
                
                # Interpret direction of effect
                if coef['odds_ratio'] > 1:
                    interpretation += f"  For each unit increase in {predictor}, the odds of being in category '{category}' rather than '{ref_category}' "
                    interpretation += f"increase by {(coef['odds_ratio']-1)*100:.1f}%.\n"
                else:
                    interpretation += f"  For each unit increase in {predictor}, the odds of being in category '{category}' rather than '{ref_category}' "
                    interpretation += f"decrease by {(1-coef['odds_ratio'])*100:.1f}%.\n"
        else:
            interpretation += "No significant predictors were found.\n"
            
        # Model performance
        if 'accuracy' in prediction_stats:
            interpretation += f"\nModel prediction accuracy: {prediction_stats['accuracy']*100:.1f}%.\n"
            
            # Add info on precision and recall
            if 'weighted_avg' in prediction_stats:
                w_avg = prediction_stats['weighted_avg']
                interpretation += f"Weighted average precision: {w_avg['precision']*100:.1f}%, "
                interpretation += f"recall: {w_avg['recall']*100:.1f}%, "
                interpretation += f"F1-score: {w_avg['f1_score']*100:.1f}%.\n"
                
            # Add info on AUC if available
            if 'macro_avg_auc' in prediction_stats:
                interpretation += f"Macro-average AUC: {prediction_stats['macro_avg_auc']:.3f}.\n"
        
        # Assumption check summary
        interpretation += "\nAssumption checks:\n"
        for name, result in assumptions.items():
            # Skip categories with their own sections
            if any(name.startswith(prefix) and name.count('_') > 1 for prefix in 
                  ['sample_size_', 'influential_points_', 'linearity_', 'model_specification_']):
                continue
                
            # Skip details keys and error keys that will be handled with their main keys
            if '_details' in name or '_vif_values' in name or name.endswith('_error'):
                continue
            
            # Handle the different types of result values
            if isinstance(result, AssumptionResult):
                # For enum values, use the name directly
                result_text = "Satisfied" if result == AssumptionResult.PASSED else "Warning" if result == AssumptionResult.WARNING else "Failed" if result == AssumptionResult.FAILED else "Not Applicable"
                detail_key = f"{name}_details"
                details = assumptions.get(detail_key, "No details available")
                interpretation += f"- {name.replace('_', ' ').title()}: {result_text}. {details}\n"
            elif isinstance(result, dict):
                if 'satisfied' in result:
                    # For the dictionary format with satisfied flag
                    result_text = "Satisfied" if result['satisfied'] else "Failed"
                    details = result.get('details', "No details available")
                    interpretation += f"- {name.replace('_', ' ').title()}: {result_text}. {details}\n"
                elif 'result' in result:
                    # For newer format with result key
                    result_text = result['result'].name if hasattr(result['result'], 'name') else str(result['result'])
                    details = result.get('details', "No details available")
                    interpretation += f"- {name.replace('_', ' ').title()}: {result_text}. {details}\n"
                elif 'error' in result:
                    # For error dictionary format
                    interpretation += f"- {name.replace('_', ' ').title()}: Error - {result['error']}\n"
            elif isinstance(result, str):
                # For simple string results
                interpretation += f"- {name.replace('_', ' ').title()}: {result}\n"
        
        # Now add categorical assumption results - group by assumption type
        category_assumptions = {
            'Sample Size': {},
            'Influential Points': {},
            'Linearity': {},
            'Model Specification': {}
        }
        
        # Collect all category-specific assumptions
        for name, result in assumptions.items():
            if name.startswith('sample_size_'):
                category = name.replace('sample_size_', '')
                category_assumptions['Sample Size'][category] = result
            elif name.startswith('influential_points_'):
                category = name.replace('influential_points_', '')
                category_assumptions['Influential Points'][category] = result
            elif name.startswith('linearity_'):
                parts = name.replace('linearity_', '').split('_', 1)
                if len(parts) == 2:
                    category, predictor = parts
                    if category not in category_assumptions['Linearity']:
                        category_assumptions['Linearity'][category] = {}
                    category_assumptions['Linearity'][category][predictor] = result
            elif name.startswith('model_specification_'):
                category = name.replace('model_specification_', '')
                category_assumptions['Model Specification'][category] = result
        
        # Add categorical assumption summaries
        for assumption_type, categories in category_assumptions.items():
            if categories:
                interpretation += f"\n{assumption_type} by category:\n"
                for category, result in categories.items():
                    if isinstance(result, dict):
                        if 'satisfied' in result:
                            result_text = "Satisfied" if result['satisfied'] else "Failed"
                            details = result.get('details', "No details available")
                            interpretation += f"- {category}: {result_text}. {details}\n"
                        elif 'result' in result:
                            result_text = result['result'].name if hasattr(result['result'], 'name') else str(result['result'])
                            details = result.get('details', "No details available")
                            interpretation += f"- {category}: {result_text}. {details}\n"
                        elif 'error' in result:
                            interpretation += f"- {category}: Error - {result['error']}\n"
                        elif isinstance(result, dict) and not any(k in result for k in ['satisfied', 'result', 'error']):
                            # For linearity which has predictors as keys
                            interpretation += f"- {category}:\n"
                            for pred, pred_result in result.items():
                                if isinstance(pred_result, dict):
                                    if 'satisfied' in pred_result:
                                        pred_text = "Satisfied" if pred_result['satisfied'] else "Failed"
                                        pred_details = pred_result.get('details', "No details available")
                                        interpretation += f"  - {pred}: {pred_text}. {pred_details}\n"
                                    elif 'result' in pred_result:
                                        pred_text = pred_result['result'].name if hasattr(pred_result['result'], 'name') else str(pred_result['result'])
                                        pred_details = pred_result.get('details', "No details available")
                                        interpretation += f"  - {pred}: {pred_text}. {pred_details}\n"
                    elif isinstance(result, AssumptionResult):
                        result_text = "Satisfied" if result == AssumptionResult.PASSED else "Warning" if result == AssumptionResult.WARNING else "Failed" if result == AssumptionResult.FAILED else "Not Applicable"
                        interpretation += f"- {category}: {result_text}\n"
                    elif isinstance(result, str):
                        interpretation += f"- {category}: {result}\n"
        
        # Create visualizations
        figures = {}
        
        # Figure 1: Coefficient Plot for Odds Ratios with CI
        try:
            # Count how many predictors and categories we have to plot
            n_predictors = len(set(pred for cat in coefficients.values() for pred in cat.keys() if pred != 'Intercept'))
            
            # Get only the categories that are in the coefficients dictionary 
            plot_categories = list(coefficients.keys())
            n_categories = len(plot_categories)
            
            if n_predictors > 0 and n_categories > 0:
                # Create a figure with subplots for each category
                fig, axes = plt.subplots(n_categories, 1, figsize=(10, n_categories * 3), sharex=True)
                
                # If there's only one category, convert axes to array
                if n_categories == 1:
                    axes = np.array([axes])
                
                # Now iterate through our categories from the coefficients dict
                for i, category in enumerate(plot_categories):
                    ax = axes[i]
                    
                    # Extract predictors and their odds ratios for this category
                    cat_coeffs = coefficients[category]
                    predictors_to_plot = [p for p in cat_coeffs.keys() if p != 'Intercept']
                    
                    # Sort predictors by odds ratio for better visualization
                    predictors_to_plot = sorted(predictors_to_plot, 
                                            key=lambda p: cat_coeffs[p]['odds_ratio'])
                    
                    # Extract odds ratios and confidence intervals
                    odds_ratios = [cat_coeffs[p]['odds_ratio'] for p in predictors_to_plot]
                    ci_lowers = [cat_coeffs[p]['ci_lower'] for p in predictors_to_plot]
                    ci_uppers = [cat_coeffs[p]['ci_upper'] for p in predictors_to_plot]
                    
                    # Create error bars for plotting
                    errors_minus = np.array([or_val - ci_l for or_val, ci_l in zip(odds_ratios, ci_lowers)])
                    errors_plus = np.array([ci_u - or_val for or_val, ci_u in zip(odds_ratios, ci_uppers)])
                    
                    # Set colors based on significance
                    colors = ['tab:blue' if cat_coeffs[p]['significant'] else 'tab:gray' 
                           for p in predictors_to_plot]
                    
                    # Plot odds ratios with error bars
                    y_pos = np.arange(len(predictors_to_plot))
                    ax.errorbar(odds_ratios, y_pos, xerr=(errors_minus, errors_plus), 
                               fmt='o', capsize=5, color='black', alpha=0.7)
                    
                    # Add scatter points with colors
                    for j, (x, y, color) in enumerate(zip(odds_ratios, y_pos, colors)):
                        ax.scatter(x, y, color=color, s=100, zorder=10)
                    
                    # Add labels
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(predictors_to_plot)
                    
                    # Add vertical line at OR=1 (no effect)
                    ax.axvline(x=1, color='red', linestyle='--', alpha=0.7)
                    
                    # Add title for this category
                    ax.set_title(f"Odds Ratios for Category: {category} (vs. {ref_category})")
                    
                    # Add labels for OR values
                    for j, (or_val, y) in enumerate(zip(odds_ratios, y_pos)):
                        p_val = cat_coeffs[predictors_to_plot[j]]['p_value']
                        sig_str = '*' if p_val < alpha else ''
                        ax.text(max(or_val + errors_plus[j], 1.1 * or_val), y, 
                               f" {or_val:.3f} {sig_str}", va='center')
                    
                    # Set reasonable x limits with log scale
                    min_or = min([l for l in ci_lowers if l > 0] or [0.1])
                    max_or = max(ci_uppers or [10])
                    
                    # Use log scale if range is large
                    if max_or / min_or > 10:
                        ax.set_xscale('log')
                        ax.set_xlim(min_or / 2, max_or * 2)
                    else:
                        ax.set_xlim(min(0.5, min_or / 2), max(1.5, max_or * 1.2))
                    
                    # Add grid
                    ax.grid(True, alpha=0.3)
                
                # Add common x-label
                fig.text(0.5, 0.01, 'Odds Ratio (log scale)', ha='center', va='center', fontsize=12)
                
                # Add legend
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', markersize=10, label='Significant'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:gray', markersize=10, label='Non-significant')
                ]
                fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99))
                
                fig.tight_layout(rect=[0, 0.02, 1, 0.98])
                figures['odds_ratio_plot'] = fig_to_svg(fig)
            
        except Exception as e:
            figures['odds_ratio_plot_error'] = str(e)
            # Import traceback has been shadowed by a local variable somewhere
            import sys
            figures['odds_ratio_plot_traceback'] = ''.join(traceback.format_exception(*sys.exc_info()))
            
        # Figure 2: Confusion Matrix Heatmap
        try:
            if 'confusion_matrix' in prediction_stats:
                conf_matrix = np.array(prediction_stats['confusion_matrix'])
                
                # Create figure
                fig, ax = plt.subplots(figsize=(8 + n_categories*0.5, 6 + n_categories*0.5))
                
                # Create heatmap
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='RdPu', 
                          xticklabels=categories, yticklabels=categories, ax=ax)
                
                # Set labels and title
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                
                # Add accuracy on top
                accuracy = prediction_stats['accuracy']
                ax.text(0.5, 1.05, f"Accuracy: {accuracy:.3f}", 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                fig.tight_layout()
                figures['confusion_matrix'] = fig_to_svg(fig)
        except Exception as e:
            figures['confusion_matrix_error'] = str(e)
            
        # Figure 3: ROC Curves
        try:
            if 'roc_curves' in prediction_stats:
                roc_data = prediction_stats['roc_curves']
                
                # Create figure
                fig, ax = plt.subplots(figsize=(8, 8))
                
                # Plot ROC curve for each class
                for i, (category, data) in enumerate(roc_data.items()):
                    fpr = data['false_positive_rate']
                    tpr = data['true_positive_rate']
                    auc_val = data['auc']
                    
                    color_idx = i % len(PASTEL_COLORS)
                    ax.plot(fpr, tpr, lw=2, color=PASTEL_COLORS[color_idx],
                            label=f'{category} (AUC = {auc_val:.3f})')
                
                # Plot diagonal reference line
                ax.plot([0, 1], [0, 1], 'k--', lw=1)
                
                # Set labels and title
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curves (One-vs-Rest)')
                
                # Add legend
                ax.legend(loc='lower right')
                
                # Set limits and grid
                ax.set_xlim([-0.01, 1.01])
                ax.set_ylim([-0.01, 1.01])
                ax.grid(True, alpha=0.3)
                
                fig.tight_layout()
                figures['roc_curves'] = fig_to_svg(fig)
        except Exception as e:
            figures['roc_curves_error'] = str(e)
            
        # Figure 4: Prediction Probability Distribution
        try:
            if 'predicted_probs' in locals():
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # If we don't have probabilities for the reference category, calculate them
                if predicted_probs.shape[1] < len(all_categories):
                    all_probs = np.column_stack((1 - predicted_probs.sum(axis=1), predicted_probs))
                else:
                    all_probs = predicted_probs
                
                # Create violin plots for each category
                data_to_plot = []
                category_labels = []
                colors = []
                
                for i, category in enumerate(all_categories):
                    # Get the probabilities for this category
                    if i < all_probs.shape[1]:
                        cat_probs = all_probs[:, i]
                        
                        # Get actual labels for this category
                        actual_labels = (true_categories == category)
                        
                        # Create separate for correct and incorrect predictions
                        correct_probs = cat_probs[actual_labels]
                        incorrect_probs = cat_probs[~actual_labels]
                        
                        if len(correct_probs) > 0:
                            data_to_plot.append(correct_probs)
                            category_labels.append(f"{category}\n(Correct)")
                            colors.append(PASTEL_COLORS[0])
                            
                        if len(incorrect_probs) > 0:
                            data_to_plot.append(incorrect_probs)
                            category_labels.append(f"{category}\n(Incorrect)")
                            colors.append(PASTEL_COLORS[2])
                
                # Create violin plots
                if data_to_plot:
                    parts = ax.violinplot(data_to_plot, showmeans=True, showmedians=True)
                    
                    # Color the violins
                    for i, pc in enumerate(parts['bodies']):
                        pc.set_facecolor(colors[i])
                        pc.set_alpha(0.7)
                    
                    # Add jittered points
                    for i, d in enumerate(data_to_plot):
                        x = np.random.normal(i+1, 0.04, size=len(d))
                        ax.scatter(x, d, alpha=0.3, edgecolor='none', s=20, c=colors[i])
                    
                    # Set labels and title
                    ax.set_xticks(np.arange(1, len(data_to_plot) + 1))
                    ax.set_xticklabels(category_labels)
                    ax.set_ylabel('Predicted Probability')
                    ax.set_title('Distribution of Predicted Probabilities')
                    
                    # Add horizontal line at 0.5
                    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.7)
                    
                    fig.tight_layout()
                    figures['prediction_distribution'] = fig_to_svg(fig)
        except Exception as e:
            figures['prediction_distribution_error'] = str(e)
            
        # Figure 5: Assumption Check Summary
        try:
            # Create a visual summary of assumptions
            assumption_names = []
            assumption_satisfied = []
            assumption_details = []
            
            for name, result in assumptions.items():
                if 'satisfied' in result:
                    assumption_names.append(name.replace('_', ' ').title())
                    satisfied = result['satisfied']
                    assumption_satisfied.append(satisfied)
                    assumption_details.append(result.get('details', ''))
            
            if assumption_names:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Set colors based on satisfaction
                colors = ['green' if s else 'red' for s in assumption_satisfied]
                
                # Create bars
                y_pos = np.arange(len(assumption_names))
                bars = ax.barh(y_pos, [1] * len(assumption_names), color=colors, alpha=0.7)
                
                # Add text for each bar
                for i, bar in enumerate(bars):
                    text = 'Satisfied' if assumption_satisfied[i] else 'Violated'
                    ax.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, 
                           text, ha='center', va='center', color='white', 
                           fontweight='bold')
                
                # Set y-tick labels
                ax.set_yticks(y_pos)
                ax.set_yticklabels(assumption_names)
                
                # Hide x-axis
                ax.set_xticks([])
                ax.set_xlim(0, 1)
                
                # Set title
                ax.set_title('Assumption Check Summary')
                
                # Add overall assessment
                all_satisfied = all(assumption_satisfied)
                most_satisfied = sum(assumption_satisfied) / len(assumption_satisfied) >= 0.75
                
                if all_satisfied:
                    assessment = "All assumptions are satisfied. Results are reliable."
                elif most_satisfied:
                    assessment = "Most assumptions are satisfied. Results should be interpreted with minor caution."
                else:
                    assessment = "Multiple assumptions are violated. Results should be interpreted with caution."
                
                ax.text(0.5, -0.1, assessment, transform=ax.transAxes, 
                       ha='center', va='center', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
                
                fig.tight_layout()
                figures['assumption_summary'] = fig_to_svg(fig)
        except Exception as e:
            figures['assumption_summary_error'] = str(e)
            
        # Figure 6: VIF Multicollinearity Plot
        try:
            if 'multicollinearity' in assumptions and 'multicollinearity_vif_values' in assumptions['multicollinearity']:
                vif_data = assumptions['multicollinearity']['multicollinearity_vif_values']
                
                if vif_data:
                    # Create figure
                    fig, ax = plt.subplots(figsize=(9, 5))
                    
                    # Sort VIFs for better visualization
                    sorted_vif = {k: v for k, v in sorted(vif_data.items(), 
                                                        key=lambda item: item[1] or 0, 
                                                        reverse=True)}
                    
                    # Extract predictors and VIF values
                    predictors_list = list(sorted_vif.keys())
                    vif_values = [v if v is not None else 0 for v in sorted_vif.values()]
                    
                    # Set colors based on VIF thresholds
                    colors = [PASTEL_COLORS[2] if v < 5 else PASTEL_COLORS[1] if v < 10 else PASTEL_COLORS[0] 
                             for v in vif_values]
                    
                    # Create bar chart
                    bars = ax.bar(predictors_list, vif_values, color=colors, alpha=0.7)
                    
                    # Add reference lines for VIF thresholds
                    ax.axhline(y=5, color=PASTEL_COLORS[1], linestyle='--', 
                              alpha=0.7, label='Moderate (VIF=5)')
                    ax.axhline(y=10, color=PASTEL_COLORS[0], linestyle='--', 
                              alpha=0.7, label='Severe (VIF=10)')
                    
                    # Add value labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:  # Only add label if VIF is positive
                            ax.text(bar.get_x() + bar.get_width() / 2, height,
                                  f'{height:.2f}', ha='center', va='bottom')
                    
                    # Set labels and title
                    ax.set_ylabel('VIF Value')
                    ax.set_title('Variance Inflation Factors (Multicollinearity)')
                    
                    # Rotate x-tick labels if needed
                    plt.xticks(rotation=45 if len(predictors_list) > 5 else 0, ha='right')
                    
                    # Add legend
                    ax.legend()
                    
                    # Set reasonable y-limit
                    ax.set_ylim(0, max(max(vif_values) * 1.1, 11))
                    
                    fig.tight_layout()
                    figures['vif_plot'] = fig_to_svg(fig)
        except Exception as e:
            figures['vif_plot_error'] = str(e)
            
        # Figure 7: Predictive Performance Metrics
        try:
            if 'per_class' in prediction_stats:
                # Create a grouped bar chart of precision, recall, F1 for each category
                metrics = ['precision', 'recall', 'f1_score']
                categories_list = list(prediction_stats['per_class'].keys())
                
                # Create data arrays
                data = {
                    metric: [prediction_stats['per_class'][cat][metric] 
                           for cat in categories_list]
                    for metric in metrics
                }
                
                # Set up the figure
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Set width of bars
                bar_width = 0.25
                index = np.arange(len(categories_list))
                
                # Create bars
                for i, metric in enumerate(metrics):
                    shift = (i - 1) * bar_width
                    bars = ax.bar(index + shift, data[metric], bar_width,
                                 label=metric.replace('_', ' ').title(),
                                 color=PASTEL_COLORS[i % len(PASTEL_COLORS)])
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2, height,
                               f'{height:.2f}', ha='center', va='bottom')
                
                # Set labels and title
                ax.set_xlabel('Category')
                ax.set_ylabel('Score')
                ax.set_title('Predictive Performance Metrics by Category')
                
                # Set x-ticks
                ax.set_xticks(index)
                ax.set_xticklabels(categories_list)
                
                # Add legend
                ax.legend()
                
                # Set y-limits
                ax.set_ylim(0, 1.1)
                
                # Add overall accuracy on the plot
                accuracy = prediction_stats['accuracy']
                ax.text(0.5, 0.02, f'Overall Accuracy: {accuracy:.3f}',
                       transform=ax.transAxes, ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                fig.tight_layout()
                figures['performance_metrics'] = fig_to_svg(fig)
        except Exception as e:
            figures['performance_metrics_error'] = str(e)
            
        # Figure 8: Effect Plot for Categorical Predictors (if any)
        try:
            # Identify categorical predictors
            categorical_predictors = []
            for pred in predictors:
                if (pd.api.types.is_categorical_dtype(data_copy[pred]) or 
                    pd.api.types.is_object_dtype(data_copy[pred]) or
                    len(data_copy[pred].unique()) < 10):  # Assumption for small number of categories
                    categorical_predictors.append(pred)
            
            if categorical_predictors:
                # Create stacked bar charts for each categorical predictor
                for pred in categorical_predictors[:3]:  # Limit to first 3 to avoid too many plots
                    # Create cross-tabulation
                    cross_tab = pd.crosstab(data_copy[pred], data_copy[outcome], normalize='index')
                    
                    # Create figure
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Create stacked bar chart
                    cross_tab.plot(kind='bar', stacked=True, ax=ax, color=PASTEL_COLORS)
                    
                    # Set labels and title
                    ax.set_xlabel(pred)
                    ax.set_ylabel('Proportion')
                    ax.set_title(f'Distribution of {outcome} by {pred}')
                    
                    # Add legend with a better location
                    ax.legend(title=outcome, bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                    # Rotate x-tick labels if needed
                    plt.xticks(rotation=45 if len(cross_tab) > 5 else 0, ha='right')
                    
                    fig.tight_layout()
                    figures[f'effect_plot_{pred}'] = fig_to_svg(fig)
        except Exception as e:
            figures['effect_plot_error'] = str(e)
            
        # Figure 9: Effect Plot for Continuous Predictors (if any)
        try:
            # Identify continuous predictors
            continuous_predictors = []
            for pred in predictors:
                if (pd.api.types.is_numeric_dtype(data_copy[pred]) and 
                    not pd.api.types.is_bool_dtype(data_copy[pred]) and
                    len(data_copy[pred].unique()) >= 10):
                    continuous_predictors.append(pred)
            
            if continuous_predictors:
                # Create box plots for each continuous predictor
                for pred in continuous_predictors[:3]:  # Limit to first 3 to avoid too many plots
                    # Create figure
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Create box plot
                    sns.boxplot(x=outcome, y=pred, data=data_copy, ax=ax,
                                palette=PASTEL_COLORS)
                    
                    # Add individual data points
                    sns.stripplot(x=outcome, y=pred, data=data_copy, ax=ax,
                                color='#666666', alpha=0.3, jitter=True, size=3)
                    
                    # Set labels and title
                    ax.set_xlabel(outcome)
                    ax.set_ylabel(pred)
                    ax.set_title(f'Distribution of {pred} by {outcome}')
                    
                    # Rotate x-tick labels if needed
                    plt.xticks(rotation=45 if len(data_copy[outcome].unique()) > 5 else 0, 
                              ha='right')
                    
                    fig.tight_layout()
                    figures[f'boxplot_{pred}'] = fig_to_svg(fig)
        except Exception as e:
            figures['boxplot_error'] = str(e)
        
        return {
            'test': 'Multinomial Logistic Regression',
            'coefficients': coefficients,
            'model_fit': model_fit,
            'prediction_stats': prediction_stats,
            'formula': formula,
            'summary': str(results.summary()),
            'assumptions': assumptions,
            'interpretation': interpretation,
            'figures': figures
        }
    except Exception as e:
        # Use the original traceback module without being affected by any local variable
        return {
            'test': 'Multinomial Logistic Regression',
            'error': str(e),
            'traceback': traceback.format_exc()
        }