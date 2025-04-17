import traceback
import pandas as pd
from scipy import stats
import numpy as np
from typing import Dict, List, Any
from statsmodels.miscmodels.ordinal_model import OrderedModel
from data.assumptions.format import AssumptionTestKeys
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import base64
import io
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from data.selection.detailed_tests.formatting import fig_to_svg, PASTEL_COLORS, PASTEL_CMAP

def is_test_passed(result) -> bool:
    """Helper function to check if a test result is passed, handling different result formats."""
    if isinstance(result, str):
        return result == 'passed'
    elif hasattr(result, 'value'):
        return result.value == 'passed'
    return False

def check_proportional_odds_manually(data: pd.DataFrame, outcome: str, covariates: List[str], results) -> Dict[str, Any]:
    """
    A custom function to check the proportional odds assumption without relying on external tests.
    
    The proportional odds assumption states that the effect of each predictor is consistent
    across different thresholds of the outcome variable.
    
    Returns:
        Dictionary with test results and interpretation.
    """
    try:
        # Convert outcome to numeric codes for analysis
        outcome_codes = pd.Series(data[outcome].cat.codes, index=data.index)
        outcome_categories = data[outcome].cat.categories
        n_categories = len(outcome_categories)
        
        # Store coefficients for each binary cutpoint model
        coefficients_by_cutpoint = {}
        violated_predictors = []
        
        # Significance threshold for testing coefficient differences (default: 0.05)
        sig_threshold = 0.05
        
        # For each possible cutpoint, fit a binary logistic regression
        for i in range(n_categories - 1):
            cutpoint_name = f"{outcome_categories[i]}"
            
            # Create binary outcome for this cutpoint
            binary_outcome = (outcome_codes > i).astype(int)
            
            # Create DataFrame for this binary model
            model_data = data[covariates].copy()
            model_data['binary_outcome'] = binary_outcome
            
            # Fit binary logistic regression
            try:
                formula = f"binary_outcome ~ {' + '.join(covariates)}"
                binary_model = smf.logit(formula=formula, data=model_data).fit(disp=False)
                
                # Store coefficients
                cutpoint_coefficients = {}
                for var in covariates:
                    try:
                        coef = binary_model.params.get(var, 0)
                        cutpoint_coefficients[var] = float(coef)
                    except Exception as e:
                        # Skip variables that cause problems
                        cutpoint_coefficients[var] = np.nan
                
                coefficients_by_cutpoint[cutpoint_name] = cutpoint_coefficients
            
            except Exception as e:
                print(f"Error fitting binary model for cutpoint {cutpoint_name}: {str(e)}")
                # Add placeholder to maintain cutpoint structure
                coefficients_by_cutpoint[cutpoint_name] = {var: np.nan for var in covariates}
        
        # Check if the coefficients across cutpoints are similar for each predictor
        coefficient_variances = {}
        significant_variations = False
        
        for var in covariates:
            # Get coefficients across all cutpoints for this variable
            var_coefficients = [
                coefficients_by_cutpoint[cutpoint][var] 
                for cutpoint in coefficients_by_cutpoint.keys()
                if not np.isnan(coefficients_by_cutpoint[cutpoint][var])
            ]
            
            if len(var_coefficients) > 1:
                # Calculate coefficient variation
                coef_std = np.std(var_coefficients)
                coef_range = np.max(var_coefficients) - np.min(var_coefficients)
                coef_mean = np.mean(var_coefficients)
                
                # Store variance statistics
                coefficient_variances[var] = {
                    'std': float(coef_std),
                    'range': float(coef_range),
                    'mean': float(coef_mean),
                    'variation_coef': float(coef_std / abs(coef_mean)) if coef_mean != 0 else float('inf')
                }
                
                # Check if variation is significant (using coefficient of variation)
                if coefficient_variances[var]['variation_coef'] > 0.5:  # Arbitrary threshold
                    violated_predictors.append(var)
                    significant_variations = True
        
        # Determine the overall result
        if significant_variations:
            result = {
                'result': 'warning',
                'details': {
                    'coefficients_by_cutpoint': {
                        str(k): {str(var): coef for var, coef in v.items()}
                        for k, v in coefficients_by_cutpoint.items()
                    },
                    'coefficient_variances': coefficient_variances,
                    'violated_predictors': violated_predictors,
                    'interpretation': 'The proportional odds assumption may be violated. Consider using multinomial logistic regression.'
                }
            }
        else:
            result = {
                'result': 'passed',
                'details': {
                    'coefficients_by_cutpoint': {
                        str(k): {str(var): coef for var, coef in v.items()}
                        for k, v in coefficients_by_cutpoint.items()
                    },
                    'coefficient_variances': coefficient_variances,
                    'interpretation': 'The proportional odds assumption appears to be satisfied.'
                }
            }
            
        return result
        
    except Exception as e:
        error_message = f"Could not test proportional odds assumption: {str(e)}"
        print(error_message)
        return {
            'result': 'warning',
            'details': {
                'error': error_message,
                'interpretation': 'Could not test proportional odds assumption. Consider fitting separate binary logistic regressions.'
            }
        }

def ordinal_regression(data: pd.DataFrame, outcome: str, covariates: List[str], alpha: float) -> Dict[str, Any]:
    """Performs Ordinal Regression with comprehensive statistics, visualizations, and assumption checks."""
    try:
        # Set matplotlib to use a non-interactive backend
        matplotlib.use('Agg')
        figures = {}
        
        # Initialize assumptions dict
        assumptions = {}
        
        # 1. Check if outcome is categorical and ordered
        try:
            data[outcome] = pd.Categorical(data[outcome], ordered=True)
            
            # Check if outcome has at least 3 categories (for ordinal regression)
            outcome_categories = data[outcome].cat.categories
            
            outcome_check_result = {
                'result': 'passed' if len(outcome_categories) >= 3 else 'warning',
                'details': {
                    'num_categories': len(outcome_categories),
                    'categories': [str(cat) for cat in outcome_categories],
                    'interpretation': ('The outcome variable has 3 or more ordered categories'
                                        if len(outcome_categories) >= 3 
                                        else 'The outcome variable has fewer than 3 categories; consider using binary logistic regression instead')
                }
            }
            assumptions['outcome_check'] = outcome_check_result
            
        except:
            raise ValueError(f"Could not convert {outcome} to ordered categorical. Ensure it has appropriate levels.")
        
        # 2. Check for multicollinearity among covariates
        if len(covariates) > 1:
            # Use AssumptionTestKeys instead of direct import
            multicollinearity_test = AssumptionTestKeys.MULTICOLLINEARITY.value["function"]()
            multicollinearity_result = multicollinearity_test.run_test(df=data, covariates=covariates)
            assumptions['multicollinearity'] = multicollinearity_result
        
        # 3. Check sample size
        # Rule of thumb: at least 10 observations per category per predictor
        n_categories = len(data[outcome].cat.categories)
        n_predictors = len(covariates)
        min_recommended = 10 * n_categories * n_predictors
        
        # Use AssumptionTestKeys instead of direct import
        sample_size_test = AssumptionTestKeys.SAMPLE_SIZE.value["function"]()
        sample_size_result = sample_size_test.run_test(data=data[outcome], min_recommended=min_recommended)
        assumptions['sample_size'] = sample_size_result
        
        # 4. Check for cell sparsity (contingency tables for each predictor)
        cell_sparsity_results = {}
        for predictor in covariates:
            if data[predictor].dtype.name in ['category', 'object']:
                # For categorical predictors, check contingency table
                contingency = pd.crosstab(data[outcome], data[predictor])
                min_cell_count = contingency.values.min()
                sparse_cells = (contingency < 5).sum().sum()
                
                cell_sparsity_results[predictor] = {
                    'result': 'passed' if min_cell_count >= 5 and sparse_cells == 0 else 'warning',
                    'details': {
                        'min_cell_count': int(min_cell_count),
                        'sparse_cells': int(sparse_cells),
                        'interpretation': ('All cells have adequate counts'
                                            if min_cell_count >= 5 and sparse_cells == 0
                                            else 'Some cells have counts < 5, which may affect model stability')
                    }
                }
        
        if cell_sparsity_results:
            assumptions['cell_sparsity'] = cell_sparsity_results

        # Create distribution plot of outcome variable
        plt.figure(figsize=(10, 6))
        outcome_counts = data[outcome].value_counts().sort_index()
        plt.bar(outcome_counts.index.astype(str), outcome_counts.values, alpha=0.7, color=PASTEL_COLORS[0])
        plt.xlabel(outcome)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {outcome}')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Save figure
        figures['outcome_distribution'] = fig_to_svg(plt.gcf())
            
        # Construct formula
        formula = f"{outcome} ~ {' + '.join(covariates)}"
        
        # Fit the model - default to logit link function
        model = OrderedModel.from_formula(formula, data, distr='logit')
        results = model.fit(method='bfgs', disp=False)
        
        # Extract model parameters
        aic = float(results.aic)
        bic = float(results.bic)
        log_likelihood = float(results.llf)
        
        # Extract coefficients (skip threshold parameters)
        coefficients = {}
        for term in results.params.index:
            if term.lower().startswith('threshold'):
                continue
            coefficients[term] = {
                'estimate': float(results.params[term]),
                'std_error': float(results.bse[term]) if term in results.bse else None,
                'z_value': float(results.tvalues[term]) if term in results.tvalues else None,
                'p_value': float(results.pvalues[term]) if term in results.pvalues else None,
                'significant': float(results.pvalues[term]) < alpha if term in results.pvalues else None,
                'odds_ratio': float(np.exp(results.params[term])),
                'ci_lower': float(np.exp(results.conf_int().loc[term, 0])) if term in results.conf_int().index else None,
                'ci_upper': float(np.exp(results.conf_int().loc[term, 1])) if term in results.conf_int().index else None
            }
            
        # Extract thresholds/intercepts (look for parameter names starting with 'threshold')
        thresholds = {}
        for term in results.params.index:
            if term.lower().startswith('threshold'):
                thresholds[term] = {
                    'estimate': float(results.params[term]),
                    'std_error': float(results.bse[term]) if term in results.bse else None,
                    'z_value': float(results.tvalues[term]) if term in results.tvalues else None,
                    'p_value': float(results.pvalues[term]) if term in results.pvalues else None
                }
        
        # Create odds ratio plot for coefficients
        plt.figure(figsize=(10, 6))
        terms = list(coefficients.keys())
        odds_ratios = [coefficients[term]['odds_ratio'] for term in terms]
        ci_lower = [coefficients[term]['ci_lower'] for term in terms]
        ci_upper = [coefficients[term]['ci_upper'] for term in terms]
        
        # Sort by odds ratio for better visualization
        sorted_idx = np.argsort(odds_ratios)
        sorted_terms = [terms[i] for i in sorted_idx]
        sorted_odds = [odds_ratios[i] for i in sorted_idx]
        sorted_ci_lower = [ci_lower[i] for i in sorted_idx]
        sorted_ci_upper = [ci_upper[i] for i in sorted_idx]
        
        # Calculate error bar widths
        err_min = [o - l for o, l in zip(sorted_odds, sorted_ci_lower)]
        err_max = [u - o for o, u in zip(sorted_odds, sorted_ci_upper)]
        
        plt.errorbar(sorted_odds, np.arange(len(sorted_terms)), xerr=[err_min, err_max], fmt='o', capsize=5, color=PASTEL_COLORS[0])
        plt.axvline(x=1, color='red', linestyle='--')
        plt.yticks(np.arange(len(sorted_terms)), sorted_terms)
        plt.xscale('log')
        plt.xlabel('Odds Ratio (log scale)')
        plt.title('Odds Ratios with 95% Confidence Intervals')
        plt.grid(True, alpha=0.3)
        
        # Save figure
        figures['odds_ratio_plot'] = fig_to_svg(plt.gcf())
        
        # Calculate model fit statistics compared to null model
        try:
            # Fit null model (intercepts only)
            null_formula = f"{outcome} ~ 1"
            null_model = OrderedModel.from_formula(null_formula, data, distr='logit')
            null_results = null_model.fit(method='bfgs', disp=False)
            
            # Calculate likelihood ratio test
            lr_stat = 2 * (results.llf - null_results.llf)
            df = len(covariates)
            lr_pval = stats.chi2.sf(lr_stat, df)
            
            # Calculate McFadden's pseudo R-squared
            mcfadden_r2 = 1 - (results.llf / null_results.llf)
            
            # Calculate Nagelkerke's R-squared (pseudo R-squared)
            n = len(data)
            cox_snell_r2 = 1 - np.exp(2 * (null_results.llf - results.llf) / n)
            nagelkerke_r2 = cox_snell_r2 / (1 - np.exp(2 * null_results.llf / n))
            
            model_fit = {
                'log_likelihood': float(results.llf),
                'null_log_likelihood': float(null_results.llf),
                'lr_statistic': float(lr_stat),
                'lr_df': int(df),
                'lr_p_value': float(lr_pval),
                'mcfadden_r2': float(mcfadden_r2),
                'cox_snell_r2': float(cox_snell_r2),
                'nagelkerke_r2': float(nagelkerke_r2),
                'aic': float(results.aic),
                'bic': float(results.bic),
                'significant': lr_pval < alpha
            }
        except:
            model_fit = {
                'log_likelihood': float(results.llf),
                'aic': float(results.aic),
                'bic': float(results.bic),
                'error': 'Could not fit null model for comparison'
            }
            
        # Calculate predicted probabilities for each outcome category
        try:
            predicted_probs = results.predict(which='prob')
            
            # Extract probabilities from DataFrame - ensure we get an array per row
            if isinstance(predicted_probs, pd.DataFrame):
                predicted_probs = [row.values for _, row in predicted_probs.iterrows()]
            
            # For easier interpretation, calculate average predicted probability for each outcome category
            avg_predicted_probs = {}
            for category in data[outcome].cat.categories:
                category_idx = list(data[outcome].cat.categories).index(category)
                avg_prob = np.mean([p[category_idx] for p in predicted_probs])
                avg_predicted_probs[str(category)] = float(avg_prob)
                
            # Create bar plot of average predicted probabilities
            plt.figure(figsize=(10, 6))
            categories = list(avg_predicted_probs.keys())
            probabilities = list(avg_predicted_probs.values())
            plt.bar(categories, probabilities, alpha=0.7, color=PASTEL_COLORS[1])
            plt.xlabel(outcome)
            plt.ylabel('Average Predicted Probability')
            plt.title(f'Average Predicted Probabilities for Each {outcome} Category')
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            plt.grid(axis='y', alpha=0.3)
            
            # Save figure
            figures['predicted_probabilities'] = fig_to_svg(plt.gcf())
                
            # Calculate classification accuracy
            predicted_categories = []
            for probs in predicted_probs:
                pred_idx = np.argmax(probs)
                predicted_categories.append(data[outcome].cat.categories[pred_idx])
                
            accuracy = np.mean(predicted_categories == data[outcome])
            
            # Create confusion matrix
            cm = confusion_matrix(data[outcome], predicted_categories)
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap=PASTEL_CMAP,
                        xticklabels=data[outcome].cat.categories,
                        yticklabels=data[outcome].cat.categories)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Normalized Confusion Matrix')
            
            # Save figure
            figures['confusion_matrix'] = fig_to_svg(plt.gcf())
            
            # Calculate per-class metrics
            class_metrics = {}
            cm_raw = confusion_matrix(data[outcome], predicted_categories)
            
            for i, category in enumerate(data[outcome].cat.categories):
                tp = cm_raw[i, i]
                fp = cm_raw[:, i].sum() - tp
                fn = cm_raw[i, :].sum() - tp
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                class_metrics[str(category)] = {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1)
                }
                
            # Calculate residuals for independence check
            # Using response residuals (observed - predicted category index)
            observed_category_idx = np.array([list(data[outcome].cat.categories).index(cat) for cat in data[outcome]])
            predicted_category_idx = np.array([list(data[outcome].cat.categories).index(cat) for cat in predicted_categories])
            residuals = observed_category_idx - predicted_category_idx
            
            # Calculate cross-validation performance
            try:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                X = data[covariates].copy()
                y = data[outcome].copy()
                cv_accuracies = []
                
                for train_idx, test_idx in cv.split(X, y):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    train_df = pd.concat([X_train, y_train], axis=1)
                    
                    cv_model = OrderedModel.from_formula(formula, train_df, distr='logit')
                    cv_results = cv_model.fit(method='bfgs', disp=False)
                    
                    test_df = pd.concat([X_test, y_test], axis=1)
                    cv_pred_probs = cv_results.model.predict(cv_results.params, exog=test_df, which='prob')
                    
                    # Extract probabilities from DataFrame - ensure we get an array per row
                    if isinstance(cv_pred_probs, pd.DataFrame):
                        cv_pred_probs = [row.values for _, row in cv_pred_probs.iterrows()]
                    
                    cv_pred_categories = []
                    for probs in cv_pred_probs:
                        pred_idx = np.argmax(probs)
                        cv_pred_categories.append(y.cat.categories[pred_idx])
                    
                    cv_accuracy = np.mean(cv_pred_categories == y_test.values)
                    cv_accuracies.append(cv_accuracy)
                
                cv_results_summary = {
                    'cv_accuracy_mean': float(np.mean(cv_accuracies)),
                    'cv_accuracy_std': float(np.std(cv_accuracies))
                }
            except Exception as e:
                cv_results_summary = {'error': f'Could not perform cross-validation: {str(e)}'}
            
            prediction_stats = {
                'avg_predicted_probabilities': avg_predicted_probs,
                'accuracy': float(accuracy),
                'class_metrics': class_metrics,
                'cross_validation': cv_results_summary
            }
            
            # Create stacked bar chart showing predicted vs actual distribution
            predicted_counts = pd.Series(predicted_categories).value_counts().sort_index()
            actual_counts = data[outcome].value_counts().sort_index()
            
            comparison_df = pd.DataFrame({
                'Predicted': predicted_counts,
                'Actual': actual_counts
            }, index=sorted(set(predicted_counts.index) | set(actual_counts.index)))
            comparison_df.fillna(0, inplace=True)
            
            # Convert index to strings to avoid categorical plotting issues
            comparison_df.index = comparison_df.index.map(str)
            
            plt.figure(figsize=(10, 6))
            comparison_df.plot(kind='bar', alpha=0.7, ax=plt.gca(), color=[PASTEL_COLORS[0], PASTEL_COLORS[1]])
            plt.xlabel(outcome)
            plt.ylabel('Count')
            plt.title(f'Predicted vs Actual Distribution of {outcome}')
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            plt.legend()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='svg')
            plt.close()
            figures['predicted_vs_actual'] = fig_to_svg(plt.gcf())
            
        except Exception as e:
            prediction_stats = {
                'error': f'Could not calculate predicted probabilities: {str(e)}'
            }
            
        # 5. Test proportional odds assumption
        try:
            # Create a custom proportional odds test instead of using the potentially problematic imported one
            assumptions['proportional_odds'] = check_proportional_odds_manually(data, outcome, covariates, results)
            
            # Get the result from our custom check
            proportional_odds_result = assumptions['proportional_odds']
            
            # If we have coefficients by cutpoint, create the visualization
            if 'coefficients_by_cutpoint' in proportional_odds_result.get('details', {}):
                coefs_by_cutpoint = proportional_odds_result['details']['coefficients_by_cutpoint']
                
                plot_data = []
                for cutpoint, coefs in coefs_by_cutpoint.items():
                    for var, coef in coefs.items():
                        plot_data.append({
                            'Cutpoint': cutpoint,
                            'Variable': var,
                            'Coefficient': coef
                        })
                
                if plot_data:
                    plot_df = pd.DataFrame(plot_data)
                    
                    plt.figure(figsize=(12, 8))
                    
                    for i, var in enumerate(plot_df['Variable'].unique()):
                        color_idx = i % len(PASTEL_COLORS)
                        var_data = plot_df[plot_df['Variable'] == var]
                        plt.plot(var_data['Cutpoint'], var_data['Coefficient'], 'o-', 
                                label=var, color=PASTEL_COLORS[color_idx])
                    
                    plt.xlabel('Cutpoint')
                    plt.ylabel('Coefficient')
                    plt.title('Coefficients Across Cutpoints\n(Lines should be approximately parallel for proportional odds)')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='svg')
                    plt.close()
                    figures['proportional_odds'] = fig_to_svg(plt.gcf())
            else:
                print("No coefficients_by_cutpoint found in proportional_odds_result")
        except Exception as e:
            # More informative error message
            error_message = f"Could not test proportional odds assumption: {str(e)}"
            print(error_message)
            assumptions['proportional_odds'] = {
                'result': 'warning',
                'details': {
                    'error': error_message,
                    'interpretation': 'Could not test proportional odds assumption. Consider fitting separate binary logistic regressions.'
                }
            }
            
        # 6. Check for independence (optional but helpful)
        try:
            # Use AssumptionTestKeys for independence check
            independence_test = AssumptionTestKeys.INDEPENDENCE.value["function"]()
            # If we have residuals, we can use them to check independence
            if 'residuals' in locals() or 'residuals' in globals():
                
                independence_result = independence_test.run_test(data=residuals)
                assumptions['independence'] = independence_result
        except:
            # Skip if not applicable or fails
            pass
            
        # 7. Check for outliers if needed
        try:
            # Use AssumptionTestKeys for outlier detection
            outlier_test = AssumptionTestKeys.OUTLIERS.value["function"]()
            # Check for outliers in continuous covariates
            for cov in covariates:
                if pd.api.types.is_numeric_dtype(data[cov]):
                    outlier_result = outlier_test.run_test(data=data[cov])
                    assumptions[f'outliers_{cov}'] = outlier_result
        except:
            # Skip if not applicable or fails
            pass
            
        # Create threshold plot
        plt.figure(figsize=(10, 6))
        try:
            # Identify threshold parameters and sort them by numeric index
            threshold_params = [param for param in results.params.index if param.lower().startswith('threshold')]
            threshold_params = sorted(threshold_params, key=lambda x: int(''.join(filter(str.isdigit, x))) if ''.join(filter(str.isdigit, x)) else 0)
            threshold_values = [float(results.params[param]) for param in threshold_params]
            
            if len(outcome_categories) >= len(threshold_values) + 1:
                # Ensure categories are converted to strings for plotting
                threshold_labels = [f"{str(outcome_categories[i])} | {str(outcome_categories[i+1])}" for i in range(len(threshold_values))]
            else:
                threshold_labels = threshold_params
            
            plt.barh(threshold_labels, threshold_values, alpha=0.7, color=PASTEL_COLORS[0])
            plt.xlabel('Threshold Value')
            plt.title('Threshold Parameters')
            plt.grid(axis='x', alpha=0.3)
            
            # Ensure proper figure saving
            figures['thresholds'] = fig_to_svg(plt.gcf())
            plt.close()
        except Exception as e:
            print(f"Could not create threshold plot: {str(e)}")
            plt.close()
        
        # Create cumulative probability curves plot for a reference individual
        try:
            reference_data = {}
            for cov in covariates:
                if pd.api.types.is_numeric_dtype(data[cov]):
                    reference_data[cov] = [data[cov].mean()]
                else:
                    reference_data[cov] = [data[cov].mode()[0]]
            
            reference_df = pd.DataFrame(reference_data)
            pred_results = results.predict(exog=reference_df, which='prob')
            
            # Debug the shape of prediction results
            print(f"Debug - Prediction result type: {type(pred_results)}, shape/length: {np.shape(pred_results) if hasattr(np, 'shape') else len(pred_results)}")
            
            # Extract probabilities properly from the DataFrame (it has shape 1,4)
            # Convert the first row to a numpy array
            reference_prediction = pred_results.iloc[0].values
            
            print(f"Debug - Reference prediction type: {type(reference_prediction)}, shape/length: {np.shape(reference_prediction)}")
            
            if len(reference_prediction) == len(outcome_categories):
                cumulative_probs = np.cumsum(reference_prediction)
                
                plt.figure(figsize=(10, 6))
                plt.plot(range(len(outcome_categories)), cumulative_probs, 'o-', 
                        linewidth=2, color=PASTEL_COLORS[0])
                plt.axhline(y=0.5, color='red', linestyle='--', label='50% probability')
                plt.xticks(range(len(outcome_categories)), [str(cat) for cat in outcome_categories])
                plt.xlabel(outcome)
                plt.ylabel('Cumulative Probability')
                plt.title('Cumulative Probability Curve for Reference Individual')
                plt.ylim(0, 1)
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='svg')
                plt.close()
                figures['cumulative_probability'] = fig_to_svg(plt.gcf())
            else:
                print(f"Shape mismatch: outcome_categories length = {len(outcome_categories)}, prediction length = {len(reference_prediction)}")
                plt.close()
        except Exception as e:
            print(f"Could not create cumulative probability plot: {str(e)}")
            plt.close()
        
        # Create interpretation
        significant_covariates = [term for term, coef in coefficients.items() if coef.get('significant', False)]
        
        interpretation = f"Ordinal Regression with {outcome} as outcome and covariates ({', '.join(covariates)}).\n\n"
        
        # Interpret model fit
        if 'lr_p_value' in model_fit:
            interpretation += f"The model is {'statistically significant' if model_fit['significant'] else 'not statistically significant'} "
            interpretation += f"compared to the null model (χ²({model_fit['lr_df']}) = {model_fit['lr_statistic']:.3f}, p = {model_fit['lr_p_value']:.5f}).\n"
            
            if 'mcfadden_r2' in model_fit:
                interpretation += f"McFadden's pseudo R² = {model_fit['mcfadden_r2']:.3f}, "
            if 'nagelkerke_r2' in model_fit:
                interpretation += f"Nagelkerke's R² = {model_fit['nagelkerke_r2']:.3f}, "
            interpretation += f"suggesting that the model explains a "
            if model_fit.get('nagelkerke_r2', 0) > 0.5:
                interpretation += "substantial "
            elif model_fit.get('nagelkerke_r2', 0) > 0.3:
                interpretation += "moderate "
            else:
                interpretation += "modest "
            interpretation += f"proportion of the variation in {outcome}.\n\n"
        
        # Interpret coefficients
        if significant_covariates:
            interpretation += "Significant covariates:\n"
            for term in significant_covariates:
                coef = coefficients[term]
                interpretation += f"- {term}: β = {coef['estimate']:.3f}, OR = {coef['odds_ratio']:.3f} "
                interpretation += f"(95% CI: {coef['ci_lower']:.3f}-{coef['ci_upper']:.3f}), p = {coef['p_value']:.5f}\n"
                if coef['odds_ratio'] > 1:
                    interpretation += f"  For each unit increase in {term}, the odds of being in a higher category of {outcome} increase by {(coef['odds_ratio']-1)*100:.1f}%.\n"
                else:
                    interpretation += f"  For each unit increase in {term}, the odds of being in a higher category of {outcome} decrease by {(1-coef['odds_ratio'])*100:.1f}%.\n"
        else:
            interpretation += "No significant covariates were found.\n"
            
        # Interpret thresholds using sorted threshold parameters
        interpretation += "\nThreshold parameters:\n"
        sorted_thresholds = sorted(thresholds.items(), key=lambda x: int(''.join(filter(str.isdigit, x[0]))) if ''.join(filter(str.isdigit, x[0])) else 0)
        for i, (key, thresh) in enumerate(sorted_thresholds):
            if i < len(outcome_categories) - 1:
                interpretation += f"- Threshold between {outcome_categories[i]} and {outcome_categories[i+1]}: {thresh['estimate']:.3f}\n"
            
        # Add classification accuracy if available
        if 'accuracy' in prediction_stats:
            interpretation += f"\nClassification Performance:\n"
            interpretation += f"- The model correctly classifies {prediction_stats['accuracy']*100:.1f}% of observations.\n"
            if 'class_metrics' in prediction_stats:
                interpretation += "- Per-class performance:\n"
                for cat, metrics in prediction_stats['class_metrics'].items():
                    interpretation += f"  {cat}: Precision = {metrics['precision']:.2f}, Recall = {metrics['recall']:.2f}, F1 = {metrics['f1']:.2f}\n"
            if 'cross_validation' in prediction_stats and 'cv_accuracy_mean' in prediction_stats['cross_validation']:
                cv = prediction_stats['cross_validation']
                interpretation += f"- 5-fold cross-validation accuracy: {cv['cv_accuracy_mean']*100:.1f}% (±{cv['cv_accuracy_std']*100:.1f}%)\n"
                if cv['cv_accuracy_mean'] < prediction_stats['accuracy'] - 0.1:
                    interpretation += "  The model shows signs of overfitting as cross-validation accuracy is substantially lower than training accuracy.\n"
                else:
                    interpretation += "  The model generalizes well as cross-validation and training accuracies are similar.\n"
            
        # Add assumption check results to interpretation
        interpretation += "\nAssumption checks:\n"
        
        # Outcome check
        outcome_check_satisfied = outcome_check_result.get('result') == 'passed'
        if outcome_check_satisfied:
            interpretation += f"- Outcome variable: {outcome} has {len(outcome_categories)} ordered categories, which is appropriate for ordinal regression.\n"
        else:
            interpretation += f"- Outcome variable: {outcome} has only {len(outcome_categories)} categories. "
            interpretation += "Ordinal regression typically works best with 3 or more ordered categories.\n"
            
        # Multicollinearity
        if 'multicollinearity' in assumptions:
            multicollinearity_satisfied = is_test_passed(assumptions['multicollinearity'].get('result', {}))
            if multicollinearity_satisfied:
                interpretation += "- Multicollinearity: No significant multicollinearity detected among predictors.\n"
            else:
                interpretation += "- Multicollinearity: Potential multicollinearity detected among predictors. "
                interpretation += "This may affect the stability and interpretation of coefficients.\n"
                if 'vif_values' in assumptions['multicollinearity']:
                    high_vif = {k: v for k, v in assumptions['multicollinearity']['vif_values'].items() if v > 5}
                    if high_vif:
                        interpretation += f"  Variables with high VIF: {', '.join([f'{k} ({v:.1f})' for k, v in high_vif.items()])}.\n"
        
        # Sample size
        sample_size_satisfied = is_test_passed(assumptions['sample_size'].get('result', {}))
        if sample_size_satisfied:
            interpretation += "- Sample size: The sample size is adequate for the model complexity.\n"
        else:
            interpretation += "- Sample size: The sample size may be insufficient for the model complexity. "
            interpretation += f"With {n_categories} outcome categories and {n_predictors} predictors, "
            interpretation += f"a minimum of {min_recommended} observations is recommended, but only {len(data)} are available.\n"
            
        # Cell sparsity
        if cell_sparsity_results:
            sparse_predictors = [pred for pred, result in cell_sparsity_results.items() 
                                 if not is_test_passed(result.get('result'))]
            if not sparse_predictors:
                interpretation += "- Cell sparsity: All categorical predictors have adequate cell counts.\n"
            else:
                interpretation += f"- Cell sparsity: Sparse cells detected for predictors: {', '.join(sparse_predictors)}. "
                interpretation += "This may affect model stability and convergence.\n"
                
        # Proportional odds
        if 'proportional_odds' in assumptions:
            proportional_odds_satisfied = is_test_passed(assumptions['proportional_odds'].get('result', {}))
                
            if proportional_odds_satisfied:
                interpretation += "- Proportional odds: The assumption appears to be satisfied. "
                interpretation += "This means the effect of each predictor is consistent across different thresholds of the outcome.\n"
            else:
                interpretation += "- Proportional odds: The assumption may be violated. "
                interpretation += "This means that the effect of predictors may not be consistent across different thresholds of the outcome variable. "
                interpretation += "Consider using multinomial logistic regression instead.\n"
                if 'details' in assumptions['proportional_odds'] and 'violated_predictors' in assumptions['proportional_odds']['details']:
                    violated_predictors = assumptions['proportional_odds']['details']['violated_predictors']
                    if violated_predictors:
                        interpretation += f"  Predictors that may violate the assumption: {', '.join(violated_predictors)}.\n"
                        
        # Add guidance for interpretation
        interpretation += "\nGuidance for interpretation:\n"
        
        if model_fit.get('nagelkerke_r2', 0) < 0.2:
            interpretation += "- The model explains a relatively small proportion of the variation in the outcome. Consider including additional relevant predictors.\n"
            
        has_violations = False
        if (('multicollinearity' in assumptions and not is_test_passed(assumptions['multicollinearity'].get('result', {}))) or
           ('proportional_odds' in assumptions and not is_test_passed(assumptions['proportional_odds'].get('result', {}))) or
           ('sample_size' in assumptions and not is_test_passed(assumptions['sample_size'].get('result', {})))):
            has_violations = True
            
        if has_violations:
            interpretation += "- Due to assumption violations, consider these alternatives:\n"
            if 'proportional_odds' in assumptions and not is_test_passed(assumptions['proportional_odds'].get('result', {})):
                interpretation += "  1. Use multinomial logistic regression which doesn't require the proportional odds assumption\n"
                interpretation += "  2. Fit separate binary logistic regression models for each threshold\n"
            if 'multicollinearity' in assumptions and not is_test_passed(assumptions['multicollinearity'].get('result', {})):
                interpretation += "  3. Remove or combine highly correlated predictors\n"
            if 'sample_size' in assumptions and not is_test_passed(assumptions['sample_size'].get('result', {})):
                interpretation += "  4. Simplify the model by reducing the number of predictors\n"
                interpretation += "  5. Collapse some outcome categories if appropriate\n"
                
        interpretation += "\nVisualization descriptions:\n"
        interpretation += "- Odds Ratio Plot: Shows the effect size of each predictor with confidence intervals\n"
        interpretation += "- Predicted Probabilities: Shows the average predicted probability for each outcome category\n"
        interpretation += "- Confusion Matrix: Displays model classification performance for each category\n"
        interpretation += "- Threshold Parameters: Shows the cutpoints between consecutive outcome categories\n"
        interpretation += "- Cumulative Probability Curve: Shows how probabilities accumulate across categories for a reference case\n"
        if 'proportional_odds' in figures:
            interpretation += "- Proportional Odds Plot: Lines should be roughly parallel if the proportional odds assumption holds\n"
            
        confidence_level = "high"
        if has_violations:
            confidence_level = "moderate"
        if 'sample_size' in assumptions and not is_test_passed(assumptions['sample_size'].get('result', {})):
            confidence_level = "low" if confidence_level == "moderate" else "moderate"
            
        interpretation += f"\nOverall confidence in model results: {confidence_level}\n"
        
        return {
            'test': 'Ordinal Regression',
            'coefficients': coefficients,
            'thresholds': thresholds,
            'model_fit': model_fit,
            'prediction_stats': prediction_stats,
            'assumptions': assumptions,
            'outcome_categories': [str(cat) for cat in data[outcome].cat.categories],
            'formula': formula,
            'summary': str(results.summary()),
            'interpretation': interpretation,
            'figures': figures
        }
    except Exception as e:
        return {
            'test': 'Ordinal Regression',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
