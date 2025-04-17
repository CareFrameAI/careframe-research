import traceback
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from scipy import stats
from sklearn.metrics import mean_squared_error, roc_curve, confusion_matrix
import statsmodels.formula.api as smf
import statsmodels.api as sm
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib
import base64
import io
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
from data.selection.detailed_tests.formatting import fig_to_svg, PASTEL_COLORS, PASTEL_CMAP
from data.assumptions.format import AssumptionTestKeys


def logistic_regression(data: pd.DataFrame, outcome: str, predictors: List[str], alpha: float) -> Dict[str, Any]:
    """Performs Logistic Regression with comprehensive statistics."""
    try:
        # Set matplotlib to use a non-interactive backend
        matplotlib.use('Agg')
        figures = {}
        
        # Validate that outcome is binary
        unique_vals = data[outcome].unique()
        if len(unique_vals) != 2:
            raise ValueError(f"Outcome variable {outcome} must be binary (have exactly 2 values)")
            
        # Convert outcome to 0/1 if not already
        if not set(unique_vals).issubset({0, 1}):
            # Create mapping of original values to 0/1
            val_map = {unique_vals[0]: 0, unique_vals[1]: 1}
            original_vals = {0: unique_vals[0], 1: unique_vals[1]}
            data = data.copy()
            data[outcome] = data[outcome].map(val_map)
            
        # Construct formula
        formula = f"{outcome} ~ {' + '.join(predictors)}"
        
        # Fit the model
        model = smf.logit(formula, data)
        results = model.fit()
        
        # Extract model parameters
        aic = float(results.aic)
        bic = float(results.bic)
        log_likelihood = float(results.llf)
        
        # Extract coefficients
        coefficients = {}
        for term in results.params.index:
            coefficients[term] = {
                'estimate': float(results.params[term]),
                'std_error': float(results.bse[term]),
                'z_value': float(results.tvalues[term]),
                'p_value': float(results.pvalues[term]),
                'significant': results.pvalues[term] < alpha,
                'odds_ratio': float(np.exp(results.params[term])),
                'ci_lower': float(np.exp(results.conf_int().loc[term, 0])),
                'ci_upper': float(np.exp(results.conf_int().loc[term, 1]))
            }
        
        # Create forest plot for coefficients
        plt.figure(figsize=(10, 6))
        terms = [term for term in results.params.index if term != 'Intercept']
        odds_ratios = [np.exp(results.params[term]) for term in terms]
        ci_lower = [np.exp(results.conf_int().loc[term, 0]) for term in terms]
        ci_upper = [np.exp(results.conf_int().loc[term, 1]) for term in terms]
        
        y_pos = np.arange(len(terms))
        plt.errorbar(odds_ratios, y_pos, xerr=[np.array(odds_ratios)-np.array(ci_lower), np.array(ci_upper)-np.array(odds_ratios)],
                    fmt='o', ecolor='black', capsize=5, color=PASTEL_COLORS[0])
        plt.axvline(x=1, color=PASTEL_COLORS[5], linestyle='--')
        plt.yticks(y_pos, terms)
        plt.xscale('log')
        plt.xlabel('Odds Ratio (log scale)')
        plt.title('Odds Ratios with 95% Confidence Intervals')
        plt.grid(True, alpha=0.3)
        
        # Save figure using fig_to_svg
        figures['odds_ratio_plot'] = fig_to_svg(plt.gcf())
        plt.close()
            
        # Calculate model fit statistics
        # Null model (intercept only)
        try:
            null_formula = f"{outcome} ~ 1"
            null_model = smf.logit(null_formula, data)
            null_results = null_model.fit()
            
            # Likelihood ratio test
            lr_stat = 2 * (results.llf - null_results.llf)
            df = len(predictors)
            lr_pval = stats.chi2.sf(lr_stat, df)
            
            # McFadden's pseudo R-squared
            mcfadden_r2 = 1 - (results.llf / null_results.llf)
            
            # Cox & Snell and Nagelkerke R-squared
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
                'aic': aic,
                'bic': bic,
                'significant': lr_pval < alpha
            }
        except:
            model_fit = {
                'log_likelihood': float(results.llf),
                'aic': aic,
                'bic': bic,
                'error': 'Could not fit null model for comparison'
            }
            
        # Calculate predictions and classification metrics
        try:
            # Get predicted probabilities
            y_pred_prob = results.predict()
            
            # Convert to binary predictions using 0.5 threshold
            y_pred = (y_pred_prob > 0.5).astype(int)
            y_true = data[outcome]
            
            # Calculate confusion matrix
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            # Calculate metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive predictive value
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
            
            # Calculate ROC AUC
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y_true, y_pred_prob))
            
            # Add predicted probabilities distribution
            probs_dist = {
                'mean': float(y_pred_prob.mean()),
                'std': float(y_pred_prob.std()),
                'min': float(y_pred_prob.min()),
                'max': float(y_pred_prob.max()),
            }
            
            # Add F1 score
            f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
            
            # Calculate Brier score (mean squared error between predictions and outcomes)
            brier_score = mean_squared_error(y_true, y_pred_prob)
            
            # Add threshold analysis
            thresholds = np.linspace(0.1, 0.9, 9)
            threshold_analysis = []
            for threshold in thresholds:
                y_pred_t = (y_pred_prob > threshold).astype(int)
                tp_t = np.sum((y_true == 1) & (y_pred_t == 1))
                tn_t = np.sum((y_true == 0) & (y_pred_t == 0))
                fp_t = np.sum((y_true == 0) & (y_pred_t == 1))
                fn_t = np.sum((y_true == 1) & (y_pred_t == 0))
                
                acc_t = (tp_t + tn_t) / (tp_t + tn_t + fp_t + fn_t)
                sens_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
                spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0
                ppv_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
                npv_t = tn_t / (tn_t + fn_t) if (tn_t + fn_t) > 0 else 0
                f1_t = 2 * (ppv_t * sens_t) / (ppv_t + sens_t) if (ppv_t + sens_t) > 0 else 0
                
                threshold_analysis.append({
                    'threshold': float(threshold),
                    'accuracy': float(acc_t),
                    'sensitivity': float(sens_t),
                    'specificity': float(spec_t),
                    'ppv': float(ppv_t),
                    'npv': float(npv_t),
                    'f1': float(f1_t)
                })
            
            # Cross-validation
            cv_results = {}
            try:
                X = data[predictors].copy()
                y = data[outcome].copy()
                # Add constant for statsmodels compatibility
                if 'Intercept' not in X.columns:
                    X['Intercept'] = 1
                
                # Set up cross-validation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
                # Initialize metrics
                cv_acc = []
                cv_sens = []
                cv_spec = []
                cv_auc = []
                
                # Perform cross-validation
                for train_idx, test_idx in cv.split(X, y):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    # Fit model
                    model_cv = sm.Logit(y_train, X_train)
                    results_cv = model_cv.fit(disp=0)
                    
                    # Predict
                    y_pred_prob_cv = results_cv.predict(X_test)
                    y_pred_cv = (y_pred_prob_cv > 0.5).astype(int)
                    
                    # Calculate metrics
                    tn_cv, fp_cv, fn_cv, tp_cv = confusion_matrix(y_test, y_pred_cv).ravel()
                    acc_cv = (tp_cv + tn_cv) / (tp_cv + tn_cv + fp_cv + fn_cv)
                    sens_cv = tp_cv / (tp_cv + fn_cv) if (tp_cv + fn_cv) > 0 else 0
                    spec_cv = tn_cv / (tn_cv + fp_cv) if (tn_cv + fp_cv) > 0 else 0
                    auc_cv = roc_auc_score(y_test, y_pred_prob_cv)
                    
                    cv_acc.append(acc_cv)
                    cv_sens.append(sens_cv)
                    cv_spec.append(spec_cv)
                    cv_auc.append(auc_cv)
                
                cv_results = {
                    'cv_accuracy_mean': float(np.mean(cv_acc)),
                    'cv_accuracy_std': float(np.std(cv_acc)),
                    'cv_sensitivity_mean': float(np.mean(cv_sens)),
                    'cv_sensitivity_std': float(np.std(cv_sens)),
                    'cv_specificity_mean': float(np.mean(cv_spec)),
                    'cv_specificity_std': float(np.std(cv_spec)),
                    'cv_auc_mean': float(np.mean(cv_auc)),
                    'cv_auc_std': float(np.std(cv_auc))
                }
            except Exception as e:
                cv_results = {'error': f'Could not perform cross-validation: {str(e)}'}
            
            prediction_stats = {
                'confusion_matrix': {
                    'tp': int(tp),
                    'tn': int(tn),
                    'fp': int(fp),
                    'fn': int(fn)
                },
                'accuracy': float(accuracy),
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'ppv': float(ppv),
                'npv': float(npv),
                'f1': float(f1),
                'auc': auc,
                'brier_score': float(brier_score),
                'probability_distribution': probs_dist,
                'threshold_analysis': threshold_analysis,
                'cross_validation': cv_results
            }
            
            # Create ROC curve figure
            fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})', color=PASTEL_COLORS[0])
            plt.plot([0, 1], [0, 1], '--', label='Random classifier', color=PASTEL_COLORS[5])
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            # Save ROC curve using fig_to_svg
            figures['roc_curve'] = fig_to_svg(plt.gcf())
            plt.close()
            
            # Create confusion matrix visualization
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap=sns.color_palette(PASTEL_COLORS), cbar=False)
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.title('Confusion Matrix')
            plt.xticks([0.5, 1.5], ['Negative (0)', 'Positive (1)'])
            plt.yticks([0.5, 1.5], ['Negative (0)', 'Positive (1)'])
            
            # Save confusion matrix using fig_to_svg
            figures['confusion_matrix'] = fig_to_svg(plt.gcf())
            plt.close()
            
            # Create calibration curve
            try:
                prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=10)
                plt.figure(figsize=(8, 8))
                plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibration curve', color=PASTEL_COLORS[0])
                plt.plot([0, 1], [0, 1], '--', label='Perfectly calibrated', color=PASTEL_COLORS[5])
                plt.xlabel('Mean predicted probability')
                plt.ylabel('Fraction of positives')
                plt.title('Calibration Curve')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                
                # Save calibration curve using fig_to_svg
                figures['calibration_curve'] = fig_to_svg(plt.gcf())
                plt.close()
            except:
                pass
            
            # Create probability distribution figure
            plt.figure(figsize=(10, 6))
            sns.histplot(data=pd.DataFrame({
                'Probability': y_pred_prob,
                'Actual': y_true.map({0: 'Negative', 1: 'Positive'})
            }), x='Probability', hue='Actual', bins=20, kde=True, palette=[PASTEL_COLORS[0], PASTEL_COLORS[1]])
            plt.title('Distribution of Predicted Probabilities by Actual Outcome')
            plt.xlabel('Predicted Probability')
            plt.ylabel('Count')
            
            # Save probability distribution using fig_to_svg
            figures['probability_distribution'] = fig_to_svg(plt.gcf())
            plt.close()
            
            # Create threshold analysis figure
            plt.figure(figsize=(12, 8))
            metrics = ['accuracy', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1']
            for i, metric in enumerate(metrics):
                values = [analysis[metric] for analysis in threshold_analysis]
                plt.plot(thresholds, values, marker='o', label=metric.capitalize(), color=PASTEL_COLORS[i % len(PASTEL_COLORS)])
            plt.axvline(x=0.5, color='gray', linestyle='--', label='Default threshold (0.5)')
            plt.xlim([0.1, 0.9])
            plt.ylim([0, 1])
            plt.xlabel('Classification Threshold')
            plt.ylabel('Metric Value')
            plt.title('Performance Metrics at Different Classification Thresholds')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save threshold analysis using fig_to_svg
            figures['threshold_analysis'] = fig_to_svg(plt.gcf())
            plt.close()
            
        except:
            prediction_stats = {
                'error': 'Could not calculate predictions'
            }
            
        # Test assumptions
        assumptions = {}
        
        # 1. Multicollinearity
        try:
            # Use the test from AssumptionTestKeys.MULTICOLLINEARITY
            multicollinearity_test = AssumptionTestKeys.MULTICOLLINEARITY.value["function"]()
            multicollinearity_result = multicollinearity_test.run_test(df=data, covariates=predictors)
            
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
        
        # 2. Linearity of the logit
        # For each continuous predictor, test linearity with the logit
        linearity_results = {}
        
        # Get predicted probabilities
        logit_values = np.log(y_pred_prob / (1 - y_pred_prob))
        
        # For each numeric predictor, test linearity with the logit
        for predictor in predictors:
            if predictor == 'Intercept':
                continue
                
            if pd.api.types.is_numeric_dtype(data[predictor]):
                # Use the test from AssumptionTestKeys.LINEARITY
                linearity_test = AssumptionTestKeys.LINEARITY.value["function"]()
                
                # Test linearity between predictor and logit
                try:
                    # The LinearityTest.run_test() method expects x and y as pandas Series or arrays
                    # This is where the error occurs - we need to pass Series objects, not use scatter directly
                    x_series = pd.Series(data[predictor])
                    y_series = pd.Series(logit_values)
                    
                    # Call the standardized test correctly
                    linearity_result = linearity_test.run_test(x=x_series, y=y_series)
                    
                    # Extract results
                    assumptions['linearity_of_logit_'+predictor] = linearity_result
                except Exception as inner_e:
                    # If there's an error, skip this predictor
                    assumptions['linearity_of_logit_'+predictor] = {
                        'error': f'Could not test linearity for {predictor}: {str(inner_e)}'
                    }
        
        # 3. Sample size check
        try:
            # Use the test from AssumptionTestKeys.SAMPLE_SIZE
            sample_size_test = AssumptionTestKeys.SAMPLE_SIZE.value["function"]()
            
            # For logistic regression, we need at least 10 events per predictor
            # Count events in the minority class
            n_minority = min(sum(data[outcome] == 0), sum(data[outcome] == 1))
            min_recommended = 10 * len(predictors)
            
            # Check if we have enough events
            sample_size_result = sample_size_test.run_test(
                data=pd.Series(range(n_minority)),  # Dummy data, just need the length
                min_recommended=min_recommended
            )
            
            # Extract results
            assumptions['sample_size'] = {
                'n_total': len(data),
                'n_events': int(sum(data[outcome] == 1)),
                'n_non_events': int(sum(data[outcome] == 0)),
                'n_minority_class': int(n_minority),
                'min_recommended': int(min_recommended),
                'events_per_predictor': float(n_minority / len(predictors)),
                'satisfied': sample_size_result['result'].value == 'passed',
                'details': sample_size_result.get('details', '')
            }
        except Exception as e:
            assumptions['sample_size'] = {
                'error': f'Could not check sample size: {str(e)}'
            }
        
        # 4. Influential observations
        try:
            # Use the test from AssumptionTestKeys.INFLUENTIAL_POINTS
            influential_points_test = AssumptionTestKeys.INFLUENTIAL_POINTS.value["function"]()
            
            # Calculate residuals, leverage, and fitted values
            influence = results.get_influence()
            cooks_d = influence.cooks_distance[0]
            leverage = influence.hat_matrix_diag
            residuals = results.resid_pearson
            fitted = results.predict()
            
            # Create a dummy X matrix for the test
            X = data[predictors].copy()
            if 'Intercept' not in X.columns:
                X['Intercept'] = 1
                
            # Run the influential points test
            influential_points_result = influential_points_test.run_test(
                residuals=residuals,
                leverage=leverage,
                fitted=fitted,
                X=X
            )
            
            # Extract results
            assumptions['influential_observations'] = {
                'satisfied': influential_points_result['result'].value == 'passed',
                'details': influential_points_result['details'],
                'influential_points': influential_points_result['influential_points'],
                'warnings': influential_points_result['warnings'],
                'figures': influential_points_result['figures'],
                'n_influential': len(influential_points_result['influential_points']),
                'threshold': influential_points_result.get('threshold', 0.5)  # Default if not provided
            }
            
        except Exception as e:
            assumptions['influential_observations'] = {
                'error': f'Could not check for influential observations: {str(e)}'
            }
        
        # 5. Goodness of fit
        try:
            # Use the test from AssumptionTestKeys.GOODNESS_OF_FIT
            goodness_of_fit_test = AssumptionTestKeys.GOODNESS_OF_FIT.value["function"]()
            
            # Run Hosmer-Lemeshow test
            goodness_of_fit_result = goodness_of_fit_test.run_test(
                observed=data[outcome],
                expected=y_pred_prob
            )
            
            # Extract results
            assumptions['goodness_of_fit'] = {
                'test': goodness_of_fit_result.get('test_used', 'Hosmer-Lemeshow'),
                'statistic': float(goodness_of_fit_result.get('statistic', 0)),
                'p_value': float(goodness_of_fit_result.get('p_value', 1)),
                'satisfied': goodness_of_fit_result['result'].value == 'passed',
                'details': goodness_of_fit_result.get('details', '')
            }
        except Exception as e:
            assumptions['goodness_of_fit'] = {
                'error': f'Could not test goodness of fit: {str(e)}'
            }
            
        # 6. Homoscedasticity of residuals
        try:
            # Use the test from AssumptionTestKeys.HOMOSCEDASTICITY
            homoscedasticity_test = AssumptionTestKeys.HOMOSCEDASTICITY.value["function"]()
            
            # Run test on residuals and predicted values
            homoscedasticity_result = homoscedasticity_test.run_test(
                residuals=results.resid_pearson,
                predicted=results.predict()
            )
            
            # Extract results
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
            
        # 7. Residual normality
        try:
            # Use the test from AssumptionTestKeys.RESIDUAL_NORMALITY
            residual_normality_test = AssumptionTestKeys.RESIDUAL_NORMALITY.value["function"]()
            
            # Run test on residuals
            residual_normality_result = residual_normality_test.run_test(
                residuals=results.resid_pearson
            )
            
            # Extract results
            assumptions['residual_normality'] = {
                'result': residual_normality_result.get('result', None),
                'message': residual_normality_result.get('message', ''),
                'details': residual_normality_result.get('details', {})
            }
        except Exception as e:
            assumptions['residual_normality'] = {
                'error': f'Could not test residual normality: {str(e)}'
            }
            
        # 8. Overdispersion (for logistic regression, check if variance > mean)
        try:
            # Use the test from AssumptionTestKeys.OVERDISPERSION
            overdispersion_test = AssumptionTestKeys.OVERDISPERSION.value["function"]()
            
            # Run test on observed and predicted values
            overdispersion_result = overdispersion_test.run_test(
                observed=data[outcome],
                predicted=y_pred_prob
            )
            
            # Extract results
            assumptions['overdispersion'] = {
                'result': overdispersion_result.get('result', None),
                'message': overdispersion_result.get('message', ''),
                'details': overdispersion_result.get('details', {})
            }
        except Exception as e:
            assumptions['overdispersion'] = {
                'error': f'Could not test for overdispersion: {str(e)}'
            }
            
        # Create interpretation
        significant_predictors = [term for term, coef in coefficients.items() 
                                 if coef.get('significant', False) and term != 'Intercept']
        
        interpretation = f"Logistic Regression with {outcome} as outcome "
        if 'original_vals' in locals():
            interpretation += f"({original_vals[1]} vs {original_vals[0]}) "
        interpretation += f"and predictors ({', '.join(predictors)}).\n\n"
        
        # Interpret model fit
        if 'lr_p_value' in model_fit:
            interpretation += f"The model is {'statistically significant' if model_fit['significant'] else 'not statistically significant'} "
            interpretation += f"compared to the null model (χ²({model_fit['lr_df']}) = {model_fit['lr_statistic']:.3f}, p = {model_fit['lr_p_value']:.5f}).\n"
            
            if 'nagelkerke_r2' in model_fit:
                interpretation += f"Nagelkerke's R² = {model_fit['nagelkerke_r2']:.3f}, suggesting that the model explains "
                interpretation += f"{model_fit['nagelkerke_r2']*100:.1f}% of the variation in the outcome.\n\n"
        
        # Interpret coefficients
        if significant_predictors:
            interpretation += "Significant predictors:\n"
            for term in significant_predictors:
                coef = coefficients[term]
                interpretation += f"- {term}: β = {coef['estimate']:.3f}, OR = {coef['odds_ratio']:.3f} "
                interpretation += f"(95% CI: {coef['ci_lower']:.3f}-{coef['ci_upper']:.3f}), p = {coef['p_value']:.5f}\n"
                
                # Interpret direction of effect
                if coef['odds_ratio'] > 1:
                    interpretation += f"  For each unit increase in {term}, the odds of the outcome increase by {(coef['odds_ratio']-1)*100:.1f}%.\n"
                else:
                    interpretation += f"  For each unit increase in {term}, the odds of the outcome decrease by {(1-coef['odds_ratio'])*100:.1f}%.\n"
        else:
            interpretation += "No significant predictors were found.\n"
            
        # Interpret classification performance if available
        if 'accuracy' in prediction_stats:
            interpretation += f"\nModel classification performance:\n"
            interpretation += f"- Accuracy: {prediction_stats['accuracy']*100:.1f}%\n"
            interpretation += f"- Sensitivity: {prediction_stats['sensitivity']*100:.1f}%\n"
            interpretation += f"- Specificity: {prediction_stats['specificity']*100:.1f}%\n"
            interpretation += f"- F1 Score: {prediction_stats['f1']*100:.1f}%\n"
            
            if 'auc' in prediction_stats:
                interpretation += f"- Area Under the ROC Curve (AUC): {prediction_stats['auc']:.3f}\n"
                
                # Interpret AUC
                if prediction_stats['auc'] > 0.9:
                    interpretation += "  The AUC indicates excellent discrimination.\n"
                elif prediction_stats['auc'] > 0.8:
                    interpretation += "  The AUC indicates good discrimination.\n"
                elif prediction_stats['auc'] > 0.7:
                    interpretation += "  The AUC indicates acceptable discrimination.\n"
                else:
                    interpretation += "  The AUC indicates poor discrimination.\n"
                    
            # Add Brier score interpretation
            if 'brier_score' in prediction_stats:
                interpretation += f"- Brier Score: {prediction_stats['brier_score']:.3f} (lower is better, perfect predictions = 0)\n"
                
            # Add cross-validation results if available
            if 'cross_validation' in prediction_stats and 'cv_accuracy_mean' in prediction_stats['cross_validation']:
                cv = prediction_stats['cross_validation']
                interpretation += f"\nCross-validation results (5-fold):\n"
                interpretation += f"- Mean Accuracy: {cv['cv_accuracy_mean']*100:.1f}% (±{cv['cv_accuracy_std']*100:.1f}%)\n"
                interpretation += f"- Mean AUC: {cv['cv_auc_mean']:.3f} (±{cv['cv_auc_std']:.3f})\n"
                
                # Add model generalization assessment
                train_auc = prediction_stats['auc']
                cv_auc = cv['cv_auc_mean']
                auc_diff = train_auc - cv_auc
                
                if auc_diff > 0.1:
                    interpretation += "  The model shows signs of overfitting as the training AUC is considerably higher than the cross-validation AUC.\n"
                elif auc_diff < 0.03:
                    interpretation += "  The model generalizes well as the training and cross-validation AUCs are very similar.\n"
                else:
                    interpretation += "  The model shows mild signs of overfitting but should still generalize reasonably well.\n"
        
        # Interpret assumptions
        interpretation += "\nAssumption tests:\n"
        
        # Multicollinearity
        if 'multicollinearity' in assumptions and 'satisfied' in assumptions['multicollinearity']:
            multicollinearity_result = assumptions['multicollinearity']
            interpretation += "- Multicollinearity: "
            
            if multicollinearity_result['satisfied']:
                interpretation += "No significant multicollinearity detected among predictors.\n"
            else:
                interpretation += f"{multicollinearity_result['details']}\n"
        
        # Linearity of logit
        if 'linearity_of_logit' in assumptions and 'overall_satisfied' in assumptions['linearity_of_logit']:
            linearity_result = assumptions['linearity_of_logit']
            interpretation += "- Linearity of the logit: "
            
            if linearity_result['overall_satisfied']:
                interpretation += "The assumption of linearity between predictors and the logit is satisfied.\n"
            else:
                nonlinear_predictors = [pred for pred, result in linearity_result['results'].items() 
                                      if 'satisfied' in result and not result['satisfied']]
                
                if nonlinear_predictors:
                    interpretation += f"Potential non-linear relationships detected for: {', '.join(nonlinear_predictors)}. "
                    interpretation += "Consider transformations or including polynomial terms for these predictors.\n"
                else:
                    interpretation += "Could not fully assess linearity of the logit.\n"
        
        # Sample size
        if 'sample_size' in assumptions and 'satisfied' in assumptions['sample_size']:
            sample_size_result = assumptions['sample_size']
            interpretation += "- Sample size: "
            
            if sample_size_result['satisfied']:
                interpretation += f"Adequate sample size with {sample_size_result['events_per_predictor']:.1f} events per predictor.\n"
            else:
                interpretation += f"Sample size may be inadequate with only {sample_size_result['events_per_predictor']:.1f} events per predictor "
                interpretation += f"(recommended minimum: 10). Results should be interpreted with caution.\n"
        
        # Influential observations
        if 'influential_observations' in assumptions and 'satisfied' in assumptions['influential_observations']:
            influential_result = assumptions['influential_observations']
            interpretation += "- Influential observations: "
            
            if influential_result['satisfied']:
                interpretation += "No influential observations detected.\n"
            else:
                # Add safety checks for missing keys
                n_influential = influential_result.get('n_influential', 'some')
                threshold = influential_result.get('threshold', 'high')
                
                interpretation += f"Found {n_influential} potentially influential observations "
                if isinstance(threshold, (int, float)):
                    interpretation += f"(Cook's distance > {threshold:.5f}). "
                else:
                    interpretation += f"(Cook's distance > {threshold}). "
                interpretation += "Consider examining these cases and potentially refitting the model without extreme outliers if they represent errors.\n"
        
        # Goodness of fit
        if 'goodness_of_fit' in assumptions and 'satisfied' in assumptions['goodness_of_fit']:
            goodness_result = assumptions['goodness_of_fit']
            interpretation += f"- Goodness of fit: {goodness_result['test']} test, p = {goodness_result['p_value']:.5f}, "
            
            if goodness_result['satisfied']:
                interpretation += "indicating the model fits the data well.\n"
            else:
                interpretation += "suggesting potential issues with model fit. Consider adding interaction terms, non-linear terms, or additional predictors.\n"
                
        # Add model calibration assessment if available
        if 'calibration_curve' in figures:
            interpretation += "\nModel Calibration: Review the calibration curve to assess whether predicted probabilities match observed rates. "
            interpretation += "Ideally, points should fall close to the diagonal line, indicating well-calibrated predictions.\n"
        
        # Add threshold analysis interpretation
        if 'threshold_analysis' in prediction_stats:
            interpretation += "\nThreshold Analysis: The default classification threshold of 0.5 can be adjusted to optimize specific performance metrics. "
            interpretation += "Review the threshold analysis graph to identify alternative thresholds that may better balance sensitivity and specificity for your specific needs.\n"
            
        # Add advice on model improvement if performance is not optimal
        if 'auc' in prediction_stats and prediction_stats['auc'] < 0.7:
            interpretation += "\nModel Improvement Suggestions:\n"
            interpretation += "- Consider additional predictors that may better explain the outcome\n"
            interpretation += "- Explore interaction terms between existing predictors\n"
            interpretation += "- For continuous predictors with non-linear relationships, consider polynomial terms or splines\n"
            interpretation += "- If class imbalance exists, consider resampling techniques or weighted models\n"
        
        return {
            'test': 'Logistic Regression',
            'coefficients': coefficients,
            'model_fit': model_fit,
            'prediction_stats': prediction_stats,
            'assumptions': assumptions,
            'formula': formula,
            'summary': str(results.summary()),
            'interpretation': interpretation,
            'figures': figures
        }
    except Exception as e:
        return {
            'test': 'Logistic Regression',
            'error': str(e),
            'traceback': traceback.format_exc()
        }