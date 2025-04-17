import traceback
import pandas as pd
from scipy import stats
import numpy as np
from typing import Dict, List, Any
import statsmodels.formula.api as smf
from typing import Dict, List, Any
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
    SampleSizeTest,
    ZeroInflationTest,
    StationarityTest
)

def negative_binomial_regression(data: pd.DataFrame, outcome: str, predictors: List[str], alpha: float) -> Dict[str, Any]:
    """Performs Negative Binomial Regression with comprehensive statistics and assumption checks."""
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
        
        # Fit the model
        model = smf.negativebinomial(formula, data)
        results = model.fit(disp=0)  # Suppress convergence messages
        
        # Extract model parameters
        aic = float(results.aic)
        bic = float(results.bic)
        log_likelihood = float(results.llf)
        alpha_param = float(results.params[-1])  # Dispersion parameter
        
        # Extract coefficients with more detailed information
        coefficients = {}
        for term in results.params.index:
            if term == 'alpha':  # Skip the dispersion parameter in coefficients
                continue
                
            coefficients[term] = {
                'estimate': float(results.params[term]),
                'std_error': float(results.bse[term]),
                'z_value': float(results.tvalues[term]),
                'p_value': float(results.pvalues[term]),
                'significant': results.pvalues[term] < alpha,
                'irr': float(np.exp(results.params[term])),
                'ci_lower': float(np.exp(results.conf_int().loc[term, 0])),
                'ci_upper': float(np.exp(results.conf_int().loc[term, 1])),
                'percent_change': float((np.exp(results.params[term]) - 1) * 100),
                'standardized_coef': float(results.params[term] * data[term].std() if term in data.columns else np.nan)
            }
            
        # Calculate model fit statistics
        # Null model (intercept only)
        try:
            null_formula = f"{outcome} ~ 1"
            null_model = smf.negativebinomial(null_formula, data)
            null_results = null_model.fit(disp=0)
            
            # Likelihood ratio test
            lr_stat = 2 * (results.llf - null_results.llf)
            df = len(predictors)
            lr_pval = stats.chi2.sf(lr_stat, df)
            
            # McFadden's pseudo R-squared
            mcfadden_r2 = 1 - (results.llf / null_results.llf)
            
            # Calculate additional pseudo R-squared measures
            # Cox & Snell R²
            n = len(data)
            cox_snell_r2 = 1 - np.exp(-(2/n) * (results.llf - null_results.llf))
            
            # Nagelkerke/Cragg & Uhler R²
            nagelkerke_r2 = cox_snell_r2 / (1 - np.exp((2/n) * null_results.llf))
            
            # Manually calculate deviance and pearson chi2
            y_obs = data[outcome]
            y_pred = results.predict()
            # Pearson chi-square: sum of squared Pearson residuals
            pearson_residuals = (y_obs - y_pred) / np.sqrt(y_pred)
            pearson_chi2 = np.sum(pearson_residuals**2)
            
            # Deviance for negative binomial: can be approximated with the scaled deviance
            # Using formula: 2 * sum(y_obs * log(y_obs/y_pred) - (y_obs + alpha) * log((y_obs + alpha)/(y_pred + alpha)))
            # For simplicity, using 2 * (saturated_llf - results.llf)
            # where saturated_llf is approximated as the model perfectly fitting the data
            deviance = 2 * (results.llf - null_results.llf)
            
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
                'alpha_parameter': alpha_param,
                'significant': lr_pval < alpha,
                'deviance': float(deviance),
                'pearson_chi2': float(pearson_chi2),
                'df_resid': int(results.df_resid),
                'df_model': int(results.df_model)
            }
        except Exception as e:
            # Manually calculate deviance and pearson chi2 even if null model fails
            y_obs = data[outcome]
            y_pred = results.predict()
            # Pearson chi-square
            pearson_residuals = (y_obs - y_pred) / np.sqrt(y_pred)
            pearson_chi2 = np.sum(pearson_residuals**2)
            
            # For deviance when null model fails, use a different approximation
            deviance = 2 * np.sum(y_obs * np.log(np.maximum(y_obs, 0.1) / y_pred) - (y_obs - y_pred))
            
            model_fit = {
                'log_likelihood': float(results.llf),
                'aic': aic,
                'bic': bic,
                'alpha_parameter': alpha_param,
                'deviance': float(deviance),
                'pearson_chi2': float(pearson_chi2),
                'df_resid': int(results.df_resid),
                'df_model': int(results.df_model),
                'error': f'Could not fit null model for comparison: {str(e)}'
            }
            
        # Compare with Poisson model to test significance of dispersion parameter
        try:
            poisson_model = smf.poisson(formula, data)
            poisson_results = poisson_model.fit(disp=0)
            
            # Likelihood ratio test for dispersion
            lr_stat_disp = 2 * (results.llf - poisson_results.llf)
            df_disp = 1  # Testing one parameter (alpha)
            lr_pval_disp = stats.chi2.sf(lr_stat_disp, df_disp)
            
            # Calculate dispersion statistics
            pearson_chi2 = np.sum((data[outcome] - poisson_results.predict())**2 / poisson_results.predict())
            dispersion_stat = pearson_chi2 / poisson_results.df_resid
            
            dispersion_test = {
                'lr_statistic': float(lr_stat_disp),
                'p_value': float(lr_pval_disp),
                'significant': lr_pval_disp < alpha,
                'poisson_aic': float(poisson_results.aic),
                'nb_aic': float(results.aic),
                'poisson_bic': float(poisson_results.bic),
                'nb_bic': float(results.bic),
                'pearson_dispersion': float(dispersion_stat),
                'alpha_parameter': float(alpha_param),
                'alpha_p_value': float(results.pvalues.get('alpha', np.nan))
            }
        except Exception as e:
            dispersion_test = {
                'error': f'Could not compare with Poisson model: {str(e)}'
            }
            
        # Calculate predicted counts and compare to observed
        y_pred = results.predict()
        residuals = data[outcome] - y_pred
        pearson_residuals = residuals / np.sqrt(y_pred)
        deviance_residuals = np.sign(residuals) * np.sqrt(2 * (data[outcome] * np.log(data[outcome] / y_pred) - (data[outcome] - y_pred)))
        deviance_residuals = np.where(np.isnan(deviance_residuals), 0, deviance_residuals)  # Handle zeros
        
        # Calculate additional prediction metrics
        sum_squared_error = np.sum(residuals**2)
        total_sum_squares = np.sum((data[outcome] - data[outcome].mean())**2)
        r_squared_raw = 1 - (sum_squared_error / total_sum_squares)
        
        prediction_stats = {
            'mean_observed': float(data[outcome].mean()),
            'mean_predicted': float(y_pred.mean()),
            'min_predicted': float(y_pred.min()),
            'max_predicted': float(y_pred.max()),
            'correlation_obs_pred': float(np.corrcoef(data[outcome], y_pred)[0, 1]),
            'mean_absolute_error': float(np.mean(np.abs(residuals))),
            'root_mean_squared_error': float(np.sqrt(np.mean(residuals**2))),
            'mean_squared_error': float(np.mean(residuals**2)),
            'median_absolute_error': float(np.median(np.abs(residuals))),
            'r_squared_raw': float(r_squared_raw),
            'sum_squared_error': float(sum_squared_error),
            'total_sum_squares': float(total_sum_squares),
            'mean_pearson_residual': float(np.mean(pearson_residuals)),
            'max_pearson_residual': float(np.max(np.abs(pearson_residuals))),
            'mean_deviance_residual': float(np.mean(deviance_residuals)),
            'max_deviance_residual': float(np.max(np.abs(deviance_residuals))),
            'zero_count': int(np.sum(data[outcome] == 0)),
            'zero_percent': float(np.mean(data[outcome] == 0) * 100)
        }
        
        # Run assumption tests
        assumptions = {}
        
        # 1. Multicollinearity test
        multicollinearity_test = MulticollinearityTest()
        assumptions['multicollinearity'] = multicollinearity_test.run_test(
            df=data,
            covariates=predictors
        )
        
        # 2. Overdispersion test
        overdispersion_test = OverdispersionTest()
        # Get number of parameters for degrees of freedom calculation
        num_params = len(results.params)
        assumptions['overdispersion'] = overdispersion_test.run_test(
            observed=data[outcome],
            predicted=y_pred,
            model_type='poisson',
            num_params=num_params,
            alpha=alpha
        )
        
        # 3. Outlier test
        outlier_test = OutlierTest()
        assumptions['outliers'] = outlier_test.run_test(
            data=pearson_residuals
        )
        
        # 4. Independence test
        independence_test = IndependenceTest()
        assumptions['independence'] = independence_test.run_test(
            data=residuals
        )
        
        # 5. Goodness of fit test
        goodness_test = GoodnessOfFitTest()
        try:
            # Normalize expected values to ensure they sum to the same total as observed values
            # This prevents the "sum of frequencies must agree" error
            normalized_expected = y_pred * (data[outcome].sum() / y_pred.sum())
            assumptions['goodness_of_fit'] = goodness_test.run_test(
                observed=data[outcome],
                expected=normalized_expected
            )
        except Exception as e:
            # Fallback if goodness of fit test fails
            assumptions['goodness_of_fit'] = {
                'result': 'not_applicable',
                'details': f"Goodness of fit test could not be performed: {str(e)}",
                'warnings': [f"Error in goodness of fit test: {str(e)}"]
            }
        
        # 6. Sample size test
        sample_size_test = SampleSizeTest()
        # Calculate minimum recommended sample size - common rule is 10-15 observations per parameter
        min_recommended_size = (len(predictors) + 1) * 10  # +1 for intercept
        assumptions['sample_size'] = sample_size_test.run_test(
            data=data[outcome].values,
            min_recommended=min_recommended_size
        )
        
        # 7. Add Zero Inflation test
        zero_inflation_test = ZeroInflationTest()
        assumptions['zero_inflation'] = zero_inflation_test.run_test(
            data=data[outcome].values
        )
        
        # 8. Add Stationarity test if data can be considered time series
        # This is optional and depends on the data - may need user specification
        # assumptions['stationarity'] = StationarityTest().run_test(data=data[outcome].values) - TO BE ADDED IN FUTURE - DO NOT REMOVE COMMENT
        
        # Create detailed interpretation
        significant_predictors = [term for term, coef in coefficients.items() 
                                 if coef.get('significant', False) and term != 'Intercept']
        
        interpretation = f"Negative Binomial Regression with {outcome} as outcome and predictors ({', '.join(predictors)}).\n\n"
        
        # Interpret model fit
        if 'lr_p_value' in model_fit:
            interpretation += f"The model is {'statistically significant' if model_fit['significant'] else 'not statistically significant'} "
            interpretation += f"compared to the null model (χ²({model_fit['lr_df']}) = {model_fit['lr_statistic']:.3f}, p = {model_fit['lr_p_value']:.5f}).\n"
            
            if 'mcfadden_r2' in model_fit:
                interpretation += f"McFadden's pseudo R² = {model_fit['mcfadden_r2']:.3f}, suggesting that the model explains "
                interpretation += f"{model_fit['mcfadden_r2']*100:.1f}% of the null deviance.\n"
                
            if 'cox_snell_r2' in model_fit and 'nagelkerke_r2' in model_fit:
                interpretation += f"Cox & Snell R² = {model_fit['cox_snell_r2']:.3f}, Nagelkerke R² = {model_fit['nagelkerke_r2']:.3f}.\n"
                
            interpretation += f"AIC = {model_fit['aic']:.2f}, BIC = {model_fit['bic']:.2f}.\n\n"
        
        # Interpret dispersion test
        if 'p_value' in dispersion_test:
            interpretation += f"Compared to Poisson regression, the negative binomial model {'is' if dispersion_test['significant'] else 'is not'} significantly better "
            interpretation += f"(χ²(1) = {dispersion_test['lr_statistic']:.3f}, p = {dispersion_test['p_value']:.5f}).\n"
            
            if 'pearson_dispersion' in dispersion_test:
                interpretation += f"Pearson dispersion statistic = {dispersion_test['pearson_dispersion']:.3f} "
                interpretation += f"(values > 1 indicate overdispersion).\n"
                
            if 'alpha_parameter' in dispersion_test:
                interpretation += f"Dispersion parameter alpha = {dispersion_test['alpha_parameter']:.5f}.\n"
            
            if dispersion_test['significant']:
                interpretation += "This indicates that overdispersion is present and the negative binomial model is appropriate.\n\n"
            else:
                interpretation += "This suggests that overdispersion may not be significant and a simpler Poisson model might be sufficient.\n\n"
        
        # Interpret coefficients
        if significant_predictors:
            interpretation += "Significant predictors:\n"
            for term in significant_predictors:
                coef = coefficients[term]
                interpretation += f"- {term}: β = {coef['estimate']:.3f}, IRR = {coef['irr']:.3f} "
                interpretation += f"(95% CI: {coef['ci_lower']:.3f}-{coef['ci_upper']:.3f}), p = {coef['p_value']:.5f}\n"
                
                # Interpret incident rate ratio
                if coef['irr'] > 1:
                    interpretation += f"  For each unit increase in {term}, the expected count increases by {coef['percent_change']:.1f}%.\n"
                else:
                    interpretation += f"  For each unit increase in {term}, the expected count decreases by {abs(coef['percent_change']):.1f}%.\n"
        else:
            interpretation += "No significant predictors were found.\n"
            
        # Model performance
        interpretation += f"\nModel performance: The correlation between observed and predicted counts is {prediction_stats['correlation_obs_pred']:.3f}.\n"
        interpretation += f"Mean observed count: {prediction_stats['mean_observed']:.3f}, Mean predicted count: {prediction_stats['mean_predicted']:.3f}.\n"
        interpretation += f"Root mean squared error: {prediction_stats['root_mean_squared_error']:.3f}.\n"
        interpretation += f"Zero counts: {prediction_stats['zero_count']} ({prediction_stats['zero_percent']:.1f}% of observations).\n\n"
        
        # Interpret assumption tests
        interpretation += "Assumption checks:\n"
        
        # Multicollinearity
        if 'multicollinearity' in assumptions:
            result = assumptions['multicollinearity'].get('result', 'not_applicable')
            if result == 'passed':
                interpretation += "- Multicollinearity: No problematic multicollinearity detected.\n"
            elif result == 'warning':
                interpretation += "- Multicollinearity: Some moderate multicollinearity detected. Consider centering variables or using regularization.\n"
            elif result == 'failed':
                interpretation += "- Multicollinearity: Severe multicollinearity detected. Consider removing or combining highly correlated predictors.\n"
        
        # Overdispersion
        if 'overdispersion' in assumptions:
            result = assumptions['overdispersion'].get('result', 'not_applicable')
            if result == 'passed':
                interpretation += "- Overdispersion: No significant overdispersion detected. The negative binomial model may be unnecessarily complex.\n"
            elif result == 'warning':
                interpretation += "- Overdispersion: Mild overdispersion detected. The negative binomial model is appropriate.\n"
            elif result == 'failed':
                interpretation += "- Overdispersion: Significant overdispersion detected. The negative binomial model is appropriate.\n"
        
        # Outliers
        if 'outliers' in assumptions:
            result = assumptions['outliers'].get('result', 'not_applicable')
            if result == 'passed':
                interpretation += "- Outliers: No influential outliers detected.\n"
            elif result == 'warning':
                interpretation += "- Outliers: Some potential outliers detected. Consider examining these cases.\n"
            elif result == 'failed':
                interpretation += "- Outliers: Significant outliers detected that may be influencing the model. Consider robust regression methods.\n"
        
        # Independence
        if 'independence' in assumptions:
            result = assumptions['independence'].get('result', 'not_applicable')
            if result == 'passed':
                interpretation += "- Independence: Residuals appear to be independent.\n"
            elif result == 'warning':
                interpretation += "- Independence: Some potential autocorrelation in residuals. Consider time series methods if data is temporal.\n"
            elif result == 'failed':
                interpretation += "- Independence: Significant autocorrelation detected. Consider time series methods or mixed models.\n"
        
        # Goodness of fit
        if 'goodness_of_fit' in assumptions:
            result = assumptions['goodness_of_fit'].get('result', 'not_applicable')
            if result == 'passed':
                interpretation += "- Goodness of fit: The model fits the data well.\n"
            elif result == 'warning':
                interpretation += "- Goodness of fit: The model fit is acceptable but could be improved.\n"
            elif result == 'failed':
                interpretation += "- Goodness of fit: The model does not fit the data well. Consider adding interaction terms or non-linear terms.\n"
        
        # Sample size
        if 'sample_size' in assumptions:
            result = assumptions['sample_size'].get('result', 'not_applicable')
            if result == 'passed':
                interpretation += "- Sample size: Adequate sample size for the number of predictors.\n"
            elif result == 'warning':
                interpretation += "- Sample size: Sample size is minimally adequate. Results should be interpreted with caution.\n"
            elif result == 'failed':
                interpretation += "- Sample size: Sample size may be too small for reliable estimation with this many predictors.\n"
        
        # Zero inflation
        if 'zero_inflation' in assumptions:
            result = assumptions['zero_inflation'].get('result', 'not_applicable') 
            if result == 'passed':
                interpretation += "- Zero inflation: The amount of zeros in the data is consistent with the negative binomial distribution.\n"
            elif result == 'warning':
                interpretation += "- Zero inflation: Some evidence of excess zeros. Consider a zero-inflated negative binomial model.\n"
            elif result == 'failed':
                interpretation += "- Zero inflation: Significant excess zeros detected. A zero-inflated or hurdle model is recommended.\n"
        
        # Create a dictionary with all residuals for potential plotting
        residuals_data = {
            'raw_residuals': residuals.tolist(),
            'pearson_residuals': pearson_residuals.tolist(),
            'deviance_residuals': deviance_residuals.tolist(),
            'predicted': y_pred.tolist(),
            'observed': data[outcome].tolist()
        }
        
        # Create diagnostic plots
        figures = {}
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from data.selection.detailed_tests.formatting import fig_to_svg, PASTEL_COLORS
            
            # Residuals vs Predicted plot
            try:
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                ax1.scatter(y_pred, pearson_residuals, alpha=0.5, color=PASTEL_COLORS[0])
                ax1.axhline(y=0, color=PASTEL_COLORS[2], linestyle='-')
                ax1.set_xlabel('Predicted Values')
                ax1.set_ylabel('Pearson Residuals')
                ax1.set_title('Residuals vs Predicted Values')
                
                figures['residuals_vs_predicted'] = fig_to_svg(fig1)
            except Exception as e:
                figures['residuals_vs_predicted_error'] = str(e)
            
            # QQ plot of residuals
            try:
                import statsmodels.api as sm
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                sm.qqplot(np.array(pearson_residuals), line='45', ax=ax2, 
                         marker='o', markerfacecolor=PASTEL_COLORS[1], 
                         markeredgecolor='none', alpha=0.5)
                ax2.set_title('Q-Q Plot of Pearson Residuals')
                
                figures['qq_plot'] = fig_to_svg(fig2)
            except Exception as e:
                figures['qq_plot_error'] = str(e)
            
            # Observed vs Predicted plot
            try:
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                ax3.scatter(data[outcome], y_pred, alpha=0.5, color=PASTEL_COLORS[3])
                max_val = max(data[outcome].max(), y_pred.max())
                ax3.plot([0, max_val], [0, max_val], '--', color=PASTEL_COLORS[4])
                ax3.set_xlabel('Observed Values')
                ax3.set_ylabel('Predicted Values')
                ax3.set_title('Observed vs Predicted Values')
                
                figures['observed_vs_predicted'] = fig_to_svg(fig3)
            except Exception as e:
                figures['observed_vs_predicted_error'] = str(e)
                
        except Exception as e:
            figures['plot_generation_error'] = str(e)
        
        return {
            'test': 'Negative Binomial Regression',
            'coefficients': coefficients,
            'model_fit': model_fit,
            'overdispersion': dispersion_test,
            'prediction_stats': prediction_stats,
            'assumptions': assumptions,
            'formula': formula,
            'summary': str(results.summary()),
            'interpretation': interpretation,
            'figures': figures,
            'alpha_parameter': alpha_param,
            'n_observations': len(data),
            'n_predictors': len(predictors),
            'zero_count': prediction_stats['zero_count'],
            'zero_percent': prediction_stats['zero_percent']
        }
    except Exception as e:
        return {
            'test': 'Negative Binomial Regression',
            'error': str(e),
            'traceback': traceback.format_exc()
        }