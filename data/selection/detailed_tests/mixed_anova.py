from typing import Dict, List, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import traceback
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pingouin as pg
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from typing import Dict, Any, List
from data.selection.detailed_tests.formatting import fig_to_svg, PASTEL_COLORS, PASTEL_CMAP
from data.assumptions.tests import AssumptionResult


def mixed_anova(data: pd.DataFrame, between_factors: List[str], within_factors: List[str], subject_id: str, alpha: float, outcome: str) -> Dict[str, Any]:
    """
    Performs Mixed ANOVA using statsmodels with comprehensive statistics, assumption checks, and visualizations.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataset containing all variables
    between_factors : List[str]
        List of column names for between-subjects factors
    within_factors : List[str]
        List of column names for within-subjects (repeated measures) factors
    subject_id : str
        Column name identifying subjects
    alpha : float
        Significance level
    outcome : str
        Name of the outcome/dependent variable
    
    Returns:
    --------
    Dict[str, Any]
        Comprehensive results including ANOVA statistics, post-hoc tests, assumption checks, and visualizations
    """
    try:
        # Filter out any rows with missing values
        data_clean = data.dropna(subset=[outcome] + between_factors + within_factors + [subject_id])
        
        if len(data_clean) == 0:
            return {
                'test': 'Mixed ANOVA',
                'error': 'No valid data after removing missing values',
                'satisfied': False
            }

        # Check for sufficient data
        n_subjects = data_clean[subject_id].nunique()
        if n_subjects < 3:
            return {
                'test': 'Mixed ANOVA',
                'error': f'Insufficient number of subjects (n = {n_subjects}). Need at least 3.',
                'satisfied': False
            }
        
        # Summary information
        design_info = {
            'between_factors': between_factors,
            'within_factors': within_factors,
            'n_subjects': n_subjects,
            'cells_per_subject': len(data_clean) / n_subjects,
            'balanced': True  # Will check this later
        }
        
        # Check if design is balanced
        counts = data_clean.groupby(subject_id).size()
        design_info['balanced'] = counts.std() == 0
        
        if not design_info['balanced']:
            warnings.warn("Unbalanced design detected. Some subjects have more observations than others.")
        
        # Run the mixed ANOVA
        # We'll use two approaches:
        # 1. statsmodels mixedlm for the full model including random effects (as in the original function)
        # 2. pingouin for direct computation of sphericity-corrected ANOVAs for RM factors
        
        # Method 1: statsmodels Mixed LM approach (original)
        try:
            # Construct formula.  Within-subject factors are crossed,
            # and between-subject factors are separate terms.
            within_part = ' * '.join([f'C({factor})' for factor in within_factors])
            between_part = ' + '.join([f'C({factor})' for factor in between_factors])

            if within_part and between_part:
                formula = f"{outcome} ~ {within_part} * {between_part}"
            elif within_part:
                formula = f"{outcome} ~ {within_part}"
            elif between_part:
                formula = f"{outcome} ~ {between_part}"
            else:
                formula = f"{outcome} ~ 1"  # Intercept-only model

            model = smf.mixedlm(formula, data_clean, groups=data_clean[subject_id])
            lmm_results = model.fit()

            # Identify relevant terms (between, within, and their interactions)
            between_terms = [f"C({factor})" for factor in between_factors]
            within_terms = [f"C({factor})" for factor in within_factors]
            interaction_terms = []

            # Between-subject interactions
            for i in range(len(between_factors)):
                for j in range(i + 1, len(between_factors)):
                    interaction_terms.append(f"C({between_factors[i]}):C({between_factors[j]})")

            # Within-subject interactions
            for i in range(len(within_factors)):
                for j in range(i + 1, len(within_factors)):
                    interaction_terms.append(f"C({within_factors[i]}):C({within_factors[j]})")
            if len(within_factors) > 2:  # Higher-order within interactions
                 interaction_terms.append(":".join([f"C({factor})" for factor in within_factors]))

            # Between-Within interactions
            for b_factor in between_factors:
                for w_factor in within_factors:
                    interaction_terms.append(f"C({b_factor}):C({w_factor})")
                # Add higher order interactions with the between factor
                if len(within_factors) > 1:
                    for i in range(len(within_factors)):
                        for j in range(i + 1, len(within_factors)):
                            interaction_terms.append(f"C({b_factor}):C({within_factors[i]}):C({within_factors[j]})")

            # Extract relevant p-values and test statistics
            lmm_stats = []
            
            for term in between_terms + within_terms + interaction_terms:
                for key, p_value in lmm_results.pvalues.items():
                    if term in key:  # Check if term is part of the key
                        coef = lmm_results.params.get(key, None)
                        se = lmm_results.bse.get(key, None)
                        t_stat = None if coef is None or se is None else coef / se
                        
                        # Clean up the term name for display
                        display_term = key.replace('C(', '').replace(')', '').replace(':', ' × ')
                        
                        lmm_stats.append({
                            'term': key,
                            'display_term': display_term,
                            'coefficient': float(coef) if coef is not None else None,
                            'std_error': float(se) if se is not None else None,
                            't_value': float(t_stat) if t_stat is not None else None,
                            'p_value': float(p_value),
                            'significant': p_value < alpha,
                            'term_type': ('between' if any(bf in key for bf in between_terms) else 
                                        'within' if any(wf in key for wf in within_terms) else
                                        'interaction')
                        })
            
            lmm_summary = {
                'method': 'Mixed Linear Model (statsmodels)',
                'formula': formula,
                'aic': float(lmm_results.aic),
                'bic': float(lmm_results.bic),
                'log_likelihood': float(lmm_results.llf),
                'term_results': lmm_stats
            }
        except Exception as e:
            lmm_summary = {
                'method': 'Mixed Linear Model (statsmodels)',
                'error': str(e)
            }
        
        # Method 2: pingouin approach (for proper sphericity correction on RM factors)
        try:
            # For pingouin, we need a different data structure - wide format for RM factors
            # This approach gives more accurate results for within-subjects factors 
            # by applying sphericity corrections
            
            # Initialize results storage
            pg_results = []
            
            # First, analyze main effects and interactions involving within-factors
            if within_factors:
                # Use pingouin's rm_anova function for repeated measures
                pg_aov = pg.rm_anova(
                    data=data_clean,
                    dv=outcome,
                    within=within_factors,
                    between=between_factors,
                    subject=subject_id,
                    detailed=True
                )
                
                # Process results
                for _, row in pg_aov.iterrows():
                    term = row['Source']
                    p_value = row['p-unc']  # Uncorrected p-value
                    p_gg = row.get('p-GG', None)  # Greenhouse-Geisser corrected p
                    p_hf = row.get('p-HF', None)  # Huynh-Feldt corrected p
                    
                    # Get effect size measures
                    partial_eta_sq = row.get('np2', None)  # Partial eta-squared
                    
                    # Determine sphericity
                    sphericity = True
                    if 'sphericity' in row:
                        sphericity = row['sphericity']
                    elif 'W-spher' in row and not pd.isna(row['W-spher']):
                        sphericity = row['W-spher'] > 0.7  # Rule of thumb
                    
                    # Use GG correction if sphericity violated
                    if not sphericity and p_gg is not None:
                        p_value = p_gg
                    
                    # Store results
                    pg_results.append({
                        'term': term,
                        'df': row['ddof1'],
                        'df_res': row['ddof2'],
                        'F': row['F'],
                        'p_value': p_value,
                        'p_gg': p_gg,
                        'p_hf': p_hf,
                        'significant': p_value < alpha,
                        'partial_eta_squared': partial_eta_sq,
                        'sphericity': sphericity
                    })
            
            # For between-subjects factors only (if no within-subjects factors used above)
            if between_factors and not within_factors:
                # Use pingouin's anova function for between-subjects
                pg_aov = pg.anova(
                    data=data_clean,
                    dv=outcome,
                    between=between_factors,
                    detailed=True
                )
                
                # Process results
                for _, row in pg_aov.iterrows():
                    term = row['Source']
                    p_value = row['p-unc']
                    
                    # Get effect size measures
                    partial_eta_sq = row.get('np2', None)  # Partial eta-squared
                    
                    # Store results
                    pg_results.append({
                        'term': term,
                        'df': row['DF'],
                        'df_res': row['DF2'] if 'DF2' in row else None,
                        'F': row['F'],
                        'p_value': p_value,
                        'significant': p_value < alpha,
                        'partial_eta_squared': partial_eta_sq,
                        'sphericity': True  # Not applicable for between-subjects
                    })
            
            pg_summary = {
                'method': 'ANOVA with Sphericity Correction (pingouin)',
                'term_results': pg_results
            }
        except Exception as e:
            pg_summary = {
                'method': 'ANOVA with Sphericity Correction (pingouin)',
                'error': str(e)
            }
        
        # Merge results from both methods, with preference to pingouin for RM factors
        anova_results = {}
        
        # Check if we have valid results from pingouin
        if 'term_results' in pg_summary and pg_summary['term_results']:
            # Use these as primary results
            for result in pg_summary['term_results']:
                anova_results[result['term']] = result
        
        # Fill in any missing results from lmm_summary
        if 'term_results' in lmm_summary and lmm_summary['term_results']:
            for result in lmm_summary['term_results']:
                term = result['term']
                # If term wasn't in pingouin results, add it
                if term not in anova_results:
                    cleaned_term = term.replace('C(', '').replace(')', '').replace(':', ' × ')
                    anova_results[cleaned_term] = {
                        'term': cleaned_term,
                        'p_value': result['p_value'],
                        'significant': result['significant'],
                        't_value': result.get('t_value', None),
                        'coefficient': result.get('coefficient', None),
                        'std_error': result.get('std_error', None),
                        'sphericity': True  # Assume true as it's from LMM
                    }
        
        # Post-hoc tests for significant effects
        posthoc_results = {}
        
        # Only do post-hoc for factors with more than 2 levels
        for factor in between_factors + within_factors:
            n_levels = data_clean[factor].nunique()
            
            if n_levels <= 2:
                continue  # Skip factors with only 2 levels
                
            # Check if this factor had a significant effect
            factor_term = f"C({factor})" if factor in between_factors else factor
            
            is_significant = False
            for term, result in anova_results.items():
                if term == factor or term == factor_term:
                    is_significant = result['significant']
                    break
            
            if not is_significant:
                continue  # Skip non-significant factors
            
            try:
                # Perform post-hoc test
                posthoc = pairwise_tukeyhsd(data_clean[outcome], data_clean[factor], alpha=alpha)
                
                # Extract results
                posthoc_data = []
                for i, (group1, group2, reject, _, _, _) in enumerate(zip(
                    posthoc.groupsunique[posthoc.pairindices[0]],
                    posthoc.groupsunique[posthoc.pairindices[1]],
                    posthoc.reject,
                    posthoc.meandiffs,
                    posthoc.confint,
                    posthoc.std_pairs
                )):
                    mean_diff = posthoc.meandiffs[i]
                    lower_ci = posthoc.confint[i, 0]
                    upper_ci = posthoc.confint[i, 1]
                    p_value = None  # Tukeyhsd doesn't provide exact p-values
                    
                    # Calculate Cohen's d for this comparison
                    group1_data = data_clean[data_clean[factor] == group1][outcome]
                    group2_data = data_clean[data_clean[factor] == group2][outcome]
                    
                    # Pooled standard deviation
                    n1, n2 = len(group1_data), len(group2_data)
                    s1, s2 = np.std(group1_data, ddof=1), np.std(group2_data, ddof=1)
                    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
                    
                    # Cohen's d
                    cohens_d = mean_diff / pooled_sd if pooled_sd != 0 else 0
                    
                    posthoc_data.append({
                        'group1': str(group1),
                        'group2': str(group2),
                        'mean_difference': float(mean_diff),
                        'ci_lower': float(lower_ci),
                        'ci_upper': float(upper_ci),
                        'p_adjust': None,  # Not available in statsmodels implementation
                        'significant': bool(reject),
                        'cohens_d': float(cohens_d)
                    })
                
                # Try to calculate exact p-values using pingouin
                try:
                    pg_posthoc = pg.pairwise_ttests(
                        data=data_clean, 
                        dv=outcome, 
                        between=factor if factor in between_factors else None,
                        within=factor if factor in within_factors else None,
                        subject=subject_id,
                        padjust='bonf'  # Bonferroni correction
                    )
                    
                    # Match p-values to our existing results
                    for i, row in pg_posthoc.iterrows():
                        a = str(row['A'])
                        b = str(row['B'])
                        p_adj = row['p-corr']

                        # Find corresponding entry in our posthoc_data
                        for entry in posthoc_data:
                            if ((entry['group1'] == a and entry['group2'] == b) or
                                (entry['group1'] == b and entry['group2'] == a)):  # Check both orderings
                                entry['p_adjust'] = float(p_adj)
                                # Update significance based on adjusted p-value
                                entry['significant'] = p_adj < alpha
                                break
                except Exception:
                    # If pingouin posthoc fails, stick with statsmodels results
                    pass
                
                posthoc_results[factor] = {
                    'method': 'Tukey HSD',
                    'comparisons': posthoc_data
                }
            except Exception as e:
                posthoc_results[factor] = {
                    'method': 'Tukey HSD',
                    'error': str(e)
                }
        
        # Test assumptions using standardized format
        assumptions = {}
        
        # 1. Normality tests - test for each cell in the design
        try:
            # Create design cells
            if between_factors and within_factors:
                design_cells = data_clean.groupby(between_factors + within_factors)
            elif between_factors:
                design_cells = data_clean.groupby(between_factors)
            elif within_factors:
                design_cells = data_clean.groupby(within_factors)
            else:
                design_cells = [('all', data_clean)]
            
            # Test normality in each cell
            for name, group in design_cells:
                # Skip cells with insufficient data
                if len(group) < 3:
                    continue
                
                # If name is not a tuple, convert it to one
                if not isinstance(name, tuple):
                    name = (name,)
                
                # Create a name for this cell
                if len(name) == 1:
                    cell_name = str(name[0])
                else:
                    cell_name = '_'.join(str(n) for n in name)
                
                # Use NormalityTest from format.py
                normality_key = f"normality_{cell_name}"
                statistic, p_value = stats.shapiro(group[outcome])
                
                assumptions[normality_key] = {
                    "result": AssumptionResult.PASSED if p_value >= 0.05 else AssumptionResult.FAILED,
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "details": f"Shapiro-Wilk test for cell {cell_name}: {'normal' if p_value >= 0.05 else 'non-normal'} distribution",
                    "warnings": [] if p_value >= 0.05 else [f"Cell {cell_name} violates normality assumption (p={p_value:.5f})"],
                    "figures": {}
                }
                
                # Create QQ plot for this cell
                fig, ax = plt.subplots(figsize=(8, 6))
                stats.probplot(group[outcome], plot=ax)
                ax.set_title(f'Q-Q Plot for {cell_name}')
                assumptions[normality_key]["figures"]["qq_plot"] = fig_to_svg(fig)
                plt.close(fig)
        except Exception as e:
            assumptions["normality_error"] = {
                "result": AssumptionResult.FAILED,
                "details": f"Error testing normality: {str(e)}",
                "warnings": [f"Normality test failed with error: {str(e)}"],
                "figures": {}
            }
        
        # 2. Homogeneity of variance for between-subjects factors
        if between_factors:
            try:
                # Group by between-subjects factors
                groups = []
                group_names = []
                
                for name, group in data_clean.groupby(between_factors):
                    # If name is not a tuple, convert it to one
                    if not isinstance(name, tuple):
                        name = (name,)
                    
                    # Create a name for this group
                    if len(name) == 1:
                        group_name = str(name[0])
                    else:
                        group_name = '_'.join(str(n) for n in name)
                    
                    groups.append(group[outcome])
                    group_names.append(group_name)
                
                # Test for homogeneity of variance if we have enough groups
                if len(groups) >= 2:
                # Levene's test
                    levene_stat, levene_p = stats.levene(*groups)
                    
                    assumptions["homogeneity"] = {
                        "result": AssumptionResult.PASSED if levene_p >= 0.05 else AssumptionResult.FAILED,
                        "statistic": float(levene_stat),
                        "p_value": float(levene_p),
                        "details": "Levene's test for homogeneity of variance",
                        "test_used": "Levene",
                        "group_variances": {name: float(np.var(group, ddof=1)) for name, group in zip(group_names, groups)},
                        "warnings": [] if levene_p >= 0.05 else ["Homogeneity of variance assumption is violated. Consider using Welch's ANOVA or robust methods."],
                        "figures": {}
                    }
                    
                    # Create variance comparison plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    variances = [np.var(g, ddof=1) for g in groups]
                    ax.bar(group_names, variances, color=PASTEL_COLORS)
                    ax.set_xlabel('Group')
                    ax.set_ylabel('Variance')
                    ax.set_title('Variance Comparison Between Groups')
                    
                    # Add text for Levene's test result
                    ax.text(0.5, 0.9, f"Levene's test p-value: {levene_p:.5f}", 
                           transform=ax.transAxes, ha='center',
                           bbox=dict(facecolor='white', alpha=0.5))
                    
                    assumptions["homogeneity"]["figures"]["variance_plot"] = fig_to_svg(fig)
                    plt.close(fig)
                else:
                    assumptions["homogeneity"] = {
                        "result": AssumptionResult.NOT_APPLICABLE,
                        "details": "Homogeneity of variance test requires at least 2 groups",
                        "warnings": ["Not enough groups to test homogeneity of variance"],
                        "figures": {}
                    }
            except Exception as e:
                assumptions["homogeneity_error"] = {
                    "result": AssumptionResult.FAILED,
                    "details": f"Error testing homogeneity of variance: {str(e)}",
                    "warnings": [f"Homogeneity test failed with error: {str(e)}"],
                    "figures": {}
                }
        
        # 3. Sphericity for within-subjects factors
        if within_factors and len(within_factors) >= 2:
            try:
                # Use pingouin to test sphericity
                import pingouin as pg
                
                # Test sphericity for each within-subject factor with >= 3 levels
                for factor in within_factors:
                    if data_clean[factor].nunique() >= 3:
                        # Reshape data to wide format
                        wide_data = data_clean.pivot_table(
                            index=subject_id, 
                            columns=factor, 
                            values=outcome
                        )
                        
                        # Run Mauchly's test
                        spher_result = pg.sphericity(wide_data.values)
                        w_value = spher_result[0]
                        chi2_value = spher_result[1]
                        dof = spher_result[2]
                        p_value = spher_result[3]
                        
                        assumptions[f"sphericity_{factor}"] = {
                            "result": AssumptionResult.PASSED if p_value >= 0.05 else AssumptionResult.FAILED,
                            "statistic": float(w_value),
                            "p_value": float(p_value),
                            "details": f"Mauchly's test of sphericity for factor {factor}",
                            "test_used": "Mauchly",
                            "warnings": [] if p_value >= 0.05 else [f"Sphericity assumption violated for factor {factor}. Use Greenhouse-Geisser or Huynh-Feldt correction."],
                            "figures": {}
                    }
            except Exception as e:
                assumptions["sphericity_error"] = {
                    "result": AssumptionResult.FAILED,
                    "details": f"Error testing sphericity: {str(e)}",
                    "warnings": [f"Sphericity test failed with error: {str(e)}"],
                    "figures": {}
                }
        
        # 4. Sample size adequacy for each group
        try:
            # Determine cell counts
            if between_factors:
                cell_counts = data_clean.groupby(between_factors).size()
            else:
                cell_counts = pd.Series({'overall': len(data_clean)})
            
            # Check sample size for each cell/group
            for name, count in cell_counts.items():
                # Convert name to string if it's a tuple
                if isinstance(name, tuple):
                    group_name = '_'.join(str(n) for n in name)
                else:
                    group_name = str(name)
                
                min_recommended = 15  # Rule of thumb for parametric tests
                
                assumptions[f"sample_size_{group_name}"] = {
                    "result": AssumptionResult.PASSED if count >= min_recommended else 
                             AssumptionResult.WARNING if count >= 5 else 
                             AssumptionResult.FAILED,
                    "details": f"Sample size check for group {group_name}: n={count}",
                    "sample_size": int(count),
                    "minimum_required": min_recommended,
                    "power": None,  # Would need effect size to calculate
                    "warnings": [] if count >= min_recommended else [f"Sample size for group {group_name} (n={count}) is below recommended minimum ({min_recommended})"]
            }
        except Exception as e:
            assumptions["sample_size_error"] = {
                "result": AssumptionResult.FAILED,
                "details": f"Error checking sample size: {str(e)}",
                "warnings": [f"Sample size check failed with error: {str(e)}"]
            }
        
        # 5. Outliers check for each group
        try:
            if between_factors:
                # Check outliers within each between-subjects group
                for name, group in data_clean.groupby(between_factors):
                    # Convert name to string if it's a tuple
                    if isinstance(name, tuple):
                        group_name = '_'.join(str(n) for n in name)
                    else:
                        group_name = str(name)
                    
            # Z-score method for outlier detection
                    z_scores = np.abs(stats.zscore(group[outcome]))
                    outlier_indices = np.where(z_scores > 3)[0]
                    
                    assumptions[f"outliers_{group_name}"] = {
                        "result": AssumptionResult.PASSED if len(outlier_indices) == 0 else AssumptionResult.WARNING,
                        "outliers": outlier_indices.tolist()[:20] if len(outlier_indices) > 0 else [],
                        "details": f"Outlier check for group {group_name} using z-score > 3",
                        "test_used": "Z-score",
                        "warnings": [] if len(outlier_indices) == 0 else [f"Found {len(outlier_indices)} outliers in group {group_name}"],
                        "figures": {}
                    }
                    
                    # Create boxplot for outlier visualization
                    if len(group) >= 5:  # Only create if we have enough data
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.boxplot(y=group[outcome], ax=ax, color=PASTEL_COLORS[0])
                        ax.set_title(f'Boxplot for {group_name}')
                        assumptions[f"outliers_{group_name}"]["figures"]["boxplot"] = fig_to_svg(fig)
                        plt.close(fig)
            else:
                # Check outliers in the entire dataset
                z_scores = np.abs(stats.zscore(data_clean[outcome]))
                outlier_indices = np.where(z_scores > 3)[0]
                
                assumptions["outliers"] = {
                    "result": AssumptionResult.PASSED if len(outlier_indices) == 0 else AssumptionResult.WARNING,
                    "outliers": outlier_indices.tolist()[:20] if len(outlier_indices) > 0 else [],
                    "details": "Outlier check using z-score > 3",
                    "test_used": "Z-score",
                    "warnings": [] if len(outlier_indices) == 0 else [f"Found {len(outlier_indices)} outliers in the data"],
                    "figures": {}
                }
                
                # Create boxplot for outlier visualization
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.boxplot(y=data_clean[outcome], ax=ax, color=PASTEL_COLORS[0])
                ax.set_title('Boxplot for Outlier Detection')
                assumptions["outliers"]["figures"]["boxplot"] = fig_to_svg(fig)
                plt.close(fig)
        except Exception as e:
            assumptions["outliers_error"] = {
                "result": AssumptionResult.FAILED,
                "details": f"Error checking outliers: {str(e)}",
                "warnings": [f"Outlier check failed with error: {str(e)}"],
                "figures": {}
            }
        
        # 6. Independence check (for within-subjects design)
        # This is often assumed rather than tested, but we can check for autocorrelation in residuals
        if 'lmm_summary' in locals() and 'term_results' in lmm_summary:
            try:
                residuals = data_clean[outcome] - lmm_results.fittedvalues
                
                # Durbin-Watson test
                dw_stat = sm.stats.durbin_watson(residuals)
                
                assumptions["independence"] = {
                    "result": AssumptionResult.PASSED if 1.5 < dw_stat < 2.5 else AssumptionResult.WARNING,
                    "message": "Durbin-Watson test for independence of residuals",
                    "statistic": float(dw_stat),
                    "details": {
                        "durbin_watson": float(dw_stat),
                        "interpretation": "Value close to 2 indicates no autocorrelation. Values <1.5 or >2.5 may indicate autocorrelation."
                    }
                }
                
                # Create autocorrelation plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sm.graphics.tsa.plot_acf(residuals, lags=min(20, len(residuals)//5), ax=ax)
                ax.set_title('Autocorrelation of Residuals')
                assumptions["independence"]["figures"] = {"autocorrelation": fig_to_svg(fig)}
                plt.close(fig)
            except Exception as e:
                assumptions["independence_error"] = {
                    "result": AssumptionResult.FAILED,
                    "message": f"Error testing independence: {str(e)}",
                    "details": {"error": str(e)}
                }
        
        # Calculate descriptive statistics per cell
        descriptive_stats = []
        
        try:
            # Determine grouping variables based on the design
            if between_factors and within_factors:
                grouping = between_factors + within_factors
            elif between_factors:
                grouping = between_factors
            elif within_factors:
                grouping = within_factors
            else:
                # No factors - just overall statistics
                grouping = None
            
            if grouping:
                # Group statistics
                for name, group in data_clean.groupby(grouping):
                    # Convert name to string representation
                    if not isinstance(name, tuple):
                        name = (name,)
                    
                    # Construct display name
                    display_name = ' / '.join(str(part) for part in name)
                    
                    descriptive_stats.append({
                        'cell': display_name,
                        'n': len(group),
                        'mean': float(group[outcome].mean()),
                        'std': float(group[outcome].std()),
                        'se': float(group[outcome].std() / np.sqrt(len(group))),
                        'ci95_lower': float(group[outcome].mean() - 1.96 * group[outcome].std() / np.sqrt(len(group))),
                        'ci95_upper': float(group[outcome].mean() + 1.96 * group[outcome].std() / np.sqrt(len(group))),
                        'min': float(group[outcome].min()),
                        'max': float(group[outcome].max()),
                        'median': float(group[outcome].median()),
                        'skewness': float(stats.skew(group[outcome])),
                        'kurtosis': float(stats.kurtosis(group[outcome]))
                    })
            else:
                # Overall statistics
                descriptive_stats.append({
                    'cell': 'Overall',
                    'n': len(data_clean),
                    'mean': float(data_clean[outcome].mean()),
                    'std': float(data_clean[outcome].std()),
                    'se': float(data_clean[outcome].std() / np.sqrt(len(data_clean))),
                    'ci95_lower': float(data_clean[outcome].mean() - 1.96 * data_clean[outcome].std() / np.sqrt(len(data_clean))),
                    'ci95_upper': float(data_clean[outcome].mean() + 1.96 * data_clean[outcome].std() / np.sqrt(len(data_clean))),
                    'min': float(data_clean[outcome].min()),
                    'max': float(data_clean[outcome].max()),
                    'median': float(data_clean[outcome].median()),
                    'skewness': float(stats.skew(data_clean[outcome])),
                    'kurtosis': float(stats.kurtosis(data_clean[outcome]))
                })
        except Exception as e:
            descriptive_stats = [{
                'error': f'Could not calculate descriptive statistics: {str(e)}'
            }]
        
        # Create interpretation
        interpretation = f"Mixed ANOVA Analysis\n\n"
        
        # Study design
        interpretation += f"Study Design:\n"
        if between_factors:
            interpretation += f"- Between-subjects factors: {', '.join(between_factors)}\n"
        if within_factors:
            interpretation += f"- Within-subjects factors: {', '.join(within_factors)}\n"
        interpretation += f"- Number of subjects: {n_subjects}\n"
        interpretation += f"- Outcome variable: {outcome}\n\n"
        
        # ANOVA results
        interpretation += f"ANOVA Results:\n"
        
        # Sort effects: main effects first, then interactions
        main_effects = []
        interaction_effects = []
        
        for term, result in anova_results.items():
            term_str = f"- {term}: "
            
            # Add F-statistic and df if available
            if 'F' in result and 'df' in result and 'df_res' in result:
                term_str += f"F({result['df']:.0f}, {result['df_res']:.0f}) = {result['F']:.3f}, "
            
            # Add p-value
            term_str += f"p = {result['p_value']:.5f}"
            
            # Add significance indicator
            if result['significant']:
                term_str += " *"
            
            # Add effect size if available
            if 'partial_eta_squared' in result and result['partial_eta_squared'] is not None:
                eta_sq = result['partial_eta_squared']
                term_str += f", ηp² = {eta_sq:.3f}"
                
                # Add effect size interpretation
                if eta_sq < 0.01:
                    term_str += " (negligible effect)"
                elif eta_sq < 0.06:
                    term_str += " (small effect)"
                elif eta_sq < 0.14:
                    term_str += " (medium effect)"
                else:
                    term_str += " (large effect)"
            
            # Add sphericity correction note if applicable
            if 'sphericity' in result and not result['sphericity']:
                term_str += " (Greenhouse-Geisser corrected)"
            
            # Categorize as main effect or interaction
            if ':' in term or '×' in term:
                interaction_effects.append(term_str)
            else:
                main_effects.append(term_str)
        
        # Display main effects first, then interactions
        if main_effects:
            interpretation += "Main Effects:\n"
            interpretation += '\n'.join(main_effects) + '\n\n'
            
        if interaction_effects:
            interpretation += "Interaction Effects:\n"
            interpretation += '\n'.join(interaction_effects) + '\n\n'
        
        # Post-hoc results
        if posthoc_results:
            interpretation += f"Post-hoc Tests (for significant effects):\n"
            
            for factor, result in posthoc_results.items():
                interpretation += f"Factor: {factor}\n"
                
                if 'error' in result:
                    interpretation += f"  Error: {result['error']}\n"
                    continue
                
                sig_comparisons = [c for c in result['comparisons'] if c['significant']]
                if sig_comparisons:
                    interpretation += f"  Significant pairwise differences:\n"
                    
                    for comp in sig_comparisons:
                        p_str = f", p = {comp['p_adjust']:.5f}" if comp['p_adjust'] is not None else ""
                        d_str = f", d = {comp['cohens_d']:.2f}" if 'cohens_d' in comp else ""
                        
                        interpretation += f"  - {comp['group1']} vs. {comp['group2']}: mean diff = {comp['mean_difference']:.3f}"
                        interpretation += f", 95% CI [{comp['ci_lower']:.3f}, {comp['ci_upper']:.3f}]{p_str}{d_str}\n"
                else:
                    interpretation += f"  No significant pairwise differences found.\n"
                
                interpretation += "\n"
        
        # Assumption checks
        interpretation += f"Assumption Checks:\n"
        
        # Go through each assumption by type
        for key, value in assumptions.items():
            if "_error" in key:
                continue  # Skip error entries
                
            if "result" in value:
                # Only add if it has a result
                if "normality" in key:
                    interpretation += f"- {key}: {value['result'].value}, p-value: {value.get('p_value', 'N/A')}\n"
                elif "homogeneity" in key:
                    interpretation += f"- {key}: {value['result'].value}, p-value: {value.get('p_value', 'N/A')}\n"
                elif "sphericity" in key:
                    interpretation += f"- {key}: {value['result'].value}, p-value: {value.get('p_value', 'N/A')}\n"
                elif "sample_size" in key:
                    interpretation += f"- {key}: {value['result'].value}, n: {value.get('sample_size', 'N/A')}\n"
                elif "outliers" in key:
                    interpretation += f"- {key}: {value['result'].value}, outliers found: {len(value.get('outliers', []))}\n"
                elif "independence" in key:
                    interpretation += f"- {key}: {value['result'].value}, Durbin-Watson: {value.get('statistic', 'N/A')}\n"
        
        # Overall conclusion
        interpretation += f"\nConclusion:\n"
        
        # Count significant effects
        sig_main_effects = [term for term, result in anova_results.items() 
                         if result['significant'] and (':' not in term and '×' not in term)]
        
        sig_interactions = [term for term, result in anova_results.items() 
                         if result['significant'] and (':' in term or '×' in term)]
        
        if sig_main_effects or sig_interactions:
            if sig_main_effects:
                interpretation += f"The analysis found significant main effects for: {', '.join(sig_main_effects)}. "
            
            if sig_interactions:
                interpretation += f"Significant interactions were found for: {', '.join(sig_interactions)}. "
                interpretation += "When interactions are significant, main effects should be interpreted with caution. "
            
            if posthoc_results:
                interpretation += f"Post-hoc tests revealed specific group differences as detailed above. "
        else:
            interpretation += f"The analysis did not find any significant main effects or interactions. "
            
        # Add caution about assumptions if needed
        assumption_issues = []
        
        for name, result in assumptions.items():
            if 'overall_satisfied' in result and not result['overall_satisfied']:
                assumption_issues.append(name)
        
        if assumption_issues:
            interpretation += f"\nCaution: The following assumptions were violated: {', '.join(assumption_issues)}. "
            interpretation += "Results should be interpreted with caution. Consider alternative analyses or robust methods."
        
        # Create visualizations
        figures = {}
        
        # Figure 1: Main Effects Plot
        try:
            # For each main effect, create a plot showing means by factor
            for factor in between_factors + within_factors:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Group data by factor
                means = data_clean.groupby(factor)[outcome].mean()
                std_errs = data_clean.groupby(factor)[outcome].sem()
                
                # Create bar plot
                x = np.arange(len(means))
                bars = ax.bar(x, means, yerr=std_errs, capsize=10, alpha=0.7, color=PASTEL_COLORS[0])
                
                # Add value labels on top of bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + std_errs.iloc[i],
                           f'{height:.2f}', ha='center', va='bottom')
                
                # Add labels and title
                ax.set_xlabel(factor)
                ax.set_ylabel(outcome)
                ax.set_title(f'Main Effect of {factor}')
                
                # Set x-tick labels
                ax.set_xticks(x)
                ax.set_xticklabels(means.index)
                
                # Add significance indicator
                is_significant = False
                p_value = None
                for term, result in anova_results.items():
                    if term == factor or term == f"C({factor})":
                        is_significant = result['significant']
                        p_value = result['p_value']
                        break
                
                if is_significant and p_value is not None:
                    ax.annotate(f"Significant effect (p = {p_value:.4f})", 
                              xy=(0.5, 0.95), xycoords='axes fraction',
                              ha='center', va='top',
                              bbox=dict(boxstyle='round', fc='lightgreen', alpha=0.3))
                elif p_value is not None:
                    ax.annotate(f"Non-significant effect (p = {p_value:.4f})", 
                              xy=(0.5, 0.95), xycoords='axes fraction',
                              ha='center', va='top',
                              bbox=dict(boxstyle='round', fc='lightgray', alpha=0.3))
                
                fig.tight_layout()
                figures[f'main_effect_{factor}'] = fig_to_svg(fig)
        except Exception as e:
            figures['main_effects_error'] = str(e)
        
        # Figure 2: Interaction Plot (for 2-way interactions)
        try:
            # Check for 2-way interactions
            if len(between_factors + within_factors) >= 2:
                # Plot interactions between each pair of factors
                for i, factor1 in enumerate(between_factors + within_factors):
                    for factor2 in (between_factors + within_factors)[i+1:]:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Calculate means for the interaction
                        interaction_means = data_clean.groupby([factor1, factor2])[outcome].mean().unstack()
                        
                        # Create line plot
                        interaction_means.plot(marker='o', ax=ax)
                        
                        # Add labels and title
                        ax.set_xlabel(factor1)
                        ax.set_ylabel(outcome)
                        ax.set_title(f'Interaction between {factor1} and {factor2}')
                        
                        # Add legend
                        ax.legend(title=factor2)
                        
                        # Add grid
                        ax.grid(True, linestyle='--', alpha=0.3)
                        
                        # Add significance indicator
                        is_significant = False
                        p_value = None
                        for term, result in anova_results.items():
                            if ((f"{factor1}:{factor2}" in term) or 
                                (f"{factor2}:{factor1}" in term) or
                                (f"{factor1} × {factor2}" in term) or
                                (f"{factor2} × {factor1}" in term)):
                                is_significant = result['significant']
                                p_value = result['p_value']
                                break
                        
                        if is_significant and p_value is not None:
                            ax.annotate(f"Significant interaction (p = {p_value:.4f})", 
                                      xy=(0.5, 0.05), xycoords='axes fraction',
                                      ha='center', va='bottom',
                                      bbox=dict(boxstyle='round', fc='lightgreen', alpha=0.3))
                        elif p_value is not None:
                            ax.annotate(f"Non-significant interaction (p = {p_value:.4f})", 
                                      xy=(0.5, 0.05), xycoords='axes fraction',
                                      ha='center', va='bottom',
                                      bbox=dict(boxstyle='round', fc='lightgray', alpha=0.3))
                        
                        fig.tight_layout()
                        figures[f'interaction_{factor1}_{factor2}'] = fig_to_svg(fig)
        except Exception as e:
            figures['interaction_plot_error'] = str(e)
        
        # Figure 3: Boxplots for between-subjects factors
        try:
            if between_factors:
                for factor in between_factors:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Create boxplot
                    sns.boxplot(x=factor, y=outcome, data=data_clean, ax=ax, palette=PASTEL_CMAP)
                    
                    # Add individual data points with jitter
                    sns.stripplot(x=factor, y=outcome, data=data_clean, 
                                color='black', alpha=0.4, jitter=True, size=4, ax=ax)
                    
                    # Add labels and title
                    ax.set_xlabel(factor)
                    ax.set_ylabel(outcome)
                    ax.set_title(f'Distribution of {outcome} by {factor}')
                    
                    fig.tight_layout()
                    figures[f'boxplot_{factor}'] = fig_to_svg(fig)
        except Exception as e:
            figures['boxplot_error'] = str(e)
        
        # Figure 4: Within-subjects profile plots
        try:
            if within_factors:
                for factor in within_factors:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Reshape data for profile plot
                    # We'll plot individual subject profiles and the mean
                    
                    # Group data by subject and factor
                    subject_profiles = data_clean.pivot_table(
                        index=subject_id, 
                        columns=factor, 
                        values=outcome
                    )
                    
                    # Plot individual subject profiles with transparency
                    for idx, row in subject_profiles.iterrows():
                        ax.plot(row.index, row.values, 'o-', alpha=0.2, color=PASTEL_COLORS[idx % len(PASTEL_COLORS)])
                    
                    # Calculate and plot the mean profile
                    mean_profile = subject_profiles.mean()
                    se_profile = subject_profiles.sem()
                    
                    ax.errorbar(mean_profile.index, mean_profile.values, 
                               yerr=se_profile.values, fmt='o-', color=PASTEL_COLORS[0], 
                               linewidth=2, capsize=6, label='Mean')
                    
                    # Add labels and title
                    ax.set_xlabel(factor)
                    ax.set_ylabel(outcome)
                    ax.set_title(f'Within-subjects Profile Plot for {factor}')
                    
                    # Add grid
                    ax.grid(True, linestyle='--', alpha=0.3)
                    
                    # Add legend
                    ax.legend()
                    
                    fig.tight_layout()
                    figures[f'profile_{factor}'] = fig_to_svg(fig)
        except Exception as e:
            figures['profile_plot_error'] = str(e)
        
        # Figure 5: Assumption checks
        try:
            # Create a 2x2 grid of assumption check plots
            fig, axs = plt.subplots(2, 2, figsize=(14, 12))
            
            # 1. QQ plot of residuals (top left)
            # Calculate residuals (observed - predicted)
            if 'lmm_summary' in locals() and 'term_results' in lmm_summary:
                residuals = data_clean[outcome] - lmm_results.fittedvalues
                
                # Create QQ plot
                sorted_residuals = np.sort(residuals)
                n = len(sorted_residuals)
                quantiles = np.arange(1, n + 1) / (n + 1)
                theoretical_quantiles = stats.norm.ppf(quantiles)
                
                axs[0, 0].scatter(theoretical_quantiles, sorted_residuals, alpha=0.6)
                
                # Add reference line
                min_val = min(np.min(theoretical_quantiles), np.min(sorted_residuals))
                max_val = max(np.max(theoretical_quantiles), np.max(sorted_residuals))
                axs[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
                
                # Add title and labels
                axs[0, 0].set_title('Normal Q-Q Plot of Residuals')
                axs[0, 0].set_xlabel('Theoretical Quantiles')
                axs[0, 0].set_ylabel('Sample Quantiles')
                
                # Add normality test result if available
                if 'normality' in assumptions and 'overall_satisfied' in assumptions['normality']:
                    satisfied = assumptions['normality']['overall_satisfied']
                    axs[0, 0].text(0.05, 0.95, 
                                f"Normality: {'Satisfied' if satisfied else 'Violated'}", 
                                transform=axs[0, 0].transAxes,
                                va='top', ha='left',
                                bbox=dict(boxstyle='round', fc='white' if satisfied else 'pink', alpha=0.8))
            else:
                axs[0, 0].text(0.5, 0.5, "Residuals not available", 
                              ha='center', va='center',
                              transform=axs[0, 0].transAxes)
            
            # 2. Homogeneity of variance (top right)
            if between_factors:
                # Create a boxplot of residuals by between-subjects groups
                if 'lmm_summary' in locals() and 'term_results' in lmm_summary:
                    # Create a dataframe with residuals and group info
                    resid_df = pd.DataFrame({
                        'residuals': residuals,
                        'group': data_clean[between_factors[0]] if len(between_factors) == 1 else 
                                '_'.join(data_clean[f].astype(str) for f in between_factors)
                    })
                    
                    # Create boxplot
                    sns.boxplot(x='group', y='residuals', data=resid_df, ax=axs[0, 1])
                    
                    # Add reference line at 0
                    axs[0, 1].axhline(y=0, color='r', linestyle='--')
                    
                    # Add title and labels
                    axs[0, 1].set_title('Residuals by Group')
                    axs[0, 1].set_xlabel('Group')
                    axs[0, 1].set_ylabel('Residuals')
                    
                    # Add homogeneity test result if available
                    if ('homogeneity_of_variance' in assumptions and 
                        'overall_satisfied' in assumptions['homogeneity_of_variance']):
                        satisfied = assumptions['homogeneity_of_variance']['overall_satisfied']
                        axs[0, 1].text(0.05, 0.95, 
                                     f"Homogeneity: {'Satisfied' if satisfied else 'Violated'}", 
                                     transform=axs[0, 1].transAxes,
                                     va='top', ha='left',
                                     bbox=dict(boxstyle='round', fc='white' if satisfied else 'pink', alpha=0.8))
                else:
                    axs[0, 1].text(0.5, 0.5, "Residuals not available", 
                                  ha='center', va='center',
                                  transform=axs[0, 1].transAxes)
            else:
                axs[0, 1].text(0.5, 0.5, "No between-subjects factors", 
                              ha='center', va='center',
                              transform=axs[0, 1].transAxes)
            
            # 3. Outliers (bottom left)
            # Create a boxplot of standardized residuals
            if 'lmm_summary' in locals() and 'term_results' in lmm_summary:
                # Standardize residuals
                std_residuals = residuals / np.std(residuals)
                
                # Create histogram with KDE
                sns.histplot(std_residuals, kde=True, ax=axs[1, 0])
                
                # Add reference lines for outlier thresholds
                axs[1, 0].axvline(x=-3, color='r', linestyle='--')
                axs[1, 0].axvline(x=3, color='r', linestyle='--')
                
                # Add title and labels
                axs[1, 0].set_title('Distribution of Standardized Residuals')
                axs[1, 0].set_xlabel('Standardized Residuals')
                axs[1, 0].set_ylabel('Frequency')
                
                # Add outlier test result if available
                if 'outliers' in assumptions and 'satisfied' in assumptions['outliers']:
                    satisfied = assumptions['outliers']['satisfied']
                    axs[1, 0].text(0.05, 0.95, 
                                  f"Outliers: {'None detected' if satisfied else 'Present'}", 
                                  transform=axs[1, 0].transAxes,
                                  va='top', ha='left',
                                  bbox=dict(boxstyle='round', fc='white' if satisfied else 'pink', alpha=0.8))
            else:
                axs[1, 0].text(0.5, 0.5, "Residuals not available", 
                              ha='center', va='center',
                              transform=axs[1, 0].transAxes)
            
            # 4. Sample Size (bottom right)
            # Create a bar chart of sample sizes per cell
            axs[1, 1].set_title('Sample Size per Cell')
            
            if grouping:
                # Count observations per cell
                cell_counts = data_clean.groupby(grouping).size()

                # If too many cells, aggregate
                if len(cell_counts) > 10:
                    if between_factors:
                        grouping = between_factors + [subject_id]
                    elif within_factors:
                        grouping = [subject_id] + within_factors
                    else:
                        grouping = [subject_id]
                    cell_counts = data_clean.groupby(grouping).size()

                # Create bar chart
                cell_counts.plot.bar(ax=axs[1, 1])

                # Add reference line for minimum recommended sample size
                axs[1, 1].axhline(y=15, color='r', linestyle='--', label='Recommended Min')

                # Add title and labels
                axs[1, 1].set_ylabel('Count')
                axs[1, 1].legend()

                # Rotate x-tick labels if needed
                if len(cell_counts) > 5:
                    plt.xticks(rotation=45, ha='right')

                # Add sample size assessment if available
                if 'sample_size' in assumptions and 'adequate' in assumptions['sample_size']:
                    satisfied = assumptions['sample_size']['adequate']
                    axs[1, 1].text(0.05, 0.95,
                                  f"Sample Size: {'Adequate' if satisfied else 'Small'}",
                                  transform=axs[1, 1].transAxes,
                                  va='top', ha='left',
                                  bbox=dict(boxstyle='round', fc='white' if satisfied else 'pink', alpha=0.8))
            else:
                axs[1, 1].text(0.5, 0.5, "Grouping information not available",
                              ha='center', va='center',
                              transform=axs[1, 1].transAxes)
            
            # Overall title
            fig.suptitle('Assumption Checks', fontsize=16)
            
            fig.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
            figures['assumption_checks'] = fig_to_svg(fig)
        except Exception as e:
            figures['assumption_checks_error'] = str(e)
        
        # Figure 6: Post-hoc test results visualization
        try:
            if posthoc_results:
                for factor, result in posthoc_results.items():
                    if 'error' in result or not result['comparisons']:
                        continue
                    
                    # Create heatmap of pairwise comparisons
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Extract unique groups
                    groups = list(set([c['group1'] for c in result['comparisons']] + 
                                   [c['group2'] for c in result['comparisons']]))
                    
                    # Create a matrix of p-values
                    n_groups = len(groups)
                    p_matrix = np.ones((n_groups, n_groups))
                    sig_matrix = np.zeros((n_groups, n_groups), dtype=bool)
                    
                    for comp in result['comparisons']:
                        i = groups.index(comp['group1'])
                        j = groups.index(comp['group2'])
                        
                        # If p-value is available, use it
                        if comp['p_adjust'] is not None:
                            p_matrix[i, j] = comp['p_adjust']
                            p_matrix[j, i] = comp['p_adjust']  # Matrix is symmetric
                        
                        # Mark significant comparisons
                        sig_matrix[i, j] = comp['significant']
                        sig_matrix[j, i] = comp['significant']
                    
                    # Set diagonal to 1 (no comparison with self)
                    np.fill_diagonal(p_matrix, 1)
                    
                    # Create heatmap
                    cmap = plt.cm.YlOrRd_r  # Reversed YlOrRd (yellow=high, red=low)
                    heatmap = ax.imshow(p_matrix, cmap=cmap, vmin=0, vmax=1)
                    
                    # Add colorbar
                    cbar = plt.colorbar(heatmap, ax=ax, label='p-value')
                    
                    # Add grid
                    ax.set_xticks(np.arange(n_groups))
                    ax.set_yticks(np.arange(n_groups))
                    ax.set_xticklabels(groups)
                    ax.set_yticklabels(groups)
                    
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                    
                    # Add text annotations
                    for i in range(n_groups):
                        for j in range(n_groups):
                            if i != j:
                                # Show p-value with significance indicator
                                text = ax.text(j, i, f"{p_matrix[i, j]:.3f}" + 
                                            ('*' if sig_matrix[i, j] else ''),
                                            ha="center", va="center", 
                                            color="white" if p_matrix[i, j] < 0.5 else "black")
                    
                    # Add title
                    ax.set_title(f'Post-hoc Tests for {factor}')
                    
                    fig.tight_layout()
                    figures[f'posthoc_{factor}'] = fig_to_svg(fig)
        except Exception as e:
            figures['posthoc_plot_error'] = str(e)
        
        # Figure 7: Summary of Significant Effects
        try:
            # Create a table of significant effects
            sig_effects = [term for term, result in anova_results.items() if result['significant']]
            
            if sig_effects:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Hide axes
                ax.axis('off')
                
                # Create a table with effect information
                table_data = []
                table_colors = []
                
                # Headers
                headers = ['Effect', 'F-value', 'df', 'p-value', 'Effect Size']
                table_data.append(headers)
                table_colors.append(['#f2f2f2'] * len(headers))  # Header row background
                
                for term in sig_effects:
                    result = anova_results[term]
                    
                    # Extract values
                    f_value = result.get('F', '-')
                    df1 = result.get('df', '-')
                    df2 = result.get('df_res', '-')
                    p_value = result['p_value']
                    effect_size = result.get('partial_eta_squared', '-')
                    
                    # Format values
                    if isinstance(f_value, (int, float)):
                        f_value = f"{f_value:.3f}"
                    
                    df_str = f"{df1:.0f}, {df2:.0f}" if (isinstance(df1, (int, float)) and 
                                                     isinstance(df2, (int, float))) else '-'
                    
                    p_str = f"{p_value:.5f}"
                    
                    if isinstance(effect_size, (int, float)):
                        effect_size = f"{effect_size:.3f}"
                    
                    row = [term, f_value, df_str, p_str, effect_size]
                    table_data.append(row)
                    
                    # Set row color based on p-value
                    if p_value < 0.001:
                        row_color = '#d4edda'  # Very significant (green)
                    elif p_value < 0.01:
                        row_color = '#e6f3e6'  # Significant (light green)
                    else:
                        row_color = '#f8f9fa'  # Less significant (light gray)
                    
                    table_colors.append([row_color] * len(headers))
                
                # Create table
                table = ax.table(
                    cellText=table_data,
                    cellColours=table_colors,
                    loc='center',
                    cellLoc='center'
                )
                
                # Adjust table properties
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.5)  # Adjust table scaling
                
                # Add title
                ax.set_title('Summary of Significant Effects', fontsize=14, pad=20)
                
                fig.tight_layout()
                figures['significant_effects_summary'] = fig_to_svg(fig)
            else:
                # Create a simple figure stating no significant effects
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Hide axes
                ax.axis('off')
                
                # Add text
                ax.text(0.5, 0.5, "No significant effects were found", 
                      ha='center', va='center', fontsize=14)
                
                # Add title
                ax.set_title('Summary of Significant Effects', fontsize=14, pad=20)
                
                fig.tight_layout()
                figures['significant_effects_summary'] = fig_to_svg(fig)
        except Exception as e:
            figures['significant_effects_summary_error'] = str(e)
        
        # Compile full results
        return {
            'test': 'Mixed ANOVA',
            'design': design_info,
            'anova_results': anova_results,
            'descriptive_statistics': descriptive_stats,
            'posthoc_tests': posthoc_results,
            'assumptions': assumptions,
            'interpretation': interpretation,
            'figures': figures,
            'satisfied': True
        }
    except Exception as e:
        return {
            'test': 'Mixed ANOVA',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'satisfied': False
        }