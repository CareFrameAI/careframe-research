import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# ---------------------------
# STUDY DESIGN DATASETS
# ---------------------------

# 1. BETWEEN_SUBJECTS: Required: outcome_SysBP6Weeks, group; Optional: subject_id, covariates_age
n = 20
df_between_subjects = pd.DataFrame({
    'subject_id': np.arange(1, n + 1),
    'group': np.random.choice(['control', 'intervention'], size=n),
    'outcome_SysBP6Weeks': np.random.normal(loc=120, scale=15, size=n),
    'covariates_age': np.random.randint(20, 70, size=n)
})
# Add effect: for group 'intervention', add a positive effect (e.g., +10 mmHg)
df_between_subjects.loc[df_between_subjects['group'] == 'intervention', 'outcome_SysBP6Weeks'] += 10

# 2. WITHIN_SUBJECTS: (no group variable so no renaming)
n_subjects = 10
time_points = [1, 2]
rows = []
for subj in range(1, n_subjects + 1):
    for t in time_points:
        rows.append({
            'subject_id': subj,
            'time': t,
            'outcome_SysBP6Weeks': np.random.normal(loc=120, scale=15),
            'pair_id': subj,  # using subject id as pair_id
            'covariates_age': np.random.randint(20, 70)
        })
df_within_subjects = pd.DataFrame(rows)

# 3. MIXED: Required: outcome_SysBP6Weeks, group, subject_id, time; Optional: pair_id, covariates_age
n_subjects = 15
time_points = [1, 2]
rows = []
for subj in range(1, n_subjects + 1):
    grp = np.random.choice(['control', 'intervention'])
    pair_id = subj if np.random.rand() > 0.5 else None
    for t in time_points:
        rows.append({
            'subject_id': subj,
            'group': grp,
            'time': t,
            'outcome_SysBP6Weeks': np.random.normal(loc=120, scale=15),
            'pair_id': pair_id,
            'covariates_age': np.random.randint(20, 70)
        })
df_mixed = pd.DataFrame(rows)
# Add effect for group 'intervention'
df_mixed.loc[df_mixed['group'] == 'intervention', 'outcome_SysBP6Weeks'] += 10

# 4. SINGLE_SUBJECT: (no group variable)
n_time_points = 10
df_single_subject = pd.DataFrame({
    'subject_id': [1] * n_time_points,
    'time': list(range(1, n_time_points + 1)),
    'outcome_SysBP6Weeks': np.random.normal(loc=120, scale=15, size=n_time_points),
    'covariates_age': [np.random.randint(20, 70)] * n_time_points
})

# 5. CROSS_OVER: Required: outcome_SysBP6Weeks, subject_id, time; Optional: group, covariates_age
n_subjects = 10
time_points = [1, 2]
rows = []
for subj in range(1, n_subjects + 1):
    # Use meaningful names directly
    grp = np.random.choice(['control', 'intervention'])
    for t in time_points:
        rows.append({
            'subject_id': subj,
            'group': grp,
            'time': t,
            'outcome_SysBP6Weeks': np.random.normal(loc=120, scale=15),
            'covariates_age': np.random.randint(20, 70)
        })
df_cross_over = pd.DataFrame(rows)
# For cross-over, apply effect to intervention group:
df_cross_over.loc[df_cross_over['group'] == 'intervention', 'outcome_SysBP6Weeks'] += 10

# 6. FACTORIAL: Required: outcome_SysBP6Weeks, group; Optional: subject_id, time, covariates_age
n = 30
df_factorial = pd.DataFrame({
    'subject_id': np.arange(1, n + 1),
    'group': np.random.choice(['control', 'intervention', 'comparator', 'additional'], size=n),
    'time': np.random.choice([1, 2], size=n),  # optional
    'outcome_SysBP6Weeks': np.random.normal(loc=120, scale=15, size=n),
    'covariates_age': np.random.randint(20, 70, size=n)
})
# Assume intervention group gets the treatment effect:
df_factorial.loc[df_factorial['group'] == 'intervention', 'outcome_SysBP6Weeks'] += 10

# 7. REPEATED_CROSS_SECTIONAL: Required: outcome_SysBP6Weeks, time; Optional: group, subject_id, covariates_age
n = 40
df_repeated_cross_sectional = pd.DataFrame({
    'subject_id': np.arange(1, n + 1),
    'group': np.random.choice(['control', 'intervention'], size=n),
    'time': np.random.choice([1, 2, 3], size=n),
    'outcome_SysBP6Weeks': np.random.normal(loc=120, scale=15, size=n),
    'covariates_age': np.random.randint(20, 70, size=n)
})
# Add effect for intervention group
df_repeated_cross_sectional.loc[df_repeated_cross_sectional['group'] == 'intervention', 'outcome_SysBP6Weeks'] += 10

# 8. NESTED: Required: outcome_SysBP6Weeks, group, subject_id; Optional: time, covariates_age; NOT_USED: pair_id
n_subjects = 30
df_nested = pd.DataFrame({
    'subject_id': np.arange(1, n_subjects + 1),
    'group': np.random.choice(['control', 'intervention', 'comparator'], size=n_subjects),
    'outcome_SysBP6Weeks': np.random.normal(loc=120, scale=15, size=n_subjects),
    'time': np.random.choice([1, 2], size=n_subjects),  # optional
    'covariates_age': np.random.randint(20, 70, size=n_subjects)
})
# Assume 'intervention' is the exposure (with a higher mean):
df_nested.loc[df_nested['group'] == 'intervention', 'outcome_SysBP6Weeks'] += 10

# 9. LATIN_SQUARE: Required: outcome_SysBP6Weeks, group, subject_id, time; Optional: covariates_age; NOT_USED: pair_id
n = 20
df_latin_square = pd.DataFrame({
    'subject_id': np.arange(1, n + 1),
    'group': np.random.choice(['control', 'intervention', 'comparator', 'additional'], size=n),
    'time': np.random.choice([1, 2, 3, 4], size=n),
    'outcome_SysBP6Weeks': np.random.normal(loc=120, scale=15, size=n),
    'covariates_age': np.random.randint(20, 70, size=n)
})
# Designate intervention group as having the effect:
df_latin_square.loc[df_latin_square['group'] == 'intervention', 'outcome_SysBP6Weeks'] += 10

# 10. ONE_SAMPLE: (no group variable)
n = 15
df_one_sample = pd.DataFrame({
    'subject_id': np.arange(1, n + 1),
    'outcome_SysBP6Weeks': np.random.normal(loc=120, scale=15, size=n)
})

# ---------------------------
# TEST DATASETS
# ---------------------------

# 1. ONE_SAMPLE_T_TEST: (no group)
n = 15
df_one_sample_t_test = pd.DataFrame({
    'subject_id': np.arange(1, n + 1),
    'outcome_SysBP6Weeks': np.random.normal(loc=120, scale=15, size=n)
})

# 2. INDEPENDENT_T_TEST: Required: group, outcome_SysBP6Weeks; Optional: subject_id, covariates_age
n = 20
df_independent_t_test = pd.DataFrame({
    'subject_id': np.arange(1, n + 1),
    'group': np.random.choice(['control', 'intervention'], size=n),
    'outcome_SysBP6Weeks': np.random.normal(loc=120, scale=15, size=n),
    'covariates_age': np.random.randint(20, 70, size=n)
})
df_independent_t_test.loc[df_independent_t_test['group'] == 'intervention', 'outcome_SysBP6Weeks'] += 10
df_independent_t_test['group'] = df_independent_t_test['group'].map({'control': 'control', 'intervention': 'intervention'})

# 3. PAIRED_T_TEST: (no group)
n_subjects = 10
time_points = [1, 2]
rows = []
for subj in range(1, n_subjects + 1):
    for t in time_points:
        rows.append({
            'subject_id': subj,
            'time': t,
            'outcome_SysBP6Weeks': np.random.normal(loc=120, scale=15),
            'pair_id': subj,
            'covariates_age': np.random.randint(20, 70)
        })
df_paired_t_test = pd.DataFrame(rows)

# 4. ONE_WAY_ANOVA: Required: group, outcome_SysBP6Weeks; Optional: subject_id, covariates_age
n = 25
df_one_way_anova = pd.DataFrame({
    'subject_id': np.arange(1, n + 1),
    'group': np.random.choice(['control', 'intervention', 'comparator'], size=n),
    'outcome_SysBP6Weeks': np.random.normal(loc=120, scale=15, size=n),
    'covariates_age': np.random.randint(20, 70, size=n)
})
df_one_way_anova.loc[df_one_way_anova['group'] == 'intervention', 'outcome_SysBP6Weeks'] += 10
df_one_way_anova['group'] = df_one_way_anova['group'].map({'control': 'control', 'intervention': 'intervention', 'comparator': 'comparator'})

# 5. REPEATED_MEASURES_ANOVA: (no group)
n_subjects = 10
time_points = [1, 2, 3]
rows = []
for subj in range(1, n_subjects + 1):
    for t in time_points:
        rows.append({
            'subject_id': subj,
            'time': t,
            'outcome_SysBP6Weeks': np.random.normal(loc=120, scale=15)
        })
df_repeated_measures_anova = pd.DataFrame(rows)

# 6. MIXED_ANOVA: Required: group, subject_id, time, outcome_SysBP6Weeks
n_subjects = 12
time_points = [1, 2]
rows = []
for subj in range(1, n_subjects + 1):
    grp = np.random.choice(['control', 'intervention'])
    for t in time_points:
        rows.append({
            'subject_id': subj,
            'group': grp,
            'time': t,
            'outcome_SysBP6Weeks': np.random.normal(loc=120, scale=15)
        })
df_mixed_anova = pd.DataFrame(rows)
df_mixed_anova.loc[df_mixed_anova['group'] == 'intervention', 'outcome_SysBP6Weeks'] += 10
df_mixed_anova['group'] = df_mixed_anova['group'].map({'control': 'control', 'intervention': 'intervention'})

# 7. LINEAR_REGRESSION: Required: group, outcome_SysBP6Weeks, covariates_age
n = 30
df_linear_regression = pd.DataFrame({
    'subject_id': np.arange(1, n + 1),
    'group': np.random.choice(['control', 'intervention', 'comparator'], size=n),
    'outcome_SysBP6Weeks': np.random.normal(loc=120, scale=15, size=n),
    'covariates_age': np.random.randint(20, 70, size=n)
})
df_linear_regression.loc[df_linear_regression['group'] == 'intervention', 'outcome_SysBP6Weeks'] += 10
df_linear_regression['group'] = df_linear_regression['group'].map({'control': 'control', 'intervention': 'intervention', 'comparator': 'comparator'})

# 8. ANCOVA: Required: group, outcome_SysBP6Weeks, covariates_age
n = 25
df_ancova = pd.DataFrame({
    'subject_id': np.arange(1, n + 1),
    'group': np.random.choice(['control', 'intervention', 'comparator'], size=n),
    'outcome_SysBP6Weeks': np.random.normal(loc=120, scale=15, size=n),
    'covariates_age': np.random.randint(20, 70, size=n)
})
df_ancova.loc[df_ancova['group'] == 'intervention', 'outcome_SysBP6Weeks'] += 10
df_ancova['group'] = df_ancova['group'].map({'control': 'control', 'intervention': 'intervention', 'comparator': 'comparator'})

# 9. LINEAR_MIXED_EFFECTS_MODEL: Required: subject_id, time, outcome_SysBP6Weeks; Optional: group
n_subjects = 15
time_points = [1, 2]
rows = []
for subj in range(1, n_subjects + 1):
    grp = np.random.choice(['control', 'intervention'])
    for t in time_points:
        rows.append({
            'subject_id': subj,
            'group': grp,
            'time': t,
            'outcome_SysBP6Weeks': np.random.normal(loc=120, scale=15)
        })
df_linear_mixed_effects_model = pd.DataFrame(rows)
df_linear_mixed_effects_model.loc[df_linear_mixed_effects_model['group'] == 'intervention', 'outcome_SysBP6Weeks'] += 10
df_linear_mixed_effects_model['group'] = df_linear_mixed_effects_model['group'].map({'control': 'control', 'intervention': 'intervention'})

# 10. MANN_WHITNEY_U_TEST: Required: group, outcome_SysBP6Weeks
n = 20
df_mann_whitney_u_test = pd.DataFrame({
    'group': np.random.choice(['control', 'intervention'], size=n),
    'outcome_SysBP6Weeks': np.random.normal(loc=120, scale=15, size=n)
})
df_mann_whitney_u_test.loc[df_mann_whitney_u_test['group'] == 'intervention', 'outcome_SysBP6Weeks'] += 10
df_mann_whitney_u_test['group'] = df_mann_whitney_u_test['group'].map({'control': 'control', 'intervention': 'intervention'})

# 11. WILCOXON_SIGNED_RANK_TEST: (no group)
n_subjects = 10
time_points = [1, 2]
rows = []
for subj in range(1, n_subjects + 1):
    for t in time_points:
        rows.append({
            'subject_id': subj,
            'time': t,
            'outcome_SysBP6Weeks': np.random.normal(loc=120, scale=15)
        })
df_wilcoxon_signed_rank_test = pd.DataFrame(rows)

# 12. KRUSKAL_WALLIS_TEST: Required: group, outcome_SysBP6Weeks
n = 30
df_kruskal_wallis_test = pd.DataFrame({
    'group': np.random.choice(['control', 'intervention', 'comparator'], size=n),
    'outcome_SysBP6Weeks': np.random.normal(loc=120, scale=15, size=n)
})
df_kruskal_wallis_test.loc[df_kruskal_wallis_test['group'] == 'intervention', 'outcome_SysBP6Weeks'] += 10
df_kruskal_wallis_test['group'] = df_kruskal_wallis_test['group'].map({'control': 'control', 'intervention': 'intervention', 'comparator': 'comparator'})

# 13. PEARSON_CORRELATION: (no group)
n = 30
df_pearson_correlation = pd.DataFrame({
    'subject_id': np.arange(1, n + 1),
    'group': np.random.choice(['control', 'intervention'], size=n),
    'outcome_SysBP6Weeks': np.random.normal(loc=120, scale=15, size=n),
    'covariates_age': np.random.randint(20, 70, size=n)
})

# 14. SPEARMAN_CORRELATION: (no group)
n = 30
df_spearman_correlation = pd.DataFrame({
    'subject_id': np.arange(1, n + 1),
    'group': np.random.choice(['control', 'intervention'], size=n),
    'outcome_SysBP6Weeks': np.random.normal(loc=120, scale=15, size=n),
    'covariates_age': np.random.randint(20, 70, size=n)
})

# 15. KENDALL_TAU_CORRELATION: (no group)
n = 30
df_kendall_tau_correlation = pd.DataFrame({
    'subject_id': np.arange(1, n + 1),
    'group': np.random.choice(['control', 'intervention'], size=n),
    'outcome_SysBP6Weeks': np.random.normal(loc=120, scale=15, size=n),
    'covariates_age': np.random.randint(20, 70, size=n)
})

# 16. SURVIVAL_ANALYSIS: Required: group, subject_id, time, outcome_SysBP6Weeks
n = 20
df_survival_analysis = pd.DataFrame({
    'subject_id': np.arange(1, n + 1),
    'group': np.random.choice(['control', 'intervention'], size=n),
    'time': np.random.choice([1, 2, 3], size=n),
    'event': np.random.choice([0, 1], size=n),
    'outcome_SysBP6Weeks': np.random.normal(loc=120, scale=15, size=n)
})
df_survival_analysis.loc[df_survival_analysis['group'] == 'intervention', 'outcome_SysBP6Weeks'] += 10
# Make intervention group have higher event rate (better survival outcome)
df_survival_analysis.loc[df_survival_analysis['group'] == 'intervention', 'event'] = np.random.choice(
    [0, 1], size=sum(df_survival_analysis['group'] == 'intervention'), p=[0.25, 0.75]
)
df_survival_analysis['group'] = df_survival_analysis['group'].map({'control': 'control', 'intervention': 'intervention'})

# 17. CHI_SQUARE_TEST: Required: group, outcome_SysBP6Weeks (categorical)
n = 25
df_chi_square_test = pd.DataFrame({
    'subject_id': np.arange(1, n + 1),
    'group': np.random.choice(['control', 'intervention'], size=n),
    # initially create outcomes uniformly...
    'outcome_SysBP6Weeks': np.random.choice(['High', 'Low'], size=n),
    'covariates_age': np.random.randint(20, 70, size=n)
})
def simulate_chi_square(g):
    if g == 'intervention':
        return np.random.choice(['High', 'Low'], p=[0.8, 0.2])
    else:
        return np.random.choice(['High', 'Low'], p=[0.4, 0.6])
df_chi_square_test['outcome_SysBP6Weeks'] = df_chi_square_test['group'].apply(simulate_chi_square)
df_chi_square_test['group'] = df_chi_square_test['group'].map({'control': 'control', 'intervention': 'intervention'})

# 18. FISHERS_EXACT_TEST: Required: group, outcome_SysBP6Weeks (categorical)
n = 25
df_fishers_exact_test = pd.DataFrame({
    'subject_id': np.arange(1, n + 1),
    'group': np.random.choice(['control', 'intervention'], size=n),
    # initially random...
    'outcome_SysBP6Weeks': np.random.choice(['Yes', 'No'], size=n),
    'covariates_age': np.random.randint(20, 70, size=n)
})
def simulate_fishers(g):
    if g == 'intervention':
        return np.random.choice(['Yes', 'No'], p=[0.8, 0.2])
    else:
        return np.random.choice(['Yes', 'No'], p=[0.4, 0.6])
df_fishers_exact_test['outcome_SysBP6Weeks'] = df_fishers_exact_test['group'].apply(simulate_fishers)
df_fishers_exact_test['group'] = df_fishers_exact_test['group'].map({'control': 'control', 'intervention': 'intervention'})

# 19. LOGISTIC_REGRESSION: Required: group, outcome_SysBP6Weeks (binary), covariates_age
n = 30
df_logistic_regression = pd.DataFrame({
    'subject_id': np.arange(1, n + 1),
    'group': np.random.choice(['control', 'intervention'], size=n),
    # placeholder for outcome; will re-simulate using probabilities below
    'outcome_SysBP6Weeks': np.nan,
    'covariates_age': np.random.randint(20, 70, size=n)
})
def simulate_logistic(g):
    # For exposed group ('intervention') assign outcome 1 with higher probability (e.g., 80% vs 40%)
    if g == 'intervention':
        return np.random.binomial(1, 0.8)
    else:
        return np.random.binomial(1, 0.4)
df_logistic_regression['outcome_SysBP6Weeks'] = df_logistic_regression['group'].apply(simulate_logistic)
df_logistic_regression['group'] = df_logistic_regression['group'].map({'control': 'control', 'intervention': 'intervention'})

# 20. POINT_BISERIAL_CORRELATION: (requires one binary and one continuous variable)
n = 30
# Create a binary variable that will have a real effect on the outcome
treatment = np.random.choice([0, 1], size=n)
# Create continuous outcome with an effect for the treatment group
baseline = np.random.normal(loc=120, scale=15, size=n)
# Add effect: treatment group (1) has +10 mmHg higher values
outcome = baseline + (treatment * 10)

df_point_biserial_correlation = pd.DataFrame({
    'subject_id': np.arange(1, n + 1),
    'group': treatment,  # Binary predictor (0/1) - renamed from binary_treatment to match UI expectations
    'outcome_SysBP6Weeks': outcome,  # Continuous outcome affected by the treatment
    'covariates_age': np.random.randint(20, 70, size=n)  # Continuous covariate
})

# 21. MULTINOMIAL_LOGISTIC_REGRESSION: Required: group, outcome_SysBP6Weeks (categorical with >2 levels), covariates_age
n = 30
df_multinomial_logistic_regression = pd.DataFrame({
    'subject_id': np.arange(1, n + 1),
    'group': np.random.choice(['control', 'intervention', 'comparator'], size=n),
    # placeholder for outcome; will re-simulate using probabilities
    'outcome_SysBP6Weeks': np.nan,
    'covariates_age': np.random.randint(20, 70, size=n)
})
def simulate_multinomial(g):
    # For group 'intervention', assume higher chance of "High" outcome.
    if g == 'intervention':
        return np.random.choice(['Low', 'Medium', 'High'], p=[0.1, 0.2, 0.7])
    else:
        return np.random.choice(['Low', 'Medium', 'High'])
df_multinomial_logistic_regression['outcome_SysBP6Weeks'] = df_multinomial_logistic_regression['group'].apply(simulate_multinomial)
df_multinomial_logistic_regression['group'] = df_multinomial_logistic_regression['group'].map({'control': 'control', 'intervention': 'intervention', 'comparator': 'comparator'})

# 22. POISSON_REGRESSION: Required: group, outcome_SysBP6Weeks (count), covariates_age
n = 30
df_poisson_regression = pd.DataFrame({
    'subject_id': np.arange(1, n + 1),
    'group': np.random.choice(['control', 'intervention'], size=n),
    # placeholder; will simulate with different lambda
    'outcome_SysBP6Weeks': np.nan,
    'covariates_age': np.random.randint(20, 70, size=n)
})
def simulate_poisson(g):
    if g == 'intervention':
        return np.random.poisson(lam=8)
    else:
        return np.random.poisson(lam=5)
df_poisson_regression['outcome_SysBP6Weeks'] = df_poisson_regression['group'].apply(simulate_poisson)
df_poisson_regression['group'] = df_poisson_regression['group'].map({'control': 'control', 'intervention': 'intervention'})

# 23. NEGATIVE_BINOMIAL_REGRESSION: Required: group, outcome_SysBP6Weeks (overdispersed count), covariates_age
n = 30
df_negative_binomial_regression = pd.DataFrame({
    'subject_id': np.arange(1, n + 1),
    'group': np.random.choice(['control', 'intervention'], size=n),
    # placeholder; simulate with different parameters
    'outcome_SysBP6Weeks': np.nan,
    'covariates_age': np.random.randint(20, 70, size=n)
})
def simulate_negative_binomial(g):
    if g == 'intervention':
        return np.random.negative_binomial(n=5, p=0.4)
    else:
        return np.random.negative_binomial(n=5, p=0.5)
df_negative_binomial_regression['outcome_SysBP6Weeks'] = df_negative_binomial_regression['group'].apply(simulate_negative_binomial)
df_negative_binomial_regression['group'] = df_negative_binomial_regression['group'].map({'control': 'control', 'intervention': 'intervention'})

# 24. ORDINAL_REGRESSION: Required: group, outcome_SysBP6Weeks (ordinal), covariates_age
n = 30
df_ordinal_regression = pd.DataFrame({
    'subject_id': np.arange(1, n + 1),
    'group': np.random.choice(['control', 'intervention'], size=n),
    # placeholder; outcome as ordinal category
    'outcome_SysBP6Weeks': np.nan,
    'covariates_age': np.random.randint(20, 70, size=n)
})
def simulate_ordinal(g):
    if g == 'intervention':
        return np.random.choice([1, 2, 3, 4], p=[0.1, 0.2, 0.3, 0.4])
    else:
        return np.random.choice([1, 2, 3, 4])
df_ordinal_regression['outcome_SysBP6Weeks'] = df_ordinal_regression['group'].apply(simulate_ordinal)
df_ordinal_regression['group'] = df_ordinal_regression['group'].map({'control': 'control', 'intervention': 'intervention'})



# ---------------------------
# SUBGROUP ANALYSIS DATASET
# ---------------------------

# Create a comprehensive dataset for subgroup analysis with realistic treatment effect heterogeneity
n = 500  # Large sample size for robust testing

# Create base dataframe
df_subgroup_analysis = pd.DataFrame({
    'subject_id': np.arange(1, n + 1),
    'treatment': np.random.choice([0, 1], size=n),  # Binary treatment
    'age': np.random.randint(18, 85, size=n),
    'sex': np.random.choice(['Male', 'Female'], size=n),
    'bmi': np.random.normal(27, 5, size=n),  # Mean BMI of 27 with SD of 5
    'smoking_status': np.random.choice(['Never', 'Former', 'Current'], size=n, p=[0.5, 0.3, 0.2]),
    'diabetes': np.random.choice([0, 1], size=n, p=[0.85, 0.15]),  # 15% prevalence
    'hypertension': np.random.choice([0, 1], size=n, p=[0.7, 0.3]),  # 30% prevalence
    'baseline_sbp': np.random.normal(130, 15, size=n),  # Baseline systolic blood pressure
    'cholesterol': np.random.normal(200, 40, size=n),
    'hdl': np.random.normal(50, 15, size=n),
    'ldl': np.random.normal(120, 30, size=n),
    'prior_cvd': np.random.choice([0, 1], size=n, p=[0.9, 0.1]),  # 10% prevalence
    'medication_adherence': np.random.choice(['Low', 'Medium', 'High'], size=n, p=[0.2, 0.3, 0.5])
})

# Create age groups for easier subgroup analysis
df_subgroup_analysis['age_group'] = pd.cut(
    df_subgroup_analysis['age'], 
    bins=[0, 40, 65, 100], 
    labels=['Young', 'Middle-aged', 'Elderly']
)

# Create BMI categories
df_subgroup_analysis['bmi_category'] = pd.cut(
    df_subgroup_analysis['bmi'], 
    bins=[0, 18.5, 25, 30, 100], 
    labels=['Underweight', 'Normal', 'Overweight', 'Obese']
)

# Base effect (main treatment effect) - 10 mmHg reduction
base_effect = -10

# Generate outcome with heterogeneous treatment effects across subgroups
# Initialize outcome with baseline values and noise
df_subgroup_analysis['outcome_sbp_reduction'] = df_subgroup_analysis['baseline_sbp'] * 0.05 + np.random.normal(0, 5, size=n)

# Apply treatment effect with heterogeneity across subgroups
treatment_mask = df_subgroup_analysis['treatment'] == 1

# 1. Age-based heterogeneity: Elderly have stronger treatment effect
df_subgroup_analysis.loc[treatment_mask & (df_subgroup_analysis['age_group'] == 'Elderly'), 'outcome_sbp_reduction'] += (base_effect - 5)
df_subgroup_analysis.loc[treatment_mask & (df_subgroup_analysis['age_group'] == 'Middle-aged'), 'outcome_sbp_reduction'] += base_effect
df_subgroup_analysis.loc[treatment_mask & (df_subgroup_analysis['age_group'] == 'Young'), 'outcome_sbp_reduction'] += (base_effect + 3)

# 2. Sex-based heterogeneity: Males respond slightly better
df_subgroup_analysis.loc[treatment_mask & (df_subgroup_analysis['sex'] == 'Male'), 'outcome_sbp_reduction'] += -2
df_subgroup_analysis.loc[treatment_mask & (df_subgroup_analysis['sex'] == 'Female'), 'outcome_sbp_reduction'] += 2

# 3. Diabetes-based heterogeneity: Diabetics have weaker response
df_subgroup_analysis.loc[treatment_mask & (df_subgroup_analysis['diabetes'] == 1), 'outcome_sbp_reduction'] += 4

# 4. Smoking status heterogeneity: Current smokers have weakest response
df_subgroup_analysis.loc[treatment_mask & (df_subgroup_analysis['smoking_status'] == 'Current'), 'outcome_sbp_reduction'] += 5
df_subgroup_analysis.loc[treatment_mask & (df_subgroup_analysis['smoking_status'] == 'Former'), 'outcome_sbp_reduction'] += 2

# 5. Medication adherence effect: High adherence strengthens treatment effect
df_subgroup_analysis.loc[treatment_mask & (df_subgroup_analysis['medication_adherence'] == 'High'), 'outcome_sbp_reduction'] += -3
df_subgroup_analysis.loc[treatment_mask & (df_subgroup_analysis['medication_adherence'] == 'Low'), 'outcome_sbp_reduction'] += 4

# 6. BMI effect: Obese have weaker response
df_subgroup_analysis.loc[treatment_mask & (df_subgroup_analysis['bmi_category'] == 'Obese'), 'outcome_sbp_reduction'] += 3

# 7. Hypertension status affects baseline but not treatment effect much
df_subgroup_analysis.loc[df_subgroup_analysis['hypertension'] == 1, 'outcome_sbp_reduction'] += -2  # All hypertensives have slightly better response

# 8. Interaction effect between diabetes and age
df_subgroup_analysis.loc[treatment_mask & (df_subgroup_analysis['diabetes'] == 1) & 
                        (df_subgroup_analysis['age_group'] == 'Elderly'), 'outcome_sbp_reduction'] += -3  # Elderly diabetics do better than expected

# Generate a binary outcome for testing binary outcome subgroup analysis
# Lower outcome_sbp_reduction values mean better response, so negative values are "responders"
df_subgroup_analysis['responder'] = (df_subgroup_analysis['outcome_sbp_reduction'] < -5).astype(int)

# ---------------------------
# MEDIATION ANALYSIS DATASET
# ---------------------------

# Create a dataset for validating mediation analysis with known effects
n = 200  # Sample size for mediation analysis

# 1. Generate independent variables 
X1 = np.random.normal(loc=5, scale=1.5, size=n)  # Primary predictor
X2 = np.random.normal(loc=3, scale=1.0, size=n)  # Secondary predictor

# 2. Generate moderator with some correlation to X1 (r~0.2)
W = 0.2 * X1 + np.random.normal(loc=4, scale=1.2, size=n)

# 3. Generate error terms
error_m1 = np.random.normal(loc=0, scale=1.0, size=n)
error_m2 = np.random.normal(loc=0, scale=1.0, size=n)
error_y = np.random.normal(loc=0, scale=1.0, size=n)

# 4. Known path coefficients (for validation)
# X1 -> M1 path
a1 = 0.6
# X1 -> M2 path 
a2 = 0.3
# X1*W -> M1 (moderation effect)
a3 = 0.15
# M1 -> Y path
b1 = 0.5
# M2 -> Y path
b2 = 0.4
# M1 -> M2 path (for serial mediation)
d = 0.35
# X1 -> Y direct path
c_prime = 0.3
# X2 -> Y direct path
c2 = 0.25

# 5. Generate mediators
# First mediator with moderation effect
M1 = a1 * X1 + a3 * (X1 * W) + error_m1
# Second mediator with influence from X1 and M1 (for serial mediation)
M2 = a2 * X1 + d * M1 + error_m2

# 6. Generate dependent variable Y
Y = c_prime * X1 + c2 * X2 + b1 * M1 + b2 * M2 + error_y

# 7. Add covariates
age = np.random.randint(18, 65, size=n)
gender = np.random.choice(['Male', 'Female'], size=n)
# Create a small effect of age on Y
Y = Y + 0.02 * age

# 8. Create final mediation dataset
df_mediation = pd.DataFrame({
    'subject_id': np.arange(1, n + 1),
    'X1': X1,
    'X2': X2,
    'M1': M1,
    'M2': M2,
    'W': W,
    'Y': Y,
    'age': age,
    'gender': gender
})

# 9. Create a binary version of X1 for treatment/control scenarios
df_mediation['treatment'] = (X1 > np.median(X1)).astype(int)

# Store the true effects for validation
mediation_true_effects = {
    'a1': a1,                      # X1 -> M1
    'a2': a2,                      # X1 -> M2
    'a3': a3,                      # X1*W -> M1 (moderation)
    'b1': b1,                      # M1 -> Y
    'b2': b2,                      # M2 -> Y
    'd': d,                        # M1 -> M2 (serial path)
    'c_prime': c_prime,            # X1 -> Y (direct effect)
    'c2': c2,                      # X2 -> Y
    'indirect_effect_m1': a1 * b1,       # Simple indirect effect through M1
    'indirect_effect_m2': a2 * b2,       # Simple indirect effect through M2
    'serial_indirect': a1 * d * b2,      # Serial indirect effect X1 -> M1 -> M2 -> Y
    'moderated_indirect_w_low': (a1 + a3 * (np.mean(W) - np.std(W))) * b1,  # Indirect at low W
    'moderated_indirect_w_mean': (a1 + a3 * np.mean(W)) * b1,               # Indirect at mean W
    'moderated_indirect_w_high': (a1 + a3 * (np.mean(W) + np.std(W))) * b1, # Indirect at high W
    'total_effect': c_prime + a1 * b1 + a2 * b2 + a1 * d * b2              # Total effect
}

# Add total effect as attribute for easier access
df_mediation.attrs['true_effects'] = mediation_true_effects

# ---------------------------
# FINAL DICTIONARY: Combine Study and Test Datasets
# ---------------------------
final_datasets = {
    # Study Designs
    'study_between_subjects': df_between_subjects,
    'study_within_subjects': df_within_subjects,
    'study_mixed': df_mixed,
    'study_single_subject': df_single_subject,
    'study_cross_over': df_cross_over,
    'study_factorial': df_factorial,
    'study_repeated_cross_sectional': df_repeated_cross_sectional,
    'study_nested': df_nested,
    'study_latin_square': df_latin_square,
    'study_one_sample': df_one_sample,
    # Statistical Tests
    'test_one_sample_t_test': df_one_sample_t_test,
    'test_independent_t_test': df_independent_t_test,
    'test_paired_t_test': df_paired_t_test,
    'test_one_way_anova': df_one_way_anova,
    'test_repeated_measures_anova': df_repeated_measures_anova,
    'test_mixed_anova': df_mixed_anova,
    'test_linear_regression': df_linear_regression,
    'test_ancova': df_ancova,
    'test_linear_mixed_effects_model': df_linear_mixed_effects_model,
    'test_mann_whitney_u_test': df_mann_whitney_u_test,
    'test_wilcoxon_signed_rank_test': df_wilcoxon_signed_rank_test,
    'test_kruskal_wallis_test': df_kruskal_wallis_test,
    'test_pearson_correlation': df_pearson_correlation,
    'test_spearman_correlation': df_spearman_correlation,
    'test_kendall_tau_correlation': df_kendall_tau_correlation,
    'test_survival_analysis': df_survival_analysis,
    'test_chi_square_test': df_chi_square_test,
    'test_fishers_exact_test': df_fishers_exact_test,
    'test_logistic_regression': df_logistic_regression,
    'test_point_biserial_correlation': df_point_biserial_correlation,
    'test_multinomial_logistic_regression': df_multinomial_logistic_regression,
    'test_poisson_regression': df_poisson_regression,
    'test_negative_binomial_regression': df_negative_binomial_regression,
    'test_ordinal_regression': df_ordinal_regression,

    # Subgroup Analysis
    'subgroup_analysis': df_subgroup_analysis,

    # Mediation Analysis
    'mediation_analysis': df_mediation
}

# # (Optional) Print the keys and the first few rows of each dataset for verification
# for key, df in final_datasets.items():
#     print(f"Dataset: {key}")
#     print(df.head(), "\n")



# Complete Picture: Combining All Analyses
# When we integrate findings from all three analyses:
# Simple Mediation: Identified the basic mediation structure but with inflated estimates
# Multiple Mediators: Correctly modeled both mediators' effects on Y
# Moderated Mediation: Accurately captured the moderation of the X→M1 path
# Serial Mediation: Correctly identified the M1→M2 pathway and direct effect
# Together, these analyses reveal the true underlying data structure, which combines:
# Moderated mediation (W moderates X→M1)
# Serial mediation (X→M1→M2→Y)
# Direct effect (X→Y)
# Conclusion
# This serial mediation analysis provides an excellent representation of the true data generation process, particularly for:
# ✓ The direct effect (0.3087 vs 0.3)
# ✓ The X→M2 path (0.3961 vs 0.3)
# ✓ The M1→M2 serial path (0.2673 vs 0.35)
# ✓ Both mediator→Y paths (0.4644 vs 0.5 and 0.4074 vs 0.4)
# The only aspect not fully captured is the moderation effect on the X→M1 path. For a complete match with the data generation process, a moderated serial mediation model would be needed, combining both the moderation and serial aspects.
# Overall, your comprehensive approach to analyzing this dataset through multiple complementary methods shows an excellent understanding of mediation analysis techniques and their application.