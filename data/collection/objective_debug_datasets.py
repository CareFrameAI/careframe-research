import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# ---------------------------
# HEALTHCARE OBJECTIVE DATASETS
# ---------------------------

# 1. READMISSION_STUDY: Inpatient readmission data
n = 500
admission_dates = [datetime.now() - timedelta(days=np.random.randint(30, 730)) for _ in range(n)]

df_readmission_study = pd.DataFrame({
    'patient_id': np.arange(1001, 1001 + n),
    'age': np.random.randint(18, 95, size=n),
    'sex': np.random.choice(['Male', 'Female'], size=n),
    'admission_date': admission_dates,
    'length_of_stay': np.random.randint(1, 30, size=n),
    'primary_diagnosis': np.random.choice(['Heart Failure', 'Pneumonia', 'COPD', 'Diabetes', 'Stroke'], size=n),
    'comorbidity_count': np.random.randint(0, 6, size=n),
    'insurance_type': np.random.choice(['Medicare', 'Medicaid', 'Private', 'Self-Pay'], size=n),
    'discharge_disposition': np.random.choice(['Home', 'SNF', 'Home Health', 'LTAC'], size=n),
    'readmitted_30days': np.random.choice([0, 1], size=n, p=[0.85, 0.15])  # 15% readmission rate
})

# Add intervention effect: Patients with "Home Health" have lower readmission rates
home_health_mask = df_readmission_study['discharge_disposition'] == 'Home Health'
df_readmission_study.loc[home_health_mask, 'readmitted_30days'] = np.random.choice(
    [0, 1], size=sum(home_health_mask), p=[0.92, 0.08]  # Only 8% readmission rate
)

# Age group effect: Elderly patients have higher readmission
elderly_mask = df_readmission_study['age'] > 75
df_readmission_study.loc[elderly_mask, 'readmitted_30days'] = np.random.choice(
    [0, 1], size=sum(elderly_mask), p=[0.80, 0.20]  # 20% readmission rate for elderly
)

# 2. LOS_PREDICTION: Length of stay prediction
n = 600
df_los_prediction = pd.DataFrame({
    'patient_id': np.arange(2001, 2001 + n),
    'age': np.random.randint(18, 95, size=n),
    'sex': np.random.choice(['Male', 'Female'], size=n),
    'emergency_admission': np.random.choice([0, 1], size=n, p=[0.3, 0.7]),
    'surgical_procedure': np.random.choice([0, 1], size=n, p=[0.6, 0.4]),
    'primary_diagnosis': np.random.choice(['Heart Failure', 'Pneumonia', 'COPD', 'Diabetes', 'Stroke', 'Sepsis'], size=n),
    'comorbidity_count': np.random.randint(0, 8, size=n),
    'icu_stay': np.random.choice([0, 1], size=n, p=[0.85, 0.15]),
    'hospital_teaching_status': np.random.choice(['Teaching', 'Non-Teaching'], size=n),
    'weekend_admission': np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
})

# Base length of stay
base_los = np.random.gamma(shape=2.0, scale=2.0, size=n)

# Add effects for various factors
los = base_los.copy()
los[df_los_prediction['emergency_admission'] == 1] += 0.8  # Emergency adds days
los[df_los_prediction['surgical_procedure'] == 1] += 1.5   # Surgery adds days
los[df_los_prediction['icu_stay'] == 1] += 3.0             # ICU adds days
los[df_los_prediction['weekend_admission'] == 1] += 0.5    # Weekend adds days

# Add comorbidity effect
los += df_los_prediction['comorbidity_count'] * 0.4

# Add age effect
age_effect = (df_los_prediction['age'] - 50) / 50  # Normalized around 50 years
los += age_effect

# Add diagnosis effect
diagnosis_effect = {
    'Heart Failure': 1.0,
    'Pneumonia': 1.2,
    'COPD': 1.5,
    'Diabetes': 0.7,
    'Stroke': 2.5,
    'Sepsis': 3.0
}
for diagnosis, effect in diagnosis_effect.items():
    los[df_los_prediction['primary_diagnosis'] == diagnosis] += effect

# Ensure all LOS values are positive and convert to integer
los = np.maximum(los, 1)
df_los_prediction['length_of_stay'] = np.round(los).astype(int)

# 3. READMISSION_INTERVENTION: Testing different interventions
n = 400
df_readmission_intervention = pd.DataFrame({
    'patient_id': np.arange(3001, 3001 + n),
    'age': np.random.randint(18, 95, size=n),
    'sex': np.random.choice(['Male', 'Female'], size=n),
    'length_of_stay': np.random.randint(1, 21, size=n),
    'primary_diagnosis': np.random.choice(['Heart Failure', 'Pneumonia', 'COPD'], size=n),
    'comorbidity_count': np.random.randint(0, 6, size=n),
    'intervention_group': np.random.choice(['Control', 'Medication Reconciliation', 'Follow-up Call', 'Home Visit'], size=n),
    'risk_score': np.random.uniform(0, 100, size=n),
})

# Base readmission probability
base_readmission_prob = 0.15

# Intervention effects (reduction in readmission probability)
intervention_effects = {
    'Control': 0.0,
    'Medication Reconciliation': 0.05,
    'Follow-up Call': 0.07,
    'Home Visit': 0.10
}

# Risk factors
# Higher risk score increases readmission probability
risk_effect = df_readmission_intervention['risk_score'] / 100

# Comorbidity effect
comorbidity_effect = df_readmission_intervention['comorbidity_count'] * 0.02

# Calculate final readmission probability
readmission_prob = base_readmission_prob + risk_effect + comorbidity_effect

# Apply intervention effects
for intervention, effect in intervention_effects.items():
    idx = df_readmission_intervention['intervention_group'] == intervention
    readmission_prob[idx] -= effect

# Ensure probabilities are between 0 and 1
readmission_prob = np.clip(readmission_prob, 0.01, 0.99)

# Generate readmission outcome
df_readmission_intervention['readmitted_30days'] = np.random.binomial(1, readmission_prob)

# 4. ED_TO_INPATIENT_CONVERSION: Emergency Department to Inpatient Conversion
n = 550
df_ed_conversion = pd.DataFrame({
    'patient_id': np.arange(4001, 4001 + n),
    'age': np.random.randint(18, 95, size=n),
    'sex': np.random.choice(['Male', 'Female'], size=n),
    'arrival_method': np.random.choice(['Ambulance', 'Walk-in'], size=n, p=[0.3, 0.7]),
    'chief_complaint': np.random.choice(['Chest Pain', 'Shortness of Breath', 'Abdominal Pain', 
                                        'Fever', 'Injury', 'Neurological'], size=n),
    'triage_level': np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.15, 0.40, 0.30, 0.10]),
    'weekend_arrival': np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
    'hour_of_day': np.random.randint(0, 24, size=n),
    'lab_tests_ordered': np.random.randint(0, 10, size=n),
    'imaging_ordered': np.random.choice([0, 1], size=n, p=[0.4, 0.6]),
})

# Base conversion probability
base_conversion_prob = 0.25

# Triage effect (higher acuity = higher conversion)
triage_effect = (6 - df_ed_conversion['triage_level']) * 0.05  # Level 1 adds 0.25, Level 5 adds 0.05

# Arrival method effect
ambulance_effect = np.zeros(n)
ambulance_effect[df_ed_conversion['arrival_method'] == 'Ambulance'] = 0.15

# Chief complaint effect
complaint_effects = {
    'Chest Pain': 0.10,
    'Shortness of Breath': 0.12,
    'Abdominal Pain': 0.08,
    'Fever': 0.05,
    'Injury': 0.03,
    'Neurological': 0.15
}
complaint_effect = np.zeros(n)
for complaint, effect in complaint_effects.items():
    complaint_effect[df_ed_conversion['chief_complaint'] == complaint] = effect

# Age effect
age_effect = (df_ed_conversion['age'] - 50) / 150  # Normalized around 50

# Calculate conversion probability
conversion_prob = base_conversion_prob + triage_effect + ambulance_effect + complaint_effect + age_effect
conversion_prob = np.clip(conversion_prob, 0.01, 0.99)

# Generate admission outcome
df_ed_conversion['admitted_to_inpatient'] = np.random.binomial(1, conversion_prob)

# Add ED length of stay
base_ed_los = np.random.gamma(shape=1.5, scale=2.0, size=n)
ed_los = base_ed_los.copy()

# Higher triage (more severe) gets seen faster but stays longer
ed_los += (6 - df_ed_conversion['triage_level']) * 0.5

# Admitted patients stay longer in ED
ed_los[df_ed_conversion['admitted_to_inpatient'] == 1] += 2.0

# Weekend and night effect
ed_los[df_ed_conversion['weekend_arrival'] == 1] += 1.0
night_mask = (df_ed_conversion['hour_of_day'] >= 22) | (df_ed_conversion['hour_of_day'] <= 6)
ed_los[night_mask] += 0.5

# Lab and imaging add time
ed_los += df_ed_conversion['lab_tests_ordered'] * 0.2
ed_los[df_ed_conversion['imaging_ordered'] == 1] += 1.0

# Convert to hours, minimum 1 hour
df_ed_conversion['ed_length_of_stay_hours'] = np.maximum(np.round(ed_los), 1).astype(int)

# 5. HOSPITAL_ACQUIRED_CONDITION: Hospital-acquired conditions dataset
n = 350
df_hac = pd.DataFrame({
    'patient_id': np.arange(5001, 5001 + n),
    'age': np.random.randint(18, 95, size=n),
    'sex': np.random.choice(['Male', 'Female'], size=n),
    'length_of_stay': np.random.randint(1, 30, size=n),
    'icu_stay': np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
    'surgical_patient': np.random.choice([0, 1], size=n, p=[0.6, 0.4]),
    'central_line': np.random.choice([0, 1], size=n, p=[0.75, 0.25]),
    'urinary_catheter': np.random.choice([0, 1], size=n, p=[0.65, 0.35]),
    'ventilator': np.random.choice([0, 1], size=n, p=[0.85, 0.15]),
    'intervention_bundle': np.random.choice([0, 1], size=n, p=[0.5, 0.5]),
})

# Base HAC probability
base_hac_prob = 0.08

# Risk factors
los_effect = df_hac['length_of_stay'] * 0.005  # Each day adds 0.5% risk
icu_effect = df_hac['icu_stay'] * 0.05
central_line_effect = df_hac['central_line'] * 0.07
urinary_catheter_effect = df_hac['urinary_catheter'] * 0.06
ventilator_effect = df_hac['ventilator'] * 0.1
surgical_effect = df_hac['surgical_patient'] * 0.03

# Intervention bundle reduces risk by 50%
intervention_effect = -0.5 * base_hac_prob * df_hac['intervention_bundle']

# Calculate HAC probability
hac_prob = base_hac_prob + los_effect + icu_effect + central_line_effect + urinary_catheter_effect + ventilator_effect + surgical_effect
hac_prob = np.clip(hac_prob, 0.01, 0.99)

# Apply intervention effect (multiplicative)
intervention_mask = df_hac['intervention_bundle'] == 1
hac_prob[intervention_mask] = hac_prob[intervention_mask] * 0.5

# Generate HAC outcome
df_hac['developed_hac'] = np.random.binomial(1, hac_prob)

# Assign specific HAC types
hac_types = ['CLABSI', 'CAUTI', 'VAP', 'SSI', 'C. diff']
hac_type_probabilities = {
    'CLABSI': 0.6 if central_line_effect.mean() > 0 else 0.1,
    'CAUTI': 0.6 if urinary_catheter_effect.mean() > 0 else 0.1,
    'VAP': 0.6 if ventilator_effect.mean() > 0 else 0.1,
    'SSI': 0.6 if surgical_effect.mean() > 0 else 0.1,
    'C. diff': 0.2  # Base probability for C. diff
}

# Normalize probabilities
hac_type_probs = np.array(list(hac_type_probabilities.values()))
hac_type_probs = hac_type_probs / hac_type_probs.sum()

df_hac['hac_type'] = np.nan
hac_positive = df_hac['developed_hac'] == 1
df_hac.loc[hac_positive, 'hac_type'] = np.random.choice(hac_types, size=sum(hac_positive), p=hac_type_probs)

# ---------------------------
# FINAL DICTIONARY: Healthcare Objective Datasets
# ---------------------------
objective_datasets = {
    'obj_readmission_study': df_readmission_study,
    'obj_los_prediction': df_los_prediction,
    'obj_readmission_intervention': df_readmission_intervention,
    'obj_ed_to_inpatient': df_ed_conversion,
    'obj_hospital_acquired_condition': df_hac
} 