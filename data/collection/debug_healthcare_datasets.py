import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Research Objectives and Hypotheses for Healthcare Datasets
# 1. Inpatient Admissions Dataset
# Research Objective: Identify key determinants of 30-day readmission among hospital inpatients to develop a targeted intervention strategy.
# Primary Hypothesis: Patients discharged against medical advice (AMA) have significantly higher 30-day readmission rates compared to those with planned discharges, independent of comorbidity burden.
# Secondary Hypothesis: The relationship between length of stay and readmission risk follows a U-shaped curve, with both very short and very long stays associated with increased readmission probability.
# 2. Length of Stay Prediction Dataset
# Research Objective: Develop a predictive model for hospital length of stay using admission data to optimize resource allocation and improve discharge planning.
# Primary Hypothesis: A combination of clinical factors (vital signs, lab values) and social determinants (housing status) predicts length of stay more accurately than clinical factors alone.
# Secondary Hypothesis: The impact of comorbidities on length of stay is modified by admission type, with emergency admissions showing stronger associations between comorbidity index and extended stays.
# 3. Readmission Risk Dataset
# Research Objective: Evaluate the effectiveness of discharge planning interventions on reducing 30-day readmission rates.
# Primary Hypothesis: Patients receiving both follow-up appointment scheduling and medication reconciliation have significantly lower readmission rates compared to those receiving only one or neither intervention.
# Secondary Hypothesis: The protective effect of discharge planning interventions is greatest among patients with limited social support and multiple prior hospitalizations.
# 4. ED-to-Inpatient Conversion Dataset
# Research Objective: Identify early predictors of emergency department patients requiring inpatient admission to improve ED flow and resource utilization.
# Primary Hypothesis: A predictive model incorporating triage level, arrival method, and vital signs can identify patients requiring admission with >80% accuracy within the first hour of ED presentation.
# Secondary Hypothesis: The predictive value of chief complaint for admission varies significantly by age group, with respiratory complaints having higher admission predictive value in older patients compared to younger patients.


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def generate_patient_ids(n):
    """Generate patient IDs with proper padding"""
    return [f"P{str(i).zfill(5)}" for i in range(1, n+1)]

def generate_visit_ids(n):
    """Generate visit IDs with proper padding"""
    return [f"V{str(i).zfill(6)}" for i in range(1, n+1)]

def generate_admission_dates(n, start_date='2022-01-01', end_date='2022-12-31'):
    """Generate random admission dates within a date range"""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    date_range = (end - start).days
    
    return [start + timedelta(days=np.random.randint(0, date_range)) for _ in range(n)]

def generate_icd10_codes(n, primary=True):
    """Generate ICD-10 diagnosis codes"""
    # Common ICD-10 codes for various conditions
    common_codes = [
        # Cardiovascular
        'I21.4', 'I50.9', 'I25.10', 'I10', 'I48.91',
        # Respiratory
        'J44.9', 'J18.9', 'J45.909', 'J96.01',
        # Gastrointestinal
        'K70.30', 'K92.2', 'K85.9', 'K57.30',
        # Endocrine
        'E11.9', 'E86.0', 'E87.1',
        # Neurological
        'G40.909', 'G45.9', 'G20'
    ]
    
    # If generating primary diagnoses, weight towards more serious conditions
    if primary:
        weights = np.ones(len(common_codes))
        # Increase weight for common admission diagnoses
        weights[0:5] = 3.0  # Higher weight for cardiovascular conditions
        weights = weights / weights.sum()
        return np.random.choice(common_codes, size=n, p=weights)
    else:
        return np.random.choice(common_codes, size=n)

def generate_medications(n):
    """Generate common medications"""
    medications = [
        'Lisinopril', 'Metoprolol', 'Amlodipine', 'Atorvastatin', 'Furosemide',
        'Aspirin', 'Clopidogrel', 'Insulin', 'Metformin', 'Albuterol',
        'Prednisone', 'Levothyroxine', 'Omeprazole', 'Pantoprazole', 'Gabapentin'
    ]
    
    # Each patient might be on multiple medications
    return [', '.join(np.random.choice(medications, size=np.random.randint(1, 6), replace=False)) 
            for _ in range(n)]

def generate_comorbidity_scores(n):
    """Generate Charlson Comorbidity Index scores (0-25)"""
    # Weighted towards lower scores, but some high ones
    raw_probs = np.array([0.20, 0.18, 0.15, 0.12, 0.10, 
                          0.07, 0.05, 0.04, 0.03, 0.02,
                          0.01, 0.01, 0.01, 0.005, 0.005,
                          0.005, 0.000])
    # Normalize to ensure sum is exactly 1.0 
    probabilities = raw_probs / raw_probs.sum()
    return np.random.choice(range(17), size=n, p=probabilities)

def generate_admission_types(n):
    """Generate admission types"""
    types = ['Emergency', 'Elective', 'Urgent', 'Observation']
    return np.random.choice(types, size=n, p=[0.60, 0.25, 0.10, 0.05])

def generate_discharge_dispositions(n):
    """Generate discharge dispositions"""
    dispositions = ['Home', 'Home Health', 'SNF', 'Rehab', 'AMA', 'Expired', 'Hospice']
    return np.random.choice(dispositions, size=n, p=[0.65, 0.15, 0.10, 0.05, 0.02, 0.02, 0.01])

def generate_hospitals(n):
    """Generate hospital names"""
    hospitals = ['Memorial Hospital', 'University Medical Center', 'Community Hospital', 
                'Regional Medical Center', 'Metro General Hospital']
    return np.random.choice(hospitals, size=n)

def generate_los(admission_type, age, cci_score):
    """Generate length of stay based on factors"""
    # Base LOS
    if admission_type == 'Emergency':
        base_los = np.random.normal(4.5, 2)
    elif admission_type == 'Elective':
        base_los = np.random.normal(3, 1.5)
    elif admission_type == 'Urgent':
        base_los = np.random.normal(5, 2.5)
    else:  # Observation
        base_los = np.random.normal(1.5, 0.8)
    
    # Age adjustment - older patients stay longer
    age_factor = 1.0 + (max(0, age - 50) / 100)
    
    # CCI adjustment - sicker patients stay longer
    cci_factor = 1.0 + (cci_score / 10)
    
    # Calculate final LOS
    los = max(1, round(base_los * age_factor * cci_factor))
    
    return los

def generate_discharge_date(admission_date, los):
    """Generate discharge date based on admission date and LOS"""
    return admission_date + timedelta(days=los)

def generate_readmission_status(age, los, cci_score, hospital, admission_type, discharge_disposition):
    """Generate 30-day readmission status based on risk factors"""
    # Base readmission probability
    base_prob = 0.12  # 12% baseline
    
    # Age factor (older → higher risk)
    age_factor = 1.0 + (max(0, age - 65) / 100)
    
    # CCI factor (higher score → higher risk)
    cci_factor = 1.0 + (cci_score / 10)
    
    # LOS factor (longer stay → slightly higher risk, U-shaped)
    if los <= 2:
        los_factor = 1.1  # Short stays can mean premature discharge
    elif los >= 10:
        los_factor = 1.2  # Very long stays indicate complexity
    else:
        los_factor = 1.0
    
    # Hospital factor (some variation by hospital)
    hospital_factors = {
        'Memorial Hospital': 1.0,
        'University Medical Center': 0.9,  # Teaching hospital, lower readmissions
        'Community Hospital': 1.2,         # Small community hospital, higher readmissions
        'Regional Medical Center': 1.0,
        'Metro General Hospital': 1.1
    }
    hospital_factor = hospital_factors.get(hospital, 1.0)
    
    # Admission type factor
    admission_type_factors = {
        'Emergency': 1.3,     # Emergency admissions have higher readmission rates
        'Elective': 0.7,      # Planned admissions have lower rates
        'Urgent': 1.2,
        'Observation': 0.9
    }
    admission_factor = admission_type_factors.get(admission_type, 1.0)
    
    # Discharge disposition factor
    disposition_factors = {
        'Home': 1.0,
        'Home Health': 0.9,   # Support at home lowers risk
        'SNF': 0.8,           # Continued care in SNF lowers risk
        'Rehab': 0.8,         # Continued care in rehab lowers risk
        'AMA': 2.0,           # Against medical advice greatly increases risk
        'Expired': 0.0,       # Can't be readmitted if expired
        'Hospice': 0.0        # Counting hospice as not readmitted
    }
    disposition_factor = disposition_factors.get(discharge_disposition, 1.0)
    
    # Calculate final probability
    readmission_prob = base_prob * age_factor * cci_factor * los_factor * hospital_factor * admission_factor * disposition_factor
    
    # Cap probability at 90%
    readmission_prob = min(0.9, readmission_prob)
    
    # Set to 0 for deceased patients or hospice discharges
    if discharge_disposition in ['Expired', 'Hospice']:
        readmission_prob = 0
    
    # Generate readmission status
    return np.random.choice([1, 0], p=[readmission_prob, 1-readmission_prob])

def generate_time_to_readmission(readmission_status):
    """Generate days to readmission if readmitted, else NaN"""
    if readmission_status == 1:
        # Most readmissions happen in first two weeks, with peak in first week
        return int(np.random.exponential(7)) + 1
    else:
        return np.nan

def generate_insurance(n):
    """Generate insurance types"""
    insurance_types = ['Medicare', 'Medicaid', 'Private', 'Self-Pay', 'Other']
    return np.random.choice(insurance_types, size=n, p=[0.45, 0.15, 0.30, 0.05, 0.05])


# -----------------------------
# MAIN DATASETS
# -----------------------------

# 1. Inpatient Admissions Dataset
def create_inpatient_dataset(n=1000):
    """Create a comprehensive inpatient dataset with readmissions"""
    # Generate demographic data
    patient_ids = generate_patient_ids(n)
    visit_ids = generate_visit_ids(n)
    ages = np.random.normal(65, 15, size=n).astype(int)
    ages = np.clip(ages, 18, 100)
    genders = np.random.choice(['Male', 'Female'], size=n)
    races = np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], 
                             size=n, p=[0.65, 0.15, 0.12, 0.05, 0.03])
    
    # Generate hospital and admission data
    hospitals = generate_hospitals(n)
    admission_types = generate_admission_types(n)
    admission_dates = generate_admission_dates(n)
    primary_diagnoses = generate_icd10_codes(n, primary=True)
    secondary_diagnoses = generate_icd10_codes(n, primary=False)
    cci_scores = generate_comorbidity_scores(n)
    medications = generate_medications(n)
    insurance = generate_insurance(n)
    
    # Generate length of stay
    los_values = [generate_los(admission_types[i], ages[i], cci_scores[i]) for i in range(n)]
    
    # Generate discharge data
    discharge_dispositions = generate_discharge_dispositions(n)
    discharge_dates = [generate_discharge_date(admission_dates[i], los_values[i]) for i in range(n)]
    
    # Generate readmission data
    readmission_status = [generate_readmission_status(
        ages[i], los_values[i], cci_scores[i], hospitals[i], 
        admission_types[i], discharge_dispositions[i]) for i in range(n)]
    
    days_to_readmission = [generate_time_to_readmission(status) for status in readmission_status]
    
    # Create dataframe
    df_inpatient = pd.DataFrame({
        'visit_id': visit_ids,
        'patient_id': patient_ids,
        'age': ages,
        'gender': genders,
        'race': races,
        'hospital': hospitals,
        'insurance': insurance,
        'admission_type': admission_types,
        'admission_date': admission_dates,
        'primary_diagnosis': primary_diagnoses,
        'secondary_diagnosis': secondary_diagnoses,
        'charlson_index': cci_scores,
        'medications': medications,
        'length_of_stay': los_values,
        'discharge_date': discharge_dates,
        'discharge_disposition': discharge_dispositions,
        'readmission_30day': readmission_status,
        'days_to_readmission': days_to_readmission
    })
    
    return df_inpatient

# 2. Length of Stay Prediction Dataset
def create_los_prediction_dataset(n=800):
    """Create a dataset specifically for LOS prediction modeling"""
    # Generate basic data
    patient_ids = generate_patient_ids(n)
    ages = np.random.normal(65, 15, size=n).astype(int)
    ages = np.clip(ages, 18, 100)
    genders = np.random.choice(['Male', 'Female'], size=n)
    
    # Generate medical data with more details
    cci_scores = generate_comorbidity_scores(n)
    admission_types = generate_admission_types(n)
    primary_diagnoses = generate_icd10_codes(n, primary=True)
    
    # Generate additional predictive factors
    # Vital signs at admission
    systolic_bp = np.random.normal(130, 20, size=n).astype(int)
    diastolic_bp = np.random.normal(80, 10, size=n).astype(int)
    heart_rate = np.random.normal(80, 15, size=n).astype(int)
    respiratory_rate = np.random.normal(18, 3, size=n).astype(int)
    temperature = np.random.normal(98.6, 1, size=n)
    oxygen_saturation = np.random.normal(96, 3, size=n).astype(int)
    oxygen_saturation = np.clip(oxygen_saturation, 70, 100)
    
    # Lab values
    wbc = np.random.normal(9, 3, size=n)
    hemoglobin = np.random.normal(12, 2, size=n)
    platelets = np.random.normal(250, 100, size=n).astype(int)
    sodium = np.random.normal(138, 4, size=n).astype(int)
    potassium = np.random.normal(4.0, 0.5, size=n)
    creatinine = np.random.normal(1.1, 0.5, size=n)
    glucose = np.random.normal(120, 40, size=n).astype(int)
    
    # Social factors
    insurance = generate_insurance(n)
    # Housing security (impacts discharge planning)
    housing_status = np.random.choice(['Stable', 'Unstable', 'Homeless'], 
                                    size=n, p=[0.85, 0.10, 0.05])
    
    # Comorbidities (1=present, 0=absent)
    hypertension = np.random.binomial(1, 0.6, size=n)
    diabetes = np.random.binomial(1, 0.3, size=n)
    copd = np.random.binomial(1, 0.2, size=n)
    chf = np.random.binomial(1, 0.15, size=n)
    renal_disease = np.random.binomial(1, 0.15, size=n)
    liver_disease = np.random.binomial(1, 0.08, size=n)
    cancer = np.random.binomial(1, 0.1, size=n)
    
    # Generate target variable: length of stay
    # Base LOS from admission type
    base_los = np.zeros(n)
    for i, admission_type in enumerate(admission_types):
        if admission_type == 'Emergency':
            base_los[i] = np.random.normal(4.5, 2)
        elif admission_type == 'Elective':
            base_los[i] = np.random.normal(3, 1.5)
        elif admission_type == 'Urgent':
            base_los[i] = np.random.normal(5, 2.5)
        else:  # Observation
            base_los[i] = np.random.normal(1.5, 0.8)
    
    # Modify LOS based on factors that would realistically affect it
    # Age effect (older → longer stay)
    age_effect = (ages - 50) / 100
    age_effect = np.clip(age_effect, -0.2, 0.5)
    
    # Comorbidity effect
    comorbidity_effect = cci_scores / 10
    
    # Abnormal vitals effect
    vitals_effect = np.zeros(n)
    # High heart rate indicates acuity
    vitals_effect += np.where(heart_rate > 100, 0.2, 0)
    # Low O2 sat indicates respiratory issues
    vitals_effect += np.where(oxygen_saturation < 90, 0.3, 0)
    # Fever
    vitals_effect += np.where(temperature > 100.4, 0.2, 0)
    
    # Abnormal labs effect
    labs_effect = np.zeros(n)
    # High WBC suggests infection
    labs_effect += np.where(wbc > 12, 0.2, 0)
    # Abnormal sodium suggests metabolic issues
    labs_effect += np.where((sodium < 135) | (sodium > 145), 0.15, 0)
    # High creatinine suggests kidney problems
    labs_effect += np.where(creatinine > 1.5, 0.25, 0)
    
    # Housing status effect (harder to discharge homeless patients)
    housing_effect = np.zeros(n)
    housing_effect[housing_status == 'Unstable'] = 0.2
    housing_effect[housing_status == 'Homeless'] = 0.4
    
    # Calculate final LOS
    los = base_los * (1 + age_effect + comorbidity_effect + vitals_effect + labs_effect + housing_effect)
    los = np.maximum(1, np.round(los)).astype(int)
    
    # Create dataframe
    df_los_prediction = pd.DataFrame({
        'patient_id': patient_ids,
        'age': ages,
        'gender': genders,
        'charlson_index': cci_scores,
        'admission_type': admission_types,
        'primary_diagnosis': primary_diagnoses,
        'insurance': insurance,
        'housing_status': housing_status,
        
        # Vitals
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'heart_rate': heart_rate,
        'respiratory_rate': respiratory_rate,
        'temperature': temperature,
        'oxygen_saturation': oxygen_saturation,
        
        # Labs
        'wbc': wbc,
        'hemoglobin': hemoglobin,
        'platelets': platelets,
        'sodium': sodium,
        'potassium': potassium,
        'creatinine': creatinine,
        'glucose': glucose,
        
        # Comorbidities
        'hypertension': hypertension,
        'diabetes': diabetes,
        'copd': copd,
        'chf': chf,
        'renal_disease': renal_disease,
        'liver_disease': liver_disease,
        'cancer': cancer,
        
        # Target variable
        'length_of_stay': los
    })
    
    return df_los_prediction

# 3. Readmission Risk Dataset
def create_readmission_risk_dataset(n=1200):
    """Create a dataset for readmission risk prediction modeling"""
    # Generate patient information
    patient_ids = generate_patient_ids(n)
    ages = np.random.normal(65, 15, size=n).astype(int)
    ages = np.clip(ages, 18, 100)
    genders = np.random.choice(['Male', 'Female'], size=n)
    
    # Generate healthcare utilization history
    prior_admissions_6mo = np.random.poisson(0.5, size=n)
    prior_ed_visits_6mo = np.random.poisson(0.8, size=n)
    
    # Generate index admission details
    index_los = np.random.geometric(1/4, size=n) + 1  # Geometric distribution for LOS
    admission_types = generate_admission_types(n)
    discharge_dispositions = generate_discharge_dispositions(n)
    
    # Generate clinical risk factors
    cci_scores = generate_comorbidity_scores(n)
    polypharmacy = np.random.poisson(4, size=n)  # Number of medications
    
    # Generate additional risk factors
    # LACE score components (validated readmission risk tool)
    # L = Length of stay
    # A = Acuity of admission (already in admission_types)
    # C = Comorbidity (already in cci_scores)
    # E = ED visits in last 6 months (already in prior_ed_visits_6mo)
    
    # Additional predictive factors
    # Social determinants
    insurance = generate_insurance(n)
    social_support = np.random.choice(['None', 'Limited', 'Adequate', 'Strong'], 
                                    size=n, p=[0.05, 0.25, 0.50, 0.20])
    depression_score = np.random.poisson(3, size=n)  # PHQ-9 score approximation
    
    # Clinical factors
    had_surgery = np.random.binomial(1, 0.3, size=n)
    icu_stay = np.random.binomial(1, 0.15, size=n)
    
    # Discharge planning
    followup_scheduled = np.random.binomial(1, 0.7, size=n)
    medication_reconciliation = np.random.binomial(1, 0.85, size=n)
    
    # Generate target: 30-day readmission
    # Base probability
    base_prob = 0.12
    
    # Risk adjustments
    risk_multipliers = np.ones(n)
    
    # Prior utilization increases risk
    risk_multipliers += prior_admissions_6mo * 0.15
    risk_multipliers += prior_ed_visits_6mo * 0.08
    
    # Longer stays can mean more complex cases
    risk_multipliers += np.clip((index_los - 3) * 0.03, 0, 0.3)
    
    # Comorbidities increase risk
    risk_multipliers += cci_scores * 0.05
    
    # Polypharmacy increases risk
    risk_multipliers += np.clip((polypharmacy - 5) * 0.04, 0, 0.2)
    
    # Admission type - emergency increases risk
    risk_multipliers[admission_types == 'Emergency'] += 0.2
    
    # Discharge disposition - facility decreases risk vs home
    risk_multipliers[discharge_dispositions == 'SNF'] -= 0.1
    risk_multipliers[discharge_dispositions == 'Rehab'] -= 0.1
    risk_multipliers[discharge_dispositions == 'Home Health'] -= 0.05
    risk_multipliers[discharge_dispositions == 'AMA'] += 0.5
    risk_multipliers[discharge_dispositions == 'Expired'] = 0
    risk_multipliers[discharge_dispositions == 'Hospice'] = 0
    
    # Social factors
    risk_multipliers[social_support == 'None'] += 0.2
    risk_multipliers[social_support == 'Limited'] += 0.1
    
    # Depression increases risk
    risk_multipliers += np.clip((depression_score - 5) * 0.04, 0, 0.2)
    
    # ICU stay indicates higher acuity
    risk_multipliers[icu_stay == 1] += 0.15
    
    # Good discharge planning reduces risk
    risk_multipliers[followup_scheduled == 1] -= 0.1
    risk_multipliers[medication_reconciliation == 1] -= 0.08
    
    # Calculate final probabilities
    readmission_prob = base_prob * risk_multipliers
    readmission_prob = np.clip(readmission_prob, 0, 0.9)
    
    # Set to 0 for deceased patients or hospice discharges
    readmission_prob[discharge_dispositions == 'Expired'] = 0
    readmission_prob[discharge_dispositions == 'Hospice'] = 0
    
    # Generate readmission outcome
    readmission = np.random.binomial(1, readmission_prob)
    
    # Create dataframe
    df_readmission_risk = pd.DataFrame({
        'patient_id': patient_ids,
        'age': ages,
        'gender': genders,
        'insurance': insurance,
        
        # Index admission
        'admission_type': admission_types,
        'index_los': index_los,
        'discharge_disposition': discharge_dispositions,
        'had_surgery': had_surgery,
        'icu_stay': icu_stay,
        
        # Clinical factors
        'charlson_index': cci_scores,
        'polypharmacy_count': polypharmacy,
        
        # Utilization history
        'prior_admissions_6mo': prior_admissions_6mo,
        'prior_ed_visits_6mo': prior_ed_visits_6mo,
        
        # Social determinants
        'social_support': social_support,
        'depression_score': depression_score,
        
        # Discharge planning
        'followup_scheduled': followup_scheduled,
        'med_reconciliation_completed': medication_reconciliation,
        
        # Target
        'readmission_30day': readmission,
        'readmission_probability': readmission_prob
    })
    
    return df_readmission_risk

# 4. ED-to-Inpatient Conversion Dataset
def create_ed_to_inpatient_dataset(n=900):
    """Create a dataset modeling emergency department to inpatient conversion"""
    # Generate patient information
    patient_ids = generate_patient_ids(n)
    visit_ids = generate_visit_ids(n)
    ages = np.random.normal(60, 20, size=n).astype(int)
    ages = np.clip(ages, 18, 100)
    genders = np.random.choice(['Male', 'Female'], size=n)
    
    # Generate ED arrival information
    arrival_options = ['Self', 'Ambulance', 'Transfer']
    raw_probs = np.array([0.65, 0.30, 0.05])
    # Normalize to ensure sum is exactly 1.0
    arrival_probs = raw_probs / raw_probs.sum()
    arrival_methods = np.random.choice(arrival_options, size=n, p=arrival_probs)
    
    # Triage information
    triage_options = [1, 2, 3, 4, 5]
    raw_probs = np.array([0.05, 0.15, 0.40, 0.30, 0.10])
    # Normalize to ensure sum is exactly 1.0
    triage_probs = raw_probs / raw_probs.sum()
    triage_levels = np.random.choice(triage_options, size=n, p=triage_probs)
    
    # Vitals
    systolic_bp = np.random.normal(130, 20, size=n).astype(int)
    diastolic_bp = np.random.normal(80, 10, size=n).astype(int)
    heart_rate = np.random.normal(85, 15, size=n).astype(int)
    respiratory_rate = np.random.normal(18, 3, size=n).astype(int)
    temperature = np.random.normal(98.6, 1, size=n)
    oxygen_saturation = np.random.normal(96, 3, size=n).astype(int)
    oxygen_saturation = np.clip(oxygen_saturation, 70, 100)
    
    # Chief complaints
    complaint_options = [
        'Chest Pain', 'Shortness of Breath', 'Abdominal Pain', 
        'Fever', 'Altered Mental Status', 'Headache', 
        'Back Pain', 'Fall', 'Syncope', 'Nausea/Vomiting'
    ]
    raw_probs = np.array([0.15, 0.15, 0.20, 0.10, 0.05, 0.10, 0.10, 0.05, 0.05, 0.05])
    # Normalize to ensure sum is exactly 1.0
    complaint_probs = raw_probs / raw_probs.sum()
    chief_complaints = np.random.choice(complaint_options, size=n, p=complaint_probs)
    
    # ED interventions
    iv_fluids = np.random.binomial(1, 0.7, size=n)
    oxygen_therapy = np.random.binomial(1, 0.3, size=n)
    imaging_performed = np.random.binomial(1, 0.6, size=n)
    labs_performed = np.random.binomial(1, 0.8, size=n)
    
    # Clinical factors
    cci_scores = generate_comorbidity_scores(n)
    
    # Generate target: ED-to-Inpatient conversion
    # Base probability
    base_prob = 0.25  # 25% base admission rate
    
    # Risk adjustments
    risk_multipliers = np.ones(n)
    
    # Age increases admission risk
    age_effect = (ages - 50) / 100
    age_effect = np.clip(age_effect, -0.1, 0.4)
    risk_multipliers += age_effect
    
    # Arrival method - ambulance suggests higher acuity
    risk_multipliers[arrival_methods == 'Ambulance'] += 0.3
    risk_multipliers[arrival_methods == 'Transfer'] += 0.4
    
    # Triage level - lower is more severe (1 most severe, 5 least)
    triage_effect = np.zeros(n)
    triage_effect[triage_levels == 1] = 0.6
    triage_effect[triage_levels == 2] = 0.4
    triage_effect[triage_levels == 3] = 0.2
    triage_effect[triage_levels == 4] = -0.1
    triage_effect[triage_levels == 5] = -0.2
    risk_multipliers += triage_effect
    
    # Abnormal vitals increase admission risk
    vitals_effect = np.zeros(n)
    # High heart rate indicates acuity
    vitals_effect += np.where(heart_rate > 100, 0.2, 0)
    # Low O2 sat indicates respiratory issues
    vitals_effect += np.where(oxygen_saturation < 92, 0.3, 0)
    # Fever
    vitals_effect += np.where(temperature > 100.5, 0.2, 0)
    # Hypotension
    vitals_effect += np.where(systolic_bp < 100, 0.3, 0)
    risk_multipliers += vitals_effect
    
    # Chief complaint effect
    complaint_effect = np.zeros(n)
    high_risk_complaints = ['Chest Pain', 'Shortness of Breath', 'Altered Mental Status', 'Syncope']
    for complaint in high_risk_complaints:
        complaint_effect[chief_complaints == complaint] += 0.2
    risk_multipliers += complaint_effect
    
    # Comorbidities increase risk
    risk_multipliers += cci_scores * 0.04
    
    # ED interventions suggest higher acuity
    risk_multipliers[oxygen_therapy == 1] += 0.3
    
    # Calculate final probabilities
    admission_prob = base_prob * risk_multipliers
    admission_prob = np.clip(admission_prob, 0.05, 0.95)
    
    # Generate admission decision
    admission = np.random.binomial(1, admission_prob)
    
    # Create dataframe
    df_ed_conversion = pd.DataFrame({
        'visit_id': visit_ids,
        'patient_id': patient_ids,
        'age': ages,
        'gender': genders,
        
        # ED arrival
        'arrival_method': arrival_methods,
        'triage_level': triage_levels,
        'chief_complaint': chief_complaints,
        
        # Vitals
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'heart_rate': heart_rate,
        'respiratory_rate': respiratory_rate,
        'temperature': temperature,
        'oxygen_saturation': oxygen_saturation,
        
        # ED interventions
        'iv_fluids': iv_fluids,
        'oxygen_therapy': oxygen_therapy,
        'imaging_performed': imaging_performed,
        'labs_performed': labs_performed,
        
        # Clinical factors
        'charlson_index': cci_scores,
        
        # Target
        'admitted_to_inpatient': admission
    })
    
    return df_ed_conversion

# Create all datasets
healthcare_datasets = {
    'inpatient_admissions': create_inpatient_dataset(),
    'los_prediction': create_los_prediction_dataset(),
    'readmission_risk': create_readmission_risk_dataset(),
    'ed_to_inpatient_conversion': create_ed_to_inpatient_dataset()
} 