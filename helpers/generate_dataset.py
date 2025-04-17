import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string

def generate_id(prefix, length=8):
    """Generate random ID with given prefix."""
    return f"{prefix}_{''.join(random.choices(string.digits, k=length))}"

def generate_phone():
    """Generate random phone number."""
    return f"{random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"

def generate_address():
    """Generate random address."""
    streets = ['Main St', 'Oak Ave', 'Maple Rd', 'Cedar Ln', 'Pine Dr']
    return f"{random.randint(100,9999)} {random.choice(streets)}"

def create_base_dataset(size=1000):
    """Create a base dataset with standard columns."""
    now = datetime.now()
    data = {
        'patient_id': [generate_id('P', 6) for _ in range(size)],
        'patient_mrn': [generate_id('MRN', 8) for _ in range(size)],
        'date_of_birth': [
            (now - timedelta(days=random.randint(365*18, 365*90))).strftime('%Y-%m-%d')
            for _ in range(size)
        ],
        'sex': np.random.choice(['M', 'F'], size=size),
        'race': np.random.choice(['White', 'Black', 'Asian', 'Hispanic', 'Other'], size=size),
        'address': [generate_address() for _ in range(size)],
        'zip_code': np.random.randint(10000, 99999, size=size),
        'phone_number': [generate_phone() for _ in range(size)]
    }
    return pd.DataFrame(data)

def create_test_datasets():
    """Create various test datasets with different characteristics."""
    
    # 1. Clean, small dataset
    df_clean = create_base_dataset(size=100)
    df_clean.to_csv('test_clean_small.tsv', sep='\t', index=False)
    
    # 2. Large dataset with duplicates
    df_large = create_base_dataset(size=5000)
    duplicates = df_large.sample(n=1000, replace=True)
    df_large = pd.concat([df_large, duplicates], ignore_index=True)
    df_large.to_csv('test_large_duplicates.tsv', sep='\t', index=False)
    
    # 3. Dataset with different column names
    df_alt_names = create_base_dataset(size=200)
    df_alt_names = df_alt_names.rename(columns={
        'patient_id': 'pat_id',
        'patient_mrn': 'medical_record_number',
        'date_of_birth': 'birthdate',
        'phone_number': 'contact_number'
    })
    df_alt_names.to_csv('test_alt_names.tsv', sep='\t', index=False)
    
    # 4. Dataset with missing values
    df_missing = create_base_dataset(size=300)
    mask = np.random.random(df_missing.shape) < 0.2
    df_missing[mask] = np.nan
    df_missing.to_csv('test_missing_values.tsv', sep='\t', index=False)
    
    # 5. Dataset with additional columns
    df_extra = create_base_dataset(size=250)
    df_extra['blood_type'] = np.random.choice(['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-'], size=len(df_extra))
    df_extra['weight_kg'] = np.random.normal(70, 15, size=len(df_extra))
    df_extra['height_cm'] = np.random.normal(170, 20, size=len(df_extra))
    df_extra.to_csv('test_extra_columns.tsv', sep='\t', index=False)
    
    # 6. Dataset with encounter information
    df_encounters = create_base_dataset(size=150)
    encounters_per_patient = np.random.randint(1, 5, size=len(df_encounters))
    rows = []
    for idx, row in df_encounters.iterrows():
        for _ in range(encounters_per_patient[idx]):
            new_row = row.copy()
            new_row['encounter_id'] = generate_id('E', 8)
            new_row['visit_date'] = (datetime.now() - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d')
            new_row['department'] = random.choice(['Cardiology', 'Neurology', 'Internal Medicine', 'Surgery'])
            rows.append(new_row)
    df_encounters = pd.DataFrame(rows)
    df_encounters.to_csv('test_encounters.tsv', sep='\t', index=False)
    
    # 7. Dataset with mixed data quality issues
    df_messy = create_base_dataset(size=400)
    # Add some duplicates with slight variations
    duplicates = df_messy.sample(n=50)
    duplicates['address'] = duplicates['address'].apply(lambda x: x.upper())
    duplicates['phone_number'] = duplicates['phone_number'].apply(lambda x: x.replace('-', ''))
    # Add some typos in sex and race
    df_messy.loc[df_messy.sample(n=20).index, 'sex'] = ['M ', ' F', 'Male', 'Female']
    df_messy.loc[df_messy.sample(n=20).index, 'race'] = ['white', 'BLACK', 'asian', 'HISPANIC']
    df_messy = pd.concat([df_messy, duplicates], ignore_index=True)
    df_messy.to_csv('test_messy_data.tsv', sep='\t', index=False)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    create_test_datasets()
    print("Test datasets generated successfully!")
    print("\nGenerated files:")
    print("1. test_clean_small.tsv - Clean, small dataset (100 records)")
    print("2. test_large_duplicates.tsv - Large dataset with duplicates (6000 records)")
    print("3. test_alt_names.tsv - Dataset with alternative column names (200 records)")
    print("4. test_missing_values.tsv - Dataset with missing values (300 records)")
    print("5. test_extra_columns.tsv - Dataset with additional columns (250 records)")
    print("6. test_encounters.tsv - Dataset with encounter information (~450-600 records)")
    print("7. test_messy_data.tsv - Dataset with mixed data quality issues (450 records)")