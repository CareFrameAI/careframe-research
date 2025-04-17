
import numpy as np
import pandas as pd
from study_model.study_model import StudyType


def analyze_dataset_structure(df):
    """
    Analyze dataset structure to determine compatible study designs
    and potential variable roles.
    """
    analysis = {
        "has_time_variable": False,
        "has_subject_id": False,
        "has_group_variable": False,
        "has_outcome_variable": False,
        "potential_time_vars": [],
        "potential_subject_ids": [],
        "potential_group_vars": [],
        "potential_outcomes": [],
        "compatible_designs": [],
        "is_wide_format": False
    }
    
    # Check for time variables - more strictly
    time_terms = ["time", "visit", "day", "week", "month", "year", "date", "period", "session", "wave"]
    possible_time_vars = []
    outcome_with_time = []
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Check if it could be a datetime column
        if pd.api.types.is_datetime64_dtype(df[col]):
            analysis["has_time_variable"] = True
            analysis["potential_time_vars"].append(col)
            continue
            
        # Direct name matching for PURE time variables (the entire name is about time)
        if any(col_lower == term or col_lower.startswith(term + "_") or col_lower.endswith("_" + term) for term in time_terms):
            possible_time_vars.append(col)
            continue
            
        # Check for outcome measurements with embedded time information
        # Pattern like: MeasurementName_TimePoint
        if "_" in col_lower:
            parts = col_lower.split("_")
            if any(time_term in parts[-1] for time_term in time_terms):
                outcome_with_time.append(col)
    
    # Detect if we have wide format data (multiple variables measuring same thing at different times)
    base_measures = {}
    for col in outcome_with_time:
        base_name = col.split("_")[0]
        if base_name not in base_measures:
            base_measures[base_name] = []
        base_measures[base_name].append(col)
    
    # If we have multiple columns for the same base measure, this is likely wide format
    wide_format_measures = {base: cols for base, cols in base_measures.items() if len(cols) > 1}
    if wide_format_measures:
        analysis["is_wide_format"] = True
        
    # Only set time variables if they appear to be pure time variables
    if possible_time_vars:
        analysis["has_time_variable"] = True
        analysis["potential_time_vars"].extend(possible_time_vars)
    
    # Check for subject ID variables
    id_terms = ["id", "subject", "participant", "patient", "person", "respondent"]
    for col in df.columns:
        col_lower = col.lower()
        # Look for ID-like names with high cardinality
        if any(term in col_lower for term in id_terms) and df[col].nunique() > min(20, len(df) * 0.5):
            analysis["potential_subject_ids"].append(col)
    
    # Validate if any potential subject_id is actually used for repeated measures
    valid_subject_id = False
    for subject_id_col in analysis["potential_subject_ids"]:
        # Helper function to validate if a column is used for repeated measures
        def validate_within_subjects_format(df, subject_id_column):
            if subject_id_column not in df.columns:
                return False
                
            # Count observations per subject
            counts = df[subject_id_column].value_counts()
            
            # Check if at least some subjects have multiple observations
            multiple_obs_subjects = (counts > 1).sum()
            
            # If most subjects have only one observation, this is likely not a within-subjects design
            if multiple_obs_subjects < len(counts) * 0.5:  # Less than half of subjects have multiple observations
                return False
                
            return True
            
        if validate_within_subjects_format(df, subject_id_col):
            valid_subject_id = True
            break
    
    # Only set has_subject_id if we have a valid subject_id with repeated measures
    analysis["has_subject_id"] = valid_subject_id
    
    # Check for group variables
    group_terms = ["group", "treatment", "arm", "condition", "intervention", "control", "experimental"]
    for col in df.columns:
        col_lower = col.lower()
        # Group variables usually have few unique values (2-10)
        if any(term in col_lower for term in group_terms) and 1 < df[col].nunique() <= 10:
            analysis["has_group_variable"] = True
            analysis["potential_group_vars"].append(col)
        # Also check for binary/categorical variables that could be groups
        elif df[col].nunique() <= 5 and not pd.api.types.is_numeric_dtype(df[col]):
            analysis["potential_group_vars"].append(col)
    
    # Look for potential outcome variables (numeric, not already identified)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and col not in (
            analysis["potential_time_vars"] + 
            analysis["potential_subject_ids"] + 
            analysis["potential_group_vars"]
        ):
            # Outcomes usually have reasonable variance
            if df[col].std() > 0:
                analysis["has_outcome_variable"] = True
                analysis["potential_outcomes"].append(col)
    
    # Determine compatible study designs
    # For wide format data without explicit time column, avoid recommending mixed design
    if analysis["is_wide_format"] and not analysis["has_time_variable"]:
        if analysis["has_group_variable"]:
            analysis["compatible_designs"].append(StudyType.BETWEEN_SUBJECTS)
    else:
        if analysis["has_group_variable"]:
            analysis["compatible_designs"].append(StudyType.BETWEEN_SUBJECTS)
            analysis["compatible_designs"].append(StudyType.FACTORIAL)
        
        # Only suggest within-subjects designs if we have valid subject_id with repeated measures
        if analysis["has_time_variable"] and analysis["has_subject_id"]:
            analysis["compatible_designs"].append(StudyType.WITHIN_SUBJECTS)
            analysis["compatible_designs"].append(StudyType.CROSS_OVER)
            
            if analysis["has_group_variable"]:
                analysis["compatible_designs"].append(StudyType.MIXED)
        
        if analysis["has_time_variable"] and not analysis["has_subject_id"]:
            analysis["compatible_designs"].append(StudyType.REPEATED_CROSS_SECTIONAL)
    
    # If no standard designs match, default to between-subjects if we have any grouping
    if not analysis["compatible_designs"] and analysis["has_group_variable"]:
        analysis["compatible_designs"].append(StudyType.BETWEEN_SUBJECTS)
    
    # If still no compatible designs, default to the simplest option
    if not analysis["compatible_designs"]:
        analysis["compatible_designs"].append(StudyType.BETWEEN_SUBJECTS)
    
    return analysis

def sanitize_result(value):
    """Recursively sanitize result values."""
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, dict):
        return {k: sanitize_result(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [sanitize_result(item) for item in value]
    else:
        return value
