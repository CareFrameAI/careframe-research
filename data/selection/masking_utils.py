import re
import numpy as np
import pandas as pd
from typing import List, Any, Dict, Union
import random

def get_masked_value(val: Any) -> str:
    """
    Generate a privacy-preserving masked pattern for a value that preserves structural information.
    
    Args:
        val: The value to mask
        
    Returns:
        str: A masked pattern representing the structure of the value
    """
    if pd.isna(val):
        return "NULL"
    
    # Handle numeric values
    if isinstance(val, (int, np.int64, np.int32)):
        # For integers, show N for each digit
        return "N" * len(str(val))
    elif isinstance(val, (float, np.float64, np.float32)):
        # For floats, show N.N format
        int_part, dec_part = str(val).split('.')
        return "N" * len(int_part) + "." + "N" * len(dec_part)
    elif pd.api.types.is_numeric_dtype(type(val)):
        return f"[NUM:{type(val).__name__}]"
    
    # Handle string values
    elif isinstance(val, str):
        # Check for common healthcare data patterns first
        
        # Phone numbers
        if re.match(r'^\d{3}-\d{3}-\d{4}$', val):
            return "PHONE:NNN-NNN-NNNN"
        elif re.match(r'^\(\d{3}\)\s*\d{3}-\d{4}$', val):
            return "PHONE:(NNN)NNN-NNNN"
        elif re.match(r'^\d{10}$', val) and len(val) == 10:
            return "PHONE:NNNNNNNNNN"
        
        # ZIP/Postal codes
        elif re.match(r'^\d{5}(-\d{4})?$', val):
            return "ZIP:" + ("NNNNN" if len(val) == 5 else "NNNNN-NNNN")
        
        # Dates
        elif re.match(r'^(19|20)\d{2}-\d{1,2}-\d{1,2}$', val):
            return "DATE:YYYY-MM-DD"
        elif re.match(r'^\d{1,2}/\d{1,2}/(19|20)\d{2}$', val):
            return "DATE:MM/DD/YYYY"
        elif re.match(r'^(19|20)\d{2}/\d{1,2}/\d{1,2}$', val):
            return "DATE:YYYY/MM/DD"
        
        # Email
        elif re.match(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$', val):
            return "EMAIL"
        
        # Names
        elif re.match(r'^([A-Z][a-z]+\s)+([A-Z][a-z]+)$', val):
            return "NAME"
        
        # Medical Record Numbers (MRNs) - various formats
        elif re.match(r'^MRN\d{6,}$', val):
            return "MRN:MRNNNNNNNN"
        elif re.match(r'^[A-Z]{1,3}\d{6,}$', val):
            return f"MRN:{val[:len(val)-6]}NNNNNN"
        
        # ICD-10 codes
        elif re.match(r'^[A-Z]\d{2}(\.\d{1,2})?$', val):
            if '.' in val:
                parts = val.split('.')
                return f"ICD10:{parts[0]}.{'N' * len(parts[1])}"
            else:
                return f"ICD10:{val[0]}NN"
        
        # CPT codes
        elif re.match(r'^\d{5}$', val) and not re.match(r'^\d{5}$', val): # Not a ZIP code (context check)
            return "CPT:NNNNN"
        
        # LOINC codes
        elif re.match(r'^\d{4,5}-\d$', val):
            parts = val.split('-')
            return f"LOINC:{'N' * len(parts[0])}-N"
        
        # HCPCS codes
        elif re.match(r'^[A-Z]\d{4}$', val):
            return f"HCPCS:{val[0]}NNNN"
        
        # NDC codes (National Drug Codes)
        elif re.match(r'^\d{5}-\d{4}-\d{2}$', val) or re.match(r'^\d{5}-\d{3}-\d{2}$', val):
            parts = val.split('-')
            return f"NDC:{'N' * len(parts[0])}-{'N' * len(parts[1])}-{'N' * len(parts[2])}"
        
        # Provider NPI (National Provider Identifier)
        elif re.match(r'^\d{10}$', val) and len(val) == 10:
            return "NPI:NNNNNNNNNN"
        
        # Medical Measurements
        elif re.match(r'^\d+(\.\d+)?\s*(mg|g|kg|ml|L|cm|mm|mmHg|mm Hg)$', val):
            num_part = re.match(r'^(\d+(\.\d+)?)', val).group(1)
            unit_part = val[len(num_part):].strip()
            if '.' in num_part:
                int_part, dec_part = num_part.split('.')
                return f"MEASUREMENT:{'N' * len(int_part)}.{'N' * len(dec_part)} {unit_part}"
            else:
                return f"MEASUREMENT:{'N' * len(num_part)} {unit_part}"
        
        # For other strings, generate character-by-character pattern
        pattern = ""
        for char in val:
            if char.isupper():
                pattern += "A"
            elif char.islower():
                pattern += "a"
            elif char.isdigit():
                pattern += "N"
            elif char == ".":
                pattern += "."
            elif char == "-":
                pattern += "-"
            elif char == "_":
                pattern += "_"
            elif char == " ":
                pattern += " "
            elif char == "@":
                pattern += "@"
            elif char == "/":
                pattern += "/"
            else:
                pattern += "?"
        
        return pattern
    
    # Handle other types
    else:
        return f"[{type(val).__name__}]"


def get_masked_values_list(values: List[Any]) -> List[str]:
    """
    Generate masked patterns for a list of values.
    
    Args:
        values: List of values to mask
        
    Returns:
        List[str]: List of masked patterns
    """
    return [get_masked_value(val) for val in values]


def get_masked_distribution(df: pd.DataFrame, column: str, include_actual: bool = False) -> Dict:
    """
    Generate a masked distribution of values for a column.
    
    Args:
        df: DataFrame containing the data
        column: Column name to analyze
        include_actual: Whether to include actual values alongside masked values
        
    Returns:
        Dict: A dictionary containing the masked distribution information
    """
    # Get unique values
    unique_values = df[column].dropna().unique()
    
    # Determine column data type
    is_numeric = pd.api.types.is_numeric_dtype(df[column])
    all_strings = df[column].apply(lambda x: isinstance(x, str) if not pd.isna(x) else True).all()
    mixed_types = not (is_numeric or all_strings)
    
    column_info = {
        "type": "numeric" if is_numeric else "string" if all_strings else "mixed",
        "unique_count": len(unique_values),
        "null_count": int(df[column].isna().sum()),
        "null_percentage": float((df[column].isna().sum()/len(df))*100)
    }
    
    # Add masked value information
    if len(unique_values) <= 10:
        # Show all unique values with masking
        column_info["value_distribution"] = get_masked_values_list(unique_values)
        
        # Optionally include actual values
        if include_actual:
            column_info["actual_values"] = [str(v) for v in unique_values]
    else:
        # Take a random sample rather than first/last values
        sample_size = min(5, len(unique_values))
        sampled_values = random.sample(list(unique_values), sample_size)
        
        column_info["value_distribution"] = {
            "sample_values": get_masked_values_list(sampled_values),
            "total_unique": len(unique_values)
        }
        
        # Optionally include actual values
        if include_actual:
            column_info["actual_values"] = {
                "sample_values": [str(v) for v in sampled_values],
                "total_unique": len(unique_values)
            }
    
    return column_info


def get_column_mapping(df: pd.DataFrame, include_actual: bool = False) -> Dict:
    """
    Generate a complete column mapping for a DataFrame with masked value distributions.
    
    Args:
        df: DataFrame to analyze
        include_actual: Whether to include actual values alongside masked values
        
    Returns:
        Dict: A dictionary containing column mapping information with masked distributions
    """
    column_names = df.columns.tolist()
    
    mapping_info = {
        "column_names": column_names,
        "row_count": len(df),
        "column_mappings": {}
    }
    
    for column in column_names:
        mapping_info["column_mappings"][column] = get_masked_distribution(df, column, include_actual)
    
    return mapping_info 