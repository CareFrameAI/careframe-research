#!/usr/bin/env python3
"""
Script to recursively scan Python files for import statements and
compile a list of required modules for requirements.txt
"""

import os
import re
import ast
import sys
from collections import defaultdict

# Patterns to recognize import statements
IMPORT_RE = re.compile(r"^\s*import\s+([^#\n]+)")
FROM_IMPORT_RE = re.compile(r"^\s*from\s+([^\.#\n]+)\s+import\s+([^#\n]+)")

# Standard library modules to exclude from requirements
STDLIB_MODULES = set([
    "abc", "argparse", "asyncio", "base64", "collections", "copy", "csv", 
    "datetime", "dataclasses", "enum", "functools", "glob", "hashlib", "io", "itertools", 
    "json", "logging", "math", "multiprocessing", "os", "pathlib", "random", 
    "re", "shutil", "signal", "socket", "subprocess", "sys", "tempfile", 
    "threading", "time", "traceback", "types", "typing", "uuid", "warnings",
    "ast", "pickle", "difflib", "secrets", "platform", "textwrap", "string",
    "contextlib", "gc", "html", "unittest", "xml", "sip"
])

# Project-specific internal modules to exclude
PROJECT_MODULES = set([
    "data", "helpers", "llms", "study_model", "literature_search", "plan", 
    "admin", "exchange", "qt_sections", "protocols", "model_builder", "privacy",
    "agent", "bionlp", "common", "hypotheses", "server", "db_ops", "qt_workers",
    "splash_loader"
])

# Module name mapping (package import name -> PyPI name with version if known)
MODULE_TO_PACKAGE = {
    # Core UI & Web Framework
    "PyQt6": "PyQt6>=6.3.0",
    "fastapi": "fastapi>=0.95.0,<0.100.0",
    "uvicorn": "uvicorn>=0.22.0,<0.23.0",
    "qasync": "qasync>=0.24.0",
    "starlette": "starlette>=0.27.0", 
    "qt_material": "qt-material>=2.14",
    "websockets": "websockets>=10.4",
    
    # Data Analysis & Scientific Libraries
    "numpy": "numpy>=1.22.0",
    "pandas": "pandas>=1.5.0",
    "scipy": "scipy>=1.8.0",
    "matplotlib": "matplotlib>=3.5.0",
    "seaborn": "seaborn>=0.12.0",
    "statsmodels": "statsmodels>=0.13.0",
    "sklearn": "scikit-learn>=1.0.0",
    "pingouin": "pingouin>=0.5.0",
    "lifelines": "lifelines>=0.27.0",
    "scikit_posthocs": "scikit-posthocs>=0.7.0",
    "pymc": "pymc>=5.0.0",
    "arviz": "arviz>=0.14.0",
    "pyqtgraph": "pyqtgraph>=0.13.0",
    
    # NLP & BioNLP Tools
    "spacy": "spacy>=3.5.0",
    "scispacy": "scispacy==0.5.3",
    "better_profanity": "better-profanity>=0.7.0",
    "presidio_analyzer": "presidio-analyzer>=2.2.0",
    "presidio_anonymizer": "presidio-anonymizer>=2.2.0",
    "scrubadub": "scrubadub>=2.0.0",
    "spellchecker": "pyspellchecker>=0.7.0,<0.8.0",
    "nltk": "nltk>=3.8.0",
    "icd10": "icd10>=1.0.0,<2.0.0",
    "sentence_transformers": "sentence-transformers>=2.2.0",
    "rank_bm25": "rank-bm25>=0.2.2",
    
    # AI & LLM libraries
    "anthropic": "anthropic>=0.15.0",
    "google.generativeai": "google-generativeai>=0.3.0",
    
    # Utilities
    "dotenv": "python-dotenv>=0.21.0",
    "requests": "requests>=2.28.0",
    "pydantic": "pydantic<2.0.0,>=1.8.2",
    "sqlalchemy": "SQLAlchemy>=2.0.0",
    "couchdb": "couchdb>=1.2",
    "cryptography": "cryptography>=40.0.0",
    "PIL": "Pillow>=10.0.0",
    "aiohttp": "aiohttp>=3.8.4",
    "faiss": "faiss-cpu>=1.7.0",
    "psutil": "psutil>=5.9.0",
    "lxml": "lxml>=4.9.0",
    "habanero": "habanero>=1.2.0",
    "cairosvg": "cairosvg>=2.7.0",
    "networkx": "networkx>=3.0.0",
    "Bio": "biopython>=1.81",
    
    # Part of standard library or not needed
    "dateutil": None,
    "setuptools": None,
    "sqlite3": None,
}

# Simplified package names for setup.py (without version specifiers)
def get_setup_py_package_name(full_req):
    """Convert a version-specific requirement to a simple package name for setup.py."""
    # Handle special cases first
    if "pydantic<2.0.0" in full_req:
        return "pydantic"
    
    # Extract the basic package name by removing version specifiers
    package_name = full_req.split('>=')[0].split('==')[0].split('<')[0].strip()
    return package_name

def find_imports_with_ast(file_path):
    """Use AST to find imports in a Python file."""
    imports = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
            
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.add(name.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0:  # Only add external imports, not relative ones
                    if node.module:  # Skip empty modules
                        imports.add(node.module.split('.')[0])
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return imports

def find_imports_with_regex(file_path):
    """Use regex to find imports in a Python file."""
    imports = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Split into lines and process each line
        lines = content.split('\n')
        for line in lines:
            # Look for "import X" statements
            match = IMPORT_RE.match(line)
            if match:
                # Get the modules and clean them
                modules = match.group(1).split(',')
                for module in modules:
                    module = module.strip().split(' as ')[0].split('.')[0]
                    imports.add(module)
                continue
                
            # Look for "from X import Y" statements
            match = FROM_IMPORT_RE.match(line)
            if match:
                module = match.group(1).strip().split('.')[0]
                imports.add(module)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return imports

def scan_directory(directory):
    """Recursively scan a directory for Python files and find imports."""
    all_imports = defaultdict(int)
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                # Try AST first for more accurate parsing
                imports = find_imports_with_ast(file_path)
                if not imports:
                    # Fall back to regex if AST fails
                    imports = find_imports_with_regex(file_path)
                
                for imp in imports:
                    all_imports[imp] += 1
    
    return all_imports

def generate_requirements(imports_counter):
    """Generate requirements.txt content from import counter."""
    requirements = []
    
    for module, count in sorted(imports_counter.items(), key=lambda x: (-x[1], x[0])):
        # Skip standard library modules and project-specific modules
        if (module in STDLIB_MODULES or module in PROJECT_MODULES or 
            module.startswith('_') or not module):
            continue
            
        # Map module name to package name if known
        if module in MODULE_TO_PACKAGE:
            package = MODULE_TO_PACKAGE[module]
            if package:  # Only add if it maps to a package
                requirements.append((package, count))
        else:
            # Use module name as is if not in mapping
            requirements.append((module, count))
    
    return requirements

def generate_setup_py_requirements(requirements):
    """Generate setup.py compatible install_requires list."""
    setup_reqs = []
    for package, _ in requirements:
        simple_name = get_setup_py_package_name(package)
        if simple_name and simple_name not in setup_reqs:
            setup_reqs.append(simple_name)
    return sorted(setup_reqs)

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = '.'
    
    print(f"Scanning directory: {directory}")
    imports_counter = scan_directory(directory)
    
    requirements = generate_requirements(imports_counter)
    
    print("\nFound external dependencies (sorted by frequency):")
    for package, count in requirements:
        print(f"{package:<35} ({count} occurrences)")
    
    # Group requirements by type for the output file
    ui_web = []
    data_science = []
    nlp = []
    ai = []
    utils = []
    
    for package, _ in requirements:
        if any(package.startswith(prefix) for prefix in ["PyQt", "fastapi", "uvicorn", "qasync", "qt-material", "websockets", "starlette"]):
            ui_web.append(package)
        elif any(package.startswith(prefix) for prefix in ["numpy", "pandas", "scipy", "matplotlib", "seaborn", "statsmodels", "scikit", "pingouin", "lifelines", "pymc", "arviz", "pyqtgraph"]):
            data_science.append(package)
        elif any(package.startswith(prefix) for prefix in ["spacy", "scispacy", "better-profanity", "presidio", "scrubadub", "pyspellchecker", "nltk", "icd10", "sentence-transformers", "rank-bm25"]):
            nlp.append(package)
        elif any(package.startswith(prefix) for prefix in ["anthropic", "google", "openai"]):
            ai.append(package)
        else:
            utils.append(package)
    
    # Write the categorized requirements.txt file
    with open('generated_requirements.txt', 'w') as f:
        f.write("# Core UI and Web Framework\n")
        f.write("\n".join(sorted(ui_web)) + "\n\n")
        
        f.write("# Data Analysis and Scientific Libraries\n")
        f.write("\n".join(sorted(data_science)) + "\n\n")
        
        f.write("# NLP and BioNLP Tools\n")
        f.write("\n".join(sorted(nlp)) + "\n\n")
        
        if ai:
            f.write("# AI and LLM Integration\n")
            f.write("\n".join(sorted(ai)) + "\n\n")
        
        f.write("# Utilities\n")
        f.write("\n".join(sorted(utils)) + "\n\n")
        
        f.write("# Note: The following packages need to be installed manually:\n")
        f.write("# spaCy models: python -m spacy download en_core_web_sm\n")
        f.write("# SciSpaCy models: \n")
        f.write("# - https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_lg-0.5.3.tar.gz\n")
        f.write("# - https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_scibert-0.5.3.tar.gz\n")
    
    # Generate setup.py compatible install_requires
    setup_reqs = generate_setup_py_requirements(requirements)
    
    # Write the setup.py compatible install_requires
    with open('setup_requires.txt', 'w') as f:
        f.write("# Generated install_requires for setup.py\n")
        f.write("install_requires=[\n")
        for req in sorted(setup_reqs):
            f.write(f"    '{req}',\n")
        f.write("]\n")
    
    print("\nWritten to generated_requirements.txt")
    print("Written to setup_requires.txt for setup.py usage")

if __name__ == "__main__":
    main() 