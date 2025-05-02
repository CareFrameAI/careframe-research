import os
import json
import pickle
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
import uuid
from datetime import datetime
from collections import namedtuple
import pandas as pd
from pandas import DataFrame

from study_model.study_model import StudyDesign
from literature_search.model_calls import GroundedReview
from protocols.builder import ProtocolBuilder

# Import the workflow JSON creation functions
from qt_sections.model_builder import (
    create_parallel_group_json,
    create_crossover_json, 
    create_factorial_json,
    create_single_arm_json,
    create_prepost_json,
    create_case_control_json
)


@dataclass
class Results:
    """Stores results data for a specific outcome measure."""
    outcome_name: str
    timepoint_index: Optional[int] = None # default; index of the timepoint in the study design 
    timepoint_comparator: Optional[int] = None # pre-post - index of the timepoint to compare to
    raw_data: Optional[Dict] = None
    analysis_results: Optional[Dict] = None
    covariates_data: Optional[Dict] = None
    groups_data: Optional[Dict] = None
    test_input_data: Optional[Dict] = None
    
    # Add new fields for statistical test results
    statistical_test_key: Optional[str] = None  # Store the test key (e.g., "INDEPENDENT_T_TEST")
    statistical_test_name: Optional[str] = None  # Human-readable test name
    test_results: Optional[Dict] = None  # Store the sanitized test results
    test_timestamp: Optional[str] = None  # When the test was run
    design_type: Optional[str] = None  # "between", "within", or "mixed"
    variables: Optional[Dict] = None  # Store variable roles (outcome, group, covariates, etc.)
    assumptions_check: Optional[Dict] = None  # Results of assumption tests
    dataset_name: Optional[str] = None  # Name of the dataset used
    # Add model diagnostics field to store additional model fit information
    model_diagnostics: Optional[Dict] = None  # For residuals, VIFs, random effects, predicted values, etc.


@dataclass
class Interpretation:
    """Stores interpretation data for a specific outcome measure."""
    outcome_name: str
    clinical_significance: Optional[str] = None
    limitations: Optional[str] = None
    future_directions: Optional[str] = None
    comparison_to_literature: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class Study:
    """Represents a complete study with all its components."""
    id: str
    name: str
    created_at: str
    updated_at: str
    study_design: StudyDesign
    available_datasets: Optional[List[Tuple[str, DataFrame]]] = field(default_factory=list)  # Will store (name, dataframe) tuples
    study_data: Optional[Dict] = None
    literature_review: Optional[GroundedReview] = None
    protocol_sections: Optional[Dict[str, Dict[str, str]]] = None
    results: Optional[List[Results]] = field(default_factory=list)
    interpretations: Optional[List[Interpretation]] = field(default_factory=list)
    
    # Fields for test selection workflow
    current_test_data: Optional[Dict] = field(default_factory=dict)  # Stores current test selection data
    current_assumption_results: Optional[Dict] = field(default_factory=dict)  # Stores assumption check results
    
    # Field for dataset metadata
    datasets_metadata: Optional[Dict[str, Dict]] = field(default_factory=dict)  # Stores metadata for datasets
    
    # Field for literature search results
    literature_searches: Optional[List[Dict]] = field(default_factory=list)  # Stores search queries and results
    
    # Field for hypotheses
    hypotheses: Optional[List[Dict]] = field(default_factory=list)  # Stores research hypotheses
    
    # Field for paper rankings
    paper_rankings: Optional[List[Dict]] = field(default_factory=list)  # Stores paper ranking results

    # Add these fields
    evidence_claims: Optional[List[Dict]] = field(default_factory=list)
    claims_hypothesis_mapping: Optional[Dict] = field(default_factory=dict)
    
    def add_hypothesis(self, hypothesis_data: Dict) -> str:
        """Add a hypothesis to this study.
        
        Args:
            hypothesis_data: Dictionary containing hypothesis data
            
        Returns:
            String hypothesis ID
            
        Raises:
            ValueError: If required fields are missing
        """
        # Initialize hypotheses list if needed
        if self.hypotheses is None:
            self.hypotheses = []
            
        # Validate required fields
        required_fields = ['title', 'null_hypothesis', 'alternative_hypothesis']
        missing_fields = [field for field in required_fields if not hypothesis_data.get(field)]
        
        if missing_fields:
            raise ValueError(f"Missing required hypothesis fields: {', '.join(missing_fields)}")
            
        # Create a copy of the hypothesis data to avoid modifying the original
        hypothesis = hypothesis_data.copy()
        
        # Ensure required metadata fields
        if 'id' not in hypothesis:
            hypothesis['id'] = str(uuid.uuid4())
        if 'created_at' not in hypothesis:
            hypothesis['created_at'] = datetime.now().isoformat()
        if 'updated_at' not in hypothesis:
            hypothesis['updated_at'] = datetime.now().isoformat()
        if 'status' not in hypothesis:
            hypothesis['status'] = 'untested'
            
        # Add to list
        self.hypotheses.append(hypothesis)
        
        # Update study timestamp
        self.updated_at = datetime.now().isoformat()
        
        return hypothesis['id']
        
    def get_hypothesis(self, hypothesis_id: str) -> Optional[Dict]:
        """Get a hypothesis by its ID.
        
        Args:
            hypothesis_id: ID of the hypothesis to retrieve
            
        Returns:
            Dictionary containing hypothesis data or None if not found
        """
        if not self.hypotheses:
            return None
            
        for hyp in self.hypotheses:
            if hyp.get('id') == hypothesis_id:
                return hyp
                
        return None


@dataclass
class Project:
    """Contains multiple studies and project metadata."""
    id: str
    name: str
    description: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    studies: Dict[str, Study] = field(default_factory=dict)
    # Project-level metadata and settings
    settings: Dict = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


class StudiesManager:
    """Manages multiple projects, each containing multiple studies."""
    
    def __init__(self):
        self.projects: Dict[str, Project] = {}
        self.debug_config: Optional[Dict] = None
        self.debug_data: Optional[Dict] = None
        self.active_project_id: Optional[str] = None
        self.active_study_id: Optional[str] = None
    
    def create_project(self, name: str, description: Optional[str] = None) -> Project:
        """Create a new project."""
        project_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        project = Project(
            id=project_id,
            name=name,
            description=description,
            created_at=timestamp,
            updated_at=timestamp
        )
        
        # Check if self.projects is a list and convert it to a dictionary if needed
        if isinstance(self.projects, list):
            # Convert list to dictionary
            projects_dict = {}
            for p in self.projects:
                if hasattr(p, 'id'):
                    projects_dict[p.id] = p
            self.projects = projects_dict
            
        self.projects[project_id] = project
        self.active_project_id = project_id
        
        return project
    
    def get_project(self, project_id: str) -> Optional[Project]:
        """Get a project by its ID."""
        # Check if self.projects is a list and handle accordingly
        if isinstance(self.projects, list):
            for project in self.projects:
                if hasattr(project, 'id') and project.id == project_id:
                    return project
            return None
        # Original behavior when self.projects is a dictionary
        return self.projects.get(project_id)
    
    def get_active_project(self) -> Optional[Project]:
        """Get the active project."""
        if self.active_project_id:
            # Check if self.projects is a list and handle accordingly
            if isinstance(self.projects, list):
                for project in self.projects:
                    if hasattr(project, 'id') and project.id == self.active_project_id:
                        return project
                return None
            # Original behavior when self.projects is a dictionary
            return self.projects.get(self.active_project_id)
        return None
    
    def set_active_project(self, project_id: str) -> bool:
        """Set the active project by ID."""
        # Check if self.projects is a list and handle accordingly
        if isinstance(self.projects, list):
            for project in self.projects:
                if hasattr(project, 'id') and project.id == project_id:
                    self.active_project_id = project_id
                    # Reset active study when changing projects
                    self.active_study_id = None
                    return True
            return False
        # Original behavior when self.projects is a dictionary
        if project_id in self.projects:
            self.active_project_id = project_id
            # Reset active study when changing projects
            self.active_study_id = None
            return True
        return False
    
    def list_projects(self) -> List[Dict]:
        """List all projects with basic information."""
        # Check if self.projects is a list and handle accordingly
        if isinstance(self.projects, list):
            return [
                {
                    "id": project.id,
                    "name": project.name,
                    "description": project.description,
                    "created_at": project.created_at,
                    "updated_at": project.updated_at,
                    "study_count": len(project.studies) if hasattr(project, 'studies') else 0,
                    "is_active": project.id == self.active_project_id
                }
                for project in self.projects if hasattr(project, 'id')
            ]
        # Original behavior when self.projects is a dictionary
        return [
            {
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "created_at": project.created_at,
                "updated_at": project.updated_at,
                "study_count": len(project.studies),
                "is_active": project.id == self.active_project_id
            }
            for project in self.projects.values()
        ]
    
    def create_study(self, name: str, study_design: StudyDesign, 
                    project_id: Optional[str] = None) -> Study:
        """Create a new study within a project."""
        # Use specified project or active project
        if project_id:
            project = self.get_project(project_id)
        else:
            project = self.get_active_project()
            
        # If no active project, create a new one
        if not project:
            project = self.create_project(f"Project for {name}")
        
        study_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Initialize study with empty results and interpretations
        study = Study(
            id=study_id,
            name=name,
            created_at=timestamp,
            updated_at=timestamp,
            study_design=study_design,
        )
        
        # Add study to project
        project.studies[study_id] = study
        project.updated_at = timestamp
        
        # Set as active study
        self.active_study_id = study_id
        
        return study
    
    def get_study(self, study_id: str, project_id: Optional[str] = None) -> Optional[Study]:
        """Get a study by its ID within a project."""
        try:
            # If project_id is specified, search only that project
            if project_id:
                project = self.get_project(project_id)
                if project:
                    # Check if project.studies is a dictionary and has the study_id as a key
                    if isinstance(project.studies, dict) and study_id in project.studies:
                        return project.studies.get(study_id)
                return None
                
            # If no project_id, first try the active project
            active_project = self.get_active_project()
            if active_project:
                # Check if active_project.studies is a dictionary and has the study_id as a key
                if isinstance(active_project.studies, dict) and study_id in active_project.studies:
                    return active_project.studies.get(study_id)
                
            # If not found in active project, search all projects
            if isinstance(self.projects, dict):
                for project in self.projects.values():
                    if isinstance(project.studies, dict) and study_id in project.studies:
                        return project.studies.get(study_id)
            elif isinstance(self.projects, list):
                for project in self.projects:
                    if hasattr(project, 'studies'):
                        if isinstance(project.studies, dict) and study_id in project.studies:
                            return project.studies.get(study_id)
                        
            return None
        except Exception as e:
            # Silently handle exceptions
            return None
    
    def get_active_study(self) -> Optional[Study]:
        """Get the active study from the active project."""
        active_project = self.get_active_project()
        if active_project and self.active_study_id:
            return active_project.studies.get(self.active_study_id)
        return None
    
    def set_active_study(self, study_id: str, project_id: Optional[str] = None) -> bool:
        """Set the active study by ID within a project."""
        # If project_id is specified, verify the study exists in that project
        if project_id:
            project = self.get_project(project_id)
            if project and study_id in project.studies:
                self.active_project_id = project_id
                self.active_study_id = study_id
                return True
            return False
            
        # If no project_id, first check the active project
        active_project = self.get_active_project()
        if active_project and study_id in active_project.studies:
            self.active_study_id = study_id
            return True
            
        # If not in active project, search all projects
        for project_id, project in self.projects.items():
            if study_id in project.studies:
                self.active_project_id = project_id
                self.active_study_id = study_id
                return True
                
        return False
    
    def list_studies(self, project_id: Optional[str] = None) -> List[Dict]:
        """List all studies within a project with basic information."""
        try:
            if project_id:
                project = self.get_project(project_id)
                if not project:
                    return []
                studies = project.studies
            else:
                # Get studies from active project
                active_project = self.get_active_project()
                if not active_project:
                    return []
                studies = active_project.studies
            
            # Check the type of studies collection and handle accordingly
            result = []
            
            if isinstance(studies, dict):
                # Dictionary of studies
                for study_id, study in studies.items():
                    try:
                        result.append({
                            "id": study.id,
                            "name": study.name,
                            "created_at": study.created_at,
                            "updated_at": study.updated_at,
                            "is_active": study.id == self.active_study_id
                        })
                    except:
                        # Skip studies that can't be properly serialized
                        pass
            elif isinstance(studies, list):
                # List of studies
                for study in studies:
                    try:
                        if hasattr(study, 'id') and hasattr(study, 'name'):
                            result.append({
                                "id": study.id,
                                "name": study.name,
                                "created_at": getattr(study, 'created_at', datetime.now().isoformat()),
                                "updated_at": getattr(study, 'updated_at', datetime.now().isoformat()),
                                "is_active": study.id == self.active_study_id
                            })
                    except:
                        # Skip studies that can't be properly serialized
                        pass
            
            return result
        except Exception as e:
            # Return empty list on any error
            return []
    
    def add_hypothesis_to_study(self, hypothesis_text: str, 
                             related_outcome: Optional[str] = None,
                             hypothesis_data: Optional[Dict] = None,
                             study_id: Optional[str] = None) -> Union[str, bool]:
        """Add a research hypothesis to a study.
        
        Args:
            hypothesis_text: Text of the hypothesis
            related_outcome: Optional related outcome variable
            hypothesis_data: Optional full hypothesis data dict
            study_id: Optional study ID (uses active study if None)
            
        Returns:
            String hypothesis ID if successful, False if failed
        """
        study = self.get_study(study_id) if study_id else self.get_active_study()
        if not study:
            return False
        
        # Initialize hypotheses list if needed
        if not hasattr(study, 'hypotheses') or study.hypotheses is None:
            study.hypotheses = []
            
        # Check if we have full hypothesis data or need to create one
        if hypothesis_data is None:
            # Create a new hypothesis with the text
            hypothesis_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            hypothesis_data = {
                'id': hypothesis_id,
                'title': hypothesis_text,
                'description': '',
                'null_hypothesis': '',
                'alternative_hypothesis': '',
                'related_outcome': related_outcome,
                'created_at': timestamp,
                'updated_at': timestamp,
                'status': 'untested'
            }
        else:
            # Ensure required fields are present in provided data
            required_fields = ['id', 'title']
            if any(field not in hypothesis_data for field in required_fields):
                print(f"Error: hypothesis_data missing required fields: {required_fields}")
                return False
                
            # Check if this hypothesis already exists (by ID)
            if any(h.get('id') == hypothesis_data.get('id') for h in study.hypotheses if isinstance(h, dict)):
                print(f"Hypothesis with ID {hypothesis_data.get('id')} already exists in study")
                return hypothesis_data.get('id')
                
            # Make sure updated_at is set
            if 'updated_at' not in hypothesis_data:
                hypothesis_data['updated_at'] = datetime.now().isoformat()
        
        # Add to list
        study.hypotheses.append(hypothesis_data)
        study.updated_at = datetime.now().isoformat()
        
        return hypothesis_data.get('id')
    
    def get_study_hypotheses(self, study_id: Optional[str] = None) -> List[Dict]:
        """Get all hypotheses for a study."""
        study = self.get_study(study_id) if study_id else self.get_active_study()
        if not study:
            print("No study found when loading hypotheses")
            return []
            
        # Debug output
        if hasattr(study, 'hypotheses'):
            hyp_type = type(study.hypotheses).__name__
            hyp_count = len(study.hypotheses) if study.hypotheses else 0
            print(f"Found {hyp_count} hypotheses in study '{study.name}', type: {hyp_type}")
            
            # If we have hypotheses, print the first one for debugging
            if hyp_count > 0:
                first_hyp = study.hypotheses[0]
                if isinstance(first_hyp, dict):
                    print(f"First hypothesis: {first_hyp.get('title', first_hyp.get('text', 'Unnamed'))}")
                    print(f"Keys: {list(first_hyp.keys())}")
                else:
                    print(f"First hypothesis is not a dict: {type(first_hyp).__name__}")
        else:
            print(f"Study '{study.name}' has no hypotheses attribute")
            return []
            
        # Ensure hypotheses is initialized
        if study.hypotheses is None:
            print("Hypotheses is None, initializing empty list")
            study.hypotheses = []
            
        # If hypotheses is not a list, try to convert it
        if not isinstance(study.hypotheses, list):
            print(f"Converting hypotheses from {type(study.hypotheses).__name__} to list")
            try:
                # Try to convert to list if it's another iterable
                study.hypotheses = list(study.hypotheses)
            except Exception as e:
                print(f"Error converting hypotheses to list: {e}")
                study.hypotheses = []
            
        # Make sure each hypothesis has required fields
        valid_hypotheses = []
        for hyp in study.hypotheses:
            if not isinstance(hyp, dict):
                print(f"Skipping non-dict hypothesis: {type(hyp).__name__}")
                continue
                
            # Check for required fields and provide defaults if missing
            if 'id' not in hyp:
                hyp['id'] = str(uuid.uuid4())
                
            if 'title' not in hyp and 'text' in hyp:
                # Support legacy format where hypothesis text was in 'text' field
                hyp['title'] = hyp['text']
                
            if 'title' not in hyp:
                hyp['title'] = "Unnamed Hypothesis"
                
            if 'created_at' not in hyp:
                hyp['created_at'] = datetime.now().isoformat()
                
            if 'updated_at' not in hyp:
                hyp['updated_at'] = datetime.now().isoformat()
                
            if 'status' not in hyp:
                hyp['status'] = 'untested'
                
            # Add to valid list
            valid_hypotheses.append(hyp)
            
        # Update study.hypotheses with the validated list
        study.hypotheses = valid_hypotheses
        print(f"Returning {len(valid_hypotheses)} validated hypotheses")
            
        return study.hypotheses
    
    def update_hypothesis(self, hypothesis_id: str, update_data: Dict, study_id: Optional[str] = None) -> bool:
        """Update a hypothesis entry with new data."""
        study = self.get_study(study_id) if study_id else self.get_active_study()
        if not study or not hasattr(study, 'hypotheses') or not study.hypotheses:
            return False
        
        # Find the hypothesis
        for i, hyp in enumerate(study.hypotheses):
            if hyp.get('id') == hypothesis_id:
                # Preserve existing data that shouldn't be overwritten
                preserved_data = {
                    'id': hyp['id'],
                    'created_at': hyp.get('created_at'),
                    'test_results': hyp.get('test_results'),
                    'literature_evidence': hyp.get('literature_evidence')
                }
                
                # Ensure update_data is a dictionary
                if not isinstance(update_data, dict):
                    # Assuming update_data is a HypothesisConfig or similar dataclass
                    update_dict = asdict(update_data) 
                else:
                    update_dict = update_data
                
                # Update with new data while preserving existing fields
                updated_hyp = {**hyp, **update_dict} # Use the converted dict
                
                # Restore preserved data
                for key, value in preserved_data.items():
                    if value is not None:
                        updated_hyp[key] = value
                
                # Always update the modified timestamp
                updated_hyp['updated_at'] = datetime.now().isoformat()
                
                # Update the hypothesis
                study.hypotheses[i] = updated_hyp
                
                # Update study timestamp
                study.updated_at = datetime.now().isoformat()
                
                return True
        
        return False

    def update_hypothesis_with_test_results(self, hypothesis_id: str, test_results: Dict, study_id: Optional[str] = None) -> bool:
        """Update a hypothesis with statistical test results."""
        study = self.get_study(study_id) if study_id else self.get_active_study()
        if not study or not hasattr(study, 'hypotheses') or not study.hypotheses:
            return False
        
        # Find the hypothesis
        for i, hyp in enumerate(study.hypotheses):
            if hyp.get('id') == hypothesis_id:
                # Add test results
                hyp['test_results'] = test_results
                hyp['test_date'] = datetime.now().isoformat()
                
                # Determine status based on test results
                p_value = test_results.get('p_value')
                alpha = hyp.get('alpha_level', 0.05)
                
                if p_value is not None:
                    # Consider both statistical and literature evidence
                    lit_status = hyp.get('literature_evidence', {}).get('status')
                    
                    if p_value < alpha:
                        if lit_status == 'rejected':
                            hyp['status'] = 'inconclusive'  # Statistical and literature evidence conflict
                        else:
                            hyp['status'] = 'confirmed'
                    else:
                        if lit_status == 'confirmed':
                            hyp['status'] = 'inconclusive'  # Statistical and literature evidence conflict
                        else:
                            hyp['status'] = 'rejected'
                
                # Update timestamp
                hyp['updated_at'] = datetime.now().isoformat()
                
                # Update the hypothesis in the list
                study.hypotheses[i] = hyp
                
                # Update study timestamp
                study.updated_at = datetime.now().isoformat()
                
                return True
        
        return False

    def update_hypothesis_with_literature(self, hypothesis_id: str, literature_evidence: Dict, study_id: Optional[str] = None) -> bool:
        """Update a hypothesis with literature evidence."""
        study = self.get_study(study_id) if study_id else self.get_active_study()
        if not study or not hasattr(study, 'hypotheses') or not study.hypotheses:
            return False
        
        # Find the hypothesis
        for i, hyp in enumerate(study.hypotheses):
            if hyp.get('id') == hypothesis_id:
                # Add literature evidence
                hyp['literature_evidence'] = literature_evidence
                
                # Update status based on both literature and statistical evidence
                lit_status = literature_evidence.get('status')
                test_results = hyp.get('test_results')
                
                if test_results:
                    # If we have statistical test results, consider both
                    p_value = test_results.get('p_value')
                    alpha = hyp.get('alpha_level', 0.05)
                    
                    if p_value is not None:
                        if p_value < alpha:
                            if lit_status == 'rejected':
                                hyp['status'] = 'inconclusive'
                            else:
                                hyp['status'] = 'confirmed'
                        else:
                            if lit_status == 'confirmed':
                                hyp['status'] = 'inconclusive'
                            else:
                                hyp['status'] = 'rejected'
                else:
                    # If no statistical results, use literature status
                    hyp['status'] = lit_status if lit_status else 'untested'
                
                # Update timestamp
                hyp['updated_at'] = datetime.now().isoformat()
                
                # Update the hypothesis in the list
                study.hypotheses[i] = hyp
                
                # Update study timestamp
                study.updated_at = datetime.now().isoformat()
                
                return True
        
        return False

    def get_hypothesis(self, hypothesis_id: str, study_id: Optional[str] = None) -> Optional[Dict]:
        """Get a specific hypothesis by ID."""
        study = self.get_study(study_id) if study_id else self.get_active_study()
        if not study or not hasattr(study, 'hypotheses') or not study.hypotheses:
            return None
        
        # Find and return the hypothesis
        for hyp in study.hypotheses:
            if hyp.get('id') == hypothesis_id:
                return hyp
        
        return None

    def get_hypothesis_evidence(self, hypothesis_id: str, study_id: Optional[str] = None) -> Optional[Dict]:
        """Get all evidence (statistical and literature) for a hypothesis."""
        hypothesis = self.get_hypothesis(hypothesis_id, study_id)
        if not hypothesis:
            return None
        
        return {
            'statistical': hypothesis.get('test_results'),
            'literature': hypothesis.get('literature_evidence'),
            'status': hypothesis.get('status'),
            'updated_at': hypothesis.get('updated_at')
        }
    
    def delete_hypothesis(self, hypothesis_id: str, study_id: Optional[str] = None) -> bool:
        """Delete a hypothesis from a study."""
        study = self.get_study(study_id) if study_id else self.get_active_study()
        if not study or not hasattr(study, 'hypotheses') or not study.hypotheses:
            return False
            
        # Find and remove the hypothesis
        for i, hyp in enumerate(study.hypotheses):
            if hyp.get('id') == hypothesis_id:
                study.hypotheses.pop(i)
                
                # Update study timestamp
                study.updated_at = datetime.now().isoformat()
                
                return True
                
        return False
    
    def add_papers_to_hypothesis(self, hypothesis_id: str, papers: List[Dict], 
                               study_id: Optional[str] = None) -> bool:
        """Add supporting papers to a hypothesis."""
        study = self.get_study(study_id) if study_id else self.get_active_study()
        if not study or not hasattr(study, 'hypotheses') or not study.hypotheses:
            return False
            
        # Find the hypothesis
        for hyp in study.hypotheses:
            if hyp.get('id') == hypothesis_id:
                # Initialize supporting_papers if needed
                if 'supporting_papers' not in hyp or hyp['supporting_papers'] is None:
                    hyp['supporting_papers'] = []
                    
                # Add papers (checking for duplicates by ID)
                existing_ids = {p.get('id') for p in hyp['supporting_papers']}
                for paper in papers:
                    if paper.get('id') not in existing_ids:
                        hyp['supporting_papers'].append(paper)
                        existing_ids.add(paper.get('id'))
                        
                # Update study timestamp
                study.updated_at = datetime.now().isoformat()
                
                return True
                
        return False
    
    def add_dataset_to_active_study(self, dataset_name: str, dataframe: DataFrame, metadata: Optional[Dict] = None) -> bool:
        """
        Add a dataset to the active study. If no active study exists, create one.
        
        Args:
            dataset_name: Name to identify the dataset
            dataframe: The pandas DataFrame to add
            metadata: Optional metadata dictionary for the dataset
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Get the active study or create one if it doesn't exist
        study = self.get_active_study()
        if not study:
            # We need a StudyDesign to create a study, so create a minimal one
            from study_model.study_model import StudyDesign
            study_design = StudyDesign(
                study_id=str(uuid.uuid4()),
            )
            study = self.create_study(f"Study with {dataset_name}", study_design)
        
        # Initialize available_datasets if needed
        if not hasattr(study, 'available_datasets') or study.available_datasets is None:
            study.available_datasets = []
        
        # Create a dictionary to store dataset information
        dataset_entry = {
            'name': dataset_name,
            'data': dataframe,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        # Check if dataset already exists (by name)
        for i, existing_dataset in enumerate(study.available_datasets):
            if isinstance(existing_dataset, dict) and existing_dataset.get('name') == dataset_name:
                # Update existing dataset
                dataset_entry['created_at'] = existing_dataset.get('created_at', dataset_entry['created_at'])
                study.available_datasets[i] = dataset_entry
                break
        else:
            # Add new dataset
            study.available_datasets.append(dataset_entry)
        
        # Update the timestamp
        study.updated_at = datetime.now().isoformat()
        
        return True
    
    def remove_dataset_from_active_study(self, dataset_name: str) -> bool:
        """
        Remove a dataset from the active study by name.
        
        Args:
            dataset_name: Name of the dataset to remove
            
        Returns:
            bool: True if successful, False if dataset not found or no active study
        """
        study = self.get_active_study()
        if not study:
            return False
        
        # Find the dataset by name
        for i, dataset in enumerate(study.available_datasets):
            # Handle both dict and namedtuple format for backward compatibility
            name = dataset.get('name') if isinstance(dataset, dict) else dataset.name
            if name == dataset_name:
                # Remove the dataset
                study.available_datasets.pop(i)
                # Update the timestamp
                study.updated_at = datetime.now().isoformat()
                return True
        
        # Dataset not found
        return False
    
    def rename_dataset_in_active_study(self, old_name: str, new_name: str) -> bool:
        """
        Rename a dataset in the active study.
        
        Args:
            old_name: Current name of the dataset
            new_name: New name for the dataset
            
        Returns:
            bool: True if successful, False if dataset not found or no active study
        """
        study = self.get_active_study()
        if not study:
            return False
        
        # Find the dataset by name
        for i, dataset in enumerate(study.available_datasets):
            if dataset.name == old_name:
                # Create a new named tuple with the updated name but same data
                DatasetEntry = namedtuple('DatasetEntry', ['name', 'data'])
                updated_dataset = DatasetEntry(name=new_name, data=dataset.data)
                
                # Replace the dataset
                study.available_datasets[i] = updated_dataset
                
                # Update the timestamp
                study.updated_at = datetime.now().isoformat()
                return True
        
        # Dataset not found
        return False
    
    def update_dataset_in_active_study(self, dataset_name: str, dataframe: DataFrame, metadata: Optional[Dict] = None) -> bool:
        """
        Update an existing dataset in the active study with new data.
        
        Args:
            dataset_name: Name of the dataset to update
            dataframe: The new pandas DataFrame
            metadata: Optional metadata dictionary to update
            
        Returns:
            bool: True if successful, False if dataset not found or no active study
        """
        study = self.get_active_study()
        if not study:
            return False
            
        # Find the dataset by name
        for i, dataset in enumerate(study.available_datasets):
            # Handle different data structures
            if isinstance(dataset, dict):
                name = dataset.get('name')
                if name == dataset_name:
                    # Update existing metadata if provided 
                    updated_metadata = dataset.get('metadata', {})
                    if metadata:
                        updated_metadata.update(metadata)
                    
                    # Create updated dataset entry
                    updated_dataset = {
                        'name': dataset_name,
                        'data': dataframe,
                        'created_at': dataset.get('created_at', datetime.now().isoformat()),
                        'updated_at': datetime.now().isoformat(),
                        'metadata': updated_metadata
                    }
                    
                    # Replace the dataset
                    study.available_datasets[i] = updated_dataset
                    
                    # Update the timestamp
                    study.updated_at = datetime.now().isoformat()
                    return True
            elif isinstance(dataset, tuple) and len(dataset) >= 2:
                name = dataset[0]
                if name == dataset_name:
                    # Try to get metadata from tuple structure
                    updated_metadata = None
                    if len(dataset) > 2:
                        updated_metadata = dataset[2] if len(dataset) > 2 else None
                    if updated_metadata is not None:
                        # Update existing metadata if provided 
                        updated_metadata.update(metadata or {})
                        
                        # Create updated dataset entry
                        updated_dataset = (name, dataframe, updated_metadata)
                        
                        # Replace the dataset
                        study.available_datasets[i] = updated_dataset
                    
                    # Update the timestamp
                    study.updated_at = datetime.now().isoformat()
                    return True
            # Add handling for other types here if needed
        
        # Dataset not found - add it as a new dataset
        return self.add_dataset_to_active_study(dataset_name, dataframe, metadata)

    def update_dataset_metadata(self, dataset_name: str, metadata: Dict) -> bool:
        """
        Update metadata for a dataset in the active study.
        
        Args:
            dataset_name: Name of the dataset
            metadata: Updated metadata dictionary
            
        Returns:
            bool: True if successful, False if the dataset was not found
        """
        study = self.get_active_study()
        if not study:
            return False
        
        # Ensure metadata is a dictionary
        if metadata is None:
            metadata = {}
        
        if not isinstance(metadata, dict):
            # If metadata is not a dictionary (e.g., it's a list), convert to dict
            try:
                metadata_dict = {'raw_data': metadata}
                metadata = metadata_dict
            except:
                # If conversion fails, use empty dict
                metadata = {}
                
        # Find the dataset by name
        for i, dataset in enumerate(study.available_datasets):
            # Handle different data structures
            if isinstance(dataset, dict):
                name = dataset.get('name')
                if name == dataset_name:
                    # Update the metadata
                    dataset['metadata'] = metadata
                    dataset['updated_at'] = datetime.now().isoformat()
                    
                    # Replace the dataset in the list
                    study.available_datasets[i] = dataset
                    
                    # Update the timestamp
                    study.updated_at = datetime.now().isoformat()
                    
                    return True
            elif isinstance(dataset, tuple) and len(dataset) >= 2:
                name = dataset[0]
                data = dataset[1]
                if name == dataset_name:
                    # Create a new tuple with the updated metadata
                    study.available_datasets[i] = (name, data, metadata)
                    
                    # Update the timestamp
                    study.updated_at = datetime.now().isoformat()
                    return True
        
        return False
    
    def get_datasets_from_active_study(self) -> List[tuple]:
        """
        Get all datasets from the active study.
        
        Returns:
            List of (name, dataframe) tuples, empty list if no active study
        """
        study = self.get_active_study()
        if not study:
            return []
        
        result = []
        for dataset in study.available_datasets:
            # Handle different data structures
            if isinstance(dataset, dict):
                result.append((dataset.get('name'), dataset.get('data')))
            elif isinstance(dataset, tuple) and len(dataset) >= 2:
                name = dataset[0]
                result.append((name, dataset[1]))
            # Add handling for other types here if needed
        
        return result
    
    def get_dataset_metadata(self, dataset_name: str) -> Optional[Dict]:
        """
        Get metadata for a specific dataset in the active study.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dict or None: Dataset metadata if found, None otherwise
        """
        study = self.get_active_study()
        if not study:
            return None
        
        # Find the dataset by name
        for dataset in study.available_datasets:
            # Handle different data structures
            if isinstance(dataset, dict):
                name = dataset.get('name')
                if name == dataset_name:
                    metadata = dataset.get('metadata', {})
                    if metadata is None:
                        return {}
                    return metadata
            elif isinstance(dataset, tuple) and len(dataset) >= 2:
                name = dataset[0]
                if name == dataset_name:
                    # Try to get metadata from tuple structure
                    metadata = None
                    if len(dataset) > 2:
                        metadata = dataset[2] if len(dataset) > 2 else None
                    return metadata if metadata is not None else {}
            # Add handling for other types here if needed
        
        return None
    
    def add_statistical_results_to_study(self, 
                                     outcome_name: str, 
                                     test_details: Dict, 
                                     study_id: Optional[str] = None) -> bool:
        """
        Add statistical test results to a study for a specific outcome.
        
        Args:
            outcome_name: Name of the outcome measure
            test_details: Dictionary containing test results and metadata
                Should include:
                    - test_key: The identified test key
                    - test_name: Human-readable test name
                    - results: The main test results
                    - timestamp: When the test was run
                    - design: Study design type
                    - dataset: Name of the dataset used
                    - variables: Detailed variable metadata including:
                        - outcome: Outcome variable name
                        - group: Group variable name(s)
                        - covariates: List of covariate variables
                        - subject_id: Subject ID variable
                        - time: Time variable
                        - role_definitions: Dictionary mapping variable names to roles
                    - model_diagnostics: Dict with model fit information (residuals, VIFs, etc.)
            study_id: The study ID (uses active study if None)
        
        Returns:
            bool: True if successful, False otherwise
        """
        study = self.get_study(study_id) if study_id else self.get_active_study()
        if not study:
            return False
        
        # Check if results already exist for this outcome
        existing_result = None
        for result in study.results:
            if result.outcome_name == outcome_name:
                existing_result = result
                break
        
        if not existing_result:
            # Create new Results object if none exists
            result = Results(outcome_name=outcome_name)
            study.results.append(result)
        else:
            result = existing_result
        
        # Update fields with test details
        result.statistical_test_key = test_details.get('test_key')
        result.statistical_test_name = test_details.get('test_name')
        result.test_results = test_details.get('results')
        result.test_timestamp = test_details.get('timestamp')
        result.design_type = test_details.get('design')
        result.dataset_name = test_details.get('dataset_name')
        
        # Store variable information with enhanced metadata
        if 'variables' in test_details and isinstance(test_details['variables'], dict):
            # Store the complete variables metadata structure
            result.variables = test_details['variables']
        else:
            # Backward compatibility: create variables structure from individual fields
            result.variables = {
                'outcome': test_details.get('outcome'),
                'group': test_details.get('group', []),
                'covariates': test_details.get('covariates', []),
                'subject_id': test_details.get('subject_id'),
                'time': test_details.get('time'),
                'role_definitions': {}
            }
            
            # Build role definitions if not provided
            if 'role_definitions' not in result.variables:
                role_defs = {}
                if result.variables['outcome']:
                    role_defs[result.variables['outcome']] = 'outcome'
                
                if isinstance(result.variables['group'], list):
                    for var in result.variables['group']:
                        if var:
                            role_defs[var] = 'group'
                elif result.variables['group']:
                    role_defs[result.variables['group']] = 'group'
                    
                if result.variables['subject_id']:
                    role_defs[result.variables['subject_id']] = 'subject_id'
                    
                if result.variables['time']:
                    role_defs[result.variables['time']] = 'time'
                    
                for var in result.variables.get('covariates', []):
                    if var:
                        role_defs[var] = 'covariate'
                        
                result.variables['role_definitions'] = role_defs
        
        # Store model diagnostics information
        result.model_diagnostics = test_details.get('model_diagnostics')
        
        # Extract and store assumption check results
        if 'results' in test_details and isinstance(test_details['results'], dict):
            if 'assumptions' in test_details['results']:
                result.assumptions_check = test_details['results']['assumptions']
                print(f"Stored assumptions_check: {list(result.assumptions_check.keys())}")
        
        # Update timestamp
        study.updated_at = datetime.now().isoformat()
        
        return True

    def get_statistical_results(self, 
                               outcome_name: Optional[str] = None, 
                               study_id: Optional[str] = None) -> List[Dict]:
        """
        Get statistical test results from a study.
        
        Args:
            outcome_name: Name of the outcome measure (None for all outcomes)
            study_id: The study ID (uses active study if None)
        
        Returns:
            List of result dictionaries, empty list if none found
        """
        study = self.get_study(study_id) if study_id else self.get_active_study()
        if not study or not study.results:
            return []
        
        # Filter results by outcome name if specified
        filtered_results = []
        for result in study.results:
            if outcome_name and result.outcome_name != outcome_name:
                continue
            
            # Skip results that don't have statistical test data
            if not hasattr(result, 'test_results') or not result.test_results:
                continue
            
            # Convert to dictionary for easier consumption
            result_dict = {
                'outcome_name': result.outcome_name,
                'statistical_test_key': getattr(result, 'statistical_test_key', None),
                'statistical_test_name': getattr(result, 'statistical_test_name', None),
                'test_results': getattr(result, 'test_results', None),
                'test_timestamp': getattr(result, 'test_timestamp', None),
                'design_type': getattr(result, 'design_type', None),
                'variables': getattr(result, 'variables', None),
                'dataset_name': getattr(result, 'dataset_name', None),
                'model_diagnostics': getattr(result, 'model_diagnostics', None)
            }
            filtered_results.append(result_dict)
                    
        return filtered_results
    
    def store_current_test_data(self, test_data: Dict) -> bool:
        """
        Store the current test identification data from the select module.
        
        Args:
            test_data: Dictionary containing test information including:
                - test_key: The identified test key (e.g., "INDEPENDENT_T_TEST")
                - test_name: Human-readable test name
                - outcome: Name of the outcome variable
                - group: Name of the group variable(s)
                - covariates: List of covariate variable names
                - subject_id: Name of the subject ID variable
                - time: Name of the time variable
                - dataset_name: Name of the dataset being used
        
        Returns:
            bool: True if successful, False otherwise
        """
        study = self.get_active_study()
        if not study:
            return False
        
        # Store the test data in the current_test_data field
        study.current_test_data = test_data
        
        # Update timestamp
        study.updated_at = datetime.now().isoformat()
        
        return True
    
    def get_current_test_data(self) -> Optional[Dict]:
        """
        Get the current test identification data.
        
        Returns:
            Dict containing test information or None if no data exists
        """
        study = self.get_active_study()
        if not study or not hasattr(study, 'current_test_data') or not study.current_test_data:
            return None
        
        return study.current_test_data
    
    def get_assumption_results(self):
        """Get assumption check results from the active study."""
        print("\n\n==== DEBUG: get_assumption_results ====")
        active_study = self.get_active_study()
        if not active_study:
            print("DEBUG: No active study found")
            return {}
            
        # Check if the active study has results with assumption checks
        if not hasattr(active_study, 'results') or not active_study.results:
            print("DEBUG: Active study has no results")
            return {}
            
        print(f"DEBUG: Active study has {len(active_study.results)} results")
        
        # Collect all assumption results from all test results
        assumption_results = {}
        
        for i, result in enumerate(active_study.results):
            print(f"\nDEBUG: Examining result {i+1}/{len(active_study.results)}")
            print(f"DEBUG: Result outcome_name: {getattr(result, 'outcome_name', 'Unknown')}")
            print(f"DEBUG: Result test_type: {getattr(result, 'statistical_test_key', 'Unknown')}")
            
            # Check if result has test_results
            if hasattr(result, 'test_results'):
                print(f"DEBUG: Result has test_results: {type(result.test_results)}")
                if result.test_results:
                    print(f"DEBUG: test_results keys: {result.test_results.keys() if isinstance(result.test_results, dict) else 'Not a dict'}")
                    
                    # Check for assumptions in test_results
                    if isinstance(result.test_results, dict) and 'assumptions' in result.test_results:
                        assumptions = result.test_results['assumptions']
                        print(f"DEBUG: Found assumptions in test_results: {assumptions.keys() if isinstance(assumptions, dict) else 'Not a dict'}")
                        
                        # Process each assumption type
                        if isinstance(assumptions, dict):
                            for assumption_type, check_result in assumptions.items():
                                print(f"DEBUG: Processing assumption type: {assumption_type}")
                                print(f"DEBUG: Check result type: {type(check_result)}")
                                print(f"DEBUG: Check result keys: {check_result.keys() if isinstance(check_result, dict) else 'Not a dict'}")
                                
                                if assumption_type not in assumption_results:
                                    assumption_results[assumption_type] = {}
                                
                                # Use outcome name as variable name
                                var_name = result.outcome_name if hasattr(result, 'outcome_name') else "Unknown"
                                
                                # Store the result
                                assumption_results[assumption_type][var_name] = {
                                    'result': check_result.get('satisfied', False),
                                    'details': {
                                        'test_name': result.statistical_test_name if hasattr(result, 'statistical_test_name') else "Unknown",
                                        'dataset': result.dataset_name if hasattr(result, 'dataset_name') else "Unknown",
                                        'p_value': check_result.get('p_value'),
                                        'statistic': check_result.get('statistic'),
                                        'test': check_result.get('test')
                                    },
                                    'recommendation': "Review the assumption check details in the statistical test results."
                                }
                    else:
                        print(f"DEBUG: No 'assumptions' key in test_results")
                else:
                    print("DEBUG: test_results is empty")
            else:
                print("DEBUG: Result has no test_results attribute")
                
            # Check if result has assumptions_check
            if hasattr(result, 'assumptions_check'):
                print(f"DEBUG: Result has assumptions_check: {type(result.assumptions_check)}")
                if result.assumptions_check:
                    print(f"DEBUG: assumptions_check keys: {result.assumptions_check.keys() if isinstance(result.assumptions_check, dict) else 'Not a dict'}")
                    
                    # Process assumptions_check if it's a dictionary
                    if isinstance(result.assumptions_check, dict):
                        for assumption_type, check_result in result.assumptions_check.items():
                            print(f"DEBUG: Processing assumption_check type: {assumption_type}")
                            print(f"DEBUG: assumption_check result type: {type(check_result)}")
                            
                            if assumption_type not in assumption_results:
                                assumption_results[assumption_type] = {}
                            
                            # Use outcome name as variable name
                            var_name = result.outcome_name if hasattr(result, 'outcome_name') else "Unknown"
                            
                            # Store the result
                            assumption_results[assumption_type][var_name] = {
                                'result': check_result.get('satisfied', False) if isinstance(check_result, dict) else check_result,
                                'details': {
                                    'test_name': result.statistical_test_name if hasattr(result, 'statistical_test_name') else "Unknown",
                                    'dataset': result.dataset_name if hasattr(result, 'dataset_name') else "Unknown"
                                },
                                'recommendation': "Review the assumption check details in the statistical test results."
                            }
                else:
                    print("DEBUG: assumptions_check is empty")
            else:
                print("DEBUG: Result has no assumptions_check attribute")
        
        print(f"\nDEBUG: Final assumption_results keys: {assumption_results.keys()}")
        for key, value in assumption_results.items():
            print(f"DEBUG: {key} has {len(value)} results")
        
        return assumption_results
        
    def get_all_studies_assumption_results(self):
        """
        Get assumption check results from all studies.
        
        Returns:
            dict: Dictionary with study IDs as keys and assumption results as values.
                  Each value is a dictionary with assumption types as keys and results as values.
        """
        all_assumption_results = {}
        
        # Iterate through all studies
        for study_id, study in self.studies.items():
            # Skip studies without results
            if not hasattr(study, 'results') or not study.results:
                continue
                
            # Collect assumption results for this study
            study_assumption_results = {}
            
            for result in study.results:
                if hasattr(result, 'assumptions_check') and result.assumptions_check:
                    # Process each assumption check
                    for assumption_type, check_result in result.assumptions_check.items():
                        # Convert to the format expected by the assumptions widget
                        if assumption_type not in study_assumption_results:
                            study_assumption_results[assumption_type] = {}
                            
                        # The variable name might be in the result or we use a default
                        var_name = result.outcome_name if hasattr(result, 'outcome_name') else "Unknown"
                        
                        # Store the result with details
                        study_assumption_results[assumption_type][var_name] = {
                            'result': check_result,  # This should be an AssumptionResult enum value
                            'details': {
                                'study_name': study.name,
                                'test_name': result.statistical_test_name if hasattr(result, 'statistical_test_name') else "Unknown",
                                'dataset': result.dataset_name if hasattr(result, 'dataset_name') else "Unknown",
                                'timestamp': result.test_timestamp if hasattr(result, 'test_timestamp') else "Unknown"
                            },
                            'recommendation': "Review the assumption check details in the statistical test results."
                        }
            
            # Add this study's results to the overall dictionary if it has any
            if study_assumption_results:
                all_assumption_results[study_id] = {
                    'study_name': study.name,
                    'assumption_results': study_assumption_results
                }
                
        return all_assumption_results
        
    def store_assumption_results(self, assumption_results):
        """Store assumption results in the active study."""
        active_study = self.get_active_study()
        if not active_study:
            return False
            
        # This would need to be implemented based on how your Study class stores assumption results
        # For example:
        if hasattr(active_study, 'set_assumption_results'):
            return active_study.set_assumption_results(assumption_results)
            
        return False
    
    def update_test_based_on_assumptions(self, new_test_data: Dict) -> bool:
        """
        Update the test selection based on assumption check results.
        
        Args:
            new_test_data: Dictionary containing updated test information
        
        Returns:
            bool: True if successful, False otherwise
        """
        study = self.get_active_study()
        if not study:
            return False
        
        # Update the test data
        study.current_test_data = new_test_data
        
        # Update timestamp
        study.updated_at = datetime.now().isoformat()
        
        return True
    
    def encode_ordinal_column(self, dataset_name: str, column_name: str, custom_mapping: Optional[Dict] = None) -> Tuple[Optional[pd.Series], Optional[Dict]]:
        """
        Encode an ordinal column in a dataset by mapping tokens to integers.
        
        Args:
            dataset_name: Name of the dataset containing the column
            column_name: Name of the column to encode
            custom_mapping: Optional custom mapping dictionary {value: integer}. 
                           If not provided, will create mapping automatically.
                           
        Returns:
            Tuple of (encoded_series, encoding_mapping) or (None, None) if failed
        """
        study = self.get_active_study()
        if not study:
            return None, None
            
        # Find the dataset
        for dataset in study.available_datasets:
            # Handle different data structures
            if isinstance(dataset, dict):
                name = dataset.get('name')
                if name == dataset_name:
                    df = dataset.get('data')
                    if df is None or column_name not in df.columns:
                        return None, None
                        
                    # Get current metadata or initialize empty dict
                    metadata = dataset.get('metadata', {})
                    
                    # Initialize encodings dict if it doesn't exist
                    if 'encodings' not in metadata:
                        metadata['encodings'] = {}
                        
                    # Create mapping if not provided
                    if custom_mapping is None:
                        unique_values = sorted(df[column_name].unique())
                        mapping = {val: i for i, val in enumerate(unique_values)}
                    else:
                        mapping = custom_mapping
                        
                    # Store both forward and reverse mappings
                    metadata['encodings'][column_name] = {
                        'value_to_int': mapping,
                        'int_to_value': {i: val for val, i in mapping.items()},
                        'encoded_at': datetime.now().isoformat()
                    }
                    
                    # Update metadata
                    dataset['metadata'] = metadata
                    
                    # Create encoded series
                    encoded_series = df[column_name].map(mapping)
                    
                    # Update timestamp
                    study.updated_at = datetime.now().isoformat()
                    
                    return encoded_series, mapping
            elif isinstance(dataset, tuple) and len(dataset) >= 2:
                name = dataset[0]
                if name == dataset_name:
                    # Try to get metadata from tuple structure
                    metadata = None
                    if len(dataset) > 2:
                        metadata = dataset[2] if len(dataset) > 2 else None
                    if metadata is not None:
                        # Update existing metadata if provided 
                        updated_metadata = metadata.copy()
                        if custom_mapping:
                            updated_metadata.update(custom_mapping)
                        
                        # Create updated dataset entry
                        updated_dataset = (name, df, updated_metadata)
                        
                        # Replace the dataset
                        study.available_datasets[i] = updated_dataset
                        
                        # Create encoded series
                        encoded_series = df[column_name].map(mapping)
                        
                        # Update timestamp
                        study.updated_at = datetime.now().isoformat()
                        
                        return encoded_series, mapping
            # Add handling for other types here if needed
        
        # Dataset not found
        return None, None
    
    def get_ordinal_encoding(self, dataset_name: str, column_name: str) -> Optional[Dict]:
        """
        Get the ordinal encoding mapping for a column in a dataset.
        
        Args:
            dataset_name: Name of the dataset
            column_name: Name of the encoded column
            
        Returns:
            Dict containing 'value_to_int' and 'int_to_value' mappings or None if not found
        """
        metadata = self.get_dataset_metadata(dataset_name)
        if not metadata or 'encodings' not in metadata:
            return None
            
        encodings = metadata['encodings']
        return encodings.get(column_name)
    
    def save_literature_search(self, query: str, filters: Dict, papers: List[Dict], description: Optional[str] = None) -> bool:
        """
        Save literature search results to the active study.
        
        Args:
            query: The search query string
            filters: Dictionary of filters that were applied
            papers: List of paper dictionaries from the search
            description: Optional description of the search purpose/hypothesis
            
        Returns:
            bool: True if successful, False otherwise
        """
        study = self.get_active_study()
        if not study:
            return False
        
        # Initialize literature_searches list if needed
        if not hasattr(study, 'literature_searches') or study.literature_searches is None:
            study.literature_searches = []
        
        # Create a search entry
        search_entry = {
            'id': str(uuid.uuid4()),
            'query': query,
            'filters': filters,
            'description': description or f"Search on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            'timestamp': datetime.now().isoformat(),
            'papers': papers,
            'paper_count': len(papers)
        }
        
        # Add to searches list
        study.literature_searches.append(search_entry)
        
        # Update timestamp
        study.updated_at = datetime.now().isoformat()
        
        return True

    def get_literature_searches(self) -> List[Dict]:
        """
        Get all literature searches from the active study.
        
        Returns:
            List of search entries, each containing id, query, description, timestamp and paper_count.
            The papers are not included to avoid large data transfer.
        """
        study = self.get_active_study()
        if not study or not hasattr(study, 'literature_searches') or not study.literature_searches:
            return []
        
        # Return summary information for each search (without papers to reduce data size)
        return [
            {
                'id': search.get('id'),
                'query': search.get('query'),
                'filters': search.get('filters'),
                'description': search.get('description'),
                'timestamp': search.get('timestamp'),
                'paper_count': search.get('paper_count', 0)
            }
            for search in study.literature_searches
        ]

    def get_literature_search_by_id(self, search_id: str) -> Optional[Dict]:
        """
        Get a specific literature search by ID, including the papers.
        
        Args:
            search_id: The ID of the search to retrieve
            
        Returns:
            Dict containing the full search information including papers, or None if not found
        """
        study = self.get_active_study()
        if not study or not hasattr(study, 'literature_searches') or not study.literature_searches:
            return None
        
        # Find the search with the matching ID
        for search in study.literature_searches:
            if search.get('id') == search_id:
                return search
        
        return None

    def update_literature_search(self, search_id: str, update_data: Dict) -> bool:
        """
        Update a literature search entry.
        
        Args:
            search_id: The ID of the search to update
            update_data: Dictionary of fields to update (can include 'description', 'papers', etc.)
            
        Returns:
            bool: True if successful, False if not found
        """
        study = self.get_active_study()
        if not study or not hasattr(study, 'literature_searches') or not study.literature_searches:
            return False
        
        # Find and update the search
        for i, search in enumerate(study.literature_searches):
            if search.get('id') == search_id:
                # Update fields
                for key, value in update_data.items():
                    search[key] = value
                
                # Update timestamp if papers are modified
                if 'papers' in update_data:
                    search['paper_count'] = len(update_data['papers'])
                    search['timestamp'] = datetime.now().isoformat()
                
                # Update the search in the list
                study.literature_searches[i] = search
                
                # Update study timestamp
                study.updated_at = datetime.now().isoformat()
                
                return True
        
        return False

    def delete_literature_search(self, search_id: str) -> bool:
        """
        Delete a literature search entry.
        
        Args:
            search_id: The ID of the search to delete
            
        Returns:
            bool: True if successful, False if not found
        """
        study = self.get_active_study()
        if not study or not hasattr(study, 'literature_searches') or not study.literature_searches:
            return False
        
        # Find and remove the search
        for i, search in enumerate(study.literature_searches):
            if search.get('id') == search_id:
                study.literature_searches.pop(i)
                
                # Update study timestamp
                study.updated_at = datetime.now().isoformat()
                
                return True
        
        return False

    def get_paper_rankings(self) -> List[Dict]:
        """
        Get all paper rankings from the active study.
        
        Returns:
            List of ranking entries, each containing id, description, timestamp and paper_count.
            The papers are not included to reduce data size.
        """
        study = self.get_active_study()
        if not study or not hasattr(study, 'paper_rankings') or not study.paper_rankings:
            return []
        
        # Return summary information for each ranking (without papers to reduce data size)
        return [
            {
                'id': ranking.get('id'),
                'description': ranking.get('description'),
                'query': ranking.get('query'),
                'timestamp': ranking.get('timestamp'),
                'paper_count': ranking.get('paper_count', 0),
                'source_dataset': ranking.get('source_dataset', '')
            }
            for ranking in study.paper_rankings
        ]

    def get_paper_ranking_by_id(self, ranking_id: str) -> Optional[Dict]:
        """
        Get a specific paper ranking by ID, including the papers.
        
        Args:
            ranking_id: The ID of the ranking to retrieve
            
        Returns:
            Dict containing the full ranking information including papers, or None if not found
        """
        study = self.get_active_study()
        if not study or not hasattr(study, 'paper_rankings') or not study.paper_rankings:
            return None
        
        # Find the ranking with the matching ID
        for ranking in study.paper_rankings:
            if ranking.get('id') == ranking_id:
                return ranking
        
        return None

    def save_paper_ranking(self, description: str, query: str, papers: List[Dict], 
                         source_dataset: Optional[str] = None) -> bool:
        """
        Save paper ranking results to the active study.
        
        Args:
            description: Description of the ranking
            query: The query used for ranking
            papers: List of ranked paper dictionaries
            source_dataset: Name of the source dataset used
            
        Returns:
            bool: True if successful, False otherwise
        """
        study = self.get_active_study()
        if not study:
            return False
        
        # Initialize paper_rankings list if needed
        if not hasattr(study, 'paper_rankings') or study.paper_rankings is None:
            study.paper_rankings = []
        
        # Create a ranking entry
        ranking_entry = {
            'id': str(uuid.uuid4()),
            'description': description,
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'papers': papers,
            'paper_count': len(papers),
            'source_dataset': source_dataset
        }
        
        # Add to rankings list
        study.paper_rankings.append(ranking_entry)
        
        # Update timestamp
        study.updated_at = datetime.now().isoformat()
        
        return True
    
    def save_to_file(self, filepath: str) -> bool:
        """
        Save all projects and studies to a file.
        
        Args:
            filepath: Path to save the data to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create a serializable representation of the data
            data = {
                'active_project_id': self.active_project_id,
                'active_study_id': self.active_study_id,
                'projects': {}
            }
            
            # Process each project
            for project_id, project in self.projects.items():
                project_data = {
                    'id': project.id,
                    'name': project.name,
                    'description': project.description,
                    'created_at': project.created_at,
                    'updated_at': project.updated_at,
                    'settings': project.settings,
                    'tags': project.tags,
                    'studies': {}
                }
                
                # Process each study in the project
                for study_id, study in project.studies.items():
                    # Convert datasets to a serializable format
                    serialized_datasets = []
                    if hasattr(study, 'available_datasets') and study.available_datasets:
                        for dataset in study.available_datasets:
                            if isinstance(dataset, dict):
                                # Create a copy without the DataFrame
                                dataset_copy = dataset.copy()
                                if 'data' in dataset_copy:
                                    # Convert DataFrame to CSV string
                                    df = dataset_copy.pop('data')
                                    dataset_copy['data_csv'] = df.to_csv(index=False)
                                serialized_datasets.append(dataset_copy)
                            else:
                                # Handle namedtuple format
                                dataset_dict = {
                                    'name': dataset.name,
                                    'data_csv': dataset.data.to_csv(index=False)
                                }
                                serialized_datasets.append(dataset_dict)
                    
                    # Convert study design to dict
                    study_design_dict = asdict(study.study_design) if study.study_design else None
                    
                    # Create serializable study data
                    study_data = {
                        'id': study.id,
                        'name': study.name,
                        'created_at': study.created_at,
                        'updated_at': study.updated_at,
                        'study_design': study_design_dict,
                        'available_datasets': serialized_datasets,
                        'study_data': study.study_data,
                        'hypotheses': study.hypotheses,
                        'literature_searches': study.literature_searches,
                        'paper_rankings': study.paper_rankings
                    }
                    
                    # Add study to project data
                    project_data['studies'][study_id] = study_data
                
                # Add project to output data
                data['projects'][project_id] = project_data
            
            # Write to file (create directory if needed)
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
            return True
        
        except Exception as e:
            print(f"Error saving data: {e}")
            return False
    
    def load_from_file(self, filepath: str) -> bool:
        """
        Load projects and studies from a file.
        
        Args:
            filepath: Path to the data file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                return False
                
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Clear existing data
            self.projects = {}
            self.active_project_id = data.get('active_project_id')
            self.active_study_id = data.get('active_study_id')
            
            # Process each project
            for project_id, project_data in data.get('projects', {}).items():
                # Create project
                project = Project(
                    id=project_data['id'],
                    name=project_data['name'],
                    description=project_data.get('description'),
                    created_at=project_data.get('created_at', datetime.now().isoformat()),
                    updated_at=project_data.get('updated_at', datetime.now().isoformat()),
                    settings=project_data.get('settings', {}),
                    tags=project_data.get('tags', [])
                )
                
                # Process each study in the project
                for study_id, study_data in project_data.get('studies', {}).items():
                    # Convert study_design dict back to object
                    from study_model.study_model import StudyDesign
                    study_design = None
                    if study_data.get('study_design'):
                        # Create a StudyDesign object with the essential fields
                        study_design = StudyDesign(
                            study_id=study_data['id'],
                            title=study_data.get('study_design', {}).get('title', ''),
                            description=study_data.get('study_design', {}).get('description', '')
                        )
                        # Add other attributes from saved data
                        for key, value in study_data.get('study_design', {}).items():
                            if key not in ['study_id', 'title', 'description']:
                                setattr(study_design, key, value)
                    
                    # Create the study object
                    study = Study(
                        id=study_data['id'],
                        name=study_data['name'],
                        created_at=study_data.get('created_at', datetime.now().isoformat()),
                        updated_at=study_data.get('updated_at', datetime.now().isoformat()),
                        study_design=study_design,
                        study_data=study_data.get('study_data'),
                        hypotheses=study_data.get('hypotheses', []),
                        literature_searches=study_data.get('literature_searches', []),
                        paper_rankings=study_data.get('paper_rankings', [])
                    )
                    
                    # Process datasets
                    study.available_datasets = []
                    for dataset_dict in study_data.get('available_datasets', []):
                        if 'data_csv' in dataset_dict:
                            # Convert CSV string back to DataFrame
                            from io import StringIO
                            csv_data = dataset_dict.pop('data_csv')
                            df = pd.read_csv(StringIO(csv_data))
                            dataset_dict['data'] = df
                            study.available_datasets.append(dataset_dict)
                    
                    # Add study to project
                    project.studies[study_id] = study
                
                # Add project to manager
                self.projects[project_id] = project
            
            return True
                
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def save_evidence_claims(self, claims: List[Dict], mapping: Dict) -> bool:
        """Save evidence claims directly to the study."""
        study = self.get_active_study()
        if not study:
            return False
        
        study.evidence_claims = claims
        study.claims_hypothesis_mapping = mapping
        study.updated_at = datetime.now().isoformat()
        return True

    def get_evidence_claims(self) -> Tuple[Optional[List[Dict]], Optional[Dict]]:
        """Get evidence claims and their hypothesis mapping."""
        study = self.get_active_study()
        if not study:
            return None, None
        return study.evidence_claims, study.claims_hypothesis_mapping
    
    def generate_model_plan(self, study_id: Optional[str] = None) -> Optional[Dict]:
        """
        Generate a model plan from a study design that's compatible with WorkflowScene.
        
        Args:
            study_id: ID of the study to use (defaults to active study)
            
        Returns:
            Dict: A workflow JSON that can be loaded by WorkflowScene, or None if not possible
        """
        study = self.get_study(study_id) if study_id else self.get_active_study()
        if not study or not study.study_design:
            return None
            
        design = study.study_design
        
        # Calculate total sample size from arms if available
        total_sample_size = 1000  # Default value
        arm_sample_sizes = []
        
        if hasattr(design, 'arms') and isinstance(design.arms, list) and len(design.arms) > 0:
            # Sum up all arm cohort sizes
            for arm in design.arms:
                if hasattr(arm, 'cohort_size') and arm.cohort_size:
                    arm_sample_sizes.append({
                        'name': arm.name,
                        'size': arm.cohort_size,
                        'interventions': [i.name for i in arm.interventions] if hasattr(arm, 'interventions') else []
                    })
            
            # Calculate total from arms if we found any with sizes
            if arm_sample_sizes:
                total_sample_size = sum(arm['size'] for arm in arm_sample_sizes if arm['size'])
        
        # If no arm sizes, try to get from design sample size
        if not arm_sample_sizes and hasattr(design, 'sample_size') and design.sample_size:
            total_sample_size = design.sample_size
        elif not arm_sample_sizes and hasattr(design, 'total_sample_size') and design.total_sample_size:
            total_sample_size = design.total_sample_size
            
        # Determine the type of study design based on available attributes
        design_type = None
        
        # Check for design type indicators
        if hasattr(design, 'design_type') and design.design_type:
            design_type = design.design_type.lower()
        elif hasattr(design, 'groups') and isinstance(design.groups, list) and len(design.groups) > 1:
            # Multiple groups typically indicates parallel group design
            design_type = 'parallel'
        elif hasattr(design, 'periods') and design.periods > 1:
            # Multiple periods typically indicates crossover design
            design_type = 'crossover'
        elif hasattr(design, 'factors') and isinstance(design.factors, list) and len(design.factors) > 0:
            # Presence of factors indicates factorial design
            design_type = 'factorial'
        elif hasattr(design, 'arms') and isinstance(design.arms, list) and len(design.arms) == 1:
            # Single arm study
            design_type = 'single_arm'
            
        # Map the design type to the corresponding preset in WorkflowScene
        preset_map = {
            'parallel': create_parallel_group_json,
            'crossover': create_crossover_json,
            'factorial': create_factorial_json,
            'single_arm': create_single_arm_json,
            'pre_post': create_prepost_json,
            'case_control': create_case_control_json
        }
        
        # Create the basic workflow JSON - will be customized based on study data
        workflow_json = {
            "nodes": [],
            "edges": []
        }
        
        # If we can determine the design type, use the corresponding preset as a starting point
        if design_type and design_type in preset_map:
            # Create the preset JSON
            create_func = preset_map[design_type]
            if callable(create_func):
                try:
                    # Try to get the preset JSON
                    workflow_json = create_func(total_sample_size)
                except Exception as e:
                    print(f"Error creating workflow JSON from preset: {e}")
                    # Fall back to empty structure if creation fails
                    workflow_json = {"nodes": [], "edges": []}
        
        # If we don't have nodes yet from a preset, build custom workflow
        if not workflow_json.get("nodes"):
            print("Creating custom workflow structure from study design data")
            
            # Create IDs for nodes
            node_id_prefix = study.id[:8] if hasattr(study, 'id') else "study"
            node_ids = {
                "target": f"{node_id_prefix}_target",
                "eligible": f"{node_id_prefix}_eligible"
            }
            
            # Always create target population node
            workflow_json["nodes"].append({
                "id": node_ids["target"],
                "type": "target_population",
                "label": f"Population ({total_sample_size})",
                "x": 0,
                "y": 0,
                "patient_count": total_sample_size
            })
            
            # Create eligible population node
            eligible_count = sum(arm['size'] for arm in arm_sample_sizes) if arm_sample_sizes else total_sample_size
            workflow_json["nodes"].append({
                "id": node_ids["eligible"],
                "type": "eligible_population",
                "label": f"Eligible Subjects ({eligible_count})",
                "x": 300,
                "y": 0,
                "patient_count": eligible_count
            })
            
            # Create edge from target to eligible
            workflow_json["edges"].append({
                "source": node_ids["target"],
                "target": node_ids["eligible"],
                "patient_count": eligible_count,
                "label": "Screening"
            })
            
            # If we have arms, create subgroup nodes for each arm
            arm_nodes = []
            if arm_sample_sizes:
                # Position calculation for multiple arms
                arm_count = len(arm_sample_sizes)
                arm_spacing = 300 if arm_count < 3 else 200  # Adjust spacing based on number of arms
                
                for i, arm in enumerate(arm_sample_sizes):
                    # Calculate position - arrange in a row with equal spacing
                    y_pos = -((arm_count-1) * arm_spacing / 2) + (i * arm_spacing)
                    
                    # Create unique ID for this arm
                    arm_id = f"{node_id_prefix}_arm_{i}"
                    node_ids[f"arm_{i}"] = arm_id
                    
                    # Always use subgroup nodes for arms as they have proper output ports for outcomes
                    node_type = "subgroup"
                    
                    # Create node
                    arm_node = {
                        "id": arm_id,
                        "type": node_type,
                        "label": arm['name'],
                        "x": 600,
                        "y": y_pos,
                        "patient_count": arm['size'],
                        "description": ", ".join(arm['interventions']) if arm['interventions'] else ""
                    }
                    workflow_json["nodes"].append(arm_node)
                    arm_nodes.append(arm_node)
                    
                    # Create edge from eligible to arm
                    workflow_json["edges"].append({
                        "source": node_ids["eligible"],
                        "target": arm_id,
                        "patient_count": arm['size'],
                        "label": f"Allocated to {arm['name']}"
                    })
            
            # If we have outcome measures, create outcome nodes
            outcome_nodes = []
            if hasattr(design, 'outcome_measures') and design.outcome_measures:
                # Use only primary outcomes to avoid cluttering the diagram
                primary_outcomes = [
                    o for o in design.outcome_measures 
                    if hasattr(o, 'category') and o.category and o.category.value == 'primary'
                ]
                
                # If no primary outcomes, use the first few outcome measures
                if not primary_outcomes and design.outcome_measures:
                    primary_outcomes = design.outcome_measures[:2]  # Limit to 2 outcomes
                
                # Calculate position for outcomes
                outcome_count = len(primary_outcomes)
                outcome_spacing = 300 if outcome_count < 3 else 200
                
                for i, outcome in enumerate(primary_outcomes):
                    # Calculate position - arrange in a row with equal spacing
                    y_pos = -((outcome_count-1) * outcome_spacing / 2) + (i * outcome_spacing)
                    
                    # Create unique ID for this outcome
                    outcome_id = f"{node_id_prefix}_outcome_{i}"
                    node_ids[f"outcome_{i}"] = outcome_id
                    
                    # Create node
                    outcome_node = {
                        "id": outcome_id,
                        "type": "outcome",
                        "label": outcome.name,
                        "x": 900,
                        "y": y_pos,
                        "patient_count": eligible_count  # Default to all eligible patients
                    }
                    workflow_json["nodes"].append(outcome_node)
                    outcome_nodes.append(outcome_node)
                    
                    # Connect to arm nodes
                    for arm_node in arm_nodes:
                        workflow_json["edges"].append({
                            "source": arm_node["id"],
                            "target": outcome_id,
                            "patient_count": arm_node["patient_count"],
                            "label": f"Measure {outcome.name}"
                        })
            
            # If we have timepoints, create timepoint nodes
            if hasattr(design, 'timepoints') and design.timepoints:
                # Sort timepoints by order
                timepoints = sorted(design.timepoints, key=lambda t: t.order if hasattr(t, 'order') else 0)
                
                # Limit to 3 timepoints to avoid cluttering
                if len(timepoints) > 3:
                    timepoints = [
                        next((t for t in timepoints if t.point_type.value == 'baseline'), timepoints[0]),
                        timepoints[len(timepoints)//2],  # Middle timepoint
                        timepoints[-1]  # Last timepoint
                    ]
                
                # Calculate position for timepoints
                timepoint_count = len(timepoints)
                timepoint_spacing = 150
                
                for i, timepoint in enumerate(timepoints):
                    # Calculate position - arrange in a column
                    y_pos = -((timepoint_count-1) * timepoint_spacing / 2) + (i * timepoint_spacing)
                    
                    # Create unique ID for this timepoint
                    timepoint_id = f"{node_id_prefix}_timepoint_{i}"
                    node_ids[f"timepoint_{i}"] = timepoint_id
                    
                    # Get descriptive name
                    timepoint_name = timepoint.name
                    if hasattr(timepoint, 'offset_days') and timepoint.offset_days is not None:
                        timepoint_name += f" (Day {timepoint.offset_days})"
                    
                    # Create node
                    timepoint_node = {
                        "id": timepoint_id,
                        "type": "timepoint",
                        "label": timepoint_name,
                        "x": 1200,
                        "y": y_pos,
                        "patient_count": eligible_count  # Default to all eligible patients
                    }
                    workflow_json["nodes"].append(timepoint_node)
                    
                    # Connect to outcome nodes - each timepoint connects to all outcomes
                    for outcome_node in outcome_nodes:
                        workflow_json["edges"].append({
                            "source": outcome_node["id"],
                            "target": timepoint_id,
                            "patient_count": outcome_node["patient_count"],
                            "label": f"Measured at {timepoint_name}"
                        })
        
        # Ensure all node types match the expected NodeCategory enum values
        for node in workflow_json.get('nodes', []):
            # Convert StudyModel to target_population if needed
            if node.get('type') == 'StudyModel':
                node['type'] = 'target_population'
                
            # Update target population node properties
            if node.get('type') == 'target_population':
                # Move properties to top level and add study details
                if 'properties' in node:
                    props = node.pop('properties', {})
                    # Add these properties to the node directly
                    for key, value in props.items():
                        node[key] = value
                
                # Add study info
                node['label'] = study.name
                if hasattr(design, 'description') and design.description:
                    node['description'] = design.description
        
        # Extra safety check - ensure all edges reference valid nodes
        if workflow_json.get('nodes') and workflow_json.get('edges'):
            # Get all node IDs
            valid_node_ids = {node.get('id') for node in workflow_json.get('nodes', [])}
            
            # Filter edges to only include those with valid source and target IDs
            valid_edges = []
            for edge in workflow_json.get('edges', []):
                if edge.get('source') in valid_node_ids and edge.get('target') in valid_node_ids:
                    # Ensure patient_count is present and is an integer
                    if 'patient_count' in edge:
                        edge['patient_count'] = int(edge['patient_count'])
                    valid_edges.append(edge)
                else:
                    print(f"Skipping edge with invalid source/target: {edge.get('source')} -> {edge.get('target')}")
            
            # Update edges in workflow_json
            workflow_json['edges'] = valid_edges
        
        # Log what we're returning for debugging
        node_count = len(workflow_json.get('nodes', []))
        edge_count = len(workflow_json.get('edges', []))
        print(f"Returning workflow with {node_count} nodes and {edge_count} edges")
        
        return workflow_json
    