from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, 
                            QTableWidgetItem, QPushButton, QLabel, QGroupBox, 
                            QSplitter, QTabWidget, QTextEdit, QComboBox, 
                            QFormLayout, QTreeWidget, QTreeWidgetItem, QDialog,
                            QInputDialog, QLineEdit, QMessageBox, QStackedWidget)
from PyQt6.QtCore import Qt, pyqtSignal, QItemSelectionModel
from PyQt6.QtGui import QFont, QBrush, QColor

import pandas as pd
import json
from datetime import datetime
import uuid
import requests
import os

class StudiesManagerWidget(QWidget):
    """Widget for viewing and managing studies and projects."""
    
    # Signals to notify when active study/project changes
    active_study_changed = pyqtSignal(str)
    active_project_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.studies_manager = None
        self.user_access = None  # Will be set by MainWindow
        self.current_user = None  # Will be updated when user logs in
        self.db_url = "http://localhost:5984"  # Default CouchDB URL
        self.db_auth = ("admin", "cfpwd")  # Default CouchDB auth
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        
        # Create project selector at the top
        project_layout = QHBoxLayout()
        project_layout.addWidget(QLabel("Current Project:"))
        self.project_combo = QComboBox()
        self.project_combo.currentIndexChanged.connect(self.on_project_selected)
        project_layout.addWidget(self.project_combo)
        
        # Buttons for project management
        self.new_project_btn = QPushButton("New Project")
        self.new_project_btn.clicked.connect(self.create_new_project)
        project_layout.addWidget(self.new_project_btn)
        
        self.edit_project_btn = QPushButton("Edit Project")
        self.edit_project_btn.clicked.connect(self.edit_project)
        project_layout.addWidget(self.edit_project_btn)
        
        # Add Save Project button
        self.save_project_btn = QPushButton("Save Project")
        self.save_project_btn.clicked.connect(self.save_current_project)
        project_layout.addWidget(self.save_project_btn)
        
        # Add Load Projects button
        self.load_projects_btn = QPushButton("Load Projects")
        self.load_projects_btn.clicked.connect(self.load_projects_from_db)
        project_layout.addWidget(self.load_projects_btn)
        
        # Add Debug button
        self.debug_btn = QPushButton("Debug Info")
        self.debug_btn.clicked.connect(self.debug_project_contents)
        project_layout.addWidget(self.debug_btn)
        
        main_layout.addLayout(project_layout)
        
        # Create a splitter for the left and right panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - list of studies
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Studies list
        studies_group = QGroupBox("Studies in Project")
        studies_layout = QVBoxLayout(studies_group)
        
        self.studies_table = QTableWidget()
        self.studies_table.setColumnCount(4)
        self.studies_table.setHorizontalHeaderLabels(["Name", "Created", "Last Updated", "Active"])
        self.studies_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.studies_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.studies_table.itemClicked.connect(self.on_study_selected)
        studies_layout.addWidget(self.studies_table)
        
        # Buttons for study actions
        buttons_layout = QHBoxLayout()
        self.set_active_btn = QPushButton("Set as Active")
        self.set_active_btn.clicked.connect(self.set_active_study)
        
        self.new_study_btn = QPushButton("New Study")
        self.new_study_btn.clicked.connect(self.create_new_study)
        
        self.save_study_btn = QPushButton("Save Study")
        self.save_study_btn.clicked.connect(self.save_current_study)
        
        self.refresh_btn = QPushButton("Refresh List")
        self.refresh_btn.clicked.connect(self.refresh_studies_list)
        
        buttons_layout.addWidget(self.new_study_btn)
        buttons_layout.addWidget(self.set_active_btn)
        buttons_layout.addWidget(self.save_study_btn)
        buttons_layout.addWidget(self.refresh_btn)
        studies_layout.addLayout(buttons_layout)
        
        left_layout.addWidget(studies_group)
        
        # Right panel - study details
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.study_details_label = QLabel("No study selected")
        self.study_details_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.study_details_label)
        
        # Stacked widget for empty state vs study content
        self.content_stack = QStackedWidget()
        right_layout.addWidget(self.content_stack)
        
        # Empty state widget
        empty_widget = QWidget()
        empty_layout = QVBoxLayout(empty_widget)
        empty_layout.addWidget(QLabel("Select a study or create a new one to get started."))
        self.content_stack.addWidget(empty_widget)
        
        # Study details widget with tabs
        study_details_widget = QWidget()
        study_details_layout = QVBoxLayout(study_details_widget)
        
        # Tabs for different aspects of a study
        self.details_tabs = QTabWidget()
        
        # Tab for study design
        self.design_tab = QWidget()
        design_layout = QVBoxLayout(self.design_tab)
        self.design_tree = QTreeWidget()
        self.design_tree.setHeaderLabels(["Property", "Value"])
        self.design_tree.setColumnWidth(0, 200)
        design_layout.addWidget(self.design_tree)
        self.details_tabs.addTab(self.design_tab, "Study Design")
        
        # Tab for hypotheses (simplified)
        self.hypotheses_tab = QWidget()
        hypotheses_layout = QVBoxLayout(self.hypotheses_tab)
        hypotheses_layout.addWidget(QLabel("Hypotheses management is handled in dedicated modules."))
        self.details_tabs.addTab(self.hypotheses_tab, "Hypotheses")
        
        # Add the existing tabs
        self.datasets_tab = QWidget()
        self.details_tabs.addTab(self.datasets_tab, "Datasets")
        
        self.results_tab = QWidget()
        self.details_tabs.addTab(self.results_tab, "Results")
        
        self.llm_tab = QWidget()
        self.details_tabs.addTab(self.llm_tab, "LLM Analyses")
        
        self.chats_tab = QWidget()
        self.details_tabs.addTab(self.chats_tab, "Chat Histories")
        
        study_details_layout.addWidget(self.details_tabs)
        self.content_stack.addWidget(study_details_widget)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 700])  # Initial sizes
        
        # Add database management buttons
        db_buttons_layout = QHBoxLayout()
        
        self.reset_db_btn = QPushButton("Reset/Create Databases")
        self.reset_db_btn.clicked.connect(self.reset_databases)
        self.reset_db_btn.setToolTip("Create or reset the project and study databases")
        
        db_buttons_layout.addWidget(self.reset_db_btn)
        db_buttons_layout.addStretch()
        
        right_layout.addLayout(db_buttons_layout)
        
    def set_user_access(self, user_access):
        """Set the user access manager and update current user."""
        self.user_access = user_access
        if self.user_access and self.user_access.current_user:
            self.current_user = self.user_access.current_user
        
    def set_studies_manager(self, studies_manager):
        """Set the studies manager and refresh the projects list."""
        self.studies_manager = studies_manager
        # Update database connection settings from studies manager if available
        if hasattr(self.studies_manager, 'db_url'):
            self.db_url = self.studies_manager.db_url
        if hasattr(self.studies_manager, 'db_auth'):
            self.db_auth = self.studies_manager.db_auth
        
        # Ensure databases exist
        self.ensure_databases_exist()
        
        # # Try to automatically load projects from database
        # print("Studies Manager Widget: Attempting to load projects on initialization")
        # try:
        #     self.load_projects_from_db()
        # except Exception as e:
        #     print(f"Error auto-loading projects: {str(e)}")
        
        # self.refresh_projects_list()
        
    def ensure_databases_exist(self):
        """Check if required databases exist and create them if they don't."""
        required_dbs = ['projects', 'studies']
        databases_created = []
        errors = []
        
        try:
            # First try local mock storage if CouchDB is not available
            mock_dir = os.path.join(os.path.expanduser("~"), '.careframe', 'mock_couchdb')
            success = True
            
            # Check for mock directories and create if needed
            for db_name in required_dbs:
                db_dir = os.path.join(mock_dir, db_name)
                docs_dir = os.path.join(db_dir, 'docs')
                if not os.path.exists(docs_dir):
                    os.makedirs(docs_dir, exist_ok=True)
                    print(f"Created mock storage directory for {db_name}")
            
            # Now try CouchDB
            try:
                # Get list of all existing databases
                response = requests.get(
                    f"{self.db_url}/_all_dbs", 
                    auth=self.db_auth,
                    timeout=1  # Use a short timeout to fail fast
                )
                
                if response.status_code == 200:
                    existing_dbs = response.json()
                    
                    for db_name in required_dbs:
                        if db_name not in existing_dbs:
                            # Create the database
                            create_response = requests.put(
                                f"{self.db_url}/{db_name}",
                                auth=self.db_auth,
                                timeout=1  # Short timeout
                            )
                            
                            if create_response.status_code in [201, 202]:
                                print(f"Created database: {db_name}")
                                databases_created.append(db_name)
                                
                                # Create necessary views and indexes
                                if db_name == 'projects':
                                    self.create_project_views(db_name)
                                elif db_name == 'studies':
                                    self.create_study_views(db_name)
                            else:
                                print(f"Error creating CouchDB database {db_name}, but mock storage is available")
                else:
                    print(f"Error getting CouchDB database list, but mock storage is available")
            except requests.RequestException as e:
                print(f"CouchDB not available: {str(e)}, but mock storage is ready")
            
            return True  # Return success since mock DB is always available
                
        except Exception as e:
            print(f"Error checking/creating databases: {str(e)}")
            # Only show an error message if this is an explicit user action
            if hasattr(self, '_show_db_errors') and self._show_db_errors:
                QMessageBox.warning(self, "Connection Error", f"Error checking/creating databases: {str(e)}")
            return False

    def create_project_views(self, db_name):
        """Create views for the projects database."""
        design_doc = {
            "_id": "_design/projects",
            "views": {
                "by_name": {
                    "map": "function(doc) { if (doc.name) { emit(doc.name, doc); } }"
                },
                "active_projects": {
                    "map": "function(doc) { if (doc.is_active === true) { emit(doc._id, doc); } }"
                }
            }
        }
        
        try:
            response = requests.put(
                f"{self.db_url}/{db_name}/_design/projects",
                auth=self.db_auth,
                json=design_doc
            )
            
            if response.status_code in [201, 202]:
                print("Created project views")
            else:
                print(f"Error creating project views: {response.text}")
                
        except requests.RequestException as e:
            print(f"Error creating project views: {str(e)}")
            
    def create_study_views(self, db_name):
        """Create views for the studies database."""
        design_doc = {
            "_id": "_design/studies",
            "views": {
                "by_name": {
                    "map": "function(doc) { if (doc.name) { emit(doc.name, doc); } }"
                },
                "by_project": {
                    "map": "function(doc) { if (doc.project_id) { emit(doc.project_id, doc); } }"
                },
                "active_studies": {
                    "map": "function(doc) { if (doc.is_active === true) { emit(doc._id, doc); } }"
                }
            }
        }
        
        try:
            response = requests.put(
                f"{self.db_url}/{db_name}/_design/studies",
                auth=self.db_auth,
                json=design_doc
            )
            
            if response.status_code in [201, 202]:
                print("Created study views")
            else:
                print(f"Error creating study views: {response.text}")
                
        except requests.RequestException as e:
            print(f"Error creating study views: {str(e)}")
        
    def refresh_projects_list(self):
        """Refresh the list of projects."""
        if not self.studies_manager:
            return
            
        # Block signals to prevent triggering currentIndexChanged
        self.project_combo.blockSignals(True)
        self.project_combo.clear()
        
        try:
            # Check if studies_manager.projects is a list or dict and handle accordingly
            projects = []
            if hasattr(self.studies_manager, 'list_projects'):
                try:
                    projects = self.studies_manager.list_projects()
                except Exception as e:
                    print(f"Error calling list_projects: {str(e)}")
                    # Fallback to direct access if list_projects fails
                    if hasattr(self.studies_manager, 'projects'):
                        if isinstance(self.studies_manager.projects, dict):
                            projects = [
                                {
                                    'id': project_id,
                                    'name': project.name if hasattr(project, 'name') else f"Project {project_id}",
                                    'study_count': len(project.studies) if hasattr(project, 'studies') else 0,
                                    'is_active': project.is_active if hasattr(project, 'is_active') else False
                                }
                                for project_id, project in self.studies_manager.projects.items()
                            ]
                        elif isinstance(self.studies_manager.projects, list):
                            projects = [
                                {
                                    'id': project.id if hasattr(project, 'id') else str(i),
                                    'name': project.name if hasattr(project, 'name') else f"Project {i}",
                                    'study_count': len(project.studies) if hasattr(project, 'studies') else 0,
                                    'is_active': project.is_active if hasattr(project, 'is_active') else False
                                }
                                for i, project in enumerate(self.studies_manager.projects)
                            ]
        
            if not projects:
                self.project_combo.addItem("No projects available")
                self.project_combo.setEnabled(False)
                self.edit_project_btn.setEnabled(False)
                self.save_project_btn.setEnabled(False)
            else:
                self.project_combo.setEnabled(True)
                self.edit_project_btn.setEnabled(True)
                self.save_project_btn.setEnabled(True)
                
                active_index = 0
                for i, project in enumerate(projects):
                    display_text = f"{project['name']} ({project['study_count']} studies)"
                    self.project_combo.addItem(display_text, project["id"])
                    
                    if project["is_active"]:
                        active_index = i
                        
                # Set the active project
                self.project_combo.setCurrentIndex(active_index)
        except Exception as e:
            print(f"Error refreshing projects list: {str(e)}")
            self.project_combo.addItem(f"Error loading projects: {str(e)}")
            self.project_combo.setEnabled(False)
            self.edit_project_btn.setEnabled(False)
            self.save_project_btn.setEnabled(False)
        
        # Unblock signals
        self.project_combo.blockSignals(False)
        
        # Refresh studies list for the selected project
        self.refresh_studies_list()
        
    def refresh_studies_list(self):
        """Refresh the list of studies for the current project."""
        if not self.studies_manager:
            return
            
        active_project_id = None
        if self.project_combo.currentData():
            active_project_id = self.project_combo.currentData()
            
        studies = self.studies_manager.list_studies(active_project_id)
        self.studies_table.setRowCount(len(studies))
        
        for i, study in enumerate(studies):
            # Set background color for active study
            bg_color = QBrush(QColor(240, 255, 240)) if study["is_active"] else None
            
            # Name
            name_item = QTableWidgetItem(study["name"])
            name_item.setData(Qt.ItemDataRole.UserRole, study["id"])  # Store ID
            if bg_color:
                name_item.setBackground(bg_color)
            self.studies_table.setItem(i, 0, name_item)
            
            # Created date
            created_date = datetime.fromisoformat(study["created_at"])
            created_item = QTableWidgetItem(created_date.strftime("%Y-%m-%d %H:%M"))
            if bg_color:
                created_item.setBackground(bg_color)
            self.studies_table.setItem(i, 1, created_item)
            
            # Updated date
            updated_date = datetime.fromisoformat(study["updated_at"])
            updated_item = QTableWidgetItem(updated_date.strftime("%Y-%m-%d %H:%M"))
            if bg_color:
                updated_item.setBackground(bg_color)
            self.studies_table.setItem(i, 2, updated_item)
            
            # Active status
            active_item = QTableWidgetItem("✓" if study["is_active"] else "")
            active_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if study["is_active"]:
                active_item.setForeground(QBrush(QColor("green")))
                if bg_color:
                    active_item.setBackground(bg_color)
            self.studies_table.setItem(i, 3, active_item)
        
        self.studies_table.resizeColumnsToContents()
        
        # Show empty state if no studies
        if len(studies) == 0:
            self.content_stack.setCurrentIndex(0)  # Show empty state
        elif self.studies_manager.get_active_study():
            # If there's an active study, select and display it
            self.content_stack.setCurrentIndex(1)  # Show study details
            self.display_study_details(self.studies_manager.get_active_study())
            
            # Select the active study in the table
            for i in range(self.studies_table.rowCount()):
                if self.studies_table.item(i, 3).text() == "✓":
                    self.studies_table.selectRow(i)
                    break
        
    def on_project_selected(self, index):
        """Handle selection of a project in the combo box."""
        if index < 0 or not self.project_combo.itemData(index):
            return
            
        project_id = self.project_combo.itemData(index)
        if self.studies_manager.set_active_project(project_id):
            self.active_project_changed.emit(project_id)
            self.refresh_studies_list()
    
    def save_project_to_db(self, project):
        """Save project to database with current user information."""
        if not self.studies_manager or not project:
            return False
            
        try:
            # Add user metadata
            if not hasattr(project, 'created_by') and self.current_user:
                project.created_by = self.current_user.get('email')
                
            if self.current_user:
                project.updated_by = self.current_user.get('email')
                
            project.updated_at = datetime.now().isoformat()
            
            # Use studies manager to save if it has the method
            if hasattr(self.studies_manager, 'save_project_to_db'):
                result = self.studies_manager.save_project_to_db(project)
                return result
            
            # Otherwise, save directly to database
            # First convert the project to a serializable dictionary
            project_data = {}
            
            # Manual serialization of common attributes
            for attr in ['id', 'name', 'description', 'created_at', 'updated_at', 
                       'created_by', 'updated_by', 'is_active']:
                if hasattr(project, attr):
                    project_data[attr] = getattr(project, attr)
            
            # Add any special database attributes
            if hasattr(project, '_id'):
                project_data['_id'] = project._id
            if hasattr(project, '_rev'):
                project_data['_rev'] = project._rev
                
            # Add type identifier for CouchDB
            project_data['type'] = 'project'
            
            # Check if project exists
            if hasattr(project, '_id') and project._id:
                response = requests.put(
                    f"{self.db_url}/projects/{project._id}",
                    auth=self.db_auth,
                    json=project_data
                )
            else:
                response = requests.post(
                    f"{self.db_url}/projects",
                    auth=self.db_auth,
                    json=project_data
                )
                
            if response.status_code in [200, 201]:
                result = response.json()
                if not hasattr(project, '_id'):
                    project._id = result.get('id')
                if not hasattr(project, '_rev'):
                    project._rev = result.get('rev')
                return True
            else:
                print(f"Error saving project: {response.text}")
                return False
                
        except Exception as e:
            print(f"Error saving project to database: {str(e)}")
            return False
    
    def delete_project_from_db(self, project_id):
        """Delete project from database."""
        if not self.studies_manager or not project_id:
            return False
            
        try:
            # Use studies manager to delete if it has the method
            if hasattr(self.studies_manager, 'delete_project'):
                result = self.studies_manager.delete_project(project_id)
                return result
                
            # First get the project to get the revision
            response = requests.get(
                f"{self.db_url}/projects/{project_id}",
                auth=self.db_auth
            )
            
            if response.status_code == 200:
                project = response.json()
                rev = project.get('_rev')
                
                # Then delete it
                delete_response = requests.delete(
                    f"{self.db_url}/projects/{project_id}?rev={rev}",
                    auth=self.db_auth
                )
                
                return delete_response.status_code in [200, 202]
            return False
                
        except Exception as e:
            print(f"Error deleting project from database: {str(e)}")
            return False
    
    def save_study_to_db(self, study):
        """Save study to database with current user information."""
        if not self.studies_manager or not study:
            return False
            
        try:
            # Add user metadata
            if not hasattr(study, 'created_by') and self.current_user:
                study.created_by = self.current_user.get('email')
                
            if self.current_user:
                study.updated_by = self.current_user.get('email')
                
            study.updated_at = datetime.now().isoformat()
            
            # Use studies manager to save if it has the method
            if hasattr(self.studies_manager, 'save_study_to_db'):
                result = self.studies_manager.save_study_to_db(study)
                return result
                
            # Otherwise, save directly to database
            # First convert the study to a serializable dictionary
            study_data = {}
            
            # Manual serialization of common attributes
            for attr in ['id', 'name', 'description', 'created_at', 'updated_at', 
                       'created_by', 'updated_by', 'is_active', 'project_id']:
                if hasattr(study, attr):
                    study_data[attr] = getattr(study, attr)
            
            # Add any special database attributes
            if hasattr(study, '_id'):
                study_data['_id'] = study._id
            if hasattr(study, '_rev'):
                study_data['_rev'] = study._rev
                
            # Add type identifier for CouchDB
            study_data['type'] = 'study'
            
            # Handle the study design separately
            if hasattr(study, 'study_design') and study.study_design:
                design_data = {}
                design = study.study_design
                
                # Serialize basic design properties
                for attr in ['study_id', 'title', 'description', 'design_type', 'created_at', 'updated_at']:
                    if hasattr(design, attr):
                        design_data[attr] = getattr(design, attr)
                
                # Handle complex objects like outcome measures, groups, etc.
                if hasattr(design, 'outcome_measures') and design.outcome_measures:
                    outcome_measures = []
                    for measure in design.outcome_measures:
                        measure_dict = {}
                        for attr in ['name', 'description', 'data_type', 'units']:
                            if hasattr(measure, attr):
                                measure_dict[attr] = getattr(measure, attr)
                        outcome_measures.append(measure_dict)
                    design_data['outcome_measures'] = outcome_measures
                
                if hasattr(design, 'groups') and design.groups:
                    groups = []
                    for group in design.groups:
                        group_dict = {}
                        for attr in ['name', 'description', 'size']:
                            if hasattr(group, attr):
                                group_dict[attr] = getattr(group, attr)
                        groups.append(group_dict)
                    design_data['groups'] = groups
                
                if hasattr(design, 'timepoints') and design.timepoints:
                    timepoints = []
                    for timepoint in design.timepoints:
                        timepoint_dict = {}
                        for attr in ['name', 'description', 'day', 'time']:
                            if hasattr(timepoint, attr):
                                timepoint_dict[attr] = getattr(timepoint, attr)
                        timepoints.append(timepoint_dict)
                    design_data['timepoints'] = timepoints
                
                # Add the serialized design to the study data
                study_data['study_design'] = design_data
            
            # Serialize any other complex objects or collections
            for complex_attr in ['hypotheses', 'literature_searches', 'results']:
                if hasattr(study, complex_attr) and getattr(study, complex_attr):
                    attr_value = getattr(study, complex_attr)
                    if isinstance(attr_value, list):
                        # Handle lists of objects
                        serialized_list = []
                        for item in attr_value:
                            if hasattr(item, '__dict__'):
                                serialized_list.append(item.__dict__)
                            else:
                                serialized_list.append(item)
                        study_data[complex_attr] = serialized_list
                    elif isinstance(attr_value, dict):
                        # Handle dictionaries directly
                        study_data[complex_attr] = attr_value
                    else:
                        # Try to get __dict__ or convert to string as fallback
                        try:
                            study_data[complex_attr] = attr_value.__dict__
                        except (AttributeError, TypeError):
                            study_data[complex_attr] = str(attr_value)
            
            # Special handling for available_datasets
            if hasattr(study, 'available_datasets') and study.available_datasets:
                datasets = []
                for dataset in study.available_datasets:
                    dataset_dict = {}
                    
                    # If dataset is a dict, use it directly
                    if isinstance(dataset, dict):
                        dataset_dict = dict(dataset)
                        # Remove any DataFrame that can't be serialized
                        if 'data' in dataset_dict and not isinstance(dataset_dict['data'], (str, list, dict)):
                            shape = dataset_dict['data'].shape if hasattr(dataset_dict['data'], 'shape') else (0, 0)
                            dataset_dict['data_shape'] = shape
                            del dataset_dict['data']
                    else:
                        # Otherwise try to convert to dict
                        for attr in ['name', 'description', 'source', 'last_updated']:
                            if hasattr(dataset, attr):
                                dataset_dict[attr] = getattr(dataset, attr)
                        
                        # Handle the data separately since it's usually a DataFrame
                        if hasattr(dataset, 'data') and dataset.data is not None:
                            # Store just the shape info instead of the data
                            shape = dataset.data.shape if hasattr(dataset.data, 'shape') else (0, 0)
                            dataset_dict['data_shape'] = shape
                    
                    datasets.append(dataset_dict)
                
                study_data['available_datasets'] = datasets
            
            # Check if study exists in database
            if hasattr(study, '_id') and study._id:
                response = requests.put(
                    f"{self.db_url}/studies/{study._id}",
                    auth=self.db_auth,
                    json=study_data
                )
            else:
                response = requests.post(
                    f"{self.db_url}/studies",
                    auth=self.db_auth,
                    json=study_data
                )
                
            if response.status_code in [200, 201]:
                result = response.json()
                if not hasattr(study, '_id'):
                    study._id = result.get('id')
                if not hasattr(study, '_rev'):
                    study._rev = result.get('rev')
                return True
            else:
                print(f"Error saving study: {response.text}")
                return False
                
        except Exception as e:
            print(f"Error saving study to database: {str(e)}")
            return False
    
    def delete_study_from_db(self, study_id):
        """Delete study from database."""
        if not self.studies_manager or not study_id:
            return False
            
        try:
            # Use studies manager to delete if it has the method
            if hasattr(self.studies_manager, 'delete_study'):
                result = self.studies_manager.delete_study(study_id)
                return result
                
            # First get the study to get the revision
            response = requests.get(
                f"{self.db_url}/studies/{study_id}",
                auth=self.db_auth
            )
            
            if response.status_code == 200:
                study = response.json()
                rev = study.get('_rev')
                
                # Then delete it
                delete_response = requests.delete(
                    f"{self.db_url}/studies/{study_id}?rev={rev}",
                    auth=self.db_auth
                )
                
                return delete_response.status_code in [200, 202]
            return False
                
        except Exception as e:
            print(f"Error deleting study from database: {str(e)}")
            return False
    
    def create_new_project(self):
        """Create a new project."""
        if not self.studies_manager:
            return
            
        # Show dialog to get project name and description
        dialog = QDialog(self)
        dialog.setWindowTitle("Create New Project")
        dialog.resize(400, 200)
        
        layout = QVBoxLayout(dialog)
        
        form_layout = QFormLayout()
        name_input = QLineEdit()
        name_input.setPlaceholderText("Enter project name")
        form_layout.addRow("Project Name:", name_input)
        
        description_input = QTextEdit()
        description_input.setPlaceholderText("Enter project description (optional)")
        description_input.setMaximumHeight(80)
        form_layout.addRow("Description:", description_input)
        layout.addLayout(form_layout)
        
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        create_btn = QPushButton("Create")
        create_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(create_btn)
        layout.addLayout(button_layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            name = name_input.text().strip()
            description = description_input.toPlainText().strip()
            
            if not name:
                QMessageBox.warning(self, "Input Error", "Project name cannot be empty.")
                return
                
            # Create new project
            project = self.studies_manager.create_project(name, description)
            if project:
                # No auto-saving - user must use Save button
                QMessageBox.information(self, "Success", f"Project '{name}' created. Use the Save button to save to database.")
                self.refresh_projects_list()
                self.active_project_changed.emit(project.id)
    
    def edit_project(self):
        """Edit the current project."""
        if not self.studies_manager:
            return
            
        project = self.studies_manager.get_active_project()
        if not project:
            QMessageBox.warning(self, "No Project", "No active project to edit.")
            return
            
        # Show dialog to edit project name and description
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Project")
        dialog.resize(400, 200)
        
        layout = QVBoxLayout(dialog)
        
        form_layout = QFormLayout()
        name_input = QLineEdit()
        name_input.setText(project.name)
        form_layout.addRow("Project Name:", name_input)
        
        description_input = QTextEdit()
        description_input.setText(project.description or "")
        description_input.setMaximumHeight(80)
        form_layout.addRow("Description:", description_input)
        layout.addLayout(form_layout)
        
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        save_btn = QPushButton("Apply Changes")
        save_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(save_btn)
        
        delete_btn = QPushButton("Delete Project")
        delete_btn.setStyleSheet("background-color: #ff6b6b;")
        delete_btn.clicked.connect(lambda: self.confirm_delete_project(dialog, project))
        button_layout.addWidget(delete_btn)
        
        layout.addLayout(button_layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            name = name_input.text().strip()
            description = description_input.toPlainText().strip()
            
            if not name:
                QMessageBox.warning(self, "Input Error", "Project name cannot be empty.")
                return
                
            # Update project
            project.name = name
            project.description = description
            project.updated_at = datetime.now().isoformat()
            
            # No auto-saving - user must use Save button
            QMessageBox.information(self, "Success", f"Project '{name}' updated. Use the Save button to save changes to database.")
            
            self.refresh_projects_list()
    
    def confirm_delete_project(self, parent_dialog, project):
        """Confirm deletion of a project."""
        reply = QMessageBox.question(
            parent_dialog,
            'Confirm Project Deletion',
            f'Are you sure you want to delete the project "{project.name}"?\n\n'
            'This will permanently delete the project and all its studies!',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Close the edit dialog first
            parent_dialog.reject()
            
            # Delete the project from memory
            if hasattr(self.studies_manager, 'remove_project'):
                self.studies_manager.remove_project(project.id)
                QMessageBox.information(self, "Success", f"Project '{project.name}' has been removed from memory. Database not updated.")
                self.refresh_projects_list()
            else:
                QMessageBox.warning(self, "Error", f"Failed to remove project '{project.name}' from memory.")
    
    def create_new_study(self):
        """Create a new study in the current project."""
        from study_model.study_model import StudyDesign
        
        if not self.studies_manager:
            return
            
        # Get active project ID
        project_id = None
        if self.project_combo.currentData():
            project_id = self.project_combo.currentData()
        
        # Show dialog to get study name
        dialog = QDialog(self)
        dialog.setWindowTitle("Create New Study")
        dialog.resize(400, 150)
        
        layout = QVBoxLayout(dialog)
        
        form_layout = QFormLayout()
        name_input = QLineEdit()
        name_input.setPlaceholderText("Enter study name")
        form_layout.addRow("Study Name:", name_input)
        layout.addLayout(form_layout)
        
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        create_btn = QPushButton("Create")
        create_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(create_btn)
        layout.addLayout(button_layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            name = name_input.text().strip()
            
            if not name:
                QMessageBox.warning(self, "Input Error", "Study name cannot be empty.")
                return
                
            # Create a basic study design
            study_design = StudyDesign(
                study_id=str(uuid.uuid4()),
                title=name,
                description=f"New study created on {datetime.now().strftime('%Y-%m-%d')}",
            )
            
            # Create new study
            study = self.studies_manager.create_study(name, study_design, project_id)
            if study:
                # No auto-saving - user must use Save button
                QMessageBox.information(self, "Success", f"Study '{name}' created. Use the Save button to save to database.")
                self.refresh_studies_list()
                self.active_study_changed.emit(study.id)
    
    def on_study_selected(self, item):
        """Handle selection of a study in the table."""
        row = item.row()
        study_id = self.studies_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        
        # Get the active project ID
        project_id = None
        if self.project_combo.currentData():
            project_id = self.project_combo.currentData()
            
        # Load the study details
        study = self.studies_manager.get_study(study_id, project_id)
        if study:
            self.display_study_details(study)
            self.content_stack.setCurrentIndex(1)  # Show study details
            
            # Add buttons for edit and delete
            if not hasattr(self, 'study_action_buttons'):
                self.study_action_buttons = QHBoxLayout()
                edit_study_btn = QPushButton("Edit Study")
                edit_study_btn.clicked.connect(self.edit_current_study)
                
                delete_study_btn = QPushButton("Delete Study")
                delete_study_btn.setStyleSheet("background-color: #ff6b6b;")
                delete_study_btn.clicked.connect(self.delete_current_study)
                
                self.study_action_buttons.addWidget(edit_study_btn)
                self.study_action_buttons.addWidget(delete_study_btn)
                
                # Add to layout below study details label
                layout_index = self.content_stack.currentWidget().layout().indexOf(self.study_details_label)
                self.content_stack.currentWidget().layout().insertLayout(layout_index + 1, self.study_action_buttons)
            
            # Store current study ID
            self.current_study_id = study_id
    
    def edit_current_study(self):
        """Edit the currently displayed study."""
        if not hasattr(self, 'current_study_id') or not self.current_study_id:
            QMessageBox.warning(self, "Error", "No study selected.")
            return
            
        # Get active project ID
        project_id = None
        if self.project_combo.currentData():
            project_id = self.project_combo.currentData()
            
        # Get current study
        study = self.studies_manager.get_study(self.current_study_id, project_id)
        if not study:
            QMessageBox.warning(self, "Error", "Could not load study.")
            return
            
        # Show dialog to edit study name
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Study")
        dialog.resize(400, 150)
        
        layout = QVBoxLayout(dialog)
        
        form_layout = QFormLayout()
        name_input = QLineEdit()
        name_input.setText(study.name)
        form_layout.addRow("Study Name:", name_input)
        
        description_input = QTextEdit()
        if hasattr(study, 'study_design') and study.study_design and hasattr(study.study_design, 'description'):
            description_input.setText(study.study_design.description)
        description_input.setMaximumHeight(80)
        form_layout.addRow("Description:", description_input)
        
        layout.addLayout(form_layout)
        
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        save_btn = QPushButton("Apply Changes")
        save_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(save_btn)
        layout.addLayout(button_layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            name = name_input.text().strip()
            description = description_input.toPlainText().strip()
            
            if not name:
                QMessageBox.warning(self, "Input Error", "Study name cannot be empty.")
                return
                
            # Update study
            study.name = name
            study.updated_at = datetime.now().isoformat()
            
            # Update study design description if it exists
            if hasattr(study, 'study_design') and study.study_design:
                study.study_design.description = description
                study.study_design.title = name  # Keep title in sync with name
            
            # No auto-saving - user must use Save button
            QMessageBox.information(self, "Success", f"Study '{name}' updated. Use the Save button to save changes to database.")
            
            # Refresh display
            self.refresh_studies_list()
            self.display_study_details(study)
    
    def delete_current_study(self):
        """Delete the currently displayed study."""
        if not hasattr(self, 'current_study_id') or not self.current_study_id:
            QMessageBox.warning(self, "Error", "No study selected to delete.")
            return
            
        # Get active project ID
        project_id = None
        if self.project_combo.currentData():
            project_id = self.project_combo.currentData()
            
        # Get current study
        study = self.studies_manager.get_study(self.current_study_id, project_id)
        if not study:
            QMessageBox.warning(self, "Error", "Could not load study.")
            return
            
        # Confirm deletion
        reply = QMessageBox.question(
            self,
            'Confirm Study Deletion',
            f'Are you sure you want to delete the study "{study.name}"?\n\n'
            'This will remove it from memory only. Database will not be updated.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Remove the study from memory
            if hasattr(self.studies_manager, 'remove_study'):
                self.studies_manager.remove_study(self.current_study_id, project_id)
                QMessageBox.information(self, "Success", f"Study '{study.name}' has been removed from memory. Database not updated.")
                self.current_study_id = None
                self.content_stack.setCurrentIndex(0)  # Show empty state
                self.refresh_studies_list()
            else:
                QMessageBox.warning(self, "Error", f"Failed to remove study '{study.name}' from memory.")
    
    def set_active_study(self):
        """Set the selected study as active."""
        selected_rows = self.studies_table.selectedItems()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select a study first.")
            return
            
        row = selected_rows[0].row()
        study_id = self.studies_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        
        # Get the active project ID
        project_id = None
        if self.project_combo.currentData():
            project_id = self.project_combo.currentData()
            
        # Set as active
        if self.studies_manager.set_active_study(study_id, project_id):
            self.active_study_changed.emit(study_id)
            self.refresh_studies_list()
    
    def display_study_details(self, study):
        """Display the details of a study."""
        if not study:
            self.study_details_label.setText("No study selected")
            self.content_stack.setCurrentIndex(0)  # Show empty state
            return
            
        # Set study title
        self.study_details_label.setText(f"Study: {study.name}")
        
        # Display study design details
        self.display_study_design(study)
            
        # Switch to study details view
        self.content_stack.setCurrentIndex(1)
    
    def display_study_design(self, study):
        """Display the study design in the design tree."""
        self.design_tree.clear()
        
        # Basic study information
        basic_info = QTreeWidgetItem(self.design_tree, ["Study Information", ""])
        QTreeWidgetItem(basic_info, ["Name", study.name])
        QTreeWidgetItem(basic_info, ["Created", datetime.fromisoformat(study.created_at).strftime("%Y-%m-%d %H:%M")])
        QTreeWidgetItem(basic_info, ["Last Updated", datetime.fromisoformat(study.updated_at).strftime("%Y-%m-%d %H:%M")])
        
        # Study design information
        if hasattr(study, 'study_design') and study.study_design:
            design = study.study_design
            design_info = QTreeWidgetItem(self.design_tree, ["Design", ""])
            
            if hasattr(design, 'title') and design.title:
                QTreeWidgetItem(design_info, ["Title", design.title])
                
            if hasattr(design, 'description') and design.description:
                QTreeWidgetItem(design_info, ["Description", design.description])
                
            if hasattr(design, 'design_type') and design.design_type:
                QTreeWidgetItem(design_info, ["Design Type", design.design_type])
                
            # Outcome measures
            if hasattr(design, 'outcome_measures') and design.outcome_measures:
                outcomes_info = QTreeWidgetItem(self.design_tree, ["Outcome Measures", f"{len(design.outcome_measures)} measures"])
                for i, outcome in enumerate(design.outcome_measures):
                    outcome_item = QTreeWidgetItem(outcomes_info, [f"Outcome {i+1}", outcome.name if hasattr(outcome, 'name') else "Unnamed"])
                    if hasattr(outcome, 'description') and outcome.description:
                        QTreeWidgetItem(outcome_item, ["Description", outcome.description])
                    if hasattr(outcome, 'data_type') and outcome.data_type:
                        QTreeWidgetItem(outcome_item, ["Data Type", str(outcome.data_type)])
            
            # Groups
            if hasattr(design, 'groups') and design.groups:
                groups_info = QTreeWidgetItem(self.design_tree, ["Groups", f"{len(design.groups)} groups"])
                for i, group in enumerate(design.groups):
                    group_item = QTreeWidgetItem(groups_info, [f"Group {i+1}", group.name if hasattr(group, 'name') else "Unnamed"])
                    if hasattr(group, 'description') and group.description:
                        QTreeWidgetItem(group_item, ["Description", group.description])
            
            # Timepoints
            if hasattr(design, 'timepoints') and design.timepoints:
                timepoints_info = QTreeWidgetItem(self.design_tree, ["Timepoints", f"{len(design.timepoints)} timepoints"])
                for i, timepoint in enumerate(design.timepoints):
                    timepoint_item = QTreeWidgetItem(timepoints_info, [f"Timepoint {i+1}", timepoint.name if hasattr(timepoint, 'name') else "Unnamed"])
                    if hasattr(timepoint, 'description') and timepoint.description:
                        QTreeWidgetItem(timepoint_item, ["Description", timepoint.description])
        
        # Datasets information if available
        datasets_count = len(study.available_datasets) if hasattr(study, 'available_datasets') and study.available_datasets else 0
        if datasets_count > 0:
            datasets_info = QTreeWidgetItem(self.design_tree, ["Datasets", f"{datasets_count} datasets"])
            for i, dataset in enumerate(study.available_datasets):
                # Handle both dict and namedtuple format for backward compatibility
                if isinstance(dataset, dict):
                    name = dataset.get('name', f"Dataset {i+1}")
                    rows, cols = dataset.get('data').shape if 'data' in dataset else (0, 0)
                else:
                    name = dataset.name if hasattr(dataset, 'name') else f"Dataset {i+1}"
                    rows, cols = dataset.data.shape if hasattr(dataset, 'data') else (0, 0)
                    
                dataset_item = QTreeWidgetItem(datasets_info, [name, f"{rows} rows × {cols} columns"])

        # Hypotheses information if available
        hypotheses_count = len(study.hypotheses) if hasattr(study, 'hypotheses') and study.hypotheses else 0
        if hypotheses_count > 0:
            QTreeWidgetItem(self.design_tree, ["Hypotheses", f"{hypotheses_count} hypotheses"])
        
        # Literature searches if available
        lit_search_count = len(study.literature_searches) if hasattr(study, 'literature_searches') and study.literature_searches else 0
        if lit_search_count > 0:
            QTreeWidgetItem(self.design_tree, ["Literature Searches", f"{lit_search_count} searches"])
            
        # Results if available
        results_count = len(study.results) if hasattr(study, 'results') and study.results else 0
        if results_count > 0:
            QTreeWidgetItem(self.design_tree, ["Results", f"{results_count} outcome results"])
        
        # Expand top-level items
        for i in range(self.design_tree.topLevelItemCount()):
            self.design_tree.topLevelItem(i).setExpanded(True)

    def save_current_project(self):
        """Save the current project to database."""
        if not self.studies_manager:
            QMessageBox.warning(self, "Error", "Studies manager not initialized.")
            return
            
        # Ensure databases exist before saving
        if not self.ensure_databases_exist():
            QMessageBox.warning(self, "Error", "Cannot save project without valid databases.")
            return
            
        project = self.studies_manager.get_active_project()
        if not project:
            QMessageBox.warning(self, "Error", "No active project to save.")
            return
            
        # Save the project to database
        if self.save_project_to_db(project):
            QMessageBox.information(self, "Success", f"Project '{project.name}' saved to database.")
            
            # Now save all studies associated with this project
            studies_saved = 0
            studies_failed = 0
            
            # Get all studies for this project
            if hasattr(self.studies_manager, 'list_studies'):
                studies = self.studies_manager.list_studies(project.id)
                
                for study_info in studies:
                    # Get the full study object
                    study = self.studies_manager.get_study(study_info['id'], project.id)
                    if study:
                        # Make sure the study has the project ID
                        if not hasattr(study, 'project_id'):
                            study.project_id = project.id
                            
                        # Save the study
                        if self.save_study_to_db(study):
                            studies_saved += 1
                        else:
                            studies_failed += 1
                
                if studies_saved > 0:
                    QMessageBox.information(
                        self, 
                        "Success", 
                        f"Saved {studies_saved} studies associated with this project."
                    )
                    
                if studies_failed > 0:
                    QMessageBox.warning(
                        self,
                        "Warning",
                        f"Failed to save {studies_failed} studies."
                    )
        else:
            QMessageBox.warning(self, "Error", f"Failed to save project '{project.name}' to database.")
            
    def save_current_study(self):
        """Save the currently selected study to database."""
        # Ensure databases exist before saving
        if not self.ensure_databases_exist():
            QMessageBox.warning(self, "Error", "Cannot save study without valid databases.")
            return
        
        # Check if we have a current study ID
        if hasattr(self, 'current_study_id') and self.current_study_id:
            study_id = self.current_study_id
        else:
            # Try to get the selected study from the table
            selected_rows = self.studies_table.selectedItems()
            if not selected_rows:
                QMessageBox.warning(self, "Error", "No study selected to save.")
                return
                
            row = selected_rows[0].row()
            study_id = self.studies_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            
        # Get active project ID
        project_id = None
        if self.project_combo.currentData():
            project_id = self.project_combo.currentData()
            
        # Get the study
        study = self.studies_manager.get_study(study_id, project_id)
        if not study:
            QMessageBox.warning(self, "Error", "Could not load study.")
            return
            
        # Save to database
        if self.save_study_to_db(study):
            QMessageBox.information(self, "Success", f"Study '{study.name}' saved to database.")
        else:
            QMessageBox.warning(self, "Error", f"Failed to save study '{study.name}' to database.")

    def reset_databases(self):
        """Reset (or create) the projects and studies databases."""
        reply = QMessageBox.question(
            self,
            'Reset Databases',
            'This will delete and recreate the projects and studies databases. Continue?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.No:
            return
        
        # Set flag to show errors during database operations
        self._show_db_errors = True
        
        # If user wants to proceed, actually delete and recreate the databases
        required_dbs = ['projects', 'studies']
        databases_created = []
        errors = []
        
        try:
            # Check which databases exist first
            response = requests.get(f"{self.db_url}/_all_dbs", auth=self.db_auth)
            
            if response.status_code == 200:
                existing_dbs = response.json()
                
                # First delete if they exist
                for db_name in required_dbs:
                    if db_name in existing_dbs:
                        print(f"Deleting database: {db_name}")
                        delete_response = requests.delete(
                            f"{self.db_url}/{db_name}",
                            auth=self.db_auth
                        )
                        
                        if delete_response.status_code not in [200, 202]:
                            error_msg = f"Error deleting database {db_name}: {delete_response.text}"
                            print(error_msg)
                            errors.append(error_msg)
                
                # Now create them all
                for db_name in required_dbs:
                    print(f"Creating database: {db_name}")
                    create_response = requests.put(
                        f"{self.db_url}/{db_name}",
                        auth=self.db_auth
                    )
                    
                    if create_response.status_code in [201, 202]:
                        print(f"Successfully created database: {db_name}")
                        databases_created.append(db_name)
                        
                        # Create necessary views and indexes
                        if db_name == 'projects':
                            self.create_project_views(db_name)
                        elif db_name == 'studies':
                            self.create_study_views(db_name)
                    else:
                        error_msg = f"Error creating database {db_name}: {create_response.text}"
                        print(error_msg)
                        errors.append(error_msg)
            else:
                error_msg = f"Error getting database list: {response.text}"
                print(error_msg)
                errors.append(error_msg)
                
            # Show errors if any occurred
            if errors:
                QMessageBox.warning(
                    self,
                    "Database Reset Errors",
                    "\n".join(errors)
                )
                return False
            
            # Success message
            QMessageBox.information(
                self,
                "Success",
                f"Databases have been reset successfully: {', '.join(databases_created)}"
            )
            return True
                
        except Exception as e:
            error_msg = f"Error resetting databases: {str(e)}"
            print(error_msg)
            QMessageBox.warning(self, "Error", error_msg)
            return False
            
        # Reset the flag
        self._show_db_errors = False

    def load_projects_from_db(self):
        """Load projects from the database."""
        # First ensure databases exist
        if not self.ensure_databases_exist():
            QMessageBox.warning(self, "Error", "Cannot load projects without valid databases.")
            return
            
        # Add compatibility for get_project method that assumes a dict
        if hasattr(self.studies_manager, 'projects') and isinstance(self.studies_manager.projects, list):
            if hasattr(self.studies_manager, 'get_project'):
                # Create a safer version of get_project that works with lists
                original_get_project = self.studies_manager.get_project
                
                def safe_get_project(project_id):
                    try:
                        # First try the original method
                        return original_get_project(project_id)
                    except AttributeError:
                        # If it fails with AttributeError, implement our own
                        for project in self.studies_manager.projects:
                            if hasattr(project, 'id') and str(project.id) == str(project_id):
                                return project
                        return None
                
                # Monkey patch the get_project method
                self.studies_manager.get_project = safe_get_project
        
        try:
            # Get all projects from database
            response = requests.get(
                f"{self.db_url}/projects/_all_docs?include_docs=true",
                auth=self.db_auth
            )
            
            if response.status_code == 200:
                # Parse the JSON response safely
                try:
                    data = response.json()
                except ValueError:
                    QMessageBox.warning(self, "Error", "Failed to parse database response as JSON.")
                    return
                
                # Safely handle the response structure
                if not isinstance(data, dict) or 'rows' not in data:
                    QMessageBox.warning(self, "Error", "Unexpected database response format.")
                    print(f"Response data: {data}")
                    return
                
                rows = data['rows']
                if not isinstance(rows, list):
                    QMessageBox.warning(self, "Error", "Database response 'rows' is not a list.")
                    return
                
                # Clear existing projects first if requested
                reply = QMessageBox.question(
                    self,
                    'Load Projects',
                    'Replace current projects with ones from database?',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
                )
                
                if reply == QMessageBox.StandardButton.Cancel:
                    return
                
                # When replacing, we need to clear the projects to avoid duplicates and ID mismatches
                if reply == QMessageBox.StandardButton.Yes:
                    if hasattr(self.studies_manager, 'clear_projects'):
                        print("Clearing existing projects")
                        self.studies_manager.clear_projects()
                    elif hasattr(self.studies_manager, 'projects'):
                        print("Clearing existing projects list")
                        self.studies_manager.projects = []
                
                projects_loaded = 0
                project_ids_map = {}  # Maps database IDs to memory project IDs
                
                # Process each project document
                for row in rows:
                    # Make sure row is a dictionary and has 'doc' key
                    if not isinstance(row, dict) or 'doc' not in row:
                        print(f"Skipping invalid row: {row}")
                        continue
                    
                    doc = row['doc']
                    if not isinstance(doc, dict):
                        print(f"Skipping invalid doc: {doc}")
                        continue
                    
                    # Skip design documents
                    if '_id' in doc and isinstance(doc['_id'], str) and doc['_id'].startswith('_design'):
                        continue
                    
                    # Get project ID from various possible fields
                    db_project_id = None
                    for id_field in ['id', '_id', 'project_id']:
                        if id_field in doc and doc[id_field]:
                            db_project_id = doc[id_field]
                            break
                            
                    if not db_project_id:
                        print(f"Skipping project without ID: {doc.get('name', 'Unknown')}")
                        continue
                    
                    print(f"Processing project: {doc.get('name', 'Unnamed')} (ID: {db_project_id})")
                    
                    # Check if this project already exists to prevent duplicates
                    existing_project = None
                    if hasattr(self.studies_manager, 'get_project_by_id'):
                        existing_project = self.studies_manager.get_project_by_id(db_project_id)
                    elif hasattr(self.studies_manager, 'get_project'):
                        existing_project = self.studies_manager.get_project(db_project_id)
                    
                    if existing_project:
                        print(f"Updating existing project: {existing_project.name}")
                        # Update existing project instead of creating a new one
                        for key, value in doc.items():
                            if key not in ['_id', '_rev'] and hasattr(existing_project, key):
                                setattr(existing_project, key, value)
                                
                        # Set database attributes
                        existing_project._id = doc.get('_id')
                        existing_project._rev = doc.get('_rev')
                        
                        # Store the project ID mapping
                        project_ids_map[db_project_id] = existing_project.id
                        
                        projects_loaded += 1
                    else:
                        print(f"No matching project found for ID: {db_project_id}")
                        
                        # Create a new project for this study if no matching project exists
                        print(f"Creating new project for study: {doc.get('name', 'Unnamed Study')}")
                        if hasattr(self.studies_manager, 'create_project'):
                            project_name = f"Project for {doc.get('name', 'Unnamed Study')}"
                            project_obj = self.studies_manager.create_project(project_name)
                            if project_obj:
                                project_id = project_obj.id
                                project_exists = True
                                # Add this mapping so future studies with this db_project_id will use our new project
                                project_ids_map[db_project_id] = project_id
                                print(f"Created new project '{project_name}' with ID: {project_id} and mapped DB ID {db_project_id} to it")
                                
                                # Now that we've created a project, we can try to create the study
                                if hasattr(self.studies_manager, 'create_study'):
                                    print(f"Creating study in new project")
                                    continue
                        else:
                            print("Cannot create new project - studies_manager doesn't have create_project method")
                
                print(f"Project ID mappings: {project_ids_map}")
                
                # Now load all studies for these projects with the ID mappings
                studies_loaded = self.load_studies_for_projects(project_ids_map)
                
                # Refresh the display
                self.refresh_projects_list()
                
                # Show success message
                if studies_loaded > 0:
                    QMessageBox.information(
                        self,
                        "Success",
                        f"Loaded {projects_loaded} projects and {studies_loaded} studies from database."
                    )
                else:
                    QMessageBox.information(
                        self,
                        "Success",
                        f"Loaded {projects_loaded} projects from database."
                    )
            else:
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Failed to load projects: {response.text}"
                )
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(
                self,
                "Connection Error",
                f"Failed to load projects: {str(e)}"
            )
            
    def load_studies_for_projects(self, project_ids_map=None):
        """Load all studies from database for the currently loaded projects.
        
        Args:
            project_ids_map: Optional mapping from database project IDs to memory project IDs
        """
        studies_loaded = 0
        
        # Ensure project_ids_map is initialized
        if project_ids_map is None:
            project_ids_map = {}
        
        try:
            # Get all studies from database
            response = requests.get(
                f"{self.db_url}/studies/_all_docs?include_docs=true",
                auth=self.db_auth
            )
            
            if response.status_code == 200:
                # Parse the JSON response safely
                try:
                    data = response.json()
                except ValueError:
                    print("Failed to parse database response as JSON.")
                    return 0
                
                # Safely handle the response structure
                if not isinstance(data, dict) or 'rows' not in data:
                    print(f"Unexpected database response format: {data}")
                    return 0
                
                rows = data['rows']
                if not isinstance(rows, list):
                    print("Database response 'rows' is not a list.")
                    return 0
                
                print(f"Found {len(rows)} study documents in database")
                
                # Process each study document
                for row in rows:
                    # Make sure row is a dictionary and has 'doc' key
                    if not isinstance(row, dict) or 'doc' not in row:
                        print(f"Skipping invalid row: {row}")
                        continue
                    
                    doc = row['doc']
                    if not isinstance(doc, dict):
                        print(f"Skipping invalid doc: {doc}")
                        continue
                    
                    # Skip design documents
                    if '_id' in doc and isinstance(doc['_id'], str) and doc['_id'].startswith('_design'):
                        continue
                    
                    # Get study ID - check multiple fields for compatibility
                    study_id = None
                    for id_field in ['id', '_id', 'study_id']:
                        if id_field in doc and doc[id_field]:
                            study_id = doc[id_field]
                            break
                            
                    if not study_id:
                        print(f"Skipping study without ID: {doc.get('name', 'Unknown')}")
                        continue
                    
                    # Get project ID - check multiple fields for compatibility
                    db_project_id = None
                    for project_field in ['project_id', 'projectId', 'project']:
                        if project_field in doc and doc[project_field]:
                            db_project_id = doc[project_field]
                            break
                            
                    if not db_project_id:
                        print(f"Skipping study {study_id} without project ID")
                        continue
                        
                    print(f"Processing study: {doc.get('name', 'Unnamed')} (ID: {study_id}, DB Project ID: {db_project_id})")
                    
                    # Convert the database project ID to the memory project ID if we have a mapping
                    project_id = db_project_id
                    if db_project_id in project_ids_map:
                        project_id = project_ids_map[db_project_id]
                        print(f"Mapped DB project ID {db_project_id} to memory project ID {project_id}")
                    
                    # Check if this project exists
                    project_exists = False
                    project_obj = None
                    
                    # Use existing methods to check if project exists - try with both IDs
                    if hasattr(self.studies_manager, 'get_project_by_id'):
                        project_obj = self.studies_manager.get_project_by_id(project_id)
                        if not project_obj and project_id != db_project_id:
                            # Try with the database ID as a fallback
                            project_obj = self.studies_manager.get_project_by_id(db_project_id)
                        project_exists = (project_obj is not None)
                        
                    if not project_exists and hasattr(self.studies_manager, 'get_project'):
                        project_obj = self.studies_manager.get_project(project_id)
                        if not project_obj and project_id != db_project_id:
                            # Try with the database ID as a fallback
                            project_obj = self.studies_manager.get_project(db_project_id)
                        project_exists = (project_obj is not None)
                        
                    # Last resort - check if project is in the projects list
                    if not project_exists and hasattr(self.studies_manager, 'projects'):
                        if isinstance(self.studies_manager.projects, dict):
                            # Dictionary-based projects
                            if project_id in self.studies_manager.projects:
                                project_exists = True
                                project_obj = self.studies_manager.projects[project_id]
                            elif db_project_id in self.studies_manager.projects:
                                project_exists = True
                                project_obj = self.studies_manager.projects[db_project_id]
                                project_id = db_project_id  # Use the matching ID
                        else:
                            # List-based projects
                            for proj in self.studies_manager.projects:
                                # Try both project IDs (memory and database)
                                if (hasattr(proj, 'id') and 
                                   (str(proj.id) == str(project_id) or 
                                    str(proj.id) == str(db_project_id))):
                                    project_exists = True
                                    project_obj = proj
                                    if str(proj.id) == str(db_project_id):
                                        project_id = db_project_id  # Use the matching ID
                                    break
                    
                    if project_exists:
                        project_name = project_obj.name if hasattr(project_obj, 'name') else project_id
                        print(f"Found matching project: {project_name}")
                        
                        # Check if study already exists to avoid duplicates
                        existing_study = None
                        if hasattr(self.studies_manager, 'get_study'):
                            existing_study = self.studies_manager.get_study(study_id, project_id)
                            # Try again with the database project ID if the first attempt failed
                            if not existing_study and project_id != db_project_id:
                                existing_study = self.studies_manager.get_study(study_id, db_project_id)
                        
                        if existing_study:
                            print(f"Updating existing study: {existing_study.name}")
                            # Update existing study
                            for key, value in doc.items():
                                if key not in ['_id', '_rev', 'study_design'] and hasattr(existing_study, key):
                                    # Special handling for hypotheses and other list data
                                    if key in ['hypotheses', 'literature_searches', 'results', 'paper_rankings', 'evidence_claims']:
                                        # Ensure the attribute is initialized as a list
                                        if getattr(existing_study, key) is None:
                                            setattr(existing_study, key, [])
                                        
                                        # Only add new items that don't already exist
                                        existing_ids = []
                                        if key == 'hypotheses':
                                            existing_ids = [h.get('id') for h in getattr(existing_study, key) if isinstance(h, dict) and 'id' in h]
                                        
                                        # Add the items that don't exist yet
                                        if isinstance(value, list):
                                            for item in value:
                                                if key == 'hypotheses':
                                                    # Only add if not already exists by ID
                                                    if isinstance(item, dict) and 'id' in item and item['id'] not in existing_ids:
                                                        getattr(existing_study, key).append(item)
                                                else:
                                                    # For other lists, just append
                                                    getattr(existing_study, key).append(item)
                                    else:
                                        # Normal attribute assignment
                                        setattr(existing_study, key, value)
                            
                            # Set database attributes
                            existing_study._id = doc.get('_id')
                            existing_study._rev = doc.get('_rev')
                            
                            # Ensure project ID is set correctly
                            if hasattr(existing_study, 'project_id'):
                                existing_study.project_id = project_id
                            
                            # Update study design if it exists
                            if 'study_design' in doc and hasattr(existing_study, 'study_design'):
                                design_data = doc['study_design']
                                for key, value in design_data.items():
                                    if hasattr(existing_study.study_design, key):
                                        setattr(existing_study.study_design, key, value)
                            
                            studies_loaded += 1
                        else:
                            print(f"Creating new study: {doc.get('name', 'Unnamed Study')}")
                            
                            # Load the study using load_study_from_doc if available
                            if hasattr(self.studies_manager, 'load_study_from_doc'):
                                print(f"Using load_study_from_doc method")
                                # Make sure to use the memory project ID, not the database ID
                                self.studies_manager.load_study_from_doc(doc, project_id)
                                studies_loaded += 1
                            
                            # Otherwise use create_study method
                            elif hasattr(self.studies_manager, 'create_study'):
                                print(f"Using create_study method")
                                # Create a basic study with the study design if available
                                study_design = None
                                
                                # Try to create study design object if data is available
                                if 'study_design' in doc:
                                    try:
                                        from study_model.study_model import StudyDesign
                                        design_data = doc['study_design']
                                        study_design = StudyDesign(
                                            study_id=study_id,
                                            title=design_data.get('title', doc.get('name', 'Unnamed Study')),
                                            description=design_data.get('description', '')
                                        )
                                    except Exception as e:
                                        print(f"Error creating study design: {str(e)}")
                                        # Continue without study design if it fails
                                
                                # Try to create the study
                                try:
                                    study = self.studies_manager.create_study(
                                        doc.get('name', 'Unnamed Study'),
                                        study_design,
                                        project_id  # Use the memory project ID
                                    )
                                    
                                    if study:
                                        # Update properties from document
                                        for key, value in doc.items():
                                            if key not in ['_id', '_rev', 'study_design'] and hasattr(study, key):
                                                # Special handling for hypotheses and other list data
                                                if key in ['hypotheses', 'literature_searches', 'results', 'paper_rankings', 'evidence_claims']:
                                                    # Ensure the attribute is initialized as a list
                                                    if getattr(study, key) is None:
                                                        setattr(study, key, [])
                                                    
                                                    # Only add new items that don't already exist
                                                    existing_ids = []
                                                    if key == 'hypotheses':
                                                        existing_ids = [h.get('id') for h in getattr(study, key) if isinstance(h, dict) and 'id' in h]
                                                    
                                                    # Add the items that don't exist yet
                                                    if isinstance(value, list):
                                                        for item in value:
                                                            if key == 'hypotheses':
                                                                # Only add if not already exists by ID
                                                                if isinstance(item, dict) and 'id' in item and item['id'] not in existing_ids:
                                                                    getattr(study, key).append(item)
                                                            else:
                                                                # For other lists, just append
                                                                getattr(study, key).append(item)
                                                else:
                                                    # Normal attribute assignment
                                                    setattr(study, key, value)
                                        
                                        # Set database attributes
                                        study._id = doc.get('_id')
                                        study._rev = doc.get('_rev')
                                        
                                        # Make sure the study is correctly associated with the project
                                        if hasattr(study, 'project_id'):
                                            study.project_id = project_id
                                            
                                        # Add study to project if needed
                                        if hasattr(self.studies_manager, 'add_study_to_project'):
                                            self.studies_manager.add_study_to_project(study, project_id)
                                        
                                        studies_loaded += 1
                                        print(f"Successfully created study: {study.name}")
                                    else:
                                        print(f"Failed to create study, returned None")
                                        
                                except Exception as e:
                                    print(f"Error creating study: {str(e)}")
                    else:
                        print(f"No matching project found for ID: {project_id} (DB ID: {db_project_id})")
            
            print(f"Total studies loaded: {studies_loaded}")
            return studies_loaded
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error loading studies: {str(e)}")
            return studies_loaded

    def debug_project_contents(self):
        """Display detailed information about projects and studies for debugging."""
        if not self.studies_manager:
            QMessageBox.warning(self, "Error", "Studies manager not initialized.")
            return
            
        debug_info = []
        
        # Check projects structure
        if hasattr(self.studies_manager, 'projects'):
            if isinstance(self.studies_manager.projects, dict):
                debug_info.append(f"Projects is a dictionary with {len(self.studies_manager.projects)} items")
            elif isinstance(self.studies_manager.projects, list):
                debug_info.append(f"Projects is a list with {len(self.studies_manager.projects)} items")
            else:
                debug_info.append(f"Projects is a {type(self.studies_manager.projects)}")
        else:
            debug_info.append("No projects attribute found")
            
        # Get all projects
        projects = []
        try:
            if hasattr(self.studies_manager, 'list_projects'):
                projects = self.studies_manager.list_projects()
            elif hasattr(self.studies_manager, 'projects'):
                if isinstance(self.studies_manager.projects, dict):
                    projects = [
                        {
                            "id": project_id,
                            "name": project.name if hasattr(project, 'name') else f"Project {project_id}",
                            "study_count": len(project.studies) if hasattr(project, 'studies') else 0
                        }
                        for project_id, project in self.studies_manager.projects.items()
                    ]
                elif isinstance(self.studies_manager.projects, list):
                    projects = [
                        {
                            "id": project.id if hasattr(project, 'id') else str(i),
                            "name": project.name if hasattr(project, 'name') else f"Project {i}",
                            "study_count": len(project.studies) if hasattr(project, 'studies') else 0
                        }
                        for i, project in enumerate(self.studies_manager.projects)
                    ]
        except Exception as e:
            debug_info.append(f"Error getting projects: {str(e)}")
            
        # Add project info
        debug_info.append(f"Found {len(projects)} projects:")
        for project in projects:
            debug_info.append(f"  - Project '{project['name']}' (ID: {project['id']}) with {project['study_count']} studies")
            
            # Get all studies for this project
            try:
                if hasattr(self.studies_manager, 'list_studies'):
                    studies = self.studies_manager.list_studies(project['id'])
                    
                    for study in studies:
                        study_id = study.get('id')
                        study_obj = self.studies_manager.get_study(study_id, project['id'])
                        
                        # Count hypotheses and other elements
                        hyp_count = len(study_obj.hypotheses) if hasattr(study_obj, 'hypotheses') and study_obj.hypotheses else 0
                        dataset_count = len(study_obj.available_datasets) if hasattr(study_obj, 'available_datasets') and study_obj.available_datasets else 0
                        
                        debug_info.append(f"      - Study '{study['name']}' (ID: {study_id})")
                        debug_info.append(f"          Hypotheses: {hyp_count}")
                        debug_info.append(f"          Datasets: {dataset_count}")
                        
                        # Show first few hypotheses
                        if hyp_count > 0:
                            debug_info.append(f"          Hypothesis list:")
                            for i, hyp in enumerate(study_obj.hypotheses[:5]):
                                if isinstance(hyp, dict):
                                    title = hyp.get('title', hyp.get('text', 'Unnamed'))
                                    hyp_id = hyp.get('id', 'Unknown')
                                    debug_info.append(f"            - {i+1}. {title} (ID: {hyp_id})")
                                else:
                                    debug_info.append(f"            - {i+1}. {str(hyp)}")
                                    
                            if hyp_count > 5:
                                debug_info.append(f"            ... and {hyp_count - 5} more")
            except Exception as e:
                debug_info.append(f"      Error getting studies: {str(e)}")
                
        # Display debug information
        dialog = QDialog(self)
        dialog.setWindowTitle("Project and Study Debug Information")
        dialog.resize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setText("\n".join(debug_info))
        layout.addWidget(text_edit)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        # Add to the project management buttons layout
        if not hasattr(self, 'debug_btn'):
            self.debug_btn = QPushButton("Debug Projects/Studies")
            self.debug_btn.clicked.connect(self.debug_project_contents)
            # Find the project buttons layout
            for i in range(self.layout().count()):
                item = self.layout().itemAt(i)
                if isinstance(item, QHBoxLayout):
                    for j in range(item.count()):
                        widget = item.itemAt(j).widget()
                        if isinstance(widget, QPushButton) and widget.text() == "New Project":
                            item.addWidget(self.debug_btn)
                            break
                    break
        
        dialog.exec()