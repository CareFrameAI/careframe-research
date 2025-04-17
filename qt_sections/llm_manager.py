# qt_sections/llm_manager.py
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QComboBox, QLabel, QTableWidget,
    QHeaderView, QTableWidgetItem, QGroupBox, QPushButton, QCheckBox,
    QDialog, QLineEdit, QDialogButtonBox, QMessageBox, QHBoxLayout, QTabWidget,
    QScrollArea, QFrame, QPlainTextEdit, QApplication
)
from PyQt6.QtCore import pyqtSignal, QObject, Qt
from PyQt6.QtGui import QFont, QColor
import datetime
import json
import os
import sqlite3
import traceback

# Import PHI manager
from privacy.phi_manager import PHIManagerWidget, phi_config

# --- Configuration ---
class LLMConfig(QObject):
    config_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        # Define available models (fetch dynamically or list known ones)
        self.AVAILABLE_MODELS = {
            "Claude": [
                "claude-3-7-sonnet-20250219",
                "claude-3-5-sonnet-20241022"
            ],
            "Gemini": [
                "gemini-2.0-flash",      
                "gemini-2.0-flash-lite"  
            ],
            "Local": [
                "local-llama3",
                "local-mistral",
                "local-llava"
            ]
        }
        self.ALL_MODELS = sorted(list(set(model for family in self.AVAILABLE_MODELS.values() for model in family)))

        # Default selections (ensure they are valid based on the updated list)
        self._default_text_model = "gemini-2.0-flash"
        self._default_json_model = "gemini-2.0-flash"
        self._default_vision_model = "claude-3-7-sonnet-20250219" # Claude 3.5 supports vision

        # Cost metrics per 1000 tokens (default estimated costs)
        self._cost_metrics = {
            # Claude models
            "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},
            "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
            # Gemini models
            "gemini-2.0-flash": {"input": 0.15, "output": 0.60},
            "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
            # Local models (free)
            "local-llama3": {"input": 0.0, "output": 0.0},
            "local-mistral": {"input": 0.0, "output": 0.0},
            "local-llava": {"input": 0.0, "output": 0.0}
        }
        
        # Token limits for models
        self._token_limits = {
            # Claude models
            "claude-3-7-sonnet-20250219": {"input": 200000, "output": 4096},
            "claude-3-5-sonnet-20241022": {"input": 200000, "output": 4096},
            # Gemini models
            "gemini-2.0-flash": {"input": 1000000, "output": 8192},
            "gemini-2.0-flash-lite": {"input": 1000000, "output": 8192},
            # Local models
            "local-llama3": {"input": 8192, "output": 4096},
            "local-mistral": {"input": 8192, "output": 4096},
            "local-llava": {"input": 8192, "output": 4096}
        }

        # Flag for saving LLM calls
        self._save_calls = False
        
        # Local LLM configuration
        self._use_local_llms = False
        self._local_llm_endpoints = {
            "text": os.environ.get("LOCAL_TEXT_LLM_ENDPOINT", "http://localhost:11434/v1"),
            "vision": os.environ.get("LOCAL_VISION_LLM_ENDPOINT", "http://localhost:11434/v1"),
            "json": os.environ.get("LOCAL_JSON_LLM_ENDPOINT", "http://localhost:11434/v1")
        }
        self._local_llm_models = {
            "text": os.environ.get("LOCAL_TEXT_LLM_MODEL", "llama3"),
            "vision": os.environ.get("LOCAL_VISION_LLM_MODEL", "llava"),
            "json": os.environ.get("LOCAL_JSON_LLM_MODEL", "llama3")
        }
        
        # Validate defaults against the new list
        if self._default_text_model not in self.ALL_MODELS:
            self._default_text_model = self.ALL_MODELS[0] if self.ALL_MODELS else "" # Fallback
        if self._default_json_model not in self.ALL_MODELS:
            self._default_json_model = self.ALL_MODELS[0] if self.ALL_MODELS else "" # Fallback
        if self._default_vision_model not in self.ALL_MODELS:
             # Fallback to a known vision model or the first model
            vision_options = [m for m in self.ALL_MODELS if "vision" in m or "sonnet" in m or "opus" in m] # Simple heuristic
            self._default_vision_model = vision_options[0] if vision_options else (self.ALL_MODELS[0] if self.ALL_MODELS else "")
            
        # Check if local LLM is enabled via environment variable
        if os.environ.get("USE_LOCAL_LLM", "").lower() in ("true", "1", "yes"):
            self._use_local_llms = True
            print(f"Local LLM mode enabled via environment variable")

    @property
    def default_text_model(self):
        return self._default_text_model

    @default_text_model.setter
    def default_text_model(self, value):
        if value in self.ALL_MODELS:
            self._default_text_model = value
            self.config_changed.emit()
        else:
            print(f"Warning: Attempted to set invalid default text model: {value}")

    @property
    def default_json_model(self):
        return self._default_json_model

    @default_json_model.setter
    def default_json_model(self, value):
        if value in self.ALL_MODELS:
            self._default_json_model = value
            self.config_changed.emit()
        else:
             print(f"Warning: Attempted to set invalid default json model: {value}")

    @property
    def default_vision_model(self):
        return self._default_vision_model

    @default_vision_model.setter
    def default_vision_model(self, value):
        # Add check for vision capabilities if known, otherwise just check existence
        if value in self.ALL_MODELS: # Simple check for now
            self._default_vision_model = value
            self.config_changed.emit()
        else:
            print(f"Warning: Attempted to set invalid default vision model: {value}")
            
    @property
    def save_calls(self):
        return self._save_calls
        
    @save_calls.setter
    def save_calls(self, value):
        self._save_calls = bool(value)
        self.config_changed.emit()
        
    @property
    def use_local_llms(self):
        return self._use_local_llms
        
    @use_local_llms.setter
    def use_local_llms(self, value):
        self._use_local_llms = bool(value)
        self.config_changed.emit()
        
    @property
    def local_llm_endpoints(self):
        return self._local_llm_endpoints
        
    @property
    def local_llm_models(self):
        return self._local_llm_models
    
    def set_local_llm_endpoint(self, endpoint_type, url):
        """Set a local LLM endpoint URL"""
        if endpoint_type in self._local_llm_endpoints:
            self._local_llm_endpoints[endpoint_type] = url
            self.config_changed.emit()
            return True
        return False
        
    def set_local_llm_model(self, model_type, model_name):
        """Set a local LLM model name"""
        if model_type in self._local_llm_models:
            self._local_llm_models[model_type] = model_name
            self.config_changed.emit()
            return True
        return False
        
    def is_local_model(self, model_name):
        """Check if a model is a local LLM"""
        return model_name.startswith("local-") or model_name in self._local_llm_models.values()
        
    def get_model_cost(self, model_name):
        """Get the cost metrics for a specific model"""
        return self._cost_metrics.get(model_name, {"input": 0.0, "output": 0.0})
        
    def set_model_cost(self, model_name, input_cost, output_cost):
        """Set cost metrics for a model"""
        if model_name in self.ALL_MODELS:
            self._cost_metrics[model_name] = {
                "input": float(input_cost),
                "output": float(output_cost)
            }
            self.config_changed.emit()
            return True
        return False
    
    def get_token_limits(self, model_name):
        """Get token limits for a specific model"""
        return self._token_limits.get(model_name, {"input": 1000000, "output": 4096})
    
    def set_token_limits(self, model_name, input_limit, output_limit):
        """Set token limits for a model"""
        if model_name in self.ALL_MODELS:
            self._token_limits[model_name] = {
                "input": int(input_limit),
                "output": int(output_limit)
            }
            self.config_changed.emit()
            return True
        return False
        
    def add_custom_model(self, family, model_name, input_cost=0.0, output_cost=0.0, input_limit=1000000, output_limit=4096):
        """Add a custom model to the available models"""
        if model_name in self.ALL_MODELS:
            return False  # Model already exists
            
        # Add to family or create new family
        if family in self.AVAILABLE_MODELS:
            if model_name not in self.AVAILABLE_MODELS[family]:
                self.AVAILABLE_MODELS[family].append(model_name)
        else:
            self.AVAILABLE_MODELS[family] = [model_name]
            
        # Update ALL_MODELS list
        self.ALL_MODELS = sorted(list(set(model for family in self.AVAILABLE_MODELS.values() for model in family)))
        
        # Add cost metrics
        self._cost_metrics[model_name] = {"input": float(input_cost), "output": float(output_cost)}
        
        # Add token limits
        self._token_limits[model_name] = {"input": int(input_limit), "output": int(output_limit)}
        
        self.config_changed.emit()
        return True

# Instantiate the config (this instance will be shared)
llm_config = LLMConfig()

# --- Signal for Call Tracking ---
class LLMTrackerSignals(QObject):
    # Signal arguments: timestamp, model, duration, input_tokens, output_tokens, status, error_msg, prompt_preview, response_preview, phi_scanned, phi_found_count, phi_types, phi_redacted
    call_logged = pyqtSignal(datetime.datetime, str, float, int, int, str, str, str, str, bool, int, dict, bool)
    
    # Add new signal for PHI blocking
    phi_blocked = pyqtSignal(str, str, dict)  # model, prompt, phi_report

llm_tracker_signals = LLMTrackerSignals()

# --- Database class for LLM call storage ---
class LLMDatabase:
    def __init__(self):
        self.db_path = os.path.join(os.path.expanduser("~"), ".careframe", "llm_calls.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()
        
    def init_db(self):
        """Initialize the database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS llm_calls (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            model TEXT,
            duration REAL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            status TEXT,
            error_msg TEXT,
            prompt TEXT,
            response TEXT,
            phi_scanned INTEGER,
            phi_found_count INTEGER,
            phi_types TEXT,
            phi_redacted INTEGER,
            estimated_cost REAL
        )
        ''')
        conn.commit()
        conn.close()
        
    def save_call(self, timestamp, model, duration, input_tokens, output_tokens, 
                  status, error_msg, prompt, response, phi_scanned, phi_found_count, phi_types, phi_redacted=False):
        """Save an LLM call to the database"""
        # Calculate estimated cost
        model_costs = llm_config.get_model_cost(model)
        input_cost = (input_tokens / 1000) * model_costs["input"]
        output_cost = (output_tokens / 1000) * model_costs["output"]
        total_cost = input_cost + output_cost
        
        # Convert phi_types dict to JSON string
        phi_types_json = json.dumps(phi_types) if phi_types else "{}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO llm_calls 
        (timestamp, model, duration, input_tokens, output_tokens, status, 
         error_msg, prompt, response, phi_scanned, phi_found_count, phi_types, phi_redacted, estimated_cost)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp.isoformat(), model, duration, input_tokens, output_tokens,
            status, error_msg, prompt, response, 1 if phi_scanned else 0, 
            phi_found_count, phi_types_json, 1 if phi_redacted else 0, total_cost
        ))
        conn.commit()
        conn.close()
        
    def get_calls(self, limit=100):
        """Retrieve the most recent calls from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        SELECT * FROM llm_calls ORDER BY timestamp DESC LIMIT ?
        ''', (limit,))
        results = cursor.fetchall()
        conn.close()
        return results
        
    def get_recent_calls(self, limit=100):
        """Alias for get_calls for backward compatibility"""
        return self.get_calls(limit=limit)
        
    def clear_calls(self):
        """Delete all records from the llm_calls table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM llm_calls')
        conn.commit()
        conn.close()

# Initialize database
llm_database = LLMDatabase()

# --- UI Widget ---
class LlmManagerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create tabs
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)
        
        # Create tab 1: Model Selection
        tab_models = QWidget()
        tab_widget.addTab(tab_models, "Model Selection")
        
        # Create tab 2: Call History
        tab_history = QWidget()
        tab_widget.addTab(tab_history, "Call History")
        
        # Create tab 3: Settings
        tab_settings = QWidget()
        tab_widget.addTab(tab_settings, "Settings")
        
        # Create tab 4: Local LLMs
        tab_local = QWidget()
        tab_widget.addTab(tab_local, "Local LLMs")
        
        # Setup each tab's content
        self.setup_models_tab(tab_models)
        self.setup_history_tab(tab_history)
        self.setup_settings_tab(tab_settings)
        self.setup_local_llm_tab(tab_local)
    
    def setup_models_tab(self, tab):
        # Vertical layout for the tab
        layout = QVBoxLayout(tab)
        
        # Group for model selection
        group_model = QGroupBox("Default Model Selection")
        layout.addWidget(group_model)
        
        model_layout = QFormLayout(group_model)
        
        # Text model selection
        self.text_model_combo = QComboBox()
        for model in llm_config.ALL_MODELS:
            self.text_model_combo.addItem(model)
        index = self.text_model_combo.findText(llm_config.default_text_model)
        if index >= 0:
            self.text_model_combo.setCurrentIndex(index)
        model_layout.addRow("Default Text Model:", self.text_model_combo)
        self.text_model_combo.currentTextChanged.connect(self.on_text_model_changed)
        
        # JSON model selection
        self.json_model_combo = QComboBox()
        for model in llm_config.ALL_MODELS:
            self.json_model_combo.addItem(model)
        index = self.json_model_combo.findText(llm_config.default_json_model)
        if index >= 0:
            self.json_model_combo.setCurrentIndex(index)
        model_layout.addRow("Default JSON Model:", self.json_model_combo)
        self.json_model_combo.currentTextChanged.connect(self.on_json_model_changed)
        
        # Vision model selection
        self.vision_model_combo = QComboBox()
        for model in llm_config.ALL_MODELS:
            self.vision_model_combo.addItem(model)
        index = self.vision_model_combo.findText(llm_config.default_vision_model)
        if index >= 0:
            self.vision_model_combo.setCurrentIndex(index)
        model_layout.addRow("Default Vision Model:", self.vision_model_combo)
        self.vision_model_combo.currentTextChanged.connect(self.on_vision_model_changed)
        
        # Group for model information/status
        group_info = QGroupBox("Model Info")
        layout.addWidget(group_info)
        
        info_layout = QVBoxLayout(group_info)
        
        # Table for model information
        self.model_info_table = QTableWidget()
        self.model_info_table.setColumnCount(4)
        self.model_info_table.setHorizontalHeaderLabels(["Model", "Input Cost", "Output Cost", "Max Tokens"])
        info_layout.addWidget(self.model_info_table)
        
        # Populate table
        self.update_model_info_table()
        
        # Stretcher at the bottom
        layout.addStretch()
    
    def setup_history_tab(self, tab):
        # Vertical layout for the tab
        layout = QVBoxLayout(tab)
        
        # Group for call history
        group_history = QGroupBox("LLM Call History")
        layout.addWidget(group_history)
        
        history_layout = QVBoxLayout(group_history)
        
        # Table for call history
        self.call_history_table = QTableWidget()
        self.call_history_table.setColumnCount(7)
        self.call_history_table.setHorizontalHeaderLabels(["Timestamp", "Model", "Duration", "Input Tokens", "Output Tokens", "Status", "Details"])
        history_layout.addWidget(self.call_history_table)
        
        # Connect to call logged signal
        llm_tracker_signals.call_logged.connect(self.on_llm_call_logged)
        
        # Refresh button and clear button
        button_layout = QHBoxLayout()
        history_layout.addLayout(button_layout)
        
        refresh_button = QPushButton("Refresh History")
        button_layout.addWidget(refresh_button)
        refresh_button.clicked.connect(self.refresh_call_history)
        
        clear_button = QPushButton("Clear History")
        button_layout.addWidget(clear_button)
        clear_button.clicked.connect(self.clear_call_history)
        
        # Load call history if database is available
        self.refresh_call_history()
        
        # Stretcher at the bottom
        layout.addStretch()
    
    def setup_settings_tab(self, tab):
        # Vertical layout for the tab
        layout = QVBoxLayout(tab)
        
        # Group for general settings
        group_settings = QGroupBox("General Settings")
        layout.addWidget(group_settings)
        
        settings_layout = QFormLayout(group_settings)
        
        # Save calls toggle
        self.save_calls_checkbox = QCheckBox()
        self.save_calls_checkbox.setChecked(llm_config.save_calls)
        settings_layout.addRow("Save LLM Calls:", self.save_calls_checkbox)
        self.save_calls_checkbox.stateChanged.connect(self.on_save_calls_changed)
        
        # Add more settings as needed
        # ...
        
        # Stretcher at the bottom
        layout.addStretch()
        
    def setup_local_llm_tab(self, tab):
        # Vertical layout for the tab
        layout = QVBoxLayout(tab)
        
        # Group for local LLM settings
        group_local = QGroupBox("Local LLM Configuration")
        layout.addWidget(group_local)
        
        local_layout = QFormLayout(group_local)
        
        # Enable local LLMs toggle
        self.use_local_llms_checkbox = QCheckBox()
        self.use_local_llms_checkbox.setChecked(llm_config.use_local_llms)
        local_layout.addRow("Use Local LLMs:", self.use_local_llms_checkbox)
        self.use_local_llms_checkbox.stateChanged.connect(self.on_use_local_llms_changed)
        
        # Endpoint configuration section
        endpoints_group = QGroupBox("Endpoints")
        local_layout.addRow(endpoints_group)
        endpoints_layout = QFormLayout(endpoints_group)
        
        # Text endpoint
        self.text_endpoint_edit = QLineEdit()
        self.text_endpoint_edit.setText(llm_config.local_llm_endpoints.get("text", "http://localhost:11434/v1"))
        endpoints_layout.addRow("Text API Endpoint:", self.text_endpoint_edit)
        self.text_endpoint_edit.editingFinished.connect(lambda: self.on_endpoint_changed("text"))
        
        # Vision endpoint
        self.vision_endpoint_edit = QLineEdit()
        self.vision_endpoint_edit.setText(llm_config.local_llm_endpoints.get("vision", "http://localhost:11434/v1"))
        endpoints_layout.addRow("Vision API Endpoint:", self.vision_endpoint_edit)
        self.vision_endpoint_edit.editingFinished.connect(lambda: self.on_endpoint_changed("vision"))
        
        # JSON endpoint
        self.json_endpoint_edit = QLineEdit()
        self.json_endpoint_edit.setText(llm_config.local_llm_endpoints.get("json", "http://localhost:11434/v1"))
        endpoints_layout.addRow("JSON API Endpoint:", self.json_endpoint_edit)
        self.json_endpoint_edit.editingFinished.connect(lambda: self.on_endpoint_changed("json"))
        
        # Model configuration section
        models_group = QGroupBox("Models")
        local_layout.addRow(models_group)
        models_layout = QFormLayout(models_group)
        
        # Text model
        self.text_model_edit = QLineEdit()
        self.text_model_edit.setText(llm_config.local_llm_models.get("text", "llama3"))
        models_layout.addRow("Text Model:", self.text_model_edit)
        self.text_model_edit.editingFinished.connect(lambda: self.on_local_model_changed("text"))
        
        # Vision model
        self.vision_model_edit = QLineEdit()
        self.vision_model_edit.setText(llm_config.local_llm_models.get("vision", "llava"))
        models_layout.addRow("Vision Model:", self.vision_model_edit)
        self.vision_model_edit.editingFinished.connect(lambda: self.on_local_model_changed("vision"))
        
        # JSON model
        self.json_model_edit = QLineEdit()
        self.json_model_edit.setText(llm_config.local_llm_models.get("json", "llama3"))
        models_layout.addRow("JSON Model:", self.json_model_edit)
        self.json_model_edit.editingFinished.connect(lambda: self.on_local_model_changed("json"))
        
        # Test connection section
        test_group = QGroupBox("Test Connection")
        layout.addWidget(test_group)
        test_layout = QVBoxLayout(test_group)
        
        test_button = QPushButton("Test Local LLM Connection")
        test_layout.addWidget(test_button)
        test_button.clicked.connect(self.test_local_llm_connection)
        
        self.test_result_label = QLabel("")
        test_layout.addWidget(self.test_result_label)
        
        # Update UI state based on whether local LLMs are enabled
        self.update_local_llm_ui_state()
        
        # Stretcher at the bottom
        layout.addStretch()
    
    def update_model_info_table(self):
        # Clear the table
        self.model_info_table.setRowCount(0)
        
        # Populate with model information
        for i, model in enumerate(llm_config.ALL_MODELS):
            self.model_info_table.insertRow(i)
            
            # Model name
            self.model_info_table.setItem(i, 0, QTableWidgetItem(model))
            
            # Cost metrics
            cost_metrics = llm_config.get_model_cost(model)
            self.model_info_table.setItem(i, 1, QTableWidgetItem(f"${cost_metrics['input']:.4f}"))
            self.model_info_table.setItem(i, 2, QTableWidgetItem(f"${cost_metrics['output']:.4f}"))
            
            # Token limits
            token_limits = llm_config.get_token_limits(model)
            self.model_info_table.setItem(i, 3, QTableWidgetItem(f"{token_limits['input']}"))
        
        # Resize columns to content
        self.model_info_table.resizeColumnsToContents()
    
    def on_text_model_changed(self, model_name):
        llm_config.default_text_model = model_name
        
    def on_json_model_changed(self, model_name):
        llm_config.default_json_model = model_name
        
    def on_vision_model_changed(self, model_name):
        llm_config.default_vision_model = model_name
        
    def on_save_calls_changed(self, state):
        llm_config.save_calls = (state == Qt.CheckState.Checked)
        
    def on_use_local_llms_changed(self, state):
        llm_config.use_local_llms = (state == Qt.CheckState.Checked)
        # Update UI elements based on the new state
        self.update_local_llm_ui_state()
        
        # Update the LLM client configuration
        try:
            from llms.client import LOCAL_LLM_CONFIG
            LOCAL_LLM_CONFIG["enabled"] = llm_config.use_local_llms
        except ImportError:
            print("Warning: Could not update LLM client configuration")
            
    def on_endpoint_changed(self, endpoint_type):
        if endpoint_type == "text":
            llm_config.set_local_llm_endpoint(endpoint_type, self.text_endpoint_edit.text())
        elif endpoint_type == "vision":
            llm_config.set_local_llm_endpoint(endpoint_type, self.vision_endpoint_edit.text())
        elif endpoint_type == "json":
            llm_config.set_local_llm_endpoint(endpoint_type, self.json_endpoint_edit.text())
            
        # Update the LLM client configuration
        try:
            from llms.client import LOCAL_LLM_CONFIG
            LOCAL_LLM_CONFIG["endpoints"][endpoint_type] = llm_config.local_llm_endpoints[endpoint_type]
        except ImportError:
            print("Warning: Could not update LLM client configuration")
            
    def on_local_model_changed(self, model_type):
        if model_type == "text":
            llm_config.set_local_llm_model(model_type, self.text_model_edit.text())
        elif model_type == "vision":
            llm_config.set_local_llm_model(model_type, self.vision_model_edit.text())
        elif model_type == "json":
            llm_config.set_local_llm_model(model_type, self.json_model_edit.text())
            
        # Update the LLM client configuration
        try:
            from llms.client import LOCAL_LLM_CONFIG
            LOCAL_LLM_CONFIG["models"][model_type] = llm_config.local_llm_models[model_type]
        except ImportError:
            print("Warning: Could not update LLM client configuration")
            
    def update_local_llm_ui_state(self):
        # Enable/disable UI elements based on local LLM toggle
        enabled = self.use_local_llms_checkbox.isChecked()
        
        self.text_endpoint_edit.setEnabled(enabled)
        self.vision_endpoint_edit.setEnabled(enabled)
        self.json_endpoint_edit.setEnabled(enabled)
        
        self.text_model_edit.setEnabled(enabled)
        self.vision_model_edit.setEnabled(enabled)
        self.json_model_edit.setEnabled(enabled)
        
    def test_local_llm_connection(self):
        if not llm_config.use_local_llms:
            self.test_result_label.setText("Local LLMs are not enabled")
            return
            
        self.test_result_label.setText("Testing connection...")
        QApplication.processEvents()  # Update UI immediately
        
        try:
            import asyncio
            import aiohttp
            
            async def test_connection():
                endpoint = llm_config.local_llm_endpoints["text"]
                model = llm_config.local_llm_models["text"]
                
                try:
                    async with aiohttp.ClientSession() as session:
                        # Try to get models list first
                        try:
                            async with session.get(f"{endpoint}/models") as response:
                                if response.status == 200:
                                    models = await response.json()
                                    return True, f"Connection successful. {len(models['data'])} models available."
                                else:
                                    # If models endpoint fails, try a simple completion
                                    payload = {
                                        "model": model,
                                        "messages": [{"role": "user", "content": "hello"}],
                                        "max_tokens": 10
                                    }
                                    
                                    async with session.post(f"{endpoint}/chat/completions", json=payload) as chat_response:
                                        if chat_response.status == 200:
                                            return True, "Connection successful via chat completions API."
                                        else:
                                            error_text = await chat_response.text()
                                            return False, f"API error: {chat_response.status} {error_text}"
                        except aiohttp.ClientError:
                            # Models endpoint might not be supported, try chat completions
                            payload = {
                                "model": model,
                                "messages": [{"role": "user", "content": "hello"}],
                                "max_tokens": 10
                            }
                            
                            async with session.post(f"{endpoint}/chat/completions", json=payload) as response:
                                if response.status == 200:
                                    return True, "Connection successful via chat completions API."
                                else:
                                    error_text = await response.text()
                                    return False, f"API error: {response.status} {error_text}"
                    
                except aiohttp.ClientError as e:
                    return False, f"Connection error: {str(e)}"
                except Exception as e:
                    return False, f"Error: {str(e)}"
            
            # Need to run the async test in an event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # If no event loop exists in this thread, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            success, message = loop.run_until_complete(test_connection())
            
            if success:
                self.test_result_label.setStyleSheet("color: green;")
            else:
                self.test_result_label.setStyleSheet("color: red;")
            self.test_result_label.setText(message)
            
        except ImportError as e:
            self.test_result_label.setText(f"Error: {str(e)}")
        except Exception as e:
            self.test_result_label.setText(f"Error: {str(e)}")
    
    def on_llm_call_logged(self, timestamp, model, duration, input_tokens, output_tokens, status, error_msg, prompt_preview, response_preview, phi_scanned=True, phi_found_count=0, phi_types={}, phi_redacted=False):
        # Insert a new row at the top of the table
        self.call_history_table.insertRow(0)
        
        # Format timestamp
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        # Add data to the row
        self.call_history_table.setItem(0, 0, QTableWidgetItem(timestamp_str))
        self.call_history_table.setItem(0, 1, QTableWidgetItem(model))
        self.call_history_table.setItem(0, 2, QTableWidgetItem(f"{duration:.3f}s"))
        self.call_history_table.setItem(0, 3, QTableWidgetItem(str(input_tokens)))
        self.call_history_table.setItem(0, 4, QTableWidgetItem(str(output_tokens)))
        
        status_item = QTableWidgetItem(status)
        if status == "Success":
            status_item.setBackground(QColor(200, 255, 200))  # Light green
        else:
            status_item.setBackground(QColor(255, 200, 200))  # Light red
        self.call_history_table.setItem(0, 5, status_item)
        
        # Create a details button
        details_widget = QWidget()
        details_layout = QHBoxLayout(details_widget)
        details_layout.setContentsMargins(2, 2, 2, 2)
        
        details_button = QPushButton("Details")
        details_layout.addWidget(details_button)
        
        # Store the call details as properties of the button for access when clicked
        details_button.setProperty("prompt", prompt_preview)
        details_button.setProperty("response", response_preview)
        details_button.setProperty("error", error_msg)
        details_button.setProperty("phi_count", phi_found_count)
        details_button.setProperty("phi_types", str(phi_types))
        
        details_button.clicked.connect(self.show_call_details)
        
        self.call_history_table.setCellWidget(0, 6, details_widget)
        
        # Resize columns to content
        self.call_history_table.resizeColumnsToContents()
    
    def refresh_call_history(self):
        # Clear the table
        self.call_history_table.setRowCount(0)
        
        # Load call history from database if available
        if llm_database is not None:
            try:
                calls = llm_database.get_recent_calls(limit=100)  # Get the most recent 100 calls
                
                for call in calls:
                    # call format: (timestamp, model, duration, input_tokens, output_tokens, status, error_msg, prompt, response, phi_scanned, phi_count, phi_types, phi_redacted)
                    self.on_llm_call_logged(
                        call[0], call[1], call[2], call[3], call[4], call[5], call[6], 
                        call[7][:100] + "..." if len(call[7]) > 100 else call[7],
                        call[8][:100] + "..." if len(call[8]) > 100 else call[8],
                        call[9], call[10], eval(call[11]) if call[11] else {}, call[12]
                    )
            except Exception as e:
                print(f"Error loading call history: {e}")
                traceback.print_exc()
    
    def clear_call_history(self):
        # Clear the table
        self.call_history_table.setRowCount(0)
        
        # Clear database if available
        if llm_database is not None:
            try:
                llm_database.clear_calls()
            except Exception as e:
                print(f"Error clearing call history: {e}")
    
    def show_call_details(self):
        # Get the sender button
        button = self.sender()
        
        # Get call details from button properties
        prompt = button.property("prompt")
        response = button.property("response")
        error = button.property("error")
        phi_count = button.property("phi_count")
        phi_types = button.property("phi_types")
        
        # Create details dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Call Details")
        dialog.setMinimumSize(600, 400)
        
        layout = QVBoxLayout(dialog)
        
        # Create tabs for different details
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Prompt tab
        prompt_tab = QWidget()
        prompt_layout = QVBoxLayout(prompt_tab)
        prompt_text = QPlainTextEdit()
        prompt_text.setPlainText(prompt)
        prompt_text.setReadOnly(True)
        prompt_layout.addWidget(prompt_text)
        tabs.addTab(prompt_tab, "Prompt")
        
        # Response tab
        response_tab = QWidget()
        response_layout = QVBoxLayout(response_tab)
        response_text = QPlainTextEdit()
        response_text.setPlainText(response)
        response_text.setReadOnly(True)
        response_layout.addWidget(response_text)
        tabs.addTab(response_tab, "Response")
        
        # Error tab (only if there's an error)
        if error:
            error_tab = QWidget()
            error_layout = QVBoxLayout(error_tab)
            error_text = QPlainTextEdit()
            error_text.setPlainText(error)
            error_text.setReadOnly(True)
            error_layout.addWidget(error_text)
            tabs.addTab(error_tab, "Error")
        
        # PHI tab (if PHI scanning was done)
        if phi_count > 0:
            phi_tab = QWidget()
            phi_layout = QVBoxLayout(phi_tab)
            phi_text = QPlainTextEdit()
            phi_text.setPlainText(f"PHI detected: {phi_count} instances\nPHI types: {phi_types}")
            phi_text.setReadOnly(True)
            phi_layout.addWidget(phi_text)
            tabs.addTab(phi_tab, f"PHI ({phi_count})")
        
        # Close button
        close_button = QPushButton("Close")
        layout.addWidget(close_button)
        close_button.clicked.connect(dialog.accept)
        
        # Show dialog
        dialog.exec_()

# Create a wrapper for LLM calls
def check_phi_and_process(model, prompt, process_llm_call):
    """
    Check for PHI in the prompt and process the LLM call
    
    Args:
        model: The LLM model to use
        prompt: The prompt to check for PHI
        process_llm_call: Function to process the LLM call with signature (model, processed_prompt) -> response
    
    Returns:
        The processed response from the LLM
    """
    start_time = datetime.datetime.now()
    timestamp = start_time
    
    # Default PHI values
    phi_scanned = phi_config.enabled
    phi_found_count = 0
    phi_types = {}
    phi_redacted = False
    
    # Skip PHI check if disabled
    if not phi_config.enabled:
        response = process_llm_call(model, prompt)
        
        # Log the call without PHI scanning
        duration = (datetime.datetime.now() - start_time).total_seconds()
        tokens_in = len(prompt.split()) if prompt else 0
        tokens_out = tokens_in  # Approximate if not provided by the model
        
        if isinstance(response, dict):
            status = "Error" if response.get("error", False) else "Success"
            error_msg = response.get("message", "") if response.get("error", False) else ""
            response_text = str(response.get("response", ""))
            tokens_in = response.get("input_tokens", tokens_in)
            tokens_out = response.get("output_tokens", tokens_out)
        else:
            status = "Success"
            error_msg = ""
            response_text = str(response)
        
        # Emit logging signal
        llm_tracker_signals.call_logged.emit(
            timestamp, model, duration, tokens_in, tokens_out, 
            status, error_msg, 
            prompt[:100] + "..." if len(prompt) > 100 else prompt, 
            response_text[:100] + "..." if len(response_text) > 100 else response_text,
            phi_scanned, phi_found_count, phi_types, phi_redacted
        )
        
        return response
    
    # Check for PHI in the prompt
    block_required, phi_report, redacted_prompt = phi_config.check_phi(prompt)
    
    # Update PHI values
    if phi_report:
        phi_found_count = phi_report['total_phi_count']
        phi_types = phi_report.get('phi_types', {})
    
    # If PHI is detected above threshold, block the call
    if block_required:
        # Emit signal for PHI blocking
        llm_tracker_signals.phi_blocked.emit(model, prompt, phi_report)
        
        # Calculate duration
        duration = (datetime.datetime.now() - start_time).total_seconds()
        
        # Determine if blocking is due to PHI or malicious content
        error_message = ""
        if phi_report.get('malicious_content', False) and phi_report.get('malicious_score', 0) >= 4:
            error_message = f"LLM call blocked: Malicious content detected with score {phi_report.get('malicious_score', 0)}"
        else:
            error_message = f"LLM call blocked: {phi_report['total_phi_count']} PHI elements detected above threshold of {phi_config.block_threshold}"
        
        # Log the blocked call
        llm_tracker_signals.call_logged.emit(
            timestamp, model, duration, len(prompt.split()) if prompt else 0, 0, 
            "Blocked", error_message, 
            prompt[:100] + "..." if len(prompt) > 100 else prompt, 
            "Call blocked due to content policy violation", 
            phi_scanned, phi_found_count, phi_types, False
        )
        
        # Return error message
        return {
            "error": True,
            "message": error_message,
            "phi_report": phi_report
        }
    
    # If PHI is detected but below threshold, use redacted prompt
    has_phi = phi_report and phi_report['total_phi_count'] > 0
    final_prompt = redacted_prompt if has_phi else prompt
    phi_redacted = has_phi
    
    # Process the LLM call with potentially redacted prompt
    response = process_llm_call(model, final_prompt)
    
    # Calculate duration
    duration = (datetime.datetime.now() - start_time).total_seconds()
    
    # Format response and extract metadata
    if isinstance(response, dict):
        response_status = "Error" if response.get("error", False) else "Success"
        error_message = response.get("message", "") if response.get("error", False) else ""
        response_text = str(response.get("response", ""))
        tokens_in = response.get("input_tokens", len(final_prompt.split()) if final_prompt else 0)
        tokens_out = response.get("output_tokens", len(response_text.split()) if response_text else 0)
        
        # If the response is a dictionary with metadata, add PHI info
        if not response.get("error", False):
            response["phi_detected"] = has_phi
            response["phi_redacted"] = has_phi
            response["original_prompt"] = prompt if has_phi else None
            response["phi_count"] = phi_found_count
            response["phi_types"] = phi_types
            # Add malicious content info
            response["malicious_content_checked"] = True
            response["malicious_content_detected"] = phi_report.get('malicious_content', False)
    else:
        response_status = "Success"
        error_message = ""
        response_text = str(response)
        tokens_in = len(final_prompt.split()) if final_prompt else 0
        tokens_out = len(response_text.split()) if response_text else 0
    
    # Log the call
    llm_tracker_signals.call_logged.emit(
        timestamp, model, duration, tokens_in, tokens_out,
        response_status, error_message,
        prompt[:100] + "..." if len(prompt) > 100 else prompt,
        response_text[:100] + "..." if len(response_text) > 100 else response_text,
        phi_scanned, phi_found_count, phi_types, phi_redacted
    )
    
    return response
