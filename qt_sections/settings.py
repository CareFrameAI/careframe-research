from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QLineEdit, QMessageBox, QScrollArea, QFrame,
                            QGroupBox, QTextEdit)
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QIcon, QColor, QPalette
import requests
from cryptography.fernet import Fernet
import base64
import logging
from admin.portal import secrets
import os
import json

class SettingsSection(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Initialize encryption BEFORE loading API keys
        self.init_encryption()
        self.init_ui()
        # Load API keys AFTER encryption is initialized
        self.load_api_keys()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Create a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        main_layout.addWidget(scroll)
        
        # Container for scrollable content
        container = QWidget()
        scroll_layout = QVBoxLayout(container)
        
        # Add a notification area for empty API keys
        self.notification_frame = QFrame()
        self.notification_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        self.notification_frame.setStyleSheet("background-color: #FFF3CD; padding: 10px; border-radius: 5px;")
        self.notification_frame.setVisible(False)  # Hidden by default
        
        notification_layout = QVBoxLayout(self.notification_frame)
        self.notification_label = QLabel()
        self.notification_label.setStyleSheet("color: #856404; font-weight: bold;")
        self.notification_label.setWordWrap(True)
        notification_layout.addWidget(self.notification_label)
        
        scroll_layout.addWidget(self.notification_frame)
        
        # Add description about API keys
        info_frame = QFrame()
        info_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        info_frame.setStyleSheet("background-color: #E3F2FD; padding: 10px; border-radius: 5px;")
        info_layout = QVBoxLayout(info_frame)
        
        info_title = QLabel("<h3>CareFrame API Keys Information</h3>")
        info_layout.addWidget(info_title)
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setStyleSheet("background-color: transparent; border: none;")
        info_text.setHtml("""
            <p>API keys are required for CareFrame to access various external services:</p>
            <ul>
                <li><b>Gemini API Key</b>: Used for AI-powered language capabilities</li>
                <li><b>Claude API Key</b>: Used for advanced AI language and vision capabilities</li>
                <li><b>UMLS API Key</b>: Provides access to medical terminology and concepts</li>
                <li><b>Unpaywall Email</b>: Helps retrieve open access research papers</li>
                <li><b>Zenodo API Key</b>: Access to research data repository</li>
                <li><b>CORE API Key</b>: Provides access to research paper aggregation</li>
                <li><b>Entrez API Key/Email</b>: Used for accessing PubMed and other NCBI databases</li>
            </ul>
            <p>Without these keys, certain features of CareFrame may be limited or unavailable.</p>
        """)
        info_text.setMaximumHeight(200)
        info_layout.addWidget(info_text)
        
        # Add a link to documentation
        docs_link = QLabel("<a href='#'>View Documentation on API Key Setup</a>")
        docs_link.setOpenExternalLinks(False)
        docs_link.linkActivated.connect(lambda: QDesktopServices.openUrl(QUrl("https://careframe.ai/docs/api-keys")))
        info_layout.addWidget(docs_link)
        
        scroll_layout.addWidget(info_frame)
        
        # API Keys section
        api_keys_frame = QFrame()
        api_keys_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        api_keys_layout = QVBoxLayout(api_keys_frame)
        
        title = QLabel("<h2>API Keys Settings</h2>")
        api_keys_layout.addWidget(title)
        
        # Dictionary to store API key input fields
        self.api_key_inputs = {}
        self.key_status_labels = {}
        
        # Add input fields for each API key
        key_names = [
            ("gemini_api_key", "Gemini API Key"),
            ("claude_api_key", "Claude API Key"),
            ("umls_api_key", "UMLS API Key"),
            ("unpaywall_email", "Unpaywall Email"),
            ("zenodo_api_key", "Zenodo API Key"),
            ("core_api_key", "CORE API Key"),
            ("entrez_api_key", "Entrez API Key"),
            ("entrez_email", "Entrez Email")
        ]
        
        for key_id, display_name in key_names:
            key_layout = QHBoxLayout()
            
            # Label
            label = QLabel(f"{display_name}:")
            label.setMinimumWidth(150)
            key_layout.addWidget(label)
            
            # Input field
            input_field = QLineEdit()
            input_field.setPlaceholderText(f"Enter {display_name}")
            self.api_key_inputs[key_id] = input_field
            key_layout.addWidget(input_field)
            
            # Status indicator
            status_label = QLabel("")
            status_label.setMinimumWidth(30)
            self.key_status_labels[key_id] = status_label
            key_layout.addWidget(status_label)
            
            # Remove button
            remove_btn = QPushButton("Remove")
            remove_btn.clicked.connect(lambda checked, k=key_id: self.remove_api_key(k))
            key_layout.addWidget(remove_btn)
            
            api_keys_layout.addLayout(key_layout)
        
        # Save button
        save_btn = QPushButton("Save API Keys")
        save_btn.clicked.connect(self.save_api_keys)
        api_keys_layout.addWidget(save_btn)
        
        scroll_layout.addWidget(api_keys_frame)
        scroll.setWidget(container)

    def init_encryption(self):
        """Initialize or load encryption key"""
        try:
            # First try to use the file-based encryption key
            key_file = '.encryption_key'
            if os.path.exists(key_file):
                with open(key_file, 'rb') as f:
                    encryption_key_data = f.read()
                # Validate the key format
                try:
                    self.cipher_suite = Fernet(encryption_key_data)
                    self.encryption_key = encryption_key_data
                    return
                except Exception as e:
                    logging.warning(f"Invalid encryption key format in file: {e}")
                    # Continue to try CouchDB or generate a new key
            
            # Try CouchDB next
            response = requests.get(
                "http://localhost:5984/settings/encryption_key",
                auth=("admin", "cfpwd"),
                timeout=1  # Use a short timeout to fail fast
            )
            
            if response.status_code == 200:
                key_data = response.json()
                self.encryption_key = base64.b64decode(key_data['key'])
            else:
                # Generate new key if not exists
                self.encryption_key = Fernet.generate_key()
                # Try to save to CouchDB if it's available
                try:
                    requests.put(
                        "http://localhost:5984/settings/encryption_key",
                        auth=("admin", "cfpwd"),
                        json={
                            "type": "encryption_key",
                            "key": base64.b64encode(self.encryption_key).decode()
                        },
                        timeout=1  # Use a short timeout
                    )
                except:
                    # If CouchDB fails, save to file
                    with open(key_file, 'wb') as f:
                        f.write(self.encryption_key)
            
            self.cipher_suite = Fernet(self.encryption_key)
            
        except requests.RequestException as e:
            logging.warning(f"Failed to initialize encryption from CouchDB: {e}")
            # Try to use the file-based encryption key instead of showing an error
            try:
                # Generate a new key and save to file
                self.encryption_key = Fernet.generate_key()
                key_file = '.encryption_key'
                with open(key_file, 'wb') as f:
                    f.write(self.encryption_key)
                self.cipher_suite = Fernet(self.encryption_key)
                logging.info("Using file-based encryption key")
            except Exception as file_error:
                logging.error(f"Failed to initialize encryption from file: {file_error}")
                # Only show the error message if both methods fail
                QMessageBox.warning(self, "Warning", "Failed to initialize encryption. Some features may not work correctly.")

    def encrypt_value(self, value):
        """Encrypt a value using Fernet encryption"""
        try:
            return base64.b64encode(
                self.cipher_suite.encrypt(value.encode())
            ).decode()
        except Exception as e:
            logging.error(f"Encryption error: {e}")
            return ""

    def decrypt_value(self, encrypted_value):
        """Decrypt a value using Fernet encryption"""
        try:
            return self.cipher_suite.decrypt(
                base64.b64decode(encrypted_value)
            ).decode()
        except Exception as e:
            logging.error(f"Decryption error: {e}")
            return ""

    def load_api_keys(self):
        """Load API keys from database and update UI"""
        keys_loaded = False
        try:
            # First try to load from file-based storage
            mock_dir = os.path.join(os.path.expanduser("~"), '.careframe', 'mock_couchdb', 'settings', 'docs')
            api_keys_path = os.path.join(mock_dir, 'api_keys.json')
            
            if os.path.exists(api_keys_path):
                try:
                    with open(api_keys_path, 'r') as f:
                        doc = json.load(f)
                    keys_dict = doc.get('keys', {})
                    keys_loaded = True
                    logging.info("Loaded API keys from mock storage")
                except Exception as file_e:
                    logging.warning(f"Error loading API keys from file: {file_e}")
            
            # If not loaded from file, try CouchDB
            if not keys_loaded:
                # Check if API keys exist in database
                response = requests.get(
                    "http://localhost:5984/settings/api_keys",
                    auth=("admin", "cfpwd"),
                    timeout=1  # Use a short timeout
                )
                
                if response.status_code == 200:
                    doc = response.json()
                    keys_dict = doc.get('keys', {})
                    keys_loaded = True
                else:
                    # Create new empty API keys document
                    self._create_empty_api_keys()
                    keys_dict = {}
            
            # Count missing keys
            missing_keys = []
            
            # Populate UI fields and update status indicators
            for key_id, input_field in self.api_key_inputs.items():
                # Get from secrets first (cloud values)
                value = secrets.get(key_id, "")
                
                # If not in secrets, try from database
                if not value and key_id in keys_dict:
                    encrypted_value = keys_dict.get(key_id, "")
                    if encrypted_value:
                        value = self.decrypt_value(encrypted_value)
                
                # Update field
                input_field.setText(value)
                
                # Update status indicator
                if value:
                    self.key_status_labels[key_id].setText("✅")
                    self.key_status_labels[key_id].setStyleSheet("color: green;")
                else:
                    self.key_status_labels[key_id].setText("❌")
                    self.key_status_labels[key_id].setStyleSheet("color: red;")
                    missing_keys.append(key_id)
            
            # Show notification if keys are missing
            if missing_keys:
                self.notification_frame.setVisible(True)
                self.notification_label.setText(
                    f"Warning: {len(missing_keys)} API keys are missing. "
                    "Some features may be limited or unavailable. "
                    "Please add the missing API keys below."
                )
            else:
                self.notification_frame.setVisible(False)
                
        except requests.RequestException as e:
            logging.warning(f"Failed to load API keys from CouchDB: {e}")
            # Create an empty set of keys instead of showing an error
            for key_id in self.api_key_inputs.keys():
                self.key_status_labels[key_id].setText("❌")
                self.key_status_labels[key_id].setStyleSheet("color: red;")
                
            # Show notification about API keys
            self.notification_frame.setVisible(True)
            self.notification_label.setText(
                "API keys have not been configured. "
                "Please set up your API keys to enable all features."
            )

    def _create_empty_api_keys(self):
        """Create an empty API keys document if it doesn't exist"""
        try:
            empty_keys = {key_id: "" for key_id in self.api_key_inputs.keys()}
            requests.put(
                "http://localhost:5984/settings/api_keys",
                auth=("admin", "cfpwd"),
                json={
                    "_id": "api_keys",
                    "type": "api_keys",
                    "keys": empty_keys
                }
            )
        except requests.RequestException as e:
            logging.error(f"Failed to create empty API keys: {e}")

    def save_api_keys(self):
        """Save API keys to database"""
        try:
            # Collect and encrypt API keys
            encrypted_keys = {}
            missing_keys = []
            
            for key_id, input_field in self.api_key_inputs.items():
                value = input_field.text().strip()
                if value:
                    encrypted_keys[key_id] = self.encrypt_value(value)
                    # Update status indicator
                    self.key_status_labels[key_id].setText("✅")
                    self.key_status_labels[key_id].setStyleSheet("color: green;")
                else:
                    encrypted_keys[key_id] = ""
                    missing_keys.append(key_id)
                    # Update status indicator
                    self.key_status_labels[key_id].setText("❌")
                    self.key_status_labels[key_id].setStyleSheet("color: red;")
            
            # Update notification area
            if missing_keys:
                self.notification_frame.setVisible(True)
                self.notification_label.setText(
                    f"Warning: {len(missing_keys)} API keys are still missing. "
                    "Some features may be limited or unavailable."
                )
            else:
                self.notification_frame.setVisible(False)
            
            # Get existing document if it exists
            response = requests.get(
                "http://localhost:5984/settings/api_keys",
                auth=("admin", "cfpwd")
            )
            
            if response.status_code == 200:
                doc = response.json()
                doc['keys'] = encrypted_keys
                update_response = requests.put(
                    "http://localhost:5984/settings/api_keys",
                    auth=("admin", "cfpwd"),
                    json=doc
                )
                
                if update_response.status_code in (201, 200):
                    QMessageBox.information(self, "Success", "API keys saved successfully")
                else:
                    QMessageBox.warning(self, "Error", "Failed to save API keys")
            else:
                # Create new document
                new_doc = {
                    "_id": "api_keys",
                    "type": "api_keys",
                    "keys": encrypted_keys
                }
                create_response = requests.put(
                    "http://localhost:5984/settings/api_keys",
                    auth=("admin", "cfpwd"),
                    json=new_doc
                )
                
                if create_response.status_code in (201, 200):
                    QMessageBox.information(self, "Success", "API keys saved successfully")
                else:
                    QMessageBox.warning(self, "Error", "Failed to save API keys")
                
        except requests.RequestException as e:
            logging.error(f"Failed to save API keys: {e}")
            QMessageBox.warning(self, "Error", "Failed to save API keys")

    def remove_api_key(self, key_id):
        """Remove an API key"""
        try:
            # Clear the input field
            self.api_key_inputs[key_id].clear()
            
            # Update status indicator
            self.key_status_labels[key_id].setText("❌")
            self.key_status_labels[key_id].setStyleSheet("color: red;")
            
            # Remove from database
            response = requests.get(
                "http://localhost:5984/settings/api_keys",
                auth=("admin", "cfpwd")
            )
            
            if response.status_code == 200:
                doc = response.json()
                if key_id in doc.get('keys', {}):
                    doc['keys'][key_id] = ""
                    update_response = requests.put(
                        "http://localhost:5984/settings/api_keys",
                        auth=("admin", "cfpwd"),
                        json=doc
                    )
                    
                    if update_response.status_code in (201, 200):
                        QMessageBox.information(self, "Success", f"Removed {key_id}")
                    else:
                        QMessageBox.warning(self, "Error", "Failed to remove API key")
            
            # Update notification
            self._update_missing_keys_notification()
                        
        except requests.RequestException as e:
            logging.error(f"Failed to remove API key: {e}")
            QMessageBox.warning(self, "Error", "Failed to remove API key")
    
    def _update_missing_keys_notification(self):
        """Update the notification about missing keys"""
        missing_keys = []
        for key_id, input_field in self.api_key_inputs.items():
            if not input_field.text().strip():
                missing_keys.append(key_id)
        
        if missing_keys:
            self.notification_frame.setVisible(True)
            self.notification_label.setText(
                f"Warning: {len(missing_keys)} API keys are missing. "
                "Some features may be limited or unavailable."
            )
        else:
            self.notification_frame.setVisible(False)
