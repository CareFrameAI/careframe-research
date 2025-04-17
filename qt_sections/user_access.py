from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLineEdit, 
                            QPushButton, QLabel, QMessageBox, QCheckBox,
                            QDialog, QApplication, QHBoxLayout, QFrame,
                            QListWidget, QListWidgetItem, QFileDialog,
                            QFormLayout, QInputDialog)
import requests
import json
import hashlib
import os
import base64
from datetime import datetime, timedelta
from PyQt6.QtCore import pyqtSignal, Qt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from helpers.load_icon import load_bootstrap_icon

class UserAccess:
    def __init__(self, base_url="http://localhost:5984", auth=("admin", "cfpwd")):
        self.base_url = base_url
        self.auth = auth
        self.current_user = None
        self.ensure_user_views()
        self._init_encryption()
        self.load_remembered_user()

    def ensure_user_views(self):
        """Ensure necessary views exist in the users database"""
        views = {
            '_design/auth': {
                'views': {
                    'by_email': {
                        'map': 'function (doc) { if (doc.email) emit(doc.email, doc); }'
                    },
                    'by_token': {
                        'map': 'function (doc) { if (doc.reset_token) emit(doc.reset_token, doc); }'
                    }
                }
            }
        }
        
        try:
            requests.put(
                f"{self.base_url}/users/_design/auth",
                auth=self.auth,
                json=views
            )
        except requests.RequestException as e:
            print(f"Error ensuring views: {e}")

    def hash_password(self, password):
        """Create a secure hash of the password"""
        salt = os.urandom(32)
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000
        )
        return base64.b64encode(salt + key).decode('utf-8')

    def verify_password(self, stored_hash, provided_password):
        """Verify a password against its hash"""
        try:
            decoded = base64.b64decode(stored_hash.encode('utf-8'))
            salt = decoded[:32]
            key = decoded[32:]
            new_key = hashlib.pbkdf2_hmac(
                'sha256',
                provided_password.encode('utf-8'),
                salt,
                100000
            )
            return key == new_key
        except:
            return False

    def login(self, email, password):
        """Attempt to log in a user"""
        print(f"UserAccess: Attempting login for {email}")  # Debug print
        try:
            response = requests.get(
                f"{self.base_url}/users/_design/auth/_view/by_email?key=\"{email}\"",
                auth=self.auth
            )
            print(f"UserAccess: Got response: {response.status_code}")  # Debug print
            
            if response.status_code == 200 and response.json()['rows']:
                user_doc = response.json()['rows'][0]['value']
                print(f"UserAccess: Found user document")  # Debug print
                
                if not user_doc.get('password_hash'):
                    print("UserAccess: No password set")  # Debug print
                    # User exists but hasn't set password
                    return {'status': 'no_password', 'user': user_doc}
                
                if self.verify_password(user_doc['password_hash'], password):
                    print("UserAccess: Password verified")  # Debug print
                    self.current_user = user_doc
                    return {'status': 'success', 'user': user_doc}
                print("UserAccess: Invalid password")  # Debug print
                return {'status': 'invalid_password'}
            print("UserAccess: User not found")  # Debug print
            return {'status': 'user_not_found'}
        
        except requests.RequestException as e:
            print(f"UserAccess: Error - {str(e)}")  # Debug print
            return {'status': 'error', 'message': str(e)}

    def set_password(self, email, password, admin_token=None):
        """Set or update a user's password"""
        try:
            # Verify user exists
            response = requests.get(
                f"{self.base_url}/users/_design/auth/_view/by_email?key=\"{email}\"",
                auth=self.auth
            )
            
            if not response.json()['rows']:
                return {'status': 'error', 'message': 'User not found'}
            
            user_doc = response.json()['rows'][0]['value']
            
            # Check authorization
            if not (self.current_user and self.current_user.get('is_admin')) and not admin_token:
                if user_doc.get('password_hash'):
                    return {'status': 'error', 'message': 'Unauthorized'}
            
            # Update password
            user_doc['password_hash'] = self.hash_password(password)
            user_doc['password_updated_at'] = datetime.now().isoformat()
            
            # Clear reset token if exists
            user_doc.pop('reset_token', None)
            
            response = requests.put(
                f"{self.base_url}/users/{user_doc['_id']}",
                auth=self.auth,
                json=user_doc
            )
            
            if response.status_code == 201:
                return {'status': 'success'}
            return {'status': 'error', 'message': 'Failed to update password'}
            
        except requests.RequestException as e:
            return {'status': 'error', 'message': str(e)}

    def generate_admin_token(self, admin_email):
        """Generate a secure reset token for admin"""
        try:
            response = requests.get(
                f"{self.base_url}/users/_design/auth/_view/by_email?key=\"{admin_email}\"",
                auth=self.auth
            )
            
            if not response.json()['rows']:
                return {'status': 'error', 'message': 'Admin not found'}
            
            admin_doc = response.json()['rows'][0]['value']
            if not admin_doc.get('is_admin'):
                return {'status': 'error', 'message': 'User is not an admin'}
            
            # Generate secure token
            token = base64.b64encode(os.urandom(32)).decode('utf-8')
            admin_doc['reset_token'] = token
            
            response = requests.put(
                f"{self.base_url}/users/{admin_doc['_id']}",
                auth=self.auth,
                json=admin_doc
            )
            
            if response.status_code == 201:
                return {'status': 'success', 'token': token}
            return {'status': 'error', 'message': 'Failed to generate token'}
            
        except requests.RequestException as e:
            return {'status': 'error', 'message': str(e)}

    def _init_encryption(self):
        """Initialize encryption key using system-specific data"""
        try:
            # Load or generate encryption key
            key_file = '.encryption_key'
            if os.path.exists(key_file):
                with open(key_file, 'rb') as f:
                    encryption_key_data = f.read()
                # Validate the key
                try:
                    self.fernet = Fernet(encryption_key_data)
                    self.encryption_key = encryption_key_data
                    print("Using existing encryption key from file")
                    return
                except Exception as e:
                    print(f"Invalid encryption key format in file: {e}")
                    # Continue to generate a new key
            
            # If we don't have a valid key from file, generate a new one
            # Generate a new key
            salt = os.urandom(16)
            # Use machine-specific data as part of the key generation
            system_data = f"{os.name}:{os.getlogin()}".encode()
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            self.encryption_key = base64.urlsafe_b64encode(kdf.derive(system_data))
            
            # Save the key
            with open(key_file, 'wb') as f:
                f.write(self.encryption_key)
            
            self.fernet = Fernet(self.encryption_key)
            print("Created and saved new encryption key")
        except Exception as e:
            print(f"Error initializing encryption: {e}")
            # Create a basic fallback key if all else fails
            try:
                self.encryption_key = Fernet.generate_key()
                self.fernet = Fernet(self.encryption_key)
                # Try to save it to file
                with open(key_file, 'wb') as f:
                    f.write(self.encryption_key)
                print("Created fallback encryption key")
            except:
                print("Failed to create fallback encryption key")
                self.fernet = None

    def save_remembered_user(self, email, password_hash):
        """Save user credentials securely with encryption and expiration"""
        try:
            if not self.fernet:
                raise Exception("Encryption not initialized")
                
            credentials = {
                'email': email,
                'password_hash': password_hash,
                'timestamp': datetime.now().isoformat(),
                'expiration': (datetime.now() + timedelta(days=30)).isoformat()
            }
            
            # Convert to JSON and encrypt
            json_data = json.dumps(credentials)
            encrypted_data = self.fernet.encrypt(json_data.encode())
            
            # Save encrypted data
            with open('.credentials', 'wb') as f:
                f.write(encrypted_data)
                
            print("Credentials saved securely")
        except Exception as e:
            print(f"Error saving credentials: {e}")
            self.clear_remembered_user()

    def load_remembered_user(self):
        """Load and verify saved credentials with expiration check"""
        try:
            if not os.path.exists('.credentials') or not self.fernet:
                return None
                
            # Read and decrypt credentials
            with open('.credentials', 'rb') as f:
                encrypted_data = f.read()
                
            decrypted_data = self.fernet.decrypt(encrypted_data)
            credentials = json.loads(decrypted_data)
            
            # Check expiration
            expiration = datetime.fromisoformat(credentials['expiration'])
            if datetime.now() > expiration:
                print("Saved credentials have expired")
                self.clear_remembered_user()
                return None
            
            # Verify credentials against database
            response = requests.get(
                f"{self.base_url}/users/_design/auth/_view/by_email?key=\"{credentials['email']}\"",
                auth=self.auth
            )
            
            if response.status_code == 200 and response.json()['rows']:
                user_doc = response.json()['rows'][0]['value']
                if user_doc['password_hash'] == credentials['password_hash']:
                    self.current_user = user_doc
                    
                    # Refresh expiration time
                    self.save_remembered_user(
                        credentials['email'],
                        credentials['password_hash']
                    )
                    
                    return {'status': 'success', 'user': user_doc}
            
            # If verification fails, clear saved credentials
            self.clear_remembered_user()
            
        except Exception as e:
            print(f"Error loading credentials: {e}")
            self.clear_remembered_user()
        return None

    def clear_remembered_user(self):
        """Clear saved credentials and optionally encryption key"""
        try:
            # Remove credentials file
            if os.path.exists('.credentials'):
                os.remove('.credentials')
            print("Credentials cleared")
        except Exception as e:
            print(f"Error clearing credentials: {e}")

    def logout(self):
        """Logout user and clear credentials"""
        self.clear_remembered_user()
        self.current_user = None
        return {'status': 'success'}

class LoginDialog(QDialog):
    loginSuccessful = pyqtSignal(dict)

    def __init__(self, user_access, parent=None):
        super().__init__(parent)
        self.user_access = user_access
        self.setWindowTitle("Login")
        self.setModal(True)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)  # Increased margins
        layout.setSpacing(15)  # Increased spacing
        
        # Title label with larger font
        title_label = QLabel("Login to CareFrameAI")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 15px;
                color: #333333;
            }
        """)
        layout.addWidget(title_label)
        
        # Email input with styling and icon
        email_container = QHBoxLayout()
        email_icon = QLabel()
        email_icon.setPixmap(load_bootstrap_icon("envelope").pixmap(24, 24))
        email_container.addWidget(email_icon)
        
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("Email")
        self.email_input.setMinimumHeight(40)  # Increased height
        self.email_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
                font-size: 14px;
                background-color: white;
                color: #333333;
            }
            QLineEdit:focus {
                border: 2px solid #2196F3;
            }
        """)
        email_container.addWidget(self.email_input)
        layout.addLayout(email_container)
        
        # Password input with styling and icon
        password_container = QHBoxLayout()
        password_icon = QLabel()
        password_icon.setPixmap(load_bootstrap_icon("lock").pixmap(24, 24))
        password_container.addWidget(password_icon)
        
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setMinimumHeight(40)
        self.password_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
                font-size: 14px;
                background-color: white;
                color: #333333;
            }
            QLineEdit:focus {
                border: 2px solid #2196F3;
            }
        """)
        password_container.addWidget(self.password_input)
        layout.addLayout(password_container)
        
        # Remember me checkbox with improved styling
        self.remember_me = QCheckBox("Remember me")
        self.remember_me.setStyleSheet("""
            QCheckBox {
                font-size: 13px;
                color: #333333;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #ccc;
                border-radius: 3px;
                background-color: white;
            }
            QCheckBox::indicator:unchecked:hover {
                border-color: #2196F3;
            }
            QCheckBox::indicator:checked {
                background-color: #2196F3;
                border-color: #2196F3;
                image: url(check.png);
            }
            QCheckBox::indicator:checked:hover {
                background-color: #1976D2;
                border-color: #1976D2;
            }
        """)
        layout.addWidget(self.remember_me)
        
        # Login button with improved styling and icon
        self.login_button = QPushButton("Login")
        self.login_button.setIcon(load_bootstrap_icon("box-arrow-in-right"))
        self.login_button.setMinimumHeight(40)
        self.login_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 15px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        self.login_button.clicked.connect(self.attempt_login)
        layout.addWidget(self.login_button)
        
        # Status label with styling
        self.status_label = QLabel()
        self.status_label.setStyleSheet("""
            QLabel {
                color: #f44336;
                font-size: 13px;
                min-height: 20px;
            }
        """)
        layout.addWidget(self.status_label)

        # Set minimum size and center the dialog
        self.setMinimumWidth(400)
        self.setMinimumHeight(400)  # Increased height
        
        # Apply global dialog styling with light gray background
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f5;
            }
            QLabel {
                color: #333333;
            }
        """)
        
        self.center_dialog()

    def center_dialog(self):
        """Center the dialog on the screen"""
        screen = QApplication.primaryScreen().geometry()
        dialog_size = self.geometry()
        x = (screen.width() - dialog_size.width()) // 2
        y = (screen.height() - dialog_size.height()) // 2
        self.move(x, y)

    def attempt_login(self):
        email = self.email_input.text()
        password = self.password_input.text()
        
        if not email or not password:
            self.status_label.setText("Please enter both email and password")
            return
        
        result = self.user_access.login(email, password)
        
        if result['status'] == 'success':
            # Handle "Remember Me" functionality with encryption
            if self.remember_me.isChecked():
                self.user_access.save_remembered_user(
                    email, 
                    result['user']['password_hash']
                )
            else:
                self.user_access.clear_remembered_user()
                
            self.status_label.setStyleSheet("color: #4CAF50;")
            self.status_label.setText("Login successful!")
            self.loginSuccessful.emit(result['user'])
            self.accept()
        elif result['status'] == 'no_password':
            response = QMessageBox.question(
                self,
                "Set Password",
                "Would you like to set your password now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if response == QMessageBox.StandardButton.Yes:
                self.show_set_password_dialog(email)
        else:
            self.status_label.setText(f"Login failed: {result.get('message', 'Invalid credentials')}")

    def show_set_password_dialog(self, email):
        dialog = QDialog(self)
        dialog.setWindowTitle("Set Password")
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        
        # Title label with larger font
        title = QLabel(f"Set password for {email}")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)
        
        # Password fields with icons and better styling
        new_password_container = QHBoxLayout()
        new_password_icon = QLabel()
        new_password_icon.setPixmap(load_bootstrap_icon("key").pixmap(24, 24))
        new_password_container.addWidget(new_password_icon)
        
        new_password = QLineEdit()
        new_password.setPlaceholderText("New Password")
        new_password.setEchoMode(QLineEdit.EchoMode.Password)
        new_password.setMinimumHeight(30)
        new_password.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid #2196F3;
            }
        """)
        new_password_container.addWidget(new_password)
        layout.addLayout(new_password_container)
        
        # Confirm password field with icon
        confirm_password_container = QHBoxLayout()
        confirm_password_icon = QLabel()
        confirm_password_icon.setPixmap(load_bootstrap_icon("check2-circle").pixmap(24, 24))
        confirm_password_container.addWidget(confirm_password_icon)
        
        confirm_password = QLineEdit()
        confirm_password.setPlaceholderText("Confirm Password")
        confirm_password.setEchoMode(QLineEdit.EchoMode.Password)
        confirm_password.setMinimumHeight(30)
        confirm_password.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid #2196F3;
            }
        """)
        confirm_password_container.addWidget(confirm_password)
        layout.addLayout(confirm_password_container)
        
        # Submit button with better styling and icon
        submit_btn = QPushButton("Set Password")
        submit_btn.setIcon(load_bootstrap_icon("shield-lock"))
        submit_btn.setMinimumHeight(35)
        submit_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        layout.addWidget(submit_btn)
        
        def set_password():
            if new_password.text() != confirm_password.text():
                QMessageBox.warning(dialog, "Error", "Passwords do not match!")
                return
            
            result = self.user_access.set_password(email, new_password.text())
            if result['status'] == 'success':
                QMessageBox.information(dialog, "Success", "Password set successfully! Please log in.")
                dialog.accept()
                self.email_input.setText(email)
                self.password_input.setFocus()
            else:
                QMessageBox.warning(dialog, "Error", result.get('message', 'Failed to set password'))
        
        submit_btn.clicked.connect(set_password)
        
        # Set size and center
        dialog.setMinimumWidth(400)
        dialog.setMinimumHeight(250)
        
        # Center on parent
        parent_geometry = self.geometry()
        dialog.move(
            parent_geometry.center().x() - dialog.width() // 2,
            parent_geometry.center().y() - dialog.height() // 2
        )
        
        dialog.exec()

    def logout(self):
        """Handle logout and clear credentials"""
        self.user_access.logout()  # This will clear credentials and current user
        self.email_input.clear()
        self.password_input.clear()
        self.remember_me.setChecked(False)
        self.status_label.clear()
