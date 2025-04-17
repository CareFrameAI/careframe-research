from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                            QTreeWidget, QTreeWidgetItem, QLabel, 
                            QTextEdit, QComboBox, QGroupBox, QPushButton, QLineEdit, QMessageBox, QFileDialog, QDialog, QApplication)
from PyQt6.QtCore import Qt
import requests
import json
from datetime import datetime
import logging
import os
from pathlib import Path

# Import our database setup and mock implementation
from db_ops.generate_tables import DatabaseSetup
from db_ops.json_db import MockCouchDB, MockCouchDBResourceNotFound

logger = logging.getLogger("database_section")

class DatabaseSection(QWidget):
    def __init__(self):
        super().__init__()
        self.user_access = None  # Will be set by MainWindow
        
        # Initialize the database setup
        self.db_setup = DatabaseSetup()
        
        # CouchDB connection settings
        self.base_url = "http://localhost:5984"
        self.auth = ("admin", "cfpwd")
        
        # Initialize UI
        self.init_ui()
        
        # Initial data load
        self.refresh_databases()
        
        # Add user management section
        self.add_user_management()
        
        # Show database type info
        self.show_database_info()

    def init_ui(self):
        layout = QHBoxLayout()
        
        # Left side - Database/Table tree
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        # Database type indicator
        self.db_type_label = QLabel("Database Type: Loading...")
        left_layout.addWidget(self.db_type_label)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh Databases")
        refresh_btn.clicked.connect(self.refresh_databases)
        left_layout.addWidget(refresh_btn)
        
        # Setup button
        setup_btn = QPushButton("Setup/Initialize Databases")
        setup_btn.clicked.connect(self.setup_databases)
        left_layout.addWidget(setup_btn)
        
        # Database selector
        self.db_selector = QComboBox()
        self.db_selector.currentTextChanged.connect(self.on_database_selected)
        left_layout.addWidget(QLabel("Select Database:"))
        left_layout.addWidget(self.db_selector)
        
        # Table tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Tables/Views"])
        self.tree.itemClicked.connect(self.on_item_selected)
        left_layout.addWidget(self.tree)
        
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(300)
        
        # Right side - Content viewer
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        self.content_viewer = QTextEdit()
        self.content_viewer.setReadOnly(True)  # Make it read-only
        right_layout.addWidget(QLabel("Content:"))
        right_layout.addWidget(self.content_viewer)
        
        right_panel.setLayout(right_layout)
        
        # Add panels to main layout
        layout.addWidget(left_panel)
        layout.addWidget(right_panel)
        
        self.setLayout(layout)

    def show_database_info(self):
        """Display information about the database type."""
        info = self.db_setup.get_database_info()
        
        if "error" in info:
            self.db_type_label.setText(f"Database Error: {info['error']}")
            self.db_type_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            db_type = info["server_type"]
            if db_type == "JSON Mock":
                self.db_type_label.setText("Database: JSON Mock (Local Files)")
                self.db_type_label.setStyleSheet("color: orange; font-weight: bold;")
            else:
                self.db_type_label.setText(f"Database: {db_type} (Server)")
                self.db_type_label.setStyleSheet("color: green; font-weight: bold;")

    def setup_databases(self):
        """Set up all required databases and their views."""
        try:
            result = self.db_setup.setup_databases()
            db_names = ", ".join(result.keys())
            QMessageBox.information(self, "Success", f"Databases initialized/verified: {db_names}")
            self.refresh_databases()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to set up databases: {e}")

    def refresh_databases(self):
        """Fetch and display all databases"""
        try:
            info = self.db_setup.get_database_info()
            
            if "error" in info:
                self.content_viewer.setText(f"Error connecting to database: {info['error']}")
                return
            
            # Update database type display
            self.show_database_info()
            
            # Update database selector
            self.db_selector.clear()
            self.db_selector.addItems(info["databases"])
        except Exception as e:
            self.content_viewer.setText(f"Error connecting to database: {str(e)}")

    def on_database_selected(self, db_name):
        """When a database is selected, fetch and display its views/tables"""
        if not db_name:
            return
            
        try:
            # Use our database setup to get the server
            db = self.db_setup.server[db_name]
            
            self.tree.clear()
            
            # Create root item for documents
            docs_root = QTreeWidgetItem(self.tree, ["Documents"])
            
            # Add documents (handle differently based on mock or real CouchDB)
            if self.db_setup.use_mock:
                # For mock CouchDB - directly access _documents attribute
                for doc_id in db._documents:
                    if not doc_id.startswith('_'):  # Skip design documents
                        QTreeWidgetItem(docs_root, [doc_id])
                
                # Handle design documents and views
                if db._design_docs:
                    views_root = QTreeWidgetItem(self.tree, ["Views"])
                    for design_name, design_doc in db._design_docs.items():
                        if 'views' in design_doc:
                            for view_name in design_doc['views']:
                                view_item = QTreeWidgetItem(views_root, [view_name])
                                view_item.setData(0, Qt.ItemDataRole.UserRole, 
                                                {'design_doc': design_name, 'view': view_name})
            else:
                # For real CouchDB - use the REST API
                # Get all documents in the database
                response = requests.get(
                    f"{self.db_setup.base_url}/{db_name}/_all_docs",
                    auth=self.db_setup.auth
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Add each document as a child
                    for row in data.get('rows', []):
                        doc_id = row.get('id')
                        if doc_id and not doc_id.startswith('_'):  # Skip design documents
                            QTreeWidgetItem(docs_root, [doc_id])
                    
                    # Get design documents (views)
                    design_response = requests.get(
                        f"{self.db_setup.base_url}/{db_name}/_design_docs",
                        auth=self.db_setup.auth
                    )
                    
                    if design_response.status_code == 200:
                        design_data = design_response.json()
                        views_root = QTreeWidgetItem(self.tree, ["Views"])
                        
                        for row in design_data.get('rows', []):
                            design_doc_id = row.get('id')
                            if design_doc_id:
                                # Get the actual design document
                                doc_response = requests.get(
                                    f"{self.db_setup.base_url}/{db_name}/{design_doc_id}",
                                    auth=self.db_setup.auth
                                )
                                if doc_response.status_code == 200:
                                    design_doc = doc_response.json()
                                    views = design_doc.get('views', {})
                                    
                                    for view_name in views:
                                        design_name = design_doc_id.split('/')[-1]
                                        view_item = QTreeWidgetItem(views_root, [view_name])
                                        view_item.setData(0, Qt.ItemDataRole.UserRole, 
                                                        {'design_doc': design_name, 'view': view_name})
            
            self.tree.expandAll()
                
        except Exception as e:
            self.content_viewer.setText(f"Error fetching database content: {str(e)}")

    def on_item_selected(self, item):
        """When a tree item is selected, display its content"""
        if not item.parent():  # Root items
            return
            
        db_name = self.db_selector.currentText()
        
        try:
            # Get database from our setup
            db = self.db_setup.server[db_name]
            
            if item.parent().text(0) == "Documents":
                # Fetch individual document
                doc_id = item.text(0)
                
                if self.db_setup.use_mock:
                    # Use direct access for mock
                    try:
                        doc = db[doc_id]
                        content = json.dumps(doc._data, indent=2)
                        self.content_viewer.setText(content)
                    except MockCouchDBResourceNotFound as e:
                        self.content_viewer.setText(f"Error: Document not found\n{str(e)}")
                else:
                    # Use REST API for real CouchDB
                    response = requests.get(
                        f"{self.db_setup.base_url}/{db_name}/{doc_id}",
                        auth=self.db_setup.auth
                    )
                    
                    if response.status_code == 200:
                        content = json.dumps(response.json(), indent=2)
                        self.content_viewer.setText(content)
                    else:
                        self.content_viewer.setText(f"Error: {response.status_code}\n{response.text}")
                
            elif item.parent().text(0) == "Views":
                # Fetch view results
                view_data = item.data(0, Qt.ItemDataRole.UserRole)
                design_doc = view_data['design_doc']
                view_name = view_data['view']
                
                if self.db_setup.use_mock:
                    # Use direct view query for mock
                    try:
                        results = db.view(design_doc, view_name)
                        content = json.dumps(results, indent=2)
                        self.content_viewer.setText(content)
                    except MockCouchDBResourceNotFound as e:
                        self.content_viewer.setText(f"Error: View not found\n{str(e)}")
                else:
                    # Use REST API for real CouchDB
                    response = requests.get(
                        f"{self.db_setup.base_url}/{db_name}/{design_doc}/_view/{view_name}",
                        auth=self.db_setup.auth
                    )
                    
                    if response.status_code == 200:
                        content = json.dumps(response.json(), indent=2)
                        self.content_viewer.setText(content)
                    else:
                        self.content_viewer.setText(f"Error: {response.status_code}\n{response.text}")
                    
        except Exception as e:
            self.content_viewer.setText(f"Error fetching content: {str(e)}")

    def add_user_management(self):
        user_panel = QWidget()
        user_layout = QVBoxLayout()
        
        # Login section
        login_group = QGroupBox("User Login")
        login_layout = QVBoxLayout()
        
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("Email")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        
        login_button = QPushButton("Login")
        login_button.clicked.connect(self.login_user)
        logout_button = QPushButton("Logout")
        logout_button.clicked.connect(self.logout_user)
        
        login_layout.addWidget(self.email_input)
        login_layout.addWidget(self.password_input)
        login_layout.addWidget(login_button)
        login_layout.addWidget(logout_button)
        login_group.setLayout(login_layout)
        
        user_layout.addWidget(login_group)
        
        # Add admin token management
        admin_group = QGroupBox("Admin Tools")
        admin_layout = QVBoxLayout()
        
        generate_token_button = QPushButton("Generate Admin Recovery Token")
        generate_token_button.clicked.connect(self.generate_admin_token)
        
        use_token_button = QPushButton("Use Recovery Token")
        use_token_button.clicked.connect(self.use_admin_token)
        
        admin_layout.addWidget(generate_token_button)
        admin_layout.addWidget(use_token_button)
        admin_group.setLayout(admin_layout)
        
        user_layout.addWidget(admin_group)
        
        user_panel.setLayout(user_layout)
        
        # Add to main layout
        self.layout().addWidget(user_panel)

    def login_user(self):
        if not self.user_access:
            QMessageBox.warning(self, "Error", "User access not initialized")
            return
            
        email = self.email_input.text()
        password = self.password_input.text()
        
        try:
            result = self.user_access.login(email, password)
            
            if result['status'] == 'success':
                QMessageBox.information(self, "Success", f"Welcome, {result['user']['name']}")
                # Log the action
                self.db_setup.log_user_action(email, "login", {"success": True})
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
                QMessageBox.warning(self, "Error", "Invalid credentials")
                # Log the failed attempt
                self.db_setup.log_user_action(email, "login", {"success": False})
        
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Login failed: {str(e)}")
            # Log the error
            self.db_setup.log_user_action(email, "login", {"success": False, "error": str(e)})

    def logout_user(self):
        email = self.email_input.text()
        self.db_setup.log_user_action(email, "logout", {})
        self.email_input.clear()
        self.password_input.clear()
        QMessageBox.information(self, "Success", "Logged out successfully")

    def generate_admin_token(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Generate Admin Recovery Token")
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        
        # Title with styling
        title = QLabel("Generate Recovery Token")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)
        
        # Admin email field with styling
        admin_email = QLineEdit()
        admin_email.setPlaceholderText("Admin Email")
        admin_email.setMinimumHeight(30)
        layout.addWidget(admin_email)
        
        # Generate button with styling
        generate_btn = QPushButton("Generate Token")
        generate_btn.setMinimumHeight(35)
        generate_btn.setStyleSheet("""
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
        layout.addWidget(generate_btn)
        
        def generate():
            try:
                result = self.user_access.generate_admin_token(admin_email.text())
                if result['status'] == 'success':
                    # Save token to file
                    file_dialog = QFileDialog()
                    file_path, _ = file_dialog.getSaveFileName(
                        dialog,
                        "Save Recovery Token",
                        "",
                        "Token Files (*.token)"
                    )
                    
                    if file_path:
                        with open(file_path, 'w') as f:
                            json.dump({
                                'email': admin_email.text(),
                                'token': result['token'],
                                'generated_at': datetime.now().isoformat()
                            }, f)
                        QMessageBox.information(
                            dialog,
                            "Success",
                            f"Recovery token saved to {file_path}"
                        )
                        dialog.close()
                else:
                    QMessageBox.warning(dialog, "Error", result.get('message', 'Failed to generate token'))
            except Exception as e:
                QMessageBox.warning(dialog, "Error", f"Failed to generate token: {str(e)}")
        
        generate_btn.clicked.connect(generate)
        
        # Set size and center
        dialog.setMinimumWidth(400)
        dialog.setMinimumHeight(200)
        
        # Center on screen
        screen = QApplication.primaryScreen().geometry()
        dialog_size = dialog.geometry()
        x = (screen.width() - dialog_size.width()) // 2
        y = (screen.height() - dialog_size.height()) // 2
        dialog.move(x, y)
        
        dialog.show()

    def use_admin_token(self):
        """Use a recovery token to reset admin password"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Use Recovery Token")
        layout = QVBoxLayout(dialog)
        
        # Token file selection
        select_file_btn = QPushButton("Select Token File")
        layout.addWidget(select_file_btn)
        
        # New password fields
        new_password = QLineEdit()
        new_password.setPlaceholderText("New Password")
        new_password.setEchoMode(QLineEdit.EchoMode.Password)
        
        confirm_password = QLineEdit()
        confirm_password.setPlaceholderText("Confirm New Password")
        confirm_password.setEchoMode(QLineEdit.EchoMode.Password)
        
        layout.addWidget(new_password)
        layout.addWidget(confirm_password)
        
        # Reset button
        reset_btn = QPushButton("Reset Password")
        layout.addWidget(reset_btn)
        reset_btn.setEnabled(False)
        
        token_data = {}
        
        def select_file():
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(
                dialog,
                "Select Recovery Token",
                "",
                "Token Files (*.token)"
            )
            
            if file_path:
                try:
                    with open(file_path, 'r') as f:
                        nonlocal token_data
                        token_data = json.load(f)
                    reset_btn.setEnabled(True)
                except Exception as e:
                    QMessageBox.warning(dialog, "Error", f"Failed to load token file: {str(e)}")
        
        def reset_password():
            if new_password.text() != confirm_password.text():
                QMessageBox.warning(dialog, "Error", "Passwords do not match!")
                return
            
            try:
                # Verify admin status - this would need to be updated for the mock
                if not self.user_access.current_user or not self.user_access.current_user.get('is_admin'):
                    QMessageBox.warning(dialog, "Error", "Admin privileges required!")
                    return
                
                # Reset password
                result = self.user_access.set_password(token_data['email'], new_password.text())
                if result['status'] == 'success':
                    QMessageBox.information(dialog, "Success", "Password reset successfully!")
                    dialog.close()
                else:
                    QMessageBox.warning(dialog, "Error", result.get('message', 'Failed to reset password'))
                    
            except Exception as e:
                QMessageBox.warning(dialog, "Error", f"Failed to reset password: {str(e)}")
        
        select_file_btn.clicked.connect(select_file)
        reset_btn.clicked.connect(reset_password)
        dialog.exec()

    def show_set_password_dialog(self, email):
        dialog = QDialog(self)
        dialog.setWindowTitle("Set Password")
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        
        # Title with styling
        title = QLabel(f"Set password for {email}")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)
        
        # Password fields with better styling
        new_password = QLineEdit()
        new_password.setPlaceholderText("New Password")
        new_password.setEchoMode(QLineEdit.EchoMode.Password)
        new_password.setMinimumHeight(30)
        
        confirm_password = QLineEdit()
        confirm_password.setPlaceholderText("Confirm Password")
        confirm_password.setEchoMode(QLineEdit.EchoMode.Password)
        confirm_password.setMinimumHeight(30)
        
        layout.addWidget(new_password)
        layout.addWidget(confirm_password)
        
        # Submit button with styling
        submit_btn = QPushButton("Set Password")
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
                
                # Log the password set
                self.db_setup.log_user_action(email, "password_set", {})
            else:
                QMessageBox.warning(dialog, "Error", result.get('message', 'Failed to set password'))
        
        submit_btn.clicked.connect(set_password)
        
        # Set size and center
        dialog.setMinimumWidth(400)
        dialog.setMinimumHeight(250)
        
        # Center on screen
        screen = QApplication.primaryScreen().geometry()
        dialog_size = dialog.geometry()
        x = (screen.width() - dialog_size.width()) // 2
        y = (screen.height() - dialog_size.height()) // 2
        dialog.move(x, y)
        
        dialog.exec()
