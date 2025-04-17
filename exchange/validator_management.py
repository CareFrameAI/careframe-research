from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QComboBox, QMessageBox, QCheckBox, 
                             QGroupBox, QTextEdit, QGridLayout, QMenu, QDialog, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QSplitter, QApplication)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QMetaObject
from PyQt6.QtGui import QFont, QIcon, QColor
import logging
import requests
from datetime import datetime
import json
import os

class ValidatorManagementSection(QWidget):
    """
    UI section for managing blockchain validators and validation permissions.
    Allows admins to assign validation rights to team members and manage validation requests.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.user_access = None  # Will be set by MainWindow
        self.blockchain_api = None  # Will be set by MainWindow
        
        # Base URL for CouchDB
        self.base_url = "http://localhost:5984"
        self.auth = ("admin", "cfpwd")
        
        # Validation settings
        self.required_validations = 1  # Default, admin can change this
        
        # UI setup
        self.setup_ui()
        
        # Refresh data on load
        self.refresh_teams()
        self.refresh_pending_validations()
        
        # Set update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.refresh_pending_validations)
        self.update_timer.start(30000)  # Refresh every 30 seconds
    
    def setup_ui(self):
        """Set up the UI components for validator management."""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Blockchain Validation Management")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(title)
        
        # Create a splitter for two main sections
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - Team Validator Management
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Team Validator Configuration
        team_config_group = QGroupBox("Validator Team Configuration")
        team_config_layout = QVBoxLayout(team_config_group)
        
        # Teams list
        teams_layout = QHBoxLayout()
        teams_layout.addWidget(QLabel("Select Team:"))
        self.team_combo = QComboBox()
        self.team_combo.currentTextChanged.connect(self.team_selected)
        teams_layout.addWidget(self.team_combo, 1)
        
        refresh_teams_btn = QPushButton("Refresh Teams")
        refresh_teams_btn.clicked.connect(self.refresh_teams)
        teams_layout.addWidget(refresh_teams_btn)
        
        team_config_layout.addLayout(teams_layout)
        
        # Team validators list
        team_config_layout.addWidget(QLabel("Team Members:"))
        self.members_table = QTableWidget()
        self.members_table.setColumnCount(4)
        self.members_table.setHorizontalHeaderLabels(["Name", "Email", "Role", "Validator"])
        self.members_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        
        team_config_layout.addWidget(self.members_table)
        
        # Actions
        actions_layout = QHBoxLayout()
        self.save_validators_btn = QPushButton("Save Validator Changes")
        self.save_validators_btn.clicked.connect(self.save_validator_changes)
        actions_layout.addWidget(self.save_validators_btn)
        
        team_config_layout.addLayout(actions_layout)
        
        # Validation requirements
        validation_req_layout = QGridLayout()
        validation_req_layout.addWidget(QLabel("Required Validations:"), 0, 0)
        self.required_validations_combo = QComboBox()
        self.required_validations_combo.addItems(["1", "2", "3", "All Team Members"])
        self.required_validations_combo.currentTextChanged.connect(self.update_validation_requirements)
        validation_req_layout.addWidget(self.required_validations_combo, 0, 1)
        
        team_config_layout.addLayout(validation_req_layout)
        
        left_layout.addWidget(team_config_group)
        
        # Right side - Pending Validations
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Pending Validations
        pending_group = QGroupBox("Pending Block Validations")
        pending_layout = QVBoxLayout(pending_group)
        
        self.pending_table = QTableWidget()
        self.pending_table.setColumnCount(5)
        self.pending_table.setHorizontalHeaderLabels(["Block Hash", "Creator", "Transactions", "Time", "Status"])
        self.pending_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.pending_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        
        pending_layout.addWidget(self.pending_table)
        
        # Actions for pending blocks
        pending_actions = QHBoxLayout()
        self.validate_selected_btn = QPushButton("Validate Selected")
        self.validate_selected_btn.clicked.connect(self.validate_selected_block)
        pending_actions.addWidget(self.validate_selected_btn)
        
        self.assign_validator_btn = QPushButton("Assign Validator")
        self.assign_validator_btn.clicked.connect(self.assign_validator_to_block)
        pending_actions.addWidget(self.assign_validator_btn)
        
        self.refresh_pending_btn = QPushButton("Refresh")
        self.refresh_pending_btn.clicked.connect(self.refresh_pending_validations)
        pending_actions.addWidget(self.refresh_pending_btn)
        
        pending_layout.addLayout(pending_actions)
        
        right_layout.addWidget(pending_group)
        
        # Add validation history
        history_group = QGroupBox("Validation History")
        history_layout = QVBoxLayout(history_group)
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(["Block Hash", "Validator", "Team", "Time", "Status"])
        self.history_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        
        history_layout.addWidget(self.history_table)
        
        refresh_history_btn = QPushButton("Refresh History")
        refresh_history_btn.clicked.connect(self.refresh_validation_history)
        history_layout.addWidget(refresh_history_btn)
        
        right_layout.addWidget(history_group)
        
        # Add widgets to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 600])  # Set initial sizes
        
        # Add splitter to main layout
        layout.addWidget(splitter)
        
        # Admin controls
        admin_group = QGroupBox("Admin Controls (Admin Only)")
        admin_layout = QVBoxLayout(admin_group)
        
        admin_actions = QHBoxLayout()
        self.force_validate_btn = QPushButton("Force Validate All Pending Blocks")
        self.force_validate_btn.clicked.connect(self.force_validate_all_blocks)
        admin_actions.addWidget(self.force_validate_btn)
        
        self.reset_validators_btn = QPushButton("Reset Validator Registry")
        self.reset_validators_btn.clicked.connect(self.reset_validator_registry)
        admin_actions.addWidget(self.reset_validators_btn)
        
        admin_layout.addLayout(admin_actions)
        
        layout.addWidget(admin_group)
        
        # Disable admin controls for non-admins
        self.update_admin_controls()
    
    def update_admin_controls(self):
        """Update visibility of admin controls based on user role."""
        is_admin = False
        if self.user_access and self.user_access.current_user:
            is_admin = self.user_access.current_user.get('is_admin', False)
        
        # Show/hide admin controls
        for btn in [self.force_validate_btn, self.reset_validators_btn]:
            btn.setEnabled(is_admin)
            btn.setVisible(is_admin)
    
    def refresh_teams(self):
        """Refresh the list of teams from CouchDB."""
        try:
            response = requests.get(
                f"{self.base_url}/teams/_all_docs?include_docs=true",
                auth=self.auth,
                timeout=1  # Use a short timeout to fail fast
            )
            
            if response.status_code == 200:
                teams = [row['doc'] for row in response.json()['rows'] if 'doc' in row]
                
                self.team_combo.clear()
                
                for team in teams:
                    if 'name' in team:
                        self.team_combo.addItem(team['name'], team['_id'])
                
                if self.team_combo.count() > 0:
                    self.team_combo.setCurrentIndex(0)
                    self.team_selected(self.team_combo.currentText())
                else:
                    # Add a placeholder if no teams found
                    self.team_combo.addItem("No teams available")
            else:
                # Handle non-successful response
                self.team_combo.clear()
                self.team_combo.addItem("Error loading teams")
        
        except Exception as e:
            # Just log the error without showing a popup
            print(f"Failed to refresh teams: {str(e)}")
            
            # Add a placeholder
            self.team_combo.clear()
            self.team_combo.addItem("Database unavailable")
    
    def team_selected(self, team_name):
        """Handle team selection from dropdown."""
        # Skip loading if team_name is a placeholder message
        if not team_name or team_name in ["No teams available", "Error loading teams", "Database unavailable"]:
            self.members_table.setRowCount(0)
            return
            
        team_id = self.team_combo.currentData()
        if not team_id:
            self.members_table.setRowCount(0)
            return
        
        try:
            response = requests.get(
                f"{self.base_url}/teams/{team_id}",
                auth=self.auth,
                timeout=1  # Use a short timeout to fail fast
            )
            
            if response.status_code == 200:
                team = response.json()
                self.current_team = team
                self.populate_members_table(team.get('members', []))
            else:
                # Just clear the members table
                self.members_table.setRowCount(0)
        
        except Exception as e:
            # Just log the error without showing a popup
            print(f"Failed to load team details: {str(e)}")
            self.members_table.setRowCount(0)
    
    def populate_members_table(self, members):
        """Populate the members table with team members."""
        self.members_table.setRowCount(0)
        
        # Get current validators if blockchain API is available
        current_validators = []
        if self.blockchain_api:
            try:
                current_validators = self.blockchain_api.get_validation_teams()
            except Exception as e:
                print(f"Error getting validators: {e}")
        
        for i, member in enumerate(members):
            self.members_table.insertRow(i)
            
            # Name
            self.members_table.setItem(i, 0, QTableWidgetItem(member.get('name', '')))
            
            # Email
            email = member.get('email', '')
            self.members_table.setItem(i, 1, QTableWidgetItem(email))
            
            # Role
            self.members_table.setItem(i, 2, QTableWidgetItem(member.get('role', '')))
            
            # Validator checkbox
            validator_checkbox = QCheckBox()
            validator_checkbox.setChecked(email in current_validators)
            
            # Create a widget to hold the checkbox
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.addWidget(validator_checkbox)
            checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            
            # Add the widget to the table
            self.members_table.setCellWidget(i, 3, checkbox_widget)
    
    def save_validator_changes(self):
        """Save the validator changes to the blockchain registry."""
        if not self.blockchain_api:
            QMessageBox.warning(self, "Error", "Blockchain API not initialized")
            return
            
        try:
            # Clear current validators
            self.blockchain_api.team_validator.validators = {}
            
            # Add selected validators
            for i in range(self.members_table.rowCount()):
                # Get email
                email = self.members_table.item(i, 1).text()
                
                # Get checkbox widget
                checkbox_widget = self.members_table.cellWidget(i, 3)
                checkbox = checkbox_widget.findChild(QCheckBox)
                
                if checkbox and checkbox.isChecked():
                    # Add to validators
                    self.blockchain_api.team_validator.add_validator(
                        email, 
                        self.current_team.get('name', ''), 
                        self.current_team.get('_id', '')
                    )
            
            # Get validation requirements from combo
            if self.blockchain_api.consensus:
                text = self.required_validations_combo.currentText()
                if text == "All Team Members":
                    count = len([cb for cb in [self.members_table.cellWidget(i, 3).findChild(QCheckBox) 
                                 for i in range(self.members_table.rowCount())] 
                                 if cb and cb.isChecked()])
                    self.blockchain_api.consensus.required_validations = max(1, count)
                else:
                    self.blockchain_api.consensus.required_validations = int(text)
            
            # Save validators to file
            self.blockchain_api.team_validator.save_validators()
            
            QMessageBox.information(self, "Success", "Validator changes saved successfully")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save validator changes: {str(e)}")
    
    def update_validation_requirements(self, text):
        """Update the validation requirements."""
        try:
            if not self.blockchain_api or not self.blockchain_api.consensus:
                return
                
            if text == "All Team Members":
                count = len([cb for cb in [self.members_table.cellWidget(i, 3).findChild(QCheckBox) 
                             for i in range(self.members_table.rowCount())] 
                             if cb and cb.isChecked()])
                self.blockchain_api.consensus.required_validations = max(1, count)
            else:
                self.blockchain_api.consensus.required_validations = int(text)
                
            print(f"Updated required validations to: {self.blockchain_api.consensus.required_validations}")
            
        except Exception as e:
            print(f"Error updating validation requirements: {e}")
    
    def refresh_pending_validations(self):
        """Refresh the list of pending blocks for validation."""
        if not self.blockchain_api:
            return
            
        try:
            self.pending_table.setRowCount(0)
            
            # Get pending blocks
            pending_blocks = self.blockchain_api.get_pending_blocks()
            
            # Get current user ID for validation check
            current_user_id = ""
            if self.user_access and self.user_access.current_user:
                current_user_id = self.user_access.current_user.get('email', '')
            
            i = 0
            for block_hash, block in pending_blocks.items():
                self.pending_table.insertRow(i)
                
                # Block hash
                hash_item = QTableWidgetItem(block_hash[:15] + "...")
                hash_item.setData(Qt.ItemDataRole.UserRole, block_hash)  # Store full hash
                self.pending_table.setItem(i, 0, hash_item)
                
                # Creator - get from the first transaction
                creator = "Unknown"
                if hasattr(block, 'transactions') and block.transactions:
                    tx = block.transactions[0]
                    if hasattr(tx, 'user_id'):
                        creator = tx.user_id
                    elif isinstance(tx, dict):
                        creator = tx.get('submitter', tx.get('creator', 'Unknown'))
                self.pending_table.setItem(i, 1, QTableWidgetItem(str(creator)))
                
                # Transaction count
                tx_count = len(block.transactions) if hasattr(block, 'transactions') else 0
                self.pending_table.setItem(i, 2, QTableWidgetItem(str(tx_count)))
                
                # Time - Convert datetime object to string if needed
                time_str = block.timestamp if hasattr(block, 'timestamp') else "Unknown"
                if isinstance(time_str, datetime):
                    time_str = time_str.isoformat()
                self.pending_table.setItem(i, 3, QTableWidgetItem(str(time_str)))
                
                # Status with validation count and user validation status
                validation_count = len(block.validations) if hasattr(block, 'validations') else 0
                required = self.blockchain_api.consensus.required_validations
                
                # Check if current user has validated this block
                user_validated = False
                validators = []
                if hasattr(block, 'validations'):
                    for validation in block.validations:
                        validator = validation.get('validator_id', '')
                        validators.append(validator)
                        if validator == current_user_id:
                            user_validated = True
                
                # Create status text
                if user_validated:
                    status = f"✓ {validation_count}/{required} validations (You validated)"
                else:
                    status = f"{validation_count}/{required} validations"
                    
                # Add validator list as tooltip
                tooltip = "Validators:\n" + "\n".join(validators) if validators else "No validators yet"
                
                status_item = QTableWidgetItem(status)
                status_item.setToolTip(tooltip)
                
                if user_validated:
                    status_item.setForeground(QColor("blue"))
                elif validation_count >= required:
                    status_item.setForeground(QColor("green"))
                else:
                    status_item.setForeground(QColor("red"))
                
                # Store validation status as user data for sorting
                status_item.setData(Qt.ItemDataRole.UserRole, validation_count)
                    
                self.pending_table.setItem(i, 4, status_item)
                
                i += 1
            
            # Update message if no pending blocks
            if i == 0:
                self.pending_table.insertRow(0)
                no_blocks_item = QTableWidgetItem("No pending blocks")
                no_blocks_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.pending_table.setSpan(0, 0, 1, 5)
                self.pending_table.setItem(0, 0, no_blocks_item)
                
            # Resize columns
            self.pending_table.resizeColumnsToContents()
            
        except Exception as e:
            print(f"Error refreshing pending validations: {e}")
    
    def refresh_validation_history(self):
        """Refresh the validation history table."""
        if not self.blockchain_api:
            return
            
        try:
            self.history_table.setRowCount(0)
            
            # Get validated blocks from the chain
            chain = self.blockchain_api.get_chain()
            
            i = 0
            for block in chain:
                if hasattr(block, 'validations') and block.validations:
                    for validation in block.validations:
                        self.history_table.insertRow(i)
                        
                        # Block hash
                        hash_item = QTableWidgetItem(block.hash[:15] + "...")
                        self.history_table.setItem(i, 0, hash_item)
                        
                        # Validator
                        validator = validation.get('validator_id', 'Unknown')
                        self.history_table.setItem(i, 1, QTableWidgetItem(str(validator)))
                        
                        # Team
                        team = validation.get('team_name', 'Unknown')
                        self.history_table.setItem(i, 2, QTableWidgetItem(str(team)))
                        
                        # Time
                        time_str = validation.get('timestamp', 'Unknown')
                        # Convert datetime object to string if needed
                        if isinstance(time_str, datetime):
                            time_str = time_str.isoformat()
                        self.history_table.setItem(i, 3, QTableWidgetItem(str(time_str)))
                        
                        # Status
                        status_item = QTableWidgetItem("Validated")
                        status_item.setForeground(QColor("green"))
                        self.history_table.setItem(i, 4, status_item)
                        
                        i += 1
            
            # Update message if no history
            if i == 0:
                self.history_table.insertRow(0)
                no_history_item = QTableWidgetItem("No validation history")
                no_history_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.history_table.setSpan(0, 0, 1, 5)
                self.history_table.setItem(0, 0, no_history_item)
                
            # Resize columns
            self.history_table.resizeColumnsToContents()
            
        except Exception as e:
            print(f"Error refreshing validation history: {e}")
    
    def validate_selected_block(self):
        """Validate the selected block using the current user's credentials."""
        if not self.blockchain_api:
            QMessageBox.warning(self, "Error", "Blockchain API not initialized")
            return
            
        selected_rows = self.pending_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select a block to validate")
            return
            
        row = selected_rows[0].row()
        block_hash_item = self.pending_table.item(row, 0)
        if not block_hash_item:
            return
            
        # Get the full block hash
        block_hash = block_hash_item.data(Qt.ItemDataRole.UserRole)
        
        # Get current user for validation
        if not self.user_access or not self.user_access.current_user:
            QMessageBox.warning(self, "Not Logged In", "You must be logged in to validate blocks")
            return
            
        validator_id = self.user_access.current_user.get('email', '')
        
        # Check if this block is in pending blocks
        if not hasattr(self.blockchain_api.blockchain, 'pending_blocks') or \
           block_hash not in self.blockchain_api.blockchain.pending_blocks:
            QMessageBox.information(self, "Block Not Pending", 
                                  "This block is no longer pending validation.")
            self.refresh_pending_validations()
            return
            
        # Check if user already validated this block
        block = self.blockchain_api.blockchain.pending_blocks[block_hash]
        if hasattr(block, 'validations'):
            for validation in block.validations:
                if validation.get('validator_id') == validator_id:
                    QMessageBox.information(self, "Already Validated", 
                                          f"You have already validated this block.")
                    return
        
        # Show validation dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Validate Block")
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel(f"You are about to validate block:\n{block_hash}"))
        
        # Show block information
        block_info = QTextEdit()
        block_info.setReadOnly(True)
        tx_count = len(block.transactions) if hasattr(block, 'transactions') else 0
        
        # Format transactions preview
        tx_preview = ""
        if hasattr(block, 'transactions'):
            for i, tx in enumerate(block.transactions[:5]):  # Show first 5 transactions
                if hasattr(tx, 'tx_type'):
                    tx_preview += f"- {tx.tx_type}"
                    if hasattr(tx, 'user_id'):
                        tx_preview += f" by {tx.user_id}"
                    tx_preview += "\n"
            
            if tx_count > 5:
                tx_preview += f"...and {tx_count - 5} more transactions"
        
        # Block info text
        info_text = f"""
        Block Hash: {block_hash}
        Index: {block.index if hasattr(block, 'index') else 'Unknown'}
        Timestamp: {block.timestamp if hasattr(block, 'timestamp') else 'Unknown'}
        Transaction Count: {tx_count}
        
        Transactions:
        {tx_preview}
        
        Current Validations: {len(block.validations) if hasattr(block, 'validations') else 0}
        Required Validations: {self.blockchain_api.consensus.required_validations}
        """
        
        block_info.setText(info_text)
        layout.addWidget(block_info)
        
        key_layout = QHBoxLayout()
        key_layout.addWidget(QLabel("Private Key:"))
        
        key_input = QLineEdit()
        key_input.setEchoMode(QLineEdit.EchoMode.Password)
        
        # If we have admin keys and the user is admin, pre-fill
        if hasattr(self.blockchain_api, 'admin_keys') and self.user_access.current_user.get('is_admin', False):
            key_input.setText(self.blockchain_api.admin_keys.get('private', ''))
        
        key_layout.addWidget(key_input)
        layout.addLayout(key_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        validate_btn = QPushButton("Validate")
        cancel_btn = QPushButton("Cancel")
        button_layout.addWidget(validate_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        # Connect buttons
        validate_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        
        # Show dialog
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
            
        private_key = key_input.text()
        if not private_key:
            QMessageBox.warning(self, "Missing Key", "Private key is required for validation")
            return
            
        # Update admin_keys user_id if user is admin
        if self.user_access.current_user.get('is_admin', False) and hasattr(self.blockchain_api, 'admin_keys'):
            self.blockchain_api.admin_keys['user_id'] = validator_id
        
        # Perform validation
        try:
            # Execute the validation directly without showing a progress dialog
            result = self.blockchain_api.validate_block(block_hash, validator_id, private_key)
            
            if result:
                # Update UI immediately
                self.refresh_pending_validations()
                self.refresh_validation_history()
                
                # Show a status message in the UI if needed
                status_bar = self.parent().statusBar() if hasattr(self.parent(), 'statusBar') else None
                if status_bar:
                    status_bar.showMessage("Block validation successful!", 3000)  # Show for 3 seconds
            else:
                QMessageBox.warning(self, "Validation Failed", 
                                 "Failed to validate block. Check console for details.")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during validation: {str(e)}")
            
        # Final refresh to ensure UI is in sync with actual state
        self.refresh_pending_validations()
    
    def assign_validator_to_block(self):
        """Assign a validator to the selected block."""
        if not self.blockchain_api:
            QMessageBox.warning(self, "Error", "Blockchain API not initialized")
            return
        
        # Check if user is admin
        if not self.user_access or not self.user_access.current_user or not self.user_access.current_user.get('is_admin', False):
            QMessageBox.warning(self, "Permission Denied", "Only administrators can assign validators")
            return
            
        selected_rows = self.pending_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select a block to assign validators")
            return
            
        row = selected_rows[0].row()
        block_hash_item = self.pending_table.item(row, 0)
        if not block_hash_item:
            return
            
        # Get the full block hash
        block_hash = block_hash_item.data(Qt.ItemDataRole.UserRole)
        
        # Get validators
        validators = list(self.blockchain_api.team_validator.validators.keys())
        if not validators:
            QMessageBox.warning(self, "No Validators", "No validators are configured")
            return
            
        # Show validator selection dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Assign Validator")
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel(f"Assign validator for block:\n{block_hash}"))
        
        # Validator selection
        validator_layout = QHBoxLayout()
        validator_layout.addWidget(QLabel("Validator:"))
        
        validator_combo = QComboBox()
        for validator in validators:
            team_name = self.blockchain_api.team_validator.validators[validator].get('team_name', '')
            validator_combo.addItem(f"{validator} ({team_name})", validator)
            
        validator_layout.addWidget(validator_combo)
        layout.addLayout(validator_layout)
        
        # Notification message
        layout.addWidget(QLabel("The selected validator will receive a notification to validate this block."))
        
        # Buttons
        button_layout = QHBoxLayout()
        assign_btn = QPushButton("Assign")
        cancel_btn = QPushButton("Cancel")
        button_layout.addWidget(assign_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        # Connect buttons
        assign_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        
        # Show dialog
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
            
        validator_id = validator_combo.currentData()
        
        # Create validation request
        try:
            # In a real implementation, this would send a notification
            # For now, we'll just create a record in the requests database
            
            request = {
                "type": "validation_request",
                "block_hash": block_hash,
                "validator_id": validator_id,
                "requested_by": self.user_access.current_user.get('email', ''),
                "status": "pending",
                "created_at": datetime.now().isoformat()
            }
            
            # Store in validation_requests directory
            validation_dir = os.path.join(os.path.expanduser("~"), ".blockchain_data", "validation_requests")
            os.makedirs(validation_dir, exist_ok=True)
            
            request_file = os.path.join(validation_dir, f"{block_hash[:10]}_{validator_id}.json")
            with open(request_file, 'w') as f:
                json.dump(request, f)
            
            QMessageBox.information(self, "Success", 
                                   f"Validation request assigned to {validator_id}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error assigning validator: {str(e)}")
    
    def force_validate_all_blocks(self):
        """Force validate all pending blocks as admin."""
        if not self.blockchain_api:
            QMessageBox.warning(self, "Error", "Blockchain API not initialized")
            return
            
        # Check if user is admin
        if not self.user_access or not self.user_access.current_user or not self.user_access.current_user.get('is_admin', False):
            QMessageBox.warning(self, "Permission Denied", "Only administrators can force validate blocks")
            return
            
        # Get pending blocks
        pending_blocks = self.blockchain_api.get_pending_blocks()
        if not pending_blocks:
            QMessageBox.information(self, "No Blocks", "No pending blocks to validate")
            return
            
        # Confirm with the user
        reply = QMessageBox.question(
            self,
            "Confirm Force Validation",
            f"You are about to force validate {len(pending_blocks)} pending blocks as admin. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.No:
            return
            
        # Get admin ID and key
        admin_id = self.user_access.current_user.get('email', 'admin')
        
        # Update admin_keys
        if hasattr(self.blockchain_api, 'admin_keys'):
            self.blockchain_api.admin_keys['user_id'] = admin_id
            admin_key = self.blockchain_api.admin_keys.get('private', '')
        else:
            # If no admin keys, ask for private key
            key_dialog = QDialog(self)
            key_dialog.setWindowTitle("Admin Private Key")
            key_layout = QVBoxLayout(key_dialog)
            
            key_layout.addWidget(QLabel("Enter your admin private key:"))
            
            key_input = QLineEdit()
            key_input.setEchoMode(QLineEdit.EchoMode.Password)
            key_layout.addWidget(key_input)
            
            # Buttons
            button_layout = QHBoxLayout()
            ok_btn = QPushButton("OK")
            cancel_btn = QPushButton("Cancel")
            button_layout.addWidget(ok_btn)
            button_layout.addWidget(cancel_btn)
            key_layout.addLayout(button_layout)
            
            # Connect buttons
            ok_btn.clicked.connect(key_dialog.accept)
            cancel_btn.clicked.connect(key_dialog.reject)
            
            # Show dialog
            if key_dialog.exec() != QDialog.DialogCode.Accepted:
                return
                
            admin_key = key_input.text()
            
        if not admin_key:
            QMessageBox.warning(self, "Missing Key", "Admin private key is required")
            return
        
        # Show status in status bar if available
        status_bar = self.parent().statusBar() if hasattr(self.parent(), 'statusBar') else None
        if status_bar:
            status_bar.showMessage("Validating pending blocks...", 1000)
            
        # Validate all blocks
        success_count = 0
        fail_count = 0
        
        # Make a copy of the keys since we'll be modifying the dict during iteration
        block_hashes = list(pending_blocks.keys())
        
        for i, block_hash in enumerate(block_hashes):
            # Update progress in console
            print(f"Validating block {i+1}/{len(block_hashes)}...")
            
            # Update status bar
            if status_bar:
                status_bar.showMessage(f"Validating block {i+1}/{len(block_hashes)}...", 500)
                
            # Check if block is still pending (might have been validated by another process)
            if block_hash not in self.blockchain_api.blockchain.pending_blocks:
                print(f"Block {block_hash[:10]}... is no longer pending, skipping")
                continue
                
            try:
                result = self.blockchain_api.validate_block(block_hash, admin_id, admin_key)
                
                if result:
                    success_count += 1
                else:
                    fail_count += 1
                    
            except Exception as e:
                print(f"Error validating block {block_hash}: {e}")
                fail_count += 1
            
            # Periodic refresh to update UI during lengthy operations
            if (i + 1) % 5 == 0 or i == len(block_hashes) - 1:
                # Use invoke method for thread safety
                QMetaObject.invokeMethod(self, "refresh_pending_validations", 
                                       Qt.ConnectionType.QueuedConnection)
        
        # Show a status message in the UI
        if status_bar:
            status_bar.showMessage(f"Force validation complete. Successful: {success_count}, Failed: {fail_count}", 5000)
        
        # Log to console
        print(f"Force validation complete. Successful: {success_count}, Failed: {fail_count}")
        
        # Refresh views
        self.refresh_pending_validations()
        self.refresh_validation_history()
    
    def reset_validator_registry(self):
        """Reset the validator registry."""
        if not self.blockchain_api:
            QMessageBox.warning(self, "Error", "Blockchain API not initialized")
            return
            
        # Check if user is admin
        if not self.user_access or not self.user_access.current_user or not self.user_access.current_user.get('is_admin', False):
            QMessageBox.warning(self, "Permission Denied", "Only administrators can reset the validator registry")
            return
            
        # Confirm with the user
        reply = QMessageBox.question(
            self,
            "Confirm Reset",
            "You are about to reset the validator registry. All validator assignments will be cleared. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.No:
            return
            
        # Reset validators
        try:
            self.blockchain_api.team_validator.validators = {}
            self.blockchain_api.team_validator.save_validators()
            
            # Reset required validations to 1
            if self.blockchain_api.consensus:
                self.blockchain_api.consensus.required_validations = 1
                
            # Update UI
            self.required_validations_combo.setCurrentText("1")
            self.populate_members_table(self.current_team.get('members', []))
            
            # Show a status message in the UI if needed
            status_bar = self.parent().statusBar() if hasattr(self.parent(), 'statusBar') else None
            if status_bar:
                status_bar.showMessage("Validator registry reset successfully", 3000)
            
            # Log to console
            print("Validator registry reset successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error resetting validator registry: {str(e)}") 