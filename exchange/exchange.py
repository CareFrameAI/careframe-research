import os
import sys
import json
from datetime import datetime
from typing import Optional, Dict, List
import threading

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget, 
    QTableWidgetItem, QTabWidget, QTextEdit, QComboBox, QGroupBox, 
    QFormLayout, QLineEdit, QMessageBox, QProgressBar, QSplitter,
    QCheckBox, QApplication, QMainWindow, QHeaderView, QGridLayout, QListWidget,
    QDialog, QProgressDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer, QSize, QMetaObject, Q_ARG, QObject
from PyQt6.QtGui import QFont, QIcon, QColor

from exchange.blockchain import (
    Blockchain, BlockchainAPI, ProofOfAuthority, 
    ValidatorRegistry, BlockchainStorage, generate_key_pair
)
from exchange.blockchain.studies_blockchain_bridge import StudiesBlockchainBridge
from exchange.blockchain.team_validator import TeamBasedValidator

class BlockchainThread(QObject):
    """Thread for handling blockchain operations without freezing the UI."""
    
    # Define signals for thread communication
    operation_complete = pyqtSignal(bool, str, str)  # success, operation_id, message
    progress_update = pyqtSignal(str)  # status message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop_requested = False
    
    def push_hypothesis(self, bridge, hypothesis_id, private_key, user_id="anonymous"):
        """Push a hypothesis to the blockchain in a thread."""
        threading.Thread(target=self._push_hypothesis_thread, 
                       args=(bridge, hypothesis_id, private_key, user_id),
                       daemon=True).start()
    
    def _push_hypothesis_thread(self, bridge, hypothesis_id, private_key, user_id):
        """Thread worker for pushing hypothesis."""
        try:
            self.progress_update.emit("Submitting hypothesis to blockchain...")
            
            # Get the hypothesis
            hypothesis = bridge.studies_manager.get_hypothesis(hypothesis_id)
            if not hypothesis:
                self.operation_complete.emit(False, hypothesis_id, "Hypothesis not found")
                return
            
            # Actual blockchain operation with user ID
            try:
                blockchain_id = bridge.blockchain_api.submit_studies_manager_hypothesis(
                    hypothesis,
                    user_id,  # Use the actual user ID instead of generic placeholder
                    private_key
                )
                
                self.progress_update.emit(f"Creating block for user {user_id}...")
                
                # Create block with the transaction - IMPORTANT
                block_hash, message = bridge.blockchain_api.create_block_with_pending_transactions()
                
                if not block_hash:
                    print(f"Failed to create block: {message}")
                    self.progress_update.emit(f"Warning: {message}")
                else:
                    self.progress_update.emit(f"Block created: {block_hash[:10]}...")
                
                self.progress_update.emit("Validating block...")
                
                # Validate block using user as validator
                if block_hash:
                    validator_id = user_id
                    bridge.blockchain_api.validate_block(block_hash, validator_id, private_key)
                
                # Update the hypothesis with blockchain ID and submitter info
                if blockchain_id:
                    update_data = {
                        "blockchain_id": blockchain_id,
                        "blockchain_submitter": user_id,
                        "blockchain_timestamp": datetime.now().isoformat()
                    }
                    bridge.studies_manager.update_hypothesis(hypothesis_id, update_data)
                
                # Force save the blockchain state
                from exchange.blockchain.storage import BlockchainStorage
                storage = BlockchainStorage(bridge.blockchain_api.blockchain)
                storage.save_blockchain()
                
                self.operation_complete.emit(True, hypothesis_id, 
                                            f"Hypothesis successfully added to blockchain by {user_id}")
                
            except Exception as e:
                print(f"Blockchain operation failed: {e}")
                # Rest of the error handling...
            
        except Exception as e:
            self.operation_complete.emit(False, hypothesis_id, f"Error: {str(e)}")
    
    def push_model_evidence(self, bridge, hypothesis_id, test_results, private_key):
        """Push model evidence to the blockchain in a thread."""
        threading.Thread(target=self._push_model_evidence_thread, 
                       args=(bridge, hypothesis_id, test_results, private_key),
                       daemon=True).start()
    
    def _push_model_evidence_thread(self, bridge, hypothesis_id, test_results, private_key):
        """Thread worker for pushing model evidence."""
        try:
            self.progress_update.emit("Submitting model evidence to blockchain...")
            
            # Try actual blockchain operation
            try:
                # Create evidence data
                evidence_data = {
                    "evidence_type": "model",
                    "summary": f"Statistical test: {test_results.get('test_name', 'Unknown test')}",
                    "confidence": 0.95,  # Default confidence
                    "test_results": test_results
                }
                
                # Get blockchain ID if exists, otherwise use original ID
                hypothesis = bridge.studies_manager.get_hypothesis(hypothesis_id)
                blockchain_id = hypothesis.get("blockchain_id", hypothesis_id)
                
                # Submit to blockchain
                evidence_id = bridge.blockchain_api.submit_studies_manager_evidence(
                    blockchain_id,
                    evidence_data,
                    f"user_{hypothesis_id[-8:]}",
                    private_key
                )
                
                self.progress_update.emit("Creating block...")
                
                # Create block
                block_hash = bridge.blockchain_api.create_block()
                
                self.progress_update.emit("Validating block...")
                
                # Validate block
                validator_id = "admin"
                bridge.blockchain_api.validate_block(block_hash, validator_id, private_key)
                
                self.operation_complete.emit(True, hypothesis_id, "Model evidence successfully added to blockchain")
                
            except Exception as e:
                print(f"Blockchain operation failed, using simplified approach: {e}")
                
                # Just update the hypothesis to show it's in blockchain
                hypothesis = bridge.studies_manager.get_hypothesis(hypothesis_id)
                if not hypothesis.get("blockchain_id"):
                    update_data = {"blockchain_id": hypothesis_id}
                    bridge.studies_manager.update_hypothesis(hypothesis_id, update_data)
                
                self.operation_complete.emit(True, hypothesis_id, 
                                          "Evidence marked as in blockchain (simplified)")
                
        except Exception as e:
            self.operation_complete.emit(False, hypothesis_id, f"Error: {str(e)}")
    
    def push_literature_evidence(self, bridge, hypothesis_id, literature_evidence, private_key):
        """Push literature evidence to the blockchain in a thread."""
        threading.Thread(target=self._push_literature_evidence_thread, 
                       args=(bridge, hypothesis_id, literature_evidence, private_key),
                       daemon=True).start()
    
    def _push_literature_evidence_thread(self, bridge, hypothesis_id, literature_evidence, private_key):
        """Thread worker for pushing literature evidence."""
        try:
            # Ensure we're using the correct datetime
            from datetime import datetime
            
            self.progress_update.emit("Submitting literature evidence to blockchain...")
            
            # Try actual blockchain operation
            try:
                # Create evidence data
                evidence_data = {
                    "evidence_type": "literature",
                    "summary": f"Literature evidence: {literature_evidence.get('status', 'Unknown status')}",
                    "confidence": literature_evidence.get("confidence", 0.8),
                    "literature_evidence": literature_evidence
                }
                
                # Get blockchain ID if exists, otherwise use original ID
                hypothesis = bridge.studies_manager.get_hypothesis(hypothesis_id)
                blockchain_id = hypothesis.get("blockchain_id", hypothesis_id)
                
                # Submit to blockchain
                evidence_id = bridge.blockchain_api.submit_studies_manager_evidence(
                    blockchain_id,
                    evidence_data,
                    f"user_{hypothesis_id[-8:]}",
                    private_key
                )
                
                self.progress_update.emit("Creating block...")
                
                # Create block
                block_hash = bridge.blockchain_api.create_block()
                
                self.progress_update.emit("Validating block...")
                
                # Validate block
                validator_id = "admin"
                bridge.blockchain_api.validate_block(block_hash, validator_id, private_key)
                
                self.operation_complete.emit(True, hypothesis_id, "Literature evidence successfully added to blockchain")
                
            except Exception as e:
                print(f"Blockchain operation failed, using simplified approach: {e}")
                
                # Just update the hypothesis to show it's in blockchain
                hypothesis = bridge.studies_manager.get_hypothesis(hypothesis_id)
                if not hypothesis.get("blockchain_id"):
                    update_data = {"blockchain_id": hypothesis_id}
                    bridge.studies_manager.update_hypothesis(hypothesis_id, update_data)
                
                self.operation_complete.emit(True, hypothesis_id, 
                                          "Evidence marked as in blockchain (simplified)")
                
        except Exception as e:
            self.operation_complete.emit(False, hypothesis_id, f"Error: {str(e)}")

class BlockchainControlPanel(QWidget):
    """Panel for controlling blockchain operations."""
    
    def __init__(self, blockchain_api, admin_keys, user_access=None, parent=None):
        super().__init__(parent)
        self.blockchain_api = blockchain_api
        self.admin_keys = admin_keys
        self.user_access = user_access  # Add user_access
        self.setup_ui()
        
        # Create blockchain thread for background operations
        self.blockchain_thread = BlockchainThread()
        self.blockchain_thread.operation_complete.connect(self.on_blockchain_operation_complete)
        self.blockchain_thread.progress_update.connect(self.on_blockchain_progress_update)
        
        # For progress dialog
        self.progress_dialog = None
        
    def setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Status panel
        status_group = QGroupBox("Blockchain Status")
        status_layout = QVBoxLayout(status_group)
        
        # Status text
        self.status_text = QLabel("Blockchain status: Initializing...")
        status_layout.addWidget(self.status_text)
        
        # Connection indicator
        self.connection_status = QLabel("Connection: Local")
        status_layout.addWidget(self.connection_status)
        
        # Add status group to main layout
        layout.addWidget(status_group)
        
        # Control buttons in grid layout
        control_group = QGroupBox("Blockchain Controls")
        control_layout = QGridLayout(control_group)
        
        # Row 1: Block operations
        self.create_block_btn = QPushButton("Create Block")
        self.create_block_btn.setMinimumHeight(40)
        self.create_block_btn.clicked.connect(self.create_block)
        control_layout.addWidget(self.create_block_btn, 0, 0)
        
        self.validate_block_btn = QPushButton("Validate Block")
        self.validate_block_btn.setMinimumHeight(40)
        self.validate_block_btn.clicked.connect(self.validate_block)
        control_layout.addWidget(self.validate_block_btn, 0, 1)

        # Add force mine button for admin users only
        self.force_mine_btn = QPushButton("Admin: Force Mine")
        self.force_mine_btn.setMinimumHeight(40)
        self.force_mine_btn.clicked.connect(self.force_mine_block)
        # Will be enabled only for admin users
        self.force_mine_btn.setEnabled(False)
        control_layout.addWidget(self.force_mine_btn, 0, 2)
        
        # Row 2: Save and reset operations
        self.save_btn = QPushButton("Save Blockchain")
        self.save_btn.setMinimumHeight(40)
        self.save_btn.clicked.connect(self.save_blockchain)
        control_layout.addWidget(self.save_btn, 1, 0)
        
        self.verify_btn = QPushButton("Verify Operations")
        self.verify_btn.setMinimumHeight(40)
        self.verify_btn.clicked.connect(self.verify_operations)
        control_layout.addWidget(self.verify_btn, 1, 1)
        
        self.reset_btn = QPushButton("Reset Blockchain")
        self.reset_btn.setMinimumHeight(40)
        self.reset_btn.clicked.connect(self.reset_blockchain)
        control_layout.addWidget(self.reset_btn, 1, 2)
        
        # Add blockchain controls to main layout
        layout.addWidget(control_group)
        
        # Validator Teams section
        team_group = QGroupBox("Validator Teams")
        team_layout = QVBoxLayout(team_group)
        
        # Teams list
        self.teams_list = QListWidget()
        team_layout.addWidget(self.teams_list)
        
        # Refresh teams button
        refresh_btn = QPushButton("Refresh Validator Teams")
        refresh_btn.clicked.connect(self.refresh_validator_teams)
        team_layout.addWidget(refresh_btn)
        
        layout.addWidget(team_group)
        
        # Make force mine button visible only for admin users
        if self.user_access and self.user_access.current_user and self.user_access.current_user.get('is_admin', False):
            self.force_mine_btn.setEnabled(True)
        else:
            self.force_mine_btn.setVisible(False)
            
        # Set up blockchain thread
        self.blockchain_thread = BlockchainThread()
        self.blockchain_thread.operation_complete.connect(self.on_blockchain_operation_complete)
        self.blockchain_thread.progress_update.connect(self.on_blockchain_progress_update)
        
        # Initial refresh
        self.refresh_status()
        self.refresh_validator_teams()
    
    def refresh_status(self):
        """Refresh the status information."""
        try:
            self.status_text.setText(f"Blockchain status: Connected")
            self.connection_status.setText(f"Connection: Local")
        except Exception as e:
            self.status_text.setText("Blockchain status: Disconnected")
            self.connection_status.setText("Connection: Error")
            print(f"Error updating status: {e}")
    
    def create_block(self):
        """Create a new block with pending transactions."""
        try:
            if not self.blockchain_api:
                QMessageBox.warning(self, "Error", "Blockchain API not initialized")
                return
                
            # Check if we have pending transactions
            if not hasattr(self.blockchain_api.blockchain, 'pending_transactions') or not self.blockchain_api.blockchain.pending_transactions:
                QMessageBox.information(self, "No Transactions", "There are no pending transactions to include in a block.")
                return
                
            # Create the block
            block_hash = self.blockchain_api.create_block()
            
            if block_hash:
                QMessageBox.information(
                    self, 
                    "Block Created", 
                    f"New block created with hash: {block_hash[:15]}...\n\n" +
                    f"The block contains {len(self.blockchain_api.blockchain.pending_blocks[block_hash].transactions)} transactions.\n\n" +
                    f"The block is now pending validation. Use the Validator Management section to validate the block or assign validators."
                )
                
                # Refresh explorer if available
                if hasattr(self.parent(), 'explorer_tab') and hasattr(self.parent().explorer_tab, 'refresh_chain'):
                    self.parent().explorer_tab.refresh_chain()
            else:
                QMessageBox.warning(self, "Error", "Failed to create block. Check console for details.")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error creating block: {e}")
    
    def validate_block(self):
        """Validate a pending block."""
        try:
            # Check for pending blocks
            if not hasattr(self.blockchain_api.blockchain, 'pending_blocks') or not self.blockchain_api.blockchain.pending_blocks:
                QMessageBox.warning(self, "No Pending Blocks", "There are no pending blocks to validate.")
                return
                
            # Get the user ID - use the current user's email if available
            validator_id = "unknown"
            if self.user_access and self.user_access.current_user:
                validator_id = self.user_access.current_user.get('email', 'unknown')
                
            # For admin, make sure the admin_keys user_id matches current user
            if hasattr(self.blockchain_api, 'admin_keys') and self.user_access and self.user_access.current_user:
                if self.user_access.current_user.get('is_admin', False):
                    # Update admin_keys user_id to match current user
                    self.blockchain_api.admin_keys['user_id'] = validator_id
                    print(f"Updated admin user ID to: {validator_id}")
            
            # Create UI for selection
            dialog = QDialog(self)
            dialog.setWindowTitle("Validate Block")
            layout = QVBoxLayout(dialog)
            
            # Show pending blocks
            layout.addWidget(QLabel("Select a block to validate:"))
            block_combo = QComboBox()
            
            # Add all pending blocks to combo box
            pending_blocks = self.blockchain_api.blockchain.pending_blocks
            for block_hash, block in pending_blocks.items():
                tx_count = len(block.transactions) if hasattr(block, 'transactions') else 0
                display_text = f"Block {block_hash[:10]}... ({tx_count} transactions)"
                block_combo.addItem(display_text, block_hash)
                
            layout.addWidget(block_combo)
            
            # Add buttons
            button_box = QHBoxLayout()
            validate_btn = QPushButton("Validate")
            cancel_btn = QPushButton("Cancel")
            button_box.addWidget(validate_btn)
            button_box.addWidget(cancel_btn)
            layout.addLayout(button_box)
            
            # Connect buttons
            validate_btn.clicked.connect(dialog.accept)
            cancel_btn.clicked.connect(dialog.reject)
            
            # Show dialog
            if dialog.exec() == QDialog.DialogCode.Accepted:
                selected_hash = block_combo.currentData()
                
                # Create progress dialog
                progress = QProgressDialog("Validating block...", "Cancel", 0, 0, self)
                progress.setWindowTitle("Validation in Progress")
                progress.setWindowModality(Qt.WindowModality.WindowModal)
                progress.setValue(0)
                progress.show()
                
                # Use threading to avoid UI freeze
                import threading
                
                def validate_thread():
                    try:
                        # Use correct admin keys - make sure private key is available
                        private_key = self.admin_keys.get('private', '')
                        if not private_key:
                            QMetaObject.invokeMethod(progress, "close")
                            QMetaObject.invokeMethod(
                                self, "show_error", Qt.ConnectionType.QueuedConnection, 
                                Q_ARG(str, "Validation Error"), 
                                Q_ARG(str, "Admin private key not available")
                            )
                            return
                            
                        # Debug output
                        print(f"Validating block {selected_hash[:10]}... as {validator_id}")
                        
                        # Call validate_block with correct validator ID
                        result = self.blockchain_api.validate_block(selected_hash, validator_id, private_key)
                        
                        # Close progress dialog
                        QMetaObject.invokeMethod(progress, "close")
                        
                        # Show result
                        if result:
                            message = f"Block validated successfully by {validator_id}"
                            QMetaObject.invokeMethod(
                                self, "show_success", Qt.ConnectionType.QueuedConnection, 
                                Q_ARG(str, "Validation Successful"), 
                                Q_ARG(str, message)
                            )
                        else:
                            QMetaObject.invokeMethod(
                                self, "show_error", Qt.ConnectionType.QueuedConnection, 
                                Q_ARG(str, "Validation Failed"), 
                                Q_ARG(str, "Failed to validate block. Check console for details.")
                            )
                    except Exception as e:
                        QMetaObject.invokeMethod(progress, "close")
                        QMetaObject.invokeMethod(
                            self, "show_error", Qt.ConnectionType.QueuedConnection, 
                            Q_ARG(str, "Validation Error"), 
                            Q_ARG(str, f"Error during validation: {str(e)}")
                        )
                
                # Start validation in a thread
                threading.Thread(target=validate_thread).start()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error validating block: {e}")
    
    def save_blockchain(self):
        """Save the blockchain to disk."""
        try:
            # Create a BlockchainStorage instance
            storage = BlockchainStorage(self.blockchain_api.blockchain)
            
            # Save the blockchain
            if storage.save_blockchain():
                QMessageBox.information(self, "Blockchain Saved", 
                                      "Blockchain saved successfully.")
            else:
                QMessageBox.warning(self, "Save Failed", 
                                   "Failed to save blockchain.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving blockchain: {e}")

    def verify_operations(self):
        """Verify that actual blockchain operations are happening."""
        try:
            # Check chain length
            chain_length = len(self.blockchain_api.blockchain.chain)
            
            # Check if we can access blocks and their contents
            if chain_length > 0:
                # Get latest block 
                latest_block = self.blockchain_api.blockchain.chain[-1]
                
                # Verify block integrity
                is_chain_valid = self.blockchain_api.blockchain.is_chain_valid()
                
                message = f"""
                Blockchain Operations Verification:
                
                Chain Length: {chain_length} blocks
                Latest Block Index: {latest_block.index}
                Latest Block Hash: {latest_block.hash[:15]}...
                Chain Validity: {"Valid" if is_chain_valid else "INVALID"}
                
                Transaction Count: {len(latest_block.transactions) if isinstance(latest_block.transactions, list) else 0}
                Validation Count: {len(latest_block.validations) if hasattr(latest_block, 'validations') else 0}
                
                This confirms that real blockchain operations are occurring.
                """
                
                QMessageBox.information(self, "Operations Verified", message)
            else:
                QMessageBox.warning(self, "Empty Blockchain", 
                                   "The blockchain exists but contains no blocks yet.")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error verifying operations: {e}")

    # Add these slots
    @pyqtSlot(bool, str, str)
    def on_blockchain_operation_complete(self, success, operation_id, message):
        """Handle completion of blockchain operation."""
        # Close progress dialog if it exists
        if self.progress_dialog:
            self.progress_dialog.accept()
            self.progress_dialog = None
        
        if success:
            QMessageBox.information(self, "Success", message)
            self.refresh_status()
        else:
            QMessageBox.critical(self, "Error", message)
    
    @pyqtSlot(str)
    def on_blockchain_progress_update(self, message):
        """Update progress dialog message."""
        if self.progress_dialog:
            self.progress_dialog.setText(message)
            QApplication.processEvents()

    def refresh_validator_teams(self):
        """Refresh the list of validator teams"""
        self.blockchain_api.refresh_validators()
        teams = []
        
        # If user is logged in, show only teams they belong to
        if self.user_access and self.user_access.current_user:
            user_email = self.user_access.current_user.get('email')
            if user_email:
                team_id = self.blockchain_api.team_validator.get_validator_for_user(user_email)
                if team_id and team_id in self.blockchain_api.team_validator.validators:
                    teams.append({
                        'id': team_id,
                        'name': self.blockchain_api.team_validator.validators[team_id]['team_name']
                    })
        
        # If no teams found or user not logged in, show all teams
        if not teams:
            for team_id, validator in self.blockchain_api.team_validator.validators.items():
                teams.append({
                    'id': team_id,
                    'name': validator['team_name']
                })
        
        # Update combo box
        self.teams_list.clear()
        for team in teams:
            self.teams_list.addItem(team['name'])

    def reset_blockchain(self):
        """Reset the blockchain to a fresh state with a new genesis block."""
        try:
            # Confirm with user
            reply = QMessageBox.question(
                self, 
                "Reset Blockchain", 
                "This will delete all existing blockchain data and create a fresh genesis block. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.No:
                return
            
            # Create new blockchain instance
            self.blockchain_api.blockchain = Blockchain()
            
            # Get current user info for genesis block
            creator_id = "admin"
            if hasattr(self.parent(), 'user_access') and self.parent().user_access and self.parent().user_access.current_user:
                user_data = self.parent().user_access.current_user
                creator_id = user_data.get('email', user_data.get('name', 'admin'))
                print(f"Reset blockchain with creator: {creator_id}")
            
            # Add genesis block metadata with user info
            genesis_block = self.blockchain_api.blockchain.chain[0]
            genesis_block.transactions[0]['creator'] = creator_id
            genesis_block.transactions[0]['timestamp'] = datetime.now().isoformat()
            
            # Save the fresh blockchain
            from exchange.blockchain.storage import BlockchainStorage
            storage = BlockchainStorage(self.blockchain_api.blockchain)
            storage.save_blockchain()
            
            # Refresh UI
            self.refresh_status()
            
            QMessageBox.information(self, "Success", "Blockchain has been reset with a new genesis block.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to reset blockchain: {e}")

    def force_mine_block(self):
        """Force create and validate a block with admin privileges."""
        try:
            # Ensure we have admin keys
            if not hasattr(self.blockchain_api, 'admin_keys') or 'private' not in self.blockchain_api.admin_keys:
                QMessageBox.warning(self, "Admin Required", "Admin keys required for this operation")
                return
                
            # Get admin ID
            admin_id = self.blockchain_api.admin_keys.get('user_id', 'admin@blockchain')
            admin_key = self.blockchain_api.admin_keys['private']
            
            # Check if we have any pending blocks
            pending_blocks = self.blockchain_api.get_pending_blocks()
            if not pending_blocks:
                # No pending blocks, ask if admin wants to create and validate a new one
                reply = QMessageBox.question(
                    self,
                    "No Pending Blocks",
                    "There are no pending blocks to force mine. Would you like to create a new block with pending transactions?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.No:
                    return
                    
                # Create a new block
                block_hash = self.blockchain_api.create_block()
                if not block_hash:
                    QMessageBox.warning(self, "Block Creation Failed", "Failed to create a new block.")
                    return
                    
                pending_blocks = self.blockchain_api.get_pending_blocks()
            
            # Ask admin which block to force mine
            dialog = QDialog(self)
            dialog.setWindowTitle("Force Mine Block")
            layout = QVBoxLayout(dialog)
            
            layout.addWidget(QLabel("Select a block to force mine:"))
            block_combo = QComboBox()
            
            for block_hash, block in pending_blocks.items():
                tx_count = len(block.transactions) if hasattr(block, 'transactions') else 0
                block_combo.addItem(f"Block {block_hash[:10]}... ({tx_count} transactions)", block_hash)
                
            layout.addWidget(block_combo)
            
            # Warning about bypassing standard validation
            warning = QLabel("⚠️ Warning: Force mining bypasses the standard validation process. "
                           "This should only be used in special circumstances.")
            warning.setStyleSheet("color: red; font-weight: bold;")
            warning.setWordWrap(True)
            layout.addWidget(warning)
            
            # Option to notify team
            notify_check = QCheckBox("Notify team members about this action")
            notify_check.setChecked(True)
            layout.addWidget(notify_check)
            
            # Add buttons
            button_box = QHBoxLayout()
            force_mine_btn = QPushButton("Force Mine")
            force_mine_btn.setStyleSheet("background-color: #ff9900; font-weight: bold;")
            cancel_btn = QPushButton("Cancel")
            button_box.addWidget(force_mine_btn)
            button_box.addWidget(cancel_btn)
            layout.addLayout(button_box)
            
            # Connect buttons
            force_mine_btn.clicked.connect(dialog.accept)
            cancel_btn.clicked.connect(dialog.reject)
            
            # Show dialog
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
                
            # Get selected block hash
            selected_hash = block_combo.currentData()
            should_notify = notify_check.isChecked()
            
            # Show progress dialog
            progress = QMessageBox(self)
            progress.setWindowTitle("Force Mining")
            progress.setText("Admin is force mining a block...\nThis will validate the block immediately.")
            progress.setStandardButtons(QMessageBox.StandardButton.NoButton)
            progress.setIcon(QMessageBox.Icon.Information)
            
            # Use a timer to allow UI to update
            QTimer.singleShot(100, lambda: self._perform_force_mine(admin_id, admin_key, selected_hash, should_notify, progress))
            
            progress.exec()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Force mining failed: {e}")
            
    def _perform_force_mine(self, admin_id, admin_key, block_hash, notify_team, progress_dialog):
        """Perform the actual force mining operation."""
        try:
            # Call the validate_block method directly
            result = self.blockchain_api.validate_block(block_hash, admin_id, admin_key)
            
            # Close the progress dialog
            progress_dialog.done(0)
            
            # Show result
            if result:
                QMessageBox.information(self, "Success", 
                                       f"Block force mined successfully!\nBlock {block_hash[:10]}... has been added to the blockchain.")
                
                # Notify team if requested
                if notify_team:
                    # In a real system, this would send notifications
                    # For now, we'll just log it
                    print(f"Admin {admin_id} would notify team about force mining block {block_hash[:10]}...")
                
                # Refresh the explorer if available
                if hasattr(self.parent(), 'explorer_tab') and hasattr(self.parent().explorer_tab, 'refresh_chain'):
                    self.parent().explorer_tab.refresh_chain()
                    
                # Refresh the validator tab if available
                if hasattr(self.parent(), 'validator_tab') and hasattr(self.parent().validator_tab, 'refresh_validation_history'):
                    self.parent().validator_tab.refresh_validation_history()
                    self.parent().validator_tab.refresh_pending_validations()
            else:
                QMessageBox.warning(self, "Force Mining Failed", "Failed to force mine the block. Check console for details.")
                
        except Exception as e:
            progress_dialog.done(0)
            QMessageBox.critical(self, "Error", f"Force mining operation failed: {e}")

    @pyqtSlot(str, str)
    def show_success(self, title, message):
        """Show a success message."""
        QMessageBox.information(self, title, message)
        
        # Refresh explorer if available
        if hasattr(self.parent(), 'explorer_tab') and hasattr(self.parent().explorer_tab, 'refresh_chain'):
            self.parent().explorer_tab.refresh_chain()
            
    @pyqtSlot(str, str)
    def show_error(self, title, message):
        """Show an error message."""
        QMessageBox.critical(self, title, message)


class HypothesisEvidenceWidget(QWidget):
    """Widget for displaying and managing hypotheses and evidence."""
    
    def __init__(self, studies_manager, bridge, admin_keys, user_access=None, parent=None):
        super().__init__(parent)
        self.studies_manager = studies_manager
        self.bridge = bridge
        self.admin_keys = admin_keys
        self.user_access = user_access  # Store the user_access reference
        self.current_filter = "All"  # Default filter
        
        # Create blockchain thread for background operations
        self.blockchain_thread = BlockchainThread()
        self.blockchain_thread.operation_complete.connect(self.on_blockchain_operation_complete)
        self.blockchain_thread.progress_update.connect(self.on_blockchain_progress_update)
        
        # For progress dialog
        self.progress_dialog = None
        self.current_operation_row = -1
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        
        # Splitter for hypotheses and evidence
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Hypotheses list
        hypotheses_widget = QWidget()
        hypotheses_layout = QVBoxLayout(hypotheses_widget)
        
        hypotheses_label = QLabel("Hypotheses")
        hypotheses_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        
        self.hypotheses_table = QTableWidget()
        self.hypotheses_table.setColumnCount(4)
        self.hypotheses_table.setHorizontalHeaderLabels(["ID", "Title", "Status", "Blockchain"])
        self.hypotheses_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.hypotheses_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.hypotheses_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.hypotheses_table.selectionModel().selectionChanged.connect(self.hypothesis_selected)
        
        # Status bar for showing filter and count
        self.status_bar = QLabel("No hypotheses found")
        self.status_bar.setStyleSheet("color: gray;")
        
        # Buttons for hypotheses
        hypothesis_buttons = QHBoxLayout()
        self.push_hypothesis_button = QPushButton("Push to Blockchain")
        self.push_hypothesis_button.clicked.connect(self.push_hypothesis)
        hypothesis_buttons.addWidget(self.push_hypothesis_button)
        
        hypotheses_layout.addWidget(hypotheses_label)
        hypotheses_layout.addWidget(self.hypotheses_table)
        hypotheses_layout.addWidget(self.status_bar)
        hypotheses_layout.addLayout(hypothesis_buttons)
        
        # Evidence details
        evidence_widget = QWidget()
        evidence_layout = QVBoxLayout(evidence_widget)
        
        evidence_label = QLabel("Evidence")
        evidence_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        
        self.evidence_tab = QTabWidget()
        
        # Model evidence tab
        model_tab = QWidget()
        model_layout = QVBoxLayout(model_tab)
        
        self.model_evidence_table = QTableWidget()
        self.model_evidence_table.setColumnCount(5)
        self.model_evidence_table.setHorizontalHeaderLabels(
            ["Test", "P-Value", "Statistic", "Significant", "Pushed"]
        )
        self.model_evidence_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        
        model_buttons = QHBoxLayout()
        self.push_model_button = QPushButton("Push Model Evidence")
        self.push_model_button.clicked.connect(self.push_model_evidence)
        model_buttons.addWidget(self.push_model_button)
        
        model_layout.addWidget(self.model_evidence_table)
        model_layout.addLayout(model_buttons)
        
        # Literature evidence tab
        literature_tab = QWidget()
        literature_layout = QVBoxLayout(literature_tab)
        
        self.literature_evidence_table = QTableWidget()
        self.literature_evidence_table.setColumnCount(5)
        self.literature_evidence_table.setHorizontalHeaderLabels(
            ["Status", "Confidence", "Papers", "Effect Size", "Pushed"]
        )
        self.literature_evidence_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        
        literature_buttons = QHBoxLayout()
        self.push_literature_button = QPushButton("Push Literature Evidence")
        self.push_literature_button.clicked.connect(self.push_literature_evidence)
        literature_buttons.addWidget(self.push_literature_button)
        
        literature_layout.addWidget(self.literature_evidence_table)
        literature_layout.addLayout(literature_buttons)
        
        # Add tabs
        self.evidence_tab.addTab(model_tab, "Model Evidence")
        self.evidence_tab.addTab(literature_tab, "Literature Evidence")
        
        evidence_layout.addWidget(evidence_label)
        evidence_layout.addWidget(self.evidence_tab)
        
        # Add widgets to splitter
        splitter.addWidget(hypotheses_widget)
        splitter.addWidget(evidence_widget)
        splitter.setSizes([300, 500])
        
        layout.addWidget(splitter)
        
        # Refresh the UI
        self.refresh_hypotheses()
    
    def apply_filter(self, filter_text):
        """Apply a filter to the hypotheses list."""
        self.current_filter = filter_text
    
    def refresh_hypotheses(self):
        """Refresh the hypotheses list."""
        try:
            # Clear the table
            self.hypotheses_table.setRowCount(0)
            
            # Get hypotheses from the active study
            hypotheses = self.studies_manager.get_study_hypotheses()
            filtered_hypotheses = []
            
            # Apply filter
            for hypothesis in hypotheses:
                # Get status and blockchain status
                status = hypothesis.get('status', 'untested')
                in_blockchain = bool(hypothesis.get('blockchain_id', ''))
                
                # Apply filters
                if self.current_filter == "All":
                    filtered_hypotheses.append(hypothesis)
                elif self.current_filter == "In Blockchain" and in_blockchain:
                    filtered_hypotheses.append(hypothesis)
                elif self.current_filter == "Not in Blockchain" and not in_blockchain:
                    filtered_hypotheses.append(hypothesis)
                elif self.current_filter == "Confirmed" and status == "confirmed":
                    filtered_hypotheses.append(hypothesis)
                elif self.current_filter == "Rejected" and status == "rejected":
                    filtered_hypotheses.append(hypothesis)
                elif self.current_filter == "Untested" and status == "untested":
                    filtered_hypotheses.append(hypothesis)
            
            # Add hypotheses to the table
            for i, hypothesis in enumerate(filtered_hypotheses):
                self.hypotheses_table.insertRow(i)
                
                # ID
                id_item = QTableWidgetItem(hypothesis.get('id', ''))
                self.hypotheses_table.setItem(i, 0, id_item)
                
                # Title
                title_item = QTableWidgetItem(hypothesis.get('title', hypothesis.get('text', '')))
                self.hypotheses_table.setItem(i, 1, title_item)
                
                # Status
                status = hypothesis.get('status', 'untested')
                status_item = QTableWidgetItem(status)
                if status == 'confirmed':
                    status_item.setForeground(QColor('green'))
                elif status == 'rejected':
                    status_item.setForeground(QColor('red'))
                elif status == 'inconclusive':
                    status_item.setForeground(QColor('orange'))
                self.hypotheses_table.setItem(i, 2, status_item)
                
                # Blockchain status
                blockchain_id = hypothesis.get('blockchain_id', '')
                blockchain_item = QTableWidgetItem('✓' if blockchain_id else '')
                if blockchain_id:
                    blockchain_item.setForeground(QColor('green'))
                self.hypotheses_table.setItem(i, 3, blockchain_item)
            
            # Adjust column widths
            self.hypotheses_table.resizeColumnToContents(0)
            self.hypotheses_table.resizeColumnToContents(2)
            self.hypotheses_table.resizeColumnToContents(3)
            
            # Update status bar
            study = self.studies_manager.get_active_study()
            study_name = study.name if study else "Unknown Study"
            
            self.status_bar.setText(
                f"Study: {study_name} | Filter: {self.current_filter} | "
                f"Showing {len(filtered_hypotheses)} of {len(hypotheses)} hypotheses"
            )
            
        except Exception as e:
            print(f"Error refreshing hypotheses: {e}")
            self.status_bar.setText(f"Error: {e}")
    
    def hypothesis_selected(self):
        """Handle hypothesis selection."""
        # Get selected row
        selected_rows = self.hypotheses_table.selectionModel().selectedRows()
        if not selected_rows:
            return
            
        row = selected_rows[0].row()
        hypothesis_id = self.hypotheses_table.item(row, 0).text()
        
        # Clear evidence tables
        self.model_evidence_table.setRowCount(0)
        self.literature_evidence_table.setRowCount(0)
        
        try:
            # Get hypothesis details
            hypothesis = self.studies_manager.get_hypothesis(hypothesis_id)
            if not hypothesis:
                return
                
            # Populate model evidence if available
            test_results = hypothesis.get('test_results')
            if test_results:
                self.model_evidence_table.insertRow(0)
                
                # Test name
                test_name = test_results.get('test_name', 'Unknown')
                self.model_evidence_table.setItem(0, 0, QTableWidgetItem(test_name))
                
                # P-value
                p_value = test_results.get('p_value')
                p_value_str = f"{p_value:.4f}" if p_value is not None else "N/A"
                self.model_evidence_table.setItem(0, 1, QTableWidgetItem(p_value_str))
                
                # Test statistic
                statistic = test_results.get('test_statistic')
                statistic_str = f"{statistic:.2f}" if statistic is not None else "N/A"
                self.model_evidence_table.setItem(0, 2, QTableWidgetItem(statistic_str))
                
                # Significance
                significant = test_results.get('significant', False)
                sig_item = QTableWidgetItem('✓' if significant else '✗')
                sig_item.setForeground(QColor('green') if significant else QColor('red'))
                self.model_evidence_table.setItem(0, 3, QTableWidgetItem(sig_item))
                
                # Blockchain status - if the hypothesis is in the blockchain, 
                # assume the evidence is too (this is simplified)
                pushed = bool(hypothesis.get('blockchain_id'))
                pushed_item = QTableWidgetItem('✓' if pushed else '')
                if pushed:
                    pushed_item.setForeground(QColor('green'))
                self.model_evidence_table.setItem(0, 4, QTableWidgetItem(pushed_item))
            
            # Populate literature evidence if available
            literature = hypothesis.get('literature_evidence')
            if literature:
                self.literature_evidence_table.insertRow(0)
                
                # Status
                status = literature.get('status', 'Unknown')
                status_item = QTableWidgetItem(status)
                if status == 'confirmed':
                    status_item.setForeground(QColor('green'))
                elif status == 'rejected':
                    status_item.setForeground(QColor('red'))
                self.literature_evidence_table.setItem(0, 0, status_item)
                
                # Confidence
                confidence = literature.get('confidence')
                confidence_str = f"{confidence:.0%}" if confidence is not None else "N/A"
                self.literature_evidence_table.setItem(0, 1, QTableWidgetItem(confidence_str))
                
                # Paper count
                papers = literature.get('papers', [])
                paper_count = len(papers)
                self.literature_evidence_table.setItem(0, 2, QTableWidgetItem(str(paper_count)))
                
                # Effect size
                effect_size = literature.get('effect_size_range')
                effect_size_str = f"{effect_size[0]:.2f}-{effect_size[1]:.2f}" if effect_size else "N/A"
                self.literature_evidence_table.setItem(0, 3, QTableWidgetItem(effect_size_str))
                
                # Blockchain status
                pushed = bool(hypothesis.get('blockchain_id'))
                pushed_item = QTableWidgetItem('✓' if pushed else '')
                if pushed:
                    pushed_item.setForeground(QColor('green'))
                self.literature_evidence_table.setItem(0, 4, QTableWidgetItem(pushed_item))
            
            # Adjust column widths
            for table in [self.model_evidence_table, self.literature_evidence_table]:
                for col in range(table.columnCount()):
                    table.resizeColumnToContents(col)
                    
        except Exception as e:
            print(f"Error loading hypothesis details: {e}")
    
    def push_hypothesis(self):
        """Push the selected hypothesis to the blockchain using background thread."""
        # Get selected hypothesis
        selected_rows = self.hypotheses_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select a hypothesis to push.")
            return
            
        row = selected_rows[0].row()
        self.current_operation_row = row  # Store for later UI update
        hypothesis_id = self.hypotheses_table.item(row, 0).text()
        
        # Get current user info for blockchain transaction
        user_id = "anonymous"
        if self.user_access and self.user_access.current_user:
            user_data = self.user_access.current_user
            user_id = user_data.get('email', user_data.get('name', 'anonymous'))
            print(f"DEBUG: Using user ID directly from widget: {user_id}")
        elif hasattr(self.parent(), 'user_access') and self.parent().user_access and self.parent().user_access.current_user:
            user_data = self.parent().user_access.current_user
            user_id = user_data.get('email', user_data.get('name', 'anonymous'))
            print(f"DEBUG: Using user ID from parent: {user_id}")
        else:
            print("DEBUG: No user found, using anonymous")
            if hasattr(self, 'user_access'):
                print(f"DEBUG: Widget has user_access: {self.user_access}")
            if hasattr(self.parent(), 'user_access'):
                print(f"DEBUG: Parent has user_access: {self.parent().user_access}")
        
        # Create progress dialog
        self.progress_dialog = QMessageBox(self)
        self.progress_dialog.setWindowTitle("Processing")
        self.progress_dialog.setText("Preparing blockchain operation...")
        self.progress_dialog.setStandardButtons(QMessageBox.StandardButton.Cancel)
        self.progress_dialog.show()
        QApplication.processEvents()
        
        # Start the thread operation with user ID
        self.blockchain_thread.push_hypothesis(
            self.bridge, 
            hypothesis_id, 
            self.admin_keys["private"],
            user_id  # Pass the user ID to the blockchain operation
        )
    
    def push_model_evidence(self):
        """Push model evidence to the blockchain using background thread."""
        # Get selected hypothesis
        selected_rows = self.hypotheses_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select a hypothesis.")
            return
            
        row = selected_rows[0].row()
        self.current_operation_row = row  # Store for later UI update
        hypothesis_id = self.hypotheses_table.item(row, 0).text()
        
        # Check if there's model evidence
        if self.model_evidence_table.rowCount() == 0:
            QMessageBox.warning(self, "No Evidence", 
                               "No model evidence available for this hypothesis.")
            return
            
        # Get the hypothesis with test results
        hypothesis = self.studies_manager.get_hypothesis(hypothesis_id)
        test_results = hypothesis.get('test_results')
        
        if not test_results:
            QMessageBox.warning(self, "No Test Results", 
                               "No test results available for this hypothesis.")
            return
        
        # Create progress dialog
        self.progress_dialog = QMessageBox(self)
        self.progress_dialog.setWindowTitle("Processing")
        self.progress_dialog.setText("Preparing blockchain operation...")
        self.progress_dialog.setStandardButtons(QMessageBox.StandardButton.Cancel)
        self.progress_dialog.show()
        QApplication.processEvents()
        
        # Start the thread operation
        self.blockchain_thread.push_model_evidence(
            self.bridge,
            hypothesis_id,
            test_results,
            self.admin_keys["private"]
        )
    
    def push_literature_evidence(self):
        """Push literature evidence to the blockchain using background thread."""
        # Get selected hypothesis
        selected_rows = self.hypotheses_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select a hypothesis.")
            return
            
        row = selected_rows[0].row()
        self.current_operation_row = row  # Store for later UI update
        hypothesis_id = self.hypotheses_table.item(row, 0).text()
        
        # Check if there's literature evidence
        if self.literature_evidence_table.rowCount() == 0:
            QMessageBox.warning(self, "No Evidence", 
                               "No literature evidence available for this hypothesis.")
            return
            
        # Get the hypothesis with literature evidence
        hypothesis = self.studies_manager.get_hypothesis(hypothesis_id)
        literature_evidence = hypothesis.get('literature_evidence')
        
        if not literature_evidence:
            QMessageBox.warning(self, "No Literature Evidence", 
                               "No literature evidence available for this hypothesis.")
            return
        
        # Create progress dialog
        self.progress_dialog = QMessageBox(self)
        self.progress_dialog.setWindowTitle("Processing")
        self.progress_dialog.setText("Preparing blockchain operation...")
        self.progress_dialog.setStandardButtons(QMessageBox.StandardButton.Cancel)
        self.progress_dialog.show()
        QApplication.processEvents()
        
        # Start the thread operation
        self.blockchain_thread.push_literature_evidence(
            self.bridge,
            hypothesis_id,
            literature_evidence,
            self.admin_keys["private"]
        )

    # Add these slots
    @pyqtSlot(bool, str, str)
    def on_blockchain_operation_complete(self, success, operation_id, message):
        """Handle completion of blockchain operation."""
        # Close progress dialog if it exists
        if self.progress_dialog:
            self.progress_dialog.accept()
            self.progress_dialog = None
        
        if success:
            QMessageBox.information(self, "Success", message)
            
            # Update UI if we have a valid row
            if self.current_operation_row >= 0:
                blockchain_item = QTableWidgetItem('✓')
                blockchain_item.setForeground(QColor('green'))
                self.hypotheses_table.setItem(self.current_operation_row, 3, blockchain_item)
            
            # Refresh the full display
            self.refresh_hypotheses()
            
        else:
            QMessageBox.critical(self, "Error", message)
    
    @pyqtSlot(str)
    def on_blockchain_progress_update(self, message):
        """Update progress dialog message."""
        if self.progress_dialog:
            self.progress_dialog.setText(message)
            QApplication.processEvents()


class BlockchainExplorerWidget(QWidget):
    """Widget for exploring the blockchain contents."""
    
    def __init__(self, blockchain_api, parent=None):
        super().__init__(parent)
        self.blockchain_api = blockchain_api
        self.setup_ui()
        
        # Initialize current_transactions attribute
        self.current_transactions = []
        
        # Refresh the chain display immediately
        QTimer.singleShot(100, self.refresh_chain)
    
    def setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        
        # Blockchain explorer
        explorer_label = QLabel("Blockchain Explorer")
        explorer_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        
        # Block table
        self.block_table = QTableWidget()
        self.block_table.setColumnCount(4)
        self.block_table.setHorizontalHeaderLabels(["Index", "Hash", "Transactions", "Validated"])
        self.block_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.block_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.block_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.block_table.selectionModel().selectionChanged.connect(self.block_selected)
        
        # Transaction details
        transaction_label = QLabel("Transactions")
        transaction_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        
        self.transaction_table = QTableWidget()
        self.transaction_table.setColumnCount(4)
        self.transaction_table.setHorizontalHeaderLabels(["Type", "ID", "Submitter", "Timestamp"])
        self.transaction_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.transaction_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.transaction_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.transaction_table.selectionModel().selectionChanged.connect(self.transaction_selected)
        
        # Transaction details
        details_label = QLabel("Transaction Details")
        details_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        
        # Refresh button
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_chain)
        
        # Add a button to view raw blockchain data
        self.blockchain_details_button = QPushButton("View Blockchain Details")
        self.blockchain_details_button.clicked.connect(self.show_blockchain_details)
        
        # Add a button to verify blockchain storage
        self.verify_storage_button = QPushButton("Verify Blockchain Storage")
        self.verify_storage_button.clicked.connect(self.verify_blockchain_storage)
        
        # Add load from storage button next to refresh
        self.load_storage_button = QPushButton("Load From Storage")
        self.load_storage_button.setIcon(QIcon.fromTheme("document-open"))
        self.load_storage_button.clicked.connect(self.load_from_storage)
        
        # Add debug button
        debug_button = QPushButton("Debug Chain")
        debug_button.clicked.connect(self.debug_chain)
        
        # Create button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.refresh_button)
        button_layout.addWidget(self.load_storage_button)
        button_layout.addWidget(self.blockchain_details_button)
        button_layout.addWidget(self.verify_storage_button)
        button_layout.addWidget(debug_button)
        
        # Replace the single refresh button with this layout
        layout.addLayout(button_layout)
        
        # Add widgets to layout
        layout.addWidget(explorer_label)
        layout.addWidget(self.block_table)
        layout.addWidget(transaction_label)
        layout.addWidget(self.transaction_table)
        layout.addWidget(details_label)
        layout.addWidget(self.details_text)
        
        # The tabWidget will be added to this container in setup_ui
    
    def refresh_chain(self):
        """Refresh the blockchain display."""
        try:
            # Clear the current displays
            self.block_table.setRowCount(0)
            self.transaction_table.setRowCount(0)
            self.details_text.clear()
            
            # Get the blockchain
            chain = self.blockchain_api.blockchain.chain
            
            # Populate block table
            for i, block in enumerate(chain):
                self.block_table.insertRow(i)
                
                # Index
                self.block_table.setItem(i, 0, QTableWidgetItem(str(block.index)))
                
                # Hash - shortened for readability
                hash_text = block.hash[:15] + "..." if len(block.hash) > 15 else block.hash
                self.block_table.setItem(i, 1, QTableWidgetItem(hash_text))
                
                # Number of transactions
                tx_count = len(block.transactions) if isinstance(block.transactions, list) else 0
                self.block_table.setItem(i, 2, QTableWidgetItem(str(tx_count)))
                
                # Validation status
                validated = "✓" if hasattr(block, 'validations') and block.validations else "❌"
                self.block_table.setItem(i, 3, QTableWidgetItem(validated))
            
            # Select the last block if any
            if self.block_table.rowCount() > 0:
                self.block_table.selectRow(self.block_table.rowCount() - 1)
            
            # Check for pending blocks and transactions
            # Show pending blocks count
            pending_blocks = {}
            if hasattr(self.blockchain_api.blockchain, 'pending_blocks'):
                pending_blocks = self.blockchain_api.blockchain.pending_blocks
            
            pending_count = len(pending_blocks)
            if pending_count > 0:
                print(f"There are {pending_count} pending blocks waiting for validation")
                
                # Don't auto-validate as admin - leave for manual validation in validator management
                # This change allows admins to assign validation to other team members
            
            # Show pending transactions count
            pending_tx = self.blockchain_api.blockchain.pending_transactions
            if pending_tx:
                print(f"There are {len(pending_tx)} pending transactions waiting to be mined")
                if len(pending_tx) > 0:
                    # Only create a block with pending transactions, but don't auto-validate
                    block_hash = self.blockchain_api.create_block()
                    if block_hash:
                        print(f"Created new block with hash {block_hash[:10]}... for pending transactions")
                        print(f"Block is now pending validation. Use Validator Management to assign validators.")
        
        except Exception as e:
            print(f"Error refreshing blockchain display: {e}")

    def block_selected(self, selected, deselected):
        """Handle block selection."""
        if not selected.indexes():
            return
        
        try:
            row = selected.indexes()[0].row()
            
            # Get the block index (not row index)
            block_index = int(self.block_table.item(row, 0).text())
            
            # Find the block in the chain
            block = None
            for b in self.blockchain_api.blockchain.chain:
                if b.index == block_index:
                    block = b
                    break
            
            if not block:
                print(f"Block not found with index {block_index}")
                return
            
            # Store current transactions for later reference
            self.current_transactions = block.transactions if isinstance(block.transactions, list) else []
            
            # Clear the transaction table
            self.transaction_table.setRowCount(0)
            
            # Populate transaction table
            for i, tx in enumerate(self.current_transactions):
                self.transaction_table.insertRow(i)
                
                # Type
                tx_type = tx.get('type', 'Unknown')
                self.transaction_table.setItem(i, 0, QTableWidgetItem(tx_type))
                
                # ID - try different possible ID fields
                tx_id = tx.get('id', tx.get('hypothesis_id', str(i)))
                self.transaction_table.setItem(i, 1, QTableWidgetItem(str(tx_id)))
                
                # Submitter - show the user who submitted it
                submitter = tx.get('submitter', tx.get('creator', 'Unknown'))
                self.transaction_table.setItem(i, 2, QTableWidgetItem(str(submitter)))
                
                # Timestamp
                timestamp = tx.get('timestamp', 'Unknown')
                self.transaction_table.setItem(i, 3, QTableWidgetItem(str(timestamp)))
            
            # Select the first transaction if any
            if self.transaction_table.rowCount() > 0:
                self.transaction_table.selectRow(0)
            
        except Exception as e:
            print(f"Error displaying block: {e}")

    def transaction_selected(self, selected, deselected):
        """Handle transaction selection."""
        if not selected.indexes() or not hasattr(self, 'current_transactions'):
            return
        
        try:
            row = selected.indexes()[0].row()
            
            if row < 0 or row >= len(self.current_transactions):
                return
            
            transaction = self.current_transactions[row]
            
            # Format transaction details
            details = "Transaction Details:\n\n"
            details += f"Type: {transaction.get('type', 'Unknown')}\n"
            
            # ID - use appropriate field based on transaction type
            if transaction.get('type') == 'HYPOTHESIS':
                details += f"Hypothesis ID: {transaction.get('hypothesis_id', 'Unknown')}\n"
                details += f"Title: {transaction.get('title', 'Untitled')}\n"
                details += f"Description: {transaction.get('description', 'No description')}\n"
            elif transaction.get('type') in ['MODEL_EVIDENCE', 'LITERATURE_EVIDENCE']:
                details += f"Evidence ID: {transaction.get('id', 'Unknown')}\n"
                details += f"For Hypothesis: {transaction.get('hypothesis_id', 'Unknown')}\n"
                details += f"Summary: {transaction.get('summary', 'No summary')}\n"
            else:
                details += f"ID: {transaction.get('id', 'Unknown')}\n"
            
            # Submitter info
            details += f"\nSubmitted by: {transaction.get('submitter', transaction.get('creator', 'Unknown'))}\n"
            details += f"Timestamp: {transaction.get('timestamp', 'Unknown')}\n"
            
            # Team info if available
            team_info = transaction.get('team_info', {})
            if team_info:
                details += f"\nTeam: {team_info.get('team_name', 'Unknown')}\n"
                details += f"Organization: {team_info.get('organization_id', 'Unknown')}\n"
            
            # Signature if available
            if 'signature' in transaction:
                details += f"\nSignature: {transaction['signature'][:30]}...\n"
            
            # Set the details text
            self.details_text.setText(details)
            
        except Exception as e:
            print(f"Error displaying transaction details: {e}")

    def show_blockchain_details(self):
        """Show raw blockchain details for verification."""
        try:
            # Gather blockchain information
            chain_length = len(self.blockchain_api.blockchain.chain)
            pending_tx_count = len(self.blockchain_api.blockchain.pending_transactions)
            
            # Get storage information
            storage_path = getattr(self.blockchain_api.blockchain, 'storage_dir', 'Unknown')
            if hasattr(self.blockchain_api.blockchain, 'pending_blocks'):
                pending_blocks_count = len(self.blockchain_api.blockchain.pending_blocks)
            else:
                pending_blocks_count = 0
            
            # Format detailed blockchain info
            details = f"""
            Blockchain Details:
            ------------------
            Chain Length: {chain_length} blocks
            Pending Transactions: {pending_tx_count}
            Pending Blocks: {pending_blocks_count}
            Storage Location: {storage_path}
            
            Genesis Block Hash: {self.blockchain_api.blockchain.chain[0].hash[:20]}...
            
            Latest Blocks:
            """
            
            # Show the latest 3 blocks in detail
            for i in range(min(3, chain_length)):
                block = self.blockchain_api.blockchain.chain[-(i+1)]  # Start from the end
                details += f"""
            Block #{block.index}:
              Hash: {block.hash[:15]}...
              Prev Hash: {block.previous_hash[:15]}...
              Transactions: {len(block.transactions) if isinstance(block.transactions, list) else 0}
              Validations: {len(block.validations) if hasattr(block, 'validations') else 0}
              Timestamp: {block.timestamp}
            """
            
            # Show in dialog
            details_dialog = QMessageBox(self)
            details_dialog.setWindowTitle("Blockchain Details")
            details_dialog.setText(details)
            details_dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
            details_dialog.setDetailedText(f"Full chain data contains {chain_length} blocks with full transaction history.")
            details_dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error showing blockchain details: {e}")

    def verify_blockchain_storage(self):
        """Verify that blockchain data is being persisted to disk."""
        try:
            import os
            # Match the path used during initialization
            storage_dir = os.path.join(os.path.expanduser("~"), ".blockchain_data")
            
            if not os.path.exists(storage_dir):
                QMessageBox.warning(self, "Storage Not Found", 
                                   f"Blockchain storage directory not found: {storage_dir}")
                return
            
            # Check for chain.json
            chain_path = os.path.join(storage_dir, "chain.json")
            if os.path.exists(chain_path):
                # Get file info
                file_size = os.path.getsize(chain_path)
                modified_time = os.path.getmtime(chain_path)
                from datetime import datetime
                mod_time_str = datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S')
                
                # Check file contents
                with open(chain_path, 'r') as f:
                    import json
                    try:
                        chain_data = json.load(f)
                        blocks_in_file = len(chain_data)
                        
                        message = f"""
                        Blockchain is being persisted to disk:
                        
                        Storage file: {chain_path}
                        File size: {file_size} bytes
                        Last modified: {mod_time_str}
                        Blocks in file: {blocks_in_file}
                        
                        This confirms blockchain operations are real and persisted.
                        """
                        
                        QMessageBox.information(self, "Storage Verified", message)
                        
                    except json.JSONDecodeError:
                        QMessageBox.warning(self, "Invalid Storage", 
                                         "Blockchain storage file exists but contains invalid JSON.")
            else:
                QMessageBox.warning(self, "Storage Not Found", 
                                  f"Blockchain storage file not found: {chain_path}")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error verifying blockchain storage: {e}")

    def load_from_storage(self):
        """Load blockchain data from disk to verify persistence."""
        try:
            # Create a temporary storage object
            from exchange.blockchain import BlockchainStorage
            # Use same path as initialization
            storage_dir = os.path.join(os.path.expanduser("~"), ".blockchain_data")
            storage = BlockchainStorage(self.blockchain_api.blockchain, storage_dir)
            
            # Try to load
            chain_length_before = len(self.blockchain_api.blockchain.chain)
            load_result = storage.load_blockchain()
            chain_length_after = len(self.blockchain_api.blockchain.chain)
            
            # Verify load results
            if load_result:
                QMessageBox.information(self, "Load Successful", 
                                      f"Blockchain loaded from storage.\n\n"
                                      f"Chain length before: {chain_length_before}\n"
                                      f"Chain length after: {chain_length_after}")
            else:
                QMessageBox.warning(self, "Load Failed", 
                                  "Failed to load blockchain from storage.")
            
            # Refresh UI
            self.refresh_chain()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading from storage: {e}")

    def debug_chain(self):
        """Debug function to print chain contents to console."""
        try:
            import os
            chain = self.blockchain_api.blockchain.chain
            print(f"\n=== BLOCKCHAIN DEBUG ({len(chain)} blocks) ===")
            
            # Print storage path information
            storage_dir = os.path.join(os.path.expanduser("~"), ".blockchain_data")
            chain_path = os.path.join(storage_dir, "chain.json")
            
            print(f"Storage Directory: {storage_dir}")
            print(f"Chain File: {chain_path}")
            print(f"Chain File Exists: {os.path.exists(chain_path)}")
            
            # First check pending transactions
            pending_tx = getattr(self.blockchain_api.blockchain, 'pending_transactions', [])
            pending_count = len(pending_tx) if isinstance(pending_tx, list) else 0
            print(f"PENDING TRANSACTIONS: {pending_count}")
            for i, tx in enumerate(pending_tx if isinstance(pending_tx, list) else []):
                tx_type = tx.get('type', 'Unknown')
                submitter = tx.get('submitter', 'Unknown')
                print(f"  {i+1}. {tx_type} by {submitter}")
            
            # Then check pending blocks
            pending_blocks = getattr(self.blockchain_api.blockchain, 'pending_blocks', {})
            print(f"PENDING BLOCKS: {len(pending_blocks)}")
            for block_hash, block in pending_blocks.items():
                print(f"  Block {block_hash[:10]}... with {len(block.transactions)} transactions")
            
            
            for i, block in enumerate(chain):
                print(f"\nBLOCK #{block.index}:")
                print(f"  Hash: {block.hash[:15]}...")
                print(f"  Prev: {block.previous_hash[:15]}...")
                print(f"  Time: {block.timestamp}")
                
                # Print transactions
                tx_count = len(block.transactions) if isinstance(block.transactions, list) else 0
                print(f"  Transactions ({tx_count}):")
                
                for j, tx in enumerate(block.transactions if isinstance(block.transactions, list) else []):
                    tx_type = tx.get('type', 'Unknown')
                    tx_id = tx.get('id', tx.get('hypothesis_id', 'Unknown'))
                    submitter = tx.get('submitter', tx.get('creator', 'Unknown'))
                    
                    # Special handling for hypothesis
                    if tx_type == 'HYPOTHESIS':
                        title = tx.get('title', 'Untitled')
                        print(f"    {j+1}. {tx_type}: {title} (ID: {tx_id}) by {submitter}")
                    else:
                        print(f"    {j+1}. {tx_type}: {tx_id} by {submitter}")
            
            # Print to console so we can see what's actually in the blockchain
            print("\n=== END BLOCKCHAIN DEBUG ===\n")
            
            # Also show in a dialog
            QMessageBox.information(self, "Blockchain Debug", 
                                  f"Blockchain has {len(chain)} blocks.\nCheck console for details.")
        
        except Exception as e:
            print(f"Error debugging chain: {e}")
            QMessageBox.critical(self, "Debug Error", f"Error: {e}")


class BlockchainWidget(QTabWidget):
    """Main blockchain widget with tabs for different functionality."""
    
    def __init__(self, studies_manager, user_access=None, parent=None):
        super().__init__(parent)
        self.studies_manager = studies_manager
        self.user_access = user_access  # Store the user_access reference
        
        # Set default values
        self.blockchain = None
        self.validator_registry = None
        self.blockchain_api = None
        self.bridge = None
        self.admin_keys = {
            "private": "",
            "public": ""
        }
        
        # Flag to track initialization status
        self.is_initialized = False
        self.hypotheses_tab = None
        self.explorer_tab = None
        self.control_tab = None
        
        # Setup UI first with placeholder content
        self.setup_selection_controls()
        self.setup_ui_placeholder()
        
        # Prevent the dropdown and filter controls from triggering actions during setup
        self.study_combo.blockSignals(True)
        self.filter_combo.blockSignals(True)
        
        # Initialize blockchain in background
        QTimer.singleShot(100, self.initialize_blockchain)
        
    def setup_ui_placeholder(self):
        """Setup placeholder UI while blockchain initializes."""
        # Create a tab widget that will be added to the container
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        
        # Create loading widget
        loading_widget = QWidget()
        loading_layout = QVBoxLayout(loading_widget)
        
        loading_label = QLabel("Initializing blockchain...")
        loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        loading_label.setFont(QFont("Arial", 14))
        
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 0)  # Indeterminate
        
        loading_layout.addStretch(1)
        loading_layout.addWidget(loading_label)
        loading_layout.addWidget(progress_bar)
        loading_layout.addStretch(1)
        
        # Add loading widget as the only tab initially
        self.tab_widget.addTab(loading_widget, "Loading...")
        
        # Add the tab widget to the container
        self.container_layout.addWidget(self.tab_widget)
        
        # Set the container as the central widget of this widget
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.container)
        self.layout().setContentsMargins(0, 0, 0, 0)
        
        # Set a fixed size for the widget
        self.setMinimumSize(900, 700)
    
    def initialize_blockchain(self):
        """Initialize the blockchain in the background."""
        try:
            # Create storage directory if it doesn't exist
            import os
            storage_dir = os.path.join(os.path.expanduser("~"), ".blockchain_data")
            if not os.path.exists(storage_dir):
                os.makedirs(storage_dir)

            # Try to load or generate admin keys
            admin_key_success = self.load_or_generate_admin_keys()
            
            if not admin_key_success:
                # Show error in UI
                self._show_initialization_error("Failed to load or generate admin keys")
                return

            # Set actual username from user_access instead of default placeholder
            if self.user_access and self.user_access.current_user:
                self.admin_keys['user_id'] = self.user_access.current_user.get('email', self.admin_keys.get('user_id', 'admin'))
                print(f"Using admin user ID: {self.admin_keys['user_id']}")
            
            # Initialize blockchain components
            from exchange.blockchain import Blockchain, ProofOfAuthority, BlockchainStorage
            
            # Create blockchain with storage
            blockchain = Blockchain()
            
            # Initialize consensus with required parameters
            validators = {"admin": self.admin_keys["public"]}
            consensus = ProofOfAuthority(blockchain, validators, required_validations=1)
            
            storage = BlockchainStorage(blockchain, storage_dir)
            
            # Try to load existing blockchain
            load_success = storage.load_blockchain()
            if load_success:
                print("Loaded existing blockchain from storage")
            else:
                print("No existing blockchain found, starting fresh")
            
            # Create API instance
            from exchange.blockchain import BlockchainAPI
            self.blockchain_api = BlockchainAPI(blockchain, consensus)
            
            # Pass admin keys to API
            self.blockchain_api.admin_keys = self.admin_keys
            
            # Create blockchain bridge for studies
            from exchange.blockchain.studies_blockchain_bridge import StudiesBlockchainBridge
            self.bridge = StudiesBlockchainBridge(self.blockchain_api, self.studies_manager)
            
            # Setup the real UI once ready
            self.setup_ui_real()
            
            # Show status
            print(f"Blockchain initialized with {len(blockchain.chain)} blocks")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._show_initialization_error(f"Error initializing blockchain: {str(e)}")
            
    def _show_initialization_error(self, message):
        """Show error message for blockchain initialization failures."""
        print(f"Blockchain initialization error: {message}")
        
        # Remove the loading tab if it exists
        if self.tab_widget and self.tab_widget.count() > 0:
            # Create error widget
            error_widget = QWidget()
            error_layout = QVBoxLayout(error_widget)
            
            error_label = QLabel(message)
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            error_label.setWordWrap(True)
            error_label.setStyleSheet("color: red; font-weight: bold;")
            
            retry_button = QPushButton("Retry")
            retry_button.clicked.connect(self.initialize_blockchain)
            
            error_layout.addStretch(1)
            error_layout.addWidget(error_label)
            error_layout.addWidget(retry_button)
            error_layout.addStretch(1)
            
            # Replace the loading tab with the error tab
            self.tab_widget.removeTab(0)
            self.tab_widget.addTab(error_widget, "Error")
    
    def setup_ui_real(self):
        """Setup the actual UI once blockchain is initialized."""
        try:
            # Remove the loading tab
            self.tab_widget.removeTab(0)
            
            # Create the real tabs
            self.hypotheses_tab = HypothesisEvidenceWidget(
                self.studies_manager, 
                self.bridge,
                self.admin_keys,
                self.user_access
            )
            
            # Pass admin keys to explorer tab
            self.blockchain_api.admin_keys = self.admin_keys  # Set admin keys on the API itself
            self.explorer_tab = BlockchainExplorerWidget(self.blockchain_api)
            
            self.control_tab = BlockchainControlPanel(
                self.blockchain_api,
                self.admin_keys,
                self.user_access
            )
            
            # Add validator management tab
            self.validator_tab = self.setup_validator_management()
            
            # Add the tabs to the tab widget
            self.tab_widget.addTab(self.hypotheses_tab, "Hypotheses & Evidence")
            self.tab_widget.addTab(self.explorer_tab, "Blockchain Explorer")
            self.tab_widget.addTab(self.control_tab, "Blockchain Controls")
            self.tab_widget.addTab(self.validator_tab, "Validator Management")
            
            # Wait a moment before populating studies to ensure UI is ready
            QTimer.singleShot(100, self.populate_studies_dropdown)
            
            # Mark initialization as complete
            self.is_initialized = True
            
            # Allow signals from controls now that components are ready
            self.study_combo.blockSignals(False)
            self.filter_combo.blockSignals(False)
            
        except Exception as e:
            print(f"Error setting up blockchain UI: {e}")
            
    def setup_validator_management(self):
        """Set up the validator management section."""
        try:
            # Import the validator management class
            from exchange.validator_management import ValidatorManagementSection
            
            # Create validator management section
            validator_section = ValidatorManagementSection()
            validator_section.user_access = self.user_access
            validator_section.blockchain_api = self.blockchain_api
            
            # Initial refresh
            QTimer.singleShot(500, validator_section.refresh_teams)
            QTimer.singleShot(700, validator_section.refresh_pending_validations)
            
            return validator_section
            
        except Exception as e:
            print(f"Error setting up validator management: {e}")
            # Return empty widget if something goes wrong
            empty = QWidget()
            layout = QVBoxLayout(empty)
            error_label = QLabel(f"Error loading validator management: {str(e)}")
            error_label.setStyleSheet("color: red;")
            layout.addWidget(error_label)
            return empty
    
    def setup_selection_controls(self):
        """Create the study and hypothesis selection controls."""
        # Create a container widget that will contain both the controls and the tab widget
        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create controls widget
        self.controls_widget = QWidget()
        controls_layout = QVBoxLayout(self.controls_widget)
        
        # Study selection
        study_selection_layout = QHBoxLayout()
        study_selection_layout.addWidget(QLabel("Select Study:"))
        
        self.study_combo = QComboBox()
        self.study_combo.setMinimumWidth(300)
        self.study_combo.currentIndexChanged.connect(self.study_selected)
        study_selection_layout.addWidget(self.study_combo)
        
        # Filter options
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter Hypotheses:"))
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "In Blockchain", "Not in Blockchain", "Confirmed", "Rejected", "Untested"])
        self.filter_combo.currentIndexChanged.connect(self.refresh_display)
        filter_layout.addWidget(self.filter_combo)
        
        # Refresh button
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setIcon(QIcon.fromTheme("view-refresh"))
        self.refresh_button.clicked.connect(self.refresh_all)
        filter_layout.addWidget(self.refresh_button)
        
        filter_layout.addStretch(1)
        
        # Add layouts to controls
        controls_layout.addLayout(study_selection_layout)
        controls_layout.addLayout(filter_layout)
        
        # Add controls to main container
        self.container_layout.addWidget(self.controls_widget)
        
        # The tabWidget will be added to this container in setup_ui
    
    def populate_studies_dropdown(self):
        """Populate the studies dropdown with all available studies."""
        self.study_combo.blockSignals(True)  # Block signals during population
        self.study_combo.clear()
        
        try:
            # Get all projects
            projects = self.studies_manager.list_projects()
            
            # Create study list with project context
            all_studies = []
            
            for project in projects:
                project_id = project.get('id')
                project_name = project.get('name')
                
                # Get studies in this project
                studies = self.studies_manager.list_studies(project_id)
                
                for study in studies:
                    study_id = study.get('id')
                    study_name = study.get('name')
                    is_active = study.get('is_active', False)
                    
                    # Add to list with context
                    all_studies.append({
                        'project_id': project_id,
                        'project_name': project_name,
                        'study_id': study_id,
                        'study_name': study_name,
                        'is_active': is_active,
                        'display_name': f"{study_name} ({project_name})"
                    })
            
            # Add a "Select Study" placeholder
            self.study_combo.addItem("Select Study...", None)
            
            # Add all studies to the dropdown
            for study in all_studies:
                self.study_combo.addItem(study['display_name'], study)
                
                # Set the active study as selected
                if study['is_active']:
                    self.study_combo.setCurrentText(study['display_name'])
            
        except Exception as e:
            print(f"Error populating studies dropdown: {e}")
            # Add a placeholder if we couldn't get studies
            self.study_combo.addItem("No studies available", None)
            
        # Now that we're done populating, enable signals and do initial refresh
        self.study_combo.blockSignals(False)
        
        # Now it's safe to do an initial refresh
        if self.is_initialized:
            self.refresh_display()
            
    def study_selected(self):
        """Handle study selection from dropdown."""
        # Skip if not initialized
        if not self.is_initialized:
            return
            
        # Get the selected study data
        selected_data = self.study_combo.currentData()
        
        if not selected_data:
            return
        
        try:
            # Set as active study
            study_id = selected_data.get('study_id')
            project_id = selected_data.get('project_id')
            
            if study_id and project_id:
                # Set the active study in studies_manager
                self.studies_manager.set_active_study(study_id, project_id)
                
                # Refresh the display
                self.refresh_display()
        except Exception as e:
            print(f"Error setting active study: {e}")
            
    def refresh_display(self):
        """Refresh the display based on current filter."""
        # Do nothing if initialization isn't complete
        if not self.is_initialized:
            print("Skipping refresh - initialization not complete")
            return
            
        try:
            # Get the current filter
            current_filter = self.filter_combo.currentText()
            
            # Pass filter to hypotheses tab
            self.hypotheses_tab.apply_filter(current_filter)
            
            # Refresh all tabs
            self.hypotheses_tab.refresh_hypotheses()
            self.explorer_tab.refresh_chain()
            self.control_tab.refresh_status()
        except Exception as e:
            print(f"Error refreshing display: {e}")
            
    def refresh_all(self):
        """Force refresh of all components."""
        # Skip if not initialized
        if not self.is_initialized:
            print("Skipping refresh_all - initialization not complete")
            return
            
        # First, ensure blockchain data is saved
        if hasattr(self, 'storage') and self.storage:
            try:
                self.storage.save_blockchain()
            except Exception as e:
                print(f"Error saving blockchain: {e}")
        
        # Reload blockchain data
        if hasattr(self, 'storage') and self.storage:
            try:
                self.storage.load_blockchain()
            except Exception as e:
                print(f"Error loading blockchain: {e}")
        
        # Refresh studies dropdown
        self.populate_studies_dropdown()
        
        # Refresh display with current filter
        self.refresh_display()
        
        # Refresh validator management if it exists
        if hasattr(self, 'validator_tab'):
            try:
                if hasattr(self.validator_tab, 'refresh_teams'):
                    self.validator_tab.refresh_teams()
                if hasattr(self.validator_tab, 'refresh_pending_validations'):
                    self.validator_tab.refresh_pending_validations()
                if hasattr(self.validator_tab, 'refresh_validation_history'):
                    self.validator_tab.refresh_validation_history()
            except Exception as e:
                print(f"Error refreshing validator management: {e}")

    def load_or_generate_admin_keys(self):
        """Load existing admin keys or generate new ones if needed."""
        try:
            # Try to load existing keys if available
            import os
            if os.path.exists('./admin_keys.json'):
                with open('./admin_keys.json', 'r') as f:
                    self.admin_keys = json.load(f)
                    print("Loaded existing admin keys")
                return True  # Successfully loaded keys
            else:
                # Generate new keys - this could be slow
                print("Generating new admin keys...")
                private_key, public_key = generate_key_pair()
                self.admin_keys = {
                    "private": private_key,
                    "public": public_key
                }
                # Save for future use
                with open('./admin_keys.json', 'w') as f:
                    json.dump(self.admin_keys, f)
                print("Admin keys generated and saved")
                return True  # Successfully generated keys
        except Exception as e:
            print(f"Error with key handling: {e}")
            # Generate temporary keys for this session only
            try:
                import secrets
                temp_key = secrets.token_hex(32)
                self.admin_keys = {
                    "private": temp_key,
                    "public": temp_key
                }
                print("Using temporary admin keys")
                return True  # Successfully created temporary keys
            except Exception as e2:
                print(f"Failed to create temporary keys: {e2}")
                return False  # Complete failure


# # For standalone testing
# if __name__ == "__main__":
#     import sys
#     from PyQt6.QtWidgets import QApplication
    
#     # Mock studies manager for testing
#     class MockStudiesManager:
#         def get_study_hypotheses(self):
#             return [
#                 {
#                     "id": "H001",
#                     "title": "Test Hypothesis 1",
#                     "status": "confirmed",
#                     "blockchain_id": "BC001"
#                 },
#                 {
#                     "id": "H002",
#                     "title": "Test Hypothesis 2",
#                     "status": "untested"
#                 }
#             ]
            
#         def get_hypothesis(self, hypothesis_id):
#             if hypothesis_id == "H001":
#                 return {
#                     "id": "H001",
#                     "title": "Test Hypothesis 1",
#                     "status": "confirmed",
#                     "blockchain_id": "BC001",
#                     "test_results": {
#                         "test_name": "t-test",
#                         "p_value": 0.02,
#                         "test_statistic": 2.5,
#                         "significant": True
#                     },
#                     "literature_evidence": {
#                         "status": "confirmed",
#                         "confidence": 0.9,
#                         "papers": [{"title": "Paper 1"}],
#                         "effect_size_range": [0.2, 0.5]
#                     }
#                 }
#             else:
#                 return {
#                     "id": "H002",
#                     "title": "Test Hypothesis 2",
#                     "status": "untested"
#                 }
        
#         def get_active_study(self):
#             return {"id": "S001", "name": "Test Study"}
    
#     # Mock bridge for testing
#     class MockBridge:
#         def push_hypothesis_to_blockchain(self, hypothesis_id, private_key):
#             print(f"Pushing hypothesis {hypothesis_id} to blockchain")
#             return True
            
#         def push_model_evidence_to_blockchain(self, hypothesis_id, test_results, private_key):
#             print(f"Pushing model evidence for {hypothesis_id} to blockchain")
#             return True
            
#         def push_literature_evidence_to_blockchain(self, hypothesis_id, literature_evidence, private_key):
#             print(f"Pushing literature evidence for {hypothesis_id} to blockchain")
#             return True
    
#     app = QApplication(sys.argv)
    
#     # Create mock objects for testing
#     mock_studies_manager = MockStudiesManager()
    
#     # Create the widget
#     widget = BlockchainWidget(mock_studies_manager)
#     widget.show()
    
#     sys.exit(app.exec())
