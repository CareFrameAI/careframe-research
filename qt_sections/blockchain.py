from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QTextEdit, QSplitter, QComboBox, QMessageBox, QGroupBox, QSplitter, QTabWidget)
from PyQt6.QtCore import Qt
import logging
import json
from exchange.blockchain_ops import Blockchain
from datetime import datetime
from qt_sections.evidence_network import EvidenceNetworkView

class BlockchainSection(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.blockchain = Blockchain()  # Initialize blockchain
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create Chain Overview tab
        self.chain_overview_widget = QWidget()
        self.create_chain_overview_tab()
        self.tab_widget.addTab(self.chain_overview_widget, "Chain Overview")
        
        # Create Evidence Network tab
        self.evidence_network = EvidenceNetworkView()
        self.tab_widget.addTab(self.evidence_network, "Evidence Network")
        
        # Add tab widget to main layout
        self.layout.addWidget(self.tab_widget)

    def create_chain_overview_tab(self):
        """Creates the chain overview tab UI"""
        self.chain_layout = QVBoxLayout(self.chain_overview_widget)
        
        # Create main splitter
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - Chain Overview controls
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        
        # Chain status
        self.status_group = QGroupBox("Blockchain Status")
        status_layout = QVBoxLayout()
        self.chain_length_label = QLabel("Chain Length: 1")
        self.chain_valid_label = QLabel("Chain Valid: Yes")
        status_layout.addWidget(self.chain_length_label)
        status_layout.addWidget(self.chain_valid_label)
        self.status_group.setLayout(status_layout)
        
        # Search/Filter controls
        self.filter_group = QGroupBox("Search & Filter")
        filter_layout = QVBoxLayout()
        
        # Filter type selector
        self.filter_type = QComboBox()
        self.filter_type.addItems([
            "Population Indicators",
            "Data Source Name",
            "Intervention Text",
            "Protocol Section Title",
            "Statistical Test",
            "Literature Review Section"
        ])
        
        self.filter_input = QLineEdit()
        self.filter_button = QPushButton("Apply Filter")
        self.filter_button.clicked.connect(self.apply_filter)
        
        filter_layout.addWidget(QLabel("Filter Type:"))
        filter_layout.addWidget(self.filter_type)
        filter_layout.addWidget(QLabel("Filter Value:"))
        filter_layout.addWidget(self.filter_input)
        filter_layout.addWidget(self.filter_button)
        self.filter_group.setLayout(filter_layout)
        
        # Add sample data button
        self.add_sample_button = QPushButton("Add Sample Block")
        self.add_sample_button.clicked.connect(self.add_sample_data)
        
        # File operations
        self.file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout()
        
        self.save_button = QPushButton("Save Blockchain")
        self.save_button.clicked.connect(self.save_blockchain_dialog)
        self.load_button = QPushButton("Load Blockchain")
        self.load_button.clicked.connect(self.load_blockchain_dialog)
        
        file_layout.addWidget(self.save_button)
        file_layout.addWidget(self.load_button)
        self.file_group.setLayout(file_layout)
        
        # Add components to left layout
        self.left_layout.addWidget(self.status_group)
        self.left_layout.addWidget(self.filter_group)
        self.left_layout.addWidget(self.add_sample_button)
        self.left_layout.addWidget(self.file_group)
        self.left_layout.addStretch()
        
        # Right side - Block Details
        self.block_details = QTextEdit()
        self.block_details.setReadOnly(True)
        
        # Add both sides to splitter
        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.block_details)
        self.splitter.setSizes([300, 700])  # Set initial sizes
        
        # Add splitter to chain layout
        self.chain_layout.addWidget(self.splitter)
        
        # Initialize with genesis block
        self.update_chain_status()

    def update_chain_status(self):
        """Updates the blockchain status display"""
        chain_length = len(self.blockchain.chain)
        is_valid = self.blockchain.is_chain_valid()
        
        self.chain_length_label.setText(f"Chain Length: {chain_length}")
        self.chain_valid_label.setText(f"Chain Valid: {'Yes' if is_valid else 'No'}")
        
        # Update block details with all blocks
        details = "Blockchain Contents:\n\n"
        for block in self.blockchain.chain:
            details += f"Block {block.index}:\n"
            details += f"Timestamp: {datetime.fromtimestamp(block.timestamp)}\n"
            details += f"Hash: {block.hash[:16]}...\n"
            if block.index > 0:  # Skip genesis block evidence details
                evidence = block.evidence
                details += f"Evidence ID: {evidence.get('evidence_id', 'N/A')}\n"
                details += f"Study ID: {evidence.get('study_id', 'N/A')}\n"
                details += f"Title: {evidence.get('title', 'N/A')}\n"
                details += f"Intervention: {evidence.get('intervention_text', 'N/A')[:100]}...\n"
                details += "Claims: "
                if 'evidence_claims' in evidence and evidence['evidence_claims']:
                    details += f"{len(evidence['evidence_claims'])} claims\n"
                else:
                    details += "None\n"
                
                details += "Components: "
                if 'study_components' in evidence and evidence['study_components']:
                    component_types = [c.get('component_type', 'unknown') for c in evidence['study_components'] 
                                      if isinstance(c, dict)]
                    details += f"{', '.join(component_types)}\n"
                else:
                    details += "None\n"
            details += "-" * 40 + "\n"
        
        self.block_details.setText(details)
        
        # Also update the evidence network if it exists
        if hasattr(self, 'evidence_network'):
            self.evidence_network.load_blockchain(self.blockchain)

    def add_sample_data(self):
        """Adds a sample block to the blockchain"""
        try:
            self.blockchain = self.blockchain.add_sample_block()
            self.update_chain_status()
            QMessageBox.information(self, "Success", "Sample block added successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add sample block: {str(e)}")

    def apply_filter(self):
        """Applies the selected filter to the blockchain"""
        filter_type = self.filter_type.currentText()
        filter_value = self.filter_input.text()
        
        if not filter_value:
            QMessageBox.warning(self, "Warning", "Please enter a filter value")
            return
            
        try:
            filtered_blocks = []
            
            if filter_type == "Population Indicators":
                # Example: expect JSON format like {"age_group": "adults"}
                try:
                    indicators = json.loads(filter_value)
                    filtered_blocks = self.blockchain.get_studies_by_population_indicators(indicators)
                except json.JSONDecodeError:
                    QMessageBox.warning(self, "Warning", "Invalid JSON format. Use format: {\"key\": \"value\"}")
                    return
            
            elif filter_type == "Data Source Name":
                filtered_blocks = self.blockchain.get_studies_by_data_source_name(filter_value)
            
            elif filter_type == "Intervention Text":
                filtered_blocks = self.blockchain.get_studies_by_intervention_text(filter_value)
            
            elif filter_type == "Protocol Section Title":
                filtered_blocks = self.blockchain.get_studies_by_protocol_section_title(filter_value)
            
            elif filter_type == "Statistical Test":
                filtered_blocks = self.blockchain.get_studies_by_statistical_test_name(filter_value)
            
            elif filter_type == "Literature Review Section":
                filtered_blocks = self.blockchain.get_studies_by_literature_review_thematic_section(filter_value)
            
            # Display filtered results
            details = f"Filter Results for {filter_type}: {filter_value}\n\n"
            if filtered_blocks:
                for block in filtered_blocks:
                    details += f"Block {block.index}:\n"
                    details += f"Timestamp: {datetime.fromtimestamp(block.timestamp)}\n"
                    details += f"Hash: {block.hash[:16]}...\n"
                    evidence = block.evidence
                    details += f"Title: {evidence.get('title', 'N/A')}\n"
                    details += f"Intervention: {evidence.get('intervention_text', 'N/A')[:100]}...\n"
                    details += "-" * 40 + "\n"
            else:
                details += "No matching blocks found.\n"
            
            self.block_details.setText(details)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Filter error: {str(e)}")

    def save_blockchain_dialog(self):
        """Open dialog to save blockchain to file"""
        try:
            from PyQt6.QtWidgets import QFileDialog
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Blockchain", "", "JSON Files (*.json);;All Files (*)"
            )
            if filename:
                self.save_blockchain(filename)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error opening save dialog: {str(e)}")

    def load_blockchain_dialog(self):
        """Open dialog to load blockchain from file"""
        try:
            from PyQt6.QtWidgets import QFileDialog
            filename, _ = QFileDialog.getOpenFileName(
                self, "Load Blockchain", "", "JSON Files (*.json);;All Files (*)"
            )
            if filename:
                self.load_blockchain(filename)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error opening load dialog: {str(e)}")

    def save_blockchain(self, filename):
        try:
            self.blockchain.save_blockchain(filename)
            logging.info(f"Blockchain saved to {filename}")
            QMessageBox.information(self, "Success", f"Blockchain saved to: {filename}")
        except Exception as e:
            logging.error(f"Error saving blockchain: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save blockchain: {e}")

    def load_blockchain(self, filename):
        try:
            self.blockchain = Blockchain.load_blockchain(filename)
            self.update_chain_status()
            
            # Update the evidence network visualization
            self.evidence_network.load_blockchain(self.blockchain)
            
            logging.info(f"Blockchain loaded from {filename}")
            QMessageBox.information(self, "Success", f"Blockchain loaded from: {filename}")
        except Exception as e:
            logging.error(f"Error loading blockchain: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load blockchain: {e}")
