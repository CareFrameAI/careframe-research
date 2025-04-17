from PyQt6.QtWidgets import (
    QVBoxLayout,
    QPushButton, QWidget, QDialog, 
    QLabel, QLineEdit, QHBoxLayout, QTextEdit, QComboBox, QDialogButtonBox, QTableWidget, QTableWidgetItem, 
    QHeaderView, QAbstractItemView, QSpinBox, QScrollArea, QListWidget, QGroupBox
    
)

class TargetPopulationDialog(QDialog):
    """Dialog for configuring a target population node"""
    
    def __init__(self, parent=None, node=None):
        super().__init__(parent)
        self.node = node
        
        # Setup UI
        self.setWindowTitle("Configure Target Population")
        self.resize(600, 400)
        
        layout = QVBoxLayout(self)
        
        # Group name and description
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Group Name:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Target Population")
        name_layout.addWidget(self.name_input)
        layout.addLayout(name_layout)
        
        desc_layout = QVBoxLayout()
        desc_layout.addWidget(QLabel("Description:"))
        self.desc_input = QTextEdit()
        self.desc_input.setPlaceholderText("Describe the target population...")
        desc_layout.addWidget(self.desc_input)
        layout.addLayout(desc_layout)
        
        # Inclusion/exclusion criteria section
        layout.addWidget(QLabel("Inclusion Criteria:"))
        
        self.inclusion_layout = QVBoxLayout()
        inclusion_scroll = QScrollArea()
        inclusion_scroll.setWidgetResizable(True)
        inclusion_container = QWidget()
        inclusion_container.setLayout(self.inclusion_layout)
        inclusion_scroll.setWidget(inclusion_container)
        inclusion_scroll.setMinimumHeight(100)
        layout.addWidget(inclusion_scroll)
        
        add_inclusion_btn = QPushButton("Add Inclusion Criterion")
        add_inclusion_btn.clicked.connect(self.add_inclusion)
        layout.addWidget(add_inclusion_btn)
        
        layout.addWidget(QLabel("Exclusion Criteria:"))
        
        self.exclusion_layout = QVBoxLayout()
        exclusion_scroll = QScrollArea()
        exclusion_scroll.setWidgetResizable(True)
        exclusion_container = QWidget()
        exclusion_container.setLayout(self.exclusion_layout)
        exclusion_scroll.setWidget(exclusion_container)
        exclusion_scroll.setMinimumHeight(100)
        layout.addWidget(exclusion_scroll)
        
        add_exclusion_btn = QPushButton("Add Exclusion Criterion")
        add_exclusion_btn.clicked.connect(self.add_exclusion)
        layout.addWidget(add_exclusion_btn)
        
        # Additional notes
        layout.addWidget(QLabel("Notes:"))
        self.notes_input = QTextEdit()
        self.notes_input.setPlaceholderText("Any additional notes about this patient group...")
        layout.addWidget(self.notes_input)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Load data if node is provided
        if self.node:
            if hasattr(self.node, 'display_name'):
                self.name_input.setText(self.node.display_name)
            
            if hasattr(self.node, 'config_details'):
                self.notes_input.setPlainText(self.node.config_details)
            
            # Load custom data if available
            if hasattr(self.node, 'custom_data'):
                if 'inclusion_criteria' in self.node.custom_data:
                    for criterion in self.node.custom_data['inclusion_criteria']:
                        self.add_inclusion(criterion)
                
                if 'exclusion_criteria' in self.node.custom_data:
                    for criterion in self.node.custom_data['exclusion_criteria']:
                        self.add_exclusion(criterion)
    
    def add_inclusion(self, text=""):
        """Add an inclusion criterion input field"""
        criterion = QLineEdit()
        criterion.setPlaceholderText("Enter inclusion criterion...")
        criterion.setText(text)
        self.inclusion_layout.addWidget(criterion)
        
    def add_exclusion(self, text=""):
        """Add an exclusion criterion input field"""
        criterion = QLineEdit()
        criterion.setPlaceholderText("Enter exclusion criterion...")
        criterion.setText(text)
        self.exclusion_layout.addWidget(criterion)
    
    def accept(self):
        """Store the data in the node when OK is clicked"""
        if self.node:
            # Set display name
            self.node.display_name = self.name_input.text() or "Patient Group"
            
            # Set config details
            self.node.config_details = self.notes_input.toPlainText()
            
            # Create custom data if not exists
            if not hasattr(self.node, 'custom_data'):
                self.node.custom_data = {}
            
            # Store inclusion criteria
            inclusion_criteria = []
            for i in range(self.inclusion_layout.count()):
                widget = self.inclusion_layout.itemAt(i).widget()
                if isinstance(widget, QLineEdit) and widget.text().strip():
                    inclusion_criteria.append(widget.text())
            self.node.custom_data['inclusion_criteria'] = inclusion_criteria
            
            # Store exclusion criteria
            exclusion_criteria = []
            for i in range(self.exclusion_layout.count()):
                widget = self.exclusion_layout.itemAt(i).widget()
                if isinstance(widget, QLineEdit) and widget.text().strip():
                    exclusion_criteria.append(widget.text())
            self.node.custom_data['exclusion_criteria'] = exclusion_criteria
            
            # Update node appearance
            self.node.update()
        
        super().accept()

class EligiblePopulationDialog(QDialog):
    """Dialog for configuring eligibility criteria (inclusion/exclusion)"""
    
    def __init__(self, parent=None, node=None):
        super().__init__(parent)
        self.node = node
        
        # Setup UI
        self.setWindowTitle("Configure Eligible Population")
        self.resize(600, 500)
        
        layout = QVBoxLayout(self)
        
        # Description of this step
        desc_layout = QVBoxLayout()
        desc_layout.addWidget(QLabel("Description:"))
        self.desc_input = QTextEdit()
        self.desc_input.setPlaceholderText("Describe the eligible population...")
        self.desc_input.setMaximumHeight(80)
        desc_layout.addWidget(self.desc_input)
        layout.addLayout(desc_layout)
        
        # Inclusion/exclusion criteria section
        layout.addWidget(QLabel("Inclusion Criteria:"))
        
        self.inclusion_layout = QVBoxLayout()
        inclusion_scroll = QScrollArea()
        inclusion_scroll.setWidgetResizable(True)
        inclusion_container = QWidget()
        inclusion_container.setLayout(self.inclusion_layout)
        inclusion_scroll.setWidget(inclusion_container)
        inclusion_scroll.setMinimumHeight(150)
        layout.addWidget(inclusion_scroll)
        
        add_inclusion_btn = QPushButton("Add Inclusion Criterion")
        add_inclusion_btn.clicked.connect(self.add_inclusion)
        layout.addWidget(add_inclusion_btn)
        
        layout.addWidget(QLabel("Exclusion Criteria:"))
        
        self.exclusion_layout = QVBoxLayout()
        exclusion_scroll = QScrollArea()
        exclusion_scroll.setWidgetResizable(True)
        exclusion_container = QWidget()
        exclusion_container.setLayout(self.exclusion_layout)
        exclusion_scroll.setWidget(exclusion_container)
        exclusion_scroll.setMinimumHeight(150)
        layout.addWidget(exclusion_scroll)
        
        add_exclusion_btn = QPushButton("Add Exclusion Criterion")
        add_exclusion_btn.clicked.connect(self.add_exclusion)
        layout.addWidget(add_exclusion_btn)
        
        # Additional notes
        layout.addWidget(QLabel("Notes:"))
        self.notes_input = QTextEdit()
        self.notes_input.setPlaceholderText("Any additional notes about eligible population...")
        layout.addWidget(self.notes_input)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Load data if node is provided
        if self.node:
            if hasattr(self.node, 'display_name'):
                self.desc_input.setPlainText(self.node.display_name)
            
            if hasattr(self.node, 'config_details'):
                self.notes_input.setPlainText(self.node.config_details)
            
            # Load custom data if available
            if hasattr(self.node, 'custom_data'):
                if 'inclusion_criteria' in self.node.custom_data:
                    for criterion in self.node.custom_data['inclusion_criteria']:
                        self.add_inclusion(criterion)
                
                if 'exclusion_criteria' in self.node.custom_data:
                    for criterion in self.node.custom_data['exclusion_criteria']:
                        self.add_exclusion(criterion)
        else:
            # Add some default empty criteria
            self.add_inclusion()
            self.add_exclusion()
    
    def add_inclusion(self, text=""):
        """Add an inclusion criterion input field"""
        criterion_layout = QHBoxLayout()
        
        criterion = QLineEdit()
        criterion.setPlaceholderText("Enter inclusion criterion...")
        criterion.setText(text)
        criterion_layout.addWidget(criterion)
        
        remove_btn = QPushButton("×")
        remove_btn.setMaximumWidth(30)
        remove_btn.clicked.connect(lambda: self.remove_criterion(criterion_layout))
        criterion_layout.addWidget(remove_btn)
        
        self.inclusion_layout.addLayout(criterion_layout)
        
    def add_exclusion(self, text=""):
        """Add an exclusion criterion input field"""
        criterion_layout = QHBoxLayout()
        
        criterion = QLineEdit()
        criterion.setPlaceholderText("Enter exclusion criterion...")
        criterion.setText(text)
        criterion_layout.addWidget(criterion)
        
        remove_btn = QPushButton("×")
        remove_btn.setMaximumWidth(30)
        remove_btn.clicked.connect(lambda: self.remove_criterion(criterion_layout))
        criterion_layout.addWidget(remove_btn)
        
        self.exclusion_layout.addLayout(criterion_layout)
    
    def remove_criterion(self, layout):
        """Remove a criterion layout"""
        # Remove all widgets from layout
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            else:
                # Might be another layout
                sub_layout = item.layout()
                if sub_layout:
                    self.remove_criterion(sub_layout)
        
        # Remove layout itself
        if layout.parent():
            layout.parent().removeItem(layout)
    
    def accept(self):
        """Store the data in the node when OK is clicked"""
        if self.node:
            # Set display name
            self.node.display_name = "Eligibility Criteria"
            
            # Set config details
            self.node.config_details = self.notes_input.toPlainText()
            
            # Create custom data if not exists
            if not hasattr(self.node, 'custom_data'):
                self.node.custom_data = {}
            
            # Store description
            self.node.custom_data['description'] = self.desc_input.toPlainText()
            
            # Store inclusion criteria
            inclusion_criteria = []
            for i in range(self.inclusion_layout.count()):
                layout_item = self.inclusion_layout.itemAt(i)
                if layout_item and layout_item.layout():
                    criterion_layout = layout_item.layout()
                    for j in range(criterion_layout.count()):
                        widget_item = criterion_layout.itemAt(j)
                        if widget_item and isinstance(widget_item.widget(), QLineEdit) and widget_item.widget().text().strip():
                            inclusion_criteria.append(widget_item.widget().text())
                            break
            self.node.custom_data['inclusion_criteria'] = inclusion_criteria
            
            # Store exclusion criteria
            exclusion_criteria = []
            for i in range(self.exclusion_layout.count()):
                layout_item = self.exclusion_layout.itemAt(i)
                if layout_item and layout_item.layout():
                    criterion_layout = layout_item.layout()
                    for j in range(criterion_layout.count()):
                        widget_item = criterion_layout.itemAt(j)
                        if widget_item and isinstance(widget_item.widget(), QLineEdit) and widget_item.widget().text().strip():
                            exclusion_criteria.append(widget_item.widget().text())
                            break
            self.node.custom_data['exclusion_criteria'] = exclusion_criteria
            
            # Update node appearance
            self.node.update()
        
        super().accept()

class TimePointDialog(QDialog):
    """Dialog for configuring measurement timepoints"""
    
    def __init__(self, parent=None, node=None):
        super().__init__(parent)
        self.node = node
        
        # Setup UI
        self.setWindowTitle("Configure Measurement Timepoint")
        self.resize(500, 400)
        
        layout = QVBoxLayout(self)
        
        # Timepoint name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Timepoint Name:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., Baseline, Week 4, Follow-up")
        name_layout.addWidget(self.name_input)
        layout.addLayout(name_layout)
        
        # Timing information
        timing_group = QGroupBox("Timing")
        timing_layout = QVBoxLayout()
        
        # Time unit
        time_unit_layout = QHBoxLayout()
        time_unit_layout.addWidget(QLabel("Time Unit:"))
        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItems(["Days", "Weeks", "Months", "Years"])
        time_unit_layout.addWidget(self.time_unit_combo)
        timing_layout.addLayout(time_unit_layout)
        
        # Time value
        time_value_layout = QHBoxLayout()
        time_value_layout.addWidget(QLabel("Time Value:"))
        self.time_value_spin = QSpinBox()
        self.time_value_spin.setRange(0, 10000)
        time_value_layout.addWidget(self.time_value_spin)
        timing_layout.addLayout(time_value_layout)
        
        # Reference point
        ref_point_layout = QHBoxLayout()
        ref_point_layout.addWidget(QLabel("Reference Point:"))
        self.ref_point_combo = QComboBox()
        self.ref_point_combo.addItems(["From Baseline", "From Randomization", "From Intervention Start", "From Study Start"])
        ref_point_layout.addWidget(self.ref_point_combo)
        timing_layout.addLayout(ref_point_layout)
        
        timing_group.setLayout(timing_layout)
        layout.addWidget(timing_group)
        
        # Measurements to collect
        layout.addWidget(QLabel("Measurements to collect at this timepoint:"))
        self.measurements_list = QListWidget()
        layout.addWidget(self.measurements_list)
        
        # Add measurement input
        measurement_layout = QHBoxLayout()
        self.measurement_input = QLineEdit()
        self.measurement_input.setPlaceholderText("Enter measurement name...")
        measurement_layout.addWidget(self.measurement_input)
        
        add_measurement_btn = QPushButton("Add")
        add_measurement_btn.clicked.connect(self.add_measurement)
        measurement_layout.addWidget(add_measurement_btn)
        
        layout.addLayout(measurement_layout)
        
        # Additional notes
        layout.addWidget(QLabel("Notes:"))
        self.notes_input = QTextEdit()
        self.notes_input.setPlaceholderText("Any additional notes about this timepoint...")
        self.notes_input.setMaximumHeight(100)
        layout.addWidget(self.notes_input)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Load data if node is provided
        if self.node:
            if hasattr(self.node, 'custom_data'):
                if 'name' in self.node.custom_data:
                    self.name_input.setText(self.node.custom_data['name'])
                
                if 'time_unit' in self.node.custom_data:
                    index = self.time_unit_combo.findText(self.node.custom_data['time_unit'])
                    if index >= 0:
                        self.time_unit_combo.setCurrentIndex(index)
                
                if 'time_value' in self.node.custom_data:
                    self.time_value_spin.setValue(self.node.custom_data['time_value'])
                
                if 'reference_point' in self.node.custom_data:
                    index = self.ref_point_combo.findText(self.node.custom_data['reference_point'])
                    if index >= 0:
                        self.ref_point_combo.setCurrentIndex(index)
                
                if 'measurements' in self.node.custom_data:
                    for measurement in self.node.custom_data['measurements']:
                        self.measurements_list.addItem(measurement)
                
                if 'notes' in self.node.custom_data:
                    self.notes_input.setPlainText(self.node.custom_data['notes'])
    
    def add_measurement(self):
        """Add a measurement to collect at this timepoint"""
        text = self.measurement_input.text().strip()
        if text:
            self.measurements_list.addItem(text)
            self.measurement_input.clear()
    
    def accept(self):
        """Store the data in the node when OK is clicked"""
        if self.node:
            # Set display name based on timepoint name
            timepoint_name = self.name_input.text().strip() or "Timepoint"
            self.node.display_name = timepoint_name
            
            # Create custom data if not exists
            if not hasattr(self.node, 'custom_data'):
                self.node.custom_data = {}
            
            # Store all configuration
            self.node.custom_data['name'] = timepoint_name
            self.node.custom_data['time_unit'] = self.time_unit_combo.currentText()
            self.node.custom_data['time_value'] = self.time_value_spin.value()
            self.node.custom_data['reference_point'] = self.ref_point_combo.currentText()
            
            # Store measurements
            measurements = []
            for i in range(self.measurements_list.count()):
                measurements.append(self.measurements_list.item(i).text())
            self.node.custom_data['measurements'] = measurements
            
            # Store notes
            self.node.custom_data['notes'] = self.notes_input.toPlainText()
            
            # Update node appearance
            self.node.update()
        
        super().accept()

class InterventionDialog(QDialog):
    """Dialog for configuring interventions."""
    
    def __init__(self, parent=None, node=None):
        super().__init__(parent)
        self.node = node
        
        self.setWindowTitle("Intervention Configuration")
        self.resize(500, 400)
        
        layout = QVBoxLayout()
        
        # Info label
        info_label = QLabel("Configure this intervention in your study model")
        info_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(info_label)
        
        # Intervention name
        name_layout = QHBoxLayout()
        name_label = QLabel("Intervention Name:")
        name_layout.addWidget(name_label)
        
        self.name_input = QLineEdit()
        if node and hasattr(node, 'display_name') and node.display_name:
            self.name_input.setText(node.display_name)
        else:
            self.name_input.setPlaceholderText("e.g., Drug A, Surgery, Behavioral Therapy")
        name_layout.addWidget(self.name_input)
        
        layout.addLayout(name_layout)
        
        # Intervention description
        description_label = QLabel("Description:")
        layout.addWidget(description_label)
        
        self.description_input = QTextEdit()
        if node and hasattr(node, 'config_details') and node.config_details:
            self.description_input.setText(node.config_details)
        else:
            self.description_input.setPlaceholderText("Describe the intervention details")
        layout.addWidget(self.description_input)
        
        # Intervention type
        type_layout = QHBoxLayout()
        type_label = QLabel("Type:")
        type_layout.addWidget(type_label)
        
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Drug", "Device", "Procedure", "Behavioral", "Other"])
        type_layout.addWidget(self.type_combo)
        
        layout.addLayout(type_layout)
        
        # Intervention dose/schedule
        dosage_label = QLabel("Dosage/Schedule:")
        layout.addWidget(dosage_label)
        
        self.dosage_input = QLineEdit()
        self.dosage_input.setPlaceholderText("e.g., 10mg daily, twice weekly sessions")
        layout.addWidget(self.dosage_input)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
        # Load existing data if present
        if node and hasattr(node, 'custom_data'):
            if 'type' in node.custom_data:
                index = self.type_combo.findText(node.custom_data['type'])
                if index >= 0:
                    self.type_combo.setCurrentIndex(index)
            if 'dosage' in node.custom_data:
                self.dosage_input.setText(node.custom_data['dosage'])
    
    def accept(self):
        """Handle dialog acceptance"""
        # Store the data for the node
        if self.node:
            self.node.display_name = self.name_input.text()
            self.node.config_details = self.description_input.toPlainText()
            
            # Store additional details in custom data
            self.node.custom_data['type'] = self.type_combo.currentText()
            self.node.custom_data['dosage'] = self.dosage_input.text()
            
            # Update node appearance
            self.node.update()
        
        super().accept()

class OutcomeDialog(QDialog):
    """Dialog for configuring outcomes."""
    
    def __init__(self, parent=None, node=None):
        super().__init__(parent)
        self.node = node
        
        self.setWindowTitle("Outcome Configuration")
        self.resize(500, 400)
        
        layout = QVBoxLayout()
        
        # Info label
        info_label = QLabel("Configure this outcome measure in your study model")
        info_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(info_label)
        
        # Outcome name
        name_layout = QHBoxLayout()
        name_label = QLabel("Outcome Name:")
        name_layout.addWidget(name_label)
        
        self.name_input = QLineEdit()
        if node and hasattr(node, 'display_name') and node.display_name:
            self.name_input.setText(node.display_name)
        else:
            self.name_input.setPlaceholderText("e.g., Mortality, HbA1c, Quality of Life")
        name_layout.addWidget(self.name_input)
        
        layout.addLayout(name_layout)
        
        # Outcome description
        description_label = QLabel("Description:")
        layout.addWidget(description_label)
        
        self.description_input = QTextEdit()
        if node and hasattr(node, 'config_details') and node.config_details:
            self.description_input.setText(node.config_details)
        else:
            self.description_input.setPlaceholderText("Describe the outcome measure details")
        layout.addWidget(self.description_input)
        
        # Outcome type
        type_layout = QHBoxLayout()
        type_label = QLabel("Type:")
        type_layout.addWidget(type_label)
        
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Primary", "Secondary", "Safety", "Exploratory"])
        type_layout.addWidget(self.type_combo)
        
        layout.addLayout(type_layout)
        
        # Outcome measurement method
        method_label = QLabel("Measurement Method:")
        layout.addWidget(method_label)
        
        self.method_input = QLineEdit()
        self.method_input.setPlaceholderText("e.g., Blood test, Survey, Clinical assessment")
        layout.addWidget(self.method_input)
        
        # Time of measurement
        time_layout = QHBoxLayout()
        time_label = QLabel("Time of Measurement:")
        time_layout.addWidget(time_label)
        
        self.time_combo = QComboBox()
        self.time_combo.addItems([
            "At baseline",
            "During intervention",
            "Post-intervention",
            "During follow-up",
            "Multiple timepoints"
        ])
        time_layout.addWidget(self.time_combo)
        
        layout.addLayout(time_layout)
        
        # Statistical analysis
        analysis_label = QLabel("Statistical Analysis:")
        layout.addWidget(analysis_label)
        
        self.analysis_input = QLineEdit()
        self.analysis_input.setPlaceholderText("e.g., t-test, ANOVA, Cox regression")
        layout.addWidget(self.analysis_input)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
        # Load existing data if present
        if node and hasattr(node, 'custom_data'):
            if 'type' in node.custom_data:
                index = self.type_combo.findText(node.custom_data['type'])
                if index >= 0:
                    self.type_combo.setCurrentIndex(index)
            if 'method' in node.custom_data:
                self.method_input.setText(node.custom_data['method'])
            if 'time' in node.custom_data:
                index = self.time_combo.findText(node.custom_data['time'])
                if index >= 0:
                    self.time_combo.setCurrentIndex(index)
            if 'analysis' in node.custom_data:
                self.analysis_input.setText(node.custom_data['analysis'])
    
    def accept(self):
        """Handle dialog acceptance"""
        # Store the data for the node
        if self.node:
            self.node.display_name = self.name_input.text()
            self.node.config_details = self.description_input.toPlainText()
            
            # Store additional details in custom data
            self.node.custom_data['type'] = self.type_combo.currentText()
            self.node.custom_data['method'] = self.method_input.text()
            self.node.custom_data['time'] = self.time_combo.currentText()
            self.node.custom_data['analysis'] = self.analysis_input.text()
            
            # Update node appearance
            self.node.update()
        
        super().accept()

class RandomizationDialog(QDialog):
    """Dialog for configuring randomization."""
    
    def __init__(self, parent=None, node=None):
        super().__init__(parent)
        self.node = node
        
        self.setWindowTitle("Randomization Configuration")
        self.resize(600, 500)
        
        layout = QVBoxLayout()
        
        # Info label
        info_label = QLabel("Configure the randomization for your study model")
        info_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(info_label)
        
        # Randomization method
        method_layout = QHBoxLayout()
        method_label = QLabel("Randomization Method:")
        method_layout.addWidget(method_label)
        
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Simple", 
            "Block", 
            "Stratified", 
            "Cluster", 
            "Minimization", 
            "Adaptive"
        ])
        if node and hasattr(node, 'custom_data') and 'method' in node.custom_data:
            index = self.method_combo.findText(node.custom_data['method'])
            if index >= 0:
                self.method_combo.setCurrentIndex(index)
        method_layout.addWidget(self.method_combo)
        
        layout.addLayout(method_layout)
        
        # Allocation ratio
        ratio_layout = QHBoxLayout()
        ratio_label = QLabel("Allocation Ratio:")
        ratio_layout.addWidget(ratio_label)
        
        self.ratio_input = QLineEdit()
        if node and hasattr(node, 'custom_data') and 'ratio' in node.custom_data:
            self.ratio_input.setText(node.custom_data['ratio'])
        else:
            self.ratio_input.setPlaceholderText("e.g., 1:1, 2:1")
        ratio_layout.addWidget(self.ratio_input)
        
        layout.addLayout(ratio_layout)
        
        # Arms/groups
        arms_label = QLabel("Treatment Arms/Groups:")
        layout.addWidget(arms_label)
        
        self.arms_list = QTableWidget()
        self.arms_list.setColumnCount(2)
        self.arms_list.setHorizontalHeaderLabels(["Name", "Description"])
        self.arms_list.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.arms_list.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self.arms_list)
        
        # Load existing arms if available
        if node and hasattr(node, 'custom_data') and 'arms' in node.custom_data:
            arms = node.custom_data['arms']
            self.arms_list.setRowCount(len(arms))
            for i, arm in enumerate(arms):
                self.arms_list.setItem(i, 0, QTableWidgetItem(arm.get('name', f"Group {i+1}")))
                self.arms_list.setItem(i, 1, QTableWidgetItem(arm.get('description', "")))
        else:
            # Add default rows
            self.arms_list.setRowCount(2)
            self.arms_list.setItem(0, 0, QTableWidgetItem("Group 1"))
            self.arms_list.setItem(0, 1, QTableWidgetItem(""))
            self.arms_list.setItem(1, 0, QTableWidgetItem("Group 2"))
            self.arms_list.setItem(1, 1, QTableWidgetItem(""))
        
        # Buttons for arms
        arms_buttons_layout = QHBoxLayout()
        add_arm_btn = QPushButton("Add Arm")
        add_arm_btn.clicked.connect(self.add_arm)
        arms_buttons_layout.addWidget(add_arm_btn)
        
        remove_arm_btn = QPushButton("Remove Selected Arm")
        remove_arm_btn.clicked.connect(self.remove_arm)
        arms_buttons_layout.addWidget(remove_arm_btn)
        
        layout.addLayout(arms_buttons_layout)
        
        # Stratification factors
        strat_label = QLabel("Stratification Factors:")
        layout.addWidget(strat_label)
        
        self.strat_list = QListWidget()
        layout.addWidget(self.strat_list)
        
        # Load existing stratification factors if available
        if node and hasattr(node, 'custom_data') and 'stratification' in node.custom_data:
            for factor in node.custom_data['stratification']:
                self.strat_list.addItem(factor)
        
        # Stratification input
        strat_input_layout = QHBoxLayout()
        self.strat_input = QLineEdit()
        self.strat_input.setPlaceholderText("Add stratification factor")
        strat_input_layout.addWidget(self.strat_input)
        
        add_strat_btn = QPushButton("Add")
        add_strat_btn.clicked.connect(self.add_stratification)
        strat_input_layout.addWidget(add_strat_btn)
        
        layout.addLayout(strat_input_layout)
        
        # Notes
        notes_label = QLabel("Additional Notes:")
        layout.addWidget(notes_label)
        
        self.notes_input = QTextEdit()
        if node and hasattr(node, 'config_details') and node.config_details:
            self.notes_input.setText(node.config_details)
        layout.addWidget(self.notes_input)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def add_arm(self):
        """Add a new arm to the table"""
        row_count = self.arms_list.rowCount()
        self.arms_list.insertRow(row_count)
        self.arms_list.setItem(row_count, 0, QTableWidgetItem(f"Group {row_count+1}"))
        self.arms_list.setItem(row_count, 1, QTableWidgetItem(""))
    
    def remove_arm(self):
        """Remove the selected arm"""
        selected_items = self.arms_list.selectedItems()
        if selected_items and self.arms_list.rowCount() > 2:  # Keep at least 2 arms
            row = selected_items[0].row()
            self.arms_list.removeRow(row)
        
    def add_stratification(self):
        """Add a stratification factor"""
        text = self.strat_input.text().strip()
        if text:
            self.strat_list.addItem(text)
            self.strat_input.clear()
    
    def accept(self):
        # Store the data for the node
        if self.node:
            self.node.display_name = f"Randomization ({self.method_combo.currentText()})"
            self.node.config_details = self.notes_input.toPlainText()
            
            # Store additional details in custom data
            self.node.custom_data['method'] = self.method_combo.currentText()
            self.node.custom_data['ratio'] = self.ratio_input.text()
            
            # Store arms/groups
            arms = []
            for row in range(self.arms_list.rowCount()):
                name_item = self.arms_list.item(row, 0)
                desc_item = self.arms_list.item(row, 1)
                
                name = name_item.text() if name_item else f"Group {row+1}"
                description = desc_item.text() if desc_item else ""
                
                arms.append({
                    'name': name,
                    'description': description
                })
                
            self.node.custom_data['arms'] = arms
            
            # Store stratification factors
            stratification = []
            for i in range(self.strat_list.count()):
                stratification.append(self.strat_list.item(i).text())
                
            self.node.custom_data['stratification'] = stratification
            
            # Update node appearance
            self.node.update()
        
        super().accept()



class StudyDesignTypeDialog(QDialog):
    """Dialog for selecting study design type when creating a new workflow"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Study Design Type")
        self.resize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # Info label
        info_label = QLabel("Select the type of study design you want to create:")
        info_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(info_label)
        
        # Design type options as radio buttons
        self.rct_radio = QPushButton("Randomized Controlled Trial (RCT)")
        self.rct_radio.setCheckable(True)
        self.rct_radio.setChecked(True)
        self.rct_radio.clicked.connect(self.show_rct_info)
        layout.addWidget(self.rct_radio)
        
        self.prepost_radio = QPushButton("Pre-Post Design")
        self.prepost_radio.setCheckable(True)
        self.prepost_radio.clicked.connect(self.show_prepost_info)
        layout.addWidget(self.prepost_radio)
        
        self.crossover_radio = QPushButton("Crossover Design")
        self.crossover_radio.setCheckable(True)
        self.crossover_radio.clicked.connect(self.show_crossover_info)
        layout.addWidget(self.crossover_radio)
        
        # Info text area
        layout.addSpacing(15)
        self.info_label = QLabel("Design Info:")
        layout.addWidget(self.info_label)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        layout.addWidget(self.info_text)
        
        # Show initial info for RCT
        self.show_rct_info()
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def show_rct_info(self):
        """Show information about RCT design"""
        self.rct_radio.setChecked(True)
        self.prepost_radio.setChecked(False)
        self.crossover_radio.setChecked(False)
        
        info = """
        <b>Randomized Controlled Trial</b>
        
        In this design:
        • Participants are randomly assigned to groups
        • One or more groups receive the intervention
        • One or more groups serve as controls
        • Outcomes are compared between groups
        
        <b>Typical Flow:</b>
        Patient Group → Eligibility → Randomization → Intervention/Control → Outcome
        """
        self.info_text.setHtml(info)
    
    def show_prepost_info(self):
        """Show information about Pre-Post design"""
        self.rct_radio.setChecked(False)
        self.prepost_radio.setChecked(True)
        self.crossover_radio.setChecked(False)
        
        info = """
        <b>Pre-Post Design</b>
        
        In this design:
        • All participants receive the same intervention
        • Measurements are taken before and after intervention
        • Each participant serves as their own control
        • Pre-intervention vs post-intervention outcomes are compared
        
        <b>Typical Flow:</b>
        Patient Group → Eligibility → Baseline Measurement → Intervention → Follow-up Measurement
        """
        self.info_text.setHtml(info)
    
    def show_crossover_info(self):
        """Show information about Crossover design"""
        self.rct_radio.setChecked(False)
        self.prepost_radio.setChecked(False)
        self.crossover_radio.setChecked(True)
        
        info = """
        <b>Crossover Design</b>
        
        In this design:
        • Participants receive multiple interventions in sequence
        • Each participant receives all interventions
        • Order of interventions is often randomized
        • "Washout" periods may separate interventions
        
        <b>Typical Flow:</b>
        Patient Group → Eligibility → Randomize Sequence → Intervention Sequence → Multiple Timepoints
        """
        self.info_text.setHtml(info)
    
    def get_design_type(self):
        """Return the selected design type"""
        if self.rct_radio.isChecked():
            return "rct"
        elif self.prepost_radio.isChecked():
            return "prepost"
        elif self.crossover_radio.isChecked():
            return "crossover"
        return "rct"  # Default to RCT if none selected
