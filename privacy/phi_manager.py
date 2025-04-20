# privacy/phi_manager.py
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, QCheckBox,
    QComboBox, QLabel, QSlider, QPushButton, QSpinBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QScrollArea
)
from PyQt6.QtCore import Qt, QSettings, pyqtSignal, QObject
import json
import os

from privacy.privacy_filter import PrivacyFilter

class PHIConfig(QObject):
    """Configuration for PHI detection and handling"""
    config_changed = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self._enabled = True
        self._block_threshold = 3  # Block calls with this many or more PHI elements
        self._replacement_format = "[{type}]"  # Format for replacement tokens
        self._regions = ["US", "CANADA"]
        self._use_presidio = True
        self._use_scrubadub = True
        self._use_regex = True  # Add regex option
        self._phi_types_to_check = [
            "NAME", "SSN", "PHONE_NUMBER", "EMAIL_ADDRESS", "ADDRESS", 
            "MEDICAL_RECORD", "HEALTH_CARD", "SIN", "DATE", "AGE", 
            "CREDIT_CARD", "IP_ADDRESS", "URL", "PASSPORT_NUMBER", "LICENSE_NUMBER",
            "INAPPROPRIATE_LANGUAGE"
        ]
        self._load_settings()
        
        # Create the privacy filter
        self._privacy_filter = PrivacyFilter(
            use_presidio=self._use_presidio,
            use_scrubadub=self._use_scrubadub,
            use_regex=self._use_regex,
            regions=self._regions
        )
    
    def _load_settings(self):
        """Load settings from QSettings"""
        settings = QSettings("CareFrame", "PrivacyFilter")
        self._enabled = settings.value("enabled", True, bool)
        self._block_threshold = settings.value("block_threshold", 3, int)
        self._replacement_format = settings.value("replacement_format", "[{type}]", str)
        
        regions = settings.value("regions", None)
        if regions:
            self._regions = json.loads(regions)
            
        self._use_presidio = settings.value("use_presidio", True, bool)
        self._use_scrubadub = settings.value("use_scrubadub", True, bool)
        self._use_regex = settings.value("use_regex", True, bool)
        
        phi_types = settings.value("phi_types_to_check", None)
        if phi_types:
            self._phi_types_to_check = json.loads(phi_types)
    
    def save_settings(self):
        """Save settings to QSettings"""
        settings = QSettings("CareFrame", "PrivacyFilter")
        settings.setValue("enabled", self._enabled)
        settings.setValue("block_threshold", self._block_threshold)
        settings.setValue("replacement_format", self._replacement_format)
        settings.setValue("regions", json.dumps(self._regions))
        settings.setValue("use_presidio", self._use_presidio)
        settings.setValue("use_scrubadub", self._use_scrubadub)
        settings.setValue("use_regex", self._use_regex)
        settings.setValue("phi_types_to_check", json.dumps(self._phi_types_to_check))
        
    def reinitialize_filter(self):
        """Reinitialize the privacy filter with current settings"""
        self._privacy_filter = PrivacyFilter(
            use_presidio=self._use_presidio,
            use_scrubadub=self._use_scrubadub,
            use_regex=self._use_regex,
            regions=self._regions
        )
    
    @property
    def enabled(self):
        return self._enabled
        
    @enabled.setter
    def enabled(self, value):
        self._enabled = bool(value)
        self.config_changed.emit()
        self.save_settings()
    
    @property
    def block_threshold(self):
        return self._block_threshold
        
    @block_threshold.setter
    def block_threshold(self, value):
        self._block_threshold = int(value)
        self.config_changed.emit()
        self.save_settings()
    
    @property
    def replacement_format(self):
        return self._replacement_format
        
    @replacement_format.setter
    def replacement_format(self, value):
        self._replacement_format = value
        self.config_changed.emit()
        self.save_settings()
    
    @property
    def regions(self):
        return self._regions
        
    @regions.setter
    def regions(self, value):
        self._regions = value
        self.reinitialize_filter()
        self.config_changed.emit()
        self.save_settings()
    
    @property
    def use_presidio(self):
        return self._use_presidio
        
    @use_presidio.setter
    def use_presidio(self, value):
        self._use_presidio = bool(value)
        self.reinitialize_filter()
        self.config_changed.emit()
        self.save_settings()
    
    @property
    def use_scrubadub(self):
        return self._use_scrubadub
        
    @use_scrubadub.setter
    def use_scrubadub(self, value):
        self._use_scrubadub = bool(value)
        self.reinitialize_filter()
        self.config_changed.emit()
        self.save_settings()
    
    @property
    def use_regex(self):
        return self._use_regex
        
    @use_regex.setter
    def use_regex(self, value):
        self._use_regex = bool(value)
        self.reinitialize_filter()
        self.config_changed.emit()
        self.save_settings()
    
    @property
    def phi_types_to_check(self):
        return self._phi_types_to_check
        
    @phi_types_to_check.setter
    def phi_types_to_check(self, value):
        self._phi_types_to_check = value
        self.config_changed.emit()
        self.save_settings()
    
    @property
    def privacy_filter(self):
        return self._privacy_filter
    
    def check_phi(self, text):
        """
        Check for PHI in text
        
        Args:
            text: Text to check for PHI
        
        Returns:
            tuple: (has_phi_above_threshold, phi_report, redacted_text)
        """
        if not self._enabled or not text:
            return False, None, text
            
        # Get PHI report
        phi_report = self._privacy_filter.get_phi_report(text)
        
        # Check if PHI count is above threshold
        phi_count = phi_report['total_phi_count']
        block_required = phi_count >= self._block_threshold
        
        # Also check for malicious content
        if phi_report.get('malicious_content', False):
            # If malicious content has a high score, block it
            if phi_report.get('malicious_score', 0) >= 4:
                block_required = True
        
        # Generate redacted text with token replacements
        redacted_text = self._privacy_filter.replace_phi(text, self._replacement_format)
        
        return block_required, phi_report, redacted_text

# Global instance
phi_config = PHIConfig()

class PHIManagerWidget(QWidget):
    """Widget for managing PHI detection settings"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._connect_signals()
        self.update_display()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Main enable/disable checkbox
        self.enable_checkbox = QCheckBox("Enable PHI detection and redaction")
        layout.addWidget(self.enable_checkbox)
        
        # Settings group
        settings_group = QGroupBox("PHI Detection Settings")
        settings_layout = QFormLayout(settings_group)
        
        # Blocking threshold
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setMinimum(1)
        self.threshold_spin.setMaximum(20)
        settings_layout.addRow("Block threshold (# of PHI elements):", self.threshold_spin)
        
        # Replacement format
        self.format_combo = QComboBox()
        self.format_combo.addItems(["[{type}]", "[REDACTED]", "[PHI]", "***", "{type}"])
        self.format_combo.setEditable(True)
        settings_layout.addRow("Replacement format:", self.format_combo)
        
        # Regions selection
        self.us_checkbox = QCheckBox("United States (HIPAA)")
        self.canada_checkbox = QCheckBox("Canada (PHIPA)")
        region_layout = QVBoxLayout()
        region_layout.addWidget(self.us_checkbox)
        region_layout.addWidget(self.canada_checkbox)
        settings_layout.addRow("Regions:", region_layout)
        
        # Detection libraries
        self.presidio_checkbox = QCheckBox("Microsoft Presidio")
        self.scrubadub_checkbox = QCheckBox("Scrubadub")
        self.regex_checkbox = QCheckBox("Custom Regex Patterns")
        engines_layout = QVBoxLayout()
        engines_layout.addWidget(self.presidio_checkbox)
        engines_layout.addWidget(self.scrubadub_checkbox)
        engines_layout.addWidget(self.regex_checkbox)
        # Add an informational label about enhanced detection
        enhanced_label = QLabel("Enhanced detection is always enabled for PHI, inappropriate language, and malicious content.")
        enhanced_label.setStyleSheet("color: darkgreen;")
        engines_layout.addWidget(enhanced_label)
        settings_layout.addRow("Detection engines:", engines_layout)
        
        layout.addWidget(settings_group)
        
        # === PHI Types selection group with scrolling ===
        types_group = QGroupBox("PHI Types to Detect")
        types_outer_layout = QVBoxLayout(types_group)
        
        # Add scrolling capability
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Scrollable widget for PHI types
        types_widget = QWidget()
        types_layout = QVBoxLayout(types_widget)
        types_layout.setSpacing(2)  # Reduce spacing to make it more compact
        
        # Add checkboxes for PHI types from both US and Canada
        self.phi_type_checkboxes = {}
        
        # Get PHI categories from PrivacyFilter
        all_phi_types = {}
        all_phi_types.update(PrivacyFilter.PHI_CATEGORIES['US'])
        all_phi_types.update(PrivacyFilter.PHI_CATEGORIES['CANADA'])
        
        # Add additional regex-based types
        regex_types = {
            "CREDIT_CARD": "Credit card number",
            "IP_ADDRESS": "IP address",
            "URL": "Web URL",
            "PASSPORT_NUMBER": "Passport number",
            "LICENSE_NUMBER": "License number"
        }
        all_phi_types.update(regex_types)
        
        # Create a checkbox for each PHI type
        for phi_type, description in all_phi_types.items():
            checkbox = QCheckBox(f"{phi_type}: {description}")
            checkbox.setProperty("phi_type", phi_type)
            self.phi_type_checkboxes[phi_type] = checkbox
            types_layout.addWidget(checkbox)
        
        # Add stretch to bottom of the scrollable area
        types_layout.addStretch()
        
        # Set up scroll area
        scroll_area.setWidget(types_widget)
        types_outer_layout.addWidget(scroll_area)
        
        # Set a reasonable fixed height for the scrollable area
        scroll_area.setMinimumHeight(150)
        scroll_area.setMaximumHeight(200)
        
        layout.addWidget(types_group)
        
        # Test section
        test_group = QGroupBox("Test PHI Detection")
        test_layout = QVBoxLayout(test_group)
        
        # Add text entry and test button
        self.test_button = QPushButton("Test Text for PHI")
        test_layout.addWidget(self.test_button)
        
        layout.addWidget(test_group)
    
    def _connect_signals(self):
        # Connect UI signals to config changes
        self.enable_checkbox.toggled.connect(self._update_enabled)
        self.threshold_spin.valueChanged.connect(self._update_threshold)
        self.format_combo.currentTextChanged.connect(self._update_format)
        self.us_checkbox.toggled.connect(self._update_regions)
        self.canada_checkbox.toggled.connect(self._update_regions)
        self.presidio_checkbox.toggled.connect(self._update_presidio)
        self.scrubadub_checkbox.toggled.connect(self._update_scrubadub)
        self.regex_checkbox.toggled.connect(self._update_regex)
        
        # Connect PHI type checkboxes
        for checkbox in self.phi_type_checkboxes.values():
            checkbox.toggled.connect(self._update_phi_types)
        
        # Connect test button
        self.test_button.clicked.connect(self._test_phi_detection)
        
        # Connect config changes to UI updates
        phi_config.config_changed.connect(self.update_display)
    
    def _update_enabled(self, enabled):
        phi_config.enabled = enabled
    
    def _update_threshold(self, value):
        phi_config.block_threshold = value
    
    def _update_format(self, format_str):
        phi_config.replacement_format = format_str
    
    def _update_regions(self):
        regions = []
        if self.us_checkbox.isChecked():
            regions.append("US")
        if self.canada_checkbox.isChecked():
            regions.append("CANADA")
        phi_config.regions = regions
    
    def _update_presidio(self, enabled):
        phi_config.use_presidio = enabled
    
    def _update_scrubadub(self, enabled):
        phi_config.use_scrubadub = enabled
    
    def _update_regex(self, enabled):
        phi_config.use_regex = enabled
    
    def _update_phi_types(self):
        selected_types = []
        for phi_type, checkbox in self.phi_type_checkboxes.items():
            if checkbox.isChecked():
                selected_types.append(phi_type)
        phi_config.phi_types_to_check = selected_types
    
    def _test_phi_detection(self):
        # Show a dialog to test PHI detection
        from PyQt6.QtWidgets import QDialog, QTextEdit, QVBoxLayout, QHBoxLayout
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Test PHI Detection")
        dialog.resize(600, 400)
        
        layout = QVBoxLayout(dialog)
        
        # Input text
        input_label = QLabel("Enter text to scan for PHI:")
        input_text = QTextEdit()
        layout.addWidget(input_label)
        layout.addWidget(input_text)
        
        # Results
        results_label = QLabel("Results:")
        results_text = QTextEdit()
        results_text.setReadOnly(True)
        layout.addWidget(results_label)
        layout.addWidget(results_text)
        
        # Buttons
        button_layout = QHBoxLayout()
        scan_button = QPushButton("Scan")
        close_button = QPushButton("Close")
        button_layout.addWidget(scan_button)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)
        
        # Connect signals
        close_button.clicked.connect(dialog.accept)
        
        def scan_text():
            text = input_text.toPlainText()
            if not text:
                results_text.setPlainText("Please enter text to scan")
                return
                
            # Scan for PHI
            block_required, phi_report, redacted_text = phi_config.check_phi(text)
            
            # Display results
            results = f"Found {phi_report['total_phi_count']} PHI elements\n"
            results += f"PHI types detected: {phi_report['phi_types']}\n"
            results += f"Compliance risk: {phi_report['compliance_risk']}\n"
            
            # Add malicious content info
            results += f"Malicious content detected: {phi_report.get('malicious_content', False)}\n"
            if phi_report.get('malicious_content', False):
                results += f"Malicious score: {phi_report.get('malicious_score', 0)}\n"
                results += "Malicious detections:\n"
                for detection in phi_report.get('malicious_detections', []):
                    results += f"  - {detection.get('type', 'Unknown')}: '{detection.get('text', '')}'\n"
            
            results += f"Block required: {block_required}\n\n"
            results += f"Redacted text:\n{redacted_text}"
            
            results_text.setPlainText(results)
        
        scan_button.clicked.connect(scan_text)
        
        # Show dialog
        dialog.exec()
    
    def update_display(self):
        # Update UI to reflect current config
        self.enable_checkbox.blockSignals(True)
        self.threshold_spin.blockSignals(True)
        self.format_combo.blockSignals(True)
        self.us_checkbox.blockSignals(True)
        self.canada_checkbox.blockSignals(True)
        self.presidio_checkbox.blockSignals(True)
        self.scrubadub_checkbox.blockSignals(True)
        self.regex_checkbox.blockSignals(True)
        
        # Set values
        self.enable_checkbox.setChecked(phi_config.enabled)
        self.threshold_spin.setValue(phi_config.block_threshold)
        self.format_combo.setEditText(phi_config.replacement_format)
        self.us_checkbox.setChecked("US" in phi_config.regions)
        self.canada_checkbox.setChecked("CANADA" in phi_config.regions)
        self.presidio_checkbox.setChecked(phi_config.use_presidio)
        self.scrubadub_checkbox.setChecked(phi_config.use_scrubadub)
        self.regex_checkbox.setChecked(phi_config.use_regex)
        
        # Update PHI type checkboxes
        for phi_type, checkbox in self.phi_type_checkboxes.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(phi_type in phi_config.phi_types_to_check)
            checkbox.blockSignals(False)
        
        self.enable_checkbox.blockSignals(False)
        self.threshold_spin.blockSignals(False)
        self.format_combo.blockSignals(False)
        self.us_checkbox.blockSignals(False)
        self.canada_checkbox.blockSignals(False)
        self.presidio_checkbox.blockSignals(False)
        self.scrubadub_checkbox.blockSignals(False)
        self.regex_checkbox.blockSignals(False) 