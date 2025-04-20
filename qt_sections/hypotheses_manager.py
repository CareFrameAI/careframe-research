from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, 
                             QTableWidgetItem, QPushButton, QLabel, QGroupBox, 
                             QSplitter, QFormLayout, QTextEdit, QComboBox,
                             QDialog, QListWidget, QListWidgetItem,
                             QGridLayout, QCheckBox, QMessageBox,
                             QHeaderView, QTabWidget, QFrame, QToolButton, QScrollArea, QSizePolicy, QDialogButtonBox, QFileDialog, QSlider)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QDateTime
from PyQt6.QtGui import QFont, QIcon, QColor

# Import the icon loading function
from helpers.load_icon import load_bootstrap_icon

# Import standardized hypothesis states
from plan.research_goals import HypothesisState, resolve_hypothesis_status, get_hypothesis_state

import json
from datetime import datetime
import uuid
import logging

# Import TEST_REGISTRY
from data.selection.stat_tests import TEST_REGISTRY

class HypothesisDialog(QDialog):
    """Dialog for creating or editing a hypothesis."""
    
    def __init__(self, parent=None, hypothesis_data=None):
        super().__init__(parent)
        self.setWindowTitle("Create Hypothesis" if hypothesis_data is None else "Edit Hypothesis")
        self.setMinimumWidth(600)
        self.hypothesis_data = hypothesis_data or {}
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Form for hypothesis entry
        form_layout = QFormLayout()
        
        # Hypothesis title
        title_label = QLabel("Title: *")
        title_label.setStyleSheet("font-weight: bold;")
        self.title_edit = QTextEdit()
        self.title_edit.setPlaceholderText("Enter hypothesis title (e.g., 'Treatment effect on primary outcome')")
        self.title_edit.setMaximumHeight(60)
        if 'title' in self.hypothesis_data:
            self.title_edit.setText(self.hypothesis_data['title'])
        form_layout.addRow(title_label, self.title_edit)
        
        # Hypothesis statement (null and alternative)
        null_label = QLabel("Null Hypothesis (H₀): *")
        null_label.setStyleSheet("font-weight: bold;")
        self.null_edit = QTextEdit()
        self.null_edit.setPlaceholderText("Enter null hypothesis (H₀) statement (e.g., 'There is no difference between treatment and control groups')")
        if 'null_hypothesis' in self.hypothesis_data:
            self.null_edit.setText(self.hypothesis_data['null_hypothesis'])
        form_layout.addRow(null_label, self.null_edit)
        
        alt_label = QLabel("Alternative Hypothesis (H₁): *")
        alt_label.setStyleSheet("font-weight: bold;")
        self.alternative_edit = QTextEdit()
        self.alternative_edit.setPlaceholderText("Enter alternative hypothesis (H₁) statement (e.g., 'The treatment group shows improved outcomes compared to control')")
        if 'alternative_hypothesis' in self.hypothesis_data:
            self.alternative_edit.setText(self.hypothesis_data['alternative_hypothesis'])
        form_layout.addRow(alt_label, self.alternative_edit)
        
        # Directionality
        direction_label = QLabel("Directionality:")
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["Non-directional (two-tailed)", 
                                      "Directional - greater than (right-tailed)", 
                                      "Directional - less than (left-tailed)"])
        if 'directionality' in self.hypothesis_data:
            direction_index = 0
            if self.hypothesis_data['directionality'] == 'greater':
                direction_index = 1
            elif self.hypothesis_data['directionality'] == 'less':
                direction_index = 2
            self.direction_combo.setCurrentIndex(direction_index)
        form_layout.addRow(direction_label, self.direction_combo)
        
        # Additional notes
        notes_label = QLabel("Notes:")
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText("Enter any additional notes or context for this hypothesis")
        if 'notes' in self.hypothesis_data:
            self.notes_edit.setText(self.hypothesis_data['notes'])
        form_layout.addRow(notes_label, self.notes_edit)
        
        # Add a note about required fields
        required_note = QLabel("* Required fields")
        required_note.setStyleSheet("color: #dc3545; font-style: italic;")
        layout.addLayout(form_layout)
        layout.addWidget(required_note)
        
        # Buttons
        button_layout = QHBoxLayout()
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.accept)
        button_layout.addWidget(save_button)
        
        layout.addLayout(button_layout)
    
    def get_hypothesis_data(self):
        """Get the hypothesis data from the form."""
        # Map directionality dropdown to internal representation
        direction_map = {
            0: 'non-directional',
            1: 'greater',
            2: 'less'
        }
        
        # Create or update the hypothesis data
        data = self.hypothesis_data.copy() if self.hypothesis_data else {}
        
        # Add/update fields
        data.update({
            'title': self.title_edit.toPlainText().strip(),
            'null_hypothesis': self.null_edit.toPlainText().strip(),
            'alternative_hypothesis': self.alternative_edit.toPlainText().strip(),
            'directionality': direction_map[self.direction_combo.currentIndex()],
            'notes': self.notes_edit.toPlainText().strip()
        })
        
        # Preserve existing variables if they exist
        if 'outcome_variables' in self.hypothesis_data:
            data['outcome_variables'] = self.hypothesis_data['outcome_variables']
        else:
            data['outcome_variables'] = ''
            
        if 'predictor_variables' in self.hypothesis_data:
            data['predictor_variables'] = self.hypothesis_data['predictor_variables']
        else:
            data['predictor_variables'] = ''
        
        # Preserve existing statistical test settings if they exist
        if 'expected_test' in self.hypothesis_data:
            data['expected_test'] = self.hypothesis_data['expected_test']
        else:
            # Default value if needed
            data['expected_test'] = 'T-Test (Independent Samples)'
            
        if 'alpha_level' in self.hypothesis_data:
            data['alpha_level'] = self.hypothesis_data['alpha_level']
        else:
            # Default value
            data['alpha_level'] = 0.05
        
        # If it's a new hypothesis, add ID and timestamps
        if 'id' not in data:
            data['id'] = str(uuid.uuid4())
            data['created_at'] = datetime.now().isoformat()
        
        # Always update the modified timestamp
        data['updated_at'] = datetime.now().isoformat()
        
        # Initialize status if not present
        if 'status' not in data:
            data['status'] = 'untested'
            
        return data


class HypothesisCard(QFrame):
    """Card widget displaying a hypothesis with action buttons."""
    
    clicked = pyqtSignal(object)
    edit_requested = pyqtSignal(object)
    delete_requested = pyqtSignal(object)
    
    def __init__(self, hypothesis_data, parent=None):
        super().__init__(parent)
        self.hypothesis_data = hypothesis_data
        self.selected = False
        self.init_ui()
        
    def init_ui(self):
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setObjectName("hypothesisCard")
        
        # Apply custom styling
        self.setStyleSheet("""
            QFrame#hypothesisCard {
                border: 1px solid palette(mid);
                border-radius: 8px;
                margin: 5px;
                padding: 12px;
            }
            QFrame#hypothesisCard:hover {
                border-color: palette(highlight);
            }
        """)
        
        # Set fixed size for consistent grid layout
        self.setMinimumWidth(300)
        self.setMaximumWidth(400)
        self.setFixedHeight(264)  # Increased by 20% from 220
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(6)  # Reduce spacing between elements
        layout.setContentsMargins(10, 10, 10, 10)  # Consistent margins
        
        # Status indicator and title in header
        header_layout = QHBoxLayout()
        
        # Status indicator
        status_text = self.hypothesis_data.get('status', 'untested').capitalize()
        
        # Select icon based on status
        status_icon = QLabel()
        
        # Use the same colors as in the HypothesisNode for icons only
        status_colors = {
            'proposed': "#b4b4b4",       # Gray
            'testing': "#ffc107",        # Amber
            'untested': "#6c757d",       # Gray
            'confirmed': "#28a745",      # Green
            'validated': "#28a745",      # Green (same as confirmed)
            'rejected': "#dc3545",       # Red
            'inconclusive': "#fd7e14",   # Orange
            'modified': "#17a2b8"        # Cyan
        }
        
        # Icon selection based on status
        icon_name = "question-circle-fill"  # Default
        if status_text.lower() == 'confirmed' or status_text.lower() == 'validated':
            icon_name = "check-circle-fill"
        elif status_text.lower() == 'rejected':
            icon_name = "x-circle-fill"
        elif status_text.lower() == 'inconclusive':
            icon_name = "dash-circle-fill"
        elif status_text.lower() == 'testing':
            icon_name = "hourglass-split"
        elif status_text.lower() == 'modified':
            icon_name = "pencil-square"
            
        # Use colored icons for visual appeal
        status_color = status_colors.get(status_text.lower(), "#6c757d")
        icon = load_bootstrap_icon(icon_name, color=status_color, size=24)
        status_icon.setPixmap(icon.pixmap(24, 24))
        header_layout.addWidget(status_icon)
        
        # Add status text without color override
        status_label = QLabel(status_text)
        header_layout.addWidget(status_label)
        
        header_layout.addStretch()
        
        # Add buttons at the right side
        button_layout = QHBoxLayout()
        button_layout.setSpacing(2)
        
        # Edit button
        edit_button = QToolButton()
        edit_button.setIcon(load_bootstrap_icon("pencil", size=24))
        edit_button.setIconSize(QSize(24, 24))
        edit_button.setToolTip("Edit")
        edit_button.clicked.connect(lambda: self.edit_requested.emit(self.hypothesis_data))
        button_layout.addWidget(edit_button)
        
        # Delete button
        delete_button = QToolButton()
        delete_button.setIcon(load_bootstrap_icon("trash", size=24))
        delete_button.setIconSize(QSize(24, 24))
        delete_button.setToolTip("Delete")
        delete_button.clicked.connect(lambda: self.delete_requested.emit(self.hypothesis_data))
        button_layout.addWidget(delete_button)
        
        header_layout.addLayout(button_layout)
        layout.addLayout(header_layout)
        
        # Title
        title = self.hypothesis_data.get('title', 'Untitled Hypothesis')
        title_label = QLabel(f"<b>{title}</b>")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)
        
        # Hypothesis summary - truncated null and alt statements
        null_hypothesis = self.hypothesis_data.get('null_hypothesis', '')
        alt_hypothesis = self.hypothesis_data.get('alternative_hypothesis', '')
        
        if null_hypothesis or alt_hypothesis:
            summary_frame = QFrame()
            summary_frame.setStyleSheet("border-radius: 4px; padding: 2px;")
            summary_layout = QVBoxLayout(summary_frame)
            summary_layout.setContentsMargins(6, 4, 6, 4)  # Reduce margins
            summary_layout.setSpacing(2)  # Reduce spacing
            
            if null_hypothesis:
                # Show shorter snippet
                null_text = null_hypothesis[:60] + "..." if len(null_hypothesis) > 60 else null_hypothesis
                null_label = QLabel(f"<span style='font-size: 10px;'>H₀: {null_text}</span>")
                null_label.setWordWrap(True)
                summary_layout.addWidget(null_label)
                
            if alt_hypothesis:
                # Show shorter snippet
                alt_text = alt_hypothesis[:60] + "..." if len(alt_hypothesis) > 60 else alt_hypothesis
                alt_label = QLabel(f"<span style='font-size: 10px;'>H₁: {alt_text}</span>")
                alt_label.setWordWrap(True)
                summary_layout.addWidget(alt_label)
                
            layout.addWidget(summary_frame)
        
        # Evidence indicators if available
        has_model_evidence = 'test_results' in self.hypothesis_data and self.hypothesis_data['test_results']
        has_claims_evidence = 'literature_evidence' in self.hypothesis_data and self.hypothesis_data['literature_evidence']
        
        if has_model_evidence or has_claims_evidence:
            evidence_frame = QFrame()
            evidence_frame.setStyleSheet("border-radius: 4px; padding: 4px;")
            evidence_frame.setMaximumHeight(40)  # Set a maximum height for the evidence status bar
            evidence_layout = QHBoxLayout(evidence_frame)
            evidence_layout.setContentsMargins(6, 2, 6, 2)  # Reduce vertical padding
            evidence_layout.setSpacing(8)
            
            # Model evidence indicator
            if has_model_evidence:
                p_value = self.hypothesis_data['test_results'].get('p_value')
                alpha = self.hypothesis_data.get('alpha_level', 0.05)
                
                if p_value is not None:
                    model_icon = QLabel()
                    if p_value < alpha:
                        icon = load_bootstrap_icon("bar-chart-fill", color="#28a745", size=20)  # Green for significant
                    else:
                        icon = load_bootstrap_icon("bar-chart-fill", color="#dc3545", size=20)  # Red for not significant
                    model_icon.setPixmap(icon.pixmap(20, 20))
                    
                    model_layout = QHBoxLayout()
                    model_layout.setSpacing(2)
                    model_layout.addWidget(model_icon)
                    p_value_label = QLabel(f"p={p_value:.4f}")
                    p_value_label.setStyleSheet("font-size: 10px;")
                    model_layout.addWidget(p_value_label)
                    evidence_layout.addLayout(model_layout)
            
            # Claims evidence indicator
            if has_claims_evidence:
                lit_evidence = self.hypothesis_data['literature_evidence']
                supporting = lit_evidence.get('supporting', 0)
                refuting = lit_evidence.get('refuting', 0)
                neutral = lit_evidence.get('neutral', 0)
                
                if supporting + refuting + neutral > 0:
                    claims_icon = QLabel()
                    icon = load_bootstrap_icon("journal-text", color="#0275d8", size=20)  # Blue for literature
                    claims_icon.setPixmap(icon.pixmap(20, 20))
                    
                    claims_layout = QHBoxLayout()
                    claims_layout.setSpacing(2)
                    claims_layout.addWidget(claims_icon)
                    # Use spans without color properties
                    claims_count_label = QLabel(
                        f"<span style='font-size: 10px;'><span style='color: #28a745;'>{supporting}↑</span> <span style='color: #dc3545;'>{refuting}↓</span> <span style='color: #6c757d;'>{neutral}○</span></span>"
                    )
                    claims_layout.addWidget(claims_count_label)
                    evidence_layout.addLayout(claims_layout)
            
            evidence_layout.addStretch()
            layout.addWidget(evidence_frame)
            
        # Parent manager (for undo functionality)
        parent_manager = None
        parent = self.parent()
        while parent and not parent_manager:
            if hasattr(parent, 'change_history'):
                parent_manager = parent
            parent = parent.parent()
            
        # Add undo button if history exists
        if (parent_manager is not None and 
            hasattr(parent_manager, 'change_history') and 
            any(change.hypothesis_id == self.hypothesis_data.get('id') 
                for change in parent_manager.change_history)):
                
            undo_btn = QToolButton()
            undo_btn.setIcon(load_bootstrap_icon("arrow-counterclockwise", size=24))
            undo_btn.setIconSize(QSize(24, 24))
            undo_btn.setToolTip("Undo Last Change")
            undo_btn.clicked.connect(lambda: parent_manager.undo_last_change(self.hypothesis_data.get('id')))
            
            undo_layout = QHBoxLayout()
            undo_layout.addStretch()
            undo_layout.addWidget(undo_btn)
            layout.addLayout(undo_layout)
            
        # Remove the mouse press event that triggers clicked signal
        # self.mousePressEvent = lambda event: self.clicked.emit(self.hypothesis_data)

    def set_selected(self, selected):
        """Set the selected state of this card and update its styling."""
        self.selected = selected
        
        # Update visual styling to show selected state
        if selected:
            self.setStyleSheet("""
                QFrame#hypothesisCard {
                    border: 4px solid palette(dark);
                    border-radius: 8px;
                    margin: 5px;
                    padding: 12px;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame#hypothesisCard {
                    border: 1px solid palette(mid);
                    border-radius: 8px;
                    margin: 5px;
                    padding: 12px;
                }
                QFrame#hypothesisCard:hover {
                    border-color: palette(highlight);
                }
            """)
            
    def mousePressEvent(self, event):
        """Handle mouse press events to show details."""
        self.clicked.emit(self.hypothesis_data)
        super().mousePressEvent(event)


class HypothesisDetailView(QWidget):
    """Detailed view for a selected hypothesis."""
    
    # Add signal for evidence conflicts request
    evidence_conflicts_requested = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_hypothesis = None
        self.studies_manager = None
        self.init_ui()
        
    def set_studies_manager(self, studies_manager):
        """Set the studies manager for accessing cross-study data."""
        self.studies_manager = studies_manager
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Add size policies to control widget expansion
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        
        # Title
        self.title_label = QLabel("Select a hypothesis to view details")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.title_label)
        
        # Status
        status_layout = QHBoxLayout()
        self.status_icon = QLabel()
        self.status_label = QLabel("")
        status_layout.addWidget(self.status_icon)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)
        
        # Create tab widget for information
        self.tab_widget = QTabWidget()
        
        # Create the Main tab (previously Evidence) with basic info
        self.main_tab = QWidget()
        main_layout = QVBoxLayout(self.main_tab)
        
        # Basic hypothesis information
        info_group = QGroupBox("Hypothesis Information")
        info_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        info_layout = QFormLayout(info_group)
        
        self.null_label = QLabel()
        self.null_label.setWordWrap(True)
        info_layout.addRow("Null Hypothesis (H₀):", self.null_label)
        
        self.alternative_label = QLabel()
        self.alternative_label.setWordWrap(True)
        info_layout.addRow("Alternative Hypothesis (H₁):", self.alternative_label)
        
        self.outcome_label = QLabel()
        self.outcome_label.setWordWrap(True)
        info_layout.addRow("Outcome Variable(s):", self.outcome_label)
        
        self.predictors_label = QLabel()
        self.predictors_label.setWordWrap(True)
        info_layout.addRow("Predictor Variable(s):", self.predictors_label)
        
        self.notes_label = QLabel()
        self.notes_label.setWordWrap(True)
        info_layout.addRow("Notes:", self.notes_label)
        
        main_layout.addWidget(info_group)
        
        # Evidence sections
        # Model-based evidence section
        self.model_evidence_group = QGroupBox("Model-Based Evidence")
        model_layout = QVBoxLayout(self.model_evidence_group)
        
        self.no_model_evidence_label = QLabel("No model-based test results available")
        self.no_model_evidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_model_evidence_label.setStyleSheet("color: #6c757d; font-style: italic;")
        model_layout.addWidget(self.no_model_evidence_label)
        
        # Model results display will be created dynamically when results are available
        self.model_results_content = QWidget()
        self.model_results_layout = QVBoxLayout(self.model_results_content)
        model_layout.addWidget(self.model_results_content)
        self.model_results_content.hide()
        
        main_layout.addWidget(self.model_evidence_group)
        
        # Claims-based evidence section
        self.claims_evidence_group = QGroupBox("Claims-Based Evidence")
        claims_layout = QVBoxLayout(self.claims_evidence_group)
        
        self.no_claims_evidence_label = QLabel("No claims-based evidence available")
        self.no_claims_evidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_claims_evidence_label.setStyleSheet("color: #6c757d; font-style: italic;")
        claims_layout.addWidget(self.no_claims_evidence_label)
        
        # Claims results display will be created dynamically when results are available
        self.claims_results_content = QWidget()
        self.claims_results_layout = QVBoxLayout(self.claims_results_content)
        claims_layout.addWidget(self.claims_results_content)
        self.claims_results_content.hide()
        
        main_layout.addWidget(self.claims_evidence_group)
        
        # Evidence summary section
        self.evidence_summary_group = QGroupBox("Evidence Summary")
        summary_layout = QVBoxLayout(self.evidence_summary_group)
        
        self.evidence_summary_label = QLabel("No evidence available")
        self.evidence_summary_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.evidence_summary_label.setWordWrap(True)
        self.evidence_summary_label.setStyleSheet("color: #6c757d; font-style: italic;")
        summary_layout.addWidget(self.evidence_summary_label)
        
        self.view_analysis_btn = QPushButton("View Detailed Evidence Analysis")
        self.view_analysis_btn.clicked.connect(self._request_evidence_conflicts)
        self.view_analysis_btn.setVisible(False)
        summary_layout.addWidget(self.view_analysis_btn)
        
        main_layout.addWidget(self.evidence_summary_group)
        
        # Remove unused widget declarations that were only used by the removed _display_test_results method
        # For backward compatibility with existing code
        # self.results_content = QWidget()
        # self.results_content_layout = QVBoxLayout(self.results_content)
        # self.results_content.hide()
        # 
        # self.no_results_label = QLabel("No test results available")
        
        # Add stretch at the end to push everything to the top
        main_layout.addStretch()
        
        # Add the main tab
        self.tab_widget.addTab(self.main_tab, "Evidence")
        
        # Create the Related Hypotheses tab
        self.related_hypotheses_widget = QWidget()
        related_layout = QVBoxLayout(self.related_hypotheses_widget)
        
        # Header for related hypotheses
        related_header = QLabel("Hypotheses from other studies with shared variables or evidence")
        related_header.setStyleSheet("font-weight: bold;")
        related_layout.addWidget(related_header)
        
        # Scroll area for related hypotheses cards
        related_scroll = QScrollArea()
        related_scroll.setWidgetResizable(True)
        
        self.related_content = QWidget()
        self.related_content_layout = QVBoxLayout(self.related_content)
        
        related_scroll.setWidget(self.related_content)
        related_layout.addWidget(related_scroll)
        
        # Add the related hypotheses tab
        self.tab_widget.addTab(self.related_hypotheses_widget, "Related Hypotheses")
        
        # Add tab widget to main layout
        layout.addWidget(self.tab_widget)
    
    def display_hypothesis(self, hypothesis_data):
        """Display detailed information about a hypothesis."""
        if not hypothesis_data:
            return
        
        self.current_hypothesis = hypothesis_data
        
        # Clear any previous dynamic content
        self.title_label.setText("Select a hypothesis to view details")
        self.status_icon.clear()
        self.status_label.clear()
        self.null_label.clear()
        self.alternative_label.clear()
        self.outcome_label.clear()
        self.predictors_label.clear()
        self.notes_label.clear()
        
        # Update title
        title = hypothesis_data.get('title', 'Untitled Hypothesis')
        self.title_label.setText(title)
        
        # Update status
        status_text = hypothesis_data.get('status', 'untested').capitalize()
        
        # Select icon based on status
        status_icon = QLabel()
        
        # Use the same colors as in the HypothesisNode for icons only
        status_colors = {
            'proposed': "#b4b4b4",       # Gray
            'testing': "#ffc107",        # Amber
            'untested': "#6c757d",       # Gray
            'confirmed': "#28a745",      # Green
            'validated': "#28a745",      # Green (same as confirmed)
            'rejected': "#dc3545",       # Red
            'inconclusive': "#fd7e14",   # Orange
            'modified': "#17a2b8"        # Cyan
        }
        
        # Icon selection based on status
        icon_name = "question-circle-fill"  # Default
        if status_text.lower() == 'confirmed' or status_text.lower() == 'validated':
            icon_name = "check-circle-fill"
        elif status_text.lower() == 'rejected':
            icon_name = "x-circle-fill"
        elif status_text.lower() == 'inconclusive':
            icon_name = "dash-circle-fill"
        elif status_text.lower() == 'testing':
            icon_name = "hourglass-split"
        elif status_text.lower() == 'modified':
            icon_name = "pencil-square"
            
        # Use colored icons for visual appeal
        status_color = status_colors.get(status_text.lower(), "#6c757d")
        icon = load_bootstrap_icon(icon_name, color=status_color, size=24)
        status_icon.setPixmap(icon.pixmap(24, 24))
        self.status_icon.setPixmap(icon.pixmap(24, 24))
        
        # Add status text without color override
        self.status_label.setText(status_text)
        
        # Update hypothesis information
        self.null_label.setText(hypothesis_data.get('null_hypothesis', ''))
        self.alternative_label.setText(hypothesis_data.get('alternative_hypothesis', ''))
        self.outcome_label.setText(hypothesis_data.get('outcome_variables', ''))
        self.predictors_label.setText(hypothesis_data.get('predictor_variables', ''))
        self.notes_label.setText(hypothesis_data.get('notes', ''))
        
        # Display model-based test results if available
        has_model_evidence = False
        if 'test_results' in hypothesis_data and hypothesis_data['test_results']:
            has_model_evidence = True
            
            # Hide the no evidence label
            self.no_model_evidence_label.hide()
            
            # Clear previous model results
            while self.model_results_layout.count():
                item = self.model_results_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                elif item.layout():
                    # Recursively clear the layout
                    while item.layout().count():
                        sub_item = item.layout().takeAt(0)
                        if sub_item.widget():
                            sub_item.widget().deleteLater()
                    # Remove the layout itself
                    item.layout().deleteLater()
            
            # Create form layout for model results
            form_layout = QFormLayout()
            
            # Display test type
            # First check for 'expected_test' from generation, then 'test_results'
            test_type = hypothesis_data.get('expected_test') 
            if not test_type and 'test_results' in hypothesis_data and hypothesis_data['test_results']:
                test_type = hypothesis_data['test_results'].get('test', 'Unknown Test')
            elif not test_type:
                 test_type = 'Unknown Test' # Default if neither is found
            
            test_label = QLabel(test_type)
            test_label.setStyleSheet("font-weight: bold;")
            form_layout.addRow("Test Type:", test_label)
            
            # Display p-value
            p_value = hypothesis_data['test_results'].get('p_value')
            if p_value is not None:
                p_label = QLabel(f"{p_value:.4f}")
                significant = p_value < hypothesis_data.get('alpha_level', 0.05)
                if significant:
                    p_label.setStyleSheet("color: #28a745; font-weight: bold;")  # green
                else:
                    p_label.setStyleSheet("color: #dc3545;")  # red
                form_layout.addRow("p-value:", p_label)
            
            # Display other test statistics
            for key, label_text in [
                ('statistic', 'Test Statistic:'), 
                ('df', 'Degrees of Freedom:'),
                ('effect_size', 'Effect Size:')
            ]:
                value = hypothesis_data['test_results'].get(key)
                if value is not None:
                    value_text = f"{value:.4f}" if isinstance(value, float) else str(value)
                    form_layout.addRow(label_text, QLabel(value_text))
            
            # Display confidence interval
            ci_lower = hypothesis_data['test_results'].get('ci_lower')
            ci_upper = hypothesis_data['test_results'].get('ci_upper')
            if ci_lower is not None and ci_upper is not None:
                ci_label = QLabel(f"[{ci_lower:.4f}, {ci_upper:.4f}]")
                form_layout.addRow("95% Confidence Interval:", ci_label)
            
            # Display conclusion
            conclusion = hypothesis_data['test_results'].get('conclusion', '')
            if conclusion:
                conclusion_label = QLabel(conclusion)
                conclusion_label.setWordWrap(True)
                conclusion_label.setStyleSheet("font-weight: bold;")
                form_layout.addRow("Conclusion:", conclusion_label)
            
            # Add form layout to model results content
            self.model_results_layout.addLayout(form_layout)
            self.model_results_content.show()
        else:
            self.no_model_evidence_label.show()
            self.model_results_content.hide()
        
        # Display claims-based evidence if available
        has_claims_evidence = False
        if 'literature_evidence' in hypothesis_data and hypothesis_data['literature_evidence']:
            has_claims_evidence = True
            lit_evidence = hypothesis_data['literature_evidence']
            
            # Update the claims evidence section
            self.no_claims_evidence_label.hide()
            
            # Clear previous claims results
            while self.claims_results_layout.count():
                item = self.claims_results_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                elif item.layout():
                    # Clear the layout
                    while item.layout().count():
                        sub_item = item.layout().takeAt(0)
                        if sub_item.widget():
                            sub_item.widget().deleteLater()
            
            # Create grid layout for claims results
            claims_layout = QGridLayout()
            
            # Add counts with colored boxes
            supporting = lit_evidence.get('supporting', 0)
            refuting = lit_evidence.get('refuting', 0)
            neutral = lit_evidence.get('neutral', 0)
            total = supporting + refuting + neutral
            
            if total > 0:
                # Summary of studies
                summary_label = QLabel(f"<b>Summary:</b> {supporting} supporting, {refuting} refuting, {neutral} neutral studies")
                claims_layout.addWidget(summary_label, 0, 0, 1, 2)
                
                # Create a bar visualization
                vis_widget = QWidget()
                vis_layout = QHBoxLayout(vis_widget)
                vis_layout.setContentsMargins(0, 10, 0, 10)
                vis_layout.setSpacing(0)
                
                # Only add bars for categories with values
                if supporting > 0:
                    supporting_bar = QFrame()
                    supporting_bar.setStyleSheet("background-color: #28a745;")  # green
                    supporting_bar.setFixedHeight(20)
                    supporting_bar.setFixedWidth(int(300 * (supporting / total)))
                    supporting_bar.setToolTip(f"Supporting: {supporting} studies")
                    vis_layout.addWidget(supporting_bar)
                
                if neutral > 0:
                    neutral_bar = QFrame()
                    neutral_bar.setStyleSheet("background-color: #6c757d;")  # gray
                    neutral_bar.setFixedHeight(20)
                    neutral_bar.setFixedWidth(int(300 * (neutral / total)))
                    neutral_bar.setToolTip(f"Neutral: {neutral} studies")
                    vis_layout.addWidget(neutral_bar)
                
                if refuting > 0:
                    refuting_bar = QFrame()
                    refuting_bar.setStyleSheet("background-color: #dc3545;")  # red
                    refuting_bar.setFixedHeight(20)
                    refuting_bar.setFixedWidth(int(300 * (refuting / total)))
                    refuting_bar.setToolTip(f"Refuting: {refuting} studies")
                    vis_layout.addWidget(refuting_bar)
                
                claims_layout.addWidget(vis_widget, 1, 0, 1, 2)
                
                # Add legend
                legend_widget = QWidget()
                legend_layout = QHBoxLayout(legend_widget)
                legend_layout.setContentsMargins(0, 0, 0, 0)
                
                # Functions to create legend items
                def create_legend_item(color, text, count):
                    item_layout = QHBoxLayout()
                    color_box = QWidget()
                    color_box.setFixedSize(12, 12)
                    color_box.setStyleSheet(f"background-color: {color};")
                    item_layout.addWidget(color_box)
                    item_layout.addWidget(QLabel(f"{text}: {count}"))
                    return item_layout
                
                if supporting > 0:
                    legend_layout.addLayout(create_legend_item("palette(link)", "Supporting", supporting))
                if neutral > 0:
                    legend_layout.addLayout(create_legend_item("palette(mid)", "Neutral", neutral))
                if refuting > 0:
                    legend_layout.addLayout(create_legend_item("palette(negative-text)", "Refuting", refuting))
                
                legend_layout.addStretch()
                claims_layout.addWidget(legend_widget, 2, 0, 1, 2)
            
            # Display literature status
            status = lit_evidence.get('status', 'Unknown')
            status_label = QLabel(status.capitalize())
            if status.lower() == "confirmed":
                status_label.setStyleSheet("font-weight: bold;")
            elif status.lower() == "rejected":
                status_label.setStyleSheet("font-weight: bold;")
            elif status.lower() == "inconclusive":
                status_label.setStyleSheet("font-weight: bold;")
            
            claims_layout.addWidget(QLabel("<b>Status:</b>"), 3, 0)
            claims_layout.addWidget(status_label, 3, 1)
            
            # Display literature conclusion
            if lit_evidence.get('conclusion'):
                conclusion_label = QLabel(lit_evidence['conclusion'])
                conclusion_label.setWordWrap(True)
                
                claims_layout.addWidget(QLabel("<b>Conclusion:</b>"), 4, 0, Qt.AlignmentFlag.AlignTop)
                claims_layout.addWidget(conclusion_label, 4, 1)
            
            # Add layout to claims results content
            self.claims_results_layout.addLayout(claims_layout)
            self.claims_results_content.show()
        else:
            self.no_claims_evidence_label.show()
            self.claims_results_content.hide()
        
        # Update evidence summary
        if has_model_evidence or has_claims_evidence:
            summary_text = ""
            
            if has_model_evidence and has_claims_evidence:
                model_status = "confirmed" if hypothesis_data['test_results'].get('p_value', 1.0) < hypothesis_data.get('alpha_level', 0.05) else "rejected"
                claims_status = hypothesis_data['literature_evidence'].get('status', 'unknown').lower()
                
                if model_status == claims_status:
                    if model_status == "confirmed":
                        summary_text = "<b>✓ Strong evidence supports this hypothesis</b><br>Both model-based and claims-based evidence agree."
                    else:
                        summary_text = "<b>✗ Strong evidence against this hypothesis</b><br>Both model-based and claims-based evidence agree."
                else:
                    summary_text = "<b>⚠️ Conflicting evidence for this hypothesis</b><br>Model-based and claims-based evidence disagree."
            elif has_model_evidence:
                p_value = hypothesis_data['test_results'].get('p_value')
                alpha = hypothesis_data.get('alpha_level', 0.05)
                if p_value is not None and p_value < alpha:
                    summary_text = "<b>✓ Model-based evidence supports this hypothesis</b><br>Consider adding claims-based evidence."
                else:
                    summary_text = "<b>✗ Model-based evidence does not support this hypothesis</b><br>Consider adding claims-based evidence."
            elif has_claims_evidence:
                claims_status = hypothesis_data['literature_evidence'].get('status', '').lower()
                if claims_status == "confirmed":
                    summary_text = "<b>✓ Claims-based evidence supports this hypothesis</b><br>Consider adding model-based evidence."
                elif claims_status == "rejected":
                    summary_text = "<b>✗ Claims-based evidence does not support this hypothesis</b><br>Consider adding model-based evidence."
                else:
                    summary_text = "<b>⚠️ Claims-based evidence is inconclusive</b><br>Consider adding model-based evidence."
            
            self.evidence_summary_label.setText(summary_text)
            self.evidence_summary_label.setStyleSheet("")
            self.view_analysis_btn.setVisible(True)
        else:
            self.evidence_summary_label.setText("No evidence available")
            self.evidence_summary_label.setStyleSheet("font-style: italic;")
            self.view_analysis_btn.setVisible(False)
        
        # Remove the backward compatibility call that creates a popup
        # For backward compatibility with original _display_test_results method
        # if 'test_results' in hypothesis_data:
        #     self._display_test_results(hypothesis_data['test_results'])
        
        # Load related hypotheses
        self._load_related_hypotheses(hypothesis_data)
    
    def _load_related_hypotheses(self, hypothesis_data):
        """Load and display hypotheses from other studies that are related to this one."""
        # Clear previous content
        while self.related_content_layout.count():
            item = self.related_content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        if not self.studies_manager:
            # Show message that studies manager is required
            message = QLabel("Studies manager not available")
            message.setAlignment(Qt.AlignmentFlag.AlignCenter)
            message.setStyleSheet("font-style: italic;")
            self.related_content_layout.addWidget(message)
            return
            
        # Get active study to know which study we're currently viewing
        active_study = self.studies_manager.get_active_study()
        if not active_study:
            message = QLabel("No active study")
            message.setAlignment(Qt.AlignmentFlag.AlignCenter)
            message.setStyleSheet("font-style: italic;")
            self.related_content_layout.addWidget(message)
            return
            
        # Get all studies
        all_studies = self.studies_manager.list_studies()
        if not all_studies:
            message = QLabel("No studies found")
            message.setAlignment(Qt.AlignmentFlag.AlignCenter)
            message.setStyleSheet("font-style: italic;")
            self.related_content_layout.addWidget(message)
            return
            
        # Extract variables from current hypothesis
        current_outcome_vars = set(var.strip() for var in hypothesis_data.get('outcome_variables', '').split(',') if var.strip())
        current_predictor_vars = set(var.strip() for var in hypothesis_data.get('predictor_variables', '').split(',') if var.strip())
        
        # Track related hypotheses
        related_by_variables = []
        related_by_evidence = []
        
        # Check each study for related hypotheses
        for study_info in all_studies:
            # Skip the current study
            if study_info["id"] == active_study.id:
                continue
                
            # Get hypotheses from this study
            self.studies_manager.set_active_study(study_info["id"])
            study_hypotheses = self.studies_manager.get_study_hypotheses()
            
            for hyp in study_hypotheses:
                # Check if this hypothesis shares variables with our current one
                hyp_outcome_vars = set(var.strip() for var in hyp.get('outcome_variables', '').split(',') if var.strip())
                hyp_predictor_vars = set(var.strip() for var in hyp.get('predictor_variables', '').split(',') if var.strip())
                
                # Calculate variable overlap
                shared_outcome = current_outcome_vars.intersection(hyp_outcome_vars)
                shared_predictors = current_predictor_vars.intersection(hyp_predictor_vars)
                
                if shared_outcome or shared_predictors:
                    # Copy and add study info
                    related_hyp = hyp.copy()
                    related_hyp["study_name"] = study_info["name"]
                    related_hyp["study_id"] = study_info["id"]
                    related_hyp["relation_type"] = "variables"
                    related_hyp["shared_vars"] = list(shared_outcome.union(shared_predictors))
                    related_by_variables.append(related_hyp)
                    
                # Check if this hypothesis references our current one as evidence
                for evidence in hyp.get("supporting_evidence", []) + hyp.get("contradicting_evidence", []):
                    if "cross_reference" in evidence:
                        ref = evidence["cross_reference"]
                        if ref.get("hypothesis_id") == hypothesis_data.get("id"):
                            # This hypothesis references our current one
                            related_hyp = hyp.copy()
                            related_hyp["study_name"] = study_info["name"]
                            related_hyp["study_id"] = study_info["id"]
                            related_hyp["relation_type"] = "evidence_target"
                            related_hyp["evidence_type"] = "supporting" if evidence in hyp.get("supporting_evidence", []) else "contradicting"
                            related_by_evidence.append(related_hyp)
                
                # Check if our current hypothesis references this one as evidence
                for evidence in hypothesis_data.get("supporting_evidence", []) + hypothesis_data.get("contradicting_evidence", []):
                    if "cross_reference" in evidence:
                        ref = evidence["cross_reference"]
                        if ref.get("hypothesis_id") == hyp.get("id"):
                            # Our current hypothesis references this one
                            related_hyp = hyp.copy()
                            related_hyp["study_name"] = study_info["name"]
                            related_hyp["study_id"] = study_info["id"]
                            related_hyp["relation_type"] = "evidence_source"
                            related_hyp["evidence_type"] = "supporting" if evidence in hypothesis_data.get("supporting_evidence", []) else "contradicting"
                            related_by_evidence.append(related_hyp)
        
        # Reset to original active study
        self.studies_manager.set_active_study(active_study.id)
        
        # Display related hypotheses
        if not related_by_variables and not related_by_evidence:
            message = QLabel("No related hypotheses found in other studies")
            message.setAlignment(Qt.AlignmentFlag.AlignCenter)
            message.setStyleSheet("font-style: italic;")
            self.related_content_layout.addWidget(message)
            return
            
        # First show evidence-related hypotheses
        if related_by_evidence:
            # Create a visible section with background
            evidence_section = QFrame()
            evidence_section.setStyleSheet("border-radius: 5px; margin: 5px;")
            evidence_layout = QVBoxLayout(evidence_section)
            
            # Add header with icon
            header_layout = QHBoxLayout()
            icon_label = QLabel()
            icon_label.setPixmap(load_bootstrap_icon("journal-check", size=24).pixmap(24, 24))
            header_layout.addWidget(icon_label)
            
            evidence_header = QLabel("<h3>Hypotheses Connected by Evidence</h3>")
            evidence_header.setStyleSheet("font-weight: bold;")
            header_layout.addWidget(evidence_header)
            header_layout.addStretch()
            evidence_layout.addLayout(header_layout)
            
            # Add explanation
            explanation = QLabel("These hypotheses are linked through shared evidence relationships.")
            explanation.setWordWrap(True)
            explanation.setStyleSheet("margin-bottom: 10px;")
            evidence_layout.addWidget(explanation)
            
            # Add each related hypothesis
            for hyp in related_by_evidence:
                card = self._create_related_hypothesis_card(hyp)
                evidence_layout.addWidget(card)
            
            # Add the section to the main layout
            self.related_content_layout.addWidget(evidence_section)
        
        # Then show variable-related hypotheses
        if related_by_variables:
            # Add a separator between sections if we already added evidence cards
            if related_by_evidence:
                self.related_content_layout.addSpacing(20)
            
            # Create a visible section with background
            variables_section = QFrame()
            variables_section.setStyleSheet("border-radius: 5px; margin: 5px;")
            variables_layout = QVBoxLayout(variables_section)
            
            # Add header with icon
            header_layout = QHBoxLayout()
            icon_label = QLabel()
            icon_label.setPixmap(load_bootstrap_icon("diagram-3", size=24).pixmap(24, 24))
            header_layout.addWidget(icon_label)
            
            variables_header = QLabel("<h3>Hypotheses with Shared Variables</h3>")
            variables_header.setStyleSheet("font-weight: bold;")
            header_layout.addWidget(variables_header)
            header_layout.addStretch()
            variables_layout.addLayout(header_layout)
            
            # Add explanation
            explanation = QLabel("These hypotheses are linked through shared outcome or predictor variables.")
            explanation.setWordWrap(True)
            explanation.setStyleSheet("margin-bottom: 10px;")
            variables_layout.addWidget(explanation)
            
            # Add each related hypothesis
            for hyp in related_by_variables:
                card = self._create_related_hypothesis_card(hyp)
                variables_layout.addWidget(card)
            
            # Add the section to the main layout
            self.related_content_layout.addWidget(variables_section)
        
        # Add a stretch at the end to push everything to the top
        self.related_content_layout.addStretch()
    
    def _create_related_hypothesis_card(self, hyp_data):
        """Create a card displaying a related hypothesis with its relationship to the current one."""
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setObjectName("relatedHypothesisCard")
        
        # Apply custom styling
        card.setStyleSheet("""
            QFrame#relatedHypothesisCard {
                border: 1px solid palette(mid);
                border-radius: 8px;
                margin: 5px;
                padding: 10px;
            }
            QFrame#relatedHypothesisCard:hover {
                border-color: palette(highlight);
            }
        """)
        
        layout = QVBoxLayout(card)
        
        # Study name
        study_name = QLabel(f"Study: {hyp_data.get('study_name', 'Unknown')}")
        study_name.setStyleSheet("font-size: 11px;")
        layout.addWidget(study_name)
        
        # Hypothesis title
        title = QLabel(f"<b>{hyp_data.get('title', 'Untitled Hypothesis')}</b>")
        title.setWordWrap(True)
        layout.addWidget(title)
        
        # Relationship type
        relation_layout = QHBoxLayout()
        
        if hyp_data.get("relation_type") == "variables":
            relation_icon = QLabel()
            relation_icon.setPixmap(load_bootstrap_icon("link", color="#007bff", size=24).pixmap(24, 24))
            relation_layout.addWidget(relation_icon)
            
            shared_vars = ", ".join(hyp_data.get("shared_vars", []))
            relation_text = QLabel(f"Shares variables: <b>{shared_vars}</b>")
            relation_layout.addWidget(relation_text)
            
        elif hyp_data.get("relation_type") == "evidence_target":
            # This hypothesis uses our current one as evidence
            if hyp_data.get("evidence_type") == "supporting":
                relation_icon = QLabel()
                relation_icon.setPixmap(load_bootstrap_icon("arrow-up-circle", color="#28a745", size=24).pixmap(24, 24))
                relation_layout.addWidget(relation_icon)
                
                relation_text = QLabel("Uses current hypothesis as supporting evidence")
            else:
                relation_icon = QLabel()
                relation_icon.setPixmap(load_bootstrap_icon("arrow-down-circle", color="#dc3545", size=24).pixmap(24, 24))
                relation_layout.addWidget(relation_icon)
                
                relation_text = QLabel("Uses current hypothesis as contradicting evidence")
                
            relation_layout.addWidget(relation_text)
            
        elif hyp_data.get("relation_type") == "evidence_source":
            # Our current hypothesis uses this one as evidence
            if hyp_data.get("evidence_type") == "supporting":
                relation_icon = QLabel()
                relation_icon.setPixmap(load_bootstrap_icon("arrow-down-circle", color="#28a745", size=24).pixmap(24, 24))
                relation_layout.addWidget(relation_icon)
                
                relation_text = QLabel("Used as supporting evidence by current hypothesis")
            else:
                relation_icon = QLabel()
                relation_icon.setPixmap(load_bootstrap_icon("arrow-up-circle", color="#dc3545", size=24).pixmap(24, 24))
                relation_layout.addWidget(relation_icon)
                
                relation_text = QLabel("Used as contradicting evidence by current hypothesis")
                
            relation_layout.addWidget(relation_text)
            
        relation_layout.addStretch()
        layout.addLayout(relation_layout)
        
        # Status
        status_text = hyp_data.get('status', 'untested').capitalize()
        status_colors = {
            'proposed': "#b4b4b4",       # Gray
            'testing': "#ffc107",        # Amber
            'untested': "#6c757d",       # Dark Gray
            'validated': "#28a745",      # Green
            'confirmed': "#28a745",      # Green (same as validated)
            'rejected': "#dc3545",       # Red
            'inconclusive': "#fd7e14",   # Orange
            'modified': "#007bff",       # Blue
        }
        status_color = status_colors.get(status_text.lower(), "#6c757d")
        status_label = QLabel(f"Status: {status_text}")
        status_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(status_label)
        
        # Add view button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        view_button = QPushButton("View Hypothesis")
        view_button.setStyleSheet("""
            font-size: 11px; 
            padding: 3px 10px;
            border: 1px solid palette(mid);
            border-radius: 4px;
        """)
        view_button.clicked.connect(lambda: self._view_related_hypothesis(hyp_data))
        button_layout.addWidget(view_button)
        
        layout.addLayout(button_layout)
        
        return card
        
    def _view_related_hypothesis(self, hyp_data):
        """Switch to the study containing this hypothesis and display it."""
        if not self.studies_manager:
            return
            
        # Get the parent HypothesesManagerWidget to access its methods
        parent_manager = self.parent()
        while parent_manager and not parent_manager.__class__.__name__ == 'HypothesesManagerWidget':
            parent_manager = parent_manager.parent()
            
        if not parent_manager:
            # No message box - just return silently
            return
            
        # Set the study as active without confirmation
        self.studies_manager.set_active_study(hyp_data.get('study_id'))
        
        # Reload hypotheses in the manager
        parent_manager.load_hypotheses()
        
        # Find and display the hypothesis
        for i in range(parent_manager.grid_layout.count()):
            item = parent_manager.grid_layout.itemAt(i)
            widget = item.widget()
            if (widget and hasattr(widget, 'hypothesis_data') and 
                widget.hypothesis_data.get('id') == hyp_data.get('id')):
                # Simulate clicking on this card
                parent_manager.show_hypothesis_details(widget.hypothesis_data)
                # Set focus to this widget
                widget.setFocus()
                return
    
    def _request_evidence_conflicts(self):
        """Request showing evidence conflicts for the current hypothesis."""
        if self.current_hypothesis and 'id' in self.current_hypothesis:
            self.evidence_conflicts_requested.emit(self.current_hypothesis['id'])


class HypothesisChange:
    """Represents a change to a hypothesis for undo/redo functionality."""
    
    def __init__(self, hypothesis_id, field, old_value, new_value, timestamp=None):
        self.hypothesis_id = hypothesis_id
        self.field = field  # e.g., 'status', 'test_results', 'literature_evidence'
        self.old_value = old_value
        self.new_value = new_value
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self):
        return {
            'hypothesis_id': self.hypothesis_id,
            'field': self.field,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            data['hypothesis_id'],
            data['field'],
            data['old_value'],
            data['new_value'],
            datetime.fromisoformat(data['timestamp'])
        )


class HypothesesManagerWidget(QWidget):
    """Widget for managing research hypotheses."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.studies_manager = None
        self.data_testing_widget = None
        self.interpretation_widget = None
        self.change_history = []  # Track changes for undo/redo
        self.max_history = 50  # Maximum number of changes to track
        self.selected_card = None  # Track currently selected card
        self.cards = []  # Keep track of all cards for selection management
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI components."""
        main_layout = QVBoxLayout(self)
        
        # Header section with study selector and filter buttons
        header_layout = QHBoxLayout()
        
        # Study selector will be populated when studies_manager is set
        self.study_selector = QComboBox()
        self.study_selector.setMinimumWidth(250)
        self.study_selector.currentIndexChanged.connect(self._on_study_changed)
        header_layout.addWidget(QLabel("Study:"))
        header_layout.addWidget(self.study_selector)
        
        # Filter button
        filter_btn = QPushButton("Filter")
        filter_btn.setIcon(load_bootstrap_icon("funnel"))
        filter_btn.clicked.connect(self.show_filter_dialog)
        header_layout.addWidget(filter_btn)
        
        # Add hypothesis button
        add_btn = QPushButton("Add Hypothesis")
        add_btn.setIcon(load_bootstrap_icon("plus-circle"))
        add_btn.clicked.connect(self.create_hypothesis)
        header_layout.addWidget(add_btn)
        
        # Add cross-study linking button
        link_btn = QPushButton("Cross-Study Links")
        link_btn.setIcon(load_bootstrap_icon("link"))
        link_btn.setToolTip("Link hypotheses across studies")
        link_btn.clicked.connect(self.show_cross_study_linking)
        header_layout.addWidget(link_btn)
        
        # Add refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setIcon(load_bootstrap_icon("arrow-clockwise"))
        refresh_btn.setToolTip("Refresh hypotheses and variable data")
        refresh_btn.clicked.connect(self.update_study_selector)
        header_layout.addWidget(refresh_btn)
        
        # Add shared variables button
        shared_vars_btn = QPushButton("Shared Variables")
        shared_vars_btn.setIcon(load_bootstrap_icon("intersection"))
        shared_vars_btn.setToolTip("View variables shared across multiple studies")
        shared_vars_btn.clicked.connect(self.debug_and_show_shared_variables)
        header_layout.addWidget(shared_vars_btn)
        
        # Add multi-study demo button for testing
        demo_btn = QPushButton("Create Multi-Study Demo")
        demo_btn.clicked.connect(self.create_multi_study_demo)
        header_layout.addWidget(demo_btn)
        
        header_layout.addStretch()
        
        main_layout.addLayout(header_layout)
        
        # Split view with hypotheses grid on left and details on right
        split_layout = QHBoxLayout()
        
        # Hypotheses grid (scrollable)
        grid_container = QScrollArea()
        grid_container.setWidgetResizable(True)
        
        grid_widget = QWidget()
        self.grid_layout = QGridLayout(grid_widget)
        self.grid_layout.setSpacing(15)  # Increased spacing between cards
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)  # Align to top-left
        
        grid_container.setWidget(grid_widget)
        split_layout.addWidget(grid_container, 1)
        
        # Details panel
        self.detail_view = HypothesisDetailView()
        self.detail_view.evidence_conflicts_requested.connect(self.show_evidence_conflicts)
        split_layout.addWidget(self.detail_view, 1)
        
        main_layout.addLayout(split_layout, 1)
        
        # Status bar at bottom
        self.status_bar = QLabel("Ready")
        self.status_bar.setStyleSheet("color: #6c757d; border-top: 1px solid #dee2e6; padding: 5px;")
        main_layout.addWidget(self.status_bar)
        
        # Initialize change history
        self.change_history = []
        self.max_history = 50
        
        # Initialize filter state
        self.current_filter = ['confirmed', 'rejected', 'inconclusive', 'untested', 'testing', 'modified', 'proposed']
    
    def set_studies_manager(self, studies_manager):
        """Set the studies manager instance."""
        self.studies_manager = studies_manager
        # Also set it on the detail view
        self.detail_view.set_studies_manager(studies_manager)
        
        # Populate study selector if it exists
        if hasattr(self, 'study_selector'):
            self.study_selector.clear()
            studies = self.studies_manager.list_studies()
            for study in studies:
                self.study_selector.addItem(study["name"], study["id"])
            
            # Set current to active study
            active_study = self.studies_manager.get_active_study()
            if active_study:
                for i in range(self.study_selector.count()):
                    if self.study_selector.itemData(i) == active_study.id:
                        self.study_selector.setCurrentIndex(i)
                        break
        
        self.load_hypotheses()
        
    def set_data_testing_widget(self, widget):
        """Set the data testing widget for integration."""
        self.data_testing_widget = widget
        
    def set_interpretation_widget(self, widget):
        """Set the interpretation widget for integration."""
        self.interpretation_widget = widget
        
    def load_hypotheses(self):
        """Load hypotheses from the active study."""
        if not self.studies_manager:
            print("Cannot load hypotheses - studies manager not set")
            return
        
        # Clear existing cards
        self._clear_grid()
        self.cards = []  # Reset cards list
        self.selected_card = None  # Reset selected card
        
        # Get hypotheses using the studies_manager method instead of direct access
        # This ensures all the validation and formatting happens
        hypotheses = self.studies_manager.get_study_hypotheses()
        
        print(f"Loaded {len(hypotheses)} hypotheses from studies_manager.get_study_hypotheses()")
        
        if not hypotheses:
            empty_label = QLabel("No hypotheses found in this study.")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_label.setStyleSheet("color: #6c757d; font-style: italic;")
            self.grid_layout.addWidget(empty_label, 0, 0, 1, 3)
            
            self.status_bar.setText("No hypotheses found in this study.")
            return
        
        # Calculate grid dimensions based on container width
        max_cols = 3  # Default to 3 columns, but could be adjusted dynamically
        
        # Add hypothesis cards to grid
        for i, hypothesis in enumerate(hypotheses):
            row = i // max_cols
            col = i % max_cols
            
            # Debug the hypothesis data
            print(f"Adding hypothesis card {i+1}: {hypothesis.get('title', 'Unnamed')}")
            
            card = HypothesisCard(hypothesis)
            card.clicked.connect(self.show_hypothesis_details)
            card.edit_requested.connect(self.edit_hypothesis)
            card.delete_requested.connect(self.delete_hypothesis)
            
            self.grid_layout.addWidget(card, row, col)
            self.cards.append(card)  # Add to tracked cards
        
        # Update status bar
        self.status_bar.setText(f"Found {len(hypotheses)} hypotheses")
    
    def _clear_grid(self):
        """Clear all items from the grid layout."""
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        self.cards = []  # Clear cards list
        self.selected_card = None  # Reset selected card
    
    def create_hypothesis(self):
        """Create a new hypothesis."""
        dialog = HypothesisDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            hypothesis_data = dialog.get_hypothesis_data()
            
            if not self.studies_manager:
                QMessageBox.warning(self, "Error", "No studies manager available")
                return
            
            # Validate required fields
            if not hypothesis_data['title'] or not hypothesis_data['null_hypothesis'] or not hypothesis_data['alternative_hypothesis']:
                QMessageBox.warning(self, "Missing Information", 
                                   "Please fill out all required fields (Title, Null and Alternative Hypotheses)")
                return
            
            try:
                # Extract the simplest form of hypothesis text
                hypothesis_text = hypothesis_data['alternative_hypothesis']
                
                # Use the StudiesManager method to add the hypothesis
                hypothesis_id = self.studies_manager.add_hypothesis_to_study(
                    hypothesis_text=hypothesis_text,
                    hypothesis_data=hypothesis_data
                )
                
                if hypothesis_id:
                    # Reload the hypotheses
                    self.load_hypotheses()
                    
                    # Find the newly created hypothesis to display it
                    new_hypothesis = self.studies_manager.get_hypothesis(hypothesis_id)
                    
                    if new_hypothesis:
                        self.show_hypothesis_details(new_hypothesis)
                    
                    # Update status
                    self.status_bar.setText(f"Created new hypothesis: {hypothesis_data['title']}")
                else:
                    QMessageBox.warning(self, "Error", "Failed to create hypothesis")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to create hypothesis: {str(e)}")
    
    def edit_hypothesis(self, hypothesis_data):
        """Edit an existing hypothesis."""
        if not self.studies_manager:
            QMessageBox.warning(self, "Error", "No studies manager available")
            return
        
        # Show dialog to edit the hypothesis
        dialog = HypothesisDialog(self, hypothesis_data)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Get updated hypothesis data
            updated_data = dialog.get_hypothesis_data()
            
            # Use the StudiesManager to update the hypothesis
            success = self.studies_manager.update_hypothesis(
                hypothesis_id=updated_data.get('id'),
                update_data=updated_data
            )
            
            if success:
                # Refresh display
                self.load_hypotheses()
                
                # Display the edited hypothesis
                self.show_hypothesis_details(updated_data)
                
                # Update status
                self.status_bar.setText(f"Edited hypothesis: {updated_data['title']}")
            else:
                QMessageBox.warning(self, "Error", "Failed to update hypothesis")
    
    def delete_hypothesis(self, hypothesis_data):
        """Delete a hypothesis."""
        if not self.studies_manager:
            QMessageBox.warning(self, "Error", "No studies manager available")
            return
        
        # Confirm deletion
        confirm = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete the hypothesis '{hypothesis_data.get('title')}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if confirm != QMessageBox.StandardButton.Yes:
            return
        
        # Use the StudiesManager to delete the hypothesis
        success = self.studies_manager.delete_hypothesis(hypothesis_data.get('id'))
        
        if success:
            # Refresh display
            self.load_hypotheses()
            
            # Update status
            self.status_bar.setText(f"Deleted hypothesis: {hypothesis_data.get('title')}")
        else:
            QMessageBox.warning(self, "Error", "Failed to delete hypothesis")
    
    def test_hypothesis(self, hypothesis_data):
        """Test a hypothesis using the data testing widget."""
        if not self.data_testing_widget:
            return
        
        # Extract variables from hypothesis
        outcome_vars = hypothesis_data.get('outcome_variables', '').split(',')
        outcome_vars = [var.strip() for var in outcome_vars if var.strip()]
        
        predictor_vars = hypothesis_data.get('predictor_variables', '').split(',')
        predictor_vars = [var.strip() for var in predictor_vars if var.strip()]
        
        expected_test = hypothesis_data.get('expected_test', '')
        
        # TODO: Integrate with the data testing widget
        # This would involve setting the variables and test type in the data testing widget
        # and switching to that tab
        
        # Just update the status bar instead of showing a message box
        self.status_bar.setText(f"Testing hypothesis: {hypothesis_data['title']}")
    
    def show_hypothesis_details(self, hypothesis_data):
        """Display detailed information about a hypothesis."""
        # Update selected card styling
        for card in self.cards:
            if card.hypothesis_data.get('id') == hypothesis_data.get('id'):
                if self.selected_card:
                    self.selected_card.set_selected(False)
                card.set_selected(True)
                self.selected_card = card
            else:
                card.set_selected(False)
                
        # Display the hypothesis details
        self.detail_view.display_hypothesis(hypothesis_data)
    
    def show_filter_dialog(self):
        """Show dialog to filter hypotheses by status."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Filter Hypotheses")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout(dialog)
        
        # Status checkboxes
        status_group = QGroupBox("Filter by Status")
        status_layout = QVBoxLayout(status_group)
        
        all_check = QCheckBox("All")
        all_check.setChecked(True)
        status_layout.addWidget(all_check)
        
        # Create checkboxes for the standardized states
        status_checks = {}
        for state in HypothesisState:
            status_name = state.value.replace("_", " ").title()
            checkbox = QCheckBox(status_name)
            checkbox.setChecked(True)
            status_layout.addWidget(checkbox)
            status_checks[state.value] = checkbox
            
        # Also add the legacy "confirmed" status that matches validated
        confirmed_check = QCheckBox("Confirmed")
        confirmed_check.setChecked(True)
        status_layout.addWidget(confirmed_check)
        status_checks['confirmed'] = confirmed_check
        
        # Connect "All" checkbox to control others
        def on_all_toggled(checked):
            for checkbox in status_checks.values():
                checkbox.setChecked(checked)
                checkbox.setEnabled(not checked)
            
        all_check.toggled.connect(on_all_toggled)
        
        layout.addWidget(status_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_button)
        
        apply_button = QPushButton("Apply Filter")
        apply_button.clicked.connect(dialog.accept)
        button_layout.addWidget(apply_button)
        
        layout.addLayout(button_layout)
        
        # Show dialog
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Get selected filters
            selected_statuses = []
            
            if all_check.isChecked():
                # Include all states including both enum values and legacy string values
                selected_statuses = [state.value for state in HypothesisState]
                selected_statuses.append('confirmed')  # legacy value
            else:
                # Add selected states
                for state_value, checkbox in status_checks.items():
                    if checkbox.isChecked():
                        selected_statuses.append(state_value)
            
            # Apply filter
            self._apply_filter(selected_statuses)
    
    def _apply_filter(self, statuses):
        """Filter displayed hypotheses by status."""
        if not self.studies_manager:
            return
        
        active_study = self.studies_manager.get_active_study()
        if not active_study:
            return
        
        # Clear existing cards
        self._clear_grid()
        self.cards = []  # Reset cards list
        
        # Get hypotheses from study
        all_hypotheses = getattr(active_study, 'hypotheses', [])
        
        # Filter by status
        filtered_hypotheses = [h for h in all_hypotheses if h.get('status', 'untested') in statuses]
        
        if not filtered_hypotheses:
            self.status_bar.setText("No hypotheses match the selected filters")
            empty_label = QLabel("No hypotheses match the current filter.")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_label.setStyleSheet("color: #6c757d; font-style: italic;")
            self.grid_layout.addWidget(empty_label, 0, 0, 1, 3)
            return
        
        # Calculate grid dimensions based on container width
        max_cols = 3  # Default to 3 columns, but could be adjusted dynamically
        
        # Add hypothesis cards to grid
        for i, hypothesis in enumerate(filtered_hypotheses):
            row = i // max_cols
            col = i % max_cols
            
            card = HypothesisCard(hypothesis)
            card.clicked.connect(self.show_hypothesis_details)
            card.edit_requested.connect(self.edit_hypothesis)
            card.delete_requested.connect(self.delete_hypothesis)
            
            self.grid_layout.addWidget(card, row, col)
            self.cards.append(card)  # Add to tracked cards
        
        self.status_bar.setText(f"Displaying {len(filtered_hypotheses)} of {len(all_hypotheses)} hypotheses")
    
    def create_hypothesis_from_data(self, hypothesis_data):
        """Create a new hypothesis from provided data."""
        if not self.studies_manager:
            QMessageBox.warning(self, "Error", "No studies manager available")
            return None
        
        active_study = self.studies_manager.get_active_study()
        if not active_study:
            QMessageBox.warning(self, "Error", "No active study")
            return None
        
        # Ensure hypothesis has an ID and timestamps
        if 'id' not in hypothesis_data:
            hypothesis_data['id'] = str(uuid.uuid4())
        if 'created_at' not in hypothesis_data:
            hypothesis_data['created_at'] = datetime.now().isoformat()
        if 'updated_at' not in hypothesis_data:
            hypothesis_data['updated_at'] = datetime.now().isoformat()
        if 'status' not in hypothesis_data:
            hypothesis_data['status'] = 'untested'
        
        # Initialize hypotheses list if it doesn't exist
        if not hasattr(active_study, 'hypotheses'):
            active_study.hypotheses = []
        
        # Add the new hypothesis
        active_study.hypotheses.append(hypothesis_data)
        
        # Update the modified timestamp of the study
        active_study.updated_at = datetime.now().isoformat()
        
        # Reload the hypotheses
        self.load_hypotheses()
        
        # Display the new hypothesis
        self.show_hypothesis_details(hypothesis_data)
        
        # Update status
        self.status_bar.setText(f"Created new hypothesis: {hypothesis_data['title']}")
        
        return hypothesis_data['id']
    
    def update_hypothesis_with_results(self, hypothesis_id, test_results):
        """Update a hypothesis with statistical test results."""
        if not self.studies_manager:
            return False
        
        active_study = self.studies_manager.get_active_study()
        if not active_study:
            return False
        
        # Find the hypothesis
        for i, hyp in enumerate(active_study.hypotheses):
            if hyp.get('id') == hypothesis_id:
                # Store old values for history
                old_results = hyp.get('test_results', {})
                old_status = hyp.get('status', 'untested')
                
                # Add test results
                active_study.hypotheses[i]['test_results'] = test_results
                
                # Track test results change in history
                self._add_to_history(
                    hypothesis_id,
                    'test_results',
                    old_results,
                    test_results
                )
                
                # Update test date
                active_study.hypotheses[i]['test_date'] = datetime.now().isoformat()
                
                # Determine statistical status based on p-value
                p_value = test_results.get('p_value')
                alpha = active_study.hypotheses[i].get('alpha_level', 0.05)
                
                statistical_evidence = None
                if p_value is not None:
                    statistical_evidence = {
                        'p_value': p_value,
                        'alpha_level': alpha
                    }
                
                # Get literature evidence if it exists
                literature_evidence = active_study.hypotheses[i].get('literature_evidence')
                
                # Use the standardized function to determine final status
                hypothesis_state = resolve_hypothesis_status(
                    statistical_evidence=statistical_evidence,
                    literature_evidence=literature_evidence
                )
                
                # Convert to string representation for storage in JSON
                status_map = {
                    HypothesisState.PROPOSED: 'untested',
                    HypothesisState.UNTESTED: 'untested',
                    HypothesisState.TESTING: 'testing',
                    HypothesisState.VALIDATED: 'confirmed',
                    HypothesisState.REJECTED: 'rejected',
                    HypothesisState.INCONCLUSIVE: 'inconclusive',
                    HypothesisState.MODIFIED: 'modified'
                }
                
                new_status = status_map.get(hypothesis_state, 'untested')
                
                # Update if the status has changed
                if new_status != old_status:
                    active_study.hypotheses[i]['status'] = new_status
                    
                    # Track status change in history
                    self._add_to_history(
                        hypothesis_id,
                        'status',
                        old_status,
                        new_status
                    )
                    
                    # Update cross-study links when hypothesis state changes
                    self.on_hypothesis_state_changed(hypothesis_id, new_status)
                
                # Update the modified timestamp
                active_study.hypotheses[i]['updated_at'] = datetime.now().isoformat()
                
                # Refresh display
                self.load_hypotheses()
                
                return True
        
        return False

    def update_hypothesis_with_literature(self, hypothesis_id, literature_evidence):
        """Update a hypothesis with literature evidence results."""
        if not self.studies_manager:
            return False
        
        active_study = self.studies_manager.get_active_study()
        if not active_study:
            return False
        
        # Validate literature evidence format
        required_fields = ['supporting', 'refuting', 'neutral', 'status', 'conclusion']
        if not all(field in literature_evidence for field in required_fields):
            logging.error("Invalid literature evidence format")
            return False
        
        # Find the hypothesis
        for i, hyp in enumerate(active_study.hypotheses):
            if hyp.get('id') == hypothesis_id:
                # Store old values for history
                old_evidence = hyp.get('literature_evidence', {})
                old_status = hyp.get('status', 'untested')
                
                # Add literature evidence
                active_study.hypotheses[i]['literature_evidence'] = literature_evidence
                
                # Track this change in history
                self._add_to_history(
                    hypothesis_id,
                    'literature_evidence',
                    old_evidence,
                    literature_evidence
                )
                
                # Get statistical evidence if it exists
                statistical_evidence = active_study.hypotheses[i].get('test_results')
                
                # Use the standardized function to determine final status
                hypothesis_state = resolve_hypothesis_status(
                    statistical_evidence=statistical_evidence,
                    literature_evidence=literature_evidence
                )
                
                # Convert to string representation for storage in JSON
                status_map = {
                    HypothesisState.PROPOSED: 'untested',
                    HypothesisState.UNTESTED: 'untested',
                    HypothesisState.TESTING: 'testing',
                    HypothesisState.VALIDATED: 'confirmed',
                    HypothesisState.REJECTED: 'rejected',
                    HypothesisState.INCONCLUSIVE: 'inconclusive',
                    HypothesisState.MODIFIED: 'modified'
                }
                
                new_status = status_map.get(hypothesis_state, 'untested')
                
                # Update if the status has changed
                if new_status != old_status:
                    active_study.hypotheses[i]['status'] = new_status
                    
                    # Track status change in history
                    self._add_to_history(
                        hypothesis_id,
                        'status',
                        old_status,
                        new_status
                    )
                    
                    # Update cross-study links when hypothesis state changes
                    self.on_hypothesis_state_changed(hypothesis_id, new_status)
                
                # Update the modified timestamp
                active_study.hypotheses[i]['updated_at'] = datetime.now().isoformat()
                
                # Refresh display
                self.load_hypotheses()
                
                return True
        
        return False

    def get_hypothesis_evidence(self, hypothesis_id):
        """Get all evidence (statistical and literature) for a hypothesis."""
        if not self.studies_manager:
            return None
        
        active_study = self.studies_manager.get_active_study()
        if not active_study:
            return None
        
        # Find the hypothesis
        for hyp in active_study.hypotheses:
            if hyp.get('id') == hypothesis_id:
                evidence = {
                    'statistical': hyp.get('test_results'),
                    'literature': hyp.get('literature_evidence'),
                    'status': hyp.get('status'),
                    'updated_at': hyp.get('updated_at')
                }
                return evidence
        
        return None

    def _resolve_evidence_status(self, statistical_status, literature_status):
        """
        Resolve potential conflicts between statistical and literature evidence.
        Returns the final status to use.
        """
        # Convert to standard format for resolve_hypothesis_status
        statistical_evidence = None
        literature_evidence = None
        
        if statistical_status:
            statistical_evidence = {
                'p_value': 0.01 if statistical_status == 'confirmed' else 0.1,
                'alpha_level': 0.05
            }
            
        if literature_status:
            literature_evidence = {
                'status': literature_status
            }
            
        # Use the standardized function from chain_studies
        final_status = resolve_hypothesis_status(
            statistical_evidence=statistical_evidence,
            literature_evidence=literature_evidence
        )
        
        # Convert back to string format used by the manager
        status_map = {
            HypothesisState.PROPOSED: 'untested',
            HypothesisState.UNTESTED: 'untested',
            HypothesisState.TESTING: 'untested',
            HypothesisState.VALIDATED: 'confirmed',
            HypothesisState.REJECTED: 'rejected',
            HypothesisState.INCONCLUSIVE: 'inconclusive',
            HypothesisState.MODIFIED: 'untested'
        }
        
        return status_map.get(final_status, 'untested')

    def _add_to_history(self, hypothesis_id, field, old_value, new_value):
        """Add a change to the history."""
        change = HypothesisChange(hypothesis_id, field, old_value, new_value)
        self.change_history.append(change)
        
        # Trim history if it exceeds max size
        if len(self.change_history) > self.max_history:
            self.change_history.pop(0)
    
    def export_evidence_history(self, hypothesis_id=None):
        """
        Export the evidence history for one or all hypotheses.
        Returns a dictionary with full evidence history.
        """
        if not self.studies_manager:
            return None
            
        active_study = self.studies_manager.get_active_study()
        if not active_study:
            return None
            
        evidence_history = {
            'export_date': datetime.now().isoformat(),
            'study_id': active_study.id,
            'hypotheses': []
        }
        
        hypotheses = active_study.hypotheses
        if hypothesis_id:
            hypotheses = [h for h in hypotheses if h.get('id') == hypothesis_id]
            
        for hyp in hypotheses:
            hyp_history = {
                'id': hyp.get('id'),
                'title': hyp.get('title'),
                'statistical_evidence': {
                    'results': hyp.get('test_results'),
                    'test_date': hyp.get('test_date'),
                },
                'literature_evidence': hyp.get('literature_evidence'),
                'status_history': [
                    change.to_dict() for change in self.change_history
                    if change.hypothesis_id == hyp.get('id')
                ],
                'current_status': hyp.get('status'),
                'last_updated': hyp.get('updated_at')
            }
            evidence_history['hypotheses'].append(hyp_history)
            
        return evidence_history
    
    def show_evidence_conflicts(self, hypothesis_id):
        """Show a dialog displaying evidence analysis for both claims-based and model-based evidence."""
        print(f"show_evidence_conflicts called with hypothesis_id: {hypothesis_id}")
        if not self.studies_manager:
            return
            
        active_study = self.studies_manager.get_active_study()
        if not active_study:
            return
            
        # Find the hypothesis
        hypothesis = None
        for hyp in active_study.hypotheses:
            if hyp.get('id') == hypothesis_id:
                hypothesis = hyp
                break
                
        if not hypothesis:
            return
            
        # Create evidence analysis dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Evidence Analysis")
        dialog.setMinimumWidth(600)
        
        layout = QVBoxLayout(dialog)
        
        # Add hypothesis title
        title_label = QLabel(f"<h3>{hypothesis.get('title', 'Untitled Hypothesis')}</h3>")
        layout.addWidget(title_label)
        
        # Add hypothesis details
        details_text = f"<b>Null Hypothesis:</b> {hypothesis.get('null_hypothesis', 'Not defined')}<br>"
        details_text += f"<b>Alternative Hypothesis:</b> {hypothesis.get('alternative_hypothesis', 'Not defined')}"
        details_label = QLabel(details_text)
        details_label.setWordWrap(True)
        layout.addWidget(details_label)
        layout.addSpacing(10)
        
        # Create evidence comparison table
        table = QTableWidget(5, 3)  # 5 rows (type, status, details, strength, date), 3 columns (model-based, claims-based, final)
        table.setHorizontalHeaderLabels(["Model-Based Evidence", "Claims-Based Evidence", "Final Decision"])
        
        # Get evidence details
        model_results = hypothesis.get('test_results', {})
        claims_evidence = hypothesis.get('literature_evidence', {})
        
        # Row 0: Evidence Type
        table.setVerticalHeaderItem(0, QTableWidgetItem("Type"))
        table.setItem(0, 0, QTableWidgetItem("Statistical tests"))
        table.setItem(0, 1, QTableWidgetItem("Literature review"))
        table.setItem(0, 2, QTableWidgetItem("Combined analysis"))
        
        # Row 1: Status
        has_model = False
        has_claims = False
        
        p_value = model_results.get('p_value')
        alpha = hypothesis.get('alpha_level', 0.05)
        
        # Determine if we have valid model-based evidence
        if p_value is not None:
            has_model = True
            model_status = "Confirmed" if p_value < alpha else "Rejected"
            model_status_item = QTableWidgetItem(model_status)
            model_status_item.setForeground(QColor("#28a745" if p_value < alpha else "#dc3545"))
        else:
            model_status = "Unknown"
            model_status_item = QTableWidgetItem(model_status)
        
        # Determine if we have valid claims-based evidence
        claims_status_raw = claims_evidence.get('status')
        if claims_status_raw:
            has_claims = True
            claims_status = claims_status_raw.capitalize()
            claims_status_item = QTableWidgetItem(claims_status)
            if claims_status.lower() == "confirmed":
                claims_status_item.setForeground(QColor("#28a745"))
            elif claims_status.lower() == "rejected":
                claims_status_item.setForeground(QColor("#dc3545"))
            elif claims_status.lower() == "inconclusive":
                claims_status_item.setForeground(QColor("#fd7e14"))
        else:
            claims_status = "Unknown"
            claims_status_item = QTableWidgetItem(claims_status)
        
        final_status = hypothesis.get('status', 'Unknown').capitalize()
        final_status_item = QTableWidgetItem(final_status)
        if final_status.lower() == "confirmed":
            final_status_item.setForeground(QColor("#28a745"))
        elif final_status.lower() == "rejected":
            final_status_item.setForeground(QColor("#dc3545"))
        elif final_status.lower() == "inconclusive":
            final_status_item.setForeground(QColor("#fd7e14"))
        
        table.setVerticalHeaderItem(1, QTableWidgetItem("Status"))
        table.setItem(1, 0, model_status_item)
        table.setItem(1, 1, claims_status_item)
        table.setItem(1, 2, final_status_item)
        
        # Row 2: Details
        model_details = model_results.get('test', 'Unknown test')
        if p_value is not None:
            model_details += f" (p={p_value:.4f})"
        
        claims_details = ""
        if has_claims:
            claims_support = claims_evidence.get('supporting', 0)
            claims_refute = claims_evidence.get('refuting', 0)
            claims_neutral = claims_evidence.get('neutral', 0)
            claims_details = f"Supporting: {claims_support}, Refuting: {claims_refute}, Neutral: {claims_neutral}"
        else:
            claims_details = "No literature evidence"
        
        final_details = "Based on combined evidence"
        if has_model and has_claims:
            if model_status.lower() == claims_status.lower():
                final_details = "Both evidence types agree"
            else:
                final_details = "Evidence types conflict - see analysis below"
        elif has_model:
            final_details = "Based on model evidence only"
        elif has_claims:
            final_details = "Based on claims evidence only"
        
        table.setVerticalHeaderItem(2, QTableWidgetItem("Details"))
        table.setItem(2, 0, QTableWidgetItem(model_details))
        table.setItem(2, 1, QTableWidgetItem(claims_details))
        table.setItem(2, 2, QTableWidgetItem(final_details))
        
        # Row 3: Strength/Confidence
        model_strength = ""
        if p_value is not None:
            effect_size = model_results.get('effect_size')
            if effect_size is not None:
                model_strength = f"Effect size: {effect_size:.4f}"
            else:
                model_strength = "Strength not reported"
        else:
            model_strength = "Unknown"
        
        claims_strength = "Unknown"
        if has_claims:
            # Calculate ratio if we have enough data
            total_studies = sum([
                claims_evidence.get('supporting', 0),
                claims_evidence.get('refuting', 0),
                claims_evidence.get('neutral', 0)
            ])
            if total_studies > 0:
                support_ratio = claims_evidence.get('supporting', 0) / total_studies
                claims_strength = f"Support ratio: {support_ratio:.2f} ({total_studies} studies)"
        
        final_strength = "Based on available evidence"
        
        table.setVerticalHeaderItem(3, QTableWidgetItem("Strength"))
        table.setItem(3, 0, QTableWidgetItem(model_strength))
        table.setItem(3, 1, QTableWidgetItem(claims_strength))
        table.setItem(3, 2, QTableWidgetItem(final_strength))
        
        # Row 4: Date
        model_date = hypothesis.get('test_date', 'Not tested')
        claims_date = claims_evidence.get('updated_at', 'No date')
        final_date = hypothesis.get('updated_at', 'Unknown')
        
        table.setVerticalHeaderItem(4, QTableWidgetItem("Date"))
        table.setItem(4, 0, QTableWidgetItem(model_date))
        table.setItem(4, 1, QTableWidgetItem(claims_date))
        table.setItem(4, 2, QTableWidgetItem(final_date))
        
        # Adjust table properties
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        
        layout.addWidget(table)
        
        # Add conclusion section
        conclusion_group = QGroupBox("Evidence Analysis")
        conclusion_layout = QVBoxLayout(conclusion_group)
        
        # Only show conflict warning if both evidence types exist and they disagree
        if has_model and has_claims and model_status.lower() != claims_status.lower():
            warning_label = QLabel(
                "<p style='color: #dc3545;'><b>⚠️ Conflict Detected:</b> "
                "Model-based and claims-based evidence suggest different conclusions.</p>"
            )
            conclusion_layout.addWidget(warning_label)
            
            # Add potential reasons for conflict
            reasons_label = QLabel("<p><b>Potential reasons for conflict:</b></p>"
                                  "<ul>"
                                  "<li>Statistical power issues in model-based evidence</li>"
                                  "<li>Publication bias in claims-based evidence</li>"
                                  "<li>Different populations or contexts</li>"
                                  "<li>Methodological differences</li>"
                                  "</ul>")
            conclusion_layout.addWidget(reasons_label)
            
        elif not has_model and not has_claims:
            info_label = QLabel(
                "<p style='color: #6c757d;'><i>No evidence available for this hypothesis. "
                "Consider adding both model-based (statistical tests) and claims-based (literature) evidence.</i></p>"
            )
            conclusion_layout.addWidget(info_label)
        elif has_model and not has_claims:
            info_label = QLabel(
                "<p style='color: #0275d8;'><i>Only model-based evidence is available. "
                "Consider adding claims-based evidence from literature review for a more complete analysis.</i></p>"
            )
            conclusion_layout.addWidget(info_label)
        elif not has_model and has_claims:
            info_label = QLabel(
                "<p style='color: #0275d8;'><i>Only claims-based evidence is available. "
                "Consider adding model-based evidence from statistical tests for a more complete analysis.</i></p>"
            )
            conclusion_layout.addWidget(info_label)
        else:
            # Both exist and agree
            info_label = QLabel(
                "<p style='color: #28a745;'><b>✓ Evidence Aligned:</b> "
                "Both model-based and claims-based evidence support the same conclusion, "
                "strengthening overall confidence in the result.</p>"
            )
            conclusion_layout.addWidget(info_label)
        
        # Add specific conclusion from literature if available
        if has_claims and claims_evidence.get('conclusion'):
            lit_conclusion = QLabel(f"<p><b>Claims-based conclusion:</b> {claims_evidence.get('conclusion')}</p>")
            lit_conclusion.setWordWrap(True)
            conclusion_layout.addWidget(lit_conclusion)
            
        # Add specific conclusion from model if available
        if has_model and model_results.get('conclusion'):
            model_conclusion = QLabel(f"<p><b>Model-based conclusion:</b> {model_results.get('conclusion')}</p>")
            model_conclusion.setWordWrap(True)
            conclusion_layout.addWidget(model_conclusion)
        
        layout.addWidget(conclusion_group)
        
        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(dialog.reject)
        
        # Add export button
        export_btn = QPushButton("Export Evidence History")
        export_btn.clicked.connect(lambda: self._export_evidence_dialog(hypothesis_id))
        button_box.addButton(export_btn, QDialogButtonBox.ButtonRole.ActionRole)
        
        layout.addWidget(button_box)
        
        dialog.exec()
    
    def _export_evidence_dialog(self, hypothesis_id):
        """Show dialog to export evidence history."""
        evidence_history = self.export_evidence_history(hypothesis_id)
        if not evidence_history:
            return
            
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export Evidence History",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_name:
            try:
                with open(file_name, 'w') as f:
                    json.dump(evidence_history, f, indent=2)
                QMessageBox.information(self, "Success", "Evidence history exported successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export evidence history: {str(e)}")
    
    def undo_last_change(self, hypothesis_id):
        """Undo the last change for a specific hypothesis."""
        if not self.change_history:
            return False
            
        # Find the last change for this hypothesis
        for i in reversed(range(len(self.change_history))):
            change = self.change_history[i]
            if change.hypothesis_id == hypothesis_id:
                # Revert the change
                if not self.studies_manager:
                    return False
                    
                active_study = self.studies_manager.get_active_study()
                if not active_study:
                    return False
                    
                # Find the hypothesis
                for j, hyp in enumerate(active_study.hypotheses):
                    if hyp.get('id') == hypothesis_id:
                        # Revert the field to its old value
                        active_study.hypotheses[j][change.field] = change.old_value
                        active_study.hypotheses[j]['updated_at'] = datetime.now().isoformat()
                        
                        # Remove the change from history
                        self.change_history.pop(i)
                        
                        # Refresh display
                        self.load_hypotheses()
                        return True
                        
        return False

    def create_multi_study_demo(self):
        """
        Create a set of related hypotheses with shared variables across multiple studies
        for demonstration purposes.
        """
        if not self.studies_manager:
            QMessageBox.warning(self, "Error", "No studies manager available")
            return
            
        # Get all existing studies
        studies = self.studies_manager.list_studies()
        
        if not studies or len(studies) < 2:
            # Create at least two studies if they don't exist
            QMessageBox.information(
                self, 
                "Creating Demo Studies", 
                "Creating multiple studies to demonstrate hypothesis sharing...",
            )
            
            # Create studies
            study_names = [
                "Cognitive Effects of Exercise",
                "Neural Correlates of Exercise",
                "Long-term Effects of Physical Activity"
            ]
            
            # We'll create a new project for our demo studies
            active_project = self.studies_manager.get_active_project()
            if not active_project:
                # Create a new project for the demo
                project = self.studies_manager.create_project("Exercise and Cognition Project", 
                                                             "A multi-study project examining the effects of exercise on cognition")
            
            # Create a simple study design for each
            from study_model.study_model import StudyDesign
            
            for name in study_names:
                study_design = StudyDesign(
                    study_id=str(uuid.uuid4()),
                    title=name
                )
                self.studies_manager.create_study(name, study_design)
                
            # Get the newly created studies
            studies = self.studies_manager.list_studies()
            
        # Define shared variables
        shared_vars = {
            "exercise_intensity": {
                "description": "Intensity of physical exercise (low, moderate, high)",
                "type": "independent"
            },
            "exercise_duration": {
                "description": "Duration of exercise sessions in minutes",
                "type": "independent"
            },
            "cognitive_performance": {
                "description": "Scores on cognitive performance tests",
                "type": "dependent"
            },
            "neural_activation": {
                "description": "Neural activation patterns from fMRI",
                "type": "dependent"
            },
            "cortisol_levels": {
                "description": "Salivary cortisol levels",
                "type": "dependent"
            },
            "participant_age": {
                "description": "Age of study participants",
                "type": "control"
            }
        }
        
        # List of related hypotheses
        hypotheses_by_study = {
            0: [  # First study - Cognitive focus
                {
                    "title": "Exercise Intensity and Cognitive Performance",
                    "null_hypothesis": "There is no relationship between exercise intensity and cognitive performance.",
                    "alternative_hypothesis": "Higher exercise intensity is associated with improved cognitive performance.",
                    "directionality": "greater",
                    "outcome_variables": "cognitive_performance",
                    "predictor_variables": "exercise_intensity",
                    "expected_test": "Linear Regression",
                    "status": "untested"
                },
                {
                    "title": "Exercise Duration and Cognitive Performance",
                    "null_hypothesis": "Exercise duration has no effect on cognitive performance.",
                    "alternative_hypothesis": "Longer exercise duration improves cognitive performance up to a threshold.",
                    "directionality": "greater",
                    "outcome_variables": "cognitive_performance",
                    "predictor_variables": "exercise_duration",
                    "expected_test": "Linear Regression",
                    "status": "untested"
                }
            ],
            1: [  # Second study - Neural focus
                {
                    "title": "Exercise Intensity and Neural Activation",
                    "null_hypothesis": "Exercise intensity does not affect neural activation patterns.",
                    "alternative_hypothesis": "Higher exercise intensity increases neural activation in prefrontal areas.",
                    "directionality": "greater",
                    "outcome_variables": "neural_activation",
                    "predictor_variables": "exercise_intensity",
                    "expected_test": "ANOVA",
                    "status": "untested"
                },
                {
                    "title": "Cortisol and Neural Activation Relationship",
                    "null_hypothesis": "There is no relationship between cortisol levels and neural activation during exercise.",
                    "alternative_hypothesis": "Cortisol levels are negatively correlated with neural activation in the hippocampus.",
                    "directionality": "less",
                    "outcome_variables": "neural_activation",
                    "predictor_variables": "cortisol_levels",
                    "expected_test": "Pearson Correlation",
                    "status": "untested"
                }
            ],
            2: [  # Third study - Long-term focus
                {
                    "title": "Long-term Exercise and Cognitive Performance",
                    "null_hypothesis": "Long-term exercise has no effect on cognitive performance.",
                    "alternative_hypothesis": "Regular exercise over 6 months improves cognitive performance.",
                    "directionality": "greater",
                    "outcome_variables": "cognitive_performance",
                    "predictor_variables": "exercise_duration",
                    "expected_test": "Repeated Measures ANOVA",
                    "status": "untested"
                },
                {
                    "title": "Age, Exercise, and Cognitive Benefits",
                    "null_hypothesis": "Age does not moderate the relationship between exercise and cognitive benefits.",
                    "alternative_hypothesis": "Older adults show greater cognitive benefits from regular exercise than younger adults.",
                    "directionality": "greater",
                    "outcome_variables": "cognitive_performance",
                    "predictor_variables": "exercise_intensity, participant_age",
                    "expected_test": "Multiple Regression",
                    "status": "untested"
                }
            ]
        }
        
        # Add literature evidence and statistical evidence for some hypotheses
        
        # Literature evidence sample format
        literature_evidence_samples = [
            {
                "study_idx": 0,
                "hyp_idx": 0,
                "evidence": {
                    "supporting": 3,
                    "refuting": 1,
                    "neutral": 1,
                    "status": "validated",
                    "conclusion": "Meta-analyses support the positive effect of exercise intensity on cognition"
                }
            },
            {
                "study_idx": 1,
                "hyp_idx": 0,
                "evidence": {
                    "supporting": 2,
                    "refuting": 2,
                    "neutral": 3,
                    "status": "inconclusive",
                    "conclusion": "Mixed results on exercise intensity and neural activation"
                }
            },
            {
                "study_idx": 2,
                "hyp_idx": 1,
                "evidence": {
                    "supporting": 5,
                    "refuting": 0,
                    "neutral": 1,
                    "status": "validated",
                    "conclusion": "Strong evidence for age-related effects on exercise benefits"
                }
            }
        ]
        
        # Statistical evidence sample format
        statistical_evidence_samples = [
            {
                "study_idx": 0,
                "hyp_idx": 1,
                "evidence": {
                    "test": "Linear Regression",
                    "p_value": 0.032,
                    "statistic": 2.28,
                    "df": 38,
                    "effect_size": 0.42,
                    "ci_lower": 0.05,
                    "ci_upper": 0.78,
                    "conclusion": "Significant positive relationship between exercise duration and cognitive performance"
                }
            },
            {
                "study_idx": 1,
                "hyp_idx": 1,
                "evidence": {
                    "test": "Pearson Correlation",
                    "p_value": 0.068,
                    "statistic": -0.31,
                    "df": 24,
                    "effect_size": 0.31,
                    "ci_lower": -0.62,
                    "ci_upper": 0.02,
                    "conclusion": "Trend towards negative correlation between cortisol and neural activation"
                }
            }
        ]
        
        # Shared evidence for hypothesis pairs
        # For example, add evidence from study 1 to a hypothesis in study 2
        cross_study_evidence = [
            {
                "from_study_idx": 0,
                "from_hyp_idx": 0,
                "to_study_idx": 2,
                "to_hyp_idx": 0,
                "evidence_type": "supporting",
                "description": "Initial study found significant effects of exercise intensity on cognitive performance (p=0.032)",
                "source": "Cross-study reference",
                "confidence": 0.75
            },
            {
                "from_study_idx": 1,
                "from_hyp_idx": 0,
                "to_study_idx": 0,
                "to_hyp_idx": 0,
                "evidence_type": "supporting",
                "description": "Neural mechanisms observed in neuroimaging study support cognitive performance improvements",
                "source": "Cross-study reference",
                "confidence": 0.8
            }
        ]
        
        created_hypotheses = []
        
        # Add the hypotheses to each study
        for study_idx, hyp_list in hypotheses_by_study.items():
            if study_idx < len(studies):
                study_info = studies[study_idx]
                study_id = study_info["id"]
                self.studies_manager.set_active_study(study_id)
                
                for hyp_idx, hyp_data in enumerate(hyp_list):
                    # Ensure all required fields are present
                    hypothesis_data = hyp_data.copy()
                    
                    # Add required fields if not present
                    if 'id' not in hypothesis_data:
                        hypothesis_data['id'] = str(uuid.uuid4())
                    if 'created_at' not in hypothesis_data:
                        hypothesis_data['created_at'] = datetime.now().isoformat()
                    if 'updated_at' not in hypothesis_data:
                        hypothesis_data['updated_at'] = datetime.now().isoformat()
                    
                    # These fields are required by the updated API
                    if 'null_hypothesis' not in hypothesis_data:
                        hypothesis_data['null_hypothesis'] = "No effect"
                    if 'alternative_hypothesis' not in hypothesis_data:
                        hypothesis_data['alternative_hypothesis'] = hypothesis_data.get('title', "Unknown hypothesis")
                    
                    # Additional fields for testing
                    hypothesis_data["alpha_level"] = 0.05
                    hypothesis_data["notes"] = f"Part of multi-study research project on exercise effects"
                    
                    # Get the hypothesis text to use as title if missing
                    if 'title' not in hypothesis_data:
                        hypothesis_data['title'] = hypothesis_data.get('alternative_hypothesis', "Unnamed Hypothesis")
                    
                    # Add hypothesis to the current active study
                    hypothesis_id = self.studies_manager.add_hypothesis_to_study(
                        hypothesis_text=hypothesis_data['alternative_hypothesis'],
                        hypothesis_data=hypothesis_data
                    )
                    
                    # Check if successful
                    if not hypothesis_id:
                        print(f"Failed to add hypothesis: {hypothesis_data['title']}")
                        continue
                    
                    # Store created hypothesis reference with the returned ID
                    created_hypotheses.append({
                        "study_idx": study_idx,
                        "hyp_idx": hyp_idx,
                        "study_id": study_id,
                        "hypothesis_id": hypothesis_id
                    })
        
        # Add literature evidence
        for lit_ev in literature_evidence_samples:
            study_idx = lit_ev["study_idx"]
            hyp_idx = lit_ev["hyp_idx"]
            
            # Find the target hypothesis from our created references
            hyp_ref = None
            for ref in created_hypotheses:
                if ref["study_idx"] == study_idx and ref["hyp_idx"] == hyp_idx:
                    hyp_ref = ref
                    break
            
            if hyp_ref:
                # Activate the study containing the hypothesis
                self.studies_manager.set_active_study(hyp_ref["study_id"])
                
                # Add the literature evidence using the stored hypothesis ID
                result = self.studies_manager.update_hypothesis_with_literature(
                    hypothesis_id=hyp_ref["hypothesis_id"],
                    literature_evidence=lit_ev["evidence"]
                )
                
                if not result:
                    print(f"Warning: Failed to add literature evidence to hypothesis ID {hyp_ref['hypothesis_id']}")
            else:
                print(f"Warning: Could not find hypothesis reference for literature evidence (study_idx: {study_idx}, hyp_idx: {hyp_idx})")
        
        # Add statistical evidence
        for stat_ev in statistical_evidence_samples:
            study_idx = stat_ev["study_idx"]
            hyp_idx = stat_ev["hyp_idx"]
            
            # Find the target hypothesis from our created references
            hyp_ref = None
            for ref in created_hypotheses:
                if ref["study_idx"] == study_idx and ref["hyp_idx"] == hyp_idx:
                    hyp_ref = ref
                    break
            
            if hyp_ref:
                # Activate the study containing the hypothesis
                self.studies_manager.set_active_study(hyp_ref["study_id"])
                
                # Add the statistical evidence using the stored hypothesis ID
                result = self.studies_manager.update_hypothesis_with_test_results(
                    hypothesis_id=hyp_ref["hypothesis_id"],
                    test_results=stat_ev["evidence"]
                )
                
                if not result:
                    print(f"Warning: Failed to add statistical evidence to hypothesis ID {hyp_ref['hypothesis_id']}")
            else:
                print(f"Warning: Could not find hypothesis reference for statistical evidence (study_idx: {study_idx}, hyp_idx: {hyp_idx})")
        
        # Add cross-study evidence
        for cross_ev in cross_study_evidence:
            from_study_idx = cross_ev["from_study_idx"]
            from_hyp_idx = cross_ev["from_hyp_idx"]
            to_study_idx = cross_ev["to_study_idx"]
            to_hyp_idx = cross_ev["to_hyp_idx"]
            
            # Find the source hypothesis from our created references
            from_hyp_ref = None
            for ref in created_hypotheses:
                if ref["study_idx"] == from_study_idx and ref["hyp_idx"] == from_hyp_idx:
                    from_hyp_ref = ref
                    break
                    
            # Find the target hypothesis from our created references
            to_hyp_ref = None
            for ref in created_hypotheses:
                if ref["study_idx"] == to_study_idx and ref["hyp_idx"] == to_hyp_idx:
                    to_hyp_ref = ref
                    break
            
            # If we found both hypotheses, proceed with linking them
            if from_hyp_ref and to_hyp_ref:
                # Get the source hypothesis
                self.studies_manager.set_active_study(from_hyp_ref["study_id"])
                from_hyp = self.studies_manager.get_hypothesis(from_hyp_ref["hypothesis_id"])
                
                # Get the target hypothesis
                self.studies_manager.set_active_study(to_hyp_ref["study_id"])
                to_hyp = self.studies_manager.get_hypothesis(to_hyp_ref["hypothesis_id"])
                
                if from_hyp and to_hyp:
                    # Create evidence object
                    evidence = {
                        "id": str(uuid.uuid4()),
                        "type": "literature",  # Cross-study references are treated as literature
                        "description": cross_ev["description"],
                        "source": cross_ev["source"],
                        "confidence": cross_ev["confidence"],
                        "cross_reference": {
                            "study_id": from_hyp_ref["study_id"],
                            "hypothesis_id": from_hyp["id"],
                            "hypothesis_title": from_hyp.get("title", "Unknown")
                        }
                    }
                    
                    # Update with the new evidence
                    evidence_type = cross_ev["evidence_type"]
                    if evidence_type == "supporting":
                        supporting_evidence = to_hyp.get("supporting_evidence", [])
                        supporting_evidence.append(evidence)
                        update_data = {"supporting_evidence": supporting_evidence}
                    else:
                        contradicting_evidence = to_hyp.get("contradicting_evidence", [])
                        contradicting_evidence.append(evidence)
                        update_data = {"contradicting_evidence": contradicting_evidence}
                    
                    # Update the hypothesis
                    result = self.studies_manager.update_hypothesis(
                        hypothesis_id=to_hyp["id"],
                        update_data=update_data
                    )
                    if not result:
                        print(f"Warning: Failed to update hypothesis with ID {to_hyp['id']}")
                else:
                    print(f"Warning: Could not retrieve one of the hypotheses for cross-study linking")
            else:
                print(f"Warning: Could not find cross-study reference hypothesis in created hypotheses")
        
        # Set the active study to the first one
        if studies:
            self.studies_manager.set_active_study(studies[0]["id"])
            
        # Load hypotheses to display
        self.load_hypotheses()
        
        # Show a message with instructions
        QMessageBox.information(
            self, 
            "Multi-Study Demo Created",
            f"Created {sum(len(hyps) for hyps in hypotheses_by_study.values())} hypotheses across {len(studies)} studies.\n\n"
            f"The hypotheses share variables, literature evidence, and cross-references.\n\n"
            f"Use the study selector to switch between studies and examine hypotheses."
        )

    def update_cross_study_links(self, hypothesis_id=None):
        """
        Update cross-reference links between hypotheses across studies
        to reflect their current states. This ensures cross-referenced
        evidence is consistent with the current state of each hypothesis.
        
        Args:
            hypothesis_id: Optional ID of specific hypothesis to update links for.
                          If None, updates all cross-referenced hypotheses.
        """
        if not self.studies_manager:
            return
            
        # Get all studies
        all_studies = self.studies_manager.list_studies()
        if not all_studies:
            return
            
        # Track all hypotheses with cross-references
        cross_references = {}
        
        # First pass: collect all hypotheses with cross-references
        for study_info in all_studies:
            study_id = study_info["id"]
            self.studies_manager.set_active_study(study_id)
            study_hypotheses = self.studies_manager.get_study_hypotheses()
            
            for hyp in study_hypotheses:
                # If we're only updating a specific hypothesis
                if hypothesis_id and hyp["id"] != hypothesis_id:
                    continue
                    
                # Check for cross-references in supporting evidence
                for evidence in hyp.get("supporting_evidence", []):
                    if "cross_reference" in evidence:
                        source_study_id = evidence["cross_reference"]["study_id"]
                        source_hyp_id = evidence["cross_reference"]["hypothesis_id"]
                        key = f"{source_study_id}:{source_hyp_id}"
                        
                        if key not in cross_references:
                            cross_references[key] = []
                            
                        cross_references[key].append({
                            "target_study_id": study_id,
                            "target_hyp_id": hyp["id"],
                            "evidence_id": evidence["id"],
                            "type": "supporting"
                        })
                
                # Check for cross-references in contradicting evidence
                for evidence in hyp.get("contradicting_evidence", []):
                    if "cross_reference" in evidence:
                        source_study_id = evidence["cross_reference"]["study_id"]
                        source_hyp_id = evidence["cross_reference"]["hypothesis_id"]
                        key = f"{source_study_id}:{source_hyp_id}"
                        
                        if key not in cross_references:
                            cross_references[key] = []
                            
                        cross_references[key].append({
                            "target_study_id": study_id,
                            "target_hyp_id": hyp["id"],
                            "evidence_id": evidence["id"],
                            "type": "contradicting"
                        })
        
        # Second pass: update each cross-reference based on source hypothesis state
        for source_key, references in cross_references.items():
            source_study_id, source_hyp_id = source_key.split(":")
            
            # Get the source hypothesis
            self.studies_manager.set_active_study(source_study_id)
            source_hyp = self.studies_manager.get_hypothesis(source_hyp_id)
            
            if not source_hyp:
                continue
                
            # Get the current source hypothesis state
            current_state = source_hyp.get("status", "untested")
            
            # Update all targets referencing this source
            for ref in references:
                self.studies_manager.set_active_study(ref["target_study_id"])
                target_hyp = self.studies_manager.get_hypothesis(ref["target_hyp_id"])
                
                if not target_hyp:
                    continue
                    
                # Determine if the evidence type should change based on the source's state
                current_type = ref["type"]
                correct_type = self._determine_evidence_type(current_state, source_hyp, target_hyp)
                
                if current_type != correct_type:
                    # We need to move this evidence to the other list
                    # First, get both evidence lists
                    supporting_evidence = target_hyp.get("supporting_evidence", [])
                    contradicting_evidence = target_hyp.get("contradicting_evidence", [])
                    
                    # Find the evidence in the current list
                    evidence_to_move = None
                    evidence_index = -1
                    
                    if current_type == "supporting":
                        for i, evidence in enumerate(supporting_evidence):
                            if evidence.get("id") == ref["evidence_id"]:
                                evidence_to_move = evidence
                                evidence_index = i
                                break
                                
                        if evidence_to_move and evidence_index >= 0:
                            # Remove from supporting
                            supporting_evidence.pop(evidence_index)
                            # Add to contradicting
                            evidence_to_move["type"] = "contradicting"
                            contradicting_evidence.append(evidence_to_move)
                    else:
                        for i, evidence in enumerate(contradicting_evidence):
                            if evidence.get("id") == ref["evidence_id"]:
                                evidence_to_move = evidence
                                evidence_index = i
                                break
                                
                        if evidence_to_move and evidence_index >= 0:
                            # Remove from contradicting
                            contradicting_evidence.pop(evidence_index)
                            # Add to supporting
                            evidence_to_move["type"] = "supporting"
                            supporting_evidence.append(evidence_to_move)
                    
                    # Update the hypothesis
                    update_data = {
                        "supporting_evidence": supporting_evidence,
                        "contradicting_evidence": contradicting_evidence
                    }
                    self.studies_manager.update_hypothesis(
                        hypothesis_id=target_hyp["id"],
                        update_data=update_data
                    )

    def _determine_evidence_type(self, source_state, source_hyp, target_hyp):
        """
        Determine whether a cross-reference should be supporting or contradicting
        based on the current states of source and target hypotheses.
        
        Args:
            source_state: Current state of the source hypothesis
            source_hyp: The source hypothesis data
            target_hyp: The target hypothesis data
            
        Returns:
            str: "supporting" or "contradicting"
        """
        # Get the directionality of both hypotheses
        source_dir = source_hyp.get("directionality", "greater")
        target_dir = target_hyp.get("directionality", "greater")
        
        # If the source is validated/confirmed
        if source_state in ["validated", "confirmed"]:
            # If directionalities match, it's supporting
            if source_dir == target_dir:
                return "supporting"
            else:
                return "contradicting"
                
        # If the source is rejected/refuted
        elif source_state in ["rejected", "refuted"]:
            # If directionalities match, it's contradicting (rejected same direction)
            if source_dir == target_dir:
                return "contradicting"
            else:
                return "supporting"
                
        # For other states, keep the current type
        return "supporting"
        
    def on_hypothesis_state_changed(self, hypothesis_id, new_state):
        """
        Called when a hypothesis state changes to update cross-references.
        
        Args:
            hypothesis_id: The ID of the hypothesis that changed
            new_state: The new state of the hypothesis
        """
        # Update any cross-references involving this hypothesis
        self.update_cross_study_links(hypothesis_id)
        
        # Reload the display to show the updated links
        self.load_hypotheses()

    def show_cross_study_linking(self):
        """Show dialog for linking hypotheses across studies."""
        if not self.studies_manager:
            QMessageBox.warning(self, "Error", "Studies manager not available")
            return
            
        # Get all studies
        studies = self.studies_manager.list_studies()
        if len(studies) < 2:
            QMessageBox.warning(
                self, 
                "Not Enough Studies", 
                "You need at least 2 studies to create cross-study links. Currently you have " + 
                str(len(studies)) + " study/studies."
            )
            return
            
        # Create a dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Cross-Study Hypothesis Linking")
        dialog.setMinimumSize(900, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Create tabs for different linking methods
        tabs = QTabWidget()
        
        # Manual linking tab
        manual_tab = QWidget()
        manual_layout = QVBoxLayout(manual_tab)
        
        # Instructions
        instructions = QLabel(
            "<p>Manually link hypotheses across different studies to show relationships "
            "and track evidence propagation.</p>"
            "<p>Select source and target studies, then choose hypotheses to link.</p>"
        )
        instructions.setWordWrap(True)
        manual_layout.addWidget(instructions)
        
        # Source and target study selection
        selection_layout = QHBoxLayout()
        
        # Source study
        source_group = QGroupBox("Source Study")
        source_layout = QVBoxLayout(source_group)
        
        source_selector = QComboBox()
        for study in studies:
            source_selector.addItem(study["name"], study["id"])
        source_layout.addWidget(source_selector)
        
        source_hyp_list = QListWidget()
        source_layout.addWidget(source_hyp_list, 1)
        
        selection_layout.addWidget(source_group)
        
        # Target study
        target_group = QGroupBox("Target Study")
        target_layout = QVBoxLayout(target_group)
        
        target_selector = QComboBox()
        for study in studies:
            target_selector.addItem(study["name"], study["id"])
        target_layout.addWidget(target_selector)
        
        target_hyp_list = QListWidget()
        target_layout.addWidget(target_hyp_list, 1)
        
        selection_layout.addWidget(target_group)
        
        manual_layout.addLayout(selection_layout)
        
        # Linking type
        link_group = QGroupBox("Link Properties")
        link_layout = QFormLayout(link_group)
        
        link_type = QComboBox()
        link_type.addItems(["Supporting Evidence", "Contradicting Evidence"])
        link_layout.addRow("Link Type:", link_type)
        
        confidence = QSlider(Qt.Orientation.Horizontal)
        confidence.setMinimum(0)
        confidence.setMaximum(100)
        confidence.setValue(70)
        confidence.setTickPosition(QSlider.TickPosition.TicksBelow)
        confidence.setTickInterval(10)
        confidence_label = QLabel("70%")
        confidence.valueChanged.connect(lambda v: confidence_label.setText(f"{v}%"))
        
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(confidence)
        conf_layout.addWidget(confidence_label)
        link_layout.addRow("Confidence:", conf_layout)
        
        description = QTextEdit()
        description.setMaximumHeight(100)
        description.setPlaceholderText("Describe how this evidence relates to the target hypothesis...")
        link_layout.addRow("Description:", description)
        
        manual_layout.addWidget(link_group)
        
        # Create link button
        create_link_btn = QPushButton("Create Link")
        create_link_btn.clicked.connect(lambda: self._create_manual_link(
            source_selector.currentData(),
            source_hyp_list.currentItem().data(Qt.ItemDataRole.UserRole) if source_hyp_list.currentItem() else None,
            target_selector.currentData(),
            target_hyp_list.currentItem().data(Qt.ItemDataRole.UserRole) if target_hyp_list.currentItem() else None,
            link_type.currentText(),
            confidence.value() / 100.0,
            description.toPlainText()
        ))
        manual_layout.addWidget(create_link_btn)
        
        tabs.addTab(manual_tab, "Manual Linking")
        
        # AI-assisted linking tab
        ai_tab = QWidget()
        ai_layout = QVBoxLayout(ai_tab)
        
        # Instructions
        ai_instructions = QLabel(
            "<p>Use AI to automatically identify potential links between hypotheses across studies.</p>"
            "<p>Select which studies to include in the analysis, then click 'Find Potential Links'.</p>"
        )
        ai_instructions.setWordWrap(True)
        ai_layout.addWidget(ai_instructions)
        
        # Study selection
        study_selection_group = QGroupBox("Studies to Include")
        study_selection_layout = QVBoxLayout(study_selection_group)
        
        study_list = QListWidget()
        study_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for study in studies:
            item = QListWidgetItem(study["name"])
            item.setData(Qt.ItemDataRole.UserRole, study["id"])
            study_list.addItem(item)
        study_selection_layout.addWidget(study_list)
        
        ai_layout.addWidget(study_selection_group)
        
        # AI settings
        ai_settings_group = QGroupBox("Analysis Settings")
        ai_settings_layout = QFormLayout(ai_settings_group)
        
        min_confidence = QSlider(Qt.Orientation.Horizontal)
        min_confidence.setMinimum(0)
        min_confidence.setMaximum(100)
        min_confidence.setValue(50)
        min_confidence.setTickPosition(QSlider.TickPosition.TicksBelow)
        min_confidence.setTickInterval(10)
        min_conf_label = QLabel("50%")
        min_confidence.valueChanged.connect(lambda v: min_conf_label.setText(f"{v}%"))
        
        min_conf_layout = QHBoxLayout()
        min_conf_layout.addWidget(min_confidence)
        min_conf_layout.addWidget(min_conf_label)
        ai_settings_layout.addRow("Minimum Confidence:", min_conf_layout)
        
        analyze_btn = QPushButton("Find Potential Links")
        analyze_btn.clicked.connect(lambda: self._analyze_cross_study_links(
            [study_list.item(i).data(Qt.ItemDataRole.UserRole) 
             for i in range(study_list.count()) 
             if study_list.item(i).isSelected()],
            min_confidence.value() / 100.0
        ))
        
        ai_layout.addWidget(ai_settings_group)
        ai_layout.addWidget(analyze_btn)
        
        # Results area
        results_group = QGroupBox("Potential Links")
        results_layout = QVBoxLayout(results_group)
        
        self.ai_results_list = QListWidget()
        results_layout.addWidget(self.ai_results_list)
        
        apply_selected_btn = QPushButton("Apply Selected Links")
        apply_selected_btn.clicked.connect(self._apply_ai_links)
        results_layout.addWidget(apply_selected_btn)
        
        ai_layout.addWidget(results_group)
        
        tabs.addTab(ai_tab, "AI-Assisted Linking")
        
        layout.addWidget(tabs)
        
        # Load hypotheses when study selections change
        def load_source_hypotheses():
            source_hyp_list.clear()
            study_id = source_selector.currentData()
            self.studies_manager.set_active_study(study_id)
            hypotheses = self.studies_manager.get_study_hypotheses()
            
            for hyp in hypotheses:
                item = QListWidgetItem(hyp.get('title', 'Untitled Hypothesis'))
                item.setData(Qt.ItemDataRole.UserRole, hyp)
                status = hyp.get('status', 'untested')
                if status == 'confirmed':
                    item.setIcon(load_bootstrap_icon("check-circle-fill", color="#28a745"))
                elif status == 'rejected':
                    item.setIcon(load_bootstrap_icon("x-circle-fill", color="#dc3545"))
                else:
                    item.setIcon(load_bootstrap_icon("question-circle", color="#6c757d"))
                source_hyp_list.addItem(item)
        

        def load_target_hypotheses():
            target_hyp_list.clear()
            study_id = target_selector.currentData()
            self.studies_manager.set_active_study(study_id)
            hypotheses = self.studies_manager.get_study_hypotheses()
            
            for hyp in hypotheses:
                item = QListWidgetItem(hyp.get('title', 'Untitled Hypothesis'))
                item.setData(Qt.ItemDataRole.UserRole, hyp)
                status = hyp.get('status', 'untested')
                if status == 'confirmed':
                    item.setIcon(load_bootstrap_icon("check-circle-fill", color="#28a745"))
                elif status == 'rejected':
                    item.setIcon(load_bootstrap_icon("x-circle-fill", color="#dc3545"))
                else:
                    item.setIcon(load_bootstrap_icon("question-circle", color="#6c757d"))
                target_hyp_list.addItem(item)
        
        # Initial load
        load_source_hypotheses()
        load_target_hypotheses()
        
        # Connect signals
        source_selector.currentIndexChanged.connect(load_source_hypotheses)
        target_selector.currentIndexChanged.connect(load_target_hypotheses)
        
        # Reset active study when done
        active_study_id = self.studies_manager.get_active_study().id
        
        # Restore active study when dialog closes
        def restore_active_study():
            self.studies_manager.set_active_study(active_study_id)
            self.load_hypotheses()
        
        dialog.finished.connect(restore_active_study)
        
        # Show dialog
        dialog.exec()
    
    def _create_manual_link(self, source_study_id, source_hyp, target_study_id, target_hyp, 
                          link_type, confidence, description):
        """Create a manual link between two hypotheses."""
        if not source_hyp or not target_hyp:
            QMessageBox.warning(self, "Error", "Please select both source and target hypotheses")
            return
            
        if source_study_id == target_study_id and source_hyp['id'] == target_hyp['id']:
            QMessageBox.warning(self, "Error", "Cannot link a hypothesis to itself")
            return
            
        try:
            # Create evidence object
            evidence = {
                "id": str(uuid.uuid4()),
                "type": "literature",
                "description": description,
                "source": "Cross-study reference",
                "confidence": confidence,
                "cross_reference": {
                    "study_id": source_study_id,
                    "hypothesis_id": source_hyp['id'],
                    "hypothesis_title": source_hyp.get('title', 'Unknown')
                }
            }
            
            # Get the current hypothesis data
            self.studies_manager.set_active_study(target_study_id)
            current_hyp_data = self.studies_manager.get_hypothesis(target_hyp['id'])
            
            if not current_hyp_data:
                QMessageBox.warning(self, "Error", "Target hypothesis not found")
                return
                
            # Update with the new evidence
            evidence_type = "supporting" if "Supporting" in link_type else "contradicting"
            if evidence_type == "supporting":
                supporting_evidence = current_hyp_data.get("supporting_evidence", [])
                supporting_evidence.append(evidence)
                update_data = {"supporting_evidence": supporting_evidence}
            else:
                contradicting_evidence = current_hyp_data.get("contradicting_evidence", [])
                contradicting_evidence.append(evidence)
                update_data = {"contradicting_evidence": contradicting_evidence}
            
            # Update the hypothesis
            success = self.studies_manager.update_hypothesis(
                hypothesis_id=target_hyp['id'],
                update_data=update_data
            )
            
            if success:
                QMessageBox.information(
                    self, 
                    "Link Created", 
                    f"Successfully created {evidence_type} evidence link from {source_hyp.get('title')} to {target_hyp.get('title')}"
                )
            else:
                QMessageBox.warning(self, "Error", "Failed to create link")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
    
    def _analyze_cross_study_links(self, study_ids, min_confidence):
        """Use AI to analyze and suggest links between hypotheses across studies."""
        if not study_ids or len(study_ids) < 2:
            QMessageBox.warning(self, "Error", "Please select at least 2 studies to analyze")
            return
            
        self.ai_results_list.clear()
            
        # Collect all hypotheses from selected studies
        all_hypotheses = []
        study_names = {}
        study_results = {}  # To store variable information from each study's statistical tests
        
        try:
            # First, get the shared variables data to improve linking accuracy
            shared_variables = self.find_shared_variables()
            
            # Get all studies' hypotheses and statistical results
            for study_id in study_ids:
                self.studies_manager.set_active_study(study_id)
                study = self.studies_manager.get_active_study()
                study_names[study_id] = study.name
                
                # Get statistical results for this study (contains variable info)
                study_results[study_id] = self.studies_manager.get_statistical_results()
                
                # Get hypotheses for this study
                hypotheses = self.studies_manager.get_study_hypotheses()
                for hyp in hypotheses:
                    hyp_copy = hyp.copy()
                    hyp_copy['study_id'] = study_id
                    hyp_copy['study_name'] = study.name
                    
                    # Enrich with variable data where possible
                    hyp_copy['tracked_variables'] = self._extract_hypothesis_variables(
                        hyp_copy, study_results[study_id]
                    )
                    
                    all_hypotheses.append(hyp_copy)
            
            if len(all_hypotheses) < 2:
                QMessageBox.warning(self, "Error", "Not enough hypotheses found in the selected studies")
                return
                
            # This would be the AI analysis part
            # For now, using enhanced variable tracking information
            potential_links = []
            
            for i, source_hyp in enumerate(all_hypotheses):
                for j, target_hyp in enumerate(all_hypotheses):
                    # Skip same hypothesis
                    if i == j:
                        continue
                        
                    # Skip same study - we're looking for cross-study links
                    if source_hyp['study_id'] == target_hyp['study_id']:
                        continue
                    
                    # Check for variable overlap using tracked variables and shared variables data
                    var_overlap_score, shared_vars = self._calculate_variable_overlap(
                        source_hyp, target_hyp, shared_variables
                    )
                    
                    # Text similarity (very basic for now)
                    # In a real implementation, use embeddings/semantic similarity
                    source_text = source_hyp.get('alternative_hypothesis', '').lower()
                    target_text = target_hyp.get('alternative_hypothesis', '').lower()
                    
                    text_score = 0
                    if source_text and target_text:
                        # Simple word overlap score
                        source_words = set(source_text.split())
                        target_words = set(target_text.split())
                        overlap = len(source_words.intersection(target_words))
                        text_score = min(0.5, overlap * 0.05)  # Cap at 0.5
                    
                    # Combine scores (weighted average)
                    combined_score = var_overlap_score + text_score
                    confidence = min(0.95, combined_score)  # Cap at 0.95
                    
                    # Only suggest links with confidence >= minimum threshold
                    if confidence >= min_confidence:
                        # Determine evidence type based on status and directionality
                        evidence_type = self._determine_evidence_type(
                            source_hyp.get('status', 'untested'),
                            source_hyp,
                            target_hyp
                        )
                        
                        # Create a link suggestion with enhanced variable information
                        link = {
                            "source_study_id": source_hyp['study_id'],
                            "source_study_name": source_hyp['study_name'],
                            "source_hyp_id": source_hyp['id'],
                            "source_hyp_title": source_hyp.get('title', 'Unknown'),
                            "target_study_id": target_hyp['study_id'],
                            "target_study_name": target_hyp['study_name'],
                            "target_hyp_id": target_hyp['id'],
                            "target_hyp_title": target_hyp.get('title', 'Unknown'),
                            "confidence": confidence,
                            "evidence_type": evidence_type,
                            "shared_vars": shared_vars,
                            "suggested_description": self._generate_link_description(
                                source_hyp, target_hyp, evidence_type, shared_vars
                            )
                        }
                        
                        potential_links.append(link)
            
            # Sort by confidence (highest first)
            potential_links.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Display in results list
            for link in potential_links:
                item = QListWidgetItem()
                item.setData(Qt.ItemDataRole.UserRole, link)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                
                # Format display text
                if link['evidence_type'] == 'supporting':
                    icon = load_bootstrap_icon("arrow-up-circle", color="#28a745")
                    relationship = "Supports"
                else:
                    icon = load_bootstrap_icon("arrow-down-circle", color="#dc3545")
                    relationship = "Contradicts"
                    
                item.setIcon(icon)
                
                confidence_percent = int(link['confidence'] * 100)
                text = f"{link['source_study_name']}: {link['source_hyp_title']} {relationship} {link['target_study_name']}: {link['target_hyp_title']} ({confidence_percent}%)"
                item.setText(text)
                
                self.ai_results_list.addItem(item)
                
            # Status message
            if not potential_links:
                QMessageBox.information(
                    self, 
                    "Analysis Complete", 
                    "No potential links were found that meet the minimum confidence threshold."
                )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during analysis: {str(e)}")
    
    def _extract_hypothesis_variables(self, hypothesis, study_results):
        """
        Extract variables from statistical test results related to this hypothesis.
        
        Args:
            hypothesis: The hypothesis data
            study_results: Statistical test results for the study
            
        Returns:
            dict: Dictionary of variables and their roles
        """
        tracked_vars = {
            'outcome': set(),
            'predictors': set(),
            'covariates': set()
        }
        
        # Try to match hypothesis outcome with test outcome
        hyp_outcome = hypothesis.get('outcome_variables', '')
        if isinstance(hyp_outcome, str):
            hyp_outcome = [var.strip() for var in hyp_outcome.split(',') if var.strip()]
        elif not isinstance(hyp_outcome, list):
            hyp_outcome = []
            
        for outcome in hyp_outcome:
            tracked_vars['outcome'].add(outcome)
        
        # Extract predictors from hypothesis
        hyp_predictors = hypothesis.get('predictor_variables', '')
        if isinstance(hyp_predictors, str):
            hyp_predictors = [var.strip() for var in hyp_predictors.split(',') if var.strip()]
        elif not isinstance(hyp_predictors, list):
            hyp_predictors = []
            
        for predictor in hyp_predictors:
            tracked_vars['predictors'].add(predictor)
        
        # Look for statistical tests that match this hypothesis
        for result in study_results:
            # Skip if no variable info
            if 'variables' not in result or not result['variables']:
                continue
                
            variables = result['variables']
            
            # Check if this test's outcome matches the hypothesis outcome
            if 'outcome' in variables and variables['outcome'] in hyp_outcome:
                # Add group/predictor variables
                if 'group' in variables:
                    if isinstance(variables['group'], list):
                        for var in variables['group']:
                            if var:
                                tracked_vars['predictors'].add(var)
                    elif variables['group']:
                        tracked_vars['predictors'].add(variables['group'])
                
                # Add covariates
                if 'covariates' in variables and variables['covariates']:
                    if isinstance(variables['covariates'], list):
                        for var in variables['covariates']:
                            if var:
                                tracked_vars['covariates'].add(var)
                    else:
                        tracked_vars['covariates'].add(variables['covariates'])
                        
                # Add role definitions if available
                if 'role_definitions' in variables and variables['role_definitions']:
                    for var, role in variables['role_definitions'].items():
                        if role == 'outcome':
                            tracked_vars['outcome'].add(var)
                        elif role in ['group', 'time']:
                            tracked_vars['predictors'].add(var)
                        elif role == 'covariate':
                            tracked_vars['covariates'].add(var)
        
        # Convert sets to lists for JSON serialization
        return {
            'outcome': list(tracked_vars['outcome']),
            'predictors': list(tracked_vars['predictors']),
            'covariates': list(tracked_vars['covariates'])
        }
    
    def _calculate_variable_overlap(self, source_hyp, target_hyp, shared_variables):
        """
        Calculate variable overlap between two hypotheses.
        
        Args:
            source_hyp: Source hypothesis data
            target_hyp: Target hypothesis data
            shared_variables: Dictionary of variables shared across studies
            
        Returns:
            tuple: (overlap_score, list of shared variable names)
        """
        shared_vars = []
        
        # First check using enhanced tracked variables if available
        if 'tracked_variables' in source_hyp and 'tracked_variables' in target_hyp:
            source_vars = set()
            for var_list in source_hyp['tracked_variables'].values():
                source_vars.update(var_list)
                
            target_vars = set()
            for var_list in target_hyp['tracked_variables'].values():
                target_vars.update(var_list)
                
            # Check for direct overlap
            direct_overlap = source_vars.intersection(target_vars)
            if direct_overlap:
                shared_vars.extend(direct_overlap)
                
            # Check against known shared variables
            for var in source_vars:
                if var in shared_variables:
                    # This is a shared variable across studies
                    for study_id, study_name, role in shared_variables[var]:
                        if study_id == target_hyp['study_id'] and var not in shared_vars:
                            shared_vars.append(var)
            
            # Also do the reverse check
            for var in target_vars:
                if var in shared_variables:
                    # This is a shared variable across studies
                    for study_id, study_name, role in shared_variables[var]:
                        if study_id == source_hyp['study_id'] and var not in shared_vars:
                            shared_vars.append(var)
        
        # Fallback to traditional method if no tracked variables
        if not shared_vars:
            # Old method for backward compatibility
            source_outcome_vars = set(var.strip() for var in source_hyp.get('outcome_variables', '').split(',') if var.strip())
            source_predictor_vars = set(var.strip() for var in source_hyp.get('predictor_variables', '').split(',') if var.strip())
            
            target_outcome_vars = set(var.strip() for var in target_hyp.get('outcome_variables', '').split(',') if var.strip())
            target_predictor_vars = set(var.strip() for var in target_hyp.get('predictor_variables', '').split(',') if var.strip())
            
            # Calculate overlap
            shared_outcome = source_outcome_vars.intersection(target_outcome_vars)
            shared_predictors = source_predictor_vars.intersection(target_predictor_vars)
            
            shared_vars = list(shared_outcome.union(shared_predictors))
        
        # Calculate confidence score based on variable overlap
        var_overlap_score = 0
        if shared_vars:
            # Higher weights for outcome variables, lower for covariates
            outcome_weight = 0.3
            predictor_weight = 0.2
            covariate_weight = 0.1
            
            for var in shared_vars:
                # Check source hypothesis
                if 'tracked_variables' in source_hyp:
                    if var in source_hyp['tracked_variables']['outcome']:
                        var_overlap_score += outcome_weight
                    elif var in source_hyp['tracked_variables']['predictors']:
                        var_overlap_score += predictor_weight
                    elif var in source_hyp['tracked_variables']['covariates']:
                        var_overlap_score += covariate_weight
                
                # Check target hypothesis
                if 'tracked_variables' in target_hyp:
                    if var in target_hyp['tracked_variables']['outcome']:
                        var_overlap_score += outcome_weight
                    elif var in target_hyp['tracked_variables']['predictors']:
                        var_overlap_score += predictor_weight
                    elif var in target_hyp['tracked_variables']['covariates']:
                        var_overlap_score += covariate_weight
                
            # If we didn't find any weighted variables, assign a default score
            if var_overlap_score == 0:
                var_overlap_score = len(shared_vars) * 0.15
                
        return var_overlap_score, shared_vars    
    def _generate_link_description(self, source_hyp, target_hyp, evidence_type, shared_vars):
        """
        Generate a descriptive text for the link between hypotheses.
        
        Args:
            source_hyp: Source hypothesis data
            target_hyp: Target hypothesis data
            evidence_type: Type of evidence (supporting or contradicting)
            shared_vars: List of shared variables between hypotheses
            
        Returns:
            str: Descriptive text for the link
        """
        relation = "supports" if evidence_type == "supporting" else "contradicts"
        
        if shared_vars:
            vars_text = ", ".join(shared_vars)
            if len(shared_vars) == 1:
                vars_phrase = f"the variable {vars_text}"
            else:
                vars_phrase = f"the variables {vars_text}"
                
            description = (
                f"Evidence from '{source_hyp.get('title')}' in study '{source_hyp['study_name']}' "
                f"{relation} this hypothesis. Both hypotheses examine {vars_phrase}."
            )
        else:
            description = (
                f"Evidence from '{source_hyp.get('title')}' in study '{source_hyp['study_name']}' "
                f"{relation} this hypothesis based on text similarity."
            )
            
        # Add statistical evidence information if available
        if source_hyp.get('test_results'):
            p_value = source_hyp['test_results'].get('p_value')
            if p_value is not None:
                description += f" The source hypothesis has statistical evidence with p = {p_value:.4f}."
                
        return description
    
    def _apply_ai_links(self):
        """Apply the selected AI-suggested links."""
        selected_items = []
        for i in range(self.ai_results_list.count()):
            item = self.ai_results_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_items.append(item)
        
        if not selected_items:
            QMessageBox.warning(self, "No Links Selected", "Please select at least one link to apply")
            return
            
        success_count = 0
        error_count = 0
        
        # Restore active study after applying links
        active_study_id = self.studies_manager.get_active_study().id
        
        for item in selected_items:
            link = item.data(Qt.ItemDataRole.UserRole)
            
            try:
                # Create evidence object
                evidence = {
                    "id": str(uuid.uuid4()),
                    "type": "literature",
                    "description": link['suggested_description'],
                    "source": "AI-suggested cross-study reference",
                    "confidence": link['confidence'],
                    "cross_reference": {
                        "study_id": link['source_study_id'],
                        "hypothesis_id": link['source_hyp_id'],
                        "hypothesis_title": link['source_hyp_title']
                    }
                }
                
                # Get the current hypothesis data
                self.studies_manager.set_active_study(link['target_study_id'])
                current_hyp_data = self.studies_manager.get_hypothesis(link['target_hyp_id'])
                
                if not current_hyp_data:
                    error_count += 1
                    continue
                    
                # Update with the new evidence
                evidence_type = link['evidence_type']
                if evidence_type == "supporting":
                    supporting_evidence = current_hyp_data.get("supporting_evidence", [])
                    supporting_evidence.append(evidence)
                    update_data = {"supporting_evidence": supporting_evidence}
                else:
                    contradicting_evidence = current_hyp_data.get("contradicting_evidence", [])
                    contradicting_evidence.append(evidence)
                    update_data = {"contradicting_evidence": contradicting_evidence}
                
                # Update the hypothesis
                success = self.studies_manager.update_hypothesis(
                    hypothesis_id=link['target_hyp_id'],
                    update_data=update_data
                )
                
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    
            except Exception:
                error_count += 1
        
        # Restore active study
        self.studies_manager.set_active_study(active_study_id)
        self.load_hypotheses()
        
        # Show results
        QMessageBox.information(
            self, 
            "Links Applied", 
            f"Successfully applied {success_count} links.\n"
            f"{error_count} links could not be applied due to errors."
        )

    def _on_study_changed(self, index):
        """Handle study selection change."""
        if index < 0 or not self.studies_manager:
            return
            
        study_id = self.study_selector.itemData(index)
        if study_id:
            self.studies_manager.set_active_study(study_id)
            self.load_hypotheses()
            self.status_bar.setText(f"Switched to study: {self.study_selector.itemText(index)}")

    def find_shared_variables(self):
        """
        Identify variables that are shared across multiple studies.
        
        Returns:
            dict: A dictionary where keys are variable names and values are lists of tuples 
                 containing (study_id, study_name, variable_role) for each occurrence.
        """
        if not self.studies_manager:
            self.status_bar.setText("No studies manager available")
            return {}
        
        # Get list of all studies
        all_studies = []
        projects = self.studies_manager.list_projects()
        for project in projects:
            project_id = project.get('id')
            studies = self.studies_manager.list_studies(project_id)
            for study in studies:
                all_studies.append((project_id, study.get('id')))
        
        # Dictionary to hold variable occurrences
        shared_variables = {}
        
        # Check each study for variables
        for project_id, study_id in all_studies:
            study = self.studies_manager.get_study(study_id, project_id)
            if not study or not hasattr(study, 'results') or not study.results:
                continue
            
            study_name = study.name
            
            # Process each result in the study
            for result in study.results:
                # Skip if no variable information
                if not hasattr(result, 'variables') or not result.variables:
                    continue
                
                variables = result.variables
                
                # Extract role definitions if available
                role_definitions = variables.get('role_definitions', {})
                
                # Process each variable by explicit role fields
                for role, var_names in variables.items():
                    if role == 'role_definitions':
                        continue  # Skip this special field
                    
                    # Handle both single variables and lists
                    var_list = var_names if isinstance(var_names, list) else [var_names]
                    
                    for var_name in var_list:
                        if not var_name:
                            continue
                        
                        if var_name not in shared_variables:
                            shared_variables[var_name] = []
                        
                        # Get role from role_definitions if available, otherwise use field name
                        var_role = role_definitions.get(var_name, role) if role_definitions else role
                        
                        # Add to shared variables if not already present for this study
                        study_var_entry = (study_id, study_name, var_role)
                        if study_var_entry not in shared_variables[var_name]:
                            shared_variables[var_name].append(study_var_entry)
        
        # Filter out variables that only appear in one study
        return {var: occurrences for var, occurrences in shared_variables.items() 
                if len(occurrences) > 1}
    
    def show_shared_variables_dialog(self):
        """
        Display a dialog showing variables shared across multiple studies.
        This helps users identify potential links between studies.
        """
        shared_vars = self.find_shared_variables()
        
        if not shared_vars:
            QMessageBox.information(
                self, 
                "Shared Variables", 
                "No variables shared across studies were found.\n\n"
                "This may be because:\n"
                "- You only have one study\n"
                "- Your studies use different variable names\n"
                "- You haven't run any statistical tests\n\n"
                "Run statistical tests in multiple studies to track shared variables."
            )
            return
        
        # Create the dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Shared Variables Across Studies")
        dialog.resize(800, 600)
        
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("These variables appear in multiple studies:"))
        
        # Create a table to display the shared variables
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Variable", "Studies Count", "Roles", "Study Names"])
        table.setRowCount(len(shared_vars))
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        
        # Populate table
        for row, (var_name, occurrences) in enumerate(shared_vars.items()):
            # Variable name
            table.setItem(row, 0, QTableWidgetItem(var_name))
            
            # Count of studies
            table.setItem(row, 1, QTableWidgetItem(str(len(occurrences))))
            
            # Roles (comma separated)
            roles = sorted(set(role for _, _, role in occurrences))
            table.setItem(row, 2, QTableWidgetItem(", ".join(roles)))
            
            # Study names (comma separated)
            studies = sorted(set(study_name for _, study_name, _ in occurrences))
            table.setItem(row, 3, QTableWidgetItem(", ".join(studies)))
        
        # Resize columns to content
        table.resizeColumnsToContents()
        layout.addWidget(table)
        
        # Add a helpful message
        msg = QLabel(
            "Variables that appear in multiple studies can be used to create cross-study links.\n"
            "This helps you identify potential relationships between hypotheses across studies."
        )
        msg.setWordWrap(True)
        layout.addWidget(msg)
        
        # Add close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()

    def update_study_selector(self):
        """Update the study selector dropdown with current studies."""
        if not self.studies_manager:
            self.status_bar.setText("No studies manager available")
            return
            
        # Remember current selection if any
        current_study_id = None
        if self.study_selector.currentIndex() >= 0:
            current_study_id = self.study_selector.currentData()
            
        # Clear and repopulate the study selector
        self.study_selector.blockSignals(True)
        self.study_selector.clear()
        
        studies = self.studies_manager.list_studies()
        if not studies:
            self.status_bar.setText("No studies available")
            self.study_selector.blockSignals(False)
            return
            
        # Add all studies to dropdown
        for study in studies:
            self.study_selector.addItem(study["name"], study["id"])
        
        # Restore selection or set to active study
        if current_study_id:
            # Try to restore previous selection
            for i in range(self.study_selector.count()):
                if self.study_selector.itemData(i) == current_study_id:
                    self.study_selector.setCurrentIndex(i)
                    break
        else:
            # Set to current active study
            active_study = self.studies_manager.get_active_study()
            if active_study:
                for i in range(self.study_selector.count()):
                    if self.study_selector.itemData(i) == active_study.id:
                        self.study_selector.setCurrentIndex(i)
                        break
        
        self.study_selector.blockSignals(False)
        self.status_bar.setText(f"Found {len(studies)} studies")
        
        # Load hypotheses from current selection
        self.load_hypotheses()

    def debug_variable_tracking(self):
        """Debug method to check why shared variables aren't showing up."""
        if not self.studies_manager:
            QMessageBox.warning(self, "Error", "Studies manager not available")
            return
            
        debug_info = []
        debug_info.append("VARIABLE TRACKING DIAGNOSTIC INFORMATION")
        debug_info.append("-----------------------------------------")
        
        # Get list of all studies
        all_studies = []
        projects = self.studies_manager.list_projects()
        for project in projects:
            project_id = project.get('id')
            studies = self.studies_manager.list_studies(project_id)
            for study in studies:
                all_studies.append((project_id, study.get('id')))
        
        debug_info.append(f"Found {len(all_studies)} studies across {len(projects)} projects")
        debug_info.append("")
        
        # Check each study for statistical results and variables
        for project_id, study_id in all_studies:
            study = self.studies_manager.get_study(study_id, project_id)
            if not study:
                debug_info.append(f"Study {study_id} not found")
                continue
                
            debug_info.append(f"STUDY: {study.name} (ID: {study_id})")
            
            # Check for results attribute
            if not hasattr(study, 'results'):
                debug_info.append("  No 'results' attribute found on study object")
                debug_info.append("")
                continue
                
            # Check if results list exists and has items
            if not study.results:
                debug_info.append("  Study has empty results list")
                debug_info.append("")
                continue
                
            debug_info.append(f"  Found {len(study.results)} statistical results")
            
            # Check each result for variable information
            for i, result in enumerate(study.results):
                debug_info.append(f"  Result {i+1}:")
                debug_info.append(f"    Outcome: {getattr(result, 'outcome_name', 'Unknown')}")
                debug_info.append(f"    Test: {getattr(result, 'statistical_test_name', 'Unknown')}")
                
                # Check for variables attribute
                if not hasattr(result, 'variables'):
                    debug_info.append("    No 'variables' attribute found")
                    continue
                    
                if not result.variables:
                    debug_info.append("    Empty variables dictionary")
                    continue
                
                # Check variable dictionary structure
                debug_info.append("    Variables:")
                for role, var_value in result.variables.items():
                    if role == 'role_definitions':
                        debug_info.append(f"      Role Definitions: {len(var_value)} mappings")
                    else:
                        if isinstance(var_value, list):
                            debug_info.append(f"      {role}: {', '.join(var_value) if var_value else 'empty list'}")
                        else:
                            debug_info.append(f"      {role}: {var_value if var_value else 'None'}")
            
            debug_info.append("")
        
        # Show debug info in a dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Variable Tracking Debug Information")
        dialog.resize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Create text area with debug info
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText("\n".join(debug_info))
        layout.addWidget(text_edit)
        
        # Add close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()

    def debug_and_show_shared_variables(self):
        """
        Debug method to check why shared variables aren't showing up.
        This method will first run debug_variable_tracking and then show_shared_variables_dialog.
        """
        self.debug_variable_tracking()
        self.show_shared_variables_dialog()
        
    def show_literature_claims(self, hypothesis_data):
        """Show dialog for managing literature claims related to this hypothesis."""
        if not self.studies_manager:
            QMessageBox.warning(self, "Error", "No studies manager available")
            return
        
        active_study = self.studies_manager.get_active_study()
        if not active_study:
            QMessageBox.warning(self, "Error", "No active study")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Literature Claims: {hypothesis_data.get('title', 'Untitled Hypothesis')}")
        dialog.setMinimumWidth(600)
        dialog.setMinimumHeight(400)
        
        layout = QVBoxLayout(dialog)
        
        # Add hypothesis info at the top
        info_frame = QFrame()
        info_frame.setStyleSheet("border-radius: 5px; padding: 10px;")
        info_layout = QVBoxLayout(info_frame)
        
        title_label = QLabel(f"<h3>{hypothesis_data.get('title', 'Untitled Hypothesis')}</h3>")
        info_layout.addWidget(title_label)
        
        alt_hyp = hypothesis_data.get('alternative_hypothesis', '')
        if alt_hyp:
            alt_label = QLabel(f"<b>H₁:</b> {alt_hyp}")
            alt_label.setWordWrap(True)
            info_layout.addWidget(alt_label)
        
        layout.addWidget(info_frame)
        
        # Add placeholder for the literature claims widget
        placeholder = QLabel("Literature claims management will be loaded here")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("font-style: italic; margin: 20px;")
        layout.addWidget(placeholder)
        
        # Add message that the function is not fully implemented
        message_label = QLabel("<b>Note:</b> This is a placeholder for the Literature Claims feature. "
                             "The actual literature analysis section should be loaded here.")
        message_label.setWordWrap(True)
        message_label.setStyleSheet("margin: 10px;")
        layout.addWidget(message_label)
        
        # Add buttons
        button_layout = QHBoxLayout()
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.reject)
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)
        
        dialog.exec()
        
    def show_model_analysis(self, hypothesis_data):
        """Show dialog for managing model analysis related to this hypothesis."""
        if not self.studies_manager or not self.data_testing_widget:
            QMessageBox.warning(self, "Error", "Required components not available")
            return
        
        active_study = self.studies_manager.get_active_study()
        if not active_study:
            QMessageBox.warning(self, "Error", "No active study")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Model Analysis: {hypothesis_data.get('title', 'Untitled Hypothesis')}")
        dialog.setMinimumWidth(700)
        dialog.setMinimumHeight(500)
        
        layout = QVBoxLayout(dialog)
        
        # Add hypothesis info at the top
        info_frame = QFrame()
        info_frame.setStyleSheet("border-radius: 5px; padding: 10px;")
        info_layout = QVBoxLayout(info_frame)
        
        title_label = QLabel(f"<h3>{hypothesis_data.get('title', 'Untitled Hypothesis')}</h3>")
        info_layout.addWidget(title_label)
        
        # Add null and alternative hypothesis text
        null_hyp = hypothesis_data.get('null_hypothesis', '')
        alt_hyp = hypothesis_data.get('alternative_hypothesis', '')
        
        if null_hyp:
            null_label = QLabel(f"<b>H₀:</b> {null_hyp}")
            null_label.setWordWrap(True)
            info_layout.addWidget(null_label)
        
        if alt_hyp:
            alt_label = QLabel(f"<b>H₁:</b> {alt_hyp}")
            alt_label.setWordWrap(True)
            info_layout.addWidget(alt_label)
        
        layout.addWidget(info_frame)
        
        # Dataset selection
        dataset_group = QGroupBox("Select Dataset")
        dataset_layout = QVBoxLayout(dataset_group)
        
        dataset_combo = QComboBox()
        # The dataset combo will be populated when the dialog is shown
        dataset_layout.addWidget(dataset_combo)
        
        # Outcome variable selection
        outcome_layout = QHBoxLayout()
        outcome_layout.addWidget(QLabel("Outcome Variable:"))
        outcome_combo = QComboBox()
        outcome_layout.addWidget(outcome_combo)
        dataset_layout.addLayout(outcome_layout)
        
        # Load datasets from active study
        if active_study and hasattr(active_study, 'datasets'):
            for i, dataset in enumerate(active_study.datasets):
                dataset_name = dataset.get('name', f"Dataset {i+1}")
                dataset_id = dataset.get('id', '')
                dataset_combo.addItem(dataset_name, dataset_id)
                
            # Connect dataset selection to update outcome variables
            def on_dataset_changed(index):
                outcome_combo.clear()
                if index < 0 or not active_study.datasets or index >= len(active_study.datasets):
                    return
                    
                dataset = active_study.datasets[index]
                if 'variables' in dataset:
                    for var in dataset['variables']:
                        var_name = var.get('name', '')
                        if var_name:
                            outcome_combo.addItem(var_name, var_name)
                            
            dataset_combo.currentIndexChanged.connect(on_dataset_changed)
            
            # Initial population of outcome variables
            on_dataset_changed(dataset_combo.currentIndex())
        
        layout.addWidget(dataset_group)
        
        # Add message about the feature
        message_label = QLabel("<b>Note:</b> This dialog allows selecting the dataset and outcome variable "
                             "for model analysis. The full model analysis would be handled in the dedicated section.")
        message_label.setWordWrap(True)
        message_label.setStyleSheet("margin: 10px;")
        layout.addWidget(message_label)
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        analyze_button = QPushButton("Open in Model Analysis")
        analyze_button.clicked.connect(lambda: self._open_in_model_analysis(
            hypothesis_data, 
            dataset_combo.currentData(), 
            outcome_combo.currentText()
        ))
        button_layout.addWidget(analyze_button)
        
        button_layout.addStretch()
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.reject)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        dialog.exec()
        
    def _open_in_model_analysis(self, hypothesis_data, dataset_id, outcome_var):
        """Open the data testing widget with the selected dataset and outcome variable."""
        if not self.data_testing_widget:
            QMessageBox.warning(self, "Error", "Data testing widget not available")
            return
            
        # Simply close the dialog and show a message for now
        QMessageBox.information(
            self, 
            "Model Analysis", 
            f"Would open model analysis for '{hypothesis_data.get('title')}' "
            f"with dataset ID '{dataset_id}' and outcome '{outcome_var}'.\n\n"
            f"This functionality would load the data testing section with the appropriate settings."
        )
