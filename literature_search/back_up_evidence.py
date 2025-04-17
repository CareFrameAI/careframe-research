import asyncio
from typing import Dict, List, Optional, Any, Tuple
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QTabWidget, QFileDialog, QMessageBox, QProgressBar, QScrollArea,
    QGroupBox, QGridLayout, QSplitter, QButtonGroup, QRadioButton,
    QFrame, QDialog, QSpinBox, QCheckBox, QFormLayout, QComboBox, QTextEdit, QApplication,
    QDialogButtonBox, QToolButton, QToolBar, QListWidget, QListWidgetItem, QMenu, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QHeaderView
)
from PyQt6.QtCore import pyqtSignal, Qt, QUrl, QSize
from PyQt6.QtGui import QFont, QIcon, QDesktopServices, QAction, QColor
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import re
import json

from agent.tools import store_result
from helpers.load_icon import load_bootstrap_icon
from study_model.studies_manager import StudiesManager
from literature_search.pattern_extractor import QuoteExtractor

from qasync import asyncSlot
from literature_search.model_calls import rate_papers_with_gemini

class EvidenceItem:
    """Represents a piece of evidence extracted from a paper."""
    
    def __init__(self, text: str, paper_doi: str, paper_title: str, 
                 evidence_type: str = "neutral", confidence: int = 50,
                 context: str = "", page_number: Optional[int] = None):
        self.text = text
        self.paper_doi = paper_doi
        self.paper_title = paper_title
        self.evidence_type = evidence_type  # "support", "refute", "neutral"
        self.confidence = confidence  # 0-100
        self.context = context
        self.page_number = page_number
        self.tags = []
        self.notes = ""
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "paper_doi": self.paper_doi,
            "paper_title": self.paper_title,
            "evidence_type": self.evidence_type,
            "confidence": self.confidence,
            "context": self.context,
            "page_number": self.page_number,
            "tags": self.tags,
            "notes": self.notes,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EvidenceItem':
        """Create EvidenceItem from dictionary."""
        item = cls(
            text=data.get("text", ""),
            paper_doi=data.get("paper_doi", ""),
            paper_title=data.get("paper_title", ""),
            evidence_type=data.get("evidence_type", "neutral"),
            confidence=data.get("confidence", 50),
            context=data.get("context", ""),
            page_number=data.get("page_number")
        )
        item.tags = data.get("tags", [])
        item.notes = data.get("notes", "")
        
        # Convert timestamp back from string
        if "timestamp" in data:
            try:
                item.timestamp = datetime.fromisoformat(data["timestamp"])
            except (ValueError, TypeError):
                item.timestamp = datetime.now()
                
        return item

class Claim:
    """Represents a scientific claim that can be supported or refuted by evidence."""
    
    def __init__(self, text: str, source_doi: Optional[str] = None, 
                 source_title: Optional[str] = None):
        self.text = text
        self.source_doi = source_doi
        self.source_title = source_title
        self.evidence = []  # List of EvidenceItem objects
        self.tags = []
        self.notes = ""
        self.timestamp = datetime.now()
        self.id = datetime.now().strftime("%Y%m%d%H%M%S")
        
    def add_evidence(self, evidence: EvidenceItem):
        """Add evidence item to this claim."""
        self.evidence.append(evidence)
        
    def evidence_summary(self) -> Tuple[int, int, int]:
        """Return counts of supporting, refuting, and neutral evidence."""
        supporting = sum(1 for e in self.evidence if e.evidence_type == "support")
        refuting = sum(1 for e in self.evidence if e.evidence_type == "refute")
        neutral = sum(1 for e in self.evidence if e.evidence_type == "neutral")
        return supporting, refuting, neutral
    
    def overall_assessment(self) -> str:
        """Return an overall assessment based on the evidence."""
        supporting, refuting, neutral = self.evidence_summary()
        
        # Calculate weighted score
        total_evidence = supporting + refuting
        if total_evidence == 0:
            return "Insufficient Evidence"
            
        support_ratio = supporting / total_evidence
        
        if support_ratio >= 0.75:
            return "Strongly Supported"
        elif support_ratio >= 0.6:
            return "Supported"
        elif support_ratio <= 0.25:
            return "Strongly Refuted" 
        elif support_ratio <= 0.4:
            return "Refuted"
        else:
            return "Contested"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "source_doi": self.source_doi,
            "source_title": self.source_title,
            "evidence": [e.to_dict() for e in self.evidence],
            "tags": self.tags,
            "notes": self.notes,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Claim':
        """Create Claim from dictionary."""
        claim = cls(
            text=data.get("text", ""),
            source_doi=data.get("source_doi"),
            source_title=data.get("source_title")
        )
        
        claim.id = data.get("id", claim.id)
        claim.tags = data.get("tags", [])
        claim.notes = data.get("notes", "")
        
        # Convert timestamp back from string
        if "timestamp" in data:
            try:
                claim.timestamp = datetime.fromisoformat(data["timestamp"])
            except (ValueError, TypeError):
                claim.timestamp = datetime.now()
        
        # Add evidence items
        for ev_data in data.get("evidence", []):
            claim.evidence.append(EvidenceItem.from_dict(ev_data))
            
        return claim

class EvidenceTable(QTableWidget):
    """Table widget for displaying evidence items."""
    
    evidenceSelected = pyqtSignal(EvidenceItem)
    evidenceClassified = pyqtSignal(EvidenceItem, str)  # Item, new classification
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.evidence_items = []
        
    def setup_ui(self):
        """Setup the table UI."""
        # Set column headers
        self.setColumnCount(6)
        self.setHorizontalHeaderLabels([
            "Evidence", "Type", "Confidence", "Source", "Tags", "Actions"
        ])
        
        # Set table properties
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        
        # Set column widths
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)  # Evidence
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)  # Type
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Confidence
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)  # Source
        self.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)  # Tags
        self.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)  # Actions
        
        # Connect signals
        self.cellDoubleClicked.connect(self.on_cell_double_clicked)
    
    def set_evidence(self, evidence_items: List[EvidenceItem]):
        """Set the evidence items to display."""
        self.evidence_items = evidence_items
        self.refresh_table()
    
    def refresh_table(self):
        """Refresh the table display with current evidence items."""
        self.clearContents()
        self.setRowCount(len(self.evidence_items))
        
        for row, evidence in enumerate(self.evidence_items):
            # Evidence text (trimmed)
            text = evidence.text[:100] + "..." if len(evidence.text) > 100 else evidence.text
            self.setItem(row, 0, QTableWidgetItem(text))
            
            # Evidence type with color coding
            type_item = QTableWidgetItem(evidence.evidence_type.capitalize())
            if evidence.evidence_type == "support":
                type_item.setBackground(QColor(200, 255, 200))  # Light green
            elif evidence.evidence_type == "refute":
                type_item.setBackground(QColor(255, 200, 200))  # Light red
            else:
                type_item.setBackground(QColor(230, 230, 230))  # Light gray
            self.setItem(row, 1, type_item)
            
            # Confidence
            self.setItem(row, 2, QTableWidgetItem(f"{evidence.confidence}%"))
            
            # Source
            source_text = evidence.paper_title[:30] + "..." if len(evidence.paper_title) > 30 else evidence.paper_title
            self.setItem(row, 3, QTableWidgetItem(source_text))
            
            # Tags
            tags_text = ", ".join(evidence.tags[:3])
            if len(evidence.tags) > 3:
                tags_text += f" +{len(evidence.tags) - 3}"
            self.setItem(row, 4, QTableWidgetItem(tags_text))
            
            # Actions - handled in a separate method to create buttons
            self.add_action_buttons(row, evidence)
    
    def add_action_buttons(self, row, evidence):
        """Add action buttons to a row."""
        actions_widget = QWidget()
        actions_layout = QHBoxLayout(actions_widget)
        actions_layout.setContentsMargins(2, 2, 2, 2)
        actions_layout.setSpacing(2)
        
        # View button
        view_btn = QToolButton()
        view_btn.setIcon(load_bootstrap_icon("eye"))
        view_btn.setToolTip("View Details")
        view_btn.clicked.connect(lambda: self.evidenceSelected.emit(evidence))
        actions_layout.addWidget(view_btn)
        
        # Classification buttons
        if evidence.evidence_type != "support":
            support_btn = QToolButton()
            support_btn.setIcon(load_bootstrap_icon("check-circle"))
            support_btn.setToolTip("Mark as Supporting")
            support_btn.clicked.connect(lambda: self.classify_evidence(evidence, "support"))
            actions_layout.addWidget(support_btn)
            
        if evidence.evidence_type != "refute":
            refute_btn = QToolButton()
            refute_btn.setIcon(load_bootstrap_icon("x-circle"))
            refute_btn.setToolTip("Mark as Refuting")
            refute_btn.clicked.connect(lambda: self.classify_evidence(evidence, "refute"))
            actions_layout.addWidget(refute_btn)
            
        if evidence.evidence_type != "neutral":
            neutral_btn = QToolButton()
            neutral_btn.setIcon(load_bootstrap_icon("dash-circle"))
            neutral_btn.setToolTip("Mark as Neutral")
            neutral_btn.clicked.connect(lambda: self.classify_evidence(evidence, "neutral"))
            actions_layout.addWidget(neutral_btn)
        
        # Add widget to table
        self.setCellWidget(row, 5, actions_widget)
    
    def classify_evidence(self, evidence, new_type):
        """Change evidence classification."""
        old_type = evidence.evidence_type
        evidence.evidence_type = new_type
        self.evidenceClassified.emit(evidence, new_type)
        self.refresh_table()
    
    def on_cell_double_clicked(self, row, column):
        """Handle double-click on a cell."""
        if 0 <= row < len(self.evidence_items):
            self.evidenceSelected.emit(self.evidence_items[row])
    
    def add_evidence(self, evidence):
        """Add a new evidence item to the table."""
        self.evidence_items.append(evidence)
        self.refresh_table()
    
    def remove_evidence(self, evidence):
        """Remove an evidence item from the table."""
        if evidence in self.evidence_items:
            self.evidence_items.remove(evidence)
            self.refresh_table()
    
    def filter_evidence(self, filter_text: str, evidence_type: Optional[str] = None):
        """Filter evidence items by text and/or type."""
        if not filter_text and not evidence_type:
            # No filtering
            self.refresh_table()
            return
            
        filter_text = filter_text.lower()
        filtered_items = []
        
        for evidence in self.evidence_items:
            # Check evidence type filter
            if evidence_type and evidence.evidence_type != evidence_type:
                continue
                
            # Check text filter
            if filter_text and filter_text not in evidence.text.lower() and \
               filter_text not in evidence.paper_title.lower() and \
               not any(filter_text in tag.lower() for tag in evidence.tags):
                continue
                
            filtered_items.append(evidence)
        
        # Update display
        self.setRowCount(len(filtered_items))
        for row, evidence in enumerate(filtered_items):
            # Update table cells (reusing code from refresh_table)
            # Evidence text (trimmed)
            text = evidence.text[:100] + "..." if len(evidence.text) > 100 else evidence.text
            self.setItem(row, 0, QTableWidgetItem(text))
            
            # Evidence type with color coding
            type_item = QTableWidgetItem(evidence.evidence_type.capitalize())
            if evidence.evidence_type == "support":
                type_item.setBackground(QColor(200, 255, 200))  # Light green
            elif evidence.evidence_type == "refute":
                type_item.setBackground(QColor(255, 200, 200))  # Light red
            else:
                type_item.setBackground(QColor(230, 230, 230))  # Light gray
            self.setItem(row, 1, type_item)
            
            # Confidence
            self.setItem(row, 2, QTableWidgetItem(f"{evidence.confidence}%"))
            
            # Source
            source_text = evidence.paper_title[:30] + "..." if len(evidence.paper_title) > 30 else evidence.paper_title
            self.setItem(row, 3, QTableWidgetItem(source_text))
            
            # Tags
            tags_text = ", ".join(evidence.tags[:3])
            if len(evidence.tags) > 3:
                tags_text += f" +{len(evidence.tags) - 3}"
            self.setItem(row, 4, QTableWidgetItem(tags_text))
            
            # Actions
            self.add_action_buttons(row, evidence)

class ClaimWidget(QGroupBox):
    """Widget for displaying a claim with its evidence summary."""
    
    claimSelected = pyqtSignal(Claim)
    
    def __init__(self, claim, parent=None):
        super().__init__(parent)
        self.claim = claim
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the widget UI."""
        # Set title to truncated claim text
        title = self.claim.text[:80] + "..." if len(self.claim.text) > 80 else self.claim.text
        self.setTitle(title)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Source information
        if self.claim.source_title:
            source_label = QLabel(f"<b>Source:</b> {self.claim.source_title}")
            source_label.setWordWrap(True)
            layout.addWidget(source_label)
        
        # Evidence summary
        supporting, refuting, neutral = self.claim.evidence_summary()
        evidence_layout = QHBoxLayout()
        
        support_label = QLabel(f"<span style='color:green'><b>Supporting:</b> {supporting}</span>")
        refute_label = QLabel(f"<span style='color:red'><b>Refuting:</b> {refuting}</span>")
        neutral_label = QLabel(f"<span style='color:gray'><b>Neutral:</b> {neutral}</span>")
        
        evidence_layout.addWidget(support_label)
        evidence_layout.addWidget(refute_label)
        evidence_layout.addWidget(neutral_label)
        
        layout.addLayout(evidence_layout)
        
        # Overall assessment
        assessment = self.claim.overall_assessment()
        assessment_label = QLabel(f"<b>Assessment:</b> {assessment}")
        assessment_label.setStyleSheet(self.get_assessment_style(assessment))
        layout.addWidget(assessment_label)
        
        # Tags
        if self.claim.tags:
            tags_text = ", ".join(self.claim.tags[:5])
            if len(self.claim.tags) > 5:
                tags_text += f" and {len(self.claim.tags) - 5} more"
            
            tags_label = QLabel(f"<b>Tags:</b> {tags_text}")
            layout.addWidget(tags_label)
        
        # Button for viewing details
        view_button = QPushButton("View Evidence")
        view_button.setIcon(load_bootstrap_icon("list-ul"))
        view_button.clicked.connect(lambda: self.claimSelected.emit(self.claim))
        
        # Button for editing
        edit_button = QPushButton("Edit Claim")
        edit_button.setIcon(load_bootstrap_icon("pencil"))
        
        # Create button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(view_button)
        button_layout.addWidget(edit_button)
        
        layout.addLayout(button_layout)
        
        # Set a fixed size for consistent appearance
        self.setMinimumHeight(150)
        self.setMaximumHeight(200)
    
    def get_assessment_style(self, assessment):
        """Get CSS style for the assessment label."""
        if assessment == "Strongly Supported":
            return "color: darkgreen; font-weight: bold;"
        elif assessment == "Supported":
            return "color: green; font-weight: bold;"
        elif assessment == "Strongly Refuted":
            return "color: darkred; font-weight: bold;"
        elif assessment == "Refuted":
            return "color: red; font-weight: bold;"
        elif assessment == "Contested":
            return "color: darkorange; font-weight: bold;"
        else:
            return "color: gray; font-weight: bold;"

class EvidenceDialog(QDialog):
    """Dialog for viewing and editing evidence details."""
    
    evidenceUpdated = pyqtSignal(EvidenceItem)
    
    def __init__(self, evidence, parent=None, editable=True):
        super().__init__(parent)
        self.evidence = evidence
        self.editable = editable
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the dialog UI."""
        self.setWindowTitle("Evidence Details")
        self.resize(700, 500)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Create tabs
        tabs = QTabWidget()
        
        # === Main Tab ===
        main_tab = QWidget()
        main_layout = QVBoxLayout(main_tab)
        
        # Evidence text
        text_group = QGroupBox("Evidence Text")
        text_layout = QVBoxLayout(text_group)
        
        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(self.evidence.text)
        self.text_edit.setReadOnly(not self.editable)
        text_layout.addWidget(self.text_edit)
        
        main_layout.addWidget(text_group)
        
        # Context
        if self.evidence.context:
            context_group = QGroupBox("Context")
            context_layout = QVBoxLayout(context_group)
            
            context_text = QTextEdit()
            context_text.setPlainText(self.evidence.context)
            context_text.setReadOnly(True)
            context_layout.addWidget(context_text)
            
            main_layout.addWidget(context_group)
        
        # Source information
        source_group = QGroupBox("Source")
        source_layout = QFormLayout(source_group)
        
        title_label = QLabel(self.evidence.paper_title)
        title_label.setWordWrap(True)
        source_layout.addRow("Paper Title:", title_label)
        
        if self.evidence.paper_doi:
            doi_layout = QHBoxLayout()
            doi_label = QLabel(self.evidence.paper_doi)
            doi_layout.addWidget(doi_label)
            
            open_doi_btn = QToolButton()
            open_doi_btn.setIcon(load_bootstrap_icon("box-arrow-up-right"))
            open_doi_btn.setToolTip("Open DOI")
            open_doi_btn.clicked.connect(lambda: QDesktopServices.openUrl(
                QUrl(f"https://doi.org/{self.evidence.paper_doi}")
            ))
            doi_layout.addWidget(open_doi_btn)
            
            source_layout.addRow("DOI:", doi_layout)
        
        if self.evidence.page_number:
            source_layout.addRow("Page:", QLabel(str(self.evidence.page_number)))
        
        main_layout.addWidget(source_group)
        
        # Classification section
        class_group = QGroupBox("Classification")
        class_layout = QVBoxLayout(class_group)
        
        # Type radio buttons
        type_layout = QHBoxLayout()
        type_label = QLabel("Evidence Type:")
        type_layout.addWidget(type_label)
        
        self.type_group = QButtonGroup(self)
        
        self.support_radio = QRadioButton("Supporting")
        self.refute_radio = QRadioButton("Refuting")
        self.neutral_radio = QRadioButton("Neutral")
        
        self.type_group.addButton(self.support_radio, 1)
        self.type_group.addButton(self.refute_radio, 2)
        self.type_group.addButton(self.neutral_radio, 3)
        
        # Set initial state
        if self.evidence.evidence_type == "support":
            self.support_radio.setChecked(True)
        elif self.evidence.evidence_type == "refute":
            self.refute_radio.setChecked(True)
        else:
            self.neutral_radio.setChecked(True)
        
        # Set radio buttons to non-editable if needed
        self.support_radio.setEnabled(self.editable)
        self.refute_radio.setEnabled(self.editable)
        self.neutral_radio.setEnabled(self.editable)
        
        type_layout.addWidget(self.support_radio)
        type_layout.addWidget(self.refute_radio)
        type_layout.addWidget(self.neutral_radio)
        type_layout.addStretch()
        
        class_layout.addLayout(type_layout)
        
        # Confidence slider
        conf_layout = QHBoxLayout()
        conf_label = QLabel("Confidence:")
        conf_layout.addWidget(conf_label)
        
        self.conf_slider = QSpinBox()
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setSuffix("%")
        self.conf_slider.setValue(self.evidence.confidence)
        self.conf_slider.setEnabled(self.editable)
        
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addStretch()
        
        class_layout.addLayout(conf_layout)
        
        main_layout.addWidget(class_group)
        
        # === Notes Tab ===
        notes_tab = QWidget()
        notes_layout = QVBoxLayout(notes_tab)
        
        notes_label = QLabel("Add your notes about this evidence:")
        notes_layout.addWidget(notes_label)
        
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlainText(self.evidence.notes)
        self.notes_edit.setReadOnly(not self.editable)
        notes_layout.addWidget(self.notes_edit)
        
        # === Tags Tab ===
        tags_tab = QWidget()
        tags_layout = QVBoxLayout(tags_tab)
        
        # Current tags
        tags_label = QLabel("Current Tags:")
        tags_layout.addWidget(tags_label)
        
        self.tags_list = QListWidget()
        for tag in self.evidence.tags:
            self.tags_list.addItem(tag)
        tags_layout.addWidget(self.tags_list)
        
        # Add new tag
        if self.editable:
            add_tag_layout = QHBoxLayout()
            
            self.new_tag_edit = QLineEdit()
            self.new_tag_edit.setPlaceholderText("Enter new tag...")
            add_tag_layout.addWidget(self.new_tag_edit)
            
            add_tag_btn = QPushButton("Add Tag")
            add_tag_btn.setIcon(load_bootstrap_icon("plus"))
            add_tag_btn.clicked.connect(self.add_tag)
            add_tag_layout.addWidget(add_tag_btn)
            
            tags_layout.addLayout(add_tag_layout)
        
        # Add tabs
        tabs.addTab(main_tab, "Details")
        tabs.addTab(notes_tab, "Notes")
        tabs.addTab(tags_tab, "Tags")
        
        layout.addWidget(tabs)
        
        # Buttons
        button_box = QDialogButtonBox()
        
        if self.editable:
            button_box.addButton(QDialogButtonBox.StandardButton.Save)
            button_box.addButton(QDialogButtonBox.StandardButton.Cancel)
            button_box.accepted.connect(self.save_changes)
            button_box.rejected.connect(self.reject)
        else:
            button_box.addButton(QDialogButtonBox.StandardButton.Close)
            button_box.rejected.connect(self.reject)
        
        layout.addWidget(button_box)
    
    def add_tag(self):
        """Add a new tag to the evidence."""
        tag = self.new_tag_edit.text().strip()
        if tag:
            # Check if tag already exists
            existing_items = [self.tags_list.item(i).text() for i in range(self.tags_list.count())]
            if tag not in existing_items:
                self.tags_list.addItem(tag)
            self.new_tag_edit.clear()
    
    def save_changes(self):
        """Save changes to the evidence item."""
        # Update evidence type
        if self.support_radio.isChecked():
            self.evidence.evidence_type = "support"
        elif self.refute_radio.isChecked():
            self.evidence.evidence_type = "refute"
        else:
            self.evidence.evidence_type = "neutral"
        
        # Update other fields
        self.evidence.text = self.text_edit.toPlainText()
        self.evidence.confidence = self.conf_slider.value()
        self.evidence.notes = self.notes_edit.toPlainText()
        
        # Update tags
        self.evidence.tags = [self.tags_list.item(i).text() 
                            for i in range(self.tags_list.count())]
        
        # Emit update signal
        self.evidenceUpdated.emit(self.evidence)
        
        # Close dialog
        self.accept()

class ClaimDialog(QDialog):
    """Dialog for creating and editing claims."""
    
    claimUpdated = pyqtSignal(Claim)
    
    def __init__(self, claim=None, parent=None):
        super().__init__(parent)
        self.claim = claim or Claim("")
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the dialog UI."""
        self.setWindowTitle("Edit Claim" if self.claim.text else "New Claim")
        self.resize(700, 500)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Create tabs
        tabs = QTabWidget()
        
        # === Main Tab ===
        main_tab = QWidget()
        main_layout = QVBoxLayout(main_tab)
        
        # Claim text
        claim_group = QGroupBox("Claim Text")
        claim_layout = QVBoxLayout(claim_group)
        
        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(self.claim.text)
        claim_layout.addWidget(self.text_edit)
        
        main_layout.addWidget(claim_group)
        
        # Source information
        source_group = QGroupBox("Source (Optional)")
        source_layout = QFormLayout(source_group)
        
        self.title_edit = QLineEdit()
        if self.claim.source_title:
            self.title_edit.setText(self.claim.source_title)
        source_layout.addRow("Paper Title:", self.title_edit)
        
        self.doi_edit = QLineEdit()
        if self.claim.source_doi:
            self.doi_edit.setText(self.claim.source_doi)
        source_layout.addRow("DOI:", self.doi_edit)
        
        main_layout.addWidget(source_group)
        
        # === Notes Tab ===
        notes_tab = QWidget()
        notes_layout = QVBoxLayout(notes_tab)
        
        notes_label = QLabel("Notes about this claim:")
        notes_layout.addWidget(notes_label)
        
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlainText(self.claim.notes)
        notes_layout.addWidget(self.notes_edit)
        
        # === Tags Tab ===
        tags_tab = QWidget()
        tags_layout = QVBoxLayout(tags_tab)
        
        # Current tags
        tags_label = QLabel("Current Tags:")
        tags_layout.addWidget(tags_label)
        
        self.tags_list = QListWidget()
        for tag in self.claim.tags:
            self.tags_list.addItem(tag)
        tags_layout.addWidget(self.tags_list)
        
        # Add new tag
        add_tag_layout = QHBoxLayout()
        
        self.new_tag_edit = QLineEdit()
        self.new_tag_edit.setPlaceholderText("Enter new tag...")
        add_tag_layout.addWidget(self.new_tag_edit)
        
        add_tag_btn = QPushButton("Add Tag")
        add_tag_btn.setIcon(load_bootstrap_icon("plus"))
        add_tag_btn.clicked.connect(self.add_tag)
        add_tag_layout.addWidget(add_tag_btn)
        
        tags_layout.addLayout(add_tag_layout)
        
        # === Evidence Tab ===
        evidence_tab = QWidget()
        evidence_layout = QVBoxLayout(evidence_tab)
        
        if not self.claim.evidence:
            evidence_layout.addWidget(QLabel("No evidence items added yet."))
        else:
            # Display existing evidence
            self.evidence_table = EvidenceTable()
            self.evidence_table.set_evidence(self.claim.evidence)
            evidence_layout.addWidget(self.evidence_table)
        
        # Add tabs
        tabs.addTab(main_tab, "Details")
        tabs.addTab(notes_tab, "Notes")
        tabs.addTab(tags_tab, "Tags")
        tabs.addTab(evidence_tab, "Evidence")
        
        layout.addWidget(tabs)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.save_changes)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def add_tag(self):
        """Add a new tag to the claim."""
        tag = self.new_tag_edit.text().strip()
        if tag:
            # Check if tag already exists
            existing_items = [self.tags_list.item(i).text() for i in range(self.tags_list.count())]
            if tag not in existing_items:
                self.tags_list.addItem(tag)
            self.new_tag_edit.clear()
    
    def save_changes(self):
        """Save changes to the claim."""
        # Update claim text
        self.claim.text = self.text_edit.toPlainText()
        
        # Update source information
        self.claim.source_title = self.title_edit.text()
        self.claim.source_doi = self.doi_edit.text()
        
        # Update notes
        self.claim.notes = self.notes_edit.toPlainText()
        
        # Update tags
        self.claim.tags = [self.tags_list.item(i).text() 
                          for i in range(self.tags_list.count())]
        
        # Emit update signal
        self.claimUpdated.emit(self.claim)
        
        # Close dialog
        self.accept()

class ClaimDetailDialog(QDialog):
    """Dialog for viewing claim details and its supporting/refuting evidence."""
    
    claimUpdated = pyqtSignal(Claim)
    
    def __init__(self, claim, parent=None):
        super().__init__(parent)
        self.claim = claim
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the dialog UI."""
        self.setWindowTitle("Claim Analysis")
        self.resize(900, 700)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Claim details section
        details_group = QGroupBox("Claim Details")
        details_layout = QVBoxLayout(details_group)
        
        # Claim text
        claim_text = QTextEdit()
        claim_text.setReadOnly(True)
        claim_text.setPlainText(self.claim.text)
        claim_text.setMaximumHeight(100)
        details_layout.addWidget(claim_text)
        
        # Source and assessment
        info_layout = QHBoxLayout()
        
        if self.claim.source_title:
            source_text = f"<b>Source:</b> {self.claim.source_title}"
            if self.claim.source_doi:
                source_text += f" (<a href='https://doi.org/{self.claim.source_doi}'>{self.claim.source_doi}</a>)"
            source_label = QLabel(source_text)
            source_label.setOpenExternalLinks(True)
            info_layout.addWidget(source_label)
        
        assessment = self.claim.overall_assessment()
        assessment_label = QLabel(f"<b>Assessment:</b> {assessment}")
        style = self._get_assessment_style(assessment)
        assessment_label.setStyleSheet(style)
        info_layout.addWidget(assessment_label)
        
        details_layout.addLayout(info_layout)
        
        # Evidence summary
        supporting, refuting, neutral = self.claim.evidence_summary()
        summary_layout = QHBoxLayout()
        
        support_label = QLabel(f"<span style='color:green'><b>Supporting:</b> {supporting}</span>")
        refute_label = QLabel(f"<span style='color:red'><b>Refuting:</b> {refuting}</span>")
        neutral_label = QLabel(f"<span style='color:gray'><b>Neutral:</b> {neutral}</span>")
        
        summary_layout.addWidget(support_label)
        summary_layout.addWidget(refute_label)
        summary_layout.addWidget(neutral_label)
        summary_layout.addStretch()
        
        if self.claim.tags:
            tags_text = f"<b>Tags:</b> {', '.join(self.claim.tags)}"
            tags_label = QLabel(tags_text)
            summary_layout.addWidget(tags_label)
        
        details_layout.addLayout(summary_layout)
        
        layout.addWidget(details_group)
        
        # Evidence tabs
        evidence_tabs = QTabWidget()
        
        # Supporting evidence
        if supporting > 0:
            support_tab = QWidget()
            support_layout = QVBoxLayout(support_tab)
            
            support_table = EvidenceTable()
            support_items = [e for e in self.claim.evidence if e.evidence_type == "support"]
            support_table.set_evidence(support_items)
            support_table.evidenceSelected.connect(self.view_evidence)
            
            support_layout.addWidget(support_table)
            evidence_tabs.addTab(support_tab, "Supporting Evidence")
        
        # Refuting evidence
        if refuting > 0:
            refute_tab = QWidget()
            refute_layout = QVBoxLayout(refute_tab)
            
            refute_table = EvidenceTable()
            refute_items = [e for e in self.claim.evidence if e.evidence_type == "refute"]
            refute_table.set_evidence(refute_items)
            refute_table.evidenceSelected.connect(self.view_evidence)
            
            refute_layout.addWidget(refute_table)
            evidence_tabs.addTab(refute_tab, "Refuting Evidence")
        
        # Neutral evidence
        if neutral > 0:
            neutral_tab = QWidget()
            neutral_layout = QVBoxLayout(neutral_tab)
            
            neutral_table = EvidenceTable()
            neutral_items = [e for e in self.claim.evidence if e.evidence_type == "neutral"]
            neutral_table.set_evidence(neutral_items)
            neutral_table.evidenceSelected.connect(self.view_evidence)
            
            neutral_layout.addWidget(neutral_table)
            evidence_tabs.addTab(neutral_tab, "Neutral Evidence")
        
        # All evidence
        all_tab = QWidget()
        all_layout = QVBoxLayout(all_tab)
        
        self.all_evidence_table = EvidenceTable()
        self.all_evidence_table.set_evidence(self.claim.evidence)
        self.all_evidence_table.evidenceSelected.connect(self.view_evidence)
        self.all_evidence_table.evidenceClassified.connect(self.on_evidence_classified)
        
        all_layout.addWidget(self.all_evidence_table)
        evidence_tabs.addTab(all_tab, "All Evidence")
        
        layout.addWidget(evidence_tabs)
        
        # Button row
        button_layout = QHBoxLayout()
        
        add_evidence_btn = QPushButton("Add Evidence")
        add_evidence_btn.setIcon(load_bootstrap_icon("plus-circle"))
        add_evidence_btn.clicked.connect(self.add_evidence)
        button_layout.addWidget(add_evidence_btn)
        
        edit_claim_btn = QPushButton("Edit Claim")
        edit_claim_btn.setIcon(load_bootstrap_icon("pencil"))
        edit_claim_btn.clicked.connect(self.edit_claim)
        button_layout.addWidget(edit_claim_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def _get_assessment_style(self, assessment):
        """Get CSS style for the assessment label."""
        if assessment == "Strongly Supported":
            return "color: darkgreen; font-weight: bold;"
        elif assessment == "Supported":
            return "color: green; font-weight: bold;"
        elif assessment == "Strongly Refuted":
            return "color: darkred; font-weight: bold;"
        elif assessment == "Refuted":
            return "color: red; font-weight: bold;"
        elif assessment == "Contested":
            return "color: darkorange; font-weight: bold;"
        else:
            return "color: gray; font-weight: bold;"
    
    def view_evidence(self, evidence):
        """Open dialog to view evidence details."""
        dialog = EvidenceDialog(evidence, self, editable=True)
        dialog.evidenceUpdated.connect(self.on_evidence_updated)
        dialog.exec()
    
    def on_evidence_updated(self, evidence):
        """Handle when evidence is updated."""
        # Update in the claim's evidence list
        for i, e in enumerate(self.claim.evidence):
            if e is evidence:  # Check if it's the same object
                self.claim.evidence[i] = evidence
                break
        
        # Refresh the table
        self.all_evidence_table.refresh_table()
        
        # Emit claim updated signal
        self.claimUpdated.emit(self.claim)
    
    def on_evidence_classified(self, evidence, new_type):
        """Handle when evidence classification changes."""
        # The evidence object is already updated in the table
        # Just emit the claim updated signal
        self.claimUpdated.emit(self.claim)
    
    def add_evidence(self):
        """Add new evidence to the claim."""
        # This would typically open a dialog to select text from a paper
        # For now, just create a blank evidence item
        evidence = EvidenceItem(
            text="Enter evidence text here",
            paper_doi="",
            paper_title="Select a paper",
            evidence_type="neutral",
            confidence=50
        )
        
        dialog = EvidenceDialog(evidence, self, editable=True)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Add the evidence to the claim
            self.claim.add_evidence(evidence)
            
            # Refresh the table
            self.all_evidence_table.refresh_table()
            
            # Emit claim updated signal
            self.claimUpdated.emit(self.claim)
    
    def edit_claim(self):
        """Edit the claim details."""
        dialog = ClaimDialog(self.claim, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # The claim object is already updated in the dialog
            # Just emit the claim updated signal
            self.claimUpdated.emit(self.claim)
            
            # Refresh the dialog title and claim text
            self.setWindowTitle(f"Claim Analysis: {self.claim.text[:30]}...")

class LiteratureEvidenceSection(QWidget):
    """
    Widget for collecting and analyzing evidence from papers.
    Focuses on extracting claims, collecting supporting/refuting evidence,
    and analyzing the strength of hypotheses based on the literature.
    """
    # Define signals
    papersNeeded = pyqtSignal()  # Signal when papers are needed
    claimsSaved = pyqtSignal(list)  # Signal when claims are saved
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
        # Initialize data containers
        self.papers = []
        self.claims = []
        self.studies_manager = None
        self.quote_extractor = None
        self.hypotheses_manager = None  # Add reference to hypotheses manager
        
    def setup_ui(self):
        """Setup the UI components."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Create header
        header_layout = QHBoxLayout()
        
        title = QLabel("Evidence Collection")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Add hypothesis selector
        hypothesis_label = QLabel("Active Hypothesis:")
        header_layout.addWidget(hypothesis_label)
        
        self.hypothesis_combo = QComboBox()
        self.hypothesis_combo.setMinimumWidth(300)
        self.hypothesis_combo.currentIndexChanged.connect(self.on_hypothesis_changed)
        header_layout.addWidget(self.hypothesis_combo)
        
        # Add data source controls
        source_label = QLabel("Data Source:")
        header_layout.addWidget(source_label)
        
        self.data_source_combo = QComboBox()
        self.data_source_combo.setMinimumWidth(250)
        self.data_source_combo.currentIndexChanged.connect(self.load_selected_dataset)
        header_layout.addWidget(self.data_source_combo)
        
        refresh_btn = QToolButton()
        refresh_btn.setIcon(load_bootstrap_icon("arrow-repeat"))
        refresh_btn.setToolTip("Refresh data sources")
        refresh_btn.clicked.connect(self.refresh_data_sources)
        header_layout.addWidget(refresh_btn)
        
        main_layout.addLayout(header_layout)
        
        # Add a hypothesis view panel
        self.hypothesis_panel = QGroupBox("Current Hypothesis")
        hypothesis_layout = QVBoxLayout(self.hypothesis_panel)
        
        self.hypothesis_title = QLabel()
        self.hypothesis_title.setWordWrap(True)
        self.hypothesis_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        hypothesis_layout.addWidget(self.hypothesis_title)
        
        self.hypothesis_description = QTextEdit()
        self.hypothesis_description.setReadOnly(True)
        self.hypothesis_description.setMaximumHeight(100)
        hypothesis_layout.addWidget(self.hypothesis_description)
        
        # Add this panel to the top of the main layout
        main_layout.insertWidget(1, self.hypothesis_panel)
        
        # Add an "Update Hypothesis with Evidence" button
        hypothesis_actions_layout = QHBoxLayout()
        
        self.update_hypothesis_btn = QPushButton("Update Hypothesis with Evidence")
        self.update_hypothesis_btn.setIcon(load_bootstrap_icon("arrow-up-circle"))
        self.update_hypothesis_btn.clicked.connect(self.update_hypothesis_with_evidence)
        hypothesis_actions_layout.addWidget(self.update_hypothesis_btn)
        
        self.extract_for_hypothesis_btn = QPushButton("Extract Evidence from All Papers")
        self.extract_for_hypothesis_btn.setIcon(load_bootstrap_icon("search"))
        self.extract_for_hypothesis_btn.clicked.connect(self.extract_evidence_for_hypothesis)
        hypothesis_actions_layout.addWidget(self.extract_for_hypothesis_btn)
        
        # Add below the hypothesis panel
        main_layout.insertLayout(2, hypothesis_actions_layout)
        
        # Create splitter for top/bottom sections
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # === TOP SECTION - CLAIMS ===
        claims_widget = QWidget()
        claims_layout = QVBoxLayout(claims_widget)
        claims_layout.setSpacing(10)
        
        # Claims header with controls
        claims_header = QHBoxLayout()
        
        claims_title = QLabel("Claims Analysis")
        claims_title_font = QFont()
        claims_title_font.setPointSize(14)
        claims_title_font.setBold(True)
        claims_title.setFont(claims_title_font)
        claims_header.addWidget(claims_title)
        
        # Filter and sort controls
        filter_layout = QHBoxLayout()
        filter_label = QLabel("Filter:")
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter claims...")
        self.filter_input.textChanged.connect(self.filter_claims)
        
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.filter_input)
        
        claims_header.addLayout(filter_layout)
        claims_header.addStretch()
        
        # Add claim button
        add_claim_btn = QPushButton("New Claim")
        add_claim_btn.setIcon(load_bootstrap_icon("plus-circle"))
        add_claim_btn.clicked.connect(self.add_new_claim)
        claims_header.addWidget(add_claim_btn)
        
        # # Load/save buttons
        # load_claims_btn = QPushButton("Load Claims")
        # load_claims_btn.setIcon(load_bootstrap_icon("folder-open"))
        # load_claims_btn.clicked.connect(self.load_claims)
        # claims_header.addWidget(load_claims_btn)
        
        # save_claims_btn = QPushButton("Save Claims")
        # save_claims_btn.setIcon(load_bootstrap_icon("save"))
        # save_claims_btn.clicked.connect(self.save_claims)
        # claims_header.addWidget(save_claims_btn)
        
        claims_layout.addLayout(claims_header)
        
        # Claims scroll area
        self.claims_scroll = QScrollArea()
        self.claims_scroll.setWidgetResizable(True)
        self.claims_container = QWidget()
        self.claims_layout = QVBoxLayout(self.claims_container)
        self.claims_layout.setSpacing(10)
        self.claims_scroll.setWidget(self.claims_container)
        
        claims_layout.addWidget(self.claims_scroll)
        
        # === BOTTOM SECTION - PAPERS & EXTRACTION ===
        extraction_widget = QWidget()
        extraction_layout = QVBoxLayout(extraction_widget)
        extraction_layout.setSpacing(10)
        
        # Papers header
        papers_header = QHBoxLayout()
        
        papers_title = QLabel("Papers")
        papers_title_font = QFont()
        papers_title_font.setPointSize(14)
        papers_title_font.setBold(True)
        papers_title.setFont(papers_title_font)
        papers_header.addWidget(papers_title)
        
        papers_header.addStretch()
        
        # Import papers button
        import_papers_btn = QPushButton("Import Papers")
        import_papers_btn.setIcon(load_bootstrap_icon("cloud-download"))
        import_papers_btn.clicked.connect(lambda: self.papersNeeded.emit())
        papers_header.addWidget(import_papers_btn)
        
        # Extract claims button
        extract_btn = QPushButton("Extract Claims")
        extract_btn.setIcon(load_bootstrap_icon("file-earmark-text"))
        extract_btn.clicked.connect(self.extract_claims_from_paper)
        papers_header.addWidget(extract_btn)
        
        extraction_layout.addLayout(papers_header)
        
        # Papers list - use a tabular format
        self.papers_table = QTableWidget()
        self.papers_table.setColumnCount(4)
        self.papers_table.setHorizontalHeaderLabels(["Title", "Authors", "Journal", "Actions"])
        self.papers_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.papers_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        extraction_layout.addWidget(self.papers_table)
        
        # Add widgets to splitter
        self.main_splitter.addWidget(claims_widget)
        self.main_splitter.addWidget(extraction_widget)
        
        # Set initial sizes (60% top, 40% bottom)
        self.main_splitter.setSizes([600, 400])
        
        main_layout.addWidget(self.main_splitter)
        
        # Status bar
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        
        main_layout.addLayout(status_layout)
    
    def set_studies_manager(self, studies_manager: StudiesManager):
        """Set the studies manager for accessing saved datasets."""
        self.studies_manager = studies_manager
        self.refresh_data_sources()
        # Refresh hypotheses if the hypotheses manager is set
        if self.hypotheses_manager:
            self.refresh_hypotheses()
    
    def refresh_data_sources(self):
        """Refresh the list of available datasets from studies manager."""
        if not self.studies_manager:
            logging.warning("Studies manager not set, cannot refresh data sources")
            return
            
        # Clear existing items
        self.data_source_combo.clear()
        
        # Get active study
        active_study = self.studies_manager.get_active_study()
        if not active_study:
            self.data_source_combo.addItem("No active study")
            return
            
        # Get datasets
        datasets = self.studies_manager.get_datasets_for_study(active_study)
        if not datasets:
            self.data_source_combo.addItem("No datasets available")
            return
            
        # Add search result datasets and ranked papers datasets
        valid_datasets = [name for name in datasets if 
                         name.startswith("search_results_") or 
                         name.startswith("ranked_papers_") or
                         name.startswith("summarized_papers_")]
        
        if valid_datasets:
            for dataset in valid_datasets:
                self.data_source_combo.addItem(dataset)
        else:
            self.data_source_combo.addItem("No valid datasets available")
    
    def load_selected_dataset(self):
        """Load the selected dataset from the dropdown."""
        if not self.studies_manager:
            return
            
        dataset_name = self.data_source_combo.currentText()
        if not dataset_name or dataset_name in ["No active study", "No datasets available", "No valid datasets available"]:
            self.papers = []
            self.update_papers_table()
            return
            
        try:
            # Get active study
            active_study = self.studies_manager.get_active_study()
            if not active_study:
                return
                
            # Load dataset
            df = self.studies_manager.get_dataset(active_study, dataset_name)
            if df is None or df.empty:
                logging.warning(f"Dataset {dataset_name} is empty")
                self.papers = []
                self.update_papers_table()
                return
                
            # Convert DataFrame to list of dictionaries
            self.papers = df.to_dict('records')
            
            # Update papers table
            self.update_papers_table()
            
            # Update status
            self.status_label.setText(f"Loaded {len(self.papers)} papers from {dataset_name}")
            
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load dataset: {str(e)}")
    
    def update_papers_table(self):
        """Update the papers table with current papers."""
        self.papers_table.clearContents()
        self.papers_table.setRowCount(len(self.papers))
        
        for row, paper in enumerate(self.papers):
            # Title
            title = paper.get('title', 'No Title')
            self.papers_table.setItem(row, 0, QTableWidgetItem(title))
            
            # Authors
            authors = paper.get('authors', [])
            if isinstance(authors, list):
                authors_str = ", ".join(authors[:2])
                if len(authors) > 2:
                    authors_str += f" +{len(authors)-2}"
            else:
                authors_str = str(authors)
            self.papers_table.setItem(row, 1, QTableWidgetItem(authors_str))
            
            # Journal
            journal = paper.get('journal', '')
            self.papers_table.setItem(row, 2, QTableWidgetItem(journal))
            
            # Actions
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 2, 2, 2)
            actions_layout.setSpacing(2)
            
            # Extract claims button
            extract_btn = QToolButton()
            extract_btn.setIcon(load_bootstrap_icon("file-text"))
            extract_btn.setToolTip("Extract Claims")
            extract_btn.clicked.connect(lambda checked=False, p=paper: self.extract_claims_from_paper(p))
            actions_layout.addWidget(extract_btn)
            
            # View paper button
            view_btn = QToolButton()
            view_btn.setIcon(load_bootstrap_icon("eye"))
            view_btn.setToolTip("View Paper")
            view_btn.clicked.connect(lambda checked=False, p=paper: self.view_paper(p))
            actions_layout.addWidget(view_btn)
            
            self.papers_table.setCellWidget(row, 3, actions_widget)
    
    def update_claims_display(self):
        """Update the claims display with current claims."""
        # Clear existing claims
        self.clear_layout(self.claims_layout)
        
        if not self.claims:
            no_claims_label = QLabel("No claims have been added yet. Add a claim manually or extract from papers.")
            no_claims_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.claims_layout.addWidget(no_claims_label)
            return
            
        # Add claims to layout
        for claim in self.claims:
            claim_widget = ClaimWidget(claim)
            claim_widget.claimSelected.connect(self.view_claim)
            self.claims_layout.addWidget(claim_widget)
            
        # Add a stretch to the end
        self.claims_layout.addStretch()
    
    def clear_layout(self, layout):
        """Helper function to clear all widgets from a layout."""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            elif item.layout() is not None:
                self.clear_layout(item.layout())
    
    def add_new_claim(self):
        """Add a new claim manually."""
        dialog = ClaimDialog(parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.claims.append(dialog.claim)
            self.update_claims_display()
    
    def view_claim(self, claim):
        """View and edit a claim's details."""
        dialog = ClaimDetailDialog(claim, self)
        dialog.claimUpdated.connect(self.on_claim_updated)
        dialog.exec()
    
    def on_claim_updated(self, claim):
        """Handle when a claim is updated."""
        # The claim object is already updated because we're passing by reference
        # Just update the display
        self.update_claims_display()
    
    def filter_claims(self):
        """Filter claims based on search text."""
        filter_text = self.filter_input.text().lower()
        
        # Clear existing claims
        self.clear_layout(self.claims_layout)
        
        if not self.claims:
            no_claims_label = QLabel("No claims have been added yet. Add a claim manually or extract from papers.")
            no_claims_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.claims_layout.addWidget(no_claims_label)
            return
            
        # Filter claims
        filtered_claims = []
        for claim in self.claims:
            if (filter_text in claim.text.lower() or
                (claim.source_title and filter_text in claim.source_title.lower()) or
                any(filter_text in tag.lower() for tag in claim.tags) or
                filter_text in claim.notes.lower()):
                filtered_claims.append(claim)
        
        if not filtered_claims:
            no_match_label = QLabel(f"No claims match the filter '{filter_text}'")
            no_match_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.claims_layout.addWidget(no_match_label)
            return
            
        # Add filtered claims to layout
        for claim in filtered_claims:
            claim_widget = ClaimWidget(claim)
            claim_widget.claimSelected.connect(self.view_claim)
            self.claims_layout.addWidget(claim_widget)
            
        # Add a stretch to the end
        self.claims_layout.addStretch()
    
    def view_paper(self, paper):
        """View a paper's details."""
        from literature_search.search import PaperDetailDialog
        dialog = PaperDetailDialog(paper, self)
        dialog.exec()
    
    def set_hypotheses_manager(self, hypotheses_manager):
        """Set the hypotheses manager and update the available hypotheses."""
        self.hypotheses_manager = hypotheses_manager
        self.refresh_hypotheses()
    
    def refresh_hypotheses(self):
        """Refresh the list of available hypotheses from the hypotheses manager."""
        if not self.hypotheses_manager:
            return
            
        # Clear existing items
        self.hypothesis_combo.clear()
        
        # Get active study
        if not self.studies_manager:
            return
            
        active_study = self.studies_manager.get_active_study()
        if not active_study or not hasattr(active_study, 'hypotheses'):
            self.hypothesis_combo.addItem("No hypotheses available")
            self.hypothesis_panel.setVisible(False)
            return
        
        # Add hypotheses to dropdown
        hypotheses = active_study.hypotheses
        if not hypotheses:
            self.hypothesis_combo.addItem("No hypotheses available")
            self.hypothesis_panel.setVisible(False)
            return
        
        self.hypothesis_combo.addItem("Select a hypothesis...", None)
        
        for hypothesis in hypotheses:
            title = hypothesis.get('title', 'Untitled Hypothesis')
            self.hypothesis_combo.addItem(title, hypothesis.get('id'))
        
        self.hypothesis_panel.setVisible(False)
    
    def on_hypothesis_changed(self, index):
        """Handle when a different hypothesis is selected."""
        if index <= 0 or not self.studies_manager:
            self.hypothesis_panel.setVisible(False)
            return
        
        hypothesis_id = self.hypothesis_combo.itemData(index)
        active_study = self.studies_manager.get_active_study()
        
        # Find the hypothesis
        selected_hypothesis = None
        for hypothesis in active_study.hypotheses:
            if hypothesis.get('id') == hypothesis_id:
                selected_hypothesis = hypothesis
                break
        
        if not selected_hypothesis:
            self.hypothesis_panel.setVisible(False)
            return
        
        # Update the hypothesis panel
        self.hypothesis_title.setText(selected_hypothesis.get('title', 'Untitled Hypothesis'))
        
        # Format hypothesis details
        details = f"<b>Null Hypothesis (H₀):</b> {selected_hypothesis.get('null_hypothesis', '')}\n\n"
        details += f"<b>Alternative Hypothesis (H₁):</b> {selected_hypothesis.get('alternative_hypothesis', '')}\n\n"
        
        outcome_vars = selected_hypothesis.get('outcome_variables', '')
        if outcome_vars:
            details += f"<b>Outcome Variables:</b> {outcome_vars}\n\n"
        
        predictor_vars = selected_hypothesis.get('predictor_variables', '')
        if predictor_vars:
            details += f"<b>Predictor Variables:</b> {predictor_vars}\n\n"
        
        self.hypothesis_description.setHtml(details)
        self.hypothesis_panel.setVisible(True)
        
        # Check if we already have generated terms for this hypothesis
        if not selected_hypothesis.get('generated_terms'):
            # Show a loading message
            self.status_label.setText("Generating hypothesis-specific search terms...")
            QApplication.processEvents()  # Force UI update
            
            # Generate terms with LLM - use asyncSlot to handle this later
            self._generate_hypothesis_terms_async(selected_hypothesis)
        else:
            # Update filter and classification based on the generated terms
            self.classify_papers_by_relevance(selected_hypothesis)
        
        self.status_label.setText("Ready")
    
    def classify_papers_by_relevance(self, hypothesis):
        """Classify and highlight papers based on LLM-generated hypothesis terms."""
        if not hypothesis:
            return
        
        # Get generated terms if available
        hypothesis_terms = hypothesis.get('generated_terms', [])
        if not hypothesis_terms:
            # Fall back to simple term matching if no generated terms
            simple_terms = []
            
            # Add outcome variables
            outcome_vars = hypothesis.get('outcome_variables', '')
            if outcome_vars:
                simple_terms.extend([var.strip() for var in outcome_vars.split(',') if var.strip()])
                
            # Add predictor variables
            predictor_vars = hypothesis.get('predictor_variables', '')
            if predictor_vars:
                simple_terms.extend([var.strip() for var in predictor_vars.split(',') if var.strip()])
                
            self.highlight_relevant_papers(simple_terms)
            return
        
        # Reset all paper highlighting
        for row in range(self.papers_table.rowCount()):
            for col in range(self.papers_table.columnCount()):
                item = self.papers_table.item(row, col)
                if item:
                    item.setBackground(QColor(255, 255, 255))
        
        # Create a mapping of thematic clusters for visualization
        theme_colors = {
            "core": QColor(200, 255, 200),      # Light green
            "variable": QColor(230, 255, 230),  # Pale green
            "method": QColor(255, 255, 200),    # Light yellow
            "synonym": QColor(230, 230, 255),   # Light blue
            "field": QColor(255, 230, 230),     # Light pink
            "specific": QColor(230, 255, 255)   # Light cyan
        }
        
        # Process each paper
        paper_scores = []
        
        for row in range(self.papers_table.rowCount()):
            if row >= len(self.papers):
                continue
            
            paper = self.papers[row]
            paper_title = paper.get('title', '')
            paper_abstract = paper.get('abstract', '')
            paper_text = f"{paper_title} {paper_abstract}".lower()
            
            # Calculate overall relevance score and track term matches by category
            overall_score = 0
            category_matches = {category: 0 for category in theme_colors.keys()}
            matched_terms = []
            
            for term_data in hypothesis_terms:
                term = term_data.get('term', '').lower()
                weight = term_data.get('weight', 0.5)
                category = term_data.get('category', 'synonym')
                
                if term in paper_text:
                    # Calculate term score based on where it appears and its weight
                    term_score = weight
                    if term in paper_title.lower():
                        term_score *= 2  # Double score for title matches
                    
                    overall_score += term_score
                    category_matches[category] += 1
                    matched_terms.append(term)
            
            # Store paper score data
            paper_scores.append({
                'row': row,
                'paper': paper,
                'score': overall_score,
                'category_matches': category_matches,
                'matched_terms': matched_terms
            })
        
        # Sort papers by relevance score
        paper_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply colors based on dominant category and score
        for paper_data in paper_scores:
            row = paper_data['row']
            score = paper_data['score']
            
            if score <= 0:
                continue  # Skip papers with no matches
            
            # Determine dominant category
            categories = paper_data['category_matches']
            dominant_category = max(categories.items(), key=lambda x: x[1])[0]
            
            # Mix color intensity based on score (higher score = more intense color)
            base_color = theme_colors[dominant_category]
            
            # Apply scaling factor to normalize colors (adjust as needed)
            max_score = max(p['score'] for p in paper_scores) if paper_scores else 1
            intensity = min(1.0, score / max_score)
            
            # Create a color that's a blend between white and the base color
            blended_color = QColor(
                255 - int((255 - base_color.red()) * intensity),
                255 - int((255 - base_color.green()) * intensity),
                255 - int((255 - base_color.blue()) * intensity)
            )
            
            # Apply color to all cells in the row
            for col in range(self.papers_table.columnCount() - 1):  # Skip action column
                item = self.papers_table.item(row, col)
                if item:
                    item.setBackground(blended_color)
                    
                    # Add tooltip showing matched terms
                    if col == 0:  # Only on title column
                        matched_terms_str = ", ".join(paper_data['matched_terms'][:10])
                        if len(paper_data['matched_terms']) > 10:
                            matched_terms_str += f" (+{len(paper_data['matched_terms']) - 10} more)"
                        item.setToolTip(f"Relevance score: {score:.2f}\nMatched terms: {matched_terms_str}")
        
        # Update the papers table with the newly sorted/classified data
        self.display_classified_papers(paper_scores)
    
    def display_classified_papers(self, paper_scores):
        """Optionally reorder and display papers based on their classification."""
        # Ask user if they want to reorder papers by relevance
        reply = QMessageBox.question(
            self,
            "Reorder Papers",
            f"Would you like to reorder papers by relevance to the hypothesis?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Create a new ordering of papers based on scores
        reordered_papers = [score_data['paper'] for score_data in paper_scores]
        
        # Store original papers for restoration if needed
        self.original_papers = self.papers.copy() if not hasattr(self, 'original_papers') else self.original_papers
        
        # Update papers and refresh display
        self.papers = reordered_papers
        self.update_papers_table()
        
        # Create a color legend for the different categories
        self.show_category_legend()
    
    def show_category_legend(self):
        """Display a legend explaining the color coding by category."""
        legend_dialog = QDialog(self)
        legend_dialog.setWindowTitle("Paper Classification Legend")
        legend_dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(legend_dialog)
        
        # Title
        title = QLabel("<b>Paper Classification by Hypothesis Terms</b>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Explanation
        explanation = QLabel(
            "Papers are colored based on their relevance to different aspects "
            "of the hypothesis. Here's what the colors mean:"
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        
        # Create the legend grid
        legend_grid = QGridLayout()
        
        categories = {
            "core": "Core Hypothesis Concepts",
            "variable": "Variables and Measurements",
            "method": "Research Methods",
            "synonym": "Related Terminology",
            "field": "Broader Field Terms",
            "specific": "Specific Technical Terms"
        }
        
        theme_colors = {
            "core": QColor(200, 255, 200),      # Light green
            "variable": QColor(230, 255, 230),  # Pale green
            "method": QColor(255, 255, 200),    # Light yellow
            "synonym": QColor(230, 230, 255),   # Light blue
            "field": QColor(255, 230, 230),     # Light pink
            "specific": QColor(230, 255, 255)   # Light cyan
        }
        
        row = 0
        for category, description in categories.items():
            # Create color swatch
            swatch = QFrame()
            swatch.setFixedSize(24, 24)
            swatch.setStyleSheet(f"background-color: rgb({theme_colors[category].red()}, "
                                f"{theme_colors[category].green()}, {theme_colors[category].blue()}); "
                                f"border: 1px solid gray;")
            
            # Create description label
            label = QLabel(description)
            
            # Add to grid
            legend_grid.addWidget(swatch, row, 0)
            legend_grid.addWidget(label, row, 1)
            row += 1
        
        layout.addLayout(legend_grid)
        
        # Note about intensity
        note = QLabel(
            "<i>Note: Color intensity indicates relevance strength. "
            "Stronger colors mean higher relevance.</i>"
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        
        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(legend_dialog.reject)
        layout.addWidget(button_box)
        
        # Show dialog
        legend_dialog.exec()
    
    def restore_original_paper_order(self):
        """Restore papers to their original order."""
        if hasattr(self, 'original_papers'):
            self.papers = self.original_papers.copy()
            self.update_papers_table()
    
    @asyncSlot()
    async def extract_claims_from_paper(self, paper):
        """Extract evidence for the current hypothesis from a specific paper."""
        if not paper:
            return
            
        # Get the currently selected hypothesis
        hypothesis_index = self.hypothesis_combo.currentIndex()
        if hypothesis_index <= 0:
            QMessageBox.warning(self, "No Hypothesis Selected", 
                               "Please select a hypothesis before extracting evidence.")
            return
            
        hypothesis_id = self.hypothesis_combo.itemData(hypothesis_index)
        active_study = self.studies_manager.get_active_study()
        
        # Find the hypothesis
        selected_hypothesis = None
        for hypothesis in active_study.hypotheses:
            if hypothesis.get('id') == hypothesis_id:
                selected_hypothesis = hypothesis
                break
                
        if not selected_hypothesis:
            QMessageBox.warning(self, "Error", "Could not find the selected hypothesis.")
            return
            
        try:
            # Update status
            self.status_label.setText(f"Extracting evidence from '{paper.get('title', 'paper')}'...")
            self.progress_bar.setValue(10)
            self.progress_bar.setVisible(True)
            QApplication.processEvents()  # Force UI update
            
            # Initialize QuoteExtractor if needed
            if not self.quote_extractor:
                self.quote_extractor = QuoteExtractor(logging.getLogger(__name__))
            
            # Get paper information
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            full_text = paper.get('full_text', '')
            doi = paper.get('doi', '')
            
            # Prepare hypothesis context
            null_hypothesis = selected_hypothesis.get('null_hypothesis', '')
            alt_hypothesis = selected_hypothesis.get('alternative_hypothesis', '')
            outcome_vars = selected_hypothesis.get('outcome_variables', '')
            predictor_vars = selected_hypothesis.get('predictor_variables', '')
            
            # Get hypothesis terms if available
            hypothesis_terms = selected_hypothesis.get('generated_terms')
            
            # Extract evidence specific to this hypothesis
            evidence_items = await self._extract_evidence_with_ai(
                title, 
                abstract, 
                full_text,
                null_hypothesis,
                alt_hypothesis,
                outcome_vars,
                predictor_vars,
                hypothesis_terms
            )
            
            self.progress_bar.setValue(70)
            
            # Create evidence objects for this claim
            new_claims = []
            for evidence in evidence_items:
                claim = Claim(
                    text=f"Evidence regarding: {alt_hypothesis}",
                    source_doi=doi,
                    source_title=title
                )
                
                # Add evidence to the claim
                evidence_item = EvidenceItem(
                    text=evidence['text'],
                    paper_doi=doi,
                    paper_title=title,
                    evidence_type=evidence['type'],
                    confidence=evidence['confidence'],
                    context=evidence.get('context', '')
                )
                
                # Add relevant tags
                for tag in evidence.get('tags', []):
                    evidence_item.tags.append(tag)
                
                # Add related terms as tags with a prefix
                for term in evidence.get('related_terms', []):
                    evidence_item.tags.append(f"term:{term}")
                
                claim.add_evidence(evidence_item)
                new_claims.append(claim)
            
            # Add the claims to our list
            if new_claims:
                self.claims.extend(new_claims)
                self.update_claims_display()
                
                # Show success message
                count = len(new_claims)
                QMessageBox.information(
                    self, 
                    "Evidence Extracted", 
                    f"Successfully extracted {count} piece{'' if count == 1 else 's'} of evidence " +
                    f"from {title} related to the hypothesis."
                )
            else:
                QMessageBox.information(
                    self, 
                    "No Evidence Found", 
                    f"No relevant evidence was found in {title} for the selected hypothesis."
                )
            
            # Update status
            self.status_label.setText(f"Finished extracting evidence from '{title}'")
            self.progress_bar.setValue(100)
            
        except Exception as e:
            logging.error(f"Error extracting evidence: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to extract evidence: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
    
    async def _extract_evidence_with_ai(self, title, abstract, full_text, null_hypothesis, 
                                        alt_hypothesis, outcome_vars, predictor_vars, hypothesis_terms=None):
        """Extract evidence related to a specific hypothesis using AI, enhanced with generated terms."""
        try:
            from llms.client import call_llm_async_json
            
            # Use full text if available, otherwise use abstract
            text_to_analyze = full_text if full_text else abstract
            if not text_to_analyze:
                return []
                
            # Create a section with key terms if available
            terms_section = ""
            if hypothesis_terms:
                # Sort terms by weight
                sorted_terms = sorted(hypothesis_terms, key=lambda x: x.get('weight', 0), reverse=True)
                
                terms_by_category = {}
                for term_data in sorted_terms:
                    category = term_data.get('category', 'other')
                    term = term_data.get('term', '')
                    
                    if category not in terms_by_category:
                        terms_by_category[category] = []
                        
                    terms_by_category[category].append(term)
                
                # Format the terms for the prompt
                terms_section = "KEY TERMS BY CATEGORY:\n"
                for category, terms in terms_by_category.items():
                    terms_section += f"{category.capitalize()}: {', '.join(terms[:10])}\n"
                    if len(terms) > 10:
                        terms_section += f"  (and {len(terms) - 10} more {category} terms)\n"
                
                terms_section += "\n"
                
            # Actually limit the text to 8000 characters
            text_to_analyze_limited = text_to_analyze[:8000]
                
            prompt = f"""
            Analyze the following scientific paper to find evidence related to this specific hypothesis:

            Paper Title: {title}
            
            Text to analyze: {text_to_analyze_limited}
            
            HYPOTHESIS INFORMATION:
            Null Hypothesis (H₀): {null_hypothesis}
            Alternative Hypothesis (H₁): {alt_hypothesis}
            Outcome Variables: {outcome_vars}
            Predictor Variables: {predictor_vars}
            
            {terms_section}
            
            Your task: Extract specific quotes, data, or statements from the paper that either SUPPORT, 
            REFUTE, or provide NEUTRAL evidence for the alternative hypothesis. 
            
            For each piece of evidence:
            1. Extract the exact text that contains the evidence
            2. Classify whether it supports, refutes, or is neutral to the alternative hypothesis
            3. Provide the surrounding context if relevant
            4. Rate your confidence in this assessment (0-100)
            5. Add relevant tags that categorize the evidence (methodology, data, results, limitations, etc.)
            6. Identify which key terms from the hypothesis are addressed in this evidence
            
            Return the results in the following JSON format:
            {{
              "evidence": [
                {{
                  "text": "The exact quote or data point from the paper",
                  "type": "support|refute|neutral",
                  "confidence": 85,
                  "context": "Surrounding text that helps understand the evidence (optional)",
                  "tags": ["methodology", "results", "limitation", etc.],
                  "related_terms": ["term1", "term2"]
                }},
                // Additional evidence items...
              ]
            }}
            
            Extract only evidence that is DIRECTLY relevant to the specific hypothesis. Aim for 1-5 pieces of 
            high-quality evidence with clear relevance to the hypothesis variables.
            """
            
            # Call the LLM using the async JSON function
            result = await call_llm_async_json(prompt, model="claude-3-7-sonnet-20250219")
            
            # Extract and return the evidence
            evidence_items = []
            if "evidence" in result and isinstance(result["evidence"], list):
                evidence_items = result["evidence"]
            
            # If no evidence was found, return an empty list
            if not evidence_items:
                logging.warning(f"No evidence extracted from paper: {title}")
                
            return evidence_items
            
        except Exception as e:
            logging.error(f"Error in AI evidence extraction: {str(e)}")
            return []
    
    # Add a new method to update hypothesis status based on evidence
    def update_hypothesis_with_evidence(self):
        """Update the current hypothesis with the collected evidence."""
        hypothesis_index = self.hypothesis_combo.currentIndex()
        if hypothesis_index <= 0 or not self.claims:
            return
            
        hypothesis_id = self.hypothesis_combo.itemData(hypothesis_index)
        if not hypothesis_id:
            return
            
        # Count evidence types
        supporting = 0
        refuting = 0
        neutral = 0
        
        # Collect all evidence from claims
        all_evidence = []
        for claim in self.claims:
            all_evidence.extend(claim.evidence)
        
        # Count by type
        for evidence in all_evidence:
            if evidence.evidence_type == "support":
                supporting += 1
            elif evidence.evidence_type == "refute":
                refuting += 1
            else:
                neutral += 1
        
        # Only proceed if we have evidence
        if supporting + refuting + neutral == 0:
            return
            
        # Create a summary of the evidence
        summary = f"Based on literature evidence: {supporting} supporting, {refuting} refuting, {neutral} neutral"
        
        # Calculate confidence based on evidence
        total_directional = supporting + refuting
        if total_directional == 0:
            status = "inconclusive"
            conclusion = "Insufficient directional evidence"
        elif supporting > refuting * 2:  # More than twice as much supporting evidence
            status = "confirmed"
            conclusion = "Strongly supported by literature evidence"
        elif supporting > refuting:
            status = "confirmed"  
            conclusion = "Supported by literature evidence"
        elif refuting > supporting * 2:
            status = "rejected"
            conclusion = "Strongly refuted by literature evidence"
        elif refuting > supporting:
            status = "rejected"
            conclusion = "Refuted by literature evidence"
        else:
            status = "inconclusive"
            conclusion = "Contested in the literature"
        
        # Ask user for confirmation
        reply = QMessageBox.question(
                self,
            "Update Hypothesis Status",
            f"Evidence summary: {supporting} supporting, {refuting} refuting, {neutral} neutral.\n\n"
            f"Suggested status: {status.capitalize()}\n\n"
            f"Would you like to update the hypothesis status based on this evidence?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
            
        # Update the hypothesis
        active_study = self.studies_manager.get_active_study()
        for i, hyp in enumerate(active_study.hypotheses):
            if hyp.get('id') == hypothesis_id:
                # Add literature evidence results
                active_study.hypotheses[i]['literature_evidence'] = {
                    'supporting': supporting,
                    'refuting': refuting,
                    'neutral': neutral,
                    'status': status,
                    'conclusion': conclusion,
                    'updated_at': datetime.now().isoformat()
                }
                
                # Optionally update the overall status
                if 'status' in active_study.hypotheses[i] and active_study.hypotheses[i]['status'] == 'untested':
                    active_study.hypotheses[i]['status'] = status
                
                # Show success message
                QMessageBox.information(
                self,
                    "Hypothesis Updated",
                    f"The hypothesis has been updated with the literature evidence.\n\n"
                    f"New status: {status.capitalize()}"
                )
                
                # Refresh hypothesis display if hypotheses manager exists
                if self.hypotheses_manager:
                    self.hypotheses_manager.load_hypotheses()
                
                break

    @asyncSlot()
    async def extract_evidence_for_hypothesis(self):
        """Extract evidence for the current hypothesis from all loaded papers."""
        hypothesis_index = self.hypothesis_combo.currentIndex()
        if hypothesis_index <= 0:
            QMessageBox.warning(self, "No Hypothesis Selected", 
                               "Please select a hypothesis before extracting evidence.")
            return
            
        if not self.papers:
            QMessageBox.warning(self, "No Papers", "No papers available for evidence extraction.")
            return
            
        try:
            # Ask for confirmation
            num_papers = len(self.papers)
            reply = QMessageBox.question(
                self,
                "Extract Evidence",
                f"Do you want to analyze {num_papers} papers for evidence related to the selected hypothesis?\n"
                f"This may take some time.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.No:
                return
            
            hypothesis_id = self.hypothesis_combo.itemData(hypothesis_index)
            active_study = self.studies_manager.get_active_study()
            
            # Find the hypothesis
            selected_hypothesis = None
            for hypothesis in active_study.hypotheses:
                if hypothesis.get('id') == hypothesis_id:
                    selected_hypothesis = hypothesis
                    break
                    
            if not selected_hypothesis:
                QMessageBox.warning(self, "Error", "Could not find the selected hypothesis.")
                return
            
            # Get hypothesis details
            null_hypothesis = selected_hypothesis.get('null_hypothesis', '')
            alt_hypothesis = selected_hypothesis.get('alternative_hypothesis', '')
            outcome_vars = selected_hypothesis.get('outcome_variables', '')
            predictor_vars = selected_hypothesis.get('predictor_variables', '')
            
            # Update status
            self.status_label.setText("Extracting evidence from papers...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            QApplication.processEvents()  # Force UI update
            
            # Process papers one by one
            new_claims = []
            for i, paper in enumerate(self.papers):
                # Update progress
                progress = int((i / num_papers) * 100)
                self.progress_bar.setValue(progress)
                self.status_label.setText(f"Analyzing paper {i+1} of {num_papers}...")
                QApplication.processEvents()  # Force UI update
                
                # Skip papers without abstracts
                if not paper.get('abstract'):
                    continue
                
                # Get paper information
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                full_text = paper.get('full_text', '')
                doi = paper.get('doi', '')
                
                # Extract evidence
                evidence_items = await self._extract_evidence_with_ai(
                    title, 
                    abstract, 
                    full_text,
                    null_hypothesis,
                    alt_hypothesis,
                    outcome_vars,
                    predictor_vars
                )
                
                # Create claim and evidence for each finding
                if evidence_items:
                    claim = Claim(
                        text=f"Evidence regarding: {alt_hypothesis}",
                        source_doi=doi,
                        source_title=title
                    )
                    
                    # Add each evidence item to the claim
                    for evidence in evidence_items:
                        evidence_item = EvidenceItem(
                            text=evidence['text'],
                            paper_doi=doi,
                            paper_title=title,
                            evidence_type=evidence['type'],
                            confidence=evidence['confidence'],
                            context=evidence.get('context', '')
                        )
                        
                        # Add relevant tags
                        for tag in evidence.get('tags', []):
                            evidence_item.tags.append(tag)
                        
                        claim.add_evidence(evidence_item)
                    
                    new_claims.append(claim)
            
            # Add the claims to our list
            if new_claims:
                self.claims.extend(new_claims)
                self.update_claims_display()
                
                # Count evidence
                total_evidence = sum(len(claim.evidence) for claim in new_claims)
                
                # Show success message
                QMessageBox.information(
                    self, 
                    "Evidence Extraction Complete", 
                    f"Successfully extracted {total_evidence} pieces of evidence " +
                    f"from {len(new_claims)} papers related to the hypothesis."
                )
                
                # Ask if user wants to update hypothesis with new evidence
                update_reply = QMessageBox.question(
                    self,
                    "Update Hypothesis",
                    "Would you like to update the hypothesis status based on the collected evidence?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                
                if update_reply == QMessageBox.StandardButton.Yes:
                    self.update_hypothesis_with_evidence()
            else:
                QMessageBox.information(
                    self, 
                    "No Evidence Found", 
                    "No relevant evidence was found in the papers for the selected hypothesis."
                )
            
            # Update status
            self.status_label.setText("Finished extracting evidence")
            self.progress_bar.setValue(100)
            
        except Exception as e:
            logging.error(f"Error extracting evidence: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to extract evidence: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)

    # Add this new method to receive papers directly from the search section
    def set_papers(self, papers, query=None):
        """
        Set papers directly from the search section.
        
        Args:
            papers: List of paper dictionaries from the search section
            query: The search query that was used to find these papers
        """
        if not papers:
            self.status_label.setText("No papers received")
            self.papers = []
            self.display_papers([])
            return
            
        self.papers = papers.copy()  # Make a copy to avoid modifying the original
        self.query = query if query else ""  # Store the query for generating reviews later
        
        # Reset analysis results
        self.ranked_papers = []
        self.topics_data = {}
        
        # Display papers
        self.display_papers(self.papers)
        
        # Update status
        self.status_label.setText(f"Loaded {len(self.papers)} papers from search")
        
        # Switch to this section automatically
        # This needs to be connected in the main app to switch to this widget
        
        # Optionally store in studies manager for persistence
        if self.studies_manager:
            try:
                df = pd.DataFrame(self.papers)
                dataset_name = f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.studies_manager.add_dataset_to_active_study(
                    dataset_name,
                    df,
                    metadata={'query': query} if query else {}
                )
                self.loaded_dataset_name = dataset_name
            except Exception as e:
                logging.warning(f"Failed to store papers in studies manager: {str(e)}")
    
    async def generate_hypothesis_terms(self, hypothesis_data):
        """Use LLM to generate a comprehensive set of search terms and thematic tags for a hypothesis."""
        try:
            from llms.client import call_llm_async_json
            
            # Extract hypothesis information
            title = hypothesis_data.get('title', '')
            null_hypothesis = hypothesis_data.get('null_hypothesis', '')
            alt_hypothesis = hypothesis_data.get('alternative_hypothesis', '')
            outcome_vars = hypothesis_data.get('outcome_variables', '')
            predictor_vars = hypothesis_data.get('predictor_variables', '')
            
            prompt = f"""
            Analyze this scientific hypothesis and generate a comprehensive set of search terms and thematic tags 
            that would help identify relevant papers in the scientific literature.
            
            HYPOTHESIS INFORMATION:
            Title: {title}
            Null Hypothesis (H₀): {null_hypothesis}
            Alternative Hypothesis (H₁): {alt_hypothesis}
            Outcome Variables: {outcome_vars}
            Predictor Variables: {predictor_vars}
            
            Please generate the following for this hypothesis:
            
            1. Core Concepts: The main scientific concepts central to this hypothesis (5-10 terms)
            2. Variables and Measurements: Specific variables, metrics, or measurements related to the hypothesis (5-10 terms)
            3. Methods and Approaches: Research methods or approaches likely used to test this hypothesis (3-8 terms)
            4. Synonyms and Related Terms: Alternative terminology for the key concepts (5-15 terms)
            5. Broader Field Terms: Terms related to the broader scientific field this hypothesis belongs to (3-5 terms)
            6. Narrower Specific Terms: Very specific technical terms directly related to the hypothesis (3-8 terms)
            
            For each term, provide:
            - The term itself
            - A weight (0.0-1.0) indicating how relevant/important this term is to the hypothesis
            - A category label (one of: "core", "variable", "method", "synonym", "field", "specific")
            
            Return the results in the following JSON format:
            {{
              "hypothesis_terms": [
                {{
                  "term": "example term",
                  "weight": 0.9,
                  "category": "core"
                }},
                // Additional terms...
              ],
              "thematic_clusters": [
                {{
                  "name": "Cluster name (e.g., 'Statistical Methods')",
                  "terms": ["term1", "term2", "term3"]
                }},
                // Additional clusters...
              ]
            }}
            
            Ensure that terms are specific enough to be valuable for finding relevant papers but general enough 
            to match against standard scientific terminology. Include singular and plural forms only where truly necessary.
            """
            
            # Call the LLM
            result = await call_llm_async_json(prompt, model="claude-3-7-sonnet-20250219")
            
            # Store the generated terms in the hypothesis data
            if "hypothesis_terms" in result:
                hypothesis_data['generated_terms'] = result["hypothesis_terms"]
            
            if "thematic_clusters" in result:
                hypothesis_data['thematic_clusters'] = result["thematic_clusters"]
                
            # Update the hypothesis in the study
            active_study = self.studies_manager.get_active_study()
            for i, hyp in enumerate(active_study.hypotheses):
                if hyp.get('id') == hypothesis_data.get('id'):
                    active_study.hypotheses[i] = hypothesis_data
                    break
            
            return result
        
        except Exception as e:
            logging.error(f"Error generating hypothesis terms: {str(e)}")
            return {"hypothesis_terms": [], "thematic_clusters": []}

    # Add method to manually trigger classification
    def classify_papers_for_hypothesis(self):
        """Manually trigger paper classification for the current hypothesis."""
        hypothesis_index = self.hypothesis_combo.currentIndex()
        if hypothesis_index <= 0:
            QMessageBox.warning(self, "No Hypothesis Selected", 
                               "Please select a hypothesis to classify papers.")
            return
        
        hypothesis_id = self.hypothesis_combo.itemData(hypothesis_index)
        active_study = self.studies_manager.get_active_study()
        
        # Find the hypothesis
        selected_hypothesis = None
        for hypothesis in active_study.hypotheses:
            if hypothesis.get('id') == hypothesis_id:
                selected_hypothesis = hypothesis
                break
        
        if not selected_hypothesis:
            return
        
        # Run classification
        self.classify_papers_by_relevance(selected_hypothesis)
    
    def highlight_relevant_papers(self, terms):
        """Highlight papers in the table that contain any of the given terms."""
        if not terms:
            return
        
        # Reset all paper highlighting
        for row in range(self.papers_table.rowCount()):
            for col in range(self.papers_table.columnCount()):
                item = self.papers_table.item(row, col)
                if item:
                    item.setBackground(QColor(255, 255, 255))
        
        # Check each paper for term matches
        for row in range(self.papers_table.rowCount()):
            if row >= len(self.papers):
                continue
            
            paper = self.papers[row]
            paper_title = paper.get('title', '').lower()
            paper_abstract = paper.get('abstract', '').lower()
            
            matched = False
            for term in terms:
                term = term.lower()
                if term in paper_title or term in paper_abstract:
                    matched = True
                    break
                
            if matched:
                for col in range(self.papers_table.columnCount() - 1):  # Skip action column
                    item = self.papers_table.item(row, col)
                    if item:
                        item.setBackground(QColor(230, 230, 255))  # Light blue

    @asyncSlot()
    async def _generate_hypothesis_terms_async(self, selected_hypothesis):
        """Generate search terms for a hypothesis using LLM."""
        try:
            self.status_label.setText("Generating hypothesis-specific search terms...")
            QApplication.processEvents()  # Force UI update
            
            # Call the hypothesis term generation
            result = await self.generate_hypothesis_terms(selected_hypothesis)
            
            # Use the generated terms to classify papers by relevance
            self.classify_papers_by_relevance(selected_hypothesis)
            
            self.status_label.setText("Ready")
        except Exception as e:
            logging.error(f"Error generating hypothesis terms: {str(e)}")
            self.status_label.setText("Error generating terms")

    def display_papers(self, papers):
        """Display papers in the papers table."""
        if not papers:
            self.papers_table.clearContents()
            self.papers_table.setRowCount(0)
            return
            
        # Update the papers table
        self.papers = papers
        self.update_papers_table()

    def save_claims(self):
        """Save claims to the active study."""
        if not self.studies_manager:
            QMessageBox.warning(self, "Error", "No studies manager available.")
            return
            
        try:
            active_study = self.studies_manager.get_active_study()
            if not active_study:
                QMessageBox.warning(self, "Error", "No active study.")
                return
                
            # Convert claims to dict for serialization
            claims_data = [claim.to_dict() for claim in self.claims]
            
            # Create a name for the dataset
            dataset_name = f"evidence_claims_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create a DataFrame
            claims_df = pd.DataFrame({
                'claims_data': [claims_data],
                'timestamp': [datetime.now().isoformat()]
            })
            
            # Save to studies manager
            self.studies_manager.add_dataset_to_active_study(
                dataset_name,
                claims_df,
                metadata={'type': 'evidence_claims'}
            )
            
            QMessageBox.information(self, "Success", f"Saved {len(self.claims)} claims to study.")
            
            # Emit signal that claims were saved
            self.claimsSaved.emit(self.claims)
            
        except Exception as e:
            logging.error(f"Error saving claims: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save claims: {str(e)}")

    def load_claims(self):
        """Load claims from the active study."""
        if not self.studies_manager:
            QMessageBox.warning(self, "Error", "No studies manager available.")
            return
            
        try:
            active_study = self.studies_manager.get_active_study()
            if not active_study:
                QMessageBox.warning(self, "Error", "No active study.")
                return
                
            # Get available datasets
            datasets = self.studies_manager.get_datasets_for_study(active_study)
            
            # Filter to find evidence claims datasets
            evidence_datasets = []
            for name in datasets:
                df = self.studies_manager.get_dataset(active_study, name)
                if df is not None and not df.empty:
                    metadata = self.studies_manager.get_dataset_metadata(active_study, name)
                    if metadata and metadata.get('type') == 'evidence_claims':
                        evidence_datasets.append(name)
            
            if not evidence_datasets:
                QMessageBox.information(self, "No Data", "No saved claims found.")
                return
                
            # Create a dialog to select which dataset to load
            dialog = QDialog(self)
            dialog.setWindowTitle("Load Claims")
            layout = QVBoxLayout(dialog)
            
            # Add explanatory label
            label = QLabel("Select which saved claims to load:")
            layout.addWidget(label)
            
            # Add list widget
            list_widget = QListWidget()
            for name in evidence_datasets:
                list_widget.addItem(name)
            layout.addWidget(list_widget)
            
            # Add buttons
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | 
                QDialogButtonBox.StandardButton.Cancel
            )
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            
            # Show dialog
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
                
            # Get selected dataset
            selected = list_widget.currentItem()
            if not selected:
                return
                
            dataset_name = selected.text()
            
            # Load dataset
            df = self.studies_manager.get_dataset(active_study, dataset_name)
            if df is None or df.empty:
                QMessageBox.warning(self, "Error", "Selected dataset is empty.")
                return
                
            # Extract claims data
            claims_data = df['claims_data'].iloc[0]
            
            # Convert to Claim objects
            loaded_claims = []
            for claim_dict in claims_data:
                loaded_claims.append(Claim.from_dict(claim_dict))
                
            # Ask if user wants to replace or merge
            reply = QMessageBox.question(
                self,
                "Load Claims",
                f"Found {len(loaded_claims)} claims. Do you want to replace current claims or merge with existing?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes
            )
            
            if reply == QMessageBox.StandardButton.Cancel:
                return
            elif reply == QMessageBox.StandardButton.Yes:
                # Replace
                self.claims = loaded_claims
            else:
                # Merge (avoiding duplicates by ID)
                existing_ids = [claim.id for claim in self.claims]
                for claim in loaded_claims:
                    if claim.id not in existing_ids:
                        self.claims.append(claim)
                        
            # Update display
            self.update_claims_display()
            QMessageBox.information(self, "Success", f"Loaded {len(loaded_claims)} claims.")
            
        except Exception as e:
            logging.error(f"Error loading claims: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load claims: {str(e)}")

    def restore_original_paper_order(self):
        """Restore papers to their original order."""
        if hasattr(self, 'original_papers') and self.original_papers:
            self.papers = self.original_papers.copy()
            self.update_papers_table()
        else:
            logging.warning("No original paper order stored to restore")

    def update_hypothesis_with_evidence(self):
        """Update the current hypothesis with the collected evidence."""
        hypothesis_index = self.hypothesis_combo.currentIndex()
        if hypothesis_index <= 0 or not self.claims:
            QMessageBox.warning(self, "Warning", "No hypothesis selected or no claims available.")
            return
            
        hypothesis_id = self.hypothesis_combo.itemData(hypothesis_index)
        if not hypothesis_id:
            QMessageBox.warning(self, "Warning", "Invalid hypothesis selected.")
            return
            
        if not self.studies_manager:
            QMessageBox.warning(self, "Error", "Studies manager not available.")
            return
            
        try:
            active_study = self.studies_manager.get_active_study()
            if not active_study:
                QMessageBox.warning(self, "Error", "No active study.")
                return
                
            # Count evidence types
            supporting = 0
            refuting = 0
            neutral = 0
            
            # Collect all evidence from claims
            all_evidence = []
            for claim in self.claims:
                all_evidence.extend(claim.evidence)
            
            # Count by type
            for evidence in all_evidence:
                if evidence.evidence_type == "support":
                    supporting += 1
                elif evidence.evidence_type == "refute":
                    refuting += 1
                else:
                    neutral += 1
            
            # Only proceed if we have evidence
            if supporting + refuting + neutral == 0:
                QMessageBox.warning(self, "No Evidence", "No evidence available to update hypothesis.")
                return
                
            # Create a summary of the evidence
            summary = f"Based on literature evidence: {supporting} supporting, {refuting} refuting, {neutral} neutral"
            
            # Calculate confidence based on evidence
            total_directional = supporting + refuting
            if total_directional == 0:
                status = "inconclusive"
                conclusion = "Insufficient directional evidence"
            elif supporting > refuting * 2:  # More than twice as much supporting evidence
                status = "confirmed"
                conclusion = "Strongly supported by literature evidence"
            elif supporting > refuting:
                status = "confirmed"  
                conclusion = "Supported by literature evidence"
            elif refuting > supporting * 2:
                status = "rejected"
                conclusion = "Strongly refuted by literature evidence"
            elif refuting > supporting:
                status = "rejected"
                conclusion = "Refuted by literature evidence"
            else:
                status = "inconclusive"
                conclusion = "Contested in the literature"
            
            # Ask user for confirmation
            reply = QMessageBox.question(
                self,
                "Update Hypothesis Status",
                f"Evidence summary: {supporting} supporting, {refuting} refuting, {neutral} neutral.\n\n"
                f"Suggested status: {status.capitalize()}\n\n"
                f"Would you like to update the hypothesis status based on this evidence?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            
            if reply != QMessageBox.StandardButton.Yes:
                return
                
            # Update the hypothesis
            found_hypothesis = False
            for i, hyp in enumerate(active_study.hypotheses):
                if hyp.get('id') == hypothesis_id:
                    found_hypothesis = True
                    # Add literature evidence results
                    active_study.hypotheses[i]['literature_evidence'] = {
                        'supporting': supporting,
                        'refuting': refuting,
                        'neutral': neutral,
                        'status': status,
                        'conclusion': conclusion,
                        'updated_at': datetime.now().isoformat()
                    }
                    
                    # Optionally update the overall status
                    if 'status' in active_study.hypotheses[i] and active_study.hypotheses[i]['status'] == 'untested':
                        active_study.hypotheses[i]['status'] = status
                    
                    # Show success message
                    QMessageBox.information(
                        self,
                        "Hypothesis Updated",
                        f"The hypothesis has been updated with the literature evidence.\n\n"
                        f"New status: {status.capitalize()}"
                    )
                    
                    # Refresh hypothesis display if hypotheses manager exists
                    if self.hypotheses_manager:
                        self.hypotheses_manager.load_hypotheses()
                    
                    break
                
            if not found_hypothesis:
                QMessageBox.warning(self, "Error", "Could not find the selected hypothesis in the active study.")
                
        except Exception as e:
            logging.error(f"Error updating hypothesis: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to update hypothesis: {str(e)}")

    @asyncSlot()
    async def extract_evidence_for_hypothesis(self):
        """Extract evidence for the current hypothesis from all loaded papers."""
        try:
            hypothesis_index = self.hypothesis_combo.currentIndex()
            if hypothesis_index <= 0:
                QMessageBox.warning(self, "No Hypothesis Selected", 
                                "Please select a hypothesis before extracting evidence.")
                return
                
            if not self.papers:
                QMessageBox.warning(self, "No Papers", "No papers available for evidence extraction.")
                return
            
            # Ask for confirmation
            num_papers = len(self.papers)
            reply = QMessageBox.question(
                self,
                "Extract Evidence",
                f"Do you want to analyze {num_papers} papers for evidence related to the selected hypothesis?\n"
                f"This may take some time.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.No:
                return
            
            if not self.studies_manager:
                QMessageBox.warning(self, "Error", "Studies manager not available.")
                return
                
            hypothesis_id = self.hypothesis_combo.itemData(hypothesis_index)
            active_study = self.studies_manager.get_active_study()
            if not active_study:
                QMessageBox.warning(self, "Error", "No active study available.")
                return
            
            # Find the hypothesis
            selected_hypothesis = None
            for hypothesis in active_study.hypotheses:
                if hypothesis.get('id') == hypothesis_id:
                    selected_hypothesis = hypothesis
                    break
                    
            if not selected_hypothesis:
                QMessageBox.warning(self, "Error", "Could not find the selected hypothesis.")
                return
            
            # Get hypothesis details
            null_hypothesis = selected_hypothesis.get('null_hypothesis', '')
            alt_hypothesis = selected_hypothesis.get('alternative_hypothesis', '')
            outcome_vars = selected_hypothesis.get('outcome_variables', '')
            predictor_vars = selected_hypothesis.get('predictor_variables', '')
            
            # Update status
            self.status_label.setText("Extracting evidence from papers...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            QApplication.processEvents()  # Force UI update
            
            # Process papers one by one
            new_claims = []
            for i, paper in enumerate(self.papers):
                # Update progress
                progress = int((i / num_papers) * 100)
                self.progress_bar.setValue(progress)
                self.status_label.setText(f"Analyzing paper {i+1} of {num_papers}...")
                QApplication.processEvents()  # Force UI update
                
                # Skip papers without abstracts
                if not paper.get('abstract'):
                    continue
                
                # Get paper information
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                full_text = paper.get('full_text', '')
                doi = paper.get('doi', '')
                
                # Extract evidence
                evidence_items = await self._extract_evidence_with_ai(
                    title, 
                    abstract, 
                    full_text,
                    null_hypothesis,
                    alt_hypothesis,
                    outcome_vars,
                    predictor_vars
                )
                
                # Create claim and evidence for each finding
                if evidence_items:
                    claim = Claim(
                        text=f"Evidence regarding: {alt_hypothesis}",
                        source_doi=doi,
                        source_title=title
                    )
                    
                    # Add each evidence item to the claim
                    for evidence in evidence_items:
                        evidence_item = EvidenceItem(
                            text=evidence['text'],
                            paper_doi=doi,
                            paper_title=title,
                            evidence_type=evidence['type'],
                            confidence=evidence['confidence'],
                            context=evidence.get('context', '')
                        )
                        
                        # Add relevant tags
                        for tag in evidence.get('tags', []):
                            evidence_item.tags.append(tag)
                        
                        claim.add_evidence(evidence_item)
                    
                    new_claims.append(claim)
            
            # Add the claims to our list
            if new_claims:
                self.claims.extend(new_claims)
                self.update_claims_display()
                
                # Count evidence
                total_evidence = sum(len(claim.evidence) for claim in new_claims)
                
                # Show success message
                QMessageBox.information(
                    self, 
                    "Evidence Extraction Complete", 
                    f"Successfully extracted {total_evidence} pieces of evidence " +
                    f"from {len(new_claims)} papers related to the hypothesis."
                )
                
                # Ask if user wants to update hypothesis with new evidence
                update_reply = QMessageBox.question(
                    self,
                    "Update Hypothesis",
                    "Would you like to update the hypothesis status based on the collected evidence?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                
                if update_reply == QMessageBox.StandardButton.Yes:
                    self.update_hypothesis_with_evidence()
            else:
                QMessageBox.information(
                    self, 
                    "No Evidence Found", 
                    "No relevant evidence was found in the papers for the selected hypothesis."
                )
            
            # Update status
            self.status_label.setText("Finished extracting evidence")
            self.progress_bar.setValue(100)
            
        except Exception as e:
            logging.error(f"Error extracting evidence: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to extract evidence: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)

    @asyncSlot()
    async def extract_claims_from_paper(self, paper=None):
        """Extract evidence for the current hypothesis from a specific paper."""
        try:
            if not paper and len(self.papers) > 0:
                # Get the currently selected paper from the table
                selected_rows = self.papers_table.selectedIndexes()
                if not selected_rows:
                    QMessageBox.warning(self, "No Selection", "Please select a paper to analyze.")
                    return
                    
                row = selected_rows[0].row()
                if row < len(self.papers):
                    paper = self.papers[row]
                
            if not paper:
                QMessageBox.warning(self, "No Paper", "No paper selected for analysis.")
                return
            
            # Get the currently selected hypothesis
            hypothesis_index = self.hypothesis_combo.currentIndex()
            if hypothesis_index <= 0:
                QMessageBox.warning(self, "No Hypothesis Selected", 
                                "Please select a hypothesis before extracting evidence.")
                return
                
            if not self.studies_manager:
                QMessageBox.warning(self, "Error", "Studies manager not available.")
                return
                
            hypothesis_id = self.hypothesis_combo.itemData(hypothesis_index)
            active_study = self.studies_manager.get_active_study()
            if not active_study:
                QMessageBox.warning(self, "Error", "No active study available.")
                return
            
            # Find the hypothesis
            selected_hypothesis = None
            for hypothesis in active_study.hypotheses:
                if hypothesis.get('id') == hypothesis_id:
                    selected_hypothesis = hypothesis
                    break
                    
            if not selected_hypothesis:
                QMessageBox.warning(self, "Error", "Could not find the selected hypothesis.")
                return
                
            # Update status
            self.status_label.setText(f"Extracting evidence from '{paper.get('title', 'paper')}'...")
            self.progress_bar.setValue(10)
            self.progress_bar.setVisible(True)
            QApplication.processEvents()  # Force UI update
            
            # Initialize QuoteExtractor if needed
            if not self.quote_extractor:
                self.quote_extractor = QuoteExtractor(logging.getLogger(__name__))
            
            # Get paper information
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            full_text = paper.get('full_text', '')
            doi = paper.get('doi', '')
            
            # Prepare hypothesis context
            null_hypothesis = selected_hypothesis.get('null_hypothesis', '')
            alt_hypothesis = selected_hypothesis.get('alternative_hypothesis', '')
            outcome_vars = selected_hypothesis.get('outcome_variables', '')
            predictor_vars = selected_hypothesis.get('predictor_variables', '')
            
            # Get hypothesis terms if available
            hypothesis_terms = selected_hypothesis.get('generated_terms')
            
            # Extract evidence specific to this hypothesis
            evidence_items = await self._extract_evidence_with_ai(
                title, 
                abstract, 
                full_text,
                null_hypothesis,
                alt_hypothesis,
                outcome_vars,
                predictor_vars,
                hypothesis_terms
            )
            
            self.progress_bar.setValue(70)
            
            # Create evidence objects for this claim
            new_claims = []
            for evidence in evidence_items:
                claim = Claim(
                    text=f"Evidence regarding: {alt_hypothesis}",
                    source_doi=doi,
                    source_title=title
                )
                
                # Add evidence to the claim
                evidence_item = EvidenceItem(
                    text=evidence['text'],
                    paper_doi=doi,
                    paper_title=title,
                    evidence_type=evidence['type'],
                    confidence=evidence['confidence'],
                    context=evidence.get('context', '')
                )
                
                # Add relevant tags
                for tag in evidence.get('tags', []):
                    evidence_item.tags.append(tag)
                
                # Add related terms as tags with a prefix
                for term in evidence.get('related_terms', []):
                    evidence_item.tags.append(f"term:{term}")
                
                claim.add_evidence(evidence_item)
                new_claims.append(claim)
            
            # Add the claims to our list
            if new_claims:
                self.claims.extend(new_claims)
                self.update_claims_display()
                
                # Show success message
                count = len(new_claims)
                QMessageBox.information(
                    self, 
                    "Evidence Extracted", 
                    f"Successfully extracted {count} piece{'' if count == 1 else 's'} of evidence " +
                    f"from {title} related to the hypothesis."
                )
            else:
                QMessageBox.information(
                    self, 
                    "No Evidence Found", 
                    f"No relevant evidence was found in {title} for the selected hypothesis."
                )
            
            # Update status
            self.status_label.setText(f"Finished extracting evidence from '{title}'")
            self.progress_bar.setValue(100)
            
        except Exception as e:
            logging.error(f"Error extracting evidence: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to extract evidence: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)