from typing import Dict
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit, 
    QComboBox, QFileDialog, QMessageBox, QInputDialog, QListWidget, 
    QListWidgetItem, QStackedWidget, QSplitter, QApplication, QTreeWidget, 
    QTreeWidgetItem, QDialog, QGridLayout, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, QDateTime, QTimer
from PyQt6.QtGui import QPainter
import sys
import logging
import pyqtgraph as pg

from protocols.builder import ProtocolBuilder
from protocols.record_keeping import ProtocolRecordKeeper
from protocols.template_helpers import ProtocolTemplateManager
from qasync import asyncSlot
import asyncio
from helpers.load_icon import load_bootstrap_icon

logging.basicConfig(level=logging.INFO)


# The primary endpoint is the change from baseline in [Primary Outcome Measure] at [Time Point]. Secondary endpoints include changes in [Secondary Outcome Measure 1], [Secondary Outcome Measure 2], and safety outcomes. Exploratory endpoints include [Exploratory Endpoint 1] and [Exploratory Endpoint 2].

class TrackingTextEdit(QTextEdit):
    """QTextEdit subclass that tracks paste events and text changes"""
    def __init__(self, parent=None, track_callback=None):
        super().__init__(parent)
        self.track_callback = track_callback
        self.paste_range = None
        self.paste_contributor = None
        self.is_pasting = False
        self.paste_buffer = []
        
        self.paste_timer = QTimer()
        self.paste_timer.setSingleShot(True)
        self.paste_timer.timeout.connect(self.process_paste_buffer)

    def pasteEvent(self, event):
        cursor = self.textCursor()
        start_pos = cursor.position()
        selection_start = cursor.selectionStart()
        selection_end = cursor.selectionEnd()
        has_selection = cursor.hasSelection()
        
        old_text = self.toPlainText()
        self.is_pasting = True
        super().pasteEvent(event)
        new_text = self.toPlainText()
        
        if has_selection:
            paste_length = len(new_text) - len(old_text) + (selection_end - selection_start)
            paste_start = selection_start
        else:
            paste_length = len(new_text) - len(old_text)
            paste_start = start_pos
            
        parent_widget = self.parent()
        if paste_length > 0 and hasattr(parent_widget, "current_contributor_type"):
            self.paste_buffer.append({
                'range': (paste_start, paste_start + paste_length),
                'contributor': (parent_widget.current_contributor_type,
                              parent_widget.current_contributor_id),
                'text': new_text[paste_start:paste_start + paste_length]
            })
            
            self.paste_timer.stop()
            self.paste_timer.start(100)
        
        self.is_pasting = False

    def process_paste_buffer(self):
        if not self.paste_buffer:
            return
            
        self.paste_buffer.sort(key=lambda x: x['range'][0])
        merged_pastes = []
        current = self.paste_buffer[0]
        
        for next_paste in self.paste_buffer[1:]:
            if current['range'][1] >= next_paste['range'][0]:
                if current['contributor'] == next_paste['contributor']:
                    current['range'] = (
                        current['range'][0],
                        max(current['range'][1], next_paste['range'][1])
                    )
                    current['text'] = self.toPlainText()[current['range'][0]:current['range'][1]]
                else:
                    merged_pastes.append(current)
                    current = next_paste
            else:
                merged_pastes.append(current)
                current = next_paste
                
        merged_pastes.append(current)
        
        for paste_op in merged_pastes:
            self.paste_range = paste_op['range']
            self.paste_contributor = paste_op['contributor']
            if self.track_callback:
                self.track_callback(paste_event=True)
        
        self.paste_buffer.clear()
        self.paste_range = None
        self.paste_contributor = None

class StudyTypeDialog(QDialog):
    """Dialog for selecting study type and initial sections"""
    def __init__(self, template_manager: ProtocolTemplateManager, parent=None):
        super().__init__(parent)
        self.template_manager = template_manager
        self.selected_type = None
        
        self.setWindowTitle("Create New Protocol")
        self.setMinimumWidth(1000)
        self.setMinimumHeight(700)
        
        layout = QVBoxLayout(self)
        
        # Split view for study type and sections
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - Study type selection
        type_group = QFrame()
        type_group.setFrameStyle(QFrame.Shape.StyledPanel)
        type_layout = QVBoxLayout(type_group)
        
        type_label = QLabel("Study Types:")
        type_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        type_layout.addWidget(type_label)
        
        self.type_list = QListWidget()
        for study_type in self.template_manager.get_study_types():
            item = QListWidgetItem(study_type['name'])
            item.setToolTip(study_type['description'])
            item.setData(Qt.ItemDataRole.UserRole, study_type['id'])
            item.setIcon(load_bootstrap_icon("file-earmark-text"))
            self.type_list.addItem(item)
        self.type_list.itemClicked.connect(self.update_sections)
        type_layout.addWidget(self.type_list)
        
        # Add description area
        desc_label = QLabel("Description:")
        desc_label.setStyleSheet("font-weight: bold;")
        type_layout.addWidget(desc_label)
        
        self.desc_text = QTextEdit()
        self.desc_text.setReadOnly(True)
        self.desc_text.setMaximumHeight(100)
        type_layout.addWidget(self.desc_text)
        
        splitter.addWidget(type_group)
        
        # Right side - Section tree
        section_group = QFrame()
        section_group.setFrameStyle(QFrame.Shape.StyledPanel)
        section_layout = QVBoxLayout(section_group)
        
        section_header = QHBoxLayout()
        section_label = QLabel("Protocol Sections:")
        section_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        section_header.addWidget(section_label)
        
        # Add/Remove section buttons
        button_layout = QHBoxLayout()
        self.add_section_btn = QPushButton("Add Section")
        self.add_section_btn.setIcon(load_bootstrap_icon("plus-circle"))
        self.add_section_btn.clicked.connect(self.add_section)
        self.add_section_btn.setEnabled(False)
        button_layout.addWidget(self.add_section_btn)
        
        self.remove_section_btn = QPushButton("Remove Section")
        self.remove_section_btn.setIcon(load_bootstrap_icon("trash"))
        self.remove_section_btn.clicked.connect(self.remove_section)
        self.remove_section_btn.setEnabled(False)
        button_layout.addWidget(self.remove_section_btn)
        section_header.addLayout(button_layout)
        
        section_layout.addLayout(section_header)
        
        # Section tree
        self.section_tree = QTreeWidget()
        self.section_tree.setHeaderLabels(["Sections", "Status"])
        self.section_tree.setColumnWidth(0, 300)
        self.section_tree.itemClicked.connect(self.section_selected)
        self.section_tree.itemChanged.connect(self.section_changed)
        section_layout.addWidget(self.section_tree)
        
        # Section preview
        preview_label = QLabel("Section Details:")
        preview_label.setStyleSheet("font-weight: bold;")
        section_layout.addWidget(preview_label)
        
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(150)
        section_layout.addWidget(self.preview_text)
        
        splitter.addWidget(section_group)
        splitter.setSizes([300, 700])
        
        layout.addWidget(splitter)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.create_button = QPushButton("Create Protocol")
        self.create_button.setIcon(load_bootstrap_icon("check-circle"))
        self.create_button.clicked.connect(self.accept)
        self.create_button.setEnabled(False)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.setIcon(load_bootstrap_icon("x-circle"))
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.create_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

    def update_sections(self, item):
        """Update available sections based on selected study type"""
        self.selected_type = item.data(Qt.ItemDataRole.UserRole)
        study_type = next(st for st in self.template_manager.get_study_types() 
                         if st['id'] == self.selected_type)
        
        # Update description
        self.desc_text.setPlainText(study_type['description'])
        
        # Clear existing tree
        self.section_tree.clear()
        
        # Create root items for required and optional sections
        required_root = QTreeWidgetItem(self.section_tree, ["Required Sections"])
        required_root.setFlags(required_root.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        required_root.setIcon(0, load_bootstrap_icon("lock-fill"))
        
        optional_root = QTreeWidgetItem(self.section_tree, ["Optional Sections"])
        optional_root.setFlags(optional_root.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        optional_root.setIcon(0, load_bootstrap_icon("list-check"))
        
        # Add required sections
        for section_id in self.template_manager.get_required_sections(self.selected_type):
            template = self.template_manager.get_section_template(section_id)
            section_item = QTreeWidgetItem(required_root, [template['title'], "Required"])
            section_item.setData(0, Qt.ItemDataRole.UserRole, section_id)
            section_item.setFlags(section_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            section_item.setCheckState(0, Qt.CheckState.Checked)
            section_item.setIcon(0, load_bootstrap_icon("file-earmark"))
            
            # Add subsections
            for subsection in template['subsections']:
                subsection_item = QTreeWidgetItem(section_item, [subsection.replace('_', ' ').title()])
                subsection_item.setFlags(subsection_item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                subsection_item.setIcon(0, load_bootstrap_icon("card-text"))
        
        # Add optional sections
        for section_id in self.template_manager.get_optional_sections(self.selected_type):
            template = self.template_manager.get_section_template(section_id)
            section_item = QTreeWidgetItem(optional_root, [template['title'], "Optional"])
            section_item.setData(0, Qt.ItemDataRole.UserRole, section_id)
            section_item.setFlags(section_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            section_item.setCheckState(0, Qt.CheckState.Unchecked)
            section_item.setIcon(0, load_bootstrap_icon("file-earmark-text"))
            
            # Add subsections
            for subsection in template['subsections']:
                subsection_item = QTreeWidgetItem(section_item, [subsection.replace('_', ' ').title()])
                subsection_item.setFlags(subsection_item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                subsection_item.setIcon(0, load_bootstrap_icon("card-text"))
        
        self.section_tree.expandAll()
        self.add_section_btn.setEnabled(True)
        self.create_button.setEnabled(True)

    def section_selected(self, item, column):
        """Show details for the selected section"""
        section_id = item.data(0, Qt.ItemDataRole.UserRole)
        if not section_id:  # Root or subsection item
            self.preview_text.clear()
            self.remove_section_btn.setEnabled(False)
            return
            
        template = self.template_manager.get_section_template(section_id)
        
        details = [
            f"Title: {template['title']}",
            "\nSubsections:",
            *[f"• {sub}" for sub in template['subsections']],
            "\nGuidance:",
            *[f"• {prompt}" for prompt in template['prompts']]
        ]
        
        self.preview_text.setPlainText("\n".join(details))
        self.remove_section_btn.setEnabled(item.parent().text(0) == "Optional Sections")

    def section_changed(self, item, column):
        """Handle section checkbox changes"""
        if not item.data(0, Qt.ItemDataRole.UserRole):  # Root or subsection item
            return
            
        # Update child items
        for i in range(item.childCount()):
            child = item.child(i)
            child.setDisabled(item.checkState(0) == Qt.CheckState.Unchecked)

    def add_section(self):
        """Add a custom section"""
        title, ok = QInputDialog.getText(self, "Add Section", 
                                       "Enter section title:")
        if ok and title:
            # Create new section under optional
            optional_root = self.section_tree.findItems("Optional Sections", 
                                                      Qt.MatchFlag.MatchExactly)[0]
            section_item = QTreeWidgetItem(optional_root, [title, "Optional"])
            section_item.setFlags(section_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            section_item.setCheckState(0, Qt.CheckState.Unchecked)
            section_item.setIcon(0, load_bootstrap_icon("file-earmark-plus"))
            
            # Add default subsections
            default_subsections = ["Overview", "Methods", "Results", "Discussion"]
            for subsection in default_subsections:
                subsection_item = QTreeWidgetItem(section_item, [subsection])
                subsection_item.setFlags(subsection_item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                subsection_item.setIcon(0, load_bootstrap_icon("card-text"))

    def remove_section(self):
        """Remove selected optional section"""
        item = self.section_tree.currentItem()
        if item and item.parent() and item.parent().text(0) == "Optional Sections":
            item.parent().removeChild(item)

    def get_selections(self):
        """Get selected study type and sections"""
        selected_sections = set()
        
        # Helper function to process tree items
        def process_root_item(root_item):
            for i in range(root_item.childCount()):
                section_item = root_item.child(i)
                section_id = section_item.data(0, Qt.ItemDataRole.UserRole)
                
                # Add all sections
                if section_id:  # Standard section
                    selected_sections.add(section_id)
                else:  # Custom section
                    selected_sections.add(section_item.text(0))
        
        # Process required and optional sections
        required_root = self.section_tree.findItems("Required Sections", 
                                                  Qt.MatchFlag.MatchExactly)[0]
        optional_root = self.section_tree.findItems("Optional Sections", 
                                                  Qt.MatchFlag.MatchExactly)[0]
        
        process_root_item(required_root)
        process_root_item(optional_root)
        
        return self.selected_type, selected_sections

class ProtocolTab(QWidget):
    """Individual protocol section with version tracking"""
    def __init__(self, section_id: str, template_manager: ProtocolTemplateManager, title="New Section", initial_content: Dict[str, str] = None):
        super().__init__()
        self.section_id = section_id
        self.template_manager = template_manager
        self.title = title
        self.edit_count = 0
        self.version_number = 1
        self.versions = {1: initial_content or {}}
        self.char_contributors = {}  # Dict mapping subsection to its contributors
        self.current_contributor_type = 'human'
        self.current_contributor_id = 'user'
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(10)

        # Header with title
        header_layout = QHBoxLayout()
        
        title_layout = QVBoxLayout()
        self.title_edit = QTextEdit()
        self.title_edit.setPlainText(self.title)
        self.title_edit.setStyleSheet("font-size: 14px; border-radius: 4px; padding: 4px;")
        self.title_edit.setFixedHeight(50)
        self.title_edit.textChanged.connect(self.update_title)
        title_layout.addWidget(self.title_edit)
        
        header_layout.addLayout(title_layout)
        self.layout.addLayout(header_layout)

        # Content organized by subsections
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        subsections = self.template_manager.get_section_subsections(section_id)
        self.subsection_editors = {}
        
        # Add icons to subsection labels
        for subsection in subsections:
            group = QFrame()
            group.setFrameStyle(QFrame.Shape.StyledPanel)
            group_layout = QVBoxLayout(group)
            
            label_layout = QHBoxLayout()
            subsection_icon = QLabel()
            subsection_icon.setPixmap(load_bootstrap_icon("card-text").pixmap(16, 16))
            label_layout.addWidget(subsection_icon)
            
            label = QLabel(subsection.replace('_', ' ').title())
            label.setStyleSheet("font-weight: bold;")
            label_layout.addWidget(label)
            label_layout.addStretch()
            
            group_layout.addLayout(label_layout)
            
            editor = TrackingTextEdit(track_callback=lambda s=subsection: self.track_changes(subsection))
            editor.setMinimumHeight(100)
            self.subsection_editors[subsection] = editor
            group_layout.addWidget(editor)
            
            content_layout.addWidget(group)
            
            # Initialize char_contributors for this subsection
            self.char_contributors[subsection] = []
            if initial_content and subsection in initial_content:
                editor.setPlainText(initial_content[subsection])
                self.char_contributors[subsection] = [(i, 'human', 'user') 
                                                    for i in range(len(initial_content[subsection]))]
        
        scroll = QScrollArea()
        scroll.setWidget(content_widget)
        scroll.setWidgetResizable(True)
        self.layout.addWidget(scroll)

        # Version control and contribution tracking
        footer_layout = QHBoxLayout()
        
        # Version controls
        version_layout = QVBoxLayout()
        self.version_tree = QTreeWidget()
        self.version_tree.setHeaderLabels(["Version", "Timestamp"])
        self.version_tree.setMaximumHeight(120)
        self.version_tree.itemClicked.connect(self.load_version)
        root = QTreeWidgetItem(["Version 1", "Base Version"])
        root.setIcon(0, load_bootstrap_icon("file-earmark"))
        root.setData(0, Qt.ItemDataRole.UserRole, 1)
        self.version_tree.addTopLevelItem(root)
        
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save Version")
        self.save_button.setIcon(load_bootstrap_icon("save"))
        self.save_button.clicked.connect(self.save_version)
        self.highlight_button = QPushButton("Show Contributors")
        self.highlight_button.setIcon(load_bootstrap_icon("people"))
        self.highlight_button.setCheckable(True)
        self.highlight_button.clicked.connect(self.toggle_highlights)
        
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.highlight_button)
        button_layout.addStretch()
        
        version_layout.addLayout(button_layout)
        version_layout.addWidget(self.version_tree)
        
        # Contribution visualization
        contrib_layout = QVBoxLayout()
        self.contrib_viz = pg.PlotWidget(enableMenu=False, title=None)
        self.contrib_viz.setFixedHeight(14)
        self.contrib_viz.hideAxis('left')
        self.contrib_viz.hideAxis('bottom')
        self.contrib_viz.setMouseEnabled(x=False, y=False)
        self.contrib_viz.getPlotItem().setContentsMargins(0, 0, 0, 0)
        self.contrib_viz.getPlotItem().setMenuEnabled(False)
        self.contrib_viz.getPlotItem().hideButtons()
        self.contrib_viz.plotItem.vb.setDefaultPadding(0)
        self.contrib_viz.plotItem.vb.setRange(xRange=(0, 100), yRange=(0, 0.1), padding=0)
        
        self.human_color = '#32CD32'
        self.ai_color = '#00FFFF'
        
        self.human_bar = pg.BarGraphItem(x=[0], height=[0], width=1.0, brush=self.human_color)
        self.ai_bar = pg.BarGraphItem(x=[0], height=[0], width=1.0, brush=self.ai_color)
        self.contrib_viz.addItem(self.human_bar)
        self.contrib_viz.addItem(self.ai_bar)
        
        contrib_layout.addWidget(self.contrib_viz)
        
        footer_layout.addLayout(version_layout)
        footer_layout.addLayout(contrib_layout)
        self.layout.addLayout(footer_layout)

    def get_content(self) -> Dict[str, str]:
        """Get content from all subsection editors"""
        return {
            subsection: editor.toPlainText()
            for subsection, editor in self.subsection_editors.items()
        }

    def set_content(self, content: Dict[str, str]):
        """Set content for all subsection editors"""
        for subsection, text in content.items():
            if subsection in self.subsection_editors:
                self.subsection_editors[subsection].setPlainText(text)

    def update_title(self):
        self.title = self.title_edit.toPlainText()
        if hasattr(self.parent(), 'section_table'):
            for i in range(self.parent().section_table.topLevelItemCount()):
                item = self.parent().section_table.topLevelItem(i)
                if item.text(0) == self.title:
                    item.setText(0, self.title)
                    break

    def track_changes(self, subsection: str, paste_event=False):
        """Track changes for a specific subsection"""
        editor = self.subsection_editors[subsection]
        current_text = editor.toPlainText()
        
        # Get previous text from char_contributors
        prev_chars = [c[2] for c in self.char_contributors[subsection]]
        prev_text = ''.join(prev_chars) if prev_chars else ""
        
        if current_text == prev_text:
            return
            
        current_contributor = (self.current_contributor_type, self.current_contributor_id)
        paste_range = editor.paste_range if paste_event else None
        
        self.char_contributors[subsection] = ProtocolRecordKeeper.track_changes(
            prev_text, current_text, self.char_contributors[subsection],
            current_contributor, paste_range
        )
        
        self.update_contributor_badges()

    def update_contributor_badges(self):
        """Update contribution visualization based on all subsections"""
        total_chars = 0
        human_chars = 0
        ai_chars = 0
        
        for contributors in self.char_contributors.values():
            for _, contrib_type, _ in contributors:
                total_chars += 1
                if contrib_type == 'human':
                    human_chars += 1
                else:
                    ai_chars += 1
        
        if total_chars == 0:
            self.human_bar.setOpts(width=[0])
            self.ai_bar.setOpts(width=[0])
            return
        
        human_width = (human_chars / total_chars) * 100
        ai_width = (ai_chars / total_chars) * 100
        
        self.human_bar.setOpts(x=[human_width/2], width=[human_width], height=[0.1])
        self.ai_bar.setOpts(x=[human_width + ai_width/2], width=[ai_width], height=[0.1])

    def save_version(self):
        """Save current version of all subsections"""
        self.version_number += 1
        current_content = self.get_content()
        self.versions[self.version_number] = current_content
        
        version_entry = ProtocolRecordKeeper.create_version_entry(
            self.version_number,
            current_content
        )
        
        new_version = QTreeWidgetItem([
            f"Version {self.version_number}",
            version_entry["timestamp"]
        ])
        new_version.setIcon(0, load_bootstrap_icon("file-earmark-check"))
        new_version.setData(0, Qt.ItemDataRole.UserRole, self.version_number)
        self.version_tree.addTopLevelItem(new_version)
        self.edit_count += 1
        logging.info(f"Saved version {self.version_number} for {self.title}")

    def load_version(self, item, column):
        """Load a specific version"""
        version = item.data(0, Qt.ItemDataRole.UserRole)
        if version in self.versions:
            self.set_content(self.versions[version])
            logging.info(f"Loaded version {version} for {self.title}")

    def toggle_highlights(self):
        """Toggle contributor highlighting for all subsections"""
        if self.highlight_button.isChecked():
            for subsection, editor in self.subsection_editors.items():
                current_text = editor.toPlainText()
                editor.setVisible(False)
                
                if not hasattr(self, f'highlight_display_{subsection}'):
                    highlight_display = QTextEdit()
                    highlight_display.setReadOnly(True)
                    editor.parent().layout().insertWidget(
                        editor.parent().layout().indexOf(editor),
                        highlight_display
                    )
                    setattr(self, f'highlight_display_{subsection}', highlight_display)
                
                highlight_display = getattr(self, f'highlight_display_{subsection}')
                highlight_display.setVisible(True)
                highlight_display.clear()
                
                html_text = []
                contributors = self.char_contributors[subsection]
                for i, char in enumerate(current_text):
                    if i < len(contributors):
                        _, contrib_type, contrib_id = contributors[i]
                        if contrib_type == 'human':
                            html_text.append(f'<span style="background-color: rgba(0, 255, 0, 0.3)" title="{contrib_id}">{char}</span>')
                        elif contrib_type == 'ai':
                            html_text.append(f'<span style="background-color: rgba(0, 255, 255, 0.3)" title="{contrib_id}">{char}</span>')
                    else:
                        html_text.append(char)
                
                highlight_display.setHtml(''.join(html_text))
        else:
            for subsection, editor in self.subsection_editors.items():
                if hasattr(self, f'highlight_display_{subsection}'):
                    getattr(self, f'highlight_display_{subsection}').setVisible(False)
                editor.setVisible(True)

class ProtocolSection(QWidget):
    """Main protocol UI widget"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.template_manager = ProtocolTemplateManager()
        self.setWindowTitle("Protocol Editor")
        self.resize(900, 600)
        
        # Set default study type
        self.study_type = "case_control"
        self.sections = []
        
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(5)
        
        # Study type and sections
        study_group = QFrame()
        study_group.setFrameStyle(QFrame.Shape.StyledPanel)
        study_layout = QVBoxLayout(study_group)
        
        # Study controls
        study_header = QHBoxLayout()
        manage_protocol_button = QPushButton("Manage Protocol Sections")
        manage_protocol_button.setIcon(load_bootstrap_icon("gear"))
        manage_protocol_button.clicked.connect(self.manage_protocol)
        study_header.addWidget(manage_protocol_button)
        study_layout.addLayout(study_header)
        
        section_label = QLabel("Protocol Sections:")
        section_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        study_layout.addWidget(section_label)
        
        # Replace list with table
        self.section_table = QTreeWidget()
        self.section_table.setHeaderLabels(["Section", "Status", "Actions"])
        self.section_table.setColumnWidth(0, 150)
        self.section_table.setColumnWidth(1, 70)
        self.section_table.itemClicked.connect(self.change_section)
        study_layout.addWidget(self.section_table)
        
        left_layout.addWidget(study_group)
        
        # AI generation controls
        ai_group = QFrame()
        ai_group.setFrameStyle(QFrame.Shape.StyledPanel)
        ai_layout = QVBoxLayout(ai_group)
        
        ai_label = QLabel("AI Protocol Generation")
        ai_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        ai_layout.addWidget(ai_label)
        
        desc_label = QLabel("Study Description:")
        self.desc_edit = QTextEdit()
        self.desc_edit.setMaximumHeight(100)
        ai_layout.addWidget(desc_label)
        ai_layout.addWidget(self.desc_edit)
        
        lit_label = QLabel("Literature Review (optional):")
        self.lit_edit = QTextEdit()
        self.lit_edit.setMaximumHeight(100)
        ai_layout.addWidget(lit_label)
        ai_layout.addWidget(self.lit_edit)
        
        select_header = QHBoxLayout()
        select_label = QLabel("Select sections to generate:")
        select_header.addWidget(select_label)
        
        select_all_btn = QPushButton("Select All")
        select_all_btn.setIcon(load_bootstrap_icon("check-all"))
        select_all_btn.clicked.connect(self.select_all_sections)
        select_header.addWidget(select_all_btn)
        ai_layout.addLayout(select_header)
        
        self.section_selection = QListWidget()
        self.section_selection.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        ai_layout.addWidget(self.section_selection)
        
        # Populate section_selection with default sections
        self.populate_default_sections()
        
        self.generate_button = QPushButton("Generate Selected Sections")
        self.generate_button.setIcon(load_bootstrap_icon("stars"))
        self.generate_button.clicked.connect(self.generate_ai_sections)
        ai_layout.addWidget(self.generate_button)
        
        left_layout.addWidget(ai_group)
        splitter.addWidget(left_widget)

        self.stacked_widget = QStackedWidget()
        splitter.addWidget(self.stacked_widget)
        splitter.setSizes([250, 650])
        
        self.sections = []
        self.study_type = None

    def select_all_sections(self):
        """Select all sections in the generation list"""
        for i in range(self.section_selection.count()):
            self.section_selection.item(i).setSelected(True)

    def manage_protocol(self):
        """Open dialog to manage protocol sections"""
        dialog = StudyTypeDialog(self.template_manager, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            study_type, sections = dialog.get_selections()
            self.study_type = study_type
            
            # Clear existing sections
            self.sections.clear()
            self.section_table.clear()
            while self.stacked_widget.count():
                self.stacked_widget.removeWidget(self.stacked_widget.widget(0))
            
            # Create selected sections
            for section_id in sections:
                template = self.template_manager.get_section_template(section_id)
                self.create_section(section_id, template['title'])
            
            # Update AI generation section list
            self.section_selection.clear()
            for section_id in sections:
                template = self.template_manager.get_section_template(section_id)
                item = QListWidgetItem(template['title'])
                item.setData(Qt.ItemDataRole.UserRole, section_id)
                if section_id in self.template_manager.get_required_sections(study_type):
                    item.setText(f"{template['title']} (Required)")
                    item.setIcon(load_bootstrap_icon("file-earmark-lock"))
                else:
                    item.setIcon(load_bootstrap_icon("file-earmark-text"))
                self.section_selection.addItem(item)

    def create_section(self, section_id: str, content: Dict[str, str]):
        """Create a new protocol section"""
        template = self.template_manager.get_section_template(section_id.lower())
        new_section = ProtocolTab(section_id, self.template_manager, template['title'], content)
        self.sections.append(new_section)
        self.stacked_widget.addWidget(new_section)
        
        # Create table item
        item = QTreeWidgetItem()
        item.setText(0, template['title'])
        
        # Handle case when study_type is None
        if not hasattr(self, 'study_type') or self.study_type is None:
            # Set a default study type (case_control is common)
            self.study_type = "case_control"
            print(f"Setting default study type to: {self.study_type}")
            
        is_required = section_id in self.template_manager.get_required_sections(self.study_type)
        item.setText(1, "Required" if is_required else "Optional")
        item.setData(0, Qt.ItemDataRole.UserRole, section_id)
        
        # Add icons based on section status
        if is_required:
            item.setIcon(0, load_bootstrap_icon("file-earmark-lock"))
        else:
            item.setIcon(0, load_bootstrap_icon("file-earmark-text"))
        
        # Add remove button for optional sections
        if not is_required:
            remove_btn = QPushButton("Remove")
            remove_btn.setIcon(load_bootstrap_icon("trash"))
            remove_btn.clicked.connect(lambda: self.remove_section(item))
            self.section_table.setItemWidget(item, 2, remove_btn)
        
        self.section_table.addTopLevelItem(item)
        logging.info(f"Added protocol section: {template['title']}")

    def change_section(self, item, column=0):
        """Change to selected section"""
        index = self.section_table.indexOfTopLevelItem(item)
        if index >= 0:
            self.stacked_widget.setCurrentIndex(index)

    def remove_section(self, item):
        """Remove an optional section"""
        index = self.section_table.indexOfTopLevelItem(item)
        if index >= 0:
            self.section_table.takeTopLevelItem(index)
            widget = self.stacked_widget.widget(index)
            self.stacked_widget.removeWidget(widget)
            self.sections.pop(index)
            
            # Update AI generation section list
            section_id = item.data(0, Qt.ItemDataRole.UserRole)
            for i in range(self.section_selection.count()):
                list_item = self.section_selection.item(i)
                if list_item.data(Qt.ItemDataRole.UserRole) == section_id:
                    self.section_selection.takeItem(i)
                    break

    @asyncSlot()
    async def generate_ai_sections(self):
        """Generate selected protocol sections using AI"""
        study_description = self.desc_edit.toPlainText().strip()
        if not study_description:
            QMessageBox.warning(self, "Missing Input", 
                              "Please provide a study description")
            return
            
        selected_items = self.section_selection.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", 
                              "Please select sections to generate")
            return
            
        selected_sections = {item.data(Qt.ItemDataRole.UserRole) for item in selected_items}
        literature_review = self.lit_edit.toPlainText().strip() or None
        
        try:
            # Update button to show progress
            original_text = self.generate_button.text()
            original_icon = self.generate_button.icon()
            self.generate_button.setText("Generating...")
            self.generate_button.setIcon(load_bootstrap_icon("hourglass-split"))
            self.generate_button.setEnabled(False)
            
            # Generate sections
            sections = await ProtocolBuilder.generate_ai_protocol_sections(
                study_description, selected_sections, literature_review
            )
            
            if not sections:
                self.generate_button.setText(original_text)
                self.generate_button.setIcon(original_icon)
                self.generate_button.setEnabled(True)
                QMessageBox.warning(self, "Generation Failed", 
                                  "No valid sections were generated")
                return
            
            # Create or update sections
            for section_id, content in sections.items():
                # Check if section already exists
                existing_items = []
                for i in range(self.section_table.topLevelItemCount()):
                    item = self.section_table.topLevelItem(i)
                    if item.data(0, Qt.ItemDataRole.UserRole) == section_id:
                        existing_items.append(item)
                
                if existing_items:
                    # Update existing section
                    index = self.section_table.indexOfTopLevelItem(existing_items[0])
                    section = self.sections[index]
                    # Set AI as contributor before updating content
                    section.current_contributor_type = 'ai'
                    section.current_contributor_id = 'gemini'
                    # Update each subsection and track changes
                    for subsection, text in content.items():
                        if subsection in section.subsection_editors:
                            section.subsection_editors[subsection].setPlainText(text)
                            section.track_changes(subsection)
                else:
                    # Create new section
                    self.create_section(section_id, content)
            
            # Restore button state
            self.generate_button.setText(original_text)
            self.generate_button.setIcon(original_icon)
            self.generate_button.setEnabled(True)
            QMessageBox.information(self, "Success", 
                                  f"Generated {len(sections)} protocol sections")
            
        except Exception as e:
            self.generate_button.setText(original_text)
            self.generate_button.setIcon(original_icon)
            self.generate_button.setEnabled(True)
            QMessageBox.critical(self, "Generation Failed", str(e))

    def populate_default_sections(self):
        """Populate the section selection list with default sections for the current study type"""
        # Clear existing items
        self.section_selection.clear()
        
        # Get sections for the current study type
        if hasattr(self, 'study_type') and self.study_type:
            required_sections = self.template_manager.get_required_sections(self.study_type)
            optional_sections = self.template_manager.get_optional_sections(self.study_type)
            
            # Add required sections
            for section_id in required_sections:
                try:
                    template = self.template_manager.get_section_template(section_id)
                    item = QListWidgetItem(f"{template['title']} (Required)")
                    item.setData(Qt.ItemDataRole.UserRole, section_id)
                    item.setIcon(load_bootstrap_icon("file-earmark-lock"))
                    self.section_selection.addItem(item)
                except Exception as e:
                    print(f"Error adding required section {section_id}: {str(e)}")
            
            # Add optional sections
            for section_id in optional_sections:
                try:
                    template = self.template_manager.get_section_template(section_id)
                    item = QListWidgetItem(template['title'])
                    item.setData(Qt.ItemDataRole.UserRole, section_id)
                    item.setIcon(load_bootstrap_icon("file-earmark-text"))
                    self.section_selection.addItem(item)
                except Exception as e:
                    print(f"Error adding optional section {section_id}: {str(e)}")
        else:
            print("No study type set, cannot populate default sections")

# Uncomment to run standalone
if __name__ == "__main__":
    import sys
    import qasync
    import asyncio
    
    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    window = ProtocolSection()
    window.show()
    
    with loop:
        loop.run_forever()
