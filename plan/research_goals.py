import os
import math
import json
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set, Tuple, Union
import uuid
import asyncio
from datetime import datetime
import pandas as pd

from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsObject, QGraphicsItem, QGraphicsPathItem,
    QGraphicsTextItem, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QTextEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QGraphicsDropShadowEffect, QColorDialog, QMenu, QDialogButtonBox, QSplitter,
    QFrame, QListWidget, QListWidgetItem, QGroupBox, QGridLayout, QFileDialog, QMessageBox, QInputDialog, QRadioButton, QWidget, QTabWidget, QSlider, QGraphicsLineItem
)
from PyQt6.QtGui import (
    QPen, QBrush, QColor, QFont, QPainterPath, QPainter, QPainterPathStroker,
    QLinearGradient, QRadialGradient, QCursor, QIcon
)
from PyQt6.QtCore import (
    Qt, QPointF, QRectF, pyqtSignal, QPropertyAnimation, QEasingCurve, 
    QTimer, QByteArray, QJsonDocument, QLineF
)

from PyQt6.QtCore import Property

# First add these imports at the top with other imports
from helpers.load_icon import load_bootstrap_icon
from plan.plan_config import ConnectionType, Evidence, EvidenceSourceType, HypothesisConfig, HypothesisState, ObjectiveConfig, ObjectiveType
from plan.plan_dialogs import HypothesisEditorDialog, ObjectiveEditorDialog
from plan.plan_grid import GridPosition, ResearchGridScene, ResearchGridView
from plan.plan_nodes import BaseNode, EvidencePanel, HypothesisNode, NodeConnection, ObjectiveNode



# ======================
# Utility Functions
# ======================

def get_hypothesis_state(state_value):
    """
    Convert string state value to HypothesisState enum.
    Handles both string values and enum values for compatibility.
    """
    if isinstance(state_value, HypothesisState):
        return state_value
        
    # Handle string values from JSON or qt_sections
    if isinstance(state_value, str):
        # Map from qt_sections strings to our enum
        mapping = {
            "untested": HypothesisState.UNTESTED,
            "confirmed": HypothesisState.VALIDATED,
            "rejected": HypothesisState.REJECTED,
            "inconclusive": HypothesisState.INCONCLUSIVE,
            "proposed": HypothesisState.PROPOSED,
            "testing": HypothesisState.TESTING,
            "validated": HypothesisState.VALIDATED,
            "modified": HypothesisState.MODIFIED
        }
        return mapping.get(state_value.lower(), HypothesisState.PROPOSED)
    
    # Default if we can't determine the state
    return HypothesisState.PROPOSED


def resolve_hypothesis_status(statistical_evidence=None, literature_evidence=None, other_evidence=None):
    """
    Determine hypothesis status based on available evidence.
    Returns the most appropriate HypothesisState.
    """
    # If no evidence exists
    if not statistical_evidence and not literature_evidence and not other_evidence:
        return HypothesisState.UNTESTED
        
    # Start with inconclusive as default when evidence exists
    final_status = HypothesisState.INCONCLUSIVE
    
    # Statistical evidence takes precedence if it exists
    if statistical_evidence:
        p_value = statistical_evidence.get('p_value')
        alpha = statistical_evidence.get('alpha_level', 0.05)
        
        if p_value is not None:
            if p_value < alpha:
                # Strong statistical evidence supporting
                final_status = HypothesisState.VALIDATED
            else:
                # Statistical evidence against
                final_status = HypothesisState.REJECTED
    
    # Consider literature evidence if it exists and statistical is inconclusive
    if literature_evidence and final_status == HypothesisState.INCONCLUSIVE:
        lit_status = literature_evidence.get('status')
        if lit_status == 'validated':
            final_status = HypothesisState.VALIDATED
        elif lit_status == 'rejected':
            final_status = HypothesisState.REJECTED
    
    return final_status


# Node type icons - for visual distinction
NODE_TYPE_ICONS = {
    "objective": "bullseye",
    "hypothesis": "lightbulb"
}


class ResearchPlanningWidget(QSplitter):
    """Main widget for research planning with objectives and hypotheses"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOrientation(Qt.Orientation.Horizontal)
        self.current_theme = "light"
        self.studies_manager = None

        # Create grid scene and view
        self.grid_scene = ResearchGridScene()
        self.grid_view = ResearchGridView(self.grid_scene)
        
        # Create research manager
        from plan.research_goals import ResearchManager
        self.research_manager = ResearchManager(self.grid_scene)
        
        # Create default lanes
        self.grid_scene.create_type_lanes()
        
        # Connect signals
        self.grid_scene.nodeActivated.connect(self.on_node_activated)
        self.grid_scene.nodeSelected.connect(self.on_node_selected)
        
        # Create control panel
        self.control_panel = QFrame()
        self.control_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self.control_panel.setMinimumWidth(250)
        self.control_panel.setMaximumWidth(350)
        
        # Create evidence panel
        self.evidence_panel = EvidencePanel()
        self.evidence_panel.hide()
        self.evidence_panel.evidenceChanged.connect(self.on_evidence_changed)
        
        # Setup UI components
        self.setup_control_panel()
        
        # Create right-side layout with view and evidence panel
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add view to right panel
        right_layout.addWidget(self.grid_view, 1)
        
        # Add evidence panel to right panel (initially hidden)
        right_layout.addWidget(self.evidence_panel)
        
        # Add widgets to splitter
        self.addWidget(self.control_panel)
        self.addWidget(self.right_panel)
        
        # Set default sizes
        self.setSizes([300, 700])
        
        # Update control buttons initial state
        self.update_control_buttons()
    
    def setup_control_panel(self):
        """Setup control panel with buttons and options"""
        layout = QVBoxLayout(self.control_panel)
        
        # Title
        title_label = QLabel("Research Planning")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title_label)
        
        # Add Objective Group
        self.add_objective_group(layout)
        
        # Add Hypothesis Group
        self.add_hypothesis_group(layout)
        
        # Add View Options Group
        self.add_view_options_group(layout)
        
        # Spacer
        layout.addStretch()
        
        # Bottom buttons
        bottom_layout = QHBoxLayout()
        
        # Reset button
        self.reset_btn = QPushButton()
        self.reset_btn.setIcon(load_bootstrap_icon("file-earmark-plus"))
        self.reset_btn.setToolTip("Create a new research plan")
        self.reset_btn.setFixedSize(40, 40)
        self.reset_btn.clicked.connect(self.on_new_plan)
        bottom_layout.addWidget(self.reset_btn)
        
        # Sample workflow button
        self.sample_btn = QPushButton()
        self.sample_btn.setIcon(load_bootstrap_icon("bug"))
        self.sample_btn.setToolTip("Generate sample workflow for debugging")
        self.sample_btn.setFixedSize(40, 40)
        self.sample_btn.clicked.connect(self.on_generate_sample)
        bottom_layout.addWidget(self.sample_btn)
        
        # Export button
        self.export_btn = QPushButton()
        self.export_btn.setIcon(load_bootstrap_icon("download"))
        self.export_btn.setToolTip("Export research plan to JSON")
        self.export_btn.setFixedSize(40, 40)
        self.export_btn.clicked.connect(self.on_export_plan)
        bottom_layout.addWidget(self.export_btn)
        
        layout.addLayout(bottom_layout)
    
    def add_objective_group(self, layout):
        """Add objective controls group"""
        group = QGroupBox("Objectives")
        group_layout = QGridLayout(group)
        
        # Add Objective button
        self.add_obj_btn = QPushButton()
        self.add_obj_btn.setIcon(load_bootstrap_icon("bullseye"))
        self.add_obj_btn.setToolTip("Add new research objective")
        self.add_obj_btn.setFixedSize(40, 40)
        self.add_obj_btn.clicked.connect(self.on_add_objective)
        group_layout.addWidget(self.add_obj_btn, 0, 0)
        
        # Add Sub-Objective button
        self.add_subobj_btn = QPushButton()
        self.add_subobj_btn.setIcon(load_bootstrap_icon("diagram-3"))
        self.add_subobj_btn.setToolTip("Add sub-objective to selected objective")
        self.add_subobj_btn.setFixedSize(40, 40)
        self.add_subobj_btn.clicked.connect(self.on_add_sub_objective)
        self.add_subobj_btn.setEnabled(False)  # Initially disabled
        group_layout.addWidget(self.add_subobj_btn, 0, 1)
        
        # Edit Objective button
        self.edit_obj_btn = QPushButton()
        self.edit_obj_btn.setIcon(load_bootstrap_icon("pencil-square"))
        self.edit_obj_btn.setToolTip("Edit selected objective")
        self.edit_obj_btn.setFixedSize(40, 40)
        self.edit_obj_btn.clicked.connect(self.on_edit_objective)
        self.edit_obj_btn.setEnabled(False)  # Initially disabled
        group_layout.addWidget(self.edit_obj_btn, 1, 0)
        
        # Toggle Collapse button
        self.toggle_collapse_btn = QPushButton()
        self.toggle_collapse_btn.setIcon(load_bootstrap_icon("arrows-collapse"))
        self.toggle_collapse_btn.setToolTip("Collapse/expand selected objective")
        self.toggle_collapse_btn.setFixedSize(40, 40)
        self.toggle_collapse_btn.clicked.connect(self.on_toggle_collapse)
        self.toggle_collapse_btn.setEnabled(False)  # Initially disabled
        group_layout.addWidget(self.toggle_collapse_btn, 1, 1)
        
        # Objective type selector
        self.objective_type_combo = QComboBox()
        self.objective_type_combo.setToolTip("Select objective type")
        for obj_type in ObjectiveType:
            self.objective_type_combo.addItem(obj_type.value.replace("_", " ").title(), obj_type)
        group_layout.addWidget(self.objective_type_combo, 2, 0, 1, 2)
        
        layout.addWidget(group)
    
    def add_hypothesis_group(self, layout):
        """Add hypothesis controls group"""
        group = QGroupBox("Hypotheses")
        group_layout = QGridLayout(group)
        
        # Add Hypothesis button
        self.add_hyp_btn = QPushButton()
        self.add_hyp_btn.setIcon(load_bootstrap_icon("lightbulb"))
        self.add_hyp_btn.setToolTip("Add new hypothesis to selected objective")
        self.add_hyp_btn.setFixedSize(40, 40)
        self.add_hyp_btn.clicked.connect(self.on_add_hypothesis)
        self.add_hyp_btn.setEnabled(False)  # Initially disabled
        group_layout.addWidget(self.add_hyp_btn, 0, 0)
        
        # Auto-Generate Hypothesis button
        self.generate_hyp_btn = QPushButton()
        self.generate_hyp_btn.setIcon(load_bootstrap_icon("cpu"))
        self.generate_hyp_btn.setToolTip("Auto-generate hypotheses for selected objective")
        self.generate_hyp_btn.setFixedSize(40, 40)
        self.generate_hyp_btn.clicked.connect(self.on_generate_hypotheses)
        self.generate_hyp_btn.setEnabled(False)  # Initially disabled
        group_layout.addWidget(self.generate_hyp_btn, 0, 1)
        
        # Auto-Generate and Test Hypothesis button
        self.generate_test_hyp_btn = QPushButton()
        self.generate_test_hyp_btn.setIcon(load_bootstrap_icon("robot"))
        self.generate_test_hyp_btn.setToolTip("Auto-generate and test a hypothesis for selected objective")
        self.generate_test_hyp_btn.setFixedSize(40, 40)
        self.generate_test_hyp_btn.clicked.connect(self.on_auto_test_hypothesis)
        self.generate_test_hyp_btn.setEnabled(False)  # Initially disabled
        group_layout.addWidget(self.generate_test_hyp_btn, 0, 2)
        
        # Edit Hypothesis button
        self.edit_hyp_btn = QPushButton()
        self.edit_hyp_btn.setIcon(load_bootstrap_icon("pencil-square"))
        self.edit_hyp_btn.setToolTip("Edit selected hypothesis")
        self.edit_hyp_btn.setFixedSize(40, 40)
        self.edit_hyp_btn.clicked.connect(self.on_edit_hypothesis)
        self.edit_hyp_btn.setEnabled(False)  # Initially disabled
        group_layout.addWidget(self.edit_hyp_btn, 1, 0)
        
        # Evidence button
        self.evidence_btn = QPushButton()
        self.evidence_btn.setIcon(load_bootstrap_icon("journal-text"))
        self.evidence_btn.setToolTip("Manage evidence for selected hypothesis")
        self.evidence_btn.setFixedSize(40, 40)
        self.evidence_btn.clicked.connect(self.on_manage_evidence)
        self.evidence_btn.setEnabled(False)  # Initially disabled
        group_layout.addWidget(self.evidence_btn, 1, 1)
        
        # Hypothesis state selector
        self.hypothesis_state_combo = QComboBox()
        self.hypothesis_state_combo.setToolTip("Set hypothesis state")
        for state in HypothesisState:
            self.hypothesis_state_combo.addItem(state.value.replace("_", " ").title(), state)
        self.hypothesis_state_combo.setEnabled(False)  # Initially disabled
        self.hypothesis_state_combo.currentIndexChanged.connect(self.on_hypothesis_state_changed)
        group_layout.addWidget(self.hypothesis_state_combo, 2, 0, 1, 2)
        
        # Remove button
        self.remove_node_btn = QPushButton()
        self.remove_node_btn.setIcon(load_bootstrap_icon("trash"))
        self.remove_node_btn.setToolTip("Remove selected node")
        self.remove_node_btn.setFixedSize(40, 40)
        self.remove_node_btn.clicked.connect(self.on_remove_node)
        self.remove_node_btn.setEnabled(False)  # Initially disabled
        group_layout.addWidget(self.remove_node_btn, 3, 0, 1, 2)
        
        layout.addWidget(group)
    
    def add_view_options_group(self, layout):
        """Add view options group to control panel"""
        group = QGroupBox("View Options")
        group_layout = QGridLayout(group)
        
        # Toggle grid lines button
        self.grid_lines_check = QCheckBox("Show Grid Lines")
        self.grid_lines_check.setChecked(True)
        self.grid_lines_check.stateChanged.connect(self.on_toggle_grid_lines)
        group_layout.addWidget(self.grid_lines_check, 0, 0, 1, 2)
        
        # Toggle snap to lanes
        self.lane_snap_check = QCheckBox("Snap to Lanes")
        self.lane_snap_check.setChecked(True)
        self.lane_snap_check.stateChanged.connect(self.on_toggle_lane_snap)
        group_layout.addWidget(self.lane_snap_check, 1, 0, 1, 2)
        
        # Center view button
        self.center_btn = QPushButton()
        self.center_btn.setIcon(load_bootstrap_icon("arrows-fullscreen"))
        self.center_btn.setToolTip("Center view on all nodes")
        self.center_btn.setFixedSize(40, 40)
        self.center_btn.clicked.connect(self.on_center_view)
        group_layout.addWidget(self.center_btn, 2, 0)
        
        # Auto arrange button
        self.arrange_btn = QPushButton()
        self.arrange_btn.setIcon(load_bootstrap_icon("grid-3x3"))
        self.arrange_btn.setToolTip("Auto arrange nodes")
        self.arrange_btn.setFixedSize(40, 40)
        self.arrange_btn.clicked.connect(self.on_auto_arrange)
        group_layout.addWidget(self.arrange_btn, 2, 1)
        
        # Create lanes button
        self.lanes_btn = QPushButton()
        self.lanes_btn.setIcon(load_bootstrap_icon("layout-three-columns"))
        self.lanes_btn.setToolTip("Create type lanes")
        self.lanes_btn.setFixedSize(40, 40)
        self.lanes_btn.clicked.connect(self.on_create_type_lanes)
        group_layout.addWidget(self.lanes_btn, 3, 0)
        
        # Import from studies manager
        self.import_studies_btn = QPushButton()
        self.import_studies_btn.setIcon(load_bootstrap_icon("arrow-bar-down"))
        self.import_studies_btn.setToolTip("Import from Studies Manager")
        self.import_studies_btn.setFixedSize(40, 40)
        self.import_studies_btn.clicked.connect(self.on_import_from_studies)
        group_layout.addWidget(self.import_studies_btn, 3, 1)
        
        # Grid spacing slider
        spacing_layout = QHBoxLayout()
        spacing_label = QLabel("Spacing:")
        spacing_layout.addWidget(spacing_label)
        
        self.spacing_slider = QSlider(Qt.Orientation.Horizontal)
        self.spacing_slider.setMinimum(50)
        self.spacing_slider.setMaximum(200)
        self.spacing_slider.setValue(100)
        self.spacing_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.spacing_slider.setTickInterval(25)
        self.spacing_slider.valueChanged.connect(self.on_spacing_changed)
        spacing_layout.addWidget(self.spacing_slider)
        
        group_layout.addLayout(spacing_layout, 4, 0, 1, 2)
        
        layout.addWidget(group)
    
    def update_control_buttons(self, selected_node=None):
        """Update control buttons based on selected node"""
        # Enable/disable buttons based on selected node type
        if selected_node:
            # Objective-specific buttons
            if selected_node.node_type == "objective":
                self.add_subobj_btn.setEnabled(True)
                self.edit_obj_btn.setEnabled(True)
                self.toggle_collapse_btn.setEnabled(True)
                self.add_hyp_btn.setEnabled(True)
                self.generate_hyp_btn.setEnabled(True)
                self.generate_test_hyp_btn.setEnabled(True)
                
                self.edit_hyp_btn.setEnabled(False)
                self.evidence_btn.setEnabled(False)
                self.hypothesis_state_combo.setEnabled(False)
                
                # Set the button to collapse/expand based on current state
                is_collapsed = getattr(selected_node, 'is_collapsed', False)
                self.toggle_collapse_btn.setIcon(
                    load_bootstrap_icon("arrows-expand" if is_collapsed else "arrows-collapse")
                )
                self.toggle_collapse_btn.setToolTip(
                    "Expand" if is_collapsed else "Collapse"
                )
            
            # Hypothesis-specific buttons
            elif selected_node.node_type == "hypothesis":
                self.add_subobj_btn.setEnabled(False)
                self.edit_obj_btn.setEnabled(False)
                self.toggle_collapse_btn.setEnabled(False)
                self.add_hyp_btn.setEnabled(False)
                self.generate_hyp_btn.setEnabled(False)
                
                self.edit_hyp_btn.setEnabled(True)
                self.evidence_btn.setEnabled(True)
                self.hypothesis_state_combo.setEnabled(True)
                
                # Set correct state in combo box
                current_state = selected_node.hypothesis_config.state
                for i in range(self.hypothesis_state_combo.count()):
                    if self.hypothesis_state_combo.itemData(i) == current_state:
                        self.hypothesis_state_combo.blockSignals(True)
                        self.hypothesis_state_combo.setCurrentIndex(i)
                        self.hypothesis_state_combo.blockSignals(False)
                        break
            
            # Enable remove button for all node types
            self.remove_node_btn.setEnabled(True)
        else:
            # Nothing selected, disable most buttons
            self.add_subobj_btn.setEnabled(False)
            self.edit_obj_btn.setEnabled(False)
            self.toggle_collapse_btn.setEnabled(False)
            self.add_hyp_btn.setEnabled(False)
            self.generate_hyp_btn.setEnabled(False)
            self.edit_hyp_btn.setEnabled(False)
            self.evidence_btn.setEnabled(False)
            self.hypothesis_state_combo.setEnabled(False)
            self.remove_node_btn.setEnabled(False)
            
            # Hide evidence panel if no hypothesis selected
            self.evidence_panel.hide()
    
    def set_studies_manager(self, studies_manager):
        """Set the studies manager"""
        self.studies_manager = studies_manager

    def on_node_selected(self, node):
        """Handle node selection"""
        self.update_control_buttons(node)
        
        # Update evidence panel if hypothesis selected
        if node.node_type == "hypothesis":
            self.evidence_panel.set_hypothesis(node)
        else:
            self.evidence_panel.hide()
    
    def on_node_activated(self, node):
        """Handle node activation (double-click)"""
        if node.node_type == "objective":
            self.on_edit_objective()
        elif node.node_type == "hypothesis":
            self.on_edit_hypothesis()
    
    def on_evidence_changed(self, hypothesis):
        """Handle evidence changes"""
        if hypothesis:
            hypothesis.update_evidence_summary()
    
    def on_new_plan(self):
        """Create a new research plan"""
        # Confirm if there are existing nodes
        if self.grid_scene.nodes_grid:
            confirm = QMessageBox.question(
                self,
                "Confirm New Plan",
                "Are you sure you want to create a new plan? All current data will be lost.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if confirm != QMessageBox.StandardButton.Yes:
                return
        
        # Clear the scene
        self.research_manager.create_new_plan()
        
        # Update control buttons
        self.update_control_buttons()
        
        # Hide evidence panel
        self.evidence_panel.hide()
    
    def on_add_objective(self):
        """Add a new objective"""
        # Get objective type
        objective_type = self.objective_type_combo.currentData()
        
        # Show dialog for objective text
        text, ok = QInputDialog.getText(
            self, 
            "New Objective", 
            f"Enter {objective_type.value.replace('_', ' ')} text:"
        )
        
        if ok and text:
            # Create objective config
            config = ObjectiveConfig(
                id=f"o{len(self.grid_scene.nodes_grid) + 1}",
                text=text,
                type=objective_type
            )
            
            # Find an empty position
            pos = GridPosition(row=0, column=0)
            while pos in self.grid_scene.nodes_grid:
                pos = GridPosition(row=pos.row + 1, column=pos.column)
            
            # Add objective
            node = self.grid_scene.add_node("objective", pos, config)
            
            if node:
                # Center view on the new node
                self.grid_view.center_on_node(node)
                
                # Update control buttons
                self.update_control_buttons(node)
    
    def on_add_sub_objective(self):
        """Add a sub-objective to selected objective"""
        # Get selected objective
        selected = self.grid_scene.selectedItems()
        
        objective_node = None
        for item in selected:
            if isinstance(item, ObjectiveNode):
                objective_node = item
                break
        
        if not objective_node:
            return
        
        # Get objective type
        objective_type = self.objective_type_combo.currentData()
        
        # Show dialog for objective text
        text, ok = QInputDialog.getText(
            self, 
            "New Sub-Objective", 
            f"Enter {objective_type.value.replace('_', ' ')} text:"
        )
        
        if ok and text:
            # Create sub-objective
            sub_obj = self.research_manager.add_sub_objective(
                objective_node, text, objective_type
            )
            
            if sub_obj:
                # Center view on the new sub-objective
                self.grid_view.center_on_node(sub_obj)
                
                # Update control buttons
                self.update_control_buttons(sub_obj)
    
    def on_edit_objective(self):
        """Edit the selected objective"""
        # Get selected objective
        selected = self.grid_scene.selectedItems()
        
        objective_node = None
        for item in selected:
            if isinstance(item, ObjectiveNode):
                objective_node = item
                break
        
        if not objective_node:
            return
        
        # Show editor dialog
        dialog = ObjectiveEditorDialog(objective_node.objective_config, self)
        
        if dialog.exec():
            # Update the objective with edited config
            updated_config = dialog.get_updated_config()
            objective_node.objective_config = updated_config
            
            # Update node data
            objective_node.node_data.update({
                'text': updated_config.text,
                'type': updated_config.type.value,
                'progress': updated_config.progress,
                'auto_generate': updated_config.auto_generate
            })
            
            # Refresh the node text
            objective_node.setup_objective_text()
            
            # Update appearance
            objective_node.update()
    
    def on_toggle_collapse(self):
        """Toggle collapse/expand of selected objective"""
        # Get selected objective
        selected = self.grid_scene.selectedItems()
        
        objective_node = None
        for item in selected:
            if isinstance(item, ObjectiveNode):
                objective_node = item
                break
        
        if not objective_node:
            return
        
        # Toggle collapse state
        objective_node.toggle_collapse()
        
        # Update button icon
        if objective_node.is_collapsed:
            self.toggle_collapse_btn.setIcon(load_bootstrap_icon("arrows-expand"))
        else:
            self.toggle_collapse_btn.setIcon(load_bootstrap_icon("arrows-collapse"))
    
    def on_add_hypothesis(self):
        """Add a hypothesis to selected objective"""
        # Get selected objective
        selected = self.grid_scene.selectedItems()
        
        objective_node = None
        for item in selected:
            if isinstance(item, ObjectiveNode):
                objective_node = item
                break
        
        if not objective_node:
            return
        
        # Show dialog for hypothesis text
        text, ok = QInputDialog.getText(
            self, "New Hypothesis", "Enter hypothesis text:"
        )
        
        if ok and text:
            # Create hypothesis
            hypothesis = self.research_manager.add_hypothesis_to_objective(
                objective_node, text
            )
            
            if hypothesis:
                # Center view on the new hypothesis
                self.grid_view.center_on_node(hypothesis)
                
                # Update control buttons
                self.update_control_buttons(hypothesis)
    
    def on_generate_hypotheses(self):
        """Auto-generate hypotheses for the selected objective"""
        # Get selected objective
        selected = self.grid_scene.selectedItems()
        
        objective_node = None
        for item in selected:
            if isinstance(item, ObjectiveNode):
                objective_node = item
                break
        
        if not objective_node:
            return
        
        # Use a default of 3 hypotheses instead of asking
        number = 3
        
        # Simple placeholder implementation for hypothesis generation
        # In a real implementation, this would call an AI service
        generated_hypotheses = self.generate_placeholder_hypotheses(
            objective_node.objective_config.text, number
        )
        
        # Create hypotheses
        created_nodes = []
        for hypothesis_text in generated_hypotheses:
            hypothesis = self.research_manager.add_hypothesis_to_objective(
                objective_node, hypothesis_text
            )
            if hypothesis:
                created_nodes.append(hypothesis)
        
        if created_nodes:
            # Auto arrange nodes
            self.grid_scene.auto_sort_nodes()
            
            # Center on the first hypothesis
            self.grid_view.center_on_node(created_nodes[0])
            
            # Update control buttons
            self.update_control_buttons(created_nodes[0])
            
            # Show success message
            QMessageBox.information(
                self,
                "Hypotheses Generated",
                f"Successfully generated {len(created_nodes)} hypotheses."
            )
    
    def generate_placeholder_hypotheses(self, objective_text, count):
        """Generate placeholder hypotheses based on objective text
        This is a simple implementation that would be replaced with actual AI calls"""
        prefixes = [
            "Changes in ",
            "Increased levels of ",
            "Decreasing ",
            "Optimization of ",
            "Combination of ",
            "Interaction between ",
            "Correlation of ",
            "Absence of ",
            "Presence of ",
            "Modification of "
        ]
        
        suffixes = [
            " leads to improved outcomes",
            " causes significant changes",
            " results in measurable differences",
            " has no effect on the system",
            " correlates with increased efficiency",
            " contributes to decreased performance",
            " is mediated by external factors",
            " depends on environmental conditions",
            " varies based on participant characteristics",
            " shows different effects across populations"
        ]
        
        # Extract keywords from objective
        words = objective_text.split()
        keywords = [w for w in words if len(w) > 4 and w.isalpha()]
        
        if not keywords:
            keywords = ["factor", "variable", "component", "element", "parameter"]
        
        # Generate hypotheses
        hypotheses = []
        for i in range(count):
            prefix = prefixes[i % len(prefixes)]
            suffix = suffixes[i % len(suffixes)]
            keyword = keywords[i % len(keywords)]
            
            hypothesis = f"{prefix}{keyword}{suffix}"
            hypotheses.append(hypothesis)
        
        return hypotheses

    async def _generate_and_test_hypothesis(self, objective_node):
        """Generate a single hypothesis from objective and automatically test it
        
        Args:
            objective_node: The objective node to generate hypothesis for
            
        Returns:
            HypothesisNode: The created and tested hypothesis node
        """
        if not hasattr(self, 'studies_manager') or not self.studies_manager:
            QMessageBox.warning(self, "Error", "Studies manager not available")
            return None
        
        # Get the active study to check available datasets
        active_study = self.studies_manager.get_active_study()
        if not active_study or not hasattr(active_study, 'available_datasets') or not active_study.available_datasets:
            QMessageBox.warning(self, "Error", "No datasets available for testing")
            return None
        
        # Get all available datasets and their metadata
        datasets_info = []
        for dataset in active_study.available_datasets:
            if isinstance(dataset, dict):
                name = dataset.get('name')
                data = dataset.get('data')
                metadata = dataset.get('metadata', {})
            elif isinstance(dataset, tuple) and len(dataset) >= 2:
                name = dataset[0]
                data = dataset[1]
                metadata = dataset[2] if len(dataset) > 2 else {}
            else:
                continue
            
            if name and isinstance(data, pd.DataFrame) and not data.empty:
                datasets_info.append({
                    'name': name,
                    'shape': data.shape,
                    'columns': list(data.columns),
                    'metadata': metadata
                })
        
        if not datasets_info:
            QMessageBox.warning(self, "Error", "No valid datasets available")
            return None

        # Get existing hypotheses related to this objective
        existing_hypotheses_text = []
        if hasattr(self.research_manager, 'get_hypotheses_for_objective'):
            existing_hypotheses = self.research_manager.get_hypotheses_for_objective(objective_node.node_data['id'])
            for hyp_node in existing_hypotheses:
                if hasattr(hyp_node, 'hypothesis_config') and hasattr(hyp_node.hypothesis_config, 'text'):
                    existing_hypotheses_text.append(hyp_node.hypothesis_config.text)
        elif hasattr(objective_node, 'child_hypotheses'): # Fallback using node connections
            for hyp_node in objective_node.child_hypotheses:
                 if hasattr(hyp_node, 'hypothesis_config') and hasattr(hyp_node.hypothesis_config, 'text'):
                    existing_hypotheses_text.append(hyp_node.hypothesis_config.text)

        print(f"Existing hypotheses for objective '{objective_node.objective_config.text}': {existing_hypotheses_text}")

        # Generate the hypothesis and select dataset using LLM
        try:
            # Prepare prompt for LLM
            objective_text = objective_node.objective_config.text
            datasets_json = json.dumps(datasets_info, indent=2)
            existing_hypotheses_list_str = "\n".join([f"- {h}" for h in existing_hypotheses_text]) if existing_hypotheses_text else "None"

            prompt = f"""Given a research objective, a list of existing hypotheses for that objective, and available datasets, generate the *next* testable hypothesis or the first one if none exist, and identify the most suitable dataset for testing it.

Research Objective: "{objective_text}"

Existing Hypotheses for this Objective:
{existing_hypotheses_list_str}

Available Datasets:
{datasets_json}

Your task:
1. Analyze the research objective (e.g., "Test outcomes at months 3, 6, 9"). Identify *all* distinct sub-hypotheses implied by the objective's text or description.
2. Compare the *full list* of implied sub-hypotheses with the 'Existing Hypotheses' provided.
3. Determine the *next logical sub-hypothesis* that is implied by the objective but is *not* present in the 'Existing Hypotheses' list.
   - Prioritize based on any apparent sequence (e.g., time points like 3, 6, 9; components listed).
   - If no existing hypotheses match any implied sub-hypotheses, generate the first logical one (e.g., for 'months 3, 6, 9', the first might be 'month 9' or 'month 3' depending on the desired order).
   - If all implied sub-hypotheses from the objective are already present in the 'Existing Hypotheses' list, state that no new hypothesis is needed.
4. If a new sub-hypothesis is identified for generation:
   a. Formulate it as a specific, testable hypothesis statement.
   b. Select the most appropriate dataset for testing it.
   c. Explain why this dataset is the best choice.
   d. Identify the dependent (outcome) and independent (predictors) variables from this dataset.
   e. Provide formal null and alternative hypothesis statements.

Response format (JSON):
If a new hypothesis is generated:
{{
  "hypothesis": "The specific hypothesis statement (e.g., Test outcome at month 6)",
  "dataset_name": "name of the most appropriate dataset",
  "explanation": "brief explanation of why this dataset is appropriate",
  "variables": {{
    "outcome": "the dependent/outcome variable name",
    "predictors": ["list of independent/predictor variable names"]
  }},
  "null_hypothesis": "formal null hypothesis statement",
  "alternative_hypothesis": "formal alternative hypothesis statement",
  "status": "generate_next" // Indicate a hypothesis was generated
}}
If no new hypothesis is needed:
{{
  "status": "all_generated",
  "message": "All logical hypotheses for this objective appear to be generated already."
}}

Provide ONLY the JSON as your response with no additional text.
"""

            # Call LLM
            result = None
            from llms.client import call_llm_async
            
            result_text = await call_llm_async(prompt)
            
            # Parse LLM response
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                # Attempt to extract JSON from response if it contains extra text
                import re
                json_match = re.search(r'({.*})', result_text.replace('\n', ' '), re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                    except:
                        pass
                
                if not result:
                    raise ValueError("Could not parse LLM response as JSON")
            
            # Validate the response
            required_fields = ['hypothesis', 'dataset_name', 'variables']
            missing_fields = [field for field in required_fields if field not in result]
            
            if missing_fields:
                raise ValueError(f"LLM response missing required fields: {', '.join(missing_fields)}")
            
            # Create the hypothesis
            hypothesis_text = result['hypothesis']
            dataset_name = result['dataset_name']
            null_hypothesis = result.get('null_hypothesis', '')
            alternative_hypothesis = result.get('alternative_hypothesis', '')
            
            # Create hypothesis config with detailed data
            hypothesis_config = HypothesisConfig(
                id=str(uuid.uuid4()),
                text=hypothesis_text,
                state=HypothesisState.PROPOSED,
                confidence=0.5,  # Initial confidence
            )
            
            # Add additional data from LLM
            hypothesis_data = {
                'id': hypothesis_config.id,
                'title': hypothesis_text,
                'null_hypothesis': null_hypothesis,
                'alternative_hypothesis': alternative_hypothesis,
                'dataset_name': dataset_name,
                'variables': result.get('variables', {}),
                'explanation': result.get('explanation', ''),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'status': 'proposed'
            }
            
            # Add outcome_variables field explicitly to make it findable by select.py's matching function
            outcome_var = result['variables'].get('outcome')
            if outcome_var:
                hypothesis_data['outcome_variables'] = outcome_var
                print(f"Added outcome_variables: {outcome_var} to hypothesis for matching")
            
            # Add to studies manager
            hypothesis_id = self.studies_manager.add_hypothesis_to_study(
                hypothesis_text=hypothesis_text,
                hypothesis_data=hypothesis_data
            )
            
            if not hypothesis_id:
                raise ValueError("Failed to add hypothesis to studies manager")
            
            # Add to research grid
            hypothesis_node = self.research_manager.add_hypothesis_to_objective(
                objective_node, hypothesis_text
            )
            
            if not hypothesis_node:
                raise ValueError("Failed to create hypothesis node")
            
            # Set the ID to match what was stored in studies manager
            hypothesis_node.hypothesis_config.id = hypothesis_id
            hypothesis_node.node_data['id'] = hypothesis_id
            
            # Run the test on the selected dataset
            from plan.hypothesis_generator import HypothesisGeneratorWidget
            
            # Create a temporary generator widget for testing
            # Use None as parent to prevent it from being added to the UI
            generator = HypothesisGeneratorWidget(None)
            generator.set_studies_manager(self.studies_manager)
            
            # Set up the test data
            outcome_var = result['variables'].get('outcome')
            predictor_vars = result['variables'].get('predictors', [])
            group_var = predictor_vars[0] if predictor_vars else None
            
            # Perform the test silently in automated mode
            test_results = await generator.run_analysis_workflow(dataset_name=dataset_name, automated_mode=True)
            
            # Update the hypothesis with test results if available
            if test_results:
                print(f"Test results received: {test_results.get('success')}")
                
                test_result_data = {}
                
                # Get actual test result data from the testing widget directly
                if hasattr(generator, 'testing_widget') and generator.testing_widget:
                    print("Accessing test data from testing widget attributes")
                    
                    # Get current test key and name directly from the testing widget
                    test_key = None
                    test_name = "Unknown Test"
                    
                    if hasattr(generator.testing_widget, 'current_test_key'):
                        test_key = generator.testing_widget.current_test_key
                        print(f"Found test_key from current_test_key: {test_key}")
                    
                    if hasattr(generator.testing_widget, 'test_combo'):
                        # Get test name from combo box
                        test_name = generator.testing_widget.test_combo.currentText().strip()
                        print(f"Final test label update: {test_name}")
                        
                        # If we don't have a test key yet, try to get it from the combo box
                        if not test_key and hasattr(generator.testing_widget, 'available_tests'):
                            test_index = generator.testing_widget.test_combo.currentIndex()
                            if test_index >= 0:
                                test_keys = list(generator.testing_widget.available_tests.keys())
                                if test_index < len(test_keys):
                                    test_key = test_keys[test_index]
                    
                    # First try to get the latest test results directly
                    if hasattr(generator.testing_widget, 'current_test_result') and generator.testing_widget.current_test_result:
                        test_result_data = generator.testing_widget.current_test_result
                        print("Retrieved test results from testing_widget.current_test_result")
                    elif hasattr(generator.testing_widget, 'last_test_result') and generator.testing_widget.last_test_result:
                        test_result_data = generator.testing_widget.last_test_result
                        print("Retrieved test results from testing_widget.last_test_result")
                    # Fall back to the results parameter if needed
                    elif test_results.get('test_result'):
                        test_result_data = test_results['test_result']
                        print("Using test_result from run_analysis_workflow return value")
                    
                    # Make sure we have a test key and name in the result data
                    if test_result_data:
                        if 'test_key' not in test_result_data and test_key:
                            test_result_data['test_key'] = test_key
                        if 'test_name' not in test_result_data and test_name:
                            test_result_data['test_name'] = test_name
                        if 'test' not in test_result_data and test_name:
                            test_result_data['test'] = test_name
                
                # Fallback if we couldn't get results directly from the widget
                if not test_result_data and test_results.get('test_result'):
                    test_result_data = test_results['test_result']
                
                # Update hypothesis in studies manager if we have data
                if test_result_data:
                    print(f"Updating hypothesis {hypothesis_id} in studies manager")
                self.studies_manager.update_hypothesis_with_test_results(
                    hypothesis_id=hypothesis_id,
                        test_results=test_result_data
                )
                
                # Update the hypothesis node state based on the test result
                p_value = test_result_data.get('p_value')
                if p_value is not None:
                    # Default alpha level of 0.05
                    if p_value < 0.05:
                        # --- START MODIFICATION ---
                        # Use VALIDATED to match the state_colors map in HypothesisNode.paint
                        hypothesis_node.change_state(HypothesisState.VALIDATED)
                        # --- END MODIFICATION ---
                    else:
                        hypothesis_node.change_state(HypothesisState.REJECTED)
                else:
                    print("No test results data available to update hypothesis")
            
            # Return the created hypothesis node
            return hypothesis_node
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Error", f"Failed to generate hypothesis: {str(e)}")
            return None
    
    def on_import_from_studies(self):
        """Import hypotheses from the studies manager"""
        if not self.studies_manager:
            QMessageBox.warning(
                self,
                "Import Failed",
                "No studies manager available. Please set up the studies manager first."
            )
            return
            
        # Get active study
        active_study = self.studies_manager.get_active_study()
        if not active_study:
            QMessageBox.warning(
                self,
                "Import Failed",
                "No active study found. Please select a study first."
            )
            return
            
        # Confirm import
        confirm = QMessageBox.question(
            self,
            "Import Study",
            f"Import hypotheses from study '{getattr(active_study, 'title', 'Untitled Study')}'?\n\n"
            f"This will replace the current research plan.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if confirm != QMessageBox.StandardButton.Yes:
            return
            
        # Import from studies manager
        nodes = self.research_manager.import_from_studies_manager(self.studies_manager)
        
        if not nodes or not nodes['hypotheses']:
            QMessageBox.warning(
                self,
                "Import Failed",
                "No hypotheses found in the active study or import failed."
            )
            return
            
        # Show success message
        QMessageBox.information(
            self,
            "Import Successful",
            f"Successfully imported {len(nodes['hypotheses'])} hypotheses from study '{getattr(active_study, 'title', 'Untitled Study')}'."
        )
        
        # Center view on all nodes
        self.on_center_view()

    def save(self, filepath):
        """Save the research plan to a file"""
        # Save scene to JSON
        json_data = self.grid_scene.to_json()
        
        # Write to file
        with open(filepath, 'w') as f:
            f.write(json_data)
        
        self.last_save_path = filepath
        return True

    def on_export_plan(self):
        """Export the research plan"""
        # Ask for export format
        format_dialog = QDialog(self)
        format_dialog.setWindowTitle("Export Format")
        dialog_layout = QVBoxLayout(format_dialog)
        
        dialog_layout.addWidget(QLabel("Select export format:"))
        
        # Format buttons
        json_btn = QPushButton("JSON")
        json_btn.clicked.connect(lambda: format_dialog.done(1))  # JSON = 1
        dialog_layout.addWidget(json_btn)
        
        protocol_btn = QPushButton("Research Protocol")
        protocol_btn.clicked.connect(lambda: format_dialog.done(2))  # Protocol = 2
        dialog_layout.addWidget(protocol_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(lambda: format_dialog.reject())
        dialog_layout.addWidget(cancel_btn)
        
        # Show dialog and get result
        result = format_dialog.exec()
        
        if result == 1:
            # Export as JSON
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Plan", "", "JSON Files (*.json)"
            )
            
            if filename:
                try:
                    data = self.research_manager.export_to_json()
                    with open(filename, 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    QMessageBox.information(
                        self, "Export Successful", 
                        f"Research plan exported to {filename}"
                    )
                except Exception as e:
                    QMessageBox.critical(
                        self, "Export Error", 
                        f"Error exporting plan: {str(e)}"
                    )
        
        elif result == 2:
            # Export as research protocol
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Protocol", "", "JSON Files (*.json)"
            )
            
            if filename:
                try:
                    protocol = self.research_manager.export_to_protocol()
                    with open(filename, 'w') as f:
                        json.dump(protocol, f, indent=2)
                    
                    QMessageBox.information(
                        self, "Export Successful", 
                        f"Protocol exported to {filename}"
                    )
                except Exception as e:
                    QMessageBox.critical(
                        self, "Export Error", 
                        f"Error exporting protocol: {str(e)}"
                    )

    def on_auto_test_hypothesis(self):
        """Generate and test a hypothesis for the selected objective"""
        # Get selected objective
        selected = self.grid_scene.selectedItems()
        
        objective_node = None
        for item in selected:
            if isinstance(item, ObjectiveNode):
                objective_node = item
                break
        
        if not objective_node:
            return
        
        # Use the existing event loop if it's running, otherwise create a new one
        import asyncio
        
        # Show progress message - safely
        self.show_status_message("Generating and testing hypothesis...")
        
        # Check if event loop is already running
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a task
                task = asyncio.create_task(self._generate_and_test_hypothesis(objective_node))
                # We need to set up a callback to handle the result
                task.add_done_callback(self._on_hypothesis_generated)
                return
                
            # If we get here, loop exists but is not running
            # Run the task in the existing loop
            try:
                hypothesis_node = loop.run_until_complete(self._generate_and_test_hypothesis(objective_node))
                self._handle_generated_hypothesis(hypothesis_node)
            except Exception as e:
                import traceback
                traceback.print_exc()
                QMessageBox.warning(self, "Error", f"Error generating and testing hypothesis: {str(e)}")
            finally:
                self.clear_status_message()
                
        except RuntimeError:
            # If we get here, no event loop exists in this thread
            # Create a new loop and run the task
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            
            try:
                hypothesis_node = new_loop.run_until_complete(self._generate_and_test_hypothesis(objective_node))
                self._handle_generated_hypothesis(hypothesis_node)
            except Exception as e:
                import traceback
                traceback.print_exc()
                QMessageBox.warning(self, "Error", f"Error generating and testing hypothesis: {str(e)}")
            finally:
                new_loop.close()
                self.clear_status_message()
    
    def _on_hypothesis_generated(self, task):
        """Handle the result of an async hypothesis generation task"""
        try:
            hypothesis_node = task.result()
            self._handle_generated_hypothesis(hypothesis_node)
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Error", f"Error generating and testing hypothesis: {str(e)}")
        finally:
            self.clear_status_message()
            
    def _handle_generated_hypothesis(self, hypothesis_node):
        """Process a successfully generated hypothesis node"""
        if hypothesis_node:
            # Auto arrange nodes
            self.grid_scene.auto_sort_nodes()
            
            # Center on the hypothesis
            self.grid_view.center_on_node(hypothesis_node)
            
            # Update control buttons
            self.update_control_buttons(hypothesis_node)
            
            # Show success message
            # Use HypothesisState.VALIDATED to match the state set in _generate_and_test_hypothesis
            state_text = "CONFIRMED" if hypothesis_node.hypothesis_config.state == HypothesisState.VALIDATED else "REJECTED"
            QMessageBox.information(
                self,
                "Hypothesis Generated and Tested",
                f"Successfully generated and tested hypothesis.\nResult: {state_text}"
            )
    
    def show_status_message(self, message):
        """Safely show a status message if a status bar is available"""
        try:
            # Try to get status bar from parent window
            main_window = self.window()
            if main_window and hasattr(main_window, 'statusBar'):
                main_window.statusBar().showMessage(message)
                return
                
            # Log the message instead if no status bar
            print(f"Status: {message}")
        except Exception as e:
            # If all else fails, just print to console
            print(f"Status (error showing message): {message}, Error: {str(e)}")
            
    def clear_status_message(self):
        """Safely clear the status message if a status bar is available"""
        try:
            # Try to get status bar from parent window
            main_window = self.window()
            if main_window and hasattr(main_window, 'statusBar'):
                main_window.statusBar().clearMessage()
        except Exception as e:
            # Just log the error
            print(f"Error clearing status message: {str(e)}")

    def on_edit_hypothesis(self):
        """Edit the selected hypothesis"""
        # Get selected hypothesis
        selected = self.grid_scene.selectedItems()
        
        hypothesis_node = None
        for item in selected:
            if isinstance(item, HypothesisNode):
                hypothesis_node = item
                break
        
        if not hypothesis_node:
            return
        
        # Show editor dialog
        dialog = HypothesisEditorDialog(hypothesis_node.hypothesis_config, self)
        
        # Pass studies manager to the dialog if available
        if hasattr(self, 'studies_manager') and self.studies_manager:
            dialog.studies_manager = self.studies_manager
        
        if dialog.exec():
            # Update the hypothesis with edited config
            updated_config = dialog.get_updated_config()
            hypothesis_node.hypothesis_config = updated_config
            
            # Update node data
            hypothesis_node.node_data.update({
                'text': updated_config.text,
                'state': updated_config.state.value,
                'confidence': updated_config.confidence
            })
            
            # Refresh the node text
            hypothesis_node.setup_hypothesis_text()
            
            # Update appearance
            hypothesis_node.update()
            
            # Update evidence panel if visible
            if self.evidence_panel.isVisible():
                self.evidence_panel.set_hypothesis(hypothesis_node)
    
    def on_manage_evidence(self):
        """Manage evidence for the selected hypothesis"""
        # Get selected hypothesis
        selected = self.grid_scene.selectedItems()
        
        hypothesis_node = None
        for item in selected:
            if isinstance(item, HypothesisNode):
                hypothesis_node = item
                break
        
        if not hypothesis_node:
            return
        
        # Update evidence panel
        self.evidence_panel.set_hypothesis(hypothesis_node)
        
        # Show evidence panel if hidden
        if not self.evidence_panel.isVisible():
            self.evidence_panel.show()
    
    def on_hypothesis_state_changed(self, index):
        """Handle hypothesis state change from combo box"""
        # Only process if enabled (prevents triggering during setup)
        if not self.hypothesis_state_combo.isEnabled():
            return
            
        # Get new state
        new_state = self.hypothesis_state_combo.itemData(index)
        
        # Get selected hypothesis
        selected = self.grid_scene.selectedItems()
        
        hypothesis_node = None
        for item in selected:
            if isinstance(item, HypothesisNode):
                hypothesis_node = item
                break
        
        if not hypothesis_node:
            return
        
        # Update hypothesis state
        hypothesis_node.change_state(new_state)
    
    def on_remove_node(self):
        """Remove the selected node"""
        # Get selected node
        selected = self.grid_scene.selectedItems()
        
        for item in selected:
            if isinstance(item, BaseNode) and item.grid_position:
                self.grid_scene.remove_node(item.grid_position)
            elif isinstance(item, NodeConnection):
                item.delete_connection()
        
        # Update control buttons
        self.update_control_buttons()
        
        # Hide evidence panel
        self.evidence_panel.hide()
    
    def on_toggle_grid_lines(self, state):
        """Toggle grid lines visibility"""
        self.grid_scene.grid_visible = state == Qt.CheckState.Checked
        self.grid_scene.draw_grid_lines()
    
    def on_center_view(self):
        """Center view on the scene"""
        if self.grid_scene.nodes_grid:
            first_node = next(iter(self.grid_scene.nodes_grid.values()))
            self.grid_view.center_on_node(first_node)
        else:
            self.grid_view.centerOn(0, 0)
    
    def on_toggle_lane_snap(self, state):
        """Toggle lane snapping"""
        is_enabled = self.grid_scene.toggle_snap_to_lanes()
        self.lane_snap_check.setChecked(is_enabled)
    
    def on_create_type_lanes(self):
        """Create lanes for each node type"""
        self.grid_scene.create_type_lanes()
    
    def on_auto_arrange(self):
        """Auto-arrange nodes in the grid"""
        try:
            # Reset view to normal scale
            self.grid_view.resetTransform()
            
            # First create lanes if they don't exist
            if not self.grid_scene.lanes:
                self.grid_scene.create_type_lanes()
            
            # Sort nodes by type
            result = self.grid_scene.auto_sort_nodes()
                
            if result:
                # Find the center point of all nodes
                if self.grid_scene.nodes_grid:
                    min_x = float('inf')
                    max_x = float('-inf')
                    min_y = float('inf')
                    max_y = float('-inf')
                    
                    # Create a copy of nodes_grid values to safely iterate
                    nodes_to_check = list(self.grid_scene.nodes_grid.values())
                    
                    for node in nodes_to_check:
                        try:
                            node_rect = node.sceneBoundingRect()
                            min_x = min(min_x, node_rect.left())
                            max_x = max(max_x, node_rect.right())
                            min_y = min(min_y, node_rect.top())
                            max_y = max(max_y, node_rect.bottom())
                        except (RuntimeError, AttributeError, ReferenceError):
                            # Skip if node has become invalid
                            continue
                    
                    # Only center if we have valid bounds
                    if min_x != float('inf') and max_x != float('-inf') and min_y != float('inf') and max_y != float('-inf'):
                        # Center on the middle of all nodes
                        center_x = (min_x + max_x) / 2
                        center_y = (min_y + max_y) / 2
                        self.grid_view.centerOn(center_x, center_y)
        except Exception as e:
            print(f"Error in auto-arrange: {e}")
            import traceback
            traceback.print_exc()
    
    def update_theme(self, is_dark):
        """Update the theme for the widget and components"""
        self.current_theme = "dark" if is_dark else "light"
        
        # Update grid scene theme
        self.grid_scene.update_theme(self.current_theme)
        
        # Update control panel styling
        button_style = """
            QPushButton {
                background-color: %s;
                color: %s;
                border: none;
                border-radius: 4px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: %s;
            }
            QPushButton:disabled {
                background-color: %s;
                color: %s;
            }
        """ % (
            "#2D2D2D" if is_dark else "#F0F0F0",  # Button background
            "#FFFFFF" if is_dark else "#000000",  # Text color
            "#3D3D3D" if is_dark else "#E0E0E0",  # Hover color
            "#1D1D1D" if is_dark else "#CCCCCC",  # Disabled background
            "#666666" if is_dark else "#888888"   # Disabled text
        )
        
        # Update group box styling
        group_style = """
            QGroupBox {
                border: 1px solid %s;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 8px;
                color: %s;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 3px;
            }
        """ % (
            "#404040" if is_dark else "#CCCCCC",  # Border color
            "#FFFFFF" if is_dark else "#000000"   # Text color
        )
        
        # Apply styles
        for button in self.control_panel.findChildren(QPushButton):
            button.setStyleSheet(button_style)
        
        for group in self.control_panel.findChildren(QGroupBox):
            group.setStyleSheet(group_style)
            
        # Update evidence panel
        self.evidence_panel.setStyleSheet(f"background-color: {('#333333' if is_dark else '#F8F8F8')}")
    
    def on_spacing_changed(self, value):
        """Handle spacing slider value change"""
        self.grid_scene.minimum_node_distance = value
        
        # Apply new spacing to existing nodes
        nodes = [item for item in self.grid_scene.items() if isinstance(item, BaseNode)]
        
        # Apply collision prevention to all nodes
        for node in nodes:
            self.grid_scene.prevent_node_collision(node)

    def on_generate_sample(self):
        """Generate a sample research plan for testing"""
        # Clear existing plan
        self.on_new_plan()
        
        # Create main research question directly instead of using dialog
        question_config = ObjectiveConfig(
            id=str(uuid.uuid4()),
            text="How do environmental factors affect neural development in early childhood?",
            type=ObjectiveType.RESEARCH_QUESTION,
            progress=0.3
        )
        question = self.grid_scene.add_node("objective", GridPosition(0, 2), question_config)
        question.setPos(0, 0)
        
        # Add sub-goals to the main question
        goal1_config = ObjectiveConfig(
            id=str(uuid.uuid4()),
            text="Identify key environmental pollutants that impact brain development",
            type=ObjectiveType.GOAL,
            progress=0.65,
            parent_id=question_config.id
        )
        goal1 = self.grid_scene.add_node("objective", GridPosition(1, 0), goal1_config)
        goal1.setPos(-300, 200)
        
        goal2_config = ObjectiveConfig(
            id=str(uuid.uuid4()),
            text="Determine critical developmental windows for environmental sensitivity",
            type=ObjectiveType.GOAL,
            progress=0.45,
            parent_id=question_config.id
        )
        goal2 = self.grid_scene.add_node("objective", GridPosition(1, 2), goal2_config)
        goal2.setPos(0, 200)
        
        goal3_config = ObjectiveConfig(
            id=str(uuid.uuid4()),
            text="Assess socioeconomic factors as mediators of environmental exposure",
            type=ObjectiveType.GOAL,
            progress=0.2,
            parent_id=question_config.id
        )
        goal3 = self.grid_scene.add_node("objective", GridPosition(1, 4), goal3_config)
        goal3.setPos(300, 200)
        
        # Connect goals to the main question
        self.grid_scene.create_connection(
            goal1.output_ports[0], 
            question.input_ports[0],
            ConnectionType.CONTRIBUTES_TO
        )
        self.grid_scene.create_connection(
            goal2.output_ports[0], 
            question.input_ports[0],
            ConnectionType.CONTRIBUTES_TO
        )
        self.grid_scene.create_connection(
            goal3.output_ports[0], 
            question.input_ports[0],
            ConnectionType.CONTRIBUTES_TO
        )
        
        # Create hypotheses for goal 1 (pollution)
        h1_config = HypothesisConfig(
            id=str(uuid.uuid4()),
            text="Lead exposure during pregnancy significantly impairs cognitive development",
            state=HypothesisState.TESTING,
            confidence=0.7
        )
        h1 = self.grid_scene.add_node("hypothesis", GridPosition(3, 0), h1_config)
        h1.setPos(-300, 350)
        
        # Create chained hypothesis
        h1_1_config = HypothesisConfig(
            id=str(uuid.uuid4()),
            text="The effects of lead exposure are mediated by disruption of NMDA receptors",
            state=HypothesisState.PROPOSED,
            confidence=0.4
        )
        h1_1 = self.grid_scene.add_node("hypothesis", GridPosition(4, 0), h1_1_config)
        h1_1.setPos(-300, 500)
        
        # Connect hypotheses to their respective goals
        self.grid_scene.create_connection(
            goal1.output_ports[0], 
            h1.input_ports[0],
            ConnectionType.CONTRIBUTES_TO
        )
        self.grid_scene.create_connection(
            h1.input_ports[1], 
            h1_1.input_ports[1],
            ConnectionType.CONTRIBUTES_TO
        )
        
        # Center view
        self.grid_view.center_on_node(question)
        
        # Create lanes
        self.grid_scene.create_type_lanes()


# ======================
# Research Manager
# ======================

class ResearchManager:
    """Manager for research plan operations"""
    
    def __init__(self, grid_scene):
        self.grid_scene = grid_scene
    
    def create_new_plan(self):
        """Create a new empty research plan"""
        # Clear the current grid
        self.grid_scene.clear()
        
        # Create default lanes
        self.grid_scene.create_type_lanes()
        
        # Add a default research question
        question_pos = GridPosition(row=2, column=1)
        config = ObjectiveConfig(
            id="q1",
            text="What is the main research question?",
            type=ObjectiveType.RESEARCH_QUESTION,
            description="Define the primary research question"
        )
        self.grid_scene.add_node("objective", question_pos, config)
        
        # Expand grid to appropriate size
        self.grid_scene.expand_grid(0, 5, 0, 5)
    
    def create_plan_with_objective(self, objective_text, objective_type=ObjectiveType.RESEARCH_QUESTION):
        """Create a plan with a single objective"""
        # Create new plan
        self.create_new_plan()
        
        # Add objective
        obj_pos = GridPosition(row=2, column=1)
        config = ObjectiveConfig(
            id="obj1",
            text=objective_text,
            type=objective_type,
            description="Main research objective"
        )
        self.grid_scene.add_node("objective", obj_pos, config)

    def import_from_studies_manager(self, studies_manager, study_id=None):
        """
        Import hypotheses from the studies manager into the research planning graph.
        
        Args:
            studies_manager: The studies manager instance
            study_id: Optional ID of specific study to import (imports active study if None)
        
        Returns:
            Dictionary mapping study IDs to lists of created nodes
        """
        if not studies_manager:
            return None
            
        # Get the study to import
        study = None
        if study_id:
            study = studies_manager.get_study(study_id)
        else:
            study = studies_manager.get_active_study()
            
        if not study or not hasattr(study, 'hypotheses') or not study.hypotheses:
            return None
            
        # Create a new plan
        self.create_new_plan()
        
        # Create a main objective for the study
        study_title = getattr(study, 'title', 'Research Study')
        obj_pos = GridPosition(row=2, column=1)
        obj_config = ObjectiveConfig(
            id=f"obj_{study.id}" if hasattr(study, 'id') else "obj_main",
            text=study_title,
            type=ObjectiveType.RESEARCH_QUESTION,
            description=f"Main research objective from study: {study_title}"
        )
        main_obj = self.grid_scene.add_node("objective", obj_pos, obj_config)
        
        # Track created nodes
        created_nodes = {
            'objectives': [main_obj],
            'hypotheses': []
        }
        
        # Process hypotheses
        row = 1
        for hyp_data in study.hypotheses:
            # Calculate position for hypothesis (organize in rows)
            hyp_pos = GridPosition(row=row, column=3)
            
            # Check if position is already occupied
            while hyp_pos in self.grid_scene.nodes_grid:
                row += 1
                hyp_pos = GridPosition(row=row, column=3)
            
            # Convert string status to enum
            status_value = hyp_data.get('status', 'untested')
            state = get_hypothesis_state(status_value)
            
            # Extract variables
            variables = []
            outcome_vars = hyp_data.get('outcome_variables', '').split(',')
            variables.extend([v.strip() for v in outcome_vars if v.strip()])
            
            predictor_vars = hyp_data.get('predictor_variables', '').split(',')
            variables.extend([v.strip() for v in predictor_vars if v.strip()])
            
            # Create hypothesis config
            hyp_config = HypothesisConfig(
                id=hyp_data.get('id', f"h{len(created_nodes['hypotheses']) + 1}"),
                text=hyp_data.get('title', 'Untitled Hypothesis'),
                state=state,
                confidence=hyp_data.get('confidence', 0.5),
                variables=variables
            )
            
            # Add supporting evidence if available
            if 'supporting_evidence' in hyp_data and hyp_data['supporting_evidence']:
                for ev_data in hyp_data['supporting_evidence']:
                    ev_type = EvidenceSourceType.LITERATURE
                    if ev_data.get('type') == 'data':
                        ev_type = EvidenceSourceType.DATA
                    elif ev_data.get('type') == 'experiment':
                        ev_type = EvidenceSourceType.EXPERIMENT
                    elif ev_data.get('type') == 'observation':
                        ev_type = EvidenceSourceType.OBSERVATION
                        
                    evidence = Evidence(
                        id=ev_data.get('id', str(uuid.uuid4())),
                        type=ev_type,
                        description=ev_data.get('description', 'No description'),
                        supports=True,
                        confidence=ev_data.get('confidence', 0.7),
                        source=ev_data.get('source', ''),
                        notes=ev_data.get('notes', ''),
                        status=ev_data.get('status', 'validated')
                    )
                    hyp_config.supporting_evidence.append(evidence)
            
            # Add contradicting evidence if available
            if 'contradicting_evidence' in hyp_data and hyp_data['contradicting_evidence']:
                for ev_data in hyp_data['contradicting_evidence']:
                    ev_type = EvidenceSourceType.LITERATURE
                    if ev_data.get('type') == 'data':
                        ev_type = EvidenceSourceType.DATA
                    elif ev_data.get('type') == 'experiment':
                        ev_type = EvidenceSourceType.EXPERIMENT
                    elif ev_data.get('type') == 'observation':
                        ev_type = EvidenceSourceType.OBSERVATION
                        
                    evidence = Evidence(
                        id=ev_data.get('id', str(uuid.uuid4())),
                        type=ev_type,
                        description=ev_data.get('description', 'No description'),
                        supports=False,
                        confidence=ev_data.get('confidence', 0.7),
                        source=ev_data.get('source', ''),
                        notes=ev_data.get('notes', ''),
                        status=ev_data.get('status', 'validated')
                    )
                    hyp_config.contradicting_evidence.append(evidence)
            
            # Add literature evidence if available
            if 'literature_evidence' in hyp_data and hyp_data['literature_evidence']:
                # Set the literature_evidence attribute on the config object
                hyp_config.literature_evidence = hyp_data['literature_evidence']
                
            # Add test results if available
            if 'test_results' in hyp_data and hyp_data['test_results']:
                # Set the test_results attribute on the config object
                hyp_config.test_results = hyp_data['test_results']
                
            # Add alpha level if available
            if 'alpha_level' in hyp_data:
                hyp_config.alpha_level = hyp_data['alpha_level']
            
            # DEBUG PRINT: Check the final evidence counts before adding node
            print(f"DEBUG: Hypothesis '{hyp_config.text}'")
            print(f"  Config Supporting Evidence Count: {len(hyp_config.supporting_evidence)}")
            print(f"  Config Contradicting Evidence Count: {len(hyp_config.contradicting_evidence)}")
            
            # Debug literature and test evidence
            if hasattr(hyp_config, 'literature_evidence') and hyp_config.literature_evidence:
                lit_evidence = hyp_config.literature_evidence
                print(f"  Literature Evidence: Supporting={lit_evidence.get('supporting', 0)}, Refuting={lit_evidence.get('refuting', 0)}, Neutral={lit_evidence.get('neutral', 0)}")
            
            if hasattr(hyp_config, 'test_results') and hyp_config.test_results:
                test_results = hyp_config.test_results
                print(f"  Test Results: p-value={test_results.get('p_value', 'N/A')}, alpha={hyp_config.alpha_level}")
            
            # Optionally print the evidence details
            # for ev in hyp_config.supporting_evidence:
            #     print(f"    Supporting: {ev.description[:50]}...")
            # for ev in hyp_config.contradicting_evidence:
            #     print(f"    Contradicting: {ev.description[:50]}...")

            # Add hypothesis node
            hyp_node = self.grid_scene.add_node("hypothesis", hyp_pos, hyp_config)
            
            if hyp_node:
                created_nodes['hypotheses'].append(hyp_node)
                
                # Connect to main objective
                for obj_port in main_obj.output_ports:
                    if "hypothesis" in obj_port.allowed_connections:
                        for hyp_port in hyp_node.input_ports:
                            if "objective" in hyp_port.allowed_connections:
                                connection = self.grid_scene.create_connection(
                                    obj_port, hyp_port,
                                    ConnectionType.CONTRIBUTES_TO
                                )
                                
                                # Update connection appearance to reflect evidence counts
                                if connection and hasattr(connection, 'update_appearance'):
                                    connection.update_appearance()
                                break
                        break
                
                row += 1
        
        # Auto arrange nodes
        self.grid_scene.auto_sort_nodes()
        
        return created_nodes
    
    def create_plan_with_objective(self, objective_text, objective_type=ObjectiveType.RESEARCH_QUESTION):
        """Create a new plan with an initial objective"""
        # Create new plan
        self.create_new_plan()
        
        # Create central objective
        center_pos = GridPosition(row=0, column=0)
        
        # Create objective config
        config = ObjectiveConfig(
            id="o1",
            text=objective_text,
            type=objective_type
        )
        
        # Add central objective
        node = self.grid_scene.add_node("objective", center_pos, config)
        
        return node
    
    def add_sub_objective(self, parent_objective, text, objective_type=ObjectiveType.GOAL):
        """Add a sub-objective to a parent objective"""
        if not parent_objective or parent_objective.node_type != "objective":
            return None
        
        # Get parent grid position from grid scene
        parent_pos = None
        for pos, node in self.grid_scene.nodes_grid.items():
            if node == parent_objective:
                parent_pos = pos
                break
                
        if not parent_pos:
            return None
        
        # Calculate sub-objective position (below parent)
        sub_pos = GridPosition(row=parent_pos.row + 2, column=parent_pos.column)
        
        # Check if position is already occupied
        if sub_pos in self.grid_scene.nodes_grid:
            # Find next available position
            for offset in range(1, 6):
                # Try columns to the right
                test_pos = GridPosition(row=sub_pos.row, column=sub_pos.column + offset)
                if test_pos not in self.grid_scene.nodes_grid:
                    sub_pos = test_pos
                    break
                
                # Try columns to the left
                test_pos = GridPosition(row=sub_pos.row, column=sub_pos.column - offset)
                if test_pos not in self.grid_scene.nodes_grid:
                    sub_pos = test_pos
                    break
                
                # Try further down
                test_pos = GridPosition(row=sub_pos.row + offset, column=sub_pos.column)
                if test_pos not in self.grid_scene.nodes_grid:
                    sub_pos = test_pos
                    break
            else:
                # Couldn't find space
                return None
        
        # Create sub-objective config
        config = ObjectiveConfig(
            id=f"o{len(self.grid_scene.nodes_grid) + 1}",
            text=text,
            type=objective_type,
            parent_id=parent_objective.node_data['id']
        )
        
        # Add sub-objective
        sub_node = self.grid_scene.add_node("objective", sub_pos, config)
        
        if not sub_node:
            return None
        
        # Connect parent to sub-objective - always use CONTRIBUTES_TO
        if parent_objective.output_ports and sub_node.input_ports:
            for parent_port in parent_objective.output_ports:
                if "objective" in parent_port.allowed_connections:
                    for sub_port in sub_node.input_ports:
                        if "objective" in sub_port.allowed_connections:
                            self.grid_scene.create_connection(parent_port, sub_port, 
                                                            ConnectionType.CONTRIBUTES_TO)
                            break
                    break
        
        return sub_node
    
    def add_hypothesis_to_objective(self, objective, text):
        """Add a hypothesis to an objective"""
        if not objective or objective.node_type != "objective":
            return None
        
        # Get objective grid position from grid scene
        obj_pos = None
        for pos, node in self.grid_scene.nodes_grid.items():
            if node == objective:
                obj_pos = pos
                break
                
        if not obj_pos:
            return None
        
        # Calculate hypothesis position (to the right of objective)
        hyp_pos = GridPosition(row=obj_pos.row, column=obj_pos.column + 2)
        
        # Check if position is already occupied
        if hyp_pos in self.grid_scene.nodes_grid:
            # Find next available position
            for offset in range(1, 6):
                # Try positions vertically down
                test_pos = GridPosition(row=hyp_pos.row + offset, column=hyp_pos.column)
                if test_pos not in self.grid_scene.nodes_grid:
                    hyp_pos = test_pos
                    break
                
                # Try positions vertically up
                test_pos = GridPosition(row=hyp_pos.row - offset, column=hyp_pos.column)
                if test_pos not in self.grid_scene.nodes_grid:
                    hyp_pos = test_pos
                    break
                
                # Try further right
                test_pos = GridPosition(row=hyp_pos.row, column=hyp_pos.column + offset)
                if test_pos not in self.grid_scene.nodes_grid:
                    hyp_pos = test_pos
                    break
            else:
                # Couldn't find space
                return None
        
        # Create hypothesis config
        config = HypothesisConfig(
            id=f"h{len(self.grid_scene.nodes_grid) + 1}",
            text=text,
            state=HypothesisState.PROPOSED
        )
        
        # Add hypothesis
        hyp_node = self.grid_scene.add_node("hypothesis", hyp_pos, config)
        
        if not hyp_node:
            return None
        
        # Connect objective to hypothesis - always use CONTRIBUTES_TO
        for obj_port in objective.output_ports:
            if "hypothesis" in obj_port.allowed_connections:
                for hyp_port in hyp_node.input_ports:
                    if "objective" in hyp_port.allowed_connections:
                        self.grid_scene.create_connection(obj_port, hyp_port,
                                                        ConnectionType.CONTRIBUTES_TO)
                        break
                break
        
        return hyp_node
    
    def export_to_json(self):
        """Export plan to JSON format"""
        data = {
            "objectives": [],
            "hypotheses": [],
            "connections": []
        }
        
        # Process nodes
        for node in self.grid_scene.nodes_grid.values():
            if node.node_type == "objective":
                # Export objective
                obj_data = {
                    "id": node.node_data['id'],
                    "text": node.node_data['text'],
                    "type": node.node_data['type'],
                    "description": node.objective_config.description,
                    "progress": node.node_data['progress'],
                    "auto_generate": node.node_data['auto_generate'],
                    "parent_id": node.objective_config.parent_id,
                    "position": {
                        "row": node.grid_position.row,
                        "column": node.grid_position.column
                    }
                }
                data["objectives"].append(obj_data)
            
            elif node.node_type == "hypothesis":
                # Export hypothesis
                hyp_data = {
                    "id": node.node_data['id'],
                    "text": node.node_data['text'],
                    "state": node.node_data['state'],
                    "confidence": node.node_data['confidence'],
                    "variables": node.hypothesis_config.variables,
                    "position": {
                        "row": node.grid_position.row,
                        "column": node.grid_position.column
                    },
                    "supporting_evidence": [],
                    "contradicting_evidence": []
                }
                
                # Add evidence
                for evidence in node.hypothesis_config.supporting_evidence:
                    ev_data = {
                        "id": evidence.id,
                        "type": evidence.type.value,
                        "description": evidence.description,
                        "confidence": evidence.confidence,
                        "source": evidence.source,
                        "notes": evidence.notes
                    }
                    hyp_data["supporting_evidence"].append(ev_data)
                
                for evidence in node.hypothesis_config.contradicting_evidence:
                    ev_data = {
                        "id": evidence.id,
                        "type": evidence.type.value,
                        "description": evidence.description,
                        "confidence": evidence.confidence,
                        "source": evidence.source,
                        "notes": evidence.notes
                    }
                    hyp_data["contradicting_evidence"].append(ev_data)
                
                data["hypotheses"].append(hyp_data)
        
        # Process connections
        for node in self.grid_scene.nodes_grid.values():
            # Process output ports
            for port in node.output_ports:
                for link in port.connected_links:
                    # Skip processed links
                    if hasattr(link, 'processed') and link.processed:
                        continue
                    
                    # Add connection data
                    conn_data = {
                        "source_id": link.start_node.node_data['id'],
                        "target_id": link.end_node.node_data['id'],
                        "type": link.connection_type.value
                    }
                    data["connections"].append(conn_data)
                    
                    # Mark as processed
                    link.processed = True
        
        # Clean up processed flag
        for node in self.grid_scene.nodes_grid.values():
            for port in node.output_ports + node.input_ports:
                for link in port.connected_links:
                    if hasattr(link, 'processed'):
                        delattr(link, 'processed')
        
        return data
        
    def export_to_protocol(self):
        """Export the plan to a research protocol format"""
        # Get all nodes
        nodes = list(self.grid_scene.nodes_grid.values())
        
        # Find all objectives
        objectives = [node for node in nodes if node.node_type == "objective"]
        
        # Find all hypotheses
        hypotheses = [node for node in nodes if node.node_type == "hypothesis"]
        
        # Build protocol
        protocol = {
            "title": "Research Protocol",
            "sections": {
                "background": "",
                "objectives": [],
                "hypotheses": [],
                "methods": "",
                "analysis": ""
            }
        }
        
        # Extract objectives
        for obj in objectives:
            protocol["sections"]["objectives"].append({
                "id": obj.node_data['id'],
                "text": obj.node_data['text'],
                "type": obj.node_data['type'],
                "description": obj.objective_config.description
            })
        
        # Extract hypotheses
        for hyp in hypotheses:
            hyp_data = {
                "id": hyp.node_data['id'],
                "text": hyp.node_data['text'],
                "state": hyp.node_data['state'],
                "variables": hyp.hypothesis_config.variables,
                "evidence": {
                    "supporting": [],
                    "contradicting": []
                }
            }
            
            # Add evidence
            for ev in hyp.hypothesis_config.supporting_evidence:
                hyp_data["evidence"]["supporting"].append({
                    "type": ev.type.value,
                    "description": ev.description,
                    "source": ev.source
                })
            
            for ev in hyp.hypothesis_config.contradicting_evidence:
                hyp_data["evidence"]["contradicting"].append({
                    "type": ev.type.value,
                    "description": ev.description,
                    "source": ev.source
                })
            
            protocol["sections"]["hypotheses"].append(hyp_data)
        
        # Create a basic background from all objectives
        if objectives:
            background = "This research aims to address the following objectives:\n\n"
            for i, obj in enumerate(objectives):
                if obj.node_data['type'] == "research_question":
                    background += f"{i+1}. {obj.node_data['text']}\n"
            
            # Add hypothesis summary
            if hypotheses:
                background += "\nThe primary hypotheses under investigation are:\n\n"
                for i, hyp in enumerate(hypotheses):
                    background += f"{i+1}. {hyp.node_data['text']}\n"
            
            protocol["sections"]["background"] = background
        
        # Create basic methods section
        methods = "The study will employ appropriate methods to test the stated hypotheses, "
        methods += "gathering data and analyzing results using rigorous techniques. "
        
        # Add method details based on hypotheses
        if hypotheses:
            methods += "The following approaches will be used for each hypothesis:\n\n"
            for i, hyp in enumerate(hypotheses):
                methods += f"For hypothesis {i+1}: "
                
                # Generate appropriate method based on hypothesis state
                if hyp.node_data['state'] == "testing":
                    methods += "Experimental design with appropriate controls and variables.\n"
                elif hyp.node_data['state'] == "validated" or hyp.node_data['state'] == "rejected":
                    methods += "Analysis of existing data and replication of previous findings.\n"
                else:
                    methods += "Development of experimental protocols and initial pilot testing.\n"
        
        protocol["sections"]["methods"] = methods
        
        # Create basic analysis section
        analysis = "Data will be analyzed using statistical methods appropriate for the hypotheses. "
        analysis += "Significance will be assessed at p<0.05 level. "
        
        # Add analysis details based on evidence
        for hyp in hypotheses:
            if hyp.hypothesis_config.supporting_evidence or hyp.hypothesis_config.contradicting_evidence:
                analysis += f"\n\nFor hypothesis '{hyp.node_data['text']}', the following evidence will be analyzed:\n"
                
                for ev in hyp.hypothesis_config.supporting_evidence:
                    analysis += f"- Supporting: {ev.description}\n"
                
                for ev in hyp.hypothesis_config.contradicting_evidence:
                    analysis += f"- Contradicting: {ev.description}\n"
        
        protocol["sections"]["analysis"] = analysis
        
        return protocol

    def get_hypotheses_for_objective(self, objective_id: str) -> List[HypothesisNode]:
        """Find all hypothesis nodes directly connected from a specific objective node."""
        connected_hypotheses = []
        objective_node = None

        # Find the objective node instance by ID
        for node in self.grid_scene.nodes_grid.values():
            if node.node_type == "objective" and node.node_data.get('id') == objective_id:
                objective_node = node
                break

        if not objective_node:
            print(f"Warning: Objective node with ID {objective_id} not found.")
            return []

        # Check connections originating from the objective node's output ports
        for port in objective_node.output_ports:
            for link in port.connected_links:
                # Ensure the link starts at this port and ends at a hypothesis node
                if link.start_port == port and link.end_node.node_type == "hypothesis":
                    connected_hypotheses.append(link.end_node)

        # Add hypotheses stored in objective_node.child_hypotheses if that attribute exists
        # This covers cases where connections might not be the only way they are linked
        if hasattr(objective_node, 'child_hypotheses'):
            for child_node in objective_node.child_hypotheses:
                 if child_node not in connected_hypotheses and child_node.node_type == "hypothesis":
                     connected_hypotheses.append(child_node)

        return connected_hypotheses
