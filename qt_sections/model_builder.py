import math
import sys
import json
import os
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import uuid
import asyncio

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QVBoxLayout,
    QGraphicsPathItem, QGraphicsTextItem, QPushButton, QWidget, QMenu, QDialog, 
    QLabel, QLineEdit, QHBoxLayout, QTextEdit, QComboBox, QGraphicsItem,
    QFileDialog, QMessageBox, QDialogButtonBox, QTableWidget, QTableWidgetItem, 
    QHeaderView, QAbstractItemView, QSpinBox, QScrollArea, QListWidget, QGraphicsRectItem,
    QGroupBox, QGraphicsEllipseItem, QStatusBar, QInputDialog, QToolBar
)

from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from PyQt6.QtGui import QPen, QBrush, QColor, QPainterPath, QPainter, QPixmap, QTransform,QFont,QCursor, QAction
from PyQt6.QtCore import (
    Qt, QPointF, QRectF, QLineF, QEvent, QTimer, 
    pyqtSignal as Signal, QVariantAnimation, QEasingCurve
)
from qasync import asyncSlot

from model_builder.config import (
    WorkspaceConstants, NodeCategory, PortType, NODE_CONFIGS, EdgeItem, PortItem, 
    WorkflowNode, EnhancedTextItem, parse_workflow_json, create_study_design_from_llm_json, validate_workflow_json
)
from model_builder.theme_support import ThemeManager, update_node_theme, update_edge_theme, update_port_theme
from PyQt6.QtWidgets import QGraphicsDropShadowEffect
from model_builder.builder_dialogs import (
    TargetPopulationDialog, EligiblePopulationDialog, InterventionDialog, 
    OutcomeDialog, RandomizationDialog, TimePointDialog
)
from PyQt6 import sip
# Import the icon loader
from helpers.load_icon import load_bootstrap_icon

# ========================
# WORKFLOW SCENE
# ======================== 

class WorkflowScene(QGraphicsScene):
    nodeActivated = Signal(object)  # Emitted when a node is activated
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Signal for when a node is activated (double-clicked)
        # self.nodeActivated = Signal(object)  # Remove this line as it's already defined at class level
        
        # Container for nodes
        self.target_population = None  # Reference to the target population node
        
        # Track if we're drawing an edge
        self.drawing_edge = False
        self.start_port = None
        self.temp_line = None  # Temporary line for edge drawing
        
        # Snap indicator for edge creation
        self.snap_indicator = None
        self.snap_animation = None
        
        # Edge tooltip
        self.edge_tooltip = None
        
        self.statusBar = None

        # Set scene rect
        self.setSceneRect(-2000, -2000, 4000, 4000)
        
        # Set rendering hints for smoother graphics
        self.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        
        # Apply theme (light by default)
        self.is_dark_theme = False
        self.update_theme_colors()
        
        # Track grid visibility
        self.show_grid = True
        
        self.workflow_data = {}  # Store workflow data
        
        # Create the initial patient group node
        # We no longer track timepoint nodes
        self.setup_initial_nodes()
        
        # Show a tooltip about edge creation after a short delay
        QTimer.singleShot(1000, self.show_edge_creation_tooltip)
        
        # Add snap distance for edge connection
        self.snap_distance = 50  # Distance in pixels for port snapping
        self.nearest_port = None  # Track the nearest port for snapping
        
        # For visual feedback during snapping
        self.snap_tooltip = None
        
        # Add connections text display at top right
        self.connections_text = QGraphicsTextItem()
        self.connections_text.setPos(-1950, -1950)  # Position at top right of the scene
        self.connections_text.setZValue(1000)  # Ensure it's above other items
        
        # Set text color based on theme
        if self.is_dark_theme:
            self.connections_text.setDefaultTextColor(QColor(220, 220, 220))
        else:
            self.connections_text.setDefaultTextColor(QColor(60, 60, 60))
            
        font = QFont("Arial", 10)
        self.connections_text.setFont(font)
        self.addItem(self.connections_text)
        self.update_connections_display()

    def update_theme_colors(self):
        """Update colors based on the current theme."""
        # Update scene background to match view background
        if self.is_dark_theme:
            background_color = QColor(30, 30, 30)  # Dark gray
        else:
            background_color = QColor(245, 245, 245)  # Light gray
        
        self.setBackgroundBrush(QBrush(background_color))
        
        # Update existing items if needed
        self.update_existing_items()
        
        # Update connections text display with theme colors
        if hasattr(self, 'connections_text') and self.connections_text:
            self.update_connections_display()
    
    def update_existing_items(self):
        """Update colors of existing items in the scene."""
        # Update all nodes
        for item in self.items():
            if isinstance(item, WorkflowNode):
                item.update_theme(self.is_dark_theme)
            elif isinstance(item, EdgeItem):
                # Update edge colors
                if hasattr(item, 'update_theme'):
                    item.update_theme(self.is_dark_theme)
            elif isinstance(item, PortItem):
                # Update port colors
                if hasattr(item, 'update_theme'):
                    item.update_theme(self.is_dark_theme)
        
        # Force a redraw
        self.update()
    
    def set_theme(self, is_dark):
        """Set the theme and update colors."""
        if self.is_dark_theme != is_dark:
            self.is_dark_theme = is_dark
            self.update_theme_colors()

    def setup_initial_nodes(self):
        """Create initial target population node for the workflow"""
        # Check if there's already a target_population node
        target_pop_exists = False
        for item in self.items():
            if isinstance(item, WorkflowNode) and hasattr(item, 'config') and item.config.category == NodeCategory.TARGET_POPULATION:
                self.target_population = item
                target_pop_exists = True
                break
                
        # Only create target population if none exists
        if not target_pop_exists:
            center_x = 100
            center_y = 100
            
            # Create a target population node
            target_population_config = NODE_CONFIGS[NodeCategory.TARGET_POPULATION]
            initial_node = self.add_node(NodeCategory.TARGET_POPULATION, QPointF(center_x, center_y))
            
            # Store reference to this node
            self.target_population = initial_node
            
            # Center the view on this node
            if hasattr(self, 'view') and self.view():
                self.view().centerOn(initial_node)

    def contextMenuEvent(self, event):
        """Handle context menu event to show add node menu."""
        # Get scene coordinates directly
        scene_pos = event.scenePos()
        
        # Create context menu
        context_menu = QMenu(self)
        
        # Add node actions
        add_node_menu = QMenu("Add Node", self)
        add_node_menu.setIcon(load_bootstrap_icon("node-plus"))
        
        # Add each node type to the menu with appropriate icons
        add_node_menu.addAction(load_bootstrap_icon("people-fill"), "Target Population").triggered.connect(
            lambda: self.add_node(NodeCategory.TARGET_POPULATION, scene_pos))
        add_node_menu.addAction(load_bootstrap_icon("funnel"), "Eligible Population").triggered.connect(
            lambda: self.add_node(NodeCategory.ELIGIBLE_POPULATION, scene_pos))
        add_node_menu.addAction(load_bootstrap_icon("capsule"), "Intervention").triggered.connect(
            lambda: self.add_node(NodeCategory.INTERVENTION, scene_pos))
        add_node_menu.addAction(load_bootstrap_icon("clipboard-data"), "Outcome").triggered.connect(
            lambda: self.add_node(NodeCategory.OUTCOME, scene_pos))
        add_node_menu.addAction(load_bootstrap_icon("person-lines-fill"), "Subgroup").triggered.connect(
            lambda: self.add_node(NodeCategory.SUBGROUP, scene_pos))
        add_node_menu.addAction(load_bootstrap_icon("tablet"), "Control Group").triggered.connect(
            lambda: self.add_node(NodeCategory.CONTROL, scene_pos))
        add_node_menu.addAction(load_bootstrap_icon("shuffle"), "Randomization").triggered.connect(
            lambda: self.add_node(NodeCategory.RANDOMIZATION, scene_pos))
        add_node_menu.addAction(load_bootstrap_icon("calendar-check"), "Timepoint").triggered.connect(
            lambda: self.add_node(NodeCategory.TIMEPOINT, scene_pos))
        
        context_menu.addMenu(add_node_menu)
        
        # Add view actions
        context_menu.addSeparator()
        context_menu.addAction(load_bootstrap_icon("zoom-out"), "Reset Zoom").triggered.connect(self.reset_zoom)
        context_menu.addAction(load_bootstrap_icon("fullscreen"), "Center View").triggered.connect(self.center_on_study_model)
        
        # Show menu - In PyQt6, we need to use event.globalPos()
        context_menu.exec(event.globalPos())
        
        # Accept the event
        event.accept()

    def add_node(self, category: NodeCategory, pos: QPointF):
        """Add a new node to the scene"""
        # Get node configuration
        config = NODE_CONFIGS.get(category)
        if not config:
            print(f"No configuration found for {category}")
            return None
            
        # Create node
        node = WorkflowNode(config, pos.x(), pos.y())
        self.addItem(node)
        
        # Connect node activated signal
        node.nodeActivated.connect(self.on_node_activated)
        
        # Handle patient group  
        if category == NodeCategory.TARGET_POPULATION:
            # If this is the first patient group, make it the main one
            if not self.target_population:
                self.target_population = node
                print(f"Set study model: {self.target_population}")
            else:
                print(f"Additional TARGET_POPULATION node (already have {len([n for n in self.items() if isinstance(n, WorkflowNode) and n.config.category == NodeCategory.TARGET_POPULATION])})")
                
        return node
        
    def split_patient_group(self, node, pos):
        """Split a patient group into subgroups"""
        # Create two subgroup nodes
        subgroup1 = self.add_node(NodeCategory.SUBGROUP, QPointF(pos.x() + 150, pos.y() - 100))
        subgroup1.display_name = "Subgroup A"
        subgroup1.config_details = "50% of original group"
        
        subgroup2 = self.add_node(NodeCategory.SUBGROUP, QPointF(pos.x() + 150, pos.y() + 100))
        subgroup2.display_name = "Subgroup B"
        subgroup2.config_details = "50% of original group"
        
        # Get the output port from the patient group
        start_port = None
        for port in node.output_ports:
            start_port = port
            break
        
        if start_port:
            # Create edges to subgroups
            if subgroup1.input_ports:
                edge1 = EdgeItem(start_port, subgroup1.input_ports[0], "patient_flow", 50)
                edge1.flow_data["label"] = "Subgroup A"
                self.addItem(edge1)
                
            if subgroup2.input_ports:
                edge2 = EdgeItem(start_port, subgroup2.input_ports[0], "patient_flow", 50)
                edge2.flow_data["label"] = "Subgroup B"
                self.addItem(edge2)

    def on_node_activated(self, node):
        """Handle node activation"""
        # Just display the node type in status bar
        if node:
            self.statusBar.showMessage(f"Selected: {node.config.category.value.replace('_', ' ').title()}")
        
        # Check if this node has an action handler
        if not hasattr(node.config, 'action_handler') or not node.config.action_handler:
            return
            
        # Get the action handler name
        action_handler = node.config.action_handler
        
        # Route to appropriate dialog based on node type
        if action_handler == "open_target_population":
            dialog = TargetPopulationDialog(None, node)
            dialog.exec()
        elif action_handler == "open_eligible_population":
            dialog = EligiblePopulationDialog(None, node)
            dialog.exec()
        elif action_handler == "open_intervention":
            dialog = InterventionDialog(None, node)
            dialog.exec()
        elif action_handler == "open_outcome":
            dialog = OutcomeDialog(None, node)
            dialog.exec()
        elif action_handler == "open_randomization":
            dialog = RandomizationDialog(None, node)
            dialog.exec()
        elif action_handler == "open_timepoint":
            dialog = TimePointDialog(None, node)
            dialog.exec()
        elif action_handler == "open_subgroup":
            # Use PatientGroupDialog for subgroups as they're similar
            dialog = EligiblePopulationDialog(None, node)
            dialog.exec()
        elif action_handler == "open_control_group":
            # Create a simpler dialog for control groups
            dialog = InterventionDialog(None, node)
            dialog.exec()
            
        # After dialog is closed, update the node
        node.update()
        
        # Emit the signal that a node was activated
        self.nodeActivated.emit(node)

    def mousePressEvent(self, event):
        """Handle mouse press events in the scene.
        Most logic is delegated to the proper handlers based on what was clicked.
        """
        pos = event.scenePos()
        
        # Check what was clicked
        clicked_item = self.itemAt(pos, QTransform())
        
        # Handle clicks on different types of items
        if isinstance(clicked_item, WorkflowNode):
            # Handle node clicks - ensure this node becomes the active one for dragging
            self.clearSelection()  # Clear any existing selection
            clicked_item.setSelected(True)  # Select only this node
            
            # Let the node handle the event directly
            # This is important for proper dragging behavior
            clicked_item.mousePressEvent(event)
            
            # Don't call super().mousePressEvent() here to avoid double processing
            return
            
        elif isinstance(clicked_item, PortItem):
            port = clicked_item
            parent_node = port.parentItem()
            
            # For ports with specific labels, create a new node
            if hasattr(port, 'label') and port.label.lower() in ["eligible population", "randomization", 
                                                                "subgroup", "outcome", "control", 
                                                                "intervention", "timepoint"]:
                # Calculate position for new node (offset from current port)
                port_pos = port.mapToScene(QPointF(0, 0))
                new_node_pos = QPointF(port_pos.x() + 200, port_pos.y())
                
                # Create a new node based on the port label
                new_node = self.create_node_from_port(port, new_node_pos)
                if new_node:
                    print(f"Created new node from port label: {port.label}")
                return
            
            # Start drawing a connection line
            self.start_port = port
            self.drawing_edge = True
                
            # Create temporary line for feedback
            if self.temp_line:
                self.removeItem(self.temp_line)
            
            self.temp_line = QGraphicsPathItem()
            self.temp_line.setPen(QPen(QColor(33, 150, 243, 180), 2, Qt.PenStyle.DashLine))
            self.addItem(self.temp_line)
            
            # Show tooltip with instructions
            self.show_edge_creation_tooltip()
            
            # Show snap indicator for the start port
            self.show_snap_indicator(port)
            
            # Accept the event
            event.accept()
            return
            
        # For other cases, use the default handler
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse movement for edge drawing"""
        # First, call parent implementation for normal scene behavior
        super().mouseMoveEvent(event)
        
        # Then handle edge drawing if active
        if self.drawing_edge and self.start_port and self.temp_line:
            # Get the start position from the start port
            start_pos = self.start_port.mapToScene(QPointF(0, 0))
            
            # Create a path for the temporary line
            path = QPainterPath(start_pos)
            
            # Calculate control points for a more natural curve
            end_pos = event.scenePos()
            
            # If we have a nearest port for snapping, use its position instead
            if self.nearest_port:
                end_pos = self.nearest_port.mapToScene(QPointF(0, 0))
                
                # Update the temporary line style to indicate snapping
                self.temp_line.setPen(QPen(QColor(76, 175, 80, 200), 3, Qt.PenStyle.DashLine))
                
                # Show snap indicator if not already shown
                self.show_snap_indicator(self.nearest_port)
            else:
                # Reset to normal style when not snapping
                self.temp_line.setPen(QPen(QColor(33, 150, 243, 180), 2, Qt.PenStyle.DashLine))
                
                # Hide snap indicator
                self.hide_snap_indicator()
            
            # Calculate the direction vector
            dx = end_pos.x() - start_pos.x()
            dy = end_pos.y() - start_pos.y()
            
            # Determine if the connection is mostly horizontal or vertical
            is_horizontal = abs(dx) > abs(dy)
            
            # Calculate control points for a more pronounced curve
            if is_horizontal:
                # For horizontal connections, use more curved control points
                ctrl_point1 = QPointF(
                    start_pos.x() + dx * 0.3,  # 30% of the way horizontally
                    start_pos.y() + dy * 0.1  # Less vertical curve
                )
                
                ctrl_point2 = QPointF(
                    end_pos.x() - dx * 0.3,    # 30% back from the end horizontally
                    end_pos.y() - dy * 0.1  # Less vertical curve
                )
            else:
                # For vertical connections, use more curved control points
                ctrl_point1 = QPointF(
                    start_pos.x() + dx * 0.1,  # Less horizontal curve
                    start_pos.y() + dy * 0.3   # 30% of the way vertically
                )
                
                ctrl_point2 = QPointF(
                    end_pos.x() - dx * 0.1,  # Less horizontal curve
                    end_pos.y() - dy * 0.3     # 30% back from the end vertically
                )
            
            # Create a cubic Bezier curve
            path.cubicTo(ctrl_point1, ctrl_point2, end_pos)
            
            # Update the path
            self.temp_line.setPath(path)
            
            # Continuously update potential connection ports
            self.highlight_potential_connection_ports(event.scenePos())

    def mouseReleaseEvent(self, event):
        """Handle mouse release for edge creation"""
        if self.drawing_edge and self.start_port and self.temp_line:
            end_port = None
            
            # If we have a nearest port for snapping, use it
            if self.nearest_port:
                end_port = self.nearest_port
            else:
                # Otherwise, find the item at the release position
                end_item = self.itemAt(event.scenePos(), QTransform())
                
                # Check if it's a port and compatible with the start port
                if isinstance(end_item, PortItem) and end_item != self.start_port:
                    if self.is_compatible_connection(self.start_port, end_item):
                        end_port = end_item
            
            # Create an edge if we have a valid end port
            if end_port and end_port != self.start_port:
                # Create an edge between the ports using our helper method
                edge = self.create_edge(self.start_port, end_port)
                
                if edge:
                # Flash effect to indicate success
                    end_port.setBrush(QBrush(QColor(76, 175, 80)))  # Green for success
                    QTimer.singleShot(300, lambda: end_port.setBrush(end_port.default_brush))
                
                    print(f"Edge created from {self.start_port.port_type} to {end_port.port_type}")
                else:
                    # Flash effect to indicate failure (edge creation failed)
                    end_port.setBrush(QBrush(QColor(244, 67, 54)))  # Red for error
                    QTimer.singleShot(300, lambda: end_port.setBrush(end_port.default_brush))
            else:
                # Show error feedback if connection attempt was invalid
                if self.start_port is not None:
                    self.start_port.setBrush(QBrush(QColor(244, 67, 54)))  # Red for error
                    QTimer.singleShot(300, lambda: self.start_port.setBrush(self.start_port.default_brush) 
                                     if self.start_port is not None else None)
            
            # Clean up
            if self.temp_line is not None:
                self.removeItem(self.temp_line)
                self.temp_line = None
            self.drawing_edge = False
            self.start_port = None
            self.nearest_port = None
            
            # Also reset highlighting on all ports
            self.reset_port_highlights()
            
            # Hide snap indicator if showing
            self.hide_snap_indicator()
            
            # Remove tooltip
            self.remove_edge_tooltip()
            
            # Accept the event
            event.accept()
            return
            
        # For other cases, use the default handler
            super().mouseReleaseEvent(event)

    def is_compatible_connection(self, start_port, end_port):
        """Check if two ports can be connected to each other."""
        if not start_port or not end_port:
            return False
            
        # Can't connect a port to itself
        if start_port == end_port:
            return False
            
        # Can't connect two ports from the same node
        if start_port.parentItem() == end_port.parentItem():
            return False
            
        # Get port types
        try:
            start_port_type = start_port.port_type if hasattr(start_port, 'port_type') else None
            end_port_type = end_port.port_type if hasattr(end_port, 'port_type') else None
            
            # Only output ports can connect to input ports
            # This is the most common case - output to input
            if start_port_type == "output" and end_port_type == "input":
                return True
                
            # In some workflows (like those created programmatically), we sometimes need 
            # to allow input to output for specialized connections
            if start_port_type == "input" and end_port_type == "output" and hasattr(start_port, 'is_bidirectional') and start_port.is_bidirectional:
                return True
                
            # All other combinations are not allowed
            return False
        except Exception as e:
            print(f"Error checking port compatibility: {e}")
            # If we can't determine port types, be cautious and disallow the connection
            return False

    def highlight_potential_connection_ports(self, pos):
        """Highlight ports that can be connected to the start port and find the nearest one for snapping"""
        # Reset all port highlights first
        self.reset_port_highlights()
        
        # If we're not drawing an edge, return
        if not self.drawing_edge or not self.start_port:
            return
            
        # Don't highlight anything if we're starting from a plus button
        start_node = self.start_port.parentItem()
        if start_node and hasattr(start_node, 'plus_button') and self.start_port == start_node.plus_button:
            return
        
        # Reset nearest port
        self.nearest_port = None
        nearest_distance = float('inf')
        
        # Find all ports in the scene
        for item in self.items():
            if isinstance(item, PortItem) and item != self.start_port:
                # Skip plus buttons
                end_node = item.parentItem()
                if end_node and hasattr(end_node, 'plus_button') and item == end_node.plus_button:
                    continue
                
                # Check if this port is compatible with the start port
                if not self.is_compatible_connection(self.start_port, item):
                    continue
                    
                # Calculate distance to this port
                port_pos = item.mapToScene(QPointF(0, 0))
                dx = port_pos.x() - pos.x()
                dy = port_pos.y() - pos.y()
                distance = math.sqrt(dx*dx + dy*dy)
                
                # If within snap distance, check if it's the nearest
                if distance <= self.snap_distance and distance < nearest_distance:
                    nearest_distance = distance
                    self.nearest_port = item
        
        # Highlight the nearest port if found
        if self.nearest_port:
            self.nearest_port.potential_connection = True
            
            # Use theme-aware colors for highlighting
            highlight_color = ThemeManager.get_color("port_highlight", self.is_dark_theme)
            highlight_fill = ThemeManager.get_color("port_highlight_fill", self.is_dark_theme)
            self.nearest_port.setBrush(QBrush(highlight_fill))
            
            # Add a subtle glow effect
            glow = QGraphicsDropShadowEffect()
            glow.setOffset(0, 0)
            glow.setBlurRadius(15)
            glow.setColor(highlight_color.lighter(120))  # Slightly lighter version of highlight color
            self.nearest_port.setGraphicsEffect(glow)
            
            self.nearest_port.update()
    
    def reset_port_highlights(self):
        """Reset all port highlights"""
        for item in self.items():
            if isinstance(item, PortItem) and item.potential_connection:
                item.potential_connection = False
                # Use theme-aware colors
                fill_color = ThemeManager.get_color("port_fill", self.is_dark_theme)
                item.setBrush(QBrush(fill_color))
                item.setGraphicsEffect(None)  # Remove any glow effect
                item.update()

    def highlight_compatible_ports(self):
        """Highlight all compatible ports for potential connections after deletion"""
        # Get all ports in the scene
        ports = [item for item in self.items() if isinstance(item, PortItem)]
        
        # For each port, find compatible ports and highlight them
        for start_port in ports:
            # Skip plus buttons and add ports
            start_node = start_port.parentItem()
            if (start_node and hasattr(start_node, 'plus_button') and start_port == start_node.plus_button) or \
               start_port.port_type == "add":
                continue
                
            # Only consider output ports as starting points
            if start_port.port_type != "output":
                continue
                
            # Find compatible end ports
            for end_port in ports:
                # Skip the same port and incompatible ports
                if end_port == start_port or not self.is_compatible_connection(start_port, end_port):
                    continue
                    
                # Check if already connected
                already_connected = False
                for item in self.items():
                    if isinstance(item, EdgeItem) and item.start_port == start_port and item.end_port == end_port:
                        already_connected = True
                        break
                        
                if already_connected:
                    continue
                    
                # Highlight the compatible port
                end_port.potential_connection = True
                
                # Use theme-aware colors for highlighting
                highlight_color = ThemeManager.get_color("port_highlight", self.is_dark_theme)
                highlight_fill = ThemeManager.get_color("port_highlight_fill", self.is_dark_theme)
                end_port.setBrush(QBrush(highlight_fill))
                
                # Add a subtle glow effect
                glow = QGraphicsDropShadowEffect()
                glow.setOffset(0, 0)
                glow.setBlurRadius(15)
                glow.setColor(highlight_color.lighter(120))  # Slightly lighter version of highlight color
                end_port.setGraphicsEffect(glow)
                end_port.update()
        
        # Set a timer to reset the highlights after a few seconds
        QTimer.singleShot(3000, self.reset_port_highlights)

    def reset_workflow_states(self):
        """Reset all nodes in the workflow to inactive state."""
        # Clear all node states
        for item in self.items():
            if isinstance(item, WorkflowNode):
                item.set_active(False)
                
        # Reset all edge colors to default
        for item in self.items():
            if isinstance(item, EdgeItem):
                item.setActionColor("default")
                
        # Set patient group as active
        if self.target_population:
            self.target_population.set_active(True)

    def find_connected_nodes(self, target_node, port_type, node_category=None):
        """Find all nodes of a specific category connected to the target node"""
        connected_nodes = []
        
        for item in self.items():
            if isinstance(item, EdgeItem):
                start_node = item.start_port.parentItem()
                end_node = item.end_port.parentItem()
                
                if port_type == PortType.INPUT and end_node == target_node:
                    if node_category is None or start_node.config.category == node_category:
                        connected_nodes.append(start_node)
                        
                elif port_type == PortType.OUTPUT and start_node == target_node:
                    if node_category is None or end_node.config.category == node_category:
                        connected_nodes.append(end_node)
                        
        return connected_nodes

    def get_workflow_edges(self):
        """Return all edges in the workflow"""
        return [item for item in self.items() if isinstance(item, EdgeItem)]

    def save_to_json(self):
        """Export workflow to JSON format"""
        workflow_data = {
            "nodes": [],
            "edges": []
        }
        
        # Save nodes
        for item in self.items():
            if isinstance(item, WorkflowNode):
                node_data = {
                    "id": id(item),
                    "category": item.config.category.value,
                    "x": item.x(),
                    "y": item.y(),
                    "expanded": item.expanded,
                    "details": item.config_details,
                    "is_active": item.is_active
                }
                workflow_data["nodes"].append(node_data)
                
        # Save edges
        for item in self.items():
            if isinstance(item, EdgeItem):
                # Get parent node IDs
                start_node = item.start_port.parentItem()
                end_node = item.end_port.parentItem()
                
                if start_node and end_node:
                    # Get port indices
                    start_port_idx = -1
                    end_port_idx = -1
                    
                    if item.start_port.port_type == "output":
                        start_port_idx = start_node.output_ports.index(item.start_port)
                    else:
                        start_port_idx = start_node.input_ports.index(item.start_port)
                        
                    if item.end_port.port_type == "input":
                        end_port_idx = end_node.input_ports.index(item.end_port)
                    else:
                        end_port_idx = end_node.output_ports.index(item.end_port)
                    
                    edge_data = {
                        "start_node": id(start_node),
                        "end_node": id(end_node),
                        "start_port_type": item.start_port.port_type,
                        "end_port_type": item.end_port.port_type,
                        "start_port_idx": start_port_idx,
                        "end_port_idx": end_port_idx,
                        "action": item.action,
                        "flow_data": item.flow_data
                    }
                    workflow_data["edges"].append(edge_data)
        
        return workflow_data
                
    def load_from_json(self, workflow_data):
        """Load workflow from JSON data"""
        try:
            # Clear existing items
            self.clear()
            
            # Reset target_population reference
            self.target_population = None
            
            # Add connections text display back after clearing
            try:
                self.connections_text = QGraphicsTextItem()
                self.connections_text.setPos(-1950, -1950)
                self.connections_text.setZValue(1000)
                
                # Set text color based on theme
                if self.is_dark_theme:
                    self.connections_text.setDefaultTextColor(QColor(220, 220, 220))
                else:
                    self.connections_text.setDefaultTextColor(QColor(60, 60, 60))
                    
                font = QFont("Arial", 10)
                self.connections_text.setFont(font)
                self.addItem(self.connections_text)
            except Exception as e:
                print(f"Error creating connections text: {e}")
                # Continue without connections text if it fails
                self.connections_text = None
            
            # Store node references by ID for edge creation
            node_references = {}
            
            # Create nodes
            for node_data in workflow_data.get("nodes", []):
                category_str = node_data.get("category")
                try:
                    category = NodeCategory(category_str)
                    
                    # Get node config
                    config = NODE_CONFIGS.get(category)
                    if not config:
                        print(f"Warning: No config found for category {category_str}")
                        continue
                    
                    # Create node
                    x = node_data.get("x", 0)
                    y = node_data.get("y", 0)
                    node = WorkflowNode(config, x, y)
                    
                    # Set expanded state
                    if node_data.get("expanded", False):
                        node.toggle_expand()
                    
                    # Set active state
                    if node_data.get("is_active", False):
                        node.set_active(True)
                    
                    # Set details if available
                    details = node_data.get("details", {})
                    if details:
                        node.config_details = details
                        
                        # Update node title if available and title_item exists
                        if "title" in details and hasattr(node, "title_item") and node.title_item:
                            try:
                                node.title_item.setPlainText(details["title"])
                            except Exception as title_error:
                                print(f"Error setting title text: {title_error}")
                        
                        # Update description if available and description_item exists
                        if "description" in details and hasattr(node, "description_item") and node.description_item:
                            try:
                                node.description_item.setPlainText(details["description"])
                            except Exception as desc_error:
                                print(f"Error setting description text: {desc_error}")
                    
                    self.addItem(node)
                    
                    # Store reference by ID
                    node_references[node_data.get("id")] = node
                    
                    # Store target population reference if applicable
                    if category == NodeCategory.TARGET_POPULATION:
                        if self.target_population is None:
                            self.target_population = node
                            print(f"Set target population: {id(node)}")
                        else:
                            print(f"Additional TARGET_POPULATION node (already have {len([n for n in self.items() if isinstance(n, WorkflowNode) and n.config.category == NodeCategory.TARGET_POPULATION])})")
                    
                except (ValueError, KeyError, RuntimeError) as e:
                    print(f"Error creating node: {e}")
            
            # If we didn't find any target population nodes, check the scene for one
            if self.target_population is None:
                for item in self.items():
                    if isinstance(item, WorkflowNode) and hasattr(item, 'config') and item.config.category == NodeCategory.TARGET_POPULATION:
                        self.target_population = item
                        print(f"Found existing target population: {id(item)}")
                        break
            
            # Continue with creating edges, but wrapped in try/except
            try:
                # Create edges
                for edge_data in workflow_data.get("edges", []):
                    try:
                        # Get node references
                        start_node_id = edge_data.get("start_node")
                        end_node_id = edge_data.get("end_node")
                        
                        start_node = node_references.get(start_node_id)
                        end_node = node_references.get(end_node_id)
                        
                        if not start_node or not end_node:
                            print(f"Warning: Could not find nodes for edge {start_node_id} -> {end_node_id}")
                            continue
                        
                        # Get port types and indices
                        start_port_type = edge_data.get("start_port_type")
                        end_port_type = edge_data.get("end_port_type")
                        start_port_idx = edge_data.get("start_port_idx", 0)
                        end_port_idx = edge_data.get("end_port_idx", 0)
                        
                        # Get port references
                        start_port = None
                        end_port = None
                        
                        if start_port_type == "output":
                            if 0 <= start_port_idx < len(start_node.output_ports):
                                start_port = start_node.output_ports[start_port_idx]
                        else:
                            if 0 <= start_port_idx < len(start_node.input_ports):
                                start_port = start_node.input_ports[start_port_idx]
                        
                        if end_port_type == "input":
                            if 0 <= end_port_idx < len(end_node.input_ports):
                                end_port = end_node.input_ports[end_port_idx]
                        else:
                            if 0 <= end_port_idx < len(end_node.output_ports):
                                end_port = end_node.output_ports[end_port_idx]
                        
                        if not start_port or not end_port:
                            print(f"Warning: Could not find ports for edge")
                            continue
                        
                        # Create edge with action and flow data if available
                        action = edge_data.get("action", "default")
                        flow_data = edge_data.get("flow_data", {})
                        patient_count = flow_data.get("patient_count", 100) if flow_data else 100
                        
                        edge = EdgeItem(start_port, end_port, action, patient_count)
                        if flow_data:
                            edge.flow_data = flow_data
                        
                        self.addItem(edge)
                        
                        # Set edge properties
                        try:
                            edge.setPatientCount(patient_count)
                            if edge_data.get("flow_data", {}).get("label"):
                                edge.flow_data["label"] = edge_data.get("flow_data", {}).get("label")
                            
                            # Print success message
                            source_category = start_node.config.category.value if hasattr(start_node, 'config') else "Unknown"
                            target_category = end_node.config.category.value if hasattr(end_node, 'config') else "Unknown"
                            print(f"✅ Created edge {start_node_id}({source_category}) -> {end_node_id}({target_category})")
                            
                            # Track subgroup patients for validation
                            if start_node.config.category == NodeCategory.SUBGROUP:
                                if 'total_subgroup_patients' not in locals():
                                    total_subgroup_patients = 0
                                total_subgroup_patients += patient_count
                            
                            # If edge is to a target node that is an OUTCOME, track for validation
                            if end_node.config.category == NodeCategory.OUTCOME:
                                if 'node_patient_counts' not in locals():
                                    node_patient_counts = {}
                                if "outcomes" not in node_patient_counts:
                                    node_patient_counts["outcomes"] = {}
                                
                                source_name = start_node_id
                                if "outcomes" not in node_patient_counts["outcomes"]:
                                    node_patient_counts["outcomes"][source_name] = 0
                                
                                node_patient_counts["outcomes"][source_name] += patient_count
                        except Exception as e:
                            print(f"Error setting edge properties: {e}")
                            # Continue with the next edge even if setting properties fails
                    except Exception as e:
                        print(f"Error creating edge: {e}")
            except Exception as edge_error:
                print(f"Error creating edges: {edge_error}")
                
            # Update connections display at the end
            try:
                self.update_connections_display()
            except Exception as conn_error:
                print(f"Error updating connections display: {conn_error}")
                
            # Verify target_population is valid before returning
            if self.target_population and not self.target_population.scene():
                print("Warning: target_population was invalidated during loading, resetting reference")
                self.target_population = None
                
                # Try to find a valid target population
                for item in self.items():
                    if isinstance(item, WorkflowNode) and hasattr(item, 'config') and item.config.category == NodeCategory.TARGET_POPULATION:
                        self.target_population = item
                        print(f"Found replacement target population: {id(item)}")
                        break
                
            return True
        except Exception as e:
            print(f"Error in load_from_json: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_from_workflow_json(self, workflow_json):
        """
        Load workflow from a simplified JSON structure designed for dynamic graph creation.
        
        Example JSON format:
        {
            "design_type": "parallel_group", 
            "nodes": [
                {
                    "id": "population",
                    "type": "target_population",
                    "label": "Initial Population",
                    "x": 0,
                    "y": 0,
                    "patient_count": 1000
                },
                {
                    "id": "eligible",
                    "type": "eligible_population",
                    "label": "Eligible Subjects",
                    "x": 300,
                    "y": 0,
                    "patient_count": 500
                },
                ...
            ],
            "edges": [
                {
                    "source": "population",
                    "target": "eligible",
                    "patient_count": 500,
                    "label": "Screening"
                },
                ...
            ]
        }
        
        Returns:
            bool: True if successful, False otherwise
            dict: Validation results with warnings/errors if any
        """
        validation = {"valid": True, "warnings": [], "errors": []}
        
        # Print debug info about the input
        print(f"Loading workflow with {len(workflow_json.get('nodes', []))} nodes and {len(workflow_json.get('edges', []))} edges")
        
        try:
            # Clear existing items
            self.clear()
            
            # Reset target_population reference to avoid duplicates
            self.target_population = None
            
            # Add connections text display back after clearing
            self.connections_text = QGraphicsTextItem()
            self.connections_text.setPos(-1950, -1950)
            self.connections_text.setZValue(1000)
            
            # Set text color based on theme
            if self.is_dark_theme:
                self.connections_text.setDefaultTextColor(QColor(220, 220, 220))
            else:
                self.connections_text.setDefaultTextColor(QColor(60, 60, 60))
                
            font = QFont("Arial", 10)
            self.connections_text.setFont(font)
            self.addItem(self.connections_text)
            
            # Basic validation
            if not isinstance(workflow_json, dict):
                validation["valid"] = False
                validation["errors"].append("Workflow JSON must be a dictionary")
                return False, validation
                
            if "nodes" not in workflow_json or not isinstance(workflow_json["nodes"], list):
                validation["valid"] = False
                validation["errors"].append("Workflow JSON must contain a 'nodes' list")
                return False, validation
                
            if "edges" not in workflow_json or not isinstance(workflow_json["edges"], list):
                validation["valid"] = False
                validation["errors"].append("Workflow JSON must contain an 'edges' list")
                return False, validation
            
            # Store node references by ID for edge creation
            node_references = {}
            node_patient_counts = {}
            
            # First pass: Create all nodes
            print("\n--- Creating nodes ---")
            for node_data in workflow_json["nodes"]:
                try:
                    node_id = node_data.get("id")
                    if not node_id:
                        validation["warnings"].append("Node missing ID, generated random ID")
                        node_id = f"node_{len(node_references)}"
                    
                    # Get node type and convert to NodeCategory
                    node_type = node_data.get("type")
                    if not node_type:
                        validation["warnings"].append(f"Node {node_id} missing type, skipping")
                        continue
                    
                    try:
                        category = NodeCategory(node_type)
                    except ValueError:
                        validation["warnings"].append(f"Invalid node type: {node_type}, skipping")
                        continue
                    
                    # Get node config
                    config = NODE_CONFIGS.get(category)
                    if not config:
                        validation["warnings"].append(f"No config found for category {node_type}, skipping")
                        continue
                    
                    # Create node with position
                    x = node_data.get("x", 0)
                    y = node_data.get("y", 0)
                    node = WorkflowNode(config, x, y)
                    
                    # Set node label/display name if provided
                    label = node_data.get("label")
                    if label and hasattr(node, "title_item"):
                        node.display_name = label
                        if hasattr(node.title_item, "setHtml"):
                            title_html = f'<div style="text-align: center; font-weight: bold; font-family: Arial; font-size: 11pt; color: #505050;">{label}</div>'
                            node.title_item.setHtml(title_html)
                        elif hasattr(node.title_item, "setPlainText"):
                            node.title_item.setPlainText(label)
                    
                    # Set description if provided
                    description = node_data.get("description")
                    if description and hasattr(node, "desc_item"):
                        desc_html = f'<div style="text-align: center; font-family: Arial; font-size: 9pt; color: #505050;">{description}</div>'
                        node.desc_item.setHtml(desc_html)
                    
                    # Track patient count for validation
                    patient_count = node_data.get("patient_count", 0)
                    node_patient_counts[node_id] = patient_count
                    
                    self.addItem(node)
                    
                    # Store reference by ID
                    node_references[node_id] = node
                    print(f"✅ Created node {node_id} of type {category.value}")
                    
                    # Store target population reference if applicable
                    if category == NodeCategory.TARGET_POPULATION:
                        self.target_population = node
                        print(f"Set target population node: {node_id}")
                except Exception as e:
                    print(f"Error creating node {node_data.get('id')}: {e}")
                    validation["warnings"].append(f"Error creating node {node_data.get('id')}: {str(e)}")
            
            # If no target population was created, create one
            if not self.target_population and node_references:
                validation["warnings"].append("No target_population node found, creating default")
                config = NODE_CONFIGS.get(NodeCategory.TARGET_POPULATION)
                self.target_population = WorkflowNode(config, 0, 0)
                self.addItem(self.target_population)
                new_id = "default_population"
                node_references[new_id] = self.target_population
                print(f"Created default target population with ID {new_id}")
            
            # Track for subgroup validation
            subgroups = [n for n in node_references.values() if n.config.category == NodeCategory.SUBGROUP]
            output_nodes = [n for n in node_references.values() if n.config.category in [NodeCategory.OUTCOME, NodeCategory.TIMEPOINT]]
            total_subgroup_patients = 0
            
            # Second pass: Create edges
            print("\n--- Creating edges ---")
            for edge_data in workflow_json["edges"]:
                try:
                    source_id = edge_data.get("source")
                    target_id = edge_data.get("target")
                    
                    if not source_id or not target_id:
                        validation["warnings"].append(f"Edge missing source or target, skipping")
                        continue
                    
                    source_node = node_references.get(source_id)
                    target_node = node_references.get(target_id)
                    
                    if not source_node or not target_node:
                        validation["warnings"].append(f"Could not find nodes for edge {source_id} -> {target_id}, skipping")
                        continue
                    
                    print(f"Creating edge: {source_id} -> {target_id}")
                    
                    # Find appropriate ports
                    start_port = None
                    end_port = None
                    
                    # Find the appropriate output port from source node
                    target_type = target_node.config.category.value
                    for port in source_node.output_ports:
                        port_label = getattr(port, 'label', '').lower() if hasattr(port, 'label') else ''
                        
                        # If port has a label that matches target type or a simplified version of it
                        if port_label and (port_label == target_type.lower() or 
                                          port_label == target_type.lower().replace('_', ' ') or
                                          port_label == target_type.lower().split('_')[-1]):
                            start_port = port
                            break
                    
                    # If no specific port found, use first output port
                    if not start_port and source_node.output_ports:
                        start_port = source_node.output_ports[0]
                    
                    # Try to find target input port - this is usually the first input port
                    if target_node.input_ports:
                        end_port = target_node.input_ports[0]
                    
                    if not start_port or not end_port:
                        port_count_debug = f"(Source output ports: {len(source_node.output_ports)}, Target input ports: {len(target_node.input_ports)})"
                        validation["warnings"].append(f"Could not find appropriate ports for edge {source_id} -> {target_id} {port_count_debug}, skipping")
                        continue
                    
                    # Get patient count and label
                    patient_count = edge_data.get("patient_count", 100)
                    label = edge_data.get("label", "")
                    
                    # Try to create edge with a fallback approach if the first attempt fails
                    edge = self.create_edge(start_port, end_port)
                    
                    # If direct connection failed and we have more output ports, try each one
                    attempts = 1
                    while not edge and attempts < len(source_node.output_ports):
                        print(f"Attempt {attempts+1}: Trying different output port for {source_id} -> {target_id}")
                        start_port = source_node.output_ports[attempts]
                        edge = self.create_edge(start_port, end_port)
                        attempts += 1
                    
                    # If still failed and we have more input ports, try each input port
                    attempts = 1
                    while not edge and attempts < len(target_node.input_ports):
                        print(f"Attempt {attempts+1}: Trying different input port for {source_id} -> {target_id}")
                        # Reset to first output port
                        start_port = source_node.output_ports[0]
                        end_port = target_node.input_ports[attempts]
                        edge = self.create_edge(start_port, end_port)
                        attempts += 1
                        
                    if not edge:
                        # Print detailed debug info to diagnose the connection issue
                        source_category = source_node.config.category.value if hasattr(source_node, 'config') else "Unknown"
                        target_category = target_node.config.category.value if hasattr(target_node, 'config') else "Unknown"
                        print(f"⚠️ Failed to create edge {source_id}({source_category}) -> {target_id}({target_category}) after all attempts")
                        print(f"   Start port: {start_port}, type: {start_port.port_type if hasattr(start_port, 'port_type') else 'Unknown'}")
                        print(f"   End port: {end_port}, type: {end_port.port_type if hasattr(end_port, 'port_type') else 'Unknown'}")
                        validation["warnings"].append(f"Failed to create edge {source_id} -> {target_id}, incompatible connection")
                        continue
                    
                    # Set edge properties
                    try:
                        edge.setPatientCount(patient_count)
                        if label:
                            edge.flow_data["label"] = label
                        
                        # Print success message
                        source_category = source_node.config.category.value if hasattr(source_node, 'config') else "Unknown"
                        target_category = target_node.config.category.value if hasattr(target_node, 'config') else "Unknown"
                        print(f"✅ Created edge {source_id}({source_category}) -> {target_id}({target_category})")
                        
                        # Track subgroup patients for validation
                        if source_node.config.category == NodeCategory.SUBGROUP:
                            if 'total_subgroup_patients' not in locals():
                                total_subgroup_patients = 0
                            total_subgroup_patients += patient_count
                        
                        # If edge is to a target node that is an OUTCOME, track for validation
                        if target_node.config.category == NodeCategory.OUTCOME:
                            if 'node_patient_counts' not in locals():
                                node_patient_counts = {}
                            if "outcomes" not in node_patient_counts:
                                node_patient_counts["outcomes"] = {}
                            
                            source_name = source_id
                            if "outcomes" not in node_patient_counts["outcomes"]:
                                node_patient_counts["outcomes"][source_name] = 0
                            
                            node_patient_counts["outcomes"][source_name] += patient_count
                    except Exception as e:
                        print(f"Error setting edge properties: {e}")
                        # Continue with the next edge even if setting properties fails
                except Exception as e:
                    print(f"Error processing edge {edge_data}: {e}")
                    validation["warnings"].append(f"Error processing edge: {str(e)}")
            
            # Perform validation checks
            print("\n--- Validating workflow ---")
            
            # 1. Validate subgroup patient counts (wide data format)
            eligible_population = next((n for n in node_references.values() 
                                      if n.config.category == NodeCategory.ELIGIBLE_POPULATION), None)
            
            if len(subgroups) > 1 and eligible_population:
                # Get eligible population patient count from edges
                eligible_count = 0
                for item in self.items():
                    if isinstance(item, EdgeItem) and item.end_port.parentItem() == eligible_population:
                        eligible_count = item.patient_count
                        break
                
                # Check if sum of subgroups equals eligible population (wide data)
                if eligible_count > 0 and abs(total_subgroup_patients - eligible_count) > 0.001:
                    validation["warnings"].append(f"With multiple subgroups, total patient count ({total_subgroup_patients}) " +
                                               f"should match eligible population ({eligible_count})")
            
            # 2. Validate pre-post design (long data format) - more measurements than patients
            if len(subgroups) == 1 and output_nodes:
                # Check if it's likely a pre-post design
                timepoints = [n for n in node_references.values() if n.config.category == NodeCategory.TIMEPOINT]
                if len(timepoints) >= 2:
                    # Count total measurements
                    total_measurements = 0
                    for item in self.items():
                        if isinstance(item, EdgeItem) and item.end_port.parentItem() in timepoints:
                            total_measurements += item.patient_count
                    
                    # Get subgroup size
                    subgroup_size = 0
                    for item in self.items():
                        if isinstance(item, EdgeItem) and item.end_port.parentItem() == subgroups[0]:
                            subgroup_size = item.patient_count
                            break
                    
                    # In long data format, should have more measurements than subjects
                    if subgroup_size > 0 and total_measurements <= subgroup_size:
                        validation["warnings"].append(f"In a pre-post design, total measurements ({total_measurements}) " +
                                                   f"should exceed number of subjects ({subgroup_size})")
            
            # Update the connections display
            print("\n--- Updating display ---")
            self.update_connections_display()
            
            # Check if any nodes were created
            if not node_references:
                validation["warnings"].append("No nodes were created from the workflow data")
                
            print(f"Workflow loaded: {len(node_references)} nodes, {len(self.get_workflow_edges())} edges")
            return True, validation
            
        except Exception as e:
            validation["valid"] = False
            validation["errors"].append(f"Error loading workflow: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, validation

    def show_edge_creation_tooltip(self):
        """Show a tooltip with instructions for edge creation"""
        # If we already have a tooltip, remove it
        self.remove_edge_tooltip()
        
        # Create a new tooltip
        tooltip_text = "Drag to connect to another node"
        self.edge_tooltip = QGraphicsTextItem(tooltip_text)
        self.edge_tooltip.setDefaultTextColor(QColor(60, 60, 60))
        
        # Add a background
        background = QGraphicsRectItem(self.edge_tooltip.boundingRect().adjusted(-5, -5, 5, 5))
        background.setBrush(QBrush(QColor(255, 255, 255, 220)))
        background.setPen(QPen(QColor(200, 200, 200)))
        
        # Make the tooltip a child of the background
        self.edge_tooltip.setParentItem(background)
        
        # Add to scene
        self.addItem(background)
        
        # Position near the cursor
        if self.start_port:
            port_pos = self.start_port.mapToScene(QPointF(0, 0))
            background.setPos(port_pos + QPointF(20, -30))
        
        # Set a timer to remove the tooltip after a few seconds
        QTimer.singleShot(3000, self.remove_edge_tooltip)

    def remove_edge_tooltip(self):
        """Remove the edge creation tooltip"""
        if hasattr(self, 'edge_tooltip') and self.edge_tooltip:
            # Remove the parent item (background) which will also remove the tooltip
            if self.edge_tooltip.parentItem():
                self.removeItem(self.edge_tooltip.parentItem())
            self.edge_tooltip = None

    def show_snap_indicator(self, port):
        """Show a visual indicator when snapping to a port"""
        # If we already have a snap indicator, remove it
        self.hide_snap_indicator()
        
        # Create a new snap indicator
        self.snap_indicator = QGraphicsEllipseItem()
        
        # Set the size and position
        port_pos = port.mapToScene(QPointF(0, 0))
        indicator_size = 40  # Larger than the port for visibility
        self.snap_indicator.setRect(
            port_pos.x() - indicator_size/2,
            port_pos.y() - indicator_size/2,
            indicator_size,
            indicator_size
        )
        
        # Set the appearance with theme-aware colors
        highlight_color = ThemeManager.get_color("port_highlight", self.is_dark_theme)
        self.snap_indicator.setPen(QPen(QColor(highlight_color.red(), highlight_color.green(), highlight_color.blue(), 0), 2))  # Transparent border
        self.snap_indicator.setBrush(QBrush(QColor(highlight_color.red(), highlight_color.green(), highlight_color.blue(), 80)))  # Semi-transparent highlight
        
        # Add to scene
        self.addItem(self.snap_indicator)
        
        # Create a grow/shrink animation for the indicator
        if not hasattr(self, 'snap_animation') or self.snap_animation is None:
            self.snap_animation = QVariantAnimation()
            self.snap_animation.setDuration(800)  # 800ms for one cycle
            self.snap_animation.setLoopCount(-1)  # Loop indefinitely
            self.snap_animation.setStartValue(0.8)
            self.snap_animation.setEndValue(1.2)
            self.snap_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
            
            # Connect the animation to update the indicator size
            self.snap_animation.valueChanged.connect(self.update_snap_indicator)
        
        # Start the animation
        self.snap_animation.start()

    def hide_snap_indicator(self):
        """Hide the snap indicator"""
        if self.snap_indicator:
            self.removeItem(self.snap_indicator)
            self.snap_indicator = None
            
        # Stop the animation if it's running
        if hasattr(self, 'snap_animation') and self.snap_animation:
            self.snap_animation.stop()

    def create_edge(self, start_port, end_port):
        """Create an edge between two ports if they are compatible"""
        # Debug port information
        try:
            start_node = start_port.parentItem() if start_port else None
            end_node = end_port.parentItem() if end_port else None
            start_category = start_node.config.category.value if hasattr(start_node, 'config') else "Unknown"
            end_category = end_node.config.category.value if hasattr(end_node, 'config') else "Unknown"
            
            print(f"Attempting to create edge: {start_category} -> {end_category}")
            if not start_port or not end_port:
                print(f"❌ Null ports: start_port={start_port}, end_port={end_port}")
                return None
                
            if not isinstance(start_port, PortItem) or not isinstance(end_port, PortItem):
                print(f"❌ Ports are not PortItem instances: start={type(start_port)}, end={type(end_port)}")
                return None
        except Exception as e:
            print(f"❌ Error getting port info: {e}")
            
        # Verify compatibility
        if not self.is_compatible_connection(start_port, end_port):
            # Flash red on both ports to indicate invalid connection
            error_color = ThemeManager.get_color("edge_error", self.is_dark_theme)
            for port in [start_port, end_port]:
                port.setBrush(QBrush(QColor(error_color.red(), error_color.green(), error_color.blue(), 180)))  # Error color
                # Use a lambda with default argument to capture the current port
                fill_color = ThemeManager.get_color("port_fill", self.is_dark_theme)
                QTimer.singleShot(300, lambda p=port, c=fill_color: p.setBrush(QBrush(c)))
            
            # Show a message in the status bar if available
            for view in self.views():
                main_window = view.find_main_window()
                if main_window and hasattr(main_window, 'statusBar'):
                    main_window.statusBar().showMessage("Connection not allowed between these ports", 3000)
            
            # Print detailed debug information about the incompatible ports
            try:
                start_node = start_port.parentItem()
                end_node = end_port.parentItem()
                print(f"❌ Incompatible connection between ports:")
                print(f"   Start: {start_node.config.category.value}.{start_port.port_type} -> End: {end_node.config.category.value}.{end_port.port_type}")
                print(f"   Start port label: {getattr(start_port, 'label', 'None')}, End port label: {getattr(end_port, 'label', 'None')}")
            except Exception as e:
                print(f"Error getting detailed port info: {e}")
            
            return None
        
        # Check if edge already exists
        for item in self.items():
            if isinstance(item, EdgeItem):
                if (item.start_port == start_port and item.end_port == end_port) or \
                   (item.start_port == end_port and item.end_port == start_port):
                    print("Edge already exists between these ports")
                    
                    # Flash the existing edge to highlight it
                    item.setOpacity(0.5)
                    QTimer.singleShot(300, lambda: item.setOpacity(1.0))
                    
                    return item
        
        # Default patient count is 100, but may be set differently by callers
        patient_count = 100
        
        # Special case: For Target Population to Eligible Population, use 400 patients
        start_node = start_port.parentItem()
        end_node = end_port.parentItem()
        if (hasattr(start_node, 'config') and start_node.config.category == NodeCategory.TARGET_POPULATION and
            hasattr(end_node, 'config') and end_node.config.category == NodeCategory.ELIGIBLE_POPULATION):
            patient_count = 400
        
        # Create new edge with default patient flow
        try:
            edge = EdgeItem(start_port, end_port, "default", patient_count=patient_count)
            self.addItem(edge)
            
            # Get the parent nodes for logging
            start_category = start_node.config.category.value if hasattr(start_node, 'config') else "Unknown"
            end_category = end_node.config.category.value if hasattr(end_node, 'config') else "Unknown"
            
            # Success info
            print(f"✅ Created edge from {start_category} to {end_category} with patient count {patient_count}")
            
            # Briefly highlight the new edge with a flash effect
            edge.setOpacity(0.7)
            QTimer.singleShot(300, lambda: edge.setOpacity(1.0))
            
            # Show success message in status bar if available
            for view in self.views():
                main_window = view.find_main_window()
                if main_window and hasattr(main_window, 'statusBar'):
                    main_window.statusBar().showMessage(f"Connected {start_category} to {end_category}", 3000)
                    
            # Update the connections display safely
            try:
                self.update_connections_display()
            except Exception as e:
                print(f"Error updating connections display: {e}")
            
            return edge
        except Exception as e:
            print(f"❌ Error creating edge: {e}")
            return None

    def reset_zoom(self):
        """Reset all nodes in the workflow to inactive state."""
        # Clear all node states
        for item in self.items():
            if isinstance(item, WorkflowNode):
                item.set_active(False)
                
        # Reset all edge colors to default
        for item in self.items():
            if isinstance(item, EdgeItem):
                item.setActionColor("default")
                
        # Set patient group as active
        if self.target_population:
            self.target_population.set_active(True)
            
    def center_on_study_model(self):
        """Center the view on the study model (target population) node."""
        # Check if target_population exists
        if not hasattr(self, 'target_population') or self.target_population is None:
            # Try to find any TARGET_POPULATION node
            for item in self.items():
                if isinstance(item, WorkflowNode) and hasattr(item, 'config') and item.config.category == NodeCategory.TARGET_POPULATION:
                    self.target_population = item
                    break
            
        # If we still don't have a valid target_population, handle gracefully
        if not hasattr(self, 'target_population') or self.target_population is None or not self.target_population.scene():
            print("No valid target population found for centering")
            # Just center on all items if there's a valid bounding rect
            if self.views() and not self.itemsBoundingRect().isEmpty():
                self.views()[0].fitInView(self.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
            return
            
        # Check if target_population scene is valid
        if self.target_population.scene() != self:
            print("Target population is in a different scene")
            # Try to find another target_population in this scene
            for item in self.items():
                if isinstance(item, WorkflowNode) and hasattr(item, 'config') and item.config.category == NodeCategory.TARGET_POPULATION:
                    self.target_population = item
                    break
            else:
                # If no valid target_population found, just center on all
                if self.views() and not self.itemsBoundingRect().isEmpty():
                    self.views()[0].fitInView(self.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
                return
            
        # Find the view
        if not self.views():
            return
            
        try:
            view = self.views()[0]
            view.centerOn(self.target_population)
        except RuntimeError as e:
            # Handle case where the C++ object might have been deleted
            print(f"Warning: Could not center on target population: {e}")
            # Try to center on all items instead
            if self.views() and not self.itemsBoundingRect().isEmpty():
                self.views()[0].fitInView(self.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def drawBackground(self, painter, rect):
        """Override drawBackground to ensure proper scene rendering"""
        super().drawBackground(painter, rect)
        
        # Set rendering hints for smoother graphics
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
    def drawForeground(self, painter, rect):
        """Override drawForeground to ensure proper scene rendering"""
        super().drawForeground(painter, rect)
        
        # Set rendering hints for smoother graphics
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

    def update_snap_indicator(self, value):
        """Update snap indicator's appearance based on value"""
        if hasattr(self, "snap_indicator") and self.snap_indicator:
            # Fade opacity based on value (0-1)
            self.snap_indicator.setOpacity(value)
            
            # Scale size based on value (1.0 - 1.3)
            scale = 1.0 + (0.3 * value)
            
            # Calculate new transform
            transform = QTransform()
            transform.scale(scale, scale)
            
            # Apply transform
            self.snap_indicator.setTransform(transform)
            
            # Update color based on value (fade from blue to green)
            if hasattr(self.snap_indicator, 'setBrush'):
                # Create a gradient color from blue to green
                start_color = QColor(70, 130, 180)  # Steel blue
                end_color = QColor(76, 175, 80)    # Material green
                
                # Interpolate between colors
                r = int(start_color.red() + (end_color.red() - start_color.red()) * value)
                g = int(start_color.green() + (end_color.green() - start_color.green()) * value)
                b = int(start_color.blue() + (end_color.blue() - start_color.blue()) * value)
                
                # Create color with opacity
                brush_color = QColor(r, g, b, int(255 * (0.6 + 0.4 * value)))
                self.snap_indicator.setBrush(QBrush(brush_color))
                
                # Create matching border color
                border_color = QColor(r, g, b, int(255 * (0.8 + 0.2 * value)))
                self.snap_indicator.setPen(QPen(border_color, 2))
    
    def parse_workflow_json(self, json_string):
        """
        Parse a JSON string containing a workflow definition and load it.
        
        Args:
            json_string (str): JSON string defining a workflow
            
        Returns:
            tuple: (bool success, dict validation_results)
        """
        try:
            # Parse the JSON string
            workflow_data = json.loads(json_string)
            
            # Load the workflow using the structured loader
            return self.load_from_workflow_json(workflow_data)
        except json.JSONDecodeError as e:
            return False, {"valid": False, "errors": [f"Invalid JSON format: {str(e)}"]}
        except Exception as e:
            return False, {"valid": False, "errors": [f"Error parsing workflow: {str(e)}"]}
    
    def create_from_llm_json(self, design_description, patient_count=1000):
        """
        Create a workflow design from a natural language description using a predefined template.
        This is a placeholder that would be replaced with actual LLM-generated JSON in production.
        
        Args:
            design_description (str): Description of the study design
            patient_count (int): Total target population count
            
        Returns:
            tuple: (bool success, dict validation_results)
        """
        # For demonstration, we'll create a parallel group design
        if "parallel" in design_description.lower() or "rct" in design_description.lower():
            workflow_json = {
                "design_type": "parallel_group",
                "nodes": [
                    {
                        "id": "population",
                        "type": "target_population",
                        "label": "Initial Population",
                        "x": 0,
                        "y": 0,
                        "patient_count": patient_count
                    },
                    {
                        "id": "eligible",
                        "type": "eligible_population",
                        "label": "Eligible Subjects",
                        "x": 300,
                        "y": 0,
                        "patient_count": int(patient_count * 0.5)
                    },
                    {
                        "id": "randomization",
                        "type": "randomization",
                        "label": "Randomization",
                        "x": 600,
                        "y": 0
                    },
                    {
                        "id": "intervention",
                        "type": "intervention",
                        "label": "Treatment Group",
                        "x": 900,
                        "y": -150,
                        "description": "Active intervention"
                    },
                    {
                        "id": "control",
                        "type": "control",
                        "label": "Control Group",
                        "x": 900,
                        "y": 150,
                        "description": "Placebo control"
                    },
                    {
                        "id": "outcome_int",
                        "type": "outcome",
                        "label": "Primary Outcome",
                        "x": 1200,
                        "y": -150
                    },
                    {
                        "id": "outcome_ctrl",
                        "type": "outcome",
                        "label": "Primary Outcome",
                        "x": 1200,
                        "y": 150
                    },
                    {
                        "id": "timepoint_int",
                        "type": "timepoint",
                        "label": "Final Measurement",
                        "x": 1500,
                        "y": -150
                    },
                    {
                        "id": "timepoint_ctrl",
                        "type": "timepoint",
                        "label": "Final Measurement",
                        "x": 1500,
                        "y": 150
                    }
                ],
                "edges": [
                    {
                        "source": "population",
                        "target": "eligible",
                        "patient_count": int(patient_count * 0.5),
                        "label": "Screening"
                    },
                    {
                        "source": "eligible",
                        "target": "randomization",
                        "patient_count": int(patient_count * 0.5),
                        "label": "Randomized"
                    },
                    {
                        "source": "randomization",
                        "target": "intervention",
                        "patient_count": int(patient_count * 0.25),
                        "label": "Treatment Arm"
                    },
                    {
                        "source": "randomization",
                        "target": "control",
                        "patient_count": int(patient_count * 0.25),
                        "label": "Control Arm"
                    },
                    {
                        "source": "intervention",
                        "target": "outcome_int",
                        "patient_count": int(patient_count * 0.25),
                        "label": "Follow-up"
                    },
                    {
                        "source": "control",
                        "target": "outcome_ctrl",
                        "patient_count": int(patient_count * 0.25),
                        "label": "Follow-up"
                    },
                    {
                        "source": "outcome_int",
                        "target": "timepoint_int",
                        "patient_count": int(patient_count * 0.25),
                        "label": "12-week measurement"
                    },
                    {
                        "source": "outcome_ctrl",
                        "target": "timepoint_ctrl",
                        "patient_count": int(patient_count * 0.25),
                        "label": "12-week measurement"
                    }
                ]
            }
        elif "pre-post" in design_description.lower() or "prepost" in design_description.lower():
            workflow_json = {
                "design_type": "pre_post",
                "nodes": [
                    {
                        "id": "population",
                        "type": "target_population",
                        "label": "Initial Population",
                        "x": 0,
                        "y": 0,
                        "patient_count": patient_count
                    },
                    {
                        "id": "eligible",
                        "type": "eligible_population",
                        "label": "Study Population",
                        "x": 300,
                        "y": 0,
                        "patient_count": int(patient_count * 0.3)
                    },
                    {
                        "id": "cohort",
                        "type": "subgroup",
                        "label": "Study Cohort",
                        "x": 600,
                        "y": 0,
                        "patient_count": int(patient_count * 0.3)
                    },
                    {
                        "id": "outcome",
                        "type": "outcome",
                        "label": "Primary Outcome",
                        "x": 900,
                        "y": 0
                    },
                    {
                        "id": "intervention",
                        "type": "intervention",
                        "label": "Intervention",
                        "x": 1200,
                        "y": 0,
                        "description": "Study intervention"
                    },
                    {
                        "id": "baseline",
                        "type": "timepoint",
                        "label": "Baseline (Pre)",
                        "x": 1500,
                        "y": -150
                    },
                    {
                        "id": "followup",
                        "type": "timepoint",
                        "label": "Follow-up (Post)",
                        "x": 1500,
                        "y": 150
                    }
                ],
                "edges": [
                    {
                        "source": "population",
                        "target": "eligible",
                        "patient_count": int(patient_count * 0.3),
                        "label": "Screening"
                    },
                    {
                        "source": "eligible",
                        "target": "cohort",
                        "patient_count": int(patient_count * 0.3),
                        "label": "Enrolled"
                    },
                    {
                        "source": "cohort",
                        "target": "outcome",
                        "patient_count": int(patient_count * 0.3),
                        "label": "Baseline"
                    },
                    {
                        "source": "cohort",
                        "target": "intervention",
                        "patient_count": int(patient_count * 0.3),
                        "label": "Treatment"
                    },
                    {
                        "source": "intervention",
                        "target": "outcome",
                        "patient_count": int(patient_count * 0.27),
                        "label": "Post"
                    },
                    {
                        "source": "outcome",
                        "target": "baseline",
                        "patient_count": int(patient_count * 0.3),
                        "label": "Pre-measurement"
                    },
                    {
                        "source": "outcome",
                        "target": "followup",
                        "patient_count": int(patient_count * 0.27),
                        "label": "Post-measurement"
                    }
                ]
            }
        elif "case-control" in design_description.lower() or "case control" in design_description.lower():
            workflow_json = {
                "design_type": "case_control",
                "nodes": [
                    {
                        "id": "population",
                        "type": "target_population",
                        "label": "Source Population",
                        "x": 0,
                        "y": 0,
                        "patient_count": patient_count
                    },
                    {
                        "id": "eligible",
                        "type": "eligible_population",
                        "label": "Study Population",
                        "x": 300,
                        "y": 0,
                        "patient_count": int(patient_count * 0.5)
                    },
                    {
                        "id": "cases",
                        "type": "subgroup",
                        "label": "Cases",
                        "x": 600,
                        "y": -150,
                        "description": "Patients with condition",
                        "patient_count": int(patient_count * 0.1)
                    },
                    {
                        "id": "controls",
                        "type": "subgroup",
                        "label": "Controls",
                        "x": 600,
                        "y": 150,
                        "description": "Patients without condition",
                        "patient_count": int(patient_count * 0.4)
                    },
                    {
                        "id": "cases_outcome",
                        "type": "outcome",
                        "label": "Exposure Assessment",
                        "x": 900,
                        "y": -150
                    },
                    {
                        "id": "controls_outcome",
                        "type": "outcome",
                        "label": "Exposure Assessment",
                        "x": 900,
                        "y": 150
                    }
                ],
                "edges": [
                    {
                        "source": "population",
                        "target": "eligible",
                        "patient_count": int(patient_count * 0.5),
                        "label": "Selection"
                    },
                    {
                        "source": "eligible",
                        "target": "cases",
                        "patient_count": int(patient_count * 0.1),
                        "label": "Case Identification"
                    },
                    {
                        "source": "eligible",
                        "target": "controls",
                        "patient_count": int(patient_count * 0.4),
                        "label": "Control Selection"
                    },
                    {
                        "source": "cases",
                        "target": "cases_outcome",
                        "patient_count": int(patient_count * 0.1),
                        "label": "Exposure History"
                    },
                    {
                        "source": "controls",
                        "target": "controls_outcome",
                        "patient_count": int(patient_count * 0.4),
                        "label": "Exposure History"
                    }
                ]
            }
        else:
            # Default to a simple design
            workflow_json = {
                "design_type": "basic",
                "nodes": [
                    {
                        "id": "population",
                        "type": "target_population",
                        "label": "Target Population",
                        "x": 0,
                        "y": 0,
                        "patient_count": patient_count
                    },
                    {
                        "id": "eligible",
                        "type": "eligible_population",
                        "label": "Eligible Population",
                        "x": 300,
                        "y": 0,
                        "patient_count": int(patient_count * 0.5)
                    },
                    {
                        "id": "intervention",
                        "type": "intervention",
                        "label": "Intervention",
                        "x": 600,
                        "y": 0,
                        "description": "Study intervention"
                    },
                    {
                        "id": "outcome",
                        "type": "outcome",
                        "label": "Outcome",
                        "x": 900,
                        "y": 0
                    }
                ],
                "edges": [
                    {
                        "source": "population",
                        "target": "eligible",
                        "patient_count": int(patient_count * 0.5),
                        "label": "Screening"
                    },
                    {
                        "source": "eligible",
                        "target": "intervention",
                        "patient_count": int(patient_count * 0.5),
                        "label": "Treatment"
                    },
                    {
                        "source": "intervention",
                        "target": "outcome",
                        "patient_count": int(patient_count * 0.45),
                        "label": "Assessment"
                    }
                ]
            }
        
        # Load the workflow from the generated JSON
        return self.load_from_workflow_json(workflow_json)

    def create_node_from_port(self, port, pos):
        """Create a new node based on the clicked port"""
        # Get the parent node and its category
        parent_node = port.parentItem()
        if not parent_node or not hasattr(parent_node, 'config'):
            return None
            
        parent_category = parent_node.config.category
        port_label = port.label.lower() if hasattr(port, 'label') else ""
        
        print(f"Creating node from port: {port_label} on node {parent_category.value}")
        
        # Map port labels to node categories
        node_category_map = {
            "eligible population": NodeCategory.ELIGIBLE_POPULATION,
            "randomization": NodeCategory.RANDOMIZATION,
            "subgroup": NodeCategory.SUBGROUP,
            "outcome": NodeCategory.OUTCOME,
            "control": NodeCategory.CONTROL,
            "intervention": NodeCategory.INTERVENTION,
            "timepoint": NodeCategory.TIMEPOINT,
        }
        
        # Check if the port label directly maps to a node category
        next_category = node_category_map.get(port_label.lower())
        
        # Check if we found a valid category
        if next_category is None:
            print(f"No matching category found for port label: {port_label}")
            return None
            
        # Create the new node
        new_node = self.add_node(next_category, pos)
        
        # Create an edge between the parent node and the new node
        if new_node:
            # Special handling based on connection types
            connection_established = False
            
            # Handle Outcome to Timepoint connection (top port of Outcome node)
            if parent_category == NodeCategory.OUTCOME and next_category == NodeCategory.TIMEPOINT:
                # Connect from top port of outcome to input port of timepoint
                if hasattr(new_node, 'input_ports') and new_node.input_ports and port.label.lower() == "timepoint":
                    try:
                        edge = self.create_edge(port, new_node.input_ports[0])
                        if edge:
                            edge.setPatientCount(100)
                            connection_established = True
                            print("Created connection from Outcome to Timepoint")
                    except Exception as e:
                        print(f"Error creating edge: {e}")
            
            # Handle Subgroup to Outcome connection
            elif parent_category == NodeCategory.SUBGROUP and next_category == NodeCategory.OUTCOME:
                # Connect from right outcome port of subgroup to input port of outcome
                if hasattr(new_node, 'input_ports') and new_node.input_ports and port.label.lower() == "outcome":
                    try:
                        edge = self.create_edge(port, new_node.input_ports[0])
                        if edge:
                            edge.setPatientCount(100)
                            connection_established = True
                            print("Created connection from Subgroup to Outcome")
                    except Exception as e:
                        print(f"Error creating edge: {e}")
            
            # Handle other cases
            if not connection_established and hasattr(new_node, 'input_ports') and new_node.input_ports:
                # Try to find a compatible input port on the new node
                for input_port in new_node.input_ports:
                    if self.is_compatible_connection(port, input_port):
                        try:
                            edge = self.create_edge(port, input_port)
                            if edge:
                                # Flash the new node to highlight it
                                new_node.setOpacity(0.5)
                                QTimer.singleShot(300, lambda: new_node.setOpacity(1.0))
                                # Set a default patient count on the edge
                                edge.setPatientCount(100)
                                connection_established = True
                                print(f"Created generic connection from {parent_category.value} to {next_category.value}")
                        except Exception as e:
                            print(f"Error creating edge: {e}")
                        break
            
            # If still no connection, try alternative port matches
            if not connection_established:
                print("No direct compatible connection found, trying alternative approaches")
                
                # Check if we need to create a special connection
                if (parent_category == NodeCategory.SUBGROUP and next_category == NodeCategory.INTERVENTION):
                    # These specific connections might need special handling
                    if hasattr(new_node, 'input_ports') and new_node.input_ports:
                        try:
                            edge = self.create_edge(port, new_node.input_ports[0])
                            if edge:
                                edge.setPatientCount(100)
                                print(f"Created special connection from {parent_category.value} to {next_category.value}")
                        except Exception as e:
                            print(f"Error creating edge: {e}")
        
        # Highlight the new node by briefly flashing it
        if new_node:
            # Make the node more visually prominent by briefly changing opacity
            new_node.setOpacity(0.7)
            QTimer.singleShot(300, lambda: new_node.setOpacity(1.0))
            
            # Update connections display safely
            try:
                self.update_connections_display()
            except Exception as e:
                print(f"Error updating connections display: {e}")
        
        return new_node

    def create_preset_study(self, preset_type: str):
        """Create a preset study design based on the specified type."""
        # Clear existing nodes and edges
        self.clear()
        
        # Create a new target population node
        self.setup_initial_nodes()
        
        # Create the specified preset
        if preset_type == "pre_post":
            self.create_pre_post_design()
        elif preset_type == "case_control":
            self.create_case_control_design()
        elif preset_type == "observational":
            self.create_observational_design()
        else:
            print(f"Unknown preset type: {preset_type}")
            return
            
        # Update the connections display safely
        try:
            self.update_connections_display()
        except Exception as e:
            print(f"Error updating connections display: {e}")
            

    def create_parallel_group_design(self):
        """Create a parallel group randomized controlled trial design."""
        # Create eligible population node
        eligible = self.add_node(NodeCategory.ELIGIBLE_POPULATION, QPointF(300, 0))
        
        # Create randomization node
        randomization = self.add_node(NodeCategory.RANDOMIZATION, QPointF(600, 0))
        
        # Create intervention and control nodes
        intervention = self.add_node(NodeCategory.INTERVENTION, QPointF(900, -150))
        control = self.add_node(NodeCategory.CONTROL, QPointF(900, 150))
        
        # Create outcome nodes
        intervention_outcome = self.add_node(NodeCategory.OUTCOME, QPointF(1200, -150))
        control_outcome = self.add_node(NodeCategory.OUTCOME, QPointF(1200, 150))
        
        # Create timepoint nodes
        intervention_timepoint = self.add_node(NodeCategory.TIMEPOINT, QPointF(1500, -150))
        control_timepoint = self.add_node(NodeCategory.TIMEPOINT, QPointF(1500, 150))
        
        # Create edges with proper connections
        # Target Population to Eligible Population
        self.create_edge(self.target_population.output_ports[0], eligible.input_ports[0])
        
        # Eligible Population to Randomization
        self.create_edge(eligible.output_ports[0], randomization.input_ports[0])
        
        # Randomization to Intervention and Control
        self.create_edge(randomization.output_ports[0], intervention.input_ports[0])
        self.create_edge(randomization.output_ports[1], control.input_ports[0])
        
        # Intervention and Control to their respective Outcomes
        self.create_edge(intervention.output_ports[0], intervention_outcome.input_ports[0])
        self.create_edge(control.output_ports[0], control_outcome.input_ports[0])
        
        # Outcomes to their respective Timepoints
        self.create_edge(intervention_outcome.output_ports[0], intervention_timepoint.input_ports[0])
        self.create_edge(control_outcome.output_ports[0], control_timepoint.input_ports[0])

    def create_crossover_design(self):
        """Create a crossover trial design."""
        # Create eligible population node
        eligible = self.add_node(NodeCategory.ELIGIBLE_POPULATION, QPointF(300, 0))
        
        # Create randomization node
        randomization = self.add_node(NodeCategory.RANDOMIZATION, QPointF(600, 0))
        
        # Create intervention and control nodes for first period
        intervention1 = self.add_node(NodeCategory.INTERVENTION, QPointF(900, -150))
        control1 = self.add_node(NodeCategory.CONTROL, QPointF(900, 150))
        
        # Create outcome nodes for first period
        intervention1_outcome = self.add_node(NodeCategory.OUTCOME, QPointF(1200, -150))
        control1_outcome = self.add_node(NodeCategory.OUTCOME, QPointF(1200, 150))
        
        # Create intervention and control nodes for second period (crossover)
        intervention2 = self.add_node(NodeCategory.INTERVENTION, QPointF(1500, 150))
        control2 = self.add_node(NodeCategory.CONTROL, QPointF(1500, -150))
        
        # Create outcome nodes for second period
        intervention2_outcome = self.add_node(NodeCategory.OUTCOME, QPointF(1800, 150))
        control2_outcome = self.add_node(NodeCategory.OUTCOME, QPointF(1800, -150))
        
        # Create timepoint nodes
        timepoint1 = self.add_node(NodeCategory.TIMEPOINT, QPointF(2100, -150))
        timepoint2 = self.add_node(NodeCategory.TIMEPOINT, QPointF(2100, 150))
        
        # Create edges with proper connections
        # Initial flow
        self.create_edge(self.target_population.output_ports[0], eligible.input_ports[0])
        self.create_edge(eligible.output_ports[0], randomization.input_ports[0])
        
        # First period connections
        self.create_edge(randomization.output_ports[0], intervention1.input_ports[0])
        self.create_edge(randomization.output_ports[1], control1.input_ports[0])
        self.create_edge(intervention1.output_ports[0], intervention1_outcome.input_ports[0])
        self.create_edge(control1.output_ports[0], control1_outcome.input_ports[0])
        
        # Crossover connections
        self.create_edge(intervention1_outcome.output_ports[0], control2.input_ports[0])
        self.create_edge(control1_outcome.output_ports[0], intervention2.input_ports[0])
        
        # Second period outcomes
        self.create_edge(intervention2.output_ports[0], intervention2_outcome.input_ports[0])
        self.create_edge(control2.output_ports[0], control2_outcome.input_ports[0])
        
        # Connect to timepoints
        self.create_edge(intervention1_outcome.output_ports[0], timepoint1.input_ports[0])
        self.create_edge(control1_outcome.output_ports[0], timepoint1.input_ports[0])
        self.create_edge(intervention2_outcome.output_ports[0], timepoint2.input_ports[0])
        self.create_edge(control2_outcome.output_ports[0], timepoint2.input_ports[0])

    def create_factorial_design(self):
        """Create a factorial trial design."""
        # Create eligible population node
        eligible = self.add_node(NodeCategory.ELIGIBLE_POPULATION, QPointF(300, 0))
        
        # Create randomization node
        randomization = self.add_node(NodeCategory.RANDOMIZATION, QPointF(600, 0))
        
        # Create intervention nodes for different combinations
        intervention_a = self.add_node(NodeCategory.INTERVENTION, QPointF(900, -200))
        intervention_b = self.add_node(NodeCategory.INTERVENTION, QPointF(900, -100))
        intervention_ab = self.add_node(NodeCategory.INTERVENTION, QPointF(900, 0))
        control = self.add_node(NodeCategory.CONTROL, QPointF(900, 100))
        
        # Create outcome nodes
        outcome_a = self.add_node(NodeCategory.OUTCOME, QPointF(1200, -200))
        outcome_b = self.add_node(NodeCategory.OUTCOME, QPointF(1200, -100))
        outcome_ab = self.add_node(NodeCategory.OUTCOME, QPointF(1200, 0))
        outcome_control = self.add_node(NodeCategory.OUTCOME, QPointF(1200, 100))
        
        # Create timepoint nodes
        timepoint = self.add_node(NodeCategory.TIMEPOINT, QPointF(1500, 0))
        
        # Create edges with proper connections
        # Initial flow
        self.create_edge(self.target_population.output_ports[0], eligible.input_ports[0])
        self.create_edge(eligible.output_ports[0], randomization.input_ports[0])
        
        # Connect randomization to interventions
        self.create_edge(randomization.output_ports[0], intervention_a.input_ports[0])
        self.create_edge(randomization.output_ports[1], intervention_b.input_ports[0])
        self.create_edge(randomization.output_ports[2], intervention_ab.input_ports[0])
        self.create_edge(randomization.output_ports[3], control.input_ports[0])
        
        # Connect interventions to outcomes
        self.create_edge(intervention_a.output_ports[0], outcome_a.input_ports[0])
        self.create_edge(intervention_b.output_ports[0], outcome_b.input_ports[0])
        self.create_edge(intervention_ab.output_ports[0], outcome_ab.input_ports[0])
        self.create_edge(control.output_ports[0], outcome_control.input_ports[0])
        
        # Connect all outcomes to the timepoint
        self.create_edge(outcome_a.output_ports[0], timepoint.input_ports[0])
        self.create_edge(outcome_b.output_ports[0], timepoint.input_ports[0])
        self.create_edge(outcome_ab.output_ports[0], timepoint.input_ports[0])
        self.create_edge(outcome_control.output_ports[0], timepoint.input_ports[0])

    def create_adaptive_design(self):
        """Create an adaptive trial design with interim analysis."""
        # Create eligible population node
        eligible = self.add_node(NodeCategory.ELIGIBLE_POPULATION, QPointF(300, 0))
        
        # Create randomization node
        randomization = self.add_node(NodeCategory.RANDOMIZATION, QPointF(600, 0))
        
        # Create intervention and control nodes
        intervention = self.add_node(NodeCategory.INTERVENTION, QPointF(900, -150))
        control = self.add_node(NodeCategory.CONTROL, QPointF(900, 150))
        
        # Create interim outcome nodes
        interim_outcome_int = self.add_node(NodeCategory.OUTCOME, QPointF(1200, -150))
        interim_outcome_ctrl = self.add_node(NodeCategory.OUTCOME, QPointF(1200, 150))
        
        # Create subgroup nodes for interim analysis
        subgroup_int = self.add_node(NodeCategory.SUBGROUP, QPointF(1500, -150))
        subgroup_ctrl = self.add_node(NodeCategory.SUBGROUP, QPointF(1500, 150))
        
        # Create final outcome nodes
        final_outcome_int = self.add_node(NodeCategory.OUTCOME, QPointF(1800, -150))
        final_outcome_ctrl = self.add_node(NodeCategory.OUTCOME, QPointF(1800, 150))
        
        # Create timepoint nodes
        timepoint_interim = self.add_node(NodeCategory.TIMEPOINT, QPointF(1200, 0))
        timepoint_final = self.add_node(NodeCategory.TIMEPOINT, QPointF(1800, 0))
        
        # Create edges with proper connections
        # Initial flow
        self.create_edge(self.target_population.output_ports[0], eligible.input_ports[0])
        self.create_edge(eligible.output_ports[0], randomization.input_ports[0])
        
        # Connect to intervention arms
        self.create_edge(randomization.output_ports[0], intervention.input_ports[0])
        self.create_edge(randomization.output_ports[1], control.input_ports[0])
        
        # Connect to interim outcomes
        self.create_edge(intervention.output_ports[0], interim_outcome_int.input_ports[0])
        self.create_edge(control.output_ports[0], interim_outcome_ctrl.input_ports[0])
        
        # Connect interim outcomes to timepoint
        self.create_edge(interim_outcome_int.output_ports[0], timepoint_interim.input_ports[0])
        self.create_edge(interim_outcome_ctrl.output_ports[0], timepoint_interim.input_ports[0])
        
        # Connect to subgroups for adaptation
        self.create_edge(interim_outcome_int.output_ports[0], subgroup_int.input_ports[0])
        self.create_edge(interim_outcome_ctrl.output_ports[0], subgroup_ctrl.input_ports[0])
        
        # Connect subgroups to final outcomes
        self.create_edge(subgroup_int.output_ports[0], final_outcome_int.input_ports[0])
        self.create_edge(subgroup_ctrl.output_ports[0], final_outcome_ctrl.input_ports[0])
        
        # Connect final outcomes to final timepoint
        self.create_edge(final_outcome_int.output_ports[0], timepoint_final.input_ports[0])
        self.create_edge(final_outcome_ctrl.output_ports[0], timepoint_final.input_ports[0])

    def create_single_arm_design(self):
        """Create a single-arm trial design."""
        # Create eligible population node
        eligible = self.add_node(NodeCategory.ELIGIBLE_POPULATION, QPointF(300, 0))
        
        # Create subgroup for pre-intervention
        pre_subgroup = self.add_node(NodeCategory.SUBGROUP, QPointF(600, -100))
        
        # Create intervention node
        intervention = self.add_node(NodeCategory.INTERVENTION, QPointF(900, 0))
        
        # Create subgroup for post-intervention
        post_subgroup = self.add_node(NodeCategory.SUBGROUP, QPointF(1200, -100))
        
        # Create outcome nodes
        pre_outcome = self.add_node(NodeCategory.OUTCOME, QPointF(600, 100))
        post_outcome = self.add_node(NodeCategory.OUTCOME, QPointF(1200, 100))
        
        # Create timepoint nodes
        pre_timepoint = self.add_node(NodeCategory.TIMEPOINT, QPointF(600, 200))
        post_timepoint = self.add_node(NodeCategory.TIMEPOINT, QPointF(1200, 200))
        
        # Create edges with proper connections
        # Initial flow
        self.create_edge(self.target_population.output_ports[0], eligible.input_ports[0])
        
        # Pre-intervention connections
        self.create_edge(eligible.output_ports[0], pre_subgroup.input_ports[0])
        self.create_edge(pre_subgroup.output_ports[0], pre_outcome.input_ports[0])
        self.create_edge(pre_outcome.output_ports[0], pre_timepoint.input_ports[0])
        
        # Intervention flow
        self.create_edge(pre_subgroup.output_ports[0], intervention.input_ports[0])
        
        # Post-intervention connections
        self.create_edge(intervention.output_ports[0], post_subgroup.input_ports[0])
        self.create_edge(post_subgroup.output_ports[0], post_outcome.input_ports[0])
        self.create_edge(post_outcome.output_ports[0], post_timepoint.input_ports[0])

    def update_connections_display(self):
        """Update the text display showing current node connections"""
        try:
            # Check if connections_text exists and is valid
            needs_new_text = False
            
            if not hasattr(self, 'connections_text'):
                needs_new_text = True
            else:
                try:
                    # Check if the item is still valid
                    if not self.connections_text or sip.isdeleted(self.connections_text) or not self.connections_text.scene():
                        needs_new_text = True
                except (RuntimeError, ReferenceError):
                    # Handle case where the C++ object is already deleted
                    needs_new_text = True
            
            if needs_new_text:
                # Recreate the connections text if it doesn't exist or is invalid
                self.connections_text = QGraphicsTextItem()
                self.connections_text.setPos(-1950, -1950)
                self.connections_text.setZValue(1000)
                font = QFont("Arial", 10)
                self.connections_text.setFont(font)
                self.addItem(self.connections_text)
                
                # Create background for the text
                self.connections_text.background_rect = QGraphicsRectItem(self.connections_text.boundingRect())
                self.connections_text.background_rect.setParentItem(self.connections_text)
                self.connections_text.background_rect.setZValue(-1)
            
            # Get all edges
            edges = self.get_workflow_edges()
            if not edges:
                self.connections_text.setPlainText("No connections")
            else:
                connections_text = "Current Connections:\n"
                for i, edge in enumerate(edges, 1):
                    start_node = edge.start_node
                    end_node = edge.end_node
                    
                    if start_node and end_node:
                        start_type = start_node.config.category.value
                        end_type = end_node.config.category.value
                        
                        # Get node titles if available
                        start_title = ""
                        end_title = ""
                        
                        for item in start_node.childItems():
                            if isinstance(item, QGraphicsTextItem) and item.toPlainText() and not isinstance(item, PortItem):
                                start_title = item.toPlainText().split('\n')[0]
                                break
                                
                        for item in end_node.childItems():
                            if isinstance(item, QGraphicsTextItem) and item.toPlainText() and not isinstance(item, PortItem):
                                end_title = item.toPlainText().split('\n')[0]
                                break
                        
                        if start_title:
                            start_type = f"{start_type} ({start_title})"
                        if end_title:
                            end_type = f"{end_type} ({end_title})"
                        
                        connections_text += f"{i}. {start_type} → {end_type}\n"
                
                self.connections_text.setPlainText(connections_text)
            
            # Apply theme colors to the text
            if self.is_dark_theme:
                self.connections_text.setDefaultTextColor(QColor(220, 220, 220))
            else:
                self.connections_text.setDefaultTextColor(QColor(60, 60, 60))
            
            # Position at top right of the view
            if self.views():
                view = self.views()[0]
                view_rect = view.viewport().rect()
                scene_rect = view.mapToScene(view_rect).boundingRect()
                
                # Add padding from the edge
                padding = 20
                self.connections_text.setPos(
                    scene_rect.right() - self.connections_text.boundingRect().width() - padding,
                    scene_rect.top() + padding
                )
                
                # Update background rect
                if hasattr(self.connections_text, 'background_rect') and not sip.isdeleted(self.connections_text.background_rect):
                    self.connections_text.background_rect.setRect(self.connections_text.boundingRect())
                    
                    # Apply theme-appropriate background color
                    if self.is_dark_theme:
                        self.connections_text.background_rect.setBrush(QBrush(QColor(40, 40, 40, 220)))
                        self.connections_text.background_rect.setPen(QPen(QColor(60, 60, 60)))
                    else:
                        self.connections_text.background_rect.setBrush(QBrush(QColor(245, 245, 245, 220)))
                        self.connections_text.background_rect.setPen(QPen(Qt.PenStyle.NoPen))
        except Exception as e:
            # Log the error but don't crash the application
            print(f"Error updating connections display: {e}")
            import traceback
            traceback.print_exc()

    def create_pre_post_design(self):
        """Create a pre-post study design with one study group and two timepoints connected to one outcome."""
        # Clear any existing nodes/edges (already done in create_preset_study)
        
        # Create nodes with proper positioning for clearer visualization
        eligible = self.add_node(NodeCategory.ELIGIBLE_POPULATION, QPointF(300, 0))
        eligible.display_name = "Study Population"
        
        study_group = self.add_node(NodeCategory.SUBGROUP, QPointF(600, 0))
        study_group.display_name = "Study Cohort"
        
        # Single outcome node to represent the same outcome measured at different times
        outcome = self.add_node(NodeCategory.OUTCOME, QPointF(900, 0))
        outcome.display_name = "Primary Outcome"
        
        # Intervention between baseline and follow-up
        intervention = self.add_node(NodeCategory.INTERVENTION, QPointF(1200, 0))
        intervention.display_name = "Intervention"
        
        # Two timepoint nodes - baseline and follow-up
        baseline = self.add_node(NodeCategory.TIMEPOINT, QPointF(1500, -150))
        baseline.display_name = "Baseline (Pre)"
        
        followup = self.add_node(NodeCategory.TIMEPOINT, QPointF(1500, 150))
        followup.display_name = "Follow-up (Post)"
        
        # CONNECTIONS
        # 1. Target Population to Eligible Population (with 1000 patients)
        target_to_eligible = self.create_edge(self.target_population.output_ports[0], eligible.input_ports[0])
        if target_to_eligible:
            target_to_eligible.setPatientCount(1000)
            target_to_eligible.flow_data["label"] = "Initial Population"
        
        # 2. Eligible Population to Study Group (with 300 patients)
        eligible_to_group = None
        for port in eligible.output_ports:
            if hasattr(port, 'label') and port.label.lower() == "subgroup":
                eligible_to_group = self.create_edge(port, study_group.input_ports[0])
                break
                    
        # If no subgroup port was found, use the first output port
        if not eligible_to_group and eligible.output_ports:
            eligible_to_group = self.create_edge(eligible.output_ports[0], study_group.input_ports[0])
            
        # Set patient count for eligible to study group
        if eligible_to_group:
            eligible_to_group.setPatientCount(300)  # 75% of eligible population (400)
        
        # 3. Study Group to Primary Outcome (baseline measurement)
        group_to_outcome = None
        for port in study_group.output_ports:
            if hasattr(port, 'label') and port.label.lower() == "outcome":
                group_to_outcome = self.create_edge(port, outcome.input_ports[0])
                if group_to_outcome:
                    group_to_outcome.setPatientCount(300)  # Same as study group
                    group_to_outcome.flow_data["label"] = "Baseline Measurement"
                    break
                    
        # 4. Study Group to Intervention (use the intervention port)
        group_to_intervention = None
        for port in study_group.output_ports:
            if hasattr(port, 'label') and port.label.lower() == "intervention":
                group_to_intervention = self.create_edge(port, intervention.input_ports[0])
                break
                
        # If no specific intervention port was found, use another output port
        if not group_to_intervention and study_group.output_ports:
            for port in study_group.output_ports:
                if port != group_to_outcome.start_port:  # Use a different port than we used for outcome
                    group_to_intervention = self.create_edge(port, intervention.input_ports[0])
                    break
                    
        # Set patient count for study group to intervention
        if group_to_intervention:
            group_to_intervention.setPatientCount(300)  # Same as study group
        
        # 5. Intervention back to Outcome (for post measurement)
        intervention_to_outcome = None
        if intervention.output_ports and outcome.input_ports:
            intervention_to_outcome = self.create_edge(intervention.output_ports[0], outcome.input_ports[0])
            if intervention_to_outcome:
                intervention_to_outcome.setPatientCount(270)  # 90% follow-up rate (some loss to follow-up)
                intervention_to_outcome.flow_data["label"] = "Post Measurement"
        
        # 6. Outcome to both Timepoints (using the timepoint port if available)
        outcome_to_timepoints = []
        timepoint_port = None
        for port in outcome.output_ports:
            if hasattr(port, 'label') and port.label.lower() == "timepoint":
                timepoint_port = port
                break
        
        # Connect to baseline timepoint
        if timepoint_port and baseline.input_ports:
            baseline_edge = self.create_edge(timepoint_port, baseline.input_ports[0])
            if baseline_edge:
                baseline_edge.setPatientCount(300)  # Original number of baseline measurements
                baseline_edge.flow_data["label"] = "Baseline Data"
                outcome_to_timepoints.append(baseline_edge)
        
        # Connect to follow-up timepoint
        if timepoint_port and followup.input_ports:
            followup_edge = self.create_edge(timepoint_port, followup.input_ports[0])
            if followup_edge:
                followup_edge.setPatientCount(270)  # Reduced number at follow-up (loss to follow-up)
                followup_edge.flow_data["label"] = "Follow-up Data"
                outcome_to_timepoints.append(followup_edge)
        
        # If no specific timepoint port was found, use the first output port
        if not outcome_to_timepoints and outcome.output_ports:
            if baseline.input_ports:
                baseline_edge = self.create_edge(outcome.output_ports[0], baseline.input_ports[0])
                if baseline_edge:
                    baseline_edge.setPatientCount(300)  # Original number of baseline measurements
                    baseline_edge.flow_data["label"] = "Baseline Data"
                    outcome_to_timepoints.append(baseline_edge)
                
            if followup.input_ports:
                followup_edge = self.create_edge(outcome.output_ports[0], followup.input_ports[0])
                if followup_edge:
                    followup_edge.setPatientCount(270)  # Reduced number at follow-up (loss to follow-up)
                    followup_edge.flow_data["label"] = "Follow-up Data"
                    outcome_to_timepoints.append(followup_edge)

    def create_case_control_design(self):
        """Create a case-control study design with cases and controls connected to a single outcome."""
        # Clear any existing nodes/edges (already done in create_preset_study)
        
        # Create nodes with proper positioning for clearer visualization
        eligible = self.add_node(NodeCategory.ELIGIBLE_POPULATION, QPointF(300, 0))
        eligible.display_name = "Study Population"
        
        # Add case group (top branch)
        case_group = self.add_node(NodeCategory.SUBGROUP, QPointF(600, -150))
        case_group.display_name = "Cases"
        
        # Add control group (bottom branch)
        control_group = self.add_node(NodeCategory.SUBGROUP, QPointF(600, 150))
        control_group.display_name = "Controls"
        
        # Single outcome node for both groups
        outcome = self.add_node(NodeCategory.OUTCOME, QPointF(900, 0))
        outcome.display_name = "Primary Outcome"
        
        # Add outcome measurement timepoints
        case_outcome = self.add_node(NodeCategory.TIMEPOINT, QPointF(1200, -150))
        case_outcome.display_name = "Case Measurement"
        
        control_outcome = self.add_node(NodeCategory.TIMEPOINT, QPointF(1200, 150))
        control_outcome.display_name = "Control Measurement"
        
        # Add analysis node
        analysis = self.add_node(NodeCategory.INTERVENTION, QPointF(1500, 0))
        analysis.display_name = "Analysis"
        
        # Link target population to eligible population (with 1000 patients)
        target_eligible = self.create_edge(self.target_population.output_ports[0], eligible.input_ports[0])
        if target_eligible:
            target_eligible.setPatientCount(1000)
            target_eligible.flow_data["label"] = "Initial Population"
        
        # Create connections (with realistic patient flow)
        eligible_case = self.create_edge(eligible.output_ports[0], case_group.input_ports[0])
        if eligible_case:
            eligible_case.setPatientCount(200)  # 50% of eligible population (400)
            eligible_case.flow_data["label"] = "Case Selection"
        
        eligible_control = self.create_edge(eligible.output_ports[0], control_group.input_ports[0])
        if eligible_control:
            eligible_control.setPatientCount(200)  # 50% of eligible population (400)
            eligible_control.flow_data["label"] = "Control Selection"
        
        # Connect both groups to the single outcome node
        case_outcome_edge = self.create_edge(case_group.output_ports[0], outcome.input_ports[0])
        if case_outcome_edge:
            case_outcome_edge.setPatientCount(200)  # Maintain same patient count from case group
            case_outcome_edge.flow_data["label"] = "Case Data"
        
        control_outcome_edge = self.create_edge(control_group.output_ports[0], outcome.input_ports[0])
        if control_outcome_edge:
            control_outcome_edge.setPatientCount(200)  # Maintain same patient count from control group
            control_outcome_edge.flow_data["label"] = "Control Data"
        
        # Connect outcome to timepoints
        outcome_case_timepoint = self.create_edge(outcome.output_ports[0], case_outcome.input_ports[0])
        if outcome_case_timepoint:
            outcome_case_timepoint.setPatientCount(200)  # Maintain same patient count from case group
            outcome_case_timepoint.flow_data["label"] = "Case Measurement"
        
        outcome_control_timepoint = self.create_edge(outcome.output_ports[0], control_outcome.input_ports[0])
        if outcome_control_timepoint:
            outcome_control_timepoint.setPatientCount(200)  # Maintain same patient count from control group
            outcome_control_timepoint.flow_data["label"] = "Control Measurement"
        
        # Connect timepoints to analysis
        timepoint_port_case = None
        for port in case_outcome.output_ports:
            if hasattr(port, 'label') and port.label.lower() == "timepoint":
                timepoint_port_case = port
                break
        
        if timepoint_port_case:
            case_analysis = self.create_edge(timepoint_port_case, analysis.input_ports[0])
            if case_analysis:
                case_analysis.setPatientCount(200)  # Maintain same patient count from case group
                case_analysis.flow_data["label"] = "Case Data"
        elif case_outcome.output_ports:
            case_analysis = self.create_edge(case_outcome.output_ports[0], analysis.input_ports[0])
            if case_analysis:
                case_analysis.setPatientCount(200)  # Maintain same patient count from case group
                case_analysis.flow_data["label"] = "Case Data"
        
        timepoint_port_control = None
        for port in control_outcome.output_ports:
            if hasattr(port, 'label') and port.label.lower() == "timepoint":
                timepoint_port_control = port
                break
        
        if timepoint_port_control:
            control_analysis = self.create_edge(timepoint_port_control, analysis.input_ports[0])
            if control_analysis:
                control_analysis.setPatientCount(200)  # Maintain same patient count from control group
                control_analysis.flow_data["label"] = "Control Data"
        elif control_outcome.output_ports:
            control_analysis = self.create_edge(control_outcome.output_ports[0], analysis.input_ports[0])
            if control_analysis:
                control_analysis.setPatientCount(200)  # Maintain same patient count from control group
                control_analysis.flow_data["label"] = "Control Data"

    def create_observational_design(self):
        """Create an observational study design with multiple cohorts connected to a single outcome."""
        # Clear any existing nodes/edges (already done in create_preset_study)
        
        # Create nodes with proper positioning for clearer visualization
        eligible = self.add_node(NodeCategory.ELIGIBLE_POPULATION, QPointF(300, 0))
        eligible.display_name = "Study Population"
        
        # Create exposure cohorts
        cohort1 = self.add_node(NodeCategory.SUBGROUP, QPointF(600, -200))
        cohort1.display_name = "Exposed Cohort"
        
        cohort2 = self.add_node(NodeCategory.SUBGROUP, QPointF(600, 0))
        cohort2.display_name = "Unexposed Cohort"
        
        cohort3 = self.add_node(NodeCategory.SUBGROUP, QPointF(600, 200))
        cohort3.display_name = "Reference Cohort"
        
        # Single outcome node for all cohorts
        outcome = self.add_node(NodeCategory.OUTCOME, QPointF(900, 0))
        outcome.display_name = "Primary Outcome"
        
        # Timepoints for each cohort
        timepoint1 = self.add_node(NodeCategory.TIMEPOINT, QPointF(1200, -200))
        timepoint1.display_name = "Exposed Measurement"
        
        timepoint2 = self.add_node(NodeCategory.TIMEPOINT, QPointF(1200, 0))
        timepoint2.display_name = "Unexposed Measurement"
        
        timepoint3 = self.add_node(NodeCategory.TIMEPOINT, QPointF(1200, 200))
        timepoint3.display_name = "Reference Measurement"
        
        # Analysis node
        analysis = self.add_node(NodeCategory.INTERVENTION, QPointF(1500, 0))
        analysis.display_name = "Analysis"
        
        # Link target population to eligible population (with 1000 patients)
        target_eligible = self.create_edge(self.target_population.output_ports[0], eligible.input_ports[0])
        if target_eligible:
            target_eligible.setPatientCount(1000)
            target_eligible.flow_data["label"] = "Initial Population"
        
        # Create connections from eligible population to cohorts (with realistic patient flow)
        eligible_cohort1 = self.create_edge(eligible.output_ports[0], cohort1.input_ports[0])
        if eligible_cohort1:
            eligible_cohort1.setPatientCount(150)  # ~38% of eligible population (400)
            eligible_cohort1.flow_data["label"] = "Exposed Selection"
        
        eligible_cohort2 = self.create_edge(eligible.output_ports[0], cohort2.input_ports[0])
        if eligible_cohort2:
            eligible_cohort2.setPatientCount(125)  # ~31% of eligible population (400)
            eligible_cohort2.flow_data["label"] = "Unexposed Selection"
        
        eligible_cohort3 = self.create_edge(eligible.output_ports[0], cohort3.input_ports[0])
        if eligible_cohort3:
            eligible_cohort3.setPatientCount(125)  # ~31% of eligible population (400)
            eligible_cohort3.flow_data["label"] = "Reference Selection"
        
        # Connect all cohorts to the single outcome node
        cohort1_outcome = self.create_edge(cohort1.output_ports[0], outcome.input_ports[0])
        if cohort1_outcome:
            cohort1_outcome.setPatientCount(150)  # Maintain same patient count from exposed cohort
            cohort1_outcome.flow_data["label"] = "Exposed Data"
        
        cohort2_outcome = self.create_edge(cohort2.output_ports[0], outcome.input_ports[0])
        if cohort2_outcome:
            cohort2_outcome.setPatientCount(125)  # Maintain same patient count from unexposed cohort
            cohort2_outcome.flow_data["label"] = "Unexposed Data"
        
        cohort3_outcome = self.create_edge(cohort3.output_ports[0], outcome.input_ports[0])
        if cohort3_outcome:
            cohort3_outcome.setPatientCount(125)  # Maintain same patient count from reference cohort
            cohort3_outcome.flow_data["label"] = "Reference Data"
        
        # Connect outcome to timepoints
        outcome_timepoint1 = self.create_edge(outcome.output_ports[0], timepoint1.input_ports[0])
        if outcome_timepoint1:
            outcome_timepoint1.setPatientCount(150)  # Maintain same patient count from exposed cohort
            outcome_timepoint1.flow_data["label"] = "Exposed Measurement"
        
        outcome_timepoint2 = self.create_edge(outcome.output_ports[0], timepoint2.input_ports[0])
        if outcome_timepoint2:
            outcome_timepoint2.setPatientCount(125)  # Maintain same patient count from unexposed cohort
            outcome_timepoint2.flow_data["label"] = "Unexposed Measurement"
        
        outcome_timepoint3 = self.create_edge(outcome.output_ports[0], timepoint3.input_ports[0])
        if outcome_timepoint3:
            outcome_timepoint3.setPatientCount(125)  # Maintain same patient count from reference cohort
            outcome_timepoint3.flow_data["label"] = "Reference Measurement"
        
        # Connect timepoints to analysis
        for timepoint, label, count in [(timepoint1, "Exposed Data", 150), 
                                    (timepoint2, "Unexposed Data", 125),
                                    (timepoint3, "Reference Data", 125)]:
            timepoint_port = None
            for port in timepoint.output_ports:
                if hasattr(port, 'label') and port.label.lower() == "timepoint":
                    timepoint_port = port
                    break
            
            if timepoint_port:
                edge = self.create_edge(timepoint_port, analysis.input_ports[0])
                if edge:
                    edge.setPatientCount(count)  # Set appropriate patient count based on cohort
                    edge.flow_data["label"] = label
            elif timepoint.output_ports:
                edge = self.create_edge(timepoint.output_ports[0], analysis.input_ports[0])
                if edge:
                    edge.setPatientCount(count)  # Set appropriate patient count based on cohort
                    edge.flow_data["label"] = label

class WorkflowView(QGraphicsView):
    """View class for displaying the workflow scene"""
    
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        
        # Set view properties
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        
        # Set background color
        self.setBackgroundBrush(QBrush(QColor(245, 245, 245)))
        
        # Initialize zoom level
        self.zoom_level = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 3.0
        self.zoom_factor = 1.15  # Zoom in/out factor
        
        # Track if we're panning
        self.panning = False
        self.last_pan_point = None
        
        # Connect to selection changes
        self.scene().selectionChanged.connect(self.on_selection_changed)
        
        # Set focus policy to accept keyboard events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # Set up context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)
        
        # Update connections text position when view is resized
        self.viewport().installEventFilter(self)
        
    def eventFilter(self, obj, event):
        """Event filter to handle viewport resize events"""
        if obj == self.viewport() and event.type() == QEvent.Type.Resize:
            # Update connections text position when viewport is resized
            try:
                scene = self.scene()
                if scene and hasattr(scene, 'update_connections_display'):
                    scene.update_connections_display()
            except Exception as e:
                print(f"Error in eventFilter: {e}")
        return super().eventFilter(obj, event)
    
    def on_selection_changed(self):
        """Update status bar when selection changes"""
        scene = self.scene()
        if not scene:
            return
            
        # Get selected items
        selected_items = scene.selectedItems()
        
        # Find the main window to access the status bar
        main_window = self.find_main_window()
        if not main_window or not hasattr(main_window, 'statusBar'):
            return
            
        # Update status bar based on selection
        if selected_items:
            # Check if any selected item is a node
            has_nodes = any(isinstance(item, WorkflowNode) for item in selected_items)
            has_edges = any(isinstance(item, EdgeItem) for item in selected_items)
            
            if has_nodes or has_edges:
                main_window.statusBar().showMessage("Press Delete to remove selected items", 3000)
        else:
            # Clear status message when nothing is selected
            main_window.statusBar().clearMessage()
    
    def find_main_window(self):
        """Find the main window by traversing parent widgets"""
        parent = self.parent()
        while parent:
            if isinstance(parent, QMainWindow):
                return parent
            parent = parent.parent()
        return None

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming"""
        # Get the current mouse position in scene coordinates
        mouse_pos = self.mapToScene(event.position().toPoint())
        
        # Calculate zoom factor based on wheel delta
        zoom_in = event.angleDelta().y() > 0
        
        if zoom_in and self.zoom_level < self.max_zoom:
            # Zoom in
            self.scale(self.zoom_factor, self.zoom_factor)
            self.zoom_level *= self.zoom_factor
        elif not zoom_in and self.zoom_level > self.min_zoom:
            # Zoom out
            self.scale(1 / self.zoom_factor, 1 / self.zoom_factor)
            self.zoom_level /= self.zoom_factor
            
        # Ensure zoom level stays within bounds
        if self.zoom_level < self.min_zoom:
            self.zoom_level = self.min_zoom
        elif self.zoom_level > self.max_zoom:
            self.zoom_level = self.max_zoom
            
        # Update status bar with zoom level if available
        if self.scene().statusBar:
            self.scene().statusBar.showMessage(f"Zoom: {int(self.zoom_level * 100)}%", 2000)
            
        # Update connections text position after zooming
        if hasattr(self.scene(), 'connections_text') and self.scene().connections_text:
            self.scene().update_connections_display()
            
        # Prevent the event from being passed to parent widgets
        event.accept()

    def keyPressEvent(self, event):
        """Handle key press events"""
        # Delete selected items with Delete key
        if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            # Get the scene
            scene = self.scene()
            if not scene:
                super().keyPressEvent(event)
                return
                
            # Get selected items
            selected_items = scene.selectedItems()
            
            if selected_items:
                # Delete selected items
                self.delete_selected_items(scene, selected_items)
                
                # After deletion, highlight compatible ports for potential reconnection
                scene.highlight_compatible_ports()
                
                # Prevent event from propagating
                event.accept()
                return
        # Center view on patient group with Home key
        elif event.key() == Qt.Key.Key_0:
            self.resetTransform()
            self.zoom_level = 1.0
        # Zoom in with + key
        elif event.key() == Qt.Key.Key_Plus:
            factor = 1.1
            self.zoom_level *= factor
            self.scale(factor, factor)
        # Zoom out with - key
        elif event.key() == Qt.Key.Key_Minus:
            factor = 1.0 / 1.1
            self.zoom_level *= factor
            self.scale(factor, factor)
        # Handle other keys normally
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press events in the view"""
        # Check if we clicked on an item
        item = self.itemAt(event.pos())
        
        # Middle mouse button always activates panning
        if event.button() == Qt.MouseButton.MiddleButton:
            # Store the initial position for panning
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        
        # Left mouse button on empty space activates panning
        if event.button() == Qt.MouseButton.LeftButton and not item:
            # Store the initial position for panning
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        
        # For all other cases (including clicks on nodes), pass to parent implementation
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse movement in the view"""
        # Handle panning with middle mouse button or left mouse button on empty space
        if hasattr(self, '_pan_start') and self._pan_start is not None:
            if (event.buttons() & Qt.MouseButton.MiddleButton) or (event.buttons() & Qt.MouseButton.LeftButton):
                # Calculate the delta movement
                delta = event.position() - self._pan_start
                self._pan_start = event.position()
                
                # Pan the view by adjusting the scrollbars
                self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - delta.x()))
                self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - delta.y()))
                event.accept()
                return
        
        # For all other cases, pass to parent implementation
        super().mouseMoveEvent(event)
        
    def mouseReleaseEvent(self, event):
        """Handle mouse release events in the view"""
        # Reset panning state when either middle or left button is released
        if event.button() == Qt.MouseButton.MiddleButton or event.button() == Qt.MouseButton.LeftButton:
            if hasattr(self, '_pan_start'):
                self._pan_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        
        # Always call parent implementation to ensure proper event handling
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        """Show context menu for the view"""
        # Create the menu
        menu = QMenu(self)
        
        # Get the scene position
        scene_pos = self.mapToScene(event.pos())
        
        # Get selected items
        scene = self.scene()
        if not scene:
            return
            
        selected_items = scene.selectedItems()
        
        # Add delete option if items are selected
        if selected_items:
            delete_action = menu.addAction("Delete Selected Items")
            delete_action.triggered.connect(lambda: self.delete_selected_items(scene, selected_items))
            menu.addSeparator()
        
        # Add node creation options
        menu.addAction("Add Target Population").triggered.connect(
            lambda: self.scene().add_node(NodeCategory.TARGET_POPULATION, scene_pos))
        menu.addAction("Add Eligible Population").triggered.connect(
            lambda: self.scene().add_node(NodeCategory.ELIGIBLE_POPULATION, scene_pos))
        menu.addAction("Add Intervention").triggered.connect(
            lambda: self.scene().add_node(NodeCategory.INTERVENTION, scene_pos))
        menu.addAction("Add Outcome").triggered.connect(
            lambda: self.scene().add_node(NodeCategory.OUTCOME, scene_pos))
        menu.addAction("Add Subgroup").triggered.connect(
            lambda: self.scene().add_node(NodeCategory.SUBGROUP, scene_pos))
        menu.addAction("Add Control").triggered.connect(
            lambda: self.scene().add_node(NodeCategory.CONTROL, scene_pos))
        menu.addAction("Add Randomization").triggered.connect(
            lambda: self.scene().add_node(NodeCategory.RANDOMIZATION, scene_pos))
        menu.addAction("Add Timepoint").triggered.connect(
            lambda: self.scene().add_node(NodeCategory.TIMEPOINT, scene_pos))
            
        # Add view actions
        menu.addSeparator()
        menu.addAction("Reset Zoom").triggered.connect(self.reset_zoom)
        
        # Show the menu
        menu.exec(event.globalPos())

    def reset_zoom(self):
        """Reset the zoom level to 100%"""
        self.resetTransform()
        self.zoom_level = 1.0

    def delete_selected_items(self, scene, selected_items):
        """Delete selected items from the scene"""
        if not selected_items:
            return
            
        # Track if we need to update connections
        edges_deleted = False
            
        for item in selected_items:
            # Handle edge deletion
            if isinstance(item, EdgeItem):
                # Remove the edge
                scene.removeItem(item)
                edges_deleted = True
                
            # Handle node deletion
            elif isinstance(item, WorkflowNode):
                # First remove all connected edges
                edges_to_remove = []
                for scene_item in scene.items():
                    if isinstance(scene_item, EdgeItem):
                        if (scene_item.start_port and scene_item.start_port.parentItem() == item) or \
                           (scene_item.end_port and scene_item.end_port.parentItem() == item):
                            edges_to_remove.append(scene_item)
                
                for edge in edges_to_remove:
                    scene.removeItem(edge)
                    edges_deleted = True
                
                # Then remove the node itself
                scene.removeItem(item)
                
                # If this was the target population, clear the reference
                if scene.target_population == item:
                    scene.target_population = None
        
        # Update the connections display if edges were deleted
        if edges_deleted:
            try:
                scene.update_connections_display()
            except Exception as e:
                print(f"Error updating connections display: {e}")

class ModelBuilder(QWidget):
    """The main model builder widget that integrates into the application."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Set widget properties
        self.setMinimumSize(800, 600)
        
        # Initialize UI components
        self.setupUi()

        self.studies_manager = None
        
        # Connect signals
        self.connectSignals()
    
    def setupUi(self):
        """Set up the UI components"""
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create scene and view
        self.scene = WorkflowScene(self)
        self.view = WorkflowView(self.scene, self)
        
        # Create toolbar
        self.toolbar = QToolBar("Main Toolbar")
        
        # Add zoom buttons to toolbar
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.triggered.connect(self.zoom_in)
        self.toolbar.addAction(zoom_in_action)
        
        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.triggered.connect(self.zoom_out)
        self.toolbar.addAction(zoom_out_action)
        
        reset_view_action = QAction("Reset View", self)
        reset_view_action.triggered.connect(self.reset_view)
        self.toolbar.addAction(reset_view_action)
        
        # Add separator
        self.toolbar.addSeparator()
        
        # Add "Load from Active Study" button
        load_from_study_action = QAction("Load from Active Study", self)
        load_from_study_action.setStatusTip("Load model from the active study in StudiesManager")
        load_from_study_action.triggered.connect(self.load_from_active_study)
        self.toolbar.addAction(load_from_study_action)
        
        # Add toolbar to layout
        layout.addWidget(self.toolbar)
        
        # Add view to layout
        layout.addWidget(self.view)
        
        # Create menu bar and add it to the layout
        self.menu_widget = QWidget()
        menu_layout = QHBoxLayout(self.menu_widget)
        menu_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create file menu
        file_menu = QMenu("&File")
        
        # Add icons to file menu actions
        new_action = file_menu.addAction(load_bootstrap_icon("file-plus"), "New")
        new_action.triggered.connect(self.new_workflow)
        
        open_action = file_menu.addAction(load_bootstrap_icon("folder2-open"), "Open")
        open_action.triggered.connect(self.open_workflow)
        
        save_action = file_menu.addAction(load_bootstrap_icon("save"), "Save")
        save_action.triggered.connect(self.save_workflow)
        
        # Add a separator before the import options
        file_menu.addSeparator()
        
        # Add "Load from Active Study" action
        load_from_study_action = file_menu.addAction("Load from Active Study")
        load_from_study_action.setStatusTip("Load model from the active study in StudiesManager")
        load_from_study_action.triggered.connect(self.load_from_active_study)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction(load_bootstrap_icon("box-arrow-right"), "Exit")
        exit_action.triggered.connect(self.close)
        
        # Create file menu button
        file_button = QPushButton("&File")
        file_button.setMenu(file_menu)
        menu_layout.addWidget(file_button)
        
        # Study Design menu
        design_menu = QMenu("Study Design")
        
        # Add only simple designs
        pre_post_action = design_menu.addAction(load_bootstrap_icon("arrow-left-right"), "Pre-Post Design")
        pre_post_action.triggered.connect(lambda: self.scene.create_preset_study("pre_post"))
        
        case_control_action = design_menu.addAction(load_bootstrap_icon("people"), "Case-Control Study")
        case_control_action.triggered.connect(lambda: self.scene.create_preset_study("case_control"))
        
        parallel_group_action = design_menu.addAction(load_bootstrap_icon("diagram-3"), "Parallel Group")
        parallel_group_action.triggered.connect(lambda: self.scene.create_preset_study("parallel_group"))
        
        crossover_action = design_menu.addAction(load_bootstrap_icon("arrow-repeat"), "Crossover")
        crossover_action.triggered.connect(lambda: self.scene.create_preset_study("crossover"))
        
        factorial_action = design_menu.addAction(load_bootstrap_icon("grid-3x3"), "Factorial")
        factorial_action.triggered.connect(lambda: self.scene.create_preset_study("factorial"))
        
        single_arm_action = design_menu.addAction(load_bootstrap_icon("people-fill"), "Single Arm")
        single_arm_action.triggered.connect(lambda: self.scene.create_preset_study("single_arm"))
        
        # Create design menu button
        design_button = QPushButton("Study Design")
        design_button.setMenu(design_menu)
        menu_layout.addWidget(design_button)
        
        # AI Design menu
        ai_design_menu = QMenu("AI Design")
        
        ai_design_action = ai_design_menu.addAction(load_bootstrap_icon("magic"), "Design Study")
        ai_design_action.triggered.connect(self.show_ai_design_dialog)
        
        # Create AI design menu button
        ai_button = QPushButton("AI Design")
        ai_button.setMenu(ai_design_menu)
        menu_layout.addWidget(ai_button)
        
        # Add spacer to push status bar to the right
        menu_layout.addStretch(1)
        
        # Add status bar
        self.status_bar = QStatusBar()
        self.scene.statusBar = self.status_bar  # Give the scene access to the status bar
        menu_layout.addWidget(self.status_bar)
        
        # Add the menu widget to the main layout
        layout.insertWidget(0, self.menu_widget)
        
        # Set window properties
        self.setWindowTitle("Research Flow Designer")
        self.resize(1200, 800)
    
    def connectSignals(self):
        """Connect signals to slots"""
        self.scene.nodeActivated.connect(self.on_node_activated)
        
    def resizeEvent(self, event):
        """Handle resize events"""
        # No need to reposition the close button as it's now in a layout
        super().resizeEvent(event)
        
    def mousePressEvent(self, event):
        """Pass mouse events to the parent class"""
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Pass mouse events to the parent class"""
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Pass mouse events to the parent class"""
        super().mouseReleaseEvent(event)
        
    def on_node_activated(self, node):
        """Handle node activation"""
        # Just display the node type in status bar
        if node:
            self.scene.statusBar.showMessage(f"Selected: {node.config.category.value.replace('_', ' ').title()}")
            
    def zoom_in(self):
        """Zoom in the view"""
        self.view.scale(1.2, 1.2)
    
    def zoom_out(self):
        """Zoom out the view"""
        self.view.scale(1/1.2, 1/1.2)
    
    def clear_all(self):
        """Clear all nodes and edges from the scene"""
        from PyQt6.QtWidgets import QMessageBox
        reply = QMessageBox.question(self, "Clear All", "Are you sure you want to clear all nodes and edges?",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            # Store the current theme setting before clearing
            is_dark = self.scene.is_dark_theme
            
            # Clear the scene
            self.scene.clear()
            
            # Reset the scene's theme
            self.scene.is_dark_theme = is_dark
            
            # Setup initial nodes
            self.scene.setup_initial_nodes()
            
            # Update the connections display
            self.scene.update_connections_display()
            
            self.scene.statusBar.showMessage("All nodes and edges cleared")

    def update_theme(self, is_dark):
        """Update the theme of the workflow scene and view"""
        # Update view background first
        if is_dark:
            self.view.setBackgroundBrush(QBrush(QColor(30, 30, 30)))  # Dark gray for dark theme
        else:
            self.view.setBackgroundBrush(QBrush(QColor(245, 245, 245)))  # Light gray for light theme

        # Then safely update scene theme
        try:
            # Store current theme state before updating
            self.scene.is_dark_theme = is_dark
            
            # Update scene background to match view
            background_color = QColor(30, 30, 30) if is_dark else QColor(245, 245, 245)
            self.scene.setBackgroundBrush(QBrush(background_color))
            
            # Update existing items
            self.scene.update_existing_items()
            
            # Safely update connections display
            if hasattr(self.scene, 'connections_text') and self.scene.connections_text and not sip.isdeleted(self.scene.connections_text):
                self.scene.update_connections_display()
        except Exception as e:
            print(f"Error updating theme: {e}")

    def create_menu_bar(self):
        """This method is no longer needed as we create menus in setupUi"""
        pass

    def new_workflow(self):
        """Create a new empty workflow."""
        self.scene.clear()
        self.scene.setup_initial_nodes()
        self.scene.center_on_study_model()

    def set_studies_manager(self, studies_manager):
        """Set the StudiesManager instance."""
        self.studies_manager = studies_manager

    @asyncSlot()
    async def show_ai_design_dialog(self):
        """Show the AI study design dialog to create a study from description."""
        dialog = StudyDescriptionDialog(self)
        if dialog.exec():
            description = dialog.get_description()
            patient_count = dialog.get_patient_count()
            
            if description:
                try:
                    # Generate the design using LLM
                    workflow_json = await generate_study_design_with_llm(description, patient_count)
                    
                    if workflow_json:
                        # Store the current theme setting before clearing
                        is_dark = self.scene.is_dark_theme
                        
                        # First remove any existing connections text to prevent reference errors
                        if hasattr(self.scene, 'connections_text') and self.scene.connections_text:
                            self.scene.removeItem(self.scene.connections_text)
                            self.scene.connections_text = None
                            
                        # Clear all target population nodes first
                        self.scene.target_population = None
                        
                        # Clear the scene completely
                        self.scene.clear()
                        
                        # Reset the scene's theme
                        self.scene.is_dark_theme = is_dark
                        
                        # Force garbage collection to ensure deleted objects are properly cleaned up
                        import gc
                        gc.collect()
                        
                        # Use load_from_workflow_json instead of load_from_json for LLM-generated designs
                        success, validation = self.scene.load_from_workflow_json(workflow_json)
                        
                        if success:
                            self.scene.center_on_study_model()
                            self.statusBar().showMessage("Successfully created study design with AI", 3000)
                        else:
                            # Show validation errors if any
                            error_msg = "\n".join(validation.get("errors", []))
                            QMessageBox.warning(
                                self,
                                "Design Loading Failed",
                                f"Failed to load the generated design:\n{error_msg}"
                            )
                    else:
                        QMessageBox.warning(
                            self,
                            "AI Design Generation Failed",
                            "Unable to generate a study design from your description. Please try again with more details."
                        )
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "AI Design Error",
                        f"Error creating study design: {str(e)}"
                    )

    def open_workflow(self):
        """Open a workflow from a file."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Workflow",
            "",
            "Workflow Files (*.json);;All Files (*.*)"
        )
        if file_name:
            try:
                with open(file_name, 'r') as f:
                    workflow_data = json.load(f)
                self.scene.load_from_json(workflow_data)
                self.scene.center_on_study_model()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open workflow: {str(e)}")

    def save_workflow(self):
        """Save the current workflow to a file."""
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Workflow",
            "",
            "Workflow Files (*.json);;All Files (*.*)"
        )
        if file_name:
            try:
                workflow_data = self.scene.save_to_json()
                with open(file_name, 'w') as f:
                    json.dump(workflow_data, f, indent=2)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save workflow: {str(e)}")

    def toggle_grid(self):
        """Toggle the grid visibility."""
        self.scene.show_grid = not self.scene.show_grid
        self.scene.update()

    def show_about(self):
        """Show the about dialog."""
        QMessageBox.about(
            self,
            "About Research Flow Designer",
            "Research Flow Designer\n\n"
            "A tool for designing and visualizing clinical trial workflows.\n\n"
            "Version 1.0"
        )
        
    def reset_view(self):
        """Reset the view to the default position."""
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.view.setTransform(QTransform())
        
    def load_json_sample(self, design_type):
        """Load a predefined JSON sample workflow."""
        try:
            if self.scene:
                success, validation = create_study_design_from_llm_json(self.scene, design_type, 1000)
                
                if success:
                    self.statusBar().showMessage(f"Successfully loaded {design_type} JSON sample", 3000)
                    self.scene.center_on_study_model()
                else:
                    QMessageBox.warning(
                        self,
                        "JSON Sample Load Error",
                        f"Failed to load {design_type} JSON sample: {', '.join(validation['errors'])}"
                    )
                    
                # Show warnings if any
                if validation["warnings"]:
                    warning_msg = "Warnings:\n- " + "\n- ".join(validation["warnings"])
                    print(warning_msg) # Just log to console
        except Exception as e:
            QMessageBox.critical(
                self,
                "JSON Sample Error",
                f"Error loading {design_type} JSON sample: {str(e)}"
            )
            
    def load_json_with_custom_count(self):
        """Load a JSON sample with a custom patient count."""
        design_types = ["parallel", "prepost", "case-control"]
        design_type, ok1 = QInputDialog.getItem(
            self,
            "Select Design Type",
            "Study Design:",
            design_types,
            0,
            False
        )
        
        if ok1:
            patient_count, ok2 = QInputDialog.getInt(
                self,
                "Patient Count",
                "Enter total patient count:",
                1000,  # Default
                10,    # Min
                100000 # Max
            )
            
            if ok2:
                try:
                    if self.scene:
                        success, validation = create_study_design_from_llm_json(self.scene, design_type, patient_count)
                        
                        if success:
                            self.statusBar().showMessage(
                                f"Successfully loaded {design_type} JSON sample with {patient_count} patients", 
                                3000
                            )
                            self.scene.center_on_study_model()
                        else:
                            QMessageBox.warning(
                                self,
                                "JSON Sample Load Error",
                                f"Failed to load {design_type} JSON sample: {', '.join(validation['errors'])}"
                            )
                            
                        # Show warnings if any
                        if validation["warnings"]:
                            warning_msg = "Warnings:\n- " + "\n- ".join(validation["warnings"])
                            print(warning_msg) # Just log to console
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "JSON Sample Error",
                        f"Error loading {design_type} JSON sample: {str(e)}"
                    )



    def load_from_study_manager(self, study_manager, study_id=None):
        """
        Load a workflow directly from a study in the StudiesManager.
        
        Args:
            study_manager: StudiesManager instance
            study_id: Optional study ID to load (defaults to active study)
        """
        scene = self.scene
        if scene:
            # Clear current design
            scene.clear()
            
            # Initialize the scene
            scene.setup_initial_nodes()
            
            # Get the study to load
            study = study_manager.get_study(study_id) if study_id else study_manager.get_active_study()
            if not study:
                # Find main window to show status message
                main_window = self.window()
                if hasattr(main_window, 'statusBar'):
                    main_window.statusBar().showMessage("No study found to load", 3000)
                return False
                
            # Try to generate a model plan from the study
            try:
                # If the study manager has a method to generate model plans
                if hasattr(study_manager, 'generate_model_plan'):
                    workflow_json = study_manager.generate_model_plan(study_id)
                    if workflow_json:
                        # Use the existing load_from_workflow_json method
                        success, validation = scene.load_from_workflow_json(workflow_json)
                        if success:
                            # Find main window to show status message
                            main_window = self.window()
                            if hasattr(main_window, 'statusBar'):
                                main_window.statusBar().showMessage("Successfully loaded study design from StudiesManager", 3000)
                            # Center view on the model
                            self.reset_view()
                            
                            # Safely center on study model if target_population exists
                            try:
                                if scene.target_population and scene.target_population.scene() == scene:
                                    scene.center_on_study_model()
                                else:
                                    # If no valid target_population, just fit all in view
                                    self.view.fitInView(scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
                            except Exception as e:
                                print(f"Warning: Could not center on target population: {e}")
                                # Fallback to fitting all items in view
                                self.view.fitInView(scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
                            
                            return True
                        else:
                            err_msg = ", ".join(validation.get("errors", ["Unknown error"]))
                            # Find main window to show status message
                            main_window = self.window()
                            if hasattr(main_window, 'statusBar'):
                                main_window.statusBar().showMessage(f"Failed to load study design: {err_msg}", 3000)
                            return False
                    else:
                        # Find main window to show status message
                        main_window = self.window()
                        if hasattr(main_window, 'statusBar'):
                            main_window.statusBar().showMessage("Failed to generate workflow from study", 3000)
                        return False
                else:
                    # Find main window to show status message
                    main_window = self.window()
                    if hasattr(main_window, 'statusBar'):
                        main_window.statusBar().showMessage("StudiesManager does not support model plan generation", 3000)
                    return False
            except Exception as e:
                # Find main window to show status message
                main_window = self.window()
                if hasattr(main_window, 'statusBar'):
                    main_window.statusBar().showMessage(f"Error loading from StudiesManager: {str(e)}", 3000)
                print(f"Error loading study: {str(e)}")
                return False
        return False
    
    def load_from_active_study(self):
        """Load a workflow directly from the active study in StudiesManager."""
        if self.studies_manager:
            # Check if the studies_manager has the active study method
            try:
                active_study = self.studies_manager.get_active_study()
                if not active_study:
                    QMessageBox.information(
                        self, 
                        "No Active Study", 
                        "There is no active study in the StudiesManager.\n"
                        "Please select a study first."
                    )
                    return False
                
                # Use the direct method
                success = self.load_from_study_manager(self.studies_manager)
                return success
            except Exception as e:
                QMessageBox.warning(
                    self, 
                    "Load Failed", 
                    f"Failed to load from active study: {str(e)}"
                )
                return False
        else:
            QMessageBox.information(
                self, 
                "No Studies Manager", 
                "Studies Manager is not connected.\n"
                "Please setup Studies Manager first."
            )
            return False
    
class StudyDescriptionDialog(QDialog):
    """Dialog to get study description from user and generate workflow via LLM."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Design Study with AI")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel(
            "Describe your clinical study design in detail. Include information about:\n"
            "• Population and eligibility criteria\n"
            "• Interventions and control groups\n"
            "• Randomization approach (if applicable)\n"
            "• Outcomes and measurements\n"
            "• Timepoints for assessment\n"
            "• Patient counts or proportions"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Text input area
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Example: A parallel group RCT with 1000 patients randomized 1:1 to receive either drug X or placebo for 12 weeks, with primary outcome of blood pressure reduction measured at baseline and 12 weeks.")
        layout.addWidget(self.text_edit)
        
        # Patient count
        count_layout = QHBoxLayout()
        count_layout.addWidget(QLabel("Initial patient count:"))
        self.patient_count_spinbox = QSpinBox()
        self.patient_count_spinbox.setRange(10, 10000)
        self.patient_count_spinbox.setValue(1000)
        self.patient_count_spinbox.setSingleStep(100)
        count_layout.addWidget(self.patient_count_spinbox)
        layout.addLayout(count_layout)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def get_description(self):
        return self.text_edit.toPlainText()
    
    def get_patient_count(self):
        return self.patient_count_spinbox.value()

@asyncSlot()
async def generate_study_design_with_llm(description, patient_count):
    """
    Generate a study design workflow JSON using an LLM based on the description.
    
    Args:
        description (str): User's description of the study
        patient_count (int): Initial patient population count
        
    Returns:
        dict: Study workflow JSON or None if the request failed
    """
    # LLM prompt template
    prompt = f"""
    As a clinical trial design expert, create a JSON representation of the following study design:
    
    {description}
    
    Initial patient count: {patient_count}
    
    The JSON should follow this format:
    {{
      "design_type": "[study type - parallel_group, crossover, factorial, adaptive, single_arm, pre_post, case_control, or observational]",
      "nodes": [
        {{
          "id": "[unique identifier]",
          "type": "[one of: target_population, eligible_population, intervention, outcome, subgroup, control, randomization, timepoint]",
          "label": "[descriptive name]",
          "x": [x position - use integers starting at 0],
          "y": [y position - use integers starting at 0],
          "patient_count": [number of patients at this node]
        }},
        ...more nodes...
      ],
      "edges": [
        {{
          "source": "[source node id]",
          "target": "[target node id]",
          "patient_count": [number of patients flowing through this connection],
          "label": "[descriptive label for the flow]"
        }},
        ...more edges...
      ]
    }}
    
    CONSTRAINTS:
    1. Ensure patient flow is logical (counts should decrease or stay the same as patients move through the workflow)
    2. Position nodes in a logical left-to-right, top-to-bottom flow
    3. Node types must be one of: target_population, eligible_population, intervention, outcome, subgroup, control, randomization, timepoint
    4. Every node needs a unique ID and descriptive label
    5. Every edge must connect existing node IDs
    6. First node should always be target_population
    7. Each node should have appropriate x,y coordinates for visual layout
    
    Return ONLY the JSON with no additional explanations.
    """
    
    try:
        from llms.client import call_llm_async_json
        
        json_response = await call_llm_async_json(prompt, model="claude-3-7-sonnet-20250219")
        return json_response
        
    except ImportError:
        # Fallback if llms module is not available
        # Placeholder for demo - simulate response based on description keywords
        if "parallel" in description.lower() or "rct" in description.lower():
            return create_parallel_group_json(patient_count)
        elif "crossover" in description.lower():
            return create_crossover_json(patient_count)
        elif "factorial" in description.lower():
            return create_factorial_json(patient_count)
        elif "single arm" in description.lower():
            return create_single_arm_json(patient_count)
        elif "pre-post" in description.lower() or "prepost" in description.lower():
            return create_prepost_json(patient_count)
        elif "case-control" in description.lower() or "case control" in description.lower():
            return create_case_control_json(patient_count)
        else:
            return create_parallel_group_json(patient_count)  # Default to parallel group
            
    except Exception as e:
        print(f"Error generating LLM study design: {str(e)}")
        return None

def create_parallel_group_json(patient_count):
    """Create a parallel group design JSON template."""
    return {
        "design_type": "parallel_group",
        "nodes": [
            {
                "id": "population",
                "type": "target_population",
                "label": "Initial Population",
                "x": 0,
                "y": 0,
                "patient_count": patient_count
            },
            {
                "id": "eligible",
                "type": "eligible_population",
                "label": "Eligible Subjects",
                "x": 300,
                "y": 0,
                "patient_count": int(patient_count * 0.5)
            },
            {
                "id": "randomization",
                "type": "randomization",
                "label": "Randomization",
                "x": 600,
                "y": 0,
                "patient_count": int(patient_count * 0.5)
            },
            {
                "id": "intervention",
                "type": "intervention",
                "label": "Treatment Group",
                "x": 900,
                "y": -150,
                "patient_count": int(patient_count * 0.25)
            },
            {
                "id": "control",
                "type": "control",
                "label": "Control Group",
                "x": 900,
                "y": 150,
                "patient_count": int(patient_count * 0.25)
            },
            {
                "id": "outcome_int",
                "type": "outcome",
                "label": "Treatment Outcome",
                "x": 1200,
                "y": -150,
                "patient_count": int(patient_count * 0.23)
            },
            {
                "id": "outcome_ctrl",
                "type": "outcome",
                "label": "Control Outcome",
                "x": 1200,
                "y": 150,
                "patient_count": int(patient_count * 0.24)
            },
            {
                "id": "timepoint_int",
                "type": "timepoint",
                "label": "12-Week Assessment",
                "x": 1500,
                "y": -150,
                "patient_count": int(patient_count * 0.23)
            },
            {
                "id": "timepoint_ctrl",
                "type": "timepoint",
                "label": "12-Week Assessment",
                "x": 1500,
                "y": 150,
                "patient_count": int(patient_count * 0.24)
            }
        ],
        "edges": [
            {
                "source": "population",
                "target": "eligible",
                "patient_count": int(patient_count * 0.5),
                "label": "Screening"
            },
            {
                "source": "eligible",
                "target": "randomization",
                "patient_count": int(patient_count * 0.5),
                "label": "Randomized"
            },
            {
                "source": "randomization",
                "target": "intervention",
                "patient_count": int(patient_count * 0.25),
                "label": "Treatment Arm"
            },
            {
                "source": "randomization",
                "target": "control",
                "patient_count": int(patient_count * 0.25),
                "label": "Control Arm"
            },
            {
                "source": "intervention",
                "target": "outcome_int",
                "patient_count": int(patient_count * 0.23),
                "label": "Follow-up"
            },
            {
                "source": "control",
                "target": "outcome_ctrl",
                "patient_count": int(patient_count * 0.24),
                "label": "Follow-up"
            },
            {
                "source": "outcome_int",
                "target": "timepoint_int",
                "patient_count": int(patient_count * 0.23),
                "label": "12-week measurement"
            },
            {
                "source": "outcome_ctrl",
                "target": "timepoint_ctrl",
                "patient_count": int(patient_count * 0.24),
                "label": "12-week measurement"
            }
        ]
    }

def create_crossover_json(patient_count):
    """Create a crossover design JSON template."""
    return {
        "design_type": "crossover",
        "nodes": [
            {
                "id": "population",
                "type": "target_population",
                "label": "Initial Population",
                "x": 0,
                "y": 0,
                "patient_count": patient_count
            },
            {
                "id": "eligible",
                "type": "eligible_population",
                "label": "Eligible Subjects",
                "x": 300,
                "y": 0,
                "patient_count": int(patient_count * 0.6)
            },
            {
                "id": "randomization",
                "type": "randomization",
                "label": "Randomization",
                "x": 600,
                "y": 0,
                "patient_count": int(patient_count * 0.6)
            },
            {
                "id": "intervention_first",
                "type": "intervention",
                "label": "Treatment First",
                "x": 900,
                "y": -150,
                "patient_count": int(patient_count * 0.3)
            },
            {
                "id": "control_first",
                "type": "control",
                "label": "Control First",
                "x": 900,
                "y": 150,
                "patient_count": int(patient_count * 0.3)
            },
            {
                "id": "outcome_int_first",
                "type": "outcome",
                "label": "Period 1 Outcome",
                "x": 1200,
                "y": -150,
                "patient_count": int(patient_count * 0.28)
            },
            {
                "id": "outcome_ctrl_first",
                "type": "outcome",
                "label": "Period 1 Outcome",
                "x": 1200,
                "y": 150,
                "patient_count": int(patient_count * 0.29)
            },
            {
                "id": "int_after_ctrl",
                "type": "intervention",
                "label": "Treatment (Period 2)",
                "x": 1500,
                "y": 150,
                "patient_count": int(patient_count * 0.28)
            },
            {
                "id": "ctrl_after_int",
                "type": "control",
                "label": "Control (Period 2)",
                "x": 1500,
                "y": -150,
                "patient_count": int(patient_count * 0.27)
            },
            {
                "id": "outcome_period2_int",
                "type": "outcome",
                "label": "Period 2 Outcome",
                "x": 1800,
                "y": 150,
                "patient_count": int(patient_count * 0.27)
            },
            {
                "id": "outcome_period2_ctrl",
                "type": "outcome",
                "label": "Period 2 Outcome",
                "x": 1800,
                "y": -150,
                "patient_count": int(patient_count * 0.26)
            }
        ],
        "edges": [
            {
                "source": "population",
                "target": "eligible",
                "patient_count": int(patient_count * 0.6),
                "label": "Screening"
            },
            {
                "source": "eligible",
                "target": "randomization",
                "patient_count": int(patient_count * 0.6),
                "label": "Randomized"
            },
            {
                "source": "randomization",
                "target": "intervention_first",
                "patient_count": int(patient_count * 0.3),
                "label": "Sequence AB"
            },
            {
                "source": "randomization",
                "target": "control_first",
                "patient_count": int(patient_count * 0.3),
                "label": "Sequence BA"
            },
            {
                "source": "intervention_first",
                "target": "outcome_int_first",
                "patient_count": int(patient_count * 0.28),
                "label": "Period 1 Complete"
            },
            {
                "source": "control_first",
                "target": "outcome_ctrl_first",
                "patient_count": int(patient_count * 0.29),
                "label": "Period 1 Complete"
            },
            {
                "source": "outcome_int_first",
                "target": "ctrl_after_int",
                "patient_count": int(patient_count * 0.27),
                "label": "Washout & Crossover"
            },
            {
                "source": "outcome_ctrl_first",
                "target": "int_after_ctrl",
                "patient_count": int(patient_count * 0.28),
                "label": "Washout & Crossover"
            },
            {
                "source": "ctrl_after_int",
                "target": "outcome_period2_ctrl",
                "patient_count": int(patient_count * 0.26),
                "label": "Period 2 Complete"
            },
            {
                "source": "int_after_ctrl",
                "target": "outcome_period2_int",
                "patient_count": int(patient_count * 0.27),
                "label": "Period 2 Complete"
            }
        ]
    }

def create_factorial_json(patient_count):
    """Create a factorial design JSON template."""
    return {
        "design_type": "factorial",
        "nodes": [
            {
                "id": "population",
                "type": "target_population",
                "label": "Initial Population",
                "x": 0,
                "y": 0,
                "patient_count": patient_count
            },
            {
                "id": "eligible",
                "type": "eligible_population",
                "label": "Eligible Subjects",
                "x": 300,
                "y": 0,
                "patient_count": int(patient_count * 0.6)
            },
            {
                "id": "randomization",
                "type": "randomization",
                "label": "Randomization",
                "x": 600,
                "y": 0,
                "patient_count": int(patient_count * 0.6)
            },
            {
                "id": "drug_a_b",
                "type": "intervention",
                "label": "Drug A + Drug B",
                "x": 900,
                "y": -225,
                "patient_count": int(patient_count * 0.15)
            },
            {
                "id": "drug_a_placebo",
                "type": "intervention",
                "label": "Drug A + Placebo B",
                "x": 900,
                "y": -75,
                "patient_count": int(patient_count * 0.15)
            },
            {
                "id": "placebo_a_drug_b",
                "type": "intervention",
                "label": "Placebo A + Drug B",
                "x": 900,
                "y": 75,
                "patient_count": int(patient_count * 0.15)
            },
            {
                "id": "placebo_a_b",
                "type": "control",
                "label": "Placebo A + Placebo B",
                "x": 900,
                "y": 225,
                "patient_count": int(patient_count * 0.15)
            },
            {
                "id": "outcome_drug_a_b",
                "type": "outcome",
                "label": "A+B Outcome",
                "x": 1200,
                "y": -225,
                "patient_count": int(patient_count * 0.14)
            },
            {
                "id": "outcome_drug_a_placebo",
                "type": "outcome",
                "label": "A Only Outcome",
                "x": 1200,
                "y": -75,
                "patient_count": int(patient_count * 0.145)
            },
            {
                "id": "outcome_placebo_a_drug_b",
                "type": "outcome",
                "label": "B Only Outcome",
                "x": 1200,
                "y": 75,
                "patient_count": int(patient_count * 0.14)
            },
            {
                "id": "outcome_placebo_a_b",
                "type": "outcome",
                "label": "Placebo Outcome",
                "x": 1200,
                "y": 225,
                "patient_count": int(patient_count * 0.145)
            }
        ],
        "edges": [
            {
                "source": "population",
                "target": "eligible",
                "patient_count": int(patient_count * 0.6),
                "label": "Screening"
            },
            {
                "source": "eligible",
                "target": "randomization",
                "patient_count": int(patient_count * 0.6),
                "label": "Randomized"
            },
            {
                "source": "randomization",
                "target": "drug_a_b",
                "patient_count": int(patient_count * 0.15),
                "label": "A+B Group"
            },
            {
                "source": "randomization",
                "target": "drug_a_placebo",
                "patient_count": int(patient_count * 0.15),
                "label": "A Only Group"
            },
            {
                "source": "randomization",
                "target": "placebo_a_drug_b",
                "patient_count": int(patient_count * 0.15),
                "label": "B Only Group"
            },
            {
                "source": "randomization",
                "target": "placebo_a_b",
                "patient_count": int(patient_count * 0.15),
                "label": "Placebo Group"
            },
            {
                "source": "drug_a_b",
                "target": "outcome_drug_a_b",
                "patient_count": int(patient_count * 0.14),
                "label": "Follow-up"
            },
            {
                "source": "drug_a_placebo",
                "target": "outcome_drug_a_placebo",
                "patient_count": int(patient_count * 0.145),
                "label": "Follow-up"
            },
            {
                "source": "placebo_a_drug_b",
                "target": "outcome_placebo_a_drug_b",
                "patient_count": int(patient_count * 0.14),
                "label": "Follow-up"
            },
            {
                "source": "placebo_a_b",
                "target": "outcome_placebo_a_b",
                "patient_count": int(patient_count * 0.145),
                "label": "Follow-up"
            }
        ]
    }

def create_single_arm_json(patient_count):
    """Create a single arm design JSON template."""
    return {
        "design_type": "single_arm",
        "nodes": [
            {
                "id": "population",
                "type": "target_population",
                "label": "Initial Population",
                "x": 0,
                "y": 0,
                "patient_count": patient_count
            },
            {
                "id": "eligible",
                "type": "eligible_population",
                "label": "Eligible Subjects",
                "x": 300,
                "y": 0,
                "patient_count": int(patient_count * 0.5)
            },
            {
                "id": "intervention",
                "type": "intervention",
                "label": "Treatment",
                "x": 600,
                "y": 0,
                "patient_count": int(patient_count * 0.5)
            },
            {
                "id": "outcome_early",
                "type": "outcome",
                "label": "Early Assessment",
                "x": 900,
                "y": 0,
                "patient_count": int(patient_count * 0.45)
            },
            {
                "id": "outcome_primary",
                "type": "outcome",
                "label": "Primary Outcome",
                "x": 1200,
                "y": 0,
                "patient_count": int(patient_count * 0.4)
            },
            {
                "id": "timepoint_followup",
                "type": "timepoint",
                "label": "12-Month Follow-up",
                "x": 1500,
                "y": 0,
                "patient_count": int(patient_count * 0.35)
            }
        ],
        "edges": [
            {
                "source": "population",
                "target": "eligible",
                "patient_count": int(patient_count * 0.5),
                "label": "Screening"
            },
            {
                "source": "eligible",
                "target": "intervention",
                "patient_count": int(patient_count * 0.5),
                "label": "Enrollment"
            },
            {
                "source": "intervention",
                "target": "outcome_early",
                "patient_count": int(patient_count * 0.45),
                "label": "4-Week Assessment"
            },
            {
                "source": "outcome_early",
                "target": "outcome_primary",
                "patient_count": int(patient_count * 0.4),
                "label": "8-Week Assessment"
            },
            {
                "source": "outcome_primary",
                "target": "timepoint_followup",
                "patient_count": int(patient_count * 0.35),
                "label": "Long-term Follow-up"
            }
        ]
    }

def create_prepost_json(patient_count):
    """Create a pre-post design JSON template."""
    return {
        "design_type": "pre_post",
        "nodes": [
            {
                "id": "population",
                "type": "target_population",
                "label": "Initial Population",
                "x": 0,
                "y": 0,
                "patient_count": patient_count
            },
            {
                "id": "eligible",
                "type": "eligible_population",
                "label": "Eligible Subjects",
                "x": 300,
                "y": 0,
                "patient_count": int(patient_count * 0.6)
            },
            {
                "id": "baseline",
                "type": "outcome",
                "label": "Baseline Assessment",
                "x": 600,
                "y": 0,
                "patient_count": int(patient_count * 0.6)
            },
            {
                "id": "intervention",
                "type": "intervention",
                "label": "Intervention",
                "x": 900,
                "y": 0,
                "patient_count": int(patient_count * 0.6)
            },
            {
                "id": "post_assessment",
                "type": "outcome",
                "label": "Post-intervention Assessment",
                "x": 1200,
                "y": 0,
                "patient_count": int(patient_count * 0.55)
            },
            {
                "id": "followup",
                "type": "timepoint",
                "label": "6-Month Follow-up",
                "x": 1500,
                "y": 0,
                "patient_count": int(patient_count * 0.5)
            }
        ],
        "edges": [
            {
                "source": "population",
                "target": "eligible",
                "patient_count": int(patient_count * 0.6),
                "label": "Screening"
            },
            {
                "source": "eligible",
                "target": "baseline",
                "patient_count": int(patient_count * 0.6),
                "label": "Enrollment"
            },
            {
                "source": "baseline",
                "target": "intervention",
                "patient_count": int(patient_count * 0.6),
                "label": "Baseline Complete"
            },
            {
                "source": "intervention",
                "target": "post_assessment",
                "patient_count": int(patient_count * 0.55),
                "label": "Intervention Complete"
            },
            {
                "source": "post_assessment",
                "target": "followup",
                "patient_count": int(patient_count * 0.5),
                "label": "Post Assessment Complete"
            }
        ]
    }

def create_case_control_json(patient_count):
    """Create a case-control study JSON template."""
    cases = int(patient_count * 0.3)
    controls = int(patient_count * 0.7)
    
    return {
        "design_type": "case_control",
        "nodes": [
            {
                "id": "cases",
                "type": "target_population",
                "label": "Cases",
                "x": 0,
                "y": -150,
                "patient_count": cases
            },
            {
                "id": "controls",
                "type": "target_population",
                "label": "Controls",
                "x": 0,
                "y": 150,
                "patient_count": controls
            },
            {
                "id": "eligible_cases",
                "type": "eligible_population",
                "label": "Eligible Cases",
                "x": 300,
                "y": -150,
                "patient_count": int(cases * 0.9)
            },
            {
                "id": "eligible_controls",
                "type": "eligible_population",
                "label": "Eligible Controls",
                "x": 300,
                "y": 150,
                "patient_count": int(controls * 0.9)
            },
            {
                "id": "exposure_assessment_cases",
                "type": "outcome",
                "label": "Exposure Assessment",
                "x": 600,
                "y": -150,
                "patient_count": int(cases * 0.9)
            },
            {
                "id": "exposure_assessment_controls",
                "type": "outcome",
                "label": "Exposure Assessment",
                "x": 600,
                "y": 150,
                "patient_count": int(controls * 0.9)
            },
            {
                "id": "analysis",
                "type": "outcome",
                "label": "Statistical Analysis",
                "x": 900,
                "y": 0,
                "patient_count": int(cases * 0.9 + controls * 0.9)
            }
        ],
        "edges": [
            {
                "source": "cases",
                "target": "eligible_cases",
                "patient_count": int(cases * 0.9),
                "label": "Case Selection"
            },
            {
                "source": "controls",
                "target": "eligible_controls",
                "patient_count": int(controls * 0.9),
                "label": "Control Selection"
            },
            {
                "source": "eligible_cases",
                "target": "exposure_assessment_cases",
                "patient_count": int(cases * 0.9),
                "label": "Data Collection"
            },
            {
                "source": "eligible_controls",
                "target": "exposure_assessment_controls",
                "patient_count": int(controls * 0.9),
                "label": "Data Collection"
            },
            {
                "source": "exposure_assessment_cases",
                "target": "analysis",
                "patient_count": int(cases * 0.9),
                "label": "Case Data"
            },
            {
                "source": "exposure_assessment_controls",
                "target": "analysis",
                "patient_count": int(controls * 0.9),
                "label": "Control Data"
            }
        ]
    }
