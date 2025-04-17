import math
import json
from dataclasses import dataclass
import uuid

from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsObject, QGraphicsItem, QGraphicsTextItem, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QCheckBox,
    QGraphicsDropShadowEffect, QSplitter,
    QFrame, QGroupBox, QGridLayout, QFileDialog, QMessageBox, QInputDialog, QWidget, QSlider, QGraphicsLineItem
)
from PyQt6.QtGui import (
    QPen, QBrush, QColor, QFont, QPainterPath, QPainter, QRadialGradient
)
from PyQt6.QtCore import (
    Qt, QPointF, QRectF, pyqtSignal, QPropertyAnimation, QEasingCurve, 
    QTimer, QLineF
)


# First add these imports at the top with other imports
from helpers.load_icon import load_bootstrap_icon
from plan.plan_config import NODE_CONFIGS, ConnectionType, Evidence, EvidenceSourceType, HypothesisConfig, HypothesisState, ObjectiveConfig, ObjectiveType
from plan.plan_dialogs import HypothesisEditorDialog, ObjectiveEditorDialog
from plan.plan_nodes import BaseNode, EvidencePanel, HypothesisNode, NodeConnection, NodePortItem, ObjectiveNode

# ======================
# Grid Position
# ======================

@dataclass
class GridPosition:
    """Position in the grid system"""
    row: int
    column: int
    
    def __eq__(self, other):
        if not isinstance(other, GridPosition):
            return False
        return self.row == other.row and self.column == other.column
    
    def __hash__(self):
        return hash((self.row, self.column))

# ======================
# Research Grid Scene
# ======================

class ResearchGridScene(QGraphicsScene):
    """Grid-based scene for managing research objectives and hypotheses"""
    
    # Signals
    nodeActivated = pyqtSignal(object)
    nodeSelected = pyqtSignal(object)
    gridChanged = pyqtSignal()
    portClicked = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Grid settings
        self.grid_size = 120
        self.grid_width = 36
        self.grid_height = 20
        self.grid_visible = True
        self.nodes_grid = {}  # Map GridPosition to node
        self.snap_to_lanes = True
        
        # Minimum distance between nodes (to prevent clustering)
        self.minimum_node_distance = 150
        
        # Lane tracking
        self.vertical_lanes = []
        self.horizontal_lanes = []
        self.lanes = []  # All lanes (required by clear_lanes method)
        
        # Connection state
        self.drawing_connection = False
        self.start_port = None
        self.temp_line = None
        self.nearest_port = None
        self.snap_distance = 50
        self.recently_auto_arranged = False
        
        # Theme
        self.current_theme = "light"
        self.background_colors = {
            "light": QColor(250, 250, 250),
            "dark": QColor(33, 33, 33)
        }
        self.grid_colors = {
            "light": QColor(190, 190, 190),
            "dark": QColor(140, 140, 140)
        }
        
        # Initialize the grid
        self.initialize_grid()
        
        # Connect selection change
        self.selectionChanged.connect(self.on_selection_changed)
    
    def initialize_grid(self):
        """Initialize the grid layout"""
        self.setBackgroundBrush(QBrush(self.background_colors[self.current_theme]))
        self.nodes_grid = {}
    
    def draw_grid_lines(self):
        """Draw grid lines with current theme color"""
        # Remove existing grid lines
        items_to_remove = []
        for item in self.items():
            if hasattr(item, 'is_grid_line'):
                items_to_remove.append(item)
        
        for item in items_to_remove:
            self.removeItem(item)
        
        # If grid is not visible, return
        if not self.grid_visible:
            return
            
        # Create grid lines with theme color
        grid_pen = QPen(self.grid_colors[self.current_theme], 1, Qt.PenStyle.SolidLine)
        
        # Get scene boundaries
        rect = self.sceneRect()
        left = int(rect.left()) - (int(rect.left()) % self.grid_size)
        top = int(rect.top()) - (int(rect.top()) % self.grid_size)
        
        # Draw vertical lines
        for x in range(left, int(rect.right()), self.grid_size):
            line = self.addLine(x, rect.top(), x, rect.bottom(), grid_pen)
            line.is_grid_line = True
            line.setZValue(-2)
        
        # Draw horizontal lines
        for y in range(top, int(rect.bottom()), self.grid_size):
            line = self.addLine(rect.left(), y, rect.right(), y, grid_pen)
            line.is_grid_line = True
            line.setZValue(-2)
    
    def row_to_y(self, row):
        """Convert grid row to Y coordinate"""
        return row * self.grid_size
    
    def col_to_x(self, col):
        """Convert grid column to X coordinate"""
        return col * self.grid_size
    
    def get_cell_position(self, grid_pos):
        """Get cell center position from grid position"""
        return QPointF(
            self.col_to_x(grid_pos.column) + self.grid_size/2,
            self.row_to_y(grid_pos.row) + self.grid_size/2
        )
    
    def expand_grid(self, min_row, max_row, min_col, max_col):
        """Expand grid to accommodate new areas"""
        # Get current scene rect
        rect = self.sceneRect()
        
        # Calculate needed dimensions
        left = min(rect.left(), self.col_to_x(min_col) - self.grid_size)
        top = min(rect.top(), self.row_to_y(min_row) - self.grid_size)
        right = max(rect.right(), self.col_to_x(max_col) + self.grid_size * 2)
        bottom = max(rect.bottom(), self.row_to_y(max_row) + self.grid_size * 2)
        
        # Set new scene rect
        self.setSceneRect(left, top, right - left, bottom - top)
        
        # Redraw grid lines
        self.draw_grid_lines()
        
        # Emit signal
        self.gridChanged.emit()
    
    def add_node(self, node_type, grid_pos, config=None):
        """Add a node at a specific grid position"""
        
        # Calculate pixel position
        pos = self.get_cell_position(grid_pos)
        
        # Check if we need to expand the grid
        self.expand_grid(grid_pos.row, grid_pos.row, grid_pos.column, grid_pos.column)
        
        # Create node based on type
        if node_type == "objective":
            if not config:
                # Create default objective configuration
                config = ObjectiveConfig(
                    id=str(uuid.uuid4()),
                    text="New Research Question",
                    type=ObjectiveType.RESEARCH_QUESTION,
                )
            node = ObjectiveNode(config, pos.x(), pos.y())
            
        elif node_type == "hypothesis":
            if not config:
                # Create default hypothesis configuration
                config = HypothesisConfig(
                    id=str(uuid.uuid4()),
                    text="New Hypothesis",
                    state=HypothesisState.PROPOSED,
                )
            node = HypothesisNode(config, pos.x(), pos.y())
        
        else:
            return None
        
        # Add node to scene
        self.addItem(node)
        self.nodes_grid[grid_pos] = node
        
        # Connect signals
        node.nodeActivated.connect(self.nodeActivated)
        node.nodeSelected.connect(self.nodeSelected)
        node.portClicked.connect(self.on_port_clicked)
        
        # Check for collisions with existing nodes
        self.prevent_node_collision(node)
        
        # Trigger scene change
        self.gridChanged.emit()
        
        return node
    
    def create_connection(self, start_port, end_port, connection_type=ConnectionType.CONTRIBUTES_TO):
        """Create a connection between two ports"""
        # Validate ports
        if not start_port or not end_port or start_port == end_port:
            return None

        start_node = start_port.parentItem()
        end_node = end_port.parentItem()

        # Do not connect a node to itself
        if start_node == end_node:
            return None

        # Helper function to find directly connected objectives
        def find_directly_connected_objectives(node):
            objectives = []
            all_ports = node.input_ports + getattr(node, 'output_ports', [])
            for port in all_ports:
                for link in port.connected_links:
                    # Find the other port in the connection
                    other_port = link.start_port if link.end_port.parentItem() == node else link.end_port
                    other_node = other_port.parentItem()
                    if other_node.node_type == "objective":
                        objectives.append(other_node)
            return objectives
        
        # Helper function to recursively find all connected nodes of a type
        def find_all_connected_nodes_of_type(node, node_type, visited=None):
            if visited is None:
                visited = set()
            
            if node in visited:
                return []
            
            visited.add(node)
            result = []
            
            all_ports = node.input_ports + getattr(node, 'output_ports', [])
            for port in all_ports:
                for link in port.connected_links:
                    # Find the other node
                    other_port = link.start_port if link.end_port.parentItem() == node else link.end_port
                    other_node = other_port.parentItem()
                    
                    if other_node.node_type == node_type:
                        result.append(other_node)
                    
                    if other_node.node_type == "hypothesis":  # Only follow hypothesis chains
                        result.extend(find_all_connected_nodes_of_type(other_node, node_type, visited))
            
            return result
        
        # RULE 1: Two hypotheses connected to the same objective cannot connect
        if start_node.node_type == "hypothesis" and end_node.node_type == "hypothesis":
            # Get direct objectives for both hypotheses
            start_objectives = find_directly_connected_objectives(start_node)
            end_objectives = find_directly_connected_objectives(end_node)
            
            # If they share a common objective, don't allow connection
            if set(start_objectives) & set(end_objectives):
                return None
            
            # RULE 2: A hypothesis chain can only connect to one objective
            # If both have objectives in their chains and they're different, don't allow connection
            start_chain_objectives = find_all_connected_nodes_of_type(start_node, "objective")
            end_chain_objectives = find_all_connected_nodes_of_type(end_node, "objective")
            
            if start_chain_objectives and end_chain_objectives and set(start_chain_objectives) != set(end_chain_objectives):
                return None
        
        # RULE 2 (continued): Hypothesis-objective connections
        elif (start_node.node_type == "hypothesis" and end_node.node_type == "objective") or \
             (start_node.node_type == "objective" and end_node.node_type == "hypothesis"):
            
            hypothesis_node = end_node if end_node.node_type == "hypothesis" else start_node
            objective_node = start_node if start_node.node_type == "objective" else end_node
            
            # Check if hypothesis chain already has a different objective
            existing_objectives = find_all_connected_nodes_of_type(hypothesis_node, "objective")
            if existing_objectives and objective_node not in existing_objectives:
                return None
        
        # RULE 3: For objective-objective connections
        if start_node.node_type == "objective" and end_node.node_type == "objective":
            # Get objective types
            start_type = start_node.objective_config.type
            end_type = end_node.objective_config.type
            
            # Only allow connections from goals to research questions
            valid_connection = (
                (start_type == ObjectiveType.GOAL and end_type == ObjectiveType.RESEARCH_QUESTION) or
                (start_type == ObjectiveType.RESEARCH_QUESTION and end_type == ObjectiveType.GOAL)
            )
            
            if not valid_connection:
                return None

        # Check port compatibility
        if start_port.port_type == "output" and hasattr(start_port, "allowed_connections"):
            if end_node.node_type not in start_port.allowed_connections:
                return None
                
        if end_port.port_type == "input" and hasattr(end_port, "allowed_connections"):
            if start_node.node_type not in end_port.allowed_connections:
                return None

        # Create the connection
        connection = NodeConnection(start_port, end_port, connection_type)
        self.addItem(connection)
        return connection
    
    def on_port_clicked(self, port):
        """Handle port click for connection creation"""
        # If not already drawing a connection, start drawing
        if not self.drawing_connection:
            self.start_port = port
            self.drawing_connection = True
            
            # Create temporary line
            self.temp_line = QGraphicsLineItem()
            self.temp_line.setPen(QPen(QColor(100, 100, 255, 180), 2, Qt.PenStyle.DashLine))
            self.addItem(self.temp_line)
            
        else:
            # Finish drawing connection
            if self.start_port and port != self.start_port:
                # Determine start and end ports
                start_port = self.start_port
                end_port = port
                
                # If start is input and end is output, swap
                if start_port.port_type == "input" and end_port.port_type == "output":
                    start_port, end_port = end_port, start_port
                
                # Create connection
                self.create_connection(start_port, end_port)
            
            # Clean up
            if self.temp_line:
                self.removeItem(self.temp_line)
                self.temp_line = None
            
            self.drawing_connection = False
            self.start_port = None
            self.nearest_port = None
    
    def mouseMoveEvent(self, event):
        """Handle temporary connection lines and node movement"""
        super().mouseMoveEvent(event)
        
        # If we're dragging a temporary connection
        if hasattr(self, 'drawing_connection') and self.drawing_connection and hasattr(self, 'start_port'):
            if not hasattr(self, 'temp_line'):
                self.temp_line = QGraphicsLineItem()
                self.temp_line.setPen(QPen(QColor(120, 120, 120), 2, Qt.PenStyle.DashLine))
                self.addItem(self.temp_line)
            
            # Get the current mouse position
            mouse_pos = event.scenePos()
            
            # If snapping is enabled, try to snap to nearest compatible port
            if self.snap_to_lanes:
                self.find_nearest_compatible_port(mouse_pos)
                if self.nearest_port:
                    # Snap to the port if it's close enough
                    end_pos = self.nearest_port.get_scene_position()
                    self.temp_line.setPen(QPen(QColor(39, 174, 96), 2, Qt.PenStyle.DashLine))
                else:
                    end_pos = mouse_pos
                    self.temp_line.setPen(QPen(QColor(120, 120, 120), 2, Qt.PenStyle.DashLine))
            else:
                end_pos = mouse_pos
            
            # Update the temp line
            start_pos = self.start_port.get_scene_position()
            self.temp_line.setLine(start_pos.x(), start_pos.y(), end_pos.x(), end_pos.y())
            
        # Handle lane highlighting for moving nodes
        if self.snap_to_lanes and self.lanes and not self.recently_auto_arranged:
            try:
                # Reset lane highlighting
                valid_lanes = []
                for lane in list(self.lanes):
                    try:
                        if lane.scene() == self:
                            lane.is_hovered = False
                            valid_lanes.append(lane)
                    except:
                        continue
                
                # Check selected nodes
                for item in self.selectedItems():
                    if isinstance(item, BaseNode):
                        # Get item center
                        item_center = item.mapToScene(item.boundingRect().center())
                        
                        # Find matching lane
                        matching_lane = None
                        min_distance = float('inf')
                        
                        for lane in valid_lanes:
                            # Check if lane matches node type
                            if lane.label:
                                node_type_str = item.node_type.lower()
                                if node_type_str in lane.label.lower():
                                    # Calculate distance
                                    lane_pos = lane.scenePos()
                                    if lane.orientation == "horizontal":
                                        distance = abs(item_center.y() - lane_pos.y())
                                    else:  # vertical
                                        distance = abs(item_center.x() - lane_pos.x())
                                        
                                    if distance < min_distance:
                                        min_distance = distance
                                        matching_lane = lane
                        
                        # Highlight if close enough
                        if matching_lane and min_distance <= matching_lane.snap_margin / 2:
                            matching_lane.is_hovered = True
                            item.is_active = True
                            
                            # Add glow effect
                            if not item.graphicsEffect():
                                shadow = QGraphicsDropShadowEffect()
                                shadow.setBlurRadius(20)
                                shadow.setColor(QColor(matching_lane.color.red(), 
                                                     matching_lane.color.green(),
                                                     matching_lane.color.blue(), 
                                                     180))
                                shadow.setOffset(0, 0)
                                item.setGraphicsEffect(shadow)
                        else:
                            item.is_active = False
                            item.setGraphicsEffect(None)
                        
                        item.update()
                        item.update_connected_links()
                
                # Update lanes
                for lane in valid_lanes:
                    lane.update()
                    
            except Exception as e:
                print(f"Error in lane processing: {e}")
        
        # Handle node dragging and collision prevention
        moving_item = self.mouseGrabberItem()
        if moving_item and isinstance(moving_item, BaseNode):
            self.prevent_node_collision(moving_item)
    
    def find_nearest_compatible_port(self, pos):
        """Find the nearest compatible port for connection"""
        self.nearest_port = None
        nearest_distance = float('inf')
        
        if not self.drawing_connection or not self.start_port:
            return
        
        # Find compatible ports
        for item in self.items():
            if isinstance(item, NodePortItem) and item != self.start_port:
                # Skip if port types are the same
                if item.port_type == self.start_port.port_type:
                    continue
                
                # Get parent nodes
                start_node = self.start_port.parentItem()
                end_node = item.parentItem()
                
                # Skip if connecting to same node
                if start_node == end_node:
                    continue
                
                # Check compatibility
                can_connect = False
                
                if self.start_port.port_type == "output" and item.port_type == "input":
                    if hasattr(self.start_port, "allowed_connections") and hasattr(item, "allowed_connections"):
                        can_connect = end_node.node_type in self.start_port.allowed_connections
                elif self.start_port.port_type == "input" and item.port_type == "output":
                    if hasattr(self.start_port, "allowed_connections") and hasattr(item, "allowed_connections"):
                        can_connect = start_node.node_type in item.allowed_connections
                
                if not can_connect:
                    continue
                
                # Calculate distance
                port_pos = item.get_scene_position()
                dx = port_pos.x() - pos.x()
                dy = port_pos.y() - pos.y()
                distance = math.sqrt(dx*dx + dy*dy)
                
                # Update if closer
                if distance <= self.snap_distance and distance < nearest_distance:
                    nearest_distance = distance
                    self.nearest_port = item
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release for edge creation and lane snapping"""
        if not self.drawing_connection:
            # Snap nodes to lanes
            nodes_to_snap = []
            
            for item in self.selectedItems():
                if isinstance(item, BaseNode):
                    nodes_to_snap.append(item)
            
            for node in nodes_to_snap:
                try:
                    # Get node center
                    node_center = node.mapToScene(node.boundingRect().center())
                    
                    if self.snap_to_lanes and self.lanes:
                        # Find best matching lane
                        best_lane = None
                        min_distance = float('inf')
                        
                        for lane in self.lanes:
                            if not lane.scene() == self:
                                continue
                            
                            if lane.label and node.node_type.lower() in lane.label.lower():
                                # Calculate distance
                                lane_pos = lane.scenePos()
                                if lane.orientation == "horizontal":
                                    distance = abs(node_center.y() - lane_pos.y())
                                else:  # vertical
                                    distance = abs(node_center.x() - lane_pos.x())
                                    
                                if distance < min_distance:
                                    min_distance = distance
                                    best_lane = lane
                        
                        # Snap if close enough
                        if best_lane and min_distance <= best_lane.snap_margin / 2:
                            snap_pos = best_lane.get_snap_position(node.pos(), node.boundingRect())
                            
                            # Animate snap
                            if not hasattr(node, 'snap_animation'):
                                node.snap_animation = QPropertyAnimation(node, b"pos")
                                node.snap_animation.setDuration(200)
                                node.snap_animation.setEasingCurve(QEasingCurve.Type.OutQuad)
                            
                            node.snap_animation.setStartValue(node.pos())
                            node.snap_animation.setEndValue(snap_pos)
                            
                            # Update links when animation finishes
                            def update_node_links():
                                node.update_connected_links()
                            
                            node.snap_animation.finished.connect(update_node_links)
                            node.snap_animation.start()
                            
                            # Update grid position
                            row = int((snap_pos.y() + node.boundingRect().height()/2) / self.grid_size)
                            col = int((snap_pos.x() + node.boundingRect().width()/2) / self.grid_size)
                            
                            new_grid_pos = GridPosition(row=row, column=col)
                            
                            # Update node map
                            if hasattr(node, 'grid_position') and node.grid_position in self.nodes_grid:
                                if self.nodes_grid[node.grid_position] == node:
                                    del self.nodes_grid[node.grid_position]
                            
                            self.nodes_grid[new_grid_pos] = node
                            node.grid_position = new_grid_pos
                    
                    # Reset appearance
                    node.is_active = False
                    node.setGraphicsEffect(None)
                    node.update()
                    
                    # Update links
                    node.update_connected_links()
                except RuntimeError:
                    pass
        
        # Update all links
        try:
            for item in self.items():
                if isinstance(item, NodeConnection):
                    item.update_path()
        except:
            pass
        
        # Reset lane highlights
        try:
            for lane in list(self.lanes):
                if lane.scene() == self:
                    lane.is_hovered = False
                    lane.update()
        except:
            pass
    
        super().mouseReleaseEvent(event)
    
    def on_selection_changed(self):
        """Handle selection changes"""
        selected_nodes = []
        
        for item in self.selectedItems():
            if isinstance(item, BaseNode):
                selected_nodes.append(item)
        
        if selected_nodes:
            self.nodeSelected.emit(selected_nodes[0])
    
    def on_node_collapse_toggled(self, node):
        """Handle node collapse/expand"""
        if node.node_type == "objective":
            # Find all hypotheses connected to this objective
            children = self.get_hypotheses_for_objective(node)
            
            # Toggle visibility
            for child in children:
                child.setVisible(not node.is_collapsed)
                
                # Also hide any connections to this child
                for port in child.input_ports + child.output_ports:
                    for link in port.connected_links:
                        if node.is_collapsed:
                            # Only hide if both nodes aren't visible
                            other_node = link.start_node if link.end_node == child else link.end_node
                            if not other_node.isVisible():
                                link.setVisible(False)
                        else:
                            link.setVisible(True)
    
    def get_hypotheses_for_objective(self, objective_node):
        """Get all hypotheses connected to an objective"""
        hypotheses = []
        
        # Check all output ports
        for port in objective_node.output_ports:
            for link in port.connected_links:
                end_node = link.end_node
                
                if end_node.node_type == "hypothesis":
                    hypotheses.append(end_node)
        
        return hypotheses
    
    def toggle_grid_lines(self):
        """Toggle grid lines visibility"""
        self.grid_visible = not self.grid_visible
        self.draw_grid_lines()
        return self.grid_visible
    
    def toggle_snap_to_lanes(self):
        """Toggle lane snapping"""
        self.snap_to_lanes = not self.snap_to_lanes
        return self.snap_to_lanes
    
    def remove_node(self, grid_pos):
        """Remove a node from the grid"""
        if grid_pos not in self.nodes_grid:
            return False
        
        node = self.nodes_grid[grid_pos]
        node_id = node.node_data['id']
        
        # Delete node
        node.delete_node()
        
        # Remove from maps
        del self.nodes_grid[grid_pos]
        
        if node_id in self.nodes_grid:
            del self.nodes_grid[node_id]
        
        return True
    
    def update_theme(self, theme):
        """Update the theme for the grid and components"""
        self.current_theme = theme
        
        # Update background and grid colors
        self.setBackgroundBrush(QBrush(self.background_colors[theme]))
        
        # Redraw grid lines
        if self.grid_visible:
            self.draw_grid_lines()
        
        # Update lanes
        for lane in list(self.lanes):
            try:
                if lane.scene() == self:
                    lane.update_theme(theme)
            except:
                pass
        
        # Update nodes
        for node in list(self.nodes_grid.values()):
            try:
                if hasattr(node, 'update_theme'):
                    node.update_theme(theme)
            except:
                pass
        
        # Update connections
        for item in self.items():
            try:
                if isinstance(item, NodeConnection):
                    item.update_theme(theme)
            except:
                pass


    def create_type_lanes(self):
        """Create vertical lanes for each node type"""
        self.clear_lanes()
        
        # Get scene center and width
        scene_rect = self.sceneRect()
        scene_center_x = scene_rect.center().x()
        
        # Position lanes relative to center
        positions = [
            scene_center_x - 300,  # Objectives
            scene_center_x + 300,  # Hypotheses
        ]
        
        node_types = [
            "objective",
            "hypothesis"
        ]
        
        # Create a lane for each node type
        for i, node_type in enumerate(node_types):
            # Get color based on node type
            base_color = QColor(NODE_CONFIGS[node_type].color)
            lane_color = QColor(
                min(255, int(base_color.red() * 0.9 + 255 * 0.1)),
                min(255, int(base_color.green() * 0.9 + 255 * 0.1)),
                min(255, int(base_color.blue() * 0.9 + 255 * 0.1)),
                120
            )
            
            # Create the lane
            x_pos = positions[i]
            lane_label = f"{node_type.title()}"
            lane = self.add_vertical_lane(x_pos, lane_color, lane_label)
        
        return True

    def clear_lanes(self):
        """Remove all lanes"""
        # Create a safe copy of the lanes list
        lanes_to_remove = list(self.lanes)
        
        # Reset the lanes lists
        self.lanes = []
        self.vertical_lanes = []
        self.horizontal_lanes = []
        
        # Remove each lane safely
        for lane in lanes_to_remove:
            try:
                if lane.scene() == self:
                    self.removeItem(lane)
            except:
                continue

    def add_vertical_lane(self, x_position, color=None, label=""):
        """Add a vertical lane at the specified position"""
        lane = ResearchLane(
            orientation="vertical", 
            position=x_position, 
            color=color, 
            label=label,
            theme=self.current_theme
        )
        lane.setPos(x_position, 0)
        self.addItem(lane)
        self.vertical_lanes.append(lane)
        self.lanes.append(lane)
        return lane

    def add_horizontal_lane(self, y_position, color=None, label=""):
        """Add a horizontal lane at the specified position"""
        lane = ResearchLane(
            orientation="horizontal", 
            position=y_position, 
            color=color, 
            label=label,
            theme=self.current_theme
        )
        lane.setPos(0, y_position)
        self.addItem(lane)
        self.horizontal_lanes.append(lane)
        self.lanes.append(lane)
        return lane

    def auto_sort_nodes(self):
        """Automatically sort and arrange nodes by their type and connectivity"""
        # Get lanes by type for positioning
        lanes_by_type = {}
        for lane in self.lanes:
            if lane.label:
                lane_label = lane.label.lower()
                for node_type in ["objective", "hypothesis"]:
                    if node_type in lane_label:
                        lanes_by_type[node_type] = lane
                        break

        # Abort if we don't have lanes for objectives and hypotheses
        if "objective" not in lanes_by_type or "hypothesis" not in lanes_by_type:
            # Recreate lanes if needed
            self.create_type_lanes()
            
            # Try getting lanes again
            lanes_by_type = {}
            for lane in self.lanes:
                if lane.label:
                    lane_label = lane.label.lower()
                    for node_type in ["objective", "hypothesis"]:
                        if node_type in lane_label:
                            lanes_by_type[node_type] = lane
                            break
                            
            # If still missing lanes, abort
            if "objective" not in lanes_by_type or "hypothesis" not in lanes_by_type:
                return False

        # Get lane positions
        objective_lane_x = lanes_by_type["objective"].scenePos().x()
        hypothesis_lane_x = lanes_by_type["hypothesis"].scenePos().x()
        
        # Group nodes by type, filtering out invalid nodes
        objectives = []
        hypotheses = []
        
        # Create a copy of the nodes_grid values to safely iterate
        nodes_grid_copy = list(self.nodes_grid.values())
        
        for node in nodes_grid_copy:
            # Check if node is still valid
            try:
                # Try to access a property of the node to see if it's still valid
                node_type = node.node_type
                if node_type == "objective":
                    objectives.append(node)
                elif node_type == "hypothesis":
                    hypotheses.append(node)
            except (RuntimeError, AttributeError, ReferenceError):
                # If node is invalid (C++ object deleted), try to remove it from the grid
                try:
                    # Find the key for this node and remove it
                    keys_to_remove = [pos for pos, n in self.nodes_grid.items() if n == node]
                    for pos in keys_to_remove:
                        if pos in self.nodes_grid:
                            del self.nodes_grid[pos]
                except:
                    pass
                continue
        
        # First find main objectives (those without parent)
        main_objectives = []
        sub_objectives = []
        for obj in objectives:
            try:
                is_sub = False
                for port in obj.input_ports:
                    for link in port.connected_links:
                        if link.start_node.node_type == "objective":
                            is_sub = True
                            break
                    if is_sub:
                        break
                
                if is_sub:
                    sub_objectives.append(obj)
                else:
                    main_objectives.append(obj)
            except (RuntimeError, AttributeError, ReferenceError):
                # Skip if node has become invalid
                continue
        
        # Find research questions (main objectives with type "research_question")
        research_questions = []
        other_main_objectives = []
        for obj in main_objectives:
            try:
                if obj.objective_config.type == ObjectiveType.RESEARCH_QUESTION:
                    research_questions.append(obj)
                else:
                    other_main_objectives.append(obj)
            except (RuntimeError, AttributeError, ReferenceError):
                # Skip if node has become invalid
                continue
        
        # Base spacing - increased for better vertical separation
        vertical_spacing = 140  # Increased spacing between objectives
        vertical_spacing_hypotheses = 140  # Increased spacing between hypotheses
        
        # Start vertical position
        y_position = -1000
        
        # Track arranged nodes
        arranged_nodes = set()
        
        # First arrange research questions and their hypotheses
        for question in research_questions:
            try:
                # First verify the node is still valid
                if question.scene() != self:
                    continue
                    
                # Position research question
                question.setPos(objective_lane_x - question.boundingRect().width() / 2, y_position)
                arranged_nodes.add(question)
                
                # Update grid position
                question_grid_pos = GridPosition(
                    row=int(y_position / self.grid_size),
                    column=int(objective_lane_x / self.grid_size)
                )
                
                # Update node map
                if hasattr(question, 'grid_position') and question.grid_position in self.nodes_grid:
                    del self.nodes_grid[question.grid_position]
                
                self.nodes_grid[question_grid_pos] = question
                question.grid_position = question_grid_pos
                
                # Find hypotheses directly connected to this research question
                question_hypotheses = []
                for port in question.output_ports:
                    for link in port.connected_links:
                        try:
                            if link.end_node and link.end_node.scene() == self and link.end_node.node_type == "hypothesis":
                                question_hypotheses.append(link.end_node)
                        except (RuntimeError, AttributeError, ReferenceError):
                            # Skip if node has become invalid
                            continue
                
                # Arrange hypotheses connected to this research question
                hyp_y = y_position
                for hyp in question_hypotheses:
                    try:
                        # Verify the node is still valid
                        if hyp.scene() != self:
                            continue
                            
                        # Position hypothesis on its lane at the same height as the research question
                        hyp.setPos(hypothesis_lane_x - hyp.boundingRect().width() / 2, hyp_y)
                        arranged_nodes.add(hyp)
                        
                        # Update grid position
                        hyp_grid_pos = GridPosition(
                            row=int(hyp_y / self.grid_size),
                            column=int(hypothesis_lane_x / self.grid_size)
                        )
                        
                        # Update node map
                        if hasattr(hyp, 'grid_position') and hyp.grid_position in self.nodes_grid:
                            del self.nodes_grid[hyp.grid_position]
                        
                        self.nodes_grid[hyp_grid_pos] = hyp
                        hyp.grid_position = hyp_grid_pos
                        
                        # Move to next position with increased spacing
                        hyp_y += vertical_spacing_hypotheses
                    except (RuntimeError, AttributeError, ReferenceError):
                        # Skip if node has become invalid
                        continue
                
                # Update y_position for next research question with increased spacing
                y_position = max(y_position + vertical_spacing, hyp_y + vertical_spacing)
            except (RuntimeError, AttributeError, ReferenceError):
                # Skip if node has become invalid
                continue
        
        # Then arrange other main objectives, their sub-objectives and hypotheses
        for main_obj in other_main_objectives:
            try:
                # Verify the node is still valid
                if main_obj.scene() != self:
                    continue
                    
                # Position main objective
                main_obj.setPos(objective_lane_x - main_obj.boundingRect().width() / 2, y_position)
                arranged_nodes.add(main_obj)
                
                # Update grid position
                main_grid_pos = GridPosition(
                    row=int(y_position / self.grid_size),
                    column=int(objective_lane_x / self.grid_size)
                )
                
                # Update node map
                if hasattr(main_obj, 'grid_position') and main_obj.grid_position in self.nodes_grid:
                    del self.nodes_grid[main_obj.grid_position]
                
                self.nodes_grid[main_grid_pos] = main_obj
                main_obj.grid_position = main_grid_pos
                
                # Find connected sub-objectives
                connected_subs = []
                for port in main_obj.output_ports:
                    for link in port.connected_links:
                        try:
                            if link.end_node and link.end_node.scene() == self and link.end_node.node_type == "objective":
                                connected_subs.append(link.end_node)
                        except (RuntimeError, AttributeError, ReferenceError):
                            # Skip if node has become invalid
                            continue
                
                # Find hypotheses directly connected to main objective
                main_hypotheses = []
                for port in main_obj.output_ports:
                    for link in port.connected_links:
                        try:
                            if link.end_node and link.end_node.scene() == self and link.end_node.node_type == "hypothesis":
                                main_hypotheses.append(link.end_node)
                        except (RuntimeError, AttributeError, ReferenceError):
                            # Skip if node has become invalid
                            continue
                
                # Arrange hypotheses connected to this main objective
                hyp_y = y_position
                for hyp in main_hypotheses:
                    try:
                        # Verify the node is still valid
                        if hyp.scene() != self:
                            continue
                            
                        # Position hypothesis on its lane at same height as main objective
                        hyp.setPos(hypothesis_lane_x - hyp.boundingRect().width() / 2, hyp_y)
                        arranged_nodes.add(hyp)
                        
                        # Update grid position
                        hyp_grid_pos = GridPosition(
                            row=int(hyp_y / self.grid_size),
                            column=int(hypothesis_lane_x / self.grid_size)
                        )
                        
                        # Update node map
                        if hasattr(hyp, 'grid_position') and hyp.grid_position in self.nodes_grid:
                            del self.nodes_grid[hyp.grid_position]
                        
                        self.nodes_grid[hyp_grid_pos] = hyp
                        hyp.grid_position = hyp_grid_pos
                        
                        # Move to next position with increased spacing
                        hyp_y += vertical_spacing_hypotheses
                    except (RuntimeError, AttributeError, ReferenceError):
                        # Skip if node has become invalid
                        continue
                
                # Current vertical position for sub-objectives with increased spacing
                sub_y = max(y_position + vertical_spacing, hyp_y)
                
                # Arrange sub-objectives if any
                for sub_obj in connected_subs:
                    try:
                        # Verify the node is still valid
                        if sub_obj.scene() != self:
                            continue
                            
                        # Position sub-objective on same lane
                        sub_obj.setPos(objective_lane_x - sub_obj.boundingRect().width() / 2, sub_y)
                        arranged_nodes.add(sub_obj)
                        
                        # Update grid position
                        sub_grid_pos = GridPosition(
                            row=int(sub_y / self.grid_size),
                            column=int(objective_lane_x / self.grid_size)
                        )
                        
                        # Update node map
                        if hasattr(sub_obj, 'grid_position') and sub_obj.grid_position in self.nodes_grid:
                            del self.nodes_grid[sub_obj.grid_position]
                        
                        self.nodes_grid[sub_grid_pos] = sub_obj
                        sub_obj.grid_position = sub_grid_pos
                        
                        # Find hypotheses connected to this sub-objective
                        sub_hypotheses = []
                        for port in sub_obj.output_ports:
                            for link in port.connected_links:
                                try:
                                    if link.end_node and link.end_node.scene() == self and link.end_node.node_type == "hypothesis":
                                        sub_hypotheses.append(link.end_node)
                                except (RuntimeError, AttributeError, ReferenceError):
                                    # Skip if node has become invalid
                                    continue
                        
                        # Arrange hypotheses connected to this sub-objective
                        sub_hyp_y = sub_y
                        for hyp in sub_hypotheses:
                            try:
                                # Verify the node is still valid
                                if hyp.scene() != self:
                                    continue
                                    
                                # Position on hypothesis lane at same height as connecting sub-objective
                                hyp.setPos(hypothesis_lane_x - hyp.boundingRect().width() / 2, sub_hyp_y)
                                arranged_nodes.add(hyp)
                                
                                # Update grid position
                                hyp_grid_pos = GridPosition(
                                    row=int(sub_hyp_y / self.grid_size),
                                    column=int(hypothesis_lane_x / self.grid_size)
                                )
                                
                                # Update node map
                                if hasattr(hyp, 'grid_position') and hyp.grid_position in self.nodes_grid:
                                    del self.nodes_grid[hyp.grid_position]
                                
                                self.nodes_grid[hyp_grid_pos] = hyp
                                hyp.grid_position = hyp_grid_pos
                                
                                # Move to next position with increased spacing
                                sub_hyp_y += vertical_spacing_hypotheses
                            except (RuntimeError, AttributeError, ReferenceError):
                                # Skip if node has become invalid
                                continue
                        
                        # Calculate next position for main objectives with increased spacing
                        sub_y = max(sub_y + vertical_spacing, sub_hyp_y)
                    except (RuntimeError, AttributeError, ReferenceError):
                        # Skip if node has become invalid
                        continue
                
                # Calculate next position for main objectives with increased spacing
                y_position = sub_y + vertical_spacing
            except (RuntimeError, AttributeError, ReferenceError):
                # Skip if node has become invalid
                continue
        
        # Arrange any remaining unconnected objectives
        remaining_objectives = [obj for obj in objectives if obj not in arranged_nodes]
        for obj in remaining_objectives:
            try:
                obj.setPos(objective_lane_x - obj.boundingRect().width() / 2, y_position)
                
                # Update grid position
                grid_pos = GridPosition(
                    row=int(y_position / self.grid_size),
                    column=int(objective_lane_x / self.grid_size)
                )
                
                # Update node map
                if hasattr(obj, 'grid_position') and obj.grid_position in self.nodes_grid:
                    del self.nodes_grid[obj.grid_position]
                
                self.nodes_grid[grid_pos] = obj
                obj.grid_position = grid_pos
                
                # Next position with increased spacing
                y_position += vertical_spacing
            except (RuntimeError, AttributeError, ReferenceError):
                # Skip if node has become invalid
                continue
        
        # Arrange any remaining unconnected hypotheses
        remaining_hypotheses = [hyp for hyp in hypotheses if hyp not in arranged_nodes]
        for hyp in remaining_hypotheses:
            try:
                hyp.setPos(hypothesis_lane_x - hyp.boundingRect().width() / 2, y_position)
                
                # Update grid position
                grid_pos = GridPosition(
                    row=int(y_position / self.grid_size),
                    column=int(hypothesis_lane_x / self.grid_size)
                )
                
                # Update node map
                if hasattr(hyp, 'grid_position') and hyp.grid_position in self.nodes_grid:
                    del self.nodes_grid[hyp.grid_position]
                
                self.nodes_grid[grid_pos] = hyp
                hyp.grid_position = grid_pos
                
                # Next position with increased spacing
                y_position += vertical_spacing_hypotheses
            except (RuntimeError, AttributeError, ReferenceError):
                # Skip if node has become invalid
                continue
        
        # Update all connections
        for item in self.items():
            try:
                if isinstance(item, NodeConnection):
                    item.update_path()
            except (RuntimeError, AttributeError, ReferenceError):
                # Skip if node has become invalid
                continue
        
        # Set the recently_auto_arranged flag
        self.recently_auto_arranged = True
        
        # Start a timer to clear this flag after a delay
        QTimer.singleShot(2000, self.clear_auto_arranged_flag)
        
        return True
    

    def clear_auto_arranged_flag(self):
        """Clear the flag indicating auto-arrange was recently used"""
        self.recently_auto_arranged = False

    def prevent_node_collision(self, moving_node):
        """Prevent nodes from being placed too close to each other"""
        current_pos = moving_node.pos()
        
        # Check each node in the scene
        for item in self.items():
            if isinstance(item, BaseNode) and item != moving_node:
                # Calculate distance between nodes
                distance = QLineF(current_pos, item.pos()).length()
                
                # If too close, push the moving node away
                if distance < self.minimum_node_distance:
                    # Calculate direction vector from item to moving_node
                    direction = QPointF(current_pos.x() - item.pos().x(), 
                                       current_pos.y() - item.pos().y())
                    
                    # Normalize the direction vector
                    length = QLineF(QPointF(0, 0), direction).length()
                    if length > 0:  # Avoid division by zero
                        direction = QPointF(direction.x() / length, direction.y() / length)
                    
                    # Calculate the minimum distance needed
                    push_distance = self.minimum_node_distance - distance
                    
                    # Move the node in the direction away from the other node
                    new_pos = QPointF(
                        current_pos.x() + direction.x() * push_distance,
                        current_pos.y() + direction.y() * push_distance
                    )
                    
                    # Apply the new position
                    moving_node.setPos(new_pos)

# ======================
# Lane Class
# ======================

class ResearchLane(QGraphicsObject):
    """Lane for organizing research nodes by type"""
    
    laneHovered = pyqtSignal(object)
        
    def __init__(self, orientation="vertical", position=0, width=8, height=20, color=None, label="", theme="light"):
        super().__init__()
        # Properties
        self.orientation = orientation
        self.position = position
        self.width = 4400
        self.height = 40000
        self.label = label
        self.theme = theme
        self.text_item = None
        
        # Initialize text item if label exists
        if label:
            self.setup_text_item()
        
        # Interaction settings
        self.is_hovered = False
        self.snap_margin = 140
        
        # Set z-value to be below nodes but above grid
        self.setZValue(-1.5)
        
        # Color handling
        self.setup_color(color)
        
        # Graphics properties
        self.setAcceptHoverEvents(True)
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        
        # Cached paths
        self._lane_path = None
        self._hover_path = None
        self.updatePaths()

    def setup_text_item(self):
        """Setup text display"""
        self.text_item = QGraphicsTextItem(self)
        self.text_item.setPlainText(self.label)
        text_color = QColor(130, 130, 130) if self.theme == "light" else QColor(180, 180, 180)
        self.text_item.setDefaultTextColor(text_color)
        font = QFont("Arial", 11)
        font.setBold(True)
        self.text_item.setFont(font)
        
        if self.orientation == "vertical":
            # Position text above vertical lane
            text_width = self.text_item.boundingRect().width()
            self.text_item.setPos(-text_width/2, -30)
        else:
            # Position text to the left of horizontal lane
            self.text_item.setPos(-self.text_item.boundingRect().width() - 10, -10)

    def setup_color(self, color):
        """Setup lane color"""
        if color is None:
            # Default colors based on orientation
            if self.orientation == "vertical":
                self.color = QColor(70, 130, 180, 120)  # Steel blue with transparency
            else:
                self.color = QColor(188, 143, 143, 120)  # Rosy brown with transparency
        else:
            self.color = QColor(color)
            
        # Set highlight color (slightly brighter)
        self.highlight_color = QColor(
            min(self.color.red() + 40, 255),
            min(self.color.green() + 40, 255),
            min(self.color.blue() + 40, 255),
            self.color.alpha()
        )
        
        # Set the lane glow color
        self.glow_color = QColor(
            self.color.red(),
            self.color.green(),
            self.color.blue(),
            100  # Lower alpha for glow
        )

    def update_theme(self, theme):
        """Update the lane's theme"""
        self.theme = theme
        
        # Update color
        self.setup_color(None)
        
        # Update text color
        if self.text_item is not None:
            text_color = QColor(80, 80, 80) if theme == "light" else QColor(200, 200, 200)
            self.text_item.setDefaultTextColor(text_color)
        
        self.update()
        
    def boundingRect(self):
        """Return bounding rectangle"""
        if self.orientation == "vertical":
            return QRectF(-self.snap_margin, -self.height/2, 
                        self.snap_margin * 2, self.height)
        else:
            return QRectF(-self.width/2, -self.snap_margin, 
                        self.width, self.snap_margin * 2)
        
    def paint(self, painter, option, widget):
        """Paint the lane"""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw lane with cached paths
        if self.is_hovered:
            # Draw hover effect with highlight color
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(self.highlight_color))
            painter.drawPath(self._hover_path)
            
            # Draw glow effect for hovered lane
            glow = QRadialGradient(
                self.boundingRect().center(),
                self.boundingRect().width() / 2
            )
            glow.setColorAt(0, self.glow_color)
            glow.setColorAt(1, QColor(self.glow_color.red(), 
                                      self.glow_color.green(), 
                                      self.glow_color.blue(), 0))
            painter.setBrush(QBrush(glow))
            painter.drawPath(self._hover_path)
        
        # Draw the main lane
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.color))
        painter.drawPath(self._lane_path)
    
    def hoverEnterEvent(self, event):
        """Handle hover enter"""
        self.is_hovered = True
        self.laneHovered.emit(self)
        self.update()
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        """Handle hover leave"""
        self.is_hovered = False
        self.update()
        super().hoverLeaveEvent(event)
            
    def get_snap_position(self, item_pos, item_rect):
        """Get position to snap item to lane"""
        if self.orientation == "vertical":
            # Snap horizontally to lane
            return QPointF(self.scenePos().x() - item_rect.width()/2, item_pos.y())
        else:
            # Snap vertically to lane
            return QPointF(item_pos.x(), self.scenePos().y() - item_rect.height()/2)
        
    def is_in_snap_range(self, item_pos):
        """Check if position is within snap range"""
        if self.orientation == "vertical":
            return abs(item_pos.x() - self.scenePos().x()) <= self.snap_margin
        else:
            return abs(item_pos.y() - self.scenePos().y()) <= self.snap_margin

    def updatePaths(self):
        """Update cached paths"""
        self._lane_path = QPainterPath()
        self._hover_path = QPainterPath()
        
        if self.orientation == "vertical":
            self._lane_path.addRoundedRect(-3, -self.height/2, 6, self.height, 2, 2)
            self._hover_path.addRoundedRect(-4, -self.height/2, 8, self.height, 2, 2)
        else:
            self._lane_path.addRoundedRect(-self.width/2, -3, self.width, 6, 2, 2)
            self._hover_path.addRoundedRect(-self.width/2, -4, self.width, 8, 2, 2)

# ======================
# Research Planning Widget
# ======================


# ======================
# Research Grid View
# ======================

class ResearchGridView(QGraphicsView):
    """View for displaying and interacting with the research grid"""
    
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        
        # Enable scrollbars
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        
        # For panning
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.panning = False
        self.last_pos = None
    
    def wheelEvent(self, event):
        """Zoom in/out with mouse wheel"""
        zoom_factor = 1.15
        
        if event.angleDelta().y() > 0:
            # Zoom in
            self.scale(zoom_factor, zoom_factor)
        else:
            # Zoom out
            self.scale(1.0 / zoom_factor, 1.0 / zoom_factor)
    
    def keyPressEvent(self, event):
        """Navigation with arrow keys"""
        # Get current center
        center = self.mapToScene(self.viewport().rect().center())
        
        # Movement distance
        step = 50
        
        if event.key() == Qt.Key.Key_Left:
            self.centerOn(center.x() - step, center.y())
        elif event.key() == Qt.Key.Key_Right:
            self.centerOn(center.x() + step, center.y())
        elif event.key() == Qt.Key.Key_Up:
            self.centerOn(center.x(), center.y() - step)
        elif event.key() == Qt.Key.Key_Down:
            self.centerOn(center.x(), center.y() + step)
        elif event.key() == Qt.Key.Key_Delete:
            # Delete selected items
            selected = self.scene().selectedItems()
            for item in selected:
                if isinstance(item, BaseNode):
                    item.delete_node()
                elif isinstance(item, NodeConnection):
                    item.delete_connection()
        else:
            super().keyPressEvent(event)
    
    def center_on_node(self, node):
        """Center the view on a specific node"""
        if node:
            self.centerOn(node)

