from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
import math

from PyQt6.QtWidgets import (
    QGraphicsPathItem, QGraphicsTextItem, QMenu, QGraphicsObject, QGraphicsDropShadowEffect, QGraphicsItem, QColorDialog, QInputDialog, QToolTip
)
import os
from PyQt6.QtGui import QPen, QBrush, QColor, QFont, QPainterPath, QPainterPathStroker, QLinearGradient, QPixmap
from PyQt6.QtCore import Qt, QPointF, QRectF, QEvent, pyqtSignal as Signal, QPropertyAnimation, Property, QLineF, QTimer, QRect


from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit, QPushButton, QComboBox
from PyQt6.QtCore import Qt
from PyQt6.QtSvg import QSvgRenderer

# Import icon loading
from helpers.load_icon import load_bootstrap_icon

class WorkspaceConstants:
    """Constants used throughout the workspace."""
    # Default colors
    DEFAULT_NODE_COLOR = "#4DB6AC"
    DEFAULT_PORT_COLOR = "#444444"
    
    # File paths
    EXPORT_DIR = os.path.expanduser("~/Documents/")
    
    # UI sizes
    NODE_PADDING = 15
    PORT_RADIUS = 8
    
    # Animation durations (milliseconds)
    HOVER_ANIMATION_DURATION = 150
    NODE_ANIMATION_DURATION = 200


class NodeCategory(Enum):
    # Patient flow nodes
    TARGET_POPULATION = "target_population"  # Starting patient population
    ELIGIBLE_POPULATION = "eligible_population"  # Inclusion/exclusion criteria
    INTERVENTION = "intervention"    # Study intervention
    OUTCOME = "outcome"              # Outcome measurement
    SUBGROUP = "subgroup"            # Patient subgroup
    CONTROL = "control"              # Control group
    RANDOMIZATION = "randomization"  # Randomization node
    TIMEPOINT = "timepoint"          # Measurement timepoint
    

class PortType(Enum):
    """Enum defining the types of ports a node can have."""
    INPUT = "input"
    OUTPUT = "output"
    ADD = "add"
    

@dataclass
class NodeConfig:
    category: NodeCategory
    width: int
    height: int
    color: str
    can_connect_to: List[NodeCategory]
    can_connect_from: List[NodeCategory]
    center_node: bool = False
    description: str = ""
    action_handler: Optional[str] = None  # Name of method to call when node is activated
    icon_name: Optional[str] = None  # Name of the Bootstrap icon to use

# ========================
# NODE CONFIGURATIONS
# ========================

NODE_CONFIGS = {
    NodeCategory.TARGET_POPULATION: NodeConfig(
        category=NodeCategory.TARGET_POPULATION,
        width=220,
        height=180,
        color="#4DB6AC",  # Teal
        can_connect_to=[NodeCategory.ELIGIBLE_POPULATION],
        can_connect_from=[],
        center_node=False,
        description="",
        action_handler="open_patient_group",
        icon_name="people-fill"
    ),
    NodeCategory.ELIGIBLE_POPULATION: NodeConfig(
        category=NodeCategory.ELIGIBLE_POPULATION,
        width=200,
        height=180,  # Increased height for triangle shape
        color="#8E24AA",  # Deeper purple for better contrast
        can_connect_to=[NodeCategory.RANDOMIZATION, NodeCategory.INTERVENTION, NodeCategory.SUBGROUP],
        can_connect_from=[NodeCategory.TARGET_POPULATION],
        description="Inclusion/exclusion criteria",
        action_handler="open_eligibility",
        icon_name="funnel"
    ),
    NodeCategory.INTERVENTION: NodeConfig(
        category=NodeCategory.INTERVENTION,
        width=280,  # Wider as requested
        height=80,   # Less vertical height as requested
        color="#FFA000",  # Darker amber for better visibility
        can_connect_to=[],  # Intervention is now an end node with no outputs
        can_connect_from=[NodeCategory.ELIGIBLE_POPULATION, NodeCategory.RANDOMIZATION, NodeCategory.SUBGROUP],
        description="Study intervention or treatment",
        action_handler="open_intervention",
        icon_name="capsule"
    ),
    NodeCategory.OUTCOME: NodeConfig(
        category=NodeCategory.OUTCOME,
        width=200,
        height=120,
        color="#1976D2",  # Darker blue for better contrast
        can_connect_to=[NodeCategory.TIMEPOINT],  # Outcome can only connect to Timepoint
        can_connect_from=[NodeCategory.ELIGIBLE_POPULATION, NodeCategory.RANDOMIZATION, NodeCategory.SUBGROUP],
        description="Outcome measurement",
        action_handler="open_outcome",
        icon_name="clipboard-data"
    ),
    NodeCategory.SUBGROUP: NodeConfig(
        category=NodeCategory.SUBGROUP,
        width=200,
        height=120,
        color="#D81B60",  # Deeper magenta/pink for better contrast
        can_connect_to=[NodeCategory.OUTCOME, NodeCategory.INTERVENTION],
        can_connect_from=[NodeCategory.ELIGIBLE_POPULATION, NodeCategory.RANDOMIZATION],
        description="Patient subgroup",
        action_handler="open_subgroup",
        icon_name="person-lines-fill"
    ),
    NodeCategory.CONTROL: NodeConfig(
        category=NodeCategory.CONTROL,
        width=280,  # Match width with Intervention
        height=80,   # Match height with Intervention
        color="#78909C",  # Blue-grey for contrast with the Intervention's amber
        can_connect_to=[],  # Control is also an end node
        can_connect_from=[NodeCategory.RANDOMIZATION],
        description="Control group treatment",
        action_handler="open_control_group",
        icon_name="tablet"
    ),
    NodeCategory.RANDOMIZATION: NodeConfig(
        category=NodeCategory.RANDOMIZATION,
        width=220,
        height=130,
        color="#F06292",  # Lighter pink for better contrast
        can_connect_to=[NodeCategory.INTERVENTION, NodeCategory.CONTROL, NodeCategory.SUBGROUP, NodeCategory.OUTCOME],
        can_connect_from=[NodeCategory.ELIGIBLE_POPULATION],
        description="Randomization procedure",
        action_handler="open_randomization",
        icon_name="shuffle"
    ),
    NodeCategory.TIMEPOINT: NodeConfig(
        category=NodeCategory.TIMEPOINT,
        width=160,
        height=80,
        color="#9CCC65",  # Light green for contrast
        can_connect_to=[],  # Timepoint is an end node
        can_connect_from=[NodeCategory.OUTCOME],
        description="Measurement timepoint",
        action_handler="open_timepoint",
        icon_name="calendar-check"
    ),
}

# ========================
# EDGE ITEM FOR WORKFLOW CONNECTIONS
# ========================

class EdgeItem(QGraphicsPathItem):
    # A mapping of action names to colors.
    ACTION_COLORS = {
        "default": QColor(120, 120, 140, 170),  # Standard connection
        "active": QColor(76, 175, 80, 200),     # Active path
        "completed": QColor(33, 150, 243, 200), # Completed step
        "error": QColor(244, 67, 54, 200),      # Error
        "patient_flow": QColor(64, 196, 255, 180), # Patient flow
        "intervention": QColor(255, 152, 0, 200), # Intervention
        "outcome": QColor(156, 39, 176, 200),   # Outcome measurement
    }
    
    def __init__(self, start_port, end_port, action="default", patient_count=100):
        super().__init__()
        self.start_port = start_port
        self.end_port = end_port
        self.arrow_size = 25  # Increased arrow size
        self.ribbon_width = 40  # Default wider ribbon for better visibility
        self.action = action
        self.patient_count = patient_count  # Number of patients flowing through this edge
        self.max_width = 80  # Maximum width for ribbons
        self.min_width = 20  # Minimum width for ribbons
        
        self.setFlags(
            QGraphicsPathItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsPathItem.GraphicsItemFlag.ItemIsFocusable
        )
        self.setActionColor(action)
        
        # Store parent nodes weakly to avoid circular references
        self.start_node = start_port.parentItem()
        self.end_node = end_port.parentItem()
        
        # Connect to position changes if nodes exist
        if self.start_node:
            self.start_node.xChanged.connect(self.updatePosition)
            self.start_node.yChanged.connect(self.updatePosition)
        if self.end_node:
            self.end_node.xChanged.connect(self.updatePosition)
            self.end_node.yChanged.connect(self.updatePosition)
            
        self.updatePosition()
        
        # Store additional data
        self.flow_data = {
            "patient_count": patient_count,
            "label": "",
            "description": ""
        }
        
        # Calculate ribbon width based on patient count
        self.updateRibbonWidth()
        
        # Make the edge more clickable
        self.setAcceptHoverEvents(True)
        # Explicitly accept all mouse buttons
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton)
        # Set a wide pen for hit detection
        self.setPen(QPen(QColor(0, 0, 0, 0), self.ribbon_width + 15))  # Invisible wider pen for better hit detection
        
        # Set Z value to ensure edges are above other items for better selection
        self.setZValue(-1)  # Below nodes but above background

    def __del__(self):
        # Safely disconnect signals when edge is deleted
        if hasattr(self, 'start_node') and self.start_node:
            try:
                self.start_node.xChanged.disconnect(self.updatePosition)
                self.start_node.yChanged.disconnect(self.updatePosition)
            except:
                pass
        if hasattr(self, 'end_node') and self.end_node:
            try:
                self.end_node.xChanged.disconnect(self.updatePosition)
                self.end_node.yChanged.disconnect(self.updatePosition)
            except:
                pass

    def setActionColor(self, action):
        self.action = action
        color = self.ACTION_COLORS.get(action, QColor(0, 0, 0, 150))
        # Store the visible pen separately from the hit detection pen
        self.visible_pen = QPen(color, self.ribbon_width)  # Wide ribbon
        self.visible_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.visible_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        
        # Set a gradient brush for filled ribbons
        self.gradient = QLinearGradient()
        self.gradient.setColorAt(0, color.lighter(120))
        self.gradient.setColorAt(1, color)

    def updateRibbonWidth(self):
        """Update ribbon width based on patient count"""
        # Calculate a width between min_width and max_width based on patient count
        # Logarithmic scale to handle large numbers of patients
        if self.patient_count <= 0:
            self.ribbon_width = self.min_width
        else:
            import math
            # Use log scale with a cap at max_width
            # A count of 100 gives about half the max width
            log_factor = math.log10(max(1, self.patient_count)) / math.log10(1000)
            self.ribbon_width = self.min_width + (self.max_width - self.min_width) * log_factor
            self.ribbon_width = min(self.max_width, max(self.min_width, self.ribbon_width))
        
        # Update the pen width
        if hasattr(self, 'visible_pen'):
            self.visible_pen.setWidth(int(self.ribbon_width))
            
    def setPatientCount(self, count):
        """Set the patient count for this edge"""
        self.patient_count = count
        self.flow_data["patient_count"] = count
        self.updateRibbonWidth()
        self.update()

    def updatePosition(self):
        """Update the position of the edge path with enhanced curves"""
        if not self.start_port or not self.end_port:
            return
            
        # Get port scene positions
        start_pos = self.start_port.scenePos()
        end_pos = self.end_port.scenePos()
        
        # Convert to path coordinates
        start_point = self.start_port.scenePos()
        end_point = self.end_port.scenePos()
        
        # Create a path for the edge
        path = QPainterPath(start_point)
        
        # Calculate the direction vector
        dx = end_point.x() - start_point.x()
        dy = end_point.y() - start_point.y()
        line_length = (dx**2 + dy**2)**0.5
        
        # Ensure minimum line length to avoid division by zero
        if line_length < 1e-6:
            line_length = 1e-6
        
        # Calculate control points for a more pronounced curve
        # Determine if the connection is mostly horizontal or vertical
        is_horizontal = abs(dx) > abs(dy)
        
        if is_horizontal:
            # For horizontal connections, use more curved control points
            ctrl_point1 = QPointF(
                start_point.x() + dx * 0.3,  # 30% of the way horizontally
                start_point.y() + dy * 0.15 - 30  # Add more vertical curve
            )
            
            ctrl_point2 = QPointF(
                end_point.x() - dx * 0.3,    # 30% back from the end horizontally
                end_point.y() - dy * 0.15 + 30  # Add more vertical curve
            )
        else:
            # For vertical connections, use more curved control points
            ctrl_point1 = QPointF(
                start_point.x() + dx * 0.15 - 30,  # Add more horizontal curve
                start_point.y() + dy * 0.3   # 30% of the way vertically
            )
            
            ctrl_point2 = QPointF(
                end_point.x() - dx * 0.15 + 30,  # Add more horizontal curve
                end_point.y() - dy * 0.3     # 30% back from the end vertically
            )
        
        # Create a cubic Bezier curve
        path.cubicTo(ctrl_point1, ctrl_point2, end_point)
        
        # Update the path
        self.setPath(path)
        
        # Update gradient for the current path
        if hasattr(self, 'gradient'):
            self.gradient.setStart(start_point)
            self.gradient.setFinalStop(end_point)

    def paint(self, painter, option, widget):
        """Override paint to use the visible pen for drawing but keep the wide pen for hit detection"""
        # Save the current pen
        old_pen = self.pen()
        
        # Save painter state
        painter.save()
        
        # Enable antialiasing for smoother edges
        painter.setRenderHint(painter.RenderHint.Antialiasing)
        
        # Draw filled path with gradient for patient flow
        if self.action in ["patient_flow", "default"]:
            # Use gradient fill for ribbon
            if hasattr(self, 'gradient'):
                painter.setBrush(self.gradient)
            else:
                painter.setBrush(QBrush(self.visible_pen.color().lighter(115)))
                
            # Create a stroker to outline the path
            stroker = QPainterPathStroker()
            stroker.setWidth(self.ribbon_width)
            stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
            stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            
            # Create filled path
            fillPath = stroker.createStroke(self.path())
            
            # Draw the filled path
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setOpacity(0.85)  # Slightly transparent
            painter.drawPath(fillPath)
            
            # Draw a border around the filled path
            border_pen = QPen(QColor(60, 60, 60, 100), 1.5)
            painter.setPen(border_pen)
            painter.drawPath(fillPath)
            
            # Draw patient count if greater than 0
            if self.patient_count > 0:
                # Find midpoint of the curve
                path_length = self.path().length()
                mid_point = self.path().pointAtPercent(0.5)
                
                # Draw patient count label
                painter.setPen(QPen(QColor(30, 30, 30, 220)))
                painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                count_text = f"{self.patient_count}"
                
                # Get text bounds
                font_metrics = painter.fontMetrics()
                text_width = font_metrics.horizontalAdvance(count_text)
                text_height = font_metrics.height()
                
                # Draw background capsule
                text_rect = QRectF(mid_point.x() - text_width/2 - 5, 
                                  mid_point.y() - text_height/2 - 3,
                                  text_width + 10, text_height + 6)
                painter.setBrush(QColor(255, 255, 255, 200))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(text_rect, 10, 10)
                
                # Draw text
                painter.setPen(QPen(QColor(30, 30, 30, 220)))
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, count_text)
                
                # Also draw label text if there is one
                if "label" in self.flow_data and self.flow_data["label"]:
                    label_text = self.flow_data["label"]
                    label_width = font_metrics.horizontalAdvance(label_text)
                    label_rect = QRectF(mid_point.x() - label_width/2 - 5, 
                                       mid_point.y() + text_height/2 + 5,
                                       label_width + 10, text_height + 6)
                    
                    # Draw background capsule for label
                    painter.setBrush(QColor(200, 200, 220, 200))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawRoundedRect(label_rect, 10, 10)
                    
                    # Draw label text
                    painter.setPen(QPen(QColor(30, 30, 30, 220)))
                    painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label_text)
        else:
            # For other actions, use the standard line drawing
            painter.setPen(self.visible_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setOpacity(0.8)
            painter.drawPath(self.path())
        
        # If selected, draw a highlight
        if self.isSelected():
            highlight_pen = QPen(QColor(255, 165, 0, 200), self.ribbon_width + 4)  # Orange highlight
            highlight_pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(highlight_pen)
            painter.drawPath(self.path())
        
        # Restore painter state
        painter.restore()
        
        # Restore the wide pen for hit detection
        self.setPen(old_pen)

    def hoverEnterEvent(self, event):
        """Highlight the edge when mouse hovers over it"""
        # Make ribbon wider on hover
        self.visible_pen.setWidth(int(self.ribbon_width * 1.2))
        
        # Make color more vibrant on hover
        color = self.visible_pen.color()
        hover_color = QColor(
            min(255, int(color.red() * 1.2)),
            min(255, int(color.green() * 1.2)),
            min(255, int(color.blue() * 1.2)),
            min(255, int(color.alpha() * 1.2))
        )
        self.visible_pen.setColor(hover_color)
        
        # Update gradient
        if hasattr(self, 'gradient'):
            self.gradient.setColorAt(0, hover_color.lighter(120))
            self.gradient.setColorAt(1, hover_color)
        
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Return to normal appearance when mouse leaves"""
        # Return to normal width
        self.visible_pen.setWidth(int(self.ribbon_width))
        
        # Return to normal color
        color = self.ACTION_COLORS.get(self.action, QColor(0, 0, 0, 150))
        self.visible_pen.setColor(color)
        
        # Reset gradient
        if hasattr(self, 'gradient'):
            self.gradient.setColorAt(0, color.lighter(120))
            self.gradient.setColorAt(1, color)
        
        self.update()
        super().hoverLeaveEvent(event)
        
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.setSelected(True)
            event.accept()
        else:
            super().mousePressEvent(event)
            
    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        super().mouseReleaseEvent(event)
            
    def contextMenuEvent(self, event):
        """Show context menu for edge"""
        menu = QMenu()
        
        # Add patient count action
        set_count_action = menu.addAction("Set Patient Count")
        
        if self.action == "patient_flow":
            # Add label action
            set_label_action = menu.addAction("Set Label")
        
        # Add action to change color
        change_color_action = menu.addAction("Change Color")
        
        # Add delete action
        menu.addSeparator()
        delete_action = menu.addAction("Delete Edge")
        
        # Show menu and handle actions
        action = menu.exec(event.screenPos())
        
        if action == set_count_action:
            count, ok = QInputDialog.getInt(None, "Patient Count", 
                                           "Enter number of patients:", 
                                           self.patient_count, 0, 100000)
            if ok:
                self.setPatientCount(count)
        elif self.action == "patient_flow" and action == set_label_action:
            label, ok = QInputDialog.getText(None, "Edge Label", 
                                           "Enter label for this connection:",
                                           QLineEdit.EchoMode.Normal,
                                           self.flow_data.get("label", ""))
            if ok:
                self.flow_data["label"] = label
                self.update()
        elif action == change_color_action:
            color = QColorDialog.getColor(self.visible_pen.color(), None, "Select Edge Color")
            if color.isValid():
                self.visible_pen.setColor(color)
                
                # Update gradient
                if hasattr(self, 'gradient'):
                    self.gradient.setColorAt(0, color.lighter(120))
                    self.gradient.setColorAt(1, color)
                    
                self.update()
        elif action == delete_action:
            self.delete_edge()
            
        event.accept()

    def delete_edge(self):
        """Delete this edge"""
        if self.scene():
            self.scene().removeItem(self)
            
    def shape(self):
        """Override shape to use a stroked path for better hit detection"""
        stroker = QPainterPathStroker()
        stroker.setWidth(self.ribbon_width + 10)  # Wider for easier selection
        return stroker.createStroke(self.path())

    def update_theme(self, is_dark_theme):
        """Update the edge's colors based on the current theme."""
        from model_builder.theme_support import update_edge_theme
        update_edge_theme(self, is_dark_theme)

# ========================
# PORT ITEM (for connection points)
# ========================

class PortItem(QGraphicsObject):
    # Add scale property for animation
    _scale = 1.0
    
    # Define the scale property
    scaleChanged = Signal()
    
    @Property(float, notify=scaleChanged)
    def scale(self):
        return self._scale
    
    @scale.setter
    def scale(self, value):
        if self._scale != value:
            self._scale = value
            self.scaleChanged.emit()
            self.update()
    
    def __init__(self, x, y, port_type, label="", parent=None):
        """
        Initialize a new port item
        port_type can be: 'input', 'output', or 'connect' (new simplified connector)
        """
        super().__init__(parent)
        self.setAcceptHoverEvents(True)
        self.port_type = port_type
        self.label = label
        self.parent_item = parent
        self.hover = False
        self.potential_connection = False
        
        # Create a path for drawing
        self.path = QPainterPath()
        
        # Ensure port accepts mouse events for connection creation
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton)
        
        # Set position relative to parent
        self.setPos(x, y)
        
        # Set size for the port - increased for better visibility
        self.port_width = 32  # Increased from 24
        self.port_height = 24  # Increased from 18
        self.port_radius = 8  # Increased from 6
        self.plus_size = 12  # Increased from 10
        self.stroke_width = 3  # Increased from 2
        self.updatePath()
        
        # Set default appearance based on port type
        if port_type == "add":
            # Green for add ports
            self.default_brush = QBrush(QColor(76, 175, 80))  # Material green
            self.hover_brush = QBrush(QColor(129, 199, 132))  # Lighter material green
        elif port_type == "input" and parent and hasattr(parent, 'config') and parent.config.category == NodeCategory.TARGET_POPULATION:
            # Get node color from parent
            node_color = NODE_CONFIGS[NodeCategory.TARGET_POPULATION].color
            self.default_brush = QBrush(QColor(node_color))
            self.hover_brush = QBrush(QColor(node_color).lighter(120))
        elif port_type == "input" and parent and hasattr(parent, 'config') and parent.config.category == NodeCategory.INTERVENTION:
            # Get node color from parent
            node_color = NODE_CONFIGS[NodeCategory.INTERVENTION].color
            self.default_brush = QBrush(QColor(node_color))
            self.hover_brush = QBrush(QColor(node_color).lighter(120))
        elif port_type == "input" and parent and hasattr(parent, 'config') and parent.config.category == NodeCategory.OUTCOME:
            # Get node color from parent
            node_color = NODE_CONFIGS[NodeCategory.OUTCOME].color
            self.default_brush = QBrush(QColor(node_color))
            self.hover_brush = QBrush(QColor(node_color).lighter(120))
        elif port_type == "input" and parent and hasattr(parent, 'config') and parent.config.category == NodeCategory.SUBGROUP:
            # Get node color from parent
            node_color = NODE_CONFIGS[NodeCategory.SUBGROUP].color
            self.default_brush = QBrush(QColor(node_color))
            self.hover_brush = QBrush(QColor(node_color).lighter(120))
        elif port_type == "input" and parent and hasattr(parent, 'config') and parent.config.category == NodeCategory.CONTROL:
            # Get node color from parent
            node_color = NODE_CONFIGS[NodeCategory.CONTROL].color
            self.default_brush = QBrush(QColor(node_color))
            self.hover_brush = QBrush(QColor(node_color).lighter(120))
        elif port_type == "output":
            # Blue for output ports
            self.default_brush = QBrush(QColor(158, 158, 158))  # Material gray
            self.hover_brush = QBrush(QColor(189, 189, 189))  # Lighter material gray
        else:
            # Gray for other ports
            self.default_brush = QBrush(QColor(158, 158, 158))  # Material gray
            self.hover_brush = QBrush(QColor(189, 189, 189))  # Lighter material gray
            
        # Brush for potential connection highlight
        self.potential_brush = QBrush(QColor(139, 195, 74))  # Material light green
        self.normal_brush = self.default_brush
        self.setBrush(self.default_brush)
        
        # Add a visible border with light gray color
        self.setPen(QPen(QColor(180, 180, 180), 1))
        
        # Set Z value to ensure ports are above the node
        self.setZValue(2)
        
        # Create the label if provided
        if label:
            self.text_item = QGraphicsTextItem(label, self)
            # Use a neutral gray that will be updated by the theme system
            self.text_item.setDefaultTextColor(QColor(60, 60, 60))  # Initial color, will be updated by theme
            font = QFont("Arial", 10)  # Increased font size from 9 to 10
            font.setBold(True)  # Make the font bold
            self.text_item.setFont(font)
            
            # Position the label based on port type and position
            # For study model, check if port is on left side (x=0) to flare left
            parent_node = self.parentItem()
            if parent_node and hasattr(parent_node, 'config') and parent_node.config.category == NodeCategory.TARGET_POPULATION and self.pos().x() == 0:
                # Left side ports on study model - flare left
                self.text_item.setPos(-self.text_item.boundingRect().width() - 15, -12)  # Adjusted positioning
            elif port_type == "input":
                # Standard input ports
                self.text_item.setPos(-self.text_item.boundingRect().width() - 15, -12)  # Adjusted positioning
            else:
                # Output and other ports
                self.text_item.setPos(15, -12)  # Adjusted positioning
    
    def boundingRect(self):
        """Return the bounding rectangle of the port, accounting for scaling"""
        rect = self.path.boundingRect()
        # Expand the rect to account for scaling
        center = rect.center()
        scaled_width = rect.width() * self._scale
        scaled_height = rect.height() * self._scale
        return QRectF(
            center.x() - scaled_width / 2,
            center.y() - scaled_height / 2,
            scaled_width,
            scaled_height
        )
    
    def paint(self, painter, option, widget):
        """Paint the port as a rounded rectangle with a plus sign"""
        # Draw rounded rectangle background
        painter.setBrush(self.brush())
        painter.setPen(self.pen())
        
        # Apply scaling transformation
        painter.save()
        painter.scale(self._scale, self._scale)
        
        # Draw rounded rectangle background
        # Center the rectangle around the origin point
        rect = QRectF(-self.port_width/2, -self.port_height/2, self.port_width, self.port_height)
        painter.drawRoundedRect(rect, self.port_radius, self.port_radius)
        
        # Draw plus or minus sign in white
        plus_pen = QPen(QColor(255, 255, 255))
        plus_pen.setWidth(self.stroke_width)
        plus_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(plus_pen)
        
        # Draw plus sign using QLineF
        half_size = self.plus_size / 2
        
        # Always draw horizontal line (for both plus and minus)
        painter.drawLine(QLineF(-half_size, 0, half_size, 0))
        
        # Determine if we should draw a vertical line (plus sign) or not (minus sign)
        draw_vertical = True
        
        # For ports on node edges, determine orientation
        parent_node = self.parentItem()
        if parent_node:
            port_pos = self.pos()
            
            # For horizontal ports (left/right edges), don't draw vertical line
            if (abs(port_pos.x()) <= 10 or abs(port_pos.x() - parent_node.current_width) <= 10):
                draw_vertical = False
            
            # Check node type and port label to determine plus/minus icon
            if hasattr(parent_node, 'config'):
                if parent_node.config.category == NodeCategory.TARGET_POPULATION:
                    if self.label in ["Target Population"]:
                        draw_vertical = True
                    elif self.label == "Eligible Population":
                        # Always show plus sign for Eligible Population port on Target Population node
                        draw_vertical = True
                    elif self.label in ["Intervention", "Outcome", "Subgroup", "Control", "Randomization"]:
                        # Always show plus sign for output ports
                        if self.port_type == "output":
                            draw_vertical = True
                        else:
                            # Check if there's already a connected node
                            scene = self.scene()
                            if scene:
                                has_connection = False
                                for item in scene.items():
                                    if isinstance(item, EdgeItem):
                                        if (item.start_port.parentItem() == parent_node and 
                                            item.start_port.label == self.label):
                                            has_connection = True
                                            break
                                
                                # If already has a connection, show minus, otherwise plus
                                draw_vertical = not has_connection
                # Update port icons for non-patient group nodes
                elif parent_node.config.category != NodeCategory.TARGET_POPULATION:
                    # Allow positive ports for all nodes
                    if self.port_type == "output":
                        draw_vertical = True  # Output ports on all nodes should always show plus
                    elif self.port_type == "input":
                        # Check if there's already a connection to this input port
                        scene = self.scene()
                        if scene:
                            has_connection = False
                            for item in scene.items():
                                if isinstance(item, EdgeItem):
                                    if item.end_port == self:
                                        has_connection = True
                                        break
                            
                            # If already has a connection, show minus, otherwise plus
                            draw_vertical = not has_connection
                        else:
                            draw_vertical = True  # Default to plus if scene not available
        
        # Special case for add button - always show plus
        if self.port_type == "add":
            draw_vertical = True
        
        # Draw vertical line for plus sign if needed
        if draw_vertical:
            painter.drawLine(QLineF(0, -half_size, 0, half_size))
        
        painter.restore()
    
    def updatePath(self):
        """Update the port's path for hit testing"""
        self.path = QPainterPath()
        # Create a slightly larger area for hit testing
        rect = QRectF(-self.port_width/2, -self.port_height/2, self.port_width, self.port_height)
        self.path.addRoundedRect(rect, self.port_radius, self.port_radius)
        self.update()
    
    def setBrush(self, brush):
        """Store the brush for use in paint"""
        self._brush = brush
        self.update()
    
    def brush(self):
        """Return the current brush"""
        return self._brush
    
    def setPen(self, pen):
        """Store the pen for use in paint"""
        self._pen = pen
        self.update()
    
    def pen(self):
        """Return the current pen"""
        return self._pen
    
    def parentItem(self):
        """Override parentItem to ensure we return the correct parent node"""
        return self.parent_item if self.parent_item else super().parentItem()
        
    def hoverEnterEvent(self, event):
        # Make the port larger and brighter on hover
        self.hover = True
        
        # Create a smooth grow animation
        if not hasattr(self, 'hover_animation'):
            self.hover_animation = QPropertyAnimation(self, b"scale")
            self.hover_animation.setDuration(150)  # 150ms animation
        
        self.hover_animation.setStartValue(1.0)
        self.hover_animation.setEndValue(1.4)  # Grow to 140% size
        self.hover_animation.start()
        
        # Change color
        self.setBrush(self.hover_brush)
        
        # Remove glow effect
        self.setGraphicsEffect(None)
        
        # Update the cursor
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Force redraw
        self.update()
        
        # Call parent implementation
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        # Restore original appearance
        self.hover = False
        
        # Create a smooth shrink animation
        if hasattr(self, 'hover_animation'):
            self.hover_animation.setStartValue(1.4)
            self.hover_animation.setEndValue(1.0)  # Return to normal size
            self.hover_animation.start()
        
        # Restore original color
        self.setBrush(self.default_brush)
        
        # Restore cursor
        self.setCursor(Qt.CursorShape.ArrowCursor)
        
        # Force redraw
        self.update()
        
        # Call parent implementation
        super().hoverLeaveEvent(event)
        
    def mousePressEvent(self, event):
        """Handle mouse press on port to start edge creation"""
        if event.button() == Qt.MouseButton.LeftButton:
            print(f"Port clicked: {self.port_type}, Label: {self.label}")
            
            # Flash the port to provide visual feedback
            self.setBrush(QBrush(QColor(255, 255, 0, 255)))  # Bright yellow flash
            self.update()
            
            # Reset the brush after a short delay
            QTimer.singleShot(300, lambda: self.setBrush(self.default_brush))
            
            # If this is the plus button on a node, let the node handle it
            parent_node = self.parentItem()
            if parent_node and hasattr(parent_node, 'plus_button') and self == parent_node.plus_button:
                # Let the event propagate to the node's plus button handler
                super().mousePressEvent(event)
                return
                
            # If this is an "add" type port (plus button), don't start edge creation
            if self.port_type == "add":
                super().mousePressEvent(event)
                return
            
            # Get the scene and initiate edge drawing
            scene = self.scene()
            if scene and hasattr(scene, 'drawing_edge') and hasattr(scene, 'start_port'):
                # Always allow output ports to start edge creation, regardless of existing connections
                if self.port_type == "output":
                    # Set this port as the start of an edge
                    scene.start_port = self
                    scene.drawing_edge = True
                    
                    # Create temporary line for feedback if not already present
                    if hasattr(scene, 'temp_line'):
                        if scene.temp_line:
                            scene.removeItem(scene.temp_line)
                        
                        scene.temp_line = QGraphicsPathItem()
                        scene.temp_line.setPen(QPen(QColor(33, 150, 243, 180), 2, Qt.PenStyle.DashLine))
                        scene.addItem(scene.temp_line)
                    
                    # Show tooltip with instructions
                    if hasattr(scene, 'show_edge_creation_tooltip'):
                        scene.show_edge_creation_tooltip()
                    
                    # Highlight potential connection ports
                    if hasattr(scene, 'highlight_potential_connection_ports'):
                        scene.highlight_potential_connection_ports(event.scenePos())
                    
                    # Accept event to indicate we've handled it
                    event.accept()
                    return
                elif self.port_type == "input" and scene.drawing_edge and scene.start_port:
                    # If we click on an input port while drawing an edge, 
                    # automatically try to complete the connection
                    # This will be handled by the scene's mouseReleaseEvent
                    pass
                elif self.port_type == "input":
                    # Show message that input ports can't start connections
                    self.setBrush(QBrush(QColor(244, 67, 54)))  # Red for error
                    QTimer.singleShot(300, lambda: self.setBrush(self.default_brush))
                    
                    # Optional: Show a tooltip explaining why
                    QToolTip.showText(
                        event.screenPos(),
                        "Input ports cannot start connections.\nStart from an output port instead.",
                        None,
                        QRect(),
                        2000  # Hide after 2 seconds
                    )
        
        # Let the event propagate to the scene
        super().mousePressEvent(event)

    def update_theme(self, is_dark_theme):
        """Update the port's colors based on the current theme."""
        from model_builder.theme_support import update_port_theme
        update_port_theme(self, is_dark_theme)

class WorkflowNode(QGraphicsObject):
    # Define signals as class attributes
    xChanged = Signal()
    yChanged = Signal()
    nodeActivated = Signal(object)  # Emitted when node is activated (double-clicked)
    plusButtonClicked = Signal()  # Emitted when the plus button is clicked
    
    def __init__(self, config: NodeConfig, x: float = 0, y: float = 0):
        super().__init__()
        self.config = config
        self.config_details = ""
        self.expanded = False
        self.is_active = False  # Track active state
        self.is_hovered = False  # Track hover state
        self.input_ports = []
        self.output_ports = []
        self.title_item = None
        self.desc_item = None
        self.path_item = None
        self.glow_effect = None
        self.plus_button = None  # Top right plus button for Study Model
        self.current_theme = "light"  # Default theme
        self.custom_data = {}  # Initialize custom_data dictionary
        self.properties = {}  # Initialize properties dictionary for custom properties
        self._drag_start_pos = None  # Initialize drag start position
        
        # Set width and height from config
        self.width = config.width
        self.height = config.height
        self.current_width = self.width
        self.current_height = self.height
        self.border_padding = 2  # Border padding for inner content
        
        # Initialize icon
        self.icon = None
        
        # Load Bootstrap icon if specified in config
        if hasattr(config, 'icon_name') and config.icon_name:
            try:
                # Load the icon with node color and appropriate size
                node_color = config.color
                icon_size = min(self.width, self.height) // 4
                qicon = load_bootstrap_icon(config.icon_name, node_color, icon_size)
                
                # Convert QIcon to QPixmap for rendering
                pixmap = qicon.pixmap(icon_size, icon_size)
                # Store the icon as QSvgRenderer if it loaded successfully
                if not pixmap.isNull():
                    # Create a QPixmap from the icon for rendering
                    self.icon_pixmap = pixmap
            except ImportError:
                print(f"Warning: Could not import load_bootstrap_icon function")
            except Exception as e:
                print(f"Warning: Failed to load icon {config.icon_name}: {e}")
        
        # Fallback to direct SVG loading if bootstrap icon loading failed
        if not self.icon:
            icon_path = f"icons/{config.category.value}.svg"
            if os.path.exists(icon_path):
                self.icon = QSvgRenderer(icon_path)
        
        # Set flags
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        
        # Setup appearance
        self.setup_appearance(x, y)
        
        # Setup text items
        self.setup_text_items()
        
        # Setup ports
        self.setup_ports()

    def setup_appearance(self, x: float, y: float):
        self.updatePath(self.width, self.height)
        self.setPos(x, y)

    def boundingRect(self):
        # Extend bounding rect to include glow effect when active
        rect = self._path.boundingRect()
        
        # Make the bounding rect slightly larger to ensure it captures all mouse events
        if self.is_active:
            # Use the glow path for active nodes
            rect = self._glow_path.boundingRect()
        else:
            # For inactive nodes, add a small margin around the node
            rect.adjust(-5, -5, 5, 5)
            
        return rect

    def paint(self, painter, option, widget=None):
        # Draw the single border band with a darker shade of the node's color
        color = QColor(self.config.color)
        # Create a darker shade of the node's color for the border
        border_color = QColor(
            int(color.red() * 0.7),
            int(color.green() * 0.7),
            int(color.blue() * 0.7),
            200  # Semi-transparent
        )
        
        # If selected, use a more prominent border color for deletion indication
        if self.isSelected():
            border_color = QColor(255, 60, 60, 230)  # Bright red for selection/deletion
            painter.setPen(QPen(border_color, 4, Qt.PenStyle.DashLine))
        elif self.is_hovered:
            # Slightly brighter border for hover state
            border_color = QColor(
                min(255, int(color.red() * 1.2)),
                min(255, int(color.green() * 1.2)),
                min(255, int(color.blue() * 1.2)),
                220
            )
            painter.setPen(QPen(border_color, 3))
        else:
            painter.setPen(QPen(border_color, 3))
            
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(self._border_path)
        
        # Apply glow effect if node is active
        if self.is_active:
            glow_color = QColor(255, 200, 50, 100)  # Amber glow
            glow_pen = QPen(glow_color, 8)
            glow_pen.setStyle(Qt.PenStyle.SolidLine)
            painter.setPen(glow_pen)
            painter.drawPath(self._glow_path)
        
        # Draw inner content
        # Initialize pastel_color with a default value
        pastel_color = QColor(
            int((color.red() * 0.8 + 255 * 0.2)),
            int((color.green() * 0.8 + 255 * 0.2)),
            int((color.blue() * 0.8 + 255 * 0.2))
        )
        
        # Make center node slightly more saturated
        if self.config.center_node:
            if self.current_theme == "dark":
                pastel_color = QColor(
                    int((color.red() * 0.9 + 255 * 0.1)),
                    int((color.green() * 0.9 + 255 * 0.1)),
                    int((color.blue() * 0.9 + 255 * 0.1))
                )
            else:
                pastel_color = QColor(
                    int((color.red() * 0.8 + 255 * 0.2)),
                    int((color.green() * 0.8 + 255 * 0.2)),
                    int((color.blue() * 0.8 + 255 * 0.2))
                )
            
        # Brighten the node color when hovered
        if self.is_hovered and not self.isSelected():
            pastel_color = QColor(
                min(255, int(pastel_color.red() * 1.1)),
                min(255, int(pastel_color.green() * 1.1)),
                min(255, int(pastel_color.blue() * 1.1))
            )
        
        # Add a slight red tint when selected for deletion
        if self.isSelected():
            pastel_color = QColor(
                min(255, int(pastel_color.red() * 0.9 + 255 * 0.1)),
                int(pastel_color.green() * 0.9),
                int(pastel_color.blue() * 0.9)
            )
            
        painter.setBrush(QBrush(pastel_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(self._path)
        
        # Draw SVG icon if available
        if hasattr(self, 'icon_pixmap') and not self.icon_pixmap.isNull():
            # Calculate icon position (centered)
            icon_size = min(self.current_width, self.current_height) * 0.25  # 25% of node size
            x = (self.current_width - icon_size) / 2
            y = icon_size * 0.5  # Position higher in the node
            painter.drawPixmap(QRectF(x, y, icon_size, icon_size), self.icon_pixmap, QRectF(0, 0, self.icon_pixmap.width(), self.icon_pixmap.height()))
        elif self.icon and self.icon.isValid():
            # Calculate icon size and position (centered)
            icon_size = min(self.current_width, self.current_height) * 0.25  # 25% of node size
            x = (self.current_width - icon_size) / 2
            y = icon_size * 0.5  # Position higher in the node
            icon_rect = QRectF(x, y, icon_size, icon_size)
            self.icon.render(painter, icon_rect)

    def updatePath(self, width, height):
        # Create different shapes based on node category
        self._path = QPainterPath()
        self._border_path = QPainterPath()
        self._glow_path = QPainterPath()
        
        # Default corner radius for rounded rectangles
        corner_radius = 10
        
        if self.config.category == NodeCategory.ELIGIBLE_POPULATION:
            # Triangle shape for ELIGIBLE_POPULATION
            self._path.moveTo(width/2, 0)  # Top point
            self._path.lineTo(width, height)  # Bottom right
            self._path.lineTo(0, height)  # Bottom left
            self._path.closeSubpath()
            
            # Border path (slightly larger)
            self._border_path.moveTo(width/2, -6)
            self._border_path.lineTo(width + 6, height + 6)
            self._border_path.lineTo(-6, height + 6)
            self._border_path.closeSubpath()
            
            # Glow path (even larger)
            self._glow_path.moveTo(width/2, -8)
            self._glow_path.lineTo(width + 8, height + 8)
            self._glow_path.lineTo(-8, height + 8)
            self._glow_path.closeSubpath()
            
        elif self.config.category == NodeCategory.RANDOMIZATION:
            # Circle shape for RANDOMIZATION
            center_x = width / 2
            center_y = height / 2
            radius = min(width, height) / 2
            
            self._path.addEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
            
            # Border path (slightly larger circle)
            self._border_path.addEllipse(center_x - radius - 6, center_y - radius - 6, (radius + 6) * 2, (radius + 6) * 2)
            
            # Glow path (even larger circle)
            self._glow_path.addEllipse(center_x - radius - 8, center_y - radius - 8, (radius + 8) * 2, (radius + 8) * 2)
            
        elif self.config.category == NodeCategory.INTERVENTION:
            # Wide rectangle with rounded corners for INTERVENTION
            self._path.addRoundedRect(0, 0, width, height, corner_radius, corner_radius)
            
            # Border path
            self._border_path.addRoundedRect(-6, -6, width + 12, height + 12, corner_radius + 2, corner_radius + 2)
            
            # Glow path
            self._glow_path.addRoundedRect(-8, -8, width + 16, height + 16, corner_radius + 5, corner_radius + 5)
            
        elif self.config.category == NodeCategory.OUTCOME:
            # Diamond shape for OUTCOME
            self._path.moveTo(width/2, 0)  # Top
            self._path.lineTo(width, height/2)  # Right
            self._path.lineTo(width/2, height)  # Bottom
            self._path.lineTo(0, height/2)  # Left
            self._path.closeSubpath()
            
            # Border path
            self._border_path.moveTo(width/2, -6)
            self._border_path.lineTo(width + 6, height/2)
            self._border_path.lineTo(width/2, height + 6)
            self._border_path.lineTo(-6, height/2)
            self._border_path.closeSubpath()
            
            # Glow path
            self._glow_path.moveTo(width/2, -8)
            self._glow_path.lineTo(width + 8, height/2)
            self._glow_path.lineTo(width/2, height + 8)
            self._glow_path.lineTo(-8, height/2)
            self._glow_path.closeSubpath()
            
        elif self.config.category == NodeCategory.SUBGROUP:
            # Hexagon shape for SUBGROUP
            points = []
            for i in range(6):
                angle = i * 60 * 3.14159 / 180
                x = width/2 + (width/2 - 5) * math.cos(angle)
                y = height/2 + (height/2 - 5) * math.sin(angle)
                points.append(QPointF(x, y))
            
            self._path.moveTo(points[0].x(), points[0].y())
            for i in range(1, 6):
                self._path.lineTo(points[i].x(), points[i].y())
            self._path.closeSubpath()
            
            # Border path (slightly larger)
            border_points = []
            for i in range(6):
                angle = i * 60 * 3.14159 / 180
                x = width/2 + (width/2 + 1) * math.cos(angle)
                y = height/2 + (height/2 + 1) * math.sin(angle)
                border_points.append(QPointF(x, y))
            
            self._border_path.moveTo(border_points[0].x(), border_points[0].y())
            for i in range(1, 6):
                self._border_path.lineTo(border_points[i].x(), border_points[i].y())
            self._border_path.closeSubpath()
            
            # Glow path (even larger)
            glow_points = []
            for i in range(6):
                angle = i * 60 * 3.14159 / 180
                x = width/2 + (width/2 + 3) * math.cos(angle)
                y = height/2 + (height/2 + 3) * math.sin(angle)
                glow_points.append(QPointF(x, y))
            
            self._glow_path.moveTo(glow_points[0].x(), glow_points[0].y())
            for i in range(1, 6):
                self._glow_path.lineTo(glow_points[i].x(), glow_points[i].y())
            self._glow_path.closeSubpath()
            
        elif self.config.category == NodeCategory.CONTROL:
            # Octagon shape for CONTROL
            points = []
            for i in range(8):
                angle = i * 45 * 3.14159 / 180
                x = width/2 + (width/2 - 5) * math.cos(angle)
                y = height/2 + (height/2 - 5) * math.sin(angle)
                points.append(QPointF(x, y))
            
            self._path.moveTo(points[0].x(), points[0].y())
            for i in range(1, 8):
                self._path.lineTo(points[i].x(), points[i].y())
            self._path.closeSubpath()
            
            # Border path (slightly larger)
            border_points = []
            for i in range(8):
                angle = i * 45 * 3.14159 / 180
                x = width/2 + (width/2 + 1) * math.cos(angle)
                y = height/2 + (height/2 + 1) * math.sin(angle)
                border_points.append(QPointF(x, y))
            
            self._border_path.moveTo(border_points[0].x(), border_points[0].y())
            for i in range(1, 8):
                self._border_path.lineTo(border_points[i].x(), border_points[i].y())
            self._border_path.closeSubpath()
            
            # Glow path (even larger)
            glow_points = []
            for i in range(8):
                angle = i * 45 * 3.14159 / 180
                x = width/2 + (width/2 + 3) * math.cos(angle)
                y = height/2 + (height/2 + 3) * math.sin(angle)
                glow_points.append(QPointF(x, y))
            
            self._glow_path.moveTo(glow_points[0].x(), glow_points[0].y())
            for i in range(1, 8):
                self._glow_path.lineTo(glow_points[i].x(), glow_points[i].y())
            self._glow_path.closeSubpath()
            
        elif self.config.category == NodeCategory.TIMEPOINT:
            # Rounded rectangle with a clock-like appearance
            self._path.addRoundedRect(0, 0, width, height, corner_radius, corner_radius)
            
            # Add a small circle in the center to suggest a clock
            clock_radius = min(width, height) / 6
            self._path.addEllipse(width/2 - clock_radius, height/2 - clock_radius, 
                                 clock_radius * 2, clock_radius * 2)
            
            # Border path
            self._border_path.addRoundedRect(-6, -6, width + 12, height + 12, corner_radius + 2, corner_radius + 2)
            
            # Glow path
            self._glow_path.addRoundedRect(-8, -8, width + 16, height + 16, corner_radius + 5, corner_radius + 5)
            
        else:
            # Default rounded rectangle for other node types
            self._path.addRoundedRect(0, 0, width, height, corner_radius, corner_radius)
            
            # Border path
            self._border_path.addRoundedRect(-6, -6, width + 12, height + 12, corner_radius + 2, corner_radius + 2)
            
            # Glow path
            self._glow_path.addRoundedRect(-8, -8, width + 16, height + 16, corner_radius + 5, corner_radius + 5)
        
        # Remove shadow effect
        self.setGraphicsEffect(None)
        
        # Force redraw
        self.update()
        
        # Make sure ports are created before trying to reposition them
        if not hasattr(self, 'input_ports') or not hasattr(self, 'output_ports'):
            self.setup_ports()
            
        # Update port positions
        self.reposition_ports(height)

    def setup_text_items(self):
        # Title (always visible)
        title_text = self.config.category.value.replace("_", " ").title()
        
        # Create HTML formatted text with better styling
        if self.config.category == NodeCategory.TARGET_POPULATION:
            # Use enhanced text item for study model
            self.title_item = EnhancedTextItem("", self)
            # Enhanced styling for study model with better font and styling
            title_html = f'''<div style="text-align: center; 
                font-weight: bold; 
                font-family: 'Segoe UI', Arial, sans-serif; 
                font-size: 17pt; 
                color: #202020; 
                letter-spacing: 1.2px;">
                {title_text}</div>'''
        else:
            self.title_item = QGraphicsTextItem(self)
            title_html = f'<div style="text-align: center; font-weight: bold; font-family: Arial; font-size: 11pt; color: #505050;">{title_text}</div>'
        
        self.title_item.setHtml(title_html)
        
        # Ensure text items don't accept mouse events
        self.title_item.setAcceptHoverEvents(False)
        self.title_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.title_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.title_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, False)
        self.title_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemAcceptsInputMethod, False)
        self.title_item.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        
        # Set text width to enable wrapping
        self.title_item.setTextWidth(self.current_width - 20)
        
        # Center the title
        title_width = self.title_item.boundingRect().width()
        
        # For study model, center the title vertically as well
        if self.config.category == NodeCategory.TARGET_POPULATION:
            self.title_item.setPos((self.current_width - title_width) / 2, (self.current_height - self.title_item.boundingRect().height()) / 2)
        else:
            self.title_item.setPos((self.current_width - title_width) / 2, 10)
        
        # Description with text wrapping
        self.desc_item = QGraphicsTextItem(self)
        
        # Ensure description doesn't accept mouse events
        self.desc_item.setAcceptHoverEvents(False)
        self.desc_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.desc_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.desc_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, False)
        self.desc_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemAcceptsInputMethod, False)
        self.desc_item.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        
        # Create HTML formatted description with wrapping
        desc_html = f'<div style="text-align: center; font-family: Arial; font-size: 9pt; color: #505050;">{self.config.description}</div>'
        self.desc_item.setHtml(desc_html)
        
        # Set text width to enable wrapping
        self.desc_item.setTextWidth(self.current_width - 20)
        
        # Position description below title
        desc_y = 40
        if self.config.center_node:
            # Hide description for study model since we're centering the title
            self.desc_item.setVisible(False)
        
        self.desc_item.setPos(10, desc_y)
        
        # Detail item (for expanded view)
        self.detail_item = QGraphicsTextItem(self)
        self.detail_item.setTextWidth(self.current_width - 20)
        
        # Ensure detail item doesn't accept mouse events
        self.detail_item.setAcceptHoverEvents(False)
        self.detail_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.detail_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.detail_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, False)
        self.detail_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemAcceptsInputMethod, False)
        
        # Create HTML formatted details with wrapping
        detail_html = '<div style="font-family: Arial; font-size: 8pt; color: #505050;"></div>'
        self.detail_item.setHtml(detail_html)
        
        self.detail_item.setPos(10, 70)
        self.detail_item.setVisible(False)

    def setup_ports(self):
        # Initialize empty port lists
        self.input_ports = []
        self.output_ports = []
        
        # Border offset - distance from the node edge to place ports
        border_offset = 10  # Increased from 6 to account for larger port size
        
        # Create ports based on node type
        if self.config.category == NodeCategory.TARGET_POPULATION:
            # Right side - for Eligible Population
            eligible_population_port = PortItem(self.current_width + border_offset, self.current_height * 0.5, "output", "Eligible Population", self)
            self.output_ports.append(eligible_population_port)
            
        elif self.config.category == NodeCategory.ELIGIBLE_POPULATION:
            # Triangle shape - ports on each side
            # Left port for connection from target population
            input_port = PortItem(-border_offset, self.current_height * 0.5, "input", "TargetPopulation", self)
            self.input_ports.append(input_port)
            
            # Right port for connection to Randomization node
            randomization_port = PortItem(self.current_width + border_offset, self.current_height * 0.5, "output", "Randomization", self)
            self.output_ports.append(randomization_port)
            
            # Bottom port for connection to Subgroup
            subgroup_port = PortItem(self.current_width * 0.5, self.current_height + border_offset, "output", "Subgroup", self)
            self.output_ports.append(subgroup_port)
            
        elif self.config.category == NodeCategory.RANDOMIZATION:
            # Circular shape - radially opposite ports
            center_x = self.current_width / 2
            center_y = self.current_height / 2
            radius = min(self.current_width, self.current_height) / 2
            
            # Left port for input from Eligible Population
            input_port = PortItem(-border_offset, center_y, "input", "", self)
            self.input_ports.append(input_port)
            
            # Output ports - positioned radially on the right half
            # To Subgroup
            subgroup_port = PortItem(self.current_width + border_offset, center_y - radius * 0.5, "output", "Subgroup", self)
            self.output_ports.append(subgroup_port)
            
            # To Intervention
            intervention_port = PortItem(self.current_width + border_offset, center_y, "output", "Intervention", self)
            self.output_ports.append(intervention_port)
            
            # To Control
            control_port = PortItem(self.current_width + border_offset, center_y + radius * 0.5, "output", "Control", self)
            self.output_ports.append(control_port)
            
        elif self.config.category == NodeCategory.SUBGROUP:
            # Hexagon shape - position ports at vertices
            # Left port for input connection
            input_port = PortItem(-border_offset, self.current_height * 0.5, "input", "", self)
            self.input_ports.append(input_port)
            
            # Right-top port for outcome connection
            outcome_port = PortItem(self.current_width + border_offset, self.current_height * 0.3, "output", "Outcome", self)
            self.output_ports.append(outcome_port)
            
            # Right-bottom port for intervention connection
            intervention_port = PortItem(self.current_width + border_offset, self.current_height * 0.7, "output", "Intervention", self)
            self.output_ports.append(intervention_port)
            
        elif self.config.category == NodeCategory.OUTCOME:
            # Diamond shape - position ports at vertices
            # Left port for input connection
            input_port = PortItem(-border_offset, self.current_height * 0.5, "input", "", self)
            self.input_ports.append(input_port)
            
            # Top port for timepoint connection
            timepoint_port = PortItem(self.current_width * 0.5, -border_offset, "output", "Timepoint", self)
            self.output_ports.append(timepoint_port)
            
        elif self.config.category == NodeCategory.INTERVENTION:
            # Wide rectangle - position ports on sides
            # Left port for connection from various sources
            input_port = PortItem(-border_offset, self.current_height * 0.5, "input", "", self)
            self.input_ports.append(input_port)
            
            # Intervention has no output ports
            
        elif self.config.category == NodeCategory.CONTROL:
            # Octagon shape - position ports at vertices
            # Left port for input from randomization
            input_port = PortItem(-border_offset, self.current_height * 0.5, "input", "", self)
            self.input_ports.append(input_port)
            
            # Right port to outcome
            outcome_port = PortItem(self.current_width + border_offset, self.current_height * 0.5, "output", "Outcome", self)
            self.output_ports.append(outcome_port)
            
        elif self.config.category == NodeCategory.TIMEPOINT:
            # Clock-like shape
            # Input port (from outcome)
            input_port = PortItem(-border_offset, self.current_height * 0.5, "input", "", self)
            self.input_ports.append(input_port)
            
            # Timepoint is an end node, no output ports needed
            
        else:
            # Default port for other node types
            port = PortItem(self.current_width / 2, self.current_height + border_offset, "output", "", self)
            self.output_ports.append(port)

    def set_active(self, active=True):
        """Set node as active/highlighted"""
        self.is_active = active
        self.update()

    def update_theme(self, is_dark_theme):
        """Update the node's colors based on the current theme."""
        from model_builder.theme_support import update_node_theme
        update_node_theme(self, is_dark_theme)

    def toggle_expand(self):
        if not self.expanded:
            # Expand: increase height and show details
            new_height = self.height + 100
            self.updatePath(self.width, new_height)
            
            # Update detail item with HTML formatting
            if self.config_details:
                detail_html = f'<div style="font-family: Arial; font-size: 8pt; color: #505050;">{self.config_details}</div>'
                self.detail_item.setHtml(detail_html)
                
            self.detail_item.setVisible(True)
            # reposition ports
            self.reposition_ports(new_height)
            self.expanded = True
        else:
            # Collapse: revert to base height
            self.updatePath(self.width, self.height)
            self.detail_item.setVisible(False)
            self.reposition_ports(self.height)
            self.expanded = False

    def reposition_ports(self, new_height):
        # Border offset - distance from the node edge to place ports
        border_offset = 10  # Increased from 6 to account for larger port size
        
        if self.config.category == NodeCategory.TARGET_POPULATION:
            # Right side - for Eligible Population
            if len(self.output_ports) > 0:
                self.output_ports[0].setPos(self.current_width + border_offset, new_height * 0.5)  # Eligible Population
            
        elif self.config.category == NodeCategory.ELIGIBLE_POPULATION:
            # Triangle shape - ports on each side
            if len(self.input_ports) >= 1:
                # Left port (from target population)
                self.input_ports[0].setPos(-border_offset, new_height * 0.5)
            
            if len(self.output_ports) >= 2:
                # Right port (to randomization)
                self.output_ports[0].setPos(self.current_width + border_offset, new_height * 0.5)
                
                # Bottom port (to subgroup)
                self.output_ports[1].setPos(self.current_width * 0.5, new_height + border_offset)
            
        elif self.config.category == NodeCategory.RANDOMIZATION:
            # Circular shape - radially opposite ports
            center_x = self.current_width / 2
            center_y = new_height / 2
            radius = min(self.current_width, new_height) / 2
            
            if len(self.input_ports) >= 1:
                # Left port (from eligible population) - at 180 degrees
                self.input_ports[0].setPos(-border_offset, center_y)
            
            if len(self.output_ports) >= 3:
                # Position output ports radially
                # To Subgroup (top-right)
                self.output_ports[0].setPos(self.current_width + border_offset, center_y - radius * 0.5)
                
                # To Intervention (right)
                self.output_ports[1].setPos(self.current_width + border_offset, center_y)
                
                # To Control (bottom-right)
                self.output_ports[2].setPos(self.current_width + border_offset, center_y + radius * 0.5)
            
        elif self.config.category == NodeCategory.SUBGROUP:
            # Hexagon shape - position ports at vertices
            if len(self.input_ports) >= 1:
                # Left port (input)
                self.input_ports[0].setPos(-border_offset, new_height * 0.5)
            
            if len(self.output_ports) >= 2:
                # Right-top port (to outcome)
                self.output_ports[0].setPos(self.current_width + border_offset, new_height * 0.3)
                
                # Right-bottom port (to intervention)
                self.output_ports[1].setPos(self.current_width + border_offset, new_height * 0.7)
            
        elif self.config.category == NodeCategory.OUTCOME:
            # Diamond shape - position ports at vertices
            if len(self.input_ports) >= 1:
                # Left port (input)
                self.input_ports[0].setPos(-border_offset, new_height * 0.5)
            
            if len(self.output_ports) >= 1:
                # Top port (to timepoint)
                self.output_ports[0].setPos(self.current_width * 0.5, -border_offset)
            
        elif self.config.category == NodeCategory.INTERVENTION:
            # Wide rectangle - position ports on sides
            if len(self.input_ports) >= 1:
                # Left port (input)
                self.input_ports[0].setPos(-border_offset, new_height * 0.5)
            
            # Intervention has no output ports
            
        elif self.config.category == NodeCategory.CONTROL:
            # Octagon shape - position ports at vertices
            if len(self.input_ports) >= 1:
                # Left port (input from randomization)
                self.input_ports[0].setPos(-border_offset, new_height * 0.5)
            
            if len(self.output_ports) >= 1:
                # Right port (to outcome)
                self.output_ports[0].setPos(self.current_width + border_offset, new_height * 0.5)
            
        elif self.config.category == NodeCategory.TIMEPOINT:
            # Clock-like shape
            if len(self.input_ports) >= 1:
                # Input port (from outcome)
                self.input_ports[0].setPos(-border_offset, new_height * 0.5)
            
            # Timepoint has no output ports

    def contextMenuEvent(self, event):
        menu = QMenu()
        expand_action = menu.addAction(load_bootstrap_icon("arrows-expand"), "Expand/Collapse")
        config_action = menu.addAction(load_bootstrap_icon("gear"), "Configure Node")
        
        # Add direct action option based on node type
        if self.config.action_handler:
            action_name = f"Open {self.config.category.value.replace('_', ' ').title()}"
            direct_action = menu.addAction(load_bootstrap_icon("pencil-square"), action_name)
        
        menu.addSeparator()
        mark_active = menu.addAction(load_bootstrap_icon("check-circle"), "Set as Active Node")
        
        # Add delete option with keyboard shortcut hint
        menu.addSeparator()
        delete_action = menu.addAction(load_bootstrap_icon("trash"), "Delete Node (Delete)")
        
        action = menu.exec(event.screenPos())
        if action == expand_action:
            self.toggle_expand()
        elif action == config_action:
            self.show_configuration_dialog()
        elif self.config.action_handler and action == direct_action:
            self.nodeActivated.emit(self)
        elif action == mark_active:
            # Clear active state on all nodes in scene
            for item in self.scene().items():
                if isinstance(item, WorkflowNode) and item != self:
                    item.set_active(False)
            self.set_active(True)
        elif action == delete_action:
            scene = self.scene()
            if scene:
                scene.removeItem(self)

    def show_configuration_dialog(self):
        dialog = NodeConfigurationDialog(self.config.category)
        if dialog.exec():
            # Update configuration details and description text from dialog
            details = dialog.get_details()
            self.config_details = details
            if self.expanded:
                self.detail_item.setPlainText(self.config_details)

    def mouseDoubleClickEvent(self, event):
        # Emit signal that node was activated
        self.nodeActivated.emit(self)
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press events for node dragging"""
        # Store the initial position for dragging
        self._drag_start_pos = event.scenePos()
        
        # Bring to front during drag for better visual feedback
        self.setZValue(1000)
        
        # Ensure the event is accepted to prevent it from propagating to child items
        event.accept()
        
        # Call parent implementation
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse movement for node dragging"""
        # Only process if we're in a drag operation with left button
        if event.buttons() & Qt.MouseButton.LeftButton and self._drag_start_pos is not None:
            # Calculate the movement delta in scene coordinates
            current_scene_pos = event.scenePos()
            delta = current_scene_pos - self._drag_start_pos
            
            # Update the node position
            self.setPos(self.pos() + delta)
            
            # Update the drag start position to the current scene position
            self._drag_start_pos = current_scene_pos
            
            # Update connected edges
            if self.scene():
                for port in self.input_ports + self.output_ports:
                    for edge in self.scene().items():
                        if isinstance(edge, EdgeItem) and (edge.start_port == port or edge.end_port == port):
                            edge.updatePosition()
            
            # Emit position changed signals
            self.xChanged.emit()
            self.yChanged.emit()
            
            # Accept the event to prevent it from being processed further
            event.accept()
        else:
            # For non-drag operations, use the parent implementation
            super().mouseMoveEvent(event)

    def hoverEnterEvent(self, event):
        """Handle hover enter events"""
        self.is_hovered = True
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update()
        super().hoverEnterEvent(event)
        
    def hoverLeaveEvent(self, event):
        """Handle hover leave events"""
        self.is_hovered = False
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()
        super().hoverLeaveEvent(event)
        
    def itemChange(self, change, value):
        """Handle item changes, particularly selection changes"""
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedChange:
            # Force a redraw when selection state changes
            self.update()
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events for node dragging"""
        if self._drag_start_pos is not None:
            # Clean up after dragging
            self._drag_start_pos = None
            
            # Reset Z value to normal after a short delay to prevent flickering
            QTimer.singleShot(100, lambda: self.setZValue(0))
            
            # Emit position changed signals one final time
            self.xChanged.emit()
            self.yChanged.emit()
            
            event.accept()
        else:
            # For non-drag operations, use the parent implementation
            super().mouseReleaseEvent(event)

    def setup_plus_button(self):
        """Add a plus button to the center of the Study Model node"""
        # This method has been modified to not create a plus button
        # The plus button was previously used to add new Study Model nodes
        self.plus_button = None
        
        # Note: We're setting plus_button to None instead of creating it
        # This ensures any code that checks for self.plus_button will work correctly
        # but no button will be displayed in the center of the study model

class NodeConfigurationDialog(QDialog):
    def __init__(self, node_category: NodeCategory):
        super().__init__()
        self.node_category = node_category
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle(f"Configure {self.node_category.value.replace('_', ' ').title()}")
        self.layout = QVBoxLayout()

        # Different configuration UI based on node type
        if self.node_category == NodeCategory.TARGET_POPULATION:
            self.setup_target_population_ui()
        elif self.node_category == NodeCategory.INTERVENTION:
            self.setup_intervention_ui()
        elif self.node_category == NodeCategory.OUTCOME:
            self.setup_outcome_ui()
        elif self.node_category == NodeCategory.SUBGROUP:
            self.setup_subgroup_ui()
        elif self.node_category == NodeCategory.CONTROL:
            self.setup_control_ui()
        else:
            # Default configuration: a simple description
            self.layout.addWidget(QLabel(f"Configure {self.node_category.value.replace('_', ' ').title()}:"))
            self.desc_edit = QTextEdit()
            self.layout.addWidget(self.desc_edit)

        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        self.layout.addLayout(button_layout)
        self.setLayout(self.layout)

    def setup_target_population_ui(self):
        # Study Title
        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("Target Population Description:"))
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        self.layout.addWidget(self.description_edit)
        
        # Additional Details
        self.layout.addWidget(QLabel("Additional Details:"))
        self.details_edit = QTextEdit()
        self.details_edit.setMaximumHeight(80)
        self.layout.addWidget(self.details_edit)

    def setup_intervention_ui(self):
        self.layout.addWidget(QLabel("Intervention Description:"))
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        self.layout.addWidget(self.description_edit)
        
        self.layout.addWidget(QLabel("Additional Details:"))
        self.details_edit = QTextEdit()
        self.details_edit.setMaximumHeight(80)
        self.layout.addWidget(self.details_edit)

    def setup_outcome_ui(self):
        self.layout.addWidget(QLabel("Outcome Description:"))
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        self.layout.addWidget(self.description_edit)
        
        self.layout.addWidget(QLabel("Additional Details:"))
        self.details_edit = QTextEdit()
        self.details_edit.setMaximumHeight(80)
        self.layout.addWidget(self.details_edit)

    def setup_subgroup_ui(self):
        self.layout.addWidget(QLabel("Subgroup Description:"))
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        self.layout.addWidget(self.description_edit)
        
        self.layout.addWidget(QLabel("Additional Details:"))
        self.details_edit = QTextEdit()
        self.details_edit.setMaximumHeight(80)
        self.layout.addWidget(self.details_edit)

    def setup_control_ui(self):
        self.layout.addWidget(QLabel("Control Description:"))
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        self.layout.addWidget(self.description_edit)
        
        self.layout.addWidget(QLabel("Additional Details:"))
        self.details_edit = QTextEdit()
        self.details_edit.setMaximumHeight(80)
        self.layout.addWidget(self.details_edit)

    def get_details(self):
        """Return a string summary of the configuration."""
        if self.node_category == NodeCategory.TARGET_POPULATION:
            return (f"Description: {self.description_edit.toPlainText()}\n"
                    f"Details: {self.details_edit.toPlainText()}")
        elif self.node_category == NodeCategory.INTERVENTION:
            return (f"Description: {self.description_edit.toPlainText()}\n"
                    f"Details: {self.details_edit.toPlainText()}")
        elif self.node_category == NodeCategory.OUTCOME:
            return (f"Description: {self.description_edit.toPlainText()}\n"
                    f"Details: {self.details_edit.toPlainText()}")
        elif self.node_category == NodeCategory.SUBGROUP:
            return (f"Description: {self.description_edit.toPlainText()}\n"
                    f"Details: {self.details_edit.toPlainText()}")
        elif self.node_category == NodeCategory.CONTROL:
            return (f"Description: {self.description_edit.toPlainText()}\n"
                    f"Details: {self.details_edit.toPlainText()}")
        elif hasattr(self, 'desc_edit'):
            return self.desc_edit.toPlainText()
        else:
            return "Configuration details not available."

class EnhancedTextItem(QGraphicsTextItem):
    """Custom text item with enhanced visual effects"""
    
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        # Make the text item completely non-interactive
        self.setAcceptHoverEvents(False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemAcceptsInputMethod, False)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        
    def paint(self, painter, option, widget):
        painter.save()
        
        # Create a subtle glow effect
        painter.setRenderHint(painter.RenderHint.Antialiasing)
        
        # Draw the text with a more subtle shadow (reduced offset from 1,1 to 0.5,0.5)
        doc = self.document()
        painter.translate(0.5, 0.5)
        painter.setOpacity(0.25)  # Reduced opacity from 0.3 to 0.25
        painter.translate(self.document().documentMargin(), self.document().documentMargin())
        doc.drawContents(painter)
        painter.translate(-0.5, -0.5)
        
        # Draw the actual text
        painter.setOpacity(1.0)
        painter.translate(self.document().documentMargin(), self.document().documentMargin())
        doc.drawContents(painter)
        
        painter.restore()
 
# ============================================================================
# JSON WORKFLOW UTILITIES
# ============================================================================

import json

def parse_workflow_json(scene, json_string):
    """
    Parse a JSON string containing a workflow definition and load it into the provided scene.
    
    Args:
        scene: WorkflowScene instance to load the workflow into
        json_string (str): JSON string defining a workflow
        
    Returns:
        tuple: (bool success, dict validation_results)
    """
    try:
        # Parse the JSON string
        workflow_data = json.loads(json_string)
        
        # Load the workflow using the scene's loader
        if hasattr(scene, 'load_from_workflow_json'):
            return scene.load_from_workflow_json(workflow_data)
        else:
            return False, {"valid": False, "errors": ["Scene does not support JSON workflow loading"]}
    except json.JSONDecodeError as e:
        return False, {"valid": False, "errors": [f"Invalid JSON format: {str(e)}"]}
    except Exception as e:
        return False, {"valid": False, "errors": [f"Error parsing workflow: {str(e)}"]}

def create_study_design_from_llm_json(scene, design_description, patient_count=1000):
    """
    Create a workflow design in the given scene from a natural language description.
    This function would integrate with an LLM to generate the workflow JSON.
    
    Args:
        scene: WorkflowScene instance to load the workflow into
        design_description (str): Description of the study design
        patient_count (int): Total target population count
        
    Returns:
        tuple: (bool success, dict validation_results)
    """
    # For demonstration, we'll create different templates based on the description
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
    
    # Load the workflow into the scene
    if hasattr(scene, 'load_from_workflow_json'):
        return scene.load_from_workflow_json(workflow_json)
    else:
        return False, {"valid": False, "errors": ["Scene does not support JSON workflow loading"]}

def validate_workflow_json(workflow_json):
    """
    Validate a workflow JSON structure without loading it.
    
    Args:
        workflow_json (dict): The workflow JSON data to validate
        
    Returns:
        dict: Validation results with valid status, warnings, and errors
    """
    validation = {"valid": True, "warnings": [], "errors": []}
    
    # Basic structure validation
    if not isinstance(workflow_json, dict):
        validation["valid"] = False
        validation["errors"].append("Workflow JSON must be a dictionary")
        return validation
        
    if "nodes" not in workflow_json or not isinstance(workflow_json["nodes"], list):
        validation["valid"] = False
        validation["errors"].append("Workflow JSON must contain a 'nodes' list")
        return validation
        
    if "edges" not in workflow_json or not isinstance(workflow_json["edges"], list):
        validation["valid"] = False
        validation["errors"].append("Workflow JSON must contain an 'edges' list")
        return validation
    
    # Validate nodes have required fields
    node_ids = set()
    for i, node in enumerate(workflow_json["nodes"]):
        if "id" not in node:
            validation["warnings"].append(f"Node at index {i} is missing an 'id' field")
            continue
            
        node_id = node["id"]
        if node_id in node_ids:
            validation["warnings"].append(f"Duplicate node ID: {node_id}")
        node_ids.add(node_id)
        
        if "type" not in node:
            validation["warnings"].append(f"Node {node_id} is missing a 'type' field")
            continue
            
        # Check if type is valid
        try:
            NodeCategory(node["type"])
        except ValueError:
            validation["warnings"].append(f"Node {node_id} has invalid type: {node['type']}")
    
    # Validate edges refer to existing nodes
    for i, edge in enumerate(workflow_json["edges"]):
        if "source" not in edge:
            validation["warnings"].append(f"Edge at index {i} is missing a 'source' field")
            continue
            
        if "target" not in edge:
            validation["warnings"].append(f"Edge at index {i} is missing a 'target' field")
            continue
            
        source = edge["source"]
        target = edge["target"]
        
        if source not in node_ids:
            validation["warnings"].append(f"Edge references non-existent source node: {source}")
            
        if target not in node_ids:
            validation["warnings"].append(f"Edge references non-existent target node: {target}")
    
    # Validate patient counts if we have multiple subgroups
    subgroup_nodes = [n for n in workflow_json["nodes"] if n.get("type") == "subgroup"]
    if len(subgroup_nodes) > 1:
        # Get eligible population node if it exists
        eligible_nodes = [n for n in workflow_json["nodes"] if n.get("type") == "eligible_population"]
        if eligible_nodes:
            eligible_id = eligible_nodes[0]["id"]
            
            # Get edges to subgroups from eligible population
            subgroup_edges = [e for e in workflow_json["edges"] 
                               if e.get("source") == eligible_id and
                               any(n.get("id") == e.get("target") for n in subgroup_nodes)]
            
            # Check total patient counts
            if subgroup_edges:
                total_subgroup_patients = sum(e.get("patient_count", 0) for e in subgroup_edges)
                
                # Find eligible population patient count
                eligible_edges = [e for e in workflow_json["edges"] if e.get("target") == eligible_id]
                eligible_count = eligible_edges[0].get("patient_count", 0) if eligible_edges else 0
                
                if eligible_count > 0 and abs(total_subgroup_patients - eligible_count) > 0.001:
                    validation["warnings"].append(
                        f"With multiple subgroups, total patient count ({total_subgroup_patients}) " +
                        f"should match eligible population ({eligible_count})"
                    )
    
    return validation

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
Example of using the workflow JSON API to create and validate study designs:

1. Basic usage with a scene:

```python
from model_builder.config import create_study_design_from_llm_json

# Create a parallel group design with 500 patients
success, validation = create_study_design_from_llm_json(
    scene,  # WorkflowScene instance
    "randomized controlled trial with parallel groups",
    patient_count=500
)

if success:
    print("Successfully created workflow")
else:
    print("Failed to create workflow:", validation["errors"])
    
# Show any warnings
if validation["warnings"]:
    print("Warnings:", validation["warnings"])
```

2. Parsing a JSON string:

```python
from model_builder.config import parse_workflow_json

# JSON string from an external source or LLM
json_string = '''
{
    "design_type": "pre_post",
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
            "label": "Study Population",
            "x": 300,
            "y": 0,
            "patient_count": 300
        }
    ],
    "edges": [
        {
            "source": "population",
            "target": "eligible",
            "patient_count": 300,
            "label": "Screening"
        }
    ]
}
'''

success, validation = parse_workflow_json(scene, json_string)

if success:
    print("Successfully loaded workflow from JSON")
else:
    print("Failed to load workflow:", validation["errors"])
```

3. Integration with an LLM:

```python
import json
import requests

def create_workflow_from_llm(scene, design_description, patient_count=1000):
    # Send request to LLM API
    response = requests.post(
        "https://api.example.com/llm",
        json={
            "prompt": f"Create a JSON workflow for a {design_description} study design with {patient_count} patients",
            "format": "json"
        }
    )
    
    # Parse the response
    try:
        workflow_json = json.loads(response.text)
        
        # Validate the workflow
        validation = validate_workflow_json(workflow_json)
        if not validation["valid"]:
            return False, validation
            
        # Load the workflow
        return scene.load_from_workflow_json(workflow_json)
    except Exception as e:
        return False, {"valid": False, "errors": [f"Error processing LLM response: {str(e)}"]}
"""
 