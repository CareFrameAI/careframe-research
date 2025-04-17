import math

from PyQt6.QtWidgets import (
    QGraphicsObject, QGraphicsItem, QGraphicsPathItem,
    QGraphicsTextItem, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMenu, QListWidget, QListWidgetItem, QMessageBox, QWidget, QTabWidget
)
from PyQt6.QtGui import (
    QPen, QBrush, QColor, QFont, QPainterPath, QPainterPathStroker,
    QLinearGradient, QTextOption
)
from PyQt6.QtCore import (
    Qt, QPointF, QRectF, pyqtSignal, QPropertyAnimation, QLineF
)

from PyQt6.QtCore import Property

# First add these imports at the top with other imports
from helpers.load_icon import load_bootstrap_icon
from plan.plan_config import NODE_CONFIGS, ConnectionType, HypothesisState
from plan.plan_dialogs import EvidenceDialog

# ======================
# Evidence Panel
# ======================

class EvidencePanel(QWidget):
    """Panel for managing evidence for a hypothesis"""
    
    evidenceChanged = pyqtSignal(object)  # Signal when evidence is changed
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.hypothesis = None
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI components"""
        layout = QVBoxLayout(self)
        
        # Header with title
        header_layout = QHBoxLayout()
        self.title_label = QLabel("Evidence Panel")
        self.title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        header_layout.addWidget(self.title_label)
        
        # Close button
        self.close_btn = QPushButton()
        self.close_btn.setIcon(load_bootstrap_icon("x"))
        self.close_btn.setFixedSize(24, 24)
        self.close_btn.clicked.connect(self.hide)
        header_layout.addWidget(self.close_btn)
        
        layout.addLayout(header_layout)
        
        # Tabs for supporting vs contradicting
        self.evidence_tabs = QTabWidget()
        
        # Supporting evidence tab
        self.supporting_widget = QWidget()
        supporting_layout = QVBoxLayout(self.supporting_widget)
        
        # Supporting evidence list
        self.supporting_list = QListWidget()
        self.supporting_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.supporting_list.itemSelectionChanged.connect(self.on_supporting_selection_changed)
        supporting_layout.addWidget(self.supporting_list)
        
        # Supporting buttons
        supporting_btn_layout = QHBoxLayout()
        
        self.add_supporting_btn = QPushButton()
        self.add_supporting_btn.setIcon(load_bootstrap_icon("plus-circle"))
        self.add_supporting_btn.setText("Add")
        self.add_supporting_btn.clicked.connect(self.add_supporting_evidence)
        supporting_btn_layout.addWidget(self.add_supporting_btn)
        
        self.edit_supporting_btn = QPushButton()
        self.edit_supporting_btn.setIcon(load_bootstrap_icon("pencil"))
        self.edit_supporting_btn.setText("Edit")
        self.edit_supporting_btn.clicked.connect(self.edit_supporting_evidence)
        self.edit_supporting_btn.setEnabled(False)
        supporting_btn_layout.addWidget(self.edit_supporting_btn)
        
        self.delete_supporting_btn = QPushButton()
        self.delete_supporting_btn.setIcon(load_bootstrap_icon("trash"))
        self.delete_supporting_btn.setText("Delete")
        self.delete_supporting_btn.clicked.connect(self.delete_supporting_evidence)
        self.delete_supporting_btn.setEnabled(False)
        supporting_btn_layout.addWidget(self.delete_supporting_btn)
        
        supporting_layout.addLayout(supporting_btn_layout)
        
        # Add tab
        self.evidence_tabs.addTab(self.supporting_widget, "Supporting Evidence")
        
        # Contradicting evidence tab
        self.contradicting_widget = QWidget()
        contradicting_layout = QVBoxLayout(self.contradicting_widget)
        
        # Contradicting evidence list
        self.contradicting_list = QListWidget()
        self.contradicting_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.contradicting_list.itemSelectionChanged.connect(self.on_contradicting_selection_changed)
        contradicting_layout.addWidget(self.contradicting_list)
        
        # Contradicting buttons
        contradicting_btn_layout = QHBoxLayout()
        
        self.add_contradicting_btn = QPushButton()
        self.add_contradicting_btn.setIcon(load_bootstrap_icon("plus-circle"))
        self.add_contradicting_btn.setText("Add")
        self.add_contradicting_btn.clicked.connect(self.add_contradicting_evidence)
        contradicting_btn_layout.addWidget(self.add_contradicting_btn)
        
        self.edit_contradicting_btn = QPushButton()
        self.edit_contradicting_btn.setIcon(load_bootstrap_icon("pencil"))
        self.edit_contradicting_btn.setText("Edit")
        self.edit_contradicting_btn.clicked.connect(self.edit_contradicting_evidence)
        self.edit_contradicting_btn.setEnabled(False)
        contradicting_btn_layout.addWidget(self.edit_contradicting_btn)
        
        self.delete_contradicting_btn = QPushButton()
        self.delete_contradicting_btn.setIcon(load_bootstrap_icon("trash"))
        self.delete_contradicting_btn.setText("Delete")
        self.delete_contradicting_btn.clicked.connect(self.delete_contradicting_evidence)
        self.delete_contradicting_btn.setEnabled(False)
        contradicting_btn_layout.addWidget(self.delete_contradicting_btn)
        
        contradicting_layout.addLayout(contradicting_btn_layout)
        
        # Add tab
        self.evidence_tabs.addTab(self.contradicting_widget, "Contradicting Evidence")
        
        layout.addWidget(self.evidence_tabs)
    
    def set_hypothesis(self, hypothesis):
        """Set the hypothesis to display evidence for"""
        self.hypothesis = hypothesis
        
        if hypothesis:
            self.title_label.setText(f"Evidence for: {hypothesis.node_data['text'][:40]}...")
            self.update_evidence_lists()
        else:
            self.title_label.setText("Evidence Panel")
            self.supporting_list.clear()
            self.contradicting_list.clear()
    
    def update_evidence_lists(self):
        """Update evidence lists from hypothesis"""
        if not self.hypothesis:
            return
        
        # Update supporting evidence
        self.supporting_list.clear()
        for evidence in self.hypothesis.hypothesis_config.supporting_evidence:
            item = QListWidgetItem(f"{evidence.type.value.title()}: {evidence.description[:60]}...")
            item.setData(Qt.ItemDataRole.UserRole, evidence)
            self.supporting_list.addItem(item)
        
        # Update contradicting evidence
        self.contradicting_list.clear()
        for evidence in self.hypothesis.hypothesis_config.contradicting_evidence:
            item = QListWidgetItem(f"{evidence.type.value.title()}: {evidence.description[:60]}...")
            item.setData(Qt.ItemDataRole.UserRole, evidence)
            self.contradicting_list.addItem(item)
    
    def on_supporting_selection_changed(self):
        """Handle selection change in supporting evidence list"""
        self.edit_supporting_btn.setEnabled(len(self.supporting_list.selectedItems()) > 0)
        self.delete_supporting_btn.setEnabled(len(self.supporting_list.selectedItems()) > 0)
    
    def on_contradicting_selection_changed(self):
        """Handle selection change in contradicting evidence list"""
        self.edit_contradicting_btn.setEnabled(len(self.contradicting_list.selectedItems()) > 0)
        self.delete_contradicting_btn.setEnabled(len(self.contradicting_list.selectedItems()) > 0)
    
    def add_supporting_evidence(self):
        """Add new supporting evidence"""
        if not self.hypothesis:
            return
        
        # Create dialog
        dialog = EvidenceDialog(self)
        dialog.setWindowTitle("Add Supporting Evidence")
        
        if dialog.exec():
            # Get data from dialog
            evidence_data = dialog.get_evidence_data()
            evidence_data.supports = True
            
            # Add to hypothesis
            self.hypothesis.hypothesis_config.supporting_evidence.append(evidence_data)
            
            # Update display
            self.update_evidence_lists()
            self.hypothesis.update_evidence_summary()
            
            # Emit signal
            self.evidenceChanged.emit(self.hypothesis)
    
    def edit_supporting_evidence(self):
        """Edit selected supporting evidence"""
        selected_items = self.supporting_list.selectedItems()
        if not selected_items or not self.hypothesis:
            return
        
        # Get evidence
        evidence = selected_items[0].data(Qt.ItemDataRole.UserRole)
        
        # Show dialog
        dialog = EvidenceDialog(self, evidence)
        dialog.setWindowTitle("Edit Supporting Evidence")
        
        if dialog.exec():
            # Get updated data
            updated_data = dialog.get_evidence_data()
            updated_data.supports = True
            updated_data.id = evidence.id
            
            # Update in hypothesis
            for i, ev in enumerate(self.hypothesis.hypothesis_config.supporting_evidence):
                if ev.id == evidence.id:
                    self.hypothesis.hypothesis_config.supporting_evidence[i] = updated_data
                    break
            
            # Update display
            self.update_evidence_lists()
            self.hypothesis.update_evidence_summary()
            
            # Emit signal
            self.evidenceChanged.emit(self.hypothesis)
    
    def delete_supporting_evidence(self):
        """Delete selected supporting evidence"""
        selected_items = self.supporting_list.selectedItems()
        if not selected_items or not self.hypothesis:
            return
        
        # Get evidence
        evidence = selected_items[0].data(Qt.ItemDataRole.UserRole)
        
        # Confirm deletion
        confirm = QMessageBox.question(
            self,
            "Confirm Deletion",
            "Are you sure you want to delete this evidence?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if confirm == QMessageBox.StandardButton.Yes:
            # Remove from hypothesis
            self.hypothesis.hypothesis_config.supporting_evidence = [
                ev for ev in self.hypothesis.hypothesis_config.supporting_evidence 
                if ev.id != evidence.id
            ]
            
            # Update display
            self.update_evidence_lists()
            self.hypothesis.update_evidence_summary()
            
            # Emit signal
            self.evidenceChanged.emit(self.hypothesis)
    
    def add_contradicting_evidence(self):
        """Add new contradicting evidence"""
        if not self.hypothesis:
            return
        
        # Create dialog
        dialog = EvidenceDialog(self)
        dialog.setWindowTitle("Add Contradicting Evidence")
        
        if dialog.exec():
            # Get data from dialog
            evidence_data = dialog.get_evidence_data()
            evidence_data.supports = False
            
            # Add to hypothesis
            self.hypothesis.hypothesis_config.contradicting_evidence.append(evidence_data)
            
            # Update display
            self.update_evidence_lists()
            self.hypothesis.update_evidence_summary()
            
            # Emit signal
            self.evidenceChanged.emit(self.hypothesis)
    
    def edit_contradicting_evidence(self):
        """Edit selected contradicting evidence"""
        selected_items = self.contradicting_list.selectedItems()
        if not selected_items or not self.hypothesis:
            return
        
        # Get evidence
        evidence = selected_items[0].data(Qt.ItemDataRole.UserRole)
        
        # Show dialog
        dialog = EvidenceDialog(self, evidence)
        dialog.setWindowTitle("Edit Contradicting Evidence")
        
        if dialog.exec():
            # Get updated data
            updated_data = dialog.get_evidence_data()
            updated_data.supports = False
            updated_data.id = evidence.id
            
            # Update in hypothesis
            for i, ev in enumerate(self.hypothesis.hypothesis_config.contradicting_evidence):
                if ev.id == evidence.id:
                    self.hypothesis.hypothesis_config.contradicting_evidence[i] = updated_data
                    break
            
            # Update display
            self.update_evidence_lists()
            self.hypothesis.update_evidence_summary()
            
            # Emit signal
            self.evidenceChanged.emit(self.hypothesis)
    
    def delete_contradicting_evidence(self):
        """Delete selected contradicting evidence"""
        selected_items = self.contradicting_list.selectedItems()
        if not selected_items or not self.hypothesis:
            return
        
        # Get evidence
        evidence = selected_items[0].data(Qt.ItemDataRole.UserRole)
        
        # Confirm deletion
        confirm = QMessageBox.question(
            self,
            "Confirm Deletion",
            "Are you sure you want to delete this evidence?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if confirm == QMessageBox.StandardButton.Yes:
            # Remove from hypothesis
            self.hypothesis.hypothesis_config.contradicting_evidence = [
                ev for ev in self.hypothesis.hypothesis_config.contradicting_evidence 
                if ev.id != evidence.id
            ]
            
            # Update display
            self.update_evidence_lists()
            self.hypothesis.update_evidence_summary()
            
            # Emit signal
            self.evidenceChanged.emit(self.hypothesis)



# ======================
# Port Item
# ======================

class NodePortItem(QGraphicsObject):
    """Port for connecting nodes"""
    
    portClicked = pyqtSignal(object)
    
    def __init__(self, parent=None, port_type="input", position="left", allowed_connections=None):
        super().__init__(parent)
        self.setAcceptHoverEvents(True)
        self.port_type = port_type  # "input" or "output"
        self.position = position    # "left", "right", "top", "bottom"
        self.is_hovered = False
        self.allowed_connections = allowed_connections or []  # Allowed connection types
        
        # Port dimensions
        self.port_width = 32
        self.port_height = 24
        self.port_radius = 8
        self.plus_size = 12
        self.stroke_width = 3
        self.connected_links = set()
        
        # Make port selectable and focusable
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | 
                     QGraphicsItem.GraphicsItemFlag.ItemIsFocusable)
        
        # Accept mouse events
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        
        # Set z-value to be above nodes but below text
        self.setZValue(2)
        
        # Color configuration
        self.default_color = QColor(158, 158, 158)  # Material gray
        self.hover_color = QColor(189, 189, 189)    # Lighter gray
        self.connected_color = QColor(139, 195, 74) # Material light green
        
        # Initialize brush and pen
        self._brush = QBrush(self.default_color)
        self._pen = QPen(QColor(180, 180, 180), 1)
        
        # Scaling property for hover animation
        self._scale = 1.0
    
    # Define scale property for animation
    def get_scale(self):
        return self._scale
        
    def set_scale(self, value):
        self._scale = value
        self.update()
        
    scale = Property(float, get_scale, set_scale)
    
    # Brush and pen handlers
    def setBrush(self, brush):
        self._brush = brush
        self.update()
    
    def brush(self):
        return self._brush
    
    def setPen(self, pen):
        self._pen = pen
        self.update()
    
    def pen(self):
        return self._pen
    
    def boundingRect(self):
        return QRectF(-self.port_width/2 - 5, -self.port_height/2 - 5,
                     self.port_width + 10, self.port_height + 10)
    
    def paint(self, painter, option, widget):
        """Paint the port as a rounded rectangle with a plus/minus sign"""
        painter.setBrush(self.brush())
        painter.setPen(self.pen())
        
        # Apply scaling transformation
        painter.save()
        painter.scale(self._scale, self._scale)
        
        # Draw rounded rectangle background
        rect = QRectF(-self.port_width/2, -self.port_height/2, self.port_width, self.port_height)
        
        # Customize color based on allowed connections
        if self.allowed_connections:
            # Use a specific color based on node type
            if "objective" in self.allowed_connections:
                base_color = QColor(NODE_CONFIGS["objective"].color)
            else:
                base_color = QColor(NODE_CONFIGS["hypothesis"].color)
                
            if not self.is_hovered and not self.connected_links:
                base_color.setAlpha(120)
            painter.setBrush(QBrush(base_color))
        
        painter.drawRoundedRect(rect, self.port_radius, self.port_radius)
        
        # Draw plus or minus sign in white
        plus_pen = QPen(QColor(255, 255, 255))
        plus_pen.setWidth(self.stroke_width)
        plus_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(plus_pen)
        
        # Draw horizontal line
        half_size = self.plus_size / 2
        painter.drawLine(QLineF(-half_size, 0, half_size, 0))
        
        # Draw vertical line for output/plus sign
        if self.port_type == "output" or self.position in ["right", "bottom"]:
            painter.drawLine(QLineF(0, -half_size, 0, half_size))
        
        painter.restore()
    
    def hoverEnterEvent(self, event):
        self.is_hovered = True
        
        # Create animation if needed
        if not hasattr(self, 'hover_animation'):
            self.hover_animation = QPropertyAnimation(self, b"scale")
            self.hover_animation.setDuration(150)
        
        self.hover_animation.setStartValue(1.0)
        self.hover_animation.setEndValue(1.4)
        self.hover_animation.start()
        
        # Change color
        self.setBrush(QBrush(self.hover_color))
        self.setGraphicsEffect(None)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update()
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        self.is_hovered = False
        
        if hasattr(self, 'hover_animation'):
            self.hover_animation.setStartValue(1.4)
            self.hover_animation.setEndValue(1.0)
            self.hover_animation.start()
        
        self.setBrush(QBrush(self.default_color))
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()
        super().hoverLeaveEvent(event)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.portClicked.emit(self)
            event.accept()
        else:
            super().mousePressEvent(event)
    
    def get_scene_position(self):
        """Get the center position of the port in scene coordinates"""
        return self.mapToScene(QPointF(0, 0))
    
    def add_connected_link(self, link):
        """Track links connected to this port"""
        self.connected_links.add(link)
        self.setBrush(QBrush(self.connected_color))
    
    def remove_connected_link(self, link):
        """Remove link from tracked connections"""
        if link in self.connected_links:
            self.connected_links.remove(link)
            
        if not self.connected_links:
            self.setBrush(QBrush(self.default_color))

# ======================
# Node Connection
# ======================

class NodeConnection(QGraphicsPathItem):
    """Connection between nodes"""
    
    def __init__(self, start_port, end_port, connection_type=ConnectionType.CONTRIBUTES_TO):
        super().__init__()
        self.start_port = start_port
        self.end_port = end_port
        self.connection_type = ConnectionType.CONTRIBUTES_TO  # Always use CONTRIBUTES_TO
        self.arrow_size = 15
        self.base_ribbon_width = 20  # Base width for links
        self.ribbon_width = self.base_ribbon_width  # Actual width to be calculated based on evidence
        
        # Set flags
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemIsFocusable
        )
        
        # Store parent nodes
        self.start_node = self.start_port.parentItem()
        self.end_node = self.end_port.parentItem()
        
        # Configure appearance based on hypothesis node
        self.update_appearance()
        
        # Hit detection pen (wider than visible pen)
        self.setPen(QPen(QColor(0, 0, 0, 0), self.ribbon_width + 5))
        
        # Set z-value
        self.setZValue(-1)
        
        # Interaction settings
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton)
        
        # Connect to position change signals
        if self.start_node:
            self.start_node.xChanged.connect(self.update_path)
            self.start_node.yChanged.connect(self.update_path)
        if self.end_node:
            self.end_node.xChanged.connect(self.update_path)
            self.end_node.yChanged.connect(self.update_path)
        
        # Connect to ports
        self.connect_to_ports()
        
        # Update path
        self.update_path()
    
    def __del__(self):
        """Safely disconnect signals when deleted"""
        if hasattr(self, 'start_node') and self.start_node:
            try:
                self.start_node.xChanged.disconnect(self.update_path)
                self.start_node.yChanged.disconnect(self.update_path)
            except:
                pass
        if hasattr(self, 'end_node') and self.end_node:
            try:
                self.end_node.xChanged.disconnect(self.update_path)
                self.end_node.yChanged.disconnect(self.update_path)
            except:
                pass
    
    def connect_to_ports(self):
        """Register this connection with the connected ports"""
        if hasattr(self.start_port, 'add_connected_link'):
            self.start_port.add_connected_link(self)
        
        if hasattr(self.end_port, 'add_connected_link'):
            self.end_port.add_connected_link(self)

    def update_path(self):
        """Update the connection path"""
        if not self.start_port or not self.end_port:
            return
        try:
            start_pos = self.start_port.get_scene_position()
            end_pos = self.end_port.get_scene_position()
            path = QPainterPath(start_pos)

            dx = end_pos.x() - start_pos.x()
            dy = end_pos.y() - start_pos.y()
            line_length = math.hypot(dx, dy)
            extension = min(line_length * 0.4, 100)

            # Adjust control points based on port positions
            if self.start_port.position == "bottom":
                ctrl_point1 = QPointF(start_pos.x(), start_pos.y() + extension)
            elif self.start_port.position == "top":
                ctrl_point1 = QPointF(start_pos.x(), start_pos.y() - extension)
            elif self.start_port.position == "right":
                ctrl_point1 = QPointF(start_pos.x() + extension, start_pos.y())
            else:  # "left"
                ctrl_point1 = QPointF(start_pos.x() - extension, start_pos.y())

            if self.end_port.position == "bottom":
                ctrl_point2 = QPointF(end_pos.x(), end_pos.y() + extension)
            elif self.end_port.position == "top":
                ctrl_point2 = QPointF(end_pos.x(), end_pos.y() - extension)
            elif self.end_port.position == "right":
                ctrl_point2 = QPointF(end_pos.x() + extension, end_pos.y())
            else:  # "left"
                ctrl_point2 = QPointF(end_pos.x() - extension, end_pos.y())

            path.cubicTo(ctrl_point1, ctrl_point2, end_pos)
            self.setPath(path)
            self.update()
        except (RuntimeError, ReferenceError):
            pass
    
    def paint(self, painter, option, widget):
        """Paint the connection with proper styling"""
        old_pen = self.pen()
        
        painter.setPen(self.visible_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setOpacity(0.8)
        painter.drawPath(self.path())
        
        # Highlight if selected
        if self.isSelected():
            highlight_pen = QPen(QColor(255, 165, 0, 200), self.ribbon_width + 4)
            highlight_pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(highlight_pen)
            painter.drawPath(self.path())
        
        self.setPen(old_pen)
    
    def hoverEnterEvent(self, event):
        """Highlight connection on hover"""
        self.visible_pen.setWidth(self.ribbon_width + 5)
        self.visible_pen.setColor(QColor(self.visible_pen.color().red(), 
                                     self.visible_pen.color().green(),
                                     self.visible_pen.color().blue(), 
                                     220))
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Return to normal appearance"""
        self.visible_pen.setWidth(self.ribbon_width)
        self.visible_pen.setColor(QColor(self.visible_pen.color().red(), 
                                     self.visible_pen.color().green(),
                                     self.visible_pen.color().blue(), 
                                     150))
        self.update()
        super().hoverLeaveEvent(event)
    
    def shape(self):
        """Use a stroked path for better hit detection"""
        stroker = QPainterPathStroker()
        stroker.setWidth(self.ribbon_width)
        return stroker.createStroke(self.path())
    
    def contextMenuEvent(self, event):
        """Show context menu"""
        menu = QMenu()
        
        # Just add delete option, since we only have one connection type now
        delete_action = menu.addAction("Delete Connection")
        
        action = menu.exec(event.screenPos())
        
        if action and action.text() == "Delete Connection":
            self.delete_connection()
    
    def update_appearance(self):
        """Update connection appearance based on connected hypothesis"""
        # Find the hypothesis node
        hypothesis_node = None
        if hasattr(self.start_node, 'node_type') and self.start_node.node_type == "hypothesis":
            hypothesis_node = self.start_node
        elif hasattr(self.end_node, 'node_type') and self.end_node.node_type == "hypothesis":
            hypothesis_node = self.end_node
        
        # Define state colors (same as in HypothesisNode.paint)
        state_colors = {
            HypothesisState.PROPOSED: QColor(180, 180, 180),      # Gray
            HypothesisState.TESTING: QColor(255, 193, 7),         # Amber
            HypothesisState.UNTESTED: QColor(108, 117, 125),      # Dark Gray
            HypothesisState.VALIDATED: QColor(40, 167, 69),       # Green
            HypothesisState.REJECTED: QColor(220, 53, 69),        # Red
            HypothesisState.INCONCLUSIVE: QColor(253, 126, 20),   # Orange
            HypothesisState.MODIFIED: QColor(0, 123, 255),        # Blue
        }
        
        # Get color and adjust width based on evidence count
        if hypothesis_node and hasattr(hypothesis_node, 'hypothesis_config'):
            # Get state color
            state = hypothesis_node.hypothesis_config.state
            color = state_colors.get(state, QColor(180, 180, 180))
            
            # Set slight transparency
            color.setAlpha(180)
            
            # Calculate total evidence to adjust width
            supporting_count = 0
            contradicting_count = 0
            
            if hasattr(hypothesis_node, 'get_total_supporting_evidence_count'):
                supporting_count = hypothesis_node.get_total_supporting_evidence_count()
                
            if hasattr(hypothesis_node, 'get_total_contradicting_evidence_count'):
                contradicting_count = hypothesis_node.get_total_contradicting_evidence_count()
                
            total_evidence = supporting_count + contradicting_count
            
            # Adjust width by evidence (subtle change)
            # Min width is base_ribbon_width, max width is base_ribbon_width * 1.5
            if total_evidence > 0:
                width_factor = min(1.0 + (total_evidence / 10.0), 1.5)  # Cap at 50% increase
                self.ribbon_width = int(self.base_ribbon_width * width_factor)
            else:
                self.ribbon_width = self.base_ribbon_width
        else:
            # Default color if no hypothesis
            color = QColor(76, 175, 80, 180)  # Green
        
        # Create pen with calculated color and width
        self.visible_pen = QPen(color, self.ribbon_width)
        self.visible_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.visible_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        
        # Update hit detection pen
        self.setPen(QPen(QColor(0, 0, 0, 0), self.ribbon_width + 5))
        
        self.update()
    
    def delete_connection(self):
        """Remove this connection"""
        if self.start_port:
            self.start_port.remove_connected_link(self)
        
        if self.end_port:
            self.end_port.remove_connected_link(self)
        
        if self.scene():
            self.scene().removeItem(self)
    
    def update_theme(self, theme):
        """Update appearance based on theme"""
        # Just call update_appearance as the colors are now determined by the hypothesis
        self.update_appearance()

# ======================
# Base Node
# ======================

class BaseNode(QGraphicsObject):
    """Base class for all nodes"""
    
    # Signals
    nodeActivated = pyqtSignal(object)
    nodeSelected = pyqtSignal(object)
    xChanged = pyqtSignal()
    yChanged = pyqtSignal()
    portClicked = pyqtSignal(object)
    
    def __init__(self, node_type, config, x=0, y=0):
        super().__init__()
        self.node_type = node_type
        self.config = config
        self.is_active = False
        self.is_hovered = False
        self.is_collapsed = False
        
        # Common data
        self.detail_items = []
        
        # Node data
        self.node_data = {
            'id': str(id(self)),
            'type': node_type
        }
        
        # Size
        self.width = config.width
        self.height = config.height
        self.current_width = self.width
        self.current_height = self.height
        self.border_padding = 2
        
        # Flags
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        
        # Position
        self.setPos(x, y)
        
        # Ports
        self.input_ports = []
        self.output_ports = []
        
        # Paths
        self._path = QPainterPath()
        self._border_path = QPainterPath()
        self._glow_path = QPainterPath()
        
        from plan.research_goals import NODE_TYPE_ICONS
        # Icon
        self.icon = None
        icon_name = NODE_TYPE_ICONS.get(node_type)
        if icon_name:
            self.icon = load_bootstrap_icon(icon_name, color="#FFFFFF", size=32)
        
        # Setup appearance
        self.setup_appearance()
        
        # Setup text
        self.setup_text()
        
        # Setup ports
        self.setup_ports()
        
        # Theme
        self.current_theme = "light"
    
    def setup_appearance(self):
        """Setup node appearance"""
        corner_radius = 15
        
        # Main node path
        self._path = QPainterPath()
        self._path.addRoundedRect(0, 0, self.width, self.height, corner_radius, corner_radius)
        
        # Border path
        self._border_path = QPainterPath()
        self._border_path.addRoundedRect(-6, -6, self.width + 12, self.height + 12, corner_radius + 2, corner_radius + 2)
        
        # Glow path
        self._glow_path = QPainterPath()
        self._glow_path.addRoundedRect(-8, -8, self.width + 16, self.height + 16, corner_radius + 4, corner_radius + 4)
        
        self.update()

    def setup_ports(self):
        """Setup connection ports"""
        # To be implemented by subclasses
        pass
        
    def setup_text(self):
        """Setup text display"""
        # To be implemented by subclasses
        pass
    
    def boundingRect(self):
        """Return bounding rectangle"""
        return self._path.boundingRect().adjusted(-10, -10, 10, 10)
    
    def paint(self, painter, option, widget):
        """Paint the node"""
        # Get base color
        color = QColor(self.config.color)
        
        # Border color
        border_color = QColor(
            int(color.red() * 0.7),
            int(color.green() * 0.7),
            int(color.blue() * 0.7),
            200
        )
        
        # Selection and hover states
        if self.isSelected():
            border_color = QColor(255, 165, 0, 230)  # Orange for selection
            painter.setPen(QPen(border_color, 4))
        elif self.is_hovered:
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
        
        # Active state glow
        if self.is_active:
            glow_color = QColor(255, 200, 50, 100)
            glow_pen = QPen(glow_color, 8)
            painter.setPen(glow_pen)
            painter.drawPath(self._glow_path)
        
        # Background gradient
        gradient = QLinearGradient(0, 0, 0, self.height)
        
        # Calculate gradient colors
        pastel_color = QColor(
            int((color.red() * 0.8 + 255 * 0.2)),
            int((color.green() * 0.8 + 255 * 0.2)),
            int((color.blue() * 0.8 + 255 * 0.2))
        )
        
        lighter_color = QColor(
            min(255, int(pastel_color.red() * 1.1)),
            min(255, int(pastel_color.green() * 1.1)),
            min(255, int(pastel_color.blue() * 1.1))
        )
        
        darker_color = QColor(
            max(0, int(pastel_color.red() * 0.9)),
            max(0, int(pastel_color.green() * 0.9)),
            max(0, int(pastel_color.blue() * 0.9))
        )
        
        # Set gradient colors
        gradient.setColorAt(0, lighter_color)
        gradient.setColorAt(1, darker_color)
        
        # Modify colors for hover/selection
        if self.is_hovered and not self.isSelected():
            gradient.setColorAt(0, QColor(
                min(255, int(lighter_color.red() * 1.05)),
                min(255, int(lighter_color.green() * 1.05)),
                min(255, int(lighter_color.blue() * 1.05))
            ))
        
        if self.isSelected():
            gradient.setColorAt(0, QColor(
                min(255, int(lighter_color.red() * 0.95 + 255 * 0.05)),
                int(lighter_color.green() * 0.95),
                int(lighter_color.blue() * 0.95)
            ))
        
        # Draw main shape
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(self._path)
        
        # Draw icon if available
        if hasattr(self, 'icon') and self.icon:
            icon_size = 36
            icon_x = 10
            icon_y = (self.height - icon_size) / 2
            
            # Icon backdrop
            backdrop_color = QColor(
                int(color.red() * 0.7),
                int(color.green() * 0.7),
                int(color.blue() * 0.7),
                100
            )
            painter.setBrush(QBrush(backdrop_color))
            
            icon_rect = QRectF(
                icon_x - 2,
                icon_y - 2, 
                icon_size + 4,
                icon_size + 4
            )
            painter.drawEllipse(icon_rect)
            
            # Draw icon
            self.icon.paint(painter, int(icon_x), int(icon_y), icon_size, icon_size)
    
    def hoverEnterEvent(self, event):
        """Handle hover enter"""
        self.is_hovered = True
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update()
        super().hoverEnterEvent(event)
        
    def hoverLeaveEvent(self, event):
        """Handle hover leave"""
        self.is_hovered = False
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()
        super().hoverLeaveEvent(event)
    
    def mouseDoubleClickEvent(self, event):
        """Handle double-click activation"""
        self.nodeActivated.emit(self)
        super().mouseDoubleClickEvent(event)
    
    def mousePressEvent(self, event):
        """Handle mouse press"""
        super().mousePressEvent(event)
        self.nodeSelected.emit(self)
    
    def mouseMoveEvent(self, event):
        """Handle node movement"""
        old_pos = self.pos()
        super().mouseMoveEvent(event)
        
        # Emit signals
        self.xChanged.emit()
        self.yChanged.emit()
        
        # Update connections
        self.update_connected_links()
        
        # Clear auto-arranged state if moved significantly
        if self.scene() and hasattr(self.scene(), 'recently_auto_arranged'):
            if self.scene().recently_auto_arranged:
                if (self.pos() - old_pos).manhattanLength() > 5:
                    self.scene().recently_auto_arranged = False
    
    def on_port_clicked(self, port):
        """Handle port click"""
        self.portClicked.emit(port)

    def update_connected_links(self):
        """Update all connected links"""
        for port in self.input_ports:
            for link in port.connected_links:
                if link.scene():
                    link.update_path()
        
        for port in self.output_ports:
            for link in port.connected_links:
                if link.scene():
                    link.update_path()
    
    def add_detail_text(self, text, y_position=30):
        """Add detail text to the node"""
        detail = QGraphicsTextItem(self)
        detail.setPlainText(text)
        detail.setDefaultTextColor(QColor(80, 80, 80))
        
        # Set font
        detail_font = QFont("Arial", 9)
        detail.setFont(detail_font)
        
        # Set width for wrapping
        icon_width = 42
        detail.setTextWidth(self.width - icon_width - 10)
        
        # Position
        detail.setPos(icon_width, y_position)
        
        # Add to list
        self.detail_items.append(detail)
        
        return detail
    
    def toggle_collapse(self):
        """Toggle collapsed state"""
        self.is_collapsed = not self.is_collapsed
        self.update()
        
        if self.scene():
            # Signal the scene to update child visibility
            if hasattr(self.scene(), 'on_node_collapse_toggled'):
                self.scene().on_node_collapse_toggled(self)
    
    def delete_node(self):
        """Delete this node and its connections"""
        # Get all connected links
        connected_links = set()
        
        for port in self.input_ports:
            connected_links.update(port.connected_links)
        
        for port in self.output_ports:
            connected_links.update(port.connected_links)
        
        # Delete all connected links
        for link in list(connected_links):
            link.delete_connection()
        
        # Remove from scene
        if self.scene():
            self.scene().removeItem(self)

# ======================
# Objective Node
# ======================

class ObjectiveNode(BaseNode):
    """Node representing a research objective"""
    
    def __init__(self, config, x=0, y=0):
        # Create node config
        node_config = NODE_CONFIGS["objective"]
        super().__init__("objective", node_config, x, y)
        self.objective_config = config
        
        # Add objective data
        self.node_data.update({
            'id': config.id,
            'text': config.text,
            'type': config.type.value,
            'progress': config.progress,
            'auto_generate': config.auto_generate
        })
        
        # Setup objective text
        self.setup_objective_text()
        
        # Child hypotheses
        self.child_hypotheses = []
        
    def setup_ports(self):
        """Setup connection ports for objective node"""
        border_offset = 6
        
        # Port on left for objective connections
        # Only goal objectives should connect to research question objectives
        left_port = NodePortItem(
            self,
            "input",  # Can function as both input/output depending on objective type
            "left",
            allowed_connections=["objective"]
        )
        left_port.setPos(-border_offset, self.height / 2)
        left_port.portClicked.connect(self.on_port_clicked)
        self.input_ports.append(left_port)
        
        # Port on right for hypothesis connections
        right_port = NodePortItem(
            self,
            "output",
            "right",
            allowed_connections=["hypothesis"]
        )
        right_port.setPos(self.width + border_offset, self.height / 2)
        right_port.portClicked.connect(self.on_port_clicked)
        self.output_ports.append(right_port)
        
    def setup_objective_text(self):
        """Setup text for objective node"""
        # Set title text
        self.title_item = QGraphicsTextItem(self)
        self.title_item.setPlainText(f"{self.objective_config.type.value.replace('_', ' ').title()}")
        self.title_item.setDefaultTextColor(QColor(50, 50, 50))
        
        # Set font
        title_font = QFont("Arial", 11)
        title_font.setBold(True)
        self.title_item.setFont(title_font)
        
        # Position the title - account for icon
        icon_width = 42
        self.title_item.setPos(icon_width, 10)
        
        # Add objective text
        objective_text = self.objective_config.text
        if len(objective_text) > 80:
            objective_text = objective_text[:77] + "..."
        
        # Add text as detail
        text_item = self.add_detail_text(objective_text, 35)
        
        # Add progress indicator
        progress_text = f"Progress: {int(self.objective_config.progress * 100)}%"
        progress_item = self.add_detail_text(progress_text, 80)
        progress_item.progress_marker = True
        
        # Add type indicator if this is a sub-objective
        if self.objective_config.parent_id:
            type_text = "Sub-objective"
            self.add_detail_text(type_text, 100)

# ======================
# Hypothesis Node
# ======================

class HypothesisNode(BaseNode):
    """Node representing a research hypothesis"""
    
    def __init__(self, config, x=0, y=0):
        # Create node config
        node_config = NODE_CONFIGS["hypothesis"]
        # We'll set the color based on state in paint method
        super().__init__("hypothesis", node_config, x, y)
        self.hypothesis_config = config
        
        # Add hypothesis data
        self.node_data.update({
            'id': config.id,
            'text': config.text,
            'state': config.state.value,
            'confidence': config.confidence
        })
        
        # Setup hypothesis text
        self.setup_hypothesis_text()
        
        # Parent objective
        self.parent_objective = None
        
    def setup_ports(self):
        """Setup connection ports for hypothesis node"""
        border_offset = 6
        
        # One port on left for parent objective (input only)
        left_port = NodePortItem(
            self,
            "input",
            "left",
            allowed_connections=["objective"]
        )
        left_port.setPos(-border_offset, self.height / 2)
        left_port.portClicked.connect(self.on_port_clicked)
        self.input_ports.append(left_port)
        
        # One port on right for relationships with other hypotheses
        right_port = NodePortItem(
            self,
            "input",  # Can function as both input/output
            "right",
            allowed_connections=["hypothesis"]
        )
        right_port.setPos(self.width + border_offset, self.height / 2)
        right_port.portClicked.connect(self.on_port_clicked)
        self.input_ports.append(right_port)
    
    def setup_hypothesis_text(self):
        """Setup evidence count for hypothesis node"""
        # Get combined counts from all evidence sources
        supporting_count = self.get_total_supporting_evidence_count()
        contradicting_count = self.get_total_contradicting_evidence_count()
        
        # Use the exact format from HypothesisCard: "3↑ 1↓"
        evidence_text = f"{supporting_count}↑ {contradicting_count}↓"
        
        # Create evidence text item
        self.evidence_item = QGraphicsTextItem(self)
        self.evidence_item.setPlainText(evidence_text)
        self.evidence_item.setDefaultTextColor(QColor(255, 255, 255))  # White text for contrast
        
        # Set font
        evidence_font = QFont("Arial", 16)  # Increased from 12 to 16
        evidence_font.setBold(True)
        self.evidence_item.setFont(evidence_font)
        
        # Center the text
        self.evidence_item.setTextWidth(self.width)
        document = self.evidence_item.document()
        document.setDefaultTextOption(self.evidence_item.document().defaultTextOption())
        document.setDefaultTextOption(QTextOption(Qt.AlignmentFlag.AlignCenter))
        
        # Position in center of node
        text_width = self.evidence_item.boundingRect().width()
        text_height = self.evidence_item.boundingRect().height()
        x_pos = (self.width - text_width) / 2
        y_pos = (self.height - text_height) / 2
        self.evidence_item.setPos(x_pos, y_pos)
        
        # Track this as an evidence marker
        self.evidence_item.evidence_marker = True
        self.detail_items.append(self.evidence_item)
    
    def get_total_supporting_evidence_count(self):
        """Get total supporting evidence count from all sources"""
        # Start with explicit supporting evidence
        count = len(self.hypothesis_config.supporting_evidence)
        print(f"Supporting evidence base count: {count}")
        
        # Add literature evidence if available
        if hasattr(self.hypothesis_config, 'literature_evidence') and self.hypothesis_config.literature_evidence:
            lit_evidence = self.hypothesis_config.literature_evidence
            if isinstance(lit_evidence, dict) and 'supporting' in lit_evidence:
                lit_count = lit_evidence['supporting']
                count += lit_count
                print(f"  Added {lit_count} supporting from literature_evidence")
        
        # Add model evidence if available and supports hypothesis
        if hasattr(self.hypothesis_config, 'test_results') and self.hypothesis_config.test_results:
            test_results = self.hypothesis_config.test_results
            if isinstance(test_results, dict):
                p_value = test_results.get('p_value')
                alpha = getattr(self.hypothesis_config, 'alpha_level', 0.05)
                # If p-value is significant, add as supporting evidence
                if p_value is not None and p_value < alpha:
                    count += 1
                    print(f"  Added 1 supporting from test_results (p={p_value} < alpha={alpha})")
        
        print(f"Final supporting count: {count}")
        return count
    
    def get_total_contradicting_evidence_count(self):
        """Get total contradicting evidence count from all sources"""
        # Start with explicit contradicting evidence
        count = len(self.hypothesis_config.contradicting_evidence)
        print(f"Contradicting evidence base count: {count}")
        
        # Add literature evidence if available
        if hasattr(self.hypothesis_config, 'literature_evidence') and self.hypothesis_config.literature_evidence:
            lit_evidence = self.hypothesis_config.literature_evidence
            if isinstance(lit_evidence, dict) and 'refuting' in lit_evidence:
                lit_count = lit_evidence['refuting']
                count += lit_count
                print(f"  Added {lit_count} contradicting from literature_evidence")
        
        # Add model evidence if available and contradicts hypothesis
        if hasattr(self.hypothesis_config, 'test_results') and self.hypothesis_config.test_results:
            test_results = self.hypothesis_config.test_results
            if isinstance(test_results, dict):
                p_value = test_results.get('p_value')
                alpha = getattr(self.hypothesis_config, 'alpha_level', 0.05)
                # If p-value is not significant, add as contradicting evidence
                if p_value is not None and p_value >= alpha:
                    count += 1
                    print(f"  Added 1 contradicting from test_results (p={p_value} >= alpha={alpha})")
        
        print(f"Final contradicting count: {count}")
        return count
    
    def update_evidence_summary(self):
        """Update the evidence summary text"""
        # Get combined counts from all evidence sources
        supporting_count = self.get_total_supporting_evidence_count()
        contradicting_count = self.get_total_contradicting_evidence_count()
        
        # Use the exact format from HypothesisCard: "3↑ 1↓"
        evidence_text = f"{supporting_count}↑ {contradicting_count}↓"
        
        for item in self.detail_items:
            if hasattr(item, 'evidence_marker'):
                item.setPlainText(evidence_text)
                
                # Update font size to match setup_hypothesis_text
                evidence_font = QFont("Arial", 16)
                evidence_font.setBold(True)
                item.setFont(evidence_font)
                
                # Re-center the text
                text_width = item.boundingRect().width()
                text_height = item.boundingRect().height()
                x_pos = (self.width - text_width) / 2
                y_pos = (self.height - text_height) / 2
                item.setPos(x_pos, y_pos)
                
        self.update()
    
    def change_state(self, new_state):
        """Change hypothesis state"""
        # Update state
        self.hypothesis_config.state = new_state
        self.node_data['state'] = new_state.value
        
        # Update appearance 
        self.update()
    
    def paint(self, painter, option, widget):
        """Paint with entire node colored based on state"""
        # Define state colors
        state_colors = {
            HypothesisState.PROPOSED: QColor(180, 180, 180),      # Gray
            HypothesisState.TESTING: QColor(255, 193, 7),         # Amber
            HypothesisState.UNTESTED: QColor(108, 117, 125),      # Dark Gray
            HypothesisState.VALIDATED: QColor(40, 167, 69),       # Green
            HypothesisState.REJECTED: QColor(220, 53, 69),        # Red
            HypothesisState.INCONCLUSIVE: QColor(253, 126, 20),   # Orange
            HypothesisState.MODIFIED: QColor(0, 123, 255),        # Blue
        }
        
        # Get color based on hypothesis state
        color = state_colors.get(self.hypothesis_config.state, QColor(180, 180, 180))
        
        # Border color (darker variant of the state color)
        border_color = QColor(
            int(color.red() * 0.7),
            int(color.green() * 0.7),
            int(color.blue() * 0.7),
            200
        )
        
        # Selection and hover states
        if self.isSelected():
            painter.setPen(QPen(QColor(255, 165, 0, 230), 4))  # Orange for selection
        elif self.is_hovered:
            painter.setPen(QPen(border_color, 3))
        else:
            painter.setPen(QPen(border_color, 3))
        
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(self._border_path)
        
        # Active state glow
        if self.is_active:
            glow_color = QColor(255, 200, 50, 100)
            glow_pen = QPen(glow_color, 8)
            painter.setPen(glow_pen)
            painter.drawPath(self._glow_path)
        
        # Background gradient based on state color
        gradient = QLinearGradient(0, 0, 0, self.height)
        
        # Calculate gradient colors from state color
        pastel_color = QColor(
            int((color.red() * 0.8 + 255 * 0.2)),
            int((color.green() * 0.8 + 255 * 0.2)),
            int((color.blue() * 0.8 + 255 * 0.2))
        )
        
        lighter_color = QColor(
            min(255, int(pastel_color.red() * 1.1)),
            min(255, int(pastel_color.green() * 1.1)),
            min(255, int(pastel_color.blue() * 1.1))
        )
        
        darker_color = QColor(
            max(0, int(pastel_color.red() * 0.9)),
            max(0, int(pastel_color.green() * 0.9)),
            max(0, int(pastel_color.blue() * 0.9))
        )
        
        # Set gradient colors
        gradient.setColorAt(0, lighter_color)
        gradient.setColorAt(1, darker_color)
        
        # Modify colors for hover/selection
        if self.is_hovered and not self.isSelected():
            gradient.setColorAt(0, QColor(
                min(255, int(lighter_color.red() * 1.05)),
                min(255, int(lighter_color.green() * 1.05)),
                min(255, int(lighter_color.blue() * 1.05))
            ))
        
        if self.isSelected():
            gradient.setColorAt(0, QColor(
                min(255, int(lighter_color.red() * 0.95 + 255 * 0.05)),
                int(lighter_color.green() * 0.95),
                int(lighter_color.blue() * 0.95)
            ))
        
        # Draw main shape
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(self._path)
        
        # Removed badge in bottom right corner

