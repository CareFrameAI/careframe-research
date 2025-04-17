from PyQt6.QtCore import Qt, QPointF, QRectF, QTimer, QLineF, Signal, QObject
from PyQt6.QtGui import QPen, QBrush, QColor, QFont, QPainter, QPixmap, QIcon, QTransform, QCursor
from PyQt6.QtWidgets import (
    QGraphicsScene, QGraphicsView, QGraphicsItem, QGraphicsLineItem, 
    QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsPixmapItem
)
from model_builder.config import WorkflowNode, EdgeItem, PortItem, NodeCategory

class ThemeManager:
    """Manages theme-specific colors for the workflow UI elements."""
    
    # Default colors for light theme
    LIGHT_THEME = {
        "background": QColor(245, 245, 250),
        "node_border": QColor(120, 120, 120),
        "node_fill": QColor(255, 255, 255),
        "node_selected": QColor(230, 242, 255),
        "port_border": QColor(100, 100, 100),
        "port_fill": QColor(220, 220, 220),
        "port_highlight": QColor(76, 175, 80),
        "port_highlight_fill": QColor(76, 175, 80, 40),
        "edge_normal": QColor(100, 100, 100),
        "edge_highlight": QColor(76, 175, 80),
        "edge_error": QColor(244, 67, 54),
        "tooltip_text": QColor(66, 66, 66),
        "tooltip_background": QColor(255, 255, 255, 240),
        "tooltip_border": QColor(200, 230, 201),
        "text_primary": QColor(33, 33, 33),
        "text_secondary": QColor(66, 66, 66)
    }
    
    # Default colors for dark theme
    DARK_THEME = {
        "background": QColor(40, 44, 52),
        "node_border": QColor(150, 150, 150),
        "node_fill": QColor(60, 63, 65),
        "node_selected": QColor(44, 62, 80),
        "port_border": QColor(180, 180, 180),
        "port_fill": QColor(100, 100, 100),
        "port_highlight": QColor(76, 175, 80),
        "port_highlight_fill": QColor(76, 175, 80, 60),
        "edge_normal": QColor(180, 180, 180),
        "edge_highlight": QColor(76, 175, 80),
        "edge_error": QColor(244, 67, 54),
        "tooltip_text": QColor(220, 220, 220),
        "tooltip_background": QColor(50, 50, 50, 240),
        "tooltip_border": QColor(76, 175, 80, 100),
        "text_primary": QColor(220, 220, 220),
        "text_secondary": QColor(180, 180, 180)
    }
    
    # Edge action colors for light theme
    LIGHT_EDGE_ACTIONS = {
        "default": QColor(120, 120, 140, 170),
        "active": QColor(76, 175, 80, 200),
        "completed": QColor(33, 150, 243, 200),
        "error": QColor(244, 67, 54, 200),
    }
    
    # Edge action colors for dark theme
    DARK_EDGE_ACTIONS = {
        "default": QColor(150, 150, 170, 200),
        "active": QColor(76, 175, 80, 220),
        "completed": QColor(33, 150, 243, 220),
        "error": QColor(244, 67, 54, 220),
    }
    
    @classmethod
    def get_color(cls, color_name, is_dark_theme=False):
        """Get a color based on the current theme."""
        theme = cls.DARK_THEME if is_dark_theme else cls.LIGHT_THEME
        return theme.get(color_name, QColor(0, 0, 0))  # Default to black if color not found
    
    @classmethod
    def get_edge_action_color(cls, action, is_dark_theme=False):
        """Get an edge action color based on the current theme."""
        theme = cls.DARK_EDGE_ACTIONS if is_dark_theme else cls.LIGHT_EDGE_ACTIONS
        return theme.get(action, theme["default"])

# Extend the WorkflowNode class to support theme-aware colors
def update_node_theme(node, is_dark_theme):
    """Update a WorkflowNode's colors based on the current theme."""
    # Update the current theme
    node.current_theme = "dark" if is_dark_theme else "light"
    
    # Update text colors
    if hasattr(node, 'title_item') and node.title_item:
        text_color = ThemeManager.get_color("text_primary", is_dark_theme)
        node.title_item.setDefaultTextColor(text_color)
    
    if hasattr(node, 'desc_item') and node.desc_item:
        text_color = ThemeManager.get_color("text_secondary", is_dark_theme)
        node.desc_item.setDefaultTextColor(text_color)
    
    # Update ports
    for port in node.input_ports + node.output_ports:
        update_port_theme(port, is_dark_theme)
    
    # Force a redraw
    node.update()

# Extend the EdgeItem class to support theme-aware colors
def update_edge_theme(edge, is_dark_theme):
    """Update an EdgeItem's colors based on the current theme."""
    action_color = ThemeManager.get_edge_action_color(edge.action, is_dark_theme)
    edge.setPen(QPen(action_color, 2, Qt.PenStyle.SolidLine))
    
    # Force a redraw
    edge.update()

# Extend the PortItem class to support theme-aware colors
def update_port_theme(port, is_dark_theme):
    """Update a PortItem's colors based on the current theme."""
    border_color = ThemeManager.get_color("port_border", is_dark_theme)
    fill_color = ThemeManager.get_color("port_fill", is_dark_theme)
    
    port.setPen(QPen(border_color, 1))
    port.setBrush(QBrush(fill_color))
    
    # Update port label color if it exists
    if hasattr(port, 'label_item') and port.label_item:
        text_color = ThemeManager.get_color("text_secondary", is_dark_theme)
        port.label_item.setDefaultTextColor(text_color)
    
    # Force a redraw
    port.update()

# Monkey patch the classes to add theme support
WorkflowNode.update_theme = update_node_theme
EdgeItem.update_theme = update_edge_theme
PortItem.update_theme = update_port_theme 