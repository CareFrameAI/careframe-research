"""
Helper module to load Bootstrap icons from SVG files.
This provides standard icon loading for the application.
"""

import os
from PyQt6.QtGui import QIcon, QPixmap, QPainter
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QApplication
import re

def load_bootstrap_icon(icon_name, color=None, size=None):
    """
    Load a Bootstrap icon from the SVG files.
    
    Args:
        icon_name: Name of the icon file (without extension)
        color: Optional color to apply to the icon (hex string)
        size: Optional size to render the icon at
        
    Returns:
        QIcon object
    """
    
    # Default icon loading method
    icon_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'icons',
        f"{icon_name}.svg"
    )
    
    # Check if the icon exists
    if not os.path.exists(icon_path):
        print(f"Warning: Icon not found: {icon_path}")
        return QIcon()

    color = color or "#A0A0A0"  # Medium gray with a slight lightness
    
    # Load SVG and modify it
    with open(icon_path, 'r') as f:
        svg_content = f.read()
    
    # Replace color
    # Check for currentColor, fill or stroke attributes
    if 'fill="currentColor"' in svg_content:
        svg_content = svg_content.replace('fill="currentColor"', f'fill="{color}"')
    if 'stroke="currentColor"' in svg_content:
        svg_content = svg_content.replace('stroke="currentColor"', f'stroke="{color}"')
    
    # Add fill to any path elements that don't have a fill attribute
    path_tags = re.findall(r'<path[^>]*>', svg_content)
    for path_tag in path_tags:
        if 'fill=' not in path_tag:
            svg_content = svg_content.replace(path_tag, path_tag.replace('<path', f'<path fill="{color}"', 1))
    
    # Add a fill to the root SVG element if not already present
    if '<svg' in svg_content and 'fill=' not in svg_content.split('>')[0]:
        svg_content = svg_content.replace('<svg', f'<svg fill="{color}"', 1)
    
    # Create a renderer for the SVG
    renderer = QSvgRenderer(bytes(svg_content, 'utf-8'))
    
    # Create appropriately sized pixmap
    icon_size = QSize(16, 16) if size is None else (QSize(size, size) if isinstance(size, int) else size)
    pixmap = QPixmap(icon_size)
    pixmap.fill(Qt.GlobalColor.transparent)
    
    # Paint the SVG onto the pixmap
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    
    # Return an icon created from the pixmap
    return QIcon(pixmap)
