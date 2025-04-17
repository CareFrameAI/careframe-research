from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, 
                           QTextEdit, QSplitter, QComboBox, QMessageBox, QGroupBox, 
                           QHBoxLayout, QTableWidget, QTableWidgetItem, QRadioButton,
                           QButtonGroup, QFormLayout)
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor
import logging
import json
import math
import uuid
from exchange.blockchain_ops import Blockchain
from datetime import datetime

class NetworkGraphView(QWidget):
    """
    A custom widget to display a network graph visualization of the evidence network.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.nodes = []  # List of dictionaries with position, size, label, color, data
        self.edges = []  # List of tuples (node_idx1, node_idx2, weight, color)
        self.selected_node = None
        self.setMinimumSize(600, 400)
        self.node_radius = 30
        self.highlight_color = QColor(255, 165, 0)  # Orange for highlighting
        self.node_labels = True  # Whether to display node labels
        
    def add_node(self, label, data=None, color=None):
        """Add a node to the graph"""
        if color is None:
            color = QColor(70, 130, 180)  # Steel blue default
        
        # Find a position for the new node
        if not self.nodes:
            pos = QPointF(self.width()/2, self.height()/2)
        else:
            # Position in a circle around the center
            angle = 2 * math.pi * len(self.nodes) / 10  # Adjust divisor for spacing
            radius = min(self.width(), self.height()) * 0.35
            center_x, center_y = self.width()/2, self.height()/2
            pos = QPointF(
                center_x + radius * math.cos(angle),
                center_y + radius * math.sin(angle)
            )
        
        self.nodes.append({
            'pos': pos,
            'label': label,
            'size': self.node_radius,
            'color': color,
            'data': data or {}
        })
        self.update()
        return len(self.nodes) - 1  # Return index of the new node
    
    def add_edge(self, node1_idx, node2_idx, weight=1.0, color=None):
        """Add an edge between two nodes"""
        if color is None:
            color = QColor(200, 200, 200)  # Light gray default
            
        if 0 <= node1_idx < len(self.nodes) and 0 <= node2_idx < len(self.nodes):
            self.edges.append((node1_idx, node2_idx, weight, color))
            self.update()
            
    def clear(self):
        """Clear all nodes and edges"""
        self.nodes = []
        self.edges = []
        self.selected_node = None
        self.update()
            
    def paintEvent(self, event):
        """Draw the network graph"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw edges first
        for edge in self.edges:
            node1_idx, node2_idx, weight, color = edge
            if node1_idx >= len(self.nodes) or node2_idx >= len(self.nodes):
                continue  # Skip if nodes don't exist
                
            pen = QPen(color)
            pen.setWidth(max(1, int(weight * 3)))  # Edge thickness based on weight
            painter.setPen(pen)
            
            node1 = self.nodes[node1_idx]
            node2 = self.nodes[node2_idx]
            painter.drawLine(node1['pos'], node2['pos'])
        
        # Draw nodes
        for i, node in enumerate(self.nodes):
            # Draw node circle
            if i == self.selected_node:
                painter.setBrush(QBrush(self.highlight_color))
                pen = QPen(Qt.GlobalColor.black)
                pen.setWidth(2)
                painter.setPen(pen)
            else:
                painter.setBrush(QBrush(node['color']))
                painter.setPen(QPen(Qt.GlobalColor.black))
                
            painter.drawEllipse(node['pos'], node['size'], node['size'])
            
            # Draw node label if enabled
            if self.node_labels:
                painter.setPen(QPen(Qt.GlobalColor.black))
                label_rect = painter.fontMetrics().boundingRect(node['label'])
                label_pos = QPointF(
                    node['pos'].x() - label_rect.width() / 2,
                    node['pos'].y() + node['size'] + label_rect.height()
                )
                painter.drawText(label_pos, node['label'])
    
    def mousePressEvent(self, event):
        """Handle node selection on click"""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            # Check if a node was clicked
            for i, node in enumerate(self.nodes):
                if math.dist([pos.x(), pos.y()], [node['pos'].x(), node['pos'].y()]) <= node['size']:
                    self.selected_node = i
                    self.update()
                    # Emit signal or call callback for node selection
                    if hasattr(self.parent(), 'on_node_selected'):
                        self.parent().on_node_selected(i, node['data'])
                    return
            # If click wasn't on a node, deselect
            self.selected_node = None
            self.update()
    
    def resizeEvent(self, event):
        """Adjust node positions when widget is resized"""
        if not self.nodes:
            return
            
        old_size = event.oldSize()
        if old_size.width() <= 0 or old_size.height() <= 0:
            return
            
        # Scale factor for each dimension
        scale_x = self.width() / old_size.width()
        scale_y = self.height() / old_size.height()
        
        # Adjust each node position
        for node in self.nodes:
            node['pos'] = QPointF(node['pos'].x() * scale_x, node['pos'].y() * scale_y)
        
        self.update()


class EvidenceNetworkView(QWidget):
    """
    A widget for visualizing and interacting with evidence networks from blockchain data.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.blockchain = None
        self.network_nodes = {}  # Maps evidence_id/study_id to node index
        
        # Create horizontal layout
        self.h_layout = QHBoxLayout()
        
        # Left side - Controls and options
        self.controls_panel = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_panel)
        
        # Network visualization options
        self.options_group = QGroupBox("Visualization Options")
        options_layout = QVBoxLayout()
        
        # Type of network to display
        self.network_type_group = QButtonGroup()
        self.evidence_radio = QRadioButton("Evidence Network")
        self.evidence_radio.setChecked(True)
        self.study_radio = QRadioButton("Study Network")
        self.claim_radio = QRadioButton("Claim Network")
        
        self.network_type_group.addButton(self.evidence_radio)
        self.network_type_group.addButton(self.study_radio)
        self.network_type_group.addButton(self.claim_radio)
        
        options_layout.addWidget(QLabel("Network Type:"))
        options_layout.addWidget(self.evidence_radio)
        options_layout.addWidget(self.study_radio)
        options_layout.addWidget(self.claim_radio)
        
        # Toggle for node labels
        self.labels_checkbox = QRadioButton("Show Node Labels")
        self.labels_checkbox.setChecked(True)
        options_layout.addWidget(self.labels_checkbox)
        
        # Refresh button
        self.refresh_button = QPushButton("Refresh Network")
        self.refresh_button.clicked.connect(self.refresh_network)
        options_layout.addWidget(self.refresh_button)
        
        # Debug button for creating sample data
        self.debug_button = QPushButton("Create Sample Network")
        self.debug_button.clicked.connect(self.create_sample_network)
        options_layout.addWidget(self.debug_button)
        
        self.options_group.setLayout(options_layout)
        
        # Node details section
        self.details_group = QGroupBox("Node Details")
        self.details_layout = QVBoxLayout()
        self.node_details = QTextEdit()
        self.node_details.setReadOnly(True)
        self.details_layout.addWidget(self.node_details)
        self.details_group.setLayout(self.details_layout)
        
        # Statistics section
        self.stats_group = QGroupBox("Network Statistics")
        stats_layout = QFormLayout()
        self.node_count_label = QLabel("0")
        self.edge_count_label = QLabel("0")
        self.density_label = QLabel("0")
        
        stats_layout.addRow("Node Count:", self.node_count_label)
        stats_layout.addRow("Edge Count:", self.edge_count_label)
        stats_layout.addRow("Network Density:", self.density_label)
        
        self.stats_group.setLayout(stats_layout)
        
        # Add components to left panel
        self.controls_layout.addWidget(self.options_group)
        self.controls_layout.addWidget(self.stats_group)
        self.controls_layout.addWidget(self.details_group)
        self.controls_layout.addStretch()
        
        # Right side - Network visualization
        self.network_view = NetworkGraphView()
        
        # Add both panels to horizontal layout
        self.h_layout.addWidget(self.controls_panel, 1)  # Proportion 1
        self.h_layout.addWidget(self.network_view, 3)    # Proportion 3
        
        # Add horizontal layout to main layout
        self.layout.addLayout(self.h_layout)
        
        # Connect signals
        self.evidence_radio.toggled.connect(self.refresh_network)
        self.study_radio.toggled.connect(self.refresh_network)
        self.claim_radio.toggled.connect(self.refresh_network)
        self.labels_checkbox.toggled.connect(self.toggle_labels)

    def load_blockchain(self, blockchain):
        """Load blockchain data and update the network visualization"""
        self.blockchain = blockchain
        self.refresh_network()
    
    def create_sample_network(self):
        """
        Create a sample network with realistic data for visualization and testing
        without needing actual blockchain data
        """
        # Clear existing network
        self.network_view.clear()
        self.network_nodes = {}
        
        if self.evidence_radio.isChecked():
            self.create_sample_evidence_network()
        elif self.study_radio.isChecked():
            self.create_sample_study_network()
        elif self.claim_radio.isChecked():
            self.create_sample_claim_network()
            
        # Update statistics
        self.update_network_stats()
        
        # Show success message
        QMessageBox.information(self, "Sample Created", 
                               "Sample network created successfully. You can interact with the nodes.")
    
    def create_sample_evidence_network(self):
        """Create a sample evidence network with realistic data"""
        # Generate sample evidence nodes
        evidence_data = [
            {
                'id': f"evidence_{uuid.uuid4().hex[:8]}",
                'title': "Effect of Exercise on Blood Pressure",
                'description': "Randomized controlled trial on exercise impact on hypertension",
                'color': QColor(70, 130, 180)  # Steel blue
            },
            {
                'id': f"evidence_{uuid.uuid4().hex[:8]}",
                'title': "Mediterranean Diet and Cardiovascular Health",
                'description': "Longitudinal study on dietary patterns and heart disease prevention",
                'color': QColor(70, 130, 180)
            },
            {
                'id': f"evidence_{uuid.uuid4().hex[:8]}",
                'title': "Sleep Quality and Cognitive Performance",
                'description': "Cross-sectional study on sleep patterns and cognitive function",
                'color': QColor(70, 130, 180)
            },
            {
                'id': f"evidence_{uuid.uuid4().hex[:8]}",
                'title': "Vitamin D Supplementation and Immune Function",
                'description': "Double-blind placebo-controlled trial on vitamin D effects",
                'color': QColor(70, 130, 180)
            },
            {
                'id': f"evidence_{uuid.uuid4().hex[:8]}",
                'title': "Stress Reduction Techniques and Cortisol Levels",
                'description': "Randomized trial comparing meditation and biofeedback",
                'color': QColor(70, 130, 180)
            },
            {
                'id': f"evidence_{uuid.uuid4().hex[:8]}",
                'title': "Physical Activity and Telomere Length",
                'description': "Observational study on exercise and cellular aging markers",
                'color': QColor(70, 130, 180)
            }
        ]
        
        # Add nodes to the graph
        for i, evidence in enumerate(evidence_data):
            node_idx = self.network_view.add_node(
                f"E{i+1}",  # Short label
                data={
                    'type': 'evidence',
                    'evidence_id': evidence['id'],
                    'title': evidence['title'],
                    'block_index': i+1,
                    'full_data': {
                        'evidence_id': evidence['id'],
                        'study_id': f"study_{i+1}",
                        'title': evidence['title'],
                        'description': evidence['description'],
                        'intervention_text': f"Intervention details for {evidence['title']}",
                        'evidence_claims': [
                            {'claim_id': f"claim_{i}_{j}", 'claim_text': f"Claim {j} for {evidence['title']}"} 
                            for j in range(1, 4)
                        ],
                        'study_components': [
                            {'component_id': f"comp_{i}_{j}", 'component_type': t} 
                            for j, t in enumerate(['study_protocol', 'study_analysis', 'literature_review'])
                        ],
                        'metadata': {
                            'keywords': ['health', 'medicine', f"keyword_{i}"]
                        }
                    }
                },
                color=evidence['color']
            )
            self.network_nodes[evidence['id']] = node_idx
        
        # Create realistic relationships between evidence nodes
        # Based on typical patterns in research (related studies, follow-up studies, etc.)
        relationships = [
            (0, 5, 0.8),  # Exercise & Telomere Length (strongly related)
            (0, 1, 0.6),  # Exercise & Diet (moderately related)
            (1, 3, 0.5),  # Diet & Vitamin D (moderately related)
            (2, 4, 0.7),  # Sleep & Stress (strongly related)
            (3, 4, 0.4),  # Vitamin D & Stress (somewhat related)
            (0, 2, 0.3),  # Exercise & Sleep (somewhat related)
            (1, 5, 0.5),  # Diet & Telomere Length (moderately related)
        ]
        
        for src, dst, weight in relationships:
            if src < len(evidence_data) and dst < len(evidence_data):
                src_id = evidence_data[src]['id'] 
                dst_id = evidence_data[dst]['id']
                if src_id in self.network_nodes and dst_id in self.network_nodes:
                    self.network_view.add_edge(
                        self.network_nodes[src_id],
                        self.network_nodes[dst_id],
                        weight=weight
                    )
    
    def create_sample_study_network(self):
        """Create a sample study network with realistic data"""
        # Generate sample study nodes
        study_data = [
            {
                'id': f"study_{uuid.uuid4().hex[:8]}",
                'title': "ACTIVE Trial: Activity and Cognitive Health",
                'keywords': ['cognition', 'exercise', 'aging', 'prevention'],
                'color': QColor(50, 150, 50)  # Green
            },
            {
                'id': f"study_{uuid.uuid4().hex[:8]}",
                'title': "MEDLIFE Cohort: Mediterranean Lifestyle Study",
                'keywords': ['diet', 'mediterranean', 'lifestyle', 'cardiovascular'],
                'color': QColor(50, 150, 50)
            },
            {
                'id': f"study_{uuid.uuid4().hex[:8]}",
                'title': "DREAM Study: Deep Rest and Mental Acuity",
                'keywords': ['sleep', 'cognition', 'mental health', 'rest'],
                'color': QColor(50, 150, 50)
            },
            {
                'id': f"study_{uuid.uuid4().hex[:8]}",
                'title': "VITAL-D: Vitamin D and Total Health Assessment",
                'keywords': ['vitamin D', 'supplements', 'immune function', 'prevention'],
                'color': QColor(50, 150, 50)
            },
            {
                'id': f"study_{uuid.uuid4().hex[:8]}",
                'title': "RELAX Study: Reducing Stress Through Multiple Modalities",
                'keywords': ['stress', 'mental health', 'cortisol', 'meditation'],
                'color': QColor(50, 150, 50)
            },
            {
                'id': f"study_{uuid.uuid4().hex[:8]}",
                'title': "TELO-HEALTH: Telomere Length and Healthy Aging",
                'keywords': ['aging', 'telomeres', 'exercise', 'longevity'],
                'color': QColor(50, 150, 50)
            },
            {
                'id': f"study_{uuid.uuid4().hex[:8]}",
                'title': "BRAIN-FIT: Brain Health and Fitness Integration Trial",
                'keywords': ['brain', 'cognition', 'exercise', 'mental health'],
                'color': QColor(50, 150, 50)
            }
        ]
        
        # Add nodes to the graph
        for i, study in enumerate(study_data):
            node_idx = self.network_view.add_node(
                f"S:{study['id'][:5]}",  # Short label
                data={
                    'type': 'study',
                    'study_id': study['id'],
                    'title': study['title'],
                    'blocks': [i+1],
                    'evidence_ids': [f"evidence_{i}_{j}" for j in range(1, 3)],
                    'keywords': study['keywords']
                },
                color=study['color']
            )
            self.network_nodes[study['id']] = node_idx
        
        # Create edges based on keyword overlap (realistic connections)
        for i, study1 in enumerate(study_data):
            for j, study2 in enumerate(study_data):
                if i >= j:  # Skip self-connections and avoid duplicates
                    continue
                    
                # Calculate keyword overlap
                overlap = len(set(study1['keywords']) & set(study2['keywords']))
                if overlap > 0:
                    # Add edge with weight based on overlap
                    weight = min(1.0, overlap / 3.0)  # Scale weight
                    self.network_view.add_edge(
                        self.network_nodes[study1['id']],
                        self.network_nodes[study2['id']],
                        weight=weight
                    )
    
    def create_sample_claim_network(self):
        """Create a sample claim network with realistic data"""
        # Define claim types with associated colors
        claim_types = {
            'causal': QColor(200, 50, 50),         # Red for causal claims
            'associative': QColor(50, 150, 200),   # Light blue for associative
            'comparative': QColor(150, 100, 200),  # Purple for comparative
            'descriptive': QColor(100, 150, 100)   # Green for descriptive
        }
        
        # Generate sample claims
        claim_data = [
            {
                'id': f"claim_{uuid.uuid4().hex[:8]}",
                'text': "Regular aerobic exercise significantly reduces systolic blood pressure in hypertensive adults",
                'type': 'causal',
                'evidence_id': 'evidence_001',
                'confidence': 0.85
            },
            {
                'id': f"claim_{uuid.uuid4().hex[:8]}",
                'text': "Mediterranean diet adherence is associated with lower rates of cardiovascular events",
                'type': 'associative',
                'evidence_id': 'evidence_002',
                'confidence': 0.78
            },
            {
                'id': f"claim_{uuid.uuid4().hex[:8]}",
                'text': "Sleep quality shows stronger correlation with executive function than sleep duration",
                'type': 'comparative',
                'evidence_id': 'evidence_003',
                'confidence': 0.72
            },
            {
                'id': f"claim_{uuid.uuid4().hex[:8]}",
                'text': "Vitamin D supplementation increases NK cell activity in deficient individuals",
                'type': 'causal',
                'evidence_id': 'evidence_004',
                'confidence': 0.68
            },
            {
                'id': f"claim_{uuid.uuid4().hex[:8]}",
                'text': "Meditation practice is more effective than biofeedback for reducing cortisol levels",
                'type': 'comparative',
                'evidence_id': 'evidence_005',
                'confidence': 0.75
            },
            {
                'id': f"claim_{uuid.uuid4().hex[:8]}",
                'text': "Vigorous physical activity is associated with longer telomere length in middle-aged adults",
                'type': 'associative',
                'evidence_id': 'evidence_006',
                'confidence': 0.64
            },
            {
                'id': f"claim_{uuid.uuid4().hex[:8]}",
                'text': "Combination of diet and exercise produces greater blood pressure reduction than either alone",
                'type': 'comparative',
                'evidence_id': 'evidence_001',
                'confidence': 0.82
            },
            {
                'id': f"claim_{uuid.uuid4().hex[:8]}",
                'text': "Higher vitamin D levels are associated with lower incidence of respiratory infections",
                'type': 'associative',
                'evidence_id': 'evidence_004',
                'confidence': 0.71
            }
        ]
        
        # Add nodes to the graph
        for i, claim in enumerate(claim_data):
            node_idx = self.network_view.add_node(
                f"C:{claim['id'][:5]}",  # Short label
                data={
                    'type': 'claim',
                    'claim_id': claim['id'],
                    'claim_text': claim['text'],
                    'claim_type': claim['type'],
                    'evidence_id': claim['evidence_id'],
                    'block_index': (i % 4) + 1,  # Simulate block indices
                    'full_claim': {
                        'claim_id': claim['id'],
                        'claim_text': claim['text'],
                        'claim_type': claim['type'],
                        'source_id': f"source_{i+1}",
                        'source_type': 'primary_study',
                        'confidence_score': claim['confidence'],
                        'supporting_data': {
                            'p_value': round(0.01 + (i * 0.01), 3),
                            'effect_size': round(0.2 + (i * 0.1), 2),
                            'sample_size': 100 + (i * 20)
                        }
                    }
                },
                color=claim_types.get(claim['type'], QColor(100, 100, 200))
            )
            self.network_nodes[claim['id']] = node_idx
        
        # Group claims by evidence_id
        claims_by_evidence = {}
        for claim in claim_data:
            eid = claim['evidence_id']
            if eid not in claims_by_evidence:
                claims_by_evidence[eid] = []
            claims_by_evidence[eid].append(claim['id'])
        
        # Connect claims from the same evidence with medium-weight edges
        for evidence_id, claims in claims_by_evidence.items():
            for i, claim1 in enumerate(claims):
                for claim2 in claims[i+1:]:
                    if claim1 in self.network_nodes and claim2 in self.network_nodes:
                        self.network_view.add_edge(
                            self.network_nodes[claim1],
                            self.network_nodes[claim2],
                            weight=0.7,
                            color=QColor(180, 180, 180)
                        )
        
        # Add some cross-evidence connections based on related topics
        related_claims = [
            (0, 6, 0.6),  # Related exercise claims
            (1, 6, 0.5),  # Diet and exercise combination
            (2, 4, 0.4),  # Sleep and stress claims
            (3, 7, 0.8),  # Related vitamin D claims
            (0, 5, 0.3)   # Exercise and telomere claims
        ]
        
        for src, dst, weight in related_claims:
            if src < len(claim_data) and dst < len(claim_data):
                src_id = claim_data[src]['id']
                dst_id = claim_data[dst]['id']
                if src_id in self.network_nodes and dst_id in self.network_nodes:
                    # Create cross-evidence connections with colored edges based on claim types
                    src_type = claim_data[src]['type']
                    dst_type = claim_data[dst]['type']
                    if src_type == dst_type:
                        edge_color = claim_types.get(src_type, QColor(150, 150, 150))
                        edge_color.setAlpha(150)  # Make it semi-transparent
                    else:
                        edge_color = QColor(150, 150, 150)  # Default gray for mixed types
                    
                    self.network_view.add_edge(
                        self.network_nodes[src_id],
                        self.network_nodes[dst_id],
                        weight=weight,
                        color=edge_color
                    )
        
    def refresh_network(self):
        """Refresh the network visualization based on current settings"""
        if not self.blockchain:
            return
            
        # Clear current network
        self.network_view.clear()
        self.network_nodes = {}
        
        # Determine which network to display
        if self.evidence_radio.isChecked():
            self.build_evidence_network()
        elif self.study_radio.isChecked():
            self.build_study_network()
        elif self.claim_radio.isChecked():
            self.build_claim_network()
            
        # Update statistics
        self.update_network_stats()
            
    def build_evidence_network(self):
        """Build a network of evidence nodes with connections based on related studies"""
        # Create a node for each evidence block
        for block in self.blockchain.chain[1:]:  # Skip genesis block
            evidence = block.evidence
            if not isinstance(evidence, dict):
                continue
                
            evidence_id = evidence.get('evidence_id')
            if not evidence_id:
                continue
                
            title = evidence.get('title', 'Untitled')
            label = f"E{block.index}"  # Short label
            
            # Add node with evidence data
            node_idx = self.network_view.add_node(
                label, 
                data={
                    'type': 'evidence',
                    'evidence_id': evidence_id,
                    'title': title,
                    'block_index': block.index,
                    'full_data': evidence
                }
            )
            self.network_nodes[evidence_id] = node_idx
        
        # Create edges between related evidence nodes
        for evidence_id, node_idx in self.network_nodes.items():
            # Get related evidence
            related = self.blockchain.get_related_evidence(evidence_id, max_results=5)
            for related_item in related:
                related_evidence = related_item.get('evidence')
                if not isinstance(related_evidence, dict):
                    continue
                    
                related_id = related_evidence.get('evidence_id')
                if related_id in self.network_nodes:
                    # Add edge with weight based on relatedness score
                    weight = min(1.0, related_item.get('relatedness_score', 0) / 10.0)
                    self.network_view.add_edge(
                        node_idx, 
                        self.network_nodes[related_id],
                        weight=weight
                    )
    
    def build_study_network(self):
        """Build a network of study nodes with connections based on related studies"""
        # Track studies we've already processed
        processed_studies = set()
        
        # Create a node for each unique study
        for block in self.blockchain.chain[1:]:  # Skip genesis block
            evidence = block.evidence
            if not isinstance(evidence, dict):
                continue
                
            study_id = evidence.get('study_id')
            if not study_id or study_id in processed_studies:
                continue
                
            processed_studies.add(study_id)
            title = evidence.get('title', 'Untitled')
            label = f"S:{study_id[:5]}"  # Short label
            
            # Add node with study data
            node_idx = self.network_view.add_node(
                label, 
                data={
                    'type': 'study',
                    'study_id': study_id,
                    'title': title,
                    'blocks': [block.index],
                    'evidence_ids': [evidence.get('evidence_id')],
                },
                color=QColor(50, 150, 50)  # Green for studies
            )
            self.network_nodes[study_id] = node_idx
        
        # Create edges between related studies
        # For now, connect studies that share keywords
        study_keywords = {}
        for study_id in processed_studies:
            # Find blocks for this study
            study_blocks = self.blockchain.get_blocks_by_study_id(study_id)
            if not study_blocks:
                continue
                
            # Extract keywords from the first block
            evidence = study_blocks[0].evidence
            if 'metadata' in evidence and 'keywords' in evidence['metadata']:
                keywords = evidence['metadata']['keywords']
                study_keywords[study_id] = set(keywords)
        
        # Create edges based on keyword overlap
        for study_id, keywords in study_keywords.items():
            if not keywords:
                continue
                
            for other_id, other_keywords in study_keywords.items():
                if study_id == other_id or not other_keywords:
                    continue
                    
                # Calculate overlap
                overlap = len(keywords.intersection(other_keywords))
                if overlap > 0:
                    # Add edge with weight based on overlap
                    weight = min(1.0, overlap / 5.0)  # Scale weight
                    self.network_view.add_edge(
                        self.network_nodes[study_id],
                        self.network_nodes[other_id],
                        weight=weight
                    )
    
    def build_claim_network(self):
        """Build a network of claim nodes with connections based on related claims"""
        # Create nodes for each unique claim
        for block in self.blockchain.chain[1:]:  # Skip genesis block
            evidence = block.evidence
            if not isinstance(evidence, dict) or 'evidence_claims' not in evidence:
                continue
                
            for claim in evidence['evidence_claims']:
                if not isinstance(claim, dict):
                    continue
                    
                claim_id = claim.get('claim_id')
                if not claim_id or claim_id in self.network_nodes:
                    continue
                    
                claim_text = claim.get('claim_text', 'Unnamed Claim')[:30] + "..."
                claim_type = claim.get('claim_type', 'unknown')
                label = f"C:{claim_id[:5]}"
                
                # Color based on claim type
                color = QColor(100, 100, 200)  # Default blue
                if claim_type == 'causal':
                    color = QColor(200, 50, 50)  # Red for causal
                elif claim_type == 'associative':
                    color = QColor(50, 150, 200)  # Light blue for associative
                elif claim_type == 'comparative':
                    color = QColor(150, 100, 200)  # Purple for comparative
                
                # Add node with claim data
                node_idx = self.network_view.add_node(
                    label,
                    data={
                        'type': 'claim',
                        'claim_id': claim_id,
                        'claim_text': claim_text,
                        'claim_type': claim_type,
                        'full_claim': claim,
                        'block_index': block.index,
                        'evidence_id': evidence.get('evidence_id')
                    },
                    color=color
                )
                self.network_nodes[claim_id] = node_idx
        
        # Create edges between claims from the same evidence
        claims_by_evidence = {}
        for block in self.blockchain.chain[1:]:  # Skip genesis block
            evidence = block.evidence
            if not isinstance(evidence, dict) or 'evidence_claims' not in evidence:
                continue
                
            evidence_id = evidence.get('evidence_id')
            if not evidence_id:
                continue
                
            claims_by_evidence[evidence_id] = []
            
            for claim in evidence['evidence_claims']:
                if not isinstance(claim, dict):
                    continue
                    
                claim_id = claim.get('claim_id')
                if claim_id and claim_id in self.network_nodes:
                    claims_by_evidence[evidence_id].append(claim_id)
        
        # Connect claims from the same evidence
        for evidence_id, claims in claims_by_evidence.items():
            for i, claim1 in enumerate(claims):
                for claim2 in claims[i+1:]:
                    self.network_view.add_edge(
                        self.network_nodes[claim1],
                        self.network_nodes[claim2],
                        weight=0.7,
                        color=QColor(180, 180, 180)
                    )
    
    def update_network_stats(self):
        """Update network statistics display"""
        node_count = len(self.network_view.nodes)
        edge_count = len(self.network_view.edges)
        
        # Calculate network density (ratio of actual edges to possible edges)
        density = 0
        if node_count > 1:
            max_edges = (node_count * (node_count - 1)) / 2
            if max_edges > 0:
                density = edge_count / max_edges
        
        self.node_count_label.setText(str(node_count))
        self.edge_count_label.setText(str(edge_count))
        self.density_label.setText(f"{density:.3f}")
    
    def on_node_selected(self, node_idx, node_data):
        """Handle node selection event"""
        if not node_data:
            self.node_details.setText("No data available for this node.")
            return
            
        details = ""
        node_type = node_data.get('type', 'unknown')
        
        if node_type == 'evidence':
            details = self.format_evidence_details(node_data)
        elif node_type == 'study':
            details = self.format_study_details(node_data)
        elif node_type == 'claim':
            details = self.format_claim_details(node_data)
        else:
            details = f"Unknown node type: {node_type}"
            
        self.node_details.setText(details)
    
    def format_evidence_details(self, node_data):
        """Format evidence details for display"""
        details = f"<h3>Evidence Details</h3>"
        details += f"<p><b>ID:</b> {node_data.get('evidence_id', 'N/A')}</p>"
        details += f"<p><b>Title:</b> {node_data.get('title', 'N/A')}</p>"
        details += f"<p><b>Block:</b> {node_data.get('block_index', 'N/A')}</p>"
        
        # Extract from full data if available
        full_data = node_data.get('full_data', {})
        if full_data:
            details += f"<p><b>Study ID:</b> {full_data.get('study_id', 'N/A')}</p>"
            details += f"<p><b>Description:</b> {full_data.get('description', 'N/A')}</p>"
            
            # Count claims and components
            claim_count = len(full_data.get('evidence_claims', []))
            component_count = len(full_data.get('study_components', []))
            details += f"<p><b>Claims:</b> {claim_count}</p>"
            details += f"<p><b>Components:</b> {component_count}</p>"
            
            # Intervention text (shortened)
            intervention = full_data.get('intervention_text', '')
            if intervention:
                if len(intervention) > 100:
                    intervention = intervention[:100] + "..."
                details += f"<p><b>Intervention:</b> {intervention}</p>"
        
        return details
    
    def format_study_details(self, node_data):
        """Format study details for display"""
        details = f"<h3>Study Details</h3>"
        details += f"<p><b>Study ID:</b> {node_data.get('study_id', 'N/A')}</p>"
        details += f"<p><b>Title:</b> {node_data.get('title', 'N/A')}</p>"
        
        # List blocks containing this study
        blocks = node_data.get('blocks', [])
        if blocks:
            details += f"<p><b>Blocks:</b> {', '.join(map(str, blocks))}</p>"
        
        # List evidence IDs for this study
        evidence_ids = node_data.get('evidence_ids', [])
        if evidence_ids:
            details += f"<p><b>Evidence IDs:</b></p><ul>"
            for eid in evidence_ids:
                details += f"<li>{eid}</li>"
            details += "</ul>"
        
        return details
    
    def format_claim_details(self, node_data):
        """Format claim details for display"""
        details = f"<h3>Claim Details</h3>"
        details += f"<p><b>Claim ID:</b> {node_data.get('claim_id', 'N/A')}</p>"
        details += f"<p><b>Type:</b> {node_data.get('claim_type', 'N/A')}</p>"
        details += f"<p><b>Text:</b> {node_data.get('claim_text', 'N/A')}</p>"
        details += f"<p><b>From Evidence:</b> {node_data.get('evidence_id', 'N/A')}</p>"
        details += f"<p><b>Block:</b> {node_data.get('block_index', 'N/A')}</p>"
        
        # Extract from full claim if available
        full_claim = node_data.get('full_claim', {})
        if full_claim:
            details += f"<p><b>Source ID:</b> {full_claim.get('source_id', 'N/A')}</p>"
            details += f"<p><b>Source Type:</b> {full_claim.get('source_type', 'N/A')}</p>"
            details += f"<p><b>Confidence:</b> {full_claim.get('confidence_score', 'N/A')}</p>"
            
            # Supporting data
            supporting_data = full_claim.get('supporting_data', {})
            if supporting_data:
                details += f"<p><b>Supporting Data:</b></p><ul>"
                for key, value in supporting_data.items():
                    details += f"<li>{key}: {value}</li>"
                details += "</ul>"
        
        return details
    
    def toggle_labels(self, enabled):
        """Toggle node labels on/off"""
        self.network_view.node_labels = enabled
        self.network_view.update()
