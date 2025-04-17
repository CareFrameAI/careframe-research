import asyncio
from typing import Dict, List, Optional, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QTabWidget, QFileDialog, QMessageBox, QProgressBar, QScrollArea,
    QGroupBox, QGridLayout, QSplitter, QButtonGroup, QRadioButton,
    QFrame, QDialog, QSpinBox, QCheckBox, QFormLayout, QComboBox, QTextEdit, QApplication,
    QDialogButtonBox, QToolButton, QToolBar, QListWidget, QListWidgetItem, QMenu
)
from PyQt6.QtCore import pyqtSignal, Qt, QUrl, QSize
from PyQt6.QtGui import QFont, QIcon, QDesktopServices, QAction
import logging
import matplotlib
import pandas as pd
import numpy as np
from datetime import datetime
import re

from helpers.load_icon import load_bootstrap_icon
from qt_sections.search import PaperWidget
from study_model.studies_manager import StudiesManager
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, YearLocator, MonthLocator
import matplotlib.cm as cm

from qasync import asyncSlot
from literature_search.clustering import PaperClusterAnalyzer
from literature_search.pattern_extractor import QuoteExtractor
from literature_search.models import SearchResult
from literature_search.model_calls import rate_papers_with_gemini, generate_grounded_review
from admin.portal import secrets

class TopicVisualizationCanvas(FigureCanvas):
    """Canvas for displaying topic model visualization."""
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # Create subplots for different visualizations
        self.topics_ax = self.fig.add_subplot(121)
        self.dist_ax = self.fig.add_subplot(122)
        
        # Apply dark style to both subplots
        self.fig.patch.set_facecolor('#2E2E2E')
        self.topics_ax.set_facecolor('#3A3A3A')
        self.dist_ax.set_facecolor('#3A3A3A')
        
        # Set text color to light gray for better visibility on dark background
        text_color = '#BDBDBD'
        for ax in [self.topics_ax, self.dist_ax]:
            ax.tick_params(colors=text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.title.set_color(text_color)
            for spine in ax.spines.values():
                spine.set_color('#555555')
                
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()
    
    def plot_topic_distribution(self, topics_data, cluster_colors=None):
        """Plot topic distribution across documents."""
        self.topics_ax.clear()
        self.dist_ax.clear()
        
        if not topics_data or not isinstance(topics_data, dict):
            self._show_no_data_message()
            return
            
        # Plot topic keywords
        self._plot_topic_keywords(topics_data.get('topics', {}), cluster_colors)
        
        # Plot document distribution by topic
        self._plot_document_distribution(topics_data.get('document_distribution', {}))
        
        self.fig.tight_layout()
        self.draw()
    
    def _plot_topic_keywords(self, topics, cluster_colors=None):
        """Plot top keywords for each topic."""
        if not topics:
            self.topics_ax.text(0.5, 0.5, "No topic data available", 
                                ha='center', va='center', color='#BDBDBD')
            return
            
        # Default colors if not provided
        if not cluster_colors:
            # Create a color map for topics
            cmap = cm.get_cmap('viridis', max(3, len(topics)))
            cluster_colors = {i: cmap(i/len(topics)) for i in range(len(topics))}
        
        # Prepare data for horizontal bar chart
        all_bars = []
        all_labels = []
        all_colors = []
        y_ticks = []
        y_pos = 0
        
        # For each topic, plot its top keywords as horizontal bars
        for topic_id, topic_data in topics.items():
            keywords = topic_data.get('keywords', [])
            weights = topic_data.get('weights', [])
            
            # Skip topics with no keywords
            if not keywords or not weights:
                continue
                
            # Get color for this topic
            topic_color = cluster_colors.get(int(topic_id), 'blue')
            
            # Add a gap between topics
            if all_bars:
                y_pos += 1
            
            # Add topic label
            topic_label = f"Topic {topic_id}"
            if 'label' in topic_data:
                topic_label = f"{topic_label}: {topic_data['label']}"
            y_ticks.append(y_pos)
            
            # Add each keyword as a bar
            for i, (kw, weight) in enumerate(zip(keywords[:5], weights[:5])):  # Top 5 keywords
                all_bars.append(weight)
                all_labels.append(kw)
                all_colors.append(topic_color)
                y_pos += 1
        
        # Plot horizontal bars
        if all_bars:
            y_positions = range(len(all_bars))
            bars = self.topics_ax.barh(y_positions, all_bars, color=all_colors, alpha=0.7)
            
            # Add keyword labels to the bars
            for i, bar in enumerate(bars):
                self.topics_ax.text(
                    bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height()/2,
                    all_labels[i],
                    va='center',
                    color='#BDBDBD',
                    fontsize=8
                )
            
            # Set y-ticks for topic labels
            # self.topics_ax.set_yticks(y_ticks)
            # self.topics_ax.set_yticklabels([f"Topic {i}" for i in range(len(topics))])
            
            # Hide y-axis labels as we have the keywords as text
            self.topics_ax.set_yticks([])
            
            # Set title and labels
            self.topics_ax.set_title('Topic Keywords', color='#BDBDBD')
            self.topics_ax.set_xlabel('Weight', color='#BDBDBD')
        else:
            self.topics_ax.text(0.5, 0.5, "No keywords available", 
                                ha='center', va='center', color='#BDBDBD')
    
    def _plot_document_distribution(self, doc_distribution):
        """Plot distribution of documents across topics."""
        if not doc_distribution:
            self.dist_ax.text(0.5, 0.5, "No document distribution data", 
                             ha='center', va='center', color='#BDBDBD')
            return
        
        # Count documents per topic
        topic_counts = {}
        for doc_id, topics in doc_distribution.items():
            primary_topic = max(topics, key=lambda x: x[1])[0]  # Get topic with highest weight
            topic_counts[primary_topic] = topic_counts.get(primary_topic, 0) + 1
        
        # Plot as pie chart
        if topic_counts:
            labels = [f"Topic {topic}" for topic in topic_counts.keys()]
            sizes = list(topic_counts.values())
            
            # Use a colormap for the topics
            cmap = cm.get_cmap('viridis', max(3, len(topic_counts)))
            colors = [cmap(i/len(topic_counts)) for i in range(len(topic_counts))]
            
            # Explode the largest segment slightly
            max_idx = sizes.index(max(sizes))
            explode = [0.1 if i == max_idx else 0 for i in range(len(sizes))]
            
            self.dist_ax.pie(
                sizes, 
                explode=explode,
                labels=labels, 
                colors=colors,
                autopct='%1.1f%%',
                shadow=True, 
                startangle=140,
                textprops={'color': '#BDBDBD'}
            )
            self.dist_ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            self.dist_ax.set_title('Documents by Primary Topic', color='#BDBDBD')
        else:
            self.dist_ax.text(0.5, 0.5, "No document distribution data", 
                             ha='center', va='center', color='#BDBDBD')
    
    def _show_no_data_message(self):
        """Display a message when no data is available."""
        for ax in [self.topics_ax, self.dist_ax]:
            ax.text(0.5, 0.5, "No data available", 
                  ha='center', va='center', color='#BDBDBD')
        self.draw()


class RankedPaperWidget(QGroupBox):
    """Widget for displaying ranked paper information with summary and tags."""
    paperClicked = pyqtSignal(dict)  # Signal emitted when paper is clicked for detailed view
    
    def __init__(self, paper, parent=None):
        title = paper.get('title', 'No Title')
        super().__init__(title, parent)
        
        self.paper = paper
        self.setup_ui()
        
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        
        # Top row with metadata and score
        top_row = QHBoxLayout()
        
        # Metadata container
        metadata_widget = QWidget()
        meta_layout = QVBoxLayout(metadata_widget)
        meta_layout.setContentsMargins(0, 0, 0, 0)
        meta_layout.setSpacing(5)
        
        # Authors
        authors = self.paper.get('authors', [])
        if authors:
            if isinstance(authors, list):
                authors_str = ", ".join(authors[:3])
                if len(authors) > 3:
                    authors_str += f" and {len(authors) - 3} more"
            else:
                authors_str = str(authors)
            meta_layout.addWidget(QLabel(f"<b>Authors:</b> {authors_str}"))
        
        # Journal and date
        journal = self.paper.get('journal', 'N/A')
        pub_date = self.paper.get('publication_date', 'N/A')
        meta_layout.addWidget(QLabel(f"<b>Journal:</b> {journal} | <b>Date:</b> {pub_date}"))
        
        # DOI
        doi = self.paper.get('doi', 'N/A')
        doi_label = QLabel(f"<b>DOI:</b> <a href='https://doi.org/{doi}'>{doi}</a>")
        doi_label.setOpenExternalLinks(True)
        meta_layout.addWidget(doi_label)
        
        top_row.addWidget(metadata_widget, stretch=4)
        
        # Score display with attractive styling
        if 'composite_score' in self.paper:
            score_widget = QGroupBox("Score")
            score_layout = QVBoxLayout(score_widget)
            
            score_value = self.paper.get('composite_score', 0)
            score_label = QLabel(f"<h3>{score_value}</h3>")
            score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Style based on score
            if score_value >= 80:
                score_widget.setStyleSheet("QGroupBox { background-color: #4CAF50; color: white; border-radius: 5px; }")
            elif score_value >= 60:
                score_widget.setStyleSheet("QGroupBox { background-color: #FFC107; color: black; border-radius: 5px; }")
            else:
                score_widget.setStyleSheet("QGroupBox { background-color: #9E9E9E; color: white; border-radius: 5px; }")
                
            score_layout.addWidget(score_label)
            top_row.addWidget(score_widget, stretch=1)
        
        main_layout.addLayout(top_row)
        
        # Auto-generated summary (if available)
        if 'summary' in self.paper:
            summary_group = QGroupBox("Summary")
            summary_layout = QVBoxLayout(summary_group)
            
            summary_text = QLabel(self.paper.get('summary', ''))
            summary_text.setWordWrap(True)
            summary_layout.addWidget(summary_text)
            
            main_layout.addWidget(summary_group)
        else:
            # Abstract (truncated version if no summary)
            abstract = self.paper.get('abstract', '')
            if abstract:
                abstract_truncated = abstract[:200] + "..." if len(abstract) > 200 else abstract
                abstract_label = QLabel(f"<b>Abstract:</b> {abstract_truncated}")
                abstract_label.setWordWrap(True)
                main_layout.addWidget(abstract_label)
        
        # Bottom row with tags and action buttons
        bottom_row = QHBoxLayout()
        
        # Tags display (if available)
        if 'tags' in self.paper and self.paper['tags']:
            tags_layout = QHBoxLayout()
            tags_layout.setSpacing(5)
            
            tags_label = QLabel("Tags:")
            tags_layout.addWidget(tags_label)
            
            for tag in self.paper.get('tags', [])[:5]:  # Show first 5 tags
                tag_label = QLabel(tag)
                tag_label.setStyleSheet("""
                    background-color: #555555; 
                    color: white; 
                    padding: 2px 6px;
                    border-radius: 8px;
                """)
                tags_layout.addWidget(tag_label)
                
            tags_layout.addStretch()
            bottom_row.addLayout(tags_layout, stretch=3)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(5)
        
        view_details_btn = QPushButton("View Details")
        view_details_btn.setIcon(load_bootstrap_icon("info-circle"))
        view_details_btn.clicked.connect(lambda: self.paperClicked.emit(self.paper))
        buttons_layout.addWidget(view_details_btn)
        
        bottom_row.addLayout(buttons_layout, stretch=1)
        
        main_layout.addLayout(bottom_row)
        
        # Set minimum height for consistent appearance
        self.setMinimumHeight(200)

class PaperSummaryDialog(QDialog):
    """Dialog for generating, editing, and viewing paper summaries."""
    
    def __init__(self, paper, parent=None):
        super().__init__(parent)
        self.paper = paper
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("Paper Summary")
        self.resize(800, 600)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Paper title
        title_label = QLabel(self.paper.get('title', 'No Title'))
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setWordWrap(True)
        layout.addWidget(title_label)
        
        # Tabs for different content
        tabs = QTabWidget()
        
        # Abstract tab
        abstract_tab = QWidget()
        abstract_layout = QVBoxLayout(abstract_tab)
        
        abstract_text = QTextEdit()
        abstract_text.setReadOnly(True)
        abstract_text.setPlainText(self.paper.get('abstract', 'No abstract available'))
        abstract_layout.addWidget(abstract_text)
        
        tabs.addTab(abstract_tab, "Abstract")
        
        # Summary tab
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        
        self.summary_edit = QTextEdit()
        if 'summary' in self.paper:
            self.summary_edit.setPlainText(self.paper.get('summary', ''))
        else:
            self.summary_edit.setPlaceholderText("No summary available. Click 'Generate Summary' to create one.")
        summary_layout.addWidget(self.summary_edit)
        
        # Generate summary button
        generate_btn = QPushButton("Generate Summary")
        generate_btn.setIcon(load_bootstrap_icon("magic"))
        generate_btn.clicked.connect(self.generate_summary)
        summary_layout.addWidget(generate_btn)
        
        tabs.addTab(summary_tab, "Summary")
        
        # Tags tab
        tags_tab = QWidget()
        tags_layout = QVBoxLayout(tags_tab)
        
        tags_label = QLabel("Add tags to categorize this paper:")
        tags_layout.addWidget(tags_label)
        
        # Current tags display
        self.tags_list = QListWidget()
        for tag in self.paper.get('tags', []):
            self.tags_list.addItem(tag)
        tags_layout.addWidget(self.tags_list)
        
        # Add tag controls
        tag_input_layout = QHBoxLayout()
        
        self.tag_input = QLineEdit()
        self.tag_input.setPlaceholderText("Enter new tag...")
        tag_input_layout.addWidget(self.tag_input)
        
        add_tag_btn = QPushButton("Add")
        add_tag_btn.setIcon(load_bootstrap_icon("plus"))
        add_tag_btn.clicked.connect(self.add_tag)
        tag_input_layout.addWidget(add_tag_btn)
        
        tags_layout.addLayout(tag_input_layout)
        
        tabs.addTab(tags_tab, "Tags")
        
        # Add tabs to layout
        layout.addWidget(tabs)
        
        # Button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.save_changes)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    @asyncSlot()
    async def generate_summary(self):
        """Generate a summary of the paper using an AI model."""
        try:
            # Show loading indicator
            self.summary_edit.setPlainText("Generating summary...")
            QApplication.processEvents()  # Force UI update
            
            # Get model from the API (note: we'll need to implement this)
            title = self.paper.get('title', '')
            abstract = self.paper.get('abstract', '')
            
            if not abstract:
                QMessageBox.warning(
                    self, 
                    "Missing Abstract", 
                    "Cannot generate summary without an abstract."
                )
                self.summary_edit.setPlainText("")
                return
            
            # Call an AI model to generate summary
            # This is a placeholder - will need to implement actual API call
            summary = await self.call_summary_api(title, abstract)
            
            if summary:
                self.summary_edit.setPlainText(summary)
            else:
                self.summary_edit.setPlainText("Failed to generate summary.")
                
        except Exception as e:
            logging.error(f"Error generating summary: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to generate summary: {str(e)}")
            self.summary_edit.setPlainText("")
    
    async def call_summary_api(self, title, abstract):
        """Call an AI API to generate a paper summary."""
        # This is a placeholder - implement actual API call
        # For now, returning a simple extraction of the first few sentences
        
        # In a real implementation, you would call an LLM API here:
        # summary = await call_llm_api(prompt=f"Summarize this paper: Title: {title}\nAbstract: {abstract}")
        
        # Placeholder implementation:
        sentences = abstract.split('. ')
        if len(sentences) <= 3:
            return abstract
        
        # Return first 2-3 sentences as a summary
        summary = '. '.join(sentences[:3]) + '.'
        return summary
    
    def add_tag(self):
        """Add a new tag to the paper."""
        tag = self.tag_input.text().strip()
        if tag:
            # Check for duplicates
            items = [self.tags_list.item(i).text() for i in range(self.tags_list.count())]
            if tag not in items:
                self.tags_list.addItem(tag)
            self.tag_input.clear()
    
    def save_changes(self):
        """Save changes to the paper summary and tags."""
        # Update the paper object with new summary
        self.paper['summary'] = self.summary_edit.toPlainText()
        
        # Update tags
        tags = [self.tags_list.item(i).text() for i in range(self.tags_list.count())]
        self.paper['tags'] = tags
        
        self.accept()


class LiteratureSummarizeSection(QWidget):
    """
    Widget for summarizing, organizing, and analyzing literature search results.
    Provides tools for paper ranking, clustering, topic modeling, and summary generation.
    """
    # Define signals
    paperUpdated = pyqtSignal(dict)  # Signal when a paper is updated
    analysisCompleted = pyqtSignal(bool)  # Signal when analysis completes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
        # Initialize data containers
        self.papers = []
        self.ranked_papers = []
        self.topics_data = {}
        self.studies_manager = None
        
        # Initialize modules
        self.cluster_analyzer = None
        self.loaded_dataset_name = None
        
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Create header
        header_layout = QHBoxLayout()
        
        title = QLabel("Literature Summarization")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
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
        
        # Create main content area with tabs
        self.tabs = QTabWidget()
        
        # --- Analysis Tab ---
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        analysis_layout.setSpacing(15)
        
        # Analysis controls
        controls_group = QGroupBox("Analysis Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        # Left side - ranking options
        ranking_options = QGroupBox("Ranking Options")
        ranking_form = QFormLayout(ranking_options)
        
        self.rank_count_spin = QSpinBox()
        self.rank_count_spin.setRange(5, 100)
        self.rank_count_spin.setValue(25)
        self.rank_count_spin.setSingleStep(5)
        ranking_form.addRow("Papers to rank:", self.rank_count_spin)
        
        self.relevance_weight = QSpinBox()
        self.relevance_weight.setRange(1, 5)
        self.relevance_weight.setValue(3)
        ranking_form.addRow("Relevance weight:", self.relevance_weight)
        
        self.recency_weight = QSpinBox()
        self.recency_weight.setRange(1, 5)
        self.recency_weight.setValue(2)
        ranking_form.addRow("Recency weight:", self.recency_weight)
        
        self.citation_weight = QSpinBox()
        self.citation_weight.setRange(1, 5)
        self.citation_weight.setValue(2)
        ranking_form.addRow("Citation weight:", self.citation_weight)
        
        controls_layout.addWidget(ranking_options)
        
        # Right side - clustering options
        clustering_options = QGroupBox("Clustering Options")
        clustering_form = QFormLayout(clustering_options)
        
        self.cluster_count = QSpinBox()
        self.cluster_count.setRange(2, 10)
        self.cluster_count.setValue(3)
        clustering_form.addRow("Number of clusters:", self.cluster_count)
        
        self.topic_count = QSpinBox()
        self.topic_count.setRange(2, 10)
        self.topic_count.setValue(5)
        clustering_form.addRow("Number of topics:", self.topic_count)
        
        controls_layout.addWidget(clustering_options)
        
        # Right side - action buttons
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        self.analyze_btn = QPushButton("Analyze Papers")
        self.analyze_btn.setIcon(load_bootstrap_icon("graph-up"))
        self.analyze_btn.clicked.connect(self.analyze_papers)
        actions_layout.addWidget(self.analyze_btn)
        
        self.summarize_all_btn = QPushButton("Summarize All Papers")
        self.summarize_all_btn.setIcon(load_bootstrap_icon("file-text"))
        self.summarize_all_btn.clicked.connect(self.summarize_all_papers)
        actions_layout.addWidget(self.summarize_all_btn)
        
        controls_layout.addWidget(actions_group)
        
        analysis_layout.addWidget(controls_group)
        
        # Status and progress
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        
        analysis_layout.addLayout(status_layout)
        
        # Topic visualization
        viz_group = QGroupBox("Topic Analysis")
        viz_layout = QVBoxLayout(viz_group)
        
        self.topic_canvas = TopicVisualizationCanvas()
        viz_layout.addWidget(self.topic_canvas)
        
        analysis_layout.addWidget(viz_group)
        
        # Add to tabs
        self.tabs.addTab(analysis_tab, "Analysis")
        
        # --- Ranked Papers Tab ---
        ranked_tab = QWidget()
        ranked_layout = QVBoxLayout(ranked_tab)
        ranked_layout.setSpacing(15)
        
        # Filtering controls
        filter_layout = QHBoxLayout()
        
        filter_label = QLabel("Filter papers:")
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Enter keywords to filter papers...")
        self.filter_input.textChanged.connect(self.filter_papers)
        
        sort_label = QLabel("Sort by:")
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Score (High to Low)", "Score (Low to High)", "Date (Newest)", "Date (Oldest)"])
        self.sort_combo.currentIndexChanged.connect(self.sort_papers)
        
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.filter_input, 3)
        filter_layout.addWidget(sort_label)
        filter_layout.addWidget(self.sort_combo, 1)
        
        ranked_layout.addLayout(filter_layout)
        
        # Ranked papers list
        self.papers_scroll = QScrollArea()
        self.papers_scroll.setWidgetResizable(True)
        self.papers_container = QWidget()
        self.papers_layout = QVBoxLayout(self.papers_container)
        self.papers_layout.setSpacing(10)
        self.papers_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.papers_scroll.setWidget(self.papers_container)
        
        ranked_layout.addWidget(self.papers_scroll)
        
        # Add to tabs
        self.tabs.addTab(ranked_tab, "Ranked Papers")
        
        # --- Cluster View Tab ---
        cluster_tab = QWidget()
        cluster_layout = QVBoxLayout(cluster_tab)
        
        # Cluster selection
        cluster_controls = QHBoxLayout()
        
        cluster_label = QLabel("Select cluster:")
        self.cluster_combo = QComboBox()
        self.cluster_combo.currentIndexChanged.connect(self.display_cluster)
        
        cluster_controls.addWidget(cluster_label)
        cluster_controls.addWidget(self.cluster_combo)
        cluster_controls.addStretch()
        
        cluster_layout.addLayout(cluster_controls)
        
        # Cluster papers list
        self.cluster_scroll = QScrollArea()
        self.cluster_scroll.setWidgetResizable(True)
        self.cluster_container = QWidget()
        self.cluster_layout = QVBoxLayout(self.cluster_container)
        self.cluster_layout.setSpacing(10)
        self.cluster_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.cluster_scroll.setWidget(self.cluster_container)
        
        cluster_layout.addWidget(self.cluster_scroll)
        
        # Add to tabs
        self.tabs.addTab(cluster_tab, "Clusters")
        
        # --- Literature Review Tab ---
        review_tab = QWidget()
        review_layout = QVBoxLayout(review_tab)
        
        # Review controls
        review_controls = QHBoxLayout()
        
        generate_review_btn = QPushButton("Generate Literature Review")
        generate_review_btn.setIcon(load_bootstrap_icon("file-earmark-text"))
        generate_review_btn.clicked.connect(self.generate_literature_review)
        review_controls.addWidget(generate_review_btn)
        
        save_review_btn = QPushButton("Save Review")
        save_review_btn.setIcon(load_bootstrap_icon("save"))
        save_review_btn.clicked.connect(self.save_literature_review)
        review_controls.addWidget(save_review_btn)
        
        review_controls.addStretch()
        
        review_layout.addLayout(review_controls)
        
        # Review display
        self.review_text = QTextEdit()
        self.review_text.setReadOnly(True)
        self.review_text.setPlaceholderText("No literature review generated. Click 'Generate Literature Review' to create one.")
        review_layout.addWidget(self.review_text)
        
        # Add to tabs
        self.tabs.addTab(review_tab, "Literature Review")
        
        # Add tabs to main layout
        main_layout.addWidget(self.tabs)
    
    def set_studies_manager(self, studies_manager: StudiesManager):
        """Set the studies manager for accessing saved datasets."""
        self.studies_manager = studies_manager
        self.refresh_data_sources()
    
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
            
        # Add search result datasets
        search_datasets = [name for name in datasets if name.startswith("search_results_")]
        if search_datasets:
            for dataset in search_datasets:
                self.data_source_combo.addItem(dataset)
        else:
            self.data_source_combo.addItem("No search results available")
    
    def load_selected_dataset(self):
        """Load the selected dataset from the dropdown."""
        if not self.studies_manager:
            return
            
        dataset_name = self.data_source_combo.currentText()
        if not dataset_name or dataset_name in ["No active study", "No datasets available", "No search results available"]:
            self.papers = []
            self.display_papers([])
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
                self.display_papers([])
                return
                
            # Convert DataFrame to list of dictionaries
            self.papers = df.to_dict('records')
            self.loaded_dataset_name = dataset_name
            
            # Reset analysis results
            self.ranked_papers = []
            self.topics_data = {}
            
            # Display papers
            self.display_papers(self.papers)
            
            # Update status
            self.status_label.setText(f"Loaded {len(self.papers)} papers from {dataset_name}")
            
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load dataset: {str(e)}")
    
    def display_papers(self, papers):
        """Display papers in the papers container."""
        # Clear existing papers
        self.clear_layout(self.papers_layout)
        
        if not papers:
            no_papers_label = QLabel("No papers available. Load papers from a search result dataset.")
            no_papers_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.papers_layout.addWidget(no_papers_label)
            return
            
        # Add papers to layout
        for paper in papers:
            paper_widget = RankedPaperWidget(paper)
            paper_widget.paperClicked.connect(self.view_paper_details)
            self.papers_layout.addWidget(paper_widget)
    
    def view_paper_details(self, paper):
        """Show the paper details dialog."""
        dialog = PaperSummaryDialog(paper, self)
        if dialog.exec():
            # If dialog was accepted (Save clicked), update the paper
            # Find the paper in our collections and update it
            for collection in [self.papers, self.ranked_papers]:
                for i, p in enumerate(collection):
                    if p.get('doi') == paper.get('doi'):
                        collection[i] = paper
                        
            # Refresh the display
            if self.tabs.currentIndex() == 1:  # Ranked Papers tab
                self.display_papers(self.ranked_papers)
            elif self.tabs.currentIndex() == 2:  # Clusters tab
                self.display_cluster(self.cluster_combo.currentIndex())
                
            # Emit signal that paper was updated
            self.paperUpdated.emit(paper)
    
    def clear_layout(self, layout):
        """Helper function to clear all widgets from a layout."""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
    
    def filter_papers(self):
        """Filter papers based on search text."""
        filter_text = self.filter_input.text().lower()
        
        # Determine which papers to filter
        papers_to_filter = self.ranked_papers if self.ranked_papers else self.papers
        
        if not filter_text:
            # If no filter, show all papers
            self.display_papers(papers_to_filter)
            return
            
        # Filter papers
        filtered_papers = []
        for paper in papers_to_filter:
            title = paper.get('title', '').lower()
            abstract = paper.get('abstract', '').lower()
            authors = ' '.join(paper.get('authors', [])).lower()
            
            if (filter_text in title or 
                filter_text in abstract or 
                filter_text in authors):
                filtered_papers.append(paper)
        
        # Display filtered papers
        self.display_papers(filtered_papers)
    
    def sort_papers(self):
        """Sort papers based on selected criteria."""
        sort_option = self.sort_combo.currentText()
        
        # Determine which papers to sort
        papers_to_sort = self.ranked_papers if self.ranked_papers else self.papers
        sorted_papers = papers_to_sort.copy()
        
        try:
            if sort_option == "Score (High to Low)":
                sorted_papers.sort(key=lambda p: p.get('composite_score', 0), reverse=True)
            elif sort_option == "Score (Low to High)":
                sorted_papers.sort(key=lambda p: p.get('composite_score', 0))
            elif sort_option == "Date (Newest)":
                # Try to convert date strings to datetime objects for sorting
                sorted_papers.sort(key=lambda p: self._parse_date(p.get('publication_date', '')), reverse=True)
            elif sort_option == "Date (Oldest)":
                sorted_papers.sort(key=lambda p: self._parse_date(p.get('publication_date', '')))
        except Exception as e:
            logging.error(f"Error sorting papers: {str(e)}")
            QMessageBox.warning(self, "Sorting Error", f"Failed to sort papers: {str(e)}")
            
        # Display sorted papers
        self.display_papers(sorted_papers)
    
    def _parse_date(self, date_str):
        """Parse date string to datetime object for sorting."""
        if not date_str:
            return datetime.min
        
        try:
            # Try various date formats
            for fmt in ['%Y-%m-%d', '%Y-%m', '%Y', '%d-%m-%Y', '%m/%d/%Y']:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
                    
            # If all formats fail, extract year
            year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
            if year_match:
                return datetime.strptime(year_match.group(), '%Y')
                
        except Exception:
            pass
            
        # Return minimum date if parsing fails
        return datetime.min
    
    @asyncSlot()
    async def analyze_papers(self):
        """Analyze papers by ranking and clustering them."""
        if not self.papers:
            QMessageBox.warning(self, "No Papers", "No papers to analyze. Please load a dataset first.")
            return
            
        try:
            # Update status
            self.status_label.setText("Analyzing papers...")
            self.progress_bar.setValue(10)
            QApplication.processEvents()  # Force UI update
            
            # Initialize cluster analyzer if needed
            if not self.cluster_analyzer:
                self.cluster_analyzer = PaperClusterAnalyzer(logging.getLogger(__name__))
                
            # Rank papers
            await self.rank_papers()
            self.progress_bar.setValue(40)
            
            # Cluster papers
            await self.cluster_papers()
            self.progress_bar.setValue(70)
            
            # Generate topics
            await self.generate_topics()
            self.progress_bar.setValue(100)
            
            # Update status
            self.status_label.setText("Analysis complete")
            
            # Switch to ranked papers tab
            self.tabs.setCurrentIndex(1)  # Ranked Papers tab
            
            # Emit signal that analysis is complete
            self.analysisCompleted.emit(True)
            
        except Exception as e:
            logging.error(f"Error analyzing papers: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to analyze papers: {str(e)}")
            self.status_label.setText("Analysis failed")
            self.progress_bar.setValue(0)
            self.analysisCompleted.emit(False)
    
    @asyncSlot()
    async def rank_papers(self):
        """Rank papers based on relevance, recency, and impact."""
        if not self.papers:
            return
            
        try:
            # Update status
            self.status_label.setText("Ranking papers...")
            QApplication.processEvents()  # Force UI update
            
            # Get ranking parameters
            max_papers = self.rank_count_spin.value()
            relevance_weight = self.relevance_weight.value()
            recency_weight = self.recency_weight.value()
            citation_weight = self.citation_weight.value()
            
            # Construct ranking query (using dataset name or simple "relevant papers")
            ranking_query = self.loaded_dataset_name
            if not ranking_query or not ranking_query.startswith("search_results_"):
                ranking_query = "relevant academic papers"
                
            # Clean up the ranking query
            ranking_query = ranking_query.replace("search_results_", "").replace("_", " ")
            
            # Rank papers using LLM
            self.ranked_papers = await rate_papers_with_gemini(
                self.papers, 
                ranking_query,
                logging.getLogger(__name__)
            )
            
            # Sort by score
            self.ranked_papers.sort(key=lambda p: p.get('composite_score', 0), reverse=True)
            
            # Limit to max papers
            self.ranked_papers = self.ranked_papers[:max_papers]
            
            # Display ranked papers
            self.display_papers(self.ranked_papers)
            
            # Update status
            self.status_label.setText(f"Ranked {len(self.ranked_papers)} papers")
            
            # Store in studies manager
            if self.studies_manager and self.ranked_papers:
                ranked_df = pd.DataFrame(self.ranked_papers)
                self.studies_manager.add_dataset_to_active_study(
                    f"ranked_papers_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    ranked_df,
                    metadata={
                        'source_dataset': self.loaded_dataset_name,
                        'parameters': {
                            'max_papers': max_papers,
                            'relevance_weight': relevance_weight,
                            'recency_weight': recency_weight,
                            'citation_weight': citation_weight
                        }
                    }
                )
                
        except Exception as e:
            logging.error(f"Error ranking papers: {str(e)}")
            raise
    
    @asyncSlot()
    async def cluster_papers(self):
        """Cluster papers based on content similarity."""
        if not self.ranked_papers:
            return
            
        try:
            # Update status
            self.status_label.setText("Clustering papers...")
            QApplication.processEvents()  # Force UI update
            
            # Get number of clusters
            num_clusters = self.cluster_count.value()
            
            # Initialize cluster analyzer if needed
            if not self.cluster_analyzer:
                self.cluster_analyzer = PaperClusterAnalyzer(logging.getLogger(__name__))
                
            # Cluster papers - don't pass n_clusters parameter as it's not supported
            cluster_info, timeline_points = self.cluster_analyzer.cluster_papers_with_scores(
                self.ranked_papers
            )
            
            # Update cluster combo box
            self.cluster_combo.clear()
            
            # Get actual number of clusters from the results
            actual_clusters = len(set(p.get('cluster_id', -1) for p in self.ranked_papers))
            
            for i in range(actual_clusters):
                self.cluster_combo.addItem(f"Cluster {i+1}")
                
            # Display first cluster
            if actual_clusters > 0:
                self.display_cluster(0)
                
            # Update status
            self.status_label.setText(f"Created {actual_clusters} clusters")
            
        except Exception as e:
            logging.error(f"Error clustering papers: {str(e)}")
            raise
    
    def display_cluster(self, index):
        """Display papers in the selected cluster."""
        if index < 0 or not self.ranked_papers or not self.cluster_analyzer:
            return
            
        try:
            # Clear existing papers in cluster view
            self.clear_layout(self.cluster_layout)
            
            # Get papers in this cluster
            # Cluster IDs in PaperClusterAnalyzer are 1-based (1, 2, 3)
            # Add 1 to convert from 0-based index to 1-based cluster_id
            cluster_id = index + 1  
            papers_in_cluster = [p for p in self.ranked_papers if p.get('cluster_id', -1) == cluster_id]
            
            if not papers_in_cluster:
                no_papers_label = QLabel(f"No papers in cluster {index+1}")
                no_papers_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.cluster_layout.addWidget(no_papers_label)
                return
                
            # Add cluster info header
            cluster_title = QLabel(f"<h3>Cluster {index+1} ({len(papers_in_cluster)} papers)</h3>")
            self.cluster_layout.addWidget(cluster_title)
            
            # Add cluster description based on the fixed clustering logic in PaperClusterAnalyzer
            cluster_description = ""
            if cluster_id == 1:
                cluster_description = "High relevance papers (score >= 80)"
            elif cluster_id == 2:
                cluster_description = "Medium relevance papers (score 60-79)"
            else:
                cluster_description = "Low relevance papers (score < 60)"
                
            cluster_desc = QLabel(f"<i>{cluster_description}</i>")
            cluster_desc.setWordWrap(True)
            self.cluster_layout.addWidget(cluster_desc)
            
            # Add horizontal line
            line = QFrame()
            line.setFrameShape(QFrame.Shape.HLine)
            line.setFrameShadow(QFrame.Shadow.Sunken)
            self.cluster_layout.addWidget(line)
            
            # Add papers to layout
            for paper in papers_in_cluster:
                paper_widget = RankedPaperWidget(paper)
                paper_widget.paperClicked.connect(self.view_paper_details)
                self.cluster_layout.addWidget(paper_widget)
                
        except Exception as e:
            logging.error(f"Error displaying cluster: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to display cluster: {str(e)}")
    
    @asyncSlot()
    async def generate_topics(self):
        """Generate topic models from the papers."""
        if not self.ranked_papers:
            return
            
        try:
            # Update status
            self.status_label.setText("Generating topics...")
            QApplication.processEvents()  # Force UI update
            
            # Get number of topics
            num_topics = self.topic_count.value()
            
            # Initialize cluster analyzer if needed
            if not self.cluster_analyzer:
                self.cluster_analyzer = PaperClusterAnalyzer(logging.getLogger(__name__))
            
            # Check if cluster analyzer has generate_topics method
            if not hasattr(self.cluster_analyzer, 'generate_topics'):
                logging.warning("ClusterAnalyzer does not have generate_topics method")
                self.status_label.setText("Topic generation not supported by current analyzer")
                
                # Create a simple topic visualization based on clusters instead
                self.generate_topics_from_clusters()
                return
                
            # Extract abstracts for topic modeling
            abstracts = [p.get('abstract', '') for p in self.ranked_papers]
            
            # Generate topics using the cluster analyzer
            self.topics_data = self.cluster_analyzer.generate_topics(
                abstracts, 
                n_topics=num_topics
            )
            
            # Assign topic to papers
            if 'document_distribution' in self.topics_data:
                for i, paper in enumerate(self.ranked_papers):
                    if str(i) in self.topics_data['document_distribution']:
                        # Get primary topic (topic with highest weight)
                        topics = self.topics_data['document_distribution'][str(i)]
                        primary_topic = max(topics, key=lambda x: x[1])[0]
                        paper['primary_topic'] = primary_topic
            
            # Plot topic visualization
            self.topic_canvas.plot_topic_distribution(self.topics_data)
            
            # Update status
            self.status_label.setText(f"Generated {num_topics} topics")
            
        except Exception as e:
            logging.error(f"Error generating topics: {str(e)}")
            # Create a fallback visualization
            self.generate_topics_from_clusters()
            self.status_label.setText("Used cluster-based topics (topic modeling failed)")
    
    def generate_topics_from_clusters(self):
        """Generate a simple topic visualization based on clusters as a fallback."""
        try:
            # Create a simplified topics data structure based on clusters
            topics_data = {
                'topics': {
                    '0': {
                        'keywords': ['high', 'relevance', 'important', 'significant', 'key'],
                        'weights': [0.9, 0.8, 0.7, 0.6, 0.5],
                        'label': 'High Relevance'
                    },
                    '1': {
                        'keywords': ['medium', 'moderate', 'relevant', 'interesting', 'useful'],
                        'weights': [0.8, 0.7, 0.6, 0.5, 0.4],
                        'label': 'Medium Relevance'
                    },
                    '2': {
                        'keywords': ['low', 'relevance', 'marginal', 'peripheral', 'tangential'],
                        'weights': [0.7, 0.6, 0.5, 0.4, 0.3],
                        'label': 'Low Relevance'
                    }
                },
                'document_distribution': {}
            }
            
            # Assign documents to topics based on cluster_id
            for i, paper in enumerate(self.ranked_papers):
                cluster_id = paper.get('cluster_id', 3)
                # Convert cluster_id (1,2,3) to topic_id (0,1,2)
                topic_id = cluster_id - 1
                # Ensure topic_id is within valid range
                topic_id = max(0, min(topic_id, 2))
                
                # Create a simple distribution with 100% weight on the corresponding topic
                topics_data['document_distribution'][str(i)] = [(str(topic_id), 1.0)]
            
            # Store the topics data
            self.topics_data = topics_data
            
            # Plot the visualization
            self.topic_canvas.plot_topic_distribution(topics_data)
            
        except Exception as e:
            logging.error(f"Error generating fallback topics: {str(e)}")
            # Show a basic message in the canvas
            self.topic_canvas._show_no_data_message()
    
    @asyncSlot()
    async def summarize_all_papers(self):
        """Generate summaries for all ranked papers."""
        if not self.ranked_papers:
            QMessageBox.warning(self, "No Papers", "No ranked papers to summarize. Please analyze papers first.")
            return
            
        try:
            # Confirm with user
            reply = QMessageBox.question(
                self, 
                "Summarize Papers", 
                f"Generate summaries for {len(self.ranked_papers)} papers? This may take some time.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply != QMessageBox.StandardButton.Yes:
                return
                
            # Update status
            self.status_label.setText("Generating summaries...")
            progress_step = 100 / len(self.ranked_papers)
            self.progress_bar.setValue(0)
            QApplication.processEvents()  # Force UI update
            
            # Generate summaries one by one
            for i, paper in enumerate(self.ranked_papers):
                # Skip if already has summary
                if 'summary' in paper and paper['summary']:
                    continue
                    
                # Update progress
                self.status_label.setText(f"Summarizing paper {i+1} of {len(self.ranked_papers)}")
                self.progress_bar.setValue(int((i+1) * progress_step))
                QApplication.processEvents()  # Force UI update
                
                # Generate summary
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                
                if not abstract:
                    continue
                    
                try:
                    # Generate summary (placeholder - implement actual call)
                    summary = await self._generate_paper_summary(title, abstract)
                    
                    if summary:
                        paper['summary'] = summary
                except Exception as e:
                    logging.error(f"Error generating summary for paper: {str(e)}")
                    continue
            
            # Refresh display
            self.display_papers(self.ranked_papers)
            
            # Update status
            self.status_label.setText("Summaries generated")
            self.progress_bar.setValue(100)
            
            # Store in studies manager
            if self.studies_manager:
                summarized_df = pd.DataFrame(self.ranked_papers)
                self.studies_manager.add_dataset_to_active_study(
                    f"summarized_papers_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    summarized_df,
                    metadata={
                        'source_dataset': self.loaded_dataset_name
                    }
                )
                
        except Exception as e:
            logging.error(f"Error generating summaries: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to generate summaries: {str(e)}")
            self.status_label.setText("Summary generation failed")
            self.progress_bar.setValue(0)
    
    async def _generate_paper_summary(self, title, abstract):
        """Generate a summary for a paper using an AI model."""
        # This is a placeholder - implement actual API call
        # For now, returning a simple extraction of the first few sentences
        
        # In a real implementation, you would call an LLM API here:
        # summary = await call_llm_api(prompt=f"Summarize this paper: Title: {title}\nAbstract: {abstract}")
        
        # Placeholder implementation:
        sentences = abstract.split('. ')
        if len(sentences) <= 3:
            return abstract
        
        # Return first 2-3 sentences as a summary
        summary = '. '.join(sentences[:3]) + '.'
        return summary
    
    @asyncSlot()
    async def generate_literature_review(self):
        """Generate a comprehensive literature review from ranked papers."""
        if not self.ranked_papers:
            QMessageBox.warning(self, "No Papers", "No ranked papers for review. Please analyze papers first.")
            return
            
        try:
            # Update status
            self.status_label.setText("Generating literature review...")
            self.progress_bar.setValue(10)
            QApplication.processEvents()  # Force UI update
            
            # Create search results objects from paper dictionaries
            search_results = []
            for paper in self.ranked_papers:
                result = SearchResult(
                    title=paper.get('title', ''),
                    authors=paper.get('authors', []),
                    abstract=paper.get('abstract', ''),
                    doi=paper.get('doi', ''),
                    publication_date=paper.get('publication_date', ''),
                    journal=paper.get('journal', ''),
                    source=paper.get('source', 'Unknown'),
                    pmcid=paper.get('pmcid', ''),
                    pmid=paper.get('pmid', '')
                )
                search_results.append(result)
                
            # Create query from stored query, dataset name, or default
            query = ""
            if hasattr(self, 'query') and self.query:
                query = f"literature review on {self.query}"
            elif self.loaded_dataset_name and self.loaded_dataset_name.startswith("search_results_"):
                query = self.loaded_dataset_name.replace("search_results_", "literature review on ").replace("_", " ")
            else:
                query = "literature review on the selected papers"
                
            # Initialize quote extractor if needed
            if not hasattr(self, 'quote_extractor') or not self.quote_extractor:
                self.quote_extractor = QuoteExtractor(logging.getLogger(__name__))
                
            self.progress_bar.setValue(20)
            
            # Generate review
            review = await generate_grounded_review(
                papers=search_results,
                query=query,
                quote_extractor=self.quote_extractor,
                logger=logging.getLogger(__name__)
            )
            
            self.progress_bar.setValue(90)
            
            if review:
                # Format review for display
                formatted_review = f"<h1>{review.title}</h1>\n\n"
                formatted_review += f"<h2>Introduction</h2>\n<p>{review.introduction}</p>\n\n"
                
                for section in review.sections:
                    formatted_review += f"<h2>{section.theme}</h2>\n<p>{section.content}</p>\n\n"
                    
                    if hasattr(section, 'quotes') and section.quotes:
                        formatted_review += "<h3>Supporting Evidence</h3>\n<ul>\n"
                        for quote in section.quotes:
                            formatted_review += f"<li><i>\"{quote.text}\"</i> - {quote.paper_title}</li>\n"
                        formatted_review += "</ul>\n\n"
                        
                formatted_review += f"<h2>Conclusion</h2>\n<p>{review.conclusion}</p>\n\n"
                
                if review.citations:
                    formatted_review += "<h2>References</h2>\n<ol>\n"
                    for citation in review.citations:
                        formatted_review += f"<li>{citation}</li>\n"
                    formatted_review += "</ol>"
                
                # Display review
                self.review_text.setHtml(formatted_review)
                
                # Switch to review tab
                self.tabs.setCurrentIndex(3)  # Literature Review tab
                
                # Update status
                self.status_label.setText("Literature review generated")
                
                # Store in studies manager
                if self.studies_manager:
                    # Store as a text file
                    review_dict = {
                        'title': review.title,
                        'introduction': review.introduction,
                        'sections': [{'theme': s.theme, 'content': s.content} for s in review.sections],
                        'conclusion': review.conclusion,
                        'citations': review.citations
                    }
                    self.studies_manager.add_dataset_to_active_study(
                        f"literature_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        pd.DataFrame([review_dict]),
                        metadata={
                            'source_dataset': self.loaded_dataset_name
                        }
                    )
            else:
                QMessageBox.warning(self, "Review Generation Failed", "Could not generate a literature review.")
                self.status_label.setText("Review generation failed")
            
            self.progress_bar.setValue(100)
            
        except Exception as e:
            logging.error(f"Error generating literature review: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to generate literature review: {str(e)}")
            self.status_label.setText("Review generation failed")
            self.progress_bar.setValue(0)
    
    def save_literature_review(self):
        """Save the generated literature review to a file."""
        if not self.review_text.toPlainText():
            QMessageBox.warning(self, "No Review", "No literature review to save.")
            return
            
        try:
            # Get save location
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Save Literature Review",
                "",
                "HTML Files (*.html);;Text Files (*.txt);;Markdown Files (*.md)"
            )
            
            if not file_name:
                return
                
            # Save the file
            with open(file_name, 'w', encoding='utf-8') as f:
                if file_name.endswith('.html'):
                    # Save as HTML
                    f.write(self.review_text.toHtml())
                elif file_name.endswith('.md'):
                    # Convert to markdown and save
                    # This is simplified - would need a proper HTML to Markdown converter
                    markdown = self.review_text.toPlainText()
                    f.write(markdown)
                else:
                    # Save as plain text
                    f.write(self.review_text.toPlainText())
                    
            QMessageBox.information(self, "Review Saved", f"Literature review saved to {file_name}")
            
        except Exception as e:
            logging.error(f"Error saving literature review: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save literature review: {str(e)}")

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