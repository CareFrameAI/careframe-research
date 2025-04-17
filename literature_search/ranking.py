import asyncio
from typing import Dict, List, Optional, Any, Tuple
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QGroupBox, QGridLayout, QSplitter, QRadioButton,
    QFrame, QDialog, QSpinBox, QCheckBox, QFormLayout, QComboBox, QTextEdit, QApplication, QFileDialog, QScrollArea, QProgressBar, QInputDialog, QListWidget, QMessageBox, QListWidgetItem
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont, QColor
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import re
import uuid
import os

from helpers.load_icon import load_bootstrap_icon
from study_model.studies_manager import StudiesManager
from qasync import asyncSlot
from literature_search.model_calls import rate_papers_with_gemini
from literature_search.paper_ranker import PaperRanker
from llms.client import call_llm_async

class PaperRankingSection(QWidget):
    """Widget for ranking papers based on relevance to hypotheses."""
    
    papersRanked = pyqtSignal(list)  # Signal for when papers are ranked
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.papers = []
        self.ranked_papers = []
        self.studies_manager = None
        self.query = ""
        
        # Initialize the paper ranker
        cache_dir = os.path.join(os.path.expanduser("~"), ".researchtools", "paper_ranker_cache")
        self.paper_ranker = PaperRanker(cache_dir=cache_dir)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI components."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Create header
        header_layout = QHBoxLayout()
        
        title = QLabel("Paper Ranking")
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
        self.data_source_combo.setMinimumWidth(350)  # Make it wider for better readability
        self.data_source_combo.currentIndexChanged.connect(self.load_selected_dataset)
        header_layout.addWidget(self.data_source_combo)
        
        main_layout.addLayout(header_layout)
        
        # Ranking controls
        controls_group = QGroupBox("Ranking Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        # Query / hypothesis to use for ranking
        query_form = QFormLayout()
        self.query_input = QTextEdit()
        self.query_input.setPlaceholderText("Enter research question or hypothesis...")
        self.query_input.setMaximumHeight(80)
        
        query_input_layout = QHBoxLayout()
        query_input_layout.addWidget(self.query_input)
        
        self.ai_assist_btn = QPushButton("AI Assist")
        self.ai_assist_btn.setIcon(load_bootstrap_icon("magic"))
        self.ai_assist_btn.setToolTip("Use AI to analyze your query and suggest optimal ranking parameters")
        self.ai_assist_btn.clicked.connect(self.get_ai_ranking_assistance)
        query_input_layout.addWidget(self.ai_assist_btn)
        
        query_form.addRow("Rank papers based on:", query_input_layout)
        
        # Option to pull from hypotheses
        hypothesis_layout = QHBoxLayout()
        hypothesis_label = QLabel("Or use hypothesis:")
        self.hypothesis_combo = QComboBox()
        self.hypothesis_combo.setMinimumWidth(300)
        self.hypothesis_combo.currentIndexChanged.connect(self.on_hypothesis_selected)
        
        hypothesis_layout.addWidget(hypothesis_label)
        hypothesis_layout.addWidget(self.hypothesis_combo)
        hypothesis_layout.addStretch()
        
        query_form.addRow("", hypothesis_layout)
        controls_layout.addLayout(query_form)
        
        # Add ranking method selection
        ranking_method_layout = QHBoxLayout()
        self.use_ai_ranking = QRadioButton("Use AI ranking (requires internet)")
        self.use_local_ranking = QRadioButton("Use local ranking (faster, offline)")
        self.use_local_ranking.setChecked(True)  # Default to local ranking for speed
        
        # Add advanced options toggle
        self.show_advanced_options = QCheckBox("Show advanced ranking options")
        self.show_advanced_options.stateChanged.connect(self.toggle_advanced_options)
        
        ranking_method_layout.addWidget(QLabel("Ranking method:"))
        ranking_method_layout.addWidget(self.use_local_ranking)
        ranking_method_layout.addWidget(self.use_ai_ranking)
        ranking_method_layout.addWidget(self.show_advanced_options)
        ranking_method_layout.addStretch()
        
        controls_layout.addLayout(ranking_method_layout)
        
        # Advanced ranking options (initially hidden)
        self.advanced_options_group = QGroupBox("Advanced Ranking Options")
        self.advanced_options_group.setVisible(False)
        advanced_layout = QFormLayout(self.advanced_options_group)
        
        # Ranking weights sliders
        self.keyword_weight = QSpinBox()
        self.keyword_weight.setRange(0, 100)
        self.keyword_weight.setValue(20)
        self.keyword_weight.setSuffix("%")
        
        self.tfidf_weight = QSpinBox()
        self.tfidf_weight.setRange(0, 100)
        self.tfidf_weight.setValue(35)
        self.tfidf_weight.setSuffix("%")
        
        self.bm25_weight = QSpinBox()
        self.bm25_weight.setRange(0, 100)
        self.bm25_weight.setValue(35)
        self.bm25_weight.setSuffix("%")
        
        self.recency_weight = QSpinBox()
        self.recency_weight.setRange(0, 100)
        self.recency_weight.setValue(10)
        self.recency_weight.setSuffix("%")
        
        # Connect signals to ensure weights sum to 100%
        self.keyword_weight.valueChanged.connect(self.balance_weights)
        self.tfidf_weight.valueChanged.connect(self.balance_weights)
        self.bm25_weight.valueChanged.connect(self.balance_weights)
        self.recency_weight.valueChanged.connect(self.balance_weights)
        
        advanced_layout.addRow("Keyword matching weight:", self.keyword_weight)
        advanced_layout.addRow("TF-IDF similarity weight:", self.tfidf_weight)
        advanced_layout.addRow("BM25 relevance weight:", self.bm25_weight)
        advanced_layout.addRow("Recency weight:", self.recency_weight)
        
        # Add cache control
        cache_layout = QHBoxLayout()
        self.clear_cache_btn = QPushButton("Clear Ranking Cache")
        self.clear_cache_btn.clicked.connect(self.clear_ranking_cache)
        cache_layout.addWidget(self.clear_cache_btn)
        cache_layout.addStretch()
        
        advanced_layout.addRow("Cache management:", cache_layout)
        
        # Add a status message for AI assistance
        self.ai_assist_status = QLabel("")
        self.ai_assist_status.setVisible(False)
        self.ai_assist_status.setStyleSheet("color: blue; font-style: italic;")
        advanced_layout.addRow("", self.ai_assist_status)
        
        controls_layout.addWidget(self.advanced_options_group)
        
        # Ranking options
        options_layout = QHBoxLayout()
        
        # Number of papers to rank
        self.papers_to_rank = QSpinBox()
        self.papers_to_rank.setRange(10, 1000)
        self.papers_to_rank.setValue(50)
        self.papers_to_rank.setSingleStep(10)
        
        # Show context checkbox
        self.show_context = QCheckBox("Include rationales for rankings")
        self.show_context.setChecked(True)
        
        options_layout.addWidget(QLabel("Papers to rank:"))
        options_layout.addWidget(self.papers_to_rank)
        options_layout.addStretch()
        options_layout.addWidget(self.show_context)
        
        controls_layout.addLayout(options_layout)
        
        # Rank button
        button_layout = QHBoxLayout()
        
        self.rank_button = QPushButton("Rank Papers")
        self.rank_button.setIcon(load_bootstrap_icon("sort-numeric-down"))
        self.rank_button.clicked.connect(self.rank_papers)
        
        button_layout.addStretch()
        button_layout.addWidget(self.rank_button)
        button_layout.addStretch()
        
        controls_layout.addLayout(button_layout)
        
        main_layout.addWidget(controls_group)
        
        # Results area - will show ranked papers
        results_group = QGroupBox("Ranking Results")
        results_layout = QVBoxLayout(results_group)
        
        # Status and progress
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        
        results_layout.addLayout(status_layout)
        
        # Scrollable area for ranked papers
        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_scroll.setWidget(self.results_container)
        
        results_layout.addWidget(self.results_scroll)
        
        # Action buttons for results
        action_layout = QHBoxLayout()
        
        self.export_button = QPushButton("Export Rankings")
        self.export_button.setIcon(load_bootstrap_icon("download"))
        self.export_button.clicked.connect(self.export_rankings)
        self.export_button.setEnabled(False)
        
        self.save_to_study_button = QPushButton("Save to Study")
        self.save_to_study_button.setIcon(load_bootstrap_icon("save"))
        self.save_to_study_button.clicked.connect(self.save_to_study)
        self.save_to_study_button.setEnabled(False)
        
        self.manage_rankings_btn = QPushButton("Manage Saved Rankings")
        self.manage_rankings_btn.setIcon(load_bootstrap_icon("folder"))
        self.manage_rankings_btn.clicked.connect(self.manage_saved_rankings)
        
        self.save_ranking_btn = QPushButton("Save Ranking")
        self.save_ranking_btn.setIcon(load_bootstrap_icon("save"))
        self.save_ranking_btn.clicked.connect(self.save_current_ranking)
        self.save_ranking_btn.setEnabled(False)
        
        action_layout.addWidget(self.export_button)
        action_layout.addWidget(self.save_to_study_button)
        action_layout.addWidget(self.manage_rankings_btn)
        action_layout.addWidget(self.save_ranking_btn)
        
        results_layout.addLayout(action_layout)
        
        main_layout.addWidget(results_group)
    
    def set_studies_manager(self, studies_manager: StudiesManager):
        """Set the studies manager for accessing saved datasets."""
        self.studies_manager = studies_manager
        self.refresh_data_sources()
        self.refresh_hypotheses()
        
        # Register a method to refresh data sources when the active study changes
        if hasattr(studies_manager, 'activeStudyChanged'):
            studies_manager.activeStudyChanged.connect(self.refresh_data_sources)
    
    def refresh_data_sources(self):
        """Refresh the list of available datasets from studies manager."""
        if not self.studies_manager:
            return
        
        # Clear existing items
        self.data_source_combo.clear()
        
        # Get active study
        active_study = self.studies_manager.get_active_study()
        if not active_study:
            self.data_source_combo.addItem("No active study")
            return
        
        added_items = 0  # Track if we've added any items
        
        # First add any saved literature searches
        if hasattr(active_study, 'literature_searches') and active_study.literature_searches:
            for search in active_study.literature_searches:
                search_id = search.get('id')
                description = search.get('description', 'Untitled search')
                paper_count = search.get('paper_count', 0)
                timestamp = search.get('timestamp', '')
                if timestamp:
                    try:
                        date_str = datetime.fromisoformat(timestamp).strftime('%Y-%m-%d')
                        item_text = f"Search: {description} ({paper_count} papers, {date_str})"
                    except:
                        item_text = f"Search: {description} ({paper_count} papers)"
                else:
                    item_text = f"Search: {description} ({paper_count} papers)"
                
                # Add to combo box with search_id as user data
                self.data_source_combo.addItem(item_text, search_id)
                added_items += 1
        
        # Get available datasets
        datasets = self.studies_manager.get_datasets_from_active_study()
        
        # Then add standard datasets
        if datasets:
            # Add search result datasets
            for name, df in datasets:
                if isinstance(name, str) and (
                    name.startswith("search_results_") or 
                    name.startswith("papers_") or
                    name.startswith("ranked_papers_") or 
                    name.startswith("imported_")):
                    # Only add datasets that actually contain papers
                    if not df.empty and 'title' in df.columns:
                        if added_items > 0:
                            # Add a separator if we added literature searches before
                            self.data_source_combo.insertSeparator(self.data_source_combo.count())
                            added_items = 0  # Reset to ensure we only add separator once
                        
                        # Add a more descriptive name if possible
                        if hasattr(active_study, 'datasets_metadata') and name in active_study.datasets_metadata:
                            metadata = active_study.datasets_metadata.get(name, {})
                            description = metadata.get('description', name)
                            timestamp = metadata.get('timestamp', '')
                            if timestamp:
                                try:
                                    date_str = datetime.fromisoformat(timestamp).strftime('%Y-%m-%d')
                                    item_text = f"{description} ({len(df)} papers, {date_str})"
                                except:
                                    item_text = f"{description} ({len(df)} papers)"
                            else:
                                item_text = f"{description} ({len(df)} papers)"
                            
                            self.data_source_combo.addItem(item_text, name)
                        else:
                            # Just use the dataset name
                            self.data_source_combo.addItem(f"{name} ({len(df)} papers)", name)
                        
                        added_items += 1
        
        # Add empty option if nothing found
        if self.data_source_combo.count() == 0:
            self.data_source_combo.addItem("No paper datasets available")
    
    def refresh_hypotheses(self):
        """Refresh the list of available hypotheses."""
        if not self.studies_manager:
            return
            
        # Clear existing items
        self.hypothesis_combo.clear()
        
        # Add empty option
        self.hypothesis_combo.addItem("Select a hypothesis...", None)
        
        # Get active study
        active_study = self.studies_manager.get_active_study()
        if not active_study or not hasattr(active_study, 'hypotheses'):
            return
            
        # Get hypotheses
        hypotheses = active_study.hypotheses
        if not hypotheses:
            return
            
        # Add hypotheses to combo box
        for hypothesis in hypotheses:
            title = hypothesis.get('title', 'Untitled Hypothesis')
            self.hypothesis_combo.addItem(title, hypothesis)
    
    def on_hypothesis_selected(self, index):
        """Handle hypothesis selection with AI assistance option."""
        if index <= 0:
            return
            
        # Get selected hypothesis
        hypothesis = self.hypothesis_combo.itemData(index)
        if not hypothesis:
            return
            
        # Create query from hypothesis
        null_hypothesis = hypothesis.get('null_hypothesis', '')
        alt_hypothesis = hypothesis.get('alternative_hypothesis', '')
        outcome_vars = hypothesis.get('outcome_variables', '')
        predictor_vars = hypothesis.get('predictor_variables', '')
        
        query = f"Research Question: {hypothesis.get('title', '')}\n\n"
        if alt_hypothesis:
            query += f"Alternative Hypothesis: {alt_hypothesis}\n"
        if null_hypothesis:
            query += f"Null Hypothesis: {null_hypothesis}\n"
        if outcome_vars:
            query += f"Outcome Variables: {outcome_vars}\n"
        if predictor_vars:
            query += f"Predictor Variables: {predictor_vars}\n"
        
        # Set query text
        self.query_input.setPlainText(query)
        
        # Ask user if they want AI assistance with this hypothesis
        reply = QMessageBox.question(
            self,
            "AI Assistance",
            "Would you like to use AI to optimize ranking parameters for this hypothesis?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Trigger AI assistance
            self.get_ai_ranking_assistance()
    
    def load_selected_dataset(self):
        """Load the selected dataset of papers."""
        if not self.studies_manager:
            return
        
        # Check for no-data messages
        selected_text = self.data_source_combo.currentText()
        if selected_text in ["No active study", "No paper datasets available"]:
            self.papers = []
            self.status_label.setText("No papers available")
            return
        
        # Get the selected user data
        selected_data = self.data_source_combo.currentData()
        
        try:
            # Get active study
            active_study = self.studies_manager.get_active_study()
            if not active_study:
                self.status_label.setText("No active study")
                return
            
            # If we have a search_id, load from saved searches
            if isinstance(selected_data, str):
                # Check if it looks like a UUID for a literature search
                if len(selected_data) > 30 and '-' in selected_data:
                    # Try to find in literature searches
                    if hasattr(active_study, 'literature_searches') and active_study.literature_searches:
                        for search in active_study.literature_searches:
                            if search.get('id') == selected_data:
                                self.papers = search.get('papers', [])
                                
                                # Update query input if available
                                if 'query' in search:
                                    self.query_input.setPlainText(f"Search query: {search['query']}")
                                
                                # Update status
                                self.status_label.setText(f"Loaded {len(self.papers)} papers from saved search")
                                return
                
                # Must be a dataset name
                dataset_name = selected_data
                
                # Get the dataset
                datasets = self.studies_manager.get_datasets_from_active_study()
                for name, df in datasets:
                    if name == dataset_name:
                        # Convert DataFrame to list of dictionaries
                        self.papers = df.to_dict('records')
                        
                        # Update status
                        self.status_label.setText(f"Loaded {len(self.papers)} papers from dataset '{name}'")
                        return
            
            # If we get here, try to find by display text
            if "Search: " in selected_text:
                # This is a search entry, extract the description
                description = selected_text.split("Search: ")[1].split(" (")[0]
                
                # Find the matching search
                if hasattr(active_study, 'literature_searches') and active_study.literature_searches:
                    for search in active_study.literature_searches:
                        if search.get('description') == description:
                            self.papers = search.get('papers', [])
                            
                            # Update query input if available
                            if 'query' in search:
                                self.query_input.setPlainText(f"Search query: {search['query']}")
                            
                            # Update status
                            self.status_label.setText(f"Loaded {len(self.papers)} papers from saved search")
                            return
            
            # If we get here, we couldn't find the dataset
            self.papers = []
            self.status_label.setText("Could not find selected paper dataset")
            
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
            self.papers = []
    
    def toggle_advanced_options(self, state):
        """Show or hide advanced ranking options."""
        self.advanced_options_group.setVisible(state)
    
    def balance_weights(self):
        """Ensure weights sum to 100%."""
        sender = self.sender()
        total = (self.keyword_weight.value() + self.tfidf_weight.value() + 
                self.bm25_weight.value() + self.recency_weight.value())
        
        # If total is over 100%, adjust other weights down proportionally
        if total > 100:
            excess = total - 100
            # Don't adjust the weight that was just changed
            if sender != self.keyword_weight:
                current = self.keyword_weight.value()
                proportion = current / (total - sender.value())
                self.keyword_weight.blockSignals(True)
                self.keyword_weight.setValue(max(0, current - int(excess * proportion)))
                self.keyword_weight.blockSignals(False)
            
            if sender != self.tfidf_weight:
                current = self.tfidf_weight.value()
                proportion = current / (total - sender.value())
                self.tfidf_weight.blockSignals(True)
                self.tfidf_weight.setValue(max(0, current - int(excess * proportion)))
                self.tfidf_weight.blockSignals(False)
            
            if sender != self.bm25_weight:
                current = self.bm25_weight.value()
                proportion = current / (total - sender.value())
                self.bm25_weight.blockSignals(True)
                self.bm25_weight.setValue(max(0, current - int(excess * proportion)))
                self.bm25_weight.blockSignals(False)
            
            if sender != self.recency_weight:
                current = self.recency_weight.value()
                proportion = current / (total - sender.value())
                self.recency_weight.blockSignals(True)
                self.recency_weight.setValue(max(0, current - int(excess * proportion)))
                self.recency_weight.blockSignals(False)
    
    def clear_ranking_cache(self):
        """Clear the paper ranker cache."""
        try:
            reply = QMessageBox.question(
                self, 
                "Clear Cache", 
                "Do you want to clear all cached ranking data?\nThis will free up disk space but may slow down future ranking operations.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.paper_ranker.clear_cache(session_only=False)
                QMessageBox.information(self, "Cache Cleared", "Ranking cache has been cleared.")
        except Exception as e:
            logging.error(f"Error clearing cache: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to clear cache: {str(e)}")
    
    @asyncSlot()
    async def rank_papers(self):
        """Rank papers based on relevance to query."""
        if not self.papers:
            self.status_label.setText("No papers to rank")
            return
            
        query = self.query_input.toPlainText().strip()
        if not query:
            self.status_label.setText("Please enter a research question or hypothesis")
            return
            
        try:
            self.status_label.setText("Ranking papers...")
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            self.rank_button.setEnabled(False)
            QApplication.processEvents()  # Update UI
            
            # Get parameters
            num_papers = min(self.papers_to_rank.value(), len(self.papers))
            include_rationales = self.show_context.isChecked()
            
            # Prepare papers for ranking
            papers_to_rank = self.papers[:num_papers]
            
            # Progress update
            self.progress_bar.setValue(10)
            QApplication.processEvents()
            
            # Use local or API-based ranking based on selection
            if hasattr(self, 'use_local_ranking') and self.use_local_ranking.isChecked():
                # Update ranking weights if advanced options are used
                if self.show_advanced_options.isChecked():
                    # Update weights in the ranker
                    weights = {
                        'keyword': self.keyword_weight.value() / 100.0,
                        'tfidf': self.tfidf_weight.value() / 100.0,
                        'bm25': self.bm25_weight.value() / 100.0,
                        'recency': self.recency_weight.value() / 100.0
                    }
                    self.paper_ranker.set_weights(weights)
                
                self.progress_bar.setValue(20)
                QApplication.processEvents()
                
                # Prepare paper data for ranker
                paper_data = []
                for paper in papers_to_rank:
                    paper_info = {
                        "id": paper.get("id", ""),
                        "title": paper.get("title", ""),
                        "abstract": paper.get("abstract", ""),
                        "year": paper.get("year", None),
                        "date": paper.get("date", None)
                    }
                    paper_data.append(paper_info)
                
                self.progress_bar.setValue(40)
                QApplication.processEvents()
                
                # Rank papers
                ranking_results = self.paper_ranker.rank_papers(
                    query,
                    paper_data
                )
                
                self.progress_bar.setValue(80)
                QApplication.processEvents()
            else:
                # Use the API-based ranking
                paper_data = []
                for paper in papers_to_rank:
                    paper_info = {
                        "title": paper.get("title", ""),
                        "abstract": paper.get("abstract", "")
                    }
                    paper_data.append(paper_info)
                    
                # Call the ranking model
                ranking_results = await rate_papers_with_gemini(
                    query, 
                    paper_data,
                    include_rationales=include_rationales
                )
            
            self.progress_bar.setValue(90)
            QApplication.processEvents()
            
            # Process results
            ranked_papers = []
            for idx, result in enumerate(ranking_results):
                # Get original paper
                if idx < len(papers_to_rank):
                    paper = papers_to_rank[idx]
                    
                    # Create new dict with ranking data
                    ranked_paper = paper.copy()
                    ranked_paper["relevance_score"] = result.get("score", 0)
                    ranked_paper["relevance_rationale"] = result.get("rationale", "")
                    ranked_papers.append(ranked_paper)
            
            # Sort by relevance score
            ranked_papers.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            # Store results
            self.ranked_papers = ranked_papers
            
            # Display results
            self.display_ranked_papers(ranked_papers)
            
            # Update UI
            self.export_button.setEnabled(True)
            self.save_to_study_button.setEnabled(True)
            self.save_ranking_btn.setEnabled(True)
            self.status_label.setText(f"Ranked {len(ranked_papers)} papers")
            
        except Exception as e:
            logging.error(f"Error ranking papers: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            self.rank_button.setEnabled(True)
    
    def display_ranked_papers(self, ranked_papers):
        """Display the ranked papers in the UI."""
        # Clear existing results
        self.clear_layout(self.results_layout)
        
        if not ranked_papers:
            no_results = QLabel("No ranking results available")
            no_results.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_layout.addWidget(no_results)
            return
        
        # Add each paper with its ranking
        for i, paper in enumerate(ranked_papers):
            # Create paper card
            paper_widget = QGroupBox()
            paper_layout = QVBoxLayout(paper_widget)
            
            # Rank and score header
            header_layout = QHBoxLayout()
            
            rank_label = QLabel(f"#{i+1}")
            rank_font = QFont()
            rank_font.setPointSize(14)
            rank_font.setBold(True)
            rank_label.setFont(rank_font)
            
            score = paper.get("relevance_score", 0)
            score_label = QLabel(f"Score: {score:.1f}/10")
            score_color = self.get_score_color(score)
            score_label.setStyleSheet(f"color: {score_color}")
            
            header_layout.addWidget(rank_label)
            header_layout.addStretch()
            header_layout.addWidget(score_label)
            
            paper_layout.addLayout(header_layout)
            
            # Title
            title = paper.get("title", "Untitled")
            title_label = QLabel(title)
            title_font = QFont()
            title_font.setBold(True)
            title_label.setFont(title_font)
            title_label.setWordWrap(True)
            paper_layout.addWidget(title_label)
            
            # Authors
            authors = paper.get("authors", [])
            if authors:
                if isinstance(authors, list):
                    authors_text = ", ".join(authors[:3])
                    if len(authors) > 3:
                        authors_text += f" + {len(authors) - 3} more"
                else:
                    authors_text = str(authors)
                
                authors_label = QLabel(f"By: {authors_text}")
                authors_label.setWordWrap(True)
                paper_layout.addWidget(authors_label)
            
            # Rationale (if available)
            rationale = paper.get("relevance_rationale", "")
            if rationale:
                rationale_group = QGroupBox("Relevance Rationale")
                rationale_layout = QVBoxLayout(rationale_group)
                
                rationale_text = QTextEdit()
                rationale_text.setPlainText(rationale)
                rationale_text.setReadOnly(True)
                rationale_text.setMaximumHeight(80)
                
                rationale_layout.addWidget(rationale_text)
                paper_layout.addWidget(rationale_group)
            
            # Add to results layout
            self.results_layout.addWidget(paper_widget)
        
        # Add stretch at the end to push cards to the top
        self.results_layout.addStretch()
    
    def clear_layout(self, layout):
        """Clear all widgets from a layout."""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            elif item.layout() is not None:
                self.clear_layout(item.layout())
    
    def get_score_color(self, score):
        """Get an appropriate color for a score value."""
        if score >= 8:
            return "#006400"  # Dark green
        elif score >= 6:
            return "#008000"  # Green
        elif score >= 4:
            return "#FFA500"  # Orange
        elif score >= 2:
            return "#FF4500"  # OrangeRed
        else:
            return "#FF0000"  # Red
    
    def export_rankings(self):
        """Export ranking results to CSV."""
        if not self.ranked_papers:
            return
            
        try:
            # Convert to DataFrame for easy CSV export
            df = pd.DataFrame(self.ranked_papers)
            
            # Let user select file location
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Rankings",
                f"paper_rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "CSV Files (*.csv)"
            )
            
            if not file_path:
                return
                
            # Export to CSV
            df.to_csv(file_path, index=False)
            
            self.status_label.setText(f"Exported rankings to {file_path}")
            
        except Exception as e:
            logging.error(f"Error exporting rankings: {str(e)}")
            self.status_label.setText(f"Error exporting: {str(e)}")
    
    def save_to_study(self):
        """Save ranked papers to the active study."""
        if not self.ranked_papers or not self.studies_manager:
            return
            
        try:
            # Get active study
            active_study = self.studies_manager.get_active_study()
            if not active_study:
                self.status_label.setText("No active study")
                return
                
            # Convert to DataFrame
            df = pd.DataFrame(self.ranked_papers)
            
            # Generate dataset name
            dataset_name = f"ranked_papers_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save to study
            self.studies_manager.add_dataset_to_active_study(
                dataset_name,
                df,
                metadata={
                    'query': self.query_input.toPlainText(),
                    'original_dataset': self.data_source_combo.currentText(),
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # Update status
            self.status_label.setText(f"Saved rankings to study as '{dataset_name}'")
            
            # Get the query text to pass along with the papers
            query_text = self.query_input.toPlainText()
            
            # Emit signal about ranked papers
            self.papersRanked.emit(self.ranked_papers)
            
            # Refresh data sources
            self.refresh_data_sources()
            
        except Exception as e:
            logging.error(f"Error saving to study: {str(e)}")
            self.status_label.setText(f"Error saving: {str(e)}")

    def set_papers(self, papers):
        """Set papers from search results."""
        if not papers:
            return
        
        # Store the papers
        self.papers = papers
        
        # Update the UI to show papers are available
        self.status_label.setText(f"Received {len(papers)} papers from search")
        
        # Update data source dropdown - just set text, don't add a new item
        self.data_source_combo.setItemText(0, f"Current search results ({len(papers)} papers)")
        self.data_source_combo.setCurrentIndex(0)
        
        # If we have a studies manager and active study, we don't need to save again
        # since the LiteratureSearchSection already does this
        
        # Set a default query based on the papers
        if not self.query_input.toPlainText():
            common_words = self.extract_common_topics(papers)
            if common_words:
                self.query_input.setPlainText(
                    f"Research question: What is the current evidence regarding {', '.join(common_words)}?"
                )

    def extract_common_topics(self, papers, max_topics=5):
        """Extract common topics from paper titles and abstracts."""
        # Simple implementation - just extracts common words from titles
        from collections import Counter
        import re
        
        # Common stop words to filter out
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'with', 'by', 'about', 'as', 'of', 'is', 'are', 'was', 'were', 'be',
                     'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'should', 'can', 'could', 'may', 'might', 'must', 'shall'}
        
        # Collect all words from titles
        all_words = []
        for paper in papers:
            title = paper.get('title', '')
            if isinstance(title, str):
                # Extract words, remove punctuation and convert to lowercase
                words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
                # Filter out stop words
                words = [word for word in words if word not in stop_words]
                all_words.extend(words)
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Get the most common words
        common_words = [word for word, count in word_counts.most_common(max_topics) if count > 1]
        
        return common_words

    def save_current_ranking(self):
        """Save current ranking results to the active study."""
        if not self.ranked_papers:
            QMessageBox.warning(self, "Warning", "No ranking results to save.")
            return
        
        if not self.studies_manager:
            QMessageBox.warning(self, "Warning", "No studies manager available.")
            return
        
        # Create an input dialog for the ranking description
        description, ok = QInputDialog.getText(
            self, 
            "Save Ranking", 
            "Enter a description for this ranking:",
            text=f"Ranking based on '{self.query_input.toPlainText()[:30]}...' ({len(self.ranked_papers)} papers)"
        )
        
        if not ok or not description:
            return  # User cancelled
        
        try:
            # Get active study
            active_study = self.studies_manager.get_active_study()
            if not active_study:
                QMessageBox.warning(self, "Warning", "No active study available.")
                return
            
            # Initialize rankings list if needed
            if not hasattr(active_study, 'paper_rankings'):
                active_study.paper_rankings = []
            
            # Create a ranking entry
            ranking_entry = {
                'id': str(uuid.uuid4()),
                'description': description,
                'query': self.query_input.toPlainText(),
                'timestamp': datetime.now().isoformat(),
                'papers': self.ranked_papers,
                'paper_count': len(self.ranked_papers),
                'source_dataset': self.data_source_combo.currentText()
            }
            
            # Add to rankings list
            active_study.paper_rankings.append(ranking_entry)
            
            # Update timestamp
            active_study.updated_at = datetime.now().isoformat()
            
            # Show success message
            QMessageBox.information(
                self, 
                "Success", 
                f"Saved ranking with {len(self.ranked_papers)} papers to the active study."
            )
            
            # Also save as a dataset for backward compatibility
            self.save_to_study()
            
        except Exception as e:
            logging.error(f"Error saving ranking: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to save ranking: {str(e)}")

    def manage_saved_rankings(self):
        """Show a dialog to manage saved rankings."""
        if not self.studies_manager:
            QMessageBox.warning(self, "Warning", "No studies manager available.")
            return
        
        # Get active study
        active_study = self.studies_manager.get_active_study()
        if not active_study:
            QMessageBox.warning(self, "Warning", "No active study available.")
            return
        
        # Check if the study has any saved rankings
        if not hasattr(active_study, 'paper_rankings') or not active_study.paper_rankings:
            QMessageBox.information(self, "Information", "No saved rankings found.")
            return
        
        # Create a dialog to display saved rankings
        dialog = QDialog(self)
        dialog.setWindowTitle("Saved Paper Rankings")
        dialog.setMinimumSize(800, 400)
        
        layout = QVBoxLayout(dialog)
        
        # Create a list widget to display rankings
        list_widget = QListWidget()
        for ranking in active_study.paper_rankings:
            timestamp = datetime.fromisoformat(ranking['timestamp']).strftime('%Y-%m-%d %H:%M')
            item = QListWidgetItem(f"{ranking['description']} - {ranking['paper_count']} papers - {timestamp}")
            item.setData(Qt.ItemDataRole.UserRole, ranking['id'])
            list_widget.addItem(item)
        
        layout.addWidget(list_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        load_btn = QPushButton("Load Selected")
        load_btn.clicked.connect(lambda: self.load_saved_ranking(list_widget) and dialog.accept())
        button_layout.addWidget(load_btn)
        
        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(lambda: self.delete_saved_ranking(list_widget) and list_widget.takeItem(list_widget.currentRow()))
        button_layout.addWidget(delete_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        # Show the dialog
        dialog.exec()

    def load_saved_ranking(self, list_widget):
        """Load a saved ranking selected from the list widget."""
        current_item = list_widget.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a ranking to load.")
            return False
        
        ranking_id = current_item.data(Qt.ItemDataRole.UserRole)
        
        # Get the active study
        active_study = self.studies_manager.get_active_study()
        if not active_study or not hasattr(active_study, 'paper_rankings'):
            return False
        
        # Find the ranking with the matching ID
        ranking_data = None
        for ranking in active_study.paper_rankings:
            if ranking['id'] == ranking_id:
                ranking_data = ranking
                break
        
        if not ranking_data:
            QMessageBox.warning(self, "Warning", "Failed to load ranking data.")
            return False
        
        # Load the ranking data
        self.ranked_papers = ranking_data['papers']
        
        # Display the results
        self.display_ranked_papers(ranking_data['papers'])
        
        # Update UI elements
        self.export_button.setEnabled(True)
        self.save_to_study_button.setEnabled(True)
        
        # Set the query text
        if 'query' in ranking_data:
            self.query_input.setPlainText(ranking_data['query'])
        
        # Update status
        self.status_label.setText(f"Loaded ranking with {len(self.ranked_papers)} papers")
        
        return True

    def delete_saved_ranking(self, list_widget):
        """Delete a saved ranking selected from the list widget."""
        current_item = list_widget.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a ranking to delete.")
            return False
        
        ranking_id = current_item.data(Qt.ItemDataRole.UserRole)
        
        # Confirm deletion
        reply = QMessageBox.question(
            self, 
            "Confirm Deletion", 
            "Are you sure you want to delete this saved ranking?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return False
        
        # Get the active study
        active_study = self.studies_manager.get_active_study()
        if not active_study or not hasattr(active_study, 'paper_rankings'):
            return False
        
        # Find and remove the ranking
        for i, ranking in enumerate(active_study.paper_rankings):
            if ranking['id'] == ranking_id:
                active_study.paper_rankings.pop(i)
                
                # Update study timestamp
                active_study.updated_at = datetime.now().isoformat()
                
                return True
        
        QMessageBox.warning(self, "Warning", "Failed to delete ranking.")
        return False

    @asyncSlot()
    async def get_ai_ranking_assistance(self):
        """Use LLM to analyze query and suggest optimal ranking parameters."""
        query = self.query_input.toPlainText().strip()
        if not query:
            QMessageBox.warning(self, "Warning", "Please enter a research question or hypothesis first.")
            return
        
        try:
            # Show progress indication
            self.ai_assist_status.setText("Analyzing your research question...")
            self.ai_assist_status.setVisible(True)
            self.ai_assist_btn.setEnabled(False)
            QApplication.processEvents()
            
            # Prepare the prompt for the LLM
            prompt = f"""
            I'm trying to rank academic papers based on their relevance to a research question. 
            Given the following research question or hypothesis, please:
            
            1. Identify 5-10 key terms/concepts that are most important to search for
            2. Suggest the optimal weights (as percentages) for these ranking factors:
               - keyword matching (finding papers with the exact terms)
               - semantic similarity (TF-IDF based)
               - BM25 relevance (sophisticated term-document relevance)
               - recency (newer papers get higher scores)
            3. Explain briefly why you chose these weights for this specific research question
            
            Research question: "{query}"
            
            Format your response as a JSON object with these keys:
            - key_terms (array of strings)
            - weights (object with percentage values for keyword, tfidf, bm25, recency)
            - explanation (string explaining your reasoning)
            """
            
            # Call the LLM
            response = await call_llm_async(prompt)
            
            # Parse the response
            import json
            import re
            
            # Try to extract JSON from the response
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without markdown code blocks
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("Could not extract JSON from LLM response")
            
            data = json.loads(json_str)
            
            # Apply the suggestions
            if 'weights' in data and isinstance(data['weights'], dict):
                weights = data['weights']
                # Enable advanced options
                self.show_advanced_options.setChecked(True)
                
                # Set weights
                if 'keyword' in weights:
                    self.keyword_weight.setValue(int(weights['keyword']))
                if 'tfidf' in weights:
                    self.tfidf_weight.setValue(int(weights['tfidf']))
                if 'bm25' in weights:
                    self.bm25_weight.setValue(int(weights['bm25']))
                if 'recency' in weights:
                    self.recency_weight.setValue(int(weights['recency']))
            
            # Display key terms and explanation
            explanation = data.get('explanation', '')
            key_terms = data.get('key_terms', [])
            
            self.ai_assist_status.setText(f"AI analysis complete: Optimized for {', '.join(key_terms[:3])}" + 
                                         (", ..." if len(key_terms) > 3 else ""))
            
            # Show the full explanation
            QMessageBox.information(
                self,
                "AI Ranking Assistance",
                f"Key terms identified: {', '.join(key_terms)}\n\n"
                f"Recommended weights:\n"
                f"• Keyword matching: {weights.get('keyword', '?')}%\n"
                f"• Semantic similarity: {weights.get('tfidf', '?')}%\n"
                f"• BM25 relevance: {weights.get('bm25', '?')}%\n"
                f"• Recency: {weights.get('recency', '?')}%\n\n"
                f"Explanation: {explanation}"
            )
            
            # Update the paper ranker with these weights
            if hasattr(self, 'paper_ranker'):
                self.paper_ranker.set_weights({
                    'keyword': weights.get('keyword', 20) / 100.0,
                    'tfidf': weights.get('tfidf', 35) / 100.0,
                    'bm25': weights.get('bm25', 35) / 100.0,
                    'recency': weights.get('recency', 10) / 100.0
                })
            
            # Apply key terms to the paper ranker
            if 'key_terms' in data and isinstance(data['key_terms'], list):
                self.apply_ai_key_terms(data['key_terms'])
            
        except Exception as e:
            logging.error(f"Error getting AI ranking assistance: {str(e)}")
            self.ai_assist_status.setText("Error analyzing research question")
            QMessageBox.warning(self, "Error", f"Failed to get AI assistance: {str(e)}")
        finally:
            self.ai_assist_btn.setEnabled(True)

    def apply_ai_key_terms(self, key_terms):
        """Apply AI-identified key terms to the paper ranker."""
        if hasattr(self, 'paper_ranker') and key_terms:
            self.paper_ranker.set_key_terms(key_terms)
            return True
        return False