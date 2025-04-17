import asyncio
from typing import Dict, List, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QTabWidget, QFileDialog, QMessageBox, QProgressBar, QScrollArea,
    QGroupBox, QGridLayout, QSplitter, QButtonGroup, QRadioButton,
    QFrame, QDialog, QSpinBox, QCheckBox, QFormLayout, QComboBox, QTextEdit, QApplication,
    QDialogButtonBox, QToolButton, QToolBar, QListWidget, QListWidgetItem, QMenu, QInputDialog
)
from PyQt6.QtCore import pyqtSignal, Qt, QUrl, QSize
from PyQt6.QtGui import QFont, QIcon, QDesktopServices, QAction
import logging
import matplotlib
import pandas as pd

from helpers.load_icon import load_bootstrap_icon
from qt_sections.search import PaperWidget
from study_model.studies_manager import StudiesManager
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, YearLocator, MonthLocator
import matplotlib.cm as cm
from datetime import datetime
import numpy as np
from collections import Counter, defaultdict
import re

from qasync import asyncSlot
from literature_search.clustering import PaperClusterAnalyzer
from literature_search.doi_manager import DOIManager
from literature_search.pattern_extractor import QuoteExtractor
from literature_search.pubmed_client import PubMedClient
from literature_search.models import SearchResult
from literature_search.model_calls import rate_papers_with_gemini, generate_grounded_review
from literature_search.prompts import format_date
from admin.portal import secrets

class SearchModifiers:
    """Handles construction and validation of advanced search queries."""
    
    BOOLEAN_OPERATORS = {
        'AND': ' AND ',
        'OR': ' OR ',
        'NOT': ' NOT ',
        'NEAR': ' NEAR/',  # For proximity searches
        'ADJ': ' ADJ/'     # For adjacent terms
    }
    
    FIELD_TAGS = {
        'title': '[Title]',
        'abstract': '[Abstract]',
        'author': '[Author]',
        'journal': '[Journal]',
        'year': '[Year]',
        'mesh': '[MeSH Terms]',
        'affiliation': '[Affiliation]',
        'doi': '[DOI]',
        'pmid': '[PMID]',
        'language': '[Language]',
        'publication_type': '[Publication Type]'
    }
    
    @staticmethod
    def build_proximity_search(terms: List[str], distance: int, operator: str = 'NEAR') -> str:
        """Build proximity search query (e.g., "term1 NEAR/5 term2")."""
        if len(terms) < 2:
            raise ValueError("Proximity search requires at least 2 terms")
        return f"{operator}{distance}".join(terms)
    
    @staticmethod
    def build_phrase_search(phrase: str) -> str:
        """Wrap phrase in quotes for exact matching."""
        return f'"{phrase}"'
    
    @staticmethod
    def build_wildcard_search(term: str, wildcard: str = '*') -> str:
        """Add wildcard to term (e.g., "neuro*" for prefix search)."""
        return f"{term}{wildcard}"
    
    @staticmethod
    def add_field_tag(term: str, field: str) -> str:
        """Add field tag to search term."""
        if field in SearchModifiers.FIELD_TAGS:
            return f"{term}{SearchModifiers.FIELD_TAGS[field]}"
        raise ValueError(f"Unknown field tag: {field}")

class AdvancedSearchQuery:
    """Builds and manages complex search queries."""
    
    def __init__(self):
        self.query_parts = []
        self.modifiers = SearchModifiers()
        
    def add_term(self, term: str, field: Optional[str] = None) -> 'AdvancedSearchQuery':
        """Add a search term, optionally with a field specification."""
        if field:
            term = self.modifiers.add_field_tag(term, field)
        self.query_parts.append(term)
        return self
    
    def add_phrase(self, phrase: str, field: Optional[str] = None) -> 'AdvancedSearchQuery':
        """Add an exact phrase search."""
        phrase = self.modifiers.build_phrase_search(phrase)
        if field:
            phrase = self.modifiers.add_field_tag(phrase, field)
        self.query_parts.append(phrase)
        return self
    
    def add_proximity_search(self, terms: List[str], distance: int) -> 'AdvancedSearchQuery':
        """Add proximity search terms."""
        proximity_query = self.modifiers.build_proximity_search(terms, distance)
        self.query_parts.append(proximity_query)
        return self
    
    def add_boolean_operator(self, operator: str) -> 'AdvancedSearchQuery':
        """Add a boolean operator between terms."""
        if operator.upper() in self.modifiers.BOOLEAN_OPERATORS:
            self.query_parts.append(self.modifiers.BOOLEAN_OPERATORS[operator.upper()])
        return self
    
    def build(self) -> str:
        """Build the final search query string."""
        return ''.join(self.query_parts)

class PaperDetailDialog(QDialog):
    """
    A professionally laid out dialog for viewing detailed paper information.
    """
    
    def __init__(self, paper: Dict, parent=None):
        super().__init__(parent)
        self.paper = paper
        self.setup_ui()
        
    def setup_ui(self):
        # Set up the dialog size and title
        self.setWindowTitle("Paper Details")
        self.setMinimumSize(800, 600)
        self.resize(900, 700)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Create a tab widget for different views
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        
        # === Overview Tab ===
        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)
        overview_layout.setSpacing(15)
        overview_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title section with styling - sanitize HTML
        title = self._clean_html(self.paper.get('title', 'No Title'))
        title_label = QLabel(title)
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setWordWrap(True)
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        overview_layout.addWidget(title_label)
        
        # Add horizontal line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        overview_layout.addWidget(line)
        
        # Authors and journal - sanitize HTML
        authors = self.paper.get('authors', [])
        authors_text = ', '.join(self._clean_html(author) for author in authors) if authors else 'Unknown Authors'
        authors_label = QLabel(f"<b>Authors:</b> {authors_text}")
        authors_label.setWordWrap(True)
        overview_layout.addWidget(authors_label)
        
        journal = self._clean_html(self.paper.get('journal', ''))
        pub_date = self.paper.get('publication_date', '')
        pub_info = f"<b>Journal:</b> {journal}"
        if pub_date:
            pub_info += f" | <b>Published:</b> {pub_date}"
        pub_label = QLabel(pub_info)
        pub_label.setWordWrap(True)
        overview_layout.addWidget(pub_label)
        
        # DOI and PMID
        doi = self.paper.get('doi', '')
        pmid = self.paper.get('pmid', '')
        pmcid = self.paper.get('pmcid', '')
        
        ids_layout = QHBoxLayout()
        if doi:
            doi_label = QLabel(f"<b>DOI:</b> {doi}")
            doi_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | 
                                            Qt.TextInteractionFlag.TextSelectableByKeyboard)
            ids_layout.addWidget(doi_label)
            
            open_doi_btn = QToolButton()
            open_doi_btn.setIcon(load_bootstrap_icon("box-arrow-up-right"))
            open_doi_btn.setToolTip("Open DOI in browser")
            open_doi_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(f"https://doi.org/{doi}")))
            ids_layout.addWidget(open_doi_btn)
            
        ids_layout.addStretch()
        
        if pmid:
            pmid_label = QLabel(f"<b>PMID:</b> {pmid}")
            pmid_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | 
                                             Qt.TextInteractionFlag.TextSelectableByKeyboard)
            ids_layout.addWidget(pmid_label)
            
            open_pmid_btn = QToolButton()
            open_pmid_btn.setIcon(load_bootstrap_icon("box-arrow-up-right"))
            open_pmid_btn.setToolTip("Open PMID in PubMed")
            open_pmid_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")))
            ids_layout.addWidget(open_pmid_btn)
        
        overview_layout.addLayout(ids_layout)
        
        # Add horizontal line
        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setFrameShadow(QFrame.Shadow.Sunken)
        overview_layout.addWidget(line2)
        
        # Abstract section - sanitize HTML
        abstract_label = QLabel("<b>Abstract</b>")
        abstract_font = QFont()
        abstract_font.setPointSize(12)
        abstract_font.setBold(True)
        abstract_label.setFont(abstract_font)
        overview_layout.addWidget(abstract_label)
        
        abstract = self._clean_html(self.paper.get('abstract', 'No abstract available.'))
        abstract_text = QTextEdit()
        abstract_text.setReadOnly(True)
        abstract_text.setPlainText(abstract)
        abstract_text.setMinimumHeight(200)
        overview_layout.addWidget(abstract_text)
        
        # Add to tabs
        self.tab_widget.addTab(overview_tab, "Overview")
        
        # === Citation Tab ===
        citation_tab = QWidget()
        citation_layout = QVBoxLayout(citation_tab)
        citation_layout.setSpacing(15)
        citation_layout.setContentsMargins(15, 15, 15, 15)
        
        # Instructions
        citation_label = QLabel("Copy citations in different formats:")
        citation_layout.addWidget(citation_label)
        
        # APA Format
        apa_group = QGroupBox("APA Format")
        apa_layout = QVBoxLayout(apa_group)
        
        apa_citation = self._format_apa_citation()
        apa_text = QTextEdit()
        apa_text.setReadOnly(True)
        apa_text.setPlainText(apa_citation)
        apa_text.setMaximumHeight(100)
        apa_layout.addWidget(apa_text)
        
        citation_layout.addWidget(apa_group)
        
        # MLA Format
        mla_group = QGroupBox("MLA Format")
        mla_layout = QVBoxLayout(mla_group)
        
        mla_citation = self._format_mla_citation()
        mla_text = QTextEdit()
        mla_text.setReadOnly(True)
        mla_text.setPlainText(mla_citation)
        mla_text.setMaximumHeight(100)
        mla_layout.addWidget(mla_text)
        
        citation_layout.addWidget(mla_group)
        
        # Chicago Format
        chicago_group = QGroupBox("Chicago Format")
        chicago_layout = QVBoxLayout(chicago_group)
        
        chicago_citation = self._format_chicago_citation()
        chicago_text = QTextEdit()
        chicago_text.setReadOnly(True)
        chicago_text.setPlainText(chicago_citation)
        chicago_text.setMaximumHeight(100)
        chicago_layout.addWidget(chicago_text)
        
        citation_layout.addWidget(chicago_group)
        
        # BibTeX Format
        bibtex_group = QGroupBox("BibTeX")
        bibtex_layout = QVBoxLayout(bibtex_group)
        
        bibtex_citation = self._format_bibtex_citation()
        bibtex_text = QTextEdit()
        bibtex_text.setReadOnly(True)
        bibtex_text.setPlainText(bibtex_citation)
        bibtex_text.setMaximumHeight(120)
        bibtex_layout.addWidget(bibtex_text)
        
        citation_layout.addWidget(bibtex_group)
        
        # Add stretch to keep citation formats at the top
        citation_layout.addStretch()
        
        self.tab_widget.addTab(citation_tab, "Citation")
        
        # === Add tabs to main layout ===
        main_layout.addWidget(self.tab_widget)
               
        # Buttons at the bottom
        button_layout = QHBoxLayout()
        
        # Add button for viewing full text if available
        if pmcid:
            view_full_text_btn = QPushButton("View Full Text")
            view_full_text_btn.setIcon(load_bootstrap_icon("file-text"))
            view_full_text_btn.clicked.connect(lambda: QDesktopServices.openUrl(
                QUrl(f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/")
            ))
            button_layout.addWidget(view_full_text_btn)
        
        # Add button for getting paper via DOI
        if doi:
            open_doi_btn = QPushButton("Open in Browser")
            open_doi_btn.setIcon(load_bootstrap_icon("box-arrow-up-right"))
            open_doi_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(f"https://doi.org/{doi}")))
            button_layout.addWidget(open_doi_btn)
        
        # Add a button box with Close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        button_layout.addWidget(button_box)
        
        main_layout.addLayout(button_layout)
        
    def _clean_html(self, text):
        """Remove HTML tags and normalize whitespace"""
        if not text:
            return ""
        # Remove HTML tags
        clean_text = re.sub(r'<[^>]+>', '', text)
        # Normalize whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        # Remove any unbalanced braces
        clean_text = re.sub(r'[\{\}]', '', clean_text)
        return clean_text
    
    def _format_apa_citation(self) -> str:
        """Format citation in APA style"""
        authors = self.paper.get('authors', [])
        title = self.paper.get('title', 'No title')
        journal = self.paper.get('journal', 'Unknown journal')
        pub_date = self.paper.get('publication_date', '')
        doi = self.paper.get('doi', '')
        
        year = pub_date.split('-')[0] if pub_date and '-' in pub_date else pub_date
        
        if not authors:
            author_text = 'Unknown Author'
        elif len(authors) == 1:
            author_text = authors[0]
        elif len(authors) == 2:
            author_text = f"{authors[0]} & {authors[1]}"
        else:
            author_text = f"{authors[0]} et al."
        
        citation = f"{author_text}. ({year}). {title}. {journal}."
        if doi:
            citation += f" https://doi.org/{doi}"
            
        return citation
    
    def _format_mla_citation(self) -> str:
        """Format citation in MLA style"""
        authors = self.paper.get('authors', [])
        title = self.paper.get('title', 'No title')
        journal = self.paper.get('journal', 'Unknown journal')
        pub_date = self.paper.get('publication_date', '')
        doi = self.paper.get('doi', '')
        
        year = pub_date.split('-')[0] if pub_date and '-' in pub_date else pub_date
        
        if not authors:
            author_text = 'Unknown Author'
        elif len(authors) == 1:
            author_text = authors[0]
        elif len(authors) > 1:
            author_text = f"{authors[0]}, et al."
        
        citation = f"{author_text}. \"{title}.\" {journal}, {year}"
        if doi:
            citation += f", doi:{doi}"
            
        return citation
    
    def _format_chicago_citation(self) -> str:
        """Format citation in Chicago style"""
        authors = self.paper.get('authors', [])
        title = self.paper.get('title', 'No title')
        journal = self.paper.get('journal', 'Unknown journal')
        pub_date = self.paper.get('publication_date', '')
        doi = self.paper.get('doi', '')
        
        year = pub_date.split('-')[0] if pub_date and '-' in pub_date else pub_date
        
        if not authors:
            author_text = 'Unknown Author'
        elif len(authors) == 1:
            author_text = authors[0]
        elif len(authors) <= 3:
            author_text = ", ".join(authors[:-1]) + ", and " + authors[-1]
        else:
            author_text = f"{authors[0]} et al."
        
        citation = f"{author_text}. \"{title}.\" {journal} ({year})."
        if doi:
            citation += f" https://doi.org/{doi}."
            
        return citation
    
    def _format_bibtex_citation(self) -> str:
        """Format citation in BibTeX format"""
        authors = self.paper.get('authors', [])
        title = self.paper.get('title', 'No title')
        journal = self.paper.get('journal', 'Unknown journal')
        pub_date = self.paper.get('publication_date', '')
        doi = self.paper.get('doi', '')
        
        year = pub_date.split('-')[0] if pub_date and '-' in pub_date else pub_date
        
        # Create citation key
        if authors and year:
            last_name = authors[0].split()[-1] if ' ' in authors[0] else authors[0]
            citation_key = f"{last_name.lower()}{year}"
        else:
            citation_key = "unknown"
        
        author_text = " and ".join(authors) if authors else "Unknown Author"
        
        bibtex = f"@article{{{citation_key},\n"
        bibtex += f"  author = {{{author_text}}},\n"
        bibtex += f"  title = {{{title}}},\n"
        bibtex += f"  journal = {{{journal}}},\n"
        if year:
            bibtex += f"  year = {{{year}}},\n"
        if doi:
            bibtex += f"  doi = {{{doi}}},\n"
        bibtex += "}"
        
        return bibtex

    def add_metadata_tab(self):
        """Add a tab with additional metadata"""
        metadata_tab = QWidget()
        metadata_layout = QVBoxLayout(metadata_tab)
        metadata_layout.setSpacing(15)
        metadata_layout.setContentsMargins(15, 15, 15, 15)
        
        # Create a grid layout for metadata fields
        grid = QGridLayout()
        grid.setVerticalSpacing(10)
        grid.setHorizontalSpacing(20)
        
        # Display all metadata fields
        row = 0
        for key, value in sorted(self.paper.items()):
            # Skip fields already shown in the overview
            if key in ('title', 'abstract', 'authors', 'journal', 'doi', 'pmid', 'pmcid'):
                continue
                
            # Create label for the key
            key_label = QLabel(f"<b>{key.replace('_', ' ').title()}:</b>")
            
            # Format the value based on its type
            if isinstance(value, list):
                value_text = ", ".join(str(item) for item in value)
            elif value is None:
                value_text = "N/A"
            else:
                value_text = str(value)
                
            value_label = QLabel(value_text)
            value_label.setWordWrap(True)
            value_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse | 
                Qt.TextInteractionFlag.TextSelectableByKeyboard
            )
            
            grid.addWidget(key_label, row, 0)
            grid.addWidget(value_label, row, 1)
            row += 1
        
        metadata_layout.addLayout(grid)
        metadata_layout.addStretch()
        
        self.tab_widget.addTab(metadata_tab, "Metadata")

class PaginatedResultsWidget(QWidget):
    """
    A widget that displays paginated search results with paper viewing capability.
    """
    
    itemSelected = pyqtSignal(object)  # Signal when a paper is selected for viewing
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.papers = []
        self.current_page = 1
        self.papers_per_page = 10
        self.filtered_papers = []
        self.setup_ui()
        
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Results list area
        self.results_list = QListWidget()
        self.results_list.setSpacing(2)
        self.results_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.results_list.itemDoubleClicked.connect(self.on_item_double_clicked)
        
        # Add right-click context menu for paper items
        self.results_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.results_list.customContextMenuRequested.connect(self.show_context_menu)
        
        main_layout.addWidget(self.results_list)
        
        # Pagination controls
        pagination_layout = QHBoxLayout()
        pagination_layout.setSpacing(5)
        
        self.page_label = QLabel("Page 1 of 1")
        
        self.first_page_btn = QToolButton()
        self.first_page_btn.setIcon(load_bootstrap_icon("chevron-double-left"))
        self.first_page_btn.setToolTip("First Page")
        self.first_page_btn.clicked.connect(lambda: self.go_to_page(1))
        
        self.prev_page_btn = QToolButton()
        self.prev_page_btn.setIcon(load_bootstrap_icon("chevron-left"))
        self.prev_page_btn.setToolTip("Previous Page")
        self.prev_page_btn.clicked.connect(self.go_to_prev_page)
        
        self.next_page_btn = QToolButton()
        self.next_page_btn.setIcon(load_bootstrap_icon("chevron-right"))
        self.next_page_btn.setToolTip("Next Page")
        self.next_page_btn.clicked.connect(self.go_to_next_page)
        
        self.last_page_btn = QToolButton()
        self.last_page_btn.setIcon(load_bootstrap_icon("chevron-double-right"))
        self.last_page_btn.setToolTip("Last Page")
        self.last_page_btn.clicked.connect(lambda: self.go_to_page(self.total_pages))
        
        # Per page dropdown
        self.per_page_combo = QComboBox()
        self.per_page_combo.addItems(["10", "20", "50", "100"])
        self.per_page_combo.setCurrentText(str(self.papers_per_page))
        self.per_page_combo.currentTextChanged.connect(self.change_papers_per_page)
        
        per_page_label = QLabel("papers per page")
        
        pagination_layout.addWidget(self.first_page_btn)
        pagination_layout.addWidget(self.prev_page_btn)
        pagination_layout.addWidget(self.page_label)
        pagination_layout.addWidget(self.next_page_btn)
        pagination_layout.addWidget(self.last_page_btn)
        pagination_layout.addStretch()
        pagination_layout.addWidget(self.per_page_combo)
        pagination_layout.addWidget(per_page_label)
        
        main_layout.addLayout(pagination_layout)
        
        # Initialize empty state
        self.update_pagination_controls()
    
    def show_context_menu(self, position):
        """Show context menu for paper items"""
        item = self.results_list.itemAt(position)
        if not item:
            return
            
        idx = item.data(Qt.ItemDataRole.UserRole)
        if idx is None or idx < 0 or idx >= len(self.filtered_papers):
            return
            
        paper = self.filtered_papers[idx]
        
        # Create context menu
        menu = QMenu(self)
        
        view_action = QAction("View Paper Details", self)
        view_action.triggered.connect(lambda: self.itemSelected.emit(paper))
        menu.addAction(view_action)
        
        # Add DOI and PMID options if available
        if paper.get('doi'):
            doi_action = QAction("Open DOI in Browser", self)
            doi_action.triggered.connect(lambda: QDesktopServices.openUrl(
                QUrl(f"https://doi.org/{paper.get('doi')}")
            ))
            menu.addAction(doi_action)
            
        if paper.get('pmid'):
            pmid_action = QAction("Open in PubMed", self)
            pmid_action.triggered.connect(lambda: QDesktopServices.openUrl(
                QUrl(f"https://pubmed.ncbi.nlm.nih.gov/{paper.get('pmid')}/")
            ))
            menu.addAction(pmid_action)
            
        menu.exec(self.results_list.mapToGlobal(position))
    
    def set_papers(self, papers):
        """Set the papers to display and reset to first page"""
        self.papers = papers
        self.filtered_papers = papers.copy()
        self.current_page = 1
        self.display_current_page()
        self.update_pagination_controls()
    
    def filter_papers(self, filter_text):
        """Filter papers based on text and display first page"""
        if not filter_text:
            self.filtered_papers = self.papers.copy()
        else:
            filter_text = filter_text.lower()
            self.filtered_papers = []
            for paper in self.papers:
                # Check title (clean any HTML first)
                title = self._clean_html(paper.get('title', '')).lower()
                if filter_text in title:
                    self.filtered_papers.append(paper)
                    continue
                    
                # Check abstract (clean any HTML first)
                abstract = self._clean_html(paper.get('abstract', '')).lower()
                if filter_text in abstract:
                    self.filtered_papers.append(paper)
                    continue
                    
                # Check authors
                authors = paper.get('authors', [])
                cleaned_authors = [self._clean_html(author).lower() for author in authors]
                if any(filter_text in author for author in cleaned_authors):
                    self.filtered_papers.append(paper)
                    continue
                
                # Check journal
                journal = self._clean_html(paper.get('journal', '')).lower()
                if filter_text in journal:
                    self.filtered_papers.append(paper)
                    continue
        
        self.current_page = 1
        self.display_current_page()
        self.update_pagination_controls()
    
    @property
    def total_pages(self):
        """Calculate the total number of pages"""
        if not self.filtered_papers:
            return 1
        return (len(self.filtered_papers) + self.papers_per_page - 1) // self.papers_per_page
    
    def display_current_page(self):
        """Display the current page of papers"""
        self.results_list.clear()
        
        if not self.filtered_papers:
            item = QListWidgetItem("No results found. Try adjusting your search criteria.")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            font = item.font()
            font.setItalic(True)
            item.setFont(font)
            self.results_list.addItem(item)
            return
        
        start_idx = (self.current_page - 1) * self.papers_per_page
        end_idx = min(start_idx + self.papers_per_page, len(self.filtered_papers))
        
        for i in range(start_idx, end_idx):
            paper = self.filtered_papers[i]
            
            # Create a list item for the paper
            paper_item = QListWidgetItem()
            paper_item.setData(Qt.ItemDataRole.UserRole, i)  # Store the index for retrieval
            
            # Clean up the title - strip HTML tags
            title = self._clean_html(paper.get('title', 'No Title'))
            
            # Clean authors 
            authors = paper.get('authors', [])
            authors_text = ', '.join(self._clean_html(author) for author in authors) if authors else 'Unknown Authors'
            
            # Clean journal
            journal = self._clean_html(paper.get('journal', ''))
            pub_date = paper.get('publication_date', '')
            
            # Simple plain text format without HTML tags
            item_text = f"{title}\n{authors_text}\n"
            if journal:
                item_text += journal
            if pub_date:
                item_text += f" ({pub_date})"
                
            paper_item.setText(item_text)
            paper_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            
            # Set a fixed height for the item
            paper_item.setSizeHint(QSize(self.results_list.width(), 80))
            
            self.results_list.addItem(paper_item)
            
    def _clean_html(self, text):
        """Remove HTML tags and normalize whitespace"""
        if not text:
            return ""
        # Remove HTML tags
        clean_text = re.sub(r'<[^>]+>', '', text)
        # Normalize whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        # Remove any unbalanced braces
        clean_text = re.sub(r'[\{\}]', '', clean_text)
        return clean_text
    
    def go_to_page(self, page):
        """Go to a specific page"""
        if 1 <= page <= self.total_pages:
            self.current_page = page
            self.display_current_page()
            self.update_pagination_controls()
    
    def go_to_next_page(self):
        """Go to the next page if available"""
        if self.current_page < self.total_pages:
            self.current_page += 1
            self.display_current_page()
            self.update_pagination_controls()
    
    def go_to_prev_page(self):
        """Go to the previous page if available"""
        if self.current_page > 1:
            self.current_page -= 1
            self.display_current_page()
            self.update_pagination_controls()
    
    def change_papers_per_page(self, value):
        """Change the number of papers displayed per page"""
        self.papers_per_page = int(value)
        self.current_page = 1  # Reset to first page
        self.display_current_page()
        self.update_pagination_controls()
    
    def update_pagination_controls(self):
        """Update the pagination controls based on current state"""
        total_papers = len(self.filtered_papers)
        self.page_label.setText(f"Page {self.current_page} of {self.total_pages} ({total_papers} results)")
        
        # Enable/disable pagination buttons
        self.first_page_btn.setEnabled(self.current_page > 1)
        self.prev_page_btn.setEnabled(self.current_page > 1)
        self.next_page_btn.setEnabled(self.current_page < self.total_pages)
        self.last_page_btn.setEnabled(self.current_page < self.total_pages)
    
    def on_item_double_clicked(self, item):
        """Handle double-click on a paper item to view details"""
        idx = item.data(Qt.ItemDataRole.UserRole)
        if idx is not None and 0 <= idx < len(self.filtered_papers):
            paper = self.filtered_papers[idx]
            self.itemSelected.emit(paper)

class LiteratureSearchSection(QWidget):
    """
    Specialized widget for literature discovery and collection.
    Handles searching, filtering, and storing papers in the StudiesManager.
    """
    # Signal when papers are found/selected
    papersCollected = pyqtSignal(list)  # Emits list of papers
    searchCompleted = pyqtSignal(bool)   # Emits success status

    def __init__(self, parent=None):
        super().__init__(parent)
        self.studies_manager = None
        self.search_results = []
        self.selected_papers = []
        self.search_history = []
        self.query_builder = AdvancedSearchQuery()
        
        # Initialize clients
        self.doi_manager = None
        self.pubmed_client = None
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Main container with padding
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Create a splitter for top and bottom sections
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_splitter.setChildrenCollapsible(False)
        
        # ===== TOP SECTION - SEARCH CONTROLS =====
        self.top_widget = QWidget()
        top_layout = QVBoxLayout(self.top_widget)
        top_layout.setSpacing(15)
        top_layout.setContentsMargins(10, 10, 10, 10)
        
        # Header with title and toggle button
        header_layout = QHBoxLayout()
        
        # Title 
        title_label = QLabel("Literature Search")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Add toggle button to show/hide controls
        self.toggle_controls_btn = QToolButton()
        self.toggle_controls_btn.setIcon(load_bootstrap_icon("chevron-up"))
        self.toggle_controls_btn.setToolTip("Hide search controls")
        self.toggle_controls_btn.setIconSize(QSize(20, 20))
        self.toggle_controls_btn.clicked.connect(self.toggle_controls)
        header_layout.addWidget(self.toggle_controls_btn)
        
        top_layout.addLayout(header_layout)
        
        # Controls container that can be hidden
        self.controls_container = QWidget()
        controls_layout = QVBoxLayout(self.controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create a horizontal splitter for query and filter sections
        controls_splitter = QSplitter(Qt.Orientation.Horizontal)
        controls_splitter.setChildrenCollapsible(False)
        
        # ===== QUERY BUILDER SECTION =====
        query_group = QGroupBox("Build Search Query")
        query_layout = QVBoxLayout(query_group)
        query_layout.setSpacing(15)
        query_layout.setContentsMargins(15, 20, 15, 15)
        
        # Query input area with descriptive label
        query_label = QLabel("Enter search terms and operators to build your query:")
        query_label.setWordWrap(True)
        query_layout.addWidget(query_label)
        
        # Term input with field selection
        term_form = QFormLayout()
        term_form.setSpacing(10)
        
        term_row = QHBoxLayout()
        self.term_input = QLineEdit()
        self.term_input.setPlaceholderText("Enter keywords, phrases, or terms...")
        self.term_input.setMinimumHeight(30)
        
        self.field_combo = QComboBox()
        self.field_combo.addItems(['All Fields'] + list(SearchModifiers.FIELD_TAGS.keys()))
        self.field_combo.setMinimumHeight(30)
        self.field_combo.setMinimumWidth(200)
        
        term_row.addWidget(self.term_input)
        term_row.addWidget(self.field_combo)
        
        term_form.addRow("Search Term:", term_row)
        
        # Add term button
        add_term_btn_layout = QHBoxLayout()
        self.add_term_btn = QPushButton("Add Term")
        self.add_term_btn.setIcon(load_bootstrap_icon("plus-circle"))
        self.add_term_btn.setStyleSheet("border: none;")
        self.add_term_btn.setMinimumHeight(36)
        self.add_term_btn.setIconSize(QSize(16, 16))  # Set icon size
        self.add_term_btn.clicked.connect(self.add_search_term)
        add_term_btn_layout.addWidget(self.add_term_btn)
        add_term_btn_layout.addStretch()
        term_form.addRow("", add_term_btn_layout)
        
        query_layout.addLayout(term_form)
        
        # Operator selection
        operator_form = QFormLayout()
        operator_form.setSpacing(10)
        
        operator_row = QHBoxLayout()
        self.operator_combo = QComboBox()
        self.operator_combo.addItems(list(SearchModifiers.BOOLEAN_OPERATORS.keys()))
        self.operator_combo.setMinimumHeight(30)
        self.operator_combo.setMinimumWidth(200)
        
        self.add_operator_btn = QPushButton("Add Operator")
        self.add_operator_btn.setIcon(load_bootstrap_icon("plus-square"))
        self.add_operator_btn.setStyleSheet("border: none;")
        self.add_operator_btn.setIconSize(QSize(16, 16))  # Set icon size
        self.add_operator_btn.setMinimumHeight(36)
        self.add_operator_btn.clicked.connect(self.add_operator)
        
        operator_row.addWidget(self.operator_combo)
        operator_row.addWidget(self.add_operator_btn)
        operator_row.addStretch()
        
        operator_form.addRow("Operator:", operator_row)
        query_layout.addLayout(operator_form)
        
        # Query preview with better visibility
        preview_label = QLabel("Current Query:")
        preview_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        query_layout.addWidget(preview_label)
        
        self.query_preview = QTextEdit()
        self.query_preview.setReadOnly(True)
        self.query_preview.setMinimumHeight(70)
        self.query_preview.setStyleSheet("border: 1px solid gray; border-radius: 4px;")
        query_layout.addWidget(self.query_preview)
        
        # Add clear query button
        clear_btn_layout = QHBoxLayout()
        clear_query_btn = QPushButton("Clear Query")
        clear_query_btn.setIcon(load_bootstrap_icon("x-circle"))
        clear_query_btn.setMinimumHeight(36)
        clear_query_btn.setIconSize(QSize(16, 16))  # Set icon size
        clear_query_btn.clicked.connect(self.clear_query)
        clear_query_btn.setStyleSheet("border: none;")
        clear_btn_layout.addWidget(clear_query_btn)
        clear_btn_layout.addStretch()

        self.search_button = QPushButton("Search")
        self.search_button.setIcon(load_bootstrap_icon("search"))
        self.search_button.setMinimumHeight(36)
        self.search_button.setIconSize(QSize(16, 16))  # Set icon size
        self.search_button.setStyleSheet("border: none;")
        search_button_font = QFont()
        search_button_font.setBold(True)
        self.search_button.setFont(search_button_font)
        self.search_button.clicked.connect(self.execute_search)

        query_layout.addLayout(clear_btn_layout)
        query_layout.addWidget(self.search_button)

        # ===== FILTERS SECTION =====
        filters_group = QGroupBox("Search Filters")
        filters_layout = QVBoxLayout(filters_group)
        filters_layout.setSpacing(15)
        filters_layout.setContentsMargins(15, 20, 15, 15)
        
        # Date range filter
        date_form = QFormLayout()
        date_form.setSpacing(10)
        
        date_range_layout = QHBoxLayout()
        date_range_layout.setSpacing(10)
        
        self.start_year = QSpinBox()
        self.end_year = QSpinBox()
        current_year = datetime.now().year
        self.start_year.setRange(1900, current_year)
        self.end_year.setRange(1900, current_year)
        self.start_year.setValue(2015)  # Set starting year to 2015
        self.end_year.setValue(current_year)
        self.start_year.setMinimumHeight(30)
        self.end_year.setMinimumHeight(30)
        self.start_year.setMinimumWidth(180)
        self.end_year.setMinimumWidth(180)
        
        date_range_layout.addWidget(self.start_year)
        date_range_layout.addWidget(QLabel("to"))
        date_range_layout.addWidget(self.end_year)
        date_range_layout.addStretch()
        
        date_form.addRow("Publication Years:", date_range_layout)
        filters_layout.addLayout(date_form)
        
        # Publication type and language
        pub_form = QFormLayout()
        pub_form.setSpacing(10)
        
        self.pub_type_combo = QComboBox()
        self.pub_type_combo.addItems([
            'All Types', 'Journal Article', 'Review', 'Clinical Trial',
            'Meta-Analysis', 'Randomized Controlled Trial'
        ])
        self.pub_type_combo.setMinimumHeight(30)
        self.pub_type_combo.setMinimumWidth(350)
        pub_form.addRow("Publication Type:", self.pub_type_combo)
        
        self.language_combo = QComboBox()
        self.language_combo.addItems(['All Languages', 'English', 'French', 'German', 'Spanish'])
        self.language_combo.setMinimumHeight(30)
        self.language_combo.setMinimumWidth(350)
        pub_form.addRow("Language:", self.language_combo)
        
        filters_layout.addLayout(pub_form)
        
        # Sources selection
        sources_label = QLabel("Search Sources:")
        filters_layout.addWidget(sources_label)
        
        sources_layout = QHBoxLayout()
        sources_layout.setSpacing(15)
        
        self.pubmed_check = QCheckBox("PubMed")
        self.pubmed_check.setIcon(load_bootstrap_icon("journal-medical"))
        self.pubmed_check.setMinimumHeight(30)
        self.pubmed_check.setChecked(True)
        
        self.crossref_check = QCheckBox("CrossRef")
        self.crossref_check.setIcon(load_bootstrap_icon("database"))
        self.crossref_check.setMinimumHeight(30)
        self.crossref_check.setChecked(True)
        
        # Add debug mode checkbox
        self.debug_mode_check = QCheckBox("Debug Mode (Mock Data)")
        self.debug_mode_check.setIcon(load_bootstrap_icon("bug"))
        self.debug_mode_check.setMinimumHeight(30)
        self.debug_mode_check.setToolTip("Use mock data instead of calling APIs")
        self.debug_mode_check.stateChanged.connect(self.update_query_preview)  # Update button text when checked
        
        sources_layout.addWidget(self.pubmed_check)
        sources_layout.addWidget(self.crossref_check)
        sources_layout.addWidget(self.debug_mode_check)
        sources_layout.addStretch()
        
        filters_layout.addLayout(sources_layout)
        
        # Results limit
        limit_form = QFormLayout()
        limit_form.setSpacing(10)
        
        self.results_limit = QSpinBox()
        self.results_limit.setRange(10, 1000)
        self.results_limit.setValue(100)
        self.results_limit.setSingleStep(10)
        self.results_limit.setMinimumHeight(30)
        self.results_limit.setMinimumWidth(180)
        
        limit_form.addRow("Maximum Results:", self.results_limit)
        filters_layout.addLayout(limit_form)
        
        # Add the query and filters to the splitter
        controls_splitter.addWidget(query_group)
        controls_splitter.addWidget(filters_group)
        
        # Set equal sizes for the splitter sections
        controls_splitter.setSizes([500, 500])
        
        controls_layout.addWidget(controls_splitter)
        
        # Add the controls container to the top layout
        top_layout.addWidget(self.controls_container)
        
        # ===== RESULTS SECTION =====
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setSpacing(15)
        results_layout.setContentsMargins(10, 10, 10, 10)
        
        # Results header with controls
        self.results_header = QHBoxLayout()
        
        results_title = QLabel("Search Results")
        results_title_font = QFont()
        results_title_font.setPointSize(14)
        results_title_font.setBold(True)
        results_title.setFont(results_title_font)
        self.results_header.addWidget(results_title)
        
        self.results_header.addStretch()
        
        # Filter results
        filter_layout = QHBoxLayout()
        filter_label = QLabel("Filter:")
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter by title, abstract, author, or journal...")
        self.filter_input.setMinimumHeight(30)
        self.filter_input.setMinimumWidth(300)  # Make filter wider
        self.filter_input.textChanged.connect(self.filter_results)
        
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.filter_input)
        
        self.results_header.addLayout(filter_layout)
        
        # Export button
        self.export_button = QPushButton("Export")
        self.export_button.setIcon(load_bootstrap_icon("download"))
        self.export_button.setMinimumHeight(36)
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        self.results_header.addWidget(self.export_button)
        
        # Add literature search management buttons
        search_mgmt_layout = QHBoxLayout()
        
        # Save search button
        self.save_search_btn = QPushButton("Save Search")
        self.save_search_btn.setIcon(load_bootstrap_icon("save"))
        self.save_search_btn.setMinimumHeight(36)
        self.save_search_btn.setEnabled(False)  # Disabled until search results are available
        self.save_search_btn.clicked.connect(self.save_current_search)
        search_mgmt_layout.addWidget(self.save_search_btn)
        
        # Clear search button
        self.clear_search_btn = QPushButton("Clear")
        self.clear_search_btn.setIcon(load_bootstrap_icon("x"))
        self.clear_search_btn.setMinimumHeight(36)
        self.clear_search_btn.clicked.connect(self.clear_search_results)
        search_mgmt_layout.addWidget(self.clear_search_btn)
        
        # Load previous searches button
        self.load_search_btn = QPushButton("Load Previous")
        self.load_search_btn.setIcon(load_bootstrap_icon("folder-open"))
        self.load_search_btn.setMinimumHeight(36)
        self.load_search_btn.clicked.connect(self.show_saved_searches)
        search_mgmt_layout.addWidget(self.load_search_btn)
        
        self.results_header.addLayout(search_mgmt_layout)
        
        results_layout.addLayout(self.results_header)
        
        # Results counter
        self.results_counter = QLabel("No results")
        results_layout.addWidget(self.results_counter)
        
        # Results area with the paginated results widget
        self.results_list = PaginatedResultsWidget()
        self.results_list.itemSelected.connect(self.show_paper_details)
        results_layout.addWidget(self.results_list)
        
        # Add top and results sections to main splitter
        self.main_splitter.addWidget(self.top_widget)
        self.main_splitter.addWidget(results_widget)
        
        # Set initial sizes (40% top, 60% results)
        self.main_splitter.setSizes([400, 600])
        
        # Add splitter to main layout
        main_layout.addWidget(self.main_splitter)

    def clear_query(self):
        """Clear the current query and reset the preview."""
        self.query_builder = AdvancedSearchQuery()
        self.update_query_preview()

    def set_studies_manager(self, studies_manager: StudiesManager):
        self.studies_manager = studies_manager

    @asyncSlot()
    async def execute_search(self):
        """Execute the advanced search with all modifiers and filters."""
        query = self.query_builder.build()
        if not query:
            QMessageBox.warning(self, "Warning", "Please build a search query.")
            return
            
        # Check for debug mode
        if hasattr(self, 'debug_mode_check') and self.debug_mode_check.isChecked():
            # Use mock data instead of real API calls
            self.generate_mock_results()
            return
        
        # Initialize clients if needed
        if not self.doi_manager or not self.pubmed_client:
            try:
                self.doi_manager = DOIManager()
                self.pubmed_client = PubMedClient(
                    email=secrets.get('entrez_email', ''),
                    api_key=secrets.get('entrez_api_key'),
                    logger=logging.getLogger(__name__)
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to initialize search clients: {str(e)}")
                return
        
        try:
            # Show a loading indicator
            self.results_counter.setText("Searching...")
            QApplication.processEvents()  # Force UI update
            
            # Get search parameters and filters
            max_results = self.results_limit.value()
            filters = self.build_search_filters()
            
            # Store search in history
            self.search_history.append({
                'query': query,
                'filters': filters,
                'timestamp': datetime.now().isoformat()
            })
            
            # Execute search with filters
            results = await self.execute_filtered_search(query, filters, max_results)
            
            # Log search results for debugging
            logging.info(f"Search returned {len(results)} results")
            if len(results) > 0:
                logging.info(f"First result: {results[0].get('title', 'No title')}")
            else:
                logging.info("No results found")
            
            # Store results
            self.search_results = results
            
            # Store in StudiesManager
            if results and self.studies_manager:
                df = pd.DataFrame(results)
                self.studies_manager.add_dataset_to_active_study(
                    f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    df,
                    metadata={
                        'query': query,
                        'filters': filters,
                        'search_history': self.search_history
                    }
                )
            
            # Display results
            self.display_results(results)
            
            # Emit completion signal
            self.searchCompleted.emit(True)
            self.papersCollected.emit(results)
            
            # Enable save button
            self.save_search_btn.setEnabled(True)
            
        except Exception as e:
            logging.error(f"Search failed: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Search failed: {str(e)}")
            self.searchCompleted.emit(False)

    def display_results(self, papers):
        """Display search results in the UI."""
        # Clear existing results
        self.results_counter.setText(f"Found {len(papers)} results")
            
        # Update button states
        self.export_button.setEnabled(len(papers) > 0)
        self.save_search_btn.setEnabled(len(papers) > 0)
        
        # Add a Rank Papers button if we have results
        if not hasattr(self, 'rank_papers_btn'):
            self.rank_papers_btn = QPushButton("Rank These Papers")
            self.rank_papers_btn.setIcon(load_bootstrap_icon("sort-numeric-down"))
            self.rank_papers_btn.clicked.connect(self.send_to_ranking)
            self.results_header.addWidget(self.rank_papers_btn)
        
        self.rank_papers_btn.setEnabled(len(papers) > 0)
        
        # Use the paginated results widget instead
        self.results_list.set_papers(papers)
    
    def filter_results(self):
        """Filter displayed results based on search text."""
        filter_text = self.filter_input.text()
        self.results_list.filter_papers(filter_text)

    def export_results(self):
        """Export search results to a file."""
        if not self.search_results:
            QMessageBox.warning(self, "Warning", "No results to export.")
            return
        
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "",
            "CSV Files (*.csv);;JSON Files (*.json)"
        )
        
        if not file_name:
            return
        
        try:
            if file_name.endswith('.csv'):
                df = pd.DataFrame(self.search_results)
                df.to_csv(file_name, index=False)
            else:  # JSON
                import json
                with open(file_name, 'w') as f:
                    json.dump(self.search_results, f, indent=2)
                
            QMessageBox.information(self, "Success", "Results exported successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export results: {str(e)}")

    async def execute_filtered_search(self, query: str, filters: Dict[str, str], max_results: int) -> List[Dict]:
        """
        Execute search with filters across multiple sources.
        
        Args:
            query: The search query string
            filters: Dictionary of search filters
            max_results: Maximum number of results per source
            
        Returns:
            List of combined and deduplicated search results
        """
        combined_papers = []
        seen_dois = set()
        
        try:
            # Search CrossRef if selected
            if hasattr(self, 'crossref_check') and self.crossref_check.isChecked():
                # Apply date filter for CrossRef
                start_year = self.start_year.value()
                end_year = self.end_year.value()
                
                try:
                    await self.doi_manager.search_crossref_async(
                        query, 
                        max_results=max_results,
                        search_field='all'  # CrossRef doesn't support all our field types
                    )
                    crossref_papers = self.doi_manager.to_dict()
                    
                    # Debug output
                    logging.info(f"CrossRef returned {len(crossref_papers)} papers")
                    
                    # Post-process to apply filters
                    filtered_papers = [
                        paper for paper in crossref_papers
                        if self._paper_matches_filters(paper, filters)
                    ]
                    
                    logging.info(f"After filtering: {len(filtered_papers)} CrossRef papers")
                    
                    # Process CrossRef papers
                    for paper in filtered_papers:
                        # Accept papers even if some fields are missing
                        if paper.get('doi'):
                            doi = paper['doi'].lower()
                            if doi not in seen_dois:
                                seen_dois.add(doi)
                                # Ensure all required fields exist
                                if 'abstract' not in paper:
                                    paper['abstract'] = "Abstract not available"
                                combined_papers.append(paper)
                except Exception as e:
                    logging.error(f"CrossRef search error: {str(e)}")
            
            # Search PubMed if selected
            if hasattr(self, 'pubmed_check') and self.pubmed_check.isChecked():
                try:
                    # Construct PubMed query with filters
                    pubmed_query = query
                    for filter_value in filters.values():
                        pubmed_query += f" AND {filter_value}"
                    
                    logging.info(f"Executing PubMed query: {pubmed_query}")
                    pubmed_results = await self.pubmed_client.search_pubmed(
                        pubmed_query,
                        max_results=max_results
                    )
                    
                    # Debug output
                    logging.info(f"PubMed returned {len(pubmed_results)} papers")
                    
                    # Process PubMed papers
                    for result in pubmed_results:
                        if result.doi:  # Don't require abstract for PubMed results
                            doi = result.doi.lower()
                            if doi not in seen_dois:
                                seen_dois.add(doi)
                                # Create paper dictionary with all fields
                                paper_dict = {
                                    'title': result.title or "No title available",
                                    'authors': result.authors or [],
                                    'doi': result.doi,
                                    'publication_date': result.publication_date or "",
                                    'journal': result.journal or "Unknown journal",
                                    'abstract': result.abstract or "Abstract not available",
                                    'source': result.source or "PubMed",
                                    'pmcid': result.pmcid or "",
                                    'pmid': result.pmid or ""
                                }
                                combined_papers.append(paper_dict)
                except Exception as e:
                    logging.error(f"PubMed search error: {str(e)}")
            
            logging.info(f"Combined results: {len(combined_papers)} papers")
            
            # If we have no results but there are papers without DOIs, include them
            if not combined_papers and hasattr(self, 'pubmed_check') and self.pubmed_check.isChecked():
                try:
                    # Retry PubMed search and include papers without DOIs
                    pubmed_query = query
                    for filter_value in filters.values():
                        pubmed_query += f" AND {filter_value}"
                    
                    pubmed_results = await self.pubmed_client.search_pubmed(
                        pubmed_query,
                        max_results=max_results
                    )
                    
                    for result in pubmed_results:
                        # Accept papers without DOIs
                        paper_dict = {
                            'title': result.title or "No title available",
                            'authors': result.authors or [],
                            'doi': result.doi or "No DOI",
                            'publication_date': result.publication_date or "",
                            'journal': result.journal or "Unknown journal",
                            'abstract': result.abstract or "Abstract not available",
                            'source': result.source or "PubMed",
                            'pmcid': result.pmcid or "",
                            'pmid': result.pmid or ""
                        }
                        combined_papers.append(paper_dict)
                except Exception as e:
                    logging.error(f"Fallback PubMed search error: {str(e)}")
            
            return combined_papers
            
        except Exception as e:
            logging.error(f"Error during filtered search: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            raise

    def _paper_matches_filters(self, paper: Dict, filters: Dict[str, str]) -> bool:
        """
        Check if a paper matches the applied filters.
        
        Args:
            paper: Paper dictionary
            filters: Dictionary of filters to apply
            
        Returns:
            bool: True if paper matches all filters
        """
        # If no filters, all papers match
        if not filters:
            return True
            
        try:
            # Check if paper has an abstract
            abstract = paper.get('abstract', '')
            if not abstract or abstract == "Abstract not available":
                return False
            
            # Check date range
            if 'date_range' in filters:
                date_str = paper.get('publication_date', '')
                if date_str:
                    try:
                        # Extract year from date string
                        if '-' in date_str:
                            year = int(date_str.split('-')[0])
                        else:
                            # Try to extract year from any format
                            year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
                            if year_match:
                                year = int(year_match.group(0))
                            else:
                                return True  # Can't determine year, allow it
                        
                        range_str = filters['date_range']
                        start_year = int(range_str.split(':')[0])
                        end_year = int(range_str.split(':')[1].split('[')[0])
                        if not (start_year <= year <= end_year):
                            return False
                    except (ValueError, IndexError) as e:
                        logging.debug(f"Date filter error for '{date_str}': {e}")
                        return True  # On error, don't filter out
                else:
                    return True  # No date, don't filter out
            
            # Check language
            if 'language' in filters:
                paper_lang = paper.get('language', '')
                if not paper_lang:  # If language not specified, assume English
                    paper_lang = 'eng'
                    
                filter_lang = filters['language'].split('[')[0].strip().lower()
                if filter_lang not in paper_lang.lower():
                    return False
            
            # Check publication type
            if 'publication_type' in filters:
                paper_type = paper.get('type', '').lower()
                if not paper_type:  # If type not specified, check journal
                    paper_type = paper.get('journal', '').lower()
                    
                filter_type = filters['publication_type'].split('"')[1].lower()
                if filter_type not in paper_type:
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error checking filters: {str(e)}")
            return True  # On error, include the paper

    def show_paper_details(self, paper):
        """Show the paper details dialog when a paper is selected"""
        dialog = PaperDetailDialog(paper, self)
        # Add the metadata tab to the dialog
        dialog.add_metadata_tab()
        dialog.exec()

    def build_search_filters(self) -> Dict[str, str]:
        """Build search filters based on current selections."""
        filters = {}
        
        # Date range filter
        if self.start_year.value() != 1900 or self.end_year.value() != datetime.now().year:
            filters['date_range'] = f"{self.start_year.value()}:{self.end_year.value()}[dp]"
            
        # Language filter
        language = self.language_combo.currentText()
        if language != 'All Languages':
            filters['language'] = f"{language}[lang]"
            
        # Publication type filter
        pub_type = self.pub_type_combo.currentText()
        if pub_type != 'All Types':
            filters['publication_type'] = f'"{pub_type}"[pt]'
            
        return filters

    def add_search_term(self):
        """Add a term to the query builder."""
        term = self.term_input.text().strip()
        if not term:
            return
            
        field = self.field_combo.currentText()
        if field == 'All Fields':
            self.query_builder.add_term(term)
        else:
            self.query_builder.add_term(term, field)
            
        self.update_query_preview()
        self.term_input.clear()
    
    def add_operator(self):
        """Add a boolean operator to the query builder."""
        operator = self.operator_combo.currentText()
        self.query_builder.add_boolean_operator(operator)
        self.update_query_preview()
    
    def update_query_preview(self):
        """Update the query preview text."""
        self.query_preview.setText(self.query_builder.build())
        
        # Update search button text based on debug mode
        if hasattr(self, 'debug_mode_check') and self.debug_mode_check.isChecked():
            self.search_button.setText("Generate Mock Results")
            self.search_button.setIcon(load_bootstrap_icon("bug"))
        else:
            self.search_button.setText("Search")
            self.search_button.setIcon(load_bootstrap_icon("search"))

    def toggle_controls(self):
        """Toggle visibility of the search controls section"""
        if self.controls_container.isVisible():
            # Save the current sizes before hiding
            current_sizes = self.main_splitter.sizes()
            
            # Hide controls
            self.controls_container.setVisible(False)
            self.toggle_controls_btn.setIcon(load_bootstrap_icon("chevron-down"))
            self.toggle_controls_btn.setToolTip("Show search controls")
            
            # Adjust the splitter to give more space to results
            # Just keep a small portion for the header with the toggle button
            header_height = 50  # approximate height needed for the header
            results_height = sum(current_sizes) - header_height
            self.main_splitter.setSizes([header_height, results_height])
        else:
            # Show controls
            self.controls_container.setVisible(True)
            self.toggle_controls_btn.setIcon(load_bootstrap_icon("chevron-up"))
            self.toggle_controls_btn.setToolTip("Hide search controls")
            
            # Reset to default 40%/60% distribution when showing controls
            total_height = sum(self.main_splitter.sizes())
            self.main_splitter.setSizes([int(total_height * 0.4), int(total_height * 0.6)])

    def save_current_search(self):
        """Save the current search results to the active study."""
        if not self.search_results:
            QMessageBox.warning(self, "Warning", "No search results to save.")
            return
        
        # Get the current query and filters
        query = self.query_builder.build()
        filters = self.build_search_filters()
        
        # Create an input dialog for the search description
        description, ok = QInputDialog.getText(
            self, 
            "Save Search", 
            "Enter a description for this search:",
            text=f"Search for '{query}' on {datetime.now().strftime('%Y-%m-%d')}"
        )
        
        if not ok:
            return  # User cancelled
        
        # Make sure we have a StudiesManager
        if not self.studies_manager:
            QMessageBox.warning(
                self, 
                "Warning", 
                "No active study to save to. Please create or select a study first."
            )
            return
        
        # Save to StudiesManager
        success = self.studies_manager.save_literature_search(
            query=query,
            filters=filters,
            papers=self.search_results,
            description=description
        )
        
        if success:
            QMessageBox.information(
                self, 
                "Success", 
                f"Saved search with {len(self.search_results)} papers to the active study."
            )
        else:
            QMessageBox.warning(
                self, 
                "Warning", 
                "Failed to save search. No active study available."
            )

    def clear_search_results(self):
        """Clear the current search results and form."""
        # Ask for confirmation
        reply = QMessageBox.question(
            self, 
            "Confirm Clear", 
            "Are you sure you want to clear the current search results?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Clear results
            self.search_results = []
            self.results_list.set_papers([])
            self.results_counter.setText("No results")
            self.export_button.setEnabled(False)
            self.save_search_btn.setEnabled(False)
            self.filter_input.clear()

    def show_saved_searches(self):
        """Show a dialog to select and load a previous search."""
        if not self.studies_manager:
            QMessageBox.warning(
                self, 
                "Warning", 
                "No active study available. Please create or select a study first."
            )
            return
        
        # Get saved searches from StudiesManager
        saved_searches = self.studies_manager.get_literature_searches()
        
        if not saved_searches:
            QMessageBox.information(
                self, 
                "Information", 
                "No saved searches found in the active study."
            )
            return
        
        # Create a dialog to display saved searches
        dialog = QDialog(self)
        dialog.setWindowTitle("Saved Literature Searches")
        dialog.setMinimumSize(800, 400)
        
        layout = QVBoxLayout(dialog)
        
        # Create a list widget to display searches
        list_widget = QListWidget()
        for search in saved_searches:
            timestamp = datetime.fromisoformat(search['timestamp']).strftime('%Y-%m-%d %H:%M')
            item = QListWidgetItem(f"{search['description']} - {search['paper_count']} papers - {timestamp}")
            item.setData(Qt.ItemDataRole.UserRole, search['id'])
            list_widget.addItem(item)
        
        layout.addWidget(list_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        load_btn = QPushButton("Load Selected")
        load_btn.clicked.connect(lambda: self.load_saved_search(list_widget) and dialog.accept())
        button_layout.addWidget(load_btn)
        
        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(lambda: self.delete_saved_search(list_widget) and list_widget.takeItem(list_widget.currentRow()))
        button_layout.addWidget(delete_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        # Show the dialog
        dialog.exec()

    def load_saved_search(self, list_widget):
        """Load a saved search selected from the list widget."""
        current_item = list_widget.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a search to load.")
            return False
        
        search_id = current_item.data(Qt.ItemDataRole.UserRole)
        
        # Get the full search data including papers
        search_data = self.studies_manager.get_literature_search_by_id(search_id)
        if not search_data:
            QMessageBox.warning(self, "Warning", "Failed to load search data.")
            return False
        
        # Set the search results
        self.search_results = search_data['papers']
        
        # Display the results
        self.display_results(search_data['papers'])
        
        # Update buttons
        self.export_button.setEnabled(True)
        self.save_search_btn.setEnabled(True)
        
        # Optionally, also restore the query in the query builder
        if 'query' in search_data:
            # Clear and set the query preview (but don't rebuild the query builder)
            self.query_preview.setText(search_data['query'])
        
        return True

    def delete_saved_search(self, list_widget):
        """Delete a saved search selected from the list widget."""
        current_item = list_widget.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a search to delete.")
            return False
        
        search_id = current_item.data(Qt.ItemDataRole.UserRole)
        
        # Confirm deletion
        reply = QMessageBox.question(
            self, 
            "Confirm Deletion", 
            "Are you sure you want to delete this saved search?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Delete the search
            success = self.studies_manager.delete_literature_search(search_id)
            if success:
                return True
            else:
                QMessageBox.warning(self, "Warning", "Failed to delete search.")
                return False
        
        return False

    def send_to_ranking(self):
        """Send the current search results to the paper ranking section."""
        if not self.search_results:
            QMessageBox.warning(self, "Warning", "No search results to rank.")
            return
        
        # Emit a signal that will be connected to the main window
        # The main window will then switch to the ranking tab and pass the papers
        self.papersCollected.emit(self.search_results)
        
        QMessageBox.information(
            self, 
            "Papers Sent", 
            "Search results have been sent to the Paper Ranking section.\n"
            "Switch to the Ranking tab to continue."
        )

    def generate_mock_results(self):
        """Generate mock scientific paper data for testing purposes."""
        try:
            # Show a loading indicator
            self.results_counter.setText("Generating mock data...")
            QApplication.processEvents()  # Force UI update
            
            # Get the number of results to generate
            num_results = min(self.results_limit.value(), 100)  # Cap at 100 for performance
            
            # Generate mock data
            mock_papers = []
            mock_titles = [
                "Advances in Neural Network Architectures for Image Recognition",
                "The Role of Gut Microbiota in Neurological Disorders",
                "Climate Change Impact on Marine Ecosystems: A Systematic Review",
                "CRISPR-Cas9 Applications in Treating Genetic Disorders",
                "Quantum Computing: Current State and Future Prospects",
                "Meta-Analysis of Cognitive Behavioral Therapy Effectiveness",
                "Machine Learning Approaches to Drug Discovery",
                "The Epidemiology of Emerging Infectious Diseases",
                "Sustainable Energy Solutions: A Comprehensive Review",
                "Brain-Computer Interfaces: Clinical Applications"
            ]
            
            mock_journals = [
                "Nature", "Science", "Cell", "The Lancet", "New England Journal of Medicine",
                "JAMA", "BMJ", "PLOS ONE", "Proceedings of the National Academy of Sciences",
                "Frontiers in Neuroscience"
            ]
            
            mock_abstracts = [
                "This study presents a novel approach to neural network architectures for image recognition, demonstrating significant improvements in accuracy and processing speed compared to previous methods.",
                "Recent evidence suggests strong associations between gut microbiota composition and various neurological disorders. This review synthesizes current knowledge and identifies areas for future research.",
                "We conducted a systematic review of 150 studies examining climate change impacts on marine ecosystems. Results indicate significant shifts in species distribution and ecosystem functioning.",
                "This paper reviews recent applications of CRISPR-Cas9 technology in treating genetic disorders, highlighting both successes and challenges in clinical implementation.",
                "We provide a comprehensive overview of the current state of quantum computing, including recent breakthroughs and potential applications in cryptography, optimization, and materials science.",
                "Our meta-analysis of 42 randomized controlled trials evaluated the effectiveness of cognitive behavioral therapy across different psychological conditions, finding strong evidence for its efficacy.",
                "This review examines how machine learning algorithms are transforming drug discovery processes, accelerating identification of potential therapeutic compounds and reducing development costs.",
                "We analyze patterns in emerging infectious disease outbreaks over the past two decades, identifying key risk factors and proposing strategies for improved surveillance and response.",
                "This paper reviews sustainable energy technologies, comparing their efficiency, environmental impact, and economic viability across different geographical and socioeconomic contexts.",
                "Recent advances in brain-computer interfaces show promising results for clinical applications in paralysis, communication disorders, and rehabilitation after stroke or spinal cord injury."
            ]
            
            current_year = datetime.now().year
            
            for i in range(num_results):
                # Generate a paper with random but realistic attributes
                title_idx = i % len(mock_titles)
                title_variant = f"{mock_titles[title_idx]}{': Part ' + str(i//len(mock_titles) + 1) if i >= len(mock_titles) else ''}"
                
                # Generate mock authors (1-6 authors)
                num_authors = min(6, (i % 6) + 1)
                first_names = ["John", "Sarah", "Michael", "Emma", "David", "Jennifer", "Robert", "Maria", "James", "Emily"]
                last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Wilson"]
                authors = []
                for j in range(num_authors):
                    first_idx = (i + j) % len(first_names)
                    last_idx = (i + j + 3) % len(last_names)
                    authors.append(f"{first_names[first_idx]} {last_names[last_idx]}")
                
                # Create a random publication date within the set range
                year = self.start_year.value() + (i % (self.end_year.value() - self.start_year.value() + 1))
                month = (i % 12) + 1
                day = (i % 28) + 1
                pub_date = f"{year}-{month:02d}-{day:02d}"
                
                # Generate a mock DOI
                doi = f"10.1234/mock.{year}.{i+1000}"
                
                # Generate PMID and PMCID for some papers
                pmid = f"{30000000 + i}" if i % 3 != 0 else ""
                pmcid = f"PMC{8000000 + i}" if i % 4 == 0 else ""
                
                # Create the paper object
                paper = {
                    'title': title_variant,
                    'authors': authors,
                    'abstract': mock_abstracts[i % len(mock_abstracts)],
                    'journal': mock_journals[i % len(mock_journals)],
                    'publication_date': pub_date,
                    'doi': doi,
                    'pmid': pmid,
                    'pmcid': pmcid,
                    'source': 'Mock Database',
                    'language': 'eng',
                    'type': 'Journal Article',
                    'citations': i * 3,  # Mock citation count
                    'keywords': ['research', 'science', f'topic-{i%5}']
                }
                
                mock_papers.append(paper)
            
            # Store and display results
            self.search_results = mock_papers
            self.display_results(mock_papers)
            
            # Update UI
            self.results_counter.setText(f"Found {len(mock_papers)} mock results")
            self.export_button.setEnabled(True)
            self.save_search_btn.setEnabled(True)
            
            # Emit signals
            self.searchCompleted.emit(True)
            self.papersCollected.emit(mock_papers)
            
        except Exception as e:
            logging.error(f"Error generating mock data: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to generate mock data: {str(e)}")