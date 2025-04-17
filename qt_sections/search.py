import asyncio
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QTabWidget, QFileDialog, QMessageBox, QProgressBar, QScrollArea,
    QGroupBox, QGridLayout, QSplitter, QButtonGroup, QRadioButton,
    QFrame, QDialog, QSpinBox, QCheckBox, QFormLayout
)
from PyQt6.QtCore import pyqtSignal, Qt
import logging
import matplotlib

from helpers.load_icon import load_bootstrap_icon
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

class TimelineCanvas(FigureCanvas):
    """Canvas for displaying papers on a timeline."""
    def __init__(self, parent=None, width=10, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # Create two subplots side by side
        self.cluster_ax = self.fig.add_subplot(121)
        self.timeline_ax = self.fig.add_subplot(122)
        
        # Apply dark style to both subplots
        self.fig.patch.set_facecolor('#2E2E2E')
        self.cluster_ax.set_facecolor('#3A3A3A')
        self.timeline_ax.set_facecolor('#3A3A3A')
        
        # Set text color to light gray for better visibility on dark background
        text_color = '#BDBDBD'
        for ax in [self.cluster_ax, self.timeline_ax]:
            ax.tick_params(colors=text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.title.set_color(text_color)
            for spine in ax.spines.values():
                spine.set_color('#555555')
                
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()
        
    def plot_data(self, timeline_data, cluster_colors=None):
        """Plot cluster proportions and papers by year."""
        self.cluster_ax.clear()
        self.timeline_ax.clear()
        
        if not timeline_data:
            self._show_no_data_message()
            return
        
        # Default colors if not provided
        if not cluster_colors:
            # Create a color map for clusters
            unique_clusters = set(point['cluster_id'] for point in timeline_data)
            cmap = cm.get_cmap('viridis', max(3, len(unique_clusters)))
            cluster_colors = {cluster_id: cmap(i/len(unique_clusters)) 
                             for i, cluster_id in enumerate(unique_clusters)}
        
        # Extract paper data from timeline points
        all_papers = []
        for point in timeline_data:
            for paper in point['papers']:
                paper['cluster_id'] = point['cluster_id']
                paper['date'] = point['date']
                all_papers.append(paper)
        
        # Plot cluster proportion bar graph
        self._plot_cluster_proportions(all_papers, cluster_colors)
        
        # Plot papers by year line graph
        self._plot_papers_by_year(all_papers)
        
        self.fig.tight_layout()
        self.draw()
        
    def _plot_cluster_proportions(self, papers, cluster_colors):
        """Plot vertical bar graph showing proportion of papers in each cluster."""
        # Count papers in each cluster
        cluster_counts = Counter(paper['cluster_id'] for paper in papers)
        
        # For labels, use "Good", "Fair", "Poor" if there are 3 clusters or cluster_ids if more
        if len(cluster_counts) == 3:
            sorted_clusters = sorted(cluster_counts.keys())
            cluster_labels = {sorted_clusters[0]: "Good", sorted_clusters[1]: "Fair", sorted_clusters[2]: "Poor"}
        else:
            cluster_labels = {k: f"Cluster {k}" for k in cluster_counts.keys()}
        
        # Prepare data for plotting
        clusters = list(cluster_counts.keys())
        counts = [cluster_counts[c] for c in clusters]
        colors = [cluster_colors[c] for c in clusters]
        labels = [cluster_labels[c] for c in clusters]
        
        # Calculate percentages
        total = sum(counts)
        percentages = [count/total*100 for count in counts]
        
        # Plot
        bars = self.cluster_ax.bar(labels, percentages, color=colors)
        
        # Add percentage labels on top of bars
        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            self.cluster_ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 1,
                f'{percentage:.1f}%',
                ha='center', 
                va='bottom',
                color='#BDBDBD'
            )
        
        # Set labels and title
        self.cluster_ax.set_ylabel('Percentage of Papers', color='#BDBDBD')
        self.cluster_ax.set_title('Distribution by Cluster', color='#BDBDBD')
        self.cluster_ax.tick_params(axis='x', labelrotation=0, colors='#BDBDBD')
        
    def _plot_papers_by_year(self, papers):
        """Plot line graph showing number of papers by year."""
        # Extract years from dates
        years_count = defaultdict(int)
        for paper in papers:
            try:
                # First check if we have an explicit year field
                if 'year' in paper and paper['year'] is not None:
                    year = paper['year']
                    years_count[year] += 1
                    continue
                    
                # If no explicit year field, try to extract from various date fields
                date_str = paper.get('date', '')
                pub_date = paper.get('publication_date', '')
                
                # Try both the date field and publication_date field
                year = None
                if date_str:
                    year = date_to_year(date_str)
                
                # If we couldn't get a year from date, try publication_date
                if year is None and pub_date:
                    year = date_to_year(pub_date)
                
                # If we still don't have a year, check the formatted_date
                if year is None and 'formatted_date' in paper:
                    year = date_to_year(paper['formatted_date'])
                
                if year is not None:
                    # Ensure year is a reasonable value (not too old, not in the future)
                    current_year = datetime.now().year
                    if 1950 <= year <= current_year + 1:
                        years_count[year] += 1
                    else:
                        logging.debug(f"Skipping unreasonable year value: {year}")
            except Exception as e:
                # Just skip this paper if we can't parse the date
                logging.debug(f"Could not parse date for paper: {e}")
                continue
        
        if not years_count:
            self.timeline_ax.text(0.5, 0.5, "No valid dates available", 
                                horizontalalignment='center', verticalalignment='center',
                                color='#BDBDBD')
            return
            
        # Log the years data for debugging
        logging.debug(f"Years count for timeline: {dict(years_count)}")
        
        # Sort years
        years = sorted(years_count.keys())
        counts = [years_count[year] for year in years]
        
        # Plot line graph
        self.timeline_ax.plot(years, counts, 'o-', color='#4CAF50', linewidth=2, markersize=8)
        
        # Fill area under the line
        self.timeline_ax.fill_between(years, counts, alpha=0.3, color='#4CAF50')
        
        # Set labels and title
        self.timeline_ax.set_xlabel('Year', color='#BDBDBD')
        self.timeline_ax.set_ylabel('Number of Papers', color='#BDBDBD')
        self.timeline_ax.set_title('Publications by Year', color='#BDBDBD')
        
        # Format x-axis
        if years:
            # Set reasonable ticks based on the range of years
            year_range = max(years) - min(years)
            if year_range > 20:
                # For large ranges, show 5-year intervals
                start_year = (min(years) // 5) * 5  # Round down to nearest 5
                end_year = ((max(years) // 5) + 1) * 5  # Round up to nearest 5
                self.timeline_ax.set_xticks(range(start_year, end_year + 1, 5))
            elif year_range > 10:
                # For medium ranges, show 2-year intervals
                self.timeline_ax.set_xticks(range(min(years), max(years) + 1, 2))
            else:
                # For small ranges, show every year
                self.timeline_ax.set_xticks(range(min(years), max(years) + 1))
                
            # Make sure the min and max years are always included in ticks
            ticks = list(self.timeline_ax.get_xticks())
            if min(years) not in ticks:
                ticks.insert(0, min(years))
            if max(years) not in ticks:
                ticks.append(max(years))
            self.timeline_ax.set_xticks(sorted(ticks))
            
            # Set x-axis limits with some padding
            padding = max(1, year_range * 0.05)  # 5% padding on each side
            self.timeline_ax.set_xlim(min(years) - padding, max(years) + padding)
        
        self.timeline_ax.tick_params(axis='x', colors='#BDBDBD')
        
    def _show_no_data_message(self):
        """Display a message when no data is available."""
        for ax in [self.cluster_ax, self.timeline_ax]:
            ax.text(0.5, 0.5, "No data available", 
                  horizontalalignment='center', verticalalignment='center',
                  color='#BDBDBD')
        self.draw()

class PaperDialog(QDialog):
    """Dialog for displaying the full paper details."""
    def __init__(self, paper, parent=None):
        super().__init__(parent)
        self.setWindowTitle(paper.get('title', 'Paper Details'))
        self.setMinimumSize(700, 500)
        
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel(f"<h2>{paper.get('title', 'No Title')}</h2>")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)
        
        # Authors
        authors = paper.get('authors', [])
        if authors:
            if isinstance(authors, list):
                authors_str = ", ".join(authors)
            else:
                authors_str = str(authors)
            layout.addWidget(QLabel(f"<b>Authors:</b> {authors_str}"))
        
        # Journal and date
        journal = paper.get('journal', 'N/A')
        pub_date = paper.get('publication_date', 'N/A')
        layout.addWidget(QLabel(f"<b>Journal:</b> {journal} | <b>Date:</b> {pub_date}"))
        
        # DOI with link
        doi = paper.get('doi', 'N/A')
        doi_label = QLabel(f"<b>DOI:</b> <a href='https://doi.org/{doi}'>{doi}</a>")
        doi_label.setOpenExternalLinks(True)
        layout.addWidget(doi_label)
        
        # Score information (if available)
        if 'composite_score' in paper:
            score_label = QLabel(f"<b>Score: {paper.get('composite_score', 'N/A')}</b>")
            score_label.setStyleSheet("color: #707070; font-size: 14pt;")
            layout.addWidget(score_label)
            
            if 'scoring_breakdown' in paper:
                breakdown_label = QLabel(f"<i>{paper.get('scoring_breakdown', '')}</i>")
                breakdown_label.setWordWrap(True)
                layout.addWidget(breakdown_label)
                
            if 'comment' in paper:
                comment_label = QLabel(f"<i>{paper.get('comment', '')}</i>")
                comment_label.setWordWrap(True)
                layout.addWidget(comment_label)
        
        # Abstract
        abstract = paper.get('abstract', '')
        if abstract:
            abstract_group = QGroupBox("Abstract")
            abstract_layout = QVBoxLayout(abstract_group)
            
            abstract_text = QLabel(abstract)
            abstract_text.setWordWrap(True)
            abstract_layout.addWidget(abstract_text)
            
            layout.addWidget(abstract_group)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

class PaperWidget(QGroupBox):
    """Widget for displaying paper information."""
    def __init__(self, paper, is_ranked=False, parent=None):
        title = paper.get('title', 'No Title')
        super().__init__(title, parent)
        
        main_layout = QVBoxLayout(self)
        
        # Top row with metadata and expand button
        top_row = QHBoxLayout()
        
        # Metadata container
        metadata_widget = QWidget()
        layout = QVBoxLayout(metadata_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Add metadata
        if is_ranked:
            score_label = QLabel(f"<b>Score: {paper.get('composite_score', 'N/A')}</b>")
            score_label.setStyleSheet("color: #707070; font-size: 12pt;")
            layout.addWidget(score_label)
            
            if 'scoring_breakdown' in paper:
                breakdown_label = QLabel(f"<i>{paper.get('scoring_breakdown', '')}</i>")
                breakdown_label.setWordWrap(True)
                layout.addWidget(breakdown_label)
                
            if 'comment' in paper:
                comment_label = QLabel(f"<i>{paper.get('comment', '')}</i>")
                comment_label.setWordWrap(True)
                layout.addWidget(comment_label)
        
        # Authors
        authors = paper.get('authors', [])
        if authors:
            if isinstance(authors, list):
                authors_str = ", ".join(authors[:3])
                if len(authors) > 3:
                    authors_str += f" and {len(authors) - 3} more"
            else:
                authors_str = str(authors)
            layout.addWidget(QLabel(f"<b>Authors:</b> {authors_str}"))
        
        # Journal and date
        journal = paper.get('journal', 'N/A')
        pub_date = paper.get('publication_date', 'N/A')
        layout.addWidget(QLabel(f"<b>Journal:</b> {journal} | <b>Date:</b> {pub_date}"))
        
        # DOI
        doi = paper.get('doi', 'N/A')
        doi_label = QLabel(f"<b>DOI:</b> <a href='https://doi.org/{doi}'>{doi}</a>")
        doi_label.setOpenExternalLinks(True)
        layout.addWidget(doi_label)
        
        # Add metadata to top row
        top_row.addWidget(metadata_widget, stretch=4)
        
        # Show details button in top right
        self.details_button = QPushButton("Show Details")
        self.details_button.setIcon(load_bootstrap_icon("info-circle"))
        self.details_button.setStyleSheet("font-size: 10pt;")
        self.details_button.clicked.connect(lambda: self.show_paper_dialog(paper))
        top_row.addWidget(self.details_button, stretch=1, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        
        main_layout.addLayout(top_row)
        
        # Abstract (truncated version only)
        abstract = paper.get('abstract', '')
        if abstract:
            self.abstract_truncated = abstract[:200] + "..." if len(abstract) > 200 else abstract
            self.abstract_label = QLabel(f"<b>Abstract:</b> {self.abstract_truncated}")
            self.abstract_label.setWordWrap(True)
            main_layout.addWidget(self.abstract_label)
        
        # Dynamic height instead of fixed height
        if is_ranked:
            self.setMinimumHeight(200)  # Increase height for ranked papers
        else:
            self.setMinimumHeight(150)
        
        # Style based on score if ranked
        if is_ranked and 'composite_score' in paper:
            score = paper.get('composite_score', 0)
            if score >= 80:
                self.setStyleSheet("QGroupBox { border: 2px solid #4CAF50; border-radius: 5px; }")
            elif score >= 60:
                self.setStyleSheet("QGroupBox { border: 2px solid #FFC107; border-radius: 5px; }")
            else:
                self.setStyleSheet("QGroupBox { border: 2px solid #9E9E9E; border-radius: 5px; }")
    
    def show_paper_dialog(self, paper):
        """Show a dialog with the full paper details."""
        dialog = PaperDialog(paper, self)
        dialog.exec()

class ReviewSectionWidget(QGroupBox):
    """Widget for displaying a review section."""
    def __init__(self, section, parent=None):
        super().__init__(section.theme, parent)
        layout = QVBoxLayout(self)
        
        # Add content
        content_label = QLabel(section.content)
        content_label.setWordWrap(True)
        layout.addWidget(content_label)
        
        # Add quotes section
        if hasattr(section, 'quotes') and section.quotes:
            self.quotes_container = QWidget()
            quotes_layout = QVBoxLayout(self.quotes_container)
            
            # Add quotes header
            quotes_header = QLabel(f"<b>Supporting Quotes ({len(section.quotes)}):</b>")
            quotes_layout.addWidget(quotes_header)
            
            # Add each quote
            for i, quote in enumerate(section.quotes):
                quote_box = QGroupBox(f"Quote {i+1}")
                # Fix the overlapping title issue by adding proper styling with margin-top
                quote_box.setStyleSheet("""
                    QGroupBox { 
                        border: 1px solid gray; 
                        border-radius: 5px; 
                        padding: 15px; 
                        margin-top: 20px; 
                    }
                    QGroupBox::title {
                        subcontrol-origin: margin;
                        left: 10px;
                        padding: 0 5px 0 5px;
                    }
                """)
                quote_inner_layout = QVBoxLayout(quote_box)
                
                # Quote text - use larger font and italic styling
                quote_text = QLabel(f'"{quote.text}"')
                quote_text.setWordWrap(True)
                quote_text.setStyleSheet("font-style: italic; font-size: 11pt;")
                quote_inner_layout.addWidget(quote_text)
                
                # Citation
                # Create citation from paper title and doi instead of accessing a non-existent attribute
                citation_text = f"{quote.paper_title}"
                if hasattr(quote, 'doi') and quote.doi:
                    citation_label = QLabel(f"<a href='https://doi.org/{quote.doi}'>{citation_text}</a>")
                    citation_label.setOpenExternalLinks(True)
                elif hasattr(quote, 'paper_doi') and quote.paper_doi:
                    citation_label = QLabel(f"<a href='https://doi.org/{quote.paper_doi}'>{citation_text}</a>")
                    citation_label.setOpenExternalLinks(True)
                else:
                    citation_label = QLabel(citation_text)
                citation_label.setStyleSheet("font-size: 8pt; color: #555555;")
                quote_inner_layout.addWidget(citation_label)
                
                quotes_layout.addWidget(quote_box)
            
            # Initially hide quotes
            self.quotes_container.setVisible(False)
            layout.addWidget(self.quotes_container)
            
            # Add show/hide quotes button
            self.toggle_quotes_button = QPushButton("Show Supporting Quotes")
            self.toggle_quotes_button.setIcon(load_bootstrap_icon("quote"))
            self.toggle_quotes_button.clicked.connect(self.toggle_quotes)
            layout.addWidget(self.toggle_quotes_button)
        
        # Add papers cited (just count and DOIs)
        if section.papers_cited:
            cited_label = QLabel(f"<b>Papers cited:</b> {len(section.papers_cited)}")
            cited_label.setToolTip(", ".join(section.papers_cited))
            layout.addWidget(cited_label)
    
    def toggle_quotes(self):
        """Toggle visibility of quotes."""
        is_visible = self.quotes_container.isVisible()
        self.quotes_container.setVisible(not is_visible)
        self.toggle_quotes_button.setText(
            "Hide Supporting Quotes" if not is_visible else "Show Supporting Quotes"
        )
        self.toggle_quotes_button.setIcon(
            load_bootstrap_icon("eye-slash" if not is_visible else "quote")
        )

class SearchSection(QWidget):
    # Signal used to send the search query (e.g. to a WebSocket handler)
    searchRequested = pyqtSignal(str)
    # Add a signal for module loading status updates
    moduleLoadingStatusUpdated = pyqtSignal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize module flags and objects as None
        self.modules_loaded = False
        self.is_loading_modules = False
        self.cluster_analyzer = None
        self.quote_extractor = None
        self.doi_manager = None
        self.pubmed_client = None
        self.grounded_review = None
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize operation flags
        self.is_searching = False
        self.is_ranking = False
        self.is_generating_review = False
        # Add a lock to prevent concurrent execution
        self.search_lock = asyncio.Lock()

        # Data storage
        self.all_papers = []
        self.ranked_papers = []
        self.timeline_data = []
        self.cluster_colors = {}
        
        # Pagination settings - initialize with defaults
        self.papers_per_page = 5
        self.current_page = 0
        self.total_pages = 0
        
        # Create the UI
        self.setup_ui()

        # Connect signals
        self.moduleLoadingStatusUpdated.connect(self.update_module_loading_status)

    def setup_ui(self):
        """Set up the user interface."""
        self.main_layout = QVBoxLayout(self)
        
        # Add the search input area
        search_area = QGroupBox("Search Parameters")
        search_layout = QVBoxLayout(search_area)
        
        # Input field and button
        input_layout = QHBoxLayout()
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter search query...")
        self.prompt_input.returnPressed.connect(lambda: asyncio.create_task(self.onPromptEnter()))
        input_layout.addWidget(self.prompt_input)
        
        self.search_button = QPushButton("Search")
        self.search_button.setIcon(load_bootstrap_icon("search"))
        self.search_button.clicked.connect(lambda: asyncio.create_task(self.onSearchButtonClick()))
        input_layout.addWidget(self.search_button)
        search_layout.addLayout(input_layout)
        
        # Search options layout
        options_layout = QHBoxLayout()
        
        # Checkboxes for search workflow control
        workflow_group = QGroupBox("Search Workflow")
        workflow_layout = QVBoxLayout(workflow_group)
        
        self.rank_papers_checkbox = QCheckBox("Rank Papers")
        self.rank_papers_checkbox.setChecked(True)
        self.rank_papers_checkbox.setToolTip("Rank papers by relevance to your query")
        
        self.generate_review_checkbox = QCheckBox("Generate Literature Review")
        self.generate_review_checkbox.setChecked(True)
        self.generate_review_checkbox.setToolTip("Generate a comprehensive literature review")
        
        # Connect checkboxes to update UI state
        self.generate_review_checkbox.toggled.connect(self.update_modules_visibility)
        self.rank_papers_checkbox.toggled.connect(self.update_review_checkbox_state)
        
        workflow_layout.addWidget(self.rank_papers_checkbox)
        workflow_layout.addWidget(self.generate_review_checkbox)
        options_layout.addWidget(workflow_group)
        
        # Parameters group
        params_group = QGroupBox("Search Parameters")
        params_layout = QFormLayout(params_group)
        
        self.crossref_papers_count = QSpinBox()
        self.crossref_papers_count.setRange(10, 500)
        self.crossref_papers_count.setValue(200)
        self.crossref_papers_count.setSingleStep(10)
        
        self.pubmed_papers_count = QSpinBox()
        self.pubmed_papers_count.setRange(10, 500)
        self.pubmed_papers_count.setValue(100)
        self.pubmed_papers_count.setSingleStep(10)
        
        self.ranked_papers_count = QSpinBox()
        self.ranked_papers_count.setRange(5, 100)
        self.ranked_papers_count.setValue(40)
        self.ranked_papers_count.setSingleStep(5)
        
        params_layout.addRow("CrossRef papers:", self.crossref_papers_count)
        params_layout.addRow("PubMed papers:", self.pubmed_papers_count)
        params_layout.addRow("Ranked papers:", self.ranked_papers_count)
        options_layout.addWidget(params_group)
        
        # Modules group
        self.modules_group = QGroupBox("Modules")
        modules_layout = QVBoxLayout(self.modules_group)
        
        self.modules_status_label = QLabel("Modules not loaded")
        self.modules_button = QPushButton("Load Modules")
        self.modules_button.setIcon(load_bootstrap_icon("box"))
        self.modules_button.clicked.connect(lambda: asyncio.create_task(self.toggle_modules()))
        
        modules_layout.addWidget(self.modules_status_label)
        modules_layout.addWidget(self.modules_button)
        options_layout.addWidget(self.modules_group)
        
        search_layout.addLayout(options_layout)
        
        # Status area
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        search_layout.addLayout(status_layout)
        
        self.main_layout.addWidget(search_area)
        
        # Add tabs for different views
        self.results_tabs = QTabWidget()
        
        # Papers tab
        papers_tab = QWidget()
        papers_layout = QVBoxLayout(papers_tab)
        
        # Radio buttons for switching between views
        view_layout = QHBoxLayout()
        self.all_papers_radio = QRadioButton("All Papers")
        self.all_papers_radio.setChecked(True)
        self.all_papers_radio.toggled.connect(self.toggle_paper_view)
        
        self.ranked_papers_radio = QRadioButton("Ranked Papers")
        self.ranked_papers_radio.toggled.connect(self.toggle_paper_view)
        
        view_layout.addWidget(self.all_papers_radio)
        view_layout.addWidget(self.ranked_papers_radio)
        
        # Per page control
        per_page_layout = QHBoxLayout()
        per_page_layout.addWidget(QLabel("Papers per page:"))
        self.papers_per_page_spinner = QSpinBox()
        self.papers_per_page_spinner.setRange(5, 50)
        self.papers_per_page_spinner.setValue(10)
        self.papers_per_page_spinner.setSingleStep(5)
        self.papers_per_page_spinner.valueChanged.connect(self.change_papers_per_page)
        per_page_layout.addWidget(self.papers_per_page_spinner)
        
        # Pagination controls
        per_page_layout.addStretch()
        self.prev_page_button = QPushButton("Previous")
        self.prev_page_button.setIcon(load_bootstrap_icon("arrow-left"))
        self.prev_page_button.clicked.connect(self.previous_page)
        per_page_layout.addWidget(self.prev_page_button)
        
        self.page_label = QLabel("Page 1 of 1")
        per_page_layout.addWidget(self.page_label)
        
        self.next_page_button = QPushButton("Next")
        self.next_page_button.setIcon(load_bootstrap_icon("arrow-right"))
        self.next_page_button.clicked.connect(self.next_page)
        per_page_layout.addWidget(self.next_page_button)
        
        papers_layout.addLayout(view_layout)
        papers_layout.addLayout(per_page_layout)
        
        # Papers scroll area
        papers_scroll = QScrollArea()
        papers_scroll.setWidgetResizable(True)
        papers_container = QWidget()
        self.papers_layout = QVBoxLayout(papers_container)
        papers_scroll.setWidget(papers_container)
        papers_layout.addWidget(papers_scroll)
        
        # Timeline tab
        timeline_tab = QWidget()
        timeline_layout = QVBoxLayout(timeline_tab)
        self.timeline_canvas = TimelineCanvas()
        timeline_layout.addWidget(self.timeline_canvas)
        
        # Literature review tab
        review_tab = QWidget()
        review_scroll = QScrollArea()
        review_scroll.setWidgetResizable(True)
        review_container = QWidget()
        self.review_layout = QVBoxLayout(review_container)
        self.review_layout.addStretch()
        review_scroll.setWidget(review_container)
        review_layout = QVBoxLayout(review_tab)
        review_layout.addWidget(review_scroll)
        
        # Add the tabs with icons
        self.results_tabs.addTab(papers_tab, load_bootstrap_icon("file-text"), "Papers")
        self.results_tabs.addTab(timeline_tab, load_bootstrap_icon("graph-up"), "Timeline")
        self.results_tabs.addTab(review_tab, load_bootstrap_icon("book"), "Literature Review")
        
        self.main_layout.addWidget(self.results_tabs)
        
        # Initialize remaining variables
        self.all_papers = []
        self.ranked_papers = []
        self.timeline_data = []
        self.grounded_review = None
        self.quote_extractor = None
        self.current_page = 0
        self.total_pages = 0
        self.papers_per_page = 10
        self.cluster_analyzer = None
        self.doi_manager = None
        self.pubmed_client = None
        self.modules_loaded = False
        self.is_loading_modules = False
        
        # Set the initial state of UI components
        self.update_modules_visibility()

    def update_modules_visibility(self):
        """Update the visibility and state of modules controls based on review generation."""
        # Module loading status text
        if self.modules_loaded:
            self.modules_status_label.setText("Advanced modules loaded")
        elif self.doi_manager and self.pubmed_client:
            self.modules_status_label.setText("Basic search modules loaded")
        else:
            self.modules_status_label.setText("Modules not loaded")
        
        # Update button text based on current state
        if self.generate_review_checkbox.isChecked():
            self.modules_group.setTitle("Advanced Modules (Required for Review)")
            self.modules_group.setToolTip("Clustering and quote extraction modules are required for literature review generation")
            if not self.modules_loaded:
                self.modules_button.setText("Load Advanced Modules")
            else:
                self.modules_button.setText("Unload Modules")
        else:
            self.modules_group.setTitle("Advanced Modules (Optional)")
            self.modules_group.setToolTip("Advanced modules only needed for literature review")
            if not self.modules_loaded:
                self.modules_button.setText("Load Advanced Modules")
            else:
                self.modules_button.setText("Unload Modules")

    def update_review_checkbox_state(self, rank_checked):
        """Update the review checkbox state based on ranking checkbox."""
        # Review cannot be generated without ranking
        if not rank_checked:
            self.generate_review_checkbox.setChecked(False)
            self.generate_review_checkbox.setEnabled(False)
            self.generate_review_checkbox.setToolTip("Ranking is required for review generation")
        else:
            self.generate_review_checkbox.setEnabled(True)
            self.generate_review_checkbox.setToolTip("Generate a comprehensive literature review")

    @asyncSlot()
    async def onPromptEnter(self):
        """Handle enter key press in the prompt input."""
        await self.onSearchClicked()
        
    @asyncSlot()
    async def onSearchButtonClick(self):
        """Handle search button click."""
        await self.onSearchClicked()

    @asyncSlot()
    async def onSearchClicked(self, should_rank=None, should_generate_review=None):
        """
        Handle search button click event.
        
        Args:
            should_rank: Whether to rank the papers after searching (overrides checkbox if provided)
            should_generate_review: Whether to generate a literature review after ranking (overrides checkbox if provided)
        """
        # Prevent concurrent execution
        if self.search_lock.locked():
            logging.info("Search already in progress, ignoring request")
            return
            
        async with self.search_lock:
            # Use provided values if given, otherwise use checkbox values
            should_rank = self.rank_papers_checkbox.isChecked() if should_rank is None else should_rank
            should_generate_review = self.generate_review_checkbox.isChecked() if should_generate_review is None else should_generate_review
            
            # Update checkboxes to match what we're doing (in case values were overridden)
            self.rank_papers_checkbox.setChecked(should_rank)
            if should_rank:
                self.generate_review_checkbox.setEnabled(True)
                self.generate_review_checkbox.setChecked(should_generate_review)
            else:
                self.generate_review_checkbox.setChecked(False)
                self.generate_review_checkbox.setEnabled(False)
                
            query = self.prompt_input.text().strip()
            if not query:
                QMessageBox.warning(self, "Warning", "Please enter a search query.")
                return
            
            # Prevent multiple operations at once
            if self.is_searching or self.is_ranking or self.is_generating_review:
                QMessageBox.warning(self, "Operation in Progress", "Please wait for the current operation to complete.")
                return
                
            # Set operation flags
            self.is_searching = True
            
            try:
                # Initialize basic search components if not loaded (these are always required)
                if not self.doi_manager or not self.pubmed_client:
                    self.status_label.setText("Initializing search clients...")
                    try:
                        self.doi_manager = DOIManager()
                        self.pubmed_client = PubMedClient(
                            email=secrets.get('entrez_email', ''),
                            api_key=secrets.get('entrez_api_key'),
                            logger=logging.getLogger(__name__)
                        )
                        self.status_label.setText("Search clients initialized.")
                    except Exception as e:
                        error_msg = f"Failed to initialize search clients: {str(e)}"
                        logging.error(error_msg, exc_info=True)
                        QMessageBox.critical(self, "Error", error_msg)
                        self.is_searching = False
                        return
                
                # If review is requested but modules not loaded, we need to load them
                if should_generate_review and not self.modules_loaded:
                    modules_response = QMessageBox.question(
                        self, "Load Advanced Modules", 
                        "Generating a literature review requires loading advanced modules for clustering and quote extraction. Would you like to load them now?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    
                    if modules_response == QMessageBox.StandardButton.Yes:
                        # Load modules first
                        await self.load_modules_async()
                        if not self.modules_loaded:
                            QMessageBox.warning(self, "Warning", "Failed to load advanced modules. Cannot generate review.")
                            should_generate_review = False
                            self.generate_review_checkbox.setChecked(False)
                    else:
                        # User declined to load modules, disable review generation
                        should_generate_review = False
                        self.generate_review_checkbox.setChecked(False)
                
                # Get search parameters from UI
                max_crossref_papers = self.crossref_papers_count.value()
                max_pubmed_papers = self.pubmed_papers_count.value()
                max_ranked_papers = self.ranked_papers_count.value()
                
                # Reset pagination
                self.current_page = 0
                self.total_pages = 0
                
                # Clear previous results
                self.clear_layout(self.papers_layout)
                self.clear_layout(self.review_layout)
                self.all_papers = []
                self.ranked_papers = []
                self.timeline_data = []

                self.status_label.setText("Starting search...")
                self.update_progress(0, "Starting search...")

                # Search CrossRef
                await self.doi_manager.search_crossref_async(query, max_results=max_crossref_papers, search_field='all')
                crossref_papers = self.doi_manager.to_dict()

                # Search PubMed
                pubmed_results = await self.pubmed_client.search_pubmed(
                    query,
                    max_results=max_pubmed_papers,
                    search_field='all'
                )

                # Combine and deduplicate results
                combined_papers = []
                seen_dois = set()
                search_results = []

                # Process CrossRef papers
                for paper in crossref_papers:
                    if paper.get('doi') and paper.get('abstract'):
                        doi = paper['doi'].lower()
                        if doi not in seen_dois:
                            seen_dois.add(doi)
                            combined_papers.append(paper)
                            search_results.append(SearchResult(
                                title=paper['title'],
                                authors=paper['authors'],
                                abstract=paper['abstract'],
                                doi=doi,
                                publication_date=paper['publication_date'],
                                journal=paper['journal'],
                                source='CrossRef',
                                pmcid=None,
                                pmid=None
                            ))

                # Process PubMed papers
                for result in pubmed_results:
                    if result.doi and result.abstract:
                        doi = result.doi.lower()
                        if doi not in seen_dois:
                            seen_dois.add(doi)
                            paper_dict = {
                                'title': result.title,
                                'authors': result.authors,
                                'doi': result.doi,
                                'publication_date': result.publication_date,
                                'journal': result.journal,
                                'abstract': result.abstract,
                                'source': result.source,
                                'pmcid': result.pmcid,
                                'pmid': result.pmid
                            }
                            combined_papers.append(paper_dict)
                            search_results.append(result)

                # Store all papers
                self.all_papers = combined_papers
                                
                # Display all papers initially
                self.display_papers(self.all_papers, is_ranked=False)
                self.update_progress(25, "Search complete")
                
                # If we should stop after search, update progress and return
                if not should_rank:
                    self.status_label.setText("Search complete")
                    self.update_progress(100, "Complete")
                    return

                # Set the ranking flag before starting ranking
                self.is_ranking = True
                self.update_progress(25, "Rating papers...")

                # Rate papers using Gemini
                rated_papers = await rate_papers_with_gemini(combined_papers, query, logging.getLogger(__name__))
                
                # Update combined_papers with rating info
                doi_rating_map = {paper['doi'].lower(): paper for paper in rated_papers}
                for paper in combined_papers:
                    doi = paper['doi'].lower()
                    rating_info = doi_rating_map.get(doi, {})
                    paper['composite_score'] = rating_info.get('composite_score', 0)
                    paper['scoring_breakdown'] = rating_info.get('scoring_breakdown', '')
                    paper['comment'] = rating_info.get('comment', '')
                
                # Store ranked papers (as a subset of all papers, with complete data)
                # Sort all papers by composite_score and take the ones with valid scores
                sorted_papers = sorted(
                    [p for p in combined_papers if p.get('composite_score', 0) > 0],
                    key=lambda x: x.get('composite_score', 0), 
                    reverse=True
                )
                # Limit to max_ranked_papers
                self.ranked_papers = sorted_papers[:max_ranked_papers]
                                
                # Switch to ranked view
                self.ranked_papers_radio.setChecked(True)
                self.display_papers(self.ranked_papers, is_ranked=True)
                self.update_progress(50, "Ranking complete")
                
                # If we should stop after ranking, update progress and return
                if not should_generate_review:
                    self.status_label.setText("Ranking complete")
                    self.update_progress(100, "Complete")
                    # Reset ranking flag
                    self.is_ranking = False
                    return
                
                # Verify modules are loaded for review generation
                if not self.modules_loaded:
                    self.status_label.setText("Cannot generate review without modules")
                    self.update_progress(0, "Error")
                    QMessageBox.warning(self, "Warning", "Cannot generate literature review. Advanced modules not loaded.")
                    # Reset ranking flag
                    self.is_ranking = False
                    return
                
                # Set the generating review flag
                self.is_generating_review = True
                self.is_ranking = False
                self.update_progress(50, "Analyzing clusters...")

                # Generate clusters and timeline
                cluster_infos, timeline_points = self.cluster_analyzer.cluster_papers_with_scores(combined_papers)

                # Format timeline data
                timeline_data = []
                current_year = datetime.now().year
                for point in timeline_points:
                    # Get papers in this cluster
                    cluster_info = cluster_infos.get(point.cluster_id)
                    if not cluster_info:
                        self.logger.warning(f"Skipping point with unknown cluster_id: {point.cluster_id}")
                        continue
                        
                    # Get the paper DOIs for this cluster
                    cluster_dois = cluster_info["papers"]
                    cluster_papers = []
                    
                    # Find the full paper data for each DOI in the cluster
                    for doi in cluster_dois:
                        paper = next((p for p in combined_papers if p.get('doi') == doi), None)
                        if paper:
                            cluster_papers.append(paper)
                    
                    paper_data = []
                    paper_years = []
                    
                    # Parse dates into years for average calculation
                    for paper in cluster_papers:
                        paper_date = None
                        try:
                            if paper.get('publication_date'):
                                paper_date = date_to_year(paper['publication_date'])
                                if paper_date and 1900 <= paper_date <= current_year + 1:  # Sanity check on year
                                    paper_years.append(paper_date)
                        except (ValueError, TypeError):
                            pass  # Skip if we can't parse the date
                    
                        # Add the paper data (moved inside the loop from outside)
                        paper_data.append({
                            'title': paper.get('title', 'Unknown Title'),
                            'authors': paper.get('authors', []),
                            'journal': paper.get('journal', 'Unknown Journal'),
                            'doi': paper.get('doi', ''),
                            'publication_date': paper.get('publication_date', ''),
                            'score': paper.get('composite_score', 0)
                        })
                
                    # Get the date for this cluster point
                    # Use the date from the point, or calculate from papers
                    point_date = None
                    try:
                        # Check if point.date is already a datetime object
                        if hasattr(point, 'date') and point.date:
                            if isinstance(point.date, datetime):
                                point_date = point.date
                            else:
                                # Try to parse it if it's a string
                                date_str = str(point.date)
                                # Handle different date formats
                                for fmt in ('%Y-%m-%d', '%Y-%m', '%Y'):
                                    try:
                                        point_date = datetime.strptime(date_str, fmt)
                                        break
                                    except ValueError:
                                        continue
                    except (ValueError, AttributeError) as e:
                        self.logger.debug(f"Error parsing date: {e}")
                        pass
                    
                    # If we couldn't parse the date, use average of paper years
                    if not point_date:
                        if paper_years:
                            avg_year = int(sum(paper_years) / len(paper_years))
                            point_date = datetime(avg_year, 1, 1)  # January 1st of the average year
                        else:
                            # If no valid years, default to current year
                            point_date = datetime(current_year, 1, 1)
                    
                    # Sanity check on the point date
                    if not paper_years or (point_date.year < 1950 or point_date.year > current_year + 1):
                        # If the cluster date is unreasonable, use the average year of papers
                        if paper_years:
                            avg_year = int(sum(paper_years) / len(paper_years))
                            point_date = datetime(avg_year, 1, 1)  # January 1st of the average year
                        else:
                            # If no valid years, default to current year
                            point_date = datetime(current_year, 1, 1)
                    
                    # Only add the timeline data if we have papers for this cluster
                    if paper_data:
                        timeline_data.append({
                            'date': point_date.strftime('%Y-%m-%d'),
                            'papers': paper_data,
                            'cluster_id': point.cluster_id
                        })

                self.timeline_data = timeline_data
                self.update_timeline(timeline_data)
                self.update_progress(75, "Generating review...")

                # Sort results by composite score and take top N based on user setting
                scored_results = []
                for result in search_results:
                    paper_score = next((p['composite_score'] for p in combined_papers if p['doi'].lower() == result.doi.lower()), 0)
                    scored_results.append((result, paper_score))

                scored_results.sort(key=lambda x: x[1], reverse=True)
                top_results = [result for result, _ in scored_results[:max_ranked_papers]]

                # Generate literature review
                review = await generate_grounded_review(
                    papers=top_results,
                    query=query,
                    quote_extractor=self.quote_extractor,
                    logger=logging.getLogger(__name__)
                )
                self.grounded_review = review
                self.update_literature_review(review)
                
                # Show the literature review tab
                self.results_tabs.setCurrentIndex(2)  # Switch to literature review tab
                
                self.status_label.setText("Search complete")
                self.update_progress(100, "Complete")
                
            except Exception as e:
                error_msg = f"Error during search: {str(e)}"
                logging.error(error_msg, exc_info=True)
                self.status_label.setText("Error")
                self.update_progress(0, "Error")
                QMessageBox.critical(self, "Error", error_msg)
            finally:
                # Reset operation flags
                self.is_searching = False
                self.is_ranking = False
                self.is_generating_review = False

    def clear_layout(self, layout):
        """Helper function to clear all widgets from a layout."""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    @asyncSlot()
    async def load_modules_async(self):
        """Load analysis modules asynchronously in the correct order - basic first, then advanced."""
        self.moduleLoadingStatusUpdated.emit("Loading basic search modules...", False)
        self.is_loading_modules = True
        
        try:
            # First ensure basic search components are loaded (these are lightweight)
            if not self.doi_manager:
                self.doi_manager = await asyncio.to_thread(DOIManager)
                self.moduleLoadingStatusUpdated.emit("Initialized DOI manager...", False)
                
            if not self.pubmed_client:
                self.pubmed_client = await asyncio.to_thread(
                    PubMedClient,
                    email=secrets.get('entrez_email', ''),
                    api_key=secrets.get('entrez_api_key'),
                    logger=logging.getLogger(__name__)
                )
                self.moduleLoadingStatusUpdated.emit("Initialized PubMed client...", False)
            
            # Now load the more intensive modules for review generation
            self.moduleLoadingStatusUpdated.emit("Loading advanced modules...", False)
            
            # Set up each module asynchronously
            self.cluster_analyzer = await asyncio.to_thread(
                PaperClusterAnalyzer, logging.getLogger(__name__)
            )
            self.moduleLoadingStatusUpdated.emit("Initialized cluster analyzer...", False)
            
            self.quote_extractor = await asyncio.to_thread(
                QuoteExtractor, logging.getLogger(__name__)
            )
            self.moduleLoadingStatusUpdated.emit("Initialized quote extractor...", False)
            
            self.modules_loaded = True
            self.is_loading_modules = False
            self.moduleLoadingStatusUpdated.emit("All modules loaded successfully", True)
            logging.info("Modules loaded successfully")
            
        except Exception as e:
            self.is_loading_modules = False
            # Even if advanced modules fail, we might still have basic search capability
            if self.doi_manager and self.pubmed_client:
                error_msg = f"Failed to load advanced modules: {str(e)}"
                self.moduleLoadingStatusUpdated.emit(error_msg, True)
                logging.error(f"Failed to load advanced modules: {e}", exc_info=True)
                QMessageBox.warning(self, "Warning", f"{error_msg}\n\nBasic search functionality is still available, but literature review generation will be disabled.")
                # Update the UI to reflect that we don't have full module capability
                self.update_modules_visibility()
            else:
                error_msg = f"Failed to load modules: {str(e)}"
                self.moduleLoadingStatusUpdated.emit(error_msg, True)
                logging.error(f"Failed to load modules: {e}", exc_info=True)
                QMessageBox.critical(self, "Error", error_msg)

    def update_module_loading_status(self, message, is_complete):
        """Update UI based on module loading status"""
        self.status_label.setText(message)
        
        if is_complete:
            if self.modules_loaded:
                self.modules_button.setText("Unload Modules")
                self.modules_button.setIcon(load_bootstrap_icon("box-arrow-left"))
                self.search_button.setEnabled(True)
                self.progress_bar.setValue(100)
                self.progress_bar.setFormat("Modules loaded")
            else:
                self.modules_button.setEnabled(True)
                self.modules_button.setIcon(load_bootstrap_icon("box"))
                self.progress_bar.setValue(0)
                self.progress_bar.setFormat("Modules unloaded")
        else:
            # Update progress bar during loading
            current = self.progress_bar.value()
            if current < 90:  # Avoid reaching 100% before completion
                self.progress_bar.setValue(current + 10)
            self.progress_bar.setFormat(message)
            
        # Update button icon to reflect loading state
        if not is_complete and not self.modules_loaded:
            self.modules_button.setIcon(load_bootstrap_icon("hourglass-split"))

    @asyncSlot()
    async def toggle_modules(self):
        """Toggle loading/unloading of analysis modules"""
        # Prevent multiple clicks during loading
        if self.is_loading_modules:
            return
            
        if not self.modules_loaded:
            self.is_loading_modules = True
            self.modules_button.setEnabled(False)
            self.modules_button.setIcon(load_bootstrap_icon("hourglass-split"))
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Starting module load...")
            
            # Start the asynchronous loading process
            await self.load_modules_async()
        else:
            # Unloading is quick, can be done synchronously
            # Check if any operations are in progress that require modules
            if self.is_searching or self.is_ranking or self.is_generating_review:
                QMessageBox.warning(self, "Modules in Use", "Cannot unload modules while operations are in progress.")
                return
                
            # For basic search we keep DOI manager and PubMed client
            # Only unload advanced modules if not needed for generating reviews
            if not self.generate_review_checkbox.isChecked():
                # Keep basic search modules, only unload advanced modules
                self.cluster_analyzer = None
                self.quote_extractor = None
                self.modules_loaded = False
                self.status_label.setText("Basic search modules available")
                self.modules_button.setIcon(load_bootstrap_icon("box"))
                logging.info("Advanced modules unloaded, basic search modules still available")
            else:
                # If review generation is selected, warn the user they need modules
                response = QMessageBox.question(
                    self, 
                    "Confirm Unload", 
                    "Review generation is selected, which requires modules. Unload modules anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                
                if response == QMessageBox.StandardButton.Yes:
                    # Unload everything
                    self.cluster_analyzer = None
                    self.quote_extractor = None
                    self.doi_manager = None
                    self.pubmed_client = None
                    self.modules_loaded = False
                    self.search_button.setEnabled(False)
                    self.status_label.setText("Modules unloaded")
                    self.modules_button.setIcon(load_bootstrap_icon("box"))
                    logging.info("All modules unloaded")
                else:
                    # User canceled unloading
                    return
            
            # Update UI to reflect current state
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Idle")
            self.update_modules_visibility()

    def toggle_paper_view(self):
        """Toggle between all papers and ranked papers views."""
        # Reset pagination when toggling views
        self.current_page = 0
        
        # Reset total pages based on which view is selected
        if self.all_papers_radio.isChecked():
            self.total_pages = (len(self.all_papers) + self.papers_per_page - 1) // self.papers_per_page
            self.display_papers(self.all_papers, is_ranked=False)
        else:
            self.total_pages = (len(self.ranked_papers) + self.papers_per_page - 1) // self.papers_per_page
            self.display_papers(self.ranked_papers, is_ranked=True)
            
    def previous_page(self):
        """Go to the previous page of papers."""
        if self.current_page > 0:
            self.current_page -= 1
            
            if self.all_papers_radio.isChecked():
                self.display_papers(self.all_papers, is_ranked=False)
            else:
                self.display_papers(self.ranked_papers, is_ranked=True)
                
    def next_page(self):
        """Go to the next page of papers."""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            
            if self.all_papers_radio.isChecked():
                self.display_papers(self.all_papers, is_ranked=False)
            else:
                self.display_papers(self.ranked_papers, is_ranked=True)
                
    def update_pagination_controls(self):
        """Update pagination controls based on current state."""
        # Update page label
        self.page_label.setText(f"Page {self.current_page + 1} of {max(1, self.total_pages)}")
        
        # Enable/disable pagination buttons
        self.prev_page_button.setEnabled(self.current_page > 0)
        self.next_page_button.setEnabled(self.current_page < self.total_pages - 1)

    def display_papers(self, papers, is_ranked=False):
        """Display papers in the papers container with pagination."""
        self.clear_layout(self.papers_layout)
        
        if not papers:
            self.papers_layout.addWidget(QLabel("No papers available."))
            self.total_pages = 0
            self.update_pagination_controls()
            return
            
        # Calculate pagination
        self.total_pages = (len(papers) + self.papers_per_page - 1) // self.papers_per_page
        
        # Ensure current_page is valid
        if self.current_page >= self.total_pages:
            self.current_page = max(0, self.total_pages - 1)
            
        # Update pagination controls
        self.update_pagination_controls()
            
        # Add title with paper count
        title_text = "All Papers" if not is_ranked else "Ranked Papers"
        title_label = QLabel(f"{title_text} ({len(papers)})")
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        self.papers_layout.addWidget(title_label)
        
        # Calculate start and end indices for current page
        start_idx = self.current_page * self.papers_per_page
        end_idx = min(start_idx + self.papers_per_page, len(papers))
        
        # Get papers for current page
        current_page_papers = papers[start_idx:end_idx]
        
        # Add page info
        page_info = QLabel(f"Showing papers {start_idx + 1}-{end_idx} of {len(papers)}")
        page_info.setStyleSheet("font-style: italic;")
        self.papers_layout.addWidget(page_info)
        
        # Add papers for current page
        for paper in current_page_papers:
            paper_widget = PaperWidget(paper, is_ranked=is_ranked)
            self.papers_layout.addWidget(paper_widget)

    def update_timeline(self, timeline_data):
        """Update the timeline visualization."""
        # Create a color map for clusters
        unique_clusters = set(point['cluster_id'] for point in timeline_data)
        cmap = plt.cm.get_cmap('viridis', max(3, len(unique_clusters)))
        self.cluster_colors = {cluster_id: cmap(i/len(unique_clusters)) 
                              for i, cluster_id in enumerate(unique_clusters)}
        
        # Plot the timeline
        self.timeline_canvas.plot_data(timeline_data, self.cluster_colors)

    def update_literature_review(self, review):
        """Update the literature review display."""
        self.clear_layout(self.review_layout)
        
        if not review:
            self.review_layout.addWidget(QLabel("No literature review available."))
            return
            
        # Add title
        title_label = QLabel(review.title)
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        self.review_layout.addWidget(title_label)
        
        # Add introduction
        intro_group = QGroupBox("Introduction")
        intro_layout = QVBoxLayout(intro_group)
        intro_text = QLabel(review.introduction)
        intro_text.setWordWrap(True)
        intro_layout.addWidget(intro_text)
        self.review_layout.addWidget(intro_group)
        
        # Add sections
        for section in review.sections:
            section_widget = ReviewSectionWidget(section)
            self.review_layout.addWidget(section_widget)
        
        # Add conclusion
        conclusion_group = QGroupBox("Conclusion")
        conclusion_layout = QVBoxLayout(conclusion_group)
        conclusion_text = QLabel(review.conclusion)
        conclusion_text.setWordWrap(True)
        conclusion_layout.addWidget(conclusion_text)
        self.review_layout.addWidget(conclusion_group)
        
        # Add citations
        if review.citations:
            citations_group = QGroupBox("References")
            citations_layout = QVBoxLayout(citations_group)
            citations_label = QLabel(f"{len(review.citations)} references")
            citations_label.setToolTip("\n".join(review.citations[:5]) + 
                                      ("\n..." if len(review.citations) > 5 else ""))
            citations_layout.addWidget(citations_label)
            self.review_layout.addWidget(citations_group)

    def update_progress(self, value: int, message: str = ""):
        """Update the progress bar value and displayed message."""
        self.progress_bar.setValue(value)
        if message:
            self.progress_bar.setFormat(message)
        else:
            self.progress_bar.setFormat(f"{value}%")

    def change_papers_per_page(self):
        """Change the papers per page setting."""
        self.papers_per_page = self.papers_per_page_spinner.value()
        self.current_page = 0
        
        # Update total pages and redisplay papers
        if self.all_papers_radio.isChecked():
            self.total_pages = (len(self.all_papers) + self.papers_per_page - 1) // self.papers_per_page
            self.display_papers(self.all_papers, is_ranked=False)
        else:
            self.total_pages = (len(self.ranked_papers) + self.papers_per_page - 1) // self.papers_per_page
            self.display_papers(self.ranked_papers, is_ranked=True)

def date_to_year(date_str):
    """
    Extract year from various date formats.
    Handles formats like: '2023', '2023-05', '2023-May', '2023-05-15', '2023-May-15', etc.
    
    Args:
        date_str: String date in various formats
        
    Returns:
        int: Year extracted from the date string, or None if parsing fails
    """
    if not date_str or date_str == 'N/A':
        return None
    
    # Log input for debugging
    logging.debug(f"Attempting to extract year from date: '{date_str}'")
    
    # Try to extract just the year if it's the only thing in the string
    if re.match(r'^\d{4}$', date_str):
        year = int(date_str)
        logging.debug(f"Extracted year using single year pattern: {year}")
        return year
    
    # Handle formats like 2024-Nov-07, 2024-Mar-20
    month_pattern = r'^(\d{4})-([A-Za-z]{3,})-(\d{1,2})$'
    match = re.match(month_pattern, date_str)
    if match:
        year = int(match.group(1))
        logging.debug(f"Extracted year using month name pattern: {year}")
        return year
    
    # Try different date formats
    date_formats = [
        '%Y-%m-%d',      # 2023-05-15
        '%Y-%b-%d',      # 2023-May-15
        '%Y-%B-%d',      # 2023-May-15
        '%Y-%m',         # 2023-05
        '%Y-%b',         # 2023-May
        '%Y-%B',         # 2023-May
        '%b %Y',         # May 2023
        '%B %Y',         # May 2023
        '%m/%d/%Y',      # 05/15/2023
        '%d/%m/%Y',      # 15/05/2023
        '%Y/%m/%d',      # 2023/05/15
    ]
    
    for fmt in date_formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            logging.debug(f"Extracted year using format {fmt}: {dt.year}")
            return dt.year
        except ValueError:
            continue
    
    # Try to extract year with regex if all else fails
    year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
    if year_match:
        year = int(year_match.group(0))
        logging.debug(f"Extracted year using regex fallback: {year}")
        return year
    
    logging.debug(f"Failed to extract year from date: '{date_str}'")
    return None
