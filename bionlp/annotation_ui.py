import sys
import os
import json
import requests
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTextEdit, QPushButton, QLabel, QSplitter, QTabWidget,
    QTreeWidget, QTreeWidgetItem, QFileDialog, QMessageBox, QCheckBox,
    QComboBox, QGroupBox, QGridLayout, QProgressDialog, QFrame, QLineEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import QColor, QTextCharFormat, QFont, QSyntaxHighlighter

# Import the get_secrets_from_database function from portal.py
from admin.portal import get_secrets_from_database

class EntityHighlighter(QSyntaxHighlighter):
    """Custom syntax highlighter for medical entities in text"""
    
    def __init__(self, document=None):
        super().__init__(document)
        self.entity_positions = []
        self.entity_colors = {
            "umls": QColor(100, 200, 100, 80),  # Light green
            "negated": QColor(255, 100, 100, 80)  # Brighter red
        }
        
    def set_entity_positions(self, entity_positions):
        """Set entity positions to highlight
        
        Args:
            entity_positions: List of tuples (start, end, entity_type)
        """
        self.entity_positions = entity_positions
        self.rehighlight()
        
    def highlightBlock(self, text):
        """Highlight the entities in the text"""
        for start, end, entity_type in self.entity_positions:
            if start < len(text) and end <= len(text) and start <= end:
                fmt = QTextCharFormat()
                color = self.entity_colors.get(entity_type, QColor(200, 200, 200, 80))
                fmt.setBackground(color)
                fmt.setFontWeight(QFont.Weight.Bold)
                self.setFormat(start, end - start, fmt)

class UmlsApiClient:
    """Simple client for the UMLS API"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://uts-ws.nlm.nih.gov/rest"
        self.service_url = f"{self.base_url}/search/current"
        self.auth_ticket = None
        
    def get_auth_ticket(self):
        """Get authentication ticket for subsequent requests"""
        auth_endpoint = f"{self.base_url}/auth"
        
        try:
            # Use a timeout to prevent hanging indefinitely
            response = requests.post(
                auth_endpoint,
                data={"apikey": self.api_key},
                timeout=10  # 10-second timeout
            )
            response.raise_for_status()
            self.auth_ticket = response.json().get('ticket')
            return self.auth_ticket
        except requests.exceptions.Timeout:
            raise TimeoutError("UMLS API authentication timed out. Please check your network connection.")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"UMLS API connection error: {str(e)}")
        except Exception as e:
            raise ValueError(f"UMLS API authentication failed: {str(e)}")
            
    def search_term(self, term, search_type="words"):
        """Search for a term in UMLS
        
        Args:
            term: The search term
            search_type: Type of search ('words', 'exact', etc.)
            
        Returns:
            List of matching concepts
        """
        if not self.auth_ticket:
            self.get_auth_ticket()
            
        if not self.auth_ticket:
            return []
            
        try:
            params = {
                "string": term,
                "ticket": self.auth_ticket,
                "searchType": search_type
            }
            
            # Use a timeout to prevent hanging
            response = requests.get(
                self.service_url, 
                params=params,
                timeout=5  # 5-second timeout
            )
            response.raise_for_status()
            
            results = response.json().get('result', {}).get('results', [])
            return results
        except requests.exceptions.Timeout:
            raise TimeoutError(f"UMLS API search timed out for term: {term}")
        except Exception as e:
            raise ValueError(f"UMLS API search error: {str(e)}")

class UmlsInitThread(QThread):
    """Thread for initializing UMLS client without blocking UI"""
    success = pyqtSignal(object)  # Signal with UMLS client object
    error = pyqtSignal(str)       # Signal with error message
    
    def __init__(self, api_key):
        super().__init__()
        self.api_key = api_key
        
    def run(self):
        try:
            # Create client
            client = UmlsApiClient(self.api_key)
            
            # Try to authenticate
            if client.get_auth_ticket():
                self.success.emit(client)
            else:
                self.error.emit("Failed to authenticate with UMLS API")
        except Exception as e:
            self.error.emit(str(e))

class ProcessingThread(QThread):
    """Thread for processing text without blocking the UI"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int)  # current, total
    
    def __init__(self, umls_client, text):
        super().__init__()
        self.umls_client = umls_client
        self.text = text
        
    def run(self):
        try:
            # Simple text processing - split into words/phrases and search UMLS
            results = self.process_text()
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))
            
    def process_text(self):
        """Simple text processing to find medical terms in UMLS"""
        results = {
            "umls_concepts": []
        }
        
        # Very basic term extraction - split by spaces and search phrases
        words = self.text.split()
        total_items = len(words) + (len(words) - 1)  # Words + phrases
        processed = 0
        
        # Process single words
        for i, word in enumerate(words):
            # Skip very short words
            if len(word) < 3:
                processed += 1
                self.progress.emit(processed, total_items)
                continue
                
            # Clean the word
            clean_word = word.strip('.,;:()[]{}!?"\'-')
            if len(clean_word) < 3:
                processed += 1
                self.progress.emit(processed, total_items)
                continue
                
            # Search UMLS
            try:
                concepts = self.umls_client.search_term(clean_word, "exact")
                
                if concepts:
                    # Find the word in the original text for highlighting
                    start = self.text.find(clean_word)
                    if start >= 0:
                        # Get just one best concept
                        concept = concepts[0]
                        results["umls_concepts"].append({
                            "entity_text": clean_word,
                            "start_char": start,
                            "end_char": start + len(clean_word),
                            "cui": concept.get('ui', ''),
                            "score": 1.0,
                            "canonical_name": concept.get('name', ''),
                            "definition": "",  # Would need additional API call
                            "types": [],  # Would need additional API call
                            "linker_name": "umls"
                        })
            except Exception as e:
                # Log error but continue processing
                print(f"Error processing word '{clean_word}': {str(e)}")
                
            processed += 1
            self.progress.emit(processed, total_items)
                    
        # Process 2-word phrases
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            
            # Clean the phrase
            clean_phrase = phrase.strip('.,;:()[]{}!?"\'-')
            if len(clean_phrase) < 3:
                processed += 1
                self.progress.emit(processed, total_items)
                continue
                
            # Search UMLS
            try:
                concepts = self.umls_client.search_term(clean_phrase, "exact")
                
                if concepts:
                    # Find the phrase in the original text for highlighting
                    start = self.text.find(clean_phrase)
                    if start >= 0:
                        # Get just one best concept
                        concept = concepts[0]
                        results["umls_concepts"].append({
                            "entity_text": clean_phrase,
                            "start_char": start,
                            "end_char": start + len(clean_phrase),
                            "cui": concept.get('ui', ''),
                            "score": 1.0,
                            "canonical_name": concept.get('name', ''),
                            "definition": "",  # Would need additional API call
                            "types": [],  # Would need additional API call
                            "linker_name": "umls"
                        })
            except Exception as e:
                # Log error but continue processing
                print(f"Error processing phrase '{clean_phrase}': {str(e)}")
                
            processed += 1
            self.progress.emit(processed, total_items)
        
        return results

class BioNlpAnnotationUI(QWidget):
    """Main UI for the Biomedical Annotation tool"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Get API key from secrets but don't verify automatically
        self.secrets = get_secrets_from_database()
        self.umls_api_key = self.secrets.get('UMLS_API_KEY', '') or self.secrets.get('umls_api_key', '')
        self.umls_client = None
        self.initialization_success = False
        
        # Create UI components
        self.create_ui()
        
        # Connect signals
        self.connect_signals()
        
        # Show a helpful message for new users
        self.status_label.setText("Ready. Enter medical text and click 'Find Medical Terms' or try 'Sample' for a demo.")
        
        # Create sample medical text and results for demo purposes
        self.sample_text = """PATIENT MEDICAL RECORD
CHIEF COMPLAINT: 58-year-old male with type 2 diabetes mellitus presents with chest pain, dyspnea, and fatigue for the past 3 days.

HISTORY OF PRESENT ILLNESS: Patient reports intermittent substernal chest pain radiating to the left arm, associated with shortness of breath and diaphoresis. Pain is worse with exertion and partially relieved by rest. Patient has a history of myocardial infarction 5 years ago. He also notes polyuria and polydipsia for the past week with poor glycemic control.

MEDICATIONS: Metformin 1000mg BID, atorvastatin 40mg daily, lisinopril 20mg daily, aspirin 81mg daily."""
        
        # Sample UMLS response
        self.sample_results = {
            "umls_concepts": [
                {
                    "entity_text": "diabetes mellitus",
                    "start_char": 45,
                    "end_char": 62,
                    "cui": "C0011849",
                    "score": 1.0,
                    "canonical_name": "Diabetes Mellitus",
                    "linker_name": "umls"
                },
                {
                    "entity_text": "chest pain",
                    "start_char": 80,
                    "end_char": 90,
                    "cui": "C0008031",
                    "score": 1.0,
                    "canonical_name": "Chest Pain",
                    "linker_name": "umls"
                },
                {
                    "entity_text": "dyspnea",
                    "start_char": 92,
                    "end_char": 99,
                    "cui": "C0013404",
                    "score": 1.0,
                    "canonical_name": "Dyspnea",
                    "linker_name": "umls"
                },
                {
                    "entity_text": "fatigue",
                    "start_char": 105,
                    "end_char": 112,
                    "cui": "C0015672",
                    "score": 1.0,
                    "canonical_name": "Fatigue",
                    "linker_name": "umls"
                },
                {
                    "entity_text": "substernal chest pain",
                    "start_char": 177,
                    "end_char": 198,
                    "cui": "C0235710",
                    "score": 1.0,
                    "canonical_name": "Substernal Chest Pain",
                    "linker_name": "umls"
                },
                {
                    "entity_text": "shortness of breath",
                    "start_char": 240,
                    "end_char": 259,
                    "cui": "C0013404",
                    "score": 1.0,
                    "canonical_name": "Dyspnea",
                    "linker_name": "umls"
                },
                {
                    "entity_text": "diaphoresis",
                    "start_char": 265,
                    "end_char": 276,
                    "cui": "C0011991",
                    "score": 1.0,
                    "canonical_name": "Diaphoresis",
                    "linker_name": "umls"
                },
                {
                    "entity_text": "exertion",
                    "start_char": 300,
                    "end_char": 308,
                    "cui": "C0015259",
                    "score": 1.0,
                    "canonical_name": "Exercise",
                    "linker_name": "umls"
                },
                {
                    "entity_text": "rest",
                    "start_char": 331,
                    "end_char": 335,
                    "cui": "C0035253",
                    "score": 1.0,
                    "canonical_name": "Rest",
                    "linker_name": "umls"
                },
                {
                    "entity_text": "myocardial infarction",
                    "start_char": 363,
                    "end_char": 384,
                    "cui": "C0027051",
                    "score": 1.0,
                    "canonical_name": "Myocardial Infarction",
                    "linker_name": "umls"
                },
                {
                    "entity_text": "polyuria",
                    "start_char": 409,
                    "end_char": 417,
                    "cui": "C0032617",
                    "score": 1.0,
                    "canonical_name": "Polyuria",
                    "linker_name": "umls"
                },
                {
                    "entity_text": "polydipsia",
                    "start_char": 422,
                    "end_char": 432,
                    "cui": "C0032483",
                    "score": 1.0,
                    "canonical_name": "Polydipsia",
                    "linker_name": "umls"
                },
                {
                    "entity_text": "glycemic control",
                    "start_char": 455,
                    "end_char": 471,
                    "cui": "C0403843",
                    "score": 1.0,
                    "canonical_name": "Glycemic Control",
                    "linker_name": "umls"
                },
                {
                    "entity_text": "Metformin",
                    "start_char": 485,
                    "end_char": 494,
                    "cui": "C0025598",
                    "score": 1.0,
                    "canonical_name": "Metformin",
                    "linker_name": "umls"
                },
                {
                    "entity_text": "atorvastatin",
                    "start_char": 509,
                    "end_char": 521,
                    "cui": "C0286651",
                    "score": 1.0,
                    "canonical_name": "Atorvastatin",
                    "linker_name": "umls"
                },
                {
                    "entity_text": "lisinopril",
                    "start_char": 536,
                    "end_char": 546,
                    "cui": "C0065374",
                    "score": 1.0,
                    "canonical_name": "Lisinopril",
                    "linker_name": "umls"
                },
                {
                    "entity_text": "aspirin",
                    "start_char": 561,
                    "end_char": 568,
                    "cui": "C0004057",
                    "score": 1.0,
                    "canonical_name": "Aspirin",
                    "linker_name": "umls"
                }
            ]
        }

    def initialize_umls_client(self):
        """Initialize the API client with key in a background thread"""
        # Clear status
        self.status_label.setText("Initializing API connection...")
        self.process_btn.setEnabled(False)
        
        # Validate key
        if not self.umls_api_key:
            self.status_label.setText("No API key entered. Please provide a valid key.")
            return
        
        # Start initialization in background thread
        self.init_thread = UmlsInitThread(self.umls_api_key)
        self.init_thread.success.connect(self.handle_init_success)
        self.init_thread.error.connect(self.handle_init_error)
        self.init_thread.start()
    
    def handle_init_success(self, client):
        """Handle successful API client initialization"""
        self.umls_client = client
        self.initialization_success = True
        self.status_label.setText("API client initialized successfully")
        self.process_btn.setEnabled(True)
        
        # Show success message
        QMessageBox.information(self, "Connection Success", 
                              "Successfully connected to term recognition API.\nYou can now process medical text.")
    
    def handle_init_error(self, error_msg):
        """Handle API client initialization error"""
        self.initialization_success = False
        self.status_label.setText(f"Error initializing API client: {error_msg}")
        self.process_btn.setEnabled(False)
        
        # Show error dialog
        QMessageBox.critical(self, "Initialization Error", 
                            f"Failed to initialize API client:\n{error_msg}\n\n"
                            "Please check your API key and network connection.")

    def create_ui(self):
        """Create the main UI layout and components"""
        # Main layout for the widget
        main_layout = QVBoxLayout(self)
        
        # Create a main splitter
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Create left panel (input area)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Text input area with toolbar
        input_group = QGroupBox("Medical Text Input")
        input_layout = QVBoxLayout(input_group)
        
        # Input toolbar
        input_toolbar = QWidget()
        toolbar_layout = QHBoxLayout(input_toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        
        # Load and Save buttons
        self.load_btn = QPushButton("Load Text")
        self.save_btn = QPushButton("Save Text")
        self.process_btn = QPushButton("Find Medical Terms")
        self.sample_btn = QPushButton("Sample")
        self.clear_btn = QPushButton("Clear")
        
        # Initialize with process button disabled since we're not auto-connecting
        self.process_btn.setEnabled(False)
        
        toolbar_layout.addWidget(self.load_btn)
        toolbar_layout.addWidget(self.save_btn)
        toolbar_layout.addWidget(self.sample_btn)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.clear_btn)
        toolbar_layout.addWidget(self.process_btn)
        
        # Text input
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter or load medical text here...")
        
        # Add custom highlighter
        self.highlighter = EntityHighlighter(self.text_input.document())
        
        input_layout.addWidget(input_toolbar)
        input_layout.addWidget(self.text_input)
        
        # Options
        options_group = QGroupBox("API Configuration")
        options_layout = QGridLayout(options_group)
        
        # Search type
        search_type_label = QLabel("Search Type:")
        self.search_type_combo = QComboBox()
        self.search_type_combo.addItems(["Exact Match", "Words", "Approximate"])
        self.search_type_combo.setCurrentText("Words")
        
        # Add API key input
        api_key_label = QLabel("API Key:")
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setText(self.umls_api_key)
        self.api_key_input.setPlaceholderText("Enter your API key here")
        self.set_key_btn = QPushButton("Update Key")
        
        # Add options to layout
        options_layout.addWidget(search_type_label, 0, 0)
        options_layout.addWidget(self.search_type_combo, 0, 1)
        options_layout.addWidget(api_key_label, 1, 0)
        options_layout.addWidget(self.api_key_input, 1, 1)
        options_layout.addWidget(self.set_key_btn, 1, 2)
        
        # Add API status
        self.api_status_label = QLabel("API Status: Not connected - Use 'Update Key' to connect or 'Sample' for demo")
        options_layout.addWidget(self.api_status_label, 2, 0, 1, 3)
        
        # Add all components to left panel
        left_layout.addWidget(input_group)
        left_layout.addWidget(options_group)
        
        # Create right panel (results area)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Tab widget for different result views
        self.results_tabs = QTabWidget()
        
        # Create tabs
        self.entity_tab = QWidget()
        self.json_tab = QWidget()
        
        # Entity tab
        entity_layout = QVBoxLayout(self.entity_tab)
        self.entity_tree = QTreeWidget()
        self.entity_tree.setHeaderLabels(["Medical Term", "ID", "Name"])
        self.entity_tree.setColumnWidth(0, 300)
        self.entity_tree.setColumnWidth(1, 100)
        self.entity_tree.setColumnWidth(2, 400)
        entity_layout.addWidget(self.entity_tree)
        
        # JSON tab
        json_layout = QVBoxLayout(self.json_tab)
        self.json_view = QTextEdit()
        self.json_view.setReadOnly(True)
        json_layout.addWidget(self.json_view)
        
        # Add tabs to tab widget
        self.results_tabs.addTab(self.entity_tab, "Medical Terms")
        self.results_tabs.addTab(self.json_tab, "JSON Data")
        
        # Add tab widget to right panel
        right_layout.addWidget(self.results_tabs)
        
        # Add status bar widget
        status_frame = QFrame()
        status_frame.setFrameShape(QFrame.Shape.StyledPanel)
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(5, 2, 5, 2)
        
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        
        right_layout.addWidget(status_frame)
        
        # Add panels to splitter
        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(right_panel)
        self.main_splitter.setSizes([400, 800])
        
        # Add main splitter to main layout
        main_layout.addWidget(self.main_splitter)
            
    def connect_signals(self):
        """Connect UI signals to handler methods"""
        self.load_btn.clicked.connect(self.load_text)
        self.save_btn.clicked.connect(self.save_text)
        self.process_btn.clicked.connect(self.process_text)
        self.clear_btn.clicked.connect(self.clear_text)
        self.sample_btn.clicked.connect(self.show_sample)
        self.entity_tree.itemClicked.connect(self.highlight_entity)
        self.set_key_btn.clicked.connect(self.update_umls_api_key)
    
    def show_sample(self):
        """Show sample medical text and a pre-defined response for demonstration"""
        # Set the sample text
        self.text_input.setText(self.sample_text)
        
        # Display sample results
        self.display_results(self.sample_results, None)
        
        # Update status
        self.status_label.setText("Showing sample results. No API key required for this demo.")
    
    def process_text(self):
        """Process the input text with the medical term recognition API"""
        # Check if API client is initialized
        if not self.initialization_success or not self.umls_client:
            QMessageBox.warning(self, "API Not Connected", 
                               "Medical term recognition API is not connected. Please update your API key first.")
            return
            
        text = self.text_input.toPlainText()
        
        if not text.strip():
            QMessageBox.warning(self, "Warning", "Please enter text to process.")
            return
            
        # Show processing dialog with progress bar
        progress = QProgressDialog("Processing text...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)  # Show immediately
        progress.setCancelButton(None)  # No cancel button to prevent issues
        progress.setAutoClose(True)
        progress.show()
        
        # Run processing in a separate thread
        self.processing_thread = ProcessingThread(self.umls_client, text)
        self.processing_thread.finished.connect(lambda result: self.display_results(result, progress))
        self.processing_thread.error.connect(lambda error: self.handle_processing_error(error, progress))
        self.processing_thread.progress.connect(lambda current, total: 
            progress.setValue(int(current / total * 100) if total > 0 else 0))
        self.processing_thread.start()

    def update_umls_api_key(self):
        """Update the UMLS API key with the one entered in the UI"""
        new_key = self.api_key_input.text().strip()
        if not new_key:
            QMessageBox.warning(self, "Invalid Key", "Please enter a valid UMLS API key")
            return
            
        self.umls_api_key = new_key
        self.api_status_label.setText("API Status: Verifying key...")
        
        # Initialize client with the new key
        self.initialize_umls_client()

    def load_text(self):
        """Load text from a file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Text File", "", "Text Files (*.txt);;All Files (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    self.text_input.setText(text)
                self.status_label.setText(f"Loaded text from {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
                
    def save_text(self):
        """Save text to a file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Text", "", "Text Files (*.txt);;All Files (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(self.text_input.toPlainText())
                self.status_label.setText(f"Saved text to {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")
    
    def clear_text(self):
        """Clear the input text and reset highlighter"""
        self.text_input.clear()
        self.highlighter.set_entity_positions([])
        self.clear_results()
        
    def clear_results(self):
        """Clear all results"""
        self.entity_tree.clear()
        self.json_view.clear()
        
    def handle_processing_error(self, error, progress):
        """Handle processing error"""
        progress.close()
        QMessageBox.critical(self, "Processing Error", f"Error processing text:\n{error}")
        self.status_label.setText(f"Error processing text: {error}")
        
    def display_results(self, results, progress):
        """Display the processing results"""
        # Close progress dialog if it exists
        if progress is not None:
            progress.close()
        
        # Clear previous results
        self.clear_results()
        
        # Display JSON results
        self.json_view.setText(json.dumps(results, indent=2))
        
        # Collect entity positions for highlighting
        entity_positions = []
        
        # Display UMLS entities
        umls_root = QTreeWidgetItem(self.entity_tree, ["Medical Entities"])
        umls_root.setExpanded(True)
        
        for concept in results.get('umls_concepts', []):
            item = QTreeWidgetItem(umls_root, [
                concept.get('entity_text', ''),
                concept.get('cui', ''),
                concept.get('canonical_name', '')
            ])
            
            # Add entity position for highlighting
            entity_positions.append((
                concept.get('start_char', 0),
                concept.get('end_char', 0),
                'umls'
            ))
        
        # Apply highlighting to the text
        self.highlighter.set_entity_positions(entity_positions)
        
        # Update status
        entity_count = len(results.get('umls_concepts', []))
        self.status_label.setText(f"Found {entity_count} medical terms in text.")
    
    def highlight_entity(self, item, column):
        """When an entity is clicked in the tree, highlight it in the text"""
        if not item.parent():  # Skip if top-level category item
            return
            
        # Get entity text and find in the document
        entity_text = item.text(0)
        if not entity_text:
            return
            
        # Find entity in text
        document = self.text_input.document()
        cursor = document.find(entity_text)
        
        if not cursor.isNull():
            # Select and scroll to the entity
            self.text_input.setTextCursor(cursor)
            self.text_input.ensureCursorVisible()

class StandaloneBioNlpUI(QMainWindow):
    """Standalone window version of the Biomedical Annotation tool for independent use"""
    
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.setWindowTitle("Biomedical Annotation Tool")
        self.setMinimumSize(1200, 800)
        
        # Create and set the annotation widget as central widget
        self.annotation_widget = BioNlpAnnotationUI(self)
        self.setCentralWidget(self.annotation_widget)
    
    def closeEvent(self, event):
        """Clean up resources when window is closed"""
        if hasattr(self.annotation_widget, 'processing_thread') and self.annotation_widget.processing_thread.isRunning():
            self.annotation_widget.processing_thread.terminate()
            self.annotation_widget.processing_thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StandaloneBioNlpUI()
    window.show()
    sys.exit(app.exec()) 