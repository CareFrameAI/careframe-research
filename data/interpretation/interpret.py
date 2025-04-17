from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, 
                             QTableWidgetItem, QPushButton, QLabel, QGroupBox, 
                             QSplitter, QTabWidget, QTextEdit, QComboBox,
                             QFormLayout, QTreeWidget, QTreeWidgetItem, QScrollArea,
                             QSizePolicy, QDialog, QListWidget, QListWidgetItem,
                             QGridLayout, QCheckBox, QFrame, QToolButton)
from PyQt6.QtCore import Qt, pyqtSignal, QItemSelectionModel, QSize, QTimer
from PyQt6.QtGui import QFont, QIcon, QColor

# Import the icon loading function
from helpers.load_icon import load_bootstrap_icon

import pandas as pd
import json
import markdown
from datetime import datetime
from llms.client import call_claude_sync, call_claude_async

# Function to convert markdown to HTML
def md_to_html(md_text):
    """Convert markdown text to HTML."""
    if not md_text:
        return ""
    html = markdown.markdown(md_text, extensions=['tables', 'fenced_code', 'codehilite'])
    return html

class InterpretationCard(QFrame):
    """A card widget representing an analysis in the grid."""
    
    clicked = pyqtSignal(object)
    selected = pyqtSignal(object, bool)
    
    def __init__(self, analysis, parent=None):
        super().__init__(parent)
        self.analysis = analysis
        self.is_selected = False
        self.init_ui()
        self.setup_animations()
        
    def init_ui(self):
        """Initialize the UI for the card."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Basic frame properties
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFrameShadow(QFrame.Shadow.Plain)
        self.setFixedSize(220, 140)  # Fixed size for consistent cards
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Header: stacked title inside a horizontal layout
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 8)
        header_layout.addStretch()
        outcome_name = self.analysis.get('outcome_name', 'Unknown Outcome')
        self.type_label = QLabel(outcome_name)
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        self.type_label.setFont(font)
        self.type_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Explicitly set flat styling for title label
        self.type_label.setStyleSheet("border: none; background-color: transparent;")
        header_layout.addWidget(self.type_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        layout.addStretch()
        
        # Date at the bottom
        timestamp = self.analysis.get('timestamp', '')
        date_str = ''
        if timestamp:
            try:
                date_obj = datetime.fromisoformat(timestamp)
                date_str = date_obj.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                date_str = timestamp
        self.date_label = QLabel(date_str)
        font = QFont()
        font.setPointSize(7)
        self.date_label.setFont(font)
        # Apply distinct styling directly to date label with no border/hover
        self.date_label.setStyleSheet("color: #6c757d; border: none; background-color: transparent;")
        self.date_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.date_label)
        
        # Frame stylesheet with completely flat styling for labels
        self.setStyleSheet("""
            QFrame { 
                border: 2px solid gray; 
                border-radius: 8px;
            }
            QFrame[selected="true"] {
                border: 2px solid orange;
                background-color: rgba(0, 123, 255, 0.1);
            }
            QFrame:hover:not([selected="true"]) {
                border: 1px solid #adb5bd;
                background-color: rgba(173, 181, 189, 0.1);
            }
            QFrame[selected="true"]:hover {
                border: 2px solid yellow;
                background-color: rgba(0, 123, 255, 0.15);
            }
        """)
        
    def setup_animations(self):
        """Setup hover (handled by stylesheet)"""
        self.setProperty("hovered", False)
        self.setStyleSheet(self.styleSheet())
    
    def enterEvent(self, event):
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        super().leaveEvent(event)
        
    def set_selected(self, selected):
        """Set selection state."""
        self.is_selected = selected
        self.setProperty("selected", selected)
        self.style().unpolish(self)
        self.style().polish(self)
    
    def mousePressEvent(self, event):
        """Toggle selection and emit signals."""
        self.set_selected(not self.is_selected)
        self.selected.emit(self, self.is_selected)
        self.clicked.emit(self)
        event.accept()

class InterpretationWidget(QWidget):
    """Widget for study interpretation and QA chat."""
    
    active_study_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.studies_manager = None
        self.chat_history = []  # conversation messages (role "user" or "assistant")
        self.current_analyses = []
        self.current_groups = []
        self.selected_items = []  # currently selected cards
        self.chat_metadata = {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "title": "New Interpretation Session",
            "study_id": None,
            "chat_id": f"chat_{datetime.now().timestamp()}"
        }
        self.init_ui()
        self.load_sample_analyses()
        
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Top toolbar section
        top_section = QWidget()
        top_section.setMaximumHeight(40)
        top_layout = QHBoxLayout(top_section)
        top_layout.setContentsMargins(5, 2, 5, 2)
        header_icon = QLabel()
        header_icon.setPixmap(load_bootstrap_icon("lightbulb", size=16).pixmap(16, 16))
        top_layout.addWidget(header_icon)
        header_label = QLabel("Study Interpretation")
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        header_label.setFont(font)
        top_layout.addWidget(header_label)
        study_label = QLabel("Study:")
        top_layout.addWidget(study_label)
        self.studies_combo = QComboBox()
        self.studies_combo.setMinimumWidth(300)
        self.studies_combo.currentIndexChanged.connect(self.on_study_selected)
        top_layout.addWidget(self.studies_combo)
        self.refresh_btn = QToolButton()
        self.refresh_btn.setIcon(load_bootstrap_icon("arrow-repeat"))
        self.refresh_btn.clicked.connect(self.refresh_studies_list)
        top_layout.addWidget(self.refresh_btn)
        
        # Add a spacer
        top_layout.addSpacing(20)
        
        # Add chat control buttons to the top row
        # New chat button
        new_chat_btn = QToolButton()
        new_chat_btn.setIcon(load_bootstrap_icon("plus-circle"))
        new_chat_btn.setIconSize(QSize(20, 20))
        new_chat_btn.setToolTip("New Chat")
        new_chat_btn.clicked.connect(self.start_new_chat)
        top_layout.addWidget(new_chat_btn)
        
        # View chat histories button
        view_chats_btn = QToolButton()
        view_chats_btn.setIcon(load_bootstrap_icon("clock-history"))
        view_chats_btn.setIconSize(QSize(20, 20))
        view_chats_btn.setToolTip("View Chat Histories")
        view_chats_btn.clicked.connect(self.view_chat_histories)
        top_layout.addWidget(view_chats_btn)
        
        # Synthesize button
        synthesize_btn = QToolButton()
        synthesize_btn.setIcon(load_bootstrap_icon("intersect"))
        synthesize_btn.setIconSize(QSize(20, 20))
        synthesize_btn.setToolTip("Synthesize Selected")
        synthesize_btn.clicked.connect(self.synthesize_selected_analyses)
        top_layout.addWidget(synthesize_btn)
        
        # Save Chat History button
        save_chat_btn = QToolButton()
        save_chat_btn.setIcon(load_bootstrap_icon("save2"))
        save_chat_btn.setIconSize(QSize(20, 20))
        save_chat_btn.setToolTip("Save Chat History")
        save_chat_btn.clicked.connect(self.save_chat_history)
        top_layout.addWidget(save_chat_btn)
        
        top_layout.addStretch()
        
        main_layout.addWidget(top_section)
        
        # Main splitter: left grid and right QA/chat area
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.splitterMoved.connect(self.on_splitter_moved)
        
        # Left panel: grid of interpretation cards
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_scroll = QScrollArea()
        self.grid_scroll.setWidgetResizable(True)
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(15)
        self.grid_layout.setContentsMargins(15, 15, 15, 15)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.grid_scroll.setWidget(self.grid_container)
        
        # Add resize handling for the grid container
        self.grid_container.resizeEvent = self.on_grid_container_resize
        
        left_layout.addWidget(self.grid_scroll)
        main_splitter.addWidget(left_panel)
        
        # Right panel: QA/chat and preview area
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 0, 5, 0)

        # QA/chat area (simpler layout without right buttons)
        chat_section = QWidget()
        chat_layout = QVBoxLayout(chat_section)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(5)
        
        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        # Remove border from chat display
        self.chat_display.setStyleSheet("border: 4px solid gray; border-radius: 8px;")
        chat_layout.addWidget(self.chat_display)
        
        welcome_html = """
        <div style='text-align:center; margin:20px;'>
            <h3>Welcome to Study Interpretation</h3>
            <p>Select analyses from the left panel to review or ask questions about your study data.</p>
            <p style='font-style:italic;'>Use Ctrl+Enter to send messages</p>
        </div>
        """
        self.chat_display.setHtml(welcome_html)
        
        # Chat input area
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(5)
        self.chat_input = QTextEdit()
        self.chat_input.setPlaceholderText("Ask a question about the analyses...")
        self.chat_input.setMaximumHeight(60)
        self.chat_input.setStyleSheet("border: 4px solid gray; border-radius: 8px;")
        self.chat_input.keyPressEvent = self.input_key_press_event
        input_layout.addWidget(self.chat_input)
        send_btn = QToolButton()
        send_btn.setIcon(load_bootstrap_icon("send"))
        send_btn.clicked.connect(self.send_chat_message)
        input_layout.addWidget(send_btn)
        chat_layout.addLayout(input_layout)
        
        right_layout.addWidget(chat_section)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([400, 600])
        main_layout.addWidget(main_splitter)
        
    def set_studies_manager(self, studies_manager):
        """Set the studies manager and refresh studies list."""
        self.studies_manager = studies_manager
        self.refresh_studies_list()
        
    def refresh_studies_list(self):
        """Refresh list of studies."""
        if not self.studies_manager:
            return
        studies = self.studies_manager.list_studies()
        self.studies_combo.clear()
        for study in studies:
            display_text = f"{study['name']} {'(Active)' if study['is_active'] else ''}"
            self.studies_combo.addItem(load_bootstrap_icon("file-earmark-text"), display_text, study["id"])
            
    def on_study_selected(self, index):
        """Handle study selection."""
        if index < 0:
            return
        study_id = self.studies_combo.itemData(index)
        study = self.studies_manager.get_study(study_id)
        if study:
            self.load_study_analyses(study)
            
            # Auto-load most recent chat history if available
            if hasattr(study, 'interpretation_chats') and study.interpretation_chats:
                # Sort by updated_at (newest first)
                sorted_chats = sorted(
                    study.interpretation_chats,
                    key=lambda x: x.get("metadata", {}).get("updated_at", ""),
                    reverse=True
                )
                
                if sorted_chats:
                    latest_chat = sorted_chats[0]
                    chat_id = latest_chat.get("metadata", {}).get("chat_id")
                    if chat_id:
                        # Load the most recent chat
                        self.load_chat_history(chat_id)
                        # Update status with a notice
                        self.show_status_message(f"Loaded recent chat: {latest_chat.get('metadata', {}).get('title', 'Untitled')}")
    
    def show_status_message(self, message, duration=3000):
        """Show a temporary status message at the bottom of the chat display."""
        current_html = self.chat_display.toHtml()
        status_html = f"""
        <div style="position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); 
                    background-color: rgba(0,0,0,0.7); color: white; padding: 8px 16px; 
                    border-radius: 4px; z-index: 1000;">
            {message}
        </div>
        """
        self.chat_display.setHtml(current_html + status_html)
        
        # Remove the status message after duration
        QTimer.singleShot(duration, lambda: self.chat_display.setHtml(current_html))
            
    def load_study_analyses(self, study):
        """Load all LLM analyses for the selected study."""
        self.clear_grid()
        self.current_analyses = []
        self.selected_items = []
        if hasattr(study, 'llm_analyses') and study.llm_analyses:
            sorted_analyses = sorted(
                study.llm_analyses, 
                key=lambda x: x.get('timestamp', ''), 
                reverse=True
            )
            for analysis in sorted_analyses:
                self.current_analyses.append(analysis)
                self.add_card_to_grid(analysis)
        else:
            sample_analyses = [
                {
                    'type': 'Statistical Analysis',
                    'outcome_name': 'Sample T-Test Results',
                    'content': 'The t-test analysis revealed a significant difference (t=3.45, p<0.001).',
                    'timestamp': datetime.now().isoformat(),
                    'dataset_name': 'Sample Dataset'
                },
                {
                    'type': 'Visualization Analysis',
                    'outcome_name': 'Distribution Plot',
                    'content': 'The distribution is right-skewed with potential outliers.',
                    'timestamp': datetime.now().isoformat(),
                    'dataset_name': 'Sample Dataset'
                },
                {
                    'type': 'Assumption Check',
                    'outcome_name': 'Normality Test',
                    'content': 'Shapiro-Wilk test indicates normal distribution (W=0.98, p=0.245).',
                    'timestamp': datetime.now().isoformat(),
                    'dataset_name': 'Sample Dataset'
                }
            ]
            for analysis in sample_analyses:
                self.current_analyses.append(analysis)
                self.add_card_to_grid(analysis)
            note_label = QLabel("Note: These are sample analyses for demonstration. Select a study to see actual analyses.")
            note_label.setStyleSheet("font-style: italic; padding: 10px;")
            note_label.setWordWrap(True)
            note_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.grid_layout.addWidget(note_label, self.grid_layout.rowCount(), 0, 1, 3)
            
    def clear_grid(self):
        """Clear all items from the grid layout."""
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
                
    def add_card_to_grid(self, analysis):
        """Add an analysis card to the grid."""
        card = InterpretationCard(analysis)
        card.clicked.connect(self.on_card_clicked)
        card.selected.connect(self.on_card_selected)
        # Set a fixed size for the card to ensure proper alignment
        card.setFixedSize(220, 140)
        # Add alignment and sizing policies
        card.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.grid_layout.addWidget(card)  # We'll reposition it in reorganize_grid
        self.reorganize_grid()
        
    def get_next_grid_position(self):
        """Calculate next grid position based on available width."""
        # This method is no longer needed as we're using reorganize_grid instead
        pass
        
    def reorganize_grid(self):
        """Reorganize all cards in the grid based on available width."""
        container_width = self.grid_container.width()
        card_width = 220  # Fixed card width
        spacing = self.grid_layout.spacing()
        
        # Calculate how many columns can fit in the available width
        # Account for spacing and some margin
        available_width = container_width - 30  # Allow for margins
        max_columns = max(1, available_width // (card_width + spacing))
        
        # Get all cards from the layout
        cards = []
        labels = []
        
        # First collect all widgets
        i = 0
        while i < self.grid_layout.count():
            item = self.grid_layout.itemAt(i)
            if item is None:
                i += 1
                continue
            
            widget = item.widget()
            if widget is None:
                i += 1
                continue
            
            if isinstance(widget, InterpretationCard):
                self.grid_layout.removeWidget(widget)
                cards.append(widget)
                # Don't increment i because removeWidget reduces the count
            elif isinstance(widget, QLabel):
                self.grid_layout.removeWidget(widget)
                labels.append(widget)
                # Don't increment i because removeWidget reduces the count
            else:
                i += 1
        
        # Add cards back in a grid that respects the calculated columns
        for i, card in enumerate(cards):
            row = i // max_columns
            col = i % max_columns
            self.grid_layout.addWidget(card, row, col, Qt.AlignmentFlag.AlignTop)
        
        # Add any labels at the bottom
        if cards and labels:
            for label in labels:
                last_row = (len(cards) - 1) // max_columns + 1
                self.grid_layout.addWidget(label, last_row, 0, 1, max_columns)
        
    def on_card_clicked(self, card):
        """Update preview pane when a card is clicked."""
        pass
    
    def on_card_selected(self, card, is_selected):
        """Update the selected items list and refresh QA window."""
        if is_selected:
            if card not in self.selected_items:
                self.selected_items.append(card)
        else:
            if card in self.selected_items:
                self.selected_items.remove(card)
        self._update_chat_display()
        
    def display_analysis_preview(self, analysis):
        """Display a preview of the selected analysis."""
        pass
            
    def synthesize_selected_analyses(self):
        """Synthesize multiple selected analyses (requires at least 2 selected cards)."""
        if not self.selected_items or len(self.selected_items) < 2:
            self.show_message_dialog("Selection Required", "Please select at least two analyses to synthesize.")
            return
        selected_analyses = []
        for card in self.selected_items:
            analysis = card.analysis
            if not analysis.get('is_group', False):
                selected_analyses.append(analysis)
            else:
                group_data = analysis.get('group_data', {})
                for group_item in group_data.get('items', []):
                    selected_analyses.append(group_item)
        if not selected_analyses:
            self.show_message_dialog("No Analyses", "No valid analyses found in selection.")
            return
        self.chat_display.append("Synthesizing analyses...")
        prompt = self._create_synthesis_prompt(selected_analyses)
        try:
            response = call_claude_sync(prompt)
            self.chat_history.append({"role": "user", "content": "Synthesize these analyses and identify key insights."})
            self.chat_history.append({"role": "assistant", "content": response})
            self._update_chat_display()
        except Exception as e:
            self.chat_display.append(f"Error: {str(e)}")
            
    def _create_synthesis_prompt(self, analyses):
        """Create a synthesis prompt from multiple analyses."""
        study_id = self.studies_combo.itemData(self.studies_combo.currentIndex())
        study = self.studies_manager.get_study(study_id)
        study_name = study.name if hasattr(study, 'name') else "Unknown Study"
        prompt = f"""As a scientific research assistant, synthesize these {len(analyses)} analyses from the study "{study_name}".
        
For each analysis, I'll provide:
1. The type of analysis
2. The outcome being analyzed 
3. The analysis content

Here are the analyses:
"""
        for i, analysis in enumerate(analyses):
            analysis_type = analysis.get('type', 'Unknown Analysis')
            outcome_name = analysis.get('outcome_name', 'Unknown Outcome')
            content = analysis.get('content', 'No content available')
            prompt += f"\n--- ANALYSIS {i+1}: {analysis_type} for {outcome_name} ---\n"
            prompt += content + "\n"
        prompt += "\nPlease provide a synthesis focused on common themes and key insights."
        return prompt
        
    def send_chat_message(self):
        """Send a chat message from the input."""
        message = self.chat_input.toPlainText().strip()
        if not message:
            return
        self.chat_input.clear()
        
        # Add timestamp and structure to message
        message_data = {
            "role": "user", 
            "content": message,
            "timestamp": datetime.now().isoformat()
        }
        self.chat_history.append(message_data)
        self._update_chat_display()
        self.chat_display.append("Thinking...")
        
        # Update metadata
        self.chat_metadata["updated_at"] = datetime.now().isoformat()
        study_id = self.studies_combo.itemData(self.studies_combo.currentIndex())
        if study_id:
            self.chat_metadata["study_id"] = study_id
        
        selected_analyses = []
        for card in self.selected_items:
            analysis = card.analysis
            if not analysis.get('is_group', False):
                selected_analyses.append(analysis)
            else:
                group_data = analysis.get('group_data', {})
                for group_item in group_data.get('items', []):
                    selected_analyses.append(group_item)
        
        prompt = self._create_chat_prompt(message, selected_analyses)
        try:
            response = call_claude_sync(prompt)
            response_data = {
                "role": "assistant", 
                "content": response,
                "timestamp": datetime.now().isoformat()
            }
            self.chat_history.append(response_data)
            self._update_chat_display()
        except Exception as e:
            self.chat_display.append(f"Error: {str(e)}")
            
    def _create_chat_prompt(self, message, selected_analyses):
        """Create a chat prompt including context from selected analyses."""
        study_id = self.studies_combo.itemData(self.studies_combo.currentIndex())
        study = self.studies_manager.get_study(study_id)
        study_name = study.name if hasattr(study, 'name') else "Unknown Study"
        prompt = f"""You are an expert research assistant helping a scientist interpret analyses from the study "{study_name}".
        
"""
        if self.chat_history:
            prompt += "Here's our conversation so far:\n\n"
            for entry in self.chat_history:
                role = "User" if entry["role"] == "user" else "Assistant"
                prompt += f"{role}: {entry['content']}\n\n"
        if selected_analyses:
            prompt += f"\nThe user has selected {len(selected_analyses)} analyses for context:\n\n"
            for i, analysis in enumerate(selected_analyses):
                analysis_type = analysis.get('type', 'Unknown Analysis')
                outcome_name = analysis.get('outcome_name', 'Unknown Outcome')
                content = analysis.get('content', 'No content available')
                prompt += f"--- ANALYSIS {i+1}: {analysis_type} for {outcome_name} ---\n"
                prompt += content + "\n\n"
        else:
            prompt += "Here are summaries of some analyses in this study for context:\n\n"
            for i, analysis in enumerate(self.current_analyses[:5]):
                analysis_type = analysis.get('type', 'Unknown Analysis')
                outcome_name = analysis.get('outcome_name', 'Unknown Outcome')
                prompt += f"- Analysis {i+1}: {analysis_type} for {outcome_name}\n"
            if len(self.current_analyses) > 5:
                prompt += f"- (Plus {len(self.current_analyses) - 5} more analyses not shown)\n"
            prompt += "\n"
        prompt += f"Now please respond to the following question:\n\n{message}\n\n"
        prompt += "Provide a clear and helpful response based on the analyses and our conversation."
        return prompt
        
    def _update_chat_display(self):
        """Update the QA window with the current chat conversation and selected card grid."""
        style = """
        <style>
            body { color: gray; font-family: sans-serif; }
            h1, h2, h3, h4, h5, h6 { color: gray; }
            pre { padding: 10px; border-radius: 5px; }
            code { padding: 2px 4px; border-radius: 3px; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 15px; }
            td, th { border: 1px solid gray; padding: 5px; text-align: left; }
            .analysis-preview { border: 1px solid gray; border-radius: 5px; padding: 10px; margin-bottom: 10px; }
        </style>
        """
        html = style

        # First, show detailed content for all selected analyses (not just titles)
        if self.selected_items:
            html += "<h3>Selected Analyses</h3>"
            for i, card in enumerate(self.selected_items):
                analysis = card.analysis
                title = analysis.get('outcome_name', 'Unknown')
                analysis_type = analysis.get('type', 'Analysis')
                content = analysis.get('content', 'No content available')
                # Format content as HTML but limit length
                if len(content) > 500:
                    content = content[:500] + "... (content truncated)"
                content_html = md_to_html(content)
                
                html += f"""
                <div class="analysis-preview">
                  <h4>{title}</h4>
                  <p><strong>Type:</strong> {analysis_type}</p>
                  {content_html}
                </div>
                """
            html += "<hr>"

        # Append conversation messages (only user and assistant)
        conversation = [entry for entry in self.chat_history if entry["role"] in ("user", "assistant")]
        if conversation:
            html += "<h3>Conversation</h3>"
            for entry in conversation:
                if entry["role"] == "user":
                    html += f"<div style='padding:8px; margin:5px 0; border-radius:5px;'><strong>You:</strong> {entry['content']}</div>"
                else:
                    html += f"<div style='padding:8px; margin:5px 0;'><strong>Assistant:</strong> {md_to_html(entry['content'])}</div>"
        elif not self.selected_items:
            # Only show welcome message if no conversation AND no selected items
            welcome_html = """
            <div style='text-align:center; margin:20px;'>
                <h3>Welcome to Study Interpretation</h3>
                <p>Select analyses from the left panel to review or ask questions about your study data.</p>
                <p style='font-style:italic;'>Use Ctrl+Enter to send messages</p>
            </div>
            """
            html += welcome_html

        # Set the HTML content
        self.chat_display.setHtml(html)
        
        # Scroll to the bottom
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def create_group(self):
        """Create a group from selected items."""
        if not self.selected_items or len(self.selected_items) < 2:
            self.show_message_dialog("Selection Required", "Please select at least two items to create a group.")
            return
        study_id = self.studies_combo.itemData(self.studies_combo.currentIndex())
        if not study_id:
            self.show_message_dialog("No Study Selected", "Please select a study first.")
            return
        study = self.studies_manager.get_study(study_id)
        if not study:
            self.show_message_dialog("Study Not Found", "The selected study could not be found.")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Create Group")
        dialog.resize(400, 300)
        layout = QVBoxLayout(dialog)
        form_layout = QFormLayout()
        name_input = QTextEdit()
        name_input.setPlaceholderText("Enter group name")
        name_input.setMaximumHeight(60)
        form_layout.addRow("Group Name:", name_input)
        desc_input = QTextEdit()
        desc_input.setPlaceholderText("Enter group description")
        form_layout.addRow("Description:", desc_input)
        layout.addLayout(form_layout)
        layout.addWidget(QLabel(f"Selected Items ({len(self.selected_items)}):"))
        items_list = QListWidget()
        for card in self.selected_items:
            analysis = card.analysis
            item_text = f"{analysis.get('type', 'Analysis')}: {analysis.get('outcome_name', 'Unknown')}"
            items_list.addItem(item_text)
        layout.addWidget(items_list)
        buttons_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        buttons_layout.addWidget(cancel_btn)
        create_btn = QPushButton("Create Group")
        create_btn.clicked.connect(dialog.accept)
        buttons_layout.addWidget(create_btn)
        layout.addLayout(buttons_layout)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            group_name = name_input.toPlainText().strip() or "Untitled Group"
            group_desc = desc_input.toPlainText().strip()
            group_items = []
            cards_to_remove = []
            for card in self.selected_items:
                analysis = card.analysis
                if not analysis.get('is_group', False):
                    group_items.append(analysis)
                    if analysis in self.current_analyses:
                        self.current_analyses.remove(analysis)
                    cards_to_remove.append(card)
                else:
                    group_data = analysis.get('group_data', {})
                    for group_item in group_data.get('items', []):
                        group_items.append(group_item)
            group_data = {
                'name': group_name,
                'description': group_desc,
                'timestamp': datetime.now().isoformat(),
                'items': group_items
            }
            try:
                if not hasattr(study, 'interpretation_groups'):
                    study.interpretation_groups = []
                study.interpretation_groups.append(group_data)
                study.updated_at = datetime.now().isoformat()
                self.current_groups.append(group_data)
                for card in cards_to_remove:
                    card.deleteLater()
                self.selected_items = []
            except Exception as e:
                self.show_message_dialog("Error", f"Failed to create group: {str(e)}")
        
    def start_new_chat(self):
        """Start a new chat by clearing the conversation history."""
        # Prompt for chat title
        dialog = QDialog(self)
        dialog.setWindowTitle("New Chat")
        dialog.resize(400, 150)
        dialog_layout = QVBoxLayout(dialog)
        
        form_layout = QFormLayout()
        title_input = QTextEdit()
        title_input.setPlaceholderText("Enter chat title")
        title_input.setMaximumHeight(60)
        form_layout.addRow("Title:", title_input)
        dialog_layout.addLayout(form_layout)
        
        # Add a label explaining the purpose
        info_label = QLabel("Creating a new chat will clear your current conversation. "
                           "Make sure to manually save your current chat if needed before starting a new one.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-style: italic; color: #6c757d;")
        dialog_layout.addWidget(info_label)
        
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        create_btn = QPushButton("Create")
        create_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(create_btn)
        dialog_layout.addLayout(button_layout)
        
        # Show dialog and handle result
        if dialog.exec() == QDialog.DialogCode.Accepted:
            title = title_input.toPlainText().strip() or "New Interpretation Session"
            
            # Clear chat history
            self.chat_history = []
            self.chat_display.clear()
            
            # Reset chat metadata with new title
            self.chat_metadata = {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "title": title,
                "study_id": self.studies_combo.itemData(self.studies_combo.currentIndex()),
                "chat_id": f"chat_{datetime.now().timestamp()}"
            }
            
            welcome_html = """
            <div style='text-align:center; margin:20px;'>
                <h3>Welcome to Study Interpretation</h3>
                <p>Select analyses from the left panel to review or ask questions about your study data.</p>
                <p style='font-style:italic;'>Use Ctrl+Enter to send messages</p>
            </div>
            """
            self.chat_display.setHtml(welcome_html)
            
    def save_chat_history(self):
        """Save the current chat history after confirming with the user."""
        if not self.chat_history:
            self.show_message_dialog("Empty Chat", "There is no chat history to save.")
            return
            
        dialog = QDialog(self)
        dialog.setWindowTitle("Save Chat History")
        dialog.resize(400, 200)
        layout = QVBoxLayout(dialog)
        
        # Confirm message
        msg_label = QLabel("Do you want to save the current chat history?")
        msg_label.setWordWrap(True)
        layout.addWidget(msg_label)
        
        # Show stats
        stats_label = QLabel(f"This chat contains {len(self.chat_history)} messages with {len(self.selected_items)} analyses selected for context.")
        stats_label.setWordWrap(True)
        layout.addWidget(stats_label)
        
        # Add title field
        form_layout = QFormLayout()
        title_input = QTextEdit()
        title_input.setPlaceholderText("Enter chat title")
        title_input.setText(self.chat_metadata.get("title", ""))
        title_input.setMaximumHeight(60)
        form_layout.addRow("Title:", title_input)
        layout.addLayout(form_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(save_btn)
        layout.addLayout(button_layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Update title if changed
            new_title = title_input.toPlainText().strip()
            if new_title:
                self.chat_metadata["title"] = new_title
                
            # Create a structured format for the chat history
            selected_analyses_data = []
            for card in self.selected_items:
                # Include the full analysis data for context
                selected_analyses_data.append(card.analysis)
                
            history_data = {
                "metadata": self.chat_metadata,
                "messages": self.chat_history,
                "selected_analyses": selected_analyses_data,  # Store full analysis details
                "timestamp": datetime.now().isoformat()
            }
            
            # Update metadata
            self.chat_metadata["updated_at"] = datetime.now().isoformat()
            
            # Save to studies manager if available
            study_id = self.studies_combo.itemData(self.studies_combo.currentIndex())
            if study_id and self.studies_manager:
                study = self.studies_manager.get_study(study_id)
                if study:
                    if not hasattr(study, 'interpretation_chats'):
                        study.interpretation_chats = []
                    
                    # Check if we're updating an existing chat or creating a new one
                    existing_chat = None
                    for i, chat in enumerate(study.interpretation_chats):
                        if chat.get("metadata", {}).get("chat_id") == self.chat_metadata.get("chat_id"):
                            existing_chat = i
                            break
                    
                    if existing_chat is not None:
                        study.interpretation_chats[existing_chat] = history_data
                        message = "Chat history updated in study records."
                    else:
                        study.interpretation_chats.append(history_data)
                        message = "Chat history saved to study records."
                    
                    study.updated_at = datetime.now().isoformat()
                    self.show_message_dialog("Success", message)
                    return history_data
            
            # If no study is selected or available
            self.show_message_dialog("Error", "Could not save chat history. No study selected or study manager not available.")
            return history_data

    def apply_filter(self, filter_text):
        """Apply filter to show/hide cards."""
        for i in range(self.grid_layout.count()):
            item = self.grid_layout.itemAt(i)
            widget = item.widget()
            if isinstance(widget, InterpretationCard):
                analysis_type = widget.analysis.get('type', '').lower()
                if filter_text == "All":
                    widget.setVisible(True)
                elif filter_text == "Statistical" and "statistical" in analysis_type:
                    widget.setVisible(True)
                elif filter_text == "Visualization" and "visualization" in analysis_type:
                    widget.setVisible(True)
                elif filter_text == "Assumption" and "assumption" in analysis_type:
                    widget.setVisible(True)
                elif filter_text == "Groups" and widget.analysis.get('is_group', False):
                    widget.setVisible(True)
                else:
                    widget.setVisible(False)
        
    def show_message_dialog(self, title, message):
        """Show a simple message dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(400, 200)
        layout = QVBoxLayout(dialog)
        label = QLabel(message)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        layout.addWidget(label)
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        ok_btn = QToolButton()
        ok_btn.setIcon(load_bootstrap_icon("check-circle"))
        ok_btn.setIconSize(QSize(24, 24))
        ok_btn.setToolTip("OK")
        ok_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(ok_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        dialog.setLayout(layout)
        dialog.exec()
        
    def input_key_press_event(self, event):
        """Send message on Ctrl+Enter."""
        if event.key() == Qt.Key.Key_Return and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.send_chat_message()
        else:
            QTextEdit.keyPressEvent(self.chat_input, event)
            
    def load_sample_analyses(self):
        """Load sample analyses when no study is selected."""
        self.clear_grid()
        self.current_analyses = []
        self.current_groups = []
        sample_analyses = [
            {
                'type': 'Statistical Analysis',
                'outcome_name': 'Treatment Effect',
                'content': '''## Primary Analysis Results
                
The treatment group showed significant improvement compared to control:
- Mean difference: 12.5 points (95% CI: 8.2 to 16.8)
- t-statistic: 3.45 (p < 0.001)
- Effect size (Cohen's d): 0.82 (large effect)

This strongly supports the primary hypothesis.''',
                'timestamp': datetime.now().isoformat(),
                'dataset_name': 'Clinical Trial Data'
            },
            {
                'type': 'Visualization Analysis',
                'outcome_name': 'Response Distribution',
                'content': '''## Distribution Analysis
                
The response variable shows interesting patterns:
- Right-skewed distribution (skewness = 0.85)
- Several outliers identified in upper range
- Potential bimodal tendency

Recommendation: Consider log transformation for subsequent analyses.''',
                'timestamp': datetime.now().isoformat(),
                'dataset_name': 'Patient Outcomes'
            },
            {
                'type': 'Assumption Check',
                'outcome_name': 'Model Validation',
                'content': '''## Statistical Assumptions
                
1. Normality Test:
   - Shapiro-Wilk: W = 0.98, p = 0.245
   - Q-Q plot shows good alignment
                
2. Homogeneity of Variance:
   - Levene's test: p = 0.892
   - Variance ratio = 1.12
                
All assumptions are satisfied.''',
                'timestamp': datetime.now().isoformat(),
                'dataset_name': 'Validation Dataset'
            },
            {
                'type': 'Statistical Analysis',
                'outcome_name': 'Secondary Endpoints',
                'content': '''## Secondary Outcomes
                
1. Quality of Life Score:
   - Improvement: +8.3 points
   - p-value: 0.012
                
2. Adverse Events:
   - 15% reduction in treatment group
   - Risk ratio: 0.85 (95% CI: 0.72-0.94)''',
                'timestamp': datetime.now().isoformat(),
                'dataset_name': 'Safety Data'
            }
        ]
        
        
        for analysis in sample_analyses:
            self.current_analyses.append(analysis)
            self.add_card_to_grid(analysis)

        
    def display_analysis_preview_html(self, html_content):
        """Display HTML content directly in the preview pane."""
        self.analysis_preview.setHtml(html_content)

    def on_grid_container_resize(self, event):
        """Handle resize events for the grid container."""
        self.reorganize_grid()
        event.accept()

    def on_splitter_moved(self, pos, index):
        """Handle when the splitter is moved."""
        # Give a short delay to ensure the container has fully resized
        QTimer.singleShot(50, self.reorganize_grid)  # Increased delay for more reliability

    def load_chat_history(self, chat_id):
        """Load a specific chat history from the studies manager."""
        study_id = self.studies_combo.itemData(self.studies_combo.currentIndex())
        if not study_id or not self.studies_manager:
            self.show_message_dialog("No Study", "Please select a study first.")
            return False
        
        study = self.studies_manager.get_study(study_id)
        if not study or not hasattr(study, 'interpretation_chats'):
            self.show_message_dialog("No History", "No chat history found for this study.")
            return False
        
        for chat in study.interpretation_chats:
            if chat.get("metadata", {}).get("chat_id") == chat_id:
                self.chat_history = chat.get("messages", [])
                self.chat_metadata = chat.get("metadata", {})
                self._update_chat_display()
                
                # Clear current selections
                for i in range(self.grid_layout.count()):
                    item = self.grid_layout.itemAt(i)
                    if item and item.widget():
                        widget = item.widget()
                        if isinstance(widget, InterpretationCard):
                            widget.set_selected(False)
                
                # Try to reselect any analyses that were selected in this chat
                if "selected_analyses" in chat:
                    stored_analyses = chat.get("selected_analyses", [])
                    for saved_analysis in stored_analyses:
                        # We need a reliable way to identify the saved analyses
                        saved_id = saved_analysis.get('timestamp', '') + saved_analysis.get('outcome_name', '')
                        
                        # Find the corresponding card
                        for i in range(self.grid_layout.count()):
                            item = self.grid_layout.itemAt(i)
                            if item and item.widget():
                                widget = item.widget()
                                if isinstance(widget, InterpretationCard):
                                    # Create a matching ID for the current card
                                    current_analysis = widget.analysis
                                    current_id = current_analysis.get('timestamp', '') + current_analysis.get('outcome_name', '')
                                    
                                    # If we have a match, select the card
                                    if current_id == saved_id:
                                        widget.set_selected(True)
                                        if widget not in self.selected_items:
                                            self.selected_items.append(widget)
                
                self._update_chat_display()
                self.show_message_dialog("Success", "Chat history loaded successfully.")
                return True
        
        self.show_message_dialog("Not Found", f"Chat with ID {chat_id} not found.")
        return False

    def view_chat_histories(self):
        """Display a dialog with saved chat histories and allow user to load one."""
        study_id = self.studies_combo.itemData(self.studies_combo.currentIndex())
        if not study_id or not self.studies_manager:
            self.show_message_dialog("No Study", "Please select a study first.")
            return
        
        study = self.studies_manager.get_study(study_id)
        if not study or not hasattr(study, 'interpretation_chats') or not study.interpretation_chats:
            self.show_message_dialog("No History", "No chat histories found for this study.")
            return
        
        # Create dialog to display chat histories
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Chat Histories - {study.name}")
        dialog.resize(800, 500)
        layout = QVBoxLayout(dialog)
        
        # Splitter for list and preview
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - chat list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Header for the left panel
        header_label = QLabel("Available Chat Histories")
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        header_label.setFont(font)
        left_layout.addWidget(header_label)
        
        # List of chat histories
        self.history_list = QListWidget()
        self.history_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.history_list.itemSelectionChanged.connect(self.on_chat_history_selected)
        left_layout.addWidget(self.history_list)
        
        # Actions for chat history items
        action_layout = QHBoxLayout()
        rename_btn = QToolButton()
        rename_btn.setIcon(load_bootstrap_icon("pencil-square"))
        rename_btn.setToolTip("Rename Selected Chat")
        rename_btn.clicked.connect(lambda: self.rename_chat_history(study))
        rename_btn.setEnabled(False)
        self.history_list.itemSelectionChanged.connect(
            lambda: rename_btn.setEnabled(bool(self.history_list.selectedItems()))
        )
        action_layout.addWidget(rename_btn)
        
        delete_btn = QToolButton()
        delete_btn.setIcon(load_bootstrap_icon("trash"))
        delete_btn.setToolTip("Delete Selected Chat")
        delete_btn.clicked.connect(lambda: self.delete_chat_history(study))
        delete_btn.setEnabled(False)
        self.history_list.itemSelectionChanged.connect(
            lambda: delete_btn.setEnabled(bool(self.history_list.selectedItems()))
        )
        action_layout.addWidget(delete_btn)
        
        action_layout.addStretch()
        left_layout.addLayout(action_layout)
        
        # Sort chats by date (newest first)
        sorted_chats = sorted(
            study.interpretation_chats,
            key=lambda x: x.get("metadata", {}).get("updated_at", ""),
            reverse=True
        )
        
        # Add chat histories to the list with icons
        for chat in sorted_chats:
            metadata = chat.get("metadata", {})
            timestamp = metadata.get("updated_at", "Unknown date")
            title = metadata.get("title", "Untitled Chat")
            
            try:
                date_obj = datetime.fromisoformat(timestamp)
                date_str = date_obj.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                date_str = timestamp
            
            # Create list item with icon
            item = QListWidgetItem(load_bootstrap_icon("chat-text"), f"{date_str}\n{title}")
            item.setData(Qt.ItemDataRole.UserRole, chat)  # Store entire chat data
            self.history_list.addItem(item)
        
        splitter.addWidget(left_panel)
        
        # Right panel - chat preview
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Header for the right panel
        preview_header = QLabel("Chat Preview")
        preview_header.setFont(font)
        right_layout.addWidget(preview_header)
        
        # Preview text area
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setStyleSheet("border: 1px solid gray; border-radius: 4px;")
        right_layout.addWidget(self.preview_text)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 500])  # Initial sizes
        
        layout.addWidget(splitter)
        
        # Buttons
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        load_btn = QPushButton("Load Selected")
        load_btn.setEnabled(False)  # Disabled until selection
        self.history_list.itemSelectionChanged.connect(
            lambda: load_btn.setEnabled(bool(self.history_list.selectedItems()))
        )
        load_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(load_btn)
        layout.addLayout(button_layout)
        
        # Show dialog and handle result
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_items = self.history_list.selectedItems()
            if not selected_items:
                return
                
            selected_chat = selected_items[0].data(Qt.ItemDataRole.UserRole)
            chat_id = selected_chat.get("metadata", {}).get("chat_id")
            if chat_id:
                self.load_chat_history(chat_id)
                
    def on_chat_history_selected(self):
        """Preview the selected chat history."""
        selected_items = self.history_list.selectedItems()
        if not selected_items:
            self.preview_text.clear()
            return
            
        # Get the selected chat data
        chat = selected_items[0].data(Qt.ItemDataRole.UserRole)
        if not chat:
            return
            
        # Create HTML preview
        metadata = chat.get("metadata", {})
        messages = chat.get("messages", [])
        
        html = """
        <style>
            body { color: gray; font-family: sans-serif; }
            .message { padding: 8px; margin: 5px 0; border-radius: 5px; }
            .user { background-color: rgba(173, 216, 230, 0.2); }
            .assistant { background-color: rgba(144, 238, 144, 0.1); }
            .metadata { color: #6c757d; font-size: 0.9em; margin-bottom: 15px; }
        </style>
        """
        
        # Add metadata
        html += "<div class='metadata'>"
        html += f"<p><b>Title:</b> {metadata.get('title', 'Untitled')}</p>"
        
        created_at = metadata.get("created_at", "")
        if created_at:
            try:
                date_obj = datetime.fromisoformat(created_at)
                created_str = date_obj.strftime("%Y-%m-%d %H:%M")
                html += f"<p><b>Created:</b> {created_str}</p>"
            except (ValueError, TypeError):
                html += f"<p><b>Created:</b> {created_at}</p>"
        
        # Count messages
        user_count = sum(1 for m in messages if m.get("role") == "user")
        assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
        html += f"<p><b>Messages:</b> {len(messages)} ({user_count} user, {assistant_count} assistant)</p>"
        html += "</div>"
        
        # Add message previews (limit to first 5 messages)
        preview_messages = messages[:5]
        for msg in preview_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            # Truncate long messages
            if len(content) > 200:
                content = content[:200] + "..."
            
            css_class = "user" if role == "user" else "assistant"
            html += f"<div class='message {css_class}'>"
            html += f"<b>{'You' if role == 'user' else 'Assistant'}:</b> {content}"
            html += "</div>"
            
        # Add indicator if there are more messages
        if len(messages) > 5:
            html += f"<p><i>... and {len(messages) - 5} more messages</i></p>"
            
        self.preview_text.setHtml(html)

    def rename_chat_history(self, study):
        """Rename the selected chat history."""
        selected_items = self.history_list.selectedItems()
        if not selected_items:
            return
            
        # Get the selected chat
        selected_chat = selected_items[0].data(Qt.ItemDataRole.UserRole)
        if not selected_chat:
            return
            
        # Get current title
        metadata = selected_chat.get("metadata", {})
        current_title = metadata.get("title", "Untitled Chat")
        
        # Create dialog for new title
        dialog = QDialog(self)
        dialog.setWindowTitle("Rename Chat History")
        dialog.resize(400, 150)
        dialog_layout = QVBoxLayout(dialog)
        
        form_layout = QFormLayout()
        title_input = QTextEdit()
        title_input.setPlaceholderText("Enter new title")
        title_input.setText(current_title)
        title_input.setMaximumHeight(60)
        form_layout.addRow("Title:", title_input)
        dialog_layout.addLayout(form_layout)
        
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(save_btn)
        dialog_layout.addLayout(button_layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_title = title_input.toPlainText().strip()
            if not new_title:
                return
                
            # Update title in chat metadata
            chat_id = metadata.get("chat_id")
            if not chat_id:
                return
                
            # Find and update the chat in the study
            for chat in study.interpretation_chats:
                if chat.get("metadata", {}).get("chat_id") == chat_id:
                    chat["metadata"]["title"] = new_title
                    chat["metadata"]["updated_at"] = datetime.now().isoformat()
                    study.updated_at = datetime.now().isoformat()
                    
                    # Update the list item
                    date_str = selected_items[0].text().split("\n")[0]
                    selected_items[0].setText(f"{date_str}\n{new_title}")
                    
                    # Update preview if visible
                    self.on_chat_history_selected()
                    break
                    
    def delete_chat_history(self, study):
        """Delete the selected chat history."""
        selected_items = self.history_list.selectedItems()
        if not selected_items:
            return
            
        # Get the selected chat
        selected_chat = selected_items[0].data(Qt.ItemDataRole.UserRole)
        if not selected_chat:
            return
            
        # Confirm deletion
        dialog = QDialog(self)
        dialog.setWindowTitle("Confirm Deletion")
        dialog.resize(300, 150)
        dialog_layout = QVBoxLayout(dialog)
        
        confirm_label = QLabel("Are you sure you want to delete this chat history?")
        confirm_label.setWordWrap(True)
        dialog_layout.addWidget(confirm_label)
        
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        delete_btn = QPushButton("Delete")
        delete_btn.setStyleSheet("background-color: #dc3545; color: white;")
        delete_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(delete_btn)
        dialog_layout.addLayout(button_layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Get chat ID
            chat_id = selected_chat.get("metadata", {}).get("chat_id")
            if not chat_id:
                return
                
            # Remove the chat from the study
            original_count = len(study.interpretation_chats)
            study.interpretation_chats = [
                chat for chat in study.interpretation_chats 
                if chat.get("metadata", {}).get("chat_id") != chat_id
            ]
            
            # Check if deletion was successful
            if len(study.interpretation_chats) < original_count:
                study.updated_at = datetime.now().isoformat()
                # Remove from list widget
                row = self.history_list.row(selected_items[0])
                self.history_list.takeItem(row)
                # Clear preview
                self.preview_text.clear()
