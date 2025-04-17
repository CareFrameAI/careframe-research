from PyQt6.QtWidgets import (QWidget, QGridLayout, QLabel, QPushButton,
                             QTextEdit, QVBoxLayout, QScrollArea, QFrame, QHBoxLayout,
                             QToolButton)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette, QColor
from helpers.load_icon import load_bootstrap_icon

class HubSection(QWidget):
    """
    Dashboard-style HubSection showing key metrics and functions.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid_layout = QGridLayout(self)
        self.setLayout(self.grid_layout)
        
        # References to managers
        self.studies_manager = None

        # Define column ratios
        approval_width = 15
        research_width = 55
        overview_width = 30  # Remaining width

        # Set column stretch factors
        self.grid_layout.setColumnStretch(0, approval_width)
        self.grid_layout.setColumnStretch(1, research_width)
        self.grid_layout.setColumnStretch(2, overview_width)

        # Initialize dashboard widgets with icons
        self.num_teams_label = QLabel("Teams: 0")
        self.num_members_label = QLabel("Members: 0")
        self.num_projects_label = QLabel("Projects: 0")

        # Create containers for overview items with icons
        self.teams_container = self.create_label_with_icon("Teams: 0", "people-fill")
        self.members_container = self.create_label_with_icon("Members: 0", "person-fill")
        self.projects_container = self.create_label_with_icon("Projects: 0", "clipboard-data-fill")

        # Pinned Projects Area
        self.pinned_projects_text = QTextEdit()
        self.pinned_projects_text.setReadOnly(True)
        self.pinned_projects_text.setPlaceholderText("No pinned projects")

        # Review/Approval Area
        self.review_area_text = QTextEdit()
        self.review_area_text.setReadOnly(True)
        self.review_area_text.setPlaceholderText("No items pending review")

        # Notifications Area
        self.notifications_text = QTextEdit()
        self.notifications_text.setReadOnly(True)
        self.notifications_text.setPlaceholderText("No new notifications")

        # Research Studies Area (Scrollable)
        self.research_studies_scroll = QScrollArea()
        self.research_studies_widget = QWidget()
        self.research_studies_layout = QVBoxLayout(self.research_studies_widget)
        self.research_studies_scroll.setWidgetResizable(True)
        self.research_studies_scroll.setWidget(self.research_studies_widget)

        # Overview Area
        self.overview_widget = QWidget()
        self.overview_layout = QVBoxLayout(self.overview_widget)

        # Create header labels with icons and add refresh button for Research Studies
        pinned_header = self.create_header_with_icon("Pinned Research Projects", "pin-angle-fill")
        research_header_container = QWidget()
        research_header_layout = QHBoxLayout(research_header_container)
        research_header_layout.setContentsMargins(0, 0, 0, 0)
        research_header = self.create_header_with_icon("Research Studies", "microscope")
        
        # Create refresh button
        self.refresh_button = QPushButton()
        self.refresh_button.setIcon(load_bootstrap_icon("arrow-clockwise", size=24))
        self.refresh_button.setText("Refresh")
        self.refresh_button.setToolTip("Refresh data from database")
        self.refresh_button.clicked.connect(lambda: print("Refresh button clicked") or self.refresh_data())
        self.refresh_button.setStyleSheet("""
            QPushButton {
                padding: 4px 8px;
                border: none;
                border-radius: 4px;
            }
        """)
        
        research_header_layout.addWidget(research_header)
        research_header_layout.addStretch()
        research_header_layout.addWidget(self.refresh_button)
        
        review_header = self.create_header_with_icon("Review", "shield-check")
        overview_header = self.create_header_with_icon("Overview", "grid-3x3-gap-fill")
        notifications_header = self.create_header_with_icon("Notifications", "bell-fill")

        # Add widgets to the grid layout
        # Top Row: Pinned Projects and Research Studies
        self.grid_layout.addWidget(pinned_header, 0, 1)
        self.grid_layout.addWidget(self.pinned_projects_text, 1, 1, 1, 1)
        self.grid_layout.addWidget(research_header_container, 2, 1)
        self.grid_layout.addWidget(self.research_studies_scroll, 3, 1, 1, 1)

        # Left Column: Review/Approval Area
        self.grid_layout.addWidget(review_header, 0, 0)
        self.grid_layout.addWidget(self.review_area_text, 1, 0, 3, 1)

        # Right Column: Overview and Notifications
        self.grid_layout.addWidget(overview_header, 0, 2)
        self.overview_layout.addWidget(self.teams_container)
        self.overview_layout.addWidget(self.members_container)
        self.overview_layout.addWidget(self.projects_container)
        self.grid_layout.addWidget(self.overview_widget, 1, 2)

        self.grid_layout.addWidget(notifications_header, 2, 2)
        self.grid_layout.addWidget(self.notifications_text, 3, 2)

        # Stretch rows/columns as needed
        self.grid_layout.setRowStretch(3, 1)  # Research studies take most vertical space

        # Initialize the UI
        self._setup_ui()

    def set_studies_manager(self, studies_manager):
        """Set the studies manager but don't automatically refresh."""
        self.studies_manager = studies_manager
        # Don't auto-refresh data here - it will happen when the section is displayed

    def refresh_data(self):
        """Refresh all data from the database."""
        if not self.studies_manager:
            self.load_dummy_data()
            return
            
        # Clear existing data
        self.clear_research_studies()
        
        # Get all projects
        projects = []
        if hasattr(self.studies_manager, 'list_projects'):
            projects = self.studies_manager.list_projects()
            print(f"Hub: Found {len(projects)} projects")
        
        # Update the projects count
        self.update_num_projects(len(projects))
        
        # Get pinned projects (first 5 projects for now)
        pinned_projects = []
        for i, project in enumerate(projects[:5]):
            pinned_projects.append(f"{project['name']}")
        self.update_pinned_projects(pinned_projects)
        
        # This approach uses study_info directly instead of trying to use get_study
        all_studies = []
        total_studies = 0
        
        for project in projects:
            project_id = project['id']
            project_name = project['name']
            print(f"Hub: Getting studies for project {project_name} (ID: {project_id})")
            
            studies = self.studies_manager.list_studies(project_id)
            print(f"Hub: list_studies returned {len(studies)} studies for project {project_name}")
            total_studies += len(studies)
            
            # Use the study_info directly instead of trying to retrieve full study objects
            for study_info in studies:
                study_id = study_info["id"]
                study_name = study_info["name"]
                
                print(f"Hub: Adding study {study_id}: {study_name}")
                
                # Create a simpler study display object with just the info we need
                all_studies.append({
                    'id': study_id,
                    'project_id': project_id,
                    'project_name': project_name,
                    'study_info': study_info
                })
        
        # Display studies in the research area
        print(f"Hub: Found {len(all_studies)} studies to display")
        if all_studies:
            for study_data in all_studies:
                self.add_study_info_from_db(study_data)
            
            # Update status message with actual counts
            self.notifications_text.setText(f"Loaded {len(projects)} projects and {len(all_studies)} studies.")
        else:
            self.notifications_text.setText(f"No studies found in {len(projects)} projects. Use the Studies Manager to create studies.")
        
        # Update metrics
        self.update_num_members(len(all_studies))
        self.update_num_teams(len(projects))

    def create_label_with_icon(self, text, icon_name, size=16):
        """Creates a label with an icon on the left"""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        icon_label = QLabel()
        icon_label.setPixmap(load_bootstrap_icon(icon_name, size=size).pixmap(size, size))
        
        text_label = QLabel(text)
        # Store a reference to the text label so we can update it later
        container.text_label = text_label
        
        layout.addWidget(icon_label)
        layout.addWidget(text_label)
        layout.addStretch()
        
        return container

    def create_header_with_icon(self, text, icon_name):
        """Creates a header label with an icon"""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        icon_label = QLabel()
        icon_label.setPixmap(load_bootstrap_icon(icon_name, size=20).pixmap(20, 20))
        
        text_label = QLabel(f"<b>{text}</b>")
        
        layout.addWidget(icon_label)
        layout.addWidget(text_label)
        layout.addStretch()
        
        return container

    def _setup_ui(self):
        pass

    def load_dummy_data(self):
        """Loads sample data into the hub section for demonstration purposes."""
        # Sample pinned projects
        # Add some sample pinned projects
        pinned_projects = [
            "Long-term Diabetes Management Study",
            "Novel Drug Delivery System Trial",
            "Genetic Markers in Cardiovascular Disease"
        ]
        self.update_pinned_projects(pinned_projects)

        # Add sample review/approval items
        review_items = [
            "IRB Approval Pending: Phase II Clinical Trial Protocol",
            "Patient Consent Form Review Required",
            "New Research Site Addition - Memorial Hospital",
            "Interim Safety Data Analysis Review",
            "Protocol Amendment Review - Dosing Schedule"
        ]
        self.update_review_items(review_items)

        # Add sample notifications
        notifications = [
            "Adverse Event Report - Study ARM-2024",
            "New Patient Enrollment Milestone Reached"
        ]
        self.update_notifications(notifications)

        # Add sample research studies
        study_samples = [
            {
                "title": "Immunotherapy Response in Advanced Melanoma",
                "team_members": ["Dr. Sarah Chen", "Dr. Robert Martinez"],
                "ai_members": ["Claude 3", "Watson Health"],
                "status": "Phase II Clinical Trial",
                "last_updated": "2024-03-15",
                "outcome": "positive",
                "results": "Interim analysis shows 40% response rate"
            },
            {
                "title": "AI-Assisted Early Alzheimer's Detection",
                "team_members": ["Dr. Emily Wong", "Dr. Michael Peters"],
                "ai_members": ["Claude 3", "MedicalGPT"],
                "status": "warning",
                "last_updated": "2024-03-14",
                "outcome": "neutral",
                "results": "Patient recruitment below target - action required"
            },
            {
                "title": "Novel Antibiotic Efficacy Study",
                "team_members": ["Dr. James Wilson", "Dr. Lisa Thompson"],
                "ai_members": ["Claude 3", "DrugDiscoveryAI"],
                "status": "Phase I Clinical Trial",
                "last_updated": "2024-03-13",
                "outcome": "positive",
                "results": "Safety profile confirmed, advancing to expanded cohort"
            },
            {
                "title": "Remote Patient Monitoring in Heart Failure",
                "team_members": ["Dr. David Kim", "Dr. Rachel Greene"],
                "ai_members": ["HealthMonitorAI", "CardiacPredictAI"],
                "status": "Observational Study",
                "last_updated": "2024-03-12",
                "outcome": "positive",
                "results": "30% reduction in hospital readmissions"
            },
            {
                "title": "mRNA Vaccine Development for Influenza",
                "team_members": ["Dr. Anna Patel", "Dr. John Murphy"],
                "ai_members": ["VaccineAI", "Claude 3"],
                "status": "Pre-clinical",
                "last_updated": "2024-03-10",
                "outcome": "warning",
                "results": "Stability testing needed at higher temperatures"
            }
        ]

        # Clear any existing studies and add new ones
        self.clear_research_studies()
        for study in study_samples:
            self.add_research_study(study)

        # Update metrics
        self.update_num_teams(8)
        self.update_num_members(24)
        self.update_num_projects(12)

    def update_num_teams(self, num_teams):
        self.teams_container.text_label.setText(f"Teams: {num_teams}")

    def update_num_members(self, num_members):
        self.members_container.text_label.setText(f"Members: {num_members}")

    def update_num_projects(self, num_projects):
        self.projects_container.text_label.setText(f"Projects: {num_projects}")

    def update_pinned_projects(self, projects):
        """
        Updates the pinned projects display.  'projects' should be a list of strings.
        """
        self.pinned_projects_text.setText("\n".join(projects))

    def update_review_items(self, review_items):
        """
        Updates the review area with items needing approval.
        'review_items' should be a list of strings.
        """
        self.review_area_text.setText("\n".join(review_items))

    def update_notifications(self, notifications):
        """
        Updates the notifications area. 'notifications' should be a list of strings.
        """
        self.notifications_text.setText("\n".join(notifications))

    def add_study_info_from_db(self, study_data):
        """
        Adds a study from the database to the research studies area using just study_info.
        study_data should contain 'study_info', 'project_name', etc.
        """
        study_info = study_data['study_info']
        print(f"Hub: Adding study {study_data['id']} from project {study_data['project_name']}")
        
        # Create the study display
        study_frame = QFrame()
        study_layout = QVBoxLayout(study_frame)
        study_frame.setFrameShape(QFrame.Shape.StyledPanel)
        study_frame.setFrameShadow(QFrame.Shadow.Raised)

        # Title with project name
        title_container = QWidget()
        title_layout = QHBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        
        # Icon for the study
        icon_name = "check-circle-fill"
        icon_color = "#28a745"  # Green
        
        icon_label = QLabel()
        icon_label.setPixmap(load_bootstrap_icon(icon_name, color=icon_color, size=20).pixmap(20, 20))
        
        # Study name
        title_label = QLabel(f"<b>{study_info['name']}</b>")
        
        title_layout.addWidget(icon_label)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        
        study_layout.addWidget(title_container)

        # Project info
        project_container = QWidget()
        project_layout = QHBoxLayout(project_container)
        project_layout.setContentsMargins(0, 0, 0, 0)
        
        project_icon = QLabel()
        project_icon.setPixmap(load_bootstrap_icon("folder", size=16).pixmap(16, 16))
        
        project_label = QLabel(f"Project: {study_data['project_name']}")
        
        project_layout.addWidget(project_icon)
        project_layout.addWidget(project_label)
        project_layout.addStretch()
        
        study_layout.addWidget(project_container)

        # Created date
        if 'created_at' in study_info:
            try:
                date_container = QWidget()
                date_layout = QHBoxLayout(date_container)
                date_layout.setContentsMargins(0, 0, 0, 0)
                
                date_icon = QLabel()
                date_icon.setPixmap(load_bootstrap_icon("calendar-date", size=16).pixmap(16, 16))
                
                created_date = study_info['created_at'].split('T')[0] if 'T' in study_info['created_at'] else study_info['created_at']
                date_label = QLabel(f"Created: {created_date}")
                
                date_layout.addWidget(date_icon)
                date_layout.addWidget(date_label)
                date_layout.addStretch()
                
                study_layout.addWidget(date_container)
            except:
                pass
        
        # Updated date
        if 'updated_at' in study_info:
            try:
                update_container = QWidget()
                update_layout = QHBoxLayout(update_container)
                update_layout.setContentsMargins(0, 0, 0, 0)
                
                update_icon = QLabel()
                update_icon.setPixmap(load_bootstrap_icon("clock-history", size=16).pixmap(16, 16))
                
                updated_date = study_info['updated_at'].split('T')[0] if 'T' in study_info['updated_at'] else study_info['updated_at']
                last_updated_label = QLabel(f"Updated: {updated_date}")
                
                update_layout.addWidget(update_icon)
                update_layout.addWidget(last_updated_label)
                update_layout.addStretch()
                
                study_layout.addWidget(update_container)
            except:
                pass

        # Active status
        if 'is_active' in study_info and study_info['is_active']:
            # Highlight active studies
            palette = study_frame.palette()
            palette.setColor(QPalette.ColorRole.Window, QColor(200, 255, 200))  # Pastel green
            study_frame.setPalette(palette)
            study_frame.setAutoFillBackground(True)
            
            status_container = QWidget()
            status_layout = QHBoxLayout(status_container)
            status_layout.setContentsMargins(0, 0, 0, 0)
            
            status_icon = QLabel()
            status_icon.setPixmap(load_bootstrap_icon("check-square", size=16).pixmap(16, 16))
            
            status_label = QLabel("<b>Active Study</b>")
            
            status_layout.addWidget(status_icon)
            status_layout.addWidget(status_label)
            status_layout.addStretch()
            
            study_layout.addWidget(status_container)

        self.research_studies_layout.addWidget(study_frame)

    def add_research_study(self, study_data):
        """
        Adds a research study to the research studies area.
        'study_data' should be a dictionary containing study information.
        """
        study_frame = QFrame()
        study_layout = QVBoxLayout(study_frame)
        study_frame.setFrameShape(QFrame.Shape.StyledPanel)
        study_frame.setFrameShadow(QFrame.Shadow.Raised)

        # Title with appropriate icon based on outcome
        title_container = QWidget()
        title_layout = QHBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        
        # Select icon based on outcome
        icon_name = "check-circle-fill" if study_data['outcome'] == 'positive' else "exclamation-triangle-fill"
        icon_color = "#28a745" if study_data['outcome'] == 'positive' else "#ffc107"  # Green or yellow
        
        icon_label = QLabel()
        icon_label.setPixmap(load_bootstrap_icon(icon_name, color=icon_color, size=20).pixmap(20, 20))
        
        title_label = QLabel(f"<b>{study_data['title']}</b>")
        
        title_layout.addWidget(icon_label)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        
        study_layout.addWidget(title_container)

        # Create a widget for team members with icon
        team_container = QWidget()
        team_layout = QHBoxLayout(team_container)
        team_layout.setContentsMargins(0, 0, 0, 0)
        
        team_icon = QLabel()
        team_icon.setPixmap(load_bootstrap_icon("people", size=16).pixmap(16, 16))
        
        team_members_label = QLabel(f"Team: {', '.join(study_data['team_members'])}")
        
        team_layout.addWidget(team_icon)
        team_layout.addWidget(team_members_label)
        team_layout.addStretch()
        
        study_layout.addWidget(team_container)

        # AI members with icon
        ai_container = QWidget()
        ai_layout = QHBoxLayout(ai_container)
        ai_layout.setContentsMargins(0, 0, 0, 0)
        
        ai_icon = QLabel()
        ai_icon.setPixmap(load_bootstrap_icon("cpu", size=16).pixmap(16, 16))
        
        ai_members_label = QLabel(f"AI: {', '.join(study_data['ai_members'])}")
        
        ai_layout.addWidget(ai_icon)
        ai_layout.addWidget(ai_members_label)
        ai_layout.addStretch()
        
        study_layout.addWidget(ai_container)

        # Status with icon
        status_container = QWidget()
        status_layout = QHBoxLayout(status_container)
        status_layout.setContentsMargins(0, 0, 0, 0)
        
        status_icon = QLabel()
        status_icon.setPixmap(load_bootstrap_icon("info-circle", size=16).pixmap(16, 16))
        
        status_label = QLabel(f"Status: {study_data['status']}")
        
        status_layout.addWidget(status_icon)
        status_layout.addWidget(status_label)
        status_layout.addStretch()
        
        study_layout.addWidget(status_container)

        # Last updated with icon
        update_container = QWidget()
        update_layout = QHBoxLayout(update_container)
        update_layout.setContentsMargins(0, 0, 0, 0)
        
        update_icon = QLabel()
        update_icon.setPixmap(load_bootstrap_icon("calendar-date", size=16).pixmap(16, 16))
        
        last_updated_label = QLabel(f"Last Updated: {study_data['last_updated']}")
        
        update_layout.addWidget(update_icon)
        update_layout.addWidget(last_updated_label)
        update_layout.addStretch()
        
        study_layout.addWidget(update_container)

        # Color-code based on outcome
        if study_data['outcome'] == 'positive':
            palette = study_frame.palette()
            palette.setColor(QPalette.ColorRole.Window, QColor(200, 255, 200))  # Pastel green
            study_frame.setPalette(palette)
            study_frame.setAutoFillBackground(True)
        elif study_data['status'] == 'warning':
            palette = study_frame.palette()
            palette.setColor(QPalette.ColorRole.Window, QColor(255, 255, 200))  # Pastel yellow
            study_frame.setPalette(palette)
            study_frame.setAutoFillBackground(True)

        # Results with icon
        results_container = QWidget()
        results_layout = QHBoxLayout(results_container)
        results_layout.setContentsMargins(0, 0, 0, 0)
        
        results_icon = QLabel()
        results_icon.setPixmap(load_bootstrap_icon("clipboard-data", size=16).pixmap(16, 16))
        
        results_label = QLabel(f"Results: {study_data['results']}")
        
        results_layout.addWidget(results_icon)
        results_layout.addWidget(results_label)
        results_layout.addStretch()
        
        study_layout.addWidget(results_container)

        self.research_studies_layout.addWidget(study_frame)

    def clear_research_studies(self):
        """
        Clears all research studies from the research studies area.
        """
        for i in reversed(range(self.research_studies_layout.count())):
            widget = self.research_studies_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

