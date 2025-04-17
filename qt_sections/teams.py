from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QComboBox, QMessageBox, QCheckBox, QGroupBox, QTextEdit, QGridLayout, QMenu, QDialog, QTableWidget, QTableWidgetItem)
import logging
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QComboBox, QMessageBox)
import logging
import requests
from datetime import datetime
from PyQt6.QtCore import Qt, QLocale
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import pyqtSlot, pyqtSignal, QObject
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtCore import pyqtProperty

class TeamSection(QWidget):
    # Keep only the signals we actually use
    editMemberSignal = pyqtSignal(int)
    removeMemberSignal = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.user_access = None  # Will be set by MainWindow
        
        # Initialize member management variables
        self.members_layout = QVBoxLayout()
        self.members_count = 0
        
        # CouchDB connection settings
        self.base_url = "http://localhost:5984"
        self.auth = ("admin", "cfpwd")
        
        # Main layout as grid
        self.layout = QGridLayout(self)
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(20, 20, 20, 20)

        # Initialize team selector
        self.team_selector = QComboBox()
        self.team_selector.currentTextChanged.connect(self.load_team_members)
        
        # Title
        title = QLabel("Administration | Organizations | Teams | Members")
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                padding: 10px 0;
            }
        """)
        self.layout.addWidget(title, 0, 0, 1, 2)

        # Left side - Team Members View
        members_group = QGroupBox("Manage Members")
        members_layout = QVBoxLayout()
        
        # Add team selector to members layout
        members_layout.addWidget(QLabel("Select Team:"))
        members_layout.addWidget(self.team_selector)
        
        # Replace QTextEdit with QTableWidget
        self.members_table = QTableWidget()
        self.members_table.setColumnCount(5)
        self.members_table.setHorizontalHeaderLabels(['Name', 'Email', 'Role', 'Admin', 'Actions'])
        self.members_table.horizontalHeader().setStretchLastSection(True)
        self.members_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.members_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        members_layout.addWidget(self.members_table)
        
        refresh_btn = QPushButton("Refresh Teams")
        refresh_btn.setMinimumHeight(40)
        refresh_btn.clicked.connect(self.refresh_teams)
        members_layout.addWidget(refresh_btn)
        
        members_group.setLayout(members_layout)
        self.layout.addWidget(members_group, 1, 0, 3, 1)
        # Right side - Management Controls
        member_group = QGroupBox("Add Members | Create Team | Delete Team")
        member_layout = QVBoxLayout()
        
        # Member name row
        member_layout.addLayout(self.members_layout)
        
        # Team name input with note
        self.team_name_input = QLineEdit()
        self.team_name_input.setPlaceholderText("Enter Team Name")
        self.team_name_input.setMinimumHeight(40)
        member_layout.addWidget(self.team_name_input)
        
        # Button row
        button_layout = QHBoxLayout()
        
        add_member_btn = QPushButton("Add Row")
        add_member_btn.setMinimumHeight(40)
        add_member_btn.clicked.connect(self.add_member_row)
        
        self.create_team_button = QPushButton("Create Account") 
        self.create_team_button.setMinimumHeight(40)
        self.create_team_button.clicked.connect(self.create_team)
        
        delete_team_btn = QPushButton("Delete Team")
        delete_team_btn.setMinimumHeight(40)
        delete_team_btn.clicked.connect(self.delete_team)
        
        button_layout.addWidget(add_member_btn)
        button_layout.addWidget(self.create_team_button)
        button_layout.addWidget(delete_team_btn)
        
        member_layout.addLayout(button_layout)
        member_group.setLayout(member_layout)
        
        # Add group to main layout
        self.layout.addWidget(member_group, 1, 1, 2, 1)

        # Organization Management Section
        org_group = QGroupBox("Organization Management")
        org_layout = QVBoxLayout()
        
        self.org_name_input = QLineEdit()
        self.org_name_input.setPlaceholderText("Enter Organization Name")
        self.org_name_input.setMinimumHeight(40)
        
        delete_org_btn = QPushButton("Delete Organization")
        delete_org_btn.setMinimumHeight(40)
        delete_org_btn.clicked.connect(self.delete_organization)
        
        org_layout.addWidget(self.org_name_input)
        org_layout.addWidget(delete_org_btn)
        org_group.setLayout(org_layout)
        self.layout.addWidget(org_group, 3, 1)

        # Password Management Section
        self.add_password_management_buttons()
        
        # Database Reset (Admin only)
        reset_group = QGroupBox("Database Management (Admin Only)")
        reset_layout = QVBoxLayout()
        
        self.admin_name_input = QLineEdit()
        self.admin_name_input.setPlaceholderText("Type your admin name to confirm reset")
        self.admin_name_input.setMinimumHeight(40)
        
        self.reset_button = QPushButton("Reset Database")
        self.reset_button.setMinimumHeight(40)
        self.reset_button.clicked.connect(self.confirm_reset_database)
        
        reset_layout.addWidget(self.admin_name_input)
        reset_layout.addWidget(self.reset_button)
        reset_group.setLayout(reset_layout)
        self.layout.addWidget(reset_group, 4, 0, 1, 2)

        # Add double-click support for team selection
        self.team_selector.activated.connect(self.on_team_selected)

        # Add WebChannel for JavaScript communication
        self.web_channel = QWebChannel()
        self.web_channel.registerObject("pyObj", self)

        # Initial setup
        self.refresh_teams()

        # Add initial member row
        self.add_member_row()

        # Modify button text and style
        self.create_team_button.setText("Add Member")
        self.create_team_button.setStyleSheet("")  # Remove custom styling
        delete_team_btn.setStyleSheet("")  # Remove custom styling

        # Update other buttons to use default styling
        refresh_btn.setStyleSheet("")
        self.reset_button.setStyleSheet("")
        delete_org_btn.setStyleSheet("")
        self.change_password_button.setStyleSheet("")
        self.reset_password_button.setStyleSheet("")

    def confirm_reset_database(self):
        if not self.user_access or not self.user_access.current_user or not self.user_access.current_user.get('is_admin'):
            QMessageBox.warning(self, "Error", "Only administrators can reset the database")
            return
            
        admin_name = self.admin_name_input.text()
        if not admin_name or admin_name != self.user_access.current_user.get('name'):
            QMessageBox.warning(self, "Error", "Please type your admin name exactly to confirm reset")
            return
            
        reply = QMessageBox.question(
            self,
            'Confirm Database Reset',
            'Are you sure you want to reset the entire database? This action cannot be undone.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.reset_database()
            self.admin_name_input.clear()

    def delete_team(self):
        if not self.team_name_input.text():
            QMessageBox.warning(self, "Warning", "Please enter a team name to delete")
            return
            
        try:
            # Check if this is the last team
            response = requests.get(
                f"{self.base_url}/teams/_all_docs?include_docs=true",
                auth=self.auth
            )
            
            if response.status_code == 200:
                teams = [row['doc'] for row in response.json()['rows'] if 'doc' in row]
                if len(teams) <= 1:
                    QMessageBox.warning(self, "Error", "Cannot delete the last team")
                    return
                    
                team = next((t for t in teams if t.get('name') == self.team_name_input.text()), None)
                
                if team:
                    reply = QMessageBox.question(
                        self,
                        'Confirm Team Deletion',
                        f'Are you sure you want to delete the team "{self.team_name_input.text()}"?',
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    
                    if reply == QMessageBox.StandardButton.Yes:
                        delete_response = requests.delete(
                            f"{self.base_url}/teams/{team['_id']}?rev={team['_rev']}",
                            auth=self.auth
                        )
                        
                        if delete_response.status_code == 200:
                            QMessageBox.information(self, "Success", "Team deleted successfully")
                            self.refresh_teams()
                            self.team_name_input.clear()
                        else:
                            QMessageBox.warning(self, "Error", "Failed to delete team")
                else:
                    QMessageBox.warning(self, "Error", "Team not found")
        except requests.RequestException as e:
            QMessageBox.warning(self, "Error", f"Failed to delete team: {str(e)}")

    def delete_organization(self):
        if not self.org_name_input.text():
            QMessageBox.warning(self, "Warning", "Please enter an organization name to delete")
            return
            
        try:
            # Check if this is the last organization
            response = requests.get(
                f"{self.base_url}/organizations/_all_docs?include_docs=true",
                auth=self.auth
            )
            
            if response.status_code == 200:
                orgs = [row['doc'] for row in response.json()['rows'] if 'doc' in row]
                if len(orgs) <= 1:
                    QMessageBox.warning(self, "Error", "Cannot delete the last organization")
                    return
                    
                org = next((o for o in orgs if o.get('name') == self.org_name_input.text()), None)
                
                if org:
                    reply = QMessageBox.question(
                        self,
                        'Confirm Organization Deletion',
                        f'Are you sure you want to delete the organization "{self.org_name_input.text()}"?',
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    
                    if reply == QMessageBox.StandardButton.Yes:
                        delete_response = requests.delete(
                            f"{self.base_url}/organizations/{org['_id']}?rev={org['_rev']}",
                            auth=self.auth
                        )
                        
                        if delete_response.status_code == 200:
                            QMessageBox.information(self, "Success", "Organization deleted successfully")
                            self.org_name_input.clear()
                            self.refresh_teams()  # Refresh to update any affected teams
                        else:
                            QMessageBox.warning(self, "Error", "Failed to delete organization")
                else:
                    QMessageBox.warning(self, "Error", "Organization not found")
        except requests.RequestException as e:
            QMessageBox.warning(self, "Error", f"Failed to delete organization: {str(e)}")

    def reset_database(self):
        try:
            # Delete existing databases
            for db in ['organizations', 'teams', 'users']:
                requests.delete(f"{self.base_url}/{db}", auth=self.auth)
                requests.put(f"{self.base_url}/{db}", auth=self.auth)

            # Create initial views
            self._create_views()
            QMessageBox.information(self, "Success", "Database reset successfully")
        except requests.RequestException as e:
            QMessageBox.warning(self, "Error", f"Failed to reset database: {str(e)}")

    def _create_views(self):
        views = {
            'users': {
                '_design/auth': {
                    'views': {
                        'by_email': {
                            'map': 'function (doc) { if (doc.email) emit(doc.email, doc); }'
                        }
                    }
                }
            },
            'teams': {
                '_design/membership': {
                    'views': {
                        'by_member': {
                            'map': 'function (doc) { for(var i in doc.members) { emit(doc.members[i].email, doc); } }'
                        }
                    }
                }
            }
        }
        
        for db, designs in views.items():
            for design_id, design_doc in designs.items():
                requests.put(
                    f"{self.base_url}/{db}/{design_id}",
                    auth=self.auth,
                    json=design_doc
                )

    def create_team(self):
        if not self.team_name_input.text():
            QMessageBox.warning(self, "Warning", "Please enter a team name.")
            return
            
        if self.members_count == 0:
            QMessageBox.warning(self, "Warning", "Please add at least one team member.")
            return
            
        org_name = self.org_name_input.text() or "Independent"
        team_name = self.team_name_input.text()

        try:
            # Check if team already exists
            response = requests.get(
                f"{self.base_url}/teams/_all_docs?include_docs=true",
                auth=self.auth
            )
            
            if response.status_code == 200:
                existing_team = next(
                    (row['doc'] for row in response.json()['rows'] 
                     if 'doc' in row and row['doc'].get('name') == team_name),
                    None
                )
                
                if existing_team:
                    reply = QMessageBox.question(
                        self,
                        'Team Exists',
                        f'Team "{team_name}" already exists. Do you want to add members to this team?',
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    
                    if reply == QMessageBox.StandardButton.No:
                        return
                    
                    # Use existing team's organization
                    org_id = existing_team.get('organization_id')
                else:
                    # Create/update organization for new team
                    org_doc = {
                        "name": org_name,
                        "type": "organization",
                        "created_at": datetime.now().isoformat()
                    }
                    org_response = requests.post(
                        f"{self.base_url}/organizations",
                        auth=self.auth,
                        json=org_doc
                    )
                    org_id = org_response.json()['id']

            # Collect members data
            members_data = []
            for i in range(self.members_count):
                row_layout = self.members_layout.itemAt(i)
                name_input = row_layout.itemAt(0).widget()
                email_input = row_layout.itemAt(1).widget()
                role_combo = row_layout.itemAt(2).widget()
                is_admin_check = row_layout.itemAt(3).widget()

                member_name = name_input.text()
                member_email = email_input.text()
                member_role = role_combo.currentText()
                is_admin = is_admin_check.isChecked()

                if member_name and member_email:
                    members_data.append({
                        "name": member_name,
                        "email": member_email,
                        "role": member_role,
                        "is_admin": is_admin
                    })

            if not members_data:
                QMessageBox.warning(self, "Warning", "Please add at least one member with email.")
                return

            # Update or create team
            if existing_team:
                # Add new members to existing team
                existing_members = existing_team['members']
                for new_member in members_data:
                    if not any(m['email'] == new_member['email'] for m in existing_members):
                        existing_members.append(new_member)
                
                existing_team['members'] = existing_members
                team_response = requests.put(
                    f"{self.base_url}/teams/{existing_team['_id']}",
                    auth=self.auth,
                    json=existing_team
                )
            else:
                # Create new team
                team_doc = {
                    "name": team_name,
                    "organization_id": org_id,
                    "members": members_data,
                    "created_at": datetime.now().isoformat()
                }
                team_response = requests.post(
                    f"{self.base_url}/teams",
                    auth=self.auth,
                    json=team_doc
                )

            # Create/update users
            for member in members_data:
                user_doc = {
                    "name": member["name"],
                    "email": member["email"],
                    "roles": [member["role"]],
                    "is_admin": member["is_admin"],
                    "created_at": datetime.now().isoformat()
                }
                requests.post(
                    f"{self.base_url}/users",
                    auth=self.auth,
                    json=user_doc
                )

            QMessageBox.information(
                self,
                "Success",
                f"Team '{team_name}' {'updated' if existing_team else 'created'} successfully"
            )
            
            # Refresh the team list
            self.refresh_teams()

        except requests.RequestException as e:
            QMessageBox.warning(self, "Error", f"Failed to {'update' if existing_team else 'create'} team: {str(e)}")

    def add_member_row(self):
        member_row_layout = QHBoxLayout()
        member_row_layout.setSpacing(10)
        
        # Remove team input field from member rows
        name_input = QLineEdit()
        name_input.setPlaceholderText("Member Name")
        name_input.setMinimumHeight(30)
        
        email_input = QLineEdit()
        email_input.setPlaceholderText("Email")
        email_input.setMinimumHeight(30)
        
        role_combo = QComboBox()
        role_combo.setMinimumHeight(30)
        roles = [
            "Principal Investigator", "Co-Investigator", "Research Coordinator",
            "Research Assistant", "IRB Member", "Data Analyst", "Statistician", "Pharmacist"
        ]
        role_combo.addItems(roles)
        
        is_admin_check = QCheckBox("Admin")
        is_admin_check.setMinimumHeight(30)
        
        remove_btn = QPushButton("Remove Row")
        remove_btn.setMinimumHeight(30)
        remove_btn.clicked.connect(lambda: self.remove_member_row(member_row_layout))
        
        # Update layout without team input
        member_row_layout.addWidget(name_input)
        member_row_layout.addWidget(email_input)
        member_row_layout.addWidget(role_combo)
        member_row_layout.addWidget(is_admin_check)
        member_row_layout.addWidget(remove_btn)
        
        self.members_layout.addLayout(member_row_layout)
        self.members_count += 1

    def remove_member_row(self, row_layout):
        # Remove all widgets from the layout
        while row_layout.count():
            widget = row_layout.takeAt(0).widget()
            if widget:
                widget.deleteLater()
        # Remove the layout itself
        self.members_layout.removeItem(row_layout)
        self.members_count -= 1

    def add_password_management_buttons(self):
        password_group = QGroupBox("Password Management")
        password_layout = QHBoxLayout()
        
        self.change_password_button = QPushButton("Change Password")
        self.change_password_button.setMinimumHeight(40)
        self.change_password_button.clicked.connect(self.show_change_password_dialog)
        
        self.reset_password_button = QPushButton("Reset User Password")
        self.reset_password_button.setMinimumHeight(40)
        self.reset_password_button.clicked.connect(self.show_reset_password_dialog)
        
        password_layout.addWidget(self.change_password_button)
        password_layout.addWidget(self.reset_password_button)
        
        password_group.setLayout(password_layout)
        self.layout.addWidget(password_group, 3, 1)  # Add to grid at row 3, column 1

    def add_team_viewer(self):
        """Add a section to view existing teams and members"""
        viewer_group = QGroupBox("View Teams")
        viewer_layout = QVBoxLayout()
        
        # Team selector
        self.team_selector = QComboBox()
        self.team_selector.currentTextChanged.connect(self.load_team_members)
        viewer_layout.addWidget(QLabel("Select Team:"))
        viewer_layout.addWidget(self.team_selector)
        
        # Members display
        self.members_display = QWebEngineView()
        self.members_display.setMinimumHeight(200)
        viewer_layout.addWidget(QLabel("Team Members:"))
        viewer_layout.addWidget(self.members_display)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh Teams")
        refresh_btn.clicked.connect(self.refresh_teams)
        viewer_layout.addWidget(refresh_btn)
        
        viewer_group.setLayout(viewer_layout)
        self.layout.addWidget(viewer_group)
        
        # Initial load
        self.refresh_teams()

    def refresh_teams(self):
        """Fetch and display all teams"""
        try:
            response = requests.get(
                f"{self.base_url}/teams/_all_docs?include_docs=true",
                auth=self.auth,
                timeout=1  # Use a short timeout to fail fast
            )
            
            if response.status_code == 200:
                teams = [row['doc'] for row in response.json()['rows'] if 'doc' in row]
                self.team_selector.clear()
                # Only add teams that have a name field
                valid_team_names = [team.get('name') for team in teams if team.get('name')]
                self.team_selector.addItems(valid_team_names)
                
                # If no teams, add a placeholder
                if not valid_team_names:
                    self.team_selector.addItem("No teams available")
                
                # Return success
                return True
            else:
                # If response is not successful, add a placeholder
                self.team_selector.clear()
                self.team_selector.addItem("Error loading teams")
                return False
                
        except requests.RequestException as e:
            # Just log the error but don't show a popup
            print(f"Failed to fetch teams: {str(e)}")
            
            # Add a placeholder item
            self.team_selector.clear()
            self.team_selector.addItem("Database unavailable")
            
            # Empty the members table
            self.members_table.setRowCount(0)
            
            # Return failure
            return False

    def load_team_members(self, team_name):
        # Skip loading if the team name is one of our placeholder messages
        if team_name in ["No teams available", "Error loading teams", "Database unavailable"]:
            self.members_table.setRowCount(0)
            self.current_team = None
            return
            
        try:
            response = requests.get(
                f"{self.base_url}/teams/_all_docs?include_docs=true",
                auth=self.auth,
                timeout=1  # Use a short timeout to fail fast
            )
            
            if response.status_code == 200:
                teams = [row['doc'] for row in response.json()['rows'] if 'doc' in row]
                team = next((t for t in teams if t.get('name') == team_name), None)
                
                if team:
                    self.current_team = team
                    self.members_table.setRowCount(0)
                    
                    # Set row height
                    self.members_table.verticalHeader().setDefaultSectionSize(50)  # Increase row height to 50 pixels
                    
                    for i, member in enumerate(team.get('members', [])):
                        self.members_table.insertRow(i)
                        
                        # Add member data
                        self.members_table.setItem(i, 0, QTableWidgetItem(member.get('name', '')))
                        self.members_table.setItem(i, 1, QTableWidgetItem(member.get('email', '')))
                        self.members_table.setItem(i, 2, QTableWidgetItem(member.get('role', '')))
                        self.members_table.setItem(i, 3, QTableWidgetItem('Yes' if member.get('is_admin') else 'No'))
                        
                        # Add edit/remove buttons
                        actions_widget = QWidget()
                        actions_layout = QHBoxLayout(actions_widget)
                        actions_layout.setContentsMargins(4, 4, 4, 4)
                        
                        edit_btn = QPushButton("Edit")
                        remove_btn = QPushButton("Remove")
                        
                        edit_btn.clicked.connect(lambda checked, row=i: self.edit_team_member(row))
                        remove_btn.clicked.connect(lambda checked, row=i: self.remove_team_member(row))
                        
                        actions_layout.addWidget(edit_btn)
                        actions_layout.addWidget(remove_btn)
                        
                        self.members_table.setCellWidget(i, 4, actions_widget)
                        
                    self.members_table.resizeColumnsToContents()
                else:
                    self.members_table.setRowCount(0)
                    self.current_team = None
        except requests.RequestException as e:
            # Just log the error without showing popup
            print(f"Failed to load team members: {str(e)}")
            self.members_table.setRowCount(0)
            self.current_team = None

    def remove_team_member(self, member_index):
        """Remove a member from the current team"""
        if not self.current_team or not self.current_team.get('members'):
            QMessageBox.warning(self, "Error", "No team selected or no members to remove")
            return
        
        try:
            members = self.current_team['members']
            if len(members) <= 1:
                QMessageBox.warning(self, "Error", "Cannot remove the last team member")
                return
            
            if 0 <= member_index < len(members):
                member = members[member_index]
                reply = QMessageBox.question(
                    self,
                    'Confirm Member Removal',
                    f'Are you sure you want to remove {member.get("name", "this member")} from the team?',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    # Remove member from list
                    members.pop(member_index)
                    
                    # Update team in database
                    response = requests.put(
                        f"{self.base_url}/teams/{self.current_team['_id']}",
                        auth=self.auth,
                        json=self.current_team
                    )
                    
                    if response.status_code == 201:
                        QMessageBox.information(self, "Success", "Team member removed successfully")
                        self.load_team_members(self.current_team['name'])  # Refresh display
                    else:
                        QMessageBox.warning(self, "Error", "Failed to update team")
            else:
                QMessageBox.warning(self, "Error", "Invalid member index")
                
        except requests.RequestException as e:
            QMessageBox.warning(self, "Error", f"Failed to remove team member: {str(e)}")

    def show_change_password_dialog(self):
        if not self.user_access:
            QMessageBox.warning(self, "Error", "User access not initialized")
            return
        if not self.user_access.current_user:
            QMessageBox.warning(self, "Error", "Please log in first")
            return
            
        dialog = QWidget()
        dialog.setWindowTitle("Change Password")
        layout = QVBoxLayout(dialog)
        
        # Get current user's email
        current_user_email = self.user_access.current_user['email']
        
        # Current password field
        current_password = QLineEdit()
        current_password.setPlaceholderText("Current Password")
        current_password.setEchoMode(QLineEdit.EchoMode.Password)
        
        # New password fields
        new_password = QLineEdit()
        new_password.setPlaceholderText("New Password")
        new_password.setEchoMode(QLineEdit.EchoMode.Password)
        
        confirm_password = QLineEdit()
        confirm_password.setPlaceholderText("Confirm New Password")
        confirm_password.setEchoMode(QLineEdit.EchoMode.Password)
        
        # Add fields to layout
        layout.addWidget(QLabel(f"Change Password for {current_user_email}"))
        layout.addWidget(current_password)
        layout.addWidget(new_password)
        layout.addWidget(confirm_password)
        
        # Submit button
        submit_btn = QPushButton("Change Password")
        layout.addWidget(submit_btn)
        
        def change_password():
            if new_password.text() != confirm_password.text():
                QMessageBox.warning(dialog, "Error", "New passwords do not match!")
                return
                
            try:
                # Get current user
                response = requests.get(
                    f"{self.base_url}/users/_design/auth/_view/by_email",
                    auth=self.auth,
                    params={"key": f"\"{current_user_email}\""}
                )
                
                if response.status_code == 200 and response.json()['rows']:
                    user_doc = response.json()['rows'][0]['value']
                    
                    # Verify current password
                    if not self.user_access.verify_password(user_doc['password_hash'], current_password.text()):
                        QMessageBox.warning(dialog, "Error", "Current password is incorrect!")
                        return
                    
                    # Update password
                    result = self.user_access.set_password(current_user_email, new_password.text())
                    if result['status'] == 'success':
                        QMessageBox.information(dialog, "Success", "Password changed successfully!")
                        dialog.close()
                    else:
                        QMessageBox.warning(dialog, "Error", result.get('message', 'Failed to change password'))
                else:
                    QMessageBox.warning(dialog, "Error", "User not found!")
                    
            except requests.RequestException as e:
                QMessageBox.warning(dialog, "Error", f"Failed to change password: {str(e)}")
        
        submit_btn.clicked.connect(change_password)
        dialog.setMinimumWidth(400)  # Make dialog wider
        dialog.show()

    def show_reset_password_dialog(self):
        if not self.user_access:
            QMessageBox.warning(self, "Error", "User access not initialized")
            return
        if not self.user_access.current_user or not self.user_access.current_user.get('is_admin'):
            QMessageBox.warning(self, "Error", "Admin privileges required!")
            return
        
        dialog = QWidget()
        dialog.setWindowTitle("Reset User Password")
        layout = QVBoxLayout(dialog)
        
        # User email field
        user_email = QLineEdit()
        user_email.setPlaceholderText("User Email")
        
        # New password fields
        new_password = QLineEdit()
        new_password.setPlaceholderText("New Password")
        new_password.setEchoMode(QLineEdit.EchoMode.Password)
        
        confirm_password = QLineEdit()
        confirm_password.setPlaceholderText("Confirm New Password")
        confirm_password.setEchoMode(QLineEdit.EchoMode.Password)
        
        # Add fields to layout
        layout.addWidget(QLabel("Reset User Password"))
        layout.addWidget(user_email)
        layout.addWidget(new_password)
        layout.addWidget(confirm_password)
        
        # Submit button
        submit_btn = QPushButton("Reset Password")
        layout.addWidget(submit_btn)
        
        def reset_password():
            if new_password.text() != confirm_password.text():
                QMessageBox.warning(dialog, "Error", "Passwords do not match!")
                return
                
            try:
                # Reset password
                result = self.user_access.set_password(user_email.text(), new_password.text())
                if result['status'] == 'success':
                    QMessageBox.information(dialog, "Success", "Password reset successfully!")
                    dialog.close()
                else:
                    QMessageBox.warning(dialog, "Error", result.get('message', 'Failed to reset password'))
                    
            except requests.RequestException as e:
                QMessageBox.warning(dialog, "Error", f"Failed to reset password: {str(e)}")
        
        submit_btn.clicked.connect(reset_password)
        dialog.setMinimumWidth(400)  # Make dialog wider
        dialog.show()

    def on_team_selected(self, index):
        """Handle team selection from combo box"""
        team_name = self.team_selector.itemText(index)
        self.team_name_input.setText(team_name)
        self.load_team_members(team_name)

    @pyqtSlot(int)
    def editMember(self, member_index):
        self.editMemberSignal.emit(member_index)
        self.edit_team_member(member_index)

    @pyqtSlot(int)
    def removeMember(self, member_index):
        self.removeMemberSignal.emit(member_index)
        self.remove_team_member(member_index)

    def edit_team_member(self, member_index):
        """Edit an existing team member"""
        if not self.current_team or not self.current_team.get('members'):
            return
        
        try:
            members = self.current_team['members']
            if 0 <= member_index < len(members):
                member = members[member_index]
                
                # Create edit dialog
                dialog = QDialog(self)
                dialog.setWindowTitle("Edit Team Member")
                layout = QVBoxLayout(dialog)
                
                # Member fields
                name_input = QLineEdit(member.get('name', ''))
                name_input.setPlaceholderText("Member Name")
                email_input = QLineEdit(member.get('email', ''))
                email_input.setPlaceholderText("Email")
                
                role_combo = QComboBox()
                roles = [
                    "Principal Investigator", "Co-Investigator", "Research Coordinator",
                    "Research Assistant", "IRB Member", "Data Analyst", "Statistician", "Pharmacist"
                ]
                role_combo.addItems(roles)
                current_role = member.get('role', '')
                if current_role in roles:
                    role_combo.setCurrentText(current_role)
                
                is_admin_check = QCheckBox("Admin")
                is_admin_check.setChecked(member.get('is_admin', False))
                
                # Add fields to layout
                layout.addWidget(QLabel("Name:"))
                layout.addWidget(name_input)
                layout.addWidget(QLabel("Email:"))
                layout.addWidget(email_input)
                layout.addWidget(QLabel("Role:"))
                layout.addWidget(role_combo)
                layout.addWidget(is_admin_check)
                
                # Add buttons
                button_box = QHBoxLayout()
                save_btn = QPushButton("Save")
                cancel_btn = QPushButton("Cancel")
                button_box.addWidget(save_btn)
                button_box.addWidget(cancel_btn)
                layout.addLayout(button_box)
                
                # Connect buttons
                save_btn.clicked.connect(dialog.accept)
                cancel_btn.clicked.connect(dialog.reject)
                
                # Show dialog
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    # Update member data
                    member['name'] = name_input.text()
                    member['email'] = email_input.text()
                    member['role'] = role_combo.currentText()
                    member['is_admin'] = is_admin_check.isChecked()
                    
                    # Update team in database
                    response = requests.put(
                        f"{self.base_url}/teams/{self.current_team['_id']}",
                        auth=self.auth,
                        json=self.current_team
                    )
                    
                    if response.status_code == 201:
                        QMessageBox.information(self, "Success", "Team member updated successfully")
                        self.load_team_members(self.current_team['name'])  # Refresh display
                    else:
                        QMessageBox.warning(self, "Error", "Failed to update team member")
        
        except requests.RequestException as e:
            QMessageBox.warning(self, "Error", f"Failed to edit team member: {str(e)}")
