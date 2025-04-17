import sys
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTableWidget,
    QTableWidgetItem, QSpinBox, QCheckBox, QApplication, QFileDialog, QMessageBox, QSplitter, QComboBox, QLineEdit, QDialog, QDialogButtonBox, QTabWidget, QTextEdit
)
from PyQt6.QtCore import Qt
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5 backend for Matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import networkx as nx
from study_model.study_model import StudyDesign, OutcomeMeasure, CovariateDefinition, CFDataType, AnalysisPlan, StudyTimepoint, TimePoint, Arm, Intervention, RandomizationScheme, RandomizationMethod, EligibilityCriteria, EligibilityCriterion, EligibilityOperator # Import StudyDesign and related classes


class DataGenerationWidget(QWidget):
    """
    A PyQt6 widget for generating and visualizing data based on a StudyDesign.
    It now includes a DataSource tab for managing different data input methods.
    """

    def __init__(self, study_design: StudyDesign):
        super().__init__()
        self.study_design = study_design
        self.data_generator = None  # Initialize DataGenerator later
        self.dataframe = None # Initialize the dataframe

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Main Splitter: Controls (Left) and Data Display (Right)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Left Side: Control Panel ---
        control_layout = QVBoxLayout()

        # Data Source Type Selection
        self.source_type_combo = QComboBox()
        self.source_type_combo.addItems(["Simulation", "Upload CSV", "Database Server", "Local Directory", "SFTP"])
        self.source_type_combo.currentTextChanged.connect(self.update_data_source_inputs)
        control_layout.addWidget(QLabel("Data Source:"))
        control_layout.addWidget(self.source_type_combo)

        self.upload_button = QPushButton("Upload CSV")
        self.upload_button.clicked.connect(self.upload_csv)
        self.upload_button.hide()

        self.db_host_input = QLineEdit()
        self.db_host_input.setPlaceholderText("Host")
        self.db_port_input = QLineEdit()
        self.db_port_input.setPlaceholderText("Port")
        self.db_name_input = QLineEdit()
        self.db_name_input.setPlaceholderText("Database Name")
        self.db_user_input = QLineEdit()
        self.db_user_input.setPlaceholderText("Username")
        self.db_password_input = QLineEdit()
        self.db_password_input.setPlaceholderText("Password")
        self.db_password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.db_connect_button = QPushButton("Connect to Database")
        self.db_connect_button.clicked.connect(self.connect_to_database)

        self.db_widgets = [QLabel("Database Host:"), self.db_host_input, QLabel("Port:"), self.db_port_input,
                          QLabel("Database Name:"), self.db_name_input, QLabel("Username:"), self.db_user_input,
                          QLabel("Password:"), self.db_password_input, self.db_connect_button]
        for widget in self.db_widgets:
            widget.hide()

        self.local_dir_input = QLineEdit()
        self.local_dir_input.setPlaceholderText("Directory Path")
        self.local_dir_button = QPushButton("Browse...")
        self.local_dir_button.clicked.connect(self.browse_local_directory)
        self.local_dir_widgets = [QLabel("Local Directory:"), self.local_dir_input, self.local_dir_button]
        for widget in self.local_dir_widgets:
            widget.hide()

        self.sftp_host_input = QLineEdit()
        self.sftp_host_input.setPlaceholderText("Host")
        self.sftp_port_input = QLineEdit()
        self.sftp_port_input.setPlaceholderText("Port (Default: 22)")
        self.sftp_user_input = QLineEdit()
        self.sftp_user_input.setPlaceholderText("Username")
        self.sftp_password_input = QLineEdit()
        self.sftp_password_input.setPlaceholderText("Password")
        self.sftp_password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.sftp_path_input = QLineEdit()
        self.sftp_path_input.setPlaceholderText("Remote Path")
        self.sftp_connect_button = QPushButton("Connect to SFTP")
        self.sftp_connect_button.clicked.connect(self.connect_to_sftp)

        self.sftp_widgets = [QLabel("SFTP Host:"), self.sftp_host_input, QLabel("Port:"), self.sftp_port_input,
                            QLabel("Username:"), self.sftp_user_input, QLabel("Password:"), self.sftp_password_input,
                            QLabel("Remote Path:"), self.sftp_path_input, self.sftp_connect_button]
        for widget in self.sftp_widgets:
            widget.hide()

        # Simulation Controls (always visible when Simulation is selected)
        self.sample_size_label = QLabel("Sample Size:")
        self.sample_size_spinbox = QSpinBox()
        self.sample_size_spinbox.setRange(1, 10000)
        self.sample_size_spinbox.setValue(100)  # Default sample size

        simulation_controls_layout = QHBoxLayout()
        simulation_controls_layout.addWidget(self.sample_size_label)
        simulation_controls_layout.addWidget(self.sample_size_spinbox)
        simulation_controls_layout.addStretch()


        control_layout.addWidget(self.upload_button)
        for widget in self.db_widgets:
            control_layout.addWidget(widget)
        for widget in self.local_dir_widgets:
            control_layout.addWidget(widget)
        for widget in self.sftp_widgets:
            control_layout.addWidget(widget)
        control_layout.addLayout(simulation_controls_layout)  # Add simulation controls
        control_layout.addStretch()


        control_widget = QWidget()
        control_widget.setLayout(control_layout)
        main_splitter.addWidget(control_widget)

        # --- Right Side: Data Display (Table and Graph) ---
        data_display_layout = QVBoxLayout()

        # Data Snippet Table
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(0)
        self.table_widget.setRowCount(0)
        data_display_layout.addWidget(self.table_widget)

        # Run Button
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_action)  # Connect to a method
        data_display_layout.addWidget(self.run_button)


        data_display_widget = QWidget()
        data_display_widget.setLayout(data_display_layout)
        main_splitter.addWidget(data_display_widget)

        # Set initial splitter sizes (adjust as needed)
        main_splitter.setSizes([int(self.width() * 0.3), int(self.width() * 0.7)])

        layout.addWidget(main_splitter)
        self.setLayout(layout)

        self.update_data_source_inputs()  # Initial update

    #Removed unused functions
    # def set_vertical_orientation(self):
    #     """Sets the table/graph splitter to vertical orientation."""
    #     self.table_graph_splitter.setOrientation(Qt.Orientation.Vertical)

    # def set_horizontal_orientation(self):
    #     """Sets the table/graph splitter to horizontal orientation."""
    #     self.table_graph_splitter.setOrientation(Qt.Orientation.Horizontal)


    def update_data_source_inputs(self):
        """Shows/hides input fields based on the selected data source."""
        source_type = self.source_type_combo.currentText()

        self.upload_button.setVisible(source_type == "Upload CSV")

        for widget in self.db_widgets:
            widget.setVisible(source_type == "Database Server")
        for widget in self.local_dir_widgets:
            widget.setVisible(source_type == "Local Directory")
        for widget in self.sftp_widgets:
            widget.setVisible(source_type == "SFTP")

        # Show/hide simulation controls
        self.sample_size_label.setVisible(source_type == "Simulation")
        self.sample_size_spinbox.setVisible(source_type == "Simulation")

        # Clear table when switching data sources
        self.table_widget.clear()
        self.table_widget.setColumnCount(0)
        self.table_widget.setRowCount(0)
        self.dataframe = None  # Reset dataframe

    def upload_csv(self):
        """Handles CSV file upload."""
        options = QFileDialog.Option.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)", options=options)
        if file_name:
            try:
                self.dataframe = pd.read_csv(file_name)
                self.display_data()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load CSV: {str(e)}")

    def connect_to_database(self):
        """Connects to a database and retrieves data."""
        # Placeholder for database connection logic.  You'll need to use a
        # database library like psycopg2 (for PostgreSQL), mysql.connector
        # (for MySQL), etc.
        try:
            # Example (replace with your actual database connection)
            host = self.db_host_input.text()
            port = self.db_port_input.text()
            dbname = self.db_name_input.text()
            user = self.db_user_input.text()
            password = self.db_password_input.text()

            # This is a *very* simplified example and is NOT secure.
            # In a real application, use a proper database library and handle
            # connection errors, SQL injection, etc., appropriately.
            # conn = psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=password)
            # cursor = conn.cursor()
            # cursor.execute("SELECT * FROM your_table")  # Replace your_table
            # data = cursor.fetchall()
            # self.dataframe = pd.DataFrame(data, columns=[desc[0] for desc in cursor.description])
            # conn.close()
            # self.display_data()

            QMessageBox.information(self, "Success", "Database connection successful (Placeholder).  Replace with actual database logic.")


        except Exception as e:
            QMessageBox.critical(self, "Error", f"Database connection failed: {str(e)}")

    def browse_local_directory(self):
        """Opens a dialog to select a local directory."""
        options = QFileDialog.Option.DontUseNativeDialog
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory", options=options)
        if dir_path:
            self.local_dir_input.setText(dir_path)
            # You might want to list files in the directory, or load data
            # from a specific file within the directory.  Add that logic here.
            try:
                # Example: Load data from a file named 'data.csv' in the directory
                file_path = f"{dir_path}/data.csv"  # Assumes a file named data.csv
                self.dataframe = pd.read_csv(file_path)
                self.display_data()
            except FileNotFoundError:
                QMessageBox.warning(self, "Warning", "No 'data.csv' file found in the selected directory.")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load data from directory: {str(e)}")

    def connect_to_sftp(self):
        """Connects to an SFTP server and retrieves data."""
        # Placeholder for SFTP connection logic.  Use a library like paramiko.
        try:
            # Example (replace with your actual SFTP connection)
            host = self.sftp_host_input.text()
            port = int(self.sftp_port_input.text()) if self.sftp_port_input.text() else 22
            user = self.sftp_user_input.text()
            password = self.sftp_password_input.text()
            remote_path = self.sftp_path_input.text()

            # This is a *very* simplified example.  Use paramiko for a real
            # SFTP connection, and handle authentication, key exchange,
            # error handling, etc., properly.
            # transport = paramiko.Transport((host, port))
            # transport.connect(username=user, password=password)
            # sftp = paramiko.SFTPClient.from_transport(transport)
            # with sftp.open(remote_path) as f:
            #     self.dataframe = pd.read_csv(f)
            # sftp.close()
            # transport.close()
            # self.display_data()
            QMessageBox.information(self, "Success", "SFTP connection successful (Placeholder). Replace with actual SFTP logic using Paramiko.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"SFTP connection failed: {str(e)}")

    def run_action(self):
        """Handles the action triggered by the 'Run' button."""
        source_type = self.source_type_combo.currentText()

        if source_type == "Simulation":
            self.generate_data()  # Call generate_data for simulation
        elif source_type == "Upload CSV":
            # CSV upload already handles displaying data, so no action needed here
            pass
        elif source_type == "Database Server":
            self.connect_to_database()
        elif source_type == "Local Directory":
            self.browse_local_directory()  # Assuming you want to reload on "Run"
        elif source_type == "SFTP":
            self.connect_to_sftp()

    def generate_data(self):
        """Generates data based on the StudyDesign and current settings."""
        # Only generate if "Simulation" is selected in the Data Source tab
        if self.source_type_combo.currentText() == "Simulation":
            try:
                self.study_design.validate_design()  # Always validate
                # self.data_generator = 
                self.dataframe = self.data_generator.execute()
                self.display_data()
                self.display_graph()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Data generation failed: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Please select 'Simulation' as the data source to generate data.")

    def display_data(self):
        """Displays the generated data in the table."""
        if self.dataframe is None:
            return

        # Display only a snippet (e.g., first 10 rows)
        data_snippet = self.dataframe.head(10)

        self.table_widget.clear()
        self.table_widget.setColumnCount(len(data_snippet.columns))
        self.table_widget.setRowCount(len(data_snippet.index))
        self.table_widget.setHorizontalHeaderLabels(list(data_snippet.columns))

        for i in range(len(data_snippet.index)):
            for j in range(len(data_snippet.columns)):
                item = QTableWidgetItem(str(data_snippet.iloc[i, j]))
                self.table_widget.setItem(i, j, item)