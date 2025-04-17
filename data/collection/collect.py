import os
import json
import pandas as pd
import numpy as np 
from datetime import datetime, date, timedelta
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,QGridLayout,
    QFileDialog, QLabel, QLineEdit, QComboBox, QTextEdit, QFormLayout, 
    QSpinBox, QMessageBox, QDialog, QDialogButtonBox, 
    QGroupBox, QStackedWidget, QPlainTextEdit, QRadioButton, QButtonGroup,
    QListWidget, QListWidgetItem, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QTabWidget, QToolButton, QSizePolicy, QStatusBar, QFrame, QCheckBox,
    QMenu, QInputDialog, QApplication
)
import re
from abc import ABC, abstractmethod
import asyncio
from llms.client import call_llm_async
from qasync import asyncSlot
from PyQt6.QtGui import QIcon, QColor
from data.selection.masking_utils import get_column_mapping
from helpers.load_icon import load_bootstrap_icon
from PyQt6.QtCore import QSize

class DataSource(ABC):
    """Abstract base class for all data sources"""
    
    @abstractmethod
    async def connect(self):
        """Establish connection to the data source"""
        pass
        
    @abstractmethod
    async def load_data(self) -> pd.DataFrame:
        """Load data from the source into a pandas DataFrame"""
        pass
        
    @abstractmethod
    def get_schema(self) -> dict:
        """Return the schema information for this data source"""
        pass
        
    @property
    @abstractmethod
    def source_type(self) -> str:
        """Return the type of data source"""
        pass

class FileDataSource(DataSource):
    """Implementation for file-based data sources"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self._df = None
        
    @property
    def source_type(self) -> str:
        return "file"
        
    async def connect(self):
        # No connection needed for files
        return True
        
    async def load_data(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
            
        if self.file_path.endswith('.csv'):
            self._df = pd.read_csv(self.file_path)
        elif self.file_path.endswith('.tsv'):
            self._df = pd.read_csv(self.file_path, sep='\t')
        elif self.file_path.endswith('.xlsx') or self.file_path.endswith('.xls'):
            self._df = pd.read_excel(self.file_path)
        else:
            raise ValueError(f"Unsupported file format: {self.file_path}")
            
        return self._df
        
    def get_schema(self) -> dict:
        if self._df is None:
            asyncio.run(self.load_data())
            
        schema = {
            "columns": list(self._df.columns),
            "dtypes": {col: str(self._df[col].dtype) for col in self._df.columns},
            "row_count": len(self._df)
        }
        return schema

class SQLDataSource(DataSource):
    """Implementation for SQL database sources"""
    
    def __init__(self, server, database, username, password, query, max_rows=1000):
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.query = query
        self.max_rows = max_rows
        self._df = None
        
    @property
    def source_type(self) -> str:
        return "sql"
        
    async def connect(self):
        # In a real implementation, this would establish a database connection
        # For demo purposes, we'll just return True
        return True
        
    async def load_data(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
            
        # In a real implementation, this would execute the SQL query
        # For demo purposes, create a dummy dataframe
        self._df = pd.DataFrame({
            'ID': range(10),
            'Name': [f'Name{i}' for i in range(10)],
            'Value': [i * 3.14 for i in range(10)]
        })
        
        return self._df
        
    def get_schema(self) -> dict:
        if self._df is None:
            asyncio.run(self.load_data())
            
        schema = {
            "columns": list(self._df.columns),
            "dtypes": {col: str(self._df[col].dtype) for col in self._df.columns},
            "row_count": len(self._df),
            "query": self.query
        }
        return schema

class SFTPDataSource(DataSource):
    """Implementation for SFTP sources"""
    
    def __init__(self, host, port, username, password, path):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.path = path
        self._df = None
        
    @property
    def source_type(self) -> str:
        return "sftp"
        
    async def connect(self):
        # In a real implementation, this would establish an SFTP connection
        # For demo purposes, we'll just return True
        return True
        
    async def load_data(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
            
        # In a real implementation, this would download and load the file
        # For demo purposes, create a dummy dataframe
        self._df = pd.DataFrame({
            'ID': range(10),
            'Remote_File': [f'file_{i}.csv' for i in range(10)],
            'Size': [i * 1024 for i in range(10)]
        })
        
        return self._df
        
    def get_schema(self) -> dict:
        if self._df is None:
            asyncio.run(self.load_data())
            
        schema = {
            "columns": list(self._df.columns),
            "dtypes": {col: str(self._df[col].dtype) for col in self._df.columns},
            "row_count": len(self._df),
            "remote_path": self.path
        }
        return schema

class RESTDataSource(DataSource):
    """Implementation for REST API sources"""
    
    def __init__(self, url, method, headers=None, body=None):
        self.url = url
        self.method = method
        self.headers = headers or {}
        self.body = body or {}
        self._df = None
        
    @property
    def source_type(self) -> str:
        return "rest"
        
    async def connect(self):
        # In a real implementation, this would validate the API connection
        # For demo purposes, we'll just return True
        return True
        
    async def load_data(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
            
        # In a real implementation, this would make the API request
        # For demo purposes, create a dummy dataframe
        self._df = pd.DataFrame({
            'api_id': range(10),
            'endpoint': [f'/resource/{i}' for i in range(10)],
            'response_time': [i * 0.1 for i in range(10)]
        })
        
        return self._df
        
    def get_schema(self) -> dict:
        if self._df is None:
            asyncio.run(self.load_data())
            
        schema = {
            "columns": list(self._df.columns),
            "dtypes": {col: str(self._df[col].dtype) for col in self._df.columns},
            "row_count": len(self._df),
            "api_url": self.url,
            "method": self.method
        }
        return schema

class SourceConnection:
    def __init__(self, source_type, connection_info, file_name=None):
        self.source_type = source_type
        self.connection_info = connection_info
        self.file_name = file_name
        self.timestamp = datetime.now()
        self.data_source = None  # Will hold the DataSource implementation
        self.metadata = None  # Store dataset metadata
        
        # Initialize the appropriate data source based on type
        self.initialize_data_source()
        
    def initialize_data_source(self):
        """Create the appropriate DataSource implementation"""
        if self.source_type == "upload" and self.connection_info.get('files'):
            # For simplicity, just use the first file
            self.data_source = FileDataSource(self.connection_info['files'][0])
            
        elif self.source_type == "sql":
            self.data_source = SQLDataSource(
                server=self.connection_info.get('server', ''),
                database=self.connection_info.get('database', ''),
                username=self.connection_info.get('username', ''),
                password=self.connection_info.get('password', ''),
                query=self.connection_info.get('query', ''),
                max_rows=self.connection_info.get('max_rows', 1000)
            )
            
        elif self.source_type == "sftp":
            self.data_source = SFTPDataSource(
                host=self.connection_info.get('host', ''),
                port=self.connection_info.get('port', 22),
                username=self.connection_info.get('username', ''),
                password=self.connection_info.get('password', ''),
                path=self.connection_info.get('path', '')
            )
            
        elif self.source_type == "rest":
            self.data_source = RESTDataSource(
                url=self.connection_info.get('url', ''),
                method=self.connection_info.get('method', 'GET'),
                headers=self.connection_info.get('headers', {}),
                body=self.connection_info.get('body', {})
            )
    
    async def load_data(self) -> pd.DataFrame:
        """Load data from the underlying data source"""
        if self.data_source:
            return await self.data_source.load_data()
        return pd.DataFrame()  # Empty dataframe if no data source

    def __str__(self):
        return f"{self.source_type} - {self.file_name} ({self.timestamp.strftime('%Y-%m-%d %H:%M:%S')})"


class DataFrameDisplay(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSortingEnabled(True)
        self.horizontalHeader().setSectionsMovable(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        
    def display_dataframe(self, df: pd.DataFrame):
        self.clear()
        if df is None or df.empty:
            return
            
        # Set up table dimensions
        self.setRowCount(min(1000, len(df)))  # Limit to 1000 rows for performance
        self.setColumnCount(len(df.columns))
        
        # Set headers
        self.setHorizontalHeaderLabels(df.columns)
        
        # Add data
        for i, (_, row) in enumerate(df.iterrows()):
            if i >= 1000:  # Limit to 1000 rows
                break
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                self.setItem(i, j, item)
        
        # Resize columns to content
        self.resizeColumnsToContents()

class DataSourceDialog(QDialog):
    """Dialog for adding data sources"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Data Source")
        self.setMinimumSize(700, 500)  # Make dialog larger
        
        # Initialize source information
        self._selected_files = []
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Source type selection
        source_type_group = QGroupBox("Source Type")
        type_layout = QVBoxLayout()
        
        type_layout.addWidget(QLabel("Select Source Type:"))
        self.source_type_combo = QComboBox()
        self.source_type_combo.addItems(["File Upload", "SQL Server", "SFTP", "REST API"])
        self.source_type_combo.currentIndexChanged.connect(self.on_source_type_changed)
        type_layout.addWidget(self.source_type_combo)
        
        source_type_group.setLayout(type_layout)
        layout.addWidget(source_type_group)

        # Stacked widget for different source configurations
        self.config_stack = QStackedWidget()
        
        # Upload config
        upload_widget = QWidget()
        upload_layout = QVBoxLayout(upload_widget)
        
        # Add a label with instructions
        upload_instructions = QLabel("Select one or more files to upload:")
        upload_instructions.setWordWrap(True)
        upload_layout.addWidget(upload_instructions)
        
        # Add file selection buttons
        file_buttons_layout = QHBoxLayout()
        
        upload_button = QPushButton("Add Files")
        upload_button.setIcon(load_bootstrap_icon("file-earmark-arrow-up"))
        upload_button.clicked.connect(self.browse_files)
        file_buttons_layout.addWidget(upload_button)
        
        remove_button = QPushButton("Remove Selected")
        remove_button.setIcon(load_bootstrap_icon("trash"))
        remove_button.clicked.connect(self.remove_selected_files)
        file_buttons_layout.addWidget(remove_button)
        
        clear_files_button = QPushButton("Clear All")
        clear_files_button.setIcon(load_bootstrap_icon("x-circle"))
        clear_files_button.clicked.connect(self.clear_files)
        file_buttons_layout.addWidget(clear_files_button)
        
        upload_layout.addLayout(file_buttons_layout)
        
        # Add a list widget to display selected files
        self.files_list = QListWidget()
        self.files_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.files_list.setMinimumHeight(150)
        upload_layout.addWidget(self.files_list)
        
        # Add a summary label
        self.file_count_label = QLabel("No files selected")
        upload_layout.addWidget(self.file_count_label)
        
        upload_layout.addStretch()
        self.config_stack.addWidget(upload_widget)
        
        # SQL Server config
        sql_widget = QWidget()
        sql_layout = QFormLayout(sql_widget)
        self.server_input = QLineEdit()
        self.database_input = QLineEdit()
        self.username_input = QLineEdit()
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.max_rows_input = QSpinBox()
        self.max_rows_input.setRange(1, 1000000)
        self.max_rows_input.setValue(1000)
        self.sql_query_input = QPlainTextEdit()
        self.sql_query_input.setPlaceholderText("Enter your SQL query here...")
        self.sql_query_input.setMinimumHeight(100)
        
        sql_layout.addRow("Server:", self.server_input)
        sql_layout.addRow("Database:", self.database_input)
        sql_layout.addRow("Username:", self.username_input)
        sql_layout.addRow("Password:", self.password_input)
        sql_layout.addRow("Max Rows:", self.max_rows_input)
        sql_layout.addRow("SQL Query:", self.sql_query_input)
        
        # Add SQL formatting and save buttons
        buttons_layout = QHBoxLayout()
        format_sql_button = QPushButton("Format SQL")
        format_sql_button.setIcon(load_bootstrap_icon("text-indent-left"))
        format_sql_button.clicked.connect(self.format_sql)
        buttons_layout.addWidget(format_sql_button)
        
        save_sql_button = QPushButton("Save SQL")
        save_sql_button.setIcon(load_bootstrap_icon("save"))
        save_sql_button.clicked.connect(self.save_sql)
        buttons_layout.addWidget(save_sql_button)
        
        sql_layout.addRow("", buttons_layout)
        
        self.config_stack.addWidget(sql_widget)
        
        # SFTP config
        sftp_widget = QWidget()
        sftp_layout = QFormLayout(sftp_widget)
        self.sftp_host_input = QLineEdit()
        self.sftp_port_input = QSpinBox()
        self.sftp_port_input.setRange(1, 65535)
        self.sftp_port_input.setValue(22)
        self.sftp_username_input = QLineEdit()
        self.sftp_password_input = QLineEdit()
        self.sftp_password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.sftp_path_input = QLineEdit()
        
        sftp_layout.addRow("Host:", self.sftp_host_input)
        sftp_layout.addRow("Port:", self.sftp_port_input)
        sftp_layout.addRow("Username:", self.sftp_username_input)
        sftp_layout.addRow("Password:", self.sftp_password_input)
        sftp_layout.addRow("Remote Path:", self.sftp_path_input)
        
        # Add SFTP browse button
        browse_sftp_button = QPushButton("Browse SFTP")
        browse_sftp_button.setIcon(load_bootstrap_icon("folder2-open"))
        browse_sftp_button.clicked.connect(self.browse_sftp)
        sftp_layout.addRow("", browse_sftp_button)
        
        self.config_stack.addWidget(sftp_widget)
        
        # REST API config
        rest_widget = QWidget()
        rest_layout = QFormLayout(rest_widget)
        self.api_url_input = QLineEdit()
        self.api_method_combo = QComboBox()
        self.api_method_combo.addItems(["GET", "POST", "PUT", "DELETE"])
        self.api_headers_input = QPlainTextEdit()
        self.api_headers_input.setPlaceholderText("Enter headers in JSON format...")
        self.api_body_input = QPlainTextEdit()
        self.api_body_input.setPlaceholderText("Enter request body in JSON format...")
        
        rest_layout.addRow("URL:", self.api_url_input)
        rest_layout.addRow("Method:", self.api_method_combo)
        rest_layout.addRow("Headers:", self.api_headers_input)
        rest_layout.addRow("Body:", self.api_body_input)
        
        self.config_stack.addWidget(rest_widget)
        
        layout.addWidget(self.config_stack)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        # Add clear button
        clear_button = QPushButton("Clear Form")
        clear_button.setIcon(load_bootstrap_icon("eraser"))
        clear_button.clicked.connect(self.clear_form)
        button_box.addButton(clear_button, QDialogButtonBox.ButtonRole.ResetRole)
        
        layout.addWidget(button_box)
        
        # Initialize to first source type
        self.on_source_type_changed(0)
    
    def on_source_type_changed(self, index):
        """Handle source type changes in the dropdown"""
        self.config_stack.setCurrentIndex(index)
    
    def browse_files(self):
        """Browse for files to upload"""
        options = QFileDialog.Option.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "Select Files", "", 
                                             "All Files (*);;TSV Files (*.tsv);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls)", 
                                             options=options)
        if files:
            # Add new files to the list widget
            for file_path in files:
                # Check if file is already in the list to avoid duplicates
                existing_items = self.files_list.findItems(file_path, Qt.MatchFlag.MatchExactly)
                if not existing_items:
                    self.add_file_to_list(file_path)
            
            # Update the selected files list
            self._selected_files = [self.files_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.files_list.count())]
            
            # Update the count label
            self.file_count_label.setText(f"Selected files: {self.files_list.count()} file(s)")
    
    def add_file_to_list(self, file_path):
        """Add a file to the list widget with formatted display"""
        # Get file info
        file_info = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        
        # Format file size
        if file_size < 1024:
            size_str = f"{file_size} bytes"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        
        # Create list item
        item = QListWidgetItem(f"{file_info} ({size_str})")
        item.setData(Qt.ItemDataRole.UserRole, file_path)  # Store full path as data
        item.setToolTip(file_path)  # Show full path on hover
        
        # Add to list
        self.files_list.addItem(item)
    
    def remove_selected_files(self):
        """Remove selected files from the list"""
        selected_items = self.files_list.selectedItems()
        for item in selected_items:
            self.files_list.takeItem(self.files_list.row(item))
        
        # Update the selected files list
        self._selected_files = [self.files_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.files_list.count())]
        
        # Update the count label
        if self.files_list.count() == 0:
            self.file_count_label.setText("No files selected")
        else:
            self.file_count_label.setText(f"Selected files: {self.files_list.count()} file(s)")
    
    def clear_files(self):
        """Clear all selected files"""
        self.files_list.clear()
        self._selected_files = []
        self.file_count_label.setText("No files selected")
    
    def format_sql(self):
        """Format SQL query"""
        # Basic SQL formatting - in a real application, use a proper SQL formatter
        sql = self.sql_query_input.toPlainText()
        keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN']
        formatted_sql = sql.upper()
        for keyword in keywords:
            formatted_sql = formatted_sql.replace(keyword, f'\n{keyword}')
        self.sql_query_input.setPlainText(formatted_sql)
    
    def save_sql(self):
        """Save SQL query to file"""
        options = QFileDialog.Option.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "Save SQL Query", "", "SQL Files (*.sql);;All Files (*)", options=options)
        if file_name:
            if not file_name.endswith('.sql'):
                file_name += '.sql'
            try:
                with open(file_name, 'w') as f:
                    f.write(self.sql_query_input.toPlainText())
                QMessageBox.information(self, "Success", "SQL query saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save SQL query: {str(e)}")
    
    def browse_sftp(self):
        """Browse SFTP server for files"""
        # This would typically connect to SFTP and show a file browser
        QMessageBox.information(self, "SFTP Browser", "SFTP browsing would be implemented here")
    
    def clear_form(self):
        """Clear the form fields"""
        index = self.config_stack.currentIndex()
        
        if index == 0:  # File Upload
            self.file_count_label.setText("No files selected")
            self._selected_files = []
            self.files_list.clear()
        elif index == 1:  # SQL
            self.server_input.clear()
            self.database_input.clear()
            self.username_input.clear()
            self.password_input.clear()
            self.max_rows_input.setValue(1000)
            self.sql_query_input.clear()
        elif index == 2:  # SFTP
            self.sftp_host_input.clear()
            self.sftp_port_input.setValue(22)
            self.sftp_username_input.clear()
            self.sftp_password_input.clear()
            self.sftp_path_input.clear()
        elif index == 3:  # REST API
            self.api_url_input.clear()
            self.api_method_combo.setCurrentIndex(0)
            self.api_headers_input.clear()
            self.api_body_input.clear()
        
    def get_source_info(self):
        """Return the source information based on form data"""
        index = self.config_stack.currentIndex()
        source_types = ["upload", "sql", "sftp", "rest"]
        source_type = source_types[index]
        
        # Prepare connection info based on source type
        connection_info = {}
        
        if source_type == "upload":
            # Get files from the list widget
            files = [self.files_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.files_list.count())]
            if not files:
                return None, None
            connection_info['files'] = files
            
        elif source_type == "sql":
            if not all([self.server_input.text(), self.database_input.text(), 
                       self.username_input.text(), self.password_input.text(),
                       self.sql_query_input.toPlainText()]):
                return None, None
            connection_info = {
                'server': self.server_input.text(),
                'database': self.database_input.text(),
                'username': self.username_input.text(),
                'password': self.password_input.text(),
                'max_rows': self.max_rows_input.value(),
                'query': self.sql_query_input.toPlainText()
            }
            
        elif source_type == "sftp":
            if not all([self.sftp_host_input.text(), self.sftp_username_input.text(),
                       self.sftp_password_input.text(), self.sftp_path_input.text()]):
                return None, None
            connection_info = {
                'host': self.sftp_host_input.text(),
                'port': self.sftp_port_input.value(),
                'username': self.sftp_username_input.text(),
                'password': self.sftp_password_input.text(),
                'path': self.sftp_path_input.text()
            }
            
        elif source_type == "rest":
            if not self.api_url_input.text():
                return None, None
            connection_info = {
                'url': self.api_url_input.text(),
                'method': self.api_method_combo.currentText(),
                'headers': self.api_headers_input.toPlainText(),
                'body': self.api_body_input.toPlainText()
            }
        
        return source_type, connection_info

class StudyGenerationDialog(QDialog):
    """Dialog for generating synthetic study data based on description"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate Study Data")
        self.setMinimumSize(600, 400)
        
        layout = QVBoxLayout(self)
        
        # Description input
        description_group = QGroupBox("Study Description")
        description_layout = QVBoxLayout(description_group)
        
        description_label = QLabel("Enter a description of the study design and data you want to generate:")
        description_layout.addWidget(description_label)
        
        self.description_text = QPlainTextEdit()
        self.description_text.setPlaceholderText(
            "Example: A randomized controlled trial with 100 participants comparing a new intervention "
            "vs control group. Measure blood pressure and cholesterol at baseline, 6 weeks, and 12 weeks. "
            "Include age, sex, and BMI as covariates."
        )
        description_layout.addWidget(self.description_text)
        
        description_group.setLayout(description_layout)
        layout.addWidget(description_group)
        
        # Dataset name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Dataset Name:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter name for generated dataset")
        name_layout.addWidget(self.name_input)
        layout.addLayout(name_layout)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def get_inputs(self):
        """Return the dialog inputs"""
        return {
            'description': self.description_text.toPlainText(),
            'name': self.name_input.text()
        }

class DataCollectionWidget(QWidget):
    """
    Widget for managing data sources and connections.
    Shows all functionality directly in the UI rather than using dialogs.
    """
    source_selected = pyqtSignal(str, object)  # Signal emitted when a source is selected (name, dataframe)
    source_added = pyqtSignal(str, object)  # Signal emitted when a source is added (name, dataframe)
    source_deleted = pyqtSignal(str)  # Signal emitted when a source is deleted (name)
    source_renamed = pyqtSignal(str, str)  # Signal emitted when a source is renamed (old_name, new_name)
    source_updated = pyqtSignal(str, object)  # Signal emitted when a source is updated (name, dataframe)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Sources")
        
        # Store connections and dataframes
        self.source_connections = {}
        self.dataframes = {}
        
        # Settings
        self.settings = DataCollectionSettings()
        
        # Current selection
        self.current_source = None
        
        # Store the latest grouping result
        self.latest_grouping_result = None
        self.latest_grouping_timestamp = None
        
        # Global style for icon buttons
        self.icon_button_style = """
            QPushButton, QToolButton {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
            }
            QPushButton:hover, QToolButton:hover {
                background-color: #f0f0f0;
                border: 1px solid #ddd;
            }
            QPushButton:pressed, QToolButton:pressed {
                background-color: #e0e0e0;
            }
        """
        
        # Group colors
        self.group_colors = [
            "#4A86E8",  # Vibrant blue
            "#6AA84F",  # Vibrant green
            "#E69138",  # Vibrant orange
            "#9966CC",  # Vibrant purple
            "#CC4125",  # Vibrant red
            "#999999",  # Gray
            "#45B6BC",  # Vibrant teal
            "#E6B800",  # Vibrant yellow
        ]
        
        # Initialize status_bar directly in __init__ to prevent AttributeError
        self.status_bar = QStatusBar()
        self.status_bar.setMaximumHeight(25)  # Reduce status bar height
        
        # Connect signals to update studies manager
        self.source_added.connect(self.add_to_studies_manager)
        self.source_deleted.connect(self.delete_from_studies_manager)
        self.source_renamed.connect(self.rename_in_studies_manager)
        self.source_updated.connect(self.update_in_studies_manager)
        
        self.init_ui()
        
    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(5)  # Reduce spacing between widgets
        main_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        
        # Top toolbar with main actions
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setSpacing(8)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        
        # Left side: Title with icon
        title_layout = QHBoxLayout()
        title_icon = QLabel()
        title_icon.setPixmap(load_bootstrap_icon("database").pixmap(20, 20))  # Smaller icon
        title_layout.addWidget(title_icon)
        
        title_label = QLabel("Data Collection")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold;")  # Slightly smaller font
        title_layout.addWidget(title_label)
        toolbar_layout.addLayout(title_layout)
        
        toolbar_layout.addStretch()
        
        # Right side: Main action buttons
        # Group in a frame with horizontal layout
        actions_frame = QFrame()
        actions_frame.setFrameShape(QFrame.Shape.Panel)
        actions_frame.setFrameShadow(QFrame.Shadow.Raised)
        actions_frame.setLineWidth(1)
        actions_frame.setStyleSheet("border: none;")
        actions_layout = QHBoxLayout(actions_frame)
        actions_layout.setSpacing(5)  # Reduce spacing
        actions_layout.setContentsMargins(5, 2, 5, 2)  # Reduce margins
        
        # Add source button - now icon-based with text
        add_button = QPushButton()
        add_button.setIcon(load_bootstrap_icon("plus-circle-fill"))
        add_button.setText("Add")
        add_button.setIconSize(QSize(18, 18))  # Smaller icon
        add_button.setToolTip("Add a new data source")
        add_button.clicked.connect(self.show_add_source_dialog)
        add_button.setFixedHeight(32)
        add_button.setMinimumWidth(80)
        self.apply_icon_button_style(add_button)
        actions_layout.addWidget(add_button)
        
        # Refresh button - now icon-based
        refresh_button = QPushButton()
        refresh_button.setIcon(load_bootstrap_icon("arrow-repeat"))
        refresh_button.setText("Refresh")
        refresh_button.setIconSize(QSize(18, 18))  # Smaller icon
        refresh_button.setToolTip("Refresh datasets from current study")
        refresh_button.clicked.connect(self.refresh_datasets)
        refresh_button.setFixedHeight(32)
        refresh_button.setMinimumWidth(80)
        self.apply_icon_button_style(refresh_button)
        actions_layout.addWidget(refresh_button)
        
        # Add vertical separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setFixedHeight(24)  # Shorter separator
        actions_layout.addWidget(separator)
        
        # Generate study button - now icon-based
        generate_button = QPushButton()
        generate_button.setIcon(load_bootstrap_icon("lightbulb-fill"))
        generate_button.setText("Generate")
        generate_button.setIconSize(QSize(18, 18))  # Smaller icon
        generate_button.setToolTip("Generate synthetic study data")
        generate_button.clicked.connect(self.show_generate_study_dialog)
        generate_button.setFixedHeight(32)
        generate_button.setMinimumWidth(80)
        self.apply_icon_button_style(generate_button)
        actions_layout.addWidget(generate_button)
        
        # Debug button - now icon-based
        debug_button = QPushButton()
        debug_button.setIcon(load_bootstrap_icon("bug-fill"))
        debug_button.setText("Debug")
        debug_button.setIconSize(QSize(18, 18))  # Smaller icon
        debug_button.setToolTip("Add debug datasets for testing")
        debug_button.clicked.connect(self.add_debug_datasets)
        debug_button.setFixedHeight(32)
        debug_button.setMinimumWidth(80)
        self.apply_icon_button_style(debug_button)
        actions_layout.addWidget(debug_button)
        
        # Healthcare Debug button
        healthcare_debug_button = QPushButton()
        healthcare_debug_button.setIcon(load_bootstrap_icon("hospital-fill"))
        healthcare_debug_button.setText("Healthcare")
        healthcare_debug_button.setIconSize(QSize(18, 18))  # Smaller icon
        healthcare_debug_button.setToolTip("Add healthcare debug datasets for testing")
        healthcare_debug_button.clicked.connect(self.add_healthcare_debug_datasets)
        healthcare_debug_button.setFixedHeight(32)
        healthcare_debug_button.setMinimumWidth(100)
        self.apply_icon_button_style(healthcare_debug_button)
        actions_layout.addWidget(healthcare_debug_button)
        
        # Add Hypothesis button
        hypothesis_button = QPushButton()
        hypothesis_button.setIcon(load_bootstrap_icon("graph-up"))
        hypothesis_button.setText("Hypothesis")
        hypothesis_button.setIconSize(QSize(18, 18))
        hypothesis_button.setToolTip("Add grouping variable for hypothesis testing")
        hypothesis_button.clicked.connect(self.add_grouping_variable)
        hypothesis_button.setFixedHeight(32)
        hypothesis_button.setMinimumWidth(100)
        self.apply_icon_button_style(hypothesis_button)
        actions_layout.addWidget(hypothesis_button)
        
        # Add vertical separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.VLine)
        separator2.setFrameShadow(QFrame.Shadow.Sunken)
        separator2.setFixedHeight(24)  # Shorter separator
        actions_layout.addWidget(separator2)
        
        # Settings button
        settings_button = QPushButton()
        settings_button.setIcon(load_bootstrap_icon("gear-fill"))
        settings_button.setText("Settings")
        settings_button.setIconSize(QSize(18, 18))  # Smaller icon
        settings_button.setToolTip("Settings")
        settings_button.clicked.connect(self.show_settings_dialog)
        settings_button.setFixedHeight(32)
        settings_button.setMinimumWidth(80)
        self.apply_icon_button_style(settings_button)
        actions_layout.addWidget(settings_button)
        
        toolbar_layout.addWidget(actions_frame)
        main_layout.addLayout(toolbar_layout)
        
        # Add separator line below toolbar
        header_separator = QFrame()
        header_separator.setFrameShape(QFrame.Shape.HLine)
        header_separator.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(header_separator)
        
        # Left panel - contains sources list
        left_panel = QWidget()
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(450)
        sources_list_layout = QVBoxLayout(left_panel)
        sources_list_layout.setContentsMargins(0, 0, 0, 0)
        
        # Sources list with header
        sources_group = QGroupBox("Available Datasets")
        sources_layout = QVBoxLayout(sources_group)
        
        # Add search box for filtering datasets
        search_layout = QHBoxLayout()
        search_icon = QLabel()
        search_icon.setPixmap(load_bootstrap_icon("search").pixmap(16, 16))
        search_layout.addWidget(search_icon)
        
        self.search_datasets = QLineEdit()
        self.search_datasets.setPlaceholderText("Search datasets...")
        search_layout.addWidget(self.search_datasets)
        sources_layout.addLayout(search_layout)
        
        # Connect search box to filter function
        self.search_datasets.textChanged.connect(self.filter_datasets)
        
        # Sources list
        self.sources_list = QListWidget()
        self.sources_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.sources_list.itemClicked.connect(self.on_source_selected)
        # Set icon size to ensure icons are visible
        self.sources_list.setIconSize(QSize(20, 20))
        sources_layout.addWidget(self.sources_list)
        
        # Dataset actions toolbar
        dataset_actions = QHBoxLayout()
        
        # Selection info - just show the count, nothing else
        self.selection_label = QLabel("0")
        self.selection_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.selection_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        dataset_actions.addWidget(self.selection_label)
        
        # Quick actions for selected datasets
        delete_btn = QToolButton()
        delete_btn.setIcon(load_bootstrap_icon("trash-fill"))
        delete_btn.setText("Delete")
        delete_btn.setToolTip("Delete selected dataset(s)")
        delete_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        delete_btn.setMinimumWidth(70)
        delete_btn.clicked.connect(self.delete_selected_datasets)
        self.apply_icon_button_style(delete_btn)
        dataset_actions.addWidget(delete_btn)
        
        save_btn = QToolButton()
        save_btn.setIcon(load_bootstrap_icon("save-fill"))
        save_btn.setText("Save")
        save_btn.setToolTip("Save selected dataset to file")
        save_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        save_btn.setMinimumWidth(70)
        save_btn.clicked.connect(lambda: self.save_source_to_file(self.current_source) if self.current_source else None)
        self.apply_icon_button_style(save_btn)
        dataset_actions.addWidget(save_btn)
        
        sources_layout.addLayout(dataset_actions)
        
        # Connect selection changed signal
        self.sources_list.itemSelectionChanged.connect(self.update_selection_label)
        
        sources_list_layout.addWidget(sources_group)
        
        # Right panel - contains dataset display
        right_panel = QWidget()
        self.setup_dataset_display(right_panel)
        
        # Add panels to splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 700])  # Adjust sizes to favor right panel
        
        # Make splitter take maximum available space
        splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        main_layout.addWidget(splitter, 1)  # Give splitter a stretch factor of 1
        
        # Add status bar at bottom with fixed height
        self.status_bar.setMaximumHeight(30)
        main_layout.addWidget(self.status_bar)
        
        # Set up context menu for sources list
        self.sources_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.sources_list.customContextMenuRequested.connect(self.show_sources_context_menu)
        
        # Remove automatic loading of debug datasets
        # self.add_debug_datasets()
    
    def setup_dataset_display(self, parent):
        """Setup the dataset display section"""
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)  # Reduce spacing between elements
        
        # Dataset display header with toolbar - more compact
        header_frame = QFrame()
        header_frame.setFrameShape(QFrame.Shape.StyledPanel)
        header_frame.setFrameShadow(QFrame.Shadow.Raised)
        header_frame.setLineWidth(1)
        header_frame.setStyleSheet("border: none;")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(8, 3, 8, 3)  # Reduce margins
        header_layout.setSpacing(8)  # Adjust spacing
        
        # Set a fixed height for the header
        header_frame.setFixedHeight(45)  # Even smaller height
        
        # Info section (left)
        info_section = QHBoxLayout()  # Changed to horizontal layout
        info_section.setSpacing(10)
        
        # Dataset title with icon
        icon = load_bootstrap_icon("table")
        icon_pixmap = icon.pixmap(16, 16)
        
        if not icon_pixmap.isNull():
            dataset_icon = QLabel()
            dataset_icon.setPixmap(icon_pixmap)
            info_section.addWidget(dataset_icon)
        
        self.current_dataset_label = QLabel("No dataset selected")
        self.current_dataset_label.setStyleSheet("font-weight: bold; font-size: 14px;")  # Smaller font
        info_section.addWidget(self.current_dataset_label)
        
        # Dataset info with metadata status
        self.dataset_info_label = QLabel("")
        info_section.addWidget(self.dataset_info_label)
        
        # Metadata status indicator
        self.metadata_status_label = QLabel("")
        info_section.addWidget(self.metadata_status_label)
        info_section.addStretch()
        
        header_layout.addLayout(info_section)
        
        # Action buttons toolbar (right)
        actions_toolbar = QHBoxLayout()
        actions_toolbar.setSpacing(5)
        
        # Generate metadata button for all selected datasets (simpler naming)
        metadata_btn = QPushButton()
        metadata_btn.setIcon(load_bootstrap_icon("magic"))
        metadata_btn.setText("Metadata")
        metadata_btn.setIconSize(QSize(18, 18))
        metadata_btn.setToolTip("Generate metadata for selected dataset(s)")
        metadata_btn.clicked.connect(self.batch_generate_metadata)
        metadata_btn.setFixedHeight(32)
        metadata_btn.setMinimumWidth(90)
        self.apply_icon_button_style(metadata_btn)
        actions_toolbar.addWidget(metadata_btn)
        
        # View metadata button
        self.view_metadata_btn = QPushButton()
        self.view_metadata_btn.setIcon(load_bootstrap_icon("info-circle-fill"))
        self.view_metadata_btn.setText("View")
        self.view_metadata_btn.setIconSize(QSize(18, 18))
        self.view_metadata_btn.setEnabled(False)
        self.view_metadata_btn.clicked.connect(self.view_current_metadata)
        self.view_metadata_btn.setToolTip("View metadata for the current dataset")
        self.view_metadata_btn.setFixedHeight(32)
        self.view_metadata_btn.setMinimumWidth(70)
        self.apply_icon_button_style(self.view_metadata_btn)
        actions_toolbar.addWidget(self.view_metadata_btn)
        
        # Add a small vertical separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setFixedHeight(20)
        actions_toolbar.addWidget(separator)
        
        # Analyze relationships button (moved from top toolbar)
        analyze_btn = QPushButton()
        analyze_btn.setIcon(load_bootstrap_icon("diagram-3-fill"))
        analyze_btn.setText("Analyze")
        analyze_btn.setIconSize(QSize(18, 18))
        analyze_btn.setToolTip("Analyze relationships between datasets")
        analyze_btn.clicked.connect(self.show_dataset_relationships)
        analyze_btn.setFixedHeight(32)
        analyze_btn.setMinimumWidth(80)
        self.apply_icon_button_style(analyze_btn)
        actions_toolbar.addWidget(analyze_btn)
        
        header_layout.addLayout(actions_toolbar)
        layout.addWidget(header_frame)
        
        # Create dataset overview panel that displays a summary of the dataset
        self.dataset_overview = QFrame()
        self.dataset_overview.setFrameShape(QFrame.Shape.StyledPanel)
        self.dataset_overview.setFrameShadow(QFrame.Shadow.Sunken)
        self.dataset_overview.setStyleSheet("""
            QFrame {
                border: none;
            }
            .stat-value {
                font-weight: bold;
            }
        """)
        overview_layout = QHBoxLayout(self.dataset_overview)
        overview_layout.setContentsMargins(10, 5, 10, 5)
        
        # Creating a more structured and visual overview with cards for each stat
        self.overview_stats = QHBoxLayout()
        self.overview_stats.setSpacing(15)
        
        # We'll dynamically add stat cards here in display_dataset function
        overview_layout.addLayout(self.overview_stats)
        
        # Set fixed height for overview
        self.dataset_overview.setFixedHeight(80)
        self.dataset_overview.setVisible(False)  # Hide until data is loaded
        
        layout.addWidget(self.dataset_overview)
        
        # Create dataset display table
        self.dataset_display = DataFrameDisplay()
        layout.addWidget(self.dataset_display, 1)  # Give table all remaining space
    
    def display_dataset(self, name, dataframe):
        """Display the selected dataset in the table view"""
        self.current_dataset_label.setText(f"Dataset: {name}")
        
        # Update dataset info
        rows, cols = dataframe.shape
        self.dataset_info_label.setText(f"Rows: {rows} | Columns: {cols}")
        
        # Update metadata status
        has_metadata = False
        if hasattr(self, 'datasets_metadata') and name in self.datasets_metadata:
            has_metadata = True
        elif name in self.source_connections and hasattr(self.source_connections[name], 'metadata') and self.source_connections[name].metadata:
            has_metadata = True
            
        if has_metadata:
            self.metadata_status_label.setText("✓ Metadata Available")
            self.view_metadata_btn.setEnabled(True)
        else:
            self.metadata_status_label.setText("✗ No Metadata")
            self.view_metadata_btn.setEnabled(False)
            
        # Display in table
        self.dataset_display.display_dataframe(dataframe)
        
        # Clear previous overview stats
        for i in reversed(range(self.overview_stats.count())): 
            item = self.overview_stats.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
            self.overview_stats.removeItem(item)
            
        # Add basic stats card
        basic_card = self.create_stat_card(
            "Dataset Size", 
            f"{rows} rows × {cols} columns",
            "table"
        )
        self.overview_stats.addWidget(basic_card)
        
        # Add missing data info
        missing_count = dataframe.isna().sum().sum()
        missing_pct = (missing_count / (rows * cols)) * 100 if rows * cols > 0 else 0
        missing_card = self.create_stat_card(
            "Missing Data",
            f"{missing_count} cells ({missing_pct:.1f}%)",
            "exclamation-triangle"
        )
        self.overview_stats.addWidget(missing_card)
        
        # Add memory usage
        memory_usage = dataframe.memory_usage(deep=True).sum()
        memory_text = f"{memory_usage / 1024 / 1024:.2f} MB"
        memory_card = self.create_stat_card(
            "Memory Usage",
            memory_text,
            "hdd"
        )
        self.overview_stats.addWidget(memory_card)
        
        # Show the overview section
        self.dataset_overview.setVisible(True)
    
    def create_stat_card(self, title, value, icon_name):
        """Create a card for displaying a statistic"""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                border-radius: 4px;
            }
            QLabel[class="title"] {
                font-size: 12px;
            }
            QLabel[class="value"] {
                font-weight: bold;
                font-size: 13px;
            }
        """)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)
        
        # Title section with icon
        title_layout = QHBoxLayout()
        title_layout.setSpacing(5)
        
        # Create icon and ensure it's visible
        icon = load_bootstrap_icon(icon_name, color="#0d6efd")
        icon_pixmap = icon.pixmap(16, 16)
        
        if not icon_pixmap.isNull():
            icon_label = QLabel()
            icon_label.setPixmap(icon_pixmap)
            icon_label.setMinimumSize(16, 16)  # Ensure the label is large enough
            title_layout.addWidget(icon_label)
        
        title_label = QLabel(title)
        title_label.setProperty("class", "title")
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        
        layout.addLayout(title_layout)
        
        # Value
        value_label = QLabel(value)
        value_label.setProperty("class", "value")
        value_label.setWordWrap(True)
        layout.addWidget(value_label)
        
        return card
    
    def view_current_metadata(self):
        """View metadata for the current dataset"""
        if self.current_source:
            self.show_metadata_dialog(self.current_source)
            
    def generate_current_metadata(self):
        """Generate metadata for the current dataset - redirects to batch method"""
        if self.current_source:
            # Select only the current source in the list
            self.sources_list.clearSelection()
            items = self.sources_list.findItems(self.current_source, Qt.MatchFlag.MatchExactly)
            if items:
                items[0].setSelected(True)
                self.batch_generate_metadata()
            else:
                # Fallback to direct call
                self.generate_dataset_metadata(self.current_source)
    
    def show_add_source_dialog(self):
        """Show the add source dialog"""
        dialog = DataSourceDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            source_type, connection_info = dialog.get_source_info()
            if source_type is not None and connection_info is not None:
                # Start async task to handle the connection
                asyncio.create_task(self.handle_source_connection_wrapper(source_type, connection_info))
                self.update_status(f"Connecting to {source_type} source...")
            else:
                QMessageBox.warning(self, "Error", "Please complete all required fields")
    
    def show_generate_study_dialog(self):
        """Show dialog to generate synthetic study data"""
        dialog = StudyGenerationDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            inputs = dialog.get_inputs()
            if not inputs['description'].strip():
                QMessageBox.warning(self, "Error", "Please enter a study description")
                return
            if not inputs['name'].strip():
                QMessageBox.warning(self, "Error", "Please enter a dataset name")
                return
            
            # Start async task to generate data
            asyncio.create_task(self.generate_study_data(inputs))
            self.update_status("Generating synthetic study data...")
    
    @asyncSlot()
    async def handle_source_connection_wrapper(self, source_type, connection_info):
        """Handle the asynchronous connection to a data source and load data"""
        try:
            # Handle file uploads differently - process each file separately
            if source_type == "upload" and "files" in connection_info and connection_info["files"]:
                # Track if we've displayed at least one dataset
                displayed_dataset = False
                
                # Process each file
                for file_path in connection_info["files"]:
                    try:
                        file_name = os.path.basename(file_path)
                        self.update_status(f"Loading file: {file_name}...")
                        
                        # Create source connection for this specific file
                        file_connection = SourceConnection(source_type, {"files": [file_path]}, file_name)
                        
                        # Try to connect and load data
                        await file_connection.data_source.connect()
                        
                        # Load the data
                        df = await file_connection.load_data()
                        
                        # Add the source to our collection
                        self.add_source(file_name, file_connection, df)
                        
                        # Display the first dataset we successfully load
                        if not displayed_dataset:
                            self.display_dataset(file_name, df)
                            displayed_dataset = True
                            
                    except Exception as e:
                        self.update_status(f"Error loading file {file_name}: {str(e)}")
                        QMessageBox.critical(self, "Error", f"Failed to load file {file_name}: {str(e)}")
                
                if displayed_dataset:
                    self.update_status(f"Successfully loaded {len(connection_info['files'])} file(s)")
                else:
                    self.update_status("No files were successfully loaded")
                    
            else:
                # Handle other source types (SQL, SFTP, REST)
                # Create a default file name based on source type
                file_name = None
                if source_type == "sql":
                    file_name = f"{connection_info.get('database', 'database')}_query"
                elif source_type == "sftp":
                    file_name = os.path.basename(connection_info.get('path', 'remote_file'))
                elif source_type == "rest":
                    file_name = f"api_{connection_info.get('method', 'GET').lower()}"
                
                # Create source connection
                connection = SourceConnection(source_type, connection_info, file_name)
                
                # Try to connect and load data
                await connection.data_source.connect()
                self.update_status(f"Loading data from {source_type} source...")
                
                # Load the data
                df = await connection.load_data()
                
                # Add the source to our collection
                source_name = file_name or f"{source_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.add_source(source_name, connection, df)
                
                # Display the data
                self.display_dataset(source_name, df)
                
                self.update_status(f"Successfully loaded data from {source_type} source")
            
        except Exception as e:
            self.update_status(f"Error loading data: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
    
    # Core functionality from DataCollectionWidget
    def update_status(self, message):
        """Update status bar with message"""
        if hasattr(self, 'status_bar') and self.status_bar is not None:
            self.status_bar.showMessage(message)
        else:
            print(f"Status bar not initialized yet: {message}")
        
    def on_source_selected(self, item):
        """Handle source selection"""
        # Only update the display if a single item is selected
        if len(self.sources_list.selectedItems()) == 1:
            name = item.text()
            if name in self.dataframes:
                self.display_dataset(name, self.dataframes[name])
                self.source_selected.emit(name, self.dataframes[name])
                self.current_source = name
    
    def add_source(self, name, connection, dataframe):
        """Add a source to the list"""
        if name in self.dataframes:
            # Update existing source
            self.dataframes[name] = dataframe
            self.source_connections[name] = connection
            self.update_status(f"Updated source: {name}")
            self.source_updated.emit(name, dataframe)
        else:
            # Add new source
            self.dataframes[name] = dataframe
            self.source_connections[name] = connection
            
            # Add to list widget with dataset icon
            item = QListWidgetItem(name)
            icon = load_bootstrap_icon("table")
            # Force a larger size for icon to ensure visibility
            item.setSizeHint(QSize(item.sizeHint().width(), 28))
            item.setIcon(icon)
            self.sources_list.addItem(item)
            
            self.update_status(f"Added source: {name}")
            
            # Emit the source_added signal
            self.source_added.emit(name, dataframe)
        
        # Update the current selection
        items = self.sources_list.findItems(name, Qt.MatchFlag.MatchExactly)
        if items:
            self.sources_list.setCurrentItem(items[0])
            self.on_source_selected(items[0])
    
    def add_debug_datasets(self):
        """Add debug datasets for testing and demonstration."""
        try:
            # Import the debug datasets module
            from data.collection.debug_datasets import final_datasets
            
            # Add each dataset from the final_datasets dictionary
            for name, df in final_datasets.items():
                # Create a simple file connection for these datasets
                file_connection = SourceConnection("file", {"path": f"debug/{name}.csv"})
                self.add_source(name, file_connection, df)
            
            self.update_status(f"Added {len(final_datasets)} debug datasets")
        except Exception as e:
            self.update_status(f"Error adding debug datasets: {str(e)}")

    @asyncSlot()
    async def generate_study_data(self, inputs):
        """Generate synthetic study data based on description"""
        try:
            # Prepare prompt for Gemini with specific code structure instructions
            prompt = f"""
            Generate Python code to create a synthetic dataset for the following study:
            {inputs['description']}
            
            IMPORTANT: Use the following specific code structure to avoid any string formatting errors:
            
            1. Define variables at the top (n, seed, etc.)
            2. Create separate functions for generating each type of variable
            3. DO NOT use f-strings or string format operations
            4. Use string concatenation ('+') for ID generation
            5. Return a pandas DataFrame named 'data' (this exact variable name is required)
            6. Create variables that cover patient identifier, covariates, outcomes, grouping variables, etc.
            7. Add timepoint based outcomes such as SysBP_Baseline, SysBP_Week1, SysBP_Week2 etc.
            
            Here's the exact pattern to follow:
            
            ```python
            import pandas as pd
            import numpy as np
            
            # Set random seed for reproducibility
            np.random.seed(42)
            
            # Number of participants and study parameters
            n_participants = 100
            
            # Generate ID string without using f-strings
            def generate_patient_ids(n):
                ids = []
                for i in range(1, n + 1):
                    # Zero-pad using string operations
                    num_str = str(i)
                    while len(num_str) < 3:
                        num_str = "0" + num_str
                    ids.append("P" + num_str)
                return ids
            
            # Generate age values    
            def generate_ages(n):
                ages = np.random.normal(50, 15, n).astype(int)
                ages = np.clip(ages, 18, 90)  # Ensure reasonable age range
                return ages
                
            # Generate treatment assignment
            def generate_groups(n):
                return np.random.choice(['Control', 'Treatment'], n)
            
            # Generate outcome based on treatment and age
            def generate_outcomes(groups, ages):
                # Base outcome correlated with age
                outcomes = 0.5 * ages + np.random.normal(0, 10, len(ages))
                
                # Treatment effect (higher values = better outcomes)
                treatment_effect = 15.0
                outcomes[groups == 'Treatment'] += treatment_effect
                
                return outcomes
                
            # Generate blood pressure measurements at different timepoints
            def generate_bp_measurements(groups, ages, n_timepoints=3):
                # Dictionary to store measurements for each timepoint
                bp_data = {{}}
                
                # Base systolic BP correlated with age
                base_systolic = 110 + 0.4 * ages
                
                # Generate measurements for each timepoint
                for t in range(n_timepoints):
                    timepoint_name = "Baseline" if t == 0 else "Week" + str(t)
                    
                    # Add treatment effect that increases over time
                    treatment_effect = 5.0 * t  # Effect increases with each timepoint
                    
                    # Generate systolic values with noise
                    systolic = base_systolic + np.random.normal(0, 5, len(ages))
                    
                    # Apply treatment effect for treatment group
                    systolic[groups == 'Treatment'] -= treatment_effect
                    
                    # Store in dictionary with appropriate column name
                    bp_data["SysBP_" + timepoint_name] = systolic
                    
                    # Generate corresponding diastolic values (correlated with systolic)
                    diastolic = 0.6 * systolic + np.random.normal(20, 5, len(ages))
                    bp_data["DiaBP_" + timepoint_name] = diastolic
                
                return bp_data
            
            # Generate all variables
            patient_ids = generate_patient_ids(n_participants)
            ages = generate_ages(n_participants)
            groups = generate_groups(n_participants)
            outcomes = generate_outcomes(groups, ages)
            
            # Generate BP measurements at multiple timepoints
            bp_data = generate_bp_measurements(groups, ages, n_timepoints=3)
            
            # Create the final DataFrame
            data = pd.DataFrame({{
                'patient_id': patient_ids,
                'age': ages,
                'group': groups,
                'outcome': outcomes,
                **bp_data  # Unpack the BP measurements
            }})
            ```
            
            Follow this exact pattern but adapt it to the study description.
            NEVER use f-strings or format() method as these cause errors.
            Use simple string concatenation with '+' for any string formatting.
            Generate a reasonable number of participants (50-200) based on the study.
            
            Only return the Python code, no other text.
            """
            
            # Call Gemini API
            self.update_status("Generating synthetic data...")
            response = await call_llm_async(prompt)
            
            # Clean up the code - remove any markdown formatting or invisible characters
            code = self.sanitize_grouping_code(response)
            
            try:
                # Execute the generated code in a safe environment
                local_vars = {}
                global_vars = {
                    'pd': pd, 
                    'np': np,
                    'range': range,
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'round': round,
                    'list': list,
                    'dict': dict
                }
                
                # Execute the code
                try:
                    exec(code, global_vars, local_vars)
                except SyntaxError as se:
                    # Enhanced syntax error reporting
                    line_num = se.lineno if hasattr(se, 'lineno') else 0
                    
                    # Find the problematic line and surrounding context
                    code_lines = code.split('\n')
                    context_start = max(0, line_num - 3)
                    context_end = min(len(code_lines), line_num + 2)
                    context = '\n'.join([f"{i+1}: {line}" for i, line in enumerate(code_lines[context_start:context_end])])
                    
                    error_details = f"Syntax error at line {line_num}:\n{context}\n\nError details: {str(se)}"
                    raise ValueError(f"Generated code has syntax errors: {error_details}")
                
                # Look specifically for 'data' DataFrame
                if 'data' in local_vars and isinstance(local_vars['data'], pd.DataFrame):
                    df = local_vars['data']
                    
                    # Basic validation of the DataFrame
                    if len(df) == 0:
                        raise ValueError("Generated DataFrame is empty")
                    
                    # Convert boolean columns to 1/0 integers
                    for col in df.columns:
                        if df[col].dtype == bool:
                            df[col] = df[col].astype(int)  # True becomes 1, False becomes 0
                    
                    # Add as a new source
                    connection = SourceConnection('generated', {'code': code}, inputs['name'])
                    self.add_source(inputs['name'], connection, df)
                    self.update_status(f"Generated study dataset '{inputs['name']}' with {len(df)} rows")
                else:
                    raise ValueError("No DataFrame named 'data' was generated by the code")
                    
            except NameError as ne:
                raise ValueError(f"Generated code uses undefined variables: {str(ne)}")
            except Exception as code_error:
                raise ValueError(f"Error executing generated code: {str(code_error)}")
                
        except Exception as e:
            error_msg = str(e)
            self.update_status(f"Error generating study data: {error_msg}")
            
            # Show more detailed error message to user
            error_dialog = QMessageBox(self)
            error_dialog.setIcon(QMessageBox.Icon.Critical)
            error_dialog.setWindowTitle("Error Generating Study Data")
            error_dialog.setText("Failed to generate study data")
            error_dialog.setInformativeText(error_msg)
            
            if 'code' in locals():
                error_dialog.setDetailedText(
                    "Generated code that caused the error:\n\n" + code
                )
            
            error_dialog.exec()

    def sanitize_grouping_code(self, code, variable_name):
        """
        Sanitize the grouping variable code by removing extraneous code
        
        Args:
            code (str): The code generated by the LLM
            variable_name (str): The name of the grouping variable
            
        Returns:
            str: Cleaned code with only the essential parts
        """
        # First do basic preprocessing
        code = self.preprocess_code(code)
        
        # Split code into lines
        lines = code.split('\n')
        
        # Filter out __main__ section
        if '__main__' in code:
            main_index = -1
            for i, line in enumerate(lines):
                if '__main__' in line:
                    main_index = i
                    break
            
            if main_index > 0:
                lines = lines[:main_index]
        
        # Filter out sample dataframe creation
        filtered_lines = []
        skip_block = False
        brace_count = 0
        
        for line in lines:
            # Skip sample dataframe creation
            if 'sample' in line and 'data' in line and '=' in line and '{' in line:
                skip_block = True
                brace_count = line.count('{') - line.count('}')
                continue
                
            # Track brace count to know when the sample data block ends
            if skip_block:
                brace_count += line.count('{') - line.count('}')
                if brace_count <= 0:
                    skip_block = False
                continue
                
            # Skip example usage
            if 'example' in line.lower() and 'usage' in line.lower():
                continue
                
            # Skip print statements
            if 'print(' in line:
                continue
                
            # Skip creating sample dataframes
            if 'pd.DataFrame' in line and ('data' in line or '{' in line):
                continue
                
            # Skip function definitions if they don't add the grouping variable
            if 'def ' in line and 'add_grouping_variable' not in line and 'create_' not in line:
                continue
                
            filtered_lines.append(line)
        
        # Join remaining lines
        cleaned_code = '\n'.join(filtered_lines)
        
        # Add a comment explaining what this code does
        header = f"# Add '{variable_name}' grouping variable for hypothesis testing\n"
        
        return header + cleaned_code

    def preprocess_code(self, text):
        """
        Preprocess code to fix common issues
        
        Args:
            text (str): Raw code text from API
            
        Returns:
            str: Cleaned code
        """
        # Remove markdown code block markers
        text = text.replace('```python', '').replace('```', '').strip()
        
        # Remove any UTF-8 BOM if present
        if text.startswith('\ufeff'):
            text = text[1:]
        
        # Remove other potential invisible characters
        text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\r\t')
        
        # Ensure newlines are standardized
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Make sure the code starts with imports
        if not text.startswith('import '):
            # Look for the first import statement
            import_index = text.find('import ')
            if import_index > 0:
                # Remove anything before the first import
                text = text[import_index:]
        
        return text
        
    def show_sources_context_menu(self, position):
        """Show context menu for sources list"""
        context_menu = QMenu()
        
        # Apply consistent icon sizes for context menu
        context_menu.setStyleSheet("""
            QMenu::item:selected { background-color: #e9ecef; color: #212529; }
            QMenu::icon { min-width: 24px; }
        """)
        
        # If only one item is selected, show full options
        if len(self.sources_list.selectedItems()) == 1:
            name = self.sources_list.selectedItems()[0].text()
            
            # View metadata
            view_action = context_menu.addAction("View Metadata")
            view_action.setIcon(load_bootstrap_icon("info-circle-fill"))
            view_action.triggered.connect(lambda: self.show_metadata_dialog(name))
            
            # Generate metadata
            generate_action = context_menu.addAction("Generate Metadata")
            generate_action.setIcon(load_bootstrap_icon("magic"))
            generate_action.triggered.connect(lambda: asyncio.create_task(self.generate_dataset_metadata(name)))
            
            context_menu.addSeparator()
            
            # Save to file
            save_action = context_menu.addAction("Save to File")
            save_action.setIcon(load_bootstrap_icon("save-fill"))
            save_action.triggered.connect(lambda: self.save_source_to_file(name))
            
            # Rename
            rename_action = context_menu.addAction("Rename")
            rename_action.setIcon(load_bootstrap_icon("pencil-fill"))
            rename_action.triggered.connect(lambda: self.rename_source(name))
            
            context_menu.addSeparator()
            
            # Delete
            delete_action = context_menu.addAction("Delete")
            delete_action.setIcon(load_bootstrap_icon("trash-fill"))
            delete_action.triggered.connect(lambda: self.delete_source(name))
        
        # Multiple selection options
        else:
            count = len(self.sources_list.selectedItems())
            
            # Generate metadata for all selected
            generate_action = context_menu.addAction(f"Generate Metadata for Selected ({count})")
            generate_action.setIcon(load_bootstrap_icon("magic"))
            generate_action.triggered.connect(self.batch_generate_metadata)
            
            context_menu.addSeparator()
            
            # Delete selected option
            delete_action = context_menu.addAction(f"Delete Selected ({count})")
            delete_action.setIcon(load_bootstrap_icon("trash-fill"))
            delete_action.triggered.connect(self.delete_selected_datasets)
        
        context_menu.exec(self.sources_list.mapToGlobal(position))
    
    def delete_selected_datasets(self):
        """Delete all selected datasets"""
        selected_items = self.sources_list.selectedItems()
        if not selected_items:
            return
            
        names = [item.text() for item in selected_items]
        count = len(names)
        
        # Ask for confirmation
        confirm = QMessageBox.question(
            self, 
            "Confirm Deletion", 
            f"Are you sure you want to delete {count} selected datasets?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if confirm == QMessageBox.StandardButton.Yes:
            # Delete each dataset
            for name in names:
                self.delete_source(name)
                
            self.update_status(f"Deleted {count} datasets")
    
    def delete_source(self, name):
        """Delete a source from the list"""
        if name in self.dataframes:
            # Ask for confirmation
            confirm = QMessageBox.question(
                self, 
                "Confirm Deletion", 
                f"Are you sure you want to delete the dataset '{name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if confirm == QMessageBox.StandardButton.Yes:
                # Get reference to the main app window and studies manager
                main_window = self.window()
                if hasattr(main_window, 'studies_manager'):
                    try:
                        # Delete from studies manager first
                        if not main_window.studies_manager.remove_dataset_from_active_study(name):
                            self.update_status(f"Warning: Unable to delete dataset from study, but will remove from UI")
                    except Exception as e:
                        self.update_status(f"Warning: Error removing dataset from study: {str(e)}")
                
                # Remove from local collections
                del self.dataframes[name]
                if name in self.source_connections:
                    del self.source_connections[name]
                
                # Remove from list widget
                items = self.sources_list.findItems(name, Qt.MatchFlag.MatchExactly)
                if items:
                    self.sources_list.takeItem(self.sources_list.row(items[0]))
                
                # Update UI components that might reference this dataset
                try:
                    self.update_join_sources()
                except:
                    pass
                    
                try:
                    self.update_append_sources()
                except:
                    pass
                
                # Emit the source_deleted signal
                self.source_deleted.emit(name)
                
                # Clear display if this was the current source
                if self.current_source == name:
                    self.current_source = None
                    self.current_dataset_label.setText("No dataset selected")
                    self.dataset_info_label.setText("")
                    self.dataset_display.clear()
                
                self.update_status(f"Deleted source: {name}")
    
    def rename_source(self, name):
        """Rename a source in the list"""
        new_name, ok = QInputDialog.getText(self, "Rename Source", "Enter new name:", text=name)
        if ok and new_name and new_name != name:
            if new_name in self.dataframes:
                QMessageBox.warning(self, "Error", "Source with this name already exists")
            else:
                # Update local collections
                self.dataframes[new_name] = self.dataframes.pop(name)
                if name in self.source_connections:
                    self.source_connections[new_name] = self.source_connections.pop(name)
                
                # Update list widget
                items = self.sources_list.findItems(name, Qt.MatchFlag.MatchExactly)
                if items:
                    row = self.sources_list.row(items[0])
                    self.sources_list.takeItem(row)
                    self.sources_list.insertItem(row, new_name)
                    self.sources_list.setCurrentRow(row)
                
                # Update sources in both operations
                self.update_join_sources()
                self.update_append_sources()
                
                # Emit the source_renamed signal instead of directly updating studies manager
                self.source_renamed.emit(name, new_name)
                
                # Update current source if this was the current source
                if self.current_source == name:
                    self.current_source = new_name
                    self.current_dataset_label.setText(f"Dataset: {new_name}")
                
                self.update_status(f"Renamed source: {name} -> {new_name}")
    
    def save_source_to_file(self, name):
        """Save a dataset to a file"""
        if name in self.dataframes:
            df = self.dataframes[name]
            
            # Ask for file type and location
            options = QFileDialog.Option.DontUseNativeDialog
            file_name, selected_filter = QFileDialog.getSaveFileName(
                self, 
                "Save Dataset", 
                name, 
                "CSV Files (*.csv);;Excel Files (*.xlsx);;TSV Files (*.tsv);;JSON Files (*.json)",
                options=options
            )
            
            if file_name:
                try:
                    # Determine file format based on extension or selected filter
                    if file_name.endswith('.csv') or selected_filter == "CSV Files (*.csv)":
                        if not file_name.endswith('.csv'):
                            file_name += '.csv'
                        df.to_csv(file_name, index=False)
                    elif file_name.endswith('.xlsx') or selected_filter == "Excel Files (*.xlsx)":
                        if not file_name.endswith('.xlsx'):
                            file_name += '.xlsx'
                        df.to_excel(file_name, index=False)
                    elif file_name.endswith('.tsv') or selected_filter == "TSV Files (*.tsv)":
                        if not file_name.endswith('.tsv'):
                            file_name += '.tsv'
                        df.to_csv(file_name, sep='\t', index=False)
                    elif file_name.endswith('.json') or selected_filter == "JSON Files (*.json)":
                        if not file_name.endswith('.json'):
                            file_name += '.json'
                        df.to_json(file_name, orient='records')
                    else:
                        # Default to CSV if no recognized extension
                        df.to_csv(file_name, index=False)
                    
                    self.update_status(f"Saved dataset '{name}' to {file_name}")
                    QMessageBox.information(self, "Success", f"Dataset saved successfully to {file_name}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save dataset: {str(e)}")
                    self.update_status(f"Error saving dataset: {str(e)}")

    def update_dataset(self, name, dataframe):
        """Update an existing dataset with new data"""
        if name in self.dataframes:
            # Update the dataframe
            self.dataframes[name] = dataframe
            
            # If this is the current source, update the display
            if self.current_source == name:
                self.display_dataset(name, dataframe)
            
            # Emit the source_updated signal instead of directly updating studies manager
            self.source_updated.emit(name, dataframe)
            
            self.update_status(f"Updated dataset: {name}")
            return True
        else:
            # If dataset doesn't exist, add it as a new source
            connection = SourceConnection("updated", {}, name)
            self.add_source(name, connection, dataframe)
            return True
    def refresh_datasets(self):
        """Refresh all datasets from the studies manager"""
        # Get reference to the main app window
        main_window = self.window()
        if not hasattr(main_window, 'studies_manager'):
            QMessageBox.warning(self, "Error", "Could not access studies manager")
            return
        
        # Get active study
        study = main_window.studies_manager.get_active_study()
        if not study:
            QMessageBox.warning(self, "Error", "No active study found")
            return
        
        # Check if study has datasets
        if not hasattr(study, 'available_datasets') or not study.available_datasets:
            QMessageBox.information(self, "Info", "No datasets available in the active study")
            return
        
        # Process each dataset
        count = 0
        for dataset in study.available_datasets:
            # Handle both dictionary and namedtuple formats
            if isinstance(dataset, dict):
                name = dataset.get('name')
                dataframe = dataset.get('data')
                metadata = dataset.get('metadata')
            else:
                # Handle legacy namedtuple format
                name = dataset.name
                dataframe = dataset.data
                metadata = None
            
            if name and isinstance(dataframe, pd.DataFrame):
                # Check if the dataset is already in our list
                if name in self.dataframes:
                    # Update existing dataset only if content has changed
                    existing_df = self.dataframes[name]
                    if not existing_df.equals(dataframe):
                        # Update the existing dataset
                        self.dataframes[name] = dataframe
                        
                        # Update metadata if available
                        if metadata and name in self.source_connections:
                            self.source_connections[name].metadata = metadata
                        
                        count += 1
                else:
                    # Create a connection object for new dataset
                    connection = SourceConnection("refreshed", {}, name)
                    if metadata:
                        connection.metadata = metadata
                    
                    # Add as a new source
                    self.source_connections[name] = connection
                    self.dataframes[name] = dataframe
                    self.sources_list.addItem(name)
                    count += 1
        
        # Update the displayed dataset if needed
        if self.current_source and self.current_source in self.dataframes:
            self.display_dataset(self.current_source, self.dataframes[self.current_source])
        
        # If we have a grouping result, reapply highlighting
        if self.latest_grouping_result:
            self.highlight_grouped_datasets(self.latest_grouping_result)
        
        if count > 0:
            self.update_status(f"Refreshed {count} datasets")
            QMessageBox.information(self, "Success", f"Successfully refreshed {count} datasets")
        else:
            QMessageBox.information(self, "Info", "All datasets are already up to date")

    @asyncSlot()
    async def generate_dataset_metadata(self, name):
        """Generate metadata for a dataset"""
        if name not in self.dataframes:
            self.update_status(f"Error: Dataset '{name}' not found")
            return
        
        self.update_status(f"Generating metadata for {name}...")
        
        try:
            # Get the dataframe
            df = self.dataframes[name]
            
            # Generate masked column info using existing masking utility
            try:
                masked_mapping = get_column_mapping(df)
                if not masked_mapping:
                    raise ValueError("Failed to generate column mapping")
            except Exception as e:
                self.update_status(f"Error generating column mapping: {str(e)}")
                QMessageBox.warning(
                    self, 
                    "Error", 
                    f"Failed to analyze dataset columns: {str(e)}"
                )
                return None
                
            # Create prompt with masked dataset info
            prompt = f"""
            I have a dataset named '{name}' with the following structure:
            - Rows: {len(df)}
            - Columns: {len(df.columns)}
            
            Column information with masked values (to protect sensitive data):
            """
            
            # Add column information
            for col, info in masked_mapping.get('column_mappings', {}).items():
                dtype = str(info.get('type', 'unknown'))
                # Handle sample values which might be in different formats
                sample_values = []
                if 'value_distribution' in info:
                    value_dist = info['value_distribution']
                    if isinstance(value_dist, dict) and 'sample_values' in value_dist:
                        sample_values = value_dist['sample_values']
                    elif isinstance(value_dist, list):
                        sample_values = value_dist
                    elif isinstance(value_dist, dict) and 'first_values' in value_dist:
                        sample_values = value_dist['first_values']
                
                # Ensure sample_values is a list
                if not isinstance(sample_values, list):
                    sample_values = [sample_values] if sample_values else []
                    
                examples_str = ', '.join([str(ex) for ex in sample_values[:3]])
                null_pct = info.get('null_percentage', 0)
                
                prompt += f"\n- {col} ({dtype})"
                prompt += f"\n  - Example values: {examples_str}"
                if null_pct > 0:
                    prompt += f"\n  - Contains {null_pct:.1f}% null values"
                
                # Add unique values count for categorical
                unique_count = info.get('unique_count', 0)
                prompt += f"\n  - {unique_count} unique values"
                
                # Add range for numeric columns
                if 'min' in info and 'max' in info:
                    prompt += f"\n  - Range: {info['min']} to {info['max']}"
                    
                    if 'mean' in info:
                        prompt += f", Mean: {info['mean']:.2f}"
                
            prompt += """

            Based on this dataset, please generate comprehensive metadata in JSON format with the following structure:
            {
                "name": "dataset name",
                "description": "brief description of what this dataset contains",
                "purpose": "likely purpose of this dataset",
                "column_descriptions": {
                    "column_name": {
                        "description": "what this column represents",
                        "data_type": "inferred data type",
                        "possible_values": ["list if categorical"],
                        "constraints": "any constraints on values",
                        "notes": "any additional notes"
                    },
                    ...
                },
                "relationships": [
                    {
                        "type": "likely relationship with other data",
                        "description": "what the relationship might be"
                    }
                ],
                "quality_issues": ["list any potential quality issues observed"],
                "recommended_uses": ["potential uses for this data"],
                "data_warnings": ["any warnings about using this data"],
                "sensitivity_level": "LOW/MEDIUM/HIGH based on data content"
            }

            Important: Only include the JSON response with no additional commentary. 
            Ensure the JSON is properly formatted with quotes around all property names.
            """
            
            # Call LLM to generate metadata
            metadata_json = await call_llm_async(self.preprocess_code(prompt), max_tokens=2000)
            
            # Clean the JSON string to remove common LLM formatting issues
            try:
                # First, try with simple JSON cleaning
                cleaned_json = metadata_json.strip()
                # Remove any markdown code block markers
                if cleaned_json.startswith("```json"):
                    cleaned_json = cleaned_json[7:]
                elif cleaned_json.startswith("```"):
                    cleaned_json = cleaned_json[3:]
                if cleaned_json.endswith("```"):
                    cleaned_json = cleaned_json[:-3]
                
                # Try to parse the cleaned JSON
                metadata = json.loads(cleaned_json)
                
                # Store metadata in source connection object
                if name in self.source_connections:
                    self.source_connections[name].metadata = metadata
                
                # Create a dictionary to store metadata if it doesn't exist
                if not hasattr(self, 'datasets_metadata'):
                    self.datasets_metadata = {}
                
                # Store the metadata
                self.datasets_metadata[name] = metadata
                
                # Update to studies manager
                main_window = self.window()
                if hasattr(main_window, 'studies_manager'):
                    # Update metadata in studies manager
                    main_window.studies_manager.update_dataset_metadata(name, metadata)
                
                # Update the source's metadata status in the UI
                try:
                    items = self.sources_list.findItems(name, Qt.MatchFlag.MatchExactly)
                    if items and len(items) > 0:
                        # Try to load icon and handle potential failures
                        try:
                            icon = load_bootstrap_icon("check-square")
                            # Set a fixed size for the item to ensure the icon is visible
                            items[0].setSizeHint(QSize(items[0].sizeHint().width(), 28))
                            items[0].setIcon(icon)
                        except Exception as icon_err:
                            self.update_status(f"Warning: Could not load icon: {str(icon_err)}")
                except Exception as e:
                    self.update_status(f"Warning: Could not update dataset icon: {str(e)}")
                
                # If this is the current source, update the metadata status
                if self.current_source == name:
                    self.metadata_status_label.setText("✓ Metadata Available")
                    self.metadata_status_label.setStyleSheet("color: green; font-weight: bold;")
                    self.view_metadata_btn.setEnabled(True)
                
                self.update_status(f"Generated metadata for {name}")
                return metadata
                
            except json.JSONDecodeError:
                # If simple cleaning failed, try more aggressive sanitization
                self.update_status(f"Initial JSON parsing failed for {name}, attempting more aggressive fixes...")
                
                try:
                    # More advanced sanitization
                    import re
                    
                    # Try to extract just the JSON part
                    text = metadata_json.strip()
                    # Get everything from the first { to the last }
                    json_start = text.find('{')
                    json_end = text.rfind('}')
                    
                    if json_start >= 0 and json_end > json_start:
                        text = text[json_start:json_end+1]
                    
                    # Fix common issues:
                    # 1. Fix unquoted property names
                    text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
                    
                    # 2. Fix single quotes to double quotes
                    text = text.replace("'", '"')
                    
                    # 3. Fix trailing commas
                    text = re.sub(r',\s*}', '}', text)
                    text = re.sub(r',\s*\]', ']', text)
                    
                    # Try parsing the fixed JSON
                    metadata = json.loads(text)
                    
                    # If we get here, the sanitization worked
                    self.update_status(f"Successfully recovered metadata for {name} with advanced fixes")
                    
                    # Store and update as before
                    if name in self.source_connections:
                        self.source_connections[name].metadata = metadata
                    
                    if not hasattr(self, 'datasets_metadata'):
                        self.datasets_metadata = {}
                    
                    self.datasets_metadata[name] = metadata
                    
                    # Update studies manager
                    main_window = self.window()
                    if hasattr(main_window, 'studies_manager'):
                        main_window.studies_manager.update_dataset_metadata(name, metadata)
                    
                    # Update UI 
                    if self.current_source == name:
                        self.metadata_status_label.setText("✓ Metadata Available (Recovered)")
                        self.metadata_status_label.setStyleSheet("color: green; font-weight: bold;")
                        self.view_metadata_btn.setEnabled(True)
                    
                    try:
                        items = self.sources_list.findItems(name, Qt.MatchFlag.MatchExactly)
                        if items and len(items) > 0:
                            try:
                                icon = load_bootstrap_icon("check-square")
                                items[0].setSizeHint(QSize(items[0].sizeHint().width(), 28))
                                items[0].setIcon(icon)
                            except Exception as icon_err:
                                # Just log the error and continue without setting the icon
                                self.update_status(f"Warning: Could not load icon: {str(icon_err)}")
                    except Exception as e:
                        self.update_status(f"Warning: Could not update dataset icon: {str(e)}")
                    
                    return metadata
                    
                except Exception as e:
                    self.update_status(f"Error: Could not parse metadata JSON even with advanced fixes: {str(e)}")
                    QMessageBox.warning(
                        self, 
                        "Error", 
                        f"Failed to parse metadata for {name}. The response was not valid JSON."
                    )
                    return None
                
        except Exception as e:
            self.update_status(f"Error generating metadata: {str(e)}")
            QMessageBox.warning(
                self, 
                "Error", 
                f"Failed to generate metadata for {name}: {str(e)}"
            )
            return None

    def show_metadata_dialog(self, name):
        """Show a dialog displaying dataset metadata"""
        if name not in self.dataframes:
            QMessageBox.warning(self, "Error", f"Dataset '{name}' not found")
            return
        
        # Check if we have metadata stored locally
        metadata = None
        if hasattr(self, 'datasets_metadata') and name in self.datasets_metadata:
            metadata = self.datasets_metadata[name]
        elif name in self.source_connections and hasattr(self.source_connections[name], 'metadata'):
            metadata = self.source_connections[name].metadata
        
        # If metadata exists, show it
        if metadata:
            # Create dialog to display metadata
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Dataset Metadata: {name}")
            dialog.setMinimumSize(700, 500)
            
            # Apply stylesheet to ensure icons are visible
            dialog.setStyleSheet("""
                QLabel[class="header"] { font-weight: bold; font-size: 16px; }
                QGroupBox { font-weight: bold; }
                QTableWidget { gridline-color: #ddd; }
            """)
            
            layout = QVBoxLayout(dialog)
            
            # Header with icon - force rendering the icon
            header_layout = QHBoxLayout()
            
            # Create icon with a specific size to ensure visibility
            icon = load_bootstrap_icon("info-circle-fill", color="#0d6efd")
            icon_pixmap = icon.pixmap(32, 32)  # Use a larger size
            if not icon_pixmap.isNull():
                # Icon loaded successfully
                icon_label = QLabel()
                icon_label.setPixmap(icon_pixmap)
                header_layout.addWidget(icon_label)
            
            # Header title
            header_label = QLabel(f"Metadata for: {name}")
            header_label.setProperty("class", "header")
            header_layout.addWidget(header_label)
            header_layout.addStretch()
            
            layout.addLayout(header_layout)
            
            # Add description & purpose
            if "description" in metadata or "purpose" in metadata:
                desc_group = QGroupBox("Overview")
                desc_layout = QVBoxLayout(desc_group)
                
                # Description
                if "description" in metadata:
                    desc_text = QLabel(metadata["description"])
                    desc_text.setWordWrap(True)
                    desc_layout.addWidget(desc_text)
                
                # Purpose
                if "purpose" in metadata:
                    purpose_label = QLabel("<b>Purpose:</b>")
                    desc_layout.addWidget(purpose_label)
                    purpose_text = QLabel(metadata["purpose"])
                    purpose_text.setWordWrap(True)
                    desc_layout.addWidget(purpose_text)
                
                layout.addWidget(desc_group)
            
            # Add column information
            columns_group = QGroupBox("Column Information")
            columns_layout = QVBoxLayout(columns_group)
            
            columns_table = QTableWidget()
            columns_table.setColumnCount(3)
            columns_table.setHorizontalHeaderLabels(["Column", "Data Type", "Description"])
            columns_table.horizontalHeader().setStretchLastSection(True)
            
            # Check which format the metadata uses for column info
            if "column_definitions" in metadata:
                column_defs = metadata.get("column_definitions", {})
                columns_table.setRowCount(len(column_defs))
                
                for i, (col_name, definition) in enumerate(column_defs.items()):
                    columns_table.setItem(i, 0, QTableWidgetItem(col_name))
                    columns_table.setItem(i, 1, QTableWidgetItem(""))  # No type in old format
                    columns_table.setItem(i, 2, QTableWidgetItem(definition))
            elif "column_descriptions" in metadata:
                column_defs = metadata.get("column_descriptions", {})
                columns_table.setRowCount(len(column_defs))
                
                for i, (col_name, col_info) in enumerate(column_defs.items()):
                    columns_table.setItem(i, 0, QTableWidgetItem(col_name))
                    
                    # Get data type if available
                    data_type = ""
                    if isinstance(col_info, dict) and "data_type" in col_info:
                        data_type = col_info["data_type"]
                    columns_table.setItem(i, 1, QTableWidgetItem(data_type))
                    
                    # Get description
                    description = ""
                    if isinstance(col_info, dict) and "description" in col_info:
                        description = col_info["description"]
                    elif isinstance(col_info, str):
                        description = col_info
                    columns_table.setItem(i, 2, QTableWidgetItem(description))
            
            columns_layout.addWidget(columns_table)
            layout.addWidget(columns_group, 1)  # Give column table more space
            
            # Add sections for relationships, quality issues, etc. if available
            if "relationships" in metadata and metadata["relationships"]:
                rel_group = QGroupBox("Relationships")
                rel_layout = QVBoxLayout(rel_group)
                rel_text = QPlainTextEdit()
                
                # Format relationships
                rel_content = ""
                for rel in metadata["relationships"]:
                    if isinstance(rel, dict):
                        rel_type = rel.get("type", "")
                        rel_desc = rel.get("description", "")
                        rel_content += f"• {rel_type}: {rel_desc}\n"
                    elif isinstance(rel, str):
                        rel_content += f"• {rel}\n"
                
                rel_text.setPlainText(rel_content)
                rel_text.setReadOnly(True)
                rel_text.setMaximumHeight(100)
                rel_layout.addWidget(rel_text)
                layout.addWidget(rel_group)
            
            # Add metadata generation timestamp if available
            if "generated_at" in metadata:
                gen_time = QLabel(f"Generated: {metadata['generated_at']}")
                layout.addWidget(gen_time)
            
            # Close button
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            
            close_button = QPushButton("Close")
            close_button.setIcon(load_bootstrap_icon("x"))
            close_button.clicked.connect(dialog.accept)
            close_button.setMinimumWidth(100)
            button_layout.addWidget(close_button)
            
            layout.addLayout(button_layout)
            
            dialog.exec()
        else:
            QMessageBox.information(self, "No Metadata", f"No metadata available for {name}. Generate metadata first.")

    def format_masked_value_distribution(self, dist, prefix=""):
        """Helper method to format masked value distribution in a consistent way
        
        Args:
            dist: Value distribution dictionary or list
            prefix: Prefix for the format lines (e.g. "- " or "  * ")
            
        Returns:
            str: Formatted string with the masked values
        """
        result = ""
        if isinstance(dist, list):
            result += f"{prefix}Example masked values: {', '.join(dist)}\n"
        else:
            # Check for 'sample_values' key which is what the masking_utils actually provides
            if "sample_values" in dist:
                result += f"{prefix}Sample masked values: {', '.join(dist['sample_values'])}\n"
            # Fallback for backward compatibility
            elif "first_values" in dist:
                result += f"{prefix}First examples (masked): {', '.join(dist['first_values'])}\n"
                result += f"{prefix}Last examples (masked): {', '.join(dist['last_values'])}\n"
            # Handle other possible dictionary structures
            else:
                result += f"{prefix}Value distribution stats: {', '.join([f'{k}: {v}' for k, v in dist.items()])}\n"
        return result

    def show_settings_dialog(self):
        """Show dialog for configuring data collection settings"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Data Collection Settings")
        layout = QVBoxLayout(dialog)
        
        # Create confirmation settings group
        confirm_group = QGroupBox("Confirmation Settings")
        confirm_layout = QVBoxLayout(confirm_group)
        
        # Create checkboxes for confirmation settings
        batch_metadata_checkbox = QCheckBox("Confirm before generating metadata for multiple datasets")
        batch_metadata_checkbox.setChecked(self.settings.confirm_metadata_batch)
        batch_metadata_checkbox.setToolTip("If unchecked, metadata generation will start immediately for multiple datasets")
        confirm_layout.addWidget(batch_metadata_checkbox)
        
        relationship_checkbox = QCheckBox("Confirm before analyzing relationships")
        relationship_checkbox.setChecked(self.settings.confirm_relationship_analysis)
        relationship_checkbox.setToolTip("If unchecked, relationship analysis will start immediately")
        confirm_layout.addWidget(relationship_checkbox)
        
        # Add the group to the layout
        layout.addWidget(confirm_group)
        
        # Add buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # Show dialog and handle result
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.settings.confirm_metadata_batch = batch_metadata_checkbox.isChecked()
            self.settings.confirm_relationship_analysis = relationship_checkbox.isChecked()

    def filter_datasets(self, text):
        """Filter the datasets list based on search text"""
        text = text.lower().strip()
        for i in range(self.sources_list.count()):
            item = self.sources_list.item(i)
            if text:
                # Show items that match the search text
                item.setHidden(text not in item.text().lower())
            else:
                # Show all items when search is empty
                item.setHidden(False)

    def update_selection_label(self):
        """Update the selection count label"""
        selected_items = self.sources_list.selectedItems()
        count = len(selected_items)
        
        if count == 0:
            self.selection_label.setText("0")
        else:
            # Only show the count number, nothing else
            self.selection_label.setText(f"{count}")

    def batch_generate_metadata(self):
        """Generate metadata for selected datasets or ask to select one"""
        selected_items = self.sources_list.selectedItems()
        
        # If no items selected, ask user to select datasets
        if not selected_items:
            QMessageBox.information(
                self,
                "No Datasets Selected",
                "Please select one or more datasets to generate metadata."
            )
            return
        
        # Get selected dataset names
        dataset_names = [item.text() for item in selected_items]
        count = len(dataset_names)
        
        # If only one dataset is selected, generate metadata immediately
        if count == 1:
            name = dataset_names[0]
            # Direct call since the method is decorated with @asyncSlot
            asyncio.create_task(self.generate_dataset_metadata(name))
            self.update_status(f"Generating metadata for {name}...")
            return
        
        # Multiple datasets selected - check if confirmation is required
        if self.settings.confirm_metadata_batch:
            # Confirmation dialog with selected datasets
            message = f"Generate metadata for {count} selected datasets?"
            if count <= 5:
                message += f"\n- {', '.join(dataset_names)}"
            else:
                message += f"\n- {', '.join(dataset_names[:3])} and {count-3} more"
                
            confirm = QMessageBox.question(
                self,
                "Generate Metadata",
                message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if confirm != QMessageBox.StandardButton.Yes:
                return
        
        # Generate metadata for each selected dataset
        for name in dataset_names:
            # Create task since the method is decorated with @asyncSlot
            asyncio.create_task(self.generate_dataset_metadata(name))
        
        self.update_status(f"Generating metadata for {count} datasets...")

    def show_dataset_relationships(self):
        """Show dataset relationships, using existing analysis if available or generating a new one"""
        if not self.dataframes:
            QMessageBox.warning(self, "No Datasets", "No datasets available to analyze")
            return
            
        # Get selected datasets
        selected_items = self.sources_list.selectedItems()
        selected_datasets = [item.text() for item in selected_items] if selected_items else []
        
        # Check if we have a recent grouping result and we're not requesting a specific selection
        if self.latest_grouping_result and not selected_datasets:
            # Show the previous result with an option to refresh
            self.show_dataset_grouping_results(self.latest_grouping_result, show_reuse_info=True)
            # Apply highlighting to the sources list
            self.highlight_grouped_datasets(self.latest_grouping_result)
        else:
            # If datasets are selected, use only those
            if selected_datasets:
                # Check if confirmation is required
                if self.settings.confirm_relationship_analysis:
                    dataset_count = len(selected_datasets)
                    
                    message = f"Analyze relationships between {dataset_count} selected datasets?\n\nThis will use AI to identify connections between your selected datasets."
                    if dataset_count <= 5:
                        message += f"\n\nSelected datasets: {', '.join(selected_datasets)}"
                    else:
                        message += f"\n\nSelected datasets: {', '.join(selected_datasets[:3])} and {dataset_count-3} more"
                    
                    confirm = QMessageBox.question(
                        self,
                        "Analyze Dataset Relationships",
                        message,
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    
                    if confirm != QMessageBox.StandardButton.Yes:
                        return
                
                # Start a new analysis with selected datasets
                asyncio.create_task(self.identify_and_group_datasets(force_new=True, selected_datasets=selected_datasets))
            else:
                # No selection, analyze all datasets
                # Check if confirmation is required
                if self.settings.confirm_relationship_analysis:
                    # Count available datasets
                    dataset_count = len(self.dataframes)
                    
                    confirm = QMessageBox.question(
                        self,
                        "Analyze Dataset Relationships",
                        f"Analyze relationships between all {dataset_count} datasets?\n\nThis will use AI to identify connections between your datasets.",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    
                    if confirm != QMessageBox.StandardButton.Yes:
                        return
                
                # Start a new analysis
                asyncio.create_task(self.identify_and_group_datasets(force_new=True))

    @asyncSlot()
    async def identify_and_group_datasets(self, force_new=False, selected_datasets=None):
        """Use LLM to identify and group related datasets"""
        if not self.dataframes:
            QMessageBox.warning(self, "No Datasets", "No datasets available to analyze")
            return

        # Show waiting dialog
        waiting_msg = QMessageBox(self)
        waiting_msg.setWindowTitle("Analyzing Datasets")
        waiting_msg.setText("Analyzing datasets to identify relationships...")
        waiting_msg.setStandardButtons(QMessageBox.StandardButton.NoButton)
        waiting_msg.show()

        try:
            # Build a comprehensive description of all datasets or selected datasets
            datasets_info = {}
            dataframes_to_analyze = {}
            
            # If specific datasets are selected, only analyze those
            if selected_datasets:
                for name in selected_datasets:
                    if name in self.dataframes:
                        dataframes_to_analyze[name] = self.dataframes[name]
            else:
                dataframes_to_analyze = self.dataframes
            
            if not dataframes_to_analyze:
                waiting_msg.accept()
                QMessageBox.warning(self, "No Datasets", "No valid datasets selected for analysis")
                return
                
            for name, df in dataframes_to_analyze.items():
                # Generate masked column mappings
                try:
                    masked_mapping = get_column_mapping(df)
                    datasets_info[name] = {
                        "row_count": len(df),
                        "column_count": len(df.columns),
                        "column_mapping": masked_mapping
                    }
                except Exception as e:
                    self.update_status(f"Warning: Could not analyze columns for {name}: {str(e)}")
                    # Add basic info even if column mapping fails
                    datasets_info[name] = {
                        "row_count": len(df),
                        "column_count": len(df.columns),
                        "columns": list(df.columns)
                    }

            # Create LLM prompt
            prompt = """
            I have multiple datasets that may or may not be related. Please analyze them to:
            1. Identify datasets that likely belong together (e.g., normalized database tables)
            2. Suggest logical groupings based on column relationships
            3. Identify potential join keys between datasets
            
            Here are the datasets with masked column information to protect sensitive data:
            """

            # Add dataset information to prompt
            for name, info in datasets_info.items():
                prompt += f"\n\nDATASET: {name}\n"
                prompt += f"Rows: {info['row_count']}, Columns: {info['column_count']}\n"
                prompt += "Columns:\n"
                
                if "column_mapping" in info:
                    for column, col_info in info["column_mapping"].get("column_mappings", {}).items():
                        prompt += f"- {column} ({col_info.get('type', 'unknown')})\n"
                        prompt += f"  * Unique values: {col_info.get('unique_count', 'unknown')}\n"
                        
                        null_count = col_info.get('null_count', 0)
                        null_pct = col_info.get('null_percentage', 0)
                        prompt += f"  * Null values: {null_count} ({null_pct:.1f}%)\n"
                        
                        # Add masked value examples if available
                        if "value_distribution" in col_info:
                            dist = col_info["value_distribution"]
                            if isinstance(dist, dict) and "sample_values" in dist:
                                prompt += f"  * Sample values: {', '.join(str(x) for x in dist['sample_values'][:3])}\n"
                            elif isinstance(dist, list):
                                prompt += f"  * Sample values: {', '.join(str(x) for x in dist[:3])}\n"
                else:
                    # Just list column names if no detailed mapping
                    for column in info.get("columns", []):
                        prompt += f"- {column}\n"

            # Additional instructions for grouping and relationship identification
            prompt += """
            
            Analyze the datasets and suggest:
            
            1. GROUPINGS: Which datasets appear to belong together and why?
               - Use naming patterns, column structures, and potential join keys
               - Consider normalized data patterns (e.g., "patients" table, "visits" table)
               
            2. CRITERIA: What criteria did you use to determine relationships?
               - Column name patterns matching across tables
               - ID columns with matching masked patterns
               - Cardinality of values suggesting relationships
               - Semantic relationships based on column names
               
            3. CONFIDENCE: For each suggested grouping, indicate your confidence level (high, medium, low)
            
            Return your response in JSON format:
            {
                "groups": [
                    {
                        "name": "Group name (e.g., 'Patient Clinical Data')",
                        "datasets": ["dataset1", "dataset2"],
                        "confidence": "high/medium/low",
                        "reasoning": "Explanation of why these datasets are grouped"
                    }
                ],
                "ungrouped_datasets": ["dataset3"],
                "criteria_used": ["List of criteria used to determine relationships"]
            }
            """

            # Call LLM
            response = await call_llm_async(prompt)
            
            # Try to parse the response as JSON
            try:
                # Strip any markdown code block indicators and whitespace
                clean_response = response.strip()
                if clean_response.startswith("```json"):
                    clean_response = clean_response[7:]
                elif clean_response.startswith("```"):
                    clean_response = clean_response[3:]
                if clean_response.endswith("```"):
                    clean_response = clean_response[:-3]
                clean_response = clean_response.strip()
                
                # Parse the cleaned JSON
                grouping_result = json.loads(clean_response)
                
                # Store the result for future use
                self.latest_grouping_result = grouping_result
                self.latest_grouping_timestamp = datetime.now()
                
                # Close the waiting dialog
                waiting_msg.accept()
                
                # Display the results in a formatted dialog
                self.show_dataset_grouping_results(grouping_result)
                
                # Apply highlighting to the sources list
                self.highlight_grouped_datasets(grouping_result)
                
            except json.JSONDecodeError:
                waiting_msg.accept()
                
                # Try more aggressive JSON parsing
                try:
                    import re
                    # Extract everything between first { and last }
                    json_match = re.search(r'({[\s\S]*})', clean_response)
                    if json_match:
                        json_text = json_match.group(1)
                        # Fix unquoted property names
                        json_text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_text)
                        # Fix single quotes
                        json_text = json_text.replace("'", '"')
                        # Fix trailing commas
                        json_text = re.sub(r',\s*}', '}', json_text)
                        json_text = re.sub(r',\s*\]', ']', json_text)
                        
                        grouping_result = json.loads(json_text)
                        
                        # Store the result
                        self.latest_grouping_result = grouping_result
                        self.latest_grouping_timestamp = datetime.now()
                        
                        # Display the results in a formatted dialog
                        self.show_dataset_grouping_results(grouping_result)
                        
                        # Apply highlighting to the sources list 
                        self.highlight_grouped_datasets(grouping_result)
                        return
                except:
                    pass
                
                # If all JSON parsing fails, show the raw response
                self.show_raw_grouping_response(response)
                
        except Exception as e:
            waiting_msg.accept()
            QMessageBox.critical(self, "Error", f"Failed to analyze datasets: {str(e)}")

    def show_dataset_grouping_results(self, grouping_result, show_reuse_info=False):
        """Display the dataset grouping results in a formatted dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Dataset Relationship Analysis")
        dialog.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Add header with refresh button
        header_layout = QHBoxLayout()
        header_label = QLabel("Dataset Relationship Analysis")
        header_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        
        # Add timestamp - show whether this is a reused or new analysis
        timestamp = self.latest_grouping_timestamp.strftime("%Y-%m-%d %H:%M") if self.latest_grouping_timestamp else datetime.now().strftime("%Y-%m-%d %H:%M")
        
        if show_reuse_info:
            timestamp_label = QLabel(f"Previously generated: {timestamp}")
            timestamp_label.setStyleSheet("color: #666;")
        else:
            timestamp_label = QLabel(f"Generated: {timestamp}")
        
        header_layout.addWidget(timestamp_label)
        
        # Add refresh button - avoid using icon to prevent failures
        refresh_btn = QPushButton("New Analysis")
        refresh_btn.clicked.connect(lambda: self.regenerate_analysis(dialog))
        header_layout.addWidget(refresh_btn)
        
        layout.addLayout(header_layout)
        
        # Add tabbed view for groups
        tabs = QTabWidget()
        self.populate_grouping_tabs(tabs, grouping_result)
        layout.addWidget(tabs, 1)  # Give tabs most of the space
        
        # Add buttons at the bottom
        button_layout = QHBoxLayout()
        
        # Add button to export the analysis - avoid using icon
        export_btn = QPushButton("Export Analysis")
        export_btn.clicked.connect(lambda: self.export_grouping_analysis(grouping_result))
        button_layout.addWidget(export_btn)
        
        button_layout.addStretch()
        
        # Close button - avoid using icon
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        # Show the dialog
        dialog.exec()

    def highlight_grouped_datasets(self, grouping_result):
        """Apply visual highlighting to datasets in the sources list based on their group assignment"""
        # Clear any existing highlighting
        for i in range(self.sources_list.count()):
            item = self.sources_list.item(i)
            item.setBackground(Qt.GlobalColor.transparent)
            item.setToolTip("")
        
        # Highlight grouped datasets
        grouped_datasets = {}  # Keep track of which datasets are assigned to groups
        
        for group_idx, group in enumerate(grouping_result.get("groups", [])):
            # Select a color for this group (cycle through available colors)
            color_idx = group_idx % len(self.group_colors)
            color = self.group_colors[color_idx]
            
            # Get the group name
            group_name = group.get("name", f"Group {group_idx+1}")
            
            # Highlight each dataset in this group
            for dataset_name in group.get("datasets", []):
                # Find the item in the source list
                items = self.sources_list.findItems(dataset_name, Qt.MatchFlag.MatchExactly)
                if items:
                    # Set background color and tooltip with explicit QColor
                    items[0].setBackground(QColor(color))
                    # Make tooltip more visible too
                    items[0].setToolTip(f"Part of group: {group_name}")
                    
                    # Remember this dataset is grouped
                    grouped_datasets[dataset_name] = True
        
        # Mark ungrouped datasets
        for dataset_name in grouping_result.get("ungrouped_datasets", []):
            items = self.sources_list.findItems(dataset_name, Qt.MatchFlag.MatchExactly)
            if items:
                # Use a light gray for ungrouped datasets
                items[0].setBackground(QColor("#E0E0E0"))
                items[0].setToolTip("Ungrouped dataset")
                
                # Remember this dataset is accounted for
                grouped_datasets[dataset_name] = True
        
        # Any datasets not mentioned in the grouping result get no special marking
        for i in range(self.sources_list.count()):
            item = self.sources_list.item(i)
            if item.text() not in grouped_datasets:
                item.setToolTip("Not analyzed in grouping")

    def regenerate_analysis(self, dialog):
        """Close the current dialog and generate a new analysis"""
        dialog.accept()  # Close the current dialog
        
        # Clear highlighting before generating new analysis
        for i in range(self.sources_list.count()):
            item = self.sources_list.item(i)
            item.setBackground(Qt.GlobalColor.transparent)
            item.setToolTip("")
            
        asyncio.create_task(self.identify_and_group_datasets(force_new=True))  # Start new analysis

    def populate_grouping_tabs(self, tabs, grouping_result):
        """Populate tabs with grouping result data"""
        # Clear existing tabs
        while tabs.count() > 0:
            tabs.removeTab(0)
            
        # Add a tab for each group
        for i, group in enumerate(grouping_result.get("groups", [])):
            group_widget = QWidget()
            group_layout = QVBoxLayout(group_widget)
            
            # Group header
            group_name = group.get("name", f"Group {i+1}")
            header = QLabel(f"<b>{group_name}</b>")
            header.setStyleSheet("font-size: 14px;")
            group_layout.addWidget(header)
            
            # Confidence level
            confidence = group.get("confidence", "unknown").upper()
            conf_color = "#28a745" if confidence == "HIGH" else "#ffc107" if confidence == "MEDIUM" else "#dc3545"
            confidence_label = QLabel(f"Confidence: <span style='color: {conf_color};'>{confidence}</span>")
            group_layout.addWidget(confidence_label)
            
            # Reasoning
            if "reasoning" in group:
                reasoning_group = QGroupBox("Reasoning")
                reasoning_layout = QVBoxLayout(reasoning_group)
                reasoning_text = QPlainTextEdit(group["reasoning"])
                reasoning_text.setReadOnly(True)
                reasoning_layout.addWidget(reasoning_text)
                group_layout.addWidget(reasoning_group)
            
            # Datasets in this group
            if "datasets" in group:
                datasets_group = QGroupBox("Datasets")
                datasets_layout = QVBoxLayout(datasets_group)
                
                datasets_list = QListWidget()
                for dataset in group["datasets"]:
                    item = QListWidgetItem(dataset)
                    # Try to add icon for datasets that exist - with better error handling
                    if dataset in self.dataframes:
                        # Create data item without icon first, to avoid failures if icon is missing
                        datasets_list.addItem(item)
                
                datasets_layout.addWidget(datasets_list)
                group_layout.addWidget(datasets_group)
            
            # Add tab for this group
            tabs.addTab(group_widget, group_name)
        
        # Add a tab for ungrouped datasets
        if grouping_result.get("ungrouped_datasets"):
            ungrouped_widget = QWidget()
            ungrouped_layout = QVBoxLayout(ungrouped_widget)
            
            ungrouped_label = QLabel("The following datasets could not be confidently grouped:")
            ungrouped_layout.addWidget(ungrouped_label)
            
            ungrouped_list = QListWidget()
            for dataset in grouping_result.get("ungrouped_datasets", []):
                item = QListWidgetItem(dataset)
                # Add items directly without icons to avoid failures
                ungrouped_list.addItem(item)
            ungrouped_layout.addWidget(ungrouped_list)
            
            tabs.addTab(ungrouped_widget, "Ungrouped Datasets")
        
        # Add a tab for criteria used
        criteria_widget = QWidget()
        criteria_layout = QVBoxLayout(criteria_widget)
        
        criteria_label = QLabel("The following criteria were used to determine relationships:")
        criteria_layout.addWidget(criteria_label)
        
        criteria_list = QListWidget()
        for criterion in grouping_result.get("criteria_used", []):
            criteria_list.addItem(criterion)
        criteria_layout.addWidget(criteria_list)
        
        tabs.addTab(criteria_widget, "Criteria Used")

    def show_raw_grouping_response(self, response):
        """Show the raw LLM response when parsing fails"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Dataset Analysis - Raw Response")
        dialog.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Add warning about parsing error
        warning = QLabel("Could not parse the LLM response as valid JSON. Here is the raw response:")
        warning.setStyleSheet("color: #dc3545; font-weight: bold;")
        layout.addWidget(warning)
        
        # Show raw response in a text area
        response_text = QPlainTextEdit()
        response_text.setPlainText(response)
        response_text.setReadOnly(True)
        layout.addWidget(response_text)
        
        # Add copy button
        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(response))
        
        # Add close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(copy_btn)
        button_layout.addStretch()
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec()

    def export_grouping_analysis(self, grouping_result):
        """Export the grouping analysis to a file"""
        # Ask for file location
        options = QFileDialog.Option.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(
            self, 
            "Export Dataset Relationship Analysis", 
            "dataset_relationships.json", 
            "JSON Files (*.json);;Text Files (*.txt)",
            options=options
        )
        
        if not file_name:
            return
            
        try:
            # Add timestamp to the export
            export_data = {
                "analysis_timestamp": datetime.now().isoformat(),
                "results": grouping_result
            }
            
            with open(file_name, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            QMessageBox.information(
                self,
                "Export Successful",
                f"Dataset relationship analysis exported to {file_name}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Could not export analysis: {str(e)}"
            )

    def apply_icon_button_style(self, button):
        """Apply the icon button style to a button"""
        button.setStyleSheet(self.icon_button_style)
        return button

    # Methods to sync with StudiesManager
    def add_to_studies_manager(self, name, dataframe):
        """Add dataset to the studies manager"""
        main_window = self.window()
        if hasattr(main_window, 'studies_manager'):
            try:
                # Get metadata if available
                metadata = None
                if name in self.source_connections:
                    metadata = getattr(self.source_connections[name], 'metadata', None)
                
                # Add to studies manager
                main_window.studies_manager.add_dataset_to_active_study(
                    name, dataframe, metadata
                )
                self.update_status(f"Added dataset '{name}' to active study")
            except Exception as e:
                self.update_status(f"Failed to add dataset to studies manager: {str(e)}")
    
    def delete_from_studies_manager(self, name):
        """Delete dataset from the studies manager"""
        main_window = self.window()
        if hasattr(main_window, 'studies_manager'):
            try:
                main_window.studies_manager.remove_dataset_from_active_study(name)
                self.update_status(f"Removed dataset '{name}' from active study")
            except Exception as e:
                self.update_status(f"Failed to remove dataset from studies manager: {str(e)}")
    
    def rename_in_studies_manager(self, old_name, new_name):
        """Rename dataset in the studies manager"""
        main_window = self.window()
        if hasattr(main_window, 'studies_manager'):
            try:
                # Need to add the new dataset and then remove the old one
                if old_name in self.dataframes:
                    # Get metadata if available
                    metadata = None
                    if new_name in self.source_connections:
                        metadata = getattr(self.source_connections[new_name], 'metadata', None)
                    
                    # Add with new name
                    main_window.studies_manager.add_dataset_to_active_study(
                        new_name, self.dataframes[new_name], metadata
                    )
                    
                    # Remove old name
                    main_window.studies_manager.remove_dataset_from_active_study(old_name)
                    
                    self.update_status(f"Renamed dataset in active study: {old_name} -> {new_name}")
            except Exception as e:
                self.update_status(f"Failed to rename dataset in studies manager: {str(e)}")
    
    def update_in_studies_manager(self, name, dataframe):
        """Update dataset in the studies manager"""
        main_window = self.window()
        if hasattr(main_window, 'studies_manager'):
            try:
                # Get metadata if available
                metadata = None
                if name in self.source_connections:
                    metadata = getattr(self.source_connections[name], 'metadata', None)
                
                # Update in studies manager
                main_window.studies_manager.update_dataset_in_active_study(
                    name, dataframe, metadata
                )
                self.update_status(f"Updated dataset '{name}' in active study")
            except Exception as e:
                self.update_status(f"Failed to update dataset in studies manager: {str(e)}")

    def add_debug_datasets(self):
        """Add debug datasets for testing and demonstration."""
        try:
            # Import the debug datasets module
            from data.collection.debug_datasets import final_datasets
            
            # Add each dataset from the final_datasets dictionary
            for name, df in final_datasets.items():
                # Create a simple file connection for these datasets
                file_connection = SourceConnection("file", {"path": f"debug/{name}.csv"})
                self.add_source(name, file_connection, df)
            
            self.update_status(f"Added {len(final_datasets)} debug datasets")
        except Exception as e:
            self.update_status(f"Error adding debug datasets: {str(e)}")

    def add_healthcare_debug_datasets(self):
        """Add healthcare debug datasets for testing and demonstration."""
        try:
            # Import the healthcare debug datasets module
            from data.collection.debug_healthcare_datasets import healthcare_datasets
            
            # Add each dataset from the healthcare_datasets dictionary
            for name, df in healthcare_datasets.items():
                # Create a proper name with 'healthcare_' prefix
                dataset_name = f"healthcare_{name}"
                
                # Create a simple file connection for these datasets
                file_connection = SourceConnection("file", {"path": f"debug/{dataset_name}.csv"})
                
                # Add as a new source
                self.add_source(dataset_name, file_connection, df)
            
            self.update_status(f"Added {len(healthcare_datasets)} healthcare debug datasets")
        except Exception as e:
            self.update_status(f"Error adding healthcare debug datasets: {str(e)}")

    @asyncSlot()
    async def add_grouping_variable(self):
        """Add a grouping variable to a dataset based on a hypothesis using LLM."""
        if not self.dataframes:
            QMessageBox.warning(self, "No Datasets", "No datasets available to modify")
            return
            
        # Create dialog to select dataset and enter hypothesis
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Grouping Variable for Hypothesis Testing")
        dialog.setMinimumSize(600, 400)
        
        layout = QVBoxLayout(dialog)
        
        # Dataset selection
        dataset_group = QGroupBox("Select Dataset")
        dataset_layout = QVBoxLayout(dataset_group)
        
        dataset_combo = QComboBox()
        dataset_combo.addItems(sorted(self.dataframes.keys()))
        if self.current_source and self.current_source in self.dataframes:
            index = dataset_combo.findText(self.current_source)
            if index >= 0:
                dataset_combo.setCurrentIndex(index)
        dataset_layout.addWidget(dataset_combo)
        
        layout.addWidget(dataset_group)
        
        # Hypothesis input
        hypothesis_group = QGroupBox("Enter Research Hypothesis")
        hypothesis_layout = QVBoxLayout(hypothesis_group)
        
        hypothesis_text = QPlainTextEdit()
        hypothesis_text.setPlaceholderText(
            "Example: Patients discharged against medical advice (AMA) have significantly higher "
            "30-day readmission rates compared to those with planned discharges, independent of "
            "comorbidity burden."
        )
        hypothesis_layout.addWidget(hypothesis_text)
        
        layout.addWidget(hypothesis_group)
        
        # Group variable name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Group Variable Name:"))
        variable_name_input = QLineEdit()
        variable_name_input.setPlaceholderText("e.g., treatment_group, risk_category")
        name_layout.addWidget(variable_name_input)
        layout.addLayout(name_layout)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # Show dialog and handle result
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
            
        # Get inputs
        dataset_name = dataset_combo.currentText()
        hypothesis = hypothesis_text.toPlainText().strip()
        variable_name = variable_name_input.text().strip()
        
        if not hypothesis:
            QMessageBox.warning(self, "Error", "Please enter a research hypothesis")
            return
            
        if not variable_name:
            QMessageBox.warning(self, "Error", "Please enter a name for the group variable")
            return
            
        # Get the dataset
        if dataset_name not in self.dataframes:
            QMessageBox.warning(self, "Error", f"Dataset '{dataset_name}' not found")
            return
            
        df = self.dataframes[dataset_name]
        
        # Show waiting dialog
        waiting_msg = QMessageBox(self)
        waiting_msg.setWindowTitle("Generating Group Variable")
        waiting_msg.setText("Generating code to create group variable based on hypothesis...\n\nThis may take a few moments. Please wait.")
        
        # Add more details
        details = f"Dataset: {dataset_name}\nNew variable: {variable_name}\n\nAnalyzing data and generating appropriate code..."
        waiting_msg.setInformativeText(details)
        waiting_msg.setStandardButtons(QMessageBox.StandardButton.NoButton)
        waiting_msg.show()
        
        try:
            # Prepare column info
            columns_info = "\n".join([f"- {col}: {df[col].dtype}" for col in df.columns])
            
            # Add some sample data
            sample_data = df.head(5).to_string()
            
            # Detect patient ID columns and analyze if there are repeats
            patient_id_cols = self.detect_patient_id_columns(df)
            patient_level_info = ""
            
            if patient_id_cols:
                repeat_analysis = self.analyze_patient_repeats(df, patient_id_cols)
                
                patient_level_info = "\nPATIENT-LEVEL ANALYSIS:\n"
                for col, analysis in repeat_analysis.items():
                    patient_level_info += f"- Column '{col}' appears to be a patient identifier\n"
                    patient_level_info += f"  * Unique values: {analysis['unique_values']}\n"
                    
                    if analysis['patients_with_repeats'] > 0:
                        patient_level_info += f"  * {analysis['patients_with_repeats']} patients ({analysis['repeat_percentage']:.1f}%) have multiple entries\n"
                        patient_level_info += f"  * Maximum entries per patient: {analysis['max_repeats']}\n"
                        
                        # Suggest appropriate aggregation strategies based on the data
                        patient_level_info += "  * Consider patient-level normalization/aggregation\n"
                    else:
                        patient_level_info += "  * No patients have multiple entries - no aggregation needed\n"
            else:
                patient_level_info = "\nNo clear patient identifier columns detected. If this is incorrect, consider:\n"
                patient_level_info += "- Manually identifying the patient ID column\n"
                patient_level_info += "- Using a combination of fields to uniquely identify patients\n"
             
            # Add distribution information for categorical columns
            distribution_info = ""
            for col in df.columns:
                if df[col].dtype == 'object' or df[col].dtype.name == 'category' or len(df[col].unique()) < 10:
                    try:
                        dist = df[col].value_counts().head(10).to_string()
                        distribution_info += f"\nDistribution of {col}:\n{dist}\n"
                    except:
                        # Skip if we can't get the distribution
                        pass
            
            # Prepare prompt for LLM
            prompt = f"""
            I need Python code to add a grouping variable to an EXISTING dataset for hypothesis testing.
            
            IMPORTANT INSTRUCTIONS:
            1. The dataframe already exists as the variable 'df'
            2. DO NOT create any sample or test dataframes
            3. DO NOT include any __main__ or example usage sections
            4. Work ONLY with the existing 'df' variable
            5. Return ONLY the exact code needed to add the new column
            6. Consider patient-level normalization when creating the grouping variable
            
            The research hypothesis is:
            {hypothesis}
            
            The dataset has the following columns:
            {columns_info}
            
            Here's a sample of the first few rows of the dataset:
            {sample_data}
            
            {patient_level_info}
            
            Here's distribution information for some key variables:
            {distribution_info}
             
            Your task is to add a new column named '{variable_name}' to the existing 'df' dataframe.
            This column will be used for statistical analysis and hypothesis testing.
             
            PATIENT-LEVEL NORMALIZATION:
            - Identify patient identifier columns (patient_id, subject_id, etc.) in the dataset
            - If multiple rows exist for the same patient, ensure consistent group assignment
            - Consider whether analysis should be at patient level rather than observation level
            
            AGGREGATION STRATEGIES (if multiple rows per patient exist):
            - For numerical values: consider mean, median, max, min, or latest value per patient
            - For categorical values: consider mode, first, last, or presence/absence across records
            - Time-based strategies: first/last occurrence, value at specific timepoint
            - Summarize patient journey: create pattern-based groups (e.g., "improved then worsened")
            
            The grouping variable should:
            1. Be directly relevant to testing the stated hypothesis
            2. Divide the data into meaningful groups based on the hypothesis
            3. Include at least 2 groups (binary variable) or more if appropriate
            4. Use existing columns in the dataset to determine group membership
            5. Have descriptive group labels that clearly indicate what each group represents
            6. Be consistent for the same patient across multiple rows if applicable
             
            The code should:
            - ONLY modify the existing 'df' variable
            - ONLY add a single new column named '{variable_name}'
            - Not modify any existing data or structure
            - Include BRIEF comments explaining the logic
            - Be as concise as possible
            - Use appropriate aggregation if normalizing at patient level
             
            DO NOT:
            - Create any sample or test dataframes
            - Add any main method or test code
            - Include print statements or display code
            - Return or assign the dataframe to any variable other than 'df'
             
            PROVIDE ONLY THE MINIMAL PYTHON CODE THAT ADDS THE COLUMN TO THE EXISTING DF VARIABLE.
            """
            
            # Call LLM to generate code
            response = await call_llm_async(prompt)
            
            # Clean up the code - remove any markdown formatting or invisible characters
            code = self.sanitize_grouping_code(response, variable_name)
            
            # Create a new variable to hold the modified dataframe
            modified_df = None
            
            # Show the code to the user and ask for confirmation
            code_dialog = QDialog(self)
            code_dialog.setWindowTitle("Confirm Code for Group Variable")
            code_dialog.setMinimumSize(800, 600)
            
            code_layout = QVBoxLayout(code_dialog)
            
            code_label = QLabel("Generated code to create group variable:")
            code_layout.addWidget(code_label)
            
            code_editor = QPlainTextEdit()
            code_editor.setPlainText(code)
            code_editor.setReadOnly(False)  # Allow editing for adjustments
            code_layout.addWidget(code_editor)
            
            # Add note about editing
            edit_note = QLabel("You can edit the code if needed before applying it to the dataset.")
            edit_note.setStyleSheet("color: #666;")
            code_layout.addWidget(edit_note)
            
            code_buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Apply | 
                QDialogButtonBox.StandardButton.Cancel
            )
            # Connect the specific buttons rather than using accepted signal
            code_buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(code_dialog.accept)
            code_buttons.rejected.connect(code_dialog.reject)
            code_layout.addWidget(code_buttons)
            
            waiting_msg.accept()
            
            if code_dialog.exec() != QDialog.DialogCode.Accepted:
                return
                
            # Get the potentially edited code
            code = code_editor.toPlainText()
            
            # Execute the code to create the new variable
            try:
                # Create safe execution environment
                local_vars = {'df': df.copy(), 'pd': pd, 'np': np}
                
                # First, validate the code looks appropriate (basic check)
                if 'df[' not in code and variable_name not in code:
                    raise ValueError(f"The code doesn't appear to add the column '{variable_name}' to the dataframe")
                
                # Execute the code
                try:
                    exec(code, {'pd': pd, 'np': np}, local_vars)
                except Exception as exec_error:
                    error_msg = str(exec_error)
                    line_num = getattr(exec_error, 'lineno', 'unknown')
                    raise ValueError(f"Error executing code at line {line_num}: {error_msg}")
                
                # Check if the modified dataframe is available
                if 'df' in local_vars:
                    modified_df = local_vars['df']
                else:
                    raise ValueError("Code did not produce a modified dataframe")
                
                # Verify the new column was added
                if variable_name not in modified_df.columns:
                    raise ValueError(f"Code did not add the specified column: {variable_name}")
                
                # Make sure no columns were removed
                missing_cols = set(df.columns) - set(modified_df.columns)
                if missing_cols:
                    raise ValueError(f"Code removed existing columns: {missing_cols}")
                
                # Make sure our variable was added
                if variable_name not in modified_df.columns:
                    raise ValueError(f"Code did not add the required column '{variable_name}'")
                 
                # Update the dataset
                self.update_dataset(dataset_name, modified_df)
                
                # Analyze the created grouping variable
                analysis_results = self.analyze_grouping_variable(modified_df, variable_name, patient_id_cols)
                if analysis_results is None:
                    analysis_results = {}
                
                # Show success message with distribution of the new variable
                try:
                    # Format value counts with percentages for better interpretation
                    counts = modified_df[variable_name].value_counts()
                    percentages = modified_df[variable_name].value_counts(normalize=True) * 100
                    value_counts = pd.DataFrame({
                        'Count': counts,
                        'Percentage': percentages.map('{:.1f}%'.format)
                    }).to_string()
                except:
                    # Fallback if there's an error with the formatted display
                    value_counts = str(modified_df[variable_name].value_counts())
                
                success_msg = QMessageBox(self)
                success_msg.setWindowTitle("Group Variable Added")
                success_msg.setIcon(QMessageBox.Icon.Information)
                success_msg.setText(f"Successfully added '{variable_name}' to dataset '{dataset_name}'")
                
                # Include the analysis results in the success message
                info_text = f"Distribution of the new variable:\n{value_counts}\n"
                if analysis_results.get('patient_consistency_info'):
                    info_text += f"\n{analysis_results['patient_consistency_info']}"
                success_msg.setInformativeText(info_text)
                
                # Add example cross-tabulation if possible
                if 'discharge_disposition' in df.columns and len(df['discharge_disposition'].unique()) < 10:
                    try:
                        crosstab = pd.crosstab(
                            modified_df[variable_name], 
                            modified_df['discharge_disposition']
                        ).to_string()
                        detailed_text = f"Cross-tabulation with discharge_disposition:\n\n{crosstab}\n\n"
                        
                        # Add patient-level consistency analysis if available
                        if analysis_results.get('detailed_consistency'):
                            detailed_text += f"\nPatient-level consistency analysis:\n{analysis_results['detailed_consistency']}\n\n"
                            
                        detailed_text += f"Generated Code:\n{code}"
                        success_msg.setDetailedText(detailed_text)
                    except:
                        success_msg.setDetailedText(code)
                else:
                    success_msg.setDetailedText(code)
                
                success_msg.exec()
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to execute code: {str(e)}\n\nPlease adjust the code and try again."
                )
                
        except Exception as e:
            waiting_msg.accept()
            QMessageBox.critical(self, "Error", f"Error generating group variable: {str(e)}")

    def detect_patient_id_columns(self, df):
        """
        Detect potential patient identifier columns in the dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            list: List of column names that appear to be patient identifiers
        """
        id_columns = []
        
        # Check for common ID column names
        id_patterns = [
            'patient', 'subject', 'participant', 'person', 'client', 'member', 
            'id', 'identifier', 'mrn', 'record'
        ]
        
        for col in df.columns:
            col_lower = col.lower()
            # Check if column name contains ID patterns
            if any(pattern in col_lower for pattern in id_patterns):
                # Verify it has reasonable cardinality - not too few or too many unique values
                unique_count = df[col].nunique()
                if unique_count > 1 and unique_count <= len(df) * 0.9:
                    id_columns.append(col)
        
        return id_columns
    
    def analyze_patient_repeats(self, df, id_columns):
        """
        Analyze if there are multiple entries per patient
        
        Args:
            df (pd.DataFrame): Input dataframe
            id_columns (list): List of potential patient ID columns
            
        Returns:
            dict: Analysis results with information about repeated entries
        """
        results = {}
        
        for col in id_columns:
            # Count entries per ID
            value_counts = df[col].value_counts()
            max_repeats = value_counts.max()
            patients_with_repeats = sum(value_counts > 1)
            repeat_percentage = (patients_with_repeats / len(value_counts)) * 100 if len(value_counts) > 0 else 0
            
            results[col] = {
                'unique_values': len(value_counts),
                'max_repeats': max_repeats,
                'patients_with_repeats': patients_with_repeats,
                'repeat_percentage': repeat_percentage,
                'is_likely_id': max_repeats <= 20  # Heuristic: most patients shouldn't have more than 20 entries
            }
        
        return results

    def analyze_grouping_variable(self, df, variable_name, patient_id_cols):
        """
        Analyze the created grouping variable for patient-level consistency
        
        Args:
            df (pd.DataFrame): Input dataframe
            variable_name (str): Name of the grouping variable
            patient_id_cols (list): List of patient identifier columns
            
        Returns:
            dict: Analysis results with patient-level consistency information
        """
        analysis_results = {}
        
        try:
            # Basic stats about the grouping variable
            value_counts = df[variable_name].value_counts()
            analysis_results['value_counts'] = value_counts.to_dict()
            
            # Check for null values
            null_count = df[variable_name].isnull().sum()
            null_percentage = (null_count / len(df)) * 100 if len(df) > 0 else 0
            analysis_results['null_percentage'] = null_percentage
            
            # Check patient-level consistency if we have patient IDs
            if patient_id_cols and len(patient_id_cols) > 0:
                consistency_info = []
                detailed_consistency = []
                
                for id_col in patient_id_cols:
                    # Get patients with multiple entries
                    patient_counts = df[id_col].value_counts()
                    patients_with_multiple = sum(patient_counts > 1)
                    
                    if patients_with_multiple > 0:
                        # Analyze consistency for a sample of patients
                        consistency_stats = self._check_patient_consistency(df, id_col, variable_name)
                        
                        # Add formatted results
                        consistency_info.append(
                            f"Patient consistency check for {id_col}: "
                            f"{consistency_stats['consistent']} consistent, "
                            f"{consistency_stats['inconsistent']} inconsistent "
                            f"({consistency_stats['percentage']:.1f}% consistent)"
                        )
                        
                        # Add detailed examples
                        if consistency_stats['examples']:
                            detailed_consistency.append(f"Inconsistent {id_col} examples:")
                            for i, example in enumerate(consistency_stats['examples'][:5]):
                                detailed_consistency.append(
                                    f"  {i+1}. Patient {example['patient_id']} has groups: "
                                    f"{', '.join(map(str, example['groups']))}"
                                )
                
                # Add consistency info to results
                if consistency_info:
                    analysis_results['patient_consistency_info'] = "\n".join(consistency_info)
                if detailed_consistency:
                    analysis_results['detailed_consistency'] = "\n".join(detailed_consistency)
                
        except Exception as e:
            analysis_results['error'] = str(e)
        
        return analysis_results

    def _check_patient_consistency(self, df, id_col, variable_name):
        """
        Helper function to check consistency of group assignment for patients
        
        Args:
            df (pd.DataFrame): Dataframe to analyze
            id_col (str): Patient ID column name
            variable_name (str): Grouping variable name
            
        Returns:
            dict: Consistency statistics
        """
        # Get patients with multiple entries
        patient_counts = df[id_col].value_counts()
        patients_with_multiple = patient_counts[patient_counts > 1].index.tolist()
        
        # Limit to 20 patients for performance
        sample_patients = patients_with_multiple[:20]
        
        # Track consistency
        consistent_patients = 0
        inconsistent_patients = 0
        inconsistent_examples = []
        
        # Check each patient
        for patient_id in sample_patients:
            # Get all rows for this patient
            patient_rows = df[df[id_col] == patient_id]
            
            # Check if all entries have the same group value
            unique_groups = patient_rows[variable_name].unique()
            if len(unique_groups) == 1:
                consistent_patients += 1
            else:
                inconsistent_patients += 1
                inconsistent_examples.append({
                    'patient_id': patient_id,
                    'groups': [str(g) for g in unique_groups]  # Convert to strings for joining later
                })
        
        # Calculate percentage
        total = consistent_patients + inconsistent_patients
        consistency_percentage = (consistent_patients / total) * 100 if total > 0 else 0
        
        return {
            'consistent': consistent_patients,
            'inconsistent': inconsistent_patients,
            'percentage': consistency_percentage,
            'examples': inconsistent_examples
        }
        

class DataCollectionSettings:
    """Store settings for data collection widget"""
    def __init__(self):
        # Confirmation settings
        self.confirm_metadata_batch = True
        self.confirm_relationship_analysis = True
