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
    QHeaderView, QTabWidget, QToolButton, QSizePolicy, QStatusBar
)
import re
from abc import ABC, abstractmethod
import asyncio
from llms.client import call_llm_async
from qasync import asyncSlot
from PyQt6.QtGui import QIcon

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

class SourceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Data Source")
        self.resize(800, 600)  # Increased size
        self.source_type = None
        self.connection_info = {}
        self.init_ui()
        
    def init_ui(self):
        # Main layout with splitters for better space utilization
        main_layout = QVBoxLayout(self)
        
        # Source type selection
        source_type_group = QGroupBox("Source Type")
        type_layout = QVBoxLayout()
        
        # Replace radio buttons with dropdown
        type_layout.addWidget(QLabel("Select Source Type:"))
        self.source_type_combo = QComboBox()
        self.source_type_combo.addItems(["File Upload", "SQL Server", "SFTP", "REST API"])
        self.source_type_combo.currentIndexChanged.connect(self.on_source_type_changed)
        type_layout.addWidget(self.source_type_combo)
        
        source_type_group.setLayout(type_layout)
        main_layout.addWidget(source_type_group)

        # Stacked widget for different source configurations
        self.config_stack = QStackedWidget()
        
        # Upload config
        upload_widget = QWidget()
        upload_layout = QVBoxLayout(upload_widget)
        self.file_path_label = QLabel("Selected files: None")
        upload_layout.addWidget(self.file_path_label)
        upload_button = QPushButton("Browse Files")
        upload_button.clicked.connect(self.browse_files)
        upload_layout.addWidget(upload_button)
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
        
        sql_layout.addRow("Server:", self.server_input)
        sql_layout.addRow("Database:", self.database_input)
        sql_layout.addRow("Username:", self.username_input)
        sql_layout.addRow("Password:", self.password_input)
        sql_layout.addRow("Max Rows:", self.max_rows_input)
        sql_layout.addRow("SQL Query:", self.sql_query_input)
        
        # Add SQL formatting button
        format_sql_button = QPushButton("Format SQL")
        format_sql_button.clicked.connect(self.format_sql)
        sql_layout.addRow("", format_sql_button)
        
        # Add Save SQL button
        save_sql_button = QPushButton("Save SQL")
        save_sql_button.clicked.connect(self.save_sql)
        sql_layout.addRow("", save_sql_button)
        
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
        
        main_layout.addWidget(self.config_stack)
        
        # Connect source type dropdown to stack widget
        self.on_source_type_changed(0)  # Set default to File Upload

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

    def on_source_type_changed(self, index):
        """Handle source type changes in the dropdown"""
        self.config_stack.setCurrentIndex(index)
        source_types = ["upload", "sql", "sftp", "rest"]
        self.source_type = source_types[index]

    def browse_files(self):
        options = QFileDialog.Option.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "Select Files", "", "All Files (*);;TSV Files (*.tsv);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls)", options=options)
        if files:
            self.file_path_label.setText(f"Selected files: {len(files)} file(s)")
            self.connection_info['files'] = files

    def format_sql(self):
        # Basic SQL formatting - in a real application, use a proper SQL formatter
        sql = self.sql_query_input.toPlainText()
        keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN']
        formatted_sql = sql.upper()
        for keyword in keywords:
            formatted_sql = formatted_sql.replace(keyword, f'\n{keyword}')
        self.sql_query_input.setPlainText(formatted_sql)

    def save_sql(self):
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
        # This would typically connect to SFTP and show a file browser
        # For now, we'll just show a message
        QMessageBox.information(self, "SFTP Browser", "SFTP browsing would be implemented here")

    def accept(self):
        # Validate and collect connection info based on selected source type
        if self.source_type == "upload":
            if not self.connection_info.get('files'):
                QMessageBox.warning(self, "Error", "Please select at least one file")
                return
            
        elif self.source_type == "sql":
            if not all([self.server_input.text(), self.database_input.text(), 
                       self.username_input.text(), self.password_input.text(),
                       self.sql_query_input.toPlainText()]):
                QMessageBox.warning(self, "Error", "Please fill in all SQL connection fields")
                return
            self.connection_info = {
                'server': self.server_input.text(),
                'database': self.database_input.text(),
                'username': self.username_input.text(),
                'password': self.password_input.text(),
                'max_rows': self.max_rows_input.value(),
                'query': self.sql_query_input.toPlainText()
            }
            
        elif self.source_type == "sftp":
            if not all([self.sftp_host_input.text(), self.sftp_username_input.text(),
                       self.sftp_password_input.text(), self.sftp_path_input.text()]):
                QMessageBox.warning(self, "Error", "Please fill in all SFTP connection fields")
                return
            self.connection_info = {
                'host': self.sftp_host_input.text(),
                'port': self.sftp_port_input.value(),
                'username': self.sftp_username_input.text(),
                'password': self.sftp_password_input.text(),
                'path': self.sftp_path_input.text()
            }
            
        elif self.source_type == "rest":
            if not self.api_url_input.text():
                QMessageBox.warning(self, "Error", "Please enter API URL")
                return
            self.connection_info = {
                'url': self.api_url_input.text(),
                'method': self.api_method_combo.currentText(),
                'headers': self.api_headers_input.toPlainText(),
                'body': self.api_body_input.toPlainText()
            }
            
        super().accept()

class SourceConnection:
    def __init__(self, source_type, connection_info, file_name=None):
        self.source_type = source_type
        self.connection_info = connection_info
        self.file_name = file_name
        self.timestamp = datetime.now()
        self.data_source = None  # Will hold the DataSource implementation
        
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


class DataTransformer(QDialog):
    """
    Dialog for transforming data between different formats
    """
    transformation_applied = pyqtSignal(str, object)  # Signal emitted when a transformation is applied
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Transformer")
        self.resize(600, 400)  # Smaller size as this is now a launcher
        
        # Set dialog properties
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        
        # Store references to dataframes
        self.dataframes = {}
        self.format_cache = {}  # Cache detected formats
        self.current_dataframe = None
        self.current_name = ""
        
        # Create a status bar for feedback
        self.status_bar = QStatusBar(self)
        
        self.init_ui()
        
    def init_ui(self):
        # Main vertical layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)
        
        # Title and description
        title_label = QLabel("Data Transformation Tools")
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        main_layout.addWidget(title_label)
        
        description = QLabel(
            "Select a transformation tool below to convert your data between different formats. "
            "Each tool provides specialized functionality for different types of transformations."
        )
        description.setWordWrap(True)
        main_layout.addWidget(description)
        
        # Current dataset info
        dataset_info = QGroupBox("Current Dataset")
        dataset_layout = QVBoxLayout(dataset_info)
        
        self.current_dataset_label = QLabel("No dataset selected")
        self.current_dataset_label.setStyleSheet("font-weight: bold;")
        dataset_layout.addWidget(self.current_dataset_label)
        
        self.format_label = QLabel("Format: Unknown")
        dataset_layout.addWidget(self.format_label)
        
        # Add detect format button
        detect_button = QPushButton("Detect Format")
        detect_button.setIcon(QIcon.fromTheme("system-search"))
        detect_button.clicked.connect(self.detect_current_format)
        dataset_layout.addWidget(detect_button)
        
        main_layout.addWidget(dataset_info)
        
        # Transformation tools section
        tools_group = QGroupBox("Transformation Tools")
        tools_layout = QVBoxLayout(tools_group)
        
        # Basic format conversion button
        basic_button = QPushButton("Basic Format Conversion")
        basic_button.setIcon(QIcon.fromTheme("view-refresh"))
        basic_button.clicked.connect(self.show_basic_transformer)
        basic_button.setMinimumHeight(50)
        tools_layout.addWidget(basic_button)
        
        # Description for basic conversion
        basic_desc = QLabel("Convert between normalized, longitudinal, and columnar formats.")
        basic_desc.setWordWrap(True)
        basic_desc.setStyleSheet("color: #666;")
        tools_layout.addWidget(basic_desc)
        
        tools_layout.addSpacing(10)
        
        # Timepoint management button
        timepoint_button = QPushButton("Timepoint Management")
        timepoint_button.setIcon(QIcon.fromTheme("x-office-spreadsheet"))
        timepoint_button.clicked.connect(self.show_timepoint_transformer)
        timepoint_button.setMinimumHeight(50)
        tools_layout.addWidget(timepoint_button)
        
        # Description for timepoint management
        timepoint_desc = QLabel("Convert between long (longitudinal) and wide (columnar) formats for time-series data.")
        timepoint_desc.setWordWrap(True)
        timepoint_desc.setStyleSheet("color: #666;")
        tools_layout.addWidget(timepoint_desc)
        
        tools_layout.addSpacing(10)
        
        # Join button
        join_button = QPushButton("Multi-Source Join")
        join_button.setIcon(QIcon.fromTheme("x-office-address-book"))
        join_button.clicked.connect(self.show_join_transformer)
        join_button.setMinimumHeight(50)
        tools_layout.addWidget(join_button)
        
        # Description for join
        join_desc = QLabel("Combine multiple datasets using different join types (inner, left, right, outer).")
        join_desc.setWordWrap(True)
        join_desc.setStyleSheet("color: #666;")
        tools_layout.addWidget(join_desc)
        
        main_layout.addWidget(tools_group)
        main_layout.addStretch()
        
        # Add AI smart transform button
        smart_transform_button = QPushButton("Smart Transform (AI)")
        smart_transform_button.setIcon(QIcon.fromTheme("applications-science"))
        smart_transform_button.clicked.connect(self.smart_transform)
        smart_transform_button.setMinimumHeight(40)
        main_layout.addWidget(smart_transform_button)
        
        # Add dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)  # Close button will close the dialog
        
        # Add status bar and button box
        main_layout.addWidget(self.status_bar)
        main_layout.addWidget(button_box)
        
        self.status_bar.showMessage("Ready")
    
    def set_current_dataset(self, name, dataframe):
        """
        Set the current dataset for transformation
        """
        self.current_name = name
        self.current_dataframe = dataframe
        self.current_dataset_label.setText(f"Dataset: {name}")
        
        # Detect format
        self.detect_current_format()
    
    def update_status(self, message):
        """Update status bar with message"""
        self.status_bar.showMessage(message)
        
    def detect_current_format(self):
        """
        Detect the format of the current dataframe
        """
        if self.current_dataframe is None:
            return
            
        if self.current_name in self.format_cache:
            detected_format = self.format_cache[self.current_name]
        else:
            detected_format = self.detect_data_format(self.current_dataframe)
            self.format_cache[self.current_name] = detected_format
            
        self.format_label.setText(f"Format: {detected_format.capitalize()}")
        
    def detect_data_format(self, df):
        """
        Detect if dataframe is in normalized, longitudinal, or columnar format
        """
        # Check for common visit/timepoint pattern in column names (columnar format)
        visit_pattern = any(col for col in df.columns if re.search(r'_visit\d+|_v\d+|_timepoint\d+|_t\d+', col.lower()))
        
        # Check for visit/timepoint columns (longitudinal format)
        has_visit_column = any(col for col in df.columns if col.lower() in ['visit', 'timepoint', 'visit_id', 'visit_date'])
        
        # Check for normalized format indicators (e.g., ID columns that would be used for joins)
        has_id_columns = any(col for col in df.columns if col.lower().endswith('_id'))
        
        if visit_pattern:
            return "columnar"
        elif has_visit_column:
            return "longitudinal"
        elif has_id_columns:
            return "normalized"
        else:
            return "unknown"
    
    def set_available_sources(self, sources_dict):
        """
        Update the available sources
        """
        self.dataframes = sources_dict
        self.format_cache = {}  # Reset format cache
    
    def show_basic_transformer(self):
        """Show the basic format transformer dialog"""
        if self.current_dataframe is None:
            QMessageBox.warning(self, "Error", "No dataset selected")
            return
            
        dialog = BasicFormatTransformerDialog(self)
        dialog.set_current_dataset(self.current_name, self.current_dataframe, 
                                  self.format_cache.get(self.current_name, "unknown"))
        dialog.transformation_applied.connect(self.handle_transformation)
        dialog.exec()
    
    def show_timepoint_transformer(self):
        """Show the timepoint transformer dialog"""
        if self.current_dataframe is None:
            QMessageBox.warning(self, "Error", "No dataset selected")
            return
            
        dialog = TimepointTransformerDialog(self)
        dialog.set_current_dataset(self.current_name, self.current_dataframe, 
                                  self.format_cache.get(self.current_name, "unknown"))
        dialog.transformation_applied.connect(self.handle_transformation)
        dialog.exec()
    
    def show_join_transformer(self):
        """Show the join transformer dialog"""
        if len(self.dataframes) < 2:
            QMessageBox.warning(self, "Error", "Need at least two datasets for joining")
            return
            
        dialog = JoinTransformerDialog(self)
        dialog.set_available_sources(self.dataframes)
        dialog.transformation_applied.connect(self.handle_transformation)
        dialog.exec()
    
    def handle_transformation(self, name, dataframe):
        """Handle transformation from any of the transformer dialogs"""
        self.transformation_applied.emit(name, dataframe)
    
    @asyncSlot() 
    async def smart_transform(self):
        """Use Gemini LLM to suggest and apply the best transformation"""
        self.update_status("Analyzing dataset for smart transformation...")
        if self.current_dataframe is None:
            QMessageBox.warning(self, "Error", "No dataset selected")
            return
            
        # Create a prompt for Gemini API
        prompt = f"""
        I need to transform this dataset into the optimal format for analysis.
        
        Dataset Name: {self.current_name}
        Dataset Columns: {list(self.current_dataframe.columns)}
        Dataset Sample (first 3 rows):
        {self.current_dataframe.head(3).to_string()}
        
        I need you to:
        1. Determine if this data is in normalized, longitudinal, or columnar format
        2. Recommend the best target format for general analysis (longitudinal is often good for mixed analysis)
        3. Identify key columns for the transformation (subject IDs, timepoints, measures)
        
        Return your response as a JSON object with the following structure:
        {{
            "current_format": "normalized|longitudinal|columnar",
            "recommended_format": "normalized|longitudinal|columnar",
            "subject_id_column": "column_name",
            "timepoint_column": "column_name",  // if applicable
            "value_columns": ["column1", "column2"],  // columns with interesting values
            "explanation": "explanation of the recommendation",
            "python_code": "code to perform the transformation"  // pandas code to transform
        }}
        """
        
        # Show loading message
        QMessageBox.information(self, "Processing", "Analyzing dataset with AI. This may take a moment...")
        
        try:
            # Call Gemini API asynchronously
            response = await call_llm_async(prompt)
            
            # Parse the JSON response
            # Extract JSON from the response (in case the API returns additional text)
            json_str = re.search(r'({.*})', response, re.DOTALL)
            if json_str:
                result = json.loads(json_str.group(1))
                
                current_format = result.get("current_format", "unknown")
                recommended_format = result.get("recommended_format")
                explanation = result.get("explanation")
                
                # Update UI with detected format
                self.format_label.setText(f"Format: {current_format.capitalize()}")
                self.format_cache[self.current_name] = current_format
                
                # Ask user if they want to apply the recommended transformation
                msg_box = QMessageBox()
                msg_box.setWindowTitle("AI Transformation Recommendation")
                msg_box.setText(f"Current format: {current_format.capitalize()}\n"
                               f"Recommended format: {recommended_format.capitalize()}\n\n"
                               f"Explanation: {explanation}")
                msg_box.setInformativeText("Do you want to apply this transformation?")
                msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)
                
                if msg_box.exec() == QMessageBox.StandardButton.Yes:
                    # Apply the transformation using the provided Python code
                    try:
                        # Extract the Python code
                        python_code = result.get("python_code", "")
                        
                        # Execute the code in a controlled environment
                        # We're using exec with a dedicated namespace for safety
                        namespace = {
                            "pd": pd, 
                            "df": self.current_dataframe,
                            "np": np,
                            "re": re
                        }
                        
                        # Add the transformation code
                        exec(python_code, namespace)
                        
                        # Get the transformed dataframe (should be assigned to 'result_df' in the code)
                        if 'result_df' in namespace:
                            transformed_df = namespace['result_df']
                            
                            # Set default save name
                            save_name = f"{self.current_name}_{recommended_format}"
                            
                            # Emit the transformed dataframe
                            self.transformation_applied.emit(save_name, transformed_df)
                            
                            QMessageBox.information(self, "Success", 
                                                  f"Transformation to {recommended_format} format completed successfully.\n"
                                                  f"New dataset '{save_name}' has been created.")
                        else:
                            QMessageBox.warning(self, "Error", "Transformation did not produce a result dataframe")
                            
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Failed to apply transformation: {str(e)}")
            else:
                QMessageBox.warning(self, "Error", "Could not parse AI response")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"AI analysis failed: {str(e)}")


class BasicFormatTransformerDialog(QDialog):
    """Dialog for basic format transformations (normalized, longitudinal, columnar)"""
    
    transformation_applied = pyqtSignal(str, object)  # Signal emitted when a transformation is applied
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Basic Format Transformer")
        self.resize(800, 700)
        
        # Set dialog properties
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        
        # Store references
        self.current_dataframe = None
        self.current_name = ""
        self.current_format = "unknown"
        self._preview_df = None
        
        # Create a status bar for feedback
        self.status_bar = QStatusBar(self)
        
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)
        
        # Create a splitter for better resizing
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.setChildrenCollapsible(False)
        
        # Top panel for controls
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        # Current dataset and format
        info_group = QGroupBox("Dataset Information")
        info_layout = QVBoxLayout(info_group)
        
        self.dataset_label = QLabel("Dataset: None")
        self.dataset_label.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(self.dataset_label)
        
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Current Format:"))
        
        self.format_label = QLabel("Unknown")
        self.format_label.setStyleSheet("font-weight: bold;")  # Removed color
        self.format_label.setFrameShape(QLabel.Shape.StyledPanel)
        self.format_label.setFrameShadow(QLabel.Shadow.Sunken)
        self.format_label.setMinimumWidth(100)
        self.format_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        format_layout.addWidget(self.format_label)
        
        format_layout.addStretch()
        
        # Add helper text explaining formats - removed background color
        format_help = QLabel(
            "<b>Formats:</b><br>"
            "<b>Normalized</b>: Data split into multiple tables with keys for joins, like a relational database<br>"
            "<b>Longitudinal</b>: One row per subject per timepoint, measurements in columns<br>"
            "<b>Columnar</b>: One row per subject, measurements from each timepoint in separate columns"
        )
        format_help.setStyleSheet("padding: 5px; border-radius: 3px;")  # Removed colors
        format_help.setWordWrap(True)
        
        info_layout.addLayout(format_layout)
        info_layout.addWidget(format_help)
        
        top_layout.addWidget(info_group)
        
        # Transformation controls
        transform_group = QGroupBox("Transform Format")
        transform_layout = QVBoxLayout(transform_group)
        
        # Add helper text explaining the operation
        transform_help = QLabel(
            "Select a target format below to convert your data. The transformation will be previewed before saving."
        )
        transform_help.setWordWrap(True)
        transform_layout.addWidget(transform_help)
        
        # Button group to show which format is active
        button_layout = QHBoxLayout()
        
        self.to_normalized_btn = QPushButton("Convert to Normalized")
        self.to_normalized_btn.setIcon(QIcon.fromTheme("view-list-text"))
        self.to_normalized_btn.clicked.connect(lambda: self.convert_format("normalized"))
        self.to_normalized_btn.setCheckable(True)
        button_layout.addWidget(self.to_normalized_btn)
        
        self.to_longitudinal_btn = QPushButton("Convert to Longitudinal")
        self.to_longitudinal_btn.setIcon(QIcon.fromTheme("view-list-details"))
        self.to_longitudinal_btn.clicked.connect(lambda: self.convert_format("longitudinal"))
        self.to_longitudinal_btn.setCheckable(True)
        button_layout.addWidget(self.to_longitudinal_btn)
        
        self.to_columnar_btn = QPushButton("Convert to Columnar")
        self.to_columnar_btn.setIcon(QIcon.fromTheme("view-column"))
        self.to_columnar_btn.clicked.connect(lambda: self.convert_format("columnar"))
        self.to_columnar_btn.setCheckable(True)
        button_layout.addWidget(self.to_columnar_btn)
        
        transform_layout.addLayout(button_layout)
        
        # Add format descriptions
        self.format_description = QPlainTextEdit()
        self.format_description.setReadOnly(True)
        self.format_description.setPlaceholderText("Format conversion description will appear here.")
        self.format_description.setMaximumHeight(100)
        transform_layout.addWidget(self.format_description)
        
        top_layout.addWidget(transform_group)
        
        main_splitter.addWidget(top_panel)
        
        # Results display in bottom panel
        bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        
        results_group = QGroupBox("Transformation Preview")
        results_layout = QVBoxLayout(results_group)
        
        # Add preview dataset display
        self.preview_display = DataFrameDisplay()
        results_layout.addWidget(self.preview_display)
        
        # Save transformed dataset
        save_layout = QHBoxLayout()
        save_layout.addWidget(QLabel("Save As:"))
        
        self.save_name_input = QLineEdit()
        self.save_name_input.setPlaceholderText("Enter name for transformed dataset")
        save_layout.addWidget(self.save_name_input, 1)
        
        save_button = QPushButton("Save Transformation")
        save_button.setIcon(QIcon.fromTheme("document-save"))
        save_button.clicked.connect(self.save_transformation)
        save_layout.addWidget(save_button)
        
        results_layout.addLayout(save_layout)
        
        bottom_layout.addWidget(results_group)
        main_splitter.addWidget(bottom_panel)
        
        # Set reasonable default sizes for the splitter
        main_splitter.setSizes([300, 400])
        
        main_layout.addWidget(main_splitter, 1)
        
        # Add status bar
        main_layout.addWidget(self.status_bar)
        
        # Add dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)
        
        self.status_bar.showMessage("Ready")
        
        # Initialize format descriptions
        self.update_format_descriptions()
        
    def set_current_dataset(self, name, dataframe, format_type="unknown"):
        """Set the current dataset for transformation"""
        self.current_name = name
        self.current_dataframe = dataframe
        self.current_format = format_type
        
        # Update UI
        self.dataset_label.setText(f"Dataset: {name}")
        self.format_label.setText(format_type.capitalize())
        
        # Set save name default
        self.save_name_input.setText(f"{name}_transformed")
        
        # Update button states
        self.update_button_states()
    
    def update_button_states(self):
        """Update button states based on current format"""
        self.to_normalized_btn.setChecked(self.current_format == "normalized")
        self.to_longitudinal_btn.setChecked(self.current_format == "longitudinal")
        self.to_columnar_btn.setChecked(self.current_format == "columnar")
        
        # Disable button for current format
        self.to_normalized_btn.setEnabled(self.current_format != "normalized")
        self.to_longitudinal_btn.setEnabled(self.current_format != "longitudinal")
        self.to_columnar_btn.setEnabled(self.current_format != "columnar")
    
    def update_format_descriptions(self):
        """Update the format descriptions text area"""
        descriptions = {
            "normalized": "Normalized format splits data into multiple related tables with foreign keys for joins. "
                         "This format reduces redundancy but requires joins for analysis.",
            
            "longitudinal": "Longitudinal format has one row per subject per timepoint with measurements as columns. "
                           "This is ideal for time-series analysis and mixed models.",
            
            "columnar": "Columnar (wide) format has one row per subject with separate columns for each measure at each timepoint. "
                       "This is useful for summary statistics and many statistical tests."
        }
        
        text = "Format Conversion Information:\n\n"
        for fmt, desc in descriptions.items():
            text += f"{fmt.capitalize()}:\n{desc}\n\n"
        
        self.format_description.setPlainText(text)
    
    def update_status(self, message):
        """Update status bar with message"""
        self.status_bar.showMessage(message)
    
    def convert_format(self, target_format):
        """Convert the current dataframe to the target format"""
        if self.current_dataframe is None:
            QMessageBox.warning(self, "Error", "No dataset selected")
            return
            
        if self.current_format == target_format:
            QMessageBox.information(self, "Info", f"Dataset is already in {target_format} format")
            return
            
        self.update_status(f"Converting from {self.current_format} to {target_format}...")
        
        # Conversion logic
        try:
            if self.current_format == "normalized" and target_format == "longitudinal":
                result_df = self.normalized_to_longitudinal()
                conversion_description = "Normalized → Longitudinal"
            elif self.current_format == "normalized" and target_format == "columnar":
                # First convert to longitudinal, then to columnar
                long_df = self.normalized_to_longitudinal()
                result_df = self.longitudinal_to_columnar(long_df)
                conversion_description = "Normalized → Longitudinal → Columnar"
            elif self.current_format == "longitudinal" and target_format == "columnar":
                result_df = self.longitudinal_to_columnar()
                conversion_description = "Longitudinal → Columnar"
            elif self.current_format == "longitudinal" and target_format == "normalized":
                result_df = self.longitudinal_to_normalized()
                conversion_description = "Longitudinal → Normalized"
            elif self.current_format == "columnar" and target_format == "longitudinal":
                result_df = self.columnar_to_longitudinal()
                conversion_description = "Columnar → Longitudinal"
            elif self.current_format == "columnar" and target_format == "normalized":
                # First convert to longitudinal, then to normalized
                long_df = self.columnar_to_longitudinal()
                result_df = self.longitudinal_to_normalized(long_df)
                conversion_description = "Columnar → Longitudinal → Normalized"
            else:
                QMessageBox.warning(self, "Error", f"Conversion from {self.current_format} to {target_format} not implemented")
                return
                
            # Display the result
            self.preview_display.display_dataframe(result_df)
            
            # Set default save name
            self.save_name_input.setText(f"{self.current_name}_{target_format}")
            
            # Store the preview dataframe
            self._preview_df = result_df
            
            # Update status
            self.update_status(f"Converted from {self.current_format} to {target_format} - {len(result_df)} rows")
            
            # Remember the new format for this preview
            self.current_format = target_format
            self.format_label.setText(target_format.capitalize())
            
            # Update button states
            self.update_button_states()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Conversion failed: {str(e)}")
    
    def normalized_to_longitudinal(self):
        """
        Convert normalized data to longitudinal format using pandas merge
        This is a simplified example - assumes we have subjects and visits tables
        """
        # In a real implementation, we would need to:
        # 1. Identify the subject and visit/timepoint tables
        # 2. Join them appropriately
        # 3. Join with measurement tables
        
        # For demonstration, let's assume we have a simplified clinical dataset
        df = self.current_dataframe
        
        # Try to find subject ID column - any column ending with _id except visit_id
        subject_id_cols = [col for col in df.columns 
                         if col.lower().endswith('_id') and col.lower() != 'visit_id']
        
        if not subject_id_cols:
            # Fall back to any column with 'subject' or 'patient' in the name
            subject_id_cols = [col for col in df.columns 
                              if 'subject' in col.lower() or 'patient' in col.lower()]
            
        if not subject_id_cols:
            raise ValueError("Could not identify a subject ID column")
            
        subject_id_col = subject_id_cols[0]
        
        # Create a basic longitudinal structure if needed
        if 'Visit' not in df.columns and 'Timepoint' not in df.columns and 'visit_id' not in df.columns:
            # This is very simplified - in real implementation, we would handle real visit data
            # For demo, create visits for each subject
            subjects = df[subject_id_col].unique()
            visits = [1, 2, 3]  # Example visits
            
            # Create a cartesian product of subjects and visits
            visits_df = pd.DataFrame([(s, v) for s in subjects for v in visits], 
                                     columns=[subject_id_col, 'Visit'])
            
            # Merge with original data
            result = pd.merge(visits_df, df, on=subject_id_col, how='left')
            return result
        
        # If we already have appropriate structure, just return it
        return df.copy()
    
    def longitudinal_to_columnar(self, df=None):
        """
        Convert longitudinal data to columnar (wide) format
        """
        if df is None:
            df = self.current_dataframe
            
        # Try to find subject ID and timepoint columns
        subject_id_cols = [col for col in df.columns 
                         if 'subject' in col.lower() or 'patient' in col.lower() or col.lower().endswith('_id')]
        
        if not subject_id_cols:
            raise ValueError("Could not identify a subject ID column")
            
        subject_id_col = subject_id_cols[0]
        
        # Look for timepoint/visit column
        timepoint_cols = [col for col in df.columns 
                        if col.lower() in ['visit', 'timepoint', 'visit_id', 'visit_number']]
        
        if not timepoint_cols:
            # Try columns with date in the name
            timepoint_cols = [col for col in df.columns if 'date' in col.lower()]
            
        if not timepoint_cols:
            raise ValueError("Could not identify a timepoint/visit column")
            
        timepoint_col = timepoint_cols[0]
        
        # Find numeric columns that aren't IDs for measurements
        value_cols = [col for col in df.select_dtypes(include='number').columns 
                     if col != subject_id_col and col != timepoint_col and not col.lower().endswith('_id')]
        
        if not value_cols:
            # If no numeric columns, try all non-ID, non-timepoint columns
            value_cols = [col for col in df.columns 
                         if col != subject_id_col and col != timepoint_col and not col.lower().endswith('_id')]
            
        if not value_cols:
            raise ValueError("Could not identify any value columns to pivot")
        
        # Create a wide format for each value column
        wide_dfs = []
        
        for val_col in value_cols:
            # Create column names based on visit/timepoint
            # Format: {value_column}_visit{visit_number}
            pivot_df = df.pivot(index=subject_id_col, 
                              columns=timepoint_col, 
                              values=val_col)
            
            # Rename columns to standard format
            pivot_df.columns = [f"{val_col}_visit{int(col)}" if isinstance(col, (int, float)) 
                               else f"{val_col}_{col}" for col in pivot_df.columns]
            
            # Reset index to make the subject ID a regular column
            pivot_df.reset_index(inplace=True)
            
            wide_dfs.append(pivot_df)
        
        # Merge all the wide dataframes on subject ID
        if not wide_dfs:
            return df.copy()  # Return original if no transformation was done
            
        result = wide_dfs[0]
        for wide_df in wide_dfs[1:]:
            result = pd.merge(result, wide_df, on=subject_id_col, how='outer')
            
        return result
    
    def longitudinal_to_normalized(self, df=None):
        """
        Convert longitudinal data to normalized format
        This is a simplified example - in reality, normalization would depend on the specific database schema
        """
        if df is None:
            df = self.current_dataframe
            
        # Try to find subject ID and timepoint columns
        subject_id_cols = [col for col in df.columns 
                         if 'subject' in col.lower() or 'patient' in col.lower() or col.lower().endswith('_id')]
        
        if not subject_id_cols:
            raise ValueError("Could not identify a subject ID column")
            
        subject_id_col = subject_id_cols[0]
        
        # Look for timepoint/visit column
        timepoint_cols = [col for col in df.columns 
                        if col.lower() in ['visit', 'timepoint', 'visit_id', 'visit_number']]
        
        if not timepoint_cols:
            # Try columns with date in the name
            timepoint_cols = [col for col in df.columns if 'date' in col.lower()]
            
        if not timepoint_cols:
            raise ValueError("Could not identify a timepoint/visit column")
            
        timepoint_col = timepoint_cols[0]
        
        # Create subjects table
        subjects_df = df[[subject_id_col]].drop_duplicates()
        subjects_df['subject_key'] = range(1, len(subjects_df) + 1)  # Add a numeric key
        
        # Create visits table
        visits_df = df[[subject_id_col, timepoint_col]].drop_duplicates()
        visits_df['visit_id'] = range(1, len(visits_df) + 1)  # Add a numeric key
        
        # For demo purposes, we'll return the combined normalized tables
        # In a real implementation, you would create separate tables for each type of measurement
        
        # Join the tables back to demonstrate the normalized structure
        result = pd.merge(subjects_df, visits_df, on=subject_id_col, how='left')
        
        return result
    
    def columnar_to_longitudinal(self):
        """
        Convert columnar (wide) data to longitudinal (long) format
        """
        df = self.current_dataframe
            
        # Try to find the subject ID column
        subject_id_cols = [col for col in df.columns 
                         if 'subject' in col.lower() or 'patient' in col.lower() or col.lower().endswith('_id')]
        
        if not subject_id_cols:
            raise ValueError("Could not identify a subject ID column")
            
        subject_id_col = subject_id_cols[0]
        
        # Identify value columns with timepoints (using regex pattern)
        # Example: weight_visit1, bp_visit2, hba1c_visit3
        value_timepoint_cols = [col for col in df.columns 
                               if re.search(r'_visit\d+|_v\d+|_timepoint\d+|_t\d+', col.lower())]
        
        if not value_timepoint_cols:
            raise ValueError("No timepoint columns detected in this dataset")
            
        # Extract measurement types and create a mapping of columns to (measure, timepoint)
        measure_timepoint_map = {}
        for col in value_timepoint_cols:
            # Extract the measure name and visit number using regex
            match = re.search(r'(.+)_(visit|v|timepoint|t)(\d+)', col.lower())
            if match:
                measure = match.group(1)
                timepoint = int(match.group(3))
                measure_timepoint_map[col] = (measure, timepoint)
        
        # Prepare for the melt operation
        id_vars = [subject_id_col]
        value_vars = list(measure_timepoint_map.keys())
        
        # Basic melt to long format
        long_df = pd.melt(df, id_vars=id_vars, value_vars=value_vars, 
                         var_name='variable', value_name='value')
        
        # Extract measure and timepoint from variable column
        long_df['measure'] = long_df['variable'].apply(lambda x: measure_timepoint_map[x][0])
        long_df['visit'] = long_df['variable'].apply(lambda x: measure_timepoint_map[x][1])
        
        # Drop the variable column as we now have measure and visit
        long_df.drop('variable', axis=1, inplace=True)
        
        # For a complete implementation, we would pivot back to get all measures as columns
        # For demo, we'll return this long format
        return long_df
    
    def save_transformation(self):
        """Save the transformed dataset"""
        new_name = self.save_name_input.text()
        
        if not new_name:
            QMessageBox.warning(self, "Error", "Please enter a name for the transformed dataset")
            return
            
        if not self._preview_df is not None:
            QMessageBox.warning(self, "Error", "No transformation to save")
            return
            
        # Emit signal to add this as a new dataset
        self.transformation_applied.emit(new_name, self._preview_df)
        
        QMessageBox.information(self, "Success", f"Dataset '{new_name}' saved successfully")
        self.update_status(f"Transformation saved as '{new_name}'")


class TimepointTransformerDialog(QDialog):
    """Dialog for timepoint transformations (long/wide format conversions)"""
    
    transformation_applied = pyqtSignal(str, object)  # Signal emitted when a transformation is applied
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Timepoint Transformer")
        self.resize(1200, 800)  # Set default size, will be maximized 
        
        # Set dialog properties
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        
        # Store references
        self.current_dataframe = None
        self.current_name = ""
        self.current_format = "unknown"
        self._preview_df = None
        self.current_transform = None  # 'long_to_wide' or 'wide_to_long'
        
        # Create a status bar for feedback
        self.status_bar = QStatusBar(self)
        
        self.init_ui()
        
        # Make dialog maximized by default
        self.showMaximized()
        
    def init_ui(self):
        # Main layout is vertical, but with horizontal sections
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Top section with controls - horizontal layout
        controls_section = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Dataset info and transformation type
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(8)
        
        # Current dataset and format
        info_group = QGroupBox("Dataset Information")
        info_layout = QVBoxLayout(info_group)
        
        self.dataset_label = QLabel("Dataset: None")
        self.dataset_label.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(self.dataset_label)
        
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Current Format:"))
        
        self.format_label = QLabel("Unknown")
        self.format_label.setStyleSheet("font-weight: bold;")
        self.format_label.setFrameShape(QLabel.Shape.StyledPanel)
        self.format_label.setFrameShadow(QLabel.Shadow.Sunken)
        self.format_label.setMinimumWidth(100)
        self.format_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        format_layout.addWidget(self.format_label)
        
        format_layout.addStretch()
        
        info_layout.addLayout(format_layout)
        
        # Add helper text explaining formats
        format_help = QLabel(
            "<b>Format Types:</b><br>"
            "<b>Long Format</b> (Longitudinal): One row per subject per timepoint, with measurements in columns<br>"
            "<b>Wide Format</b> (Columnar): One row per subject, with separate columns for each measure at each timepoint"
        )
        format_help.setStyleSheet("padding: 5px; border-radius: 3px;")
        format_help.setWordWrap(True)
        info_layout.addWidget(format_help)
        
        left_layout.addWidget(info_group)
        
        # Transformation controls
        transform_group = QGroupBox("Transformation Direction")
        transform_layout = QVBoxLayout(transform_group)
        
        # Radio buttons for transformation direction
        radio_layout = QHBoxLayout()
        
        self.transform_group = QButtonGroup(self)
        
        self.long_to_wide_radio = QRadioButton("Long → Wide")
        self.long_to_wide_radio.toggled.connect(self.update_active_transform)
        self.transform_group.addButton(self.long_to_wide_radio)
        radio_layout.addWidget(self.long_to_wide_radio)
        
        self.wide_to_long_radio = QRadioButton("Wide → Long")
        self.wide_to_long_radio.toggled.connect(self.update_active_transform)
        self.transform_group.addButton(self.wide_to_long_radio)
        radio_layout.addWidget(self.wide_to_long_radio)
        
        radio_layout.addStretch()
        
        transform_layout.addLayout(radio_layout)
        
        # Add helper text explaining the operations
        transform_desc = QLabel(
            "<b>Long → Wide:</b> Converts from longitudinal format (one row per subject per timepoint) "
            "to columnar format (one row per subject with multiple columns for each timepoint).<br>"
            "<b>Wide → Long:</b> Converts from columnar format (one row per subject) "
            "to longitudinal format (multiple rows per subject, one per timepoint)."
        )
        transform_desc.setWordWrap(True)
        transform_layout.addWidget(transform_desc)
        
        left_layout.addWidget(transform_group)
        
        # AI detect button
        auto_detect_button = QPushButton("Auto-Detect Columns (AI)")
        auto_detect_button.setIcon(QIcon.fromTheme("system-search"))
        auto_detect_button.clicked.connect(self.auto_detect_timepoints)
        left_layout.addWidget(auto_detect_button)
        
        # Add apply transformation button at bottom of left panel
        transform_button = QPushButton("Apply Transformation")
        transform_button.setIcon(QIcon.fromTheme("view-refresh"))
        transform_button.clicked.connect(self.apply_transformation)
        left_layout.addWidget(transform_button)
        
        left_layout.addStretch()
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(400)
        controls_section.addWidget(left_panel)
        
        # Middle panel: Column selection and timepoint handling
        middle_panel = QWidget()
        middle_layout = QVBoxLayout(middle_panel)
        middle_layout.setContentsMargins(5, 5, 5, 5)
        middle_layout.setSpacing(8)
        
        # Basic column selection
        columns_group = QGroupBox("Basic Column Selection")
        columns_layout = QFormLayout(columns_group)
        columns_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        
        self.subject_id_combo = QComboBox()
        self.subject_id_combo.setMinimumWidth(180)
        columns_layout.addRow("Subject ID Column:", self.subject_id_combo)
        
        self.timepoint_column_combo = QComboBox()
        self.timepoint_column_combo.setMinimumWidth(180)
        self.timepoint_column_combo.currentTextChanged.connect(self.on_timepoint_column_changed)
        columns_layout.addRow("Timepoint Column:", self.timepoint_column_combo)
        
        self.value_column_combo = QComboBox()
        self.value_column_combo.setMinimumWidth(180)
        columns_layout.addRow("Value Column:", self.value_column_combo)
        
        middle_layout.addWidget(columns_group)
        
        # Advanced timepoint handling options - tabbed interface for better organization
        self.timepoint_group = QGroupBox("Timepoint Handling Options")
        timepoint_layout = QVBoxLayout(self.timepoint_group)
        
        # Option for handling date-based timepoints
        date_options_layout = QHBoxLayout()
        date_options_layout.addWidget(QLabel("Date Handling Method:"))
        
        self.date_handling_combo = QComboBox()
        self.date_handling_combo.addItems(["Use as is", "Group by intervals", "Binned timepoints", "Regular intervals with window"])
        self.date_handling_combo.currentIndexChanged.connect(self.on_date_handling_changed)
        date_options_layout.addWidget(self.date_handling_combo, 1)
        
        timepoint_layout.addLayout(date_options_layout)
        
        # Stacked widget for different date handling options
        self.date_handling_stack = QStackedWidget()
        
        # Option 1: Use as is (no additional controls needed)
        as_is_widget = QWidget()
        as_is_layout = QVBoxLayout(as_is_widget)
        as_is_layout.addWidget(QLabel("Using timepoint values as-is without modification."))
        as_is_layout.addStretch()
        self.date_handling_stack.addWidget(as_is_widget)
        
        # Option 2: Group by intervals - more compact layout
        interval_widget = QWidget()
        interval_layout = QHBoxLayout(interval_widget)
        interval_layout.setContentsMargins(0, 5, 0, 5)
        
        interval_layout.addWidget(QLabel("Group every:"))
        
        self.interval_value_spin = QSpinBox()
        self.interval_value_spin.setRange(1, 365)
        self.interval_value_spin.setValue(30)
        interval_layout.addWidget(self.interval_value_spin)
        
        self.interval_unit_combo = QComboBox()
        self.interval_unit_combo.addItems(["Days", "Weeks", "Months", "Years"])
        interval_layout.addWidget(self.interval_unit_combo)
        
        interval_layout.addStretch(1)
        
        self.date_handling_stack.addWidget(interval_widget)
        
        # Option 3: Binned timepoints - more compact layout
        binned_widget = QWidget()
        binned_layout = QGridLayout(binned_widget)
        binned_layout.setContentsMargins(0, 5, 0, 5)
        
        binned_layout.addWidget(QLabel("Start Date:"), 0, 0)
        self.bin_start_edit = QLineEdit()
        self.bin_start_edit.setPlaceholderText("YYYY-MM-DD")
        binned_layout.addWidget(self.bin_start_edit, 0, 1)
        
        binned_layout.addWidget(QLabel("End Date:"), 0, 2)
        self.bin_end_edit = QLineEdit()
        self.bin_end_edit.setPlaceholderText("YYYY-MM-DD")
        binned_layout.addWidget(self.bin_end_edit, 0, 3)
        
        binned_layout.addWidget(QLabel("Number of Bins:"), 1, 0)
        self.bin_count_spin = QSpinBox()
        self.bin_count_spin.setRange(2, 100)
        self.bin_count_spin.setValue(4)
        binned_layout.addWidget(self.bin_count_spin, 1, 1)
        
        binned_layout.setColumnStretch(1, 1)
        binned_layout.setColumnStretch(3, 1)
        
        self.date_handling_stack.addWidget(binned_widget)
        
        # Option 4: Regular intervals with window - more compact layout
        window_widget = QWidget()
        window_layout = QGridLayout(window_widget)
        window_layout.setContentsMargins(0, 5, 0, 5)
        
        window_layout.addWidget(QLabel("Every:"), 0, 0)
        self.window_interval_spin = QSpinBox()
        self.window_interval_spin.setRange(1, 52)
        self.window_interval_spin.setValue(4)
        window_layout.addWidget(self.window_interval_spin, 0, 1)
        
        self.window_unit_combo = QComboBox()
        self.window_unit_combo.addItems(["Days", "Weeks", "Months"])
        self.window_unit_combo.setCurrentText("Weeks")
        window_layout.addWidget(self.window_unit_combo, 0, 2)
        
        window_layout.addWidget(QLabel("Window Size:"), 1, 0)
        self.window_size_spin = QSpinBox()
        self.window_size_spin.setRange(1, 90)
        self.window_size_spin.setValue(14)
        window_layout.addWidget(self.window_size_spin, 1, 1)
        
        self.window_size_unit_combo = QComboBox()
        self.window_size_unit_combo.addItems(["Days", "Weeks"])
        self.window_size_unit_combo.setCurrentText("Days")
        window_layout.addWidget(self.window_size_unit_combo, 1, 2)
        
        # Add encounter/value column selection - horizontal layout
        self.encounter_id_check = QGroupBox("Use additional column for ordering")
        self.encounter_id_check.setCheckable(True)
        self.encounter_id_check.setChecked(False)
        encounter_layout = QHBoxLayout(self.encounter_id_check)
        
        encounter_layout.addWidget(QLabel("Order Column:"))
        self.encounter_id_combo = QComboBox()
        encounter_layout.addWidget(self.encounter_id_combo, 1)
        
        window_layout.addWidget(self.encounter_id_check, 2, 0, 1, 3)
        
        window_layout.setColumnStretch(1, 1)
        window_layout.setColumnStretch(2, 1)
        
        self.date_handling_stack.addWidget(window_widget)
        
        timepoint_layout.addWidget(self.date_handling_stack)
        
        # Aggregation options - horizontal layout
        agg_layout = QHBoxLayout()
        agg_layout.addWidget(QLabel("When multiple values exist, use:"))
        self.aggregation_combo = QComboBox()
        self.aggregation_combo.addItems(["First value", "Last value", "Mean", "Median", "Min", "Max", 
                                       "Closest to interval start", "Closest to interval middle", "Closest to interval end"])
        agg_layout.addWidget(self.aggregation_combo, 1)
        
        timepoint_layout.addLayout(agg_layout)
        
        middle_layout.addWidget(self.timepoint_group)
        middle_layout.addStretch()
        middle_panel.setMinimumWidth(400)
        controls_section.addWidget(middle_panel)
        
        # Add the horizontal controls section to main layout
        main_layout.addWidget(controls_section, 1)  # 1/3 of vertical space
        
        # Results preview area - full width
        results_group = QGroupBox("Transformation Preview")
        results_layout = QVBoxLayout(results_group)
        
        # Active transformation label and save controls - horizontal layout
        header_layout = QHBoxLayout()
        
        self.active_transform_label = QLabel("No transformation applied")
        self.active_transform_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(self.active_transform_label)
        
        header_layout.addStretch()
        
        # Save transformed dataset inline with header
        header_layout.addWidget(QLabel("Save As:"))
        
        self.save_name_input = QLineEdit()
        self.save_name_input.setPlaceholderText("Enter name for transformed dataset")
        self.save_name_input.setMinimumWidth(200)
        header_layout.addWidget(self.save_name_input)
        
        save_button = QPushButton("Save")
        save_button.setIcon(QIcon.fromTheme("document-save"))
        save_button.clicked.connect(self.save_transformation)
        header_layout.addWidget(save_button)
        
        results_layout.addLayout(header_layout)
        
        # Add preview dataset display
        self.preview_display = DataFrameDisplay()
        results_layout.addWidget(self.preview_display)
        
        main_layout.addWidget(results_group, 2)  # 2/3 of vertical space for results
        
        # Add status bar
        main_layout.addWidget(self.status_bar)
        
        # Add dialog buttons at bottom
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)
        
        self.status_bar.showMessage("Ready")
    
    def set_current_dataset(self, name, dataframe, format_type="unknown"):
        """Set the current dataset for transformation"""
        self.current_name = name
        self.current_dataframe = dataframe
        self.current_format = format_type
        
        # Update UI
        self.dataset_label.setText(f"Dataset: {name}")
        self.format_label.setText(format_type.capitalize())
        
        # Set save name default
        self.save_name_input.setText(f"{name}_transformed")
        
        # Update column combos
        self.update_column_combos()
        
        # Auto-select appropriate transformation based on format
        if format_type == "longitudinal":
            self.long_to_wide_radio.setChecked(True)
        elif format_type == "columnar":
            self.wide_to_long_radio.setChecked(True)
        else:
            # Default to long → wide
            self.long_to_wide_radio.setChecked(True)
    
    def update_column_combos(self):
        """Update column selection dropdowns"""
        if self.current_dataframe is None:
            return
            
        # Clear combos
        self.subject_id_combo.clear()
        self.timepoint_column_combo.clear()
        self.value_column_combo.clear()
        self.encounter_id_combo.clear()  # Clear the encounter ID combo
        
        # Add columns to combos
        for column in self.current_dataframe.columns:
            self.subject_id_combo.addItem(column)
            self.timepoint_column_combo.addItem(column)
            self.value_column_combo.addItem(column)
            self.encounter_id_combo.addItem(column)
            
        # Try to auto-select appropriate defaults
        self.auto_select_defaults()
    
    def auto_select_defaults(self):
        """Try to auto-select appropriate default columns"""
        if self.current_dataframe is None:
            return
            
        # Try to find subject ID column
        subject_id_cols = [col for col in self.current_dataframe.columns 
                           if 'subject' in col.lower() or 'patient' in col.lower() or col.lower().endswith('_id')]
        if subject_id_cols:
            self.subject_id_combo.setCurrentText(subject_id_cols[0])
            
        # Try to find visit/timepoint column
        visit_cols = [col for col in self.current_dataframe.columns 
                      if col.lower() in ['visit', 'timepoint', 'visit_id', 'visit_date']]
        if visit_cols:
            self.timepoint_column_combo.setCurrentText(visit_cols[0])
            
        # Try to find encounter column
        encounter_cols = [col for col in self.current_dataframe.columns 
                         if 'encounter' in col.lower() or 'event' in col.lower()]
        if encounter_cols:
            self.encounter_id_combo.setCurrentText(encounter_cols[0])
            self.encounter_id_check.setChecked(True)
        else:
            self.encounter_id_check.setChecked(False)
            
        # For value column, select the first numeric column that's not an ID
        value_cols = [col for col in self.current_dataframe.columns 
                     if pd.api.types.is_numeric_dtype(self.current_dataframe[col]) 
                     and not col.lower().endswith('_id')]
        if value_cols:
            self.value_column_combo.setCurrentText(value_cols[0])
    
    def update_status(self, message):
        """Update status bar with message"""
        self.status_bar.showMessage(message)
    
    def update_active_transform(self):
        """Update UI based on selected transformation direction"""
        if self.long_to_wide_radio.isChecked():
            self.current_transform = "long_to_wide"
            self.timepoint_column_combo.setEnabled(True)
            self.value_column_combo.setEnabled(True)
        else:
            self.current_transform = "wide_to_long"
            self.timepoint_column_combo.setEnabled(False)
            self.value_column_combo.setEnabled(False)
            
        # Update transformation label
        self.active_transform_label.setText(f"Selected transformation: {'Long → Wide' if self.current_transform == 'long_to_wide' else 'Wide → Long'}")
    
    def apply_transformation(self):
        """Apply the selected transformation"""
        if self.current_dataframe is None:
            QMessageBox.warning(self, "Error", "No dataset selected")
            return
            
        if self.current_transform == "long_to_wide":
            self.convert_long_to_wide()
        elif self.current_transform == "wide_to_long":
            self.convert_wide_to_long()
        else:
            QMessageBox.warning(self, "Error", "Please select a transformation direction")
    
    @asyncSlot()
    async def auto_detect_timepoints(self):
        """Use Gemini LLM to identify timepoint and subject ID columns"""
        self.update_status("Detecting timepoint columns...")
        if self.current_dataframe is None:
            QMessageBox.warning(self, "Error", "No dataset selected")
            return
            
        # Create a prompt for Gemini API
        prompt = f"""
        I need to identify timepoint-related columns in this dataset.
        
        Dataset Columns: {list(self.current_dataframe.columns)}
        Dataset Sample (first 3 rows):
        {self.current_dataframe.head(3).to_string()}
        
        Please identify:
        1. The column that likely represents subject/patient IDs
        2. The column that likely represents timepoints/visits
        3. A column that contains values of interest (numeric data)
        
        Return your response as a JSON object with the following structure:
        {{
            "subject_id_column": "column_name",
            "timepoint_column": "column_name",
            "value_column": "column_name",
            "format_type": "longitudinal|columnar|normalized",
            "explanation": "brief explanation of your recommendation"
        }}
        """
        
        # Show loading message
        QMessageBox.information(self, "Processing", "Analyzing dataset with AI. This may take a moment...")
        
        try:
            # Call Gemini API asynchronously
            response = await call_llm_async(prompt)
            
            # Parse the JSON response
            # Extract JSON from the response (in case the API returns additional text)
            json_str = re.search(r'({.*})', response, re.DOTALL)
            if json_str:
                result = json.loads(json_str.group(1))
                
                # Set the detected values in the UI
                subject_id = result.get("subject_id_column")
                timepoint = result.get("timepoint_column")
                value = result.get("value_column")
                format_type = result.get("format_type")
                explanation = result.get("explanation")
                
                # Update UI with detected values
                if subject_id in self.current_dataframe.columns:
                    self.subject_id_combo.setCurrentText(subject_id)
                if timepoint in self.current_dataframe.columns:
                    self.timepoint_column_combo.setCurrentText(timepoint)
                if value in self.current_dataframe.columns:
                    self.value_column_combo.setCurrentText(value)
                    
                # Update format detection
                if format_type:
                    self.format_label.setText(format_type.capitalize())
                    self.current_format = format_type
                    
                    # Select appropriate radio button
                    if format_type == "longitudinal":
                        self.long_to_wide_radio.setChecked(True)
                    elif format_type == "columnar":
                        self.wide_to_long_radio.setChecked(True)
                    
                # Show explanation
                QMessageBox.information(self, "AI Recommendation", 
                                       f"Detected Columns:\n"
                                       f"Subject ID: {subject_id}\n"
                                       f"Timepoint: {timepoint}\n"
                                       f"Value: {value}\n"
                                       f"Format: {format_type}\n\n"
                                       f"Explanation: {explanation}")
            else:
                QMessageBox.warning(self, "Error", "Could not parse AI response")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"AI analysis failed: {str(e)}")
        
    def convert_long_to_wide(self):
        """Convert from long to wide format using pandas pivot"""
        if self.current_dataframe is None:
            QMessageBox.warning(self, "Error", "No dataset selected")
            return
            
        # Get necessary columns from UI
        subject_id_col = self.subject_id_combo.currentText()
        timepoint_col = self.timepoint_column_combo.currentText()
        value_col = self.value_column_combo.currentText()
        
        if not all([subject_id_col, timepoint_col, value_col]):
            QMessageBox.warning(self, "Error", "Please select all required columns")
            return
            
        try:
            self.update_status("Converting from long to wide format...")
            
            # Create a working copy of the dataframe
            df = self.current_dataframe.copy()
            
            # Check if timepoint column is a date type and needs special handling
            is_date_column = False
            is_numeric_timepoint = False
            date_handling = 0  # Initialize with default value to avoid UnboundLocalError
            
            try:
                # First, check if it's a numeric column (like 1, 2, 3)
                if pd.api.types.is_numeric_dtype(df[timepoint_col]):
                    is_numeric_timepoint = True
                # Check if column is already datetime type
                elif pd.api.types.is_datetime64_any_dtype(df[timepoint_col]):
                    is_date_column = True
                else:
                    # Try to convert to datetime
                    try:
                        pd.to_datetime(df[timepoint_col], errors='raise')
                        is_date_column = True
                    except:
                        is_date_column = False
            except:
                is_date_column = False
            
            # Process the timepoint column if it's a date
            if is_date_column and self.timepoint_group.isEnabled():
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df[timepoint_col]):
                    df[timepoint_col] = pd.to_datetime(df[timepoint_col], errors='coerce')
                
                # Get selected date handling method
                date_handling = self.date_handling_combo.currentIndex()
                
                if date_handling == 1:  # Group by intervals
                    # Create bins based on interval settings
                    interval_value = self.interval_value_spin.value()
                    interval_unit = self.interval_unit_combo.currentText().lower()
                    
                    # Create a new timepoint column based on interval
                    if interval_unit == 'days':
                        freq = f'{interval_value}D'
                    elif interval_unit == 'weeks':
                        freq = f'{interval_value}W'
                    elif interval_unit == 'months':
                        freq = f'{interval_value}M'
                    else:  # years
                        freq = f'{interval_value}Y'
                    
                    # Create a binned column rounded to the specified frequency
                    df['timepoint_binned'] = df[timepoint_col].dt.to_period(freq).dt.start_time
                    # Use the binned column for pivoting
                    timepoint_col = 'timepoint_binned'
                    
                elif date_handling == 2:  # Binned timepoints
                    try:
                        start_date = pd.to_datetime(self.bin_start_edit.text())
                        end_date = pd.to_datetime(self.bin_end_edit.text())
                        bin_count = self.bin_count_spin.value()
                        
                        # Create bins
                        bins = pd.date_range(start=start_date, end=end_date, periods=bin_count + 1)
                        
                        # Cut the dates into bins
                        df['timepoint_binned'] = pd.cut(df[timepoint_col], bins=bins, labels=False)
                        # Label bins more nicely for better column names
                        df['timepoint_binned'] = df['timepoint_binned'].apply(
                            lambda x: f"Bin{int(x) + 1}" if pd.notnull(x) else "Other")
                        
                        # Use the binned column for pivoting
                        timepoint_col = 'timepoint_binned'
                    except Exception as e:
                        QMessageBox.warning(self, "Error", f"Failed to create date bins: {str(e)}")
                        return
            
                elif date_handling == 3:  # Regular intervals with window (NEW)
                    try:
                        # Get the window parameters
                        interval_value = self.window_interval_spin.value()
                        interval_unit = self.window_unit_combo.currentText().lower()
                        window_size = self.window_size_spin.value()
                        window_unit = self.window_size_unit_combo.currentText().lower()
                        
                        # Convert units to days for calculation
                        if interval_unit == 'weeks':
                            interval_days = interval_value * 7
                        elif interval_unit == 'months':
                            interval_days = interval_value * 30  # Approximate
                        else:
                            interval_days = interval_value
                            
                        if window_unit == 'weeks':
                            window_days = window_size * 7
                        else:
                            window_days = window_size
                        
                        # Use encounter column for ordering within windows if selected
                        use_encounter = self.encounter_id_check.isChecked()
                        encounter_col = self.encounter_id_combo.currentText() if use_encounter else None
                        
                        # Find the min and max dates for the dataset
                        min_date = df[timepoint_col].min()
                        max_date = df[timepoint_col].max()
                        
                        # Create the regular interval timepoints
                        interval_points = pd.date_range(
                            start=min_date, 
                            end=max_date, 
                            freq=f"{interval_days}D"
                        )
                        
                        # Function to find the closest interval point for each date
                        def assign_interval(date):
                            deltas = [(date - point).total_seconds() for point in interval_points]
                            closest_idx = min(range(len(deltas)), key=lambda i: abs(deltas[i]))
                            return interval_points[closest_idx]
                        
                        # Group by subject and find values within windows
                        result_rows = []
                        
                        # Process each subject
                        for subject, subject_data in df.groupby(subject_id_col):
                            # For each interval point
                            for interval_point in interval_points:
                                start_window = interval_point - pd.Timedelta(days=window_days/2)
                                end_window = interval_point + pd.Timedelta(days=window_days/2)
                                
                                # Find rows within the window
                                window_data = subject_data[
                                    (subject_data[timepoint_col] >= start_window) & 
                                    (subject_data[timepoint_col] <= end_window)
                                ]
                                
                                if not window_data.empty:
                                    # Apply aggregation based on selected method
                                    agg_method = self.aggregation_combo.currentText().lower()
                                    
                                    if use_encounter and encounter_col in window_data.columns:
                                        # Sort by encounter ID if available
                                        window_data = window_data.sort_values(by=encounter_col)
                                    
                                    # Apply aggregation
                                    if agg_method == 'first value':
                                        value = window_data.iloc[0][value_col]
                                    elif agg_method == 'last value':
                                        value = window_data.iloc[-1][value_col]
                                    elif agg_method == 'mean':
                                        value = window_data[value_col].mean()
                                    elif agg_method == 'median':
                                        value = window_data[value_col].median()
                                    elif agg_method == 'min':
                                        value = window_data[value_col].min()
                                    elif agg_method == 'max':
                                        value = window_data[value_col].max()
                                    elif agg_method == 'closest to interval start':
                                        window_data['dist_to_start'] = abs(window_data[timepoint_col] - start_window)
                                        value = window_data.loc[window_data['dist_to_start'].idxmin()][value_col]
                                    elif agg_method == 'closest to interval middle':
                                        window_data['dist_to_middle'] = abs(window_data[timepoint_col] - interval_point)
                                        value = window_data.loc[window_data['dist_to_middle'].idxmin()][value_col]
                                    elif agg_method == 'closest to interval end':
                                        window_data['dist_to_end'] = abs(window_data[timepoint_col] - end_window)
                                        value = window_data.loc[window_data['dist_to_end'].idxmin()][value_col]
                                    else:
                                        value = window_data.iloc[0][value_col]  # Default to first value
                                    
                                    # Add the result row
                                    result_rows.append({
                                        subject_id_col: subject,
                                        'interval_point': interval_point,
                                        'value': value
                                    })
                        
                        # Create a new dataframe with the windowed data
                        if result_rows:
                            df = pd.DataFrame(result_rows)
                            timepoint_col = 'interval_point'
                        else:
                            QMessageBox.warning(self, "Error", "No data found within specified windows")
                            return
                                
                    except Exception as e:
                        QMessageBox.warning(self, "Error", f"Failed to create interval windows: {str(e)}")
                        return
            
            # Handle aggregation for duplicate timepoints (only if not already handled by window processing)
            if date_handling != 3 or not is_date_column:
                aggregation_method = self.aggregation_combo.currentText().lower()
                
                # Check for duplicates in the combination of subject_id and timepoint
                has_duplicates = df.duplicated(subset=[subject_id_col, timepoint_col], keep=False).any()
                
                if has_duplicates:
                    # Apply aggregation based on selected method
                    if aggregation_method == 'first value':
                        df = df.sort_values(by=[subject_id_col, timepoint_col]).drop_duplicates(
                            subset=[subject_id_col, timepoint_col], keep='first')
                    elif aggregation_method == 'last value':
                        df = df.sort_values(by=[subject_id_col, timepoint_col]).drop_duplicates(
                            subset=[subject_id_col, timepoint_col], keep='last')
                    else:
                        # Group by subject_id and timepoint, then aggregate the value column
                        if aggregation_method == 'mean':
                            agg_func = 'mean'
                        elif aggregation_method == 'median':
                            agg_func = 'median'
                        elif aggregation_method == 'min':
                            agg_func = 'min'
                        elif aggregation_method == 'max':
                            agg_func = 'max'
                        else:
                            agg_func = 'mean'  # Default
                        
                        # Get a list of all columns except the value column
                        id_vars = [subject_id_col, timepoint_col]
                        # Only aggregate numeric value columns
                        if pd.api.types.is_numeric_dtype(df[value_col]):
                            # Create an aggregated dataframe
                            df = df.groupby(id_vars)[value_col].agg(agg_func).reset_index()
                        else:
                            # For non-numeric columns, just take the first value
                            df = df.sort_values(by=id_vars).drop_duplicates(subset=id_vars, keep='first')
            
            # Use pivot to convert long to wide
            # If the timepoint column is numeric (like 1, 2, 3), convert it to string to prevent date interpretation
            if is_numeric_timepoint:
                df[timepoint_col] = df[timepoint_col].astype(str)
                
            wide_df = df.pivot(index=subject_id_col, 
                              columns=timepoint_col, 
                              values='value' if date_handling == 3 and is_date_column else value_col)
            
            # Add standard naming for columns based on the source timepoint type
            if is_date_column and self.timepoint_group.isEnabled():
                if date_handling == 0:  # Use as is
                    # Format dates for column names
                    wide_df.columns = [f"{value_col}_{col.strftime('%Y-%m-%d')}" for col in wide_df.columns]
                elif date_handling == 1 or date_handling == 3:  # Group by intervals or window
                    # Format dates for column names
                    wide_df.columns = [f"{value_col}_{col.strftime('%Y-%m-%d')}" if isinstance(col, pd.Timestamp) 
                                     else f"{value_col}_{col}" for col in wide_df.columns]
                else:  # Binned timepoints
                    wide_df.columns = [f"{value_col}_{col}" for col in wide_df.columns]
            elif is_numeric_timepoint:
                # For numeric timepoints, use original numeric value for column naming
                wide_df.columns = [f"{value_col}_visit{col}" for col in wide_df.columns]
            else:
                # For any other type of column
                wide_df.columns = [f"{value_col}_{col}" for col in wide_df.columns]
            
            # Reset index to make subject_id a regular column
            wide_df.reset_index(inplace=True)
            
            # Display the result
            self.preview_display.display_dataframe(wide_df)
            
            # Set default save name
            self.save_name_input.setText(f"{self.current_name}_wide")
            
            # Store the result for saving later
            self._preview_df = wide_df
            
            # Update transformation label
            agg_method = self.aggregation_combo.currentText().lower()
            if date_handling == 3 and is_date_column:
                window_desc = f"every {self.window_interval_spin.value()} {self.window_unit_combo.currentText().lower()}"
                self.active_transform_label.setText(f"Applied transformation: Long → Wide (window intervals {window_desc}, {agg_method})")
            else:
                self.active_transform_label.setText(f"Applied transformation: Long → Wide (with {agg_method})")
            
            # Update status
            self.update_status(f"Converted from long to wide format - {len(wide_df)} rows")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Conversion failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def convert_wide_to_long(self):
        """Convert from wide to long format using pandas melt"""
        if self.current_dataframe is None:
            QMessageBox.warning(self, "Error", "No dataset selected")
            return
            
        # Get subject ID column from UI
        subject_id_col = self.subject_id_combo.currentText()
        
        if not subject_id_col:
            QMessageBox.warning(self, "Error", "Please select a subject ID column")
            return
            
        try:
            self.update_status("Converting from wide to long format...")
            
            # Identify value columns with timepoints (using regex pattern)
            value_timepoint_cols = [col for col in self.current_dataframe.columns 
                                   if re.search(r'_visit\d+|_v\d+|_timepoint\d+|_t\d+', col.lower())]
            
            if not value_timepoint_cols:
                QMessageBox.warning(self, "Error", "No timepoint columns detected in this dataset")
                return
                
            # Basic melt to long format
            long_df = pd.melt(self.current_dataframe, id_vars=[subject_id_col], 
                             value_vars=value_timepoint_cols, 
                             var_name='variable', value_name='value')
            
            # Extract measure and visit from variable column
            long_df['measure'] = long_df['variable'].apply(
                lambda x: re.search(r'(.+)_(visit|v|timepoint|t)(\d+)', x.lower()).group(1))
            long_df['visit'] = long_df['variable'].apply(
                lambda x: int(re.search(r'(.+)_(visit|v|timepoint|t)(\d+)', x.lower()).group(3)))
            
            # Drop the variable column
            long_df.drop('variable', axis=1, inplace=True)
            
            # Display the result
            self.preview_display.display_dataframe(long_df)
            
            # Set default save name
            self.save_name_input.setText(f"{self.current_name}_long")
            
            # Store the result for saving later
            self._preview_df = long_df
            
            # Update transformation label
            self.active_transform_label.setText(f"Applied transformation: Wide → Long")
            
            # Update status
            self.update_status(f"Converted from wide to long format - {len(long_df)} rows")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Conversion failed: {str(e)}")
    
    def save_transformation(self):
        """Save the transformed dataset"""
        new_name = self.save_name_input.text()
        
        if not new_name:
            QMessageBox.warning(self, "Error", "Please enter a name for the transformed dataset")
            return
            
        if not self._preview_df is not None:
            QMessageBox.warning(self, "Error", "No transformation to save")
            return
            
        # Emit signal to add this as a new dataset
        self.transformation_applied.emit(new_name, self._preview_df)
        
        QMessageBox.information(self, "Success", f"Dataset '{new_name}' saved successfully")
        self.update_status(f"Transformation saved as '{new_name}'")
    
    def on_timepoint_column_changed(self, column_name):
        """Handle timepoint column selection change"""
        if not column_name or self.current_dataframe is None:
            return
            
        # Check if selected column is a date/time type
        try:
            # Try to determine if this is a date column
            is_date = False
            col_data = self.current_dataframe[column_name]
            
            # Check if it's already a datetime type
            if pd.api.types.is_datetime64_any_dtype(col_data):
                is_date = True
            else:
                # Try to convert to datetime if it's a string that looks like a date
                try:
                    pd.to_datetime(col_data, errors='raise')
                    is_date = True
                except:
                    is_date = False
            
            # Enable/disable date handling options based on column type
            self.timepoint_group.setEnabled(is_date)
            
            # If we have a date column, try to set appropriate defaults for binning
            if is_date:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(col_data):
                    date_col = pd.to_datetime(col_data, errors='coerce')
                else:
                    date_col = col_data
                
                # Set default bin start/end dates if valid dates exist
                valid_dates = date_col.dropna()
                if not valid_dates.empty:
                    min_date = valid_dates.min()
                    max_date = valid_dates.max()
                    
                    # Format as strings for the UI
                    if isinstance(min_date, pd.Timestamp):
                        self.bin_start_edit.setText(min_date.strftime('%Y-%m-%d'))
                    if isinstance(max_date, pd.Timestamp):
                        self.bin_end_edit.setText(max_date.strftime('%Y-%m-%d'))
        except Exception as e:
            # In case of error, disable date handling
            self.timepoint_group.setEnabled(False)
            print(f"Error checking timepoint column: {str(e)}")
    
    def on_date_handling_changed(self, index):
        """Update UI based on date handling selection"""
        self.date_handling_stack.setCurrentIndex(index)


class JoinTransformerDialog(QDialog):
    """Dialog for joining multiple datasets"""
    
    transformation_applied = pyqtSignal(str, object)  # Signal emitted when a transformation is applied
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Multi-Source Join Transformer")
        self.resize(800, 700)
        
        # Set dialog properties
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        
        # Store references
        self.dataframes = {}
        self._preview_df = None
        
        # Create a status bar for feedback
        self.status_bar = QStatusBar(self)
        
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)
        
        # Create a splitter for better resizing
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.setChildrenCollapsible(False)
        
        # Top panel for controls
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        # Info and explanation
        info_group = QGroupBox("Join Information")
        info_layout = QVBoxLayout(info_group)
        
        # Add helper text explaining joins - removed background color
        join_help = QLabel(
            "<b>Joining Datasets:</b><br>"
            "Join operations combine two datasets based on matching key columns. "
            "Different join types determine which rows are included in the result.<br><br>"
            "<b>Join Types:</b><br>"
            "<b>Inner Join:</b> Includes only rows with matching keys in both datasets<br>"
            "<b>Left Join:</b> Includes all rows from the primary dataset, matching rows from secondary<br>"
            "<b>Right Join:</b> Includes all rows from the secondary dataset, matching rows from primary<br>"
            "<b>Outer Join:</b> Includes all rows from both datasets"
        )
        join_help.setStyleSheet("padding: 5px; border-radius: 3px;")  # Removed colors
        join_help.setWordWrap(True)
        info_layout.addWidget(join_help)
        
        top_layout.addWidget(info_group)
        
        # Source selection
        source_group = QGroupBox("Data Sources")
        source_layout = QFormLayout(source_group)
        
        # Primary source
        primary_layout = QHBoxLayout()
        primary_layout.addWidget(QLabel("Primary:"))
        
        self.primary_source_combo = QComboBox()
        self.primary_source_combo.setMinimumWidth(180)
        primary_layout.addWidget(self.primary_source_combo, 1)
        
        source_layout.addRow(primary_layout)
        
        # Secondary source
        secondary_layout = QHBoxLayout()
        secondary_layout.addWidget(QLabel("Secondary:"))
        
        self.secondary_source_combo = QComboBox()
        self.secondary_source_combo.setMinimumWidth(180)
        secondary_layout.addWidget(self.secondary_source_combo, 1)
        
        source_layout.addRow(secondary_layout)
        
        # Join keys
        keys_group = QGroupBox("Join Keys")
        keys_layout = QFormLayout(keys_group)
        
        # Primary key
        primary_key_layout = QHBoxLayout()
        primary_key_layout.addWidget(QLabel("Primary Key:"))
        
        self.primary_key_combo = QComboBox()
        self.primary_key_combo.setMinimumWidth(180)
        primary_key_layout.addWidget(self.primary_key_combo, 1)
        
        keys_layout.addRow(primary_key_layout)
        
        # Secondary key
        secondary_key_layout = QHBoxLayout()
        secondary_key_layout.addWidget(QLabel("Secondary Key:"))
        
        self.secondary_key_combo = QComboBox()
        self.secondary_key_combo.setMinimumWidth(180)
        secondary_key_layout.addWidget(self.secondary_key_combo, 1)
        
        keys_layout.addRow(secondary_key_layout)
        
        # Join type
        join_type_layout = QHBoxLayout()
        join_type_layout.addWidget(QLabel("Join Type:"))
        
        self.join_type_combo = QComboBox()
        self.join_type_combo.addItems(["Inner", "Left", "Right", "Outer"])
        self.join_type_combo.setMinimumWidth(120)
        join_type_layout.addWidget(self.join_type_combo, 1)
        
        keys_layout.addRow(join_type_layout)
        
        # AI detect join button
        ai_layout = QHBoxLayout()
        auto_join_button = QPushButton("Auto-Detect Join Keys (AI)")
        auto_join_button.setIcon(QIcon.fromTheme("system-search"))
        auto_join_button.clicked.connect(self.auto_detect_join_keys)
        ai_layout.addWidget(auto_join_button)
        ai_layout.addStretch()
        
        keys_layout.addRow("", ai_layout)
        
        # Add the source and keys groups
        top_layout.addWidget(source_group)
        top_layout.addWidget(keys_group)
        
        # Join button
        join_button = QPushButton("Perform Join")
        join_button.setIcon(QIcon.fromTheme("view-refresh"))
        join_button.clicked.connect(self.perform_join)
        top_layout.addWidget(join_button)
        
        main_splitter.addWidget(top_panel)
        
        # Results display in bottom panel
        bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        
        results_group = QGroupBox("Join Results")
        results_layout = QVBoxLayout(results_group)
        
        # Active join label
        self.active_join_label = QLabel("No join performed")
        self.active_join_label.setStyleSheet("font-weight: bold;")  # Removed color
        results_layout.addWidget(self.active_join_label)
        
        # Add preview dataset display
        self.preview_display = DataFrameDisplay()
        results_layout.addWidget(self.preview_display)
        
        # Save transformed dataset
        save_layout = QHBoxLayout()
        save_layout.addWidget(QLabel("Save As:"))
        
        self.save_name_input = QLineEdit()
        self.save_name_input.setPlaceholderText("Enter name for joined dataset")
        save_layout.addWidget(self.save_name_input, 1)
        
        save_button = QPushButton("Save Join Result")
        save_button.setIcon(QIcon.fromTheme("document-save"))
        save_button.clicked.connect(self.save_transformation)
        save_layout.addWidget(save_button)
        
        results_layout.addLayout(save_layout)
        
        bottom_layout.addWidget(results_group)
        main_splitter.addWidget(bottom_panel)
        
        # Set reasonable default sizes for the splitter
        main_splitter.setSizes([350, 350])
        
        main_layout.addWidget(main_splitter, 1)
        
        # Add status bar
        main_layout.addWidget(self.status_bar)
        
        # Add dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)
        
        self.status_bar.showMessage("Ready")
        
        # Connect signals for updating key combos
        self.primary_source_combo.currentTextChanged.connect(self.update_primary_keys)
        self.secondary_source_combo.currentTextChanged.connect(self.update_secondary_keys)
    
    def set_available_sources(self, sources_dict):
        """Update available sources for joining"""
        self.dataframes = sources_dict
        
        # Update source selection dropdowns
        self.primary_source_combo.clear()
        self.secondary_source_combo.clear()
        
        for name in sources_dict.keys():
            self.primary_source_combo.addItem(name)
            self.secondary_source_combo.addItem(name)
            
        # Initialize key combos if we have sources
        if self.primary_source_combo.count() > 0:
            self.update_primary_keys(self.primary_source_combo.currentText())
        if self.secondary_source_combo.count() > 0:
            self.update_secondary_keys(self.secondary_source_combo.currentText())
            
        # Set default save name if both sources are selected
        if self.primary_source_combo.count() > 0 and self.secondary_source_combo.count() > 1:
            primary = self.primary_source_combo.currentText()
            secondary = self.secondary_source_combo.currentText()
            if primary != secondary:
                self.save_name_input.setText(f"{primary}_{secondary}_joined")
    
    def update_primary_keys(self, source_name):
        """Update primary source key dropdown"""
        if not source_name or source_name not in self.dataframes:
            return
            
        self.primary_key_combo.clear()
        for column in self.dataframes[source_name].columns:
            self.primary_key_combo.addItem(column)
            
        # Set default save name
        if self.secondary_source_combo.currentText():
            primary = source_name
            secondary = self.secondary_source_combo.currentText()
            if primary != secondary:
                self.save_name_input.setText(f"{primary}_{secondary}_joined")
    
    def update_secondary_keys(self, source_name):
        """Update secondary source key dropdown"""
        if not source_name or source_name not in self.dataframes:
            return
            
        self.secondary_key_combo.clear()
        for column in self.dataframes[source_name].columns:
            self.secondary_key_combo.addItem(column)
            
        # Set default save name
        if self.primary_source_combo.currentText():
            primary = self.primary_source_combo.currentText()
            secondary = source_name
            if primary != secondary:
                self.save_name_input.setText(f"{primary}_{secondary}_joined")
    
    def update_status(self, message):
        """Update status bar with message"""
        self.status_bar.showMessage(message)
    
    @asyncSlot()
    async def auto_detect_join_keys(self):
        """Use Gemini LLM to auto-detect optimal join keys between two datasets"""
        self.update_status("Detecting join keys...")
        primary_source = self.primary_source_combo.currentText()
        secondary_source = self.secondary_source_combo.currentText()
        
        if not primary_source or not secondary_source or primary_source == secondary_source:
            QMessageBox.warning(self, "Error", "Please select two different data sources")
            return
            
        # Get the column information for both datasets
        primary_df = self.dataframes.get(primary_source)
        secondary_df = self.dataframes.get(secondary_source)
        
        if primary_df is None or secondary_df is None:
            QMessageBox.warning(self, "Error", "One or both selected datasets are not available")
            return
            
        # Create a prompt for Gemini API
        prompt = f"""
        I need to identify the best join keys between two datasets. 
        
        Dataset 1 Columns: {list(primary_df.columns)}
        Dataset 1 Sample (first 3 rows):
        {primary_df.head(3).to_string()}
        
        Dataset 2 Columns: {list(secondary_df.columns)}
        Dataset 2 Sample (first 3 rows):
        {secondary_df.head(3).to_string()}
        
        Please identify the best column(s) to join these datasets on and recommend a join type (inner, left, right, outer).
        Return your response as a JSON object with the following structure:
        {{
            "primary_key": "column_name",
            "secondary_key": "column_name",
            "join_type": "inner|left|right|outer",
            "explanation": "brief explanation of your recommendation"
        }}
        """
        
        # Show loading message
        QMessageBox.information(self, "Processing", "Analyzing datasets with AI. This may take a moment...")
        
        try:
            # Call Gemini API asynchronously
            response = await call_llm_async(prompt)
            
            # Parse the JSON response
            # Extract JSON from the response (in case the API returns additional text)
            json_str = re.search(r'({.*})', response, re.DOTALL)
            if json_str:
                result = json.loads(json_str.group(1))
                
                # Set the detected values in the UI
                primary_key = result.get("primary_key")
                secondary_key = result.get("secondary_key")
                join_type = result.get("join_type")
                explanation = result.get("explanation")
                
                # Update UI with detected values
                if primary_key in primary_df.columns:
                    self.primary_key_combo.setCurrentText(primary_key)
                if secondary_key in secondary_df.columns:
                    self.secondary_key_combo.setCurrentText(secondary_key)
                if join_type:
                    self.join_type_combo.setCurrentText(join_type.capitalize())
                    
                # Show explanation
                QMessageBox.information(self, "AI Recommendation", 
                                       f"Recommended Join:\n"
                                       f"Primary Key: {primary_key}\n"
                                       f"Secondary Key: {secondary_key}\n"
                                       f"Join Type: {join_type}\n\n"
                                       f"Explanation: {explanation}")
            else:
                QMessageBox.warning(self, "Error", "Could not parse AI response")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"AI analysis failed: {str(e)}")
    
    def perform_join(self):
        """Perform a join between two dataframes"""
        # Get selected sources and join keys
        primary_source = self.primary_source_combo.currentText()
        secondary_source = self.secondary_source_combo.currentText()
        primary_key = self.primary_key_combo.currentText()
        secondary_key = self.secondary_key_combo.currentText()
        join_type = self.join_type_combo.currentText().lower()
        
        if not all([primary_source, secondary_source, primary_key, secondary_key]):
            QMessageBox.warning(self, "Error", "Please select all join parameters")
            return
            
        if primary_source == secondary_source:
            QMessageBox.warning(self, "Error", "Please select different sources for join")
            return
            
        try:
            self.update_status(f"Performing {join_type} join...")
            
            # Get the dataframes
            primary_df = self.dataframes[primary_source]
            secondary_df = self.dataframes[secondary_source]
            
            # Perform the join
            result_df = pd.merge(primary_df, secondary_df, 
                                left_on=primary_key, 
                                right_on=secondary_key, 
                                how=join_type)
            
            # Display the result
            self.preview_display.display_dataframe(result_df)
            
            # Set default save name if not already set
            if not self.save_name_input.text():
                self.save_name_input.setText(f"{primary_source}_{secondary_source}_joined")
            
            # Store the result for saving later
            self._preview_df = result_df
            
            # Update active join label
            self.active_join_label.setText(
                f"Join type: {join_type.upper()} JOIN\n"
                f"Primary: {primary_source} [{primary_key}]\n"
                f"Secondary: {secondary_source} [{secondary_key}]\n"
                f"Result: {len(result_df)} rows, {len(result_df.columns)} columns"
            )
            
            # Update status
            self.update_status(f"Join completed - {len(result_df)} rows")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Join operation failed: {str(e)}")
    
    def save_transformation(self):
        """Save the joined dataset"""
        new_name = self.save_name_input.text()
        
        if not new_name:
            QMessageBox.warning(self, "Error", "Please enter a name for the joined dataset")
            return
            
        if not self._preview_df is not None:
            QMessageBox.warning(self, "Error", "No join result to save")
            return
            
        # Emit signal to add this as a new dataset
        self.transformation_applied.emit(new_name, self._preview_df)
        
        QMessageBox.information(self, "Success", f"Dataset '{new_name}' saved successfully")
        self.update_status(f"Join result saved as '{new_name}'")


class DataSourcesWidget(QWidget):
    """
    Widget for managing data sources and connections.
    This will be used to display available data sources and allow the user to add new ones.
    Shows sources and data inline rather than in dialogs.
    """
    source_selected = pyqtSignal(str, object)  # Signal emitted when a source is selected (name, dataframe)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Sources")
        
        # Store connections and dataframes
        self.source_connections = {}
        self.dataframes = {}
        self.current_source = None
        
        self.init_ui()
        
    def init_ui(self):
        # Main layout with splitters for better space utilization
        main_layout = QVBoxLayout(self)
        
        # Create top controls area
        top_control_layout = QHBoxLayout()
        
        # Add source button
        add_button = QPushButton("Add Data Source")
        add_button.clicked.connect(self.show_source_dialog)
        top_control_layout.addWidget(add_button)
        
        # Add debug datasets button
        debug_button = QPushButton("Add Debug Datasets")
        debug_button.setIcon(QIcon.fromTheme("help-contents"))
        debug_button.clicked.connect(self.add_debug_datasets)
        top_control_layout.addWidget(debug_button)
        
        # Add data transformer button
        transform_button = QPushButton("Data Transformer")
        transform_button.clicked.connect(self.show_data_transformer)
        top_control_layout.addWidget(transform_button)
        
        main_layout.addLayout(top_control_layout)
        
        # Main horizontal splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Source list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Sources list
        sources_group = QGroupBox("Available Sources")
        sources_layout = QVBoxLayout(sources_group)
        self.sources_list = QListWidget()
        self.sources_list.itemClicked.connect(self.on_source_selected)
        sources_layout.addWidget(self.sources_list)
        
        left_layout.addWidget(sources_group)
        left_panel.setMinimumWidth(250)
        left_panel.setMaximumWidth(400)
        main_splitter.addWidget(left_panel)
        
        # Right panel: Dataset display
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Dataset display controls
        display_controls = QHBoxLayout()
        self.current_dataset_label = QLabel("No dataset selected")
        display_controls.addWidget(self.current_dataset_label)
        display_controls.addStretch()
        
        # Dataset info
        self.dataset_info_label = QLabel("")
        display_controls.addWidget(self.dataset_info_label)
        
        right_layout.addLayout(display_controls)
        
        # Create dataset display table
        self.dataset_display = DataFrameDisplay()
        self.dataset_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout.addWidget(self.dataset_display)
        
        main_splitter.addWidget(right_panel)
        
        # Set the size proportions (30% left, 70% right)
        main_splitter.setSizes([300, 700])
        
        main_layout.addWidget(main_splitter)
        
    def show_source_dialog(self):
        """Show dialog to add a new data source"""
        dialog = SourceDialog(self)
        
        if dialog.exec():
            # Get the source type and connection info from the dialog
            source_type = dialog.source_type
            connection_info = dialog.connection_info
            
            # Handle the new source connection using asyncio
            asyncio.create_task(self.handle_source_connection_wrapper(source_type, connection_info))
            
    async def handle_source_connection_wrapper(self, source_type, connection_info):
        """Wrapper to properly await the async connection handling"""
        await self.handle_source_connection(source_type, connection_info)
            
    async def handle_source_connection(self, source_type, connection_info):
        """
        Handle a new data source connection asynchronously
        """
        # Create a source connection
        if source_type == "upload" and connection_info.get('files'):
            # Handle each file as a separate source
            for file_path in connection_info['files']:
                file_name = os.path.basename(file_path)
                
                try:
                    # Create a source connection
                    connection = SourceConnection(source_type, {"files": [file_path]}, file_name)
                    
                    # Load the data
                    df = await connection.load_data()
                    
                    # Add the source to our list
                    self.add_source(file_name, connection, df)
                    
                    # Display the first one
                    if self.current_source is None:
                        self.display_dataset(file_name, df)
                        
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to load file {file_name}: {str(e)}")
                    
        elif source_type in ["sql", "sftp", "rest"]:
            # Create a unique name for the source
            source_name = f"{source_type.upper()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            try:
                # Create a source connection
                connection = SourceConnection(source_type, connection_info, source_name)
                
                # Load the data
                df = await connection.load_data()
                
                # Add the source to our list
                self.add_source(source_name, connection, df)
                
                # Display the dataset
                self.display_dataset(source_name, df)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to connect to {source_type} source: {str(e)}")
                
        else:
            QMessageBox.warning(self, "Error", f"Unsupported source type: {source_type}")
    
    def add_source(self, name, connection, dataframe):
        """
        Add a source to the list
        """
        self.source_connections[name] = connection
        self.dataframes[name] = dataframe
        self.sources_list.addItem(name)
        
        # If this is our first source, select it
        if self.current_source is None:
            self.current_source = name
            self.sources_list.setCurrentRow(0)
    
    def on_source_selected(self, item):
        """
        Handle source selection
        """
        name = item.text()
        if name in self.dataframes:
            self.display_dataset(name, self.dataframes[name])
            self.source_selected.emit(name, self.dataframes[name])
            self.current_source = name
            
    def display_dataset(self, name, dataframe):
        """
        Display the selected dataset in the table view
        """
        self.current_dataset_label.setText(f"Dataset: {name}")
        
        # Update dataset info
        rows, cols = dataframe.shape
        self.dataset_info_label.setText(f"Rows: {rows} | Columns: {cols}")
        
        # Display in table
        self.dataset_display.display_dataframe(dataframe)
        
        # Update data transformer if available
        if hasattr(self, 'data_transformer'):
            self.data_transformer.set_current_dataset(name, dataframe)
            
    def init_data_transformer(self):
        """
        Initialize the data transformer component
        """
        self.data_transformer = DataTransformer(self)  # Ensure parent is set
        
        # Apply application stylesheet to the dialog
        if self.window() and self.window().styleSheet():
            self.data_transformer.setStyleSheet(self.window().styleSheet())
        
        self.data_transformer.set_available_sources(self.dataframes)
        
        # Connect signals
        self.data_transformer.transformation_applied.connect(self.add_transformed_dataset)
        
    def add_transformed_dataset(self, name, dataframe):
        """
        Add a transformed dataset to the sources list
        """
        # Create a source connection object (using "transformed" as source type)
        connection = SourceConnection("transformed", {}, name)
        
        # Add the source to our list
        self.add_source(name, connection, dataframe)
        
        # Update the data transformer's available sources
        self.data_transformer.set_available_sources(self.dataframes)
        
    def show_data_transformer(self):
        """
        Show the data transformer dialog
        """
        if not hasattr(self, 'data_transformer'):
            self.init_data_transformer()
        
        # Apply current application stylesheet to ensure theme updates are reflected
        if self.window() and self.window().styleSheet():
            self.data_transformer.setStyleSheet(self.window().styleSheet())
        
        # Use exec() for modal dialog behavior
        self.data_transformer.exec()
        
        # After dialog is closed, update sources if needed
        # (This will refresh the sources list if any transformations were applied)
        self.data_transformer.set_available_sources(self.dataframes)

    def add_debug_datasets(self):
        """
        Generate and add sample datasets of different formats for testing
        """
        # Create longitudinal format sample
        longitudinal_df = self.create_longitudinal_sample()
        self.add_source("Debug_Longitudinal", SourceConnection("debug", {}, "Debug_Longitudinal"), longitudinal_df)
        
        # Create columnar (wide) format sample
        columnar_df = self.create_columnar_sample()
        self.add_source("Debug_Columnar", SourceConnection("debug", {}, "Debug_Columnar"), columnar_df)
        
        # Create normalized format samples with joinable keys
        patients_df, visits_df, measurements_df = self.create_normalized_samples()
        self.add_source("Debug_Normalized_Patients", SourceConnection("debug", {}, "Debug_Normalized_Patients"), patients_df)
        self.add_source("Debug_Normalized_Visits", SourceConnection("debug", {}, "Debug_Normalized_Visits"), visits_df)
        self.add_source("Debug_Normalized_Measurements", SourceConnection("debug", {}, "Debug_Normalized_Measurements"), measurements_df)
        
        # Display the first debug dataset
        self.display_dataset("Debug_Longitudinal", longitudinal_df)
        
        # Show confirmation to the user
        QMessageBox.information(self, "Debug Datasets Added", 
                               "5 debug datasets have been added:\n"
                               "- Longitudinal format\n"
                               "- Columnar (wide) format\n"
                               "- Normalized format (3 related tables with joinable keys)")
    
    def create_longitudinal_sample(self):
        """Create a sample longitudinal dataset"""
        # Create a dataframe with:
        # - subject_id: patient identifier
        # - visit: timepoint
        # - multiple measurement columns
        
        np.random.seed(42)  # For reproducible results
        subjects = [f"SUBJ_{i:03d}" for i in range(1, 21)]  # 20 subjects
        visits = [1, 2, 3]  # 3 visits per subject
        
        # Create all combinations of subjects and visits
        data = []
        start_date = date(2023, 1, 1)
        
        for subject in subjects:
            for visit in visits:
                # Create random measurements with some missing data to make it realistic
                weight = np.random.normal(70, 10) if np.random.random() > 0.1 else np.nan
                height = np.random.normal(170, 15) if np.random.random() > 0.1 else np.nan
                heart_rate = np.random.normal(75, 10) if np.random.random() > 0.1 else np.nan
                temperature = np.random.normal(36.8, 0.4) if np.random.random() > 0.1 else np.nan
                
                visit_date = start_date + timedelta(days=(visit-1)*30 + np.random.randint(0, 10))
                
                data.append({
                    'subject_id': subject,
                    'visit': visit,
                    'weight': round(weight, 1) if not np.isnan(weight) else np.nan,
                    'height': round(height, 1) if not np.isnan(height) else np.nan,
                    'heart_rate': round(heart_rate) if not np.isnan(heart_rate) else np.nan,
                    'temperature': round(temperature, 1) if not np.isnan(temperature) else np.nan,
                    'visit_date': visit_date.strftime("%Y-%m-%d")
                })
        
        return pd.DataFrame(data)
    
    def create_columnar_sample(self):
        """Create a sample columnar (wide) dataset"""
        # Start with longitudinal data
        long_df = self.create_longitudinal_sample()
        
        # Create a separate column for each visit/measurement combination
        subjects = []
        
        for subject in long_df['subject_id'].unique():
            subject_data = {'subject_id': subject}
            
            # Filter to just this subject
            subject_df = long_df[long_df['subject_id'] == subject]
            
            # For each visit, add columns for each measurement
            for _, row in subject_df.iterrows():
                visit = row['visit']
                for measure in ['weight', 'height', 'heart_rate', 'temperature']:
                    col_name = f"{measure}_visit{visit}"
                    subject_data[col_name] = row[measure]
                
                # Add visit date
                subject_data[f"visit_date_{visit}"] = row['visit_date']
            
            subjects.append(subject_data)
        
        return pd.DataFrame(subjects)
    
    def create_normalized_samples(self):
        """Create sample normalized datasets with joinable keys"""
        # Create 3 related tables:
        # 1. Patients table with demographics
        # 2. Visits table with visit information
        # 3. Measurements table with actual measurements
        
        np.random.seed(42)  # For reproducible results
        
        # Patients table
        patient_ids = [f"SUBJ_{i:03d}" for i in range(1, 21)]  # 20 subjects
        patients_data = []
        
        for patient_id in patient_ids:
            gender = np.random.choice(['M', 'F'])
            age = np.random.randint(18, 80)
            
            patients_data.append({
                'patient_id': patient_id,
                'gender': gender,
                'age': age,
                'enrollment_date': f"2023-01-{np.random.randint(1, 29):02d}"
            })
        
        patients_df = pd.DataFrame(patients_data)
        
        # Visits table
        visits_data = []
        visit_id_counter = 1
        start_date = date(2023, 1, 1)
        
        for patient_id in patient_ids:
            for visit_num in range(1, 4):  # 3 visits per patient
                visit_id = f"VISIT_{visit_id_counter:04d}"
                visit_id_counter += 1
                
                visit_date = start_date + timedelta(days=(visit_num-1)*30 + np.random.randint(0, 10))
                
                visits_data.append({
                    'visit_id': visit_id,
                    'patient_id': patient_id,
                    'visit_number': visit_num,
                    'visit_date': visit_date.strftime("%Y-%m-%d"),
                    'visit_type': np.random.choice(['Routine', 'Follow-up', 'Urgent'])
                })
        
        visits_df = pd.DataFrame(visits_data)
        
        # Measurements table
        measurements_data = []
        measurement_id_counter = 1
        
        # Create multiple measurements per visit
        for _, visit_row in visits_df.iterrows():
            visit_id = visit_row['visit_id']
            
            # Add multiple measurements for this visit
            for measure_type in ['weight', 'height', 'heart_rate', 'temperature']:
                # Generate random values based on measure type with occasional missing data
                if np.random.random() > 0.1:  # 10% chance of missing data
                    if measure_type == 'weight':
                        value = round(np.random.normal(70, 10), 1)  # weight in kg
                        unit = 'kg'
                    elif measure_type == 'height':
                        value = round(np.random.normal(170, 15), 1)  # height in cm
                        unit = 'cm'
                    elif measure_type == 'heart_rate':
                        value = round(np.random.normal(75, 10))  # heart rate bpm
                        unit = 'bpm'
                    elif measure_type == 'temperature':
                        value = round(np.random.normal(36.8, 0.4), 1)  # body temp in C
                        unit = 'C'
                    
                    measurements_data.append({
                        'measurement_id': f"MEAS_{measurement_id_counter:05d}",
                        'visit_id': visit_id,
                        'measure_type': measure_type,
                        'value': value,
                        'unit': unit,
                        'measurement_time': f"{np.random.randint(8, 18):02d}:{np.random.choice(['00', '15', '30', '45'])}"
                    })
                    measurement_id_counter += 1
        
        measurements_df = pd.DataFrame(measurements_data)
        
        return patients_df, visits_df, measurements_df
