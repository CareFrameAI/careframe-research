import sys
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLabel,
    QListWidget, QMessageBox, QLineEdit, QRadioButton, QGroupBox, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
import subprocess
import requests
import json
from websockets.protocol import State
import os
import signal
import platform
import asyncio

from server import ConnectionManager
from helpers.load_icon import load_bootstrap_icon

class NetworkSection(QWidget):
    serverStarted = pyqtSignal()
    serverStopped = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.server_process = None
        self.port = 8889  # Default port
        self.is_host_mode = True  # Default to host mode
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Group box for mode selection
        mode_group = QGroupBox("Mode Selection")
        mode_layout = QHBoxLayout()
        self.host_mode_btn = QRadioButton("Host Mode")
        self.client_mode_btn = QRadioButton("Client Mode")
        self.host_mode_btn.setChecked(True)  # Default to host mode
        mode_layout.addWidget(self.host_mode_btn)
        mode_layout.addWidget(self.client_mode_btn)
        mode_group.setLayout(mode_layout)

        # Server control section
        server_layout = QHBoxLayout()
        self.start_server_btn = QPushButton("Start Server")
        self.start_server_btn.setIcon(load_bootstrap_icon("play-circle"))
        self.stop_server_btn = QPushButton("Stop Server")
        self.stop_server_btn.setIcon(load_bootstrap_icon("stop-circle"))
        self.server_status_label = QLabel("Server Status: Stopped")
        self.port_input = QLineEdit(str(self.port))
        self.port_input.setMaximumWidth(100)
        self.port_input.setPlaceholderText("Port")
        
        server_layout.addWidget(QLabel("Port:"))
        server_layout.addWidget(self.port_input)
        server_layout.addWidget(self.start_server_btn)
        server_layout.addWidget(self.stop_server_btn)
        server_layout.addWidget(self.server_status_label)
        server_layout.addStretch()

        # Host connection section
        host_layout = QHBoxLayout()
        self.host_input = QLineEdit()
        self.host_input.setPlaceholderText("Host IP")
        self.host_port_input = QLineEdit()
        self.host_port_input.setPlaceholderText("Host Port")
        self.connect_host_btn = QPushButton("Connect to Host")
        self.connect_host_btn.setIcon(load_bootstrap_icon("plug"))
        self.topic_input = QLineEdit()  # Dedicated topic input
        self.topic_input.setPlaceholderText("Topic")
        self.subscribe_btn = QPushButton("Subscribe")
        self.subscribe_btn.setIcon(load_bootstrap_icon("bell"))
        self.unsubscribe_btn = QPushButton("Unsubscribe")
        self.unsubscribe_btn.setIcon(load_bootstrap_icon("bell-slash"))
        
        host_layout.addWidget(QLabel("Host:"))
        host_layout.addWidget(self.host_input)
        host_layout.addWidget(self.host_port_input)
        host_layout.addWidget(self.connect_host_btn)
        host_layout.addWidget(QLabel("Topic:"))
        host_layout.addWidget(self.topic_input)
        host_layout.addWidget(self.subscribe_btn)
        host_layout.addWidget(self.unsubscribe_btn)
        host_layout.addStretch()

        # Message publishing section
        publish_layout = QVBoxLayout()
        self.message_input = QTextEdit()
        self.message_input.setMaximumHeight(100)
        self.message_input.setPlaceholderText(
            "Enter message in JSON format:\n{\n  \"topic\": \"example\",\n  \"data\": \"your data\",\n  \"context\": \"your context\"\n}"
        )
        self.publish_btn = QPushButton("Publish Message")
        self.publish_btn.setIcon(load_bootstrap_icon("send"))
        
        publish_layout.addWidget(QLabel("Publish Message:"))
        publish_layout.addWidget(self.message_input)
        publish_layout.addWidget(self.publish_btn)

        # Subscribed messages display
        self.subscribed_messages = QTextEdit()
        self.subscribed_messages.setReadOnly(True)
        self.subscribed_messages.setMaximumHeight(200)

        # WebSocket section
        websocket_layout = QHBoxLayout()
        self.connect_button = QPushButton("Connect")
        self.connect_button.setIcon(load_bootstrap_icon("plug"))
        self.ping_btn = QPushButton("Send Ping")
        self.ping_btn.setIcon(load_bootstrap_icon("arrow-repeat"))
        self.ws_status_label = QLabel("WebSocket: Disconnected")
        
        websocket_layout.addWidget(self.connect_button)
        websocket_layout.addWidget(self.ping_btn)
        websocket_layout.addWidget(self.ws_status_label)
        websocket_layout.addStretch()

        # Connected clients list
        clients_layout = QHBoxLayout()
        self.clients_list = QListWidget()
        self.clients_list.setMaximumHeight(150)
        clients_layout.addWidget(QLabel("Connected Clients:"))
        clients_layout.addWidget(self.clients_list)

        # Server output
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMaximumHeight(200)

        # Add all layouts to main layout
        layout.addWidget(mode_group)
        layout.addLayout(server_layout)
        layout.addLayout(host_layout)
        layout.addLayout(publish_layout)
        layout.addLayout(websocket_layout)
        layout.addLayout(clients_layout)
        layout.addWidget(QLabel("Server Output:"))
        layout.addWidget(self.output_text)
        layout.addWidget(QLabel("Subscribed Messages:"))
        layout.addWidget(self.subscribed_messages)
        layout.addWidget(self.server_status_label)
        layout.addStretch()

        self.setLayout(layout)

        # Connect buttons to slots
        self.start_server_btn.clicked.connect(self.start_server)
        self.stop_server_btn.clicked.connect(self.stop_server)
        self.connect_button.clicked.connect(self.connect_to_host)
        self.connect_host_btn.clicked.connect(self.connect_to_host)
        self.ping_btn.clicked.connect(self.send_ping)
        self.subscribe_btn.clicked.connect(self.subscribe_to_topic)
        self.unsubscribe_btn.clicked.connect(self.unsubscribe_from_topic)
        self.publish_btn.clicked.connect(self.send_message)

        # Connect mode buttons
        self.host_mode_btn.toggled.connect(self.set_host_mode)
        self.client_mode_btn.toggled.connect(self.set_client_mode)

        # Initially disable server controls (enable after mode selection)
        self.start_server_btn.setEnabled(False)
        self.stop_server_btn.setEnabled(False)

    def set_host_mode(self):
        """Enable host mode features"""
        if self.host_mode_btn.isChecked():
            self.is_host_mode = True
            self.client_mode_btn.setChecked(False)
            self.start_server_btn.setEnabled(True)
            self.stop_server_btn.setEnabled(False)  # Disable until server starts
            self.host_input.setEnabled(False)
            self.host_port_input.setEnabled(False)
            self.connect_host_btn.setEnabled(False)
            self.topic_input.setEnabled(True)
            self.subscribe_btn.setEnabled(True)  # Always enabled in host mode
            self.server_status_label.setText("Host Mode")
        else:
            self.set_client_mode()

    def set_client_mode(self):
        """Enable client mode features"""
        if self.client_mode_btn.isChecked():
            self.is_host_mode = False
            self.host_mode_btn.setChecked(False)
            self.start_server_btn.setEnabled(False)
            self.stop_server_btn.setEnabled(False)
            self.host_input.setEnabled(True)
            self.host_port_input.setEnabled(True)
            self.connect_host_btn.setEnabled(True)
            self.topic_input.setEnabled(True)
            self.subscribe_btn.setEnabled(False)  # Disable until connected
            self.server_status_label.setText("Client Mode")
        else:
            self.host_input.setEnabled(False)
            self.host_port_input.setEnabled(False)
            self.connect_host_btn.setEnabled(False)
            self.topic_input.setEnabled(False)
            self.subscribe_btn.setEnabled(False)

    def start_server(self):
        """Start the FastAPI server using uvicorn"""
        try:
            # Get port from input
            try:
                self.port = int(self.port_input.text())
            except ValueError:
                QMessageBox.warning(self, "Invalid Port", "Please enter a valid port number")
                return

            # First check if server is already running
            try:
                response = requests.get(f"http://127.0.0.1:{self.port}/health")
                if response.status_code == 200:
                    QMessageBox.warning(self, "Server Status", f"Server is already running on port {self.port}")
                    self.start_server_btn.setEnabled(False)
                    self.stop_server_btn.setEnabled(True)
                    self.server_status_label.setText("Server Status: Ready")
                    return
            except requests.ConnectionError:
                pass
            
            # Get the absolute path to server.py
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            server_path = os.path.join(current_dir, "server.py")
            
            # Verify server.py exists
            if not os.path.exists(server_path):
                QMessageBox.critical(self, "Server Error", f"Cannot find server.py at {server_path}")
                return
            
            self.output_text.append(f"Starting server from: {current_dir}")
            
            # Update server start command to use 0.0.0.0
            if platform.system() == 'Windows':
                self.server_process = subprocess.Popen(
                    ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", str(self.port), "--reload"],
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=current_dir
                )
            else:
                self.server_process = subprocess.Popen(
                    ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", str(self.port), "--reload"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    cwd=current_dir
                )
            
            # Update UI immediately
            self.start_server_btn.setEnabled(False)
            self.stop_server_btn.setEnabled(True)
            self.server_status_label.setText("Server Status: Starting...")
            self.output_text.append("Starting server...")
            
            # Set up multiple health check attempts
            self.health_check_attempts = 0
            self.health_check_timer = QTimer()
            self.health_check_timer.timeout.connect(self._check_server_status)
            self.health_check_timer.start(1000)  # Check every second
            
            self.serverStarted.emit()
            
        except Exception as e:
            error_msg = f"Failed to start server: {str(e)}"
            self.output_text.append(f"ERROR: {error_msg}")
            QMessageBox.critical(self, "Server Error", error_msg)

    def _check_server_status(self):
        """Check if server is responding"""
        try:
            try:
                response = requests.get(f"http://127.0.0.1:{self.port}/health")
                self.output_text.append(f"Health check response: {response.text}")
                
                if response.status_code == 200:
                    self.server_status_label.setText("Server Status: Ready")
                    self.output_text.append("Server is ready!")
                    if hasattr(self, 'health_check_timer'):
                        self.health_check_timer.stop()
                else:
                    self.output_text.append(f"Server returned status code: {response.status_code}")
            except requests.ConnectionError:
                self.output_text.append("Health check failed - server not responding")
                
            QThread.msleep(500)

            try:
                response = requests.get(f"http://127.0.0.1:{self.port}/")
                self.output_text.append(f"Root endpoint response: {response.text}")
                if response.status_code != 200:
                    self.output_text.append(f"Root endpoint status code: {response.status_code}")
            except Exception as e:
                self.output_text.append(f"Root endpoint error: {str(e)}")
                
            self.health_check_attempts += 1
            if self.health_check_attempts >= 5:
                if not hasattr(self, '_server_ready'):
                    self.server_status_label.setText("Server Status: Failed to Start")
                    self.output_text.append("Server failed to start after 5 attempts")
                    self.stop_server()
                    if hasattr(self, 'health_check_timer'):
                        self.health_check_timer.stop()
            
        except Exception as e:
            self.server_status_label.setText("Server Status: Error")
            self.output_text.append(f"Error checking server status: {str(e)}")

    def stop_server(self):
        if self.server_process:
            try:
                try:
                    if platform.system() == 'Windows':
                        subprocess.run(['taskkill', '/F', '/T', '/PID', str(self.server_process.pid)], 
                                         check=True, capture_output=True)
                    else:
                        pgid = os.getpgid(self.server_process.pid)
                        os.killpg(pgid, signal.SIGTERM)
                        try:
                            self.server_process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(pgid, signal.SIGKILL)
                    
                    self.server_process.stdout.close() if hasattr(self.server_process, 'stdout') else None
                    self.server_process.stderr.close() if hasattr(self.server_process, 'stderr') else None
                    self.server_process = None
                    
                except Exception as e:
                    self.output_text.append(f"Error stopping server: {str(e)}")
                else:
                    try:
                        os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                    except ProcessLookupError:
                        self.output_text.append("Process already terminated")
                
                self.start_server_btn.setEnabled(True)
                self.stop_server_btn.setEnabled(False)
                self.server_status_label.setText("Server Status: Stopped")
                self.output_text.append("Server stopped")
                self.serverStopped.emit()
                
            except Exception as e:
                error_msg = f"Error stopping server: {str(e)}"
                self.output_text.append(f"ERROR: {error_msg}")
                QMessageBox.warning(self, "Server Stop Error", error_msg)
            finally:
                self.server_process = None

    def connect_to_host(self):
        """Connect to a remote host, or in host mode, connect to the local FastAPI server via WebSocket."""
        if self.is_host_mode:
            # In host mode, always connect to the local server at 127.0.0.1 using the port from port_input.
            ip = "127.0.0.1"
            try:
                port = int(self.port_input.text())
            except ValueError:
                QMessageBox.warning(self, "Invalid Port", "Please enter a valid port number")
                return
            # Save connection info in main_window
            self.main_window.host = ip
            self.main_window.port = port
            # Initiate WebSocket connection (host will then receive updates like logs and client lists)
            self.main_window.toggle_websocket_connection()  # Direct call since it's already decorated with @asyncSlot
            # Optionally, subscribe to the system topic for updates
            asyncio.create_task(self.subscribe_to_system_topic())
        else:
            # Client mode: get IP and port from the user
            ip = self.host_input.text()
            port_text = self.host_port_input.text()
            if not ip or not port_text:
                QMessageBox.warning(self, "Missing Information", "Please enter both host IP and port.")
                return
            try:
                port = int(port_text)
                self.main_window.host = ip
                self.main_window.port = port
                self.main_window.toggle_websocket_connection()  # Direct call since it's already decorated with @asyncSlot
                self.subscribe_btn.setEnabled(True)
                asyncio.create_task(self.subscribe_to_system_topic())
            except ValueError:
                QMessageBox.warning(self, "Invalid Port", "Please enter a valid port number")

    async def subscribe_to_system_topic(self):
        """Subscribes to the system topic."""
        if hasattr(self, 'main_window') and self.main_window.websocket_connection:
            message = {
                "action": "subscribe",
                "payload": {"topic": ConnectionManager.SYSTEM_TOPIC}
            }
            await self.main_window.websocket_connection.send(json.dumps(message))
            self.output_text.append("Subscribed to system topic.")

    def send_message(self):
        """Send a message to the connected host"""
        message_text = self.message_input.toPlainText()
        topic = self.topic_input.text()  # Get topic from input

        if not message_text:
            QMessageBox.warning(self, "No Message", "Please enter a message to send.")
            return
        if not topic and not self.is_host_mode:
            QMessageBox.warning(self, "No Topic", "Please enter a topic to send to.")
            return
        topic_to_send = topic if topic or self.is_host_mode else topic

        try:
            if hasattr(self, 'main_window') and self.main_window.websocket_connection:
                message = {
                    "action": "publish",
                    "payload": {
                        "topic": topic_to_send,
                        "data": message_text,
                        "context": "user"
                    }
                }
                asyncio.create_task(self.main_window.websocket_connection.send(json.dumps(message)))
                self.output_text.append(f"Sent to '{topic_to_send}': {message_text}")
                self.message_input.clear()
            else:
                self.output_text.append("Not connected to a host.")
        except Exception as e:
            QMessageBox.critical(self, "Send Error", str(e))

    def handle_received_message(self, message):
        """Handles received messages"""
        try:
            data = json.loads(message)
            if data.get("action") == "client_list":
                client_list = data.get("payload", {}).get("clients", [])
                self.update_client_list(client_list)
            elif data.get("action") == "message":
                payload = data.get("payload", {})
                topic = payload.get("topic")
                message_data = payload.get("data")
                # Append to the subscribed messages widget instead of output_text
                self.subscribed_messages.append(f"Received on '{topic}': {message_data}")
            else:
                # For other actions, also log to subscribed messages if desired
                self.subscribed_messages.append(f"Received: {message}")
        except json.JSONDecodeError:
            self.subscribed_messages.append(f"Received: {message}")

    def update_client_list(self, client_list):
        """Updates the client list display"""
        self.clients_list.clear()
        for client in client_list:
            self.clients_list.addItem(f"Client {client}")

    def update_websocket_status(self, is_connected):
        """Updates the UI based on WebSocket connection status"""
        if is_connected:
            self.connect_button.setText("Disconnect")
            self.connect_button.setIcon(load_bootstrap_icon("plug-fill", "#4CAF50"))  # Green connected icon
            self.connect_host_btn.setText("Disconnect")
            self.connect_host_btn.setIcon(load_bootstrap_icon("plug-fill", "#4CAF50"))  # Green connected icon
            self.ws_status_label.setText("WebSocket: Connected")
            self.ws_status_label.setStyleSheet("color: #4CAF50;")  # Green text
            self.subscribe_btn.setEnabled(True)
            self.unsubscribe_btn.setEnabled(True)
            self.publish_btn.setEnabled(True)
            if not self.is_host_mode:
                asyncio.create_task(self.subscribe_to_system_topic())
        else:
            self.connect_button.setText("Connect")
            self.connect_button.setIcon(load_bootstrap_icon("plug", "#F44336"))  # Red disconnected icon
            self.connect_host_btn.setText("Connect to Host")
            self.connect_host_btn.setIcon(load_bootstrap_icon("plug", "#F44336"))  # Red disconnected icon
            self.ws_status_label.setText("WebSocket: Disconnected")
            self.ws_status_label.setStyleSheet("color: #F44336;")  # Red text
            self.subscribe_btn.setEnabled(False)
            self.unsubscribe_btn.setEnabled(False)
            self.publish_btn.setEnabled(False)

    def closeEvent(self, event):
        """Handle section close event"""
        if self.server_process:
            self.stop_server()
        super().closeEvent(event)

    def subscribe_to_topic(self):
        """Subscribe to a specific topic on the local server"""
        topic = self.topic_input.text()
        if not topic:
            QMessageBox.warning(self, "Invalid Topic", "Please enter a topic name")
            return

        if self.is_host_mode:
            if hasattr(self.main_window, 'websocket_connection'):
                asyncio.create_task(self.main_window.subscribe_host_to_topic(topic))
            self.output_text.append(f"Host subscribed to topic: {topic}")
        else:
            if hasattr(self, 'main_window') and self.main_window.websocket_connection:
                message = {
                    "action": "subscribe",
                    "payload": {"topic": topic}
                }
                asyncio.create_task(self.main_window.websocket_connection.send(json.dumps(message)))
                self.output_text.append(f"Subscribed to topic: {topic}")

    def unsubscribe_from_topic(self):
        """Unsubscribe from a specific topic"""
        topic = self.topic_input.text()
        if not topic:
            QMessageBox.warning(self, "No Topic", "Please enter a topic to unsubscribe from.")
            return

        if self.is_host_mode:
            if hasattr(self.main_window, 'unsubscribe_host_from_topic'):
                asyncio.create_task(self.main_window.unsubscribe_host_from_topic(topic))
            self.output_text.append(f"Host unsubscribed from topic: {topic}")
        else:
            if hasattr(self, 'main_window') and self.main_window.websocket_connection:
                message = {
                    "action": "unsubscribe",
                    "payload": {"topic": topic}
                }
                asyncio.create_task(self.main_window.websocket_connection.send(json.dumps(message)))
                self.output_text.append(f"Unsubscribed from topic: {topic}")

    def send_ping(self):
        """Send a ping message"""
        if hasattr(self, 'main_window'):
            self.main_window.send_websocket_ping()  # Direct call since it's already decorated with @asyncSlot
