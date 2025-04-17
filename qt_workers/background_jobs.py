import subprocess
from PyQt6.QtCore import QObject, pyqtSignal, QThread
import platform
import requests


class OllamaStatusWorker(QObject):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def run(self):
        """
        Check if Ollama is running by trying to list models.
        If successful, returns the list of models.
        """
        try:
            check_cmd = ["ollama", "ls"]
            res_check = subprocess.run(check_cmd, capture_output=True, text=True)
            if res_check.returncode == 0:
                self.finished.emit(f"Ollama is running.\nInstalled models:\n{res_check.stdout}")
            else:
                self.error.emit("Ollama is not running.")
        except Exception as e:
            self.error.emit(f"Unexpected error:\n{str(e)}")

class CouchDBStatusWorker(QObject):
    finished = pyqtSignal(str)  # Success message
    error = pyqtSignal(str)     # Error message
    status = pyqtSignal(bool)   # True if running, False if not
    using_mock = pyqtSignal(bool)  # Signal to indicate if using mock

    def run(self):
        """Check if CouchDB is running and accessible, or if we're using the mock."""
        # Import here to avoid circular imports
        from db_ops.generate_tables import DatabaseSetup
        
        # Create a database setup instance to check if we're using the mock
        db_setup = DatabaseSetup()
        
        if db_setup.use_mock:
            # We're using the JSON mock implementation
            self.status.emit(True)  # Mock is always "running"
            self.using_mock.emit(True)
            self.finished.emit("Using JSON-based mock CouchDB (no server required)")
            return
        
        # We're supposed to use the real CouchDB - check if it's available
        try:
            url = "http://127.0.0.1:5984"
            response = requests.get(
                f"{url}/_up",
                auth=("admin", "cfpwd"),
                timeout=5
            )
            if response.status_code == 200:
                self.status.emit(True)
                self.using_mock.emit(False)
                self.finished.emit("CouchDB is running and accessible")
            else:
                self.status.emit(False)
                self.using_mock.emit(False)
                self.error.emit(f"CouchDB returned status code: {response.status_code}")
        except requests.exceptions.ConnectionError:
            self.status.emit(False)
            self.using_mock.emit(False)
            self.error.emit("Could not connect to CouchDB - service may not be running")
        except requests.exceptions.Timeout:
            self.status.emit(False)
            self.using_mock.emit(False)
            self.error.emit("Connection to CouchDB timed out")
        except Exception as e:
            self.status.emit(False)
            self.using_mock.emit(False)
            self.error.emit(f"Error checking CouchDB: {str(e)}")

# class OllamaServerWorker(QObject):
#     finished = pyqtSignal(str)
#     error = pyqtSignal(str)

#     def run(self):
#         """
#         Start the Ollama server with './ollama serve' in the background.
#         This blocks if we use run(), so we'll use Popen to keep it alive.
#         """
#         try:
#             serve_cmd = ["ollama", "serve"]
#             # If 'ollama' is in PATH, you could do just ["ollama", "serve"].
#             # or do a check first if the user doesn't store 'ollama' in the local dir

#             process = subprocess.Popen(serve_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#             # Keep reading the output if you want to show logs in real-time

#             # You could do something like:
#             for line in iter(process.stdout.readline, ''):
#                 # emit partial logs if needed
#                 print("[ollama serve] ", line, end="")  # or store in a buffer

#             # If the server ever exits:
#             process.stdout.close()
#             process.wait()
#             self.finished.emit("Ollama server process exited.")
#         except Exception as e:
#             self.error.emit(f"Failed to start Ollama server:\n{str(e)}")

# class OllamaRunModelWorker(QObject):
#     finished = pyqtSignal(str)
#     error = pyqtSignal(str)

#     def run(self):
#         try:
#             run_cmd = ["ollama", "run", "deepseek-r1:1.5b"]  # or your custom model name
#             process = subprocess.Popen(run_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

#             output, err = process.communicate()  # Wait until it finishes
#             if process.returncode == 0:
#                 self.finished.emit(f"Ollama run completed:\n{output}")
#             else:
#                 self.error.emit(f"ollama run encountered an error (code {process.returncode}):\n{err}")
#         except Exception as e:
#             self.error.emit(f"Failed to run model:\n{str(e)}")


# class CouchDBInstallWorker(QObject):
#     finished = pyqtSignal(str)
#     error = pyqtSignal(str)
#     progress = pyqtSignal(str)  # New signal for installation progress updates

#     def __init__(self):
#         super().__init__()
#         self.admin_password = "admin123"  # Default admin password
#         self.process = None

#     def run(self):
#         """Non-blocking installation process"""
#         try:
#             system = platform.system().lower()
            
#             if system == "linux":
#                 self._install_linux_async()
#             elif system == "darwin":
#                 self._install_macos_async()
#             elif system == "windows":
#                 self._install_windows_async()
#             else:
#                 raise Exception(f"Unsupported operating system: {system}")

#         except Exception as e:
#             self.error.emit(f"Installation failed: {str(e)}")

#     def _install_linux_async(self):
#         """Non-blocking Linux installation"""
#         try:
#             # Create a shell script with all commands
#             install_script = """#!/bin/bash
# set -e  # Exit on any error

# # Add GPG key
# curl -L https://couchdb.apache.org/repo/keys.asc | gpg --dearmor | sudo tee /usr/share/keyrings/couchdb-archive-keyring.gpg > /dev/null

# # Add repository
# echo 'deb [signed-by=/usr/share/keyrings/couchdb-archive-keyring.gpg] https://apache.jfrog.io/artifactory/couchdb-deb/ jammy main' | sudo tee /etc/apt/sources.list.d/couchdb.list

# # Update and install dependencies
# sudo apt-get update
# sudo apt-get install -y libicu70 libmozjs-91-0 openssl

# # Install CouchDB
# sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-downgrades couchdb

# # Signal completion
# echo "Installation completed successfully"
# """
#             # Write script to temporary file
#             with open("/tmp/install_couchdb.sh", "w") as f:
#                 f.write(install_script)
            
#             # Make executable
#             subprocess.run(["chmod", "+x", "/tmp/install_couchdb.sh"], check=True)
            
#             # Run installation in background
#             self.process = subprocess.Popen(
#                 ["/tmp/install_couchdb.sh"],
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 text=True,
#                 bufsize=1,
#                 universal_newlines=True
#             )

#             # Monitor output in real-time
#             for line in iter(self.process.stdout.readline, ''):
#                 self.progress.emit(line.strip())
                
#             # Check final status
#             self.process.wait()
#             if self.process.returncode == 0:
#                 self._configure_couchdb()
#                 self.finished.emit("CouchDB installation completed successfully")
#             else:
#                 error = self.process.stderr.read()
#                 raise Exception(f"Installation failed with error: {error}")

#         except Exception as e:
#             self.error.emit(f"Installation failed: {str(e)}")

#     def _install_macos_async(self):
#         """Non-blocking MacOS installation"""
#         try:
#             self.process = subprocess.Popen(
#                 ["brew", "install", "couchdb"],
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 text=True
#             )
            
#             # Monitor output
#             for line in iter(self.process.stdout.readline, ''):
#                 self.progress.emit(line.strip())
            
#             self.process.wait()
#             if self.process.returncode == 0:
#                 self._configure_couchdb()
#                 self.finished.emit("CouchDB installation completed successfully")
#             else:
#                 error = self.process.stderr.read()
#                 raise Exception(f"Installation failed with error: {error}")

#         except Exception as e:
#             self.error.emit(f"Installation failed: {str(e)}")

#     def _install_windows_async(self):
#         """Non-blocking Windows installation"""
#         try:
#             # Download installer first
#             self.progress.emit("Downloading CouchDB installer...")
#             installer_url = "https://couchdb.neighbourhood.ie/downloads/3.3.2/win/apache-couchdb-3.3.2.msi"
#             response = requests.get(installer_url, stream=True)
            
#             with open("couchdb_installer.msi", "wb") as f:
#                 for chunk in response.iter_content(chunk_size=8192):
#                     if chunk:
#                         f.write(chunk)
            
#             # Run installer
#             self.progress.emit("Running installer...")
#             self.process = subprocess.Popen([
#                 "msiexec", "/i", "couchdb_installer.msi",
#                 "/quiet", "INSTALLSERVICE=1",
#                 f"ADMINUSER=admin", f"ADMINPASS={self.admin_password}"
#             ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
#             self.process.wait()
#             if self.process.returncode == 0:
#                 self._configure_couchdb()
#                 self.finished.emit("CouchDB installation completed successfully")
#             else:
#                 error = self.process.stderr.read()
#                 raise Exception(f"Installation failed with error: {error}")

#         except Exception as e:
#             self.error.emit(f"Installation failed: {str(e)}")

#     def _configure_couchdb(self):
#         config = {
#             "couchdb": {
#                 "single_node": True,
#                 "max_document_size": "4294967296"  # 4GB
#             },
#             "admins": {
#                 "admin": self.admin_password
#             },
#             "chttpd": {
#                 "require_valid_user": True,
#                 "bind_address": "127.0.0.1"
#             },
#             "httpd": {
#                 "enable_cors": False
#             },
#             "couch_httpd_auth": {
#                 "require_valid_user": True
#             }
#         }

#         # Write configuration
#         config_file = "/opt/couchdb/etc/local.ini" if platform.system().lower() != "windows" else "C:\\CouchDB\\etc\\local.ini"
        
#         with open(config_file, "w") as f:
#             for section, values in config.items():
#                 f.write(f"[{section}]\n")
#                 for key, value in values.items():
#                     f.write(f"{key} = {value}\n")
#                 f.write("\n")

#         # Restart CouchDB service
#         if platform.system().lower() == "windows":
#             subprocess.run(["net", "stop", "CouchDB"], check=True)
#             subprocess.run(["net", "start", "CouchDB"], check=True)
#         else:
#             subprocess.run(["systemctl", "restart", "couchdb"], check=True)

# def check_couchdb_status():
#     self.status_worker = CouchDBStatusWorker()
#     self.status_worker.finished.connect(self.on_status_success)
#     self.status_worker.error.connect(self.on_status_error)
#     self.status_worker.status.connect(self.update_status_indicator)
    
#     self.thread = QThread()
#     self.status_worker.moveToThread(self.thread)
#     self.thread.started.connect(self.status_worker.run)
#     self.thread.start()



class OllamaInstallWorker(QObject):
    finished = pyqtSignal(str)  # Emitted on success (with some info message)
    error = pyqtSignal(str)     # Emitted on error

    def run(self):
        """
        1) Checks if 'ollama' is on PATH.
        2) If not found, runs the installation command.
        3) Finally, runs 'ollama ls' to list available models.
        """
        try:
            # 1) Check if ollama is installed:
            check_cmd = ["which", "ollama"]
            res_check = subprocess.run(check_cmd, capture_output=True, text=True)
            already_installed = (res_check.returncode == 0)

            if already_installed:
                info = "Ollama is already installed.\n"
            else:
                # 2) Run the install script
                install_cmd = "curl -fsSL https://ollama.com/install.sh | sh"
                subprocess.run(install_cmd, shell=True, check=True)
                info = "Ollama was not installed. Successfully installed now.\n"

            # 3) Run 'ollama ls' to see what models are available
            ls_cmd = ["ollama", "ls"]
            res_ls = subprocess.run(ls_cmd, capture_output=True, text=True)
            if res_ls.returncode == 0:
                info += "\nollama ls output:\n" + res_ls.stdout
            else:
                info += "\nUnable to run 'ollama ls'."

            self.finished.emit(info)

        except subprocess.CalledProcessError as e:
            self.error.emit(f"Command failed:\n{str(e)}")
        except Exception as e:
            self.error.emit(f"Unexpected error:\n{str(e)}")

class UvicornServerWorker(QObject):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    status = pyqtSignal(bool)  # True if running, False if not

    def __init__(self, host="127.0.0.1", port=8000):
        super().__init__()
        self.host = host
        self.port = port
        self.process = None

    def run(self):
        """Start the Uvicorn server with the FastAPI app"""
        try:
            server_cmd = ["uvicorn", "server:app", 
                         f"--host={self.host}", 
                         f"--port={self.port}"]
            
            self.process = subprocess.Popen(
                server_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor the server startup
            for line in iter(self.process.stdout.readline, ''):
                if "Application startup complete" in line:
                    self.status.emit(True)
                    self.finished.emit(f"FastAPI server is running at http://{self.host}:{self.port}")
                    break
                
            # Monitor for any errors
            for line in iter(self.process.stderr.readline, ''):
                self.error.emit(f"Server error: {line.strip()}")
                self.status.emit(False)
                break

        except Exception as e:
            self.status.emit(False)
            self.error.emit(f"Failed to start FastAPI server: {str(e)}")

    def stop(self):
        """Stop the Uvicorn server"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.status.emit(False)
            self.finished.emit("FastAPI server stopped")