from PyQt6.QtCore import QObject, pyqtSignal
import asyncio
import traceback
import os
import sys

# Force non-interactive Agg backend before any matplotlib imports
# This must happen before any other matplotlib import
import matplotlib
matplotlib.use('Agg', force=True)

# Disable all interactive features
os.environ['MPLBACKEND'] = 'Agg'  # Redundant but thorough
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Prevent any Qt window creation attempts

# Import matplotlib after setting backend
import matplotlib.pyplot as plt
import re

# Ensure matplotlib is in non-interactive mode
plt.ioff()
plt.interactive(False)
plt.rcParams['interactive'] = False

# Disable SIP's bad catcher result handling - add this before importing any PyQt modules
try:
    import sip
    sip.setdestroyonexit(False)  # Don't destroy C++ objects on exit
    # Override the bad catcher result function if accessible
    if hasattr(sip, 'setBadCatcherResult'):
        original_bad_catcher = sip.setBadCatcherResult
        sip.setBadCatcherResult(lambda: None)
except ImportError:
    pass

# Constants
MAX_FIGURES_PER_SESSION = 100  # Increased limit for maximum number of figures to capture in a session

class AgentWorker(QObject):
    """Worker thread for running agents."""
    
    output = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    figure_created = pyqtSignal(object)
    figure_count_updated = pyqtSignal(int)  # New signal to report the current figure count
    temp_file_created = pyqtSignal(str)  # Signal for when a temporary file is created
    
    def __init__(self, agent=None, task=None):
        """
        Initialize the worker with an agent and task.
        
        Args:
            agent: The agent to run
            task: The task to execute
        """
        super().__init__()
        self.agent = agent
        self.task = task
        self.figures = []
        self.figure_count = 0
        self.temp_files = []  # Track temporary files
        
    def _patch_matplotlib(self):
        """
        Patch matplotlib to capture figures and prevent pop-ups.
        """
        # Make absolutely sure the backend is Agg and not Qt-related
        matplotlib.use('Agg', force=True)
        
        # Completely disable interactive mode
        plt.ioff()
        
        # Set additional parameters to prevent any pop-ups or interactive elements
        plt.rcParams['interactive'] = False
        plt.rcParams['figure.max_open_warning'] = 0
        
        # Override any Qt-specific backend imports that might have happened
        try:
            # This prevents many thread-related errors by removing Qt-specific functionality
            sys.modules['matplotlib.backends.backend_qt'] = None
            sys.modules['matplotlib.backends.backend_qt5'] = None
            sys.modules['matplotlib.backends.backend_qt5agg'] = None
        except:
            pass
            
        # Ensure no GUI event loop interaction
        # This prevents the "set_wakeup_fd" error in thread
        try:
            if hasattr(asyncio, 'set_event_loop'):
                asyncio.set_event_loop(asyncio.new_event_loop())
        except:
            pass
        
        # Keep track of all figures created so we can close them later
        self.figures = []
        
        # Track how many figures have been created
        self.figure_count = 0
        
        # Flag to track if we've emitted a warning about max figures
        max_figures_warning_emitted = [False]
        
        # Store original functions
        original_figure = plt.figure
        original_subplots = plt.subplots
        original_show = plt.show
        original_savefig = plt.savefig
        
        # Self reference for use in nested functions
        self_ref = self
        
        # Block interactive functions
        def blocked_show(*args, **kwargs):
            # Intentionally do nothing, just return
            return None
            
        # Create patched versions of matplotlib functions
        def patched_figure(*args, **kwargs):
            try:
                # Create the figure but don't render it yet
                fig = original_figure(*args, **kwargs)
                
                # Make the figure background transparent
                fig.patch.set_alpha(0.0)
                
                # Set default colors if not specified
                if not kwargs.get('facecolor'):
                    fig.set_facecolor('none')  # Transparent background
                    
                # Add to our list of figures
                self_ref.figures.append(fig)
                self_ref.figure_count += 1
                
                # Signal that a figure was created - this will be handled in the main thread
                # Use QTimer to emit signal to avoid sipBadCatcherResult errors
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(0, lambda: self_ref.figure_created.emit(fig))
                
                # Signal the updated figure count - also with QTimer
                QTimer.singleShot(0, lambda: self_ref.figure_count_updated.emit(self_ref.figure_count))
                
                return fig
            except Exception as e:
                # Catch any Qt-related errors and return a simple figure instead
                self_ref.progress.emit(f"\n⚠️ Error creating figure: {str(e)}\n")
                # Create a basic figure with no Qt dependencies
                fig = plt.Figure()
                return fig
            
        def patched_subplots(*args, **kwargs):
            try:
                # Create the figure and axes
                fig, ax = original_subplots(*args, **kwargs)
                
                # Make the figure background transparent
                fig.patch.set_alpha(0.0)
                
                # Set default colors if not specified
                if not kwargs.get('facecolor'):
                    fig.set_facecolor('none')  # Transparent background
                    
                # Add to our list of figures
                self_ref.figures.append(fig)
                self_ref.figure_count += 1
                
                # Signal that a figure was created - use QTimer to avoid threading issues
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(0, lambda: self_ref.figure_created.emit(fig))
                
                # Signal the updated figure count - also with QTimer
                QTimer.singleShot(0, lambda: self_ref.figure_count_updated.emit(self_ref.figure_count))
                
                return fig, ax
            except Exception as e:
                # Catch any Qt-related errors and return a simple figure instead
                self_ref.progress.emit(f"\n⚠️ Error creating subplot: {str(e)}\n")
                # Create basic figure and axes with no Qt dependencies
                fig = plt.Figure()
                ax = fig.add_subplot(111)
                return fig, ax
                
        def patched_savefig(filename, *args, **kwargs):
            """
            Track saved figure files
            """
            try:
                # Call the original savefig
                result = original_savefig(filename, *args, **kwargs)
                
                # Track the file
                if isinstance(filename, str) and os.path.exists(filename):
                    self_ref.temp_files.append(filename)
                    # Use QTimer to emit signal from main thread
                    from PyQt6.QtCore import QTimer
                    QTimer.singleShot(0, lambda: self_ref.temp_file_created.emit(filename))
                    
                return result
            except Exception as e:
                self_ref.progress.emit(f"\n⚠️ Error saving figure: {str(e)}\n")
                return None
            
        # Replace the original functions with our patched versions
        plt.figure = patched_figure
        plt.subplots = patched_subplots
        plt.show = blocked_show
        plt.savefig = patched_savefig
        
        # Block other potentially dangerous interactive functions
        plt.pause = lambda *args, **kwargs: None
        plt.waitforbuttonpress = lambda *args, **kwargs: False
        
    def _check_output_for_figure_files(self, text):
        """Check output text for mentions of saved figure files."""
        if not text:
            return
            
        # Look for the pattern that the modified plt.show() creates
        file_matches = re.findall(r'Figure saved to: (.*\.png)', text)
        for filename in file_matches:
            if os.path.exists(filename):
                self.temp_files.append(filename)
                
                # Use QTimer to emit signal from main thread to avoid sipBadCatcherResult
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(0, lambda: self.temp_file_created.emit(filename))
                
                # Try to load the saved figure if we need to emit it
                try:
                    # Don't try to load and emit here - just report that the file exists
                    self.progress.emit(f"\n🖼️ Figure saved to file: {os.path.basename(filename)}\n")
                except Exception as e:
                    print(f"Error handling figure file {filename}: {e}")
                    
    async def run_agent(self):
        """
        Run the agent asynchronously.
        """
        try:
            # Apply matplotlib patching
            self._patch_matplotlib()
            
            # Intercept progress signals to look for saved figure files
            def progress_handler(text):
                try:
                    self._check_output_for_figure_files(text)
                    # Use QTimer to emit signal from main thread
                    from PyQt6.QtCore import QTimer
                    QTimer.singleShot(0, lambda: self.progress.emit(text))
                except Exception as e:
                    print(f"Error in progress handler: {str(e)}")
                    # Fallback - try direct emit
                    try:
                        self.progress.emit(f"Error in progress handling: {str(e)}\n{text}")
                    except:
                        print(f"Could not emit progress signal: {text}")
            
            # Run the agent with the task
            if self.agent and self.task:
                await self.agent.execute(
                    self.task,
                    output_callback=progress_handler
                )
                
            # Use QTimer to emit finished signal from main thread
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, lambda: self.finished.emit())
            
        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            # Use QTimer to emit error from main thread
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, lambda: self.error.emit(error_msg))
            QTimer.singleShot(0, lambda: self.finished.emit())
        finally:
            # Make sure to clean up figures to prevent memory leaks
            try:
                import matplotlib.pyplot as plt
                for fig in self.figures:
                    try:
                        plt.close(fig)
                    except:
                        pass
                self.figures = []
                
                # Don't delete temp files here - let the main thread handle this
                # after it has processed the figures
            except:
                pass
    
    def run(self):
        """
        Run the agent in the current thread.
        """
        try:
            asyncio.run(self.run_agent())
        except Exception as e:
            error_msg = f"Error in worker run: {str(e)}\n{traceback.format_exc()}"
            # Use QTimer to safely emit from main thread
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, lambda: self.error.emit(error_msg))
            QTimer.singleShot(0, lambda: self.finished.emit())
        
    def cleanup_temp_files(self):
        """
        Clean up temporary files.
        """
        for filename in self.temp_files:
            try:
                if os.path.exists(filename):
                    os.remove(filename)
            except Exception as e:
                print(f"Error removing temporary file {filename}: {e}")
                
        self.temp_files = [] 