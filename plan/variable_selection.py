import asyncio
from typing import Dict, List, Optional, Tuple, Any
import json
import re

from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QComboBox, QFrame, QMessageBox
)

class HypothesisVariableSelector(QObject):
    """
    Class to handle automatic variable selection and testing for a hypothesis.
    Works with DataTestingWidget from select.py to automate the workflow.
    """
    
    # Signals for progress updates and completion
    progress_updated = pyqtSignal(str, int)  # message, percent complete
    mapping_completed = pyqtSignal(dict)     # variable mapping results
    model_built = pyqtSignal(dict)           # model building results
    test_completed = pyqtSignal(dict)        # test results
    error_occurred = pyqtSignal(str)         # error message
    
    def __init__(self, testing_widget=None, parent=None):
        super().__init__(parent)
        # Check if testing_widget is None and log warning
        if testing_widget is None:
            print("WARNING: HypothesisVariableSelector initialized with testing_widget=None")
        
        # Store the testing widget
        self.testing_widget = testing_widget
        self.dataset_name = None
        self.hypothesis_text = None
        self.mapped_variables = None
        self.selected_outcome = None
        self.test_results = None  # Add test_results attribute to store results
    
    def set_dataset(self, dataset_name: str):
        """Set the dataset to use for testing"""
        self.dataset_name = dataset_name
        
    def set_hypothesis(self, hypothesis_text: str):
        """Set the hypothesis text to use for selecting the appropriate outcome variable"""
        self.hypothesis_text = hypothesis_text
    
    async def run_workflow(self):
        """Run the full workflow: load dataset, map variables, select outcome, build model, run test"""
        if not self.dataset_name or not self.hypothesis_text:
            self.error_occurred.emit("Dataset name and hypothesis text must be set before running workflow")
            return False
        
        try:
            # 0. Verify testing_widget is available
            if not self.testing_widget:
                self.error_occurred.emit("Testing widget not initialized. Make sure the app is properly set up.")
                return False
                
            print(f"Running workflow with dataset: {self.dataset_name}, hypothesis: {self.hypothesis_text}")
                
            # 1. Load the dataset
            self.progress_updated.emit("Loading dataset...", 10)
            print(f"Step 1: Loading dataset: {self.dataset_name}")
            
            # Try to refresh datasets first
            if hasattr(self.testing_widget, 'load_dataset_from_study'):
                try:
                    await self.testing_widget.load_dataset_from_study()
                    print("Refreshed dataset list from studies manager")
                    await asyncio.sleep(1.5)  # Wait for UI to update
                except Exception as e:
                    print(f"Non-critical error refreshing datasets: {str(e)}")
            
            success = await self.load_dataset()
            if not success:
                self.error_occurred.emit(f"Failed to load dataset: {self.dataset_name}")
                return False
            
            print(f"Dataset loaded successfully: {self.dataset_name}")
            await asyncio.sleep(1.5)  # Wait for UI to stabilize
            
            # Process UI events to ensure dataset is fully loaded
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
            
            # 2. Build model (maps variables)
            self.progress_updated.emit("Building statistical model...", 30)
            print("Step 2: Building statistical model (mapping variables)...")
            
            # Clear any existing variable assignments
            self.testing_widget.clear_all_assignments()
            
            # Call the build_model function to map variables
            await self.build_model()
            
            # Wait for UI to update after build_model
            await asyncio.sleep(1.0)
            QApplication.processEvents()
            
            # 3. Identify the appropriate test
            self.progress_updated.emit("Identifying appropriate test...", 50)
            print("Step 3: Identifying appropriate statistical test...")
            
            # Don't check the test before auto-select - it will be incorrect
            
            # Call auto_select_test to select appropriate test AFTER variables are mapped
            self.testing_widget.auto_select_test()
            
            # Extended wait to ensure test selection completes
            await asyncio.sleep(3.0)
            QApplication.processEvents()
            
            # Now check and log what test was actually selected
            selected_test = "Unknown test"
            selected_index = -1
            if hasattr(self.testing_widget, 'test_combo'):
                selected_test = self.testing_widget.test_combo.currentText()
                selected_index = self.testing_widget.test_combo.currentIndex()
                print(f"Auto-selected test: '{selected_test}' at index {selected_index}")
            
            # 4. Select the appropriate outcome based on hypothesis text
            self.progress_updated.emit("Selecting outcome variable...", 70)
            print("Step 4: Selecting outcome variable based on hypothesis...")
            outcome_selected = await self.select_outcome_from_hypothesis()
            if not outcome_selected:
                self.error_occurred.emit("Failed to select an appropriate outcome variable for the hypothesis")
                return False
            
            await asyncio.sleep(1.5)  # Wait after outcome selection
            QApplication.processEvents()
            
            # 5. Run the test
            self.progress_updated.emit("Running statistical test...", 90)
            print("Step 5: Running statistical test...")
            
            # Important: Call run_test and ensure we wait for it to complete
            await self.run_test()
            
            # Ensure UI updates completed
            await asyncio.sleep(2.0)
            QApplication.processEvents()
            
            # Final check on selected test for logs
            if hasattr(self.testing_widget, 'test_combo'):
                final_test = self.testing_widget.test_combo.currentText()
                print(f"Final test after completion: '{final_test}'")
            
            # Complete - emit signal that workflow is completed
            self.progress_updated.emit("Test completed successfully", 100)
            
            # We need to explicitly emit the test_completed signal with results
            if self.test_results:
                self.test_completed.emit(self.test_results)
                
            return True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"Error during workflow execution: {str(e)}")
            return False
    
    async def load_dataset(self) -> bool:
        """Load the dataset from the studies manager"""
        try:
            print(f"Attempting to load dataset: {self.dataset_name}")
            
            # Check if testing_widget exists
            if not self.testing_widget:
                print(f"Error: testing_widget is None when trying to load dataset")
                return False
                
            # First check if we need to refresh datasets
            if hasattr(self.testing_widget, 'load_dataset_from_study'):
                print("Refreshing datasets from study...")
                try:
                    await self.testing_widget.load_dataset_from_study()
                    await asyncio.sleep(0.5)  # Wait for UI to update
                except Exception as e:
                    print(f"Error refreshing datasets: {str(e)}")
                
            # Find the dataset selector/combo - support both naming conventions
            dataset_combo = None
            if hasattr(self.testing_widget, 'dataset_selector'):
                dataset_combo = self.testing_widget.dataset_selector
                print(f"Using dataset_selector with {dataset_combo.count()} items")
            elif hasattr(self.testing_widget, 'dataset_combo'):
                dataset_combo = self.testing_widget.dataset_combo
                print(f"Using dataset_combo with {dataset_combo.count()} items")
            else:
                print("Error: testing_widget has no dataset_selector or dataset_combo")
                return False
                
            # Log available datasets for debugging
            available_datasets = []
            for i in range(dataset_combo.count()):
                available_datasets.append(dataset_combo.itemText(i))
            print(f"Available datasets: {available_datasets}")
            
            # Find the dataset in the combobox
            dataset_found = False
            for i in range(dataset_combo.count()):
                if dataset_combo.itemText(i) == self.dataset_name:
                    # Select the dataset
                    print(f"Found dataset '{self.dataset_name}' at index {i}")
                    dataset_combo.setCurrentIndex(i)
                    # Wait a moment for the dataset to load
                    await asyncio.sleep(0.5)
                    dataset_found = True
                    return True
            
            if not dataset_found:
                print(f"Error: Dataset '{self.dataset_name}' not found in dropdown")
                
                # If no exact match, try partial match
                partial_matches = []
                for i in range(dataset_combo.count()):
                    item_text = dataset_combo.itemText(i)
                    if (self.dataset_name.lower() in item_text.lower() or 
                        item_text.lower() in self.dataset_name.lower()):
                        partial_matches.append((i, item_text))
                
                if partial_matches:
                    print(f"Found {len(partial_matches)} partial matches: {partial_matches}")
                    # Use the first partial match
                    index, match_name = partial_matches[0]
                    print(f"Using partial match: '{match_name}' at index {index}")
                    dataset_combo.setCurrentIndex(index)
                    # Update our internal dataset name
                    self.dataset_name = match_name
                    await asyncio.sleep(0.5)
                    return True
                elif dataset_combo.count() > 0:
                    # If no match at all, use the first available dataset
                    print(f"No matches found. Using first available dataset: '{dataset_combo.itemText(0)}'")
                    dataset_combo.setCurrentIndex(0)
                    # Update our internal dataset name
                    self.dataset_name = dataset_combo.itemText(0)
                    await asyncio.sleep(0.5)
                    return True
            
            # If dataset not found and no alternatives available
            return False
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    async def map_variables(self):
        """Map variables automatically using the testing widget's functionality"""
        # This function is now a stub - mapping is done directly in run_workflow via build_model
        return True
    
    async def select_outcome_from_hypothesis(self) -> bool:
        """
        Use LLM to determine the most appropriate outcome variable from the hypothesis text
        when multiple outcomes are available.
        """
        try:
            if not self.mapped_variables:
                self.error_occurred.emit("Variables must be mapped before selecting an outcome")
                return False
            
            # Get current outcome variable
            current_outcome = self.mapped_variables.get('outcome')
            if not current_outcome:
                self.error_occurred.emit("No outcome variable found in mapped variables")
                return False
            
            # Get all potential outcome variables
            all_outcomes = []
            outcome_combo = self.testing_widget.outcome_combo
            for i in range(outcome_combo.count()):
                outcome_name = outcome_combo.itemText(i)
                if outcome_name:  # Skip empty entries
                    all_outcomes.append(outcome_name)
            
            # If only one outcome, just use that
            if len(all_outcomes) <= 1:
                self.selected_outcome = current_outcome
                return True
            
            # Use LLM to determine the best outcome match for the hypothesis
            best_outcome = await self.determine_best_outcome_with_llm(all_outcomes, self.hypothesis_text)
            
            if best_outcome:
                # Set the selected outcome in the UI
                for i in range(outcome_combo.count()):
                    if outcome_combo.itemText(i) == best_outcome:
                        outcome_combo.setCurrentIndex(i)
                        # Wait for the change to take effect
                        await asyncio.sleep(0.2)
                        self.selected_outcome = best_outcome
                        return True
            
            # If LLM doesn't find a good match, use the automatically mapped one
            self.selected_outcome = current_outcome
            return True
            
        except Exception as e:
            print(f"Error selecting outcome from hypothesis: {str(e)}")
            self.error_occurred.emit(f"Error selecting outcome: {str(e)}")
            # Use the current outcome as a fallback
            self.selected_outcome = self.mapped_variables.get('outcome')
            return self.selected_outcome is not None
    
    async def determine_best_outcome_with_llm(self, outcome_variables: List[str], hypothesis_text: str) -> Optional[str]:
        """
        Use LLM to determine which outcome variable best matches the hypothesis.
        Uses the testing widget's LLM function.
        """
        # Get access to LLM functions
        from data.selection.select import call_llm_async
        
        # Build the prompt
        prompt = f"""
You are analyzing a research hypothesis and need to select the most appropriate outcome variable based on the variables available in the dataset.

HYPOTHESIS: {hypothesis_text}

AVAILABLE OUTCOME VARIABLES: {', '.join(outcome_variables)}

Determine which of these variables would be the most appropriate outcome variable to test this hypothesis.
Consider the meaning of the hypothesis and match it to the most relevant variable.
ONLY return the exact name of the variable that best matches the hypothesis.
Return only one variable name with no additional explanation.
        """
        
        # Call LLM
        response = await call_llm_async(prompt)
        
        # Parse response - we expect just the variable name
        response = response.strip()
        
        # Check if response matches one of our outcome variables
        if response in outcome_variables:
            return response
        
        # If not an exact match, look for partial matches
        for var in outcome_variables:
            if var.lower() in response.lower() or response.lower() in var.lower():
                return var
        
        # If still no match, return None and we'll use the default
        return None
    
    async def build_model(self) -> bool:
        """Build the statistical model using the testing widget's functionality"""
        try:
            
            print("Building statistical model (mapping variables)...")
            
            # Pass the hypothesis text directly to build_model to avoid showing the dialog
            if self.hypothesis_text:
                await self.testing_widget.build_model(direct_hypothesis=self.hypothesis_text)
            else:
                await self.testing_widget.build_model()
            
            # Process events to ensure UI updates
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
            
            # Now collect the mapped variables after build_model has run
            outcome = self.testing_widget.outcome_combo.currentText() if self.testing_widget.outcome_combo.isEnabled() else None
            
            self.mapped_variables = {
                'outcome': outcome,
                'group': self.testing_widget.group_combo.currentText() if self.testing_widget.group_combo.isEnabled() else None,
                'subject_id': self.testing_widget.subject_id_combo.currentText() if self.testing_widget.subject_id_combo.isEnabled() else None,
                'time': self.testing_widget.time_combo.currentText() if self.testing_widget.time_combo.isEnabled() else None
            }
            
            print(f"Variables mapped successfully: {self.mapped_variables}")
            
            # If the testing widget has a selected test, update test info
            if hasattr(self.testing_widget, 'test_combo'):
                test_name = self.testing_widget.test_combo.currentText()
                print(f"Current test after build_model: '{test_name}'")
            
            # Emit signal with the mapped variables
            self.mapping_completed.emit(self.mapped_variables)
            return True
            
        except Exception as e:
            print(f"Error building model: {str(e)}")
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"Error building model: {str(e)}")
            return False
    
    async def run_test(self) -> bool:
        """Run the statistical test using the testing widget's functionality"""
        try:
            # Double-check the selected test before running
            if hasattr(self.testing_widget, 'test_combo'):
                test_before_run = self.testing_widget.test_combo.currentText()
                print(f"Before running test: Currently selected test is '{test_before_run}'")
            
            print("Running statistical test...")
            
            # Call the run_statistical_test function
            await self.testing_widget.run_statistical_test()
            
            # Process events to ensure UI updates
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
            
            # Check what test was actually run
            if hasattr(self.testing_widget, 'test_combo'):
                test_after_run = self.testing_widget.test_combo.currentText()
                print(f"After running test: Currently selected test is '{test_after_run}'")
                
                # Get test results directly from testing widget
                if hasattr(self.testing_widget, 'last_test_results'):
                    self.test_results = self.testing_widget.last_test_results
                    print(f"Retrieved test results from testing widget")
                    # Emit the test completion signal with results
                    self.test_completed.emit(self.test_results)
            
            # Extended wait with UI processing
            await asyncio.sleep(2.0)
            QApplication.processEvents()
            
            return True
            
        except Exception as e:
            print(f"Error running test: {str(e)}")
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"Error running test: {str(e)}")
            return False


class VariableSelectionWidget(QWidget):
    """Widget for the variable selection step that integrates with DataTestingWidget"""
    
    test_completed = pyqtSignal(dict)  # Signal when test is completed with results
    next_step = pyqtSignal()  # Signal to move to the next step
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset_name = None
        self.hypothesis_text = None
        self.selector = None
        self.test_results = None  # Store the latest test results
        self.init_ui()
        
        # We'll initialize the selector when a testing_widget is provided via setup_selector
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Step 2: Variable Selection")
        header.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(header)
        
        # Description
        description = QLabel("This step automatically selects variables and runs a statistical test "
                            "to evaluate your hypothesis.")
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Dataset info
        self.dataset_label = QLabel("Selected Dataset: None")
        layout.addWidget(self.dataset_label)
        
        # Hypothesis text
        self.hypothesis_label = QLabel("Hypothesis: None")
        self.hypothesis_label.setWordWrap(True)
        layout.addWidget(self.hypothesis_label)
        
        # Progress area
        progress_frame = QFrame()
        progress_frame.setFrameShape(QFrame.Shape.StyledPanel)
        progress_layout = QVBoxLayout(progress_frame)
        
        self.progress_label = QLabel("Ready to begin analysis")
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        layout.addWidget(progress_frame)
        
        # Results area
        self.results_frame = QFrame()
        self.results_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.results_frame.setVisible(False)
        results_layout = QVBoxLayout(self.results_frame)
        
        results_header = QLabel("Results")
        results_header.setStyleSheet("font-weight: bold;")
        results_layout.addWidget(results_header)
        
        self.outcome_label = QLabel("Outcome Variable: ")
        results_layout.addWidget(self.outcome_label)
        
        self.group_label = QLabel("Group Variable: ")
        results_layout.addWidget(self.group_label)
        
        self.test_label = QLabel("Selected Test: ")
        results_layout.addWidget(self.test_label)
        
        self.result_label = QLabel("Test Result: ")
        self.result_label.setWordWrap(True)
        results_layout.addWidget(self.result_label)
        
        # Add a significance indicator
        self.significance_label = QLabel("")
        self.significance_label.setStyleSheet("font-weight: bold;")
        self.significance_label.setVisible(False)
        results_layout.addWidget(self.significance_label)
        
        # Add action buttons for save, generate hypothesis, and interpretation
        action_buttons_layout = QHBoxLayout()
        
        from PyQt6.QtGui import QIcon
        
        self.save_btn = QPushButton("Save Results")
        self.save_btn.setToolTip("Save the results to the study")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_results)
        action_buttons_layout.addWidget(self.save_btn)
        
        self.hypothesis_btn = QPushButton("Generate Hypothesis")
        self.hypothesis_btn.setToolTip("Generate a hypothesis based on the selected variables")
        self.hypothesis_btn.setEnabled(False) 
        self.hypothesis_btn.clicked.connect(self.generate_hypothesis)
        # action_buttons_layout.addWidget(self.hypothesis_btn)
        
        self.interpret_btn = QPushButton("Interpret Results")
        self.interpret_btn.setToolTip("Generate an interpretation of the test results")
        self.interpret_btn.setEnabled(False)
        self.interpret_btn.clicked.connect(self.interpret_results)
        action_buttons_layout.addWidget(self.interpret_btn)
        
        results_layout.addLayout(action_buttons_layout)
        
        layout.addWidget(self.results_frame)
        
        # Buttons - only Continue button, no Start Analysis
        button_layout = QHBoxLayout()
        
        self.continue_btn = QPushButton("Continue to Step 3")
        self.continue_btn.setEnabled(False)
        self.continue_btn.clicked.connect(self.next_step.emit)  # Connect to next_step signal
        # button_layout.addWidget(self.continue_btn)
        
        layout.addLayout(button_layout)
        
        # Stretch to push everything to the top
        layout.addStretch()
    
    def set_data(self, dataset_name: str, hypothesis_text: str):
        """Set the dataset and hypothesis to analyze"""
        print(f"Setting data in VariableSelectionWidget: {dataset_name}, {hypothesis_text}")
        self.dataset_name = dataset_name
        self.hypothesis_text = hypothesis_text
        
        # Update labels
        self.dataset_label.setText(f"Selected Dataset: {dataset_name}")
        self.hypothesis_label.setText(f"Hypothesis: {hypothesis_text}")
        
        # Reset progress
        self.progress_bar.setValue(0)
        self.progress_label.setText("Ready to begin analysis")
        self.results_frame.setVisible(False)
        self.significance_label.setVisible(False)
        self.continue_btn.setEnabled(False)
        
        # Disable action buttons until results are available
        self.save_btn.setEnabled(False)
        self.hypothesis_btn.setEnabled(False)
        self.interpret_btn.setEnabled(False)
        
        # If we have a selector, update it with the new data
        if self.selector:
            self.selector.set_dataset(dataset_name)
            self.selector.set_hypothesis(hypothesis_text)
            print(f"Updated selector with dataset: {dataset_name}, hypothesis: {hypothesis_text}")
            
            # Reset the test label to be empty until auto_select_test completes
            self.test_label.setText("Selected Test: Waiting for test selection...")
    
    def setup_selector(self, testing_widget=None):
        """Set up the variable selector with the provided testing widget"""
            
        print(f"Setting up variable selector with testing widget: {testing_widget}")
        
        # Store the testing widget directly in this widget for easy access
        self.testing_widget = testing_widget
        
        # Check if testing_widget is None
        if testing_widget is None:
            print("Warning: testing_widget is None in setup_selector")
            return

        # Create the selector with the provided testing widget
        if hasattr(self, 'selector') and self.selector is not None:
            # If selector already exists, just update its testing_widget
            print("Selector already exists, updating its testing_widget")
            self.selector.testing_widget = testing_widget
        else:
            # Create a new selector
            self.selector = HypothesisVariableSelector(testing_widget)
        
        # Double check the testing_widget is properly set in the selector
        if self.selector.testing_widget is None:
            print("WARNING: selector.testing_widget is None after initialization! Setting it directly.")
            self.selector.testing_widget = testing_widget
        
        # Connect signals
        self.selector.progress_updated.connect(self.update_progress)
        self.selector.mapping_completed.connect(self.on_mapping_completed)
        self.selector.model_built.connect(self.on_model_built)
        self.selector.test_completed.connect(self.on_test_completed)
        self.selector.error_occurred.connect(self.on_error)
        
        # If we already have data, set it in the selector
        if self.dataset_name and self.hypothesis_text:
            self.selector.set_dataset(self.dataset_name)
            self.selector.set_hypothesis(self.hypothesis_text)
            print(f"Initialized selector with existing data: {self.dataset_name}, {self.hypothesis_text}")
    
    def refresh_datasets(self):
        """Refresh available datasets from the testing widget"""
        if not self.selector or not self.selector.testing_widget:
            print("Cannot refresh datasets - testing widget not initialized")
            return False
            
        testing_widget = self.selector.testing_widget
        
        # Check if the testing widget has dataset refresh capabilities
        if hasattr(testing_widget, 'load_dataset_from_study'):
            print("Refreshing datasets from studies manager")
            try:
                # Call in non-async context
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(testing_widget.load_dataset_from_study())
                else:
                    loop.run_until_complete(testing_widget.load_dataset_from_study())
                print("Dataset list refreshed")
                return True
            except Exception as e:
                print(f"Error refreshing datasets: {e}")
                return False
        else:
            print("Testing widget does not support dataset refreshing")
            return False
    
    async def run_workflow(self):
        """Run the analysis workflow"""
        try:
            success = await self.selector.run_workflow()
            if success:
                self.continue_btn.setEnabled(True)
        except Exception as e:
            self.on_error(f"Error during analysis: {str(e)}")
    
    def update_progress(self, message: str, percent: int):
        """Update the progress display"""
        self.progress_label.setText(message)
        self.progress_bar.setValue(percent)
        
        # Force UI update
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
    
    def on_mapping_completed(self, variable_mapping: Dict):
        """Handle completion of variable mapping"""
        # Show the mapped variables
        self.results_frame.setVisible(True)
        self.outcome_label.setText(f"Outcome Variable: {variable_mapping.get('outcome', 'None')}")
        self.group_label.setText(f"Group Variable: {variable_mapping.get('group', 'None')}")
        
        # Force UI update
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
    
    def on_model_built(self, result: Dict):
        """Handle completion of model building"""
        # Do not update the test type here, wait for auto_select_test to complete
        pass
    
    def on_test_completed(self, result: Dict):
        """Handle completion of statistical test"""
        print("Test completed, updating results panel...")
        
        # Store the test results
        self.test_results = result
        
        # Format the result (this is a simplified version, you may want to customize)
        if isinstance(result, dict):
            # Try to extract p-value and test statistic safely
            p_value = None
            statistic = None
            
            # Check if p_value is directly in result
            if 'p_value' in result:
                p_value = result['p_value']
            # Check if p_value is in a nested 'results' dictionary
            elif 'results' in result and isinstance(result['results'], dict) and 'p_value' in result['results']:
                p_value = result['results']['p_value']
                
            # Similarly for statistic
            if 'statistic' in result:
                statistic = result['statistic']
            elif 'results' in result and isinstance(result['results'], dict) and 'statistic' in result['results']:
                statistic = result['results']['statistic']
            
            if p_value is not None:
                result_text = f"P-value: {p_value:.4f}"
                if statistic is not None:
                    result_text += f", Test statistic: {statistic:.4f}"
                    
                # Add significance statement
                is_significant = p_value < 0.05
                if is_significant:
                    result_text += " (Statistically significant at p < 0.05)"
                    # Show significance indicator with green styling
                    self.significance_label.setText("✓ The data supports this hypothesis")
                    self.significance_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 14px;")
                    self.significance_label.setVisible(True)
                else:
                    result_text += " (Not statistically significant at p < 0.05)"
                    # Show significance indicator with red styling
                    self.significance_label.setText("✗ The data does not support this hypothesis")
                    self.significance_label.setStyleSheet("color: #F44336; font-weight: bold; font-size: 14px;")
                    self.significance_label.setVisible(True)
                    
                self.result_label.setText(f"Test Result: {result_text}")
            else:
                self.result_label.setText("Test Result: Test completed, but no p-value found in results")
        else:
            self.result_label.setText("Test Result: Test completed successfully")
            
        # CRITICAL: Always enable all action buttons unconditionally
        print("Enabling all action buttons unconditionally")
        self.save_btn.setEnabled(True)
        self.hypothesis_btn.setEnabled(True)
        self.interpret_btn.setEnabled(True)
        
        # Update the test label
        if self.selector and self.selector.testing_widget:
            testing_widget = self.selector.testing_widget
            if hasattr(testing_widget, 'test_combo'):
                test_name = testing_widget.test_combo.currentText()
                print(f"Final test selection is: '{test_name}'")
                self.test_label.setText(f"Selected Test: {test_name}")
        
        # Make sure the results frame is visible
        self.results_frame.setVisible(True)
        
        # Enable continue button
        self.continue_btn.setEnabled(True)
        
        # Force UI update
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
        
        # Emit signal with the results
        self.test_completed.emit(result)
    
    def save_results(self):
        """Save the test results to the study using testing widget's method"""
        if self.selector and self.selector.testing_widget:
            if hasattr(self.selector.testing_widget, 'save_results_to_study'):
                try:
                    # Let select.py handle the save directly
                    self.selector.testing_widget.save_results_to_study()
                    print("Results saved to study using select.py's save method")
                except Exception as e:
                    print(f"Error saving results: {str(e)}")
                    import traceback
                    traceback.print_exc()
    
    def generate_hypothesis(self):
        """Generate a hypothesis using testing widget's method"""
        if not self.selector or not self.selector.testing_widget:
            print("Testing widget not available")
            return
            
        testing_widget = self.selector.testing_widget
        
        # Get the currently selected variables
        outcome = None
        group = None
        subject_id = None
        time = None
        
        if hasattr(testing_widget, 'outcome_combo') and testing_widget.outcome_combo.isEnabled():
            outcome = testing_widget.outcome_combo.currentText()
        if hasattr(testing_widget, 'group_combo') and testing_widget.group_combo.isEnabled():
            group = testing_widget.group_combo.currentText()
        if hasattr(testing_widget, 'subject_id_combo') and testing_widget.subject_id_combo.isEnabled():
            subject_id = testing_widget.subject_id_combo.currentText()
        if hasattr(testing_widget, 'time_combo') and testing_widget.time_combo.isEnabled():
            time = testing_widget.time_combo.currentText()
            
        test_name = None
        if hasattr(testing_widget, 'test_combo'):
            test_name = testing_widget.test_combo.currentText()
            
        # Check if testing widget has the required method
        if hasattr(testing_widget, 'generate_hypothesis_for_test'):
            # Create task to call the async method
            asyncio.create_task(self._run_generate_hypothesis(
                outcome, group, subject_id, time, test_name))
            
    async def _run_generate_hypothesis(self, outcome, group, subject_id, time, test_name):
        """Run the generate hypothesis method from testing widget"""
        if not self.selector or not self.selector.testing_widget:
            return
            
        try:
            # Get the parent dialog to potentially access its hypothesis ID
            parent_dialog = self.find_parent_dialog()
            hypothesis_id = None
            
            # If we have a parent dialog with a config that has an ID, use that directly
            if parent_dialog and hasattr(parent_dialog, 'config') and hasattr(parent_dialog.config, 'id'):
                hypothesis_id = parent_dialog.config.id
                print(f"Found hypothesis ID from parent dialog: {hypothesis_id}")
            
            # Call the method with the hypothesis ID - let select.py handle the generation
            result = await self.selector.testing_widget.generate_hypothesis_for_test(
                outcome, group, subject_id, time, test_name, hypothesis_id=hypothesis_id)
                
            # Check if we got a string (hypothesis text) or a dictionary (full result with ID and status)
            hypothesis_text = None
            hypothesis_status = None
            
            if isinstance(result, str):
                # Old behavior - just got hypothesis text
                hypothesis_text = result
            elif isinstance(result, dict):
                # New behavior - got full hypothesis data
                hypothesis_text = result.get('title')
                hypothesis_status = result.get('status')
                print(f"Got hypothesis data with status: {hypothesis_status}")
            elif result:
                # Fall back to treating result as hypothesis text
                hypothesis_text = str(result)
                
            # Just update the parent dialog's hypothesis text
            if parent_dialog and hypothesis_text:
                print(f"Generated hypothesis: {hypothesis_text}")
                
                # Update the hypothesis text in the dialog
                if hasattr(parent_dialog, 'text_edit'):
                    parent_dialog.text_edit.setPlainText(hypothesis_text)
                
                # Also update the custom hypothesis edit
                if hasattr(parent_dialog, 'custom_hypothesis_edit'):
                    parent_dialog.custom_hypothesis_edit.setText(hypothesis_text)
                
                # Update the config text
                if hasattr(parent_dialog, 'config'):
                    parent_dialog.config.text = hypothesis_text
                    print("Updated hypothesis text in parent dialog")
                
                # Force update UI - switch to the edit tab to make changes visible
                if hasattr(parent_dialog, 'tabs'):
                    print("Switching to edit tab to make changes visible")
                    parent_dialog.tabs.setCurrentIndex(0)
                
                # Update the hypothesis state based on the returned status or fallback to DRAFT
                if hasattr(parent_dialog, 'state_combo'):
                    from plan.plan_config import HypothesisState
                    
                    # Map the status string to HypothesisState enum
                    target_state = HypothesisState.DRAFT  # Default fallback
                    if hypothesis_status:
                        status_map = {
                            'confirmed': HypothesisState.VALIDATED,
                            'rejected': HypothesisState.REJECTED,
                            'inconclusive': HypothesisState.INCONCLUSIVE,
                            'untested': HypothesisState.UNTESTED,
                            'proposed': HypothesisState.PROPOSED,
                            'testing': HypothesisState.TESTING,
                        }
                        target_state = status_map.get(hypothesis_status.lower(), HypothesisState.DRAFT)
                        print(f"Mapping status '{hypothesis_status}' to state {target_state}")
                    
                    # Find and select the appropriate state in the combo
                    for i in range(parent_dialog.state_combo.count()):
                        if parent_dialog.state_combo.itemData(i) == target_state:
                            parent_dialog.state_combo.setCurrentIndex(i)
                            # Also update the config state
                            if hasattr(parent_dialog, 'config'):
                                parent_dialog.config.state = target_state
                                print(f"Updated hypothesis state to {target_state}")
                            break
                
                # Force UI update
                from PyQt6.QtWidgets import QApplication
                QApplication.processEvents()
                
        except Exception as e:
            print(f"Error generating hypothesis: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def find_parent_dialog(self):
        """Find the parent HypothesisEditorDialog if it exists"""
        parent = self.parent()
        while parent:
            if parent.__class__.__name__ == 'HypothesisEditorDialog':
                return parent
            parent = parent.parent()
        return None
    
    def interpret_results(self):
        """Generate an interpretation of the test results using testing widget's method"""
        if not self.selector or not self.selector.testing_widget:
            print("Error: Testing widget not available")
            return
            
        testing_widget = self.selector.testing_widget
        
        # Get the current test key directly from testing_widget
        test_key = None
        if hasattr(testing_widget, 'current_test_key'):
            test_key = testing_widget.current_test_key
            print(f"Found test_key from current_test_key: {test_key}")
        elif hasattr(testing_widget, 'test_combo'):
            # Get currently selected test from combo box
            test_index = testing_widget.test_combo.currentIndex()
            if hasattr(testing_widget, 'available_tests') and test_index >= 0:
                test_keys = list(testing_widget.available_tests.keys())
                if test_index < len(test_keys):
                    test_key = test_keys[test_index]
                    print(f"Found test_key from combo box: {test_key}")
        
        # Get test results directly from testing_widget 
        result = None
        if hasattr(testing_widget, 'current_test_result'):
            result = testing_widget.current_test_result
            print(f"Found test results from current_test_result: {result is not None}")
        elif self.test_results:
            result = self.test_results
            print(f"Using test results from self.test_results: {result is not None}")
        
        # Debug the available parameters
        print(f"Interpretation parameters: test_key={test_key}, result={result is not None}")
        if result:
            print(f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        
        # Check if testing widget has the required method
        if hasattr(testing_widget, 'interpret_results_with_llm') and test_key and result:
            # Create task to call the async method
            asyncio.create_task(self._run_interpret_results(test_key, result))
        else:
            reasons = []
            if not hasattr(testing_widget, 'interpret_results_with_llm'):
                reasons.append("interpret_results_with_llm method not available")
            if not test_key:
                reasons.append("test_key not found")
            if not result:
                reasons.append("test results not available")
            print(f"Error: Interpretation not available - {', '.join(reasons)}")
            
    async def _run_interpret_results(self, test_key, result):
        """Run the interpret results method from testing widget"""
        if not self.selector or not self.selector.testing_widget:
            return
            
        try:
            print(f"Calling interpret_results_with_llm with test_key={test_key}")
            
            # Call the interpretation method from select.py
            await self.selector.testing_widget.interpret_results_with_llm(test_key, result)
            
            # Get the updated test results with interpretation
            if hasattr(self.selector.testing_widget, 'current_test_result'):
                updated_result = self.selector.testing_widget.current_test_result
                
                # Find parent HypothesisEditorDialog if it exists
                parent_dialog = self.find_parent_dialog()
                
                # Just update the hypothesis config with the interpretation from select.py
                if parent_dialog and hasattr(parent_dialog, 'config'):
                    if not hasattr(parent_dialog.config, 'test_results'):
                        parent_dialog.config.test_results = {}
                    
                    # Update the test_results with the updated version that includes interpretation
                    parent_dialog.config.test_results = updated_result
                    print("Updated hypothesis with interpretation from select.py")
                    
                    # Update UI elements in the parent dialog to reflect the changes
                    if hasattr(parent_dialog, 'tabs'):
                        # Switch to the edit tab to make changes visible
                        parent_dialog.tabs.setCurrentIndex(0)
        except Exception as e:
            print(f"Error interpreting results: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def on_error(self, error_message: str):
        """Handle errors during the workflow"""
        self.progress_label.setText(f"Error: {error_message}")
        QMessageBox.warning(self, "Analysis Error", error_message)
        self.continue_btn.setEnabled(True)  # Re-enable continue button
    
    async def start_analysis_workflow(self):
        """Start the analysis workflow directly - used by the HypothesisEditorDialog"""
        if not self.selector:
            print("Error: Selector not initialized")
            return False
            
        print("Starting analysis workflow directly...")
        
        # Reset progress display
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting analysis...")
        
        # Run the workflow
        success = await self.selector.run_workflow()
        
        # If successful, make sure buttons are enabled
        if success:
            self.save_btn.setEnabled(True)
            self.hypothesis_btn.setEnabled(True)
            self.interpret_btn.setEnabled(True)
            self.results_frame.setVisible(True)
            self.continue_btn.setEnabled(True)
            
            # Force UI update
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
            
        return success 