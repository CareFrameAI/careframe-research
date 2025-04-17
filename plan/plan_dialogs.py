from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QTextEdit, QComboBox, QDoubleSpinBox, QCheckBox,
    QDialogButtonBox, QTabWidget, QPushButton, QWidget
)
from PyQt6.QtCore import pyqtSlot

from helpers.load_icon import load_bootstrap_icon
from plan.plan_config import Evidence, EvidenceSourceType, HypothesisState, ObjectiveType
from plan.hypothesis_generator import HypothesisGeneratorWidget
from qasync import asyncSlot
import asyncio

class EvidenceDialog(QDialog):
    """Dialog for adding or editing evidence"""
    
    def __init__(self, parent=None, evidence=None):
        super().__init__(parent)
        self.evidence = evidence
        self.setup_ui()
        
        # Fill fields if editing
        if evidence:
            self.type_combo.setCurrentText(evidence.type.value.title())
            self.description_edit.setPlainText(evidence.description)
            self.source_edit.setText(evidence.source)
            self.confidence_spin.setValue(evidence.confidence)
            self.notes_edit.setPlainText(evidence.notes)
            
            # Set status if available
            if hasattr(evidence, 'status'):
                status_index = self.status_combo.findText(evidence.status.title())
                if status_index >= 0:
                    self.status_combo.setCurrentIndex(status_index)
    
    def setup_ui(self):
        """Setup the UI components"""
        self.setWindowTitle("Evidence")
        self.resize(400, 400)
        
        layout = QVBoxLayout(self)
        
        # Type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        
        for ev_type in EvidenceSourceType:
            self.type_combo.addItem(ev_type.value.title(), ev_type)
        
        type_layout.addWidget(self.type_combo)
        layout.addLayout(type_layout)
        
        # Description
        layout.addWidget(QLabel("Description:"))
        self.description_edit = QTextEdit()
        layout.addWidget(self.description_edit)
        
        # Source
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Source:"))
        self.source_edit = QLineEdit()
        source_layout.addWidget(self.source_edit)
        layout.addLayout(source_layout)
        
        # Confidence
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Confidence:"))
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0, 1)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.5)
        self.confidence_spin.setDecimals(2)
        confidence_layout.addWidget(self.confidence_spin)
        layout.addLayout(confidence_layout)
        
        # Status
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("Status:"))
        self.status_combo = QComboBox()
        self.status_combo.addItems(["Validated", "Rejected", "Inconclusive"])
        status_layout.addWidget(self.status_combo)
        layout.addLayout(status_layout)
        
        # Notes
        layout.addWidget(QLabel("Notes:"))
        self.notes_edit = QTextEdit()
        layout.addWidget(self.notes_edit)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def get_evidence_data(self):
        """Get evidence data from form"""
        ev_type = self.type_combo.currentData()
        description = self.description_edit.toPlainText()
        source = self.source_edit.text()
        confidence = self.confidence_spin.value()
        notes = self.notes_edit.toPlainText()
        status = self.status_combo.currentText().lower()
        
        # Create evidence object
        evidence = Evidence(
            id=str(id(self)) if not self.evidence else self.evidence.id,
            type=ev_type,
            description=description,
            supports=True,  # Default, will be overridden
            confidence=confidence,
            source=source,
            notes=notes,
            status=status
        )
        
        return evidence
    
# ======================
# Objective Editor Dialog
# ======================

class ObjectiveEditorDialog(QDialog):
    """Dialog for editing objective properties"""
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setup_ui()
    
    def setup_ui(self):
        self.setWindowTitle("Edit Objective")
        self.resize(500, 300)
        
        layout = QVBoxLayout(self)
        
        # Objective text
        text_label = QLabel("Objective Text:")
        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(self.config.text)
        layout.addWidget(text_label)
        layout.addWidget(self.text_edit)
        
        # Type
        type_layout = QHBoxLayout()
        type_label = QLabel("Type:")
        self.type_combo = QComboBox()
        for obj_type in ObjectiveType:
            self.type_combo.addItem(obj_type.value.replace("_", " ").title(), obj_type)
        
        # Set current type
        for i in range(self.type_combo.count()):
            if self.type_combo.itemData(i) == self.config.type:
                self.type_combo.setCurrentIndex(i)
                break
        
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.type_combo)
        layout.addLayout(type_layout)
        
        # Description
        desc_label = QLabel("Description:")
        self.desc_edit = QTextEdit()
        self.desc_edit.setPlainText(self.config.description)
        self.desc_edit.setMaximumHeight(100)
        layout.addWidget(desc_label)
        layout.addWidget(self.desc_edit)
        
        # Progress
        progress_layout = QHBoxLayout()
        progress_label = QLabel("Progress:")
        self.progress_spin = QDoubleSpinBox()
        self.progress_spin.setRange(0, 1)
        self.progress_spin.setSingleStep(0.05)
        self.progress_spin.setValue(self.config.progress)
        self.progress_spin.setDecimals(2)
        progress_layout.addWidget(progress_label)
        progress_layout.addWidget(self.progress_spin)
        layout.addLayout(progress_layout)
        
        # Auto-generate checkbox
        self.auto_generate_check = QCheckBox("Auto-generate hypotheses")
        self.auto_generate_check.setChecked(self.config.auto_generate)
        layout.addWidget(self.auto_generate_check)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def get_updated_config(self):
        """Return updated objective config"""
        # Update config with dialog values
        self.config.text = self.text_edit.toPlainText()
        
        # Get type
        current_index = self.type_combo.currentIndex()
        self.config.type = self.type_combo.itemData(current_index)
        
        # Get description
        self.config.description = self.desc_edit.toPlainText()
        
        # Get progress
        self.config.progress = self.progress_spin.value()
        
        # Get auto-generate
        self.config.auto_generate = self.auto_generate_check.isChecked()
        
        return self.config

# ======================
# Hypothesis Editor Dialog
# ======================

class HypothesisEditorDialog(QDialog):
    """Dialog for editing hypothesis properties"""
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.studies_manager = None
        
        # Try to get studies_manager from parent
        if parent and hasattr(parent, 'studies_manager'):
            self.studies_manager = parent.studies_manager
        
        self.setup_ui()
    
    def setup_ui(self):
        self.setWindowTitle("Edit Hypothesis")
        self.resize(800, 600)  # Larger size to accommodate the generator
        
        layout = QVBoxLayout(self)
        
        # Create tabs
        self.tabs = QTabWidget()
        self.edit_tab = QWidget()
        self.generator_tab = QWidget()
        
        # Setup edit tab
        self.setup_edit_tab()
        
        # Setup generator tab
        self.setup_generator_tab()
        
        # Add tabs to widget
        self.tabs.addTab(self.edit_tab, "Edit")
        self.tabs.addTab(self.generator_tab, "Generate")
        
        layout.addWidget(self.tabs)
        
        # Buttons at the bottom
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def setup_edit_tab(self):
        """Setup the edit tab with hypothesis editing fields"""
        edit_layout = QVBoxLayout(self.edit_tab)
        
        # Hypothesis text
        text_label = QLabel("Hypothesis:")
        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(self.config.text)
        edit_layout.addWidget(text_label)
        edit_layout.addWidget(self.text_edit)
        
        # Confidence
        conf_layout = QHBoxLayout()
        conf_label = QLabel("Confidence:")
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0, 1)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(self.config.confidence)
        self.confidence_spin.setDecimals(2)
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.confidence_spin)
        edit_layout.addLayout(conf_layout)
        
        # State
        state_layout = QHBoxLayout()
        state_label = QLabel("State:")
        self.state_combo = QComboBox()
        for state in HypothesisState:
            self.state_combo.addItem(state.value.title(), state)
        
        # Set current state
        for i in range(self.state_combo.count()):
            if self.state_combo.itemData(i) == self.config.state:
                self.state_combo.setCurrentIndex(i)
                break
        
        state_layout.addWidget(state_label)
        state_layout.addWidget(self.state_combo)
        edit_layout.addLayout(state_layout)
    
    def setup_generator_tab(self):
        """Setup the generator tab with the hypothesis generator widget"""
        generator_layout = QVBoxLayout(self.generator_tab)
        
        # Add note about using the generator
        note_label = QLabel("Use the hypothesis generator to create and test hypotheses based on your data.")
        note_label.setWordWrap(True)
        generator_layout.addWidget(note_label)
        
        # Add a custom hypothesis entry field
        hypothesis_layout = QHBoxLayout()
        hypothesis_layout.addWidget(QLabel("Custom Hypothesis:"))
        
        self.custom_hypothesis_edit = QLineEdit()
        self.custom_hypothesis_edit.setPlaceholderText("Enter a hypothesis to test...")
        
        # Set initial text from current config
        if hasattr(self, 'config') and self.config and hasattr(self.config, 'text'):
            self.custom_hypothesis_edit.setText(self.config.text)
            
        hypothesis_layout.addWidget(self.custom_hypothesis_edit)
        
        self.set_hypothesis_btn = QPushButton("Use")
        self.set_hypothesis_btn.clicked.connect(self.set_custom_hypothesis)
        hypothesis_layout.addWidget(self.set_hypothesis_btn)
        
        generator_layout.addLayout(hypothesis_layout)
        
        # Create the generator widget
        from plan.hypothesis_generator import HypothesisGeneratorWidget
        self.generator_widget = HypothesisGeneratorWidget()
        
        # Set a default hypothesis text based on the current config or custom input
        if hasattr(self, 'config') and self.config and hasattr(self.config, 'text'):
            self.generator_widget.current_hypothesis_text = self.config.text
        
        # Connect to studies manager if available
        if self.studies_manager:
            self.generator_widget.set_studies_manager(self.studies_manager)
            
            # Set up signal connections for hypothesis generation results
            if hasattr(self.generator_widget, 'variable_selection_step'):
                self.generator_widget.variable_selection_step.test_completed.connect(self.on_test_completed)
            
            # Connect to the hypothesis_tested signal
            if hasattr(self.generator_widget, 'hypothesis_tested'):
                self.generator_widget.hypothesis_tested.connect(self.on_hypothesis_tested)
                
            # Connect signals to handle dataset selection and progression
            if hasattr(self.generator_widget, 'data_sources_step'):
                self.generator_widget.data_sources_step.sources_selected.connect(self.on_sources_selected)
        
        generator_layout.addWidget(self.generator_widget)
        
        # Button area with single run analysis button
        button_layout = QHBoxLayout()
        
        # Add Run Analysis button
        self.run_analysis_btn = QPushButton("Run Analysis")
        self.run_analysis_btn.setIcon(load_bootstrap_icon("play-circle", "#333333"))
        self.run_analysis_btn.clicked.connect(self.run_analysis)
        button_layout.addWidget(self.run_analysis_btn)
        
        # Add Use Selected Hypothesis button
        self.use_generated_btn = QPushButton("Use Selected Hypothesis")
        self.use_generated_btn.setIcon(load_bootstrap_icon("check2-circle", "#333333"))
        self.use_generated_btn.clicked.connect(self.use_generated_hypothesis)
        button_layout.addWidget(self.use_generated_btn)
        
        generator_layout.addLayout(button_layout)
    
    def set_custom_hypothesis(self):
        """Set a custom hypothesis text from the edit field"""
        custom_text = self.custom_hypothesis_edit.text().strip()
        if custom_text:
            # Update the generator widget
            if hasattr(self.generator_widget, 'set_hypothesis_text'):
                self.generator_widget.set_hypothesis_text(custom_text)
            else:
                self.generator_widget.current_hypothesis_text = custom_text
                
            # If we're on the variable selection step, update it
            if hasattr(self.generator_widget, 'timeline') and \
               hasattr(self.generator_widget, 'variable_selection_step') and \
               self.generator_widget.timeline.active_step == 2:
                # Make sure we have datasets selected
                if self.generator_widget.selected_datasets:
                    dataset_name, _ = self.generator_widget.selected_datasets[0]
                    self.generator_widget.variable_selection_step.set_data(dataset_name, custom_text)
    
    @asyncSlot()
    async def run_analysis(self):
        """Run the analysis workflow on the current hypothesis and dataset"""
        # Check if we have datasets selected
        if not hasattr(self.generator_widget, 'selected_datasets') or not self.generator_widget.selected_datasets:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Dataset Selected", 
                               "Please select a dataset first before running analysis.")
            return
        
        # Get the current hypothesis text
        hypothesis_text = self.custom_hypothesis_edit.text().strip()
        if not hypothesis_text and hasattr(self.generator_widget, 'current_hypothesis_text'):
            hypothesis_text = self.generator_widget.current_hypothesis_text
        
        if not hypothesis_text:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Hypothesis", 
                               "Please enter a hypothesis to test.")
            return
            
        # Update the generator widget with the hypothesis text
        self.generator_widget.current_hypothesis_text = hypothesis_text
        
        # Get the dataset name
        dataset_name, _ = self.generator_widget.selected_datasets[0]
        print(f"Starting analysis on dataset: {dataset_name}")
        print(f"Testing hypothesis: {hypothesis_text}")
        
        # Disable the button and change text while processing
        self.run_analysis_btn.setEnabled(False)
        self.run_analysis_btn.setText("Running Analysis...")
        
        # Create a status message to update the user
        from PyQt6.QtWidgets import QLabel
        status_message = QLabel("Starting analysis...")
        status_message.setStyleSheet("color: #666; font-weight: bold;")
        status_layout = self.generator_tab.layout()
        status_layout.addWidget(status_message)
        
        # Get variable selection step reference
        var_step = None
        if hasattr(self.generator_widget, 'variable_selection_step'):
            var_step = self.generator_widget.variable_selection_step
            
            # Set the dataset and hypothesis text
            var_step.set_data(dataset_name, hypothesis_text)
        
        try:
            # Make sure we have a studies manager
            if hasattr(self, 'studies_manager') and self.studies_manager:
                # Make sure the generator widget has the studies manager
                if not hasattr(self.generator_widget, 'studies_manager') or not self.generator_widget.studies_manager:
                    status_message.setText("Setting up studies manager...")
                    self.generator_widget.set_studies_manager(self.studies_manager)
            
            # First, ensure the testing_widget is available
            testing_widget = None
            
            # Check if generator_widget has its own testing_widget
            if hasattr(self.generator_widget, 'testing_widget') and self.generator_widget.testing_widget:
                testing_widget = self.generator_widget.testing_widget
                print(f"Using testing_widget from generator_widget: {testing_widget}")
            
            # Check if var_step has a selector with testing_widget
            if not testing_widget and var_step and hasattr(var_step, 'selector') and var_step.selector:
                if hasattr(var_step.selector, 'testing_widget') and var_step.selector.testing_widget:
                    testing_widget = var_step.selector.testing_widget
                    print(f"Using testing_widget from var_step.selector: {testing_widget}")
            
            # If still no testing_widget, try to find it from the application
            if not testing_widget:
                from PyQt6.QtWidgets import QApplication
                try:
                    main_window = QApplication.instance().activeWindow()
                    if hasattr(main_window, 'data_testing_widget'):
                        testing_widget = main_window.data_testing_widget
                        print(f"Found testing_widget in main window: {testing_widget}")
                        
                        # Ensure the selector and generator widget have this testing_widget
                        if var_step and hasattr(var_step, 'setup_selector'):
                            var_step.setup_selector(testing_widget)
                        if hasattr(self.generator_widget, 'set_testing_widget'):
                            self.generator_widget.set_testing_widget(testing_widget)
                except Exception as e:
                    print(f"Error trying to find testing_widget: {e}")
            
            if testing_widget:
                # Simple, direct workflow using the testing_widget
                from PyQt6.QtWidgets import QApplication
                
                # 1. Setup the selector
                status_message.setText("Setting up analysis...")
                await asyncio.sleep(0.5)
                
                # Directly call run_workflow on the selector to start the analysis process
                if var_step and hasattr(var_step, 'selector'):
                    selector = var_step.selector
                    selector.set_dataset(dataset_name)
                    selector.set_hypothesis(hypothesis_text)
                    
                    # Make sure the selector's testing_widget is set
                    if not selector.testing_widget and testing_widget:
                        selector.testing_widget = testing_widget
                    
                    # Use the new method if it exists
                    if hasattr(var_step, 'start_analysis_workflow'):
                        # Call the new direct analysis workflow method
                        status_message.setText("Running analysis workflow...")
                        print("Using start_analysis_workflow method")
                        success = await var_step.start_analysis_workflow()
                    else:
                        # Fallback to the old method
                        status_message.setText("Running analysis workflow...")
                        print("Directly running workflow instead of clicking button")
                        success = await selector.run_workflow()
                    
                    # Get the results
                    test_result = selector.test_results if hasattr(selector, 'test_results') else None
                    
                    # Create a standard results object
                    results = {
                        "success": success,
                        "test_result": test_result,
                        "steps_completed": ["datasets_loaded", "variables_mapped", "model_built", "test_run"]
                    }
                else:
                    # Fallback if we can't access the selector
                    results = await self.generator_widget.run_analysis_workflow(dataset_name)
            else:
                # Fallback to the standard workflow
                results = await self.generator_widget.run_analysis_workflow(dataset_name)
            
            # Process results
            if results["success"]:
                status_message.setText("Analysis completed successfully!")
                
                from PyQt6.QtWidgets import QMessageBox
                
                # Safely check test_result and p_value
                test_result = results.get("test_result", {}) or {}  # Handle None case
                p_value = None
                
                # Extract p-value safely
                if isinstance(test_result, dict):
                    # Try different ways p_value might be stored
                    p_value = test_result.get('p_value')
                    if p_value is None and 'results' in test_result:
                        p_value = test_result.get('results', {}).get('p_value')
                
                # Format p-value for display
                p_value_display = "N/A"
                if p_value is not None:
                    p_value_display = f"{p_value:.4f}"
                    
                    # Add significance indication
                    if p_value < 0.05:
                        p_value_display += " (Significant)"
                    else:
                        p_value_display += " (Not significant)"
                
                # Create detailed result message
                message = (f"Analysis completed successfully!\n\n"
                          f"Dataset: {dataset_name}\n"
                          f"Hypothesis: {hypothesis_text}\n\n"
                          f"P-value: {p_value_display}\n\n"
                          f"Steps completed: {', '.join(results['steps_completed'])}")
                
                QMessageBox.information(self, "Analysis Complete", message)
                                       
                # Update confidence and state based on p-value if available
                if p_value is not None:
                    confidence = min(1.0, max(0.0, 1.0 - p_value))
                    self.confidence_spin.setValue(confidence)
                    
                    # Update state based on significance
                    if p_value < 0.05:
                        # Find the index for VALIDATED state
                        for i in range(self.state_combo.count()):
                            if self.state_combo.itemData(i) == HypothesisState.VALIDATED:
                                self.state_combo.setCurrentIndex(i)
                                break
                    else:
                        # Find the index for REJECTED state
                        for i in range(self.state_combo.count()):
                            if self.state_combo.itemData(i) == HypothesisState.REJECTED:
                                self.state_combo.setCurrentIndex(i)
                                break
                                
                # Save the test results to the hypothesis config for future reference
                self.config.test_results = test_result
                
                # Update the hypothesis state and confidence
                current_state_index = self.state_combo.currentIndex()
                if current_state_index >= 0:
                    self.config.state = self.state_combo.itemData(current_state_index)
                self.config.confidence = self.confidence_spin.value()
                
                # Save to studies manager to update the UI node
                if hasattr(self, 'studies_manager') and self.studies_manager and hasattr(self.config, 'id'):
                    print(f"Updating hypothesis node {self.config.id} in studies manager")
                    self.studies_manager.update_hypothesis(self.config.id, self.config)
                
                # Switch to the edit tab to show updated state/confidence
                self.tabs.setCurrentIndex(0)
            else:
                status_message.setText(f"Analysis failed: {results.get('error', 'Unknown error')}")
                
                from PyQt6.QtWidgets import QMessageBox
                # Format error details with completed steps
                error_detail = "Analysis failed.\n\n"
                
                if results.get('error'):
                    error_detail += f"Error: {results['error']}\n\n"
                
                if results.get('steps_completed'):
                    error_detail += f"Steps completed: {', '.join(results['steps_completed'])}"
                else:
                    error_detail += "No steps were completed."
                    
                QMessageBox.warning(self, "Analysis Failed", error_detail)
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            status_message.setText(f"Error: {str(e)}")
            
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
        finally:
            # Always force enable buttons in variable_selection_step regardless of success/failure
            if var_step:
                # Force buttons to be enabled
                if hasattr(var_step, 'save_btn'):
                    var_step.save_btn.setEnabled(True)
                if hasattr(var_step, 'hypothesis_btn'):
                    var_step.hypothesis_btn.setEnabled(True)
                if hasattr(var_step, 'interpret_btn'):
                    var_step.interpret_btn.setEnabled(True)
                
                # Force test label to update with the final test
                if testing_widget and hasattr(testing_widget, 'test_combo') and hasattr(var_step, 'test_label'):
                    test_name = testing_widget.test_combo.currentText()
                    # Only update the test label after auto_select_test has run 
                    # (this is called in the finally block, which is after all processing)
                    var_step.test_label.setText(f"Selected Test: {test_name}")
                    print(f"Final forced test label update: {test_name}")
                
                # Make results frame visible
                if hasattr(var_step, 'results_frame'):
                    var_step.results_frame.setVisible(True)
                    
                # Enable continue button
                if hasattr(var_step, 'continue_btn'):
                    var_step.continue_btn.setEnabled(True)
                
                # Force UI update
                QApplication.processEvents()
            
            # Remove status message
            status_message.setParent(None)
            status_message.deleteLater()
            
            # Re-enable the button and restore text
            self.run_analysis_btn.setEnabled(True)
            self.run_analysis_btn.setText("Run Analysis")
    
    def on_sources_selected(self, selected_sources):
        """Handle when datasets are selected in the generator's first step"""
        # Print debug information 
        print(f"Selected datasets in editor dialog: {selected_sources}")
        
        # If we have datasets selected and we're on the right step, update the variable selection
        if hasattr(self.generator_widget, 'timeline') and hasattr(self.generator_widget, 'selected_datasets'):
            # Update the generator's selected datasets
            self.generator_widget.selected_datasets = selected_sources
            
            # If we're already on step 2, update the variable selection
            if self.generator_widget.timeline.active_step == 2:
                self.generator_widget.setup_variable_selection_step()
            
            # If we're on step 1 and have selected sources, enable continue
            elif self.generator_widget.timeline.active_step == 1 and selected_sources:
                # Make sure the continue button is enabled
                if hasattr(self.generator_widget.data_sources_step, 'continue_btn'):
                    self.generator_widget.data_sources_step.continue_btn.setEnabled(True)
                    
        # Set the custom hypothesis text in the variable selection if it exists
        if self.custom_hypothesis_edit.text().strip() and hasattr(self.generator_widget, 'current_hypothesis_text'):
            self.generator_widget.current_hypothesis_text = self.custom_hypothesis_edit.text().strip()
    
    def on_hypothesis_tested(self, hypothesis_text, results):
        """Handle when a hypothesis has been fully tested"""
        print(f"Hypothesis tested: {hypothesis_text}")
        print(f"Results: {results}")
        
        # Update the hypothesis text in the editor
        self.custom_hypothesis_edit.setText(hypothesis_text)
        
        # Also update the edit tab
        self.text_edit.setPlainText(hypothesis_text)
        
        # Update the config with the hypothesis text
        self.config.text = hypothesis_text
        
        # Handle test results
        if results:
            self.on_test_completed(results)
            
            # Store results in config for later access by interpretation
            if not hasattr(self.config, 'test_results'):
                self.config.test_results = {}
            
            # Store the results in the config
            self.config.test_results = results
            
            # Store the current test key if available
            if hasattr(self.generator_widget, 'variable_selection_step') and \
               hasattr(self.generator_widget.variable_selection_step, 'selector') and \
               hasattr(self.generator_widget.variable_selection_step.selector, 'testing_widget') and \
               hasattr(self.generator_widget.variable_selection_step.selector.testing_widget, 'current_test_key'):
                
                test_key = self.generator_widget.variable_selection_step.selector.testing_widget.current_test_key
                self.config.test_key = test_key
                print(f"Stored test_key in config: {test_key}")
            
            # Force the UI to update
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
    
    def on_test_completed(self, result):
        """Handle the completion of the statistical test"""
        # Extract the p-value and update the confidence level accordingly
        if isinstance(result, dict) and 'p_value' in result:
            p_value = result['p_value']
            
            # Convert p-value to confidence level (simple inverse relationship)
            # p-value of 0.05 -> confidence of 0.95
            # p-value of 0.01 -> confidence of 0.99
            confidence = min(1.0, max(0.0, 1.0 - p_value))
            
            # Update the confidence spinner in the edit tab
            self.confidence_spin.setValue(confidence)
            
            # Set the appropriate state based on p-value
            if p_value < 0.05:
                # Find the index for VALIDATED state
                for i in range(self.state_combo.count()):
                    if self.state_combo.itemData(i) == HypothesisState.VALIDATED:
                        self.state_combo.setCurrentIndex(i)
                        break
            else:
                # Find the index for REJECTED state
                for i in range(self.state_combo.count()):
                    if self.state_combo.itemData(i) == HypothesisState.REJECTED:
                        self.state_combo.setCurrentIndex(i)
                        break
            
            # Update the hypothesis config with test results
            self.config.test_results = result
            self.config.confidence = confidence
            self.config.state = self.state_combo.itemData(self.state_combo.currentIndex())
            
            # If studies manager is available, update the hypothesis in the manager
            if hasattr(self, 'studies_manager') and self.studies_manager:
                if hasattr(self.config, 'id'):
                    print(f"Updating hypothesis {self.config.id} state in studies manager")
                    # Update the existing hypothesis node in the studies manager
                    self.studies_manager.update_hypothesis(self.config.id, self.config)
    
    def use_generated_hypothesis(self):
        """Use the selected hypothesis from the generator"""
        # Try to get the hypothesis from the variable selection step if it's been configured
        selected_text = None
        
        if hasattr(self.generator_widget, 'variable_selection_step') and \
           hasattr(self.generator_widget.variable_selection_step, 'hypothesis_text') and \
           self.generator_widget.variable_selection_step.hypothesis_text:
            selected_text = self.generator_widget.variable_selection_step.hypothesis_text
        
        # If we didn't get text from the variable selection, check if we have custom text
        if not selected_text and hasattr(self, 'custom_hypothesis_edit'):
            custom_text = self.custom_hypothesis_edit.text().strip()
            if custom_text:
                selected_text = custom_text
        
        # If we still don't have text, use the generator widget's current hypothesis
        if not selected_text and hasattr(self.generator_widget, 'current_hypothesis_text'):
            selected_text = self.generator_widget.current_hypothesis_text
        
        # Still no text? Use a default.
        if not selected_text:
            selected_text = "Higher blood pressure leads to increased risk of heart disease"
        
        if selected_text:
            # Set the hypothesis text in the edit tab
            self.text_edit.setPlainText(selected_text)
            
            # Switch to edit tab to show the update
            self.tabs.setCurrentIndex(0)
            
            # Show confirmation
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Hypothesis Updated", 
                                    "The hypothesis text has been updated. You can now edit and save it.")
    
    def get_updated_config(self):
        """Return updated hypothesis config"""
        # Update config with dialog values
        self.config.text = self.text_edit.toPlainText()
        
        # Get confidence
        self.config.confidence = self.confidence_spin.value()
        
        # Get state
        current_index = self.state_combo.currentIndex()
        self.config.state = self.state_combo.itemData(current_index)
        
        # Keep existing variables unchanged
        
        return self.config

    @asyncSlot()
    async def generate_hypothesis(self):
        """Generate a new hypothesis using LLM based on the selected variables in the testing widget"""
        # Get variable selection step and testing widget
        if not hasattr(self.generator_widget, 'variable_selection_step'):
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", "Variable selection step not available")
            return
            
        var_step = self.generator_widget.variable_selection_step
        
        # Check if var_step has selector and testing_widget
        if not hasattr(var_step, 'selector') or not hasattr(var_step.selector, 'testing_widget'):
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", "Testing widget not available")
            return
            
        testing_widget = var_step.selector.testing_widget
        
        # Get currently selected variables from testing widget
        outcome = None
        group = None
        subject_id = None
        time = None
        test_name = None
        
        if hasattr(testing_widget, 'outcome_combo') and testing_widget.outcome_combo.isEnabled():
            outcome = testing_widget.outcome_combo.currentText()
        if hasattr(testing_widget, 'group_combo') and testing_widget.group_combo.isEnabled():
            group = testing_widget.group_combo.currentText()
        if hasattr(testing_widget, 'subject_id_combo') and testing_widget.subject_id_combo.isEnabled():
            subject_id = testing_widget.subject_id_combo.currentText()
        if hasattr(testing_widget, 'time_combo') and testing_widget.time_combo.isEnabled():
            time = testing_widget.time_combo.currentText()
        if hasattr(testing_widget, 'test_combo'):
            test_name = testing_widget.test_combo.currentText()
            
        # Check if required variables are available
        if not outcome:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", "Outcome variable not selected")
            return
            
        # Set button to working state
        self.use_generated_btn.setEnabled(False)
        self.use_generated_btn.setText("Generating...")
        
        try:
            # Use testing widget's LLM function to generate hypothesis
            if hasattr(testing_widget, 'generate_hypothesis_for_test'):
                # Call in async context
                hypothesis_text = await testing_widget.generate_hypothesis_for_test(
                    outcome, group, subject_id, time, test_name)
                
                if hypothesis_text:
                    # Update the text field
                    self.custom_hypothesis_edit.setText(hypothesis_text)
                    
                    # Update the edit tab
                    self.text_edit.setPlainText(hypothesis_text)
                    
                    # Update config
                    self.config.text = hypothesis_text
                    
                    # Preserve existing state if it's already been tested
                    if not hasattr(self.config, 'state') or self.config.state == HypothesisState.UNTESTED:
                        # Set to draft state since this is a newly generated hypothesis
                        self.config.state = HypothesisState.DRAFT
                        
                        # Update the state combo box
                        for i in range(self.state_combo.count()):
                            if self.state_combo.itemData(i) == HypothesisState.DRAFT:
                                self.state_combo.setCurrentIndex(i)
                                break
                    
                    # Save to studies manager if available
                    if hasattr(self, 'studies_manager') and self.studies_manager:
                        # First check if this hypothesis already exists
                        existing_hyp = None
                        if hasattr(self.config, 'id'):
                            existing_hyp = self.studies_manager.get_hypothesis_by_id(self.config.id)
                            
                        if existing_hyp:
                            # Update existing hypothesis
                            existing_hyp.text = hypothesis_text
                            
                            # Only update state if it's untested
                            if existing_hyp.state == HypothesisState.UNTESTED:
                                existing_hyp.state = HypothesisState.DRAFT
                                
                            self.studies_manager.update_hypothesis(existing_hyp.id, existing_hyp)
                            print(f"Updated existing hypothesis {existing_hyp.id}")
                            
                            # Switch to the edit tab to show the updates
                            self.tabs.setCurrentIndex(0)
                        else:
                            # Create new hypothesis entry in studies manager
                            from plan.plan_config import HypothesisConfig, HypothesisState
                            import uuid
                            
                            # Create new hypothesis config
                            new_hyp = HypothesisConfig(
                                id=str(uuid.uuid4()),
                                text=hypothesis_text,
                                confidence=0.5,  # Default confidence
                                state=HypothesisState.DRAFT  # Set to DRAFT instead of UNTESTED
                            )
                            
                            # Copy variables from current config
                            if hasattr(self.config, 'variables'):
                                new_hyp.variables = self.config.variables
                                
                            # Add to studies manager
                            self.studies_manager.add_hypothesis(new_hyp)
                            print(f"Added new hypothesis {new_hyp.id} to studies manager")
                            
                            # Update current config with new ID and state
                            self.config.id = new_hyp.id
                            self.config.state = HypothesisState.DRAFT
                    
                    # Show confirmation
                    from PyQt6.QtWidgets import QMessageBox
                    QMessageBox.information(self, "Hypothesis Generated", 
                                          f"A new hypothesis has been generated:\n\n{hypothesis_text}")
            else:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Error", "Hypothesis generation function not available")
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to generate hypothesis: {str(e)}")
        finally:
            # Reset button state
            self.use_generated_btn.setEnabled(True)
            self.use_generated_btn.setText("Use Selected Hypothesis")
        
    # Add merged_hypothesis_with_results method to handle merging with newly generated hypotheses
    def merge_hypothesis_with_results(self, new_hypothesis_text, test_results):
        """Merge the current hypothesis with a newly generated one and update with test results"""
        # Update the text in both places
        self.custom_hypothesis_edit.setText(new_hypothesis_text)
        self.text_edit.setPlainText(new_hypothesis_text)
        
        # Update the config
        self.config.text = new_hypothesis_text
        
        # Apply test results if available
        if test_results and isinstance(test_results, dict):
            if 'p_value' in test_results:
                p_value = test_results['p_value']
                
                # Update confidence based on p-value
                confidence = min(1.0, max(0.0, 1.0 - p_value))
                self.confidence_spin.setValue(confidence)
                self.config.confidence = confidence
                
                # Update state based on significance
                if p_value < 0.05:
                    new_state = HypothesisState.VALIDATED
                else:
                    new_state = HypothesisState.REJECTED
                    
                # Find and set the state in combo box
                for i in range(self.state_combo.count()):
                    if self.state_combo.itemData(i) == new_state:
                        self.state_combo.setCurrentIndex(i)
                        break
                        
                # Update config
                self.config.state = new_state
                self.config.test_results = test_results
                
        # Save to studies manager if available
        if hasattr(self, 'studies_manager') and self.studies_manager:
            if hasattr(self.config, 'id'):
                self.studies_manager.update_hypothesis(self.config.id, self.config)
                print(f"Updated hypothesis {self.config.id} in studies manager after merging")
