import sys
import os


from llms.client import call_llm_sync, call_llm_async
import asyncio
import json
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTextEdit, QPushButton, QLabel, 
                           QSplitter, QScrollArea, QFrame, QFileDialog)
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QThread
import matplotlib.pyplot as plt
from io import StringIO
import traceback
import ast
import re

class DataAnalysisAgent:
    """Agent for performing iterative data analysis based on user queries."""
    
    def __init__(self):
        self.dataframes = {}  # Storage for dataframes between executions
        self.execution_history = []
        self.step_count = 0
        
    async def execute(self, task, output_callback=None):
        """Execute the analysis task step by step."""
        self.step_count = 0
        
        # Initial planning step
        plan = await self._get_analysis_plan(task)
        if output_callback:
            output_callback(f"Analysis Plan:\n{plan}\n")
        
        # Track our current state and progress
        current_state = {
            "task": task,
            "plan": plan,
            "dataframes": {},
            "current_step": 0,
            "completed_steps": [],
            "outputs": []
        }
        
        # Execute steps until completion
        while True:
            self.step_count += 1
            if output_callback:
                output_callback(f"\nExecuting Step {self.step_count}...\n")
            
            # Get next code to execute
            code_to_execute = await self._get_next_code(current_state)
            
            if "ANALYSIS_COMPLETE" in code_to_execute:
                if output_callback:
                    output_callback("\nAnalysis Complete\n")
                break
                
            # Show the code we're about to execute
            if output_callback:
                output_callback(f"```python\n{code_to_execute}\n```\n")
                
            # Execute the code
            result, output, error = self._execute_code(code_to_execute)
            
            # Update state with execution results
            current_state["outputs"].append({
                "step": self.step_count,
                "code": code_to_execute,
                "output": output,
                "error": error
            })
            
            # Print output and errors
            if output and output_callback:
                output_callback(f"Output:\n{output}\n")
            if error and output_callback:
                output_callback(f"Error:\n{error}\n")
                
            # Reflect on the results and plan next step
            reflection = await self._reflect_on_execution(current_state)
            if output_callback:
                output_callback(f"Reflection:\n{reflection}\n")
                
            current_state["completed_steps"].append({
                "step": self.step_count,
                "code": code_to_execute,
                "output": output,
                "error": error,
                "reflection": reflection
            })
            current_state["current_step"] += 1
            
        # Final summary
        summary = await self._get_final_summary(current_state)
        if output_callback:
            output_callback(f"\nFinal Summary:\n{summary}\n")
            
        return summary
        
    async def _get_analysis_plan(self, task):
        """Generate an analysis plan for the given task."""
        
        # Simplify to use a plain string prompt instead of messages
        prompt = f"""[SYSTEM] You are an expert data analyst creating structured analysis plans. Focus on clarity, feasibility, and methodical steps.

[TASK] Create a detailed analysis plan for this user request: "{task}"

[INSTRUCTIONS]
1. Generate a numbered step-by-step plan (5-7 steps) that addresses the full task
2. Each step should be clear, specific, and actionable
3. Include appropriate data preparation, analysis, and visualization steps
4. Start with data exploration/understanding before complex analysis
5. Cover necessary data cleaning and preprocessing
6. Include relevant statistical analyses and visualizations
7. End with conclusions or summary steps

[FORMAT]
Return a clear numbered list with brief descriptions. For example:
1. Load and explore the dataset structure
2. Clean missing values and outliers
3. Perform exploratory data analysis with visualizations
4. Run statistical tests to verify hypotheses
5. Create final visualizations of key findings
6. Summarize insights and conclusions

[OUTPUT] Provide ONLY the numbered analysis plan without additional commentary."""
        
        response_json = await call_llm_async(prompt, model="claude-3-7-sonnet-20250219")
        try:
            return json.loads(response_json)["plan"]
        except:
            # Fallback to extract plan from text
            return self._extract_plan_from_text(response_json)
            
    def _extract_plan_from_text(self, text):
        """Extract plan from text if JSON parsing fails."""
        if "Step 1:" in text or "1." in text:
            return text
        else:
            return "Could not generate structured plan. Proceeding with analysis."
    
    async def _get_next_code(self, state):
        """Generate the next code to execute based on the current state."""
        # Create a summary of available dataframes
        df_summary = ""
        for name, df in self.dataframes.items():
            if isinstance(df, pd.DataFrame):
                df_summary += f"\nDataFrame '{name}': {df.shape[0]} rows, {df.shape[1]} columns"
                df_summary += f"\nColumns: {list(df.columns)[:10]}"
                if len(df.columns) > 10:
                    df_summary += f" and {len(df.columns) - 10} more"
                df_summary += f"\nSample data: {df.head(2).to_dict()}\n"
        
        # Build execution history
        history = ""
        for step in state["completed_steps"]:
            history += f"Step {step['step']}:\n```python\n{step['code']}\n```\n"
            if step["output"]:
                history += f"Output: {step['output'][:200]}"
                if len(step["output"]) > 200:
                    history += "...(truncated)"
                history += "\n"
            if step["error"]:
                history += f"Error: {step['error']}\n"
        
        # Improved structured prompt
        prompt = f"""[SYSTEM] You are a Python data science code generator. Generate only executable Python code for the next analysis step.

[CONTEXT]
- Task: {state['task']}
- Current analysis plan: {state['plan']}
- Current step number: {state["current_step"] + 1}
- Available dataframes: {df_summary}

[EXECUTION HISTORY]
{history}

[INSTRUCTIONS]
1. Generate VALID Python code for ONLY the next step in the analysis plan
2. Use proper error handling with try/except blocks
3. Include informative print statements to show progress and key results
4. Reference existing dataframes by name if they exist
5. Create clear, informative visualizations when appropriate
6. MUST print 'ANALYSIS_COMPLETE' if this is the final step in the plan

[CODE REQUIREMENTS]
- Use pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, or statsmodels as needed
- Include ALL necessary imports at the top of the code
- Create self-contained code for the current step only
- Save outputs to variables that will be accessible in later steps
- Include descriptive comments to explain complex operations
- Use consistent variable naming that aligns with previous steps
- Format visualizations with titles, labels, and legends

[OUTPUT]
Return ONLY the Python code with no extra text, explanations, or markdown.
"""
        
        response_json = await call_llm_async(prompt, model="claude-3-7-sonnet-20250219")
        try:
            code = json.loads(response_json)["code"]
        except:
            # Extract code block if JSON parsing fails
            code = self._extract_code_from_text(response_json)
        
        return code
    
    def _extract_code_from_text(self, text):
        """Extract code block from text if JSON parsing fails."""
        # Look for code block between triple backticks
        code_match = re.search(r'```(?:python)?(.*?)```', text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        else:
            # Return the whole text if no code block found
            return text
    
    def _execute_code(self, code):
        """Execute the generated code and capture outputs."""
        # Create string buffer to capture print outputs
        output_buffer = StringIO()
        error_message = None
        result = None
        
        # Get the current dataframes into local variables
        local_vars = self.dataframes.copy()
        
        try:
            # Make dataframes directly accessible by name in the execution scope
            # This allows the code to use dataframe names directly without self.dataframes references
            for df_name, df_value in self.dataframes.items():
                exec(f"{df_name} = local_vars['{df_name}']")
            
            # Execute code and capture standard output
            sys.stdout = output_buffer
            exec(code, globals(), local_vars)
            result = "Success"
            
            # Get any new or modified dataframes
            for var_name, var_value in local_vars.items():
                if isinstance(var_value, pd.DataFrame):
                    self.dataframes[var_name] = var_value
            
        except Exception as e:
            error_message = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            result = "Error"
        finally:
            # Restore standard output
            sys.stdout = sys.__stdout__
        
        return result, output_buffer.getvalue(), error_message
    
    async def _reflect_on_execution(self, state):
        """Reflect on the execution results and suggest course corrections if needed."""
        last_output = state["outputs"][-1] if state["outputs"] else {}
        
        # Structured reflection prompt
        prompt = f"""[SYSTEM] You are an expert data analyst reviewing code execution results. Provide concise, focused reflections.

[CONTEXT]
- Task: {state['task']}
- Code executed:
```python
{last_output.get('code', 'No code executed')}
```
- Output: {last_output.get('output', 'No output')}
- Errors: {last_output.get('error', 'No errors')}

[INSTRUCTIONS]
Analyze the execution results and provide a concise reflection addressing:
1. Whether the step was successful or not (be explicit)
2. Key insights or findings discovered in the data
3. Any issues that need to be addressed in the next step
4. Whether the analysis is progressing as expected
5. Specific corrections or adjustments needed for the next step

[FORMAT]
Provide a BRIEF reflection (2-4 sentences) focusing on:
- Success/failure assessment
- Key findings
- Next steps recommendation
- Any warning signs or issues to address

[OUTPUT]
Return ONLY your concise reflection without additional commentary, markdown formatting, or explanations."""
        
        response_json = await call_llm_async(prompt, model="claude-3-7-sonnet-20250219")
        try:
            return json.loads(response_json)["reflection"]
        except:
            # Return raw text if JSON parsing fails
            return response_json
    
    async def _get_final_summary(self, state):
        """Generate a final summary of the analysis."""
        # Build execution history
        history = ""
        for step in state["completed_steps"]:
            history += f"Step {step['step']}:\n```python\n{step['code']}\n```\n"
            if step["output"]:
                output_preview = step["output"][:300]
                if len(step["output"]) > 300:
                    output_preview += "...(truncated)"
                history += f"Output: {output_preview}\n"
            if step["reflection"]:
                history += f"Reflection: {step['reflection']}\n\n"
        
        # Structured final summary prompt
        prompt = f"""[SYSTEM] You are an expert data scientist synthesizing analysis results. Create a comprehensive, well-structured summary.

[CONTEXT]
- Original task: {state['task']}
- Analysis plan executed: {state['plan']}
- Execution steps and outputs:
{history}

[INSTRUCTIONS]
Create a comprehensive summary of the data analysis that includes:
1. An overview of what was accomplished (datasets analyzed, methods used)
2. The most significant findings and insights discovered
3. Quantitative results and statistical significance where applicable
4. Visual findings and pattern interpretations
5. Limitations of the analysis and potential sources of error/bias
6. Business implications or actionable recommendations based on findings
7. Potential next steps or further analyses that could build on these results

[FORMAT]
Structure your summary with clear sections:
- Executive Summary: 1-2 sentence high-level overview
- Methodology: Brief description of analysis approach
- Key Findings: 3-5 bullet points of most important discoveries
- Detailed Results: More in-depth description of findings
- Limitations: Honest assessment of constraints
- Recommendations: Actionable next steps
- Conclusion: Brief wrap-up

[OUTPUT]
Provide a clear, concise, and data-driven summary that would be valuable to a business stakeholder."""
        
        response_json = await call_llm_async(prompt, model="claude-3-7-sonnet-20250219")
        try:
            return json.loads(response_json)["summary"]
        except:
            # Return raw text if JSON parsing fails
            return response_json

class AgentWorker(QObject):
    """Worker thread for running the agent asynchronously."""
    finished = pyqtSignal()
    progress = pyqtSignal(str)
    
    def __init__(self, agent, task):
        super().__init__()
        self.agent = agent
        self.task = task
        
    async def run_agent(self):
        """Run the agent and emit progress signals."""
        try:
            await self.agent.execute(self.task, self.progress.emit)
        except Exception as e:
            self.progress.emit(f"Error: {str(e)}\n{traceback.format_exc()}")
        finally:
            self.finished.emit()
    
    def run(self):
        """Run the agent in the current thread."""
        asyncio.run(self.run_agent())

class DataAnalysisApp(QMainWindow):
    """PyQt6 application for the data analysis agent."""
    
    def __init__(self):
        super().__init__()
        self.agent = DataAnalysisAgent()
        self.loaded_files = []
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Data Analysis Agent")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Upload file area
        file_label = QLabel("Upload Data Files:")
        self.file_list = QTextEdit()
        self.file_list.setMaximumHeight(80)
        self.file_list.setReadOnly(True)
        
        # Upload button
        upload_button = QPushButton("Upload CSV/TSV")
        upload_button.clicked.connect(self.upload_file)
        
        # File layout
        file_layout = QHBoxLayout()
        file_layout.addWidget(file_label)
        file_layout.addWidget(upload_button)
        
        # Task input area
        task_label = QLabel("Enter your data analysis task:")
        self.task_input = QTextEdit()
        self.task_input.setMinimumHeight(100)
        self.task_input.setPlaceholderText("Describe the data analysis task you want to perform...")
        
        # Example task button
        example_button = QPushButton("Insert Example Task")
        example_button.clicked.connect(self.insert_example_task)
        
        # Execute button
        execute_button = QPushButton("Run Analysis")
        execute_button.clicked.connect(self.execute_task)
        
        # Clear button
        clear_button = QPushButton("Clear Outputs")
        clear_button.clicked.connect(self.clear_outputs)
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(example_button)
        button_layout.addWidget(execute_button)
        button_layout.addWidget(clear_button)
        
        # Output area
        output_label = QLabel("Analysis Output:")
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        
        # Add widgets to main layout
        main_layout.addLayout(file_layout)
        main_layout.addWidget(self.file_list)
        main_layout.addWidget(task_label)
        main_layout.addWidget(self.task_input)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(output_label)
        main_layout.addWidget(self.output_text)
        
        self.show()
        
    def upload_file(self):
        """Open file dialog to upload CSV or TSV files."""
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("Data files (*.csv *.tsv)")
        
        if file_dialog.exec():
            filenames = file_dialog.selectedFiles()
            for filename in filenames:
                # Process each file
                try:
                    file_base = os.path.basename(filename)
                    df_name = os.path.splitext(file_base)[0].replace(" ", "_")
                    
                    # Determine separator based on file extension
                    if filename.lower().endswith('.csv'):
                        df = pd.read_csv(filename)
                        file_type = 'CSV'
                    elif filename.lower().endswith('.tsv'):
                        df = pd.read_csv(filename, sep='\t')
                        file_type = 'TSV'
                    else:
                        continue
                    
                    # Store the dataframe in the agent
                    self.agent.dataframes[df_name] = df
                    
                    # Add to list of loaded files
                    self.loaded_files.append((file_base, df_name, file_type))
                    self.update_file_list()
                    
                    # Show confirmation message
                    self.output_text.append(f"Loaded {file_type} file: {file_base} as dataframe '{df_name}' "
                                          f"with {df.shape[0]} rows and {df.shape[1]} columns\n")
                    
                except Exception as e:
                    self.output_text.append(f"Error loading file {filename}: {str(e)}\n")
    
    def update_file_list(self):
        """Update the displayed list of loaded files."""
        file_info = []
        for file_base, df_name, file_type in self.loaded_files:
            df = self.agent.dataframes.get(df_name)
            if df is not None:
                file_info.append(f"{file_base} ({file_type}, {df.shape[0]} rows, {df.shape[1]} cols) as '{df_name}'")
        
        self.file_list.setText("\n".join(file_info))
        
    def insert_example_task(self):
        """Insert an example task in the input field."""
        if self.loaded_files:
            # Create task based on first loaded file
            file_base, df_name, file_type = self.loaded_files[0]
            example_task = f"""Analyze the {df_name} dataset:
1. Explore the data structure and key statistics
2. Clean any missing or invalid data
3. Visualize the key relationships between important variables
4. Identify any interesting patterns or outliers
5. Summarize your findings"""
        else:
            example_task = """Analyze the Titanic dataset to identify factors that influenced survival rates.
1. Load the titanic dataset from seaborn
2. Explore the data structure and key statistics
3. Visualize the relationship between survival and key variables like age, gender, and passenger class
4. Build a simple predictive model to identify the most important factors
5. Summarize your findings"""
        
        self.task_input.setText(example_task)
        
    def execute_task(self):
        """Execute the data analysis task."""
        task = self.task_input.toPlainText().strip()
        if not task:
            self.output_text.append("Please enter a task description.")
            return
            
        # Clear previous output
        self.output_text.clear()
        self.output_text.append(f"Starting analysis: {task}\n")
        
        # List available datasets
        if self.loaded_files:
            self.output_text.append("Available datasets:\n")
            for file_base, df_name, file_type in self.loaded_files:
                df = self.agent.dataframes.get(df_name)
                self.output_text.append(f"- '{df_name}': {df.shape[0]} rows, {df.shape[1]} columns\n")
        
        # Create worker thread
        self.worker_thread = QThread()
        self.worker = AgentWorker(self.agent, task)
        self.worker.moveToThread(self.worker_thread)
        
        # Connect signals
        self.worker.progress.connect(self.update_output)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker_thread.started.connect(self.worker.run)
        
        # Start the worker thread
        self.worker_thread.start()
        
    def update_output(self, text):
        """Update the output text area with new content."""
        self.output_text.append(text)
        # Scroll to the bottom
        scrollbar = self.output_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def clear_outputs(self):
        """Clear the output text area."""
        self.output_text.clear()

