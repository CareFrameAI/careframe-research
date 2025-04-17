import numpy as np
import pandas as pd
import os
import tempfile
import uuid
import re
import io
import traceback
from contextlib import redirect_stdout

# Force non-interactive Agg backend before any other matplotlib import
import matplotlib
matplotlib.use('Agg', force=True)
# Disable interactive features
os.environ['MPLBACKEND'] = 'Agg'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
# Now import matplotlib.pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import asyncio
import re
import os
import uuid
import tempfile
from contextlib import redirect_stdout
import traceback

# Set matplotlib to non-interactive mode to prevent pop-ups
plt.ioff()
plt.interactive(False)
plt.rcParams['interactive'] = False
# Remove invalid parameters
# plt.rcParams['figure.raise_window'] = False
# plt.rcParams['figure.show'] = False
plt.rcParams['figure.max_open_warning'] = 0

# Constants for graph detection and generation
GRAPH_INDICATORS = [
    "plt.plot", "plt.scatter", "plt.bar", "plt.hist", 
    "plt.boxplot", "plt.violinplot", "plt.pie", "plt.heatmap",
    "sns.heatmap", "sns.lineplot", "sns.scatterplot", "sns.barplot", 
    "sns.histplot", "sns.boxplot", "sns.violinplot", "sns.pairplot"
]

# Potentially dangerous functions that could cause pop-ups
BLOCKED_FUNCTIONS = [
    "plt.pause(", "plt.waitforbuttonpress", 
    "plt.draw_all", "plt.get_current_fig_manager", 
    "QtCore.QCoreApplication", "QApplication", "eval("
]

GRAPH_TYPES = [
    "Line Chart", "Scatter Plot", "Bar Chart", "Histogram", 
    "Box Plot", "Violin Plot", "Pie Chart", "Heatmap",
    "Pair Plot", "Distribution Plot", "Regression Plot", "Time Series"
]

class DataAnalysisAgent:
    """Agent for data analysis and visualization."""
    
    MAX_DEDICATED_VISUALIZATIONS = 10  # Maximum number of dedicated visualizations to generate
    
    def __init__(self, data_source=None, execution_context=None):
        """
        Initialize the data analysis agent.
        
        Args:
            data_source: Data source to analyze
            execution_context: Optional execution context
        """
        self.data_source = data_source
        self.execution_context = execution_context or {}
        self.step_count = 0
        
        # Configure matplotlib for non-interactive mode
        self.globals = {
            "plt": plt, 
            "pd": pd, 
            "np": np, 
            "sns": sns,
            "uuid": uuid,
            "os": os,
            "tempfile": tempfile
        }
        
        # Create a temporary directory for saving figures
        self.temp_dir = tempfile.mkdtemp(prefix="data_analysis_")
        self.globals["TEMP_DIR"] = self.temp_dir
        
        # Define a safe show function that saves to file instead
        def safe_show():
            """Save current figure to temp file instead of showing it"""
            fig_id = str(uuid.uuid4())[:8]
            filename = os.path.join(self.temp_dir, f"figure_{fig_id}.png")
            plt.savefig(filename)
            print(f"Figure saved to: {filename}")
            return filename
        
        # Replace plt.show with our safe version
        self.globals["plt"].show = safe_show
        
        # Ensure matplotlib is in non-interactive mode
        self.globals["plt"].ioff()
        
        self.locals = {}
        self.code_history = []
        self.graph_count = 0  # Track number of dedicated visualizations generated
        self.temp_files = []  # Keep track of temporary files
        
        # Define default state
        self.state = {
            "goals": [],
            "data_summary": None,
            "analysis_plan": [],
            "visualization_plan": [],
            "current_step": 0,
            "executed_steps": [],
            "generated_graphs": set(),  # Track which steps have had graphs generated
            "graph_steps": [],          # Track steps identified as graph-generating
            "graph_types": {},          # Map step numbers to graph types
            "skipped_visualizations": 0  # Track skipped visualizations
        }
        
        # Style configurations for visualizations
        plt.style.use('ggplot')
        
        # Set up seaborn
        sns.set(style="whitegrid")
        sns.set_palette("pastel")
        
        # Register with execution context
        if isinstance(execution_context, dict) and "register" in execution_context:
            execution_context["register"](self)

    async def execute(self, code, step_num=None, run_actual_code=True, output_callback=None, current_state=None):
        """
        Execute the specified code.
        
        Args:
            code: Code to execute
            step_num: Step number of execution
            run_actual_code: Whether to actually execute the code
            output_callback: Callback for output
            current_state: Current state to use
            
        Returns:
            Result of execution
        """
        if current_state is None:
            current_state = self.state
            
        self.step_count += 1
        current_state["current_step"] = self.step_count
        
        # Track the code
        self.code_history.append(code)
        
        # Initialize result
        result = None
        
        # Check if this is a graph-generating step
        is_graph_step = False
        graph_type = None
        
        for indicator in GRAPH_INDICATORS:
            if indicator in code:
                is_graph_step = True
                graph_type = next((t for t in GRAPH_TYPES if t.lower() in code.lower()), "General")
                if graph_type not in current_state.get("graph_types", {}):
                    current_state.setdefault("graph_types", {})[self.step_count] = graph_type
                break
                
        if is_graph_step and self.step_count not in current_state.get("graph_steps", []):
            current_state.setdefault("graph_steps", []).append(self.step_count)
        
        # Display code in output if we have a callback
        if output_callback:
            output_callback(f"\n```python\n{code}\n```\n")
        
        if run_actual_code:
            # Replace plt.show() with file saving
            modified_code = code
            if "plt.show(" in modified_code:
                modified_code = re.sub(r'plt\.show\(\)', 'plt.show()', modified_code)
                if output_callback:
                    output_callback(f"\n⚠️ Note: plt.show() calls will save figures to files instead of displaying them.\n")
            
            # Check if code contains any potentially dangerous function calls
            if any(func in modified_code for func in BLOCKED_FUNCTIONS):
                if output_callback:
                    output_callback(f"\n⚠️ WARNING: Code contains potentially unsafe functions that could cause pop-ups or infinite loops. These will be removed.\n")
                
                # Remove dangerous function calls
                safe_code = modified_code
                for func in BLOCKED_FUNCTIONS:
                    if func in safe_code:
                        pattern = rf'{re.escape(func)}\s*\(.*?\)'
                        safe_code = re.sub(pattern, '# REMOVED: ' + func + '()', safe_code, flags=re.DOTALL)
                
                modified_code = safe_code
                if output_callback:
                    output_callback(f"\nModified code:\n```python\n{modified_code}\n```\n")
            
            result = await self._execute_code(modified_code, output_callback)
                
        # If this is a graph step, generate a dedicated visualization
        if is_graph_step and result != "Error" and self.step_count not in current_state["generated_graphs"]:
            # Check if we've reached the maximum dedicated visualizations limit
            if self.graph_count >= self.MAX_DEDICATED_VISUALIZATIONS:
                if output_callback:
                    output_callback(f"\n⚠️ LIMITING DEDICATED VISUALIZATIONS: Maximum limit of {self.MAX_DEDICATED_VISUALIZATIONS} dedicated visualizations reached. Skipping visualization for step {self.step_count}.\n")
                current_state["skipped_visualizations"] += 1
            else:
                if output_callback:
                    output_callback(f"\n📈 Generating specialized visualization for step {self.step_count} ({graph_type})...\n")
                
                graph_code = await self._generate_graph(current_state, self.step_count, graph_type=graph_type)
                
                if graph_code and output_callback:
                    output_callback(f"\n```python\n{graph_code}\n```\n")
                
                if graph_code:
                    # Check the generated graph code as well
                    if any(func in graph_code for func in BLOCKED_FUNCTIONS):
                        if output_callback:
                            output_callback(f"\n⚠️ WARNING: Generated graph code contains potentially unsafe functions. These will be removed.\n")
                        
                        # Remove dangerous function calls
                        safe_code = graph_code
                        for func in BLOCKED_FUNCTIONS:
                            if func in safe_code:
                                pattern = rf'{re.escape(func)}\s*\(.*?\)'
                                safe_code = re.sub(pattern, '# REMOVED: ' + func + '()', safe_code, flags=re.DOTALL)
                        
                        # Replace plt.show() with file saving
                        if "plt.show(" in safe_code:
                            safe_code = re.sub(r'plt\.show\(\)', 'plt.show()', safe_code)
                        
                        graph_code = safe_code
                        if output_callback:
                            output_callback(f"\nModified graph code:\n```python\n{graph_code}\n```\n")
                    
                    graph_result = await self._execute_code(graph_code, output_callback)
                    if graph_result != "Error":
                        current_state["generated_graphs"].add(self.step_count)
                        self.graph_count += 1
                        
        # Add step to executed steps
        if self.step_count not in current_state["executed_steps"]:
            current_state["executed_steps"].append(self.step_count)
            
        return result

    async def _execute_code(self, code, output_callback=None):
        """
        Execute the provided code and capture output.
        
        Args:
            code: Python code to execute
            output_callback: Callback for output messages
            
        Returns:
            Result of execution or "Error"
        """
        # Set up output capture
        output_buffer = io.StringIO()
        
        try:
            # Ensure matplotlib is in non-interactive mode
            self.globals["plt"].ioff()
            
            # Ensure default variables exist in the globals
            # This helps with common variable names like 'df' that might be expected
            if 'df' not in self.locals and 'df' not in self.globals:
                for var_name, var_value in self.locals.items():
                    if isinstance(var_value, pd.DataFrame):
                        # Use the first dataframe we find as 'df' if not already defined
                        self.globals['df'] = var_value
                        if 'df' not in self.locals:
                            self.locals['df'] = var_value
                        break
            
            # Normalize code indentation to prevent errors
            import textwrap
            normalized_code = textwrap.dedent(code).strip()
            
            # For debugging, print the code being executed
            print("\nExecuting code:\n")
            print(normalized_code)
            
            # Execute the code with captured stdout
            with redirect_stdout(output_buffer):
                exec(normalized_code, self.globals, self.locals)
                
            # Update globals with locals to preserve state between executions
            # This is critical for variable persistence between steps
            self.globals.update(self.locals)
            
            # Get the captured output
            captured_output = output_buffer.getvalue()
            
            # Check if any figure files were created
            if "Figure saved to:" in captured_output:
                # Extract filename from output
                file_matches = re.findall(r'Figure saved to: (.*\.png)', captured_output)
                for filename in file_matches:
                    self.temp_files.append(filename)
                    # Notify that a figure was saved to a file
                    if output_callback:
                        output_callback(f"\n🖼️ Figure saved to temporary file: {os.path.basename(filename)}\n")
            
            # Display output if we have a callback
            if output_callback and captured_output:
                # Remove the "Figure saved to" lines from output as we already processed them
                cleaned_output = re.sub(r'Figure saved to:.*\n', '', captured_output)
                if cleaned_output.strip():
                    output_callback(f"\nOutput:\n{cleaned_output}\n")
                
            return "Success"
            
        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            
            # Display error if we have a callback
            if output_callback:
                output_callback(f"\n❌ {error_msg}\n")
                
            return "Error"
    
    def cleanup(self):
        """Clean up temporary files created during analysis."""
        for filename in self.temp_files:
            try:
                if os.path.exists(filename):
                    os.remove(filename)
            except Exception as e:
                print(f"Error removing temporary file {filename}: {e}")
        
        # Try to remove the temporary directory
        try:
            if os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)
        except Exception as e:
            print(f"Error removing temporary directory {self.temp_dir}: {e}")

    async def _generate_graph(self, state, step_number, data_description=None, graph_type=None):
        """
        Generate improved visualization code based on the current analysis.
        
        Args:
            state: Current state
            step_number: Step number to generate graph for
            data_description: Description of the data
            graph_type: Type of graph to generate
            
        Returns:
            Code for generating the graph
        """
        # Get the original code that generated the graph
        if len(self.code_history) >= step_number:
            original_code = self.code_history[step_number - 1]
        else:
            return None
            
        # Extract any DataFrame names or variables from the original code
        import re
        df_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\.(plot|hist|scatter|bar|boxplot|violinplot|pie|heatmap)'
        df_matches = re.findall(df_pattern, original_code)
        
        df_names = [match[0] for match in df_matches]
        if not df_names and 'df' in self.locals:
            df_names = ['df']  # Default dataframe name
            
        if not df_names:
            return None  # No dataframe found to visualize
            
        # Create specialized visualization based on graph type
        figure_size = "plt.figure(figsize=(10, 6))"
        
        if graph_type and graph_type.lower() == "line chart":
            code = f"""
# Ensure matplotlib is in non-interactive mode
plt.ioff()
{figure_size}
plt.title('Enhanced Line Chart from Step {step_number}')
{df_names[0]}.plot(kind='line')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save to file instead of showing
import uuid
import os
fig_id = str(uuid.uuid4())[:8]
filename = os.path.join(TEMP_DIR, f"enhanced_line_chart_{{fig_id}}.png")
plt.savefig(filename)
print(f"Figure saved to: {{filename}}")
"""
        elif graph_type and graph_type.lower() == "scatter plot":
            code = f"""
# Ensure matplotlib is in non-interactive mode
plt.ioff()
{figure_size}
plt.title('Enhanced Scatter Plot from Step {step_number}')
if len({df_names[0]}.columns) >= 2:
    x_col = {df_names[0]}.columns[0]
    y_col = {df_names[0]}.columns[1]
    sns.scatterplot(x=x_col, y=y_col, data={df_names[0]}, alpha=0.7)
    plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save to file instead of showing
import uuid
import os
fig_id = str(uuid.uuid4())[:8]
filename = os.path.join(TEMP_DIR, f"enhanced_scatter_plot_{{fig_id}}.png")
plt.savefig(filename)
print(f"Figure saved to: {{filename}}")
"""
        elif graph_type and graph_type.lower() == "bar chart":
            code = f"""
# Ensure matplotlib is in non-interactive mode
plt.ioff()
{figure_size}
plt.title('Enhanced Bar Chart from Step {step_number}')
if len({df_names[0]}.columns) >= 2:
    x_col = {df_names[0]}.columns[0]
    y_col = {df_names[0]}.columns[1]
    sns.barplot(x=x_col, y=y_col, data={df_names[0]})
plt.tight_layout()
plt.xticks(rotation=45)

# Save to file instead of showing
import uuid
import os
fig_id = str(uuid.uuid4())[:8]
filename = os.path.join(TEMP_DIR, f"enhanced_bar_chart_{{fig_id}}.png")
plt.savefig(filename)
print(f"Figure saved to: {{filename}}")
"""
        elif graph_type and graph_type.lower() == "histogram":
            code = f"""
# Ensure matplotlib is in non-interactive mode
plt.ioff()
{figure_size}
plt.title('Enhanced Histogram from Step {step_number}')
for col in {df_names[0]}.select_dtypes(include=['number']).columns[:3]:
    sns.histplot({df_names[0]}[col], kde=True, label=col)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save to file instead of showing
import uuid
import os
fig_id = str(uuid.uuid4())[:8]
filename = os.path.join(TEMP_DIR, f"enhanced_histogram_{{fig_id}}.png")
plt.savefig(filename)
print(f"Figure saved to: {{filename}}")
"""
        elif graph_type and "box" in graph_type.lower():
            code = f"""
# Ensure matplotlib is in non-interactive mode
plt.ioff()
{figure_size}
plt.title('Enhanced Box Plot from Step {step_number}')
sns.boxplot(data={df_names[0]}.select_dtypes(include=['number']))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.xticks(rotation=45)

# Save to file instead of showing
import uuid
import os
fig_id = str(uuid.uuid4())[:8]
filename = os.path.join(TEMP_DIR, f"enhanced_boxplot_{{fig_id}}.png")
plt.savefig(filename)
print(f"Figure saved to: {{filename}}")
"""
        elif graph_type and "violin" in graph_type.lower():
            code = f"""
# Ensure matplotlib is in non-interactive mode
plt.ioff()
{figure_size}
plt.title('Enhanced Violin Plot from Step {step_number}')
sns.violinplot(data={df_names[0]}.select_dtypes(include=['number']))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.xticks(rotation=45)

# Save to file instead of showing
import uuid
import os
fig_id = str(uuid.uuid4())[:8]
filename = os.path.join(TEMP_DIR, f"enhanced_violinplot_{{fig_id}}.png")
plt.savefig(filename)
print(f"Figure saved to: {{filename}}")
"""
        elif graph_type and "heat" in graph_type.lower():
            code = f"""
# Ensure matplotlib is in non-interactive mode
plt.ioff()
{figure_size}
plt.title('Enhanced Heatmap from Step {step_number}')
numeric_df = {df_names[0]}.select_dtypes(include=['number'])
if numeric_df.shape[1] > 1:
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
else:
    plt.text(0.5, 0.5, "Not enough numeric columns for correlation", 
             horizontalalignment='center', verticalalignment='center')
plt.tight_layout()

# Save to file instead of showing
import uuid
import os
fig_id = str(uuid.uuid4())[:8]
filename = os.path.join(TEMP_DIR, f"enhanced_heatmap_{{fig_id}}.png")
plt.savefig(filename)
print(f"Figure saved to: {{filename}}")
"""
        elif graph_type and "pair" in graph_type.lower():
            code = f"""
# Ensure matplotlib is in non-interactive mode
plt.ioff()
plt.figure(figsize=(12, 10))
plt.title('Enhanced Pair Plot from Step {step_number}')
numeric_df = {df_names[0]}.select_dtypes(include=['number'])
if numeric_df.shape[1] > 1:
    sns.pairplot(numeric_df.iloc[:, :5])  # Limit to first 5 columns
plt.tight_layout()

# Save to file instead of showing
import uuid
import os
fig_id = str(uuid.uuid4())[:8]
filename = os.path.join(TEMP_DIR, f"enhanced_pairplot_{{fig_id}}.png")
plt.savefig(filename)
print(f"Figure saved to: {{filename}}")
"""
        else:
            # Default enhanced visualization
            code = f"""
# Ensure matplotlib is in non-interactive mode
plt.ioff()
{figure_size}
plt.title('Enhanced Visualization from Step {step_number}')
# Try to create a meaningful visualization based on dataframe structure
df = {df_names[0]}
if len(df.columns) > 0:
    # For numeric data
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        if len(numeric_cols) == 1:
            # Single numeric column - histogram
            sns.histplot(df[numeric_cols[0]], kde=True)
            plt.xlabel(numeric_cols[0])
        elif len(numeric_cols) == 2:
            # Two numeric columns - scatter plot
            sns.scatterplot(x=numeric_cols[0], y=numeric_cols[1], data=df)
        else:
            # Multiple numeric columns - correlation heatmap
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    else:
        # For categorical data
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            # Count plot of first categorical column
            sns.countplot(y=cat_cols[0], data=df, order=df[cat_cols[0]].value_counts().index)
            plt.ylabel(cat_cols[0])
            
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save to file instead of showing
import uuid
import os
fig_id = str(uuid.uuid4())[:8]
filename = os.path.join(TEMP_DIR, f"enhanced_visualization_{{fig_id}}.png")
plt.savefig(filename)
print(f"Figure saved to: {{filename}}")
"""
        
        return code 

    def get_plan_depth(self):
        """Get the configured plan depth from globals or use default."""
        return self.globals.get('PLAN_DEPTH', 5)  # Default to 5 if not specified
    
    async def _get_analysis_plan(self, task):
        """Generate an analysis plan based on the task.
        
        Args:
            task: The analysis task description
            
        Returns:
            A list of plan steps or a dictionary with plan and visualization information
        """
        # Get the configured plan depth
        plan_depth = self.get_plan_depth()
        
        # Implementation would depend on how you're generating plans
        # For demonstration purposes, generate a simple plan
        plan = [
            f"Step 1: Load and examine the data structure",
            f"Step 2: Generate summary statistics for key columns",
            f"Step 3: Create visualizations to understand distributions",
            f"Step 4: Analyze relationships between variables",
            f"Step 5: Summarize findings and insights"
        ]
        
        # Add more steps if plan_depth is greater than 5
        for i in range(6, plan_depth + 1):
            plan.append(f"Step {i}: Perform additional analysis based on findings")
        
        # Or truncate if plan_depth is less than 5
        if plan_depth < 5:
            plan = plan[:plan_depth]
            
        # Mock visualization identification
        graph_steps = [3]  # Step 3 is visualization by default
        if plan_depth >= 4:
            graph_steps.append(4)  # Add step 4 if depth allows
            
        # Add one more visualization step for deeper plans
        if plan_depth >= 6:
            graph_steps.append(6)
            
        # Return as a dictionary with plan and visualization info
        return {
            "plan": "\n".join(plan),
            "graph_steps": graph_steps,
            "graph_count": len(graph_steps),
            "graph_types": ["Distribution Plot" for _ in graph_steps]
        } 