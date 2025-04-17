#!/usr/bin/env python3
"""
Example script demonstrating advanced planning capabilities.
"""

import asyncio
import logging
import sys
import os
import json
from typing import Dict, Any

# Add the parent directory to the path so we can import the agent modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.advanced_agent import AdvancedPlanningAgent
from agent.tools import create_common_tools

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def pretty_print_results(results: Dict[str, Any]) -> None:
    """Print the results in a readable format."""
    print("\n" + "="*50)
    print(f"Execution {'succeeded' if results['success'] else 'failed'}")
    print(f"Execution time: {results['execution_time']:.2f} seconds")
    
    if not results['success']:
        print(f"Error: {results['error']}")
        return
    
    print(f"Steps executed: {results['steps_executed']}")
    
    print("\nResults:")
    for i, result in enumerate(results['results']):
        print(f"\nStep {i+1} result:")
        if isinstance(result, dict) and 'error' in result:
            print(f"  Error: {result['error']}")
        else:
            try:
                print(f"  {json.dumps(result, indent=2)}")
            except:
                print(f"  {result}")
    
    print("\nFinal result:")
    try:
        print(json.dumps(results['final_result'], indent=2))
    except:
        print(results['final_result'])
    
    print("="*50 + "\n")

async def run_hierarchical_example() -> None:
    """Run an example with hierarchical planning."""
    logger.info("Running hierarchical planning example")
    
    # Create an agent
    agent = AdvancedPlanningAgent("HierarchicalAgent")
    
    # Register tools
    tool_registry = create_common_tools()
    for tool_name in tool_registry.list_tools():
        agent.register_tool(tool_registry.get_tool(tool_name))
    
    # Define a complex task that would benefit from hierarchical planning
    task = "Research the impact of renewable energy on climate change, analyze the economic factors, and create a summary report"
    
    # Run the agent asynchronously
    results = await agent.run_async(task)
    
    # Print the results
    pretty_print_results(results)

async def run_goal_oriented_example() -> None:
    """Run an example with goal-oriented planning."""
    logger.info("Running goal-oriented planning example")
    
    # Create an agent
    agent = AdvancedPlanningAgent("GoalOrientedAgent")
    
    # Register tools
    tool_registry = create_common_tools()
    for tool_name in tool_registry.list_tools():
        agent.register_tool(tool_registry.get_tool(tool_name))
    
    # Define a goal-focused task
    task = "Determine if a patient with the following symptoms should be tested for diabetes: frequent urination, increased thirst, unexplained weight loss"
    
    # Run the agent asynchronously
    results = await agent.run_async(task)
    
    # Print the results
    pretty_print_results(results)

async def run_conditional_example() -> None:
    """Run an example with conditional planning."""
    logger.info("Running conditional planning example")
    
    # Create an agent
    agent = AdvancedPlanningAgent("ConditionalAgent")
    
    # Register tools
    tool_registry = create_common_tools()
    for tool_name in tool_registry.list_tools():
        agent.register_tool(tool_registry.get_tool(tool_name))
    
    # Define a task with potentially uncertain outcomes
    task = "Research the impact of renewable energy on climate change, analyze the economic factors, and create a summary report"
    
    # Run the agent asynchronously
    results = await agent.run_async(task)
    
    # Print the results
    pretty_print_results(results) 