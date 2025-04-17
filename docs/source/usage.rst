Usage
=====

Starting the Application
-----------------------

After installation, you can start CareFrame by running:

.. code-block:: bash

   python app.py

Or if you've installed it via pip:

.. code-block:: bash

   careframe

User Interface
-------------

CareFrame's user interface is organized into several main sections:

- **Research Planning**: Design studies and generate hypotheses
- **Literature**: Search and organize research literature
- **Data**: Collect, clean, and analyze data
- **Analysis**: Perform statistical analyses
- **Exchange**: Share and validate data using blockchain

Research Planning
----------------

The Research Planning section allows you to:

1. Define research questions
2. Generate hypotheses
3. Plan study designs
4. Create protocols

Literature Search
----------------

The Literature Search section provides tools to:

1. Search for relevant publications
2. Rank papers by relevance
3. Extract evidence from papers
4. Organize references

Data Management
--------------

The Data section includes:

1. **Collection**: Import data from various sources
2. **Cleaning**: Clean and preprocess data
3. **Reshaping**: Transform data for analysis
4. **Joining**: Combine datasets
5. **Filtering**: Filter data based on criteria
6. **Testing**: Test hypotheses on the data

Analysis
-------

The Analysis section offers:

1. **Interpretation**: Interpret statistical results
2. **Evaluation**: Evaluate test results
3. **Assumptions**: Check statistical assumptions
4. **Subgroup Analysis**: Analyze specific subgroups
5. **Mediation Analysis**: Investigate mediation effects
6. **Sensitivity Analysis**: Test robustness of results

AI Assistant
-----------

The built-in AI Assistant can help with:

1. Literature search and summarization
2. Data analysis suggestions
3. Research question formulation
4. Code generation for custom analyses

Command-line Interface
---------------------

CareFrame also provides a command-line interface for automation:

.. code-block:: bash

   # Run the server only
   python server.py

   # Run with specific options
   python app.py --headless --port 8889 