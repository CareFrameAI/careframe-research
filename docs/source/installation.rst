Installation
============

Prerequisites
------------

- Python 3.9+
- PyQt6
- FastAPI
- Various scientific and data analysis libraries

Basic Installation
-----------------

You can install CareFrame using pip:

.. code-block:: bash

   pip install careframe

Or by cloning the repository:

.. code-block:: bash

   git clone https://github.com/CareFrameAI/careframe-research.git
   cd careframe-research
   pip install -e .

Installing Dependencies
----------------------

Install the required packages:

.. code-block:: bash

   pip install -r requirements.txt

For development:

.. code-block:: bash

   pip install -r requirements-dev.txt

Additional Requirements
----------------------

Some NLP models need to be downloaded separately:

.. code-block:: bash

   python -m spacy download en_core_web_sm

SciSpaCy models:

.. code-block:: bash

   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_lg-0.5.3.tar.gz
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_scibert-0.5.3.tar.gz

Docker Installation
------------------

CareFrame can also be run as a Docker container:

.. code-block:: bash

   docker-compose up -d

This will start the CareFrame application and its dependencies. 