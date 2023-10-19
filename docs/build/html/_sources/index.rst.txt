.. tidyX documentation master file, created by
   sphinx-quickstart on Fri Sep 22 00:46:38 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to tidyX's documentation!
=================================

tidyX is a Python package developed to provide various utilities for text preprocessing and visualization. It leverages spaCy for several natural language processing tasks and provides a seamless interface for users to interact with and visualize text data.

Installation
------------

To install tidyX, you can use pip. Run the following command in your terminal or command prompt:

.. code-block:: sh

   pip install tidyX

Usage
-----
In the tutorial below, you will find examples for using each function within our package. Additionally, there's a tutorial on Topic Modelling utilizing this package.

.. toctree::
   :maxdepth: 2
   :caption: How to use this package?:

   examples/tutorial
   
.. toctree::
   :maxdepth: 3
   :caption: User Documentation:

   api/TextPreprocessor
   api/SpacyPreprocessor
   api/TextVisualizer