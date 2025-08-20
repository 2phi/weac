WEAC Package Overview
====================

This page provides an overview of the WEAC package structure and its main components.

Core Package
-----------

.. toctree::
   :maxdepth: 1

   weac

Main Components
--------------

.. toctree::
   :maxdepth: 2
   :caption: Core System

   weac.core
   weac.components

.. toctree::
   :maxdepth: 2
   :caption: Analysis & Visualization

   weac.analysis

.. toctree::
   :maxdepth: 2
   :caption: Utilities & Configuration

   weac.utils
   weac.logging_config

Package Structure
----------------

The WEAC package is organized into the following main modules:

**Core System (`weac.core`)**
   - System modeling and solving
   - Eigensystem calculations
   - Field quantities computation
   - Scenario management

**Components (`weac.components`)**
   - Configuration classes
   - Layer and segment definitions
   - Model input structures
   - Scenario configurations

**Analysis (`weac.analysis`)**
   - Analysis tools and algorithms
   - Criteria evaluation
   - Plotting and visualization
   - Result processing

**Utilities (`weac.utils`)**
   - Helper functions
   - Data parsers
   - Snow type utilities
   - Miscellaneous tools

**Configuration (`weac.logging_config`)**
   - Logging setup and configuration

Quick Reference
--------------

For specific functionality, see the individual module documentation:

- **Getting Started**: :doc:`weac` - Main package overview
- **System Modeling**: :doc:`weac.core` - Core computational engine
- **Input Configuration**: :doc:`weac.components` - Data structures and configs
- **Analysis Tools**: :doc:`weac.analysis` - Analysis and visualization
- **Utilities**: :doc:`weac.utils` - Helper functions and tools
