WEAC Package
===========

The main WEAC package provides closed-form analytical models for the analysis of dry-snow slab avalanche release.

Package Overview
---------------

WEAC (Weak Layer Anticrack Nucleation Model) is a comprehensive Python package for analyzing snow slab mechanics and weak layer failure. The package is organized into several key modules:

**Core System Modules**
   - :doc:`weac.core` - Core computational engine and system modeling
   - :doc:`weac.components` - Data structures and configuration classes

**Analysis & Visualization**
   - :doc:`weac.analysis` - Analysis tools, criteria evaluation, and plotting

**Utilities & Configuration**
   - :doc:`weac.utils` - Helper functions, parsers, and utilities
   - :doc:`weac.logging_config` - Logging configuration

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   weac.analysis
   weac.components
   weac.core
   weac.utils

Submodules
----------

.. toctree::
   :maxdepth: 4

   weac.constants
   weac.logging_config

Module contents
---------------

.. automodule:: weac
   :members:
   :show-inheritance:
   :undoc-members:

Getting Started
--------------

To get started with WEAC, begin with the main package overview and then explore the specific modules based on your needs:

1. **New Users**: Start with :doc:`weac.components` to understand data structures
2. **System Modeling**: Use :doc:`weac.core` for computational tasks
3. **Analysis**: Apply :doc:`weac.analysis` for results processing and visualization
4. **Utilities**: Access :doc:`weac.utils` for helper functions and data parsing

For complete examples and tutorials, see the main documentation index.
