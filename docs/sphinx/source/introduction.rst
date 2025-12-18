Introduction
============

Overview of AQUA-diagnostics
----------------------------

AQUA-diagnostics is a dedicated module of AQUA (Climate DT Applications for QUality Assessment)
for running and managing diagnostics on low-resolution and high-resolution climate model dat.
This module provides the tools and interfaces needed to configure, launch, and organise
diagnostic analyses.

.. note::
   For detailed information about AQUA's internal mechanisms (data reading, catalog management,
   regridding, etc.), please refer to the documentation of `AQUA-core`, which contains all core
   functions and base classes.

The main repository for AQUA is the `AQUA-core <https://github.com/DestinE-Climate-DT/AQUA>`_ repository.
The main repository for AQUA-diagnostics is the `AQUA-diagnostics <https://github.com/DestinE-Climate-DT/AQUA-diagnostics>`_ repository.

The AQUA-diagnostics repository contains the full set of diagnostic tools developed for the
Destination Earth Adaptation Climate Digital Twin (ClimateDT).
It is designed to be used together with the AQUA core framework, which provides data access
and preprocessing functionalities.

The design of the package is based on a clear separation between Python classes handling
the execution of analyses and those dedicated to visualization and plotting, allowing
computational and graphical representation to be managed independently.

AQUA-diagnostics provides different groups of diagnostics, including state-of-the-art diagnostics,
frontier diagnostics, and ensemble-based approaches.

These can be combined into diagnostic suites addressing specific components or thematic
aspects of the climate system (e.g. radiation, surface energy balance). Each suite brings
together analyses that describe model outputs across variables, spatial domains and
temporal scales.

- For more information on AQUA-core, see the AQUA-core documentation here: https://aqua.readthedocs.io/en/latest/.  

Purpose
-------

The purpose of AQUA-diagnostics is to offer users a centralized environment to:

- Select and configure available diagnostics
- Run analyses on climate model outputs
- Organize and access diagnostic results

Key Features
------------

- Wide selection of ready-to-use diagnostics for climate model evaluation
- Support for both state-of-the-art diagnostics on reduced-resolution data and frontier diagnostics on native high-resolution outputs
- Simple and customizable command-line interface
- Automated management of configuration and analysis parameters
- Support for parallel execution of diagnostics, including HPC cluster integration
- Organized output structure for easy result retrieval
- Integration with AQUA-core for data access and core functionalities

Contributing
------------

AQUA-diagnostics is developed within the European Union Contract 
`DE_340_CSC - Destination Earth Programme Climate Adaptation Digital Twin (Climate DT)`.  
Contributions are welcome via the GitHub repository.  
Please refer to the ``CONTRIBUTING.md`` file for guidelines and further information.