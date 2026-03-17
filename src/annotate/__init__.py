# -*- coding: utf-8 -*-
################################################################################
# annotate/__init__.py

"""The annotate package facilitates the manual annotation of the human cortex.

These tools are intended to be run using a Docker container; for information
on how to use these tools, see the README.md file in the github repository
noahbenson/cortex-annotate.
"""

# Imports ----------------------------------------------------------------------

from ._core import AnnotationTool

# Meta-Data --------------------------------------------------------------------

__version__ = "0.3.0"
__all__ = ( "AnnotationTool", )