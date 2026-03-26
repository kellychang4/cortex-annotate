# -*- coding: utf-8 -*-
################################################################################
# annotate/config/_annotations.py

"""Annotation specification configuration for cortex-annotate.

AnnotationsConfig is a dict subclass that parses the ``annotations`` section
of config.yaml. Each annotation defines a contour, boundary, or point to be
drawn on each target, along with its figure grid layout, fixed point
dependencies, and optional target filter.

The parsed annotations are stored as Annotation namedtuples and indexed by
annotation name. Derived lookup dicts (type, figure_grid, grid_shape,
fixed_heads, fixed_tails, fixed_points, fixed_dependencies) are computed
at construction time for easy access.
"""

# Imports ----------------------------------------------------------------------

import numpy as np
from functools import partial
from collections import namedtuple

from ._error import ConfigError

# Annotation <namedtuple> ------------------------------------------------------

Annotation = namedtuple(
    typename    = "Annotation",
    field_names = ( "type", "fixed_head", "fixed_tail", "figure_grid", "filter" ),
    defaults    = ( "contour", None, None, None, None )
)
Annotation.__doc__ = """A record describing a single annotation specification from config.yaml.

Fields:
    type: 'contour', 'boundary', or 'point'
    fixed_head: dict with 'calculate' and 'requires' keys, or None
    fixed_tail: dict with 'calculate' and 'requires' keys, or None  
    figure_grid: list of lists of figure name strings (or None)
    filter: fn(target) → bool, or None
"""

# Annotation Configuration -----------------------------------------------------

class AnnotationsConfig(dict):
    """Parsed annotation specifications from config.yaml.

    Parameters
    ----------
    annotations_yaml : dict
        The ``annotations`` section from config.yaml. Each key is an
        annotation name; values are either a figure_grid list or a full
        annotation specification mapping.

    init : InitConfig
        The init environment, used to compile filter and fixed point
        calculation functions.

    Attributes
    ----------
    names : list of str
        Ordered annotation names.

    type : dict of {str: str}
        Maps annotation name → type ('contour', 'boundary', 'point').

    figure_grid : dict of {str: list of list}
        Maps annotation name → figure grid matrix.

    grid_shape : dict of {str: tuple}
        Maps annotation name → (rows, cols) shape of its figure grid.

    fixed_head : dict of {str: dict or None}
        Maps annotation name → fixed_head info dict, or None.

    fixed_tail : dict of {str: dict or None}
        Maps annotation name → fixed_tail info dict, or None.

    fixed_heads : dict of {str: list of str}
        Maps annotation name → list of annotation names required by its head.

    fixed_tails : dict of {str: list of str}
        Maps annotation name → list of annotation names required by its tail.

    fixed_points : dict of {str: list of str}
        Maps annotation name → combined list of all required annotations.

    fixed_dependencies : dict of {str: list of str}
        Reverse mapping: annotation name → annotations that depend on it.
        
    figure_names : set of str
        All unique figure names referenced across all annotation grids.
    """
    
    __slots__ = ( 
        "names", "type", "figure_grid", "grid_shape", 
        "fixed_head", "fixed_tail", "fixed_heads", "fixed_tails",
        "fixed_points", "fixed_dependencies", "figure_names"
    )
    
    def __init__(self, annotations_yaml, init):
        # Validate the annotations YAML. 
        annotations_yaml = self._validate_annotations_yaml(annotations_yaml)

        # Go through and build up the annotation dictionary.
        annotations_dict = {}
        for key, value in annotations_yaml.items():
            # Prepare ConfigError arguments for errors that arise in this loop.
            err = partial(ConfigError, f"annotations.`{key}`")

            # Check that the annotation value is a list or mapping.
            if not isinstance(value, (list, dict)):
                raise err(f"annotation `{key}` must be a list or mapping.")
        
            if isinstance(value, list):
                # If the value is a list, then this is treated as a figure_grid.
                figure_grid = self._init_figure_grid(value, err)
                annotations_dict[key] = Annotation(figure_grid = figure_grid)
            else: 
                # If the value is a mapping, then this is treated as an annotation
                # specification that is processed by the `_init_annotation` method.
                annotations_dict[key] = self._init_annotation(key, value, init)
        
        # And now all the annotations are processed, update the dictionary.
        self.update(annotations_dict)

        # Prepare annotation attribute information for easy access later.
        # These are static once the annotation are configured, so compute only once here.
        self.names       = list(self.keys())
        self.type        = self._get_type()
        self.figure_grid = self._get_figure_grid()
        self.grid_shape  = self._get_grid_shape()
        self.fixed_head  = self._get_fixed_info("fixed_head")
        self.fixed_tail  = self._get_fixed_info("fixed_tail")
        self.fixed_heads = self._get_fixed_names("fixed_head")
        self.fixed_tails = self._get_fixed_names("fixed_tail")

        # Combine the fixed heads and tails into one dictionary for ease.
        # <annotation> : [ <fixed_head>, <fixed_tail> ].
        self.fixed_points = {
            k: [ *self.fixed_heads[k], *self.fixed_tails[k] ] for k in self.keys()
        }

        # Create the fixed dependencies dictionary, which is the reverse of the 
        # fixed points dictionary.
        # <annotations> : [ <annotations that have downstream dependencies> ]
        self.fixed_dependencies = { k: [] for k in self.keys() }
        for key in self.fixed_dependencies.keys():
            for src, value in self.fixed_points.items():
                if key in value: self.fixed_dependencies[key].append(src)

        # Finally, we get all the unique figure names.
        self.figure_names = set([
            x 
            for annotation in self.values()
            for row in annotation.figure_grid
            for x in row if x is not None
        ])

    # Validate YAML ------------------------------------------------------------

    @staticmethod
    def _validate_annotations_yaml(annotations_yaml):
        """DOCSTRING."""
        # Prepare ConfigError for any errors that arise in this function.
        err = partial(ConfigError, "annotations")

        # The annotations section is required.
        if annotations_yaml is None:
            raise err("annotations section is required.")

        # The annotations section must be a mapping (dictionary).
        if not isinstance(annotations_yaml, dict):
            raise err("annotations section must be a mapping.")
        
        # Return the annotations YAML if it is valid.
        return annotations_yaml

    # Initialization Methods ---------------------------------------------------

    @staticmethod
    def _init_figure_grid(figure_grid, err):
        """Validate and normalize a figure grid specification.

        Accepts a flat list (single-row shorthand) or a list of lists
        (matrix). Validates that the matrix is rectangular and that all
        elements are strings or ``None``.

        Parameters
        ----------
        figure_grid : list
            A list of figure name strings (single row) or a list of
            lists (matrix). Elements are strings or ``None``.

        err : callable
            A partial :class:`ConfigError` constructor, pre-bound with
            the relevant config section path.

        Returns
        -------
        list of list
            The normalized figure grid as a list of equal-length rows.
        """
        # Check that the figure grid is a list.
        if not isinstance(figure_grid, list):
            raise err(f"figure_grid is required and must be a list/matrix.")
        
        # Single-row shorthand: ["a", "b", None] -> [["a", "b", None]]
        if all(el is None or isinstance(el, str) for el in figure_grid):
            figure_grid = [ figure_grid ]

        # Check the elements of the figure_grid.
        cols = None
        for row in figure_grid: 
            # Check that the row is a list. 
            if not isinstance(row, list):
                raise err("figure_grid must be a list/matrix.")
            
            # Check that the row has the same number of columns.
            if cols is None: cols = len(row) # defined by the first row
            elif len(row) != cols:
                raise err(
                    f"figure_grid cannot be a ragged matrix: "
                    f"expected {cols} columns, got {len(row)}"
                )
            
            # Check that the row elements are strings or None.
            for el in row:
                if el is None: continue
                elif not isinstance(el, str):
                    raise err("figure_grid items must be null or strings.")

        # Return the figure_grid.
        return figure_grid
    

    @staticmethod
    def _init_fixed_point(key, fixed_point, err, init):
        """Validate and compile a fixed point specification.

        A fixed point can be ``None`` (no fixed point), a string
        (shorthand: use the last point of the named annotation), or a
        dict with ``calculate`` and ``requires`` fields.

        Parameters
        ----------
        key : str
            Either ``"fixed_head"`` or ``"fixed_tail"`` (for error
            messages).

        fixed_point : str, dict, or None
            The raw YAML value for this fixed point.

        err : callable
            A partial :class:`ConfigError` constructor, pre-bound with
            the relevant config section path.

        init : InitConfig
            The init environment for compiling the calculate function.

        Returns
        -------
        dict or None
            ``None`` if no fixed point, otherwise a dict with keys
            ``"calculate"`` (compiled callable taking ``annotations``)
            and ``"requires"`` (list of annotation name strings).
        """
        # If the fixed point is None, then we just return None.
        if fixed_point is None: return None

        # Check that the fixed point is a string or mapping.
        if not isinstance(fixed_point, (str, dict)):
            raise err(f"{key} must be null, strings, or mappings.")
        
        # If the fixed point is a string, we use the last point of the given
        # annotation as the fixed point. 
        if isinstance(fixed_point, str):
            fixed_point = { 
                "calculate" : f"return annotations['{fixed_point}'][-1,:]",
                "requires"  : fixed_point
            }

        # Extract the requires and calculate fields from the mapping.
        requires  = fixed_point.get("requires", [])
        calculate = fixed_point.get("calculate", None)

        # Check that the requires field is a string, if so, wrap in a list.
        if isinstance(requires, str):
            requires = [ requires ]
        
        # Check that the requires field is a list of strings.
        if isinstance(requires, list):
            if not all(isinstance(el, str) for el in requires):
                raise err(f"{key} 'requires' field must be a string or list of strings.")
            
        # Check that the calculate field is a string.
        if calculate is None:
            raise err(f"{key} must contain 'calculate' if it is a mapping.")
            
        # Compile the calculate code string into a function. 
        calculate = init.compile_fn("annotations", calculate)
        
        # Return the fixed point dictionary.
        return { "calculate": calculate, "requires": requires }


    def _init_annotation(self, annotation_name, annotation_spec, init):
        """Parse a full annotation specification mapping into an Annotation record.

        Parameters
        ----------
        annotation_name : str
            The annotation's name (used in error messages).
        annotation_spec : dict
            The annotation's YAML mapping with optional keys: ``type``,
            ``fixed_head``, ``fixed_tail``, ``figure_grid``, ``filter``.
        init : InitConfig
            The init environment for compiling filter and fixed point code.

        Returns
        -------
        Annotation
            A validated Annotation namedtuple.
        """
        # Prepare ConfigError for any errors that arise in this function.
        err = partial(ConfigError, f"annotations.`{annotation_name}`")

        # Check that the key is a valid annotation option.
        for key in annotation_spec.keys():
            if key not in Annotation._fields:
                raise err(f"Invalid annotation key: {key}")

        # Extract annotation values or assign default values.
        atype       = annotation_spec.get("type", "contour")
        fixed_head  = annotation_spec.get("fixed_head", None)
        fixed_tail  = annotation_spec.get("fixed_tail", None)
        figure_grid = annotation_spec.get("figure_grid", None)
        filter_fn   = annotation_spec.get("filter", None)
        
        # Check that the annotation type is valid.
        if atype not in ( "contour", "boundary", "point"):
            raise err("Annotation type must be one of 'contour', 'boundary', or 'point'.")

        # Check and initialize the fixed points.
        fixed_head = self._init_fixed_point("fixed_head", fixed_head, err, init)
        fixed_tail = self._init_fixed_point("fixed_tail", fixed_tail, err, init)

        # Prepare and check the figure grid.
        figure_grid = self._init_figure_grid(figure_grid, err)

        # Check that the filter is a string or None.
        if filter_fn is not None and not isinstance(filter_fn, str):
            raise err(f"filter must be null or a code string.")

        # If a filter is provided, go ahead and compile it.
        if filter_fn is not None:
            filter_fn = init.compile_fn("target", filter_fn)

        # Return the annotation as an Annotation object.    
        return Annotation(
            type        = atype,
            fixed_head  = fixed_head,
            fixed_tail  = fixed_tail,
            figure_grid = figure_grid,
            filter      = filter_fn,
        )

    # Get Methods --------------------------------------------------------------

    def _get_type(self):
        """Build a dict mapping annotation names to their types.

        Returns
        -------
        dict of {str: str}
            Keys are annotation names; values are ``'contour'``,
            ``'boundary'``, or ``'point'``.
        """
        return { key: value.type for key, value in self.items() }


    def _get_figure_grid(self):
        """Build a dict mapping annotation names to their figure grids.

        Returns
        -------
        dict of {str: list of list}
            Keys are annotation names; values are figure grid matrices.
        """
        return { key: value.figure_grid for key, value in self.items() }


    def _get_grid_shape(self):
        """Build a dict mapping annotation names to their grid shapes.

        Returns
        -------
        dict of {str: tuple of int}
            Keys are annotation names; values are ``(rows, cols)``
            tuples.
        """
        return { key: np.shape(value.figure_grid) for key, value in self.items() }
    

    def _get_fixed_info(self, fixed_point):
        """Build a dict mapping annotation names to fixed point info.

        Parameters
        ----------
        fixed_point : str
            Either ``"fixed_head"`` or ``"fixed_tail"``.

        Returns
        -------
        dict of {str: dict or None}
            Keys are annotation names; values are the fixed point info
            dicts (with ``"calculate"`` and ``"requires"``) or ``None``.
        """
        # Ensure that the name is either "fixed_head" or "fixed_tail".
        if fixed_point not in ("fixed_head", "fixed_tail"):
            raise ValueError(f"Invalid fixed point: {fixed_point}")
            
        # Returns a dict mapping annotation names to their fixed point information.
        return { 
            key: getattr(value, fixed_point) for key, value in self.items() 
        }
    

    def _get_fixed_names(self, fixed_point):
        """Build a dict mapping annotation names to their required annotations.

        Parameters
        ----------
        fixed_point : str
            Either ``"fixed_head"`` or ``"fixed_tail"``.

        Returns
        -------
        dict of {str: list of str}
            Keys are annotation names; values are lists of annotation
            names required by the given fixed point, or an empty list.
        """
        # Ensure that the name is either "fixed_head" or "fixed_tail".
        if fixed_point not in ("fixed_head", "fixed_tail"):
            raise ValueError(f"Invalid fixed point: {fixed_point}")
            
        # Returns a dict mapping annotation names to their fixed point name.
        return { 
            key: [] if getattr(value, fixed_point) is None 
            else getattr(value, fixed_point)["requires"]
            for key, value in self.items() 
        }