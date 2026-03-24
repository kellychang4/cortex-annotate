# -*- coding: utf-8 -*-
################################################################################
# annotate/config/_viewer.py

"""3D cortex viewer configuration for cortex-annotate.

ViewerConfig parses the optional ``viewer`` section of config.yaml, which
defines the geometry, coordinate systems, overlays, and coordinate transforms
needed to render a 3D cortex mesh alongside the 2D annotation canvas.

All fields are specified as Python code strings in config.yaml and compiled
into callable functions via the init environment. If the viewer section is
omitted, ViewerConfig is an empty dict and the tool operates in 2D-only mode.
"""

# Imports ----------------------------------------------------------------------

from functools import partial

from ._error import ConfigError
            
# Viewer Configuration ---------------------------------------------------------

class ViewerConfig(dict):
    """3D cortex viewer configuration from config.yaml.

    A dict subclass whose entries define the geometry, coordinate
    systems, overlays, and coordinate transforms needed to render a 3D
    cortex mesh. Each value is a compiled callable (or dict of
    callables) produced from Python code strings in config.yaml.

    If the ``viewer`` section is omitted from config.yaml, the dict is
    empty and the tool operates in 2D-only mode.

    Parameters
    ----------
    viewer_yaml : dict or None
        The ``viewer`` section from config.yaml. If ``None`` or empty,
        the resulting dict is empty.

    figure_names : set of str
        Figure names referenced by annotation grids, used to validate
        overlay coverage.

    init : InitConfig
        The init environment for compiling code strings.

    Keys
    ----
    faces : callable
        ``fn(target) → ndarray, shape (n_faces, 3)``.
        Triangle face indices into the vertex array.

    coordinates : dict of {str: callable}
        ``{name: fn(target) → ndarray, shape (n_vertices, 3)}``.
        Named vertex coordinate sets (e.g. ``'midgray'``,
        ``'inflated'``). A single code string is stored under
        ``'_default'``.

    morph_between : list of str or None
        Two coordinate set names to interpolate between with the
        morph slider. Both must exist in ``coordinates``.

    overlays : dict of {str: callable}
        ``{name: fn(target, key) → ndarray, shape (n_vertices, 3)}``.
        Per-vertex RGB colors in [0, 1]. Must include ``'curvature'``.

    canvas_to_viewer : callable
        ``fn(target, points, mesh_coordinates) → ndarray, shape
        (n_points, 3)``. Converts 2D canvas coordinates to 3D viewer
        coordinates.
    """

    __slots__ = ( )
    
    def __init__(self, viewer_yaml, figure_names, init):
        # Initialize viewer_dict.
        viewer_dict = {}

        # The viewer section is optional, return empty dict if not provided.
        if viewer_yaml is not None:
            # If viewer section is provided, it must be a dictionary.
            if not isinstance(viewer_yaml, dict):
                raise ConfigError("viewer", "viewer section must contain a mapping.")

            # Prepare the faces field.
            viewer_dict["faces"] = self._init_faces(viewer_yaml, init)

            # Prepare the coordinates field.
            viewer_dict["coordinates"] = self._init_coordinates(viewer_yaml, init)
            
            # Prepare the morph_between field, optional.
            viewer_dict["morph_between"] = self._init_morph_between(
                viewer_yaml, viewer_dict["coordinates"])

            # Prepare the overlays field.
            viewer_dict["overlays"] = self._init_overlays(
                viewer_yaml, figure_names, init)

            # Prepare the canvas_to_viewer field
            viewer_dict["canvas_to_viewer"] = self._init_canvas_to_viewer(
                viewer_yaml, init)
            
        # Update ViewerConfig class dictionary.
        self.update(viewer_dict)


    # Compile Figure Functions -------------------------------------------------

    @staticmethod
    def _compile_fn(init, argstr, code):
       """Compile a viewer code string into a callable.

        Parameters
        ----------
        init : InitConfig
            The init environment.

        argstr : str
            Comma-separated parameter names for the function.

        code : str
            The Python code string to compile.

        Returns
        -------
        callable
            The compiled function.
        """
       return init.compile_fn(argstr, f"{code}")

    # Initialize Viewer Fields -------------------------------------------------

    @staticmethod
    def _init_faces(viewer_yaml, init):
        """Validate and compile the ``faces`` code string.

        The compiled function must return an array of shape
        ``(n_faces, 3)`` containing triangle face indices.

        Parameters
        ----------
        viewer_yaml : dict
            The viewer YAML mapping.

        init : InitConfig
            The init environment.

        Returns
        -------
        callable
            ``fn(target) → ndarray, shape (n_faces, 3)``.
        """
        # Prepare ConfigError for any errors that arise in this function.
        err = partial(ConfigError, "viewer.faces")

        # If viewer_yaml is None, there is no viewer configuration. Return None.
        if viewer_yaml is None: return None

        # If there is viewer configuration...
        faces = viewer_yaml.get("faces", None)

        # The faces field is required for any viewer configuration.
        if faces is None:
            raise err("faces is a required field.")

        # Check that faces is a string/code
        if not isinstance(faces, str):
            raise err("faces must be a code string.")

        # Compile the faces code string into a function.
        faces = ViewerConfig._compile_fn(init, "target", faces)

        # Return the compiled function.
        return faces


    @staticmethod
    def _init_coordinates(viewer_yaml, init):
        """Validate and compile the ``coordinates`` field.

        Accepts a single code string (stored under ``'_default'``) or
        a mapping of named code strings. Each compiled function must
        return an array of shape ``(n_vertices, 3)``.

        Parameters
        ----------
        viewer_yaml : dict
            The viewer YAML mapping.

        init : InitConfig
            The init environment.

        Returns
        -------
        dict of {str: callable}
            Maps coordinate set names to compiled functions
            ``fn(target) → ndarray, shape (n_vertices, 3)``.
        """
        # Prepare ConfigError for any errors that arise in this function.
        err = partial(ConfigError, "viewer.coordinates")

        # If viewer_yaml is None, there is no viewer configuration. Return None.
        if viewer_yaml is None: return None

        # Extract the coordinates field from the yaml.
        coordinates = viewer_yaml.get("coordinates", None)

        # The coordinates field is required for any viewer configuration.
        if coordinates is None:
            raise err("coordinates is a required field.")

        # Check that coordinates is a string/code
        if not isinstance(coordinates, (str, dict)):
            raise err("`coordinates` must be a code string or mapping.")
        
        # If coordinates is a string, then only one coordinate set is provided, 
        # confirm to dictionary storage under the key '_default'.
        if isinstance(coordinates, str):
            coordinates = { "_default" : coordinates }
    
        # If coordinates is a dict, then we compile the code for each key.
        # Check that all mappings are strings/code.
        for key, value in coordinates.items():
            if not isinstance(value, str):
                raise ConfigError(f"viewer.coordinates.{key}", 
                    "Coordinate mapping values must be code strings.")
         
        # return dictionary of compile functions per coordinates
        return {
            key: ViewerConfig._compile_fn(init, "target", value) 
            for key, value in coordinates.items()
        }


    @staticmethod
    def _init_morph_between(viewer_yaml, coordinates):
        """Validate the optional ``morph_between`` field.

        If provided, must be a two-element list naming coordinate sets
        that exist in *coordinates*.

        Parameters
        ----------
        viewer_yaml : dict
            The viewer YAML mapping.

        coordinates : dict
            The already-parsed coordinates dict, used to validate that
            both named sets exist.

        Returns
        -------
        list of str or None
            A two-element list of coordinate set names, or ``None``.
        """
        # Prepare ConfigError for any errors that arise in this function.
        err = partial(ConfigError, "viewer.morph_between")

        # If viewer_yaml is None, there is no viewer configuration. Return None.
        if viewer_yaml is None: return None

        # Extract the coordinates field from the yaml.
        morph_between = viewer_yaml.get("morph_between", None)

        # This is an optional field, so if None, return first coordinate set.
        if morph_between is None: 
            return list(coordinates.keys())[0:1]
        
        # If provided, must be a list of length two.
        if not isinstance(morph_between, list) or len(morph_between) != 2:
            raise err("`morph_between` must be a list of length two.")

        # If provided, coordinates names must be given in the coordinates dict.
        for surface_name in morph_between: 
            if surface_name not in coordinates:
                raise err(f"Unable to find `{surface_name}` coordinates.")
        
        # Return morph_between list.
        return morph_between
        
    
    @staticmethod
    def _init_overlays(viewer_yaml, figure_names, init):
        """Validate and compile the ``overlays`` field.

        Accepts a single code string (treated as the required
        ``curvature`` overlay) or a mapping of named code strings.
        Each compiled function must return per-vertex RGB colors as an
        array of shape ``(n_vertices, 3)`` with values in [0, 1].

        A wildcard key ``'_'`` provides a fallback for figure names
        not explicitly listed.

        Parameters
        ----------
        viewer_yaml : dict
            The viewer YAML mapping.
        
        figure_names : set of str
            Figure names from annotation grids. Combined with
            ``{"curvature"}`` to form the set of required overlays.

        init : InitConfig
            The init environment.

        Returns
        -------
        dict of {str: callable}
            Maps overlay names to compiled functions
            ``fn(target, key) → ndarray, shape (n_vertices, 3)``.
        """
        # Prepare ConfigError for any errors that arise in this function.
        err = partial(ConfigError, "viewer.overlays")

        # If viewer_yaml is None, there is no viewer configuration. Return None.
        if viewer_yaml is None: return None

        # If there is a wildcard key, extract out.
        overlays = viewer_yaml.get("overlays", None)

        # The overlays field is required for any viewer configuration.
        if overlays is None:
            raise err("`overlays` is a required field.")
        
        # Check that overlays is a string/code or mapping.
        if not isinstance(overlays, (str, dict)):
            raise err("`overlays` must be a code string or mapping.")

        # If overlays is a string, then assuming it is for the curvature.
        if isinstance(overlays, str):
            overlays = { "curvature" : overlays }

        # If overlays is a dict, it must contain a curvature or a wildcard key.
        if "curvature" not in overlays and "_" not in overlays:
            raise err("Overlays must include a 'curvature' entry or a wildcard '_'.")

        # If overlays is a dict, then we compile the code for each key.
        # Check that all mapping values are strings/code.
        for key, value in overlays.items():
            if not isinstance(value, str):
                raise ConfigError(f"viewer.overlays.{key}", 
                    "Overlay mapping values must be code strings.")
         
        # Locate wildcard key, if provided.
        wildfn = overlays.get("_", None)
        if wildfn is not None:
            wildfn = ViewerConfig._compile_fn(init, "target, key", wildfn)
            
        # Prepare valid figure names for the overlays, including curvature.
        figure_names = { "curvature" } | set(figure_names)

        # Prepare overlays dictionary
        overlays_dict = {}
        for key in figure_names:
            if key not in overlays.keys():
                if wildfn is None:
                    raise ConfigError(
                        f"viewer.overlays.{key}", 
                        f"Missing code for figure '{key}' and "
                        f"no wildcard provided."
                    )
                key_fn = wildfn
            else: 
                key_fn = ViewerConfig._compile_fn(
                    init, "target, key", overlays[key])
            overlays_dict[key] = key_fn
        
        # Return compiled overlays dictionary.
        return overlays_dict


    @staticmethod
    def _init_canvas_to_viewer(viewer_yaml, init):
        """Validate and compile the ``canvas_to_viewer`` code string.

        The compiled function converts 2D canvas annotation coordinates
        to 3D viewer coordinates using the mesh geometry.

        Parameters
        ----------
        viewer_yaml : dict
            The viewer YAML mapping.
            
        init : InitConfig
            The init environment.

        Returns
        -------
        callable
            ``fn(target, points, mesh_coordinates) → ndarray, shape
            (n_points, 3)`` where *points* is ``(n_points, 2)`` and
            *mesh_coordinates* is ``(n_vertices, 3)``.
        """
        # Prepare ConfigError for any errors that arise in this function.
        err = partial(ConfigError, "viewer.canvas_to_viewer")

        # If viewer_yaml is None, there is no viewer configuration. Return None.
        if viewer_yaml is None: return None

        # Extract the canvas_to_viewer field from the yaml.
        canvas_to_viewer = viewer_yaml.get("canvas_to_viewer", None)

        # The canvas_to_viewer field is required for any viewer configuration.
        if canvas_to_viewer is None:
            raise err("`canvas_to_viewer` is a required field.")

        # Check that canvas_to_viewer is a string/code
        if not isinstance(canvas_to_viewer, str):
            raise err("`canvas_to_viewer` must be a code string.")
    
        # Compile the canvas_to_viewer code string into a function.
        # NOTE: canvas_to_viewer(target, points, mesh_coordinates <internal>)
        canvas_to_viewer = ViewerConfig._compile_fn(
            init, "target, points, mesh_coordinates", canvas_to_viewer)
         
        # Return the compiled function.
        return canvas_to_viewer