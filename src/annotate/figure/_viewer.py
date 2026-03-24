# -*- coding: utf-8 -*-
################################################################################
# annotate/figure/_viewer.py

"""3D cortex viewer renderer for cortex-annotate.
 
``CortexViewerPanel`` is a pure rendering widget.  It displays a 3D
cortical mesh with overlay coloring and annotation paths rendered as
lines and scatter points in 3D space.
 
The viewer reads 2D annotation coordinates from the shared
``AnnotationEditor`` and converts them to 3D viewer coordinates via
a configurable ``canvas_to_viewer`` function.  Annotation styles are
read from the ``PrefsManager``.  Viewer-specific style settings
(morph, overlay, point size, line width) are also read from
``PrefsManager`` at construction and updated at runtime by the
orchestrator.
 
Viewer-specific mesh data (faces, coordinate sets, overlays, the
coordinate-transform function) are stored as instance attributes and
set by the orchestrator before drawing begins.
 
k3d Layers
----------
Layer 0 : cortex mesh -> base mesh colored by curvature.
Layer 1 : overlay mesh —> optional per-vertex color overlay
          (e.g. ROI map).  Hidden when overlay is ``"curvature"``.
Layer 2 : background annotation lines (independent of active).
Layer 3 : background annotation points (independent of active).
Layer 4 : dependent annotation lines (fixed points derived from
          active annotation).
Layer 5 : dependent annotation points.
Layer 6 : active annotation lines.
Layer 7 : active annotation points.
"""

# Imports ----------------------------------------------------------------------
 
import k3d
import numpy as np
import ipywidgets as ipw
from matplotlib.colors import to_rgb
 
# Viewer Subpanel --------------------------------------------------------------

class ViewerPanel(ipw.VBox):
    """3D k3d cortex viewer for mesh display and annotation rendering.
 
    Manages a k3d ``Plot`` with 8 drawable objects (2 meshes +
    3 line/point pairs for background, dependent, and active
    annotations).
 
    The viewer reads 2D annotation coordinates from the
    ``AnnotationEditor``, converts them to 3D via the
    ``canvas_to_viewer`` transform, and renders them as colored
    lines and scatter points on the cortical surface.
 
    Viewer-specific mesh data (faces, coordinate sets, overlays, the
    transform function) are stored as instance attributes and must be
    set before the first ``redraw_viewer`` call.
 
    Parameters
    ----------
    editor : AnnotationEditor
        The shared annotation editing model (read-only access).

    prefs : PrefsManager
        User preferences for annotation and viewer styles.

    width : int, optional
        Widget width in pixels.  Defaults to ``512``.

    height : int, optional
        Widget height in pixels.  Defaults to ``512``.
 
    Attributes
    ----------
    editor : AnnotationEditor
        Reference to the annotation editing model.

    prefs : PrefsManager
        Reference to the user preferences manager.

    faces : ndarray or None
        ``(n_faces, 3)`` triangle face indices.  Set by caller.

    _coordinates : list of ndarray or None
        List of ``(n_vertices, 3)`` coordinate arrays (1 or 2 sets
        for morph blending).  Set by caller.

    overlays : dict of {str: ndarray} or None
        Per-vertex RGB color arrays keyed by overlay name.  Set by
        caller.

    canvas_to_viewer : callable or None
        ``fn(points, mesh_coords) → ndarray (n, 3)``.  Converts 2D
        canvas coordinates to 3D viewer coordinates.  Set by caller.

    annotations : dict of {str: dict}
        Cached 3D annotation data keyed by annotation name.  Each
        value is ``{"coordinates": ndarray, "point_types": ndarray}``
        or ``{"coordinates": None, "point_types": None}``.
    """

    # Define the coordinate point types. 
    _POINT_FIXED  = 2  # fixed head/tail point
    _POINT_USER   = 1  # user-placed point
    _POINT_INTERP = 0  # interpolated point (between user/fixed points)

    # Define empty arrays for initializing k3d layers 
    _EMPTY_COORDINATES = np.array([[0, 0, 0]], dtype = np.float32)
    _EMPTY_INDICES     = np.array([[0, 0, 0]], dtype = np.uint32)
    _EMPTY_COLORS      = np.array( [0x000000], dtype = np.uint32)

    __slots__ = (
        "editor", "prefs", "annot_cfg", 
        "faces", "_coordinates", "overlays", "canvas_to_viewer",
        "annotations",
        # k3d figure.
        "_figure", 
        # Cortex mesh and overlay layers.
        "_k3dmesh_cortex", "_k3dmesh_overlay",
        # Background annotation layers 
        "_k3dline_background", "_k3dpoints_background",
        # Dependent annotation layers 
        "_k3dline_dependent", "_k3dpoints_dependent",
        # Active annotation layers.
        "_k3dline_active", "_k3dpoints_active",
    )

    def __init__(self, editor, prefs):
        """Initialize the cortex viewer panel.
 
        Parameters
        ----------
        editor : AnnotationEditor
            The shared annotation editing model.
        prefs : PrefsManager
            User preferences manager.
        width : int, optional
            Widget width in pixels.  Defaults to ``512``.
        height : int, optional
            Widget height in pixels.  Defaults to ``512``.
        """
        # Store references.
        self.editor    = editor
        self.prefs     = prefs
        self.annot_cfg = editor.annot_cfg

        # Rendering variables.
        self.faces            = None
        self._coordinates     = None
        self.overlays         = None
        self.canvas_to_viewer = None
 
        # Initialize viewer annotations dictionary.
        self.annotations = {}

        # Get viewer size from preferences and set width/height.
        viewer_size = self.prefs.get_display("viewer_size")

        # Create the k3d plot.
        self._figure = k3d.plot(
            height            = viewer_size,
            grid_visible      = False,
            menu_visibility   = False,
            camera_fov        = 60,
            axes_helper       = 0,
            camera_zoom_speed = 1.5,
        )

        # Get the camera from preferences if it exists (overrides auto-fit).
        if self.prefs.get_display("viewer_camera") != []:
            self._figure.camera_auto_fit = False
            self._figure.camera = self.prefs.get_display("viewer_camera")

        # Initialize all k3d layers (start empty/invisible).
        self._k3dmesh_cortex       = self._init_mesh()
        self._k3dmesh_overlay      = self._init_mesh()
        self._k3dline_background   = self._init_line()
        self._k3dpoints_background = self._init_points()
        self._k3dline_dependent    = self._init_line()
        self._k3dpoints_dependent  = self._init_points()
        self._k3dline_active       = self._init_line()
        self._k3dpoints_active     = self._init_points()

        # Add all layers to the figure.
        self._figure += self._k3dmesh_cortex
        self._figure += self._k3dmesh_overlay
        self._figure += self._k3dline_background
        self._figure += self._k3dpoints_background
        self._figure += self._k3dline_dependent
        self._figure += self._k3dpoints_dependent
        self._figure += self._k3dline_active
        self._figure += self._k3dpoints_active

        # Initialize the VBox with the figure as the child 
        super().__init__(
            children = [ self._figure ], 
            layout = {
                "width"   : f"{viewer_size}px", 
                "height"  : f"{viewer_size}px", 
                "border"  : "1px solid magenta",
                "overflow": "hidden"
            }
        )

        # Wire internal observers
        self._figure.observe(self._on_camera_change, names = "camera")

    # Set Method ---------------------------------------------------------------

    def set_viewer(self, faces = None, coordinates = None, overlays = None, canvas_to_viewer = None):
        """Set the viewer mesh data and coordinate transform.
 
        Parameters
        ----------
        faces : ndarray, shape (n_faces, 3)
            Triangle face indices for the cortex mesh.
 
        coordinates : list of ndarray, each of shape (n_vertices, 3)
            List of vertex coordinate sets.  One set for static meshes,
            two sets for morph blending.
 
        overlays : dict of {str: ndarray}
            Per-vertex RGB color arrays keyed by overlay name.
 
        canvas_to_viewer : callable
            ``fn(points, mesh_coords) → ndarray (n, 3)``.  Converts 2D
            canvas coordinates to 3D viewer coordinates.
        """
        print("Viewer set_viewer called with data:")
        print(f"  faces: {faces.shape if faces is not None else None}")
        print(f"  coordinates: {[c.shape for c in coordinates] if coordinates is not None else None}")
        print(f"  overlays: {list(overlays.keys()) if overlays is not None else None}")
        print(f"  canvas_to_viewer: {canvas_to_viewer is not None}")
        self.faces            = faces
        self._coordinates     = coordinates
        self.overlays         = overlays
        self.canvas_to_viewer = canvas_to_viewer
    
    # Properties ---------------------------------------------------------------
 
    @property
    def coordinates(self):
        """Compute blended vertex coordinates for the current morph state.
 
        If only one coordinate set is loaded, returns it directly.
        If two sets are loaded (for morph blending), interpolates
        between them based on ``self.prefs.get_viewer_style("morph_percent")``.
 
        Returns
        -------
        ndarray, shape (n_vertices, 3)
            The blended vertex coordinates.
        """
        # If no coordinates are loaded, return None to indicate no mesh to plot.
        if self._coordinates is None: return None

        # If only one coordinate set is loaded, return the coordinates.
        if len(self._coordinates) == 1: return self._coordinates[0]
 
        # If more than one coordinates sets, interpolate between coordinates.
        start_coords, end_coords = self._coordinates[0], self._coordinates[1]
        morph_proportion = self.prefs.get_viewer_style("morph_percent") / 100.0
        return ((end_coords - start_coords) * morph_proportion) + start_coords

    # 2D → 3D Annotation Conversion -------------------------------------------
 
    def _interpolate_coordinates(self, coordinates, point_types):
        """Interpolate coordinates along an annotation path.
 
        Inserts evenly spaced points between each consecutive pair of
        annotation coordinates.  The number of interpolated points per
        segment is determined by ``self.prefs.get_viewer_style("line_interp")``.
 
        Parameters
        ----------
        coordinates : ndarray, shape (N, 2)
            The 2D annotation coordinates.
            
        point_types : ndarray of int, shape (N,)
            Point type for each coordinate (``self._POINT_FIXED``,
            ``self._POINT_USER``, or ``self._POINT_INTERP``).
 
        Returns
        -------
        interp_coords : ndarray, shape (M, 2)
            The interpolated coordinate array.

        interp_types : ndarray of int, shape (M,)
            Point type for each interpolated coordinate.
        """
        # Get the number of points to interpolate per segment.
        n_interp = self.prefs.get_viewer_style("line_interp")
        n = n_interp + 2 # add two for start/end points

        # Intialize ararys to store interpolated coordinates
        x_interp = []; y_interp = []; ptype_interp = []
 
        # Initialize point type interpolation filler
        ptype_filler = [self._POINT_INTERP] * n_interp
 
        # Iterate over each segment and interpolate points  
        n_segments = coordinates.shape[0] - 1
        for i in np.arange(n_segments): 
            # Extract start and end coordinates and point types for the segment
            xs, xe = coordinates[i, 0], coordinates[i + 1, 0]
            ys, ye = coordinates[i, 1], coordinates[i + 1, 1]
            ps, pe = point_types[i], point_types[i + 1]

            # Interpolate x and y coordinates and point types for the segment
            xn = np.linspace(xs, xe, n)
            yn = np.linspace(ys, ye, n)
            pn = [ps, *ptype_filler, pe]
 
            if i == 0:
                # First segment: include the starting point.
                x_interp.append(xn)
                y_interp.append(yn)
                ptype_interp.append(pn)
            else:
                # Subsequent segments: exclude starting point to avoid
                # duplicates at segment boundaries.
                x_interp.append(xn[1:])
                y_interp.append(yn[1:])
                ptype_interp.append(pn[1:])
 
        # Concatenate interpolated points
        x_interp     = np.concatenate(x_interp)
        y_interp     = np.concatenate(y_interp)
        ptype_interp = np.concatenate(ptype_interp)
 
        # Return interpolated coordinates (as matrix) and point types (as int)
        interp_matrix = np.vstack((x_interp, y_interp, ptype_interp)).T
        return interp_matrix[:, :-1], interp_matrix[:, -1].astype(int)
 

    def update(self, annotations = None):
        """Convert 2D canvas annotations to 3D viewer coordinates.
 
        For each annotation, determines point types (fixed vs. user),
        interpolates along the path, and transforms to 3D via the
        ``canvas_to_viewer`` function.  Results are cached in
        ``self.annotations``.
 
        Parameters
        ----------
        annotations : list of str or None, optional
            Annotation names to update.  If ``None``, updates all
            annotations.
        """
        # If no specific annotations provided, update all annotations.
        if annotations is None: annotations = list(self.annot_cfg.names)
 
        for annotation in annotations:
            # Get the canvas coordiantes for current annotation
            canvas_points = self.editor.annotations.get(annotation, None)
 
            # If no points, no viewer coordinates or point types.
            if canvas_points is None or canvas_points.shape[0] == 0:
                self.annotations[annotation] = {
                    "coordinates": None,
                    "point_types": None,
                }
                continue
 
            # Determine point types (fixed vs. user).
            n_points    = canvas_points.shape[0]
            point_types = np.full(n_points, self._POINT_USER)
            has_fixed_head = bool(self.annot_cfg.fixed_heads[annotation])
            has_fixed_tail = bool(self.annot_cfg.fixed_tails[annotation])
            if has_fixed_head: point_types[0]  = self._POINT_FIXED
            if has_fixed_tail: point_types[-1] = self._POINT_FIXED
 
            # Interpolate if there are segments and not all fixed points.
            if n_points > 1 and not np.all(point_types == self._POINT_FIXED):
                canvas_points, point_types = (
                    self._interpolate_coordinates(canvas_points, point_types))
 
            # Convert 2D → 3D with the `canvas_to_viewer` function.
            print(f"canvas_points: {canvas_points.shape}")
            print(f"point_types: {point_types.shape}")
            print(f"canvas_to_viewer: {self.canvas_to_viewer}")
            print(f"coordinates: {self.coordinates.shape}")  
            viewer_coords = self.canvas_to_viewer(
                canvas_points, self.coordinates)

            # Store the current annotation's viewer coordinates and point types.
            self.annotations[annotation] = {
                "coordinates": viewer_coords,
                "point_types": point_types,
            }

    # k3d Color Helper Method --------------------------------------------------

    @ staticmethod
    def _rgb_to_k3dcolor(colors):
        """Convert color input to k3d uint32 hex integers.
 
        Accepts matplotlib color strings, float RGB/RGBA arrays in
        [0, 1], or uint8 RGB/RGBA arrays in [0, 255].
 
        Parameters
        ----------
        colors : str, array-like
            Color input.  Scalars and 1D arrays are promoted to 2D.
 
        Returns
        -------
        ndarray of uint32
            One hex color integer per input row.
        """
        # Convert to numpy array for easier processing
        colors = np.array(colors)
        
        # Handle string color inputs (e.g. "red", "#ff0000", etc.)
        if np.issubdtype(colors.dtype, np.str_):
            if colors.ndim == 0: colors = colors.reshape(-1,)
            colors = np.array([to_rgb(x) for x in colors], dtype = float)

        # Handle single RGB or RGBA triple input (e.g. [1, 0, 0] or [1, 0, 0, 1])
        if colors.ndim == 1: colors = colors.reshape((1, -1))
            
        # Handle floating point inputs by converting to uint8
        if np.issubdtype(colors.dtype, np.floating):
            if colors.max() > 1.0: # if max is greater than 1, assume 0-255 range
                colors = colors.astype(np.uint8)
            else: # else assume 0-1 range and convert to 0-255
                colors = (colors * 255).astype(np.uint8)

        # Handle integer inputs by checking if they are within uint8 range and converting to uint8
        if np.issubdtype(colors.dtype, np.integer):
            if colors.min() < 0 or colors.max() > 255:
                raise ValueError("Color values must be within uint8 range [0-255].")    
            colors = colors.astype(np.uint8)

        # Check that colors are now uint8 and converted to 2D array
        if not np.issubdtype(colors.dtype, np.uint8):
            raise ValueError("Color values must be convertible from float [0,1] or uint8 [0,255].")
        if colors.ndim != 2:
            raise ValueError("Color input must be scalar, 1D RGB/RGBA, or 2D Nx3/Nx4.")
        
        # Convert RGB/RGBA values to k3d color integers
        colors = colors.astype(np.uint32) # ensure uint32 for bitwise operations
        if colors.shape[1] == 3: # if RGB, convert to k3d color integer
            return np.array(
                [ ((r << 16) | (g << 8) | b) for r, g, b in colors ], 
                dtype = np.uint32
            )
        elif colors.shape[1] == 4: # if RGBA, ignore the alpha channel
            # NOTE: k3d does not support alpha in the color integer, ignore
            return np.array(
                [ ((r << 16) | (g << 8) | b) for r, g, b, _ in colors ], 
                dtype = np.uint32
            )
        else:
            raise ValueError("Color matrices must be RGB (Nx3) or RGBA (Nx4).")

    # Initialize Methods -------------------------------------------------------

    def _init_mesh(self):
        """Create an empty and invisible k3d mesh. """        
        mesh = k3d.mesh(
            vertices     = self._EMPTY_COORDINATES.copy(), 
            indices      = self._EMPTY_INDICES.copy(),
            colors       = self._EMPTY_COLORS.copy(),
            wireframe    = False,
            flat_shading = False
        )
        mesh.visible = False
        return mesh


    def _init_points(self):
        """Initialize an empty and invisible k3d points object."""
        points = k3d.points(
            positions = self._EMPTY_COORDINATES.copy(),
            colors    = self._EMPTY_COLORS.copy(), 
            shader    = "3d"
        )
        points.visible = False
        return points


    def _init_line(self):
        """Initialize an empty and invisible k3d line object."""
        line = k3d.line(
            vertices = self._EMPTY_COORDINATES.copy(),
            colors   = self._EMPTY_COLORS.copy(), 
            width    = float(self.prefs.get_viewer_style("line_width")),
            shader   = "mesh"
        )
        line.visible = False
        return line

    # Prepare Cortex Methods ---------------------------------------------------

    def _prep_cortex(self):
        """Prepare vertex/face/color data for the base cortex mesh.
 
        Returns
        -------
        dict
            Keyword arguments for updating ``k3dmesh_cortex``.
        """
        curvature = self._rgb_to_k3dcolor(self.overlays["curvature"])
        return {
            "vertices": self.coordinates.astype(np.float32),
            "indices":  self.faces.astype(np.uint32),
            "colors":   curvature.astype(np.uint32),
        }
    
    # Prepare Overlay Methods --------------------------------------------------

    def _prep_overlay(self):
        """Prepare vertex/color data for the overlay mesh.
 
        Returns ``None`` when the overlay is ``"curvature"`` (the base
        mesh already shows curvature colors).
 
        Returns
        -------
        dict or None
            Keyword arguments for updating ``k3dmesh_overlay``, or
            ``None`` if no overlay is needed.
        """
        # If overlay style is curvature, no additional overlay
        overlay_name = self.prefs.get_viewer_style("overlay")
        if overlay_name == "curvature": return None

        # Else, get overlay values and return with opacity.
        overlay_values =  s[overlay_name]
        return {
            **self._prep_cortex(),
            "colors"  : self._rgb_to_k3dcolor(overlay_values),
            "opacity" : float(self.prefs.get_viewer_style("overlay_alpha"))
        }
    
    # Prepare Single Annotation ------------------------------------------------

    def _prep_single_annotation(self, annotation, style_key):
        """Prepare line and point data for a single annotation.
 
        Used for the active annotation layer.  Produces separate line
        vertex and point position arrays with per-vertex/per-point
        colors and per-point sizes (fixed points are drawn slightly
        larger).
 
        Parameters
        ----------
        annotation : str
            Annotation name.
 
        style_key : str or None
            Key for ``prefs.get_annotation_style()``.  ``None`` for
            the active annotation style.
 
        Returns
        -------
        dict or None
            ``{"line": {...}, "points": {...}}`` with k3d keyword
            arguments, or ``None`` if the annotation has no cached
            viewer data or is invisible.
        """
        # Get the current active viewer annotation
        viewer_annotation = self.annotations[annotation]

        # If no viewer annotation, return None to skip plotting.
        if viewer_annotation is None: return None

        # If no coordinates, return None to skip plotting.
        coordinates = viewer_annotation.get("coordinates", None)
        if coordinates is None or coordinates.shape[0] == 0: return None

        # Get the annotation style from the styler (active = None)
        # If not visible, return None to skip plotting.
        annotation_style = self.prefs.get_annotation_style(style_key)
        if not annotation_style["visible"]: return None

        # Get number of annotation vertex (line) coordinates and point types
        vertices    = coordinates.astype(np.float32)
        positions   = vertices.copy() # copy!
        point_types = viewer_annotation.get("point_types", None)

        # Check if vertices are all fixed points, skip lines if so. 
        if np.all(point_types == self._POINT_FIXED):
            # set vertices to empty to skip segment plotting
            vertices = self._EMPTY_COORDINATES 

        # Get annotation user points and point types
        interp_mask = point_types != self._POINT_INTERP
        positions   = positions[interp_mask]
        point_types = point_types[interp_mask]
        n_points    = positions.shape[0]

        # Prepare scatter sizes by points type (slightly larger fixed points)
        base_size   = self.prefs.get_viewer_style("point_size")
        point_sizes = np.full(n_points, base_size)
        point_sizes[point_types == self._POINT_FIXED] = base_size * 1.25

        # Prepare colors for each annotation point
        annotation_color = self._rgb_to_k3dcolor(annotation_style["color"])

        # Return the active annotation plot keyword arguments by plot type
        return { 
            "line": {
                "vertices" : vertices.astype(np.float32),
                "width"    : float(self.prefs.get_viewer_style("line_width")),
                "colors"   : np.full(vertices.shape[0], annotation_color, dtype = np.uint32)
            },
            "points": {
                "positions"   : positions.astype(np.float32), 
                "point_sizes" : point_sizes.astype(np.float32), 
                "colors"      : np.full(n_points, annotation_color, dtype = np.uint32)
            }
        }

    # Prepare Multiple Annotations ---------------------------------------------

    def _prep_multiple_annotations(self, annotation_list, size_scale = 0.5):
        """Prepare line and point data for multiple annotations.
 
        Used for background and dependent annotation layers.  All
        annotations are concatenated into single vertex and position
        arrays with NaN separators between line segments so k3d draws
        them as disconnected paths.
 
        Parameters
        ----------
        annotation_list : list of str
            Annotation names to include.
 
        size_scale : float, optional
            Scale factor applied to ``line_width`` and ``point_size``
            from viewer preferences.  Background/dependent annotations
            are drawn smaller than the active annotation.
            Defaults to ``0.5``.
 
        Returns
        -------
        dict or None
            ``{"line": {...}, "points": {...}}`` with k3d keyword
            arguments, or ``None`` if no annotations had data to draw.
        """
        # Initialize empty arrays for all coordinates and colors
        all_vertices  = np.empty((0, 3))
        all_positions = np.empty((0, 3))
        all_lcolors   = np.empty((0,), dtype = np.uint32)
        all_pcolors   = np.empty((0,), dtype = np.uint32)

        # Initialize NaN array to separate annotations (for line plotting)
        coord_sep = np.full((1, 3), np.nan)
        color_sep = np.array([0], dtype = np.uint32)

        for annotation in annotation_list: # for each annotation
            # Get the current viewer annotation.
            viewer_annotation = self.annotations[annotation]
          
            # If no viewer annotation, skip processing.
            if viewer_annotation is None: continue
        
            # If no coordinates, skip processing.
            coordinates = viewer_annotation.get("coordinates", None)
            if coordinates is None or coordinates.shape[0] == 0: continue

            # Get the annotation style from the styler (active = None)
            # If not visible, return None to skip plotting.
            annotation_style = self.prefs.get_annotation_style(annotation)
            if not annotation_style["visible"]: continue

            # Get annotation color and point types for the current annotation
            annotation_color = self._rgb_to_k3dcolor(annotation_style["color"])
            point_types = viewer_annotation.get("point_types", None)

            # Get number of annotation vertex (line) coordinates and point types
            vertices  = coordinates.astype(np.float32)
            positions = vertices.copy() # copy!

            # Check if not all vertices are all fixed points, all to lines.
            if not np.all(point_types == self._POINT_FIXED):
                # Prepare the vertices and line colors arrays
                all_vertices  = np.vstack((all_vertices, vertices, coord_sep))
                vertex_colors = np.full(vertices.shape[0], annotation_color)
                all_lcolors   = np.hstack((all_lcolors, vertex_colors, color_sep))

            # Get annotation user points and point types
            interp_mask = point_types != self._POINT_INTERP
            positions   = positions[interp_mask]
            point_types = point_types[interp_mask]

            # Prepare the positions and point colors arrays
            point_colors  = np.full(positions.shape[0], annotation_color)
            all_positions = np.vstack((all_positions, positions))
            all_pcolors   = np.hstack((all_pcolors, point_colors))
    
        # If no coordinates, return None to skip plotting.
        if all_vertices.shape[0] == 0 and all_positions.shape[0] == 0: return None

        # Define scaled down linewidth and pointsize
        line_width = float(self.prefs.get_viewer_style("line_width") * size_scale)
        point_size = float(self.prefs.get_viewer_style("point_size") * size_scale)

        return { 
            "line": {
                "vertices" : all_vertices.astype(np.float32),
                "width"    : float(line_width), 
                "colors"   : all_lcolors.astype(np.uint32)
            },
            "points": {
                "positions"  : all_positions.astype(np.float32), 
                "point_size" : float(point_size),
                "colors"     : all_pcolors.astype(np.uint32)
            }
        }
    
    # Figure Clear Method ------------------------------------------------------

    def _clear_figure(self):
        """Hide all k3d layers."""
        self._k3dmesh_cortex.visible       = False
        self._k3dmesh_overlay.visible      = False
        self._k3dline_background.visible   = False
        self._k3dpoints_background.visible = False
        self._k3dline_dependent.visible    = False
        self._k3dpoints_dependent.visible  = False
        self._k3dline_active.visible       = False
        self._k3dpoints_active.visible     = False

    # Figure Refresh Methods ---------------------------------------------------
    
    def _redraw_cortex(self):
        """Update the cortex mesh and overlay layers from current data."""
        # Update cortex mesh.
        cortex_kwargs = self._prep_cortex()
        for key, val in cortex_kwargs.items():
            setattr(self._k3dmesh_cortex, key, val)
        self._k3dmesh_cortex.visible = True
 
        # Update overlay mesh.
        overlay_kwargs = self._prep_overlay()
        if overlay_kwargs is None:
            self._k3dmesh_overlay.visible = False
        else:
            for key, val in overlay_kwargs.items():
                setattr(self._k3dmesh_overlay, key, val)
            self._k3dmesh_overlay.visible = True
    

    def _redraw_layer(self, k3d_line, k3d_points, k3d_kwargs = None):
        """Apply prepared data to a line/points k3d layer pair.
 
        Parameters
        ----------
        k3d_kwargs : dict or None
            Output from ``_prep_single_annotation`` or
            ``_prep_multiple_annotations``.  If ``None``, the layers
            are hidden.

        line_obj : k3d.Line
            The k3d line object to update.

        points_obj : k3d.Points
            The k3d points object to update.
        """
        # If no kwargs, hide both line and points layers.  
        if k3d_kwargs is None:
            k3d_line.visible   = False
            k3d_points.visible = False
        else:
            # Update the line layer (interpolated between points)
            for key, val in k3d_kwargs["line"].items():
                setattr(k3d_line, key, val)
            k3d_line.visible = True

            # Update the points layer (fixed + user points)
            for key, val in k3d_kwargs["points"].items():
                setattr(k3d_points, key, val)
            k3d_points.visible = True
 

    def _redraw_annotations(self, active = True, dependent = False,
                            background = False):
        """Refresh annotation layers from cached viewer annotations.
 
        Parameters
        ----------
        active : bool, optional
            Refresh the active annotation layers.  Defaults to ``True``.

        dependent : bool, optional
            Refresh the dependent annotation layers.
            Defaults to ``False``.

        background : bool, optional
            Refresh the independent background annotation layers.
            Defaults to ``False``.
        """

        # Seperate the annotaitons by type.
        active_annotation     = self.editor.active
        dependent_annotations = self.annot_cfg.fixed_dependencies.get(active_annotation, [])

        nonbackground_annotations = set([active_annotation, *dependent_annotations])
        background_annotations    = list(set(self.annot_cfg.names) - set(nonbackground_annotations))

        if active:
            k3d_kwargs = self._prep_single_annotation(active_annotation, None)
            self._redraw_layer(
                k3d_line   = self._k3dline_active, 
                k3d_points = self._k3dpoints_active, 
                k3d_kwargs = k3d_kwargs
            )

        if dependent: 
            # Determine which annotations depend on the active annotation.
            k3d_kwargs = self._prep_multiple_annotations(
                dependent_annotations, size_scale = 0.5)
            self._redraw_layer(
                k3d_line   = self._k3dline_dependent,
                k3d_points = self._k3dpoints_dependent,
                k3d_kwargs = k3d_kwargs
            )

        if background:
            k3d_kwargs = self._prep_multiple_annotations(
                background_annotations, size_scale = 0.5)
            self._redraw_layer(
                k3d_line   = self._k3dline_background,
                k3d_points = self._k3dpoints_background,
                k3d_kwargs = k3d_kwargs
            )


    def redraw_viewer(self, clear = False, cortex = False, 
                      active = True, dependent = False, background = False):
        """Redraw the requested viewer layers.
 
        This is the primary entry point called by ``FigurePanel``.
        All updates are batched between ``auto_rendering`` toggles
        to avoid intermediate renders.
 
        Parameters
        ----------
        cortex : bool, optional
            Refresh the cortex mesh and overlay layers.
            Defaults to ``False``.
        active : bool, optional
            Refresh the active annotation layers.
            Defaults to ``True``.
        dependent : bool, optional
            Refresh the dependent annotation layers.
            Defaults to ``False``.
        background : bool, optional
            Refresh the independent background annotation layers.
            Defaults to ``False``.
        """
        # Disable auto-rendering to batch updates and avoid intermediate renders.
        self._figure.auto_rendering = False

        # Apply updates to the figure layers based on the specified flags.
        if clear:  self._clear_figure()
        if cortex: self._redraw_cortex()
        if active or dependent or background:
            self._redraw_annotations(
                active     = active, 
                dependent  = dependent, 
                background = background
            )

        # Re-enable auto-rendering and trigger render after all updates are applied.
        self._figure.auto_rendering = True
        self._figure.render()


    def resize(self):
        """Resize the viewer figure.
 
        Parameters
        ----------
        new_size : tuple of (width, height)
            New figure size in pixels.
        """
        # Update the viewer size.
        new_size = self.prefs.get_display("viewer_size")
        
        # Set k3d figure height
        self._figure.height = new_size

        # Set new VBox layout height
        self.layout.height = f"{new_size}px"
        self.layout.width  = f"{new_size}px"

        # TODO: somethign about the camera resetting on resize? not sure yet.


    def _on_camera_change(self, change):
        """Handle camera changes from user interaction.
 
        Parameters
        ----------
        change : dict
            Camera change event from k3d.  Contains new camera parameters
            in ``change["new"]``.
        """
        # Update preferences with camera parameters as it changes.
        self.prefs.set_display("viewer_camera", change.new)