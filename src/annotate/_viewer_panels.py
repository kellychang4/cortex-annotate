# -*- coding: utf-8 -*-
################################################################################
# annotate/_viewer_panels.py

"""
Core implementation code for the CortexViewer's subpanels. 

CortexViewer is implemented in terms of two main subpanels:
1) CortexControlPanel: The control panel that contains viewer information about 
    the dataset, target keys, and annotations, as well as user controls for 
    adjusting the inflation level of the cortex plot.
2) CortexFigurePanel: The figure panel that contains the 3D cortex plot.

Each panel is implemented as an ipywidget, and the CortexViewer class
assembles these two panels into a single interface.

CortexViewer is implmented in _viewer.py.
"""

# Imports ----------------------------------------------------------------------

import k3d
import numpy as np
import ipywidgets as ipw
from matplotlib.colors import to_rgb

# The Cortex Control Panel  ----------------------------------------------------

class CortexControlPanel(ipw.VBox):
    """Control Panel for the Cortex Viewer Tool.

    The panel that contains the information and user controls for the Cortex 
    Viewer tool.
    """

    def __init__(self, state, background_color = "#f0f0f0"):
        """Initialize the Cortex Control Panel with the given state."""
        # Store the state.
        self.state = state

        # Extract the target keys
        self.target_keys = list(state.targets.keys())

        # Create information boxes
        self.infobox = {} # initialize infobox dictionary
        infobox_keys = ( "dataset", *self.target_keys, "annotation" ) 
        for key in infobox_keys: # for each key in dataset and selection
            self.infobox[key] = self._init_infobox(key)
        
        # Create the inflation slider widget
        self.inflation_slider = self._init_inflation_slider()

        # Create the overlay dropdown widget
        self.overlay_dropdown = self._init_overlay_dropdown()

        # Create the overlay alpha slider widget
        self.overlay_slider = self._init_overlay_slider()

        # Create point size slider (user-defined points)
        self.point_size_slider = self._init_point_size_slider()

        # Create line size slider (interpolated)
        self.line_size_slider = self._init_line_size_slider()

        # Create the number of line points slider widget
        self.line_points_slider = self._init_line_points_slider()

        # Assemble the control panel with children widgets
        children = [
            # Cortex Viewer title
            self._make_section_title("Selection:"),
            # Dataset infoboxes
            self.infobox["dataset"],
            # Targets infoboxes
            *[ self.infobox[key] for key in self.target_keys ],
            # Annotation infobox
            self.infobox["annotation"],
            # Horizontal line
            self._make_hline(), 
            # Style Options title
            self._make_section_title("Style:"),
            # Inflation slider
            self.inflation_slider,
            # Overlay dropdown
            self.overlay_dropdown,
            # Overlay alpha slider
            self.overlay_slider, 
            # Point size slider
            self.point_size_slider, 
            # Line size slider
            self.line_size_slider,
            # Line interpolation slider
            self.line_points_slider,
        ]
        vbox = ipw.VBox(children, layout = { "width": "250px" })

        # Wrap the whole thing in an accordion so that it can be collapsed.
        accordion = ipw.Accordion((vbox, ), selected_index = 0)
        self.add_class("cortex-control-panel")

        # Initalize the VBox with the accordion.
        super().__init__(
            children = [ self._make_html_header(background_color), accordion ],  
            layout   = { "border": "0px", "height": "100%" }
        )        


    @classmethod
    def _make_html_header(cls, background_color = "#f0f0f0"):
        return ipw.HTML(f"""
            <style>
                .cortex-control-panel .jupyter-widget-Collapse-open {{
                    background-color: white;
                }}
                .cortex-control-panel .jupyter-widget-Collapse-header {{
                    background-color: white;
                    border-width: 0px;
                    padding: 0px;
                }}
                .cortex-control-panel .jupyter-widget-Collapse-contents {{
                    background-color: {background_color};
                    padding: 2px;
                    border-width: 1px;
                    border-style: solid;
                    border-color: lightgray;
                }}
                .info-label {{
                    margin-right: 8px;
                    text-align: right;
                }}
                .info-value {{
                    background-color: white;
                    border: 1px solid rgb(158, 158, 158);
                    padding: 0px 8px 0px 8px;
                    text-align: left;
                }}
                .cortex-control-widgets {{
                    margin: 1% 3% 1% 3%;
                    width: 94%;
                }} 
                .cortex-control-widgets .widget-readout {{
                    min-width: 50px;
                }}
                .cortex-hline {{
                    border: 1px solid lightgray;
                    margin: 3%;
                    height: 0px;
                    width: 94%;
                }}
            </style>
        """)
    

    @classmethod
    def _make_section_title(cls, title):
        return ipw.HTML(f"<b style=\"margin: 0% 3% 0% 3%;\">{title}</b>")
    

    @classmethod
    def _make_hline(cls):
        return ipw.HTML("""<div class="cortex-hline"></div>""")


    def _prep_infobox_value(self, key):
        """Prepare the infobox value for display."""
        if key == "dataset":
            return self.state.dataset
        elif key == "annotation":
            return self.state.annotation
        else: # key in state.targets
            return self.state.targets[key]


    def _make_infobox_value(self, value):
        """Make the infobox value for display."""
        return f"""<div class="info-value">{value}</div>"""
    

    def _init_infobox(self, key):
        """Update an information box for the given key and state."""
        value = self._prep_infobox_value(key)
        return ipw.Box([
            ipw.HTML(f"""<div class="info-label">{key.capitalize()}:</div>""",
                     layout = { "width": "40%", "margin": "0px" }),
            ipw.HTML(self._make_infobox_value(value), 
                     layout = { "width": "60%", "margin": "0px" }),
        ], layout = ipw.Layout(
            display = "flex",
            margin  = "1% 3% 1% 3%",
        ))


    def _init_inflation_slider(self):
        inflation_slider = ipw.IntSlider(
            value             = self.state.style["inflation_percent"],
            min               = 0,
            max               = 100,
            step              = 1,
            description       = "Inflation %:",
            continuous_update = False,
            orientation       = "horizontal",
            readout           = True,
            readout_format    = "d", 
        )
        inflation_slider.add_class("cortex-control-widgets")
        return inflation_slider


    def _init_overlay_dropdown(self):
        overlay_dropdown = ipw.Dropdown(
            options     = [
                ( "None", "curvature" ), 
                ( "Polar Angle", "angle" ), 
                ( "Eccentricity", "eccen" ), 
                ( "Variance Explained", "vexpl" )
            ],
            value       = self.state.style["overlay"],    
            description = "Overlay:",
        )
        overlay_dropdown.add_class("cortex-control-widgets")
        return overlay_dropdown


    def _init_overlay_slider(self):
        overlay_slider = ipw.FloatSlider(
            value             = self.state.style["overlay_alpha"],
            min               = 0.0,
            max               = 1.0,
            step              = 0.1,
            description       = "Alpha:",
            continuous_update = False,
            orientation       = "horizontal",
            readout           = True,
            readout_format    = ".1f", 
        )
        overlay_slider.add_class("cortex-control-widgets")
        return overlay_slider
    

    def _init_point_size_slider(self):
        point_size_slider = ipw.FloatSlider(
            value             = self.state.style["point_size"],
            min               = 0.5,
            max               = 5,
            step              = 0.1,
            description       = "Point Size:",
            continuous_update = False,
            orientation       = "horizontal",
            readout           = True,
            readout_format    = ".1f", 
        )
        point_size_slider.add_class("cortex-control-widgets")
        return point_size_slider
    

    def _init_line_size_slider(self):
        line_size_slider = ipw.FloatSlider(
            value             = self.state.style["line_size"],
            min               = 0.5,
            max               = 5,
            step              = 0.1,
            description       = "Line Size:",
            continuous_update = False,
            orientation       = "horizontal",
            readout           = True,
            readout_format    = ".1f", 
        )
        line_size_slider.add_class("cortex-control-widgets")
        return line_size_slider


    def _init_line_points_slider(self):
        line_points_slider = ipw.IntSlider(
            value             = self.state.style["line_points"],
            min               = 2,
            max               = 20,
            step              = 1,
            description       = "Line Points:",
            continuous_update = False,
            orientation       = "horizontal",
            readout           = True,
            readout_format    = "d", 
        )
        line_points_slider.add_class("cortex-control-widgets")
        return line_points_slider
    

    def refresh_infobox(self, key):
        """Refresh the control panel display to reflect updated infobox values."""
        value = self._prep_infobox_value(key)
        self.infobox[key].children[1].value = self._make_infobox_value(value)


    def observe_inflation_slider(self, callback): 
        """A method to register a callback for changes in the inflation slider."""
        self.inflation_slider.observe(callback, names = "value")


    def observe_overlay_dropdown(self, callback):
        """A method to register a callback for changes in the overlay dropdown."""
        self.overlay_dropdown.observe(callback, names = "value")

    
    def observe_overlay_slider(self, callback):
        """A method to register a callback for changes in the overlay alpha slider."""
        self.overlay_slider.observe(callback, names = "value")


    def observe_point_size_slider(self, callback):
        """A method to register a callback for changes in the point size slider."""
        self.point_size_slider.observe(callback, names = "value")


    def observe_line_size_slider(self, callback):
        """A method to register a callback for changes in the line size slider."""
        self.line_size_slider.observe(callback, names = "value")

    
    def observe_line_points_slider(self, callback):
        """A method to register a callback for changes in the line points slider."""
        self.line_points_slider.observe(callback, names = "value")

# The Cortex Figure Panel ------------------------------------------------------

class CortexFigurePanel(ipw.GridBox):
    """Cortex Figure Panel.

    The panel that contains the 3D cortex plot for the Cortex Viewer tool.
    """
    
    def __init__(self, state, height = 400):
        # Store the state.
        self.state = state

        # Create a figure background (k3d plot)
        self.figure = k3d.plot(
            # K3d does not use width, just fills the space
            height            = height, 
            grid_visible      = False,
            camera_auto_fit   = False,
            menu_visibility   = False,
            camera_fov        = 60,
            axes_helper       = 0, # remove axes direction helper
            camera_zoom_speed = 1.5,
        )

        # Create and add the cortex mesh to the figure
        self.k3dmesh_cortex = self._init_cortex()

        # Create and add the overlay mesh to the figure
        self.k3dmesh_overlay = self._init_overlay()
        
        # Create active points to figure 
        self.k3dpoints_active = self._init_active_points()

        # Create background points to figure
        self.k3dpoints_background = self._init_background_points()

        # Add the meshes and points to the figure and initial render
        self.figure += self.k3dmesh_cortex
        self.figure += self.k3dmesh_overlay 
        self.figure += self.k3dpoints_active 
        self.figure += self.k3dpoints_background 

        # Set initial camera values
        self.figure.camera = [-160, -10, -6, 15, -30, 0, 0, 0, 1]

        # Define layout (GridBox as HBox)
        layout = ipw.Layout(
            grid_template_columns = "1fr",
            height = f"{height}px",
            width  = f"{width}px" if width is not None else "100%"
        )
        super().__init__([ self.figure ], layout = layout)
        self.figure.camera_auto_fit = False
        

    def _rgb_to_k3dcolor(self, colors):
        """Converts a matplotlib color (RGB) into a hex integer for k3d.

        If the given color is a matrix of RGB triples, then a list of
        integers, one per row, is returned. 
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
            # NOTE: k3d does not support alpha in the color integer, ignor
            return np.array(
                [ ((r << 16) | (g << 8) | b) for r, g, b, _ in colors ], 
                dtype = np.uint32
            )
        else:
            raise ValueError("Color matrices must be RGB (Nx3) or RGBA (Nx4).")


    def _init_mesh(self):
        """Initialize an empty and invisible mesh."""
        mesh = k3d.mesh(
            vertices     = np.array([[0, 0, 0]], dtype = np.float32), 
            indices      = np.array([[0, 0, 0]], dtype = np.uint32),
            colors       = np.array([0x000000], dtype = np.uint32),
            wireframe    = False,
            flat_shading = False
        )
        mesh.visible = False
        return mesh


    def _prep_cortex(self):
        """Prepare the data for the cortex mesh."""
        vertices  = self.state.coordinates.T.astype(np.float32) # (n_vertices, 3)
        indices   = self.state.fsaverage[self.state.hemisphere]["tesselation"].T.astype(np.uint32)
        curvature = self._rgb_to_k3dcolor(self.state.curvature)
        return { 
            "vertices" : vertices, 
            "indices"  : indices, 
            "colors"   : curvature 
        }
    

    def _prep_overlay(self):
        """Prepare the data for the cortex overlay mesh."""
        # If no overlay, return None
        if self.state.style["overlay"] == "curvature":
            return None

        # Else return the overlay mesh keyword arguments
        return {
            **self._prep_cortex(), # same vertices + indices
            "colors"   : self._rgb_to_k3dcolor(self.state.color),
            "opacity"  : self.state.style["overlay_alpha"]
        }
    

    def _init_cortex(self):
        """Initialize the cortex mesh."""
        cortex_kwargs = self._prep_cortex()
        return k3d.mesh(**cortex_kwargs, wireframe = False, flat_shading = False)          
    

    def _init_overlay(self):
        """Initialize the cortex overlay mesh."""
        overlay_kwargs = self._prep_overlay()
        if overlay_kwargs is None:
            return self._init_mesh()
        return k3d.mesh(**overlay_kwargs, wireframe = False, flat_shading = False)


    def _init_points(self):
        """Initialize an empty and invisible points plot."""
        points = k3d.points(
            positions  = np.array([[0, 0, 0]], dtype = np.float32),
            shader     = "3d"
        )
        points.visible = False
        return points
    

    def _prep_active_points(self):
        """Prepare the data for the active annotation."""
        # Get the annotation coordinates and anchor points
        annotation  = self.state.annotation
        coordinates = self.state.surface_annotations[annotation]["coordinates"] 
        anchor      = self.state.surface_annotations[annotation]["anchor"]
        
        print("Annotation:", annotation)
        print("Coordinates shape:", coordinates.shape)
        print("Coordinates shape:", coordinates.shape)
        print("Anchor shape:", anchor.shape)

        # Get the annotation style from the styler (active = None)
        annotation_style = self.state.styler(None)
        # If not visible or no coordinates, return None
        if (not annotation_style["visible"]) or (coordinates is None):
            return None
                
        # Get number of annotation points
        n_points = coordinates.shape[1]

        # Prepare scatter sizes by anchor points
        point_sizes = np.ones(n_points) * self.state.style["line_size"]
        point_sizes[anchor == 1] = self.state.style["point_size"]

        # Prepare colors
        point_colors = self._rgb_to_k3dcolor(annotation_style["color"])
        point_colors = np.tile(point_colors, (n_points, 1))

        # Return the scatter plot keyword arguments
        return { 
            "positions"   : coordinates.T.astype(np.float32), 
            "point_sizes" : point_sizes.astype(np.float32), 
            "colors"      : point_colors
        }


    def _prep_background_points(self):
        """Prepare the data for the background annotations."""
        # Get the list of annotations excluding the selected one
        annotation = self.state.annotation
        annotation_list = list(self.state.surface_annotations.keys())
        annotation_list.remove(annotation)
        
        # Initialize arrays for all colors and coordinates
        all_colors = np.empty((0, 3)) 
        all_coordinates = np.empty((3, 0)) 

        for annotation in annotation_list: # for each annotation
            coordinates = self.state.surface_annotations[annotation]["coordinates"] 

            annotation_style = self.state.styler(annotation)
            if annotation_style["visible"] and coordinates is not None: 
                # Get colors for the annotation points
                point_colors = self._rgb_to_k3dcolor(annotation_style["color"])
                point_colors = np.tile(point_colors, (coordinates.shape[-1], 1)) 

                # Concatenate coordinates and colors
                all_coordinates = np.hstack((all_coordinates, coordinates))
                all_colors = np.vstack((all_colors, point_colors))

        # If no coordinates, return None
        if all_coordinates.shape[1] == 0:
            return None
        
        # Return the scatter plot keyword arguments
        return { 
            "positions"  : all_coordinates.T.astype(np.float32),
            "point_size" : self.state.style["line_size"].astype(np.float32), 
            "colors"     : all_colors 
        }
            
    
    def _init_active_points(self):
        """Initialize the points plot for the active annotation."""
        points_kwargs = self._prep_active_points()
        if points_kwargs is None:
            return self._init_points()
        return k3d.points(**points_kwargs, shader = "3d")


    def _init_background_points(self):
        """Initialize the points plot for the background annotations."""
        points_kwargs = self._prep_background_points()
        if points_kwargs is None:
            return self._init_points()
        return k3d.points(**points_kwargs, shader = "3d")


    def refresh_figure(self):
        """Update the existing figure mesh coordinates and annotation points."""
        # Disable auto rendering for performance
        self.figure.auto_rendering = False

        # Update the cortex mesh values
        cortex_kwargs = self._prep_cortex()
        self.k3dmesh_cortex.vertices = cortex_kwargs["vertices"]
        self.k3dmesh_cortex.indices  = cortex_kwargs["indices"]
        self.k3dmesh_cortex.colors   = cortex_kwargs["colors"] # curvature

        # Update the overlay mesh values 
        overlay_kwargs = self._prep_overlay()
        if self.state.style["overlay"] == "curvature":
            self.k3dmesh_overlay.visible = False
        else:
            self.k3dmesh_overlay.vertices = overlay_kwargs["vertices"]
            self.k3dmesh_overlay.indices  = overlay_kwargs["indices"]
            self.k3dmesh_overlay.colors   = overlay_kwargs["colors"]
            self.k3dmesh_overlay.opacity  = overlay_kwargs["opacity"]
            self.k3dmesh_overlay.visible  = True
        
        # Re-enable auto rendering after cortex and overlay values
        self.figure.auto_rendering = True
        self.figure.render()

        # Update the surface active annotation
        active_kwargs = self._prep_active_points()
        if active_kwargs is None:
            self.k3dpoints_active.visible = False
        else:
            self.k3dpoints_active.positions   = active_kwargs["positions"]
            self.k3dpoints_active.point_sizes = active_kwargs["point_sizes"]
            self.k3dpoints_active.colors      = active_kwargs["colors"]
            self.k3dpoints_active.visible     = True
        
        # Update the surface background annotation
        background_kwargs = self._prep_background_points()
        if background_kwargs is None:
            self.k3dpoints_background.visible = False
        else:
            self.k3dpoints_background.positions   = background_kwargs["positions"]
            self.k3dpoints_background.point_sizes = background_kwargs["point_sizes"]
            self.k3dpoints_background.colors      = background_kwargs["colors"]  
            self.k3dpoints_background.visible     = True

