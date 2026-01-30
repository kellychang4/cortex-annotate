# -*- coding: utf-8 -*-
################################################################################
# annotate/_viewer_panels.py

"""
Core implementation code for the CortexViewer's subpanels. 

CortexViewer is implemented in terms of two main subpanels:
1) CortexControlPanel: The control panel that contains viewer information about 
    the dataset, participant, and hemisphere, as well as user controls for 
    adjusting the inflation level of the cortex plot.
2) CortexFigurePanel: The figure panel that contains the 3D cortex plot.

Each panel is implemented as an ipywidget, and the CortexViewer class
assembles these two panels into a single interface.

CortexViewer is implmented in _viewer.py.
"""

# Imports ----------------------------------------------------------------------

import numpy as np
import ipyvolume as ipv
import neuropythy as ny
import ipywidgets as ipw
from matplotlib.colors import to_rgb

# The Cortex Control Panel  ----------------------------------------------------

class CortexControlPanel(ipw.VBox):
    """Control Panel for the Cortex Viewer Tool.

    The panel that contains the information and user controls for the Cortex 
    Viewer tool.
    """

    def __init__(self, state):
        """Initialize the Cortex Control Panel with the given state."""

        # Create information boxes
        self.infobox = {} # initialize infobox dictionary
        for key in ( "dataset", "participant", "hemisphere", "annotation" ):
            self.infobox[key] = self._make_infobox(state, key)
        
        # Create the inflation slider widget
        self.inflation_slider = ipw.IntSlider(
            value             = state.style["inflation_percent"],
            min               = 0,
            max               = 100,
            step              = 1,
            description       = "Inflation %:",
            continuous_update = False,
            orientation       = "horizontal",
            readout           = True,
            readout_format    = "d", 
            layout            = { "border": "1px solid purple", "width" : "94%", "margin" : "0% 3% 3% 3%" } 
        )

        # Create the overlay dropdown widget
        self.overlay_dropdown = ipw.Dropdown(
            options     = [
                ( "None", "curvature" ), 
                ( "Polar Angle", "angle" ), 
                ( "Eccentricity", "eccen" ), 
                ( "Variance Explained", "vexpl" )
            ],
            value       = state.style["overlay"],    
            description = "Overlay:",
            layout      = { "border": "1px solid green", "width" : "94%", "margin" : "0% 3% 3% 3%" }
        )

        # Create the overlay alpha slider widget
        self.overlay_slider = ipw.FloatSlider(
            value             = state.style["overlay_alpha"],
            min               = 0.0,
            max               = 1.0,
            step              = 0.01,
            description       = "Transparency:",
            continuous_update = False,
            orientation       = "horizontal",
            readout           = True,
            readout_format    = ".1f", 
            layout            = { "border": "1px solid purple", "width" : "94%", "margin" : "0% 3% 3% 3%" } 
        )

        # Create scatter size slider (user-defined)
        self.point_size_slider = ipw.FloatSlider(
            value             = state.style["point_size"],
            min               = 0.5,
            max               = 5,
            step              = 0.1,
            description       = "Point Size:",
            continuous_update = False,
            orientation       = "horizontal",
            readout           = True,
            readout_format    = ".1f", 
            layout            = { "border": "1px solid orange", "width" : "94%", "margin" : "0% 3% 3% 3%" } 
        )

        # Create line size slider (interpolated)
        self.line_size_slider = ipw.FloatSlider(
            value             = state.style["line_size"],
            min               = 0.5,
            max               = 5,
            step              = 0.1,
            description       = "Line Point Size:",
            continuous_update = False,
            orientation       = "horizontal",
            readout           = True,
            readout_format    = ".1f", 
            layout            = { "border": "1px solid cyan", "width" : "94%", "margin" : "0% 3% 3% 3%" } 
        )

        # Create the number of line points slider widget
        self.line_points_slider = ipw.IntSlider(
            value             = state.style["line_points"],
            min               = 2,
            max               = 20,
            step              = 1,
            description       = "Line Points:",
            continuous_update = False,
            orientation       = "horizontal",
            readout           = True,
            readout_format    = "d", 
            layout            = { "border": "1px solid purple", "width" : "94%", "margin" : "0% 3% 3% 3%" } 
        )


        # Assemble the control panel with children widgets
        ch = [
            # HTML header
            self._make_html_header(),
            # Cortex Viewer title
            ipw.HTML("<b style=\"margin: 0% 3% 0% 3%;\">Selection:</b>"),
            # Dataset label
            self.infobox["dataset"],
            # Participant label
            self.infobox["participant"],
            # Hemisphere label
            self.infobox["hemisphere"],
            # Annotation label
            self.infobox["annotation"],
            # Horizontal line
            self._make_hline(), 
            # Style Options title
            ipw.HTML("<b style=\"margin: 0% 3% 0% 3%;\">Style:</b>"),
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

        # Set the overall layout into the VBox
        super().__init__(ch, layout = { "border": "1px solid blue", "width": "250px", "height": "100%" })


    @classmethod
    def _make_html_header(cls):
        return ipw.HTML(f"""
            <style>
                .info-label {{
                    text-align: right;
                }}
                .info-value {{
                    text-align: left;
                    padding-left: 5%;
                }}
            </style>
        """)


    @classmethod
    def _make_hline(cls):
        return ipw.HTML("""
            <style> 
                .cortex-viewer-hline {
                    border: 1px solid lightgray;
                    height: 0px;
                    width: 94%;
                    margin: 3%;
                } 
            </style>
            <div class="cortex-viewer-hline"></div>
        """)


    def _format_infobox_value(self, state, key):
        if key == "dataset":
            return state.dataset
        elif key == "participant":
            return state.participant
        elif key == "hemisphere":
            return state.convert_hemisphere(state.hemisphere)
        else: # key == "annotation"
            return state.selected_annotation


    def _make_infobox_value(self, value):
        """Create a value display for the given infobox content."""
        return f"""<div class="info-value">{value}</div>"""
    

    def _make_infobox(self, state, key):
        """Update an information box for the given key and state."""
        content = self._format_infobox_value(state, key)
        return ipw.Box([
            ipw.HTML(f"""<div class="info-label"><b>{key.capitalize()}</b>:</div>""",
                     layout = { "width": "40%", "margin": "0px" }),
            ipw.HTML(self._make_infobox_value(content), 
                     layout = { "width": "60%", "margin": "0px" }),
        ], layout = ipw.Layout(
            display = "flex",
            width   = "91%",
            border  = "1px solid lightgray",
            margin  = "0% 3% 0% 6%"
        ))


    def refresh_infobox(self, state, key):
        """Refresh the control panel display to reflect updated infobox values."""
        content = self._format_infobox_value(state, key)
        self.infobox[key].children[1].value = self._make_infobox_value(content)


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

class CortexFigurePanel(ipw.HBox):
    """Cortex Figure Panel.

    The panel that contains the 3D cortex plot for the Cortex Viewer tool.
    """
    
    def __init__(
            self, 
            state, 
            width     = 600, 
            height    = 400, 
            animation = 60, 
            animation_exponent = 1
        ):
        # Create a figure background
        self.figure = ipv.figure(
            width              = width, 
            height             = height, 
            animation          = animation, 
            animation_exponent = animation_exponent
        )
                    
        # Draw the cortex plot (meshes) onto the figure
        self.figure = ny.cortex_plot(
            mesh   = state.mesh, 
            view   = "left" if state.hemisphere == "lh" else "right",
            figure = self.figure, 
        )
        
        # Add scatter plots for annotations ( points and lines )
        self.figure.scatters = [
            # Active annotation points
            self._init_active_scatter(state),
            # Background annotation points
            self._init_background_scatter(state),
        ]

        super().__init__([self.figure])


    def _init_scatter(self):
        """Initialize an empty scatter plot."""
        return ipv.scatter(0, 0, 0, visible = False, marker = "sphere")
    

    def _prep_active_scatter(self, state):
        """Prepare the data for the active annotation."""
        # Get the selected annotation
        selected_annotation = state.selected_annotation
        
        # Get the annotation style, coordinates, and anchor points
        annotation_style = state.annotation_styles[selected_annotation]
        coordinates = state.surface_annotations[selected_annotation]["coordinates"] 
        anchor      = state.surface_annotations[selected_annotation]["anchor"]

        # If not visible or no coordinates, return None
        if (not annotation_style["visible"]) or (coordinates is None):
            return None
                
        # Get number of annotation points
        n_points = coordinates.shape[1]

        # Prepare scatter sizes by anchor points
        scatter_sizes = np.ones(n_points) * state.style["line_size"]
        scatter_sizes[anchor == 1] = state.style["point_size"]

        # Prepare colors
        rgb_color = np.array(to_rgb(state.style_active["color"]))
        rgb_color = np.tile(rgb_color, (n_points, 1))

        # Return the scatter plot keyword arguments
        return { 
            "x"     : coordinates[0, :], 
            "y"     : coordinates[1, :], 
            "z"     : coordinates[2, :], 
            "size"  : scatter_sizes, 
            "color" : rgb_color 
        }


    def _prep_background_scatter(self, state):
        """Prepare the data for the background annotations."""
        # Get the list of annotations excluding the selected one
        selected_annotation = state.selected_annotation
        annotation_list = list(state.surface_annotations.keys())
        annotation_list.remove(selected_annotation)
        
        # Initialize arrays for all colors and coordinates
        all_colors = np.empty((0, 3)) 
        all_coordinates = np.empty((3, 0)) 

        for annotation in annotation_list: # for each annotation
            annotation_style = state.annotation_styles[annotation]
            coordinates = state.surface_annotations[annotation]["coordinates"] 

            if annotation_style["visible"] and coordinates is not None: 
                # Get colors for the annotation points
                rgb_color = np.array(to_rgb(annotation_style["color"]))
                rgb_color = np.tile(rgb_color, (coordinates.shape[-1], 1)) 

                # Concatenate coordinates and colors
                all_coordinates = np.hstack((all_coordinates, coordinates))
                all_colors = np.vstack((all_colors, rgb_color))

        # If no coordinates, return None
        if all_coordinates.shape[1] == 0:
            return None
        
        # Return the scatter plot keyword arguments
        return { 
            "x"     : all_coordinates[0,:], 
            "y"     : all_coordinates[1,:], 
            "z"     : all_coordinates[2,:], 
            "size"  : state.style["line_size"], 
            "color" : all_colors 
        }
            
    
    def _init_active_scatter(self, state):
        """Initialize the scatter plot for the active annotation."""
        scatter_kwargs = self._prep_active_scatter(state)
        if scatter_kwargs is None:
            return self._init_scatter()
        return ipv.scatter(**scatter_kwargs, marker = "sphere")


    def _init_background_scatter(self, state):
        """Initialize the scatter plot for the background annotations."""
        scatter_kwargs = self._prep_background_scatter(state)
        if scatter_kwargs is None:
            return self._init_scatter()
        return ipv.scatter(**scatter_kwargs, marker = "sphere")


    def refresh_figure(self, state):
        """Update the existing figure mesh coordinates and annotation points."""
        # Update the figure panel's mesh values
        with self.figure.meshes[0].hold_sync():
            self.figure.meshes[0].x     = state.coordinates[0, :]
            self.figure.meshes[0].y     = state.coordinates[1, :]
            self.figure.meshes[0].z     = state.coordinates[2, :]
            self.figure.meshes[0].color = state.color
        
        # Update the surface active annotation
        active_kwargs = self._prep_active_scatter(state)
        if active_kwargs is None:
            self.figure.scatters[0].visible = False
        else:    
            with self.figure.scatters[0].hold_sync():
                self.figure.scatters[0].x   = active_kwargs["x"]
                self.figure.scatters[0].y   = active_kwargs["y"]
                self.figure.scatters[0].z   = active_kwargs["z"]
            self.figure.scatters[0].size    = active_kwargs["size"]
            self.figure.scatters[0].color   = active_kwargs["color"]
            self.figure.scatters[0].visible = True
        
        # Update the surface background annotation
        background_kwargs = self._prep_background_scatter(state)
        if background_kwargs is None:
            self.figure.scatters[1].visible = False
        else:
            with self.figure.scatters[1].hold_sync():
                self.figure.scatters[1].x   = background_kwargs["x"]
                self.figure.scatters[1].y   = background_kwargs["y"]
                self.figure.scatters[1].z   = background_kwargs["z"]
            self.figure.scatters[1].size    = background_kwargs["size"]
            self.figure.scatters[1].color   = background_kwargs["color"]
            self.figure.scatters[1].visible = True   
