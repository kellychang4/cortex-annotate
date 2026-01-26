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

import ipyvolume as ipv
import neuropythy as ny
import ipywidgets as ipw

# The Cortex Control Panel Widget ----------------------------------------------

class CortexControlPanel(ipw.VBox):
    """
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
            value             = state.inflation_value,
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
            value       = "curvature",    
            description = "Overlay:",
            layout      = { "border": "1px solid green", "width" : "94%", "margin" : "0% 3% 3% 3%" }
        )

        # Assemble the control panel with children widgets
        ch = [
            # HTML header
            self._make_html_header(),
            # Cortex Viewer title
            ipw.HTML("<b style=\"margin: 0% 3% 0% 3%;\">Cortex Viewer:</b>"),
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
            # Inflation slider
            self.inflation_slider,
            # Horizontal line
            self._make_hline(), 
            # Overlay dropdown
            self.overlay_dropdown,
            # 
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
            hemisphere = state.hemisphere.lower()
            return "Left Hemisphere" if hemisphere.startswith("l") else "Right Hemisphere"
        else: # key == "annotation"
            return state.annotation


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
        """
        A method to register a callback for changes in the inflation slider.
        """
        self.inflation_slider.observe(callback, names = "value")


    def observe_overlay_dropdown(self, callback):
        """
        A method to register a callback for changes in the overlay dropdown.
        """
        self.overlay_dropdown.observe(callback, names = "value")


# The Cortex Figure Panel -----------------------------------------------

class CortexFigurePanel(ipw.HBox):
    """
    The panel that contains the 3D cortex plot for the Cortex Viewer tool.
    """
    
    def __init__(self, state):
        # Create a figure background
        self.figure = ipv.figure(
            width     = 600, 
            height    = 600, 
            animation = 60, 
            animation_exponent = 1
        )
                    
        # Draw the cortex plot (meshes) onto the figure
        self.figure = ny.cortex_plot(
            mesh   = state.mesh, 
            view   = "left" if state.hemisphere == "lh" else "right",
            figure = self.figure, 
        )
        
        # Add scatter plots for annotations ( points and lines )
        self.figure.scatters = [
            # Annotations surface points (user selected)
            self._create_surface_annotations(state),
            # Annotation surface paths (interpolated)
            self._create_surface_paths(state)
        ]

        super().__init__([self.figure])


    def _empty_scatter(self):
        """Create an empty scatter plot."""
        return ipv.scatter(0, 0, 0, size = 0, marker = "sphere")
    

    def _create_surface_annotations(self, state):
        """Initialize the scatter plot for the surface annotations."""
        if state.surface_annotations.shape[1] == 0:
            return self._empty_scatter()
        return ipv.scatter(
            state.surface_annotations[0, :], 
            state.surface_annotations[1, :], 
            state.surface_annotations[2, :], 
            size   = 1.5, 
            color  = "magenta", 
            marker = "sphere"
        )


    def _create_surface_paths(self, state):
        """Initialize the scatter plot for the surface paths."""
        if state.surface_paths.shape[1] == 0:
            return self._empty_scatter()
        return ipv.scatter(
            state.surface_paths[0, :], 
            state.surface_paths[1, :], 
            state.surface_paths[2, :], 
            size   = 0.5, 
            color  = "blue", 
            marker = "sphere"
        )


    def refresh_figure(self, state):
        """Update the existing figure with the new mesh coordinates."""
        # Update the figure panel's mesh values
        cortex_mesh       = self.figure.meshes[0] 
        cortex_mesh.x     = state.coordinates[0, :]
        cortex_mesh.y     = state.coordinates[1, :]
        cortex_mesh.z     = state.coordinates[2, :]
        cortex_mesh.color = state.color
        
        # Update the annotation points (user selected)
        scatter_points       = self.figure.scatters[0]
        scatter_points.x     = state.surface_annotations[0, :]
        scatter_points.y     = state.surface_annotations[1, :]
        scatter_points.z     = state.surface_annotations[2, :]
        scatter_points.color = "magenta"
        scatter_points.size  = 1.5
        
        scatter_lines       = self.figure.scatters[1]
        scatter_lines.x     = state.surface_paths[0, :]
        scatter_lines.y     = state.surface_paths[1, :]
        scatter_lines.z     = state.surface_paths[2, :]
        scatter_lines.color = "blue"
        scatter_lines.size  = 0.5