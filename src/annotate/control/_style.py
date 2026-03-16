# -*- coding: utf-8 -*-
################################################################################
# annotate/control/_style.py
#
# DOCSTRING
    
# Imports ----------------------------------------------------------------------

import os.path as op
import ipywidgets as ipw
from functools import partial

from .._widgets import make_section_title, make_hline, darken_color

# The Style Subpanel -----------------------------------------------------------

class StylePanel(ipw.VBox):
    """The subpanel of the control panel containing the style controls."""
    
    _WIDGET_LAYOUT = { "width": "94%", "margin": "0% 3% 0% 3%" } 

    _SLIDER_KWARGS = {
        "readout"           : False,
        "continuous_update" : False,
        "orientation"       : "horizontal",
        "layout"            : _WIDGET_LAYOUT
    }

    _ANNOT_SLIDER_KWARGS = {
        **_SLIDER_KWARGS, 
        "value": 1, "min": 1, "max": 8, "step": 1, 
        "readout": True, 
    }

    __slots__ = (
        "state", "style_dropdown", "visible_checkbox",
        "color_picker", "markersize_slider", "linewidth_slider", 
        "linestyle_dropdown", "style_observers", "style_widgets"
    )
    
    def __init__(self, state):
        # Store the state
        self.state = state

        # Initialize the style controls (annotation).
        self.style_dropdown     = self._init_style_dropdown()
        self.visible_checkbox   = self._init_visible_checkbox()
        self.color_picker       = self._init_color_picker()
        self.markersize_slider  = self._init_markersize_slider()
        self.linewidth_slider   = self._init_linewidth_slider()
        self.linestyle_dropdown = self._init_linestyle_dropdown()

        # Initialize the style controls (cortex).
        self.morph_slider       = self._init_morph_slider()
        self.overlay_dropdown   = self._init_overlay_dropdown()
        self.overlay_slider     = self._init_overlay_alpha_slider()
        self.point_size_slider  = self._init_point_size_slider()
        self.line_width_slider  = self._init_line_width_slider()
        self.line_interp_slider = self._init_line_interp_slider()

        # Initialize the layout toggle button.
        self.layout_toggle = self._init_layout_toggle()

        # Assemble the style panel children.
        children = [
            make_section_title("Style Options"),
            self.style_dropdown,
            make_hline(),
            make_section_title("Annotation Canvas Options"),
            self.visible_checkbox,
            self.color_picker,
            self.markersize_slider,
            self.linewidth_slider,
            self.linestyle_dropdown,
            make_hline(),
            make_section_title("Cortex Viewer Options"),
            self.morph_slider,
            self.overlay_dropdown,
            self.overlay_slider, 
            self.point_size_slider, 
            self.line_width_slider,
            self.line_interp_slider,
            make_hline(),
            make_section_title("Layout Options"),
            self.layout_toggle
        ]
        super().__init__(children, layout = { "margin": "0% 0% 3% 0%" })

        # Set up our observer pattern. We track these manually so that we can
        # call the functions using a parameter order that makes sense.
        self.style_observers = []
        self.style_widgets = {
            "visible"       : self.visible_checkbox,
            "color"         : self.color_picker,
            "markersize"    : self.markersize_slider,
            "linewidth"     : self.linewidth_slider,
            "linestyle"     : self.linestyle_dropdown, 
        }
        for (key, value) in self.style_widgets.items():
            value.observe(partial(self.on_style_change, key), names = "value")

        self.viewer_style_widgets = {
            "morph_percent" : self.morph_slider,
            "overlay"       : self.overlay_dropdown,
            "overlay_alpha" : self.overlay_slider,
            "point_size"    : self.point_size_slider,
            "line_width"    : self.line_width_slider,
            "line_interp"   : self.line_interp_slider,
        }
        for (key, value) in self.viewer_style_widgets.items():
            value.observe(partial(self.on_style_change, key), names = "value")
        
        # We need to make sure that we update things when the style dropdown
        # changes also.
        self.style_dropdown.observe(self.refresh_style, names = "index")
        self.refresh_style()

    # Property Methods ---------------------------------------------------------

    @property
    def annotation(self):
        """Compute the currently selected annotation for styling."""
        dd = self.style_dropdown
        return dd.value if dd.index > 0 else None
    
    
    @property
    def preferences(self):
        """Compute the current style preferences based on the current style controls."""
        return { key: widget.value for (key, widget) in self.style_widgets.items() }
    
    # Annotation Style Widgets -------------------------------------------------

    def _init_style_dropdown(self):
        """Initializes the style dropdown menu for selecting which annotation to style."""
        # The style dropdown menu will have an "Active Annotation" option 
        # followed by an option for each annotation in the configuration.
        options  = [ "Active Annotation" ]
        options += list(self.state.config.annotations.keys())

        # Define the style dropdown menu.
        return ipw.Dropdown(
            options     = options, 
            value       = options[0],
            description = "Annotation:",
            layout      = { **self._WIDGET_LAYOUT, "margin": "3% 3% 3% 3%" }
        )
    

    def _init_visible_checkbox(self):
        """Initialize the annotation visibility checkbox widget."""
        return ipw.Checkbox(
            description = "Visible",
            value       = True,
            layout      = self._WIDGET_LAYOUT
        )


    def _init_color_picker(self):
        """Initialize the annotation color picker widget."""
        return ipw.ColorPicker(
            concise     = False,
            description = "Color:",
            value       = "blue",
            layout      = self._WIDGET_LAYOUT
        )


    def _init_markersize_slider(self):
        """Initialize the annotation point size slider widget."""
        return ipw.IntSlider(
            **{ **self._ANNOT_SLIDER_KWARGS, "max": 12 },
            description = "Point Size:",
        )
    

    def _init_linewidth_slider(self):
        """Initialize the annotation line width slider widget."""
        return ipw.IntSlider(
            **self._ANNOT_SLIDER_KWARGS,
            description = "Line Width:",
        )
    

    def _init_linestyle_dropdown(self):
        """Initialize the annotation line style dropdown widget."""
        return ipw.Dropdown(
            options     = [ "solid", "dashed", "dot-dashed", "dotted" ],
            description = "Line Style:",
            layout      = self._WIDGET_LAYOUT
        )

    # Cortex Viewer Style Widgets ----------------------------------------------

    def _init_morph_slider(self):
        """Initialize the cortex morph slider widget."""
        return ipw.IntSlider(
            **self._SLIDER_KWARGS,
            value       = 0,
            min         = 0,
            max         = 100,
            step        = 1,
            description = "Morph %:",
        )


    def _init_overlay_dropdown(self):
        """Initialize the cortex overlay dropdown widget."""
        # Get the overlay names from the config. 
        overlay_names = list(self.state.config.cortex["overlays"].keys())
        overlay_names.remove("curvature") # remove curvature, it is default
        overlay_names = sorted(overlay_names) # sort the rest alphabetically

        # Format the overlay names for the dropdown menu options.
        dd_options = [ ("None", "curvature") ] # initialize
        for overlay in overlay_names: 
            dd_options.append(( overlay.replace("_", " ").title(), overlay ))

        # Return the dropdown widget.
        return ipw.Dropdown(
            options    = dd_options,
            value       = "curvature",    
            description = "Overlay:",
        )


    def _init_overlay_alpha_slider(self):
        """Initialize the cortex overlay alpha slider widget."""
        return ipw.FloatSlider(
            **self._SLIDER_KWARGS,
            value       = 1.0,
            min         = 0.0,
            max         = 1.0,
            step        = 0.1,
            description = "Alpha:"
        )
    

    def _init_point_size_slider(self):
        """Initialize the cortex point size slider widget."""
        return ipw.FloatSlider(
            **self._SLIDER_KWARGS,
            value       = 0.5, 
            min         = 0.5,
            max         = 5,
            step        = 0.1,
            description = "Point Size:",
        )
    

    def _init_line_width_slider(self):
        """Initialize the cortex line width slider widget."""
        return ipw.FloatSlider(
            **self._SLIDER_KWARGS,
            value       = 0.2, 
            min         = 0.10,
            max         = 0.50,
            step        = 0.05,
            description = "Line Width:",
        )


    def _init_line_interp_slider(self):
        """Initialize the cortex line interpolation slider widget."""
        return ipw.IntSlider(
            **self._SLIDER_KWARGS,
            value       = 10,
            min         = 5,
            max         = 20,
            step        = 1,
            description = "Line Interp.:",
        )
    
    # Layout Toggle Widget -----------------------------------------------------

    def _init_layout_toggle(self):
        """Initialize the layout toggle button widget."""
        return ipw.ToggleButton(
            value       = True,
            description = "Horizontal Layout",
            tooltip     = "Toggle between horizontal and vertical layout of the control and figure panels.",
            layout      = self._WIDGET_LAYOUT
        )

    # Handler Methods ----------------------------------------------------------
    
    def on_style_change(self, key, change):
        """Handles a change in one of the style controls and alerts our observers."""
        # Alert our observers.
        for fn in self.style_observers:
            fn(self.annotation, key, change)


    def refresh_style(self, change = None):
        """Refreshes the style controls based on the currently selected annotation."""
        index = self.style_dropdown.index if change is None else change.new
        annot = self.style_dropdown.options[index] if index > 0 else None
        preferences = self.state.style(annot)
        for (key, widget) in self.style_widgets.items():
            widget.value = preferences[key]


    def observe_style(self, fn):
        """Registers the given function to be called when the a style changes.

        Style elements refer to the settings managed by the `StylePanel` of the
        `ControlPanel` object. A style element is considered to have changed
        when any of these controls are changed except for the style annotation
        selection dropdown, which controls which of the annotations the other
        style controls affect.

        When a style element changes, the given function is called with three
        arguments: `fn(annotation, element, change)` where `annotation` is the
        name of the annotation that is currently selected (i.e., the annotation
        that is changing), `element` is the name of the element that is
        changing, and `change` is the typical `ipywidget` change object used
        with the `observe` pattern. If the annotation representing the currently
        selected contour is edited, then the `annotation` value will be `None`.

        The possible values for `element` are as follows:
         * `"visible"`: the visibility has changed.
         * `"color"`: the draw color has changed.
         * `"linewidth"`: the line width has changed.
         * `"linestyle"`: the line style has changed.
         * `"markersize"`: the marker size has changed.
        """
        self.style_observers.append(fn)