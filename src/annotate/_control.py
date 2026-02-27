# -*- coding: utf-8 -*-
################################################################################
# annotate/_control.py

"""Core implementation code for the annotation tool's control panel.

This file contains code for managing the panel's widget and window state. The
design intention is that the `AnnotationTool` (in `_core.py`) creates and
observes changes from the control panel (such as changes in the selection or
changes in the style parameters) and passes these on to the `FigurePanel` object
in `_figure.py` as appropriate.
"""


# Imports ----------------------------------------------------------------------

import os.path as op
import ipywidgets as ipw
from functools import partial

from ._util import make_section_title, make_hline, darken_color

# The Selection Subpanel Widget ------------------------------------------------

class SelectionPanel(ipw.VBox):
    """The subpanel of the control panel for target selection."""
    
    __slots__ = (
        "state", "target_dropdowns", "annotations_dropdown", 
        "target_observers", "annotation_observers"
    )
    
    def __init__(self, state):
        # Store the state
        self.state = state

        # We have to manage an "updating" state to avoid firing observers while
        # in the middle of updating dependent dropdowns.
        self._updating = False

        # Initialize the dropdowns.
        self.target_dropdowns = {}
        
        # Create the dropdown widgets as children (excluding annotation).
        children = [ 
            make_section_title("Selection"),
        ]
        dd_layout = { "width": "94%", "margin": "1% 3% 1% 3%" }
        for key in state.config.targets.concrete_keys:
            dropdown_values = state.config.targets.items[key]

            # If dynamic target (dcit), dropdown values depend on the parent 
            # key's dropdown selection.
            if isinstance(dropdown_values, dict):        
                # Look up the parent key that this target depends on.
                parent_key = dropdown_values["depends_on"]

                # Look up the first parent selection and return options.
                parent0 = state.config.targets.items[parent_key][0]
                
                # Look up the dropdown values for this parent selection.
                dropdown_values = dropdown_values[parent0]

            dropdown_widget = ipw.Dropdown(
                options     = dropdown_values, 
                value       = dropdown_values[0], 
                layout      = dd_layout, 
                description = (key + ":")
            )
            children.append(dropdown_widget)
            self.target_dropdowns[key] = dropdown_widget
            
        # We also need the annotation dropdown.
        self.annotations_dropdown = ipw.Dropdown(
            options     = [], 
            layout      = dd_layout, 
            description = "Annotation:"
        )
        children.append(self.annotations_dropdown)

        super().__init__(children)
        
        # Because we want to control the order of a few things, we actually
        # listen to our selection items ourselves, then update them and pass
        # them along to our listeners. This is important so that, for example,
        # the Figure panel's listener does not get updated before the annotation
        # selection dropbox is changed when the user changes the target
        # selection.

        # FIRST: Wire up dependent dropdown updates.
        # These must fire before on_target_change so that downstream
        # dropdowns have valid options when the target is read.
        for key in state.config.targets.concrete_keys:
            target_item = state.config.targets.items[key]
            if isinstance(target_item, dict):
                parent_key = target_item["depends_on"]
                self.target_dropdowns[parent_key].observe(
                    partial(self.on_parent_change, key), names = "value"
                )

        # SECOND: Wire up the (non-dependent) target change observers.
        # By the time these fire, dependent dropdowns are already updated.
        for key in state.config.targets.concrete_keys:
            self.target_dropdowns[key].observe(
                partial(self.on_target_change, key), names = "value"
        )   

        # THIRD: Wire up the annotation change observer.
        # By the time this fires, the target observers have already fired.
        self.annotations_dropdown.observe(
            self.on_annotation_change, names = "value")
        
        # Initialize the observer lists.
        self.target_observers     = []
        self.annotation_observers = []
        
        # Initialize the annotations menu.
        self.refresh_annotations()


    @property
    def target(self):
        """Compute the current target selection."""
        return tuple( dd.value for dd in self.target_dropdowns.values() )
    

    @property
    def annotation(self):
        """Compute the current annotation selection."""
        return self.annotations_dropdown.value
    

    @property
    def selection(self):
        """Compute the current selection (target + annotation)."""
        return self.target + (self.annotation, )


    def refresh_annotations(self):
        """Refreshes the annotations dropdown menu based on the current target selection."""
        # Get the new target selection entirely.
        target_id = self.target
    
        # Look up the target for this selection.
        target = self.state.config.targets[target_id]
    
        # Recalculate the annotations for this target and update the menu.
        annotation_options = [ 
            annotation for ( annotation, annotation_data ) 
            in self.state.config.annotations.items()
            if annotation_data.filter is None or annotation_data.filter(target) 
        ]
        self.annotations_dropdown.options = annotation_options
        self.annotations_dropdown.value   = annotation_options[0]


    def on_parent_change(self, key, change):
        """Handles the change in a parent dropdown for a dynamic target."""
        # Set "updating" state to avoid firing dependent dropdown observers.
        self._updating = True
        try: 
            # Get the new parent selection and dropdown values.
            dependent_items = self.state.config.targets.items[key]
            dropdown_values = dependent_items[change.new]

            # Update the dependent dropdown's options and value.
            dependent_dropdown = self.target_dropdowns[key]
            dependent_dropdown.options = dropdown_values
            dependent_dropdown.value   = dropdown_values[0]
        finally: 
            # Undo "updating" state.
            self._updating = False


    def on_target_change(self, key, change):
        """Alert our observers that the target selection has changed."""
        # Prevent firing observers if we are updating dependent dropdowns.
        if self._updating: return

        # Refresh the annotations menu.
        self.refresh_annotations()

        # Alert our other observers, now that our updates are finished.
        for fn in self.target_observers:
            fn(key, change)


    def on_annotation_change(self, change):
        """Alert our observers that the annotation selection has changed."""
        # Alert our observers.
        for fn in self.annotation_observers:
            fn(change)


    def observe_target(self, fn):
        """Registers the given function to be called when the target changes.

        The selection target refers to the selection of all the concrete keys in
        the `config.yaml` file's `targets` section. In other words, the
        selection target changes when any of the selection dropdowns are changed
        except for the annotation dropdown.

        When the selection target changes, the given function is called with two
        arguments: `fn(concrete_key, change)` where `concrete_key` is the
        (string) name of one of the concrete keys and `change` is the change
        object typically used in the `ipywidget` `observe` pattern.
        """
        self.target_observers.append(fn)


    def observe_annotation(self, fn):
        """Registers the argument to be called when the annotation changes.

        The annotation selection is the currently selected annotation in the
        annotations dropdown menu of the `SelectionPanel` component of the
        `ControlPanel`.

        When the annotation selection changes, the given function is called with
        the argument `change` where `change` is the `change` object typically
        used in the `ipywidget` `observe` pattern.
        """
        self.annotation_observers.append(fn)


    def observe_selection(self, fn):
        """Registers the given function to be called when the selection changes.

        The selection refers to the combination of target and annotation
        selection; see the `observe_target` and `observe_annotation` methods for
        more information.

        When the selection changes, the given function is called with two
        arguments: `fn(concrete_key, change)` where `concrete_key` is the
        (string) concrete key that has changed and `change` is the change object
        typically used in the `ipywidget` `observe` pattern. If the annotation
        has changed, then the `key` will be `None`.
        """
        self.observe_target(fn)
        self.observe_annotation(partial(fn, None))

# The Legend Subpanel Widget ---------------------------------------------------

class LegendPanel(ipw.VBox):
    """The subpanel of the control panel containing the legend controls."""

    slots = ( "state", "hemisphere_index", "image_dir", "image_widget" )

    def __init__(self, state):
        # Store the state
        self.state = state

        # Store the hemisphere index for later use in legend updates.
        self.hemisphere_index = state.config.targets.concrete_keys.index("Hemisphere")

        # Set up the path to the annotation legend images.
        self.image_dir = op.join(op.dirname(__file__), "annotation-legends")

        # Create the image widget
        self.image_widget = ipw.Image(
            format = "png",
            layout = { "margin": "0% 3% 0% 3%", "width": "94%" }
        )

        # Initialize the VBox
        super().__init__(
            children = [ 
                make_section_title("Annotation Legend"), 
                self.image_widget 
            ],
            layout   = { "margin": "0% 0% 3% 0%" }
        )

        # Update the legend with the initial image
        target_id  = list(state.config.targets.keys())[0]
        annotation = list(state.config.annotations.keys())[0]
        self.update_legend(target_id, annotation)


    def _read_image(self, image_path):
        """Reads the image data from the given path."""
        # Read the image data and return it.
        with open(image_path, "rb") as f:
            image_data = f.read()
        return image_data
    

    def update_legend(self, target_id, annotation):
        """Updates the legend image to the given legend name."""
        hemisphere = target_id[self.hemisphere_index]
        image_path = op.join(self.image_dir, hemisphere, f"{annotation}.png")
        if not op.isfile(image_path): # if the image does not exist, use empty
            image_path = op.join(self.image_dir, "empty.png")
        self.image_widget.value = self._read_image(image_path)

# The Style Subpanel Widget ----------------------------------------------------

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
        self.inflation_slider   = self._init_inflation_slider()
        self.overlay_dropdown   = self._init_overlay_dropdown()
        self.overlay_slider     = self._init_overlay_alpha_slider()
        self.point_size_slider  = self._init_point_size_slider()
        self.line_width_slider  = self._init_line_width_slider()
        self.line_interp_slider = self._init_line_interp_slider()

        # Assemble the style panel children.
        children = [
            make_section_title("Style Options"),
            self.style_dropdown,
            make_hline(),
            make_section_title("Annotation Options"),
            self.visible_checkbox,
            self.color_picker,
            self.markersize_slider,
            self.linewidth_slider,
            self.linestyle_dropdown,
            make_hline(),
            make_section_title("Cortex Options"),
            self.inflation_slider,
            self.overlay_dropdown,
            self.overlay_slider, 
            self.point_size_slider, 
            self.line_width_slider,
            self.line_interp_slider,
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

        self.cortex_style_widgets = {
            "inflation"     : self.inflation_slider,
            "overlay"       : self.overlay_dropdown,
            "overlay_alpha" : self.overlay_slider,
            "point_size"    : self.point_size_slider,
            "line_width"    : self.line_width_slider,
            "line_interp"   : self.line_interp_slider,
        }
        # TODO: will add observers later!!!
        
        # We need to make sure that we update things when the style dropdown
        # changes also.
        self.style_dropdown.observe(self.refresh_style, names = "index")
        self.refresh_style()

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

    # Cortex Style Widgets -----------------------------------------------------

    def _init_inflation_slider(self):
        """Initialize the cortex inflation slider widget."""
        return ipw.IntSlider(
            **self._SLIDER_KWARGS,
            value       = 100,
            min         = 0,
            max         = 100,
            step        = 1,
            description = "Inflation %:",
        )


    def _init_overlay_dropdown(self):
        """Initialize the cortex overlay dropdown widget."""
        return ipw.Dropdown(
            options     = [
                ( "None", "curvature" ), 
                ( "Polar Angle", "angle" ), 
                ( "Eccentricity", "eccen" ), 
                ( "Variance Explained", "vexpl" )
            ],
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

    # Property Methods and Observer Registration -------------------------------

    @property
    def annotation(self):
        """Compute the currently selected annotation for styling."""
        dd = self.style_dropdown
        return dd.value if dd.index > 0 else None
    
    
    @property
    def preferences(self):
        """Compute the current style preferences based on the current style controls."""
        return { key: widget.value for (key, widget) in self.style_widgets.items() }
    
    
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

# The Control Panel Widget -----------------------------------------------------

class ControlPanel(ipw.VBox):
    """The panel that contains the controls for the Annotation Tool."""

    def __init__(
            self, 
            state,
            background_color = "#f0f0f0", 
            button_color     = "#e0e0e0"
        ):

        # Create the selection panel.
        self.selection_panel = SelectionPanel(state)

        # Create the figure size slider.
        self.figure_size_slider = ipw.IntSlider(
            value             = state.preferences["figure_size"],
            min               = 250,   
            max               = 1280, 
            step              = 1,
            description       = "Image Size: ",
            readout           = False,
            continuous_update = False,
            layout            = { "width": "90%", "padding": "0px" }
        )

        # Create the legend panel. 
        self.legend_panel = LegendPanel(state)

        # Create the style panel.
        self.style_panel = StylePanel(state)

        # Create the clear all button.
        self.clear_button = ipw.Button(
            description  = "Clear All",
            tooltip      = "Clear all annotations from the figure.",
            button_style = "warning"
        )

        # Create the save button.
        self.save_button = ipw.Button(
            description  = "Save",
            tooltip      = "Save all annotations and preferences."
        )
        self.save_button.style.button_color = button_color
        
        # Create the wrapper for the buttons.
        self.button_box = ipw.HBox(
            children = [  self.clear_button, self.save_button ], 
            layout   = { "margin" : "3% 3% 3% 3%", "width" : "94%" }
        )            

        # First: Selection and Annotation related panels.
        children = [
            self.selection_panel,
            make_hline(),
            self.figure_size_slider,
            make_hline(),
            self.legend_panel,
            make_hline(),
            self.button_box,
            make_hline(),
            self._make_info_message()
        ]
        selection_vbox = ipw.VBox(children)

        # Wrap the selection and style panels in tab widget. 
        control_tabs = ipw.Tab(
            children = [ selection_vbox, self.style_panel ], 
            titles   = [ "Selection", "Style" ],
            selected_index = 0,       
        )
        control_tabs.add_class("annotate-control-tabs")

        # Finally, put the whole thing in an accordion so that it can be collapsed.
        accordion = ipw.Accordion(
            children = [ control_tabs ],
            selected_index = 0,
        )

        # Finally, call the VBox initializer. 
        super().__init__(
            children = [ self._make_html_header(background_color), accordion ],  
            layout   = { "border": "0px", }
        )

    # Classmethod Methods ------------------------------------------------------

    @classmethod
    def _make_html_header(
        cls, background_color = "#f0f0f0", inactive_amount = 0.10
    ):
        inactive_color = darken_color(background_color, inactive_amount)
        return ipw.HTML(f"""
            <style>
                .jupyter-widget-Collapse-open {{
                    background-color: white;
                    width: 300px;
                }}
                .jupyter-widget-Collapse-header {{
                    background-color: white;
                    border: 0px;
                    padding: 0px;
                }}
                .jupyter-widget-Collapse-contents {{
                    background-color: white;
                    border: 0px;
                    padding: 0px;
                }}
                .annotate-control-tabs.jupyter-widget-tab.widget-tab {{
                        max-width: 300px;
                        min-height: 850px;
                }}
                .annotate-control-tabs > .jupyter-widget-TabPanel-tabContents.widget-tab-contents
                {{
                    background-color: {background_color};
                    margin: 0px;
                    padding: 5px;
                }}
                .annotate-control-tabs.jupyter-widgets.jupyter-widget-tab > .lm-TabBar .lm-TabBar-tab {{
                    background-color: rgb{inactive_color};
                    flex: 1 1 auto;
                }}
                .annotate-control-tabs.jupyter-widgets.jupyter-widget-tab > .lm-TabBar .lm-TabBar-tab.lm-mod-current {{
                    background-color: {background_color};
                }}
                .annotate-control-panel-hline {{
                    border-color: lightgray;
                    border-style: solid;
                    border-width: 1px;
                    height: 0px;
                    width: 94%;
                    margin: 3%;
                }}
            </style>
        """)

    
    @classmethod
    def _make_info_message(cls):
        return ipw.VBox([
            ipw.HTML("""
                <div style="line-height:1.2; margin: 2%;">
                <center><b>CLICK</b> to add a point to the circled end of the
                current annotation.</center></div>
                """),
            ipw.HTML("""
                <div style="line-height:1.2; margin: 2%;">
                <center><b>BACKSPACE</b> to delete the circled point.
                </center></div>
                """),
            ipw.HTML("""
                <div style="line-height:1.2; margin: 2%;">
                <center><b>TAB</b> to toggle the circled end.</center></div>
                """)
            ], layout = { "margin": "3%", "width": "88%" })

    # Property Methods ---------------------------------------------------------

    @property
    def target(self):
        """Compute the current target selection."""
        return self.selection_panel.target
    

    @property
    def annotation(self):
        """Compute the current annotation selection."""
        return self.selection_panel.annotation
    

    @property
    def selection(self):
        """Compute the current selection (targets + annotation)."""
        return self.selection_panel.selection
    
    # Observe Methods ----------------------------------------------------------

    def observe_target(self, fn):
        """Registers the given function to be called when the taget changes.

        The selection target refers to the selection of all the concrete keys in
        the `config.yaml` file's `targets` section. In other words, the
        selection target changes when any of the selection dropdowns are changed
        except for the annotation dropdown.
a
        When the selection target changes, the given function is called with two
        arguments: `fn(concrete_key, change)` where `concrete_key` is the
        (string) name of one of the concrete keys and `change` is the change
        object typically used in the `ipywidget` `observe` pattern.
        """
        self.selection_panel.observe_target(fn)


    def observe_selection(self, fn):
        """Registers the given function to be called when the selection changes.

        The selection refers to the combination of target and annotation
        selection; see the `observe_target` and `observe_annotation` methods for
        more information.

        When the selection changes, the given function is called with two
        arguments: `fn(concrete_key, change)` where `concrete_key` is the
        (string) concrete key that has changed and `change` is the change object
        typically used in the `ipywidget` `observe` pattern. If the annotation
        has changed, then the `key` will be `None`.
        """
        self.selection_panel.observe_selection(fn)


    def observe_annotation(self, fn):
        """Registers the argument to be called when the annotation changes.

        The annotation selection is the currently selected annotation in the
        annotations dropdown menu of the `SelectionPanel` component of the
        `ControlPanel`.

        When the annotation selection changes, the given function is called with
        the argument `change` where `change` is the `change` object typically
        used in the `ipywidget` `observe` pattern.
        """
        self.selection_panel.observe_annotation(fn)


    def observe_figure_size(self, fn):
        """Registers the argument to be called when the figure size changes.

        `control_panel.observe_figure_size(fn)` is equivalent to
        `control_panel.figure_size_slider.observe(fn, names="value")`.
        """
        self.figure_size_slider.observe(fn, names = "value")


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
         * `'visible'`: the visibility has changed.
         * `'color'`: the draw color has changed.
         * `'linewidth'`: the line width has changed.
         * `'linestyle'`: the line style has changed.
         * `'markersize'`: the marker size has changed.
        """
        self.style_panel.observe_style(fn)


    def observe_save(self, fn):
        """Registers the argument to be called when the save button is clicked.
        
        The function is called with a single argument, which is the save button
        instance.
        """
        self.save_button.on_click(fn)


    def observe_clear(self, fn):
        """Registers the argument to be called when the clear all button is clicked.
        
        The function is called with a single argument, which is the clear all
        button instance.
        """
        self.clear_button.on_click(fn)