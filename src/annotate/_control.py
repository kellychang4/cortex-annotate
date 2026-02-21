# -*- coding: utf-8 -*-
################################################################################
# annotate/_control_panel.py

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

        # Initialize the dropdowns.
        self.target_dropdowns = {}
        
        # Create the dropdown widgets as children (excluding annotation).
        children = [ 
            ipw.HTML("<b style=\"margin: 0% 3% 0% 3%;\">Selection:</b>") 
        ]
        dd_layout = { "width": "94%", "margin": "1% 3% 1% 3%" }
        for key in state.config.targets.concrete_keys:
            dropdown_values = state.config.targets.items[key]
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
        for key in state.config.targets.concrete_keys:
            self.target_dropdowns[key].observe(
                partial(self.on_target_change, key), names = "value")
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
        annotion_options = [ 
            annotation for (annotation, annotation_data) 
            in self.state.config.annotations.items()
            if annotation_data.filter is None or annotation_data.filter(target) 
        ]
        self.annotations_dropdown.options = annotion_options
        self.annotations_dropdown.value   = annotion_options[0]


    def on_target_change(self, key, change):
        """Alert our observers that the target selection has changed."""
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


# The Style Subpanel Widget ----------------------------------------------------

class StylePanel(ipw.VBox):
    """The subpanel of the control panel containing the style controls."""

    __slots__ = (
        "state", "style_dropdown", "visible_checkbox",
        "color_picker", "markersize_slider", "linewidth_slider", 
        "linestyle_dropdown", "style_observers", "style_widgets"
    )
    
    def __init__(self, state):
        # Store the state
        self.state = state

        # Declare default layout and kwargs options for the style controls.
        layout = { "width": "94%", "margin": "0% 3% 0% 3%" }   
        slider_kwargs = {
            "value": 1, "min": 1, "max": 8, "step": 1,
            "readout": True, "continuous_update": False,
            "layout": layout
        }

        # The style dropdown menu will have an "Active Annotation" option 
        # followed by an option for each annotation in the configuration.
        entries  = [ "Active Annotation" ]
        entries += list(state.config.annotations.keys())

        # Define the style dropdown menu.
        self.style_dropdown = ipw.Dropdown(
            options     = entries, 
            value       = entries[0],
            description = "Annotation:",
            layout      = { **layout, "margin": "3% 3% 3% 3%" }
        )

        # Define the visible checkbox.
        self.visible_checkbox = ipw.Checkbox(
            description = "Visible",
            value       = True,
            layout      = layout
        )

        # Define the color picker.
        self.color_picker = ipw.ColorPicker(
            concise     = False,
            description = "Color:",
            value       = "blue",
            layout      = layout
        )

        # Define the marker size slider.
        self.markersize_slider = ipw.IntSlider(
            **{ **slider_kwargs, "max": 12 },
            description = "Point Size:",
        )
        self.markersize_slider.add_class("annotate-style-widget")

        # Define the line width slider.
        self.linewidth_slider = ipw.IntSlider(
            **slider_kwargs,
            description = "Line Width:",
        )
        self.linewidth_slider.add_class("annotate-style-widget")

        # Define the line style dropdown.
        self.linestyle_dropdown = ipw.Dropdown(
            options     = [ "solid", "dashed", "dot-dashed", "dotted" ],
            description = "Line Style:",
            layout      = layout
        )

        # Create the VBox children of style options.
        children = [
            self._make_html_header(),
            ipw.HTML("<b style=\"margin: 0% 3% 0% 3%;\">Style Options:</b>"),
            self.style_dropdown,
            self._make_hline(),
            self.visible_checkbox,
            self.color_picker,
            self.markersize_slider,
            self.linewidth_slider,
            self.linestyle_dropdown
        ]
        super().__init__(children, layout = { "margin": "0% 0% 3% 0%" })
        
        # Set up our observer pattern. We track these manually so that we can
        # call the functions using a parameter order that makes sense.
        self.style_observers = []
        self.style_widgets = {
            "visible"   : self.visible_checkbox,
            "color"     : self.color_picker,
            "markersize": self.markersize_slider,
            "linewidth" : self.linewidth_slider,
            "linestyle" : self.linestyle_dropdown
        }
        for (key, value) in self.style_widgets.items():
            value.observe(partial(self.on_style_change, key), names = "value")
        
        # We need to make sure that we update things when the style dropdown
        # changes also.
        self.style_dropdown.observe(self.refresh_style, names = "index")
        self.refresh_style()


    @classmethod
    def _make_html_header(cls, hline_width = 85):
        hline_right = (100 - hline_width) // 2
        hline_left  = 100 - hline_width - hline_right
        return ipw.HTML(f"""
            <style>
                .annotate-style-hline {{
                    border-color: lightgray;
                    border-style: dotted;
                    border-width: 1px;
                    height: 0px;
                    width: {hline_width}%;
                    margin: 0% {hline_right}% 0% {hline_left}%;               
                }}
                .annotate-style-widget .widget-readout {{
                    min-width: 50px;
                }}
            </style>
        """)

    
    @classmethod
    def _make_hline(cls):
        return ipw.HTML("""<div class="annotate-style-hline"></div>""")
    

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

        # Create the VBox children
        children = [
            ipw.HTML("<b style=\"margin: 0% 3% 0% 3%;\">Annotation Legend:</b>"),
            self.image_widget
        ]

        # Initialize the VBox
        super().__init__(
            children = children, 
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

        # Create the VBox children; main control panel components and dividers.
        children = [
            self.selection_panel,
            self._make_hline(),
            self.figure_size_slider,
            self._make_hline(),
            self.legend_panel,
            self._make_hline(),
            self.style_panel,
            self._make_hline(),
            self.button_box,
            self._make_hline(),
            self._make_info_message()
        ]
        vbox = ipw.VBox(children, layout = { "width": "250px" })
        
        # Wrap the whole thing in an accordion so that it can be collapsed.
        accordion = ipw.Accordion((vbox, ), selected_index = 0)
        accordion.add_class("annotate-control-panel") #TODO

        # Finally, call the VBox initializer. 
        super().__init__(
            children = [ self._make_html_header(background_color), accordion ],  
            layout   = { "border": "0px", "height": "100%" }
        )
    
    # TODO: come back here to edit for the annotate-control-panel class nesting
    @classmethod
    def _make_html_header(cls, background_color = "#f0f0f0"):
        return ipw.HTML(f"""
            <style>
                .jupyter-widget-Collapse-header {{
                    background-color: white;
                    border-width: 0px;
                    padding: 0px;
                }}
                .jupyter-widget-Collapse-contents {{
                    background-color: {background_color};
                    padding: 2px;
                    border-width: 1px;
                    border-style: solid;
                    border-color: lightgray;
                }}
                .jupyter-widget-Collapse-open {{
                    background-color: white;
                }}
                .annotate-hline {{
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
    def _make_hline(cls):
        return ipw.HTML("""<div class="annotate-hline"></div>""")
    
    
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
    

    def observe_target(self, fn):
        """Registers the given function to be called when the taget changes.

        The selection target refers to the selection of all the concrete keys in
        the `config.yaml` file's `targets` section. In other words, the
        selection target changes when any of the selection dropdowns are changed
        except for the annotation dropdown.

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