# -*- coding: utf-8 -*-
################################################################################
# annotate/control/_control_panel.py
#
# DOCSTRING
    
# Imports ----------------------------------------------------------------------

import os.path as op
import ipywidgets as ipw
from functools import partial

from ._selection import SelectionPanel
from ._legend    import LegendPanel
from ._style     import StylePanel

# The Control Panel Widget -----------------------------------------------------

class ControlPanel(ipw.VBox):
    """Facade for all control sub-panels.

    AnnotationTool interacts with ControlPanel exclusively through the
    properties and methods listed here. It never accesses sub-panel
    widgets directly.
    """

    def __init__(
            self, 
            state,
            background_color = "#f0f0f0", 
            button_color     = "#e0e0e0"
        ):

        # Create the selection panel.
        self.selection_panel = SelectionPanel(state)

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
            titles   = [ "Selection", "Style", ],
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


    # # --- Properties (read-only) ---
    # @property
    # def target(self): ...          # → SelectionPanel.target

    # @property
    # def annotation(self): ...      # → SelectionPanel.annotation

    # # --- Facade actions ---
    # def lock(self): ...            # Disables all interactive widgets
    # def unlock(self): ...          # Re-enables all interactive widgets

    # # --- Observer registration ---
    # def observe_selection(self, fn): ...      # → SelectionPanel
    # def observe_style(self, fn): ...          # → StylePanel (annotation style)
    # def observe_figure_size(self, fn): ...    # → DisplayPanel
    # def observe_layout(self, fn): ...         # → DisplayPanel
    # def observe_viewer_style(self, fn): ...   # → DisplayPanel
    # def observe_save(self, fn): ...           # → save button on_click
    # def observe_clear(self, fn): ...          # → clear button on_click

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
    
    
    @property
    def layout_toggle(self):
        """Returns the layout toggle button widget."""
        return self.style_panel.layout_toggle
    
    # Observe Methods ----------------------------------------------------------

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
        self.selection_panel.observe_target(fn)


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


