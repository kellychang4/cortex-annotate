# -*- coding: utf-8 -*-
################################################################################
# annotate/_core.py
# 
# DOCSTRING

# Imports ----------------------------------------------------------------------

import ipywidgets as ipw

from .control import ControlPanel
from .figure  import FigurePanel
    
# The Annotation Tool ----------------------------------------------------------

class AnnotationTool(ipw.HBox):
    """The core annotation tool for the `cortex-annotate` project.

    The `AnnotationTool` type handles the annotation of the cortical surface
    images for the `cortex-annotate` project.
    """

    def __init__(
            self,
            config_path  = "/config/config.yaml",
            cache_path   = "/cache",
            save_path    = "/save",
            git_path     = "/git",
            username     = None,
            control_panel_background_color = "#f0f0f0",
            button_color = "#e0e0e0",
        ):        
        
        # 1. Foundation
        paths  = PathManager(cache_path, save_path, git_path, username)
        config = Config(config_path)

        # 2. State + Persistence
        state = AnnotationState(config, paths)
        prefs = PrefsManager(config, paths.get_preferences_path())

        # 3. Figure model
        figure_state = FigurePanelState(config.annotations)

        # 4. Figure UI
        canvas_panel = CanvasPanel(figure_state, size=prefs.get_display("figure_size"))
        viewer_panel = (
            CortexViewerPanel(figure_state)
            if len(config.viewer) > 0
            else None
        )
        figure_panel = FigurePanel(canvas_panel, viewer_panel)

        # 5. Cache (needs loading_context from figure_panel)
        cache = FigureCache(config, paths, prefs, figure_panel.loading_context)

        # 6. Control UI
        control_panel = ControlPanel(state, prefs, config, has_viewer=(viewer_panel is not None))

        # Go ahead and initialize the HBox component.
        super().__init__(
            children = [ self.control_panel, self.figure_panel ],
            layout   = { "border": "1px solid black", }
        )

        # 7. Wire events
        control_panel.observe_selection(self.on_selection_change)
        control_panel.observe_style(self.on_style_change)
        control_panel.observe_figure_size(self.on_figure_size_change)
        control_panel.observe_layout(self.on_layout_change)
        control_panel.observe_viewer_style(self.on_viewer_style_change)
        control_panel.observe_save(self.on_save)
        control_panel.observe_clear(self.on_clear)


    # Tool Locking Methods -----------------------------------------------------

    def _lock_tool(self):   
        """Locks the annotation tool, preventing user interaction with the figure."""
        self.state.locked = True
        self.control_panel.figure_size_slider.disabled = True

        style_panel = self.control_panel.style_panel
        style_panel.style_dropdown.disabled = True
        for widget in style_panel.style_widgets.values():
            widget.disabled = True


    def _unlock_tool(self):
        """Unlocks the annotation tool, allowing user interaction with the figure."""
        self.state.locked = False
        self.control_panel.figure_size_slider.disabled = False

        style_panel = self.control_panel.style_panel
        style_panel.style_dropdown.disabled = False
        for widget in style_panel.style_widgets.values():
            widget.disabled = False
            
    # Figure Refresh Methods ---------------------------------------------------

    def refresh_figure(self):
        # Get the target and annotation.
        target_id     = self.control_panel.target
        annotation    = self.control_panel.annotation
        target_annots = self.state.annotations[target_id]

        # Check that the selected annotation has valid fixed annotations. 
        error = None
        fixed_points = self.annot_cfg.fixed_points[annotation]
        for i, fp in enumerate(fixed_points): # for the name of the fixed point
            # Determine if the fixed point is a fixed head or tail. 
            fp_type = ( "fixed_head" if fp in 
                self.annot_cfg.fixed_heads[annotation] else "fixed_tail" )
            
            # If there is no data for this fixed point or if the fixed point is
            # the only data for the annotation, then we have an error.
            if target_annots.is_lazy(fp) or target_annots[fp].shape[0] == 0:
                error = f"Annotation '{annotation}' requires fixed point '{fp}' " \
                        f"which is not yet available for target: {target_id}."
                break

            # If there is data for this fixed point, must make sure the fixed 
            # point is valid based on the annotation type. For contours and 
            # boundary, must have data besides their own fixed points (if there).
            atype = self.annot_cfg.type[fp] 
            if atype != "point": # atype in ( "contour", "boundary" )
                n_deps = len(self.annot_cfg.fixed_points[fp])
                if target_annots[fp].shape[0] <= n_deps:
                    error = f"Annotation '{annotation}' requires fixed point '{fp}' " \
                            f"which is not yet available for target: {target_id}."
                    break

            # If there is data for this fixed point, we need to make sure that 
            # the figure panel can calculate the fixed point based on the current data.
            try:
                self.figure_panel.figure_state.calc_fixed_point(
                    annotation, target_annots, fp_type)
            except Exception as e:
                error = f"Annotation '{annotation}' requires fixed point '{fp}' " \
                        f"which cannot be calculated for target: {target_id} " \
                        f"with the current data: {e}"
                break
    
        # If there was an error, we need to put an appropriate message up. 
        # Otherwise, we can clear any messages and just show the figure.
        if error is not None:
            # Lock the annotation tool, so user cannot interact with the figure.
            self._lock_tool()

            # Write the error message. 
            self.figure_panel.write_message(error)
        else:
            # Unlock the annotation tool, so user can interact with the figure.
            self._unlock_tool()
            
            # Clear any messages that might be up from before.
            self.figure_panel.clear_message()

            # Update the figure panel state variables.
            self.figure_panel.figure_state.update(
                target_id, annotation, target_annots)
            
            # Redraw the canvas and viewer.
            self.figure_panel.redraw(
                clear = True, base = True, active = True, background = True)
            

    # Event Handler Methods ----------------------------------------------------

    def on_selection_change(self, key, change):
        """This method runs when the control panel's selection changes."""
        # First, things first: save the annotations.
        self.state.save_annotations()

        # Update the control panel legend. 
        self.control_panel.legend_panel.update(
            target_id  = self.control_panel.target,
            annotation = self.control_panel.annotation, 
        )

        # The selection has changed; we need to redraw the image and update the
        # annotations.
        self.refresh_figure()


    # def on_figure_size_change(self, change):
    #     """This method runs when the control panel's figure size slider changes."""
    #     # Only respond to changes in the value of the style elements, 
    #     if change.name != "value": return

    #     # Update the state.
    #     self.state.figure_size(change.new)

    #     # Resize the figure panel. 
    #     self.figure_panel.canvas_panel.resize_canvas(change.new)


    # def on_style_change(self, annotation, key, change):
    #     """This method runs when the control panel's style elements change."""
    #     # Only respond to changes in the value of the style elements, 
    #     if change.name != "value": return
        
    #     # Update the state.
    #     self.state.style(annotation, { key: change.new })
        
    #     # Then redraw the annotation.
    #     self.figure_panel.redraw(base = False)


    # def on_clear(self, button):
    #     """This method runs when the control panel's clear all button is clicked."""
    #     # The clear all button has a confirmation process. When the user first 
    #     # clicks it, it changes to a "Confirm Clear" button. If they click it
    #     # again, then the annotations are cleared. The button then resets to the
    #     # original "Clear All" state.
    #     if button.description == "Clear All":
    #         # Update the button to the confirmation state.
    #         button.description  = "Confirm Clear"
    #         button.button_style = "danger"
            
    #     elif button.description == "Confirm Clear":
    #         # Update the button back to the original state.
    #         button.description  = "Clear All"
    #         button.button_style = "warning"

    #         # Clear the annotations for the current target.
    #         target_id = self.control_panel.target
    #         for annotation in self.state.annotations[target_id].keys():
    #             self.state.annotations[target_id][annotation] = (
    #                 self.figure_panel.canvas_panel.empty_point_matrix())

    #         # Refresh the figure to show the cleared annotations.
    #         self.refresh_figure()
    #     else:
    #         # If the button is in some unexpected state, we just reset it to the
    #         # original state.
    #         button.description  = "Clear All"
    #         button.button_style = "warning"

        
    # def on_save(self):
    #     """This method runs when the control panel's save button is clicked."""
    #     self.state.save_annotations()
    #     self.state.save_preferences()

    
    def on_layout_change(self, change):
        """This method runs when the control panel's layout toggle button is toggled."""
        # If the button is toggled on, we want the horizontal layout. 
        if change.new:
            self.control_panel.layout_toggle.description = "Horizontal Layout"
            self.figure_panel.layout = FigurePanel._HORIZONTAL_LAYOUT

        # If the button is toggled off, we want the vertical layout.
        else:
            self.control_panel.layout_toggle.description = "Vertical Layout"
            self.figure_panel.layout = FigurePanel._VERTICAL_LAYOUT