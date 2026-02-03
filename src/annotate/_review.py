import ipycanvas as ipc
from ._config import ConfigError, _compile_fn


class ReviewConfig:
    """An object that stores the configuration of the review panel.

    The `ReviewConfig` type stores information from the `review` section of the
    `config.yaml` file for the `cortex-annotate` project. This section stores
    only a function that generates the figure to plot in the `Review` tab of the
    display panel. The review function requires the arguments `target`,
    `annotations`, `figure`, and `axes`, and it must draw the desired graphics
    on the given matplotlib axes. The `annotations` argument is a dictionary
    whose keys are the annotation names and whose values are the drawn
    annotation. Annotations may be missing if the user opens the review tab
    before completing the annotations. If an error is raised from this function,
    then the error message is printed to the display.
    """

    __slots__ = ( "code", "function", "figure_size", "dpi" )
    
    @staticmethod
    def _compile(code, initcfg):
        return _compile_fn(
            "target, annotations, figure, axes, save_hooks",
            f"{code}\n", initcfg
        )
    

    def __init__(self, yaml, init):
        if yaml is None:
            self.code = None
            self.function = None
            self.figure_size = None
            self.dpi = None
            return
        elif isinstance(yaml, str):
            yaml = {'function': yaml}
        elif not isinstance(yaml, dict):
            raise ConfigError(
                "review",
                "review section must contain a Python code string or a dict",
                yaml)
        self.code = yaml.get("function")
        if self.code is None:
            raise ConfigError(
                "review",
                "review section must contain the key function",
                yaml)
        self.figure_size = yaml.get("figure_size", (3,3))
        self.dpi = yaml.get("dpi", 256)
        self.function = ReviewConfig._compile(self.code, init)


# Stuff from the figure panel
def review_start(self, msg, wrap = True):
    from ._util import wrap as wordwrap
    self.review_msg = msg
    self.redraw_canvas(redraw_review = True)


def review_end(self):
    self.review_msg = None
    self.redraw_canvas(redraw_review = True)


def redraw_review(self, wrap = True, fontsize = 32):
    """Clears the draw and image canvases and draws the review canvas."""
    if self.review_msg is None:
        # If there's nothing to review, we do nothing.
        return
    self.image_canvas.clear()
    self.draw_canvas.clear()
    self.fg_canvas.clear()
    dc = self.image_canvas
    if isinstance(self.review_msg, str):
        with ipc.hold_canvas():
            dc.fill_style = "white"
            dc.fill_rect(0, 0, dc.width, dc.height)
            self.write_message(
                self.review_msg,
                wrap = wrap,
                fontsize = fontsize,
                canvas = dc)
    else:
        dc.draw_image(self.review_msg, 0, 0, dc.width, dc.height)


# from the control panel
 self.edit_button = ipw.Button(
            description  = "Edit",
            button_style = "",
            tooltip      = "Continue editing annotation."
        )
        

self.review_button = ipw.Button(
    description  = "Review",
    button_style = "",
    tooltip      = "Review the annotations."
)

if state.config.review.function is not None:
    buttons = [self.review_button, self.save_button, self.edit_button]
    self.review_button.disabled = False
    self.save_button.disabled = True
    self.edit_button.disabled = True
    layout = { "margin" : "3% 3% 3% 3%", "width" : "94%" }

        

def observe_review(self, fn):
    """Registers the argument to be called when the save button is clicked.
    
    The function is called with a single argument, which is the review
    button instance.
    """
    self.review_button.on_click(fn)


# from core
 
    def generate_review(self, target_id, save_hooks):
        """Generates a single figure for the given target and figure name."""
        target = self.config.targets[target_id]
        annots = self.annotations[target_id]
        fn = self.config.review.function
        if fn is None:
            raise RuntimeError("no review function found")
        # Make a figure and axes for the plots.
        figure_size = self.config.review.figure_size
        dpi = self.config.review.dpi
        (fig,ax) = plt.subplots(1, 1, figsize = figure_size, dpi = dpi)
        # Run the function from the config that draws the figure.
        fn(target, annots, fig, ax, save_hooks)
        # We can go ahead and fix the save hooks now:
        for (k,fn) in save_hooks.items():
            save_hooks[k] = (target_id, fn)
        # Tidy things up for image plotting.
        ax.axis("off")
        fig.subplots_adjust(0,0,1,1,0,0)
        b = BytesIO()
        plt.savefig(b, format = "png")
        # We can close the figure now as well.
        plt.close(fig)
        return ipw.Image(value = b.getvalue(), format = "png")
    

    def run_save_hooks(self):
        """Runs any save hooks that were registered by the review."""
        hooks = self.save_hooks
        self.save_hooks = None
        if hooks is not None:
            for (filename, (tid, fn)) in hooks.items():
                filename = os.path.join(self.target_save_path(tid), filename)
                fn(filename)



    def observe_edit(self, fn):
        """Registers the argument to be called when the edit button is clicked.
        
        The function is called with a single argument, which is the edit button
        instance.
        """
        self.edit_button.on_click(fn)

    
        def on_review(self, button):
        """This method runs when the control panel's review button is clicked."""
        self.figure_panel.clear_message()
        self.state.save_preferences()
        # This will happen no matter what:
        self.control_panel.review_button.disabled = True
        self.control_panel.edit_button.disabled = False
        # Get the review function.
        rev = self.state.config.review.function
        if rev is None:
            self.control_panel.save_button.disabled = True
        else:
            with self.state.reviewing_context:
                try:
                    save_hooks = {}
                    targ = self.control_panel.target
                    msg = self.state.generate_review(targ, save_hooks)
                    self.control_panel.save_button.disabled = False
                    self.state.save_hooks = save_hooks
                except Exception as e:
                    msg = str(e)
                    self.control_panel.save_button.disabled = True
            self.figure_panel.review_start(msg)
    
        
    def on_edit(self, button):
        """This method runs when the control panel's edit button is clicked."""
        if self.figure_panel.review_msg is not None:
            self.control_panel.review_button.disabled = False
            self.control_panel.save_button.disabled   = True
            self.control_panel.edit_button.disabled   = True
            self.figure_panel.review_end()
            self.refresh_figure()
        self.state.save_hooks = None