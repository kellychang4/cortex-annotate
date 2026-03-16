# -*- coding: utf-8 -*-
################################################################################
# annotate/config/_figures.py

"""Figure-generating function configuration for cortex-annotate.

FiguresConfig is a dict subclass that parses the ``figures`` section of
config.yaml. Each entry maps a figure name to a compiled Python function
that generates a matplotlib figure for a given target.

Special keys:
    init : Code appended to the start of every figure function body.
    term : Code appended to the end of every figure function body.
    _    : Wildcard function used for any figure name not explicitly listed.
"""

# Imports ----------------------------------------------------------------------

from functools import partial

from ._error  import ConfigError

# Figures Configuration --------------------------------------------------------

class FiguresConfig(dict):
    """Compiled figure-generating functions from config.yaml.

    A dict mapping figure names to callable functions. Each function
    has the signature ``fn(target, key, fig, axes, figsize, dpi,
    meta_data)`` and is compiled from a Python code string in the
    ``figures`` section of config.yaml, wrapped with optional ``init``
    and ``term`` blocks.

    Parameters
    ----------
    figures_yaml : dict
        The ``figures`` section from config.yaml. Keys are figure names
        (or the special keys ``init``, ``term``, ``_``); values are
        Python code strings.

    figure_names : set of str
        Figure names referenced by annotation grids. Every name must
        have a matching entry or a wildcard (``_``) fallback.

    init : InitConfig
        The init environment for compiling code strings.
    """

    __slots__ = ( )    
    
    def __init__(self, figures_yaml, figure_names, init):
        # The figures section is required as a mapping (dictionary).
        if not isinstance(figures_yaml, dict):
            raise ConfigError("figures", "figures section must contain a mapping.")
        
        # Prepare the figures yaml and the figure compiling functions.
        figures_yaml, compile_fn, wildfn = self._prep_yaml(figures_yaml, init)

        # Prepare the figure dictionary.
        figures_dict = self._init_figures_dict(
            figures_yaml, figure_names, compile_fn, wildfn)
        
        # Update FiguresConfig class dictionary.
        self.update(figures_dict)


    @staticmethod
    def _compile_fn(init, initcode, termcode, code):
        """Compile a figure code string with init/term wrappers.

        Parameters
        ----------
        init : InitConfig
            The init environment.

        initcode : str or None
            Code appended to the start of the function body.

        termcode : str or None
            Code appended to the end of the function body.

        code : str
            The figure-specific code string.

        Returns
        -------
        callable
            A function with signature
            ``fn(target, key, fig, axes, figsize, dpi, meta_data)``.
        """
        return init.compile_fn(
            "target, key, fig, axes, figsize, dpi, meta_data",
            f"{initcode}\n{code}\n{termcode}"
        )
    

    def _prep_yaml(self, figures_yaml, init):
        """Extract special fields and prepare the figure compilation function.

        Pops ``init``, ``term``, and ``_`` (wildcard) from the figures YAML
        dict. Builds a ``compile_fn`` partial that wraps each figure's code
        string with the init/term blocks. Compiles the wildcard if present.

        Parameters
        ----------
        figures_yaml : dict
            Mutable copy of the figures YAML section. Special keys are
            popped in place; remaining keys are actual figure names.

        init : InitConfig
            The init environment for compiling code strings.

        Returns
        -------
        tuple of (dict, callable, callable or None)
            ``(remaining_yaml, compile_fn, wildcard_fn)`` where
            ``remaining_yaml`` is the input dict with special keys removed,
            ``compile_fn`` wraps a code string into a figure function, and
            ``wildcard_fn`` is the compiled wildcard or None.
        """
        # Check that the all fields are code strings if they are not None.
        for key, value in figures_yaml.items():
            if not isinstance(value, str):
                raise ConfigError(
                    f"figures.{key}", 
                    f"'{key}' value must be a code string."
                )

        # Prepare the special fields (init, term, and wildcard).
        special_dict = {
            k: figures_yaml.pop(k, None) 
            for k in ( "init", "term", "_" )
        }
        
        # Prepare the figure compiling code.
        compile_fn = partial(
            self._compile_fn, init, 
            special_dict["init"], special_dict["term"]
        )

        # Compile the wildcard field if not None.
        wildfn = None
        if special_dict["_"] is not None:
            wildfn = compile_fn(special_dict["_"])

        return ( figures_yaml, compile_fn, wildfn )


    @staticmethod
    def _init_figures_dict(figures_yaml, figure_names, compile_fn, wildfn):
        """Build the figure name → function mapping.

        For each name in *figure_names*, compiles the matching code
        string from *figures_yaml*, or falls back to *wildfn* if the
        name is not explicitly listed.

        Parameters
        ----------
        figures_yaml : dict
            The figures YAML dict with special keys already removed.

        figure_names : set of str
            All figure names that must be resolvable.

        compile_fn : callable
            A partial that compiles a code string into a figure
            function (already bound with init/term blocks).

        wildfn : callable or None
            The compiled wildcard function, or ``None`` if no wildcard
            was provided.

        Returns
        -------
        dict of {str: callable}
            Maps each figure name to its compiled function.
        """
        figures_dict = {} # initialize
        for key in figure_names: # for each figure name
            if key not in figures_yaml:
                if wildfn is None:
                    # If the key and wildcard are both missing, raise an error.
                    raise ConfigError(
                        f"figures.{key}", 
                        f"Missing code for figure '{key}' and "
                        f"no wildcard provided."
                    )
                else:
                    # If the key is missing but the wildcard is present, use the wildcard.
                    figures_dict[key] = wildfn
            else: 
                # Else, the key is present, compile the code.
                code = figures_yaml.get(key, None)
                if code is not None:
                    figures_dict[key] = compile_fn(code)

        # Return the figure dictionary.
        return figures_dict