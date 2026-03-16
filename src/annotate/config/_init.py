# -*- coding: utf-8 -*-
################################################################################
# annotate/config/_init.py

"""Initialization code block support for cortex-annotate configuration.

InitConfig executes a user-provided Python code block from the ``init``
section of config.yaml, then exposes the resulting environment to all other
configuration sections. This allows shared imports, helper functions, and
constants to be defined once and reused across targets, annotations, figures,
and viewer specifications.
"""

# Imports ----------------------------------------------------------------------

import os

from ._error import ConfigError

# Init Configuration -----------------------------------------------------------

class InitConfig:
    """Shared code environment for config.yaml sections.

    Executes a user-provided Python code block (the ``init`` section of
    config.yaml), then exposes the resulting namespace to all other
    configuration sections via :meth:`compile_fn`.

    Parameters
    ----------
    code : str or None
        Python source code to execute. If ``None``, an inert ``"None"``
        expression is used and the environment starts empty.

    globals_env : dict or None
        Additional global bindings to inject before execution.

    locals_env : dict or None
        Additional local bindings to inject before execution.

    Attributes
    ----------
    code : str
        The validated init code string.

    env : dict
        The execution environment populated by running ``code``.
    """

    __slots__ = ( "code", "env" )
    
    def __init__(self, code, globals_env = None, locals_env = None):
        # Initialize the code string.
        self.code = self._init_code(code)

        # Prepare the given globals and locals for the merged environment.
        self.env = self._init_env(globals_env, locals_env)

        # Execute the code block to populate the environment.
        exec(self.code, self.env, self.env)


    @ staticmethod
    def _init_code(code):
        """Validate and normalize the init code string.

        Parameters
        ----------
        code : str or None
            Raw code from config.yaml. ``None`` is replaced with the
            inert expression ``"None"``.

        Returns
        -------
        str
            The validated code string.
        """
        # The code is optional. If None, we just use an empty code block.
        if code is None: code = "None"

        # Check that the code is a string.
        if not isinstance(code, str):
            raise ConfigError("init", f"init section must be a string: {code}")
        
        # Return the code string. 
        return code


    @staticmethod
    def _init_env(globals_env = None, locals_env = None):

        """Merge optional global and local dicts into a single environment.

        Parameters
        ----------
        globals_env : dict or None
            Extra global bindings (default None).

        locals_env : dict or None
            Extra local bindings (default None).

        Returns
        -------
        dict
            A flat dict combining both inputs.
        """
        base_globals = {} if globals_env is None else globals_env
        base_locals  = {} if locals_env is None else locals_env
        return { **base_globals, **base_locals }


    def _exec(self, code, copy = True):
        """Execute a code string in the init environment.

        Parameters
        ----------
        code : str
            Python source code to execute.

        copy : bool, optional
            If ``True`` (default), execute in a shallow copy of
            :attr:`env` so that the original is not mutated.

        Returns
        -------
        dict
            The environment dict after execution.
        """
        if copy: env = self.env.copy()
        else: env = self.env
        exec(code, env, env)
        return env
    

    def _eval(self, code, copy = True):
        """Evaluate an expression string in the init environment.

        Parameters
        ----------
        code : str
            Python expression to evaluate.

        copy : bool, optional
            If ``True`` (default), evaluate in a shallow copy of
            :attr:`env` so that the original is not mutated.

        Returns
        -------
        object
            The result of evaluating *code*.
        """
        if copy: env = self.env.copy()
        else: env = self.env
        return eval(code, env, env)
    

    def compile_fn(self, argstr, codestr):
        """Compile a code string into a callable function.

        Wraps *codestr* in a ``def`` block with the given argument
        signature, executes the definition in a copy of :attr:`env`,
        and returns the resulting function object.

        Parameters
        ----------
        argstr : str
            Comma-separated parameter names for the generated function
            (e.g. ``"target"`` or ``"target, key, fig, axes"``).
            
        codestr : str
            The function body. May span multiple lines; each line is
            auto-indented under the ``def`` statement.

        Returns
        -------
        callable
            The compiled function.
        """
        # Generate a random function name to avoid collisions
        fn_name = f"__fn_{os.urandom(8).hex()}" 
        
        # Parse the code string (add indentation for function definition).
        code = "\n".join([("    " + ln) for ln in codestr.split("\n")])

        # Execute the function definition in the `init` environment. 
        local_env = self._exec(f"def {fn_name}({argstr}):\n{code}")

        # Return the compiled function from the local environment.
        return local_env[fn_name]