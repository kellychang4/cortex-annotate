# -*- coding: utf-8 -*-
################################################################################
# annotate/_util.py
#
# Utility types and functions used in the annotation toolkit.


# Imports ----------------------------------------------------------------------

import os
import yaml
import numpy as np
from functools import partial
from collections import namedtuple
from numbers import Real, Integral

from ._util import delay, ldict, fix_style

# Configuration Error ----------------------------------------------------------

class ConfigError(Exception):
    """An exception raised due to errors in the config.yaml file.
    
    `ConfigError` is a subclass of `Exception` that is raised when an error is
    encountered while parsing the `config.yaml` file used to configure the
    `cortex-annotate` project.
    """
    
    __slots__ = ( )
   
    def __init__(self, section, message): 
        super().__init__(
            f"Invalid `config.{section}`\n"
            f"{message}"
        )


# Display Configuration --------------------------------------------------------

class DisplayConfig():
    """An object that tracks the configuration of the tool's image display.

    The `DisplayConfig` type keeps track of the `display` section of the
    `config.yaml` file for the `cortex-annotate` project.
    """
    
    __slots__ = ( 
        "figsize", "dpi", "image_size", "active_style", "default_style",
    )
    
    def __init__(self, display_yaml):
        # The display section is optional. If None, we use default values.
        if display_yaml is None: display_yaml = {}
        
        # Initialize the figure size.
        self.figsize = self._init_figsize(display_yaml, default = [4, 4])

        # Initialize the DPI. 
        self.dpi = self._init_dpi(display_yaml, default = 128)

        # Calculate the image size in pixels from the figure size and DPI.
        self.image_size = tuple([round(self.dpi * x) for x in self.figsize])
        
        # Initialize the active style.
        self.active_style = self._init_style(
            display_yaml, parameter = "active_style", default = {})

        # Initialize the default style.
        self.default_style = self._init_style(
            display_yaml, parameter = "default_style", default = {})
        

    @staticmethod
    def _init_figsize(display_yaml, default = [4, 4]):
        """Initializes the figure size from the display yaml."""
        # Prepare ConfigError arguments for any errors that may arise in this function.
        err = partial(ConfigError, "display.figsize")

        # Extract the figure size from the yaml. 
        figsize0 = figsize = display_yaml.get("figsize", default)

        # Check that the figure size is not a string.
        if isinstance(figsize, str):
            raise err(f"figsize cannot be a string: {figsize}")

        # If the figure size is a single number, use both for the dimensions.
        if isinstance(figsize, (int, float)):
            figsize = [figsize, figsize]

        # If the figure size is a list/tuple, check there are two dimensions.
        if isinstance(figsize, (list, tuple)):
            # Check that there are two dimensions.
            if len(figsize) != 2:
                raise err(
                    f"figsize must be a number or 2-element list/tuple: "
                    f"{figsize}"
                )
            # Check that the figure size elements are positive numbers.
            if not all(isinstance(u, Real) and u > 0 for u in figsize):
                raise err(
                    f"figsize elements must be positive numbers: "
                    f"{figsize0}",
                )
        
        # Return the figure size as a tuple in the form (width, height).
        return tuple(figsize)


    @staticmethod
    def _init_dpi(display_yaml, default = 128):
        """Initializes the DPI from the display yaml."""

        # Prepare ConfigError arguments for any errors that may arise in this function.
        err = partial(ConfigError, "display.dpi")

        # Extract the DPI from the yaml.
        dpi = display_yaml.get("dpi", default)

        # Check that the DPI is a positive integer.
        if not isinstance(dpi, Integral) or dpi < 1:
            raise err(f"dpi must be a positive integer: {dpi}")
        
        # Return the DPI.
        return dpi
    

    @staticmethod
    def _init_style(display_yaml, parameter, default = {}):
        """Initializes a style from the display yaml."""
        # Import for style checking.
        from ._core import AnnotationState

        # Prepare ConfigError arguments for any errors that may arise in this function.
        err = partial(ConfigError, f"display.{parameter}")

        # Extract the style from the yaml.
        style = display_yaml.get(parameter, default)

        # Check that the style is a yaml mapping (dictionary)
        if not isinstance(style, dict):
            raise err(f"{parameter} must be a mapping")
        
        # Try to make sure the style keys are valid
        try: fix_style(style)
        except RuntimeError as e: raise err(e) from e

        return style


    def as_dict(self):
        """Returns the display configuration as a dictionary."""
        return { key: getattr(self, key) for key in self.__slots__ }


# Init Configuration -----------------------------------------------------------

class InitConfig:
    """An object that keeps track of the `init` section of `config.yaml`.

    The `InitConfig` type is used to keep track of the `init` section of the
    `config.yaml` file for the `cortex-annotate` project. The `init` section
    contains a code-block whose local values (after the code block is executed)
    are made available to all other code blocks in the config file. This allows
    one to, for example, import a library in the init block that is then
    available throughout the config file.
    """
    
    __slots__ = ( "code", "env" )
    
    def __init__(self, code, globals = None, locals = None):
        # Initialize the code string.
        self.code = self._init_code(code)

        # Prepare the given globals and locals for the merged environment.
        self.env = self._init_env(globals, locals)

        # Execute the code block to populate the environment.
        exec(self.code, self.env, self.env)


    @ staticmethod
    def _init_code(code):
        """Initializes the given code string by executing it in the context of the init code."""
        # Prepare ConfigError arguments for any errors that may arise in this function.
        err = partial(ConfigError, "init")

        # The code is optional. If None, we just use an empty code block.
        if code is None: code = "None"

        # Check that the code is a string.
        if not isinstance(code, str):
            raise err(f"init section must be a string: {code}")
        
        return code


    @staticmethod
    def _init_env(globals, locals):
        """Initializes the environment by merging the given locals and globals with the init environment."""
        base_globals = {} if globals is None else globals
        base_locals  = {} if locals is None else locals
        return { **base_globals, **base_locals }


    def _exec(self, code, copy = True):
        """Executes the given code string in the `init` environment."""
        if copy:
            env = self.env.copy()
        else:
            env = self.env
        exec(code, env, env)
        return env
    

    def _eval(self, code, copy = True):
        """Evaluates the given code string in the `init` environment."""
        if copy:
            env = self.env.copy()
        else:
            env = self.env
        return eval(code, env, env)
    

    def compile_fn(self, argstr, codestr):
        """Compiles the given code string as a function in the `init` environment."""
        # Generate a random function name to avoid collisions
        fn_name = f"__fn_{os.urandom(8).hex()}" 
        
        # Parse the code string (add indentation for function definition).
        code = "\n".join([("    " + ln) for ln in codestr.split("\n")])

        # Execute the function definition in the `init` environment. 
        local_env = self._exec(f"def {fn_name}({argstr}):\n{code}")

        # Return the compiled function from the local environment.
        return local_env[fn_name]
    
    
# Targets Configuration --------------------------------------------------------

class TargetsConfig(ldict):
    """A dict-like configuration item for the annotation tool's targets.

    The `TargetsConfig` type is a (lazy) dict-like object that stores, as dict
    entries, the targets of the annotation project (i.e., dataset, participants,
    hemispheres) as well as meta-data about the targets.

    For a `TargetsConfig` object `targets`, `targets[(id1, id2...)]` evaluates
    to the `target` dictionary for the target that is identified by the values
    for the ordered concrete key `id1, id2...`.
    """
    
    __slots__ = ( "items", "concrete_keys" )    

    def __init__(self, targets_yaml, init):
        # The targets section is required.
        if targets_yaml is None:
            raise ConfigError("targets", "targets section is required.")

        # The targets section must be a mapping (dictionary).
        if not isinstance(targets_yaml, dict):
            raise ConfigError("targets", "targets section must be a mapping.")

        # First, we step through and compile the keys when necessary.
        self.items = {} # initialize
        self.concrete_keys = [] # initialize
        for (key, value) in targets_yaml.items():
            self._parse_target(key, value, init)

        # Second, we build the product of all concrete keys
        targets_keys = self._build_targets_keys()

        # Third, we then fill these out into a lazy dict that reifies each target
        # individually. We start with a dict but put the delays into this object
        # (which is a lazy dict itself).
        targets_dict = {} 
        for target_id in targets_keys:
            targets_dict[target_id] = delay(
                TargetsConfig._reify_target, 
                self.items, 
                self.concrete_keys, 
                target_id
            )
            
        # Finally, we update this object with the target data.
        self.update(targets_dict)

   
    # Parsing Methods ----------------------------------------------------------

    def _parse_dict_target(self, key, value, init):
        """Parses a dictionary target entry. 
        
        A dictionary target entry is a concrete key that depends on another
        concrete key. It must contain a "depends_on" field that specifies the
        concrete key that it depends on. 
        
        The mapping can either specify the list values for each parent key of it
        can contain a "calculate" field that is a code string that is compiled 
        into a function that takes `target` as an argument and returns a list of
        values.
        """
        # Check that "depends_on" field is present in the dictionary.
        depends_on = value.get("depends_on", None)
        if depends_on is None:
            raise ConfigError(f"targets.{key}", 
                f"Target items that are mappings must contain a "
                f"'depends_on' field: {value}"
            )

        # Check that the depends_on field is a string.
        if not isinstance(depends_on, str):
            raise ConfigError(f"targets.{key}", 
                f"'depends_on' field must be a string: {depends_on}"
            )
        
        # Check that the depends_on field refers to a valid target.
        parents = self.items.get(depends_on, None)
        if parents is None or not isinstance(parents, list):
            raise ConfigError(f"target.{key}", 
                f"'depends_on' field must refer to a valid target with "
                f"a list value: {depends_on}"
            )

        # If "calculate" field is present, compile the code. The return
        # should be a list per parent key. 
        calculate = value.get("calculate", None)
        if calculate is not None:
            return {
                "depends_on" : depends_on,
                "calculate"  : init.compile_fn("target", calculate)
            }

        # If there is no "calculate" field, then the given dictionary 
        # should have the parent keys as fields with list values.
        value_dict = { k: value.get(k, None) for k in parents }

        # Check that all parent keys are present in the value dict.
        if not all(isinstance(v, list) for v in value_dict.values()):
            raise ConfigError(f"targets.{key}", 
                f"Target items that are mappings must contain a "
                f"field for each parent key with a list value: "
                f"{self.items[depends_on]} -> "
                f"{[x for x in value.keys() if x != 'depends_on']}"
            )
        # Return the value dict with the depends_on field.
        return { "depends_on": depends_on, **value_dict }


    def _parse_target(self, key, value, init):
        """Parses a target entry.
        
        A target entry can be a list, dict, or string. If it is a list, then it
        is treated as a concrete key with the list as its values. If it is a dict,
        then it is treated as a concrete key with dependencies that are parsed by
        the `_parse_dict_target` method. If it is a string, then it is treated as
        a code block that is compiled into a function that takes `target` as an
        argument.
        """
        # If list, then this will become a concrete_key.
        if isinstance(value, list):
            self.items[key] = value 
            self.concrete_keys.append(key)

        # If dict, then this is a concrete key with dependencies. 
        elif isinstance(value, dict):
            self.items[key] = self._parse_dict_target(key, value, init)
            self.concrete_keys.append(key)

        # If string, then this is treated as a code block that is compiled 
        # into a function that takes `target` as an argument.
        elif isinstance(value, str):
            self.items[key] = init.compile_fn("target", value)

        else:
            # Error if the item value is not a list, dict, or string.
            raise ConfigError(f"targets.{key}", 
                f"Target elements must be lists, dicts, or strings: {value}"
            )                
    
    # Target Key Building Methods ----------------------------------------------

    def _resolve_concrete_items(self, concrete_key, partial_target):
        """Resolves the concrete items for a concrete key based on the partial target."""
        # Get the concrete items for this key.
        concrete_items = self.items[concrete_key].copy()

        # If the concrete items is a list, then use the list
        if isinstance(concrete_items, list):
            return concrete_items
        
        # If the concrete items is a dict, then this is a dependent concrete key.
        elif isinstance(concrete_items, dict):
            # Get the parent key and value that this concrete key depends on.
            depends_on   = concrete_items["depends_on"]
            parent_value = partial_target[depends_on]

            # Pop the "calculate" key if it exists. 
            calculate = concrete_items.get("calculate", None)
            if calculate is not None:
                # Use the calculate function to get the values based on the partial target.
                values = sorted(calculate(partial_target))

                # Store the calculated values into the original dictionary
                self.items[concrete_key].update({ parent_value : values })

            # If there is no "calculate" key, then we get the parent key and
            # values and build the values based on the parent key values.
            else:
                values = concrete_items[parent_value]
                print(values)

            # Check that the values is a list.
            if not isinstance(values, list):
                raise ConfigError(f"targets.{concrete_key}", 
                    f"Concrete key values must be lists: {values}"
                )
            
            # Return the values.
            return values
        else: 
            raise ConfigError(f"targets.{concrete_key}",
                f"Concrete key items must be lists or dicts: {concrete_items}"
            )


    def _build_targets_keys(self):
        """Builds the target keys by taking the product over the concrete keys.
        
        The target keys are the tuples of values for the concrete keys that 
        identify each target. For example, if the concrete keys are "Dataset"
        and "Participant", then a target key might be ("DatasetA", "sub-01").

        Each concrete key is resolved in order. Dict-typed keys receive a 
        partial target built from the concrete keys resolved so far.
        """

        # Initialize the target keys as a list with one empty tuple.
        targets_keys = [()] 

        # For each concrete key, we build up the target keys by taking the
        # produict of the current target with the previous target keys.
        for concrete_key in self.concrete_keys:
            # Initialize the update keys as an empty list. 
            update_keys = [] 

            # For each target key, we build up the new target keys by appending 
            for target_id in targets_keys:
                # Build a lookup for the dependencies of this key based on the 
                # current target_id.
                partial_target = dict(zip(self.concrete_keys, target_id))
                values = self._resolve_concrete_items(concrete_key, partial_target)

                # Append previous and current target key values
                for v in values: update_keys.append(target_id + (v,))
            
            # Update the target keys with the update keys.
            targets_keys = update_keys

        # Clean up "calculate" fields from the targets dictionary
        for concrete_key in self.concrete_keys:
            if "calculate" in self.items[concrete_key]:
                self.items[concrete_key].pop("calculate")

        # Return the target keys.
        return targets_keys
    
    # Target Reification Methods -----------------------------------------------

    @staticmethod
    def _reify_target(items, concrete_keys, target_id):
        """Builds up and returns an `ldict` of the all target data.

        `TargetsConfig._reify_target(items, concrete_keys, target_id)`
        takes the target-id tuple `target_id` and builds up the `ldict` 
        representation of the target data, in which all keys in the 
        `config.yaml` file have values (albeit lazy ones in the case of the keys
        that are not concrete). The parameters `items` and `concrete_keys` 
        must be the configuration's target data and the list of concrete keys 
        must be the concrete keys for the target, respectively.
        """
        d = ldict()
        target_iter = iter(target_id)
        for (key, value) in items.items():
            if key in concrete_keys:
                d[key] = next(target_iter)
            else:
                d[key] = delay(value, ldict(d))
        return d


# Annotation Configuration -----------------------------------------------------

#NOTE: might want to add `style` back in at some point.
Annotation = namedtuple(
    typename    = "Annotation",
    field_names = ( "type", "fixed_head", "fixed_tail", "figure_grid", "filter" ),
    defaults    = ( "contour", None, None, None, None )
)


class AnnotationsConfig(dict):
    """An object that stores the configuration of the annotations to be drawn.

    The `AnnotationsConfig` type tracks the contours and boundaries that are to
    be drawn on the annotation targets for the `cortex-annotate` project.
    """
    
    __slots__ = ( 
        "names", "type", "figure_grid", "grid_shape", 
        "fixed_head", "fixed_tail", "fixed_heads", "fixed_tails",
        "fixed_points", "fixed_dependencies", "figure_names"
    )
    
    def __init__(self, annotations_yaml, init):
        # The annotations section is required as a mapping (dictionary)
        if not isinstance(annotations_yaml, dict):
            raise ConfigError("annotations", "annotations must contain a mapping.")

        # Go through and build up the annotation data.
        annotations_dict = {}
        for (key, value) in annotations_yaml.items():
            # Check that the annotation value is a list or mapping.
            if not isinstance(value, (list, dict)):
                raise ConfigError(f"annotations.`{key}`",
                    f"annotation `{key}` must be a list or mapping.")
        
            if isinstance(value, list):
                # If the value is a list, then this is treated as a figure_grid.
                figure_grid = self._init_figure_grid(
                    value, partial(ConfigError, f"annotations.`{key}`"))
                annotations_dict[key] = Annotation(figure_grid = figure_grid)
            else: 
                # If the value is a mapping, then this is treated as an annotation
                # specification that is processed by the `_init_annotation` method.
                annotations_dict[key] = self._init_annotation(key, value, init)
        
        # And now all the annotations are processed, update the dictionary.
        self.update(annotations_dict)

        # Prepare annotation attribute information for easy access later.
        self.names       = list(self.keys())
        self.type        = self._get_type()
        self.figure_grid = self._get_figure_grid()
        self.grid_shape  = self._get_grid_shape()
        self.fixed_head  = self._get_fixed_info("fixed_head")
        self.fixed_tail  = self._get_fixed_info("fixed_tail")
        self.fixed_heads = self._get_fixed_names("fixed_head")
        self.fixed_tails = self._get_fixed_names("fixed_tail")

        # Combine the fixed head and tails into one dictionary for ease.
        # <key> : [ <fixed_head>, <fixed_tail> ] needed to have valid points.
        self.fixed_points = {
            k: [ *self.fixed_heads[k], *self.fixed_tails[k] ] for k in self.keys()
        }

        # Create the fixed dependencies dictionary, which is the reverse of the 
        # fixed points dictionary.
        # <key> : [ <annotations that have downstream dependencies> ]
        self.fixed_dependencies = { k: [] for k in self.keys() }
        for key in self.fixed_dependencies.keys():
            for src, value in self.fixed_points.items():
                if key in value: self.fixed_dependencies[key].append(src)

        # Finally, we get all the unique figure names.
        self.figure_names = set([
            x 
            for annotation in self.values()
            for row in annotation.figure_grid
            for x in row if x is not None
        ])

    # Initialization Methods ---------------------------------------------------

    @staticmethod
    def _init_figure_grid(figure_grid, err):
        """Initializes the figure grid from the annotation specification."""
        # Check that the figure grid is a list.
        if not isinstance(figure_grid, list):
            raise err(f"figure_grid is required and must be a list/matrix.")
        
        # Single-row shorthand: ["a", "b", None] -> [["a", "b", None]]
        if all(el is None or isinstance(el, str) for el in figure_grid):
            figure_grid = [ figure_grid ]

        # Check the elements of the figure_grid.
        cols = None
        for row in figure_grid: 
            # Check that the row is a list. 
            if not isinstance(row, list):
                raise err("figure_grid must be a list/matrix.")
            
            # Check that the row has the same number of columns.
            if cols is None: cols = len(row) # defined by the first row
            elif len(row) != cols:
                raise err(
                    f"figure_grid cannot be a ragged matrix: "
                    f"expected {cols} columns, got {len(row)}"
                )
            
            # Check that the row elements are strings or None.
            for el in row:
                if el is None: continue
                elif not isinstance(el, str):
                    raise err("figure_grid items must be null or strings.")

        # Return the figure_grid.
        return figure_grid
    

    @staticmethod
    def _init_fixed_points(key, fixed_point, err, init):
        """Initializes the fixed points from the annotation specification."""
        # If the fixed point is None, then we just return None.
        if fixed_point is None: return None

        # Check that the fixed point is a string or mapping.
        if not isinstance(fixed_point, (str, dict)):
            raise err(f"{key} must be null, strings, or mappings.")
        
        # If the fixed point is a string, we use the last point of the given
        # annotation as the fixed point. 
        if isinstance(fixed_point, str):
            fixed_point = { 
                "calculate" : f"return annotations['{fixed_point}'][-1,:]",
                "requires"  : fixed_point
            }

        # Extract the requires and calculate fields from the mapping.
        requires  = fixed_point.get("requires", [])
        calculate = fixed_point.get("calculate", None)

        # Check that the requires field is a string.
        if isinstance(requires, str):
            requires = [ requires ]
        
        # Check that the requires field is a list of strings.
        if isinstance(requires, list):
            if not all(isinstance(el, str) for el in requires):
                raise err(f"{key} 'requires' field must be a string or list of strings.")
            
        # Check that the calculate field is a string.
        if calculate is None:
            raise err(f"{key} must contain 'calculate' if it is a mapping.")
            
        # Compile the calculate code string into a function. 
        calculate = init.compile_fn("annotations", calculate)
        
        # Return the fixed point dictionary.
        return { "calculate": calculate, "requires": requires }


    def _init_annotation(self, annotation_name, annotation_spec, init):
        """Initializes the annotation from the annotation specification."""
        # Prepare ConfigError arguments for any errors that may arise in this loop.
        err = partial(ConfigError, f"annotations.`{annotation_name}`")

        # Check that the key is a valid annotation option.
        for key in annotation_spec.keys():
            if key not in Annotation._fields:
                raise err(f"Invalid annotation key: {key}")

        # Extract annotation values or assign default values.
        ctype         = annotation_spec.get("type", "contour")
        fixed_head    = annotation_spec.get("fixed_head", None)
        fixed_tail    = annotation_spec.get("fixed_tail", None)
        figure_grid   = annotation_spec.get("figure_grid", None)
        filter        = annotation_spec.get("filter", None)
        
        # Check that the annotation type is valid.
        if ctype not in ( "contour", "boundary", "point"):
            raise err("Type must be one of 'contour', 'boundary', or 'point'.")

        # Check and initialize the fixed points.
        fixed_head = self._init_fixed_points("fixed_head", fixed_head, err, init)
        fixed_tail = self._init_fixed_points("fixed_tail", fixed_tail, err, init)

        # Prepare and check the figure grid.
        figure_grid = self._init_figure_grid(figure_grid, err)

        # Check that the filter is a string or None.
        if filter is not None and not isinstance(filter, str):
            raise err(f"filter must be null or a Python code string.")

        # We have extracted the data now; go ahead and compile the filter.
        if filter is not None:
            filter = init.compile_fn("target", filter)

        # Return the annotation as an Annotation object.    
        return Annotation(
            type        = ctype,
            fixed_head  = fixed_head,
            fixed_tail  = fixed_tail,
            figure_grid = figure_grid,
            filter      = filter,
        )

    # Get Methods --------------------------------------------------------------

    def _get_type(self):
        """Returns a dictionary mapping annotation names to their types."""
        return { key: value.type for (key, value) in self.items() }


    def _get_figure_grid(self):
        """Returns a dictionary mapping annotation names to their figure grids."""
        return { key: value.figure_grid for (key, value) in self.items() }


    def _get_grid_shape(self):
        """Returns a dictionary mapping annotation names to their figure grid shapes."""
        return { key: np.shape(value.figure_grid) for (key, value) in self.items() }
    

    def _get_fixed_info(self, fixed_point):
        """Returns a dictionary mapping annotation names to their fixed point info."""
        # Ensure that the name is either "fixed_head" or "fixed_tail".
        if fixed_point not in ("fixed_head", "fixed_tail"):
            raise ValueError(f"Invalid fixed point: {fixed_point}")
            
        # Returns a dict mapping annotation names to their fixed point information.
        return { 
            key: getattr(value, fixed_point) for (key, value) in self.items() 
        }
    

    def _get_fixed_names(self, fixed_point):
        """Returns a dictionary mapping annotation names to their fixed point name."""
        # Ensure that the name is either "fixed_head" or "fixed_tail".
        if fixed_point not in ("fixed_head", "fixed_tail"):
            raise ValueError(f"Invalid fixed point: {fixed_point}")
            
        # Returns a dict mapping annotation names to their fixed point name.
        return { 
            key: [] if getattr(value, fixed_point) is None 
            else getattr(value, fixed_point)["requires"]
            for (key, value) in self.items() 
        }
        

# Figures Configuration --------------------------------------------------------

class FiguresConfig(dict):
    """An object that stores configuration information for making figures.

    The `FiguresConfig` type stores information from the `figures` section of
    the `config.yaml` file for the `cortex-annotate` project. It resembles a
    Python `dict` object whose keys are the figure names and whose values are
    Python functions (which require the arguments `target`, `key`, `fig`, 
    `axes`, `figsize`, `dpi`, and `meta_data`) that generate the appropriate 
    figure.
    """
    
    __slots__ = ( "yaml", )    
    
    def __init__(self, figures_yaml, figure_names, init):
        # The figures section is required as a mapping (dictionary).
        if not isinstance(figures_yaml, dict):
            raise ConfigError("figures", "figures section must contain a mapping.")
        
        # Store the original figures section yaml.
        self.yaml = figures_yaml
        
        # Prepare the figures yaml and the figure compiling functions.
        figures_yaml, compile_fn, wildfn = FiguresConfig._prep_yaml(
            self.yaml.copy(), init)

        # Prepare the figure dictionary.
        figures_dict = self._init_figures_dict(
            figures_yaml, figure_names, compile_fn, wildfn)
        
        # Update FiguresConfig class dictionary.
        self.update(figures_dict)


    @staticmethod
    def _compile_fn(init, initcode, termcode, code):
        """Compiles the code strings as a figure function in the `init` environment."""
        return init.compile_fn(
            "target, key, fig, axes, figsize, dpi, meta_data",
            f"{initcode}\n{code}\n{termcode}"
        )
    

    @classmethod
    def _prep_yaml(cls, figures_yaml, init):
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
            cls._compile_fn, init, 
            special_dict["init"], special_dict["term"]
        )

        # Compile the wildcard field if not None.
        wildfn = None
        if special_dict["_"] is not None:
            wildfn = compile_fn(special_dict["_"])

        return ( figures_yaml, compile_fn, wildfn )


    @staticmethod
    def _init_figures_dict(figures_yaml, figure_names, compile_fn, wildfn):
        """Initializes the figure dictionary from the figures yaml."""
        figures_dict = {}
        for key in figure_names:
            if key not in figures_yaml:
                if wildfn is None:
                    raise ConfigError(
                        f"figures.{key}", 
                        f"Missing code for figure '{key}' and "
                        f"no wildcard provided."
                    )
                else:
                    figures_dict[key] = wildfn
            else: 
                code = figures_yaml.get(key, None)
                if code is not None:
                    figures_dict[key] = compile_fn(code)
        return figures_dict
            
# Cortex Configuration ---------------------------------------------------------

class CortexConfig(dict):
    """An object that stores configuration information for the cortex viewer.

    The `CortexConfig` type stores information from the `cortex` section of
    the `config.yaml` file for the `cortex-annotate` project. It is a dictionary.
    """
    
    def __init__(self, cortex_yaml, figure_names, init):
        # The cortex section is optional.
        if cortex_yaml is None: cortex_yaml = {}
        
        # If cortex section is provided, it must be a dictionary.
        if not isinstance(cortex_yaml, dict):
            raise ConfigError("cortex", "cortex section must contain a mapping.")
        
        # If cortex section is not empty, then we prepare the cortex dictionary.
        cortex_dict = {} # initialize
        if cortex_yaml != {}: 
            # Prepare the faces field.
            cortex_dict["faces"] = self._init_faces(cortex_yaml.copy(), init)

            # Prepare the coordinates field.
            cortex_dict["coordinates"] = self._init_coordinates(
                cortex_yaml.copy(), init)
            
            # Prepare the inflate_between field, optional.
            cortex_dict["inflate_between"] = self._init_inflate_between(
                cortex_yaml.copy(), cortex_dict["coordinates"], init)

            # Prepare the overlays field.
            cortex_dict["overlays"] = self._init_overlays(
                cortex_yaml.copy(), figure_names, init)
        
        # Update CortexConfig class dictionary.
        self.update(cortex_dict)


    @staticmethod
    def _compile_fn(init, code):
        """Compiles the code strings as a cortex function in the `init` environment."""
        return init.compile_fn("target, key", f"{code}")


    @staticmethod
    def _init_faces(cortex_yaml, init):
        """Initialize the `faces` from the cortex yaml."""
        # Extract the faces field from the yaml.
        faces = cortex_yaml.get("faces", None)

        # Check that faces is a string/code
        if not isinstance(faces, str):
            raise ConfigError("cortex.faces", "faces must be a code string.")

        # Compile the faces code string into a function.
        faces = CortexConfig._compile_fn(init, faces)

        # Return the compiled function.
        return faces


    @staticmethod
    def _init_coordinates(cortex_yaml, init):
        """Initialize the `coordinates` from the cortex yaml."""
        # Extract the coordinates field from the yaml.
        coordinates = cortex_yaml.get("coordinates", None)

        # Check that coordinates is a string/code
        if not isinstance(coordinates, (str, dict)):
            raise ConfigError("cortex.coordinates", 
                "`coordinates` must be a code string or mapping."
            )
        
        # If coordinates is a string, then conform to dictionary mapping.
        if isinstance(coordinates, str):
            coordinates = { "_default" : coordinates }
    
        # If coordinates is a dict, then we compile the code for each key.
        # Check that all mappings are strings/code.
        for key, value in coordinates.items():
            if not isinstance(value, str):
                raise ConfigError(f"cortex.coordinates.{key}", 
                    "Coordinate mapping values must be code strings.")
         
        # return dictionary of compile functions per coordinates
        return {
            key: CortexConfig._compile_fn(init, value) 
            for key, value in coordinates.items()
        }


    @staticmethod
    def _init_inflate_between(cortex_yaml, coordinates, init):
        """Initialize the `inflate_between` from the cortex yaml."""
        # Extract the coordinates field from the yaml.
        inflate_between = cortex_yaml.get("inflate_between", None)

        # This is an optioanl field, so if None, return None.
        if inflate_between is None: return None
        
        # If provided, must be a list of length two.
        if not isinstance(inflate_between, list) or len(inflate_between) != 2:
            raise ConfigError("cortex.inflate_between", 
                "`inflate_between` must be a list of length two.")

        # If provided, both surfaces must be available in the coordinates dict.
        for surface_name in inflate_between: 
            if surface_name not in coordinates:
                raise ConfigError("cortex.inflate_between", 
                    f"Unable to find `{surface_name}` coordinates.")
        
        # Return inflate between list.
        return inflate_between
        
    
    @staticmethod
    def _init_overlays(cortex_yaml, figure_names, init):
        """Initialize the `overlays` from the cortex yaml."""
        # If there is a wildcard key, extract out.
        overlays = cortex_yaml.get("overlays", None)

        # Check that overlays is a string/code
        if not isinstance(overlays, (str, dict)):
            raise ConfigError("cortex.overlays", 
                "`overlays` must be a code string or mapping."
            )

        # If overlays is a string, then assuming it is for the curvature.
        if isinstance(overlays, str):
            overlays = { "curvature" : overlays }

        # If overlays is a dict, then we compile the code for each key.
        # Check that all mappings are strings/code.
        for key, value in overlays.items():
            if not isinstance(value, str):
                raise ConfigError(f"cortex.overlays.{key}", 
                    "Overlay mapping values must be code strings.")
         
        # Locate wildcard key, if provided.
        wildfn = overlays.get("_", None)
        if wildfn is not None:
            wildfn = CortexConfig._compile_fn(init, wildfn)
            
        # Prepare valid figure names for the overlays, including curvature.
        figure_names = { "curvature" } | set(figure_names)

        # Prepare and return overlays dictionary
        overlays_dict = {}
        for key in figure_names:
            if key not in overlays.keys():
                if wildfn is None:
                    raise ConfigError(
                        f"cortex.overlays.{key}", 
                        f"Missing code for figure '{key}' and "
                        f"no wildcard provided."
                    )
                key_fn = wildfn
            else: 
                key_fn = CortexConfig._compile_fn(init, overlays[key])
            overlays_dict[key] = key_fn
        return overlays_dict


# Config Object ----------------------------------------------------------------

class Config:
    """The configuration object for the `cortex-annotate` project.

    The `Config` class stores information about the configuration of the
    `cortex-annotate` project. The configuration is specified in the
    `config.yaml` file. Configuration objects store a single value per top-level
    config item in the `config.yaml` file. Additional top-level items that are
    not recognized by `Config` are not parsed, but they are available in the
    `Config.yaml` member variable.
    """
    
    __slots__ = (
        "config_path", "yaml", "display", "init", "targets", "annotations", 
        "figures", "cortex" 
    )
    
    def __init__(self, config_path = "/config/config.yaml"):
        # Load the configuration YAML file.
        self.config_path = config_path
        with open(config_path, "rt") as f:
            self.yaml = yaml.safe_load(f)

        # Parse the display section.
        self.display = DisplayConfig(self.yaml.get("display", None))

        # Parse the init section.
        self.init = InitConfig(self.yaml.get("init", None))

        # Parse the targets section.
        self.targets = TargetsConfig(self.yaml.get("targets", None), self.init)

        # Parse the annotations section.
        self.annotations = AnnotationsConfig(
            self.yaml.get("annotations", None), self.init)

        # Parse the figures section.
        self.figures = FiguresConfig(
            self.yaml.get("figures", None),
            self.annotations.figure_names, 
            self.init
        )

        # Parse the cortex section.
        self.cortex = CortexConfig(
            self.yaml.get("cortex", None),
            self.annotations.figure_names, 
            self.init
        )

