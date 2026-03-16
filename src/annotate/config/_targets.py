# -*- coding: utf-8 -*-
################################################################################
# annotate/config/_targets.py

"""Target configuration for cortex-annotate.

TargetsConfig is a lazy-dict that enumerates all annotation targets from
the ``targets`` section of config.yaml. Targets are identified by tuples
of concrete key values (e.g. ``("HCP", "sub-01", "lh")``), and their
metadata is lazily computed on first access.

Concrete keys define the combinatorial axes (datasets, participants,
hemispheres), while string-valued keys are compiled into functions that
compute derived metadata from the partially-built target dict.
"""

# Imports ----------------------------------------------------------------------

from .._util import ldict, delay
from ._error import ConfigError
    
# Targets Configuration --------------------------------------------------------

class TargetsConfig(ldict):
    """Lazy dict of annotation targets parsed from config.yaml.

    Each key is a target id tuple of concrete key values (e.g.
    ``("HCP", "sub-01", "lh")``). Values are lazily reified
    :class:`~annotate._util.ldict` instances containing all target
    metadata — both concrete and computed.

    Parameters
    ----------
    targets_yaml : dict
        The ``targets`` section from config.yaml. Keys are target
        dimension names; values are lists (concrete), dicts
        (dependent concrete), or code strings (computed metadata).

    init : InitConfig
        The init environment for compiling computed-key functions.

    Attributes
    ----------
    item_generators : dict
        Maps each target key name to its generator: a list of values
        for simple concrete keys, a dependency dict for dependent
        concrete keys, or a compiled function for computed keys.

    concrete_keys : list of str
        Ordered list of concrete key names that form the target id
        tuples.

    target_keys : list of tuple
        All target id tuples, built as the Cartesian product over
        concrete keys (respecting dependencies).
    """
    
    __slots__ = ( "item_generators", "concrete_keys", "target_keys" )    

    def __init__(self, targets_yaml, init):
        # The targets section is required.
        if targets_yaml is None:
            raise ConfigError("targets", "targets section is required.")

        # The targets section must be a mapping (dictionary).
        if not isinstance(targets_yaml, dict):
            raise ConfigError("targets", "targets section must be a mapping.")

        # First, we step through and compile the keys when necessary.
        self.item_generators = {} # initialize
        self.concrete_keys = [] # initialize
        for (key, value) in targets_yaml.items():
            self._parse_target(key, value, init)

        # Second, we build the product of all concrete keys
        self.target_keys = self._build_targets_keys()

        # Third, we then fill these out into a lazy dict that reifies each target
        # individually. We start with a dict but put the delays into this object
        # (which is a lazy dict itself).
        targets_dict = {} 
        for target_id in self.target_keys:
            targets_dict[target_id] = delay(
                TargetsConfig._reify_target, 
                self.item_generators, 
                self.concrete_keys, 
                target_id
            )
            
        # Finally, we update this object with the target data.
        self.update(targets_dict)

   
    # Parsing Methods ----------------------------------------------------------

    def _parse_dict_target(self, key, value, init):
        """Parse a dependent concrete key from config.yaml.

        A dependent key's values vary based on a parent concrete key.
        The mapping must contain a ``depends_on`` field naming the
        parent. Values are either enumerated per parent value or
        computed via a ``calculate`` code string.

        Parameters
        ----------
        key : str
            The target dimension name being parsed.
            
        value : dict
            The YAML mapping for this key, containing ``depends_on``
            and either per-parent value lists or a ``calculate`` field.

        init : InitConfig
            The init environment for compiling the calculate function.

        Returns
        -------
        dict
            A dependency dict with ``"depends_on"`` and either
            per-parent value lists or a ``"calculate"`` callable.
        """
        # Check that `depends_on` field is present in the dictionary.
        depends_on = value.get("depends_on", None)
        if depends_on is None:
            raise ConfigError(f"targets.{key}", 
                f"Target items that are mappings must contain a "
                f"'depends_on' field: {value}"
            )

        # Check that the `depends_on` field is a string.
        if not isinstance(depends_on, str):
            raise ConfigError(f"targets.{key}", 
                f"'depends_on' field must be a string: {depends_on}"
            )
        
        # Check that the `depends_on` field refers to a valid target.
        parents = self.item_generators.get(depends_on, None)
        if parents is None or not isinstance(parents, list):
            raise ConfigError(f"targets.{key}", 
                f"'depends_on' field must refer to a valid target with "
                f"a list value: {depends_on}"
            )

        # If `calculate` field is present, compile the code. The return
        # should be a list per parent key. 
        calculate = value.get("calculate", None)
        if calculate is not None:
            return {
                "depends_on" : depends_on,
                "calculate"  : init.compile_fn("target", calculate)
            }

        # If there is no `calculate` field, then the given dictionary 
        # should have the parent keys as fields with list values.
        value_dict = { k: value.get(k, None) for k in parents }

        # Check that all parent keys are present in the value dict.
        if not all(isinstance(v, list) for v in value_dict.values()):
            raise ConfigError(f"targets.{key}", 
                f"Target items that are mappings must contain a "
                f"field for each parent key with a list value: "
                f"{self.item_generators[depends_on]} -> "
                f"{[x for x in value.keys() if x != 'depends_on']}"
            )
        
        # Return the value dict with the depends_on field.
        return { "depends_on": depends_on, **value_dict }


    def _parse_target(self, key, value, init):
        """Parse a single target entry and register it.

        Dispatches on the YAML value type: lists become simple concrete
        keys, dicts become dependent concrete keys (via
        :meth:`_parse_dict_target`), and strings become compiled
        computed-metadata functions. Results are stored in
        :attr:`item_generators` and, for concrete keys, appended to
        :attr:`concrete_keys`.

        Parameters
        ----------
        key : str
            The target dimension or metadata name.

        value : list, dict, or str
            The raw YAML value for this target entry.

        init : InitConfig
            The init environment for compiling code strings.
        """
        # If list, then this will become a concrete_key.
        if isinstance(value, list):
            self.item_generators[key] = value 
            self.concrete_keys.append(key)

        # If dict, then this is a concrete key with dependencies. 
        elif isinstance(value, dict):
            self.item_generators[key] = self._parse_dict_target(key, value, init)
            self.concrete_keys.append(key)

        # If string, then this is treated as a code block that is compiled 
        # into a function that takes `target` as an argument.
        elif isinstance(value, str):
            self.item_generators[key] = init.compile_fn("target", value)

        else:
            # Error if the item value is not a list, dict, or string.
            raise ConfigError(f"targets.{key}", 
                f"Target elements must be lists, dicts, or strings: {value}"
            )                
    
    # Target Key Building Methods ----------------------------------------------

    def _resolve_concrete_items(self, concrete_key, partial_target):
        """Resolve the list of values for a concrete key.

        For simple concrete keys, returns the stored list directly. For
        dependent keys, looks up the parent value in *partial_target*
        and either calls the ``calculate`` function or retrieves the
        pre-enumerated list. Calculated results are cached back into
        :attr:`item_generators` for future lookups.

        Parameters
        ----------
        concrete_key : str
            The concrete key name to resolve.

        partial_target : dict
            A dict mapping previously resolved concrete key names to
            their values for the current target id being built.

        Returns
        -------
        list
            The concrete values for this key given the partial target.
        """
        # Get the concrete items for this key.
        concrete_items = self.item_generators[concrete_key].copy()

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
                self.item_generators[concrete_key].update({ parent_value : values })

            # If there is no "calculate" key, then we get the parent key and
            # values and build the values based on the parent key values.
            else:
                values = concrete_items[parent_value]

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
        """Builds the target id tuples as the product over the concrete keys.

        Iterates through :attr:`concrete_keys` in order, expanding each
        partial target id by all values returned from
        :meth:`_resolve_concrete_items`. For example, if the concrete keys are
        "Dataset" and "Participant", then a target key might be 
        ("DatasetA", "sub-01").
        
        After expansion, any leftover ``calculate`` callables are removed from 
        :attr:`item_generators`. 

        Returns
        -------
        list of tuple
            All target id tuples. Each tuple has one element per
            concrete key, in :attr:`concrete_keys` order.
        """
        # Initialize the target keys as a list with one empty tuple.
        targets_keys = [()] 

        # For each concrete key, we build up the target keys by taking the
        # product of the current target with the previous target keys.
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

        # Clean up `calculate` fields from the targets dictionary
        for concrete_key in self.concrete_keys:
            generator = self.item_generators[concrete_key]
            if isinstance(generator, dict) and "calculate" in generator:
                self.item_generators[concrete_key].pop("calculate")

        # Return the target keys.
        return targets_keys
    
    # Target Reification Methods -----------------------------------------------

    @staticmethod
    def _reify_target(item_generators, concrete_keys, target_id):
        """Build the full metadata dict for a single target.

        Constructs an :class:`~annotate._util.ldict` containing every
        key defined in config.yaml. Concrete keys receive their values
        from *target_id*; computed keys receive lazy thunks that will
        evaluate on first access.

        Parameters
        ----------
        item_generators : dict
            The generators dict from :attr:`item_generators`.

        concrete_keys : list of str
            Ordered concrete key names.
            
        target_id : tuple
            The concrete values identifying this target.

        Returns
        -------
        ldict
            A lazy dict with all target metadata.
        """
        d = ldict()
        target_iter = iter(target_id)
        for (key, value) in item_generators.items():
            if key in concrete_keys:
                d[key] = next(target_iter)
            else:
                d[key] = delay(value, ldict(d))
        return d