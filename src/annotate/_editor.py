# -*- coding: utf-8 -*-
################################################################################
# annotate/figure/_editor.py

"""Pure annotation editing model for cortex-annotate.
 
``AnnotationEditor`` owns the shared annotation state (target, active
annotation, coordinate arrays, cursor, editable indices, fixed heads
and tails) and exposes point manipulation operations (push, pop,
toggle).
 
The editor receives an ``AnnotationsConfig`` at construction and uses
it for annotation metadata (types, names, fixed-point functions,
dependency graphs).  Target-specific annotation coordinate data is
supplied at runtime via ``update()``.
"""

# Imports ----------------------------------------------------------------------
 
import numpy as np

# Annotation Editor ------------------------------------------------------------
 
class AnnotationEditor:
    """Annotation editor class.
 
    Maintains the annotation coordinate arrays, cursor position,
    editable point indices, fixed-head/tail points, and insertion
    direction for the currently selected target and annotation.  
    
    Exposes ``push_point``, ``pop_point``, and ``toggle_cursor`` 
    for point manipulation, and ``update`` for target/annotation switching.
  
    Parameters
    ----------
    annot_cfg : AnnotationsConfig
        Parsed annotation configuration providing types, names,
        fixed-point functions, dependency graph, and grid metadata.
 
    Attributes
    ----------
    annot_cfg : AnnotationsConfig
        Reference to the annotation configuration (read-only).

    target : tuple of str or None
        Current target identifier tuple.

    active : str or None
        Name of the annotation currently being edited.

    annotations : dict of {str: ndarray or None}
        Mapping of annotation name to ``(N, 2)`` coordinate arrays
        for the current target.

    fixed_heads : dict of {str: ndarray or None}
        Fixed head point ``(1, 2)`` for each annotation, or ``None``.

    fixed_tails : dict of {str: ndarray or None}
        Fixed tail point ``(1, 2)`` for each annotation, or ``None``.

    editable : ndarray of int
        Indices into the active annotation's point array that are
        user-editable (i.e. not fixed heads or tails).
        
    cursor : int or None
        Index of the cursor within the active annotation's point
        array, or ``None`` if no editable points exist.

    insert : bool
        Insertion direction.  When ``after`` (default),
        ``push_point`` inserts after the cursor.  When ``before``,
        inserts before the cursor. 
    """

    # Define empty constants. 
    _EMPTY_POINT_MATRIX = np.zeros((0, 2), dtype = float)
    _EMPTY_EDITABLE     = np.zeros((0,), dtype = int)

    __slots__ = (
        "annot_cfg", "target", "active", "dependents", 
        "annotations", "fixed_heads", "fixed_tails", "editable", "cursor",
        "insert"
    )

    def __init__(self, config):
        """Initialize the annotation editor."""
        # Store the annotation configuration. 
        self.annot_cfg = config.annotations

        # Initialize internal variables.
        self.target      = None
        self.active      = None
        self.dependents  = None
        self.annotations = {}
        self.fixed_heads = {}
        self.fixed_tails = {}
        self.editable    = self._EMPTY_EDITABLE.copy()
        self.cursor      = None
        self.insert      = "after"

    # Static Helpers -----------------------------------------------------------
    
    def _init_editable(self, x = None):
        """Create an initial editable-index array.
 
        Parameters
        ----------
        x : int or None
            If given, the single editable index.  If ``None``,
            returns an empty array.
 
        Returns
        -------
        ndarray of int
            Length-0 or length-1 array of editable point indices.
        """
        if x is None: return self._EMPTY_EDITABLE.copy()
        return np.array([x], dtype = int)
    
    # Fixed Point Methods ------------------------------------------------------

    def calc_fixed_point(self, annotation, target_annotations, fixed_point):
        """Calculate a fixed head or tail point for an annotation.
 
        Uses the compiled calculation function from the annotation
        config to compute the fixed point from the current target's
        annotation coordinates.
 
        Parameters
        ----------
        annotation : str
            Annotation name.

        target_annotations : dict
            Current target's annotation coordinate arrays, keyed by
            annotation name.
            
        fixed_point : {"fixed_head", "fixed_tail"}
            Which fixed point to calculate.
 
        Returns
        -------
        ndarray, shape (1, 2) or None
            The computed fixed point, or ``None`` if the annotation
            has no fixed point for *which* or if the calculation
            fails.
        """
        # Validate the fixed point type.
        if fixed_point not in ("fixed_head", "fixed_tail"):
            raise ValueError(f"Invalid fixed point: {fixed_point}")

        # Get the fixed head or tail attribute for the given annotation.
        fixed_point = getattr(self.annot_cfg, fixed_point)[annotation]

        # If there is no fixed point, return None
        if fixed_point is None: return None

        # If there is a fixed head, attempt to calculate using the compiled function.
        try:
            fixed_point = fixed_point["calculate"](target_annotations)
            return fixed_point.reshape(1, 2)
        
        # If the calculation fails, we return None. 
        except Exception:
            return None

    # Editable Index Methods ---------------------------------------------------

    def _calc_editable(self, annotation):
        """Return indices of user editable points for *annotation*.
 
        Editable points are those that do not coincide with the
        fixed head or fixed tail.
 
        Parameters
        ----------
        annotation : str
            Annotation name.
 
        Returns
        -------
        ndarray of int
            Indices into ``self.annotations[annotation]`` that are
            not fixed.
        """
        # Get the points, fixed head, and fixed tail for the given annotation
        points     = self.annotations[annotation]
        fixed_head = self.fixed_heads[annotation]
        fixed_tail = self.fixed_tails[annotation]

        # Determine which points are fixed by comparing them to the fixed head and tail.
        is_head  = np.all(points == fixed_head, axis = 1)
        is_tail  = np.all(points == fixed_tail, axis = 1)
        is_fixed = np.logical_or(is_head, is_tail)

        # Return the indices of the editable points (i.e., non-fixed points).
        return np.where(~is_fixed)[0]

    # Update (target / annotation switch) --------------------------------------

    def update(self, target_id, annotation, target_annotations):
        """Update the editor to reflect a new target and/or annotation.
 
        Recalculates fixed heads and tails, resolves the editable
        indices, and positions the cursor.  For contour/boundary
        annotations with no user points, seeds the coordinate array
        with any configured fixed head and tail.
 
        Parameters
        ----------
        target_id : tuple of str
            The new target identifier.

        annotation : str
            The new active annotation name.

        target_annotations : dict of {str: ndarray or None}
            Mapping of annotation name to ``(N, 2)`` coordinate
            arrays for *target_id*.
 
        Returns
        -------
        bool or None
            ``True`` if the target changed — callers should reload
            base data (grid image, cortex mesh).  ``False`` if only
            the annotation changed — callers should redraw annotations
            only.  ``None`` if neither changed (no-op, no redraw
            needed).
        """    
        # If neither the target nor the annotation changed, we can skip the update.
        if self.target == target_id and self.active == annotation: return None
            
        # Store the previous state.
        prev_target     = self.target
        prev_annotation = self.active

        # Update the target, active annotation, and annotations.
        self.target      = target_id
        self.active      = annotation
        self.dependents  = self.annot_cfg.fixed_dependencies[annotation]
        self.annotations = target_annotations

        # Determine if the target changed.
        target_changed = prev_target != self.target
        
        # Determine which annotations need fixed point recalculation.
        # On first call (empty dicts) or target change, recalculate all.
        if self.fixed_heads == {} or self.fixed_tails == {} or target_changed:
            self.fixed_heads = {}
            self.fixed_tails = {}
            recalc_fixed     = list(self.annot_cfg.names)

        # If the annotation is changing, we need to recalculate the fixed heads
        # tails for dependencies of the previous annotation.
        else:
            prev_deps    = self.annot_cfg.fixed_dependencies[prev_annotation]
            recalc_fixed = { self.active, *prev_deps }
            
        # Recalculate the fixed head and tails of the given annotations.
        for annotation in recalc_fixed:
            self.fixed_heads[annotation] = self.calc_fixed_point(
                annotation, self.annotations, "fixed_head")
            self.fixed_tails[annotation] = self.calc_fixed_point(
                annotation, self.annotations, "fixed_tail")
            
        # Get the annotation coordinates type for the active annotation.
        points = self.annotations[self.active]
        atype  = self.annot_cfg.type[self.active]

        # If there are no points for the current annotation, initialize.
        if points is None or points.shape[0] == 0:
            points = self._EMPTY_POINT_MATRIX.copy()

        # Determine the editable points.
        if atype == "point":
            # Points annotations either have no point or exactly one point.
            if points.shape[0] == 0: self.editable = self._init_editable()
            else: self.editable = self._init_editable(0) # one point

        else: # atype in ( "contour", "boundary" )
            # If points is empty, update the annotations with the fixed points. 
            # Annotations are saved WITH their fixed heads and tails. 
            if points.shape[0] == 0:
                if self.fixed_heads[self.active] is not None:
                    points = np.vstack([self.fixed_heads[self.active], points])
                if self.fixed_tails[self.active] is not None:
                    points = np.vstack([points, self.fixed_tails[self.active]])
                    
                # Update the annotation with the fixed points.
                self.annotations[self.active] = points

            # Calculate the editable points (non-fixed points)
            self.editable = self._calc_editable(self.active)
    
        # If there are no editable points, we set the cursor to None.
        # Otherwise, we set the cursor to the last editable point.
        if self.editable.shape[0] == 0: self.cursor = None
        else: self.cursor = self.editable[-1]

        # Return whether the target changed (for redraw purposes).
        return target_changed
    
    # Recalculate Dependencies -------------------------------------------------

    def _recalculate_deps(self, annotation):
        """Recalculate fixed points for annotations that depend on *annotation*.
 
        Iterates over annotations whose fixed head or tail is derived
        from *annotation* and updates their first/last point in place.
 
        Parameters
        ----------
        annotation : str
            The annotation whose dependents should be recalculated.
        """
        # Get the dependent annotations for the given annotation.
        fixed_deps = self.annot_cfg.fixed_dependencies[annotation]

        # If there are no dependencies, we can skip.
        if len(fixed_deps) == 0: return 

        # We need to recalculate each of the dependent annotations using their
        # provided functions and update their annotation coordinates.
        for fd in fixed_deps: 
            # Get the current points for the dependent annotation.
            points = self.annotations[fd]

            # If there are no points, we can skip the recalculation.
            if points is None or points.shape[0] == 0: continue

            # Recalculate and update the fixed head for the dependent annotation.        
            fixed_head = self.calc_fixed_point(fd, self.annotations, "fixed_head")
            if fixed_head is not None: points[0,:] = fixed_head

            # Recalculate and update the fixed tail for the dependent annotation.        
            fixed_tail = self.calc_fixed_point(fd, self.annotations, "fixed_tail")
            if fixed_tail is not None: points[-1,:] = fixed_tail

            # Update the annotation with the new points.
            self.annotations[fd] = points

    # Push Point Method --------------------------------------------------------

    def push_point(self, new_point):
        """Add a point at the current cursor position.
 
        For point annotation types, replaces the single point.  For
        contour and boundary annotations, inserts relative to the
        cursor based on the ``insert`` flag:
 
        * ``insert = "after"`` (default): inserts **after** the
          cursor and advances the cursor to the new point.
          
        * ``insert = "before"``: inserts **before** the cursor,
          placing the new point at the cursor's current index.
 
        After insertion, recalculates fixed points for any dependent
        annotations.
 
        Parameters
        ----------
        new_point : ndarray, shape (1, 2)
            The new point in figure coordinates.
 
        Returns
        -------
        list of str
            Annotation names whose fixed points were recalculated as
            a result of this push.  Empty if the active annotation
            has no dependents, or if there is no active annotation.
        """
        # We can only push points if there is an active annotation.
        if self.active is None: return 

        # Get the current points for this annotation. If None, initialize empty.
        points = self.annotations[self.active]
        if points is None: points = self._EMPTY_POINT_MATRIX.copy()

        # Get the annotation type for this annotation.
        atype = self.annot_cfg.type[self.active]
        
        # Depending on the annotation type, we add the newest point to the
        # annotation in different ways.
        if atype == "point":
            # For a point annotation, replace the current point with the new point.
            points        = new_point
            self.editable = self._init_editable(0)
            self.cursor   = 0

        else: # atype in ( "contour", "boundary" )
            # If there are no points, we just add the new point.
            if points.shape[0] == 0:
                self.editable = self._init_editable(0)
                self.cursor   = 0

            # If there are no editable points, we add the new point to the head
            # or tail depending on which one is fixed.                
            elif self.editable.shape[0] == 0:
                if self.fixed_heads[self.active] is not None:
                    self.editable = self._init_editable(1)
                elif self.fixed_tails[self.active] is not None:
                    self.editable = self._init_editable(0)
                self.cursor = self.editable[0]   

            # If there are editable points, we add the new point dependoing 
            # on the insert direction.
            else: 
                if self.insert == "before":
                    # If we are inserting a point before the cursor, all
                    # editable points at or after the cursor shift up by one,
                    # but we also keep the original cursor position. This means
                    # we can just add an editable point index at the end of
                    # the editable points and keep the curor where it is.
                    self.editable = np.append(self.editable, np.max(self.editable) + 1)
                    
                else: # self.insert == "after"1
                    # If we are inserting a point after the cursor, all editable
                    # points after the cursor need to be shifted by one index.
                    self.editable[self.editable > self.cursor] += 1

                    # We add the new cursor position to the editable points.
                    self.editable = np.sort(np.append(self.editable, self.cursor + 1))
                    
                    # Finally, we increment the cursor to move it to the next position.
                    self.cursor += 1

            # Insert the new point at the cursor position.
            points = np.insert(points, self.cursor, new_point, axis = 0)
 
        # Update the annotation with the new points.
        self.annotations[self.active] = points

        # Update dependent annotations, if this active annotation has them.
        if len(self.dependents) > 0: self._recalculate_deps(self.active)

        # Return fixed dependencies
        return self.dependents

    # Toggle Cursor Method -----------------------------------------------------
    
    def toggle_cursor(self):
        """Cycle the cursor to the next editable point.
 
        For contour and boundary annotations, advances the cursor to
        the next editable index with wraparound.  No-op for point
        annotations (only one point) or when fewer than two editable
        points exist.
        """
        # Extract current annotation type.
        atype = self.annot_cfg.type[self.active]

        # For a point annotation, there is only one point. Toggling the 
        # cursor position does not do anything, so we can skip it.
        if atype == "point": return

        # If there are less than two editable points, we cannot toggle the cursor.
        if self.editable.shape[0] < 2: return

        # For contour or boundary annotations, we toggle the cursor position by 
        # moving it to the next editable point in the annotation.
        if atype in ( "contour", "boundary" ):
            # Get the index of the current cursor position in the editable points.
            current_index = np.where(self.editable == self.cursor)[0][0]

            # Calculate the index of the next editable point with wraparound.
            next_index = np.mod(current_index + 1, self.editable.shape[0])

            # Update the cursor to the next editable point.
            self.cursor = self.editable[next_index]

    # Pop Point Method ---------------------------------------------------------
    
    def pop_point(self):
        """Remove the point at the current cursor position.
 
        For point annotation types, clears the single point. For
        contour and boundary annotations, removes the point at the
        cursor and adjusts remaining editable indices.
 
        Deletion is blocked when the active annotation has live
        dependents and only one editable point remains where removing 
        the point would orphan the dependent annotations. In that case, 
        an error message is returned instead.
 
        After removal, recalculates fixed points for any dependent
        annotations.
 
        Returns
        -------
        tuple of (list of str, str or None)
            A two-element tuple ``(fixed_deps, error_msg)``.
 
            *fixed_deps* is the list of annotation names whose fixed
            points were recalculated.  Empty if nothing was deleted
            or no dependents exist.
 
            *error_msg* is a user-facing message when the deletion
            was blocked (e.g. live dependencies prevent removal), or
            ``None`` on success or when there is nothing to delete.
        """
        # We can only pop points if there is an active annotation.
        if self.active is None: return ([], None)

        # Get the current annotation and annotation type.
        points = self.annotations[self.active]
        atype  = self.annot_cfg.type[self.active]

        # If there are no points, we cannot delete anything. Skip.
        if points is None or points.shape[0] == 0 or \
            self.editable.shape[0] == 0: return ([], None)
        
        # Check if there are any LIVE dependencies on this annotation. If so, 
        # we cannot delete the last point of this annotation because the 
        # dependent annotations rely on it. 
        if len(self.dependents) > 0 and self.editable.shape[0] == 1:
            # Determine the number of fixed points for each dependent 
            # annotation. This number is the minimum number of points that the 
            # annotation must have be considered LIVE.
            n_fixed = [ len(self.annot_cfg.fixed_points[fd]) 
                        for fd in self.dependents ]

            live_deps = [
                fd for fd, n in zip(self.dependents, n_fixed) 
                if self.annotations[fd] is not None
                and self.annotations[fd].shape[0] > n
            ]
        
            # If there are live dependencies, we cannot delete the last point 
            # of the annotation, return error message.
            if live_deps:
                error_message = (
                    f"Cannot delete: '{self.active}'. It is required by "
                    f"'{', '.join(live_deps)}'. Clear those annotations "
                    f"first."
                )
                return ([], error_message)
        
        # If there are points, we delete based on annotation type.
        if atype == "point":
            # For a point annotation, we delete the single point.
            points        = self._EMPTY_POINT_MATRIX.copy()
            self.editable = self._init_editable()
            self.cursor   = None
            
        else: # atype in ( "contour", "boundary" )
            # If there are points to delete, delete at current position.
            points = np.delete(points, self.cursor, axis = 0)

            # Remove the current cursor from the editable points.
            self.editable = self.editable[self.editable != self.cursor]
            if self.editable.shape[0] == 0:
                self.cursor = None
            else:
                # Removing an index causes all the indices larger than the 
                # current position to shift down by one, so we need to decrement
                # the editable points.
                self.editable[self.editable > self.cursor] -= 1

                # When the cursor is at the head of the editable points, we do
                # not need to decrement the cursor because it will just move 
                # down with the shift of the points. However, if the cursor is
                # anywhere else, we need to decrement the cursor.
                if self.cursor != self.editable[0]: 
                    self.cursor -= 1

        # Update the annotation with the new points.
        self.annotations[self.active] = points

        # Update dependent annotations, if this active annotation has them.
        if len(self.dependents) > 0: self._recalculate_deps(self.active)

        # Return fixed dependencies
        return ( self.dependents, None )
