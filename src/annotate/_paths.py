# -*- coding: utf-8 -*-
################################################################################
# annotate/_paths.py

"""File path management for the cortex-annotate toolkit.

PathManager centralizes all path construction and directory creation for the
project's cache (figures, grids) and save (annotations, preferences)
directories. It also handles git repository username detection for per-user
save paths.
"""

# Imports ######################################################################

import os
import os.path as op
from warnings import warn

# Path Manager Class -----------------------------------------------------------

class PathManager:
    """Manages file paths for cached figures, grids, and saved annotations, preferences.

    Centralizes all path construction so that other modules never build 
    paths directly. Automatically creates directories as needed.

    Parameters
    ----------
    cache_path : str
        Root directory for cached figure and grid images.
    save_path : str
        Root directory for saved annotation TSV files.
    git_path : str or None
        Path to a git repository used to detect the username for 
        per-user save subdirectories. If None, no git detection is attempted.
    username : str or None
        Explicit username override. If None, detected from git.
    """
    
    def __init__(
            self, 
            cache_path  = "/cache",
            save_path   = "/save",
            git_path    = "/git",
            username    = None,
        ):
        # Store the arguments. 
        self.cache_path = cache_path
        self.save_path  = save_path
        self.git_path   = git_path
        self.username   = username

        # We add the git username to the save_path if needed here.
        if self.username is None:
            (self.username, _) = self.gitdata
        if not isinstance(self.username, str):
            raise RuntimeError("username must be a string or None.")

        # Update the save_path.
        if self.username != "": 
            # if username, add as subdirectory of save path
            self.save_path = op.join(save_path, self.username)
        
        # Make sure the cache and save directories exist.
        os.makedirs(self.cache_path, mode = 0o755, exist_ok = True)
        os.makedirs(self.save_path, mode = 0o755, exist_ok = True)
        
    # Git Methods ---------------------------------------------------------------

    @property
    def gitdata(self):
        """Detect the git repository username and repository name.

        Reads the git remote origin URL from the repository at ``git_path``
        and parses out the username and repo name. If git detection fails,
        returns empty strings and issues a warning.

        Returns
        -------
        tuple of (str, str)
            ``(username, repo_name)``. Both are ``""`` if detection fails
            or ``git_path`` is None.
        """
        # If we were not given a git path, we return standard nothings.
        if self.git_path is None: return ( "" , "" )
        try:
            # For some reason, it seems that sometimes docker does not fully
            # mount the directory until we've attempted to list its contents.
            with os.popen(f"ls {self.git_path}") as f: f.read()
            # Having performed an ls, go ahead and check git's opinion about the
            # origin with git config command line calls.
            cmd  = f"cd {self.git_path}"
            cmd += f" && git config --global --add safe.directory {self.git_path}"
            cmd +=  " && git config --get remote.origin.url"
            with os.popen(cmd) as p:
                repo_url = p.read().strip()
            repo_split = repo_url.split("/")
            repo_name = repo_split.pop()
            while repo_name == "":
                repo_name = repo_split.pop()
            repo_user = repo_split.pop()
            s1 = repo_user.split("/")[-1]
            s2 = repo_user.split(":")[-1]
            repo_user = s1 if len(s1) < len(s2) else s2
            return ( repo_user, repo_name )
        except Exception as e:
            # If there was an error, we just warn and return nothings.
            warn(f"Error finding gitdata: {e}")
            return ( "" , "" ) 
        
    # Path Methods -------------------------------------------------------------

    def _build_path(self, target_id, base_dir, subdir = None, filename = None):
        """Builds a path for the given target, base directory, subdirectory, and filename.

        Example 1: 
            - target_id: ("sub-01", "ses-02")
            - base_dir: "/cache"
            - subdir: "figures"
            - filename: "name.png"
        Result: "/cache/figures/sub-01/ses-02/name.png"

        Example 2:
            - target_id: ("sub-01", "ses-02")
            - base_dir: "/cache"
            - subdir: None
            - filename: "name.png"
        Result: "/cache/sub-01/ses-02/name.png"

        Example 3: 
            - target_id: ("sub-01", "ses-02")
            - base_dir: "/cache"
            - subdir: None
            - filename: None
        Result: "/cache/sub-01/ses-02"
        """
        path = op.join(*target_id) # target relative path
        if subdir is None: path = op.join(base_dir, path)
        else: path = op.join(base_dir, subdir, path) 
        # NOTE: We always create the directory here so callers don't need to
        # worry about whether the path exists yet. This means even "get" calls
        # have the side effect of creating directories.
        os.makedirs(path, mode = 0o755, exist_ok = True)
        if filename is not None: path = op.join(path, filename)
        return path


    def get_figure_path(self, target_id, figure = None):
        """Returns the cache path for a target's figures.

        If `figure` is None, returns the path to the target's figure directory.
        Otherwise, returns the path to the figure's png file.
        """
        return self._build_path(
            target_id = target_id, 
            base_dir  = self.cache_path, 
            subdir    = "figures", 
            filename  = f"{figure}.png" if figure is not None else None
        )


    def get_grid_path(self, target_id, grid = None):
        """Returns the cache path for a target's grids.

        If `grid` is None, returns the path to the target's grid directory.
        Otherwise, returns the path to the grid's png file.
        """
        return self._build_path(
            target_id = target_id, 
            base_dir  = self.cache_path, 
            subdir    = "grids", 
            filename  = f"{grid}.png" if grid is not None else None
        )


    def get_annotation_path(self, target_id, annotation = None):
        """Returns the save path for a target's annotations.

        If `annotation` is None, returns the path to the target's annotation 
        directory. Otherwise, returns the path to the annotation's tsv file.
        """
        return self._build_path(
            target_id = target_id, 
            base_dir  = self.save_path,
            filename  = f"{annotation}.tsv" if annotation is not None else None
        )


    def get_preferences_path(self, filename = ".annot-prefs.yaml"):
        """Returns the save path for the annotation tool's preferences."""
        return op.join(self.save_path, filename)