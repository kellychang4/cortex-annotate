# -*- coding: utf-8 -*-
################################################################################
# annotate/_paths.py
# 
# DOCSTRING

# Imports ######################################################################

import os
import os.path as op
from warnings import warn

# Path Manager Class -----------------------------------------------------------

class PathsManager:
    def __init__(
            self, 
            cache_path  = "/cache",
            save_path   = "/save",
            git_path    = "/git",
            username    = None,
        ):
    
        # Store the paths. 
        self.cache_path = cache_path
        self.save_path  = save_path
        self.git_path   = git_path
        self.username   = username

        # We add the git username to the save path if needed here.
        if username is None:
            (username, git_reponame) = self.gitdata
        if not isinstance(username, str):
            raise RuntimeError("username must be a string or None")

        # Build up the save path.
        self.username = username
        if username != "": # if username, add as subdirectory of save path
            self.save_path = op.join(save_path, username)
        os.makedirs(self.save_path, mode = 0o755, exist_ok = True)


   # Git Directory Methods -----------------------------------------------------

    @property
    def gitdata(self):
        """Reads and returns the repo username and the repo name."""
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
        
    # Target Path Methods ------------------------------------------------------

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
        path = op.join(*target_id)
        if subdir is None: path = op.join(base_dir, path)
        else: path = op.join(base_dir, subdir, path) 
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
        """Returns the cache path for a target's grids."""
        return self._build_path(
            target_id = target_id, 
            base_dir  = self.cache_path, 
            subdir    = "grids", 
            filename  = f"{grid}.png" if grid is not None else None
        )


    def get_annotation_path(self, target_id, annotation = None):
        """Returns the save path for a target's annotations data."""
        return self._build_path(
            target_id = target_id, 
            base_dir  = self.save_path,
            filename  = f"{annotation}.tsv" if annotation is not None else None
        )

