# -*- coding: utf-8 -*-
################################################################################
# annotate/_paths.py
# 
# DOCSTRING

# Imports ######################################################################

import os
import os.path as op

# Path Manager Class -----------------------------------------------------------

class PathManager():
    def __init__(self):
        pass
        
    def _target_path(self, target):
        """Returns the relative path for a target."""
        if isinstance(target, tuple):
            path = target
        else:
            path = [target[k] for k in self.config.targets.concrete_keys]
        return op.join(*path)
    
    
    def target_figure_path(self, target, figure = None, ensure = True):
        """Returns the cache path for a target's figures."""
        path = self._target_path(target)
        path = op.join(self.cache_path, "figures", path)
        if ensure and not op.isdir(path):
            os.makedirs(path, mode = 0o755)
        if figure is not None:
            path = op.join(path, f"{figure}.png")
        return path
    
    
    def target_grid_path(self, target, annotation = None, ensure = True):
        """Returns the cache path for a target's grids."""
        path = self._target_path(target)
        path = op.join(self.cache_path, "grids", path)
        if ensure and not op.isdir(path):
            os.makedirs(path, mode = 0o755)
        if annotation is not None:
            path = op.join(path, f"{annotation}.png")
        return path
    
    
    def target_save_path(self, target, annotation = None, ensure = True):
        """Returns the save path for a target's annotation data."""
        path = self._target_path(target)
        path = op.join(self.save_path, path)
        if ensure and not op.isdir(path):
            os.makedirs(path, mode = 0o755)
        if annotation is not None:
            path = op.join(path, f"{annotation}.tsv")
        return path