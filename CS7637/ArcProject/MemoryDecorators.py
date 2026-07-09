"""
Decarator functions for memory management and caching in the ARC project.
"""

__date__ = "2026-07-09"
__author__ = "NedeeshaWeerasuriya"
__version__ = "0.1"

import numpy as np
import functools
from typing import Callable, Any

from ArcMemory import (
    ArcState,
    GridState,
    ObjectState,
)
from ArcRelations import RelationalGraph
from helpers import get_grid_splits, check_valid_grid

GRID_PRIMITIVES: list[callable] = []
OBJECT_PRIMITIVES: list[callable] = []
CELL_PRIMITIVES: list[callable] = []
SPLIT_GRID_PRIMITIVES: list[callable] = []
RELATIONAL_PRIMITIVES: list[callable] = []


def grid_primitive(func=None, *, tags=None):
    """
    Decorator to automatically transition an ArcState to an array and back.
    Add tags to help in pruning.
    """

    def decorator(inner_func):
        @functools.wraps(inner_func)
        def wrapper(state: ArcState, *args, **kwargs) -> ArcState:
            raw_grid = state.to_array()
            transformed_grid = inner_func(raw_grid, *args, **kwargs)
            if not check_valid_grid(transformed_grid):
                return state
            return ArcState.from_array(transformed_grid, extract_objects=True)

        wrapper.tags = frozenset(tags or ())
        GRID_PRIMITIVES.append(wrapper)
        return wrapper

    if func is None:
        return decorator
    return decorator(func)


def object_primitive(func=None, *, tags=None):
    """
    Decorator to apply a transformation directly to the object layer of an ArcState.
    """

    def decorator(inner_func):
        @functools.wraps(inner_func)
        def wrapper(state: ArcState, *args, **kwargs) -> ArcState:
            graph = None
            # Build graph if necessary
            if not any(hasattr(v, "needs_graph") for v in kwargs.values()):
                graph = RelationalGraph(state)
            transformed_objects = []
            for obj in state.objects:
                # Resolve kwargs for the current object
                resolved_kwargs = {
                    k: (
                        v.resolve_for_object(obj, graph)
                        if hasattr(v, "resolve_for_object")
                        else v
                    )
                    for k, v in kwargs.items()
                }
                # Check for None values in resolved_kwargs, if any are None, skip this object
                if any(v is None for v in resolved_kwargs.values()):
                    transformed_objects.append(obj)
                    continue
                tr_object = inner_func(obj, *args, **resolved_kwargs)
                if not tr_object.cell_positions:
                    continue
                transformed_objects.append(tr_object)

            # Update grid state based on transformed objects
            rows, cols = state.grid_state.dimensions
            grid = np.zeros((rows, cols), dtype=int)
            if state.grid_state.background_colour != 0:
                grid.fill(state.grid_state.background_colour)
            for obj in transformed_objects:
                for r, c in obj.cell_positions:
                    if 0 <= r < rows and 0 <= c < cols:
                        grid[r, c] = obj.colour

            new_grid_state = GridState.from_array(grid)
            if not check_valid_grid(grid):
                return state
            return ArcState(
                grid_state=new_grid_state, objects=tuple(transformed_objects)
            )

        wrapper.tags = frozenset(tags or ())
        OBJECT_PRIMITIVES.append(wrapper)
        return wrapper

    if func is None:
        return decorator
    return decorator(func)


def split_grid_primitive(func=None, *, tags=None):
    """
    Decorator for transformations that operate on split grids.
    """

    def decorator(inner_func):
        @functools.wraps(inner_func)
        def wrapper(state: ArcState, split_axis: str) -> ArcState:
            # Split the grid into halves or thirds
            components = get_grid_splits(state.to_array(), split_type=split_axis)
            if components is None:
                return state
            if not all(check_valid_grid(comp) for comp in components):
                return state
            return ArcState.from_array(inner_func(components), extract_objects=True)

        wrapper.tags = frozenset(tags or ())
        SPLIT_GRID_PRIMITIVES.append(wrapper)
        return wrapper

    if func is None:
        return decorator
    return decorator(func)


# def relation_primitive(func=None, *, tags=None):
#     """
#     Decorator for transformations that are based on relationships between objects or between objects and the grid.
#     """

#     def _resolve_object(val: Any, obj: "ObjectState") -> dict:
#         return val.find_kwarg(obj) if isinstance(val, PropertyKwarg) else val

#     def decorator(inner_func):
#         @functools.wraps(inner_func)
#         def wrapper(state: ArcState, *args, **kwargs) -> ArcState:
#             original_objects = state.objects
#             transformed_objects = []
#             for obj in original_objects:
#                 other_objects = [o for o in original_objects if o != obj]
#                 resolved_kwargs = {
#                     k: _resolve_object(v, obj) for k, v in kwargs.items()
#                 }
#                 # Check kwargs resolve, if not, skip this object
#                 if any(v is None for v in resolved_kwargs.values()):
#                     transformed_objects.append(obj)  # Keep the original object
#                     continue
#                 transformed_objects.append(
#                     inner_func(obj, other_objects, *args, **resolved_kwargs)
#                 )

#             # Update grid state based on transformed objects
#             rows, cols = state.grid_state.dimensions
#             grid = np.zeros((rows, cols), dtype=int)
#             if state.grid_state.background_colour != 0:
#                 grid.fill(state.grid_state.background_colour)
#             for obj in transformed_objects:
#                 for r, c in obj.cell_positions:
#                     if 0 <= r < rows and 0 <= c < cols:
#                         grid[r, c] = obj.colour

#             new_grid_state = GridState.from_array(grid)
#             if not check_valid_grid(grid):
#                 return state
#             return ArcState(
#                 grid_state=new_grid_state, objects=tuple(transformed_objects)
#             )

#         wrapper.tags = frozenset(tags or ())
#         RELATIONAL_PRIMITIVES.append(wrapper)
#         return wrapper

#     if func is None:
#         return decorator
#     return decorator(func)
