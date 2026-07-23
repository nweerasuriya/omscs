"""
Decarator functions for memory management and caching
"""

__date__ = "2026-07-09"
__author__ = "NedeeshaWeerasuriya"
__version__ = "0.1"

import numpy as np
import functools
import inspect
from typing import Callable, Any

from ArcMemory import (
    ArcState,
    GridState,
    ObjectState,
)
from ArcRelations import RelationalGraph
from helpers import get_grid_splits, check_valid_grid

PRIMITIVE_TYPES = ("grid", "object", "split_grid", "state")
WRAPPER_REGISTRY: dict[str, list[Callable]] = {
    prim_type: [] for prim_type in PRIMITIVE_TYPES
}

GRID_WRAPPER = WRAPPER_REGISTRY["grid"]
OBJECT_WRAPPER = WRAPPER_REGISTRY["object"]
SPLIT_GRID_WRAPPER = WRAPPER_REGISTRY["split_grid"]
STATE_WRAPPER = WRAPPER_REGISTRY["state"]


def all_primitives() -> list[Callable]:
    return [p for p_type in PRIMITIVE_TYPES for p in WRAPPER_REGISTRY[p_type]]


def _inspect_kwargs(inner_func: Callable, primitive_type: str):
    """
    Get the required keyword arguments for a transformation function based on its primitive type
    """
    params = list(inspect.signature(inner_func).parameters.values())
    param_names = []
    required_kwargs = []
    for param in params[1:]:
        param_names.append(param.name)
        if param.default is inspect.Parameter.empty:
            required_kwargs.append(param.name)

    # For the split_grid primitive, add the split_axis argument
    if primitive_type == "split_grid":
        param_names = ["split_axis"] + param_names
        required_kwargs = ["split_axis"] + required_kwargs
    return tuple(param_names), frozenset(required_kwargs)


def decorate_primitive(primitive_type: str, tags=None):
    """
    Decorator to register a function as a primitive of a given type.
    """
    if primitive_type not in PRIMITIVE_TYPES:
        raise ValueError(f"Invalid primitive type: {primitive_type}")

    def decorator(func):
        wrapper_builder = WRAPPER_BUILDER[primitive_type]
        wrapper = wrapper_builder(func)
        wrapper.tags = frozenset(tags or ())
        wrapper.primitive_type = primitive_type
        # Get the required keyword arguments for the function
        param_names, required_kwargs = _inspect_kwargs(func, primitive_type)
        wrapper.param_names = param_names
        wrapper.required_kwargs = required_kwargs

        # Register transformations
        WRAPPER_REGISTRY[primitive_type].append(wrapper)
        return wrapper

    return decorator


# -----------------------------------------------------------------------------
# Wrapper functions
# -----------------------------------------------------------------------------


def _validate_array(result, state: ArcState) -> ArcState:
    """
    Validate a resulting array and convert it back to an ArcState if valid.
    """
    if result is None or not check_valid_grid(np.asarray(result)):
        return state
    return ArcState.from_array(result, extract_objects=True)


def _build_grid_wrapper(inner_func: Callable):
    """
    Wrapper to automatically transition an ArcState to an array and back.
    Add tags to help in pruning.
    """

    @functools.wraps(inner_func)
    def wrapper(state: ArcState, *args, **kwargs) -> ArcState:
        raw_grid = state.to_array()
        transformed_grid = inner_func(raw_grid, *args, **kwargs)
        return _validate_array(transformed_grid, state)

    return wrapper


def _build_object_wrapper(inner_func: Callable):
    """
    Wrapper to apply a transformation directly to the object layer of an ArcState.
    """

    @functools.wraps(inner_func)
    def wrapper(state: ArcState, object_selector=None, *args, **kwargs) -> ArcState:
        graph = None
        # Build graph if necessary
        needs_graph = (
            object_selector is not None
            and getattr(object_selector, "needs_graph", False)
        ) or any(hasattr(v, "needs_graph") for v in kwargs.values())
        graph = RelationalGraph(state) if needs_graph else None

        selected_objects = (
            object_selector.select_objects(graph) if object_selector else None
        )

        transformed_objects = []
        for obj in state.objects:
            # Skip objects that are not selected
            if selected_objects is not None and obj not in selected_objects:
                transformed_objects.append(obj)
                continue

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
            # Check for if transformation returned a list of objects
            if isinstance(tr_object, list):
                transformed_objects.extend(tr_object)
            else:
                if not tr_object.cell_positions:
                    continue
                transformed_objects.append(tr_object)
        # Sort by priority to ensure consistent ordering when recreating the grid
        transformed_objects.sort(key=lambda o: o.priority, reverse=False)

        # Update grid state based on transformed objects
        rows, cols = state.grid_state.dimensions
        # drop any pixels that are out of bounds
        in_objects = []
        for obj in transformed_objects:
            inside_bounds = frozenset(
                (r, c) for r, c in obj.cell_positions if 0 <= r < rows and 0 <= c < cols
            )
            if not inside_bounds:
                continue
            in_objects.append(
                obj
                if inside_bounds == obj.cell_positions
                else obj.with_pixels(inside_bounds)
            )
        transformed_objects = in_objects

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

        return ArcState(grid_state=new_grid_state, objects=tuple(transformed_objects))

    return wrapper


def _build_split_grid_wrapper(inner_func: Callable):
    """
    Wrapper for transformations that operate on split grids.
    """

    @functools.wraps(inner_func)
    def wrapper(state: ArcState, split_axis: str) -> ArcState:
        # Split the grid into halves or thirds
        components = get_grid_splits(state.to_array(), split_type=split_axis)
        if components is None:
            return state
        if not all(check_valid_grid(comp) for comp in components):
            return state
        return _validate_array(inner_func(components), state)

    return wrapper


def _build_state_wrapper(inner_func: Callable):
    """
    Wrapper for transformations that need object-level data (via RelationalGraph)
    but produce a whole-grid result.
    """

    @functools.wraps(inner_func)
    def wrapper(state: ArcState, *args, **kwargs) -> ArcState:
        graph = RelationalGraph(state)
        resolved_list = []
        # resolve kwargs that might depend on the graph
        for obj in state.objects:
            resolved_list.append(
                {
                    k: (
                        v.resolve_for_object(obj, graph)
                        if hasattr(v, "resolve_for_object")
                        else v
                    )
                    for k, v in kwargs.items()
                }
            )
        # For multiple objects, get the largest value for each kwarg across all objects
        resolved_kwargs = {}
        for k in kwargs.keys():
            values = [res[k] for res in resolved_list if res[k] is not None]
            if values:
                resolved_kwargs[k] = max(
                    values, key=lambda x: x if isinstance(x, (int, float)) else 0
                )
            else:
                resolved_kwargs[k] = None
        result = inner_func(state, *args, **resolved_kwargs)
        if isinstance(result, ArcState):
            return result
        return _validate_array(result, state)

    return wrapper


WRAPPER_BUILDER = {
    "grid": _build_grid_wrapper,
    "object": _build_object_wrapper,
    "split_grid": _build_split_grid_wrapper,
    "state": _build_state_wrapper,
}
