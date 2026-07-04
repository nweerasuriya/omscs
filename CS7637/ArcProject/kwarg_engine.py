"""
Keyword Argument Engine
"""

__date__ = "2026-06-12"
__author__ = "NedeeshaWeerasuriya"
__version__ = "0.1"

import inspect
import itertools
from typing import Callable, Iterator
import numpy as np
from ArcMemory import ArcState, PropertyKwarg
from ArcHeuristics import HeuristicSummary

ALL_DIRECTIONS = {(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)}
COLOUR_SET = {0, 1, 2, 3, 4, 5, 6, 7, 8}


def build_kwarg_pool(
    state_cache: dict[int, dict[str, ArcState]], hs: HeuristicSummary
) -> dict[str, list]:
    """
    Get all possible parameter values from the Heuristic Summary which will be used by the DSL primitives.
    """
    pool: dict[str, set] = {}

    # Grid related parameters
    for gd in hs.grid_differences:
        in_state = state_cache[gd.set_id]["input"]
        out_state = state_cache[gd.set_id]["output"]
        # Colour related parameters
        if gd.colour_changed:
            pool.setdefault("in_colour", set()).update([c for c in gd.input_colours])
            pool.setdefault("out_colour", set()).update([c for c in gd.output_colours])
            pool.setdefault("new_colours", set()).update([c for c in gd.new_colours])
            pool.setdefault("removed_colours", set()).update(
                [c for c in gd.removed_colours]
            )
            if gd.new_colours:
                pool.setdefault("new_colour_set", set()).add(tuple(gd.new_colours))
            if gd.removed_colours:
                pool.setdefault("removed_colour_set", set()).add(
                    tuple(gd.removed_colours)
                )

        pool.setdefault("in_shape", set()).add(in_state.grid_state.dimensions)
        pool.setdefault("out_shape", set()).add(out_state.grid_state.dimensions)

    # Object related parameters
    for obj_diff_list in hs.object_differences:
        for od in obj_diff_list:
            in_state = state_cache[od.set_id]["input"]
            in_id, out_id = od.object_id
            in_obj = in_state.objects[in_id]
            out_state = state_cache[od.set_id]["output"]
            out_obj = out_state.objects[out_id]

            # Shape related parameters
            pool.setdefault("obj_in_shape", set()).add(in_obj.area)
            pool.setdefault("obj_out_shape", set()).add(out_obj.area)

            pool.setdefault("in_centroid", set()).add(in_obj.centroid)
            pool.setdefault("out_centroid", set()).add(out_obj.centroid)
            pool.setdefault("in_bbox", set()).add(in_obj.bounding_box)
            pool.setdefault("out_bbox", set()).add(out_obj.bounding_box)

            # Colour related parameters
            pool.setdefault("obj_in_colour", set()).add(in_obj.colour)
            pool.setdefault("obj_out_colour", set()).add(out_obj.colour)
            pool.setdefault("obj_colours", set()).add((in_obj.colour, out_obj.colour))

            # Mutation related parameters
            pool.setdefault("mutation_types", set()).update(in_obj.mutation_types)
            pool.setdefault("mutation_vectors", set()).update(in_obj.mutation_vectors)

    # Mutation related parameters
    for mut_list in hs.mutations:
        for mut in mut_list:
            pool.setdefault("direction_vector", set()).add(mut.direction_vector)
            pool.setdefault("scale", set()).add(mut.scale)

    # TODO: Check if direction_vector is associated with other object properties
    # (for example all objects with a certain colour are moving in the same direction)
    for prop_assoc in hs.property_associations:
        mapping_dict = {k: next(iter(v)) for k, v in prop_assoc.mapping.items()}
        property_kwarg = PropertyKwarg(
            object_property=prop_assoc.object_property,
            mapping=tuple(mapping_dict.items()),
            support_score=prop_assoc.support_score,
        )
        pool.setdefault("property_associations", set()).add(property_kwarg)

    # Split related parameters
    if hs.split_grid:
        split_map = {direction for direction, active in hs.split_grid.items() if active}
        # For now keep only a single string value
        if split_map:
            pool.setdefault("split_axis", set()).update(split_map)
    return pool


def iter_relevant_kwargs(
    func: Callable, kwarg_pool: dict[str, list]
) -> Iterator[dict[str, any]]:
    """
    Check which parameters are relevant for a given function.
    Iterate a list of possible parameter combinations from the pool.
    """
    wrapper_params = inspect.signature(func, follow_wrapped=False).parameters
    inner_params = inspect.signature(func, follow_wrapped=True).parameters
    all_params = {**wrapper_params, **inner_params}

    relevant_kwargs = [k for k in all_params if k in kwarg_pool]

    if not relevant_kwargs:
        yield {}
        return

    kwarg_values = [kwarg_pool[k] for k in relevant_kwargs]
    for combination in itertools.product(*kwarg_values):
        kwarg_comb = dict(zip(relevant_kwargs, combination))
        for kwarg, val in kwarg_comb.items():
            if kwarg.endswith("_set") and isinstance(val, tuple):
                kwarg_comb[kwarg] = set(val)
        yield kwarg_comb


def check_relevant_kwargs(func: callable, kwarg_pool: dict[str, list]) -> list[dict]:
    """
    Check which parameters are relevant for a given function and return a list of possible parameter combinations from the pool.
    """
    wrapper_params = inspect.signature(func, follow_wrapped=False)
    inner_params = inspect.signature(func, follow_wrapped=True)

    relevant_kwargs = {
        k: kwarg_pool[k] for k in inner_params.parameters if k in kwarg_pool
    }
    relevant_kwargs.update(
        {k: kwarg_pool[k] for k in wrapper_params.parameters if k in kwarg_pool}
    )
    # sig = inspect.signature(func)
    # relevant_kwargs = {k: kwarg_pool[k] for k in sig.parameters if k in kwarg_pool}

    return relevant_kwargs
