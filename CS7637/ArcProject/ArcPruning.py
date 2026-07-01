"""
Tag based pruning of DSL primitives

Two types of tags:
1. Requirement tags - for primtives requiring specific properties in the input data
2. Effect tags - The changes that the primitive makes to the input data

Pruning Engine will use the Heuristic Summary output of conserved properties to exclude primitives.
Implement weighting of primitives to prioritise primitives if more relevant.
"""

__date__ = "2026-06-19"
__author__ = "NedeeshaWeerasuriya"
__version__ = "0.1"


import numpy as np
from typing import Callable

from ArcMemory import GRID_PRIMITIVES, OBJECT_PRIMITIVES, GridState
from ArcHeuristics import (
    ConservedAllSets,
    GridDifference,
    HeuristicSummary,
    MutatatedObject,
    ObjectDifference,
)


class Require:
    SQUARE_GRID = "requires:square_grid"
    SPLIT_GRID = "requires:split_grid"
    REMOVE_COLOURS = "requires:remove_colours"


class Effect:
    COLOUR = "effect:colour"
    SHAPE = "effect:shape"
    GRID_SIZE = "effect:grid_size"
    OBJECT_COUNT = "effect:object_count"
    SYMMETRY = "effect:symmetry"
    GROWTH = "effect:growth"
    SHRINK = "effect:shrink"
    TRANSLATE = "effect:translate"


class PruningEngine:
    """
    Ingests heuristic summary and prunes primitives based on the tags and the conserved properties.

    Exclusion list (weight = 0): primitives which are irrelevant based on the conserved properties
    Inclusion list (weight > 0, < 1): primitives which are weighted based on relevance

    Heuristic summary includes:
    1. Conserved properties across all sets
    2. Grid differences across all sets
    3. Object differences across all sets
    4. Mutations present across all sets
    """

    def _get_exclusion_tags(
        self,
        conserved_properties: ConservedAllSets,
        heuristic_summary: HeuristicSummary,
    ) -> set[str]:
        """
        Get the exclusion tags based on the conserved properties.
        """
        exclusion_tags = set()
        if conserved_properties.grid_size:
            exclusion_tags.add(Effect.GRID_SIZE)
        if conserved_properties.colours:
            exclusion_tags.add(Effect.COLOUR)
        if conserved_properties.object_count:
            exclusion_tags.add(Effect.OBJECT_COUNT)
        # if conserved_properties.object_colours:
        #     exclusion_tags.add(Effect.COLOUR)
        if conserved_properties.object_shapes:
            exclusion_tags.add(Effect.SHAPE)
        # Requirements
        if not conserved_properties.all_square_grid:
            exclusion_tags.add(Require.SQUARE_GRID)
        if not heuristic_summary.split_grid:
            exclusion_tags.add(Require.SPLIT_GRID)
        gd = heuristic_summary.grid_differences
        if not any(grid_diff.removed_colours for grid_diff in gd):
            exclusion_tags.add(Require.REMOVE_COLOURS)
        return exclusion_tags

    def prune_primitives(
        self,
        conserved_properties: ConservedAllSets,
        heuristic_summary: HeuristicSummary,
        primitives: list[Callable],
    ) -> list[Callable]:
        """
        Hard prune primitives based on the exclusion tags.
        """
        exclusion_tags = self._get_exclusion_tags(
            conserved_properties, heuristic_summary
        )
        pruned_primitives = []
        for primitive in primitives:
            primitive_tags = getattr(primitive, "tags", set())
            if primitive_tags.intersection(exclusion_tags):
                pruned_primitives.append(primitive)
        return pruned_primitives

    # TODO: Improve weighting system
    def create_weights(
        self, heuristic_summary: HeuristicSummary, primitives: list[Callable]
    ) -> dict[Callable, float]:
        """
        Create weights for primitives to provide prioritisation based on relevance.
        A weight of 0 means hard pruned
        A weight of 1 means top of the priority list
        """
        excluded_tags = self._get_exclusion_tags(
            heuristic_summary.conserved_properties, heuristic_summary
        )
        grid_diff: GridDifference = heuristic_summary.grid_differences
        object_diff: list[list[ObjectDifference]] = heuristic_summary.object_differences
        mutations: list[list[MutatatedObject]] = heuristic_summary.mutations

        relevance_mapping = {
            Effect.COLOUR: any(
                grid_diff.colour_changed
                for grid_diff in heuristic_summary.grid_differences
            )
            or any(obj.colour_changed for obj_list in object_diff for obj in obj_list),
            Effect.SHAPE: any(
                obj.shape_changed for obj_list in object_diff for obj in obj_list
            ),
            Effect.GRID_SIZE: any(
                grid_diff.input_shape != grid_diff.output_shape
                for grid_diff in heuristic_summary.grid_differences
            ),
            Effect.OBJECT_COUNT: any(
                grid_diff.object_count_changed
                for grid_diff in heuristic_summary.grid_differences
            ),
            Effect.SYMMETRY: any(
                grid_diff.symmetry_changed
                for grid_diff in heuristic_summary.grid_differences
            ),
            Effect.GROWTH: any(
                obj.size_changed for obj_list in object_diff for obj in obj_list
            )
            or any(
                "growth" in mutation.mutation_types
                for mut_list in mutations
                for mutation in mut_list
                if mutation
            ),
            Effect.SHRINK: any(
                obj.size_changed for obj_list in object_diff for obj in obj_list
            )
            or any(
                "shrink" in mutation.mutation_types
                for mut_list in mutations
                for mutation in mut_list
                if mutation
            ),
            Require.SPLIT_GRID: heuristic_summary.split_grid is not None,
        }
        weights = {}
        for primitive in primitives:
            primitive_tags = getattr(primitive, "tags", set())
            if primitive_tags.intersection(excluded_tags):
                weights[primitive] = 0
            else:
                relevance_score = sum(
                    relevance_mapping.get(tag, 0.5) for tag in primitive_tags
                )
                if relevance_score > 0:
                    relevance_score = relevance_score / len(primitive_tags)

                weights[primitive] = relevance_score
        return weights

    def generate_candidate_primitives(
        self, heuristic_summary: HeuristicSummary, primitives: list[Callable]
    ) -> tuple[list[Callable], dict[Callable, float]]:
        """
        Generate candidate primitives based on the heuristic summary and the weights.
        Return pruned list and weighted list of primitives.
        """
        weights = self.create_weights(heuristic_summary, primitives)
        pruned_primitives = self.prune_primitives(
            heuristic_summary.conserved_properties, heuristic_summary, primitives
        )
        # remove pruned primitives from weights
        weights = {
            primitive: weight
            for primitive, weight in weights.items()
            if primitive not in pruned_primitives
        }
        return pruned_primitives, weights
