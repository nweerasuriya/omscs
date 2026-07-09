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

from MemoryDecorators import GRID_PRIMITIVES, OBJECT_PRIMITIVES
from ArcHeuristics import (
    ConservedAllSets,
    GridDifference,
    HeuristicSummary,
    ObjectTransformation,
)
from kwarg_engine import required_kwargs


class Require:
    SQUARE_GRID = "requires:square_grid"
    SPLIT_GRID = "requires:split_grid"
    REMOVE_COLOURS = "requires:remove_colours"
    DIRECTIONALITY = "requires:directionality"


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
        # # Remove some dsl if no mutations are present
        # if len(heuristic_summary.object_transformations) == 0:
        #     exclusion_tags.add(Require.DIRECTIONALITY)

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
        unpruned_primitives = []
        for primitive in primitives:
            primitive_tags = getattr(primitive, "tags", set())
            if primitive_tags.intersection(exclusion_tags):
                continue
            unpruned_primitives.append(primitive)
        return unpruned_primitives

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
        grid_diff: list[GridDifference] = heuristic_summary.grid_differences
        object_tran: list[list[ObjectTransformation]] = (
            heuristic_summary.object_transformations
        )

        relevance_mapping = {
            Effect.COLOUR: any(gd.colour_changed for gd in grid_diff)
            or any(obj.colour_changed for obj_list in object_tran for obj in obj_list),
            Effect.SHAPE: any(
                obj.shape_changed for obj_list in object_tran for obj in obj_list
            ),
            Effect.GRID_SIZE: any(
                gd.input_shape != gd.output_shape for gd in grid_diff
            ),
            Effect.OBJECT_COUNT: any(gd.object_count_changed for gd in grid_diff),
            Effect.SYMMETRY: any(gd.symmetry_changed for gd in grid_diff),
            Effect.GROWTH: any(
                obj.size_changed for obj_list in object_tran for obj in obj_list
            )
            or any(
                "growth" in obj.mutation_types
                for obj_list in object_tran
                for obj in obj_list
            ),
            Effect.SHRINK: any(
                obj.size_changed for obj_list in object_tran for obj in obj_list
            )
            or any(
                "shrink" in obj.mutation_types
                for obj_list in object_tran
                for obj in obj_list
            ),
            Require.SPLIT_GRID: heuristic_summary.split_grid is not None,
            Require.REMOVE_COLOURS: any(gd.removed_colours for gd in grid_diff),
            Require.DIRECTIONALITY: any(
                "directional" in obj.mutation_types
                for obj_list in object_tran
                for obj in obj_list
            ),
            Require.SQUARE_GRID: heuristic_summary.conserved_properties.all_square_grid,
        }

        UNKNOWN = 0.5
        UNTAGGED = 0.5
        BASE_WEIGHT = 0.1

        weights = {}
        for primitive in primitives:
            primitive_tags = getattr(primitive, "tags", set())
            if primitive_tags.intersection(excluded_tags):
                weights[primitive] = 0.0
                continue
            #weights[primitive] = 0.5
            if not primitive_tags:
                weights[primitive] = UNTAGGED
                continue
            score = 0.0
            for tag in primitive_tags:
                if tag in relevance_mapping:
                    score += 1.0 if relevance_mapping[tag] else BASE_WEIGHT
                else:
                    score += UNKNOWN
            weights[primitive] = max(score / len(primitive_tags), BASE_WEIGHT)
        return weights

    def generate_candidate_primitives(
        self, heuristic_summary: HeuristicSummary, primitives: list[Callable]
    ) -> tuple[list[Callable], dict[Callable, float]]:
        """
        Generate candidate primitives based on the heuristic summary and the weights.
        Return pruned list and weighted list of primitives.
        """
        weights = self.create_weights(heuristic_summary, primitives)
        unpruned_primitives = self.prune_primitives(
            heuristic_summary.conserved_properties, heuristic_summary, primitives
        )
        # remove pruned primitives from weights
        weights = {
            primitive: weight
            for primitive, weight in weights.items()
            if primitive in unpruned_primitives
        }
        return unpruned_primitives, weights
