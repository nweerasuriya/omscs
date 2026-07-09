"""
Relationship Layer

Includes:
- pairwise object relationships (distance, direction, containment)
- object relationship to it's grid (enclosed pixels, neighbours)
- global variable counts (num of colours, shapes etc.)

Build relations as a relationship graph linked to individual arc states
"""

__date__ = "2026-07-08"
__author__ = "NedeeshaWeerasuriya"
__version__ = "0.1"


from collections import Counter
import numpy as np
from typing import Any, Optional
from dataclasses import dataclass
from scipy.spatial import distance
from ArcMemory import GridState, ObjectState, ArcState, PixelSet
from helpers import (
    ALL_DIRECTIONS,
    COLOUR_SET,
    calculate_distance,
    get_direction_vector,
    exact_translation,
)

dir_coord = tuple[int, int]


SPATIAL_TARGETS = (
    "nearest",
    "nearest_same_colour",
    "container",
)

KWARG_SPACE: dict[str, tuple[tuple[str, str], ...]] = {
    "direction_vector": tuple(("direction_to", t) for t in SPATIAL_TARGETS),
    "scale": tuple(("steps_away_from", t) for t in SPATIAL_TARGETS),
}


@dataclass(frozen=True)
class ObjectPairs:
    """
    Relationships between two objects in an ArcState
    """

    obj1_i: int
    obj2_j: int

    distance: int
    direction: dir_coord

    same_colour: bool
    same_size: bool
    # containment: bool   #TODO: implement containment check
    row_overlap: bool
    col_overlap: bool
    aligned_vertical: bool  # share all x-coordinates
    aligned_horizontal: bool  # share all y-coordinates


class RelationalGraph:
    """
    Semantic Network like representation of the relationships for a given ArcState.
    """

    def __init__(self, state: ArcState):
        self.state = state
        self.grid = state.grid_state.grid
        self.background_colour = state.grid_state.background_colour
        objects = [
            obj for obj in state.objects if obj.cell_positions
        ]  # Filter out objects with no cell positions

        self.object_count = len(objects)
        self.colour_counts = Counter(obj.colour for obj in objects)

        self.pairs: dict[tuple[int, int], ObjectPairs] = {}
        for i in range(self.object_count):
            for j in range(self.object_count):
                if i == j:
                    continue
                self.pairs[(i, j)] = self._create_object_pair(i, j)

    def _lookup_obj(self, obj_id: int) -> ObjectState:
        return self.state.objects[obj_id]

    def _lookup_others(self, obj_id: int) -> list[int]:
        return [i for i in range(self.object_count) if i != obj_id]

    def _create_object_pair(self, i: int, j: int) -> ObjectPairs:
        obj1, obj2 = self.state.objects[i], self.state.objects[j]

        distance_calc = calculate_distance(obj1.cell_positions, obj2.cell_positions)

        bbox_1, bbox_2 = obj1.bounding_box, obj2.bounding_box
        row_overlap = (bbox_1[0] <= bbox_2[1]) and (bbox_2[0] <= bbox_1[1])
        col_overlap = (bbox_1[2] <= bbox_2[3]) and (bbox_2[2] <= bbox_1[3])

        aligned_vertical = obj1.x_coords == obj2.x_coords
        aligned_horizontal = obj1.y_coords == obj2.y_coords

        return ObjectPairs(
            obj1_i=i,
            obj2_j=j,
            distance=distance_calc,
            direction=get_direction_vector(obj1.centroid, obj2.centroid),
            same_colour=obj1.colour == obj2.colour,
            same_size=obj1.size == obj2.size,
            row_overlap=row_overlap,
            col_overlap=col_overlap,
            aligned_vertical=aligned_vertical,
            aligned_horizontal=aligned_horizontal,
        )

    def select_target_object(self, obj_id: int, reason: str) -> Optional[int]:
        """
        For a specific object, select a target object based on a relationship reason.
        """
        candidates = self._lookup_others(obj_id)
        if not candidates:
            return None

        # For nearest object get min distance of all pairs
        if reason == "nearest":
            nearest_obj = min(
                candidates, key=lambda other_id: self.pairs[(obj_id, other_id)].distance
            )
            return nearest_obj
        if reason == "nearest_same_colour":
            same_colour_candidates = [
                other_id
                for other_id in candidates
                if self.pairs[(obj_id, other_id)].same_colour
            ]
            if not same_colour_candidates:
                return None
            nearest_obj = min(
                same_colour_candidates,
                key=lambda other_id: self.pairs[(obj_id, other_id)].distance,
            )
            return nearest_obj
        if reason == "container":
            pass
        return None

    def move_towards_target(
        self, obj_id: int, target_reason=str
    ) -> Optional[tuple[dir_coord, int]]:
        """
        For a specific object, move it towards a target object based on the direction vector.
        Return the direction vector and the distance to move.
        """
        target_id = self.select_target_object(obj_id, target_reason)
        if target_id is None:
            return None
        pair = self.pairs[(obj_id, target_id)]
        return pair.direction, pair.distance


@dataclass(frozen=True)
class RelationalKwarg:
    """
    Uses the relational graph to derive kwarg values based on relationships between objects.
    Produces a support score to be plugged as a prior into the search engine.
    Can be used to query speciic relationships between objects and the associated kwarg values.
    """

    query: str
    target: str
    default_value: Any = None
    support_score: float = 0.0

    need_graph: bool = True

    def resolve_for_object(self, obj: ObjectState, graph: RelationalGraph) -> Any:
        index = graph.state.objects.index(obj)
        pass


def add_relational_kwargs(pool: dict[str, set], relational_scores: dict = None) -> None:
    """
    Add relational kwargs to the kwarg pool based on the collected relational scores.
    """
    support_scores = relational_scores if relational_scores is not None else {}
    for kwarg_name, item in KWARG_SPACE.items():
        relation_pool = pool.setdefault(kwarg_name, set())
        for query, target in item:
            support_score = support_scores.get((query, target), 0.0)
            relation_pool.add(
                RelationalKwarg(query=query, target=target, support_score=support_score)
            )


def extract_translation_relations(
    input_graph: RelationalGraph,
    out_state: ArcState,
    obj_in: int,
    obj_out: int,
) -> list[dict[str, RelationalKwarg]]:
    """
    Check for translation relationships between input and output objects based on their relational graphs.
    Return list of relational kwargs for the transformations.
    """
    translation_shift = exact_translation(
        input_graph.state.objects[obj_in].cell_positions,
        out_state.objects[obj_out].cell_positions,
    )
    if translation_shift is None or translation_shift == (0, 0):
        return []

    kwarg_list = []
    for target in SPATIAL_TARGETS:
        moveset = input_graph.move_towards_target(obj_in, target)
        if moveset is None:
            continue
        direction, distance = moveset
        if (direction[0], direction[1]) == translation_shift:
            kwarg_list.append(
                {
                    "direction_vector": RelationalKwarg(
                        query="direction_to",
                        default_value=target,
                        support_score=1.0,
                    ),
                    "scale": RelationalKwarg(
                        query="steps_away_from",
                        default_value=target,
                        support_score=1.0,
                    ),
                }
            )
    return kwarg_list


def collect_relation_scores(
    state_cache: dict, obj_transformations: list[list[Any]]
) -> dict:
    """
    Collect relationship scores for each transformation based on the relational graph.
    """
    support_scores = {}
    for obj_list in obj_transformations:
        matching_set = {}
        for obj_t in obj_list:
            cache_entry = state_cache.get(obj_t.set_id)
            if not cache_entry or "input_graph" not in cache_entry:
                continue
            # Get input and output matched
            matching_set.setdefault(obj_t.set_id, []).append(
                (obj_t.input_object_id, obj_t.output_object_id)
            )
            # Detect translation based changes
            translation_kwargs = extract_translation_relations(
                cache_entry["input_graph"],
                cache_entry["output"],
                obj_t.input_object_id,
                obj_t.output_object_id,
            )
            # Add support scores for each relational kwarg
            for rel_kwarg in translation_kwargs:
                for k, rel_k in rel_kwarg.items():
                    support_scores[(k, rel_k.query, rel_k.target)] += 1

    return support_scores
