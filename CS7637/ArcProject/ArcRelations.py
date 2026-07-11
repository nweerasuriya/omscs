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
from scipy import ndimage
from ArcMemory import GridState, ObjectState, ArcState, PixelSet
from helpers import (
    ALL_DIRECTIONS,
    COLOUR_SET,
    calculate_distance,
    get_direction_vector,
    exact_translation,
    pixels_to_mask,
    mask_to_pixels,
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
    "colour_count": (("colour_count_match", "any"),),
}

RELATION_FIELDS = (
    "touching",
    "row_overlap",
    "col_overlap",
    "aligned_vertical",
    "aligned_horizontal",
    "same_colour",
    "same_size",
)


def _resolve_relation_fields(pair: "ObjectPairs", relation_str: str) -> bool:
    """
    Given a pair of objects and a relation string, return the boolean value of that relation.
    """
    if relation_str == "touching":
        return pair.distance <= 1
    return bool(getattr(pair, relation_str))


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
        self.filtered_objects = [
            obj for obj in state.objects if obj.cell_positions
        ]  # Filter out objects with no cell positions

        self.object_count = len(self.filtered_objects)
        self.colour_counts = Counter(obj.colour for obj in self.filtered_objects)
        self.interior_pixels = [
            self._get_interior_pixels(obj) for obj in self.filtered_objects
        ]

        self.pairs: dict[tuple[int, int], ObjectPairs] = {}
        for i in range(self.object_count):
            for j in range(self.object_count):
                if i == j:
                    continue
                self.pairs[(i, j)] = self._create_object_pair(i, j)

    def _lookup_obj(self, obj_id: int) -> ObjectState:
        return self.filtered_objects[obj_id]

    def _lookup_others(self, obj_id: int) -> list[int]:
        return [i for i in range(self.object_count) if i != obj_id]

    def get_index(self, obj: ObjectState) -> Optional[int]:
        """
        Get the index of an object in the state.
        """
        return (
            self.filtered_objects.index(obj) if obj in self.filtered_objects else None
        )

    def _create_object_pair(self, i: int, j: int) -> ObjectPairs:
        obj1, obj2 = self.filtered_objects[i], self.filtered_objects[j]

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

    def select_target_object(self, obj_id: int, reason: str) -> Optional[list[int]]:
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
            return [nearest_obj]
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
            return [nearest_obj]

        # Aligned object selection
        if reason == "aligned_vertical":
            aligned_candidates = [
                other_id
                for other_id in candidates
                if self.pairs[(obj_id, other_id)].aligned_vertical
            ]
            if not aligned_candidates:
                return None
            return aligned_candidates
        if reason == "aligned_horizontal":
            aligned_candidates = [
                other_id
                for other_id in candidates
                if self.pairs[(obj_id, other_id)].aligned_horizontal
            ]
            if not aligned_candidates:
                return None
            return aligned_candidates

        if reason == "container":
            pass
        return None

    def move_to_target(
        self, obj_id: int, target_reason=str
    ) -> Optional[tuple[dir_coord, int]]:
        """
        For a specific object, move it towards a target object based on the direction vector.
        Return the direction vector and the distance to move.
        """
        target_id = self.select_target_object(obj_id, target_reason)
        if target_id is None:
            return None
        if len(target_id) == 1:
            pair = self.pairs[(obj_id, target_id[0])]
            return pair.direction, pair.distance
        return None

    # def aligned_targets(self, obj_id: int, alignment: str) -> Optional[list[int]]:
    #     """
    #     For a specific object, get all aligned target objects based on the alignment type.
    #     """
    #     if alignment not in ["aligned_vertical", "aligned_horizontal"]:
    #         return None
    #     target_ids = self.select_target_object(obj_id, alignment)
    #     return

    def _get_interior_pixels(self, object: ObjectState) -> PixelSet:
        """
        Get the interior pixels of an object
        """
        mask = pixels_to_mask(object.cell_positions, self.state.grid_state.dimensions)
        fill_mask = ndimage.binary_fill_holes(mask).astype(int)
        return mask_to_pixels(fill_mask)

    def interior_colour_count(self, obj_id: int) -> Counter:
        """
        Count the colours of pixels inside the bounding box of the object.
        Excludes the background colour from the count.
        """
        counter = Counter()
        for row, col in self.interior_pixels[obj_id]:
            colour = np.array(self.grid)[row, col]
            if colour != self.background_colour:
                counter[colour] += 1
        # Remove object's own colour from the count
        obj_colour = self.filtered_objects[obj_id].colour
        if obj_colour in counter:
            del counter[obj_colour]
        return counter


class RelationalDiff:
    """
    Comparison of two relational graphs to identify changes in relationships between objects.
    """

    def __init__(
        self,
        input_graph: RelationalGraph,
        output_graph: RelationalGraph,
        obj_matches: list[tuple[int, int]],
    ):
        self.input_graph = input_graph
        self.output_graph = output_graph
        # check if objects are in filtered objects of both graphs
        self.obj_matches = [
            (i, o)
            for i, o in obj_matches in input_graph.filtered_objects
            and output_graph.filtered_objects
        ]
        self.obj_mapping = {i: o for i, o in self.obj_matches}

        self.conserved: set[tuple[str, str]] = set()
        self.removed: set[tuple[str, str]] = set()
        self.added: set[tuple[str, str]] = set()

    def _compare_relationships(
        self,
        input_pair: ObjectPairs,
        output_pair: ObjectPairs,
        target_relation: str,
        conserved: set,
        removed: set,
        added: set,
    ) -> None:
        """
        Run comparison between graphs for specified relations
        """
        if input_pair is None or output_pair is None:
            return
        for relation in RELATION_FIELDS:
            input_value = _resolve_relation_fields(input_pair, relation)
            output_value = _resolve_relation_fields(output_pair, relation)
            if input_value and output_value:
                conserved.add((relation, target_relation))
            elif input_value and not output_value:
                removed.add((relation, target_relation))
            elif not input_value and output_value:
                added.add((relation, target_relation))

    def pairwise_diff(
        self,
        conserved: set,
        removed: set,
        added: set,
    ):
        """
        Summarise how relationships between objects have changed from input to output graph.
        """
        for pair_1 in self.obj_matches:
            for pair_2 in self.obj_matches:
                if pair_1[0] == pair_2[0]:
                    continue
                input_pair = self.input_graph.pairs.get(pair_1[0], pair_2[0])
                output_pair = self.output_graph.pairs.get(pair_1[1], pair_2[1])
                self._compare_relationships(
                    input_pair,
                    output_pair,
                    target_relation="any",
                    conserved=conserved,
                    removed=removed,
                    added=added,
                )

    def specific_target_diff(
        self,
        conserved: set,
        removed: set,
        added: set,
    ):
        """
        Summarise how relationships between objects have changed from input to output graph for a specific target relation.
        """
        for pair_1 in self.obj_matches:
            for reason in SPATIAL_TARGETS:
                targets = self.input_graph.select_target_object(pair_1[0], reason)

                if not targets or len(targets) != 1:
                    continue

                target_in = targets[0]
                target_out = self.obj_mapping.get(target_in)
                if target_out is None:
                    continue

                self._compare_relationships(
                    self.input_graph.pairs.get((pair_1[0], target_in)),
                    self.output_graph.pairs.get((pair_1[1], target_out)),
                    target_relation=reason,
                    conserved=conserved,
                    removed=removed,
                    added=added,
                )


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

    needs_graph: bool = True

    def resolve_for_object(self, obj: ObjectState, graph: RelationalGraph) -> Any:
        index = graph.get_index(obj)
        if index is None:
            return self.default_value

        if self.query == "colour_count_match":
            interior_count = graph.interior_colour_count(index)
            return interior_count if interior_count else self.default_value

        moveset = graph.move_to_target(index, self.target)
        # aligned_objs = graph.aligned_targets(index, self.target)
        if moveset is not None:
            direction, distance = moveset
            if self.query == "direction_to":
                return direction
            elif self.query == "steps_away_from":
                return distance

        return self.default_value


def add_relational_kwargs(pool: dict[str, set], relational_scores: dict = None) -> None:
    """
    Add relational kwargs to the kwarg pool based on the collected relational scores.
    """
    support_scores = relational_scores if relational_scores is not None else {}
    for kwarg_name, item in KWARG_SPACE.items():
        relation_pool = pool.setdefault(kwarg_name, set())
        for query, target in item:
            support_score = support_scores.get((kwarg_name, query, target), 0.0)
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
        moveset = input_graph.move_to_target(obj_in, target)
        if moveset is None:
            continue
        direction, distance = moveset
        if (direction[0] * distance, direction[1] * distance) == translation_shift:
            kwarg_list.append(
                {
                    "direction_vector": RelationalKwarg(
                        query="direction_to",
                        target=target,
                        default_value=direction,
                    ),
                    "scale": RelationalKwarg(
                        query="steps_away_from",
                        target=target,
                        default_value=distance,
                    ),
                }
            )
    return kwarg_list


def detect_counter_related_outputs(
    input_graph: RelationalGraph,
    out_state: ArcState,
    obj_in: int,
):
    """
    Check if the output state is related to any of the counter objects
    For example number of yellow pixels in output matches teh number of interior yellow pixels in input object.
    Save as relational kwargs for the transformation.
    """
    output_grid = out_state.grid_state.grid
    out_colour_count = Counter(np.array(output_grid).flatten())
    # Remove background colour from the count
    if input_graph.background_colour in out_colour_count:
        del out_colour_count[input_graph.background_colour]

    # 1. Interior colour count
    # check if any of the interior colours in input object match the colour counts in output state
    interior_colour_count = input_graph.interior_colour_count(obj_in)
    for colour, count in interior_colour_count.items():
        if colour == input_graph.background_colour:
            continue
        output_count = out_colour_count.get(colour)
        if output_count == count:
            return [
                {
                    "colour_count": RelationalKwarg(
                        query="colour_count_match",
                        target="any",
                        default_value=out_colour_count,
                    )
                }
            ]

    # 2. Object colour count
    object_colour_count = Counter(obj.colour for obj in input_graph.filtered_objects)
    for colour, count in object_colour_count.items():
        if colour == input_graph.background_colour:
            continue
        output_count = out_colour_count.get(colour)
        if output_count == count:
            return [
                {
                    "colour_count": RelationalKwarg(
                        query="colour_count_match",
                        target="any",
                        default_value=out_colour_count,
                    )
                }
            ]
    return []


def collect_relation_scores(
    state_cache: dict, obj_transformations: list[list[Any]]
) -> dict:
    """
    Collect relationship scores for each transformation based on the relational graph.
    """
    support_scores: Counter = Counter()
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
            # Detect colour kwargs
            colour_kwargs = detect_counter_related_outputs(
                cache_entry["input_graph"],
                cache_entry["output"],
                obj_t.input_object_id,
            )
            all_kwargs = translation_kwargs + colour_kwargs

            # Add support scores for each relational kwarg
            for rel_kwarg in all_kwargs:
                for k, rel_k in rel_kwarg.items():
                    support_scores[(k, rel_k.query, rel_k.target)] += 1
    return dict(support_scores)
