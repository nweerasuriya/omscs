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
    "nearest_not_same_colour",
    "nearest_line_same_colour",
    # "nearest_line_not_same_colour",
    # "aligned_vertical_horizontal",
    # "largest_same_colour",
    # "smallest_same_colour",
    "block",
    "container",
)

KWARG_SPACE: dict[str, tuple[tuple[str, str], ...]] = {
    "direction_vector": tuple(("direction_to", t) for t in SPATIAL_TARGETS),
    "scale": tuple(("steps_away_from", t) for t in SPATIAL_TARGETS),
    "target_pos": tuple(("target_position", t) for t in SPATIAL_TARGETS),
    "colour_count": tuple(
        ("colour_count_match", t) for t in ["any", "container", "block"]
    ),
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
        self.filtered_objects = [
            obj for obj in state.objects if obj.cell_positions
        ]  # Filter out objects with no cell positions

        self.object_count = len(self.filtered_objects)
        self.colour_counts = self.state.grid_state.colour_counts
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
        # bbox is min_row, min_col, max_row, max_col
        row_overlap = not (bbox_1[2] < bbox_2[0] or bbox_2[2] < bbox_1[0])
        col_overlap = not (bbox_1[3] < bbox_2[1] or bbox_2[3] < bbox_1[1])

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

        if reason == "nearest_not_same_colour":
            not_same_colour_candidates = [
                other_id
                for other_id in candidates
                if not self.pairs[(obj_id, other_id)].same_colour
            ]
            if not not_same_colour_candidates:
                return None
            nearest_obj = min(
                not_same_colour_candidates,
                key=lambda other_id: self.pairs[(obj_id, other_id)].distance,
            )
            return [nearest_obj]

        # Nearest line refers to another object that has a pixels in a straight line >4 pixels in size
        if reason == "nearest_line_same_colour":
            line_candidates = [
                other_id
                for other_id in candidates
                if self.pairs[(obj_id, other_id)].same_colour
                and self._lookup_obj(other_id).has_line
            ]
            if not line_candidates:
                return None
            nearest_obj = min(
                line_candidates,
                key=lambda other_id: self.pairs[(obj_id, other_id)].distance,
            )
            return [nearest_obj]

        if reason == "nearest_line_not_same_colour":
            line_candidates = [
                other_id
                for other_id in candidates
                if not self.pairs[(obj_id, other_id)].same_colour
                and self._lookup_obj(other_id).has_line
            ]
            if not line_candidates:
                return None
            nearest_obj = min(
                line_candidates,
                key=lambda other_id: self.pairs[(obj_id, other_id)].distance,
            )
            return [nearest_obj]

        # Aligned object selection
        if reason == "aligned_vertical_horizontal":
            aligned_candidates = [
                other_id
                for other_id in candidates
                if self.pairs[(obj_id, other_id)].aligned_vertical
                or self.pairs[(obj_id, other_id)].aligned_horizontal
            ]
            if not aligned_candidates:
                return None
            return aligned_candidates

        if reason == "block":
            block_candidates = [
                other_id
                for other_id in candidates
                if self._lookup_obj(other_id).is_block
            ]
            if not block_candidates:
                return None
            return block_candidates

        if reason == "container":
            container_candidates = [
                other_id
                for other_id in candidates
                if self.is_container(other_id)
                and obj_id in self.interior_pixels[other_id]
            ]
            if not container_candidates:
                return None
            return container_candidates
        return None

    def move_to_target(
        self, obj_id: int, target_reason=str
    ) -> Optional[tuple[dir_coord, int, tuple[float, float]]]:
        """
        For a specific object, move it towards a target object based on the direction vector.
        Return the direction vector and the distance to move and the centroid position
        """
        target_id = self.select_target_object(obj_id, target_reason)
        if target_id is None:
            return None
        if len(target_id) == 1:
            pair = self.pairs[(obj_id, target_id[0])]
            return (
                pair.direction,
                pair.distance,
                self._lookup_obj(target_id[0]).centroid,
            )
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
        rows, cols = self.state.grid_state.dimensions
        inside = frozenset(
            (r, c)
            for (r, c) in object.cell_positions
            if 0 <= r < rows and 0 <= c < cols
        )
        mask = pixels_to_mask(inside, self.state.grid_state.dimensions)
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

    def is_container(self, obj_id: int) -> bool:
        """
        Check if an object encloses any other object based on its interior pixels and the cell positions of other objects.
        """
        interior_pixels = set(self.interior_pixels[obj_id]) - set(
            self.filtered_objects[obj_id].cell_positions
        )
        if not interior_pixels:
            return False
        for i in range(self.object_count):
            if i == obj_id:
                continue
            if set(self.filtered_objects[i].cell_positions) & interior_pixels:
                return True
        return False


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
            if self.target == "container":
                interior_count = graph.interior_colour_count(index)
                return interior_count if interior_count else self.default_value
            elif self.target == "any":
                # Otherwise return object colour count ordered by ascending count
                pixel_colour_count = graph.colour_counts.copy()
                return pixel_colour_count if pixel_colour_count else self.default_value
            elif self.target == "block":
                block_colour_count = Counter(
                    obj.colour for obj in graph.filtered_objects if obj.is_block
                )
                return block_colour_count if block_colour_count else self.default_value

        moveset = graph.move_to_target(index, self.target)
        # aligned_objs = graph.aligned_targets(index, self.target)
        if moveset is not None:
            direction, distance, centroid = moveset
            if self.query == "direction_to":
                return direction
            elif self.query == "steps_away_from":
                return distance
            elif self.query == "target_position":
                return centroid

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
        direction, distance, centroid = moveset
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
                    "target_pos": RelationalKwarg(
                        query="target_position",
                        target=target,
                        default_value=input_graph._lookup_obj(obj_in).centroid,
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
    colour_count_list = []
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
            colour_count_list.append(
                {
                    "colour_count": RelationalKwarg(
                        query="colour_count_match",
                        target="container",
                        default_value=out_colour_count,
                        support_score=1.0,
                    )
                }
            )

    # 2. Object colour count
    object_colour_count = Counter(
        obj.colour for obj in input_graph.filtered_objects if obj.is_block
    )
    for colour, count in object_colour_count.items():
        if colour == input_graph.background_colour:
            continue
        output_count = out_colour_count.get(colour)
        if output_count == count:
            colour_count_list.append(
                {
                    "colour_count": RelationalKwarg(
                        query="colour_count_match",
                        target="block",
                        default_value=out_colour_count,
                    )
                }
            )

    # 3. Pixel colour count
    pixel_colour_count = input_graph.colour_counts.copy()
    for colour, count in pixel_colour_count.items():
        if colour == input_graph.background_colour:
            continue
        output_count = out_colour_count.get(colour)
        if output_count == count:
            colour_count_list.append(
                {
                    "colour_count": RelationalKwarg(
                        query="colour_count_match",
                        target="any",
                        default_value=out_colour_count,
                    )
                }
            )
    return colour_count_list


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
            if not all_kwargs:
                continue
            # Add support scores for each relational kwarg
            for rel_kwarg in all_kwargs:
                for k, rel_k in rel_kwarg.items():
                    support_scores[(k, rel_k.query, rel_k.target)] += 1
    return dict(support_scores)


# -----------------------------------------------------------------------------
# Object Selection
# -----------------------------------------------------------------------------

SELECT_REASONS: dict[str, callable] = {
    "any": lambda graph, i: True,
    "smallest": lambda graph, i: graph.filtered_objects[i].area
    == min(obj.area for obj in graph.filtered_objects),
    "largest": lambda graph, i: graph.filtered_objects[i].area
    == max(obj.area for obj in graph.filtered_objects),
    "is_closed": lambda graph, i: graph.filtered_objects[i].is_closed,
    "container": lambda graph, i: graph.is_container(i),
    "most_common_colour": lambda graph, i: graph.filtered_objects[i].colour
    == graph.colour_counts.most_common(1)[0][0],
    "least_common_colour": lambda graph, i: graph.filtered_objects[i].colour
    == graph.colour_counts.most_common()[-1][0],
    "min_colour": lambda graph, i: graph.filtered_objects[i].colour
    == min(obj.colour for obj in graph.filtered_objects),
}


@dataclass(frozen=True)
class ObjectSelector:
    """
    Select objects from a relational graph based on specific criteria.
    """

    reason: str
    support_score: float = 0.0
    needs_graph: bool = True

    def select_obj_id(self, graph: RelationalGraph) -> frozenset[int]:
        """
        Select objects from the relational graph based on the specified reason.
        """
        if self.reason not in SELECT_REASONS:
            return frozenset(
                range(graph.object_count)
            )  # Return all objects if reason is invalid
        return frozenset(
            i
            for i in range(graph.object_count)
            if SELECT_REASONS[self.reason](graph, i)
        )

    def select_objects(self, graph: RelationalGraph) -> list[ObjectState]:
        """
        Select objects from the relational graph based on the specified reason.
        """
        selected_ids = self.select_obj_id(graph)
        return [graph.filtered_objects[i] for i in selected_ids]


def collect_object_selection_scores(
    state_cache: dict, obj_transformations: list[list[Any]]
) -> dict:
    """
    Collect object selection scores for each transformation based on the relational graph.
    """
    support_scores: Counter = Counter()
    for obj_list in obj_transformations:
        changes_by_set = {}
        for obj_t in obj_list:
            changed = changes_by_set.setdefault(obj_t.set_id, set())
            if (
                obj_t.position_changed
                or obj_t.colour_changed
                or obj_t.size_changed
                or obj_t.shape_changed
            ):
                changed.add(obj_t.input_object_id)
            for obj_id, changed in changes_by_set.items():
                if not changed:
                    continue
                cache_entry = state_cache.get(obj_id)
                if not cache_entry or "input_graph" not in cache_entry:
                    continue
                input_graph = cache_entry["input_graph"]
                target = frozenset(changed)
                for reason in SELECT_REASONS:
                    if reason == "any":
                        continue
                    if target == ObjectSelector(reason=reason).select_obj_id(
                        input_graph
                    ):
                        support_scores[(reason, "object_selector")] += 1
    return dict(support_scores)


def add_object_selection_kwargs(
    pool: dict[str, set], selection_scores: dict = None
) -> None:
    """
    Add object selection kwargs to the kwarg pool based on the collected selection scores.
    """
    support_scores = selection_scores if selection_scores is not None else {}
    for reason in SELECT_REASONS:
        if reason == "any":
            continue
        support_score = support_scores.get((reason, "object_selector"), 0.0)
        pool.setdefault("object_selector", set()).add(
            ObjectSelector(reason=reason, support_score=support_score)
        )
