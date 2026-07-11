"""
Heuristic Layer for ArcAgent

Compares the input and output grids and objects to find differences.
"""

__date__ = "2026-06-11"
__author__ = "NedeeshaWeerasuriya"
__version__ = "0.1"

from typing import List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
from ArcMemory import ArcState, GridState, ObjectState, PixelSet
from ArcRelations import RelationalGraph
from ArcSet import ArcSet

HU_TOLERANCE = 1
# Mutation Related Parameters
MUTATION_TYPES = ["growth", "shrink", "shape_change", "translation", "filled"]
ASSOCIATION_KWARGS = [
    "direction_vector",
    "centroid_change",
    "scale",
    "new_colour_obj",
    "filled",
]
OBJECT_PROPERTIES = ["colour", "is_closed"]
MIN_ASSOCIATION_SUPPORT = 2


@dataclass
class GridDifference:
    """
    Comparison of two grid states
    """

    set_id: int

    # Shape
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]

    # Object count
    input_object_count: int
    output_object_count: int
    object_count_changed: bool

    # Colour differences
    input_colours: set[int]
    output_colours: set[int]
    new_colours: set[int]
    removed_colours: set[int]
    colour_changed: bool

    # Symmetry differences
    input_symmetry_h: bool
    output_symmetry_h: bool
    input_symmetry_v: bool
    output_symmetry_v: bool
    input_symmetry_diag: bool
    output_symmetry_diag: bool
    input_symmetry_anti_diag: bool
    output_symmetry_anti_diag: bool
    symmetry_changed: bool


@dataclass
class ObjectTransformation:
    """
    A single class to capture information on object transformations from input to output
    """

    set_id: int
    input_object_id: tuple[int, int]
    output_object_id: tuple[int, int]

    # Changes
    colour_changed: bool
    colours: tuple[int, int]
    position_changed: bool
    shape_changed: bool
    size_changed: bool
    filled: bool

    mutation_types: list[str]
    direction_vector: tuple[int, int] = field(default_factory=lambda: (0, 0))
    centroid_change: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    new_colour_obj: Optional[int] = None
    scale: int = 1


@dataclass
class PropertyAssociation:
    """
    Association between a mutation and a specific object property
    """

    kwarg: str
    object_property: str
    mapping: dict  # Maps object property values to the associated kwarg values
    support_score: float = 0.0


@dataclass
class ConservedAllSets:
    """
    Properties conserved across all sets in a task
    """

    grid_size: bool
    colours: bool
    object_count: bool
    object_colours: bool
    object_shapes: bool
    all_square_grid: bool


@dataclass
class HeuristicSummary:
    """
    Summary of heuristic analysis for a task
    """

    conserved_properties: ConservedAllSets
    grid_differences: list[GridDifference]
    object_transformations: list[list[ObjectTransformation]]
    split_grid: dict[str, bool]
    property_associations: list[PropertyAssociation] = field(default_factory=list)


class HeuristicEngine:
    """
    Heuristic analysis for all sets in a ArcProblem
    """

    def __init__(self):
        self.state_cache: dict[int, dict[str, ArcState]] = {}

    def run_analysis(self, training_data: list[ArcSet]):
        """
        Runs heuristic analysis on the training data and returns a summary of differences.
        """
        # Analyse each set for grid and object differences
        grid_differences: list[GridDifference] = []
        object_transformations: list[list[ObjectTransformation]] = []
        split_grid: list[dict[str, bool]] = []

        for set_id, entry in enumerate(training_data):
            input_array = entry.get_input_data().data()
            output_array = entry.get_output_data().data()

            # Create ArcState for input and output
            input_state = ArcState.from_array(input_array, extract_objects=True)
            output_state = ArcState.from_array(output_array, extract_objects=True)
            self.state_cache[set_id] = {
                "input": input_state,
                "output": output_state,
                "input_graph": RelationalGraph(input_state),
                "output_graph": RelationalGraph(output_state),
            }

            # Analyse grid differences
            grid_diff = self._compare_grids(set_id, input_array, output_array)
            grid_differences.append(grid_diff)

            # Analyse object differences
            transformations = self.match_objects(set_id, input_state, output_state)
            object_transformations.append(transformations)

            # Analyse split grid
            split_grid_result = self._check_split_grid(input_array, output_array)
            split_grid.append(split_grid_result)

        # If any one of the directions is true in all sets, then the split grid is considered true for that direction
        if any(
            all(split[direction] for split in split_grid) for direction in split_grid[0]
        ):
            split_grid = {
                direction: all(split[direction] for split in split_grid)
                for direction in split_grid[0]
            }
        else:
            split_grid = None

        # Analyse conserved properties across all sets
        conserved_properties = self._analyse_conserved_properties(
            grid_differences, object_transformations
        )

        # Object property associations with mutations
        property_associations = self.infer_property_associations(object_transformations)

        return HeuristicSummary(
            conserved_properties=conserved_properties,
            grid_differences=grid_differences,
            object_transformations=object_transformations,
            split_grid=split_grid,
            property_associations=property_associations,
        )

    # -----------------------------------------------------------------------------
    # Grid-level heuristics
    # -----------------------------------------------------------------------------
    def _compare_grids(
        self, set_id: int, input_array: np.ndarray, output_array: np.ndarray
    ) -> GridDifference:
        """
        Compares two grids and returns a GridDifference object.
        """
        # Shape differences
        input_shape = input_array.shape
        output_shape = output_array.shape
        shape_changed = input_shape != output_shape

        # Object count differences
        input_object_count = len(self.state_cache[set_id]["input"].objects)
        output_object_count = len(self.state_cache[set_id]["output"].objects)
        object_count_changed = input_object_count != output_object_count

        # Colour differences not including background colour
        background_colour = self.state_cache[set_id][
            "input"
        ].grid_state.background_colour
        input_colours = set(np.unique(input_array)) - {background_colour}
        output_colours = set(np.unique(output_array)) - {background_colour}
        new_colours = output_colours - input_colours
        removed_colours = input_colours - output_colours
        colour_changed = bool(new_colours or removed_colours)

        # Symmetry differences
        input_symmetry_h = np.array_equal(input_array, np.flipud(input_array))
        output_symmetry_h = np.array_equal(output_array, np.flipud(output_array))
        input_symmetry_v = np.array_equal(input_array, np.fliplr(input_array))
        output_symmetry_v = np.array_equal(output_array, np.fliplr(output_array))
        input_symmetry_diag = np.array_equal(input_array, np.transpose(input_array))
        output_symmetry_diag = np.array_equal(output_array, np.transpose(output_array))
        input_symmetry_anti_diag = np.array_equal(
            input_array, np.fliplr(np.transpose(input_array))
        )
        output_symmetry_anti_diag = np.array_equal(
            output_array, np.fliplr(np.transpose(output_array))
        )
        symmetry_changed = (
            input_symmetry_h != output_symmetry_h
            or input_symmetry_v != output_symmetry_v
            or input_symmetry_diag != output_symmetry_diag
            or input_symmetry_anti_diag != output_symmetry_anti_diag
        )
        return GridDifference(
            set_id=set_id,
            input_shape=input_shape,
            output_shape=output_shape,
            # Object count
            input_object_count=input_object_count,
            output_object_count=output_object_count,
            object_count_changed=object_count_changed,
            # Colour differences
            input_colours=input_colours,
            output_colours=output_colours,
            new_colours=new_colours,
            removed_colours=removed_colours,
            colour_changed=colour_changed,
            # Symmetry differences
            input_symmetry_h=input_symmetry_h,
            output_symmetry_h=output_symmetry_h,
            input_symmetry_v=input_symmetry_v,
            output_symmetry_v=output_symmetry_v,
            input_symmetry_diag=input_symmetry_diag,
            output_symmetry_diag=output_symmetry_diag,
            input_symmetry_anti_diag=input_symmetry_anti_diag,
            output_symmetry_anti_diag=output_symmetry_anti_diag,
            symmetry_changed=symmetry_changed,
        )

    def _check_split_grid(
        self, input_array: np.ndarray, output_array: np.ndarray
    ) -> dict[str, bool]:
        """
        Check if the input grid is split by a straight line anywhere in the grid populated by non zero values.
        For diagonal splits, check if the diagonal or anti diagonal is populated by non zero values.

        Also check if the output array is half the size in the same direction as the input array in the case.
        """
        rows, cols = input_array.shape
        out_rows, out_cols = output_array.shape
        # Check for vertical split
        vertical_split = any(
            np.all(input_array[:, col] != 0) for col in range(1, cols - 1)
        )
        # Check for horizontal split
        horizontal_split = any(
            np.all(input_array[row, :] != 0) for row in range(1, rows - 1)
        )
        # diagonal split
        diagonal_split = np.all(np.diag(input_array) != 0)
        # anti diagonal split
        anti_diagonal_split = np.all(np.diag(np.fliplr(input_array)) != 0)

        # Check output array for half size in the same direction as the input array
        if not vertical_split:
            if out_cols == cols // 2:
                vertical_split = True
        if not horizontal_split:
            if out_rows == rows // 2:
                horizontal_split = True

        return {
            "vertical": vertical_split,
            "horizontal": horizontal_split,
            "diagonal": diagonal_split,
            "anti_diagonal": anti_diagonal_split,
        }

    # -----------------------------------------------------------------------------
    # Object-level heuristics
    # -----------------------------------------------------------------------------
    def _hu_distance(
        self, input_object: ObjectState, output_object: ObjectState
    ) -> float:
        """
        Calculate the distance between two Hu moments.
        """
        hu1 = input_object.hu_moments
        hu2 = output_object.hu_moments
        if hu1 is None or hu2 is None:
            return float("inf")
        return float(np.linalg.norm(np.array(hu1) - np.array(hu2)))

    def match_objects(
        self, set_id: int, input_state: ArcState, output_state: ArcState
    ) -> list[ObjectTransformation]:
        """
        Two stages of object comparison:
        1. Use Hu moments to find if objects are similar in input and output
            Check in the training data, if there are objects that are similar from input to output.
            Use the invariant properties of the objects to determine if they are the same object or not.
        2. When hu moments diverge, find mutations in the objects from input to output.
        """
        transforms = []
        visited_input_objects = set()
        visited_output_objects = set()

        for i, input_object in enumerate(input_state.objects):
            candidate_matches = [
                (
                    self._hu_distance(input_object, output_object),
                    o,
                )
                for o, output_object in enumerate(output_state.objects)
                if o not in visited_output_objects
            ]
            # Get only the matches that are within the tolerance
            candidate_matches = [
                (dist, o) for dist, o in candidate_matches if dist <= HU_TOLERANCE
            ]
            if not candidate_matches:
                continue
            closest_distance, o = min(candidate_matches, key=lambda x: x[0])
            visited_input_objects.add(i)
            visited_output_objects.add(o)
            transform = self._describe_mutation(
                set_id, i, o, input_object, output_state.objects[o]
            )
            transforms.append(transform)
            # self._save_mutation(input_state, i, transform)

        for i, input_object in enumerate(input_state.objects):
            if i in visited_input_objects:
                continue
            best_iou = 0
            best_o = None
            for o, output_object in enumerate(output_state.objects):
                if o in visited_output_objects:
                    continue
                if not self._bbox_overlap(
                    input_object.bounding_box, output_object.bounding_box
                ):
                    continue
                # cUse IoU (Intersection over Union) to find the best match between the two objects
                intersection = len(
                    input_object.cell_positions & output_object.cell_positions
                )
                if intersection == 0:
                    continue
                union = len(input_object.cell_positions | output_object.cell_positions)
                iou = intersection / union
                if iou > best_iou:
                    best_iou = iou
                    best_o = o

            if best_o is not None:
                visited_input_objects.add(i)
                visited_output_objects.add(best_o)
                transform = self._describe_mutation(
                    set_id, i, best_o, input_object, output_state.objects[best_o]
                )
                transforms.append(transform)
                # self._save_mutation(input_state, i, transform)

        return transforms

    def _bbox_overlap(
        self, bbox1: tuple[int, int, int, int], bbox2: tuple[int, int, int, int]
    ) -> bool:
        """
        Check if two bounding boxes overlap.
        Bounding box format is (min_row, max_row, min_col, max_col).
        """
        min_row1, max_row1, min_col1, max_col1 = bbox1
        min_row2, max_row2, min_col2, max_col2 = bbox2

        row_overlap = min_row1 <= max_row2 and min_row2 <= max_row1
        col_overlap = min_col1 <= max_col2 and min_col2 <= max_col1

        return row_overlap and col_overlap

    def mutation_scale(
        self,
        direction_vector: tuple[int, int],
        centroid_change: tuple[float, float],
        input_pixels: PixelSet,
        output_pixels: PixelSet,
        mutation_types: list[str],
    ) -> float:
        """
        Determine the magnitude of change in the mutation based on the direction vector and centroid change.
        1. Translation: Change in the centroid position of the object
        2. Growth: Length of new pixels in the direction of the direction vector

        Length of pixels is calculated as the number of new pixels in the direction of the direction vector.
        Look for furthers point in original object in the direction of the direction vector and
        compare with new pixels in the direction of the direction vector.
        """

        if "translation" in mutation_types:
            scale = np.linalg.norm(centroid_change)

        new_pixels = output_pixels - input_pixels
        if not new_pixels:
            scale = 0
        else:
            # Find the furthest point in the original object in the direction of the direction vector
            furthest_point = max(
                input_pixels,
                key=lambda p: p[0] * direction_vector[0] + p[1] * direction_vector[1],
            )
            # Find the furthest point in the new pixels in the direction of the direction vector
            furthest_new_point = max(
                new_pixels,
                key=lambda p: p[0] * direction_vector[0] + p[1] * direction_vector[1],
            )
            # Calculate the distance between the two points
            scale = int(
                np.linalg.norm(np.array(furthest_new_point) - np.array(furthest_point))
            )
        return scale

    def _describe_mutation(
        self,
        set_id: int,
        i: int,
        o: int,
        in_object: ObjectState,
        out_object: ObjectState,
    ) -> ObjectTransformation:
        """
        Identify objects which have mutated from the input to output.
        Only consider subsets of objects for now. So one object must be a subset of the other.

        Consider shape, size and translation changes. Check as matrix so combinations of changes can be detected.
        """
        input_pixels = in_object.cell_positions
        output_pixels = out_object.cell_positions
        # Determine type of change
        mutation_types = []
        is_subset = False
        is_superset = False

        # Check if one object is a subset of the other (frozenset property)
        if len(input_pixels) < len(output_pixels):
            if input_pixels <= output_pixels:
                mutation_types.append("growth")
                is_subset = True
        elif len(input_pixels) > len(output_pixels):
            if output_pixels <= input_pixels:
                mutation_types.append("shrink")
                is_superset = True

        # Shape Change
        if len(input_pixels) != len(output_pixels) and not (is_subset or is_superset):
            mutation_types.append("shape_change")
        if (
            len(input_pixels) == len(output_pixels)
            and in_object.bounding_box != out_object.bounding_box
        ):
            mutation_types.append("shape_change")

        centroid_change = (
            (out_object.centroid[0] - in_object.centroid[0]),
            (out_object.centroid[1] - in_object.centroid[1]),
        )
        # Translation if there is a change in position but not shape or size
        if mutation_types == [] and centroid_change != (0, 0):
            mutation_types.append("translation")

        # Check direction of additional pixels for growth or shape change
        direction_vector = (0, 0)
        if "growth" in mutation_types or "shape_change" in mutation_types:
            # Find new pixels in output that are not in input
            new_pixels = output_pixels - input_pixels
            if new_pixels:
                new_avg_row = sum(r for r, c in new_pixels) / len(new_pixels)
                new_avg_col = sum(c for r, c in new_pixels) / len(new_pixels)

                # Check from centroid
                dir_row = new_avg_row - in_object.centroid[0]
                dir_col = new_avg_col - in_object.centroid[1]
                direction_vector = (
                    int(np.sign(dir_row)),
                    int(np.sign(dir_col)),
                )
        elif "translation" in mutation_types:
            direction_vector = (
                int(np.sign(centroid_change[0])),
                int(np.sign(centroid_change[1])),
            )

        # Check if object has been filled with a new colour
        if "filled" in mutation_types:
            # Check if empty pixels inside of the object have been filled with a new colour
            empty_pixels = in_object.pixels - out_object.pixels
            if empty_pixels:
                mutation_types.append("filled")

        scale = self.mutation_scale(
            direction_vector,
            centroid_change,
            input_pixels,
            output_pixels,
            mutation_types,
        )

        colour_changed = in_object.colour != out_object.colour

        return ObjectTransformation(
            set_id=set_id,
            input_object_id=i,
            output_object_id=o,
            colour_changed=colour_changed,
            colours=(in_object.colour, out_object.colour),
            position_changed=centroid_change != (0, 0),
            shape_changed=in_object.bounding_box != out_object.bounding_box,
            size_changed=len(input_pixels) != len(output_pixels),
            filled=True if "filled" in mutation_types else False,
            mutation_types=mutation_types,
            direction_vector=direction_vector,
            centroid_change=centroid_change,
            scale=scale,
            new_colour_obj=out_object.colour if colour_changed else None,
        )

    def check_association_mutation_object_prop(
        self,
        mutation_list: list[list[ObjectTransformation]],
        obj_property: str,
        kwarg: str,
    ) -> PropertyAssociation:
        """
        For all mutations gathered in the training data, check if there are any associations
        between a mutation and a specific object property.
        For example, only blue (colour=1) objects grow in direction (1, 0) in all sets.
        Or only closed objects change shape
        Check for conservation across training sets
        """
        ignore_values = {None, 0, (0, 0), (0.0, 0.0)}
        mapping = {}
        num_sets = 0
        for set_mutations in mutation_list:
            for mutation in set_mutations:
                # Get the input object from the state cache
                input_object = self.state_cache[mutation.set_id]["input"].objects[
                    mutation.input_object_id
                ]
                # Check if the object property is present in the input object
                if hasattr(input_object, obj_property):
                    prop_value = getattr(input_object, obj_property)
                    # Read the kwarg value from the mutation object
                    kwarg_value = getattr(mutation, kwarg)
                    if kwarg_value in ignore_values:
                        continue
                    if prop_value not in mapping:
                        mapping[prop_value] = set()
                    mapping[prop_value].add(kwarg_value)
                    num_sets += 1
        return PropertyAssociation(
            kwarg=kwarg,
            object_property=obj_property,
            mapping=mapping,
            support_score=num_sets,
        )

    def infer_property_associations(
        self, mutation_list: list[list[ObjectTransformation]]
    ) -> list[PropertyAssociation]:
        """
        Check if the property associations are valid and return a list of PropertyAssociation objects.
        """
        associations = []
        for obj_property in OBJECT_PROPERTIES:
            for kwarg in ASSOCIATION_KWARGS:
                association = self.check_association_mutation_object_prop(
                    mutation_list, obj_property, kwarg
                )
                # Check each prop value maps to a single kwarg
                if any(len(values) > 1 for values in association.mapping.values()):
                    continue
                if (
                    association.support_score >= MIN_ASSOCIATION_SUPPORT
                    and association.mapping
                ):
                    associations.append(association)
        return associations

    # -----------------------------------------------------------------------------
    # Conserved properties analysis
    # -----------------------------------------------------------------------------

    def _analyse_conserved_properties(
        self,
        grid_differences: list[GridDifference],
        object_transformations: list[list[ObjectTransformation]],
    ) -> ConservedAllSets:
        """
        Analyse the differences across all sets in the ArcProblem
        """
        return ConservedAllSets(
            grid_size=all(
                grid_diff.input_shape == grid_diff.output_shape
                for grid_diff in grid_differences
            ),
            colours=all(not grid_diff.colour_changed for grid_diff in grid_differences),
            object_count=all(
                len(self.state_cache[i]["input"].objects)
                == len(self.state_cache[i]["output"].objects)
                for i in range(len(grid_differences))
            ),
            object_colours=all(
                all(obj.colour_changed == False for obj in object_diff_list)
                for object_diff_list in object_transformations
            ),
            object_shapes=all(
                all(obj.shape_changed == False for obj in object_diff_list)
                for object_diff_list in object_transformations
            ),
            all_square_grid=all(
                grid_diff.input_shape[0] == grid_diff.input_shape[1]
                for grid_diff in grid_differences
            ),
        )
