"""
Heuristic Layer for ArcAgent

Compares the input and output grids and objects to find differences.
"""

__date__ = "2026-06-11"
__author__ = "NedeeshaWeerasuriya"
__version__ = "0.1"


import numpy as np
from dataclasses import dataclass, field
from skimage.measure import label, regionprops
from ArcMemory import ArcState, GridState, ObjectState
from ArcSet import ArcSet

HU_TOLERANCE = 1


@dataclass
class GridDifference:
    """
    Comparison of two grid states
    """

    set_id: int

    # Shape
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]

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
class ObjectDifference:
    """
    Comparison of two object states
    """

    set_id: int
    object_id: tuple[int, int]

    # Object counts
    input_object: ObjectState
    output_object: ObjectState
    hu_distance: float

    # Changes
    colour_changed: bool
    # only store the colours if there is a colour change
    colours: tuple[int, int]
    position_changed: bool
    shape_changed: bool
    size_changed: bool


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
    object_differences: list[list[ObjectDifference]]
    pruned_primitives: frozenset[str]


class ArcHeuristics:
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
        object_differences: list[list[ObjectDifference]] = []

        for set_id, entry in enumerate(training_data):
            input_array = entry.get_input_data().data()
            output_array = entry.get_output_data().data()

            # Create ArcState for input and output
            input_state = ArcState.from_array(input_array, extract_objects=True)
            output_state = ArcState.from_array(output_array, extract_objects=True)
            self.state_cache[set_id] = {"input": input_state, "output": output_state}

            # Analyse grid differences
            grid_diff = self._compare_grids(set_id, input_array, output_array)
            grid_differences.append(grid_diff)

            # Analyse object differences
            object_diff = self._check_similar_objects(set_id, input_state, output_state)
            object_differences.append(object_diff)

        # Analyse conserved properties across all sets
        conserved_properties = self._analyse_conserved_properties(
            grid_differences, object_differences
        )
        pruned_primitives = self._prune_primitives(conserved_properties)

        return HeuristicSummary(
            conserved_properties=conserved_properties,
            grid_differences=grid_differences,
            object_differences=object_differences,
            pruned_primitives=pruned_primitives,
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

        # Colour differences
        input_colours = set(np.unique(input_array))
        output_colours = set(np.unique(output_array))
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
            input_colours=input_colours,
            output_colours=output_colours,
            new_colours=new_colours,
            removed_colours=removed_colours,
            colour_changed=colour_changed,
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

    # -----------------------------------------------------------------------------
    # Object-level heuristics
    # -----------------------------------------------------------------------------
    def _check_similar_objects(
        self, set_id: int, input_state: ArcState, output_state: ArcState
    ) -> list[ObjectDifference]:
        """
        Check in the training data, if there are objects that are similar from input to output.
        Use the invariant properties of the objects to determine if they are the same object or not.
        """
        object_matches = []

        for i, input_object in enumerate(input_state.objects):
            for o, output_object in enumerate(output_state.objects):
                # Use the Hu moments to find if object is similar in input and output
                # Use euclidean distance between Hu moments to determine if objects are the same
                if input_object.hu_moments is not None:
                    distance = np.linalg.norm(
                        np.array(input_object.hu_moments)
                        - np.array(output_object.hu_moments)
                    )
                    # store any objects that pass the threshold as similar objects
                    if distance < HU_TOLERANCE:
                        colour_changed = input_object.colour != output_object.colour
                        colours = (input_object.colour, output_object.colour)
                        object_matches.append(
                            ObjectDifference(
                                set_id=set_id,
                                object_id=(i, o),
                                input_object=input_object,
                                output_object=output_object,
                                hu_distance=distance,
                                colour_changed=colour_changed,
                                colours=colours if colour_changed else None,
                                position_changed=input_object.centroid
                                != output_object.centroid,
                                shape_changed=input_object.bounding_box
                                != output_object.bounding_box,
                                size_changed=input_object.area != output_object.area,
                            )
                        )
        return object_matches

    # -----------------------------------------------------------------------------
    # Conserved properties analysis
    # -----------------------------------------------------------------------------

    def _analyse_conserved_properties(
        self,
        grid_differences: list[GridDifference],
        object_differences: list[list[ObjectDifference]],
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
                len(object_diff) == 0 for object_diff in object_differences
            ),
            object_colours=all(
                all(
                    obj_diff.input_object.colour == obj_diff.output_object.colour
                    for obj_diff in object_diff_list
                )
                for object_diff_list in object_differences
            ),
            object_shapes=all(
                all(obj_diff.shape_changed == False for obj_diff in object_diff_list)
                for object_diff_list in object_differences
            ),
            all_square_grid=all(
                grid_diff.input_shape[0] == grid_diff.input_shape[1]
                for grid_diff in grid_differences
            ),
        )

    # -----------------------------------------------------------------------------
    # Pruning primitives based on conserved properties
    # -----------------------------------------------------------------------------
    def _prune_primitives(
        self, conserved_properties: ConservedAllSets
    ) -> frozenset[str]:
        """
        Compute the set of primitives to prune based on conserved properties.
        """
        pruned_primitives = set()

        if conserved_properties.grid_size:
            pruned_primitives.update({"tile", "crop_object", "crop_background_out"})
        if conserved_properties.colours:
            pruned_primitives.update(
                {
                    "change_colour",
                    "update_colour",
                }
            )
        if conserved_properties.object_count:
            pruned_primitives.update({"duplicate_object", "remove_object"})
        if conserved_properties.object_colours:
            pruned_primitives.update({"change_object_colour"})
        if conserved_properties.object_shapes:
            pruned_primitives.update({"rotate_object", "flip_object"})
        if not conserved_properties.all_square_grid:
            pruned_primitives.update(
                {"mirror_diagonal_top_left", "mirror_diagonal_bottom_right"}
            )

        return frozenset(pruned_primitives)
