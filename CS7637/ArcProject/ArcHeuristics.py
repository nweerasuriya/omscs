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
MUTATION_TYPES = ["growth", "shrink", "shape_change", "translation"]


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
class ObjectDifference:
    """
    Comparison of two object states
    """

    set_id: int
    object_id: tuple[int, int]

    # Object
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
class MutatatedObject:
    """
    Object that has mutated from input to output
    """

    set_id: int
    input_object_id: int
    output_object_id: int
    mutation_types: list[str]
    direction_vector: tuple[int, int] = field(default_factory=lambda: (0, 0))
    centroid_change: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))


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
    mutations: list[list[MutatatedObject]]
    split_grid: dict[str, bool]


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
        object_differences: list[list[ObjectDifference]] = []
        mutation_list: list[list[MutatatedObject]] = []
        split_grid: list[dict[str, bool]] = []

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

            # Analyse object mutations
            mutations = self._check_object_mutations(set_id, input_state, output_state)
            mutation_list.append(mutations)

            # Analyse split grid
            split_grid_result = self._check_split_grid(input_array)
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
            grid_differences, object_differences
        )

        return HeuristicSummary(
            conserved_properties=conserved_properties,
            grid_differences=grid_differences,
            object_differences=object_differences,
            mutations=mutation_list,
            split_grid=split_grid,
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

        # Colour differences not including 0
        input_colours = set(np.unique(input_array)) - {0}
        output_colours = set(np.unique(output_array)) - {0}
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

    def _check_split_grid(self, input_array: np.ndarray) -> dict[str, bool]:
        """
        Check if the input grid is split by a straight line anywhere in the grid populated by non zero values.
        For diagonal splits, check if the diagonal or anti diagonal is populated by non zero values.
        """
        rows, cols = input_array.shape
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
        return {
            "vertical": vertical_split,
            "horizontal": horizontal_split,
            "diagonal": diagonal_split,
            "anti_diagonal": anti_diagonal_split,
        }

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

    def _check_object_mutations(
        self, set_id: int, input_state: ArcState, output_state: ArcState
    ) -> list[MutatatedObject]:
        """
        Identify objects which have mutated from the input to output.
        Only consider subsets of objects for now. So one object must be a subset of the other.

        Consider shape, size and translation changes. Check as matrix so combinations of changes can be detected.
        """
        mutations = []
        for i, input_object in enumerate(input_state.objects):
            for o, output_object in enumerate(output_state.objects):
                # Check bounding box of one object is a subset of the other
                if not (
                    self._bbox_overlap(
                        input_object.bounding_box, output_object.bounding_box
                    )
                ):
                    continue

                # Determine type of change
                mutation_types = []

                input_pixels = input_object.cell_positions
                output_pixels = output_object.cell_positions

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
                if len(input_pixels) != len(output_pixels) and not (
                    is_subset or is_superset
                ):
                    mutation_types.append("shape_change")
                if (
                    len(input_pixels) == len(output_pixels)
                    and input_object.bounding_box != output_object.bounding_box
                ):
                    mutation_types.append("shape_change")

                centroid_change = (
                    (output_object.centroid[0] - input_object.centroid[0]),
                    (output_object.centroid[1] - input_object.centroid[1]),
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
                        dir_row = new_avg_row - input_object.centroid[0]
                        dir_col = new_avg_col - input_object.centroid[1]
                        direction_vector = (
                            int(np.sign(dir_row)),
                            int(np.sign(dir_col)),
                        )

                # Add mutation direction information to input_state
                current_mutation_types = input_state.objects[i].mutation_types
                current_mutation_vectors = input_state.objects[i].mutation_vectors
                input_state.objects[i] = input_state.objects[i]._replace(
                    mutation_types=current_mutation_types.union(mutation_types),
                    mutation_vectors=current_mutation_vectors + (centroid_change,),
                )
                mutations.append(
                    MutatatedObject(
                        set_id=set_id,
                        input_object_id=i,
                        output_object_id=o,
                        mutation_types=mutation_types,
                        centroid_change=centroid_change,
                        direction_vector=direction_vector,
                    )
                )
        return mutations
    
    # def check_association_mutation_object_prop(
    #         self, mutation_list: list[list[MutatatedObject]],
    # ):
    #     """
    #     For all mutations gathered in the training data, check if there are any associations
    #     between a mutation and a specific object property.
    #     For example, only blue (colour=1) objects grow in direction (1, 0) in all sets. 
    #     Or only closed objects change shape
    #     """
    #     mutation_association = {}
    #     for set_mutations in mutation_list:
    #         for mutation in set_mutations:


                
            

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
                len(self.state_cache[i]["input"].objects)
                == len(self.state_cache[i]["output"].objects)
                for i in range(len(grid_differences))
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
