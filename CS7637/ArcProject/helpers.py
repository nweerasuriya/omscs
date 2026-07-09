"""
Helper functions
"""

__date__ = "2026-06-24"
__author__ = "NedeeshaWeerasuriya"
__version__ = "0.1"


from typing import Optional

import numpy as np
from scipy.spatial import distance

# Enums
ALL_DIRECTIONS = {(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)}
COLOUR_SET = {0, 1, 2, 3, 4, 5, 6, 7, 8}


def determine_new_obj_props(array: np.ndarray, centroid: tuple[float, float]):
    """
    Determine the new centroid and cell positions of an object after a transformation.
    """
    coords = np.array(np.where(array))
    if coords.size == 0:
        return set()
    new_centroid = (np.mean(coords[0]), np.mean(coords[1]))
    new_pixels = {
        (
            round(coord[0] + centroid[0] - new_centroid[0]),
            round(coord[1] + centroid[1] - new_centroid[1]),
        )
        for coord in zip(*coords)
    }
    return new_pixels


def check_valid_grid(grid: np.ndarray) -> bool:
    """
    Check if the grid is valid (i.e. has at least one non-zero value).
    """
    return np.any(grid != 0) and grid.ndim == 2 and grid.size > 0


def get_grid_splits(
    input_array: np.ndarray, split_type: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Check for triple split or half split in the grid.
    """
    # convert set to string if split_type is a set
    if type(split_type) is set and len(split_type) == 1:
        split_type = next(iter(split_type))
    third_splits = check_third_splits(input_array, split_type)
    if not third_splits:
        half_splits = split_half(input_array, split_type)
        return half_splits
    else:
        return split_thirds(input_array, split_type)


def check_third_splits(input_array: np.ndarray, split_type: str) -> bool:
    """
    Check for triple split in the grid by 2 straight lines at the same distance from each other populated by non zero values.
    """
    rows, cols = input_array.shape
    if str(split_type) == "horizontal":
        # Check for horizontal split
        if (rows - 2) % 3 == 0 and rows > 2:
            line1 = (rows - 2) // 3
            line2 = 2 * line1 + 1
            if np.all(input_array[line1, :] != 0) and np.all(
                input_array[line2, :] != 0
            ):
                return True

    if str(split_type) == "vertical":
        # Check for vertical split
        if (cols - 2) % 3 == 0 and cols > 2:
            line1 = (cols - 2) // 3
            line2 = 2 * line1 + 1
            if np.all(input_array[:, line1] != 0) and np.all(
                input_array[:, line2] != 0
            ):
                return True
    return False


def split_thirds(
    grid: np.ndarray, split_type: str
) -> list[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the grid into three parts based on the split axis.
    """
    rows, cols = grid.shape
    if type(split_type) is set and len(split_type) == 1:
        split_type = next(iter(split_type))

    if split_type == "horizontal":
        component_height = (rows - 2) // 3
        mid_start = component_height + 1
        mid_end = mid_start + component_height
        top_third = grid[:component_height, :]
        middle_third = grid[mid_start:mid_end, :]
        bottom_third = grid[mid_end + 1 :, :]
        return [top_third, middle_third, bottom_third]

    if split_type == "vertical":
        component_width = (cols - 2) // 3
        mid_start = component_width + 1
        mid_end = mid_start + component_width
        left_third = grid[:, :component_width]
        middle_third = grid[:, mid_start:mid_end]
        right_third = grid[:, mid_end + 1 :]
        return [left_third, middle_third, right_third]

    return None


def split_half(grid: np.ndarray, split_type: str) -> list[np.ndarray, np.ndarray]:
    """
    Split the grid into two halves based on the split axis.
    Check on the split axis whether length of the grid is even or odd. If odd, remove the middle row/column and split the remaining grid into two halves.
    """
    rows, cols = grid.shape
    if type(split_type) is set and len(split_type) == 1:
        split_type = next(iter(split_type))

    if split_type == "horizontal":
        mid_row = rows // 2
        top_half = grid[:mid_row, :]
        if rows % 2 != 0:
            bottom_half = grid[mid_row + 1 :, :]
        else:
            bottom_half = grid[mid_row:, :]
        if top_half.shape[0] != bottom_half.shape[0]:
            return None
        return [top_half, bottom_half]

    if split_type == "vertical":
        mid_col = cols // 2
        left_half = grid[:, :mid_col]
        if cols % 2 != 0:
            right_half = grid[:, mid_col + 1 :]
        else:
            right_half = grid[:, mid_col:]
        if left_half.shape[1] != right_half.shape[1]:
            return None
        return [left_half, right_half]

    # if split_type == "diagonal":
    #     if rows != cols:
    #         return None
    #     mid_row = rows // 2
    #     mid_col = cols // 2
    #     top_left = grid[:mid_row, :mid_col]
    #     bottom_right = grid[mid_row:, mid_col:]
    #     if top_left.shape != bottom_right.shape:
    #         return None
    #     return [top_left, bottom_right]

    # if split_type == "anti-diagonal":
    #     if rows != cols:
    #         return None
    #     mid_row = rows // 2
    #     mid_col = cols // 2
    #     top_right = grid[:mid_row, mid_col:]
    #     bottom_left = grid[mid_row:, :mid_col]
    #     if top_right.shape != bottom_left.shape:
    #         return None
    # return [top_right, bottom_left]


# -----------------------------------------------------------------------------
# Relation based helper functions
# -----------------------------------------------------------------------------
def exact_translation(
    a: frozenset[tuple[int, int]], b: frozenset[tuple[int, int]]
) -> Optional[tuple[int, int]]:
    """
    Check if two sets are exact translations of each other.
    Return the translation vector if they are, otherwise return None.
    """
    if len(a) != len(b):
        return None

    # Compare the minimum coordinates of both sets to determine the translation vector
    min_a, min_b = min(a), min(b)
    shift = (min_b[0] - min_a[0], min_b[1] - min_a[1])
    if {(r + shift[0], c + shift[1]) for r, c in a} == set(b):
        return shift
    return None


def calculate_distance(obj1_coords: frozenset[tuple[int, int]], obj2_coords: frozenset[tuple[int, int]]) -> int:
    """
    Use Chebyshev distance as working with a grid.
    Scipy cdist can work on sets of pixels.
    """
    return int(distance.cdist(list(obj1_coords), list(obj2_coords), metric="chebyshev").min())


def get_direction_vector(
    centroid_1: tuple[float, float], centroid_2: tuple[float, float]
) -> tuple[int, int]:
    return (
        int(np.sign(centroid_2[0] - centroid_1[0])),
        int(np.sign(centroid_2[1] - centroid_1[1])),
    )
