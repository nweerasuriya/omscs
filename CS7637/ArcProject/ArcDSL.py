"""
Primitive DSL for ArcAgi
Includes basic operations for transformation fo cells and objects
All operations should be generic and be able to ingest either grids or objects or cells if possible
"""

__date__ = "2026-06-10"
__author__ = "NedeeshaWeerasuriya"
__version__ = "0.1"

from typing import Any

import numpy as np
from collections import Counter
from itertools import groupby
import scipy.ndimage as ndi
from MemoryDecorators import decorate_primitive
from ArcMemory import (
    ArcState,
    ObjectState,
    GridState,
    PixelSet,
)
from ArcPruning import Require, Effect
from helpers import (
    determine_new_obj_pixels,
    determine_new_obj_props,
    pixels_to_mask,
    mask_to_pixels,
)


def _update_object_state(
    obj: ObjectState, new_cell_positions: PixelSet, colour: int | None = None
) -> ObjectState:
    """
    Update the object state with new cell positions
    """
    # Recalculate bounding box, centroid, and area based on new cell positions
    if not new_cell_positions:
        return obj
    else:
        rows, cols = zip(*new_cell_positions)
        if any(r < 0 or c < 0 for r, c in new_cell_positions):
            return obj  # Return the original object if any cell position is negative
        bounding_box = (min(rows), min(cols), max(rows), max(cols))
        area = len(new_cell_positions)
        centroid = (sum(rows) / area, sum(cols) / area)

    return ObjectState(
        label_id=obj.label_id,
        colour=colour if colour is not None else obj.colour,
        grid_size=obj.grid_size,
        bounding_box=bounding_box,
        centroid=centroid,
        area=area,
        cell_positions=new_cell_positions,
        hu_moments=obj.hu_moments,
    )


def _update_arcstate_grid(state: ArcState, new_objects: list[ObjectState]) -> ArcState:
    """
    Update the ArcState grid based on the new objects
    """
    new_grid = np.zeros(state.grid_state.dimensions, dtype=int)
    # Order by priority
    new_objects.sort(key=lambda o: o.priority, reverse=False)
    for obj in new_objects:
        for r, c in obj.cell_positions:
            new_grid[r, c] = obj.colour
    return ArcState.from_array(new_grid)


# -----------------------------------------------------------------------------
# Base Logical Operators
# -----------------------------------------------------------------------------
@decorate_primitive("grid")
def logic_not(array: np.ndarray) -> np.ndarray:
    return np.bitwise_not(array)


@decorate_primitive("split_grid", tags={Require.SPLIT_GRID, Effect.GRID_SIZE})
def logic_and(array_list: list, fill_colour: int = 1) -> np.ndarray:
    """
    Perform logical AND operation. Keep original colours unless overlap occurs, then fill with fill_colour
    """
    overlap_mask = np.logical_and.reduce([a != 0 for a in array_list])
    combined = np.maximum.reduce(array_list)
    return np.where(overlap_mask, fill_colour, combined)


@decorate_primitive("split_grid", tags={Require.SPLIT_GRID, Effect.GRID_SIZE})
def logic_and_first_populated(array_list: list) -> np.ndarray:
    result = np.array(array_list[0])
    for array in array_list[1:]:
        result = np.where(result == 0, array, result)
    return result


@decorate_primitive("split_grid", tags={Require.SPLIT_GRID, Effect.GRID_SIZE})
def logic_or(array_list: list, fill_colour: int = 1) -> np.ndarray:
    """
    Perform logical OR operation. Keep original colours unless overlap occurs, then fill with fill_colour
    """
    overlap_mask = np.logical_or.reduce([a != 0 for a in array_list])
    combined = np.maximum.reduce(array_list)
    return np.where(overlap_mask, fill_colour, combined)


@decorate_primitive("split_grid", tags={Require.SPLIT_GRID, Effect.GRID_SIZE})
def logic_nand(array_list: list, fill_colour: int = 1) -> np.ndarray:
    """
    Perform logical NAND operation. Keep original colours unless overlap occurs, then fill with fill_colour
    """
    overlap_mask = np.logical_and.reduce([a != 0 for a in array_list])
    combined = np.maximum.reduce(array_list)
    return np.where(overlap_mask, fill_colour, combined)


@decorate_primitive("split_grid", tags={Require.SPLIT_GRID, Effect.GRID_SIZE})
def logic_xor(array_list: list) -> np.ndarray:
    """
    Perform logical XOR operation. Keep original colours unless overlap occurs, then fill with fill_colour
    """
    count = np.sum([a != 0 for a in array_list], axis=0)
    stacked_arrays = np.stack(array_list, axis=0)
    colours = np.sum(stacked_arrays, axis=0)
    return np.where(count == 1, colours, 0)


@decorate_primitive("split_grid", tags={Require.SPLIT_GRID, Effect.GRID_SIZE})
def logic_nor(array_list: list, fill_colour: int = 1) -> np.ndarray:
    """
    Perform logical NOR operation. Keep original colours unless overlap occurs, then fill with fill_colour
    """
    overlap_mask = np.logical_or.reduce([a != 0 for a in array_list])
    combined = np.maximum.reduce(array_list)
    return np.where(overlap_mask, fill_colour, combined)


@decorate_primitive("split_grid", tags={Require.SPLIT_GRID, Effect.GRID_SIZE})
def logic_and_(array_list: list, fill_colour: int = 1) -> np.ndarray:
    mask = np.logical_and.reduce([a != 0 for a in array_list])
    return np.where(mask, fill_colour, 0)


@decorate_primitive("split_grid", tags={Require.SPLIT_GRID, Effect.GRID_SIZE})
def logic_or_(array_list: list, fill_colour: int = 1) -> np.ndarray:
    mask = np.logical_or.reduce([a != 0 for a in array_list])
    return np.where(mask, fill_colour, 0)


@decorate_primitive("split_grid", tags={Require.SPLIT_GRID, Effect.GRID_SIZE})
def logic_nand_(array_list: list, fill_colour: int = 1) -> np.ndarray:
    mask = np.logical_and.reduce([a != 0 for a in array_list])
    return np.where(mask, 0, fill_colour)


@decorate_primitive("split_grid", tags={Require.SPLIT_GRID, Effect.GRID_SIZE})
def logic_xor_(array_list: list) -> np.ndarray:
    mask = np.logical_xor.reduce([a != 0 for a in array_list])
    return np.where(mask, 1, 0)


@decorate_primitive("split_grid", tags={Require.SPLIT_GRID, Effect.GRID_SIZE})
def logic_nor_(array_list: list, fill_colour: int = 1) -> np.ndarray:
    mask = np.logical_or.reduce([a != 0 for a in array_list])
    return np.where(mask, 0, fill_colour)


@decorate_primitive("split_grid", tags={Require.SPLIT_GRID, Effect.GRID_SIZE})
def logic_xnor_(array_list: list, fill_colour: int = 1) -> np.ndarray:
    mask = np.logical_xor.reduce([a != 0 for a in array_list])
    return np.where(mask, 0, fill_colour)


# -----------------------------------------------------------------------------
# Rotations and Flips
# -----------------------------------------------------------------------------
@decorate_primitive("grid")
def rotate_grid(input_array: np.ndarray, quarter_turn: int) -> np.ndarray:
    return np.rot90(input_array, k=quarter_turn)


@decorate_primitive("object")
def rotate_object(obj: ObjectState, quarter_turn: int) -> np.ndarray:
    rotated = np.rot90(obj.mask, k=quarter_turn)
    new_pixels = determine_new_obj_pixels(rotated, obj.centroid)
    return _update_object_state(obj, PixelSet(new_pixels))


@decorate_primitive("grid")
def flip_horizontal(input_array: np.ndarray) -> np.ndarray:
    return np.fliplr(input_array)


# @decorate_primitive("object")
# def flip_object_horizontal(obj: ObjectState) -> ObjectState:
#     flipped_grid = np.fliplr(obj.mask)
#     new_pixels = determine_new_obj_pixels(flipped_grid, obj.centroid)
#     return _update_object_state(obj, PixelSet(new_pixels))


@decorate_primitive("grid")
def flip_vertical(input_array: np.ndarray) -> np.ndarray:
    return np.flipud(input_array)


# @decorate_primitive("object")
# def flip_object_vertical(obj: ObjectState) -> ObjectState:
#     flipped_grid = np.flipud(obj.mask)
#     new_pixels = determine_new_obj_pixels(flipped_grid, obj.centroid)
#     return _update_object_state(obj, PixelSet(new_pixels))


@decorate_primitive("grid", tags={Require.SQUARE_GRID})
def flip_diagonal(input_array: np.ndarray) -> np.ndarray:
    return np.transpose(input_array)


# @decorate_primitive("object", tags={Require.SQUARE_GRID})
# def flip_object_diagonal(obj: ObjectState) -> ObjectState:
#     flipped_grid = np.transpose(obj.mask)
#     new_pixels = determine_new_obj_pixels(flipped_grid, obj.centroid)
#     return _update_object_state(obj, PixelSet(new_pixels))


@decorate_primitive("grid", tags={Require.SQUARE_GRID})
def flip_anti_diagonal(input_array: np.ndarray) -> np.ndarray:
    return np.fliplr(np.transpose(input_array))


# @decorate_primitive("object", tags={Require.SQUARE_GRID})
# def flip_object_anti_diagonal(obj: ObjectState) -> ObjectState:
#     flipped_grid = np.fliplr(np.transpose(obj.mask))
#     new_pixels = determine_new_obj_pixels(flipped_grid, obj.centroid)
#     return _update_object_state(obj, PixelSet(new_pixels))


# -----------------------------------------------------------------------------
# Colour level functions
# -----------------------------------------------------------------------------
# @decorate_primitive(
#     "grid", tags={Effect.GRID_SIZE, Effect.SHRINK, Require.EMPTY_ROWS_COLS}
# )
# def crop_background_out(
#     input_array: np.ndarray, background_color: int = 0
# ) -> np.ndarray:
#     if not np.any(input_array != background_color):
#         return (
#             input_array  # Return the original array if all values are background color
#         )
#     rows = np.any(input_array != background_color, axis=1)
#     cols = np.any(input_array != background_color, axis=0)
#     return input_array[np.ix_(rows, cols)]


@decorate_primitive("grid", tags={Effect.COLOUR})
def change_colour(
    input_array: np.ndarray, in_colour: int, out_colour: int
) -> np.ndarray:
    grid = input_array.copy()
    return np.where(grid == in_colour, out_colour, grid)


@decorate_primitive("grid", tags={Effect.COLOUR})
def update_colour(
    input_array: np.ndarray, new_colours: int, removed_colours: int
) -> np.ndarray:
    """
    Update the colours in the input array by replacing the removed colours with the new colours.
    """
    grid = input_array.copy()
    grid = np.where(grid == removed_colours, new_colours, grid)
    return grid


# @decorate_primitive(
#     "grid", tags={Effect.COLOUR, Require.REMOVE_COLOURS, Require.MULTI_OBJECTS}
# )
# def remove_colours(
#     input_array: np.ndarray, removed_colour_set: set[int], background_colour: int = 0
# ) -> np.ndarray:
#     """
#     Remove the specified colours from the input array by setting them to background colour 0.
#     """
#     grid = input_array.copy()
#     for colour in removed_colour_set:
#         grid = np.where(grid == colour, background_colour, grid)
#     return grid


@decorate_primitive("grid", tags={Effect.COLOUR, Effect.BACKGROUND_COLOUR})
def change_background_colour(input_array: np.ndarray, out_colour: int) -> np.ndarray:
    """
    Change the background colour of the input array to the new background colour.
    """
    grid = input_array.copy()
    current_background_colour = 0  # Assuming 0 is the current background colour
    return np.where(grid == current_background_colour, out_colour, grid)


@decorate_primitive("grid", tags={Effect.COLOUR})
def fill_default_colour(
    input_array: np.ndarray, out_colour: int, default_colour: int = 1
) -> np.ndarray:
    """
    Fill the input array with the specified out_colour where the default_colour is present.
    """
    grid = input_array.copy()
    return np.where(grid == default_colour, out_colour, grid)


@decorate_primitive("grid", tags={Effect.COLOUR, Effect.BACKGROUND_COLOUR})
def invert_input_colours(input_array: np.ndarray, background_colour: int) -> np.ndarray:
    """
    If two colours are present in the input array, invert them.
    If one colour is present, invert with background colour.
    """
    grid = input_array.copy()
    colours = np.unique(grid)
    # remove background colour 0 from the list of colours
    colours = colours[colours != background_colour]
    if len(colours) == 1:
        # Only one colour present, invert with background colour
        mask = grid == colours[0]
        grid = np.select([mask, ~mask], [background_colour, colours[0]], default=grid)
    elif len(colours) == 2:
        mask_a = grid == colours[0]
        mask_b = grid == colours[1]
        grid = np.select([mask_a, mask_b], [colours[1], colours[0]], default=grid)
    return grid


@decorate_primitive("grid", tags={Effect.CONSERVED_COLOUR})
def swap_colours(input_array: np.ndarray, background_colour: int) -> np.ndarray:
    """
    Swap the colours in the input array. If two colours are present, swap them.
    If one colour is present, swap with background colour.
    """
    grid = input_array.copy()
    colours = np.unique(grid)
    if len(colours) == 1:
        mask = grid == colours[0]
        grid = np.select([mask, ~mask], [background_colour, colours[0]], default=grid)
    elif len(colours) == 2:
        mask_a = grid == colours[0]
        mask_b = grid == colours[1]
        grid = np.select([mask_a, mask_b], [colours[1], colours[0]], default=grid)
    return grid


@decorate_primitive("grid", tags={Effect.COLOUR, Require.BLACK_PRESENT})
def colour_grey_to_black(input_array: np.ndarray, grey: int = 5) -> np.ndarray:
    if not np.any(input_array == grey):
        return input_array
    grid = input_array.copy()
    return np.where(grid == grey, 0, grid)


@decorate_primitive("object", tags={Effect.COLOUR})
def recolour_object(obj: ObjectState, new_colour_obj: int) -> ObjectState:
    return obj._replace(colour=new_colour_obj)


@decorate_primitive("state", tags={Effect.COLOUR})
def recolour_most_common(
    state: ArcState, colour_count: Counter, background_colour: int
) -> np.ndarray:
    """
    Recolour grid to most common colour in the input array. If two colours are present, recolour to the most common colour.
    """
    grid = state.grid_state.as_array.copy()
    if not colour_count:
        return grid
    most_common_colour, _ = colour_count.most_common(1)[0]
    if most_common_colour == background_colour:
        return grid
    return np.where(grid != background_colour, most_common_colour, grid)


@decorate_primitive("grid", tags={Effect.GROWTH})
def connect_same_colour(input_array: np.ndarray, direction_str: str) -> np.ndarray:
    """
    Connects pixels of the specified colour that are not connected by filling in the gaps between them.
    Only connect in straight lines (horizontal, vertical, diagonal)
    """
    grid = input_array.copy()
    rows, cols = grid.shape
    colour_list = list(np.unique(grid))

    for out_colour in colour_list:
        if direction_str == "horizontal":
            # Check horizontal
            for row in range(rows):
                colour_pixels = np.where(grid[row, :] == out_colour)[0]
                if len(colour_pixels) >= 2:
                    start = colour_pixels[0]
                    end = colour_pixels[-1]
                    grid[row, start : end + 1] = grid[row, start]

        if direction_str == "vertical":
            # Check vertical
            for col in range(cols):
                colour_pixels = np.where(grid[:, col] == out_colour)[0]
                if len(colour_pixels) >= 2:
                    start = colour_pixels[0]
                    end = colour_pixels[-1]
                    if grid[start, col] == grid[end, col]:
                        grid[start : end + 1, col] = grid[start, col]

        if direction_str == "diagonal":
            # Check diagonal
            for row in range(rows):
                for col in range(cols):
                    if grid[row, col] != out_colour:
                        continue
                    # Check diagonal down-right
                    r, c = row + 1, col + 1
                    while r < rows and c < cols and grid[r, c] != out_colour:
                        r += 1
                        c += 1
                    if r < rows and c < cols and grid[r, c] == out_colour:
                        for i in range(r - row + 1):
                            grid[row + i, col + i] = out_colour

        if direction_str == "anti-diagonal":
            # Check anti-diagonal
            for row in range(rows):
                for col in range(cols):
                    if grid[row, col] != out_colour:
                        continue
                    # Check diagonal up-right
                    r, c = row - 1, col + 1
                    while r >= 0 and c < cols and grid[r, c] != out_colour:
                        r -= 1
                        c += 1
                    if r >= 0 and c < cols and grid[r, c] == out_colour:
                        for i in range(row - r + 1):
                            grid[row - i, col + i] = out_colour
    return grid


@decorate_primitive("grid", tags={Effect.COLOUR})
def fill_overlap_with_original_grid(
    input_array: np.ndarray,
    original_grid: GridState,
    out_colour: int,
    background_colour: int,
) -> np.ndarray:
    """
    Compare two grids. Where input_array has a non-background pixel that differs
    from original_grid, fill with out_colour. Otherwise (background pixels, or
    pixels matching original_grid), keep original_grid's colour.
    """
    if input_array.shape != original_grid.as_array.shape:
        return input_array
    diff_mask = (input_array != original_grid.as_array) & (
        original_grid.as_array != background_colour
    )
    filled_array = np.where(diff_mask, out_colour, input_array)
    return filled_array


# -----------------------------------------------------------------------------
# Mirrors
# -----------------------------------------------------------------------------
@decorate_primitive("grid")
def mirror_horizontal(input_array: np.ndarray) -> np.ndarray:
    """
    Mirror the input array horizontally
    """
    return np.fliplr(input_array)


@decorate_primitive("grid")
def mirror_vertical(input_array: np.ndarray) -> np.ndarray:
    """
    Mirror the input array vertically
    """
    return np.flipud(input_array)


@decorate_primitive("grid", tags={Require.SQUARE_GRID})
def mirror_diagonal(input_array: np.ndarray) -> np.ndarray:
    """
    Mirror the input array diagonally, reflecting the pixels across the diagonal axis
    """
    return np.transpose(input_array)


@decorate_primitive("grid", tags={Require.SQUARE_GRID})
def mirror_anti_diagonal(input_array: np.ndarray) -> np.ndarray:
    """
    Mirror the input array anti-diagonally, reflecting the pixels across the anti-diagonal axis
    """
    return np.fliplr(np.transpose(input_array))


# -----------------------------------------------------------------------------
# Other Grid Primitives
# -----------------------------------------------------------------------------
@decorate_primitive("grid", tags={Effect.GRID_SIZE, Effect.GROWTH})
def pad_grid(
    input_array: np.ndarray, pad_width: int = 1, pad_value: int = 0
) -> np.ndarray:
    """
    Pad the input array with the specified pad_width and pad_value.
    """
    return np.pad(
        input_array, pad_width=pad_width, mode="constant", constant_values=pad_value
    )


@decorate_primitive("grid", tags={Effect.GROWTH, Effect.OBJECT_COUNT})
def unfold_grid_vertical(input_array: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Unfolds the grid vertically, producing a new grid with the same number of rows but double the number of columns.
    The array is mirrored along the vertical axis, and the two halves are concatenated side by side.
    """
    return np.concatenate((input_array, np.fliplr(input_array)), axis=axis)


# @decorate_primitive("grid", tags={Effect.GROWTH, Effect.OBJECT_COUNT})
# def unfold_mirror_vertical(input_array: np.ndarray) -> np.ndarray:
#     """
#     Unfolds the grid vertically, but reflects the original grid along the vertical axis.
#     """
#     left_half = input_array.copy()
#     right_half = np.fliplr(input_array)
#     return np.concatenate((left_half, right_half), axis=1)


@decorate_primitive("grid", tags={Effect.GROWTH, Effect.OBJECT_COUNT})
def unfold_grid_horizontal(input_array: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Unfolds the grid horizontally, producing a new grid with the same number of columns but double the number of rows.
    The array is mirrored along the horizontal axis, and the two halves are concatenated one on top of the other.
    """
    return np.concatenate((input_array, np.flipud(input_array)), axis=axis)


# @decorate_primitive("grid", tags={Effect.GROWTH})
# def unfold_mirror_horizontal(
#     input_array: np.ndarray,
# ) -> np.ndarray:
#     """
#     Unfolds the grid horizontally, but reflects the original grid along the horizontal axis.
#     """
#     lower_half = input_array.copy()
#     upper_half = np.flipud(input_array)
#     return np.concatenate((upper_half, lower_half), axis=0)


# @decorate_primitive("grid", tags={Effect.GRID_SIZE, Effect.GROWTH})
# def unfold_half_grid_vertical(input_array: np.ndarray, axis: int = 1) -> np.ndarray:
#     """
#     Unfolds the grid vertically, producing a new grid with the same number of rows but double the number of columns.
#     The array is mirrored along the vertical axis, and the two halves are concatenated side by side.
#     """
#     half_cols = input_array.shape[1] // 2
#     left_half = input_array[:, :half_cols]
#     return np.concatenate((left_half, np.fliplr(left_half)), axis=axis)


# @decorate_primitive("grid", tags={Effect.GRID_SIZE, Effect.GROWTH})
# def unfold_half_grid_horizontal(input_array: np.ndarray, axis: int = 0) -> np.ndarray:
#     """
#     Unfolds the grid horizontally, producing a new grid with the same number of columns but double the number of rows.
#     The array is mirrored along the horizontal axis, and the two halves are concatenated one on top of the other.
#     """
#     half_rows = input_array.shape[0] // 2
#     top_half = input_array[:half_rows, :]
#     return np.concatenate((top_half, np.flipud(top_half)), axis=axis)


@decorate_primitive("grid", tags={Require.EMPTY_TOP_HALF})
def mirror_for_empty_top_half(input_array: np.ndarray) -> np.ndarray:
    """
    If the top half of the input array is empty (only background colour), mirror the bottom half to fill the top half.
    """
    rows, cols = input_array.shape
    bottom_half = input_array[rows // 2 :, :]

    mirrored_bottom_half = np.flipud(bottom_half)
    return np.vstack((mirrored_bottom_half, bottom_half))


@decorate_primitive(
    "grid", tags={Effect.GRID_SIZE, Effect.SHRINK, Effect.CONSERVED_COLOUR}
)
def swap_colour_shrink_grid(
    input_array: np.ndarray, background_colour: int
) -> np.ndarray:
    """
    Swap colours if two colours are present in the input array
    Shrink grid by removing the outer cells for rows and cols
    """
    colours = np.unique(input_array)
    colours = colours[colours != background_colour]
    if len(colours) == 2:
        mask_a = input_array == colours[0]
        mask_b = input_array == colours[1]
        new_array = np.select(
            [mask_a, mask_b], [colours[1], colours[0]], default=input_array
        )
        return new_array[1:-1, 1:-1]
    return input_array[1:-1, 1:-1]


@decorate_primitive(
    "grid", tags={Effect.GRID_SIZE, Effect.GROWTH, Require.SCALE_FACTOR}
)
def expand_grid_as_stair(
    input_array: np.ndarray,
    row_size: bool,
    col_size: bool,
    row_col_scale: int,
    background_colour: int,
) -> np.ndarray:
    """
    Expand the grid by the specified row_col_scale, duplicating existing cells to fill the new grid size.
    """
    # If grid is the same size no expansion needed
    if row_size and col_size:
        return input_array.copy()
    elif row_size:
        # Expand only columns
        scale_factor = input_array.shape[0] / row_col_scale
        new_grid = np.repeat(input_array, scale_factor, axis=1)
    elif col_size:
        # Expand only rows
        scale_factor = input_array.shape[1] * row_col_scale
        new_grid = np.repeat(input_array, scale_factor, axis=0)
    else:
        # Expand both rows and columns
        new_grid = np.repeat(
            np.repeat(input_array, row_col_scale, axis=0), row_col_scale, axis=1
        )
    # Change the block of pixels to a stairs shape if possible
    obj_mask = new_grid != background_colour
    if np.any(obj_mask):
        obj_colour = np.unique(new_grid[new_grid != background_colour])[0]
        # Get the bounding box of the object
        rows, cols = np.where(obj_mask)
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        for r in range(min_row + 1, max_row + 1):
            for c in range(max_col, new_grid.shape[1]):
                if new_grid[r - 1, c - 1] == obj_colour:
                    new_grid[r, c] = obj_colour

        return new_grid
    else:
        return new_grid


# @decorate_primitive("grid", tags={Effect.GRID_SIZE})
# def slice_grid(input_array: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
#     """
#     Slice the input array to the specified output shape.
#     """
#     return input_array[: out_shape[0], : out_shape[1]]


@decorate_primitive(
    "grid",
    tags={Effect.GRID_SIZE, Effect.SHAPE, Effect.SHRINK, Effect.CONSERVED_COLOUR},
)
def reshape_grid(
    input_array: np.ndarray, out_shape: tuple[int, int], background_colour: int
) -> np.ndarray:
    """
    Resize the input array to the specified output shape.
    If pixels spill over move them to the next row or column when output is smaller.
    """
    if np.unique(input_array).size > 2:
        return input_array
    out_rows, out_cols = out_shape
    if input_array.shape[0] == out_rows and input_array.shape[1] == out_cols:
        return input_array.copy()
    # Create a new output array filled with background colour
    output_array = np.full(out_shape, background_colour, dtype=input_array.dtype)

    # if input_array is larger than out_shape, we need to flatten it and then reshape it
    if input_array.shape[0] > out_rows or input_array.shape[1] > out_cols:
        flat_input = input_array.flatten()
        non_zero_pixels = flat_input[flat_input != background_colour]

        # Fill with coloured pixels overflowing to the next row or column start from top left
        for pixel in non_zero_pixels:
            for row in range(out_rows):
                for col in range(out_cols):
                    if output_array[row, col] == background_colour:
                        output_array[row, col] = pixel
                        break
                else:
                    continue
                break
    else:
        # If input_array is smaller or the same size than out_shape then add pixels in original position in new grid
        for row in range(input_array.shape[0]):
            for col in range(input_array.shape[1]):
                if input_array[row, col] != background_colour:
                    output_array[row, col] = input_array[row, col]
    return output_array


@decorate_primitive(
    "grid", tags={Effect.GRID_SIZE, Effect.SHRINK, Require.EMPTY_ROWS_COLS}
)
def remove_empty_outer_rows_and_columns(
    input_array: np.ndarray, background_colour: int
) -> np.ndarray:
    """
    Remove empty outer rows and columns from the input array. Defined as only background colour
    The removed rows and columns are only those that are on the outer edges of the array, not any empty rows or columns that may be in the middle of the array.
    """
    rows, cols = input_array.shape
    top, bottom = 0, rows - 1
    left, right = 0, cols - 1
    new_array = input_array.copy()
    # Remove empty outer rows
    while top <= bottom and np.all(new_array[top, :] == background_colour):
        new_array = np.delete(new_array, top, axis=0)
        bottom -= 1
    while bottom >= top and np.all(new_array[bottom, :] == background_colour):
        new_array = np.delete(new_array, bottom, axis=0)
        bottom -= 1
    # Remove empty outer columns
    while left <= right and np.all(new_array[:, left] == background_colour):
        new_array = np.delete(new_array, left, axis=1)
        right -= 1
    while right >= left and np.all(new_array[:, right] == background_colour):
        new_array = np.delete(new_array, right, axis=1)
        right -= 1
    return new_array


# -----------------------------------------------------------------------------
# Object Primitives
# -----------------------------------------------------------------------------


# # TODO: Account for cells not bordering bounding box
# @decorate_primitive("object", tags={Effect.SHAPE})
# def fill_bounding_box(
#     obj: ObjectState,
# ) -> ObjectState:
#     """
#     Fill the object with the specified colour within bounding box
#     """
#     bounding_box = obj.bounding_box
#     x_min, y_min, x_max, y_max = bounding_box
#     # Get new cell positions within the bounding box
#     new_cell_positions = set()
#     for x in range(x_min, x_max):
#         for y in range(y_min, y_max):
#             new_cell_positions.add((x, y))

#     return _update_object_state(obj, PixelSet(new_cell_positions))


@decorate_primitive("object")
def fill_enclosed_area(
    obj: ObjectState,
    out_colour: int,
) -> ObjectState | list[ObjectState]:
    """
    Fill any enclosed area of the object with the specified colour.
    Return a new ObjectState with the filled area (this will be a different object state than the input object as could be a different colour and area)
    """
    # Create a binary mask of the object
    binary_mask = obj.mask != 0
    filled_mask = ndi.binary_fill_holes(binary_mask)
    # Create a new array with the filled areas set to the specified colour
    filled_array = np.where(filled_mask, out_colour, obj.mask)
    new_pixels = mask_to_pixels(filled_array)
    if out_colour == obj.colour:
        combined_pixels = obj.cell_positions.union(new_pixels)
        return _update_object_state(obj, combined_pixels)
    else:
        bbox, centroid, hu = determine_new_obj_props(new_pixels)
        return [
            ObjectState(
                label_id=obj.label_id + np.random.randint(1, 1000),
                colour=out_colour,
                grid_size=obj.grid_size,
                bounding_box=bbox,
                centroid=centroid,
                hu_moments=hu,
                area=len(new_pixels),
                cell_positions=PixelSet(new_pixels),
            ),
            obj,
        ]


# @decorate_primitive("object", tags={Effect.SHAPE})
# def crop_object(obj: ObjectState) -> ObjectState:
#     """
#     Crop the object to its bounding box
#     """
#     x_min, y_min, x_max, y_max = obj.bounding_box
#     new_cell_positions = set()
#     for cell in obj.cell_positions:
#         x, y = cell
#         if x_min <= x < x_max and y_min <= y < y_max:
#             new_cell_positions.add(cell)

#     return _update_object_state(obj, PixelSet(new_cell_positions))


# @decorate_primitive("object", tags={Effect.GROWTH})
# def grow_object(obj: ObjectState, scale: int) -> ObjectState:
#     """
#     Grow the object by scale factor.
#     """
#     cell_positions = obj.cell_positions
#     new_cell_positions = set(cell_positions)
#     for cell in cell_positions:
#         x, y = cell
#         for dx in range(-scale, scale + 1):
#             for dy in range(-scale, scale + 1):
#                 new_cell_positions.add((x + dx, y + dy))
#     return _update_object_state(obj, PixelSet(new_cell_positions))


def _surround_object(obj: ObjectState, out_colour: int, fill_type: str) -> ObjectState:
    """
    Surround the object with a border of the specified colour.
    8-connectivity
    """
    cell_positions = obj.cell_positions
    new_cell_positions = set(cell_positions)

    # Find the interior of the object
    rows, cols = obj.grid_size
    mask = np.zeros((rows, cols), dtype=bool)
    for r, c in cell_positions:
        mask[r, c] = True
    inside = ndi.binary_fill_holes(mask)
    inside = set(zip(*np.where(inside)))

    # Set condition of fill type
    if fill_type == "interior":
        fill_condition = lambda neighbor: neighbor in inside
    elif fill_type == "exterior":
        fill_condition = lambda neighbor: neighbor not in inside
    else:
        return obj  # Invalid fill_type, return original object

    for cell in cell_positions:
        x, y = cell
        neighbors = [
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1),
            (x - 1, y - 1),
            (x - 1, y + 1),
            (x + 1, y - 1),
            (x + 1, y + 1),
        ]
        for neighbor in neighbors:
            if fill_condition(neighbor) and neighbor not in cell_positions:
                new_cell_positions.add(neighbor)
    if out_colour == obj.colour:
        return _update_object_state(obj, PixelSet(new_cell_positions))
    else:
        return [
            ObjectState(
                label_id=obj.label_id + np.random.randint(1, 1000),
                colour=out_colour,
                grid_size=obj.grid_size,
                bounding_box=obj.bounding_box,
                centroid=obj.centroid,
                area=len(new_cell_positions),
                cell_positions=PixelSet(new_cell_positions),
            ),
            obj,
        ]


@decorate_primitive("object", tags={Effect.GROWTH})
def surround_object_with_exterior(obj: ObjectState, out_colour: int) -> ObjectState:
    return _surround_object(obj, out_colour, fill_type="exterior")


@decorate_primitive("object", tags={Require.CLOSED_OBJECT, Effect.OBJECT_COUNT})
def surround_interior_obj(obj: ObjectState, out_colour: int) -> ObjectState:
    return _surround_object(obj, out_colour, fill_type="interior")


# @object_primitive(tags={Effect.SHRINK})
# def shrink_object(
#     obj: ObjectState, scale: int = 1, direction_vector: tuple[int, int] = (0, 0)
# ) -> ObjectState:
#     """
#     Shrink the object by scale factor.
#     If direction vector is provided, shrink in that direction
#     """
#     if direction_vector != {(0, 0)}:
#         cell_positions = obj.cell_positions
#         new_cell_positions = set()
#         for cell in cell_positions:
#             x, y = cell
#             for dx in range(-scale, scale + 1):
#                 for dy in range(-scale, scale + 1):
#                     if (dx, dy) in direction_vector and (
#                         x + dx,
#                         y + dy,
#                     ) in cell_positions:
#                         new_cell_positions.add((x + dx, y + dy))
#         return _update_object_state(obj, PixelSet(new_cell_positions))
#     else:
#         return obj


@decorate_primitive("object", tags={Effect.GROWTH, Require.DIRECTIONALITY})
def add_line_to_object_until_grid_end(
    obj: ObjectState,
    direction_vector: tuple[int, int],
):
    """
    Add a line to the object in the specified direction until the edge of the grid is reached.
    """
    cell_positions = obj.cell_positions
    grid_shape = obj.grid_size
    dx, dy = direction_vector

    if not cell_positions or (dx == 0 and dy == 0):
        return obj

    # Get furthest point in the direction of the line
    start_point = max(cell_positions, key=lambda p: (p[0] * dx + p[1] * dy))
    new_cell_positions = set(cell_positions)
    x, y = start_point
    while 0 <= x < grid_shape[0] and 0 <= y < grid_shape[1]:
        new_cell_positions.add((x, y))
        x += dx
        y += dy

    return _update_object_state(obj, PixelSet(new_cell_positions))


@decorate_primitive("object", tags={Effect.GROWTH, Require.DIRECTIONALITY})
def add_line_to_object(
    obj: ObjectState,
    direction_vector: tuple[int, int],
    out_colour: int,
    scale: int,
) -> ObjectState | list[ObjectState]:
    """
    Add a line to the object in the specified direction
    Single line should extend by the scale number.
    Start from edge of the object in the direction of the line
    """
    cell_positions = obj.cell_positions
    grid_shape = obj.grid_size
    dx, dy = direction_vector

    if not cell_positions:
        return obj

    # Get furthest point in the direction of the line
    start_point = max(cell_positions, key=lambda p: (p[0] * dx + p[1] * dy))
    new_cell_positions = set(cell_positions)
    x, y = start_point
    for i in range(1, scale + 1):
        new_x = x + i * dx
        new_y = y + i * dy
        if 0 <= new_x < grid_shape[0] and 0 <= new_y < grid_shape[1]:
            new_cell_positions.add((new_x, new_y))
        else:
            break
    if out_colour == obj.colour:
        return _update_object_state(obj, PixelSet(new_cell_positions))
    else:
        bbox, centroid, hu = determine_new_obj_props(new_cell_positions)
        return [
            ObjectState(
                label_id=obj.label_id + np.random.randint(1, 1000),
                colour=out_colour,
                grid_size=obj.grid_size,
                bounding_box=bbox,
                centroid=centroid,
                area=len(new_cell_positions),
                cell_positions=PixelSet(new_cell_positions),
                hu_moments=hu,
            ),
            obj,
        ]


@decorate_primitive("object", tags={Effect.OBJECT_COUNT})
def add_new_obj_line(obj: ObjectState, out_colour: int) -> ObjectState:
    """
    Add a new object that is a straight line extending from original object.
    """
    rows, cols = obj.grid_size
    dr, dc = obj.pointed_direction
    if (dr, dc) == (0, 0) or obj.area == 0:
        return obj
    if dr == 0:
        edge_col = (
            max(c for r, c in obj.cell_positions)
            if dc > 0
            else min(c for r, c in obj.cell_positions)
        )
        end_row = max(r for r, c in obj.cell_positions if c == edge_col)
        start = (end_row, edge_col)

    else:
        edge_row = (
            max(r for r, c in obj.cell_positions)
            if dr > 0
            else min(r for r, c in obj.cell_positions)
        )
        end_col = max(c for r, c in obj.cell_positions if r == edge_row)
        start = (edge_row, end_col)
    new_cell_positions = set()
    r, c = start
    for i in range(1, max(rows, cols)):
        nr, nc = r + i * dr, c + i * dc
        if 0 <= nr < rows and 0 <= nc < cols:
            new_cell_positions.add((nr, nc))
        else:
            break
    if not new_cell_positions:
        return obj

    bbox, centroid, hu = determine_new_obj_props(new_cell_positions)
    return [
        ObjectState(
            label_id=np.random.randint(100, 500),
            colour=out_colour,
            grid_size=obj.grid_size,
            bounding_box=bbox,
            centroid=centroid,
            area=len(new_cell_positions),
            cell_positions=PixelSet(new_cell_positions),
            hu_moments=hu,
        ),
        obj,
    ]


@decorate_primitive("object", tags={Effect.TRANSLATE, Require.DIRECTIONALITY})
def translate_object(
    obj: ObjectState, direction_vector: tuple[int, int], scale: int
) -> ObjectState:
    """
    Translate the object in the specified direction by the scale factor
    """
    cell_positions = obj.cell_positions
    new_cell_positions = set()
    for cell in cell_positions:
        x, y = cell
        dx, dy = direction_vector
        new_cell_positions.add((x + scale * dx, y + scale * dy))
    return _update_object_state(obj, PixelSet(new_cell_positions))


@decorate_primitive("object", tags={Require.DIRECTIONALITY, Effect.SHAPE})
def extend_object(obj: ObjectState, direction_str: str, scale: int) -> ObjectState:
    """
    Extend the object in the specified direction by the scale factor
    """
    cell_positions = obj.cell_positions
    new_cell_positions = set(cell_positions)
    direction_map = {
        "horizontal": (1, 0),
        "vertical": (0, 1),
        "diagonal": ((1, 1), (1, -1)),
        "anti-diagonal": ((-1, 1), (-1, -1)),
    }
    for cell in cell_positions:
        x, y = cell
        directions = direction_map[direction_str]
        if isinstance(directions[0], tuple):
            for dx, dy in directions:
                for i in range(1, scale + 1):
                    new_cell_positions.add((x + i * dx, y + i * dy))
                    new_cell_positions.add((x - i * dx, y - i * dy))
        else:
            dx, dy = directions
            for i in range(1, scale + 1):
                new_cell_positions.add((x + i * dx, y + i * dy))
                new_cell_positions.add((x - i * dx, y - i * dy))
    return _update_object_state(obj, PixelSet(new_cell_positions))


@decorate_primitive("object", tags={Effect.GROWTH, Effect.SHAPE})
def grow_all_diag(obj: ObjectState) -> ObjectState:
    """
    Grow the object in all diagonal directions until the grid boundary is reached
    """
    cell_positions = obj.cell_positions
    new_cell_positions = set(cell_positions)
    grid_shape = obj.grid_size
    rows, cols = grid_shape

    for cell in cell_positions:
        x, y = cell
        # Grow in all diagonal directions
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            new_x, new_y = x + dx, y + dy
            while 0 <= new_x < rows and 0 <= new_y < cols:
                new_cell_positions.add((new_x, new_y))
                new_x += dx
                new_y += dy

    return _update_object_state(obj, PixelSet(new_cell_positions))


@decorate_primitive(
    "object", tags={Require.DIRECTIONALITY, Effect.OBJECT_COUNT, Require.SINGLE_PIXEL}
)
def connect_objects_diag_specific(
    obj: ObjectState, direction_vector: tuple, out_colour: int, target_pos: tuple
):
    """
    Specific transformation where we create a joining line between this object to another pixel on the grid
    The line first moves diagonally in the direction of the other pixel, then moves horizontally or vertically to reach the other pixel when 1 row/col away from the other pixel
    Assume object is a single pixel
    """
    if direction_vector == (0, 0) or obj.area != 1:
        return obj
    grid_shape = obj.grid_size
    new_cell_positions = set(obj.cell_positions)
    dx, dy = direction_vector
    start_point = list(obj.cell_positions)[0]
    x, y = start_point
    target_x, target_y = int(target_pos[0]), int(target_pos[1])

    break_condition = None
    # Move diagonally until we are 1 row/col away from the target pixel
    while True:
        x += dx
        y += dy
        if 0 <= x < grid_shape[0] and 0 <= y < grid_shape[1]:
            new_cell_positions.add((x, y))
        else:
            break
        if abs(x - target_x) < 2:
            break_condition = "row"
            break
        elif abs(y - target_y) < 2:
            break_condition = "col"
            break

    # Move horizontally or vertically to reach the target pixel
    if break_condition == "row":
        step = 1 if target_y > y else -1
        for c in range(y, target_y + step, step):
            if 0 <= x < grid_shape[0] and 0 <= c < grid_shape[1]:
                new_cell_positions.add((x, c))
    elif break_condition == "col":
        step = 1 if target_x > x else -1
        for r in range(x, target_x + step, step):
            if 0 <= r < grid_shape[0] and 0 <= y < grid_shape[1]:
                new_cell_positions.add((r, y))
    else:
        return obj

    # Remove last entry in new_cell_positions regardless
    if break_condition == "row":
        new_cell_positions.remove((x, target_y))
    elif break_condition == "col":
        new_cell_positions.remove((target_x, y))

    if out_colour == obj.colour:
        return _update_object_state(
            obj, PixelSet(new_cell_positions), colour=out_colour
        )
    else:
        bbox, centroid, hu = determine_new_obj_props(new_cell_positions)
        return [
            ObjectState(
                label_id=obj.label_id + np.random.randint(1, 1000),
                colour=out_colour,
                grid_size=obj.grid_size,
                bounding_box=bbox,
                centroid=centroid,
                hu_moments=hu,
                area=len(new_cell_positions),
                cell_positions=PixelSet(new_cell_positions),
            ),
            obj,
        ]


@decorate_primitive(
    "state", tags={Effect.OBJECT_COUNT, Effect.SHAPE, Require.SINGLE_PIXEL}
)
def remove_single_pixels_grid(
    state: ArcState,
    background_colour: int,
) -> np.ndarray:
    """
    Remove single pixels from the grid, returning a new grid with no single pixels
    """
    objects = state.objects
    remove_objects = []
    for obj in objects:
        if obj.area == 1:
            remove_objects.append(obj.cell_positions)
    new_grid = state.to_array()
    rows, cols = new_grid.shape
    for cell_positions in remove_objects:
        for cell in cell_positions:
            x, y = cell
            if 0 <= x < rows and 0 <= y < cols:
                new_grid[x, y] = background_colour
    return new_grid


# @decorate_primitive("object", tags={Effect.SHAPE})
# def remove_block_object(
#     obj: ObjectState,
# ) -> ObjectState:
#     """
#     If an object is a block (all pixels are connected), remove it as an object state.
#     Return objects state with no cell positions
#     """
#     if obj.is_block:
#         return _update_object_state(obj, PixelSet(set()))
#     else:
#         return obj


@decorate_primitive("object", tags={Effect.OBJECT_COUNT})
def remove_object(
    obj: ObjectState,
) -> ObjectState:
    """
    Remove the object, returning an empty object state
    """
    return _update_object_state(obj, PixelSet(set()))


# -----------------------------------------------------------------------------
# New Patterns
# -----------------------------------------------------------------------------
@decorate_primitive("grid", tags={Require.SQUARE_GRID, Require.EMPTY_GRID})
def create_spiral_pattern(
    input_array: np.ndarray, out_colour: int, clockwise: bool, corner_position: str
) -> np.ndarray:
    """
    Create a spiral pattern starting from the top left corner of the input array
    Account non square arrays by filling by constraint of the smaller dimension with the spiral pattern
    """
    grid = input_array.copy()
    n, m = grid.shape
    dirs = (
        [(0, 1), (1, 0), (0, -1), (-1, 0)]
        if clockwise
        else [(1, 0), (0, 1), (-1, 0), (0, -1)]
    )
    # Create a list of lengths for each direction in the spiral.
    lengths = [m - 1, n - 1, m - 1] + [
        k for k in range(min(n, m) - 3, 0, -2) for _ in (0, 1)
    ]
    # Determine the starting position based on the corner_position argument
    map = {
        "top-left": (0, 0),
        "top-right": (0, m - 1),
        "bottom-left": (n - 1, 0),
        "bottom-right": (n - 1, m - 1),
    }
    r, c = map[corner_position]
    grid[r, c] = out_colour
    for i, L in enumerate(lengths):
        dr, dc = dirs[i % 4]
        for _ in range(L):
            r, c = r + dr, c + dc
            if 0 <= r < n and 0 <= c < m:
                grid[r, c] = out_colour
    return grid


# -----------------------------------------------------------------------------
# Count based primitives
# -----------------------------------------------------------------------------
@decorate_primitive(
    "state", tags={Effect.COLOUR, Effect.OBJECT_COUNT, Require.MULTI_OBJECTS}
)
def remove_most_common_colour_obj(state: ArcState, colour_count: Counter) -> ArcState:
    """
    Remove the most common colour from the arc state, returning remaining objects.
    """
    if not colour_count:
        return state
    most_common_colour = colour_count.most_common(1)[0][0]
    new_objects = [obj for obj in state.objects if obj.colour != most_common_colour]
    return _update_arcstate_grid(state, new_objects)


@decorate_primitive(
    "state", tags={Effect.GRID_SIZE, Effect.SHAPE, Require.MULTI_OBJECTS}
)
def count_colours(
    state: ArcState,
    colour_count: Counter,
    out_shape: tuple[int, int],
    background_colour: int,
) -> np.ndarray:
    """
    Use the colour counter to return a grid with the counts
    Return as a horizontal/vertical line (dependent on output shape) of pixels with a new row for each colour and the count as the length of the line
    """
    if not colour_count:
        return None
    if (
        len(colour_count) == 1
        and colour_count.most_common(1)[0][0] == background_colour
    ):
        return None
    # check if rows are greater than columns
    rows, cols = out_shape
    n_colours = len(colour_count)
    max_count = max(colour_count.values())
    # Horizontal lines
    if rows <= cols:
        colour_count = Counter(
            dict(sorted(colour_count.items(), key=lambda x: x[1], reverse=False))
        )
        output_array = np.zeros((n_colours, max_count), dtype=int)
        for i, (colour, count) in enumerate(colour_count.items()):
            output_array[i, :count] = colour
    # Otherwise, vertical lines
    else:
        colour_count = Counter(
            dict(sorted(colour_count.items(), key=lambda x: x[1], reverse=True))
        )
        output_array = np.zeros((max_count, n_colours), dtype=int)
        for i, (colour, count) in enumerate(colour_count.items()):
            output_array[:count, i] = colour
    return output_array


# @decorate_primitive(
#     "state",
#     tags={Effect.GRID_SIZE, Effect.SHAPE, Require.MULTI_OBJECTS, Require.BLOCK_OBJECTS},
# )
# def count_colours_block_objects(
#     state: ArcState, out_shape: tuple[int, int], background_colour: int
# ):
#     """
#     Count the colours of block objects in the state and return a grid with the counts
#     """
#     colour_count = Counter()
#     for obj in state.objects:
#         if obj.is_block:
#             colour_count[obj.colour] += 1
#     if len(colour_count) == 0:
#         return None
#     # Remove most common colour
#     most_common_colour = colour_count.most_common(1)[0][0]
#     colour_count.pop(most_common_colour)
#     new_grid = count_colours(
#         state,
#         colour_count,
#         out_shape,
#         max_rows=out_shape[0],
#         max_cols=out_shape[1],
#         background_colour=background_colour,
#     )
#     return new_grid


@decorate_primitive("object", tags={Require.CLOSED_OBJECT})
def fill_interior_colour(obj: ObjectState, colour_count: Counter) -> list[ObjectState]:
    """
    Fill the inside of an object with the most common colour found inside the object
    Produce a new object state which is the filled area of the original object.
    Return both the original object and the new filled object as a list of ObjectStates
    """
    if not colour_count:
        return [obj]
    most_common_colour = colour_count.most_common(1)[0][0]
    # Use binary mask to fill in original object
    mask = pixels_to_mask(obj.cell_positions, obj.grid_size)
    fill_mask = ndi.binary_fill_holes(mask).astype(int)
    fill_pixels = mask_to_pixels(fill_mask)
    # remove the orginal object pixels from the interior pixels to get only the filled area
    interior_pixels = fill_pixels - obj.cell_positions
    new_cell_positions = set(interior_pixels)

    new_object = ObjectState(
        label_id=obj.label_id + 100,
        colour=most_common_colour,
        grid_size=obj.grid_size,
        bounding_box=obj.bounding_box,
        centroid=obj.centroid,
        area=len(new_cell_positions),
        cell_positions=PixelSet(new_cell_positions),
        priority=obj.priority + 1,
    )
    return [obj, new_object]


def _unfill_object(
    obj: ObjectState,
) -> ObjectState:
    """
    Unfill the object, only keeping the border of the object and setting the inside to background colour
    """
    cell_positions = obj.cell_positions
    new_cell_positions = set()
    # Get border cells only
    for cell in cell_positions:
        x, y = cell
        neighbors = [
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1),
        ]
        if any(neighbor not in cell_positions for neighbor in neighbors):
            new_cell_positions.add(cell)
    new_cell_positions = PixelSet(new_cell_positions)
    return _update_object_state(obj, new_cell_positions)


@decorate_primitive("state", tags={Effect.SHAPE})
def unfill_all_objects(state: ArcState) -> ArcState:
    """
    Unfill all objects in the state, only keeping the borders of the objects and setting the inside to background colour
    """
    new_objects = []
    for obj in state.objects:
        new_obj = _unfill_object(obj)
        new_objects.append(new_obj)

    grid = np.zeros(state.grid_state.dimensions, dtype=int)
    for obj in new_objects:
        for r, c in obj.cell_positions:
            grid[r, c] = obj.colour
    return ArcState.from_array(grid)


# %% --------------------------------------------------------------------------
# State based primitives
# -----------------------------------------------------------------------------
@decorate_primitive("state", tags={Effect.COLOUR})
def recolour_all_objects(state: ArcState, out_colour: int) -> ArcState:
    """
    Recolour all objects in the state to the specified colour
    """
    new_objects = []
    for obj in state.objects:
        new_obj = _update_object_state(
            obj, PixelSet(obj.cell_positions), colour=out_colour
        )
        new_objects.append(new_obj)

    return _update_arcstate_grid(state, new_objects)


@decorate_primitive(
    "state", tags={Effect.TRANSLATE, Require.SINGLE_PIXEL, Require.MULTI_OBJECTS}
)
def join_single_pixels_to_large_colour_object(state: ArcState):
    """
    Move and join single pixels onto a large object of the same colour if available
    """
    grid = state.to_array()
    objects = state.objects
    new_grid = grid.copy()
    single_pixel_objects = [obj for obj in objects if obj.area == 1]
    large_colour_objects = [obj for obj in objects if obj.area > 2]

    if not single_pixel_objects or not large_colour_objects:
        return state

    for single_pixel in single_pixel_objects:
        colour = single_pixel.colour
        # Find large object of same colour
        large_object = [obj for obj in large_colour_objects if obj.colour == colour]
        if not large_object:
            continue
        else:
            large_object = large_object[0]
        single_pixel_pos = list(single_pixel.cell_positions)[0]
        # Look for shared row or column with large object
        large_object_rows = [pos[0] for pos in large_object.cell_positions]
        large_object_cols = [pos[1] for pos in large_object.cell_positions]
        if single_pixel_pos[0] in large_object_rows:
            new_grid[single_pixel_pos[0], single_pixel_pos[1]] = 0
            try:
                target_col = (
                    (min(large_object_cols) - 1)
                    if single_pixel_pos[1] < min(large_object_cols)
                    else max(large_object_cols) + 1
                )
                new_grid[single_pixel_pos[0], target_col] = colour
            except:
                target_col = (
                    (min(large_object_cols))
                    if single_pixel_pos[1] < min(large_object_cols)
                    else max(large_object_cols)
                )
                new_grid[single_pixel_pos[0], target_col] = colour
        elif single_pixel_pos[1] in large_object_cols:
            new_grid[single_pixel_pos[0], single_pixel_pos[1]] = 0
            try:
                target_row = (
                    (min(large_object_rows) - 1)
                    if single_pixel_pos[0] < min(large_object_rows)
                    else max(large_object_rows) + 1
                )
                new_grid[target_row, single_pixel_pos[1]] = colour
            except:
                target_row = (
                    (min(large_object_rows))
                    if single_pixel_pos[0] < min(large_object_rows)
                    else max(large_object_rows)
                )
                new_grid[target_row, single_pixel_pos[1]] = colour

    return ArcState.from_array(new_grid)


@decorate_primitive("state", tags={Require.MULTI_OBJECTS, Effect.TRANSLATE})
def reflect_obj_relative_to_line_obj(
    state: ArcState,
) -> ArcState:
    """
    Reflect the object relative to the nearest line object in the specified direction
    """
    objects = state.objects
    line_objects = [obj for obj in objects if obj.has_line]
    # Get all other objects that are not line objects use opposite of intersection
    other_objects = [obj for obj in objects if obj not in line_objects]

    if not line_objects or not other_objects:
        return state

    new_objects = line_objects.copy()
    for obj in other_objects:
        # Find closest line
        closest_line = min(
            line_objects,
            key=lambda line: np.min(
                [
                    np.linalg.norm(np.array(cell) - np.array(line_cell))
                    for cell in obj.cell_positions
                    for line_cell in line.cell_positions
                ]
            ),
        )
        # Reflect the object relative to the closest line
        line_cells = list(closest_line.cell_positions)
        # Find most common row or column in the line cells to determine the mirror line
        rows, cols = zip(*line_cells)
        most_row, most_row_count = Counter(rows).most_common(1)[0]
        most_col, most_col_count = Counter(cols).most_common(1)[0]
        if most_row_count > most_col_count:
            # More rows than columns, mirror line is horizontal
            mirror_line = [(most_row, min(cols)), (most_row, max(cols))]
        else:
            # More columns than rows, mirror line is vertical
            mirror_line = [(min(rows), most_col), (max(rows), most_col)]
        new_cell_positions = set()
        for cell in obj.cell_positions:
            reflected = list(cell)
            # Axis 0 is vertical line (x coordinate is the same) and vice versa
            axis = 0 if mirror_line[0][0] == mirror_line[1][0] else 1
            reflected[axis] = mirror_line[0][axis] + (mirror_line[0][axis] - cell[axis])
            new_cell_positions.add(tuple(reflected))

        # Check that the new cell positions are within the grid bounds
        rows, cols = obj.grid_size
        new_cell_positions = {
            cell
            for cell in new_cell_positions
            if 0 <= cell[0] < rows and 0 <= cell[1] < cols
        }
        new_objects.append(_update_object_state(obj, PixelSet(new_cell_positions)))

    return _update_arcstate_grid(state, new_objects)


@decorate_primitive("state", tags={Require.MULTI_OBJECTS, Effect.OBJECT_COUNT})
def connect_objects_aligned(
    state: ArcState,
    out_colour: int,
) -> ArcState:
    """
    Connect objects that are aligned in the same row or column with a line of the specified colour
    """
    objects = state.objects
    new_objects = []
    new_objects.extend(objects)
    for i, obj1 in enumerate(objects):
        for j, obj2 in enumerate(objects):
            if i >= j:
                continue
            # Check if objects are aligned in the same row or column (at least 4 cells in the same row or column)
            # If so find shortest distance between them
            rows1, cols1 = zip(*obj1.cell_positions)
            rows2, cols2 = zip(*obj2.cell_positions)
            if len(set(rows1).intersection(set(rows2))) >= 2:
                # Find row that has shortest distance between the two objects
                row_list = list(set(rows1).intersection(set(rows2)))
                row = min(
                    row_list,
                    key=lambda r: min(
                        abs(c1 - c2)
                        for c1 in [c for r1, c in obj1.cell_positions if r1 == r]
                        for c2 in [c for r2, c in obj2.cell_positions if r2 == r]
                    ),
                )
                cols_in_obj1 = [c for r1, c in obj1.cell_positions if r1 == row]
                cols_in_obj2 = [c for r2, c in obj2.cell_positions if r2 == row]
                if min(cols_in_obj1) < min(cols_in_obj2):
                    min_col = max(cols_in_obj1)
                    max_col = min(cols_in_obj2)
                else:
                    min_col = max(cols_in_obj2)
                    max_col = min(cols_in_obj1)
                # Check if there at least 4 original object cells in the row
                if len(cols_in_obj1) >= 2 and len(cols_in_obj2) >= 2:
                    new_cell_positions = set()
                    for c in range(min_col + 1, max_col):
                        new_cell_positions.add((row, c))
                    new_objects.append(
                        ObjectState(
                            label_id=obj1.label_id
                            + obj2.label_id
                            + np.random.randint(1, 1000),
                            colour=out_colour,
                            grid_size=obj1.grid_size,
                            bounding_box=(row, min_col, row + 1, max_col + 1),
                            centroid=(row, (min_col + max_col) // 2),
                            area=len(new_cell_positions),
                            cell_positions=PixelSet(new_cell_positions),
                            priority=0,
                        ),
                    )
                else:
                    continue
            elif len(set(cols1).intersection(set(cols2))) > 0:
                # Find column that has shortest distance between the two objects
                col_list = list(set(cols1).intersection(set(cols2)))
                col = min(
                    col_list,
                    key=lambda c: min(
                        abs(r1 - r2)
                        for r1 in [r for r, c1 in obj1.cell_positions if c1 == c]
                        for r2 in [r for r, c2 in obj2.cell_positions if c2 == c]
                    ),
                )
                rows_in_obj1 = [r for r, c1 in obj1.cell_positions if c1 == col]
                rows_in_obj2 = [r for r, c2 in obj2.cell_positions if c2 == col]
                if min(rows_in_obj1) < min(rows_in_obj2):
                    min_row = max(rows_in_obj1)
                    max_row = min(rows_in_obj2)
                else:
                    min_row = max(rows_in_obj2)
                    max_row = min(rows_in_obj1)
                # Check if there at least 4 original object cells in the column
                if len(rows_in_obj1) >= 2 and len(rows_in_obj2) >= 2:
                    new_cell_positions = set()
                    for r in range(min_row + 1, max_row):
                        new_cell_positions.add((r, col))
                    new_objects.append(
                        ObjectState(
                            label_id=obj1.label_id
                            + obj2.label_id
                            + np.random.randint(1, 1000),
                            colour=out_colour,
                            grid_size=obj1.grid_size,
                            bounding_box=(min_row, col, max_row + 1, col + 1),
                            centroid=((min_row + max_row) // 2, col),
                            area=len(new_cell_positions),
                            cell_positions=PixelSet(new_cell_positions),
                            priority=0,
                        )
                    )
                else:
                    continue
    return _update_arcstate_grid(state, new_objects)


@decorate_primitive("state", tags={Require.MULTI_OBJECTS})
def perfect_join_objects_to_large_object(state: ArcState, background_colour: int):
    """
    Look to perfectly join smaller objects to a larger object.
    Perfect join is defined as no empty space between the two objects
    """
    objects = state.objects
    grid = np.array(state.grid_state.grid)

    if not objects or len(objects) < 2:
        return state
    largest_object = max(objects, key=lambda obj: obj.area)
    # Check if it is a block object
    if (
        largest_object.is_block
        and largest_object.width < state.grid_state.dimensions[1]
    ):
        return state

    top_row, min_col, bottom_row, max_col = largest_object.bounding_box
    top_r_pixels = grid[top_row, min_col:max_col]
    empty_positions = np.where(top_r_pixels == background_colour)[0] + min_col

    empty_groupings = []
    for _, group in groupby(enumerate(empty_positions), lambda ix: ix[0] - ix[1]):
        cols = [col for _, col in group]
        empty_groupings.append(((top_row, cols[0]), [(top_row, col) for col in cols]))

    # Sort the empty groupings by size in descending order
    empty_groupings.sort(key=lambda x: len(x[1]), reverse=True)

    other_objects = sorted(
        [obj for obj in objects if obj != largest_object],
        key=lambda obj: obj.area,
        reverse=True,
    )
    new_objects = [largest_object]
    used_positions = set()
    placed_objects = set()

    # From the largest other object, see if it's height or width can fill the empty space
    for obj in other_objects:
        if not obj.is_block:
            continue

        if obj.label_id in placed_objects:
            continue

        for idx, ((top_row, start_col), positions) in enumerate(empty_groupings):
            if idx in used_positions:
                continue

            size = len(positions)
            if obj.width == size:
                new_cells = {
                    (r - h, c) for r, c in positions for h in range(obj.height)
                }

            elif obj.height == size:
                new_cells = {(r - w, c) for r, c in positions for w in range(obj.width)}
            else:
                continue

            new_objects.append(_update_object_state(obj, new_cells))
            used_positions.add(idx)
            placed_objects.add(obj.label_id)
            break

    # Add old objects that were not joined to the new objects
    new_objects.extend(
        obj
        for obj in other_objects
        if obj.label_id not in [o.label_id for o in new_objects]
    )
    return _update_arcstate_grid(state, new_objects)
