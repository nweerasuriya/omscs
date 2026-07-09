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
import scipy.ndimage as ndi
from MemoryDecorators import (
    grid_primitive,
    object_primitive,
    split_grid_primitive,
)
from ArcMemory import (
    ObjectState,
    GridState,
    PixelSet,
)
from ArcPruning import Require, Effect
from helpers import determine_new_obj_props


# -----------------------------------------------------------------------------
# Base Logical Operators
# -----------------------------------------------------------------------------
@grid_primitive
def logic_not(array: np.ndarray) -> np.ndarray:
    return np.bitwise_not(array)


@split_grid_primitive(tags={Require.SPLIT_GRID, Effect.GRID_SIZE})
def logic_and(array_list: list, fill_colour: int = 1) -> np.ndarray:
    """
    Perform logical AND operation. Keep original colours unless overlap occurs, then fill with fill_colour
    """
    overlap_mask = np.logical_and.reduce([a != 0 for a in array_list])
    combined = np.maximum.reduce(array_list)
    return np.where(overlap_mask, fill_colour, combined)


@split_grid_primitive(tags={Require.SPLIT_GRID, Effect.GRID_SIZE})
def logic_and_first_populated(array_list: list) -> np.ndarray:
    result = np.array(array_list[0])
    for array in array_list[1:]:
        result = np.where(result == 0, array, result)
    return result


@split_grid_primitive(tags={Require.SPLIT_GRID, Effect.GRID_SIZE})
def logic_or(array_list: list, fill_colour: int = 1) -> np.ndarray:
    """
    Perform logical OR operation. Keep original colours unless overlap occurs, then fill with fill_colour
    """
    overlap_mask = np.logical_or.reduce([a != 0 for a in array_list])
    combined = np.maximum.reduce(array_list)
    return np.where(overlap_mask, fill_colour, combined)


@split_grid_primitive(tags={Require.SPLIT_GRID, Effect.GRID_SIZE})
def logic_nand(array_list: list, fill_colour: int = 1) -> np.ndarray:
    """
    Perform logical NAND operation. Keep original colours unless overlap occurs, then fill with fill_colour
    """
    overlap_mask = np.logical_and.reduce([a != 0 for a in array_list])
    combined = np.maximum.reduce(array_list)
    return np.where(overlap_mask, fill_colour, combined)


@split_grid_primitive(tags={Require.SPLIT_GRID, Effect.GRID_SIZE})
def logic_xor(array_list: list) -> np.ndarray:
    """
    Perform logical XOR operation. Keep original colours unless overlap occurs, then fill with fill_colour
    """
    count = np.sum([a != 0 for a in array_list], axis=0)
    stacked_arrays = np.stack(array_list, axis=0)
    colours = np.sum(stacked_arrays, axis=0)
    return np.where(count == 1, colours, 0)


@split_grid_primitive(tags={Require.SPLIT_GRID, Effect.GRID_SIZE})
def logic_nor(array_list: list, fill_colour: int = 1) -> np.ndarray:
    """
    Perform logical NOR operation. Keep original colours unless overlap occurs, then fill with fill_colour
    """
    overlap_mask = np.logical_or.reduce([a != 0 for a in array_list])
    combined = np.maximum.reduce(array_list)
    return np.where(overlap_mask, fill_colour, combined)


@split_grid_primitive(tags={Require.SPLIT_GRID, Effect.GRID_SIZE})
def logic_and_(array_list: list, fill_colour: int = 1) -> np.ndarray:
    mask = np.logical_and.reduce([a != 0 for a in array_list])
    return np.where(mask, fill_colour, 0)


@split_grid_primitive(tags={Require.SPLIT_GRID, Effect.GRID_SIZE})
def logic_or_(array_list: list, fill_colour: int = 1) -> np.ndarray:
    mask = np.logical_or.reduce([a != 0 for a in array_list])
    return np.where(mask, fill_colour, 0)


@split_grid_primitive(tags={Require.SPLIT_GRID, Effect.GRID_SIZE})
def logic_nand_(array_list: list, fill_colour: int = 1) -> np.ndarray:
    mask = np.logical_and.reduce([a != 0 for a in array_list])
    return np.where(mask, 0, fill_colour)


@split_grid_primitive(tags={Require.SPLIT_GRID, Effect.GRID_SIZE})
def logic_xor_(array_list: list) -> np.ndarray:
    mask = np.logical_xor.reduce([a != 0 for a in array_list])
    return np.where(mask, 1, 0)


@split_grid_primitive(tags={Require.SPLIT_GRID, Effect.GRID_SIZE})
def logic_nor_(array_list: list, fill_colour: int = 1) -> np.ndarray:
    mask = np.logical_or.reduce([a != 0 for a in array_list])
    return np.where(mask, 0, fill_colour)


@split_grid_primitive(tags={Require.SPLIT_GRID, Effect.GRID_SIZE})
def logic_xnor_(array_list: list, fill_colour: int = 1) -> np.ndarray:
    mask = np.logical_xor.reduce([a != 0 for a in array_list])
    return np.where(mask, 0, fill_colour)


# -----------------------------------------------------------------------------
# Rotations and Flips
# -----------------------------------------------------------------------------
@grid_primitive()
def rotate_grid(input_array: np.ndarray, quarter_turn: int) -> np.ndarray:
    return np.rot90(input_array, k=quarter_turn)


@object_primitive()
def rotate_object(obj: ObjectState, quarter_turn: int) -> np.ndarray:
    rotated = np.rot90(obj.mask, k=quarter_turn)
    new_pixels = determine_new_obj_props(rotated, obj.centroid)
    return _update_object_state(obj, PixelSet(new_pixels))


@grid_primitive()
def flip_horizontal(input_array: np.ndarray) -> np.ndarray:
    return np.fliplr(input_array)


@object_primitive()
def flip_object_horizontal(obj: ObjectState) -> ObjectState:
    flipped_grid = np.fliplr(obj.mask)
    new_pixels = determine_new_obj_props(flipped_grid, obj.centroid)
    return _update_object_state(obj, PixelSet(new_pixels))


@grid_primitive()
def flip_vertical(input_array: np.ndarray) -> np.ndarray:
    return np.flipud(input_array)


@object_primitive()
def flip_object_vertical(obj: ObjectState) -> ObjectState:
    flipped_grid = np.flipud(obj.mask)
    new_pixels = determine_new_obj_props(flipped_grid, obj.centroid)
    return _update_object_state(obj, PixelSet(new_pixels))


@grid_primitive(tags={Require.SQUARE_GRID})
def flip_diagonal(input_array: np.ndarray) -> np.ndarray:
    return np.transpose(input_array)


@object_primitive(tags={Require.SQUARE_GRID})
def flip_object_diagonal(obj: ObjectState) -> ObjectState:
    flipped_grid = np.transpose(obj.mask)
    new_pixels = determine_new_obj_props(flipped_grid, obj.centroid)
    return _update_object_state(obj, PixelSet(new_pixels))


@grid_primitive(tags={Require.SQUARE_GRID})
def flip_anti_diagonal(input_array: np.ndarray) -> np.ndarray:
    return np.fliplr(np.transpose(input_array))


@object_primitive(tags={Require.SQUARE_GRID})
def flip_object_anti_diagonal(obj: ObjectState) -> ObjectState:
    flipped_grid = np.fliplr(np.transpose(obj.mask))
    new_pixels = determine_new_obj_props(flipped_grid, obj.centroid)
    return _update_object_state(obj, PixelSet(new_pixels))


# -----------------------------------------------------------------------------
# Colour level functions
# -----------------------------------------------------------------------------
@grid_primitive(tags={Effect.GRID_SIZE, Effect.SHRINK})
def crop_background_out(
    input_array: np.ndarray, background_color: int = 0
) -> np.ndarray:
    if not np.any(input_array != background_color):
        return (
            input_array  # Return the original array if all values are background color
        )
    rows = np.any(input_array != background_color, axis=1)
    cols = np.any(input_array != background_color, axis=0)
    return input_array[np.ix_(rows, cols)]


@grid_primitive(tags={Effect.COLOUR})
def change_colour(
    input_array: np.ndarray, in_colour: int, out_colour: int
) -> np.ndarray:
    grid = input_array.copy()
    return np.where(grid == in_colour, out_colour, grid)


@grid_primitive(tags={Effect.COLOUR})
def update_colour(
    input_array: np.ndarray, new_colours: int, removed_colours: int
) -> np.ndarray:
    """
    Update the colours in the input array by replacing the removed colours with the new colours.
    """
    grid = input_array.copy()
    grid = np.where(grid == removed_colours, new_colours, grid)
    return grid


@grid_primitive(tags={Effect.COLOUR, Require.REMOVE_COLOURS})
def remove_colours(
    input_array: np.ndarray, removed_colour_set: set[int], background_colour: int = 0
) -> np.ndarray:
    """
    Remove the specified colours from the input array by setting them to background colour 0.
    """
    grid = input_array.copy()
    for colour in removed_colour_set:
        grid = np.where(grid == colour, background_colour, grid)
    return grid


@grid_primitive(tags={Effect.COLOUR})
def change_background_colour(input_array: np.ndarray, out_colour: int) -> np.ndarray:
    """
    Change the background colour of the input array to the new background colour.
    """
    grid = input_array.copy()
    current_background_colour = 0  # Assuming 0 is the current background colour
    return np.where(grid == current_background_colour, out_colour, grid)


@grid_primitive(tags={Effect.COLOUR})
def fill_default_colour(
    input_array: np.ndarray, out_colour: int, default_colour: int = 1
) -> np.ndarray:
    """
    Fill the input array with the specified out_colour where the default_colour is present.
    """
    grid = input_array.copy()
    return np.where(grid == default_colour, out_colour, grid)


@grid_primitive(tags={Effect.COLOUR})
def invert_input_colours(
    input_array: np.ndarray, background_colour: int = 0
) -> np.ndarray:
    """
    If two colours are present in the input array, invert them.
    If one colour is present, invert with background colour 0.
    """
    grid = input_array.copy()
    colours = np.unique(grid)
    # remove background colour 0 from the list of colours
    colours = colours[colours != background_colour]
    if len(colours) == 1:
        # Only one colour present, invert with background colour 0
        mask = grid == colours[0]
        grid = np.select([mask, ~mask], [0, colours[0]], default=grid)
    elif len(colours) == 2:
        mask_a = grid == colours[0]
        mask_b = grid == colours[1]
        grid = np.select([mask_a, mask_b], [colours[1], colours[0]], default=grid)
    return grid


@grid_primitive(tags={Effect.COLOUR})
def colour_grey_to_black(input_array: np.ndarray, grey: int = 5) -> np.ndarray:
    if not np.any(input_array == grey):
        return input_array
    grid = input_array.copy()
    return np.where(grid == grey, 0, grid)


@object_primitive(tags={Effect.COLOUR})
def recolour_object(obj: ObjectState, new_colour_obj: int) -> ObjectState:
    return obj._replace(colour=new_colour_obj)


@grid_primitive(tags={Effect.COLOUR})
def connect_same_colour(input_array: np.ndarray, out_colour: int) -> np.ndarray:
    """
    Connects pixels of the specified colour that are not connected by filling in the gaps between them.
    Only connect in straight lines (horizontal, vertical, diagonal)
    """
    grid = input_array.copy()
    rows, cols = grid.shape

    # Check horizontal
    for row in range(rows):
        colour_pixels = np.where(grid[row, :] == out_colour)[0]
        if len(colour_pixels) >= 2:
            start = colour_pixels[0]
            end = colour_pixels[-1]
            grid[row, start : end + 1] = grid[row, start]

    # Check vertical
    for col in range(cols):
        colour_pixels = np.where(grid[:, col] == out_colour)[0]
        if len(colour_pixels) >= 2:
            start = colour_pixels[0]
            end = colour_pixels[-1]
            if grid[start, col] == grid[end, col]:
                grid[start : end + 1, col] = grid[start, col]

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


@grid_primitive(tags={Effect.COLOUR})
def fill_enclosed_area(
    input_array: np.ndarray,
    out_colour: int,
) -> np.ndarray:
    """
    Fill any enclosed area of the input array with the specified colour.
    """
    binary_mask = input_array != 0
    filled_mask = ndi.binary_fill_holes(binary_mask)
    # Create a new array with the filled areas set to the specified colour
    filled_array = np.where(filled_mask, out_colour, input_array)
    return filled_array


@grid_primitive(tags={Effect.COLOUR})
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
@grid_primitive
def mirror_horizontal(input_array: np.ndarray) -> np.ndarray:
    """
    Mirror the input array horizontally
    """
    return np.fliplr(input_array)


@grid_primitive
def mirror_vertical(input_array: np.ndarray) -> np.ndarray:
    """
    Mirror the input array vertically
    """
    return np.flipud(input_array)


@grid_primitive(tags={Require.SQUARE_GRID})
def mirror_diagonal(input_array: np.ndarray) -> np.ndarray:
    """
    Mirror the input array diagonally, reflecting the pixels across the diagonal axis
    """
    return np.transpose(input_array)


@grid_primitive(tags={Require.SQUARE_GRID})
def mirror_anti_diagonal(input_array: np.ndarray) -> np.ndarray:
    """
    Mirror the input array anti-diagonally, reflecting the pixels across the anti-diagonal axis
    """
    return np.fliplr(np.transpose(input_array))


# -----------------------------------------------------------------------------
# Other Grid Primitives
# -----------------------------------------------------------------------------
@grid_primitive(tags={Effect.GRID_SIZE, Effect.GROWTH})
def pad_grid(
    input_array: np.ndarray, pad_width: int = 1, pad_value: int = 0
) -> np.ndarray:
    """
    Pad the input array with the specified pad_width and pad_value.
    """
    return np.pad(
        input_array, pad_width=pad_width, mode="constant", constant_values=pad_value
    )


@grid_primitive(tags={Effect.GRID_SIZE, Effect.GROWTH})
def unfold_grid_vertical(input_array: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Unfolds the grid vertically, producing a new grid with the same number of rows but double the number of columns.
    The array is mirrored along the vertical axis, and the two halves are concatenated side by side.
    """
    return np.concatenate((input_array, np.fliplr(input_array)), axis=axis)


@grid_primitive(tags={Effect.GRID_SIZE, Effect.GROWTH})
def unfold_grid_horizontal(input_array: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Unfolds the grid horizontally, producing a new grid with the same number of columns but double the number of rows.
    The array is mirrored along the horizontal axis, and the two halves are concatenated one on top of the other.
    """
    return np.concatenate((input_array, np.flipud(input_array)), axis=axis)


@grid_primitive(tags={Effect.GRID_SIZE, Effect.GROWTH})
def unfold_half_grid_vertical(input_array: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Unfolds the grid vertically, producing a new grid with the same number of rows but double the number of columns.
    The array is mirrored along the vertical axis, and the two halves are concatenated side by side.
    """
    half_cols = input_array.shape[1] // 2
    left_half = input_array[:, :half_cols]
    return np.concatenate((left_half, np.fliplr(left_half)), axis=axis)


@grid_primitive(tags={Effect.GRID_SIZE, Effect.GROWTH})
def unfold_half_grid_horizontal(input_array: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Unfolds the grid horizontally, producing a new grid with the same number of columns but double the number of rows.
    The array is mirrored along the horizontal axis, and the two halves are concatenated one on top of the other.
    """
    half_rows = input_array.shape[0] // 2
    top_half = input_array[:half_rows, :]
    return np.concatenate((top_half, np.flipud(top_half)), axis=axis)


# -----------------------------------------------------------------------------
# Object Primitives
# -----------------------------------------------------------------------------
def _update_object_state(obj: ObjectState, new_cell_positions: PixelSet) -> ObjectState:
    """
    Update the object state with new cell positions
    """
    # Recalculate bounding box, centroid, and area based on new cell positions
    if not new_cell_positions:
        bounding_box = obj.bounding_box
        centroid = obj.centroid
        area = 0
    else:
        rows, cols = zip(*new_cell_positions)
        bounding_box = (min(rows), max(rows), min(cols), max(cols))
        area = len(new_cell_positions)
        centroid = (sum(rows) / area, sum(cols) / area)

    return ObjectState(
        label_id=obj.label_id,
        colour=obj.colour,
        grid_size=obj.grid_size,
        bounding_box=bounding_box,
        centroid=centroid,
        area=area,
        cell_positions=new_cell_positions,
        hu_moments=obj.hu_moments,
    )


def _create_binary_grid_from_bbox(
    bounding_box: tuple[int, int, int, int], pixel_set: PixelSet, colour: int
) -> np.ndarray:
    """
    Create a binary grid from the bounding box of an object
    """
    x_min, y_min, x_max, y_max = bounding_box
    grid = np.zeros((x_max - x_min, y_max - y_min), dtype=int)
    for pixel in pixel_set:
        x, y = pixel
        grid[x - x_min, y - y_min] = colour
    return grid


# TODO: Account for cells not bordering bounding box
@object_primitive(tags={Effect.SHAPE})
def fill_bounding_box(
    obj: ObjectState,
) -> ObjectState:
    """
    Fill the object with the specified colour within bounding box
    """
    bounding_box = obj.bounding_box
    x_min, y_min, x_max, y_max = bounding_box
    # Get new cell positions within the bounding box
    new_cell_positions = set()
    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            new_cell_positions.add((x, y))

    return _update_object_state(obj, PixelSet(new_cell_positions))


# @object_primitive(tags={Effect.SHAPE})
# def fill_enclosed_area(
#     obj: ObjectState,
#     out_colour: int,
# ):
#     """
#     Fill any enclosed area of the object with the specified colour.
#     Return a new ObjectState with the filled area (this will be a different object state than the input object as could be a different colour and area)
#     """
#     cell_positions = obj.cell_positions
#     # Create a grid of the object
#     x_min, y_min, x_max, y_max = obj.bounding_box
#     grid = _create_binary_grid_from_bbox(obj.bounding_box, cell_positions, obj.colour)
#     # Fill enclosed areas using binary fill holes
#     filled_grid = ndi.binary_fill_holes(grid).astype(int) * list(out_colour)[0]

#     # Get new cell positions from the filled grid
#     new_cell_positions = set()
#     for x in range(filled_grid.shape[0]):
#         for y in range(filled_grid.shape[1]):
#             if filled_grid[x, y] != 0:
#                 new_cell_positions.add((x + x_min, y + y_min))

#     return ObjectState(
#         label_id=obj.label_id + 100,
#         colour=list(out_colour)[0],
#         bounding_box=obj.bounding_box,
#         centroid=obj.centroid,
#         area=len(new_cell_positions),
#         cell_positions=PixelSet(new_cell_positions),
#         hu_moments=obj.hu_moments,
#     )


@object_primitive(tags={Effect.SHAPE})
def unfill_object(
    obj: ObjectState,
    background_colour: int = 0,
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


@object_primitive(tags={Effect.SHAPE})
def remove_object(
    obj: ObjectState,
) -> ObjectState:
    """
    Remove the object, returning an empty object state
    """
    return _update_object_state(obj, PixelSet(set()))


@object_primitive(tags={Effect.SHAPE})
def crop_object(obj: ObjectState) -> ObjectState:
    """
    Crop the object to its bounding box
    """
    x_min, y_min, x_max, y_max = obj.bounding_box
    new_cell_positions = set()
    for cell in obj.cell_positions:
        x, y = cell
        if x_min <= x < x_max and y_min <= y < y_max:
            new_cell_positions.add(cell)

    return _update_object_state(obj, PixelSet(new_cell_positions))


# @object_primitive(tags={Effect.GROWTH, Require.DIRECTIONALITY})
# def grow_object(
#     obj: ObjectState, scale: int = 1, direction_vector: tuple[int, int] = (0, 0)
# ) -> ObjectState:
#     """
#     Grow the object by scale factor.
#     If direction vector is provided, grow in that direction
#     """
#     if direction_vector != {(0, 0)}:
#         cell_positions = obj.cell_positions
#         new_cell_positions = set()
#         for cell in cell_positions:
#             x, y = cell
#             for dx in range(-scale, scale + 1):
#                 for dy in range(-scale, scale + 1):
#                     if (dx, dy) in direction_vector:
#                         new_cell_positions.add((x + dx, y + dy))
#         return _update_object_state(obj, PixelSet(new_cell_positions))
#     else:
#         return obj


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


@object_primitive(tags={Effect.GROWTH, Require.DIRECTIONALITY})
def add_line_to_object(
    obj: ObjectState,
    direction_vector: tuple[int, int] = (1, 0),
    scale: int = 1,
) -> ObjectState:
    """
    Add a line to the object in the middle of the bounding box in the specified direction
    Single line should extent to the edge of the grid in the specified direction
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

    return _update_object_state(obj, PixelSet(new_cell_positions))


@object_primitive(tags={Effect.TRANSLATE, Require.DIRECTIONALITY})
def translate_object(
    obj: ObjectState, direction_vector: tuple[int, int] = (1, 0), scale: int = 1
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


# -----------------------------------------------------------------------------
# Relation based primitives
# -----------------------------------------------------------------------------
# @relation_primitive(tags={Effect.TRANSLATE})
# def move_towards_obj(
#     obj: ObjectState, others: tuple[ObjectState, ...], scale: int = 1
# ) -> ObjectState:
#     """
#     Move the object towards the target object by 1 unit in the direction of the target object
#     """
#     cell_positions = obj.cell_positions
#     target_centroid = others[0].centroid if others else obj.centroid
#     obj_centroid = obj.centroid

#     # Calculate direction vector towards the target object
#     direction_vector = (
#         np.sign(target_centroid[0] - obj_centroid[0]),
#         np.sign(target_centroid[1] - obj_centroid[1]),
#     )

#     new_cell_positions = set()
#     for cell in cell_positions:
#         x, y = cell
#         dx, dy = direction_vector
#         new_cell_positions.add((x + scale * dx, y + scale * dy))

#     return _update_object_state(obj, PixelSet(new_cell_positions))
