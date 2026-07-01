"""
Primitive DSL for ArcAgi
Includes basic operations for transformation fo cells and objects
All operations should be generic and be able to ingest either grids or objects or cells if possible
"""

__date__ = "2026-06-10"
__author__ = "NedeeshaWeerasuriya"
__version__ = "0.1"

import numpy as np
import scipy.ndimage as ndi
from ArcMemory import (
    grid_primitive,
    object_primitive,
    split_grid_primitive,
    ObjectState,
    GridState,
    PixelSet,
)
from ArcPruning import Require, Effect

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


# @split_grid_primitive(tags={Require.SPLIT_GRID, Effect.GRID_SIZE})
# def logic_xnor(array_list: list, fill_colour: int = 1) -> np.ndarray:
#     """
#     Perform logical XNOR operation. Keep original colours unless overlap occurs, then fill with fill_colour
#     """
#     count = np.sum([a != 0 for a in array_list], axis=0)
#     combined = np.maximum.reduce(array_list)
#     fill_mask = (count % 2 == 0) & (count > 0)
#     return np.where(fill_mask, fill_colour, combined)


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
# Grid level functions
# -----------------------------------------------------------------------------
@grid_primitive()
def rotation_90(input_array: np.ndarray) -> np.ndarray:
    return np.rot90(input_array, k=1)


@grid_primitive()
def rotation_180(input_array: np.ndarray) -> np.ndarray:
    return np.rot90(input_array, k=2)


@grid_primitive()
def rotation_270(input_array: np.ndarray) -> np.ndarray:
    return np.rot90(input_array, k=3)


@grid_primitive()
def flip_horizontal(input_array: np.ndarray) -> np.ndarray:
    return np.fliplr(input_array)


@grid_primitive()
def flip_vertical(input_array: np.ndarray) -> np.ndarray:
    return np.flipud(input_array)


@grid_primitive(tags={Require.SQUARE_GRID})
def flip_diagonal(input_array: np.ndarray) -> np.ndarray:
    return np.transpose(input_array)


@grid_primitive(tags={Require.SQUARE_GRID})
def flip_anti_diagonal(input_array: np.ndarray) -> np.ndarray:
    return np.fliplr(np.transpose(input_array))


@grid_primitive(tags={Effect.GRID_SIZE})
def crop_background_out(
    input_array: np.ndarray, background_color: int = 0
) -> np.ndarray:
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
def remove_colours(input_array: np.ndarray, removed_colour_set: set[int]) -> np.ndarray:
    """
    Remove the specified colours from the input array by setting them to background colour 0.
    """
    grid = input_array.copy()
    for colour in removed_colour_set:
        grid = np.where(grid == colour, 0, grid)
    return grid


@grid_primitive(tags={Effect.COLOUR})
def change_background_colour(input_array: np.ndarray, out_colour: int) -> np.ndarray:
    """
    Change the background colour of the input array to the new background colour.
    """
    grid = input_array.copy()
    current_background_colour = 0  # Assuming 0 is the current background colour
    return np.where(grid == current_background_colour, out_colour, grid)


# @grid_primitive(tags={Effect.COLOUR})
# def fill_default_colour(
#     input_array: np.ndarray, out_colour: int, default_colour: int = 1
# ) -> np.ndarray:
#     """
#     Fill the input array with the specified out_colour where the default_colour is present.
#     """
#     grid = input_array.copy()
#     return np.where(grid == default_colour, out_colour, grid)


@grid_primitive(tags={Effect.COLOUR})
def invert_input_colours(input_array: np.ndarray) -> np.ndarray:
    """
    If two colours are present in the input array, invert them.
    If one colour is present, invert with background colour 0.
    """
    grid = input_array.copy()
    colours = np.unique(grid)
    # remove background colour 0 from the list of colours
    colours = colours[colours != 0]
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
    grid = input_array.copy()
    return np.where(grid == grey, 0, grid)


@grid_primitive(tags={Effect.SHAPE})
def connect_same_colour(input_array: np.ndarray) -> np.ndarray:
    """
    Connects any pixels of the same colour that are not connected by filling in the gaps between them.
    Only connect in straight lines (horizontal, vertical).
    """
    grid = input_array.copy()
    rows, cols = grid.shape

    # Check horizontal
    for row in range(rows):
        colour_pixels = np.where(grid[row, :] != 0)[0]
        # Check for two pixels of the same colour and connect them
        if len(colour_pixels) > 1:
            for i in range(len(colour_pixels) - 1):
                start = colour_pixels[i]
                end = colour_pixels[i + 1]
                if grid[row, start] == grid[row, end]:
                    grid[row, start : end + 1] = grid[row, start]

    # Check vertical
    for col in range(cols):
        colour_pixels = np.where(grid[:, col] != 0)[0]
        # Check for two pixels of the same colour and connect them
        if len(colour_pixels) > 1:
            for i in range(len(colour_pixels) - 1):
                start = colour_pixels[i]
                end = colour_pixels[i + 1]
                if grid[start, col] == grid[end, col]:
                    grid[start : end + 1, col] = grid[start, col]
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


@grid_primitive(tags={Effect.GRID_SIZE})
def unfold_grid_vertical(input_array: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Unfolds the grid vertically, producing a new grid with the same number of rows but double the number of columns.
    The array is mirrored along the vertical axis, and the two halves are concatenated side by side.
    """
    return np.concatenate((input_array, np.fliplr(input_array)), axis=axis)


@grid_primitive(tags={Effect.GRID_SIZE})
def unfold_grid_horizontal(input_array: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Unfolds the grid horizontally, producing a new grid with the same number of columns but double the number of rows.
    The array is mirrored along the horizontal axis, and the two halves are concatenated one on top of the other.
    """
    return np.concatenate((input_array, np.flipud(input_array)), axis=axis)


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
@grid_primitive(tags={Effect.GRID_SIZE})
def pad_grid(
    input_array: np.ndarray, pad_width: int = 1, pad_value: int = 0
) -> np.ndarray:
    """
    Pad the input array with the specified pad_width and pad_value.
    """
    return np.pad(
        input_array, pad_width=pad_width, mode="constant", constant_values=pad_value
    )


# -----------------------------------------------------------------------------
# Object Primitives
# -----------------------------------------------------------------------------
def _update_object_state(obj: ObjectState, new_cell_positions: PixelSet) -> ObjectState:
    """
    Update the object state with new cell positions
    """
    return ObjectState(
        label_id=obj.label_id,
        colour=obj.colour,
        grid_size=obj.grid_size,
        bounding_box=obj.bounding_box,
        centroid=obj.centroid,
        area=obj.area,
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
#     out_colour: set[int] = {1},
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
) -> ObjectState:
    """
    Unfill the object, only keeping the border of the object and setting the inside to background colour
    """
    cell_positions = obj.cell_positions
    new_cell_positions = set()
    # Get border cells only by checking if any of the 4 neighbors are background color
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


@object_primitive(tags={Effect.GROWTH})
def grow_object(
    obj: ObjectState, scale: int = 1, direction_vector: set[tuple[int, int]] = {(0, 0)}
) -> ObjectState:
    """
    Grow the object by scale factor.
    If direction vector is provided, grow in that direction
    """
    if direction_vector != {(0, 0)}:
        cell_positions = obj.cell_positions
        new_cell_positions = set()
        for cell in cell_positions:
            x, y = cell
            for dx in range(-scale, scale + 1):
                for dy in range(-scale, scale + 1):
                    if (dx, dy) in direction_vector:
                        new_cell_positions.add((x + dx, y + dy))
        return _update_object_state(obj, PixelSet(new_cell_positions))
    else:
        return obj


@object_primitive(tags={Effect.SHRINK})
def shrink_object(
    obj: ObjectState, scale: int = 1, direction_vector: set[tuple[int, int]] = {(0, 0)}
) -> ObjectState:
    """
    Shrink the object by scale factor.
    If direction vector is provided, shrink in that direction
    """
    if direction_vector != {(0, 0)}:
        cell_positions = obj.cell_positions
        new_cell_positions = set()
        for cell in cell_positions:
            x, y = cell
            for dx in range(-scale, scale + 1):
                for dy in range(-scale, scale + 1):
                    if (dx, dy) in direction_vector and (
                        x + dx,
                        y + dy,
                    ) in cell_positions:
                        new_cell_positions.add((x + dx, y + dy))
        return _update_object_state(obj, PixelSet(new_cell_positions))
    else:
        return obj


@object_primitive(tags={Effect.GROWTH})
def add_line_to_object(
    obj: ObjectState,
    direction_vector: tuple[int, int],
) -> ObjectState:
    """
    Add a line to the object in the middle of the bounding box in the specified direction
    Line should extent to the edge of the grid in the specified direction
    Start from edge of the object in the direction of the line
    """
    cell_positions = obj.cell_positions
    grid_shape = obj.grid_size
    # Get the edge of the object in the direction of the line
    edge_cells = set()
    for cell in cell_positions:
        x, y = cell
        dx, dy = direction_vector
        neighbor = (x + dx, y + dy)
        if neighbor not in cell_positions:
            edge_cells.add(cell)
    new_cell_positions = set(cell_positions)
    for cell in edge_cells:
        x, y = cell
        dx, dy = direction_vector
        for s in range(1, max(grid_shape)):
            new_cell = (x + s * dx, y + s * dy)
            if 0 <= new_cell[0] < grid_shape[0] and 0 <= new_cell[1] < grid_shape[1]:
                new_cell_positions.add(new_cell)
            else:
                break
    return _update_object_state(obj, PixelSet(new_cell_positions))


@object_primitive(tags={Effect.TRANSLATE})
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
