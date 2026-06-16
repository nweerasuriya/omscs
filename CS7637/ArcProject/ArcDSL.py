"""
Primitive DSL for ArcAgi
Includes basic operations for transformation fo cells and objects
All operations should be generic and be able to ingest either grids or objects or cells if possible
"""

__date__ = "2026-06-10"
__author__ = "NedeeshaWeerasuriya"
__version__ = "0.1"


import numpy as np
from ArcMemory import grid_primitive, object_primitive, ObjectState, GridState, PixelSet


# -----------------------------------------------------------------------------
# Grid level functions
# -----------------------------------------------------------------------------
@grid_primitive
def rotation_90(input_array: np.ndarray) -> np.ndarray:
    return np.rot90(input_array, k=1)


@grid_primitive
def rotation_180(input_array: np.ndarray) -> np.ndarray:
    return np.rot90(input_array, k=2)


@grid_primitive
def rotation_270(input_array: np.ndarray) -> np.ndarray:
    return np.rot90(input_array, k=3)


@grid_primitive
def flip_horizontal(input_array: np.ndarray) -> np.ndarray:
    return np.fliplr(input_array)


@grid_primitive
def flip_vertical(input_array: np.ndarray) -> np.ndarray:
    return np.flipud(input_array)


@grid_primitive
def flip_diagonal(input_array: np.ndarray) -> np.ndarray:
    return np.transpose(input_array)


@grid_primitive
def flip_anti_diagonal(input_array: np.ndarray) -> np.ndarray:
    return np.fliplr(np.transpose(input_array))


@grid_primitive
def crop_background_out(
    input_array: np.ndarray, background_color: int = 0
) -> np.ndarray:
    rows = np.any(input_array != background_color, axis=1)
    cols = np.any(input_array != background_color, axis=0)
    return input_array[np.ix_(rows, cols)]


@grid_primitive
def change_colour(
    input_array: np.ndarray, in_colour: int, out_colour: int
) -> np.ndarray:
    grid = input_array.copy()
    return np.where(grid == in_colour, out_colour, grid)


@grid_primitive
def update_colour(
    input_array: np.ndarray, new_colours: list[int], removed_colours: list[int]
) -> np.ndarray:
    """
    Update the colours in the input array by replacing the removed colours with the new colours.
    """
    grid = input_array.copy()
    for old_colour, new_colour in zip(removed_colours, new_colours):
        grid = np.where(grid == old_colour, new_colour, grid)
    return grid


@grid_primitive
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


@grid_primitive
def colour_grey_to_black(input_array: np.ndarray, grey: int = 5) -> np.ndarray:
    grid = input_array.copy()
    return np.where(grid == grey, 0, grid)


# -----------------------------------------------------------------------------
# Horizontal mirrors
# -----------------------------------------------------------------------------


@grid_primitive
def mirror_horizontal_top_overwrite(input_array: np.ndarray) -> np.ndarray:
    """Overwrite the bottom half rows with a flipped top half."""
    grid = input_array.copy()
    half = grid.shape[0] // 2
    if half == 0:
        return grid
    grid[grid.shape[0] - half :, :] = np.flipud(grid[:half, :])
    return grid


@grid_primitive
def mirror_horizontal_bottom_overwrite(input_array: np.ndarray) -> np.ndarray:
    """Overwrite the top half rows with a flipped bottom half."""
    grid = input_array.copy()
    half = grid.shape[0] // 2
    if half == 0:
        return grid
    grid[:half, :] = np.flipud(grid[grid.shape[0] - half :, :])
    return grid


@grid_primitive
def mirror_horizontal_top_overlap(input_array: np.ndarray) -> np.ndarray:
    """
    Keep only cells where the top half rows match the flipped bottom half
    """
    grid = input_array.copy()
    half = grid.shape[0] // 2
    if half == 0:
        return grid
    top = grid[:half, :]
    bottom = np.flipud(grid[grid.shape[0] - half :, :])
    grid[:half, :] = np.where(top == bottom, top, 0)
    return grid


@grid_primitive
def mirror_horizontal_bottom_overlap(input_array: np.ndarray) -> np.ndarray:
    """
    Keep only cells where the bottom half rows match the flipped top half
    Keep only bottom half
    """
    grid = input_array.copy()
    half = grid.shape[0] // 2
    if half == 0:
        return grid
    top = grid[:half, :]
    bottom = grid[grid.shape[0] - half :, :]
    flipped_top = np.flipud(top)
    grid[grid.shape[0] - half :, :] = np.where(bottom == flipped_top, bottom, 0)
    return grid


# -----------------------------------------------------------------------------
# Vertical mirrors — left to right
# -----------------------------------------------------------------------------
@grid_primitive
def mirror_vertical_left_overwrite(input_array: np.ndarray) -> np.ndarray:
    """Overwrite the right half with the left half flipped."""
    grid = input_array.copy()
    half = grid.shape[1] // 2
    if half == 0:
        return grid
    grid[:, grid.shape[1] - half :] = np.fliplr(grid[:, :half])
    return grid


@grid_primitive
def mirror_vertical_right_overwrite(input_array: np.ndarray) -> np.ndarray:
    """Overwrite the left half with the right half flipped."""
    grid = input_array.copy()
    half = grid.shape[1] // 2
    if half == 0:
        return grid
    grid[:, :half] = np.fliplr(grid[:, grid.shape[1] - half :])
    return grid


@grid_primitive
def mirror_vertical_left_overlap(input_array: np.ndarray) -> np.ndarray:
    """
    Keep only cells where the left half match the flipped right half
    """
    grid = input_array.copy()
    half = grid.shape[1] // 2
    if half == 0:
        return grid
    left = grid[:, :half]
    right = np.fliplr(grid[:, grid.shape[1] - half :])
    grid[:, :half] = np.where(left == right, left, 0)
    return grid


@grid_primitive
def mirror_vertical_right_overlap(input_array: np.ndarray) -> np.ndarray:
    """
    Keep only cells where the right half match the flipped left half
    """
    grid = input_array.copy()
    half = grid.shape[1] // 2
    if half == 0:
        return grid
    left = grid[:, :half]
    right = grid[:, grid.shape[1] - half :]
    flipped_left = np.fliplr(left)
    grid[:, grid.shape[1] - half :] = np.where(right == flipped_left, right, 0)
    return grid


# -----------------------------------------------------------------------------
# Diagonal mirrors
# -----------------------------------------------------------------------------


@grid_primitive
def mirror_diagonal_top_left(input_array: np.ndarray) -> np.ndarray:
    """
    Overwrite the bottom right triangle with the transpose of the top left
    triangle, reflecting across the main diagonal.
    """
    grid = input_array.copy()
    h, w = grid.shape
    rows, cols = np.indices((h, w))
    lower_mask = rows > cols
    grid[lower_mask] = grid.T[lower_mask]
    return grid


@grid_primitive
def mirror_diagonal_bottom_right(input_array: np.ndarray) -> np.ndarray:
    """
    Overwrite the top-left triangle with the transpose of the bottom right triangle.
    """
    grid = input_array.copy()
    h, w = grid.shape
    rows, cols = np.indices((h, w))
    upper_mask = rows < cols  # strictly above main diagonal
    grid[upper_mask] = grid.T[upper_mask]
    return grid


# -----------------------------------------------------------------------------
# Object Primitives
# -----------------------------------------------------------------------------
# TODO: Account for cells not bordering bounding box
@object_primitive
def fill_object(
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
    new_cell_positions = PixelSet(new_cell_positions)
    return ObjectState(
        label_id=obj.label_id,
        colour=obj.colour,
        bounding_box=bounding_box,
        centroid=obj.centroid,
        area=obj.area,
        cell_positions=new_cell_positions,
        hu_moments=obj.hu_moments,
    )


@object_primitive
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

    return ObjectState(
        label_id=obj.label_id,
        colour=obj.colour,
        bounding_box=obj.bounding_box,
        centroid=obj.centroid,
        area=obj.area,
        cell_positions=new_cell_positions,
        hu_moments=obj.hu_moments,
    )


@object_primitive
def crop_object(obj: ObjectState) -> ObjectState:
    """
    Crop the object to its bounding box
    """
    x_min, y_min, x_max, y_max = obj.bounding_box
    return ObjectState(
        label_id=obj.label_id,
        colour=obj.colour,
        bounding_box=obj.bounding_box,
        centroid=obj.centroid,
        area=obj.area,
        cell_positions=PixelSet(
            {
                (x, y)
                for (x, y) in obj.cell_positions
                if x_min <= x < x_max and y_min <= y < y_max
            }
        ),
        hu_moments=obj.hu_moments,
    )
