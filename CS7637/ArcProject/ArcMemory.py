"""
Property Memory Classes:
1. Grid Layer: Stores properties of the overall grid, such as dimensions, symmetry, and colour distribution
2. Object Layer: Stores properties of individual objects, such as shape, size, colour, and position
3. Cell layers: Stores properties of individual cells, such as colour and context. Context includes neighbouring cell properties

All state types use @dataclass(frozen=True) for immutability as these will be memory layers.
This is a requirement for downstream use in the DSL and possibly MCTS if implemented.

"""

__date__ = "2026-06-10"
__author__ = "NedeeshaWeerasuriya"
__version__ = "0.1"


import numpy as np
import functools
from dataclasses import dataclass, field
from typing import Callable, Any
from scipy import ndimage
from skimage.measure import label, regionprops

# -----------------------------------------------------------------------------
# Type Aliases
# -----------------------------------------------------------------------------
# Hashable types for DSL and possibly MCTS if implemented
GridArray = tuple[tuple[int, ...], ...]
PixelSet = frozenset[tuple[int, int]]
HuMoments = tuple[float, float, float, float, float, float, float]
Colour = int


# -----------------------------------------------------------------------------
# High level State
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ArcState:
    """
    Combined state for all memory layers.
    This class is the main interface for the DSL primitives and MCTS
    """

    grid_state: "GridState"
    objects: tuple["ObjectState"] = field(default_factory=tuple)

    def to_array(self) -> np.ndarray:
        """
        Renders the entire state back into a raw 2D numpy array.
        Used to feed existing numpy-based DSL primitives.
        """
        # Reconstruct grid from background and object layers
        rows, cols = self.grid_state.dimensions
        grid = np.zeros((rows, cols), dtype=int)
        # Fill in the background colour
        if self.grid_state.background_colour != 0:
            grid.fill(self.grid_state.background_colour)

        # Overlay objects onto the grid based on their positions
        for obj in self.objects:
            positions = getattr(obj, "cell_positions", getattr(obj, "cell_pos", []))
            colour = getattr(obj, "colour", getattr(obj, "colour", 0))
            for r, c in positions:
                if 0 <= r < rows and 0 <= c < cols:
                    grid[r, c] = colour
        return grid

    @classmethod
    def from_array(cls, array: np.ndarray, extract_objects: bool = True) -> "ArcState":
        """
        From an array, extract the grid and object layers to create an ArcState.
        If extract_objects is False, only the grid layer is created.
        """
        # Grid layer extraction
        g_state = GridState.from_array(array)
        grid_size = g_state.dimensions

        if not extract_objects:
            return cls(grid_state=g_state, objects=tuple())

        # Object layer extraction using skimage regionprops
        extracted_objects = tuple()
        unique_colours = np.unique(array)

        for colour in unique_colours:
            # Skip background
            if colour == g_state.background_colour:
                continue

            # Create a binary mask for the specific colour
            mask = array == colour
            labeled_mask, num_features = label(mask, return_num=True)
            props = regionprops(labeled_mask)

            for prop in props:
                obj_state = ObjectState.from_regionprops(
                    prop, colour=int(colour), grid_size=grid_size
                )
                extracted_objects += (obj_state,)

        return cls(grid_state=g_state, objects=extracted_objects)

    def update_grid(self, new_array: np.ndarray) -> "ArcState":
        """
        Update grid layer with new array but keep the object layer intact.
        """
        grid_tuple: GridArray = tuple(tuple(int(x) for x in row) for row in new_array)
        new_grid_state = GridState(matrix=grid_tuple, shape=new_array.shape)
        return ArcState(grid_state=new_grid_state, objects=self.objects)

    def update_object(self, index: int, new_obj_state: "ObjectState") -> "ArcState":
        """
        Update the object layer based on index but keep the grid layer intact.
        """
        obj_list = list(self.objects)
        obj_list[index] = new_obj_state
        new_objects = tuple(obj_list)

        rows, cols = self.grid_state.dimensions
        grid = np.zeros((rows, cols), dtype=int)
        if self.grid_state.background_colour != 0:
            grid.fill(self.grid_state.background_colour)
        for obj in new_objects:
            positions = getattr(obj, "cell_positions", getattr(obj, "cell_pos", []))
            colour = getattr(obj, "colour", getattr(obj, "colour", 0))
            for r, c in positions:
                if 0 <= r < rows and 0 <= c < cols:
                    grid[r, c] = colour
        new_grid_state = GridState.from_array(grid)

        return ArcState(grid_state=new_grid_state, objects=new_objects)


# -----------------------------------------------------------------------------
# Grid Level Memory
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class GridState:
    """
    Class to represent the memory of the grid properties.
    """

    grid: GridArray
    dimensions: tuple[int, int]
    colours: frozenset[Colour]
    background_colour: Colour

    # Symmetry properties
    horizontal_symmetry: bool
    vertical_symmetry: bool
    diagonal_symmetry: bool
    anti_diagonal_symmetry: bool

    # Rotational symmetry properties
    rotational_symmetry_90: bool
    rotational_symmetry_180: bool
    rotational_symmetry_270: bool

    @property
    def num_colours(self) -> int:
        return len(self.colours)

    @property
    def num_filled_cells(self) -> int:
        return sum(1 for row in self.grid for cell in row if cell != 0)

    @property
    def as_array(self) -> np.ndarray:
        return np.array(self.grid, dtype=int)

    def cell_value_at(self, row: int, col: int) -> int:
        return self.grid[row][col]

    @classmethod
    def from_array(cls, array: np.ndarray) -> "GridState":
        grid = tuple(tuple(int(cell) for cell in row) for row in array)
        dimensions = array.shape
        background_colour = cls.determine_background_colour(array)
        colours = frozenset(
            int(cell) for row in array for cell in row if cell != background_colour
        )

        # Symmetry checks using np.flip..
        horizontal_symmetry = np.array_equal(array, np.flipud(array))
        vertical_symmetry = np.array_equal(array, np.fliplr(array))
        diagonal_symmetry = np.array_equal(array, np.transpose(array))
        anti_diagonal_symmetry = np.array_equal(array, np.fliplr(np.transpose(array)))

        # Rotational symmetry checks using np.rot90..
        rotational_symmetry_90 = np.array_equal(array, np.rot90(array, k=1))
        rotational_symmetry_180 = np.array_equal(array, np.rot90(array, k=2))
        rotational_symmetry_270 = np.array_equal(array, np.rot90(array, k=3))

        return cls(
            grid=grid,
            dimensions=dimensions,
            colours=colours,
            horizontal_symmetry=horizontal_symmetry,
            vertical_symmetry=vertical_symmetry,
            diagonal_symmetry=diagonal_symmetry,
            anti_diagonal_symmetry=anti_diagonal_symmetry,
            rotational_symmetry_90=rotational_symmetry_90,
            rotational_symmetry_180=rotational_symmetry_180,
            rotational_symmetry_270=rotational_symmetry_270,
            background_colour=background_colour,
        )

    @staticmethod
    def determine_background_colour(array: np.ndarray) -> int:
        """
        Determine the background colour of the grid.
        If 0 is within the grid, it is considered the background colour.
        Otherwise, get the most frequent colour on the edges of the grid as the background colour.
        If a tie, return the colour that is at the edges of the grid.
        """
        values, counts = np.unique(array, return_counts=True)
        if 0 in values:
            return 0
        modes = values[np.where(counts == np.max(counts))]
        if len(modes) == 1:
            return int(modes[0])
        else:
            edge_values = np.concatenate(
                (
                    [
                        array[0, :],
                        array[-1, :],
                        array[:, 0],
                        array[:, -1],
                    ]
                )
            )

            edge_values, edge_counts = np.unique(edge_values, return_counts=True)
            edge_modes = edge_values[np.where(edge_counts == np.max(edge_counts))]
            return int(edge_modes[0])


# -----------------------------------------------------------------------------
# Object Level Memory
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ObjectState:
    """
    Single object in a dataset
    Frame like struture storing:
    1. cell positions of the object
    2. colour of the object
    3. General properties of the object (from scikit image regionprops)

    Create normalised and invariant version using hu moments.
    Hu moments will provide a sanity check to determine if objects are comparable
    Store raw moments to determine the transformation between the objects from input and output
    """

    label_id: int
    colour: Colour
    grid_size: tuple[int, int]  # (rows, cols) of the grid containing the object

    bounding_box: tuple[int, int, int, int]  # (min_row, max_row, min_col, max_col)
    centroid: tuple[float, float]
    area: int
    cell_positions: PixelSet
    hu_moments: HuMoments = field(default_factory=lambda: (0.0,) * 7)

    # mutation properties to be filled in heuristics analysis
    mutation_types: frozenset[str] = field(default_factory=frozenset)
    mutation_vectors: tuple[tuple[float, ...], ...] = field(default_factory=tuple)

    # For overwriting priority in grid recreation
    priority: int = 0

    def _replace(self, **kwargs) -> "ObjectState":
        """
        Helper method to update ObjectState with some properties updated.
        """
        return ObjectState(
            label_id=kwargs.get("label_id", self.label_id),
            colour=kwargs.get("colour", self.colour),
            grid_size=kwargs.get("grid_size", self.grid_size),
            bounding_box=kwargs.get("bounding_box", self.bounding_box),
            centroid=kwargs.get("centroid", self.centroid),
            area=kwargs.get("area", self.area),
            cell_positions=kwargs.get("cell_positions", self.cell_positions),
            hu_moments=kwargs.get("hu_moments", self.hu_moments),
            mutation_types=kwargs.get("mutation_types", self.mutation_types),
            mutation_vectors=kwargs.get("mutation_vectors", self.mutation_vectors),
        )

    @property
    def height(self) -> int:
        return self.bounding_box[1] - self.bounding_box[0] + 1

    @property
    def width(self) -> int:
        return self.bounding_box[3] - self.bounding_box[2] + 1

    @property
    def size(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def x_coords(self) -> frozenset[int]:
        return frozenset(col for _, col in self.cell_positions)

    @property
    def y_coords(self) -> frozenset[int]:
        return frozenset(row for row, _ in self.cell_positions)

    @property
    def isolated_grid(self) -> np.ndarray:
        """
        Get the object in a new grid that has the size of the bounding box of the object.
        """
        bounding_box = self.bounding_box
        output = np.zeros((self.height, self.width), dtype=int)
        for pos in self.cell_positions:
            output[pos[0] - bounding_box[0], pos[1] - bounding_box[2]] = self.colour
        return output

    @property
    def normalised_pixels(self) -> PixelSet:
        """
        Get the pixel positions normalised to (0,0)
        """
        min_row, max_row, min_col, max_col = self.bounding_box
        return frozenset(
            (row - min_row, col - min_col) for (row, col) in self.cell_positions
        )

    @property
    def mask(self) -> np.ndarray:
        """
        Get the binary mask of the object in its bounding box.
        """
        mask = np.zeros((self.height, self.width), dtype=bool)
        for row, col in self.normalised_pixels:
            mask[row, col] = True
        return mask

    @property
    def is_closed(self) -> bool:
        """
        Check if the object is closed (connected and has no holes but could be empty inside)
        """
        mask = self.mask
        if ndimage.label(mask)[1] != 1:
            return False
        filled = ndimage.binary_fill_holes(mask)
        return not np.array_equal(mask, filled)

    @classmethod
    def from_regionprops(
        cls,
        prop,
        grid_size: tuple[int, int],
        colour: int,
    ) -> "ObjectState":
        """
        Build an ObjectState from a skimage regionprops entry
        """
        pixels: PixelSet = frozenset(map(tuple, prop.coords))

        raw_hu = prop.moments_hu
        log_hu = tuple(np.sign(h) * np.log10(np.abs(h) + 1e-10) for h in raw_hu)

        bbox = prop.bbox
        centroid = prop.centroid

        return cls(
            label_id=prop.label,
            colour=colour,
            bounding_box=(bbox[0], bbox[2], bbox[1], bbox[3]),
            centroid=(centroid[0], centroid[1]),
            area=int(prop.area),
            cell_positions=pixels,
            hu_moments=log_hu,
            grid_size=grid_size,
        )
