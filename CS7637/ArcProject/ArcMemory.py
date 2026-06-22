"""
Property Memory Classes:
1. Grid Layer: Stores properties of the overall grid, such as dimensions, symmetry, and color distribution
2. Object Layer: Stores properties of individual objects, such as shape, size, color, and position
3. Cell layers: Stores properties of individual cells, such as color and context. Context includes neighbouring cell properties

All state types use @dataclass(frozen=True) for immutability as these will be memory layers.
This is a requirement for downstream use in the DSL and possibly MCTS if implemented.

"""

__date__ = "2026-06-10"
__author__ = "NedeeshaWeerasuriya"
__version__ = "0.1"


import numpy as np
import functools
from dataclasses import dataclass, field
from skimage.measure import label, regionprops

# -----------------------------------------------------------------------------
# Type Aliases
# -----------------------------------------------------------------------------
# Hashable types for DSL and possibly MCTS if implemented
GridArray = tuple[tuple[int, ...], ...]
PixelSet = frozenset[tuple[int, int]]
HuMoments = tuple[float, float, float, float, float, float, float]
Colour = int


GRID_PRIMITIVES: list[callable] = []
OBJECT_PRIMITIVES: list[callable] = []
CELL_PRIMITIVES: list[callable] = []
SPLIT_GRID_PRIMITIVES: list[callable] = []


def grid_primitive(func=None, *, tags=None):
    """
    Decorator to automatically transition an ArcState to a array and back.
    Add tags to help in pruning.
    """

    def decorator(inner_func):
        @functools.wraps(inner_func)
        def wrapper(state: ArcState, *args, **kwargs) -> ArcState:
            raw_grid = state.to_array()
            transformed_grid = inner_func(raw_grid, *args, **kwargs)
            return ArcState.from_array(transformed_grid, extract_objects=True)

        wrapper.tags = frozenset(tags or ())
        GRID_PRIMITIVES.append(wrapper)
        return wrapper

    if func is None:
        return decorator
    return decorator(func)


def object_primitive(func=None, *, tags=None):
    """
    Decorator to apply a transformation directly to the object layer of an ArcState.
    """

    def decorator(inner_func):
        @functools.wraps(inner_func)
        def wrapper(state: ArcState, *args, **kwargs) -> ArcState:
            transformed_objects = [
                inner_func(obj, *args, **kwargs) for obj in state.objects
            ]
            # Update grid state based on transformed objects
            rows, cols = state.grid_state.dimensions
            grid = np.zeros((rows, cols), dtype=int)
            for obj in transformed_objects:
                for r, c in obj.cell_positions:
                    if 0 <= r < rows and 0 <= c < cols:
                        grid[r, c] = obj.colour

            new_grid_state = GridState.from_array(grid)
            return ArcState(
                grid_state=new_grid_state, objects=tuple(transformed_objects)
            )

        wrapper.tags = frozenset(tags or ())
        OBJECT_PRIMITIVES.append(wrapper)
        return wrapper

    if func is None:
        return decorator
    return decorator(func)


def _get_grid_splits(
    grid: np.ndarray, split_type: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split the grid into two halves based on the split axis.
    """
    rows, cols = grid.shape
    if type(split_type) is set and len(split_type) == 1:
        split_type = split_type.pop()
    if split_type == "horizontal":
        mid_row = rows // 2
        top_half = grid[:mid_row, :]
        bottom_half = grid[mid_row:, :]
        if top_half.shape[0] != bottom_half.shape[0]:
            return None
        return top_half, bottom_half
    if split_type == "vertical":
        print("Splitting grid vertically")
        mid_col = int(cols / 2)
        left_half = grid[:, :mid_col]
        right_half = grid[:, mid_col:]
        if left_half.shape != right_half.shape:
            return None
        return left_half, right_half

    if split_type == "diagonal":
        if rows != cols:
            return None
        top_left = grid[: rows // 2, : cols // 2]
        bottom_right = grid[rows // 2 :, cols // 2 :]
        if top_left.shape != bottom_right.shape:
            return None
        return top_left, bottom_right

    if split_type == "anti-diagonal":
        if rows != cols:
            return None
        top_right = grid[: rows // 2, cols // 2 :]
        bottom_left = grid[rows // 2 :, : cols // 2]
        if top_right.shape != bottom_left.shape:
            return None
        return top_right, bottom_left


# TODO: Expand to more than just halves but split at non-midpoint too
def split_grid_primitive(func=None, *, tags=None):
    """
    Decorator for transformations that operate on split grids.
    """

    def decorator(inner_func):
        @functools.wraps(inner_func)
        def wrapper(state: ArcState, split_axis: str) -> ArcState:
            # Split the grid into halves
            halves = _get_grid_splits(state.to_array(), split_type=split_axis)
            if halves is None:
                return state
            array_a, array_b = halves
            print(
                f"Applying split grid primitive {inner_func.__name__} on axis {split_axis}"
            )
            return ArcState.from_array(
                inner_func(array_a, array_b), extract_objects=True
            )

        wrapper.tags = frozenset(tags or ())
        SPLIT_GRID_PRIMITIVES.append(wrapper)
        return wrapper

    if func is None:
        return decorator
    return decorator(func)


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
    objects: list["ObjectState"] = field(default_factory=list)

    def to_array(self) -> np.ndarray:
        """
        Renders the entire state back into a raw 2D numpy array.
        Used to feed existing numpy-based DSL primitives.
        """
        # Reconstruct grid from background and object layers
        rows, cols = self.grid_state.dimensions
        grid = np.zeros((rows, cols), dtype=int)

        # Overlay objects onto the grid based on their positions
        for obj in self.objects:
            positions = getattr(obj, "cell_positions", getattr(obj, "cell_pos", []))
            color = getattr(obj, "color", getattr(obj, "colour", 0))
            for r, c in positions:
                if 0 <= r < rows and 0 <= c < cols:
                    grid[r, c] = color
        return grid

    @classmethod
    def from_array(cls, array: np.ndarray, extract_objects: bool = True) -> "ArcState":
        """
        From an array, extract the grid and object layers to create an ArcState.
        If extract_objects is False, only the grid layer is created.
        """
        # Grid layer extraction
        g_state = GridState.from_array(array)

        if not extract_objects:
            return cls(grid_state=g_state, objects=[])

        # Object layer extraction using skimage regionprops
        extracted_objects = []
        unique_colors = np.unique(array)

        for color in unique_colors:
            # Skip background
            if color == 0:
                continue

            # Create a binary mask for the specific color
            mask = array == color
            labeled_mask, num_features = label(mask, return_num=True)
            props = regionprops(labeled_mask)

            for prop in props:
                obj_state = ObjectState.from_regionprops(prop, colour=int(color))
                extracted_objects.append(obj_state)

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
        new_objects = obj_list

        rows, cols = self.grid_state.dimensions
        grid = np.zeros((rows, cols), dtype=int)
        for obj in new_objects:
            positions = getattr(obj, "cell_positions", getattr(obj, "cell_pos", []))
            color = getattr(obj, "color", getattr(obj, "colour", 0))
            for r, c in positions:
                if 0 <= r < rows and 0 <= c < cols:
                    grid[r, c] = color
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
        colours = frozenset(int(cell) for row in array for cell in row if cell != 0)

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
        )


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

    bounding_box: tuple[int, int, int, int]  # (min_row, max_row, min_col, max_col)
    centroid: tuple[float, float]
    area: int
    cell_positions: PixelSet
    hu_moments: HuMoments

    # mutation properties to be filled in heuristics analysis
    mutation_types: frozenset[str] = field(default_factory=frozenset)
    mutation_vectors: tuple[tuple[float, ...], ...] = field(default_factory=tuple)

    def _replace(self, **kwargs) -> "ObjectState":
        """
        Helper method to update ObjectState with some properties updated.
        """
        return ObjectState(
            label_id=kwargs.get("label_id", self.label_id),
            colour=kwargs.get("colour", self.colour),
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

    @classmethod
    def from_regionprops(
        cls,
        prop,
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
        )
