import numpy as np

from ArcProblem import ArcProblem
from ArcData import ArcData
from ArcSet import ArcSet


class ArcObject:
    """
    Single object in a dataset
    """

    def __init__(self, cell_pos=None, color=None):
        # Assume single shape has a single colour for now
        self.cell_pos = cell_pos
        self.color = color

        self.num_cells = len(cell_pos) if cell_pos is not None else 0

    def bounding_box(self):
        """
        Get the bounding box of the object.
        """
        rows = [pos[0] for pos in self.cell_pos]
        cols = [pos[1] for pos in self.cell_pos]
        return (min(rows), max(rows), min(cols), max(cols))

    def shape(self):
        """
        Get the shape of the object as a tuple (num_rows, num_cols).
        """
        bounding_box = self.bounding_box()
        return (
            bounding_box[1] - bounding_box[0] + 1,
            bounding_box[3] - bounding_box[2] + 1,
        )

    def relative_position(self, other):
        """
        Get position of object within new grid the size of the bounding box of the object.
        """
        bounding_box = self.bounding_box()
        relative_pos = set()
        for pos in self.cell_pos:
            relative_pos.add((pos[0] - bounding_box[0], pos[1] - bounding_box[2]))
        return relative_pos

    def isolated_object(self) -> ArcData:
        """
        Get the object in a new grid that has the size of the bounding box of the object.
        """
        bounding_box = self.bounding_box()
        output = np.zeros(self.shape(), dtype=int)
        for pos in self.cell_pos:
            output[pos[0] - bounding_box[0], pos[1] - bounding_box[2]] = self.color
        return ArcData(output)


class ArcAgent:
    def __init__(self):
        """
        You may add additional variables to this init method. Be aware that it gets called only once
        and then the make_predictions method will get called several times.
        """
        self.colour_matches = True
        self.grid_size_matches = True
        self.all_input_kept = True
        self.num_cells_changed = None

    # --------------------------------------------------------------------------
    # Initial Checks (Level 0)
    # --------------------------------------------------------------------------
    def _check_grid_size(self, training_data: list[ArcSet]) -> bool:
        """
        Check training data if the input grid size always matches the output grid size.
        """
        for entry in training_data:
            if entry.get_input_data().shape() != entry.get_output_data().shape():
                self.grid_size_matches = False
                break

        print("Grid size matches in training data: " + str(self.grid_size_matches))

    def _check_color_count(self, training_data: list[ArcSet]) -> bool:
        """
        Check training data if the number of colors in the input always matches the number of colors in the output.
        """
        for entry in training_data:
            if len(np.unique(entry.get_input_data().data())) != len(
                np.unique(entry.get_output_data().data())
            ):
                self.colour_matches = False
                break

    def _check_input_kept(self, training_data: list[ArcSet]) -> bool:
        """
        Check training data if the output contains the same cells as the input.
        Irrespective if other cells are added.
        Note if colour has changed
        """
        for entry in training_data:
            if not set(np.unique(entry.get_input_data().data())).issubset(
                set(np.unique(entry.get_output_data().data()))
            ):
                self.all_input_kept = False
                break

    def _check_num_cells_changed(self, training_data: list[ArcSet]) -> bool:
        """
        Count the number of cells that are changed from input to output in the training data.
        """
        train_list = []
        for entry in training_data:
            num_cells_changed = np.sum(
                entry.get_input_data().data() != entry.get_output_data().data()
            )
            train_list.append(num_cells_changed)
        # Note if no change, or all same change. Otherwise feature remains None and is not used in the agent.
        if len(set(train_list)) == 1:
            self.num_cells_changed = train_list[0]

    def _identify_objects(self, grid) -> list[ArcObject]:
        objects = []
        for color in np.unique(grid):
            if color == 0:
                continue
            cell_pos = set(zip(*np.where(grid == color)))
            objects.append(ArcObject(cell_pos=cell_pos, color=color))
        return objects

    def run_initial_checks(self, training_data):
        """
        Run all initial checks on the training data and set the corresponding variables in the agent.
        """
        self._check_grid_size(training_data)
        self._check_color_count(training_data)
        self._check_input_kept(training_data)
        if self.grid_size_matches:
            self._check_num_cells_changed(training_data)

    # --------------------------------------------------------------------------
    # Level 1 Checks
    # --------------------------------------------------------------------------

    def check_objects_remained(self, training_data: list[ArcSet]) -> int:
        """
        Check in the training data, if the output contains the same objects as the input even if moved
        If all training data has the same number of objects matched, return that number.
        Otherwise return None for future implementation.
        """
        num_objects_matched = []
        for entry in training_data:
            input_objects = self._identify_objects(entry.get_input_data().data())
            output_objects = self._identify_objects(entry.get_output_data().data())
            # Account for differences in grid size between input and output
            obj_matched = []
            for i, input_obj in enumerate(input_objects):
                input_obj_rel_pos = input_obj.relative_position(input_obj)
                for output_obj in output_objects:
                    output_obj_rel_pos = output_obj.relative_position(output_obj)
                    if input_obj_rel_pos == output_obj_rel_pos:
                        obj_matched.append(True)
                    else:
                        obj_matched.append(False)
            num_objects_matched.append(sum(obj_matched))

        if len(set(num_objects_matched)) == 1:
            return num_objects_matched[0]
        else:
            return 0

    def check_propagation_pattern(self, training_data: list[ArcSet]) -> bool:
        """
        Check in the training data,
        if there is a propagation pattern where the output is a shifted version of the input.
        """
        for entry in training_data:
            input_data = entry.get_input_data()
            output_data = entry.get_output_data()
            if not np.array_equal(
                input_data.data(), np.roll(output_data.data(), shift=1, axis=0)
            ):
                return False
        return True

    # -----------------------------------------------------------------------------
    # Make Predictions
    # -----------------------------------------------------------------------------

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        """
        Write the code in this method to solve the incoming ArcProblem.
        Your agent will receive 1 problem at a time.

        You can add up to THREE (3) the predictions to the
        predictions list provided below that you need to
        return at the end of this method.

        In the Autograder, the test data output in the arc problem will be set to None
        so your agent cannot peek at the answer (even on the public problems).

        Also, if you return more than 3 predictions in the list it
        is considered an ERROR and the test will be automatically
        marked as INCORRECT.
        """
        print("Analysing Problem: " + arc_problem.problem_name())
        input_test = arc_problem.test_set().get_input_data()

        predictions: list[np.ndarray] = list()

        # Run initial checks on the training data
        self.run_initial_checks(arc_problem.training_set())

        if self.check_objects_remained(arc_problem.training_set()) > 0:
            print("Objects remained in the training data.")
            if self.grid_size_matches:
                # Produce predictions with the objects in the same position as the input
                input_object = self._identify_objects(input_test.data())[0]
                output = np.zeros_like(input_test.data())
                for pos in input_object.cell_pos:
                    output[pos] = input_object.color
                predictions.append(output)
            else:
                # Produce predictions with the objects in the same relative position as the input
                input_object = self._identify_objects(input_test.data())[0]
                predictions.append(input_object.isolated_object().data())
        return predictions
