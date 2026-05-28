import numpy as np

from ArcProblem import ArcProblem
from ArcData import ArcData
from ArcSet import ArcSet


class ArcAgent:
    def __init__(self):
        """
        You may add additional variables to this init method. Be aware that it gets called only once
        and then the make_predictions method will get called several times.
        """
        self.colour_matches = True
        self.size_matches = True
        self.all_input_kept = True
        self.num_cells_changed = None

    # --------------------------------------------------------------------------
    # Helper functions
    # --------------------------------------------------------------------------
    def _combined_training_data(self, arc_problem: ArcProblem) -> list[ArcSet]:
        """
        Helper function to zip the training input and output data into a list
        """
        training_input = arc_problem.training_set().get_input_data()
        training_output = arc_problem.training_set().get_output_data()
        return list(zip(training_input, training_output))

    # --------------------------------------------------------------------------
    # Initial Checks (Level 0)
    # --------------------------------------------------------------------------
    def _check_size(self, training_data) -> bool:
        """
        Check training data if the input size always matches the output size.
        """
        for input_data, output_data in training_data:
            if input_data.data().shape != output_data.data().shape:
                self.size_matches = False
                break

    def _check_color_count(self, training_data) -> bool:
        """
        Check training data if the number of colors in the input always matches the number of colors in the output.
        """
        for input_data, output_data in training_data:
            if len(np.unique(input_data.data())) != len(np.unique(output_data.data())):
                self.colour_matches = False
                break

    def _check_input_kept(self, training_data) -> bool:
        """
        Check training data if the output contains the same cells as the input.
        Irrespective if other cells are added.
        Note if colour has changed
        """
        for input_data, output_data in training_data:
            if not set(np.unique(input_data.data())).issubset(
                set(np.unique(output_data.data()))
            ):
                self.all_input_kept = False
                break

    def _check_num_cells_changed(self, training_data) -> bool:
        """
        Count the number of cells that are changed from input to output in the training data.
        """
        train_list = []
        for input_data, output_data in training_data:
            num_cells_changed = np.sum(input_data.data() != output_data.data())
            train_list.append(num_cells_changed)
        # Note if no change, or all same change. Otherwise feature remains None and is not used in the agent.
        if len(set(train_list)) == 1:
            self.num_cells_changed = train_list[0]

    # --------------------------------------------------------------------------
    # Level 1 Checks
    # --------------------------------------------------------------------------

    def check_propagation_pattern(self, training_data) -> bool:
        """
        Check in the training data, if there is a propagation pattern where the output is a shifted version of the input.
        """
        for input_data, output_data in training_data:
            if not np.array_equal(
                input_data.data(), np.roll(output_data.data(), shift=1, axis=0)
            ):
                return False
        return True

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
        training_data = self._combined_training_data(arc_problem)

        # Check if the size

        predictions: list[np.ndarray] = list()

        """
        The next 2 lines are only an example of how to populate the predictions list.
        This will just be an empty answer the size of the input data;
        delete it before you start adding your own predictions.
        """
        output = np.zeros_like(arc_problem.test_set().get_input_data().data())
        predictions.append(output)

        return predictions
