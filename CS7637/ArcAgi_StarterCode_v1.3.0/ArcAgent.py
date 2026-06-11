import numpy as np
from skimage.measure import label, regionprops

from ArcProblem import ArcProblem
from ArcData import ArcData
from ArcSet import ArcSet
from ArcObject import ArcObject
from ArcTransformation import ArcTransformation


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
        self.similar_objects = {}

        # Testing variables
        self.test_passed = False

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

    # %% --------------------------------------------------------------------------
    # Object Identification and Comparison (Level 1)
    # -----------------------------------------------------------------------------
    def _identify_objects(self, grid) -> list[ArcObject]:
        """
        Identify objects in a grid and return a list of ArcObjects.
        Use scikit-image's label to identify connected components using 2x2 connectivity
        (cells that touching diagonally are considered connected).
        For each connected component of the same colour, create an ArcObject
        """
        objects = []
        for color in np.unique(grid):
            if color == 0:
                continue
            # Require mask for label function, for each colour
            mask = grid == color
            labeled_mask = label(mask, connectivity=2)
            # Get properties of labeled region
            for props in regionprops(labeled_mask):
                cell_pos = props.coords
                objects.append(
                    ArcObject(color=color, cell_pos=cell_pos, general_props=props)
                )
        return objects

    def check_similar_objects(self, training_data: list[ArcSet]) -> list[dict]:
        """
        Check in the training data, if there are objects that are similar from input to output.
        Use the invariant properties of the objects to determine if they are the same object or not.
        """
        tolerance = 0.5  # Set a tolerance level for comparing Hu moments
        object_frames = []
        for id, entry in enumerate(training_data):
            single_frame = {}
            single_frame["set_id"] = id
            input_objects = self._identify_objects(entry.get_input_data().data())
            output_objects = self._identify_objects(entry.get_output_data().data())
            for input_object in input_objects:
                # Use the log Hu moments to find if object is similar in input and output
                # Use euclidean distance between log Hu moments to determine if objects are the same
                if input_object.hu_moments is not None:
                    distances = [
                        np.linalg.norm(input_object.log_hu - output_object.log_hu)
                        for output_object in output_objects
                    ]
                    # store any objects that pass the threshold as similar objects
                    for i, distance in enumerate(distances):
                        if distance < tolerance:
                            single_frame["input_object"] = input_object
                            single_frame["output_object"] = output_objects[i]
                            single_frame["distance"] = distance
            object_frames.append(single_frame)
        return object_frames

    def run_initial_checks(self, training_data):
        """
        Run all initial checks on the training data and set the corresponding variables in the agent.
        """
        self._check_grid_size(training_data)
        self._check_color_count(training_data)
        self._check_input_kept(training_data)
        if self.grid_size_matches:
            self._check_num_cells_changed(training_data)
        self.similar_objects = self.check_similar_objects(training_data)

    # --------------------------------------------------------------------------
    # Basic Transformations
    # --------------------------------------------------------------------------
    def find_transformations_between_objects(
        self, object_frames: list[dict]
    ) -> ArcTransformation:
        """
        If there are similar objects between input and output,
        find the associated transformations between the objects using the raw moments of the objects.
        """
        transformation_list = []
        for entry in object_frames:
            transformation = ArcTransformation()
            transformation.set_id = entry["set_id"]
            input_object_props = entry["input_object"].general_props
            output_object_props = entry["output_object"].general_props
            # Use the centroid of the objects to determine the translation between the objects
            translation_x = (
                output_object_props.centroid[0] - input_object_props.centroid[0]
            )
            translation_y = (
                output_object_props.centroid[1] - input_object_props.centroid[1]
            )

            # if both translations are 0, then likely a colour change or rotation
            if translation_x == 0 and translation_y == 0:
                transformation.translation = None
                # Check colour change
                if entry["input_object"].color != entry["output_object"].color:
                    transformation.color = (
                        entry["input_object"].color,
                        entry["output_object"].color,
                    )
                # Check rotations
                for i, input_rotation in enumerate(
                    entry["input_object"].all_rotations()
                ):
                    if np.array_equal(
                        input_rotation, entry["output_object"].isolated_object().data()
                    ):
                        rotation_angle = i * 90
                        transformation.rotation = rotation_angle
            else:
                transformation.translation = (translation_x, translation_y)

            transformation_list.append(transformation)

        # Check if all transformation in the list are identical
        if all(
            transformation_list[0].__dict__ == transformation.__dict__
            for transformation in transformation_list
        ):
            return transformation_list[0]
        else:
            return None

    def check_non_object_transformation(
        self, training_data: list[ArcSet], object_frames: list[dict]
    ) -> dict:
        """
        If there are similar objects but no clear object-based transformation, based on grid size matching check:
        1. If new grid size is same as bounding box of the object, then likely a cropping transformation.
        2. If object maintained but new cells are added, compare the new cells to find pattern
            temp: just add new cells to transformation dict for now, could be used to find a pattern in the new cells in future work.
        """
        # Check if grid size of output matches bounding box of object in output
        transformation_list = []
        for entry in object_frames:
            transformation = ArcTransformation()
            transformation.set_id = entry["set_id"]
            training_set = training_data[entry["set_id"]]
            # 1. Check cropping transformation based on grid size
            if training_set.get_output_data().shape() == entry["input_object"].shape():
                transformation.cropping = True
            # 2. Check if new cells are added and if so, which ones
            if self.all_input_kept and self.num_cells_changed is not None:
                input_cells = set(tuple(pos) for pos in entry["input_object"].cell_pos)
                output_cells = set(
                    tuple(pos) for pos in entry["output_object"].cell_pos
                )
                new_cells = output_cells - input_cells
                transformation.new_cells = new_cells
            transformation_list.append(transformation)
        # Check if all transformation in the list are identical
        if all(
            transformation_list[0].__dict__ == transformation.__dict__
            for transformation in transformation_list
        ):
            return transformation_list[0]
        else:
            return None

    def transform(self, obj: ArcObject, transformation: ArcTransformation) -> ArcObject:
        """
        Transform the input object based on the identified transformation.
        """
        transformed_obj = obj
        if transformation.translation is not None:
            transformed_obj = transformation.translate(transformed_obj)
        if transformation.rotation is not None:
            transformed_obj = transformation.rotate(transformed_obj)
        if transformation.color is not None:
            transformed_obj = transformation.change_color(transformed_obj)
        if transformation.cropping is not None:
            transformed_obj = transformation.crop(transformed_obj)
        if transformation.new_cells is not None:
            transformed_obj = transformation.fill_new_cells(transformed_obj)
        return transformed_obj

    def test_transform(
        self, training_data: list[ArcSet], transformation: ArcTransformation
    ):
        """
        Test the identified transformation on one of the training data input and check if the transformed input matches the output.
        """
        for entry in training_data[:1]:  # Just test on the first entry for now
            input_objects = self._identify_objects(entry.get_input_data().data())
            for input_object in input_objects:
                transformed_object = self.transform(input_object, transformation)
                if np.array_equal(transformed_object, entry.get_output_data().data()):
                    print("Transformation works on the training data.")
                    self.test_passed = True
                else:
                    print("Transformation does not work on the training data.")

    # -----------------------------------------------------------------------------
    # Make Predictions
    # -----------------------------------------------------------------------------

    def run_basic_transformations(
        self, training_data: list[ArcSet]
    ) -> ArcTransformation:
        """
        If the input cells are kept, run all simple transformations and check if they are consistent across the training data.
        """
        # First treat whole grid as ArcObject and check for transformations based on whole grid properties
        transformation_list = []
        for entry in training_data:
            input_object = ArcObject( )
            output_object = ArcObject(
                color=None,
                cell_pos=[
                    (i, j)
                    for i in range(entry.get_output_data().shape()[0])
                    for j in range(entry.get_output_data().shape()[1])
                ],
                general_props=None,
            )
            transformation = self.find_transformations_between_objects(
                [{"input_object": input_object, "output_object": output_object}]
            )
            if transformation:
                print("Found a basic transformation based on whole grid properties.")
                self.test_transform(training_data, transformation)
                transformation_list.append(transformation)

        # Check if all transformation in the list are identical
        if len(transformation_list) > 0 and all(
            transformation_list[0].__dict__ == transformation.__dict__
            for transformation in transformation_list
        ):
            return transformation_list[0]
        else:
            return None

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

        # If the input cells are kept, run all simple transformations
        if self.all_input_kept:
            transformation = self.run_basic_transformations(arc_problem.training_set())
            prediction = self.transform(input_test.data(), transformation)
            predictions.append(prediction)

        if self.test_passed:
            return predictions

        if len(self.similar_objects) > 0:
            # Determine transformation between similar objects
            transformation = self.find_transformations_between_objects(
                self.similar_objects
            )
            test_objects = self._identify_objects(input_test.data())

            if transformation:
                # Test transformation on training data
                self.test_transform(arc_problem.training_set(), transformation)
                # Apply transformation to test input
                prediction = self.transform(
                    input_test.data(), test_objects, transformation
                )
                predictions.append(prediction)
            else:
                # Assume non object-based transformation, could be grid-based
                transformation = self.check_non_object_transformation(
                    arc_problem.training_set(), self.similar_objects
                )
                if transformation:
                    # Test transformation on training data
                    self.test_transform(arc_problem.training_set(), transformation)
                    prediction = self.transform(
                        input_test.data(), test_objects, transformation
                    )
                    predictions.append(prediction)

        return predictions
