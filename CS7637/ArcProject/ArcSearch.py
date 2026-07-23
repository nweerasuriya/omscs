"""
Search Engine

Initial Implementation: Bredth first search over all primtives in a 3 stage loop
MCTS: Monte Carlo Tree Search implementation of the search engine
"""

__date__ = "2026-06-24"
__author__ = "NedeeshaWeerasuriya"
__version__ = "0.1"

from dataclasses import dataclass
import numpy as np
from typing import Any, Callable, Iterable, Optional, Tuple

from ArcMemory import ArcState
from ArcSet import ArcSet
from ArcHeuristics import HeuristicSummary
from kwarg_engine import check_relevant_kwargs, iter_relevant_kwargs, SetSpecificKwarg


class BoundTransformation:
    """
    Class that holds a transformation function and its associated keyword arguments.
    """

    def __init__(self, transformation: Callable, kwargs: dict, prior: float):
        self.transformation = transformation
        self.kwargs = kwargs
        self.prior = self.boost_prior_kwargs(kwargs, prior)

    def _resolve_references(self, kwargs: dict, original_state: ArcState) -> dict:
        """
        Resolve any references in the kwargs to the original state.
        """
        if not any(isinstance(v, (SetSpecificKwarg)) for v in kwargs.values()):
            return kwargs
        return {
            k: (v.resolve(original_state) if isinstance(v, SetSpecificKwarg) else v)
            for k, v in kwargs.items()
        }

    def apply(self, state: ArcState, original_state: ArcState = None) -> ArcState:
        """
        Apply the transformation to the given state.
        """
        kwargs = self.kwargs.copy()
        if original_state is not None:
            kwargs = self._resolve_references(kwargs, original_state)
        return self.transformation(state, **kwargs)

    def boost_prior_kwargs(self, kwargs: dict, prior: float) -> float:
        """
        If kwargs is a property association,
        boost the prior of the transformation based on the support score of the property association.
        """
        boost = 1
        for _, value in kwargs.items():
            support_score = getattr(value, "support_score", None)
            if support_score is not None:
                # scale by log of support score to avoid over boosting
                boost *= 1 + np.log(support_score + 1)
        return prior * boost


class ArcSearch:
    """
    Search context which wraps the inputs to the search engine.
    """

    def __init__(
        self,
        sorted_primitives: list[Tuple[Callable, float]],
        kwarg_pool: dict[str, list],
        training_data: list[ArcSet],
        iter_relevant_kwargs: Callable[[Callable, dict], Iterable[dict]],
        state_from_array: Callable[[np.ndarray], ArcState],
        state_to_array: Callable[[ArcState], np.ndarray],
    ):
        self.transformations = sorted_primitives
        self.kwarg_pool = kwarg_pool
        self.training_data = training_data
        self.iter_relevant_kwargs = iter_relevant_kwargs
        self.state_from_array = state_from_array
        self.state_to_array = state_to_array

        input_states, target_states = [], []
        for train_set in training_data:
            input_array = train_set.get_input_data().data()
            input_states.append(self.state_from_array(input_array))
            target_array = train_set.get_output_data().data()
            target_states.append(self.state_from_array(target_array))

        self._input_state: Tuple[np.ndarray, ...] = tuple(input_states)
        self._target_state: Tuple[np.ndarray, ...] = tuple(target_states)
        self.baseline_error = self._calculate_baseline_error()

    def get_input_state(self) -> Tuple[np.ndarray, ...]:
        return self._input_state

    def get_target_state(self) -> Tuple[np.ndarray, ...]:
        return self._target_state

    def _iter_kwargs(self, transformation: Callable) -> Iterable[dict]:
        return self.iter_relevant_kwargs(transformation, self.kwarg_pool)

    def is_solved(self, states: list[ArcState]) -> bool:
        """
        Check if the current states match the target states.
        """
        unique_states = []
        unique_targets = []
        for state in states:
            if state not in unique_states:
                unique_states.append(state)
        for target in self._target_state:
            if target not in unique_targets:
                unique_targets.append(target)
        for state, target in zip(states, self._target_state):
            if not np.array_equal(
                self.state_to_array(state), self.state_to_array(target)
            ):
                return False
        return True

    def candidate_transformations(self) -> list[BoundTransformation]:
        """
        Generate a list of candidate transformations for the given state.
        Each transformation is paired with its relevant keyword arguments and prior.
        Split prior over all relevant kwargs for the transformation.
        """
        candidates = []
        for transformation, prior in self.transformations:
            combinations = list(self._iter_kwargs(transformation))
            # Share the prior over all relevant kwargs for the transformation
            share_prior = prior / (1 + np.log(len(combinations) + 1))
            for kwargs in combinations:
                candidates.append(
                    BoundTransformation(transformation, kwargs, share_prior)
                )
        return candidates

    def apply_transformation(
        self, action: BoundTransformation, states: tuple[ArcState]
    ) -> Tuple:
        new_states = []
        for i, arc_state in enumerate(states):
            original_state = self._input_state[i]
            try:
                if action.transformation.__name__ == "fill_overlap_with_original_grid":
                    result_state = action.apply(arc_state, original_state)
                new_states.append(action.apply(arc_state, original_state))
            except Exception as e:
                print(
                    f"Error applying transformation {action.transformation.__name__}: {e}"
                )
                return None
        return tuple(new_states)

    def predict(
        self, input_array: np.ndarray, program_list: list[list[BoundTransformation]]
    ) -> list[np.ndarray]:
        """
        Apply a sequence of transformations to the input array and return the predicted output array.
        """
        original_state = self.state_from_array(input_array)
        state = original_state
        predictions = []
        for program in program_list:
            for action in program:
                try:
                    state = action.apply(state, original_state)
                except Exception as e:
                    print(f"Error in predict for {action.transformation.__name__}: {e}")
                    return None
            predictions.append(self.state_to_array(state))
        return predictions

    def _calculate_baseline_error(self) -> float:
        """
        Calculate the baseline error of the input states against the target states.
        """
        baseline_error = []
        for input_state, target_state in zip(self._input_state, self._target_state):
            baseline_error.append(
                self.grid_diff_error(
                    self.state_to_array(input_state), self.state_to_array(target_state)
                )
                # + 0.5 * self.object_diff_error(input_state, target_state)
            )
        return sum(baseline_error) / max(len(baseline_error), 1)

    @staticmethod
    def grid_diff_error(predicted: np.ndarray, target: np.ndarray) -> float:
        """
        Calculates the error between the predicted output and the actual output.
        Normalised between 0 and 1, where 0 is a perfect match and 1 is a complete mismatch.

        1. Account for diferences in grid shape by calculating the absolute difference in shape normalised by the maximum shape.
        2. Account for pixel differences by calculating the number of differing pixels normalised by the maximum number of pixels.

        TODO: Improve the error calculation to consider objects
        """
        if predicted is None:
            return 1.0

        if predicted.shape == target.shape and np.array_equal(predicted, target):
            return 0.0

        # 1. First check grid shape
        pred_h, pred_w = predicted.shape
        target_h, target_w = target.shape
        min_h, min_w = min(pred_h, target_h), min(pred_w, target_w)
        max_h, max_w = max(pred_h, target_h), max(pred_w, target_w)

        # Absolute difference in shape normalised by the maximum shape
        shape_penalty = (
            abs(pred_h - target_h) / max_h + abs(pred_w - target_w) / max_w
        ) / 2

        # 2. Check pixel difference
        # Calculate on the minimum shape to avoid index errors
        pixel_diff = np.count_nonzero(
            predicted[:min_h, :min_w] != target[:min_h, :min_w]
        )
        max_size = max(predicted.size, target.size)
        extra_pixels = max_size - min_h * min_w
        norm_pixel_diff = (pixel_diff + extra_pixels) / max_size

        if predicted.shape == target.shape:
            return norm_pixel_diff

        return 0.5 * shape_penalty + 0.5 * norm_pixel_diff

    @staticmethod
    def object_diff_error(predicted: ArcState, target: ArcState) -> float:
        """
        Calculates error in the object differences between the predicted output and the actual output.
        Check for number, colour and size differences between the objects in the predicted and target states.
        Do pairwise comparison by checking if target object is in the predicted objects
        Return a normalised error between 0 and 1, where 0 is a perfect match and 1 is a complete mismatch.
        """
        pass

    def evaluate_cost(
        self, states: list[ArcState], depth: int = 0, complexity_penalty: float = 0.0
    ) -> float:
        """
        Evaluate the cost of the current state based on the difference between predicted output and actual output.
        Account for complexity of the transformation sequence by adding a penalty for depth.
        """
        # Check if length of states is the same as length of target states
        if len(states) != len(self._target_state):
            print(
                f"Warning: Length of states ({len(states)}) does not match length of target states ({len(self._target_state)})."
            )
            return 1.0  # Return maximum error if lengths do not match

        error = 0.0
        for state, target in zip(states, self._target_state):
            grid_error = self.grid_diff_error(
                self.state_to_array(state), self.state_to_array(target)
            )
            # obj_error = self.object_diff_error(state, target)
            error += grid_error

        error = error / len(states)  # Average error over all states

        # Check if input state is the same as target state
        if self.baseline_error < 1e-6:
            norm_error = error
        else:
            norm_error = error / self.baseline_error

        if depth > 5:
            complexity_penalty = complexity_penalty * depth // 2
        else:
            complexity_penalty = 0.0
        return norm_error + complexity_penalty


def breadth_first_search(
    sorted_primitives: list[Tuple[Callable, float]],
    kwarg_pool: dict[str, list],
    training_data: list[ArcSet],
) -> dict[tuple[Callable, ...], int]:
    ranked_hypotheses: dict[tuple[Callable, ...], int] = {}
    hypotheses = [primitive for primitive, weight in sorted_primitives]

    for first_hypothesis in hypotheses:
        for train_set in training_data:
            input_grid = train_set.get_input_data()
            output_grid = train_set.get_output_data()

            input_state = ArcState.from_array(input_grid.data(), extract_objects=True)

            # Round 1: First set of hypotheses
            for first_kwargs in iter_relevant_kwargs(first_hypothesis, kwarg_pool):
                try:
                    result_state = first_hypothesis(input_state, **first_kwargs)
                    predicted_output = result_state.grid_state.as_array
                except Exception as e:
                    print(f"Error testing hypothesis {first_hypothesis.__name__}: {e}")
                    continue

                if np.array_equal(predicted_output, output_grid.data()):
                    h = (first_hypothesis,)
                    ranked_hypotheses[h] = ranked_hypotheses.get(h, 0) + 1
                    continue

                # Round 2: Second set of hypotheses
                for second_hypothesis in hypotheses:
                    if second_hypothesis == first_hypothesis:
                        continue

                    for second_kwargs in iter_relevant_kwargs(
                        second_hypothesis, kwarg_pool
                    ):
                        # Count number of times 2 appears in the result state for debugging purposes
                        count_2 = (result_state.grid_state.as_array == 2).sum()
                        try:
                            second_state = second_hypothesis(
                                result_state, **second_kwargs
                            )
                            second_output = second_state.grid_state.as_array
                            if (
                                "connect" in first_hypothesis.__name__
                                and count_2 > 2
                                and second_kwargs["out_colour"] == 3
                            ):
                                print(
                                    "Testing:",
                                    second_hypothesis.__name__,
                                    "Kwargs:",
                                    second_kwargs,
                                    "count_2:",
                                    count_2,
                                )
                                print(second_output)
                        except Exception as e:
                            continue

                        if np.array_equal(second_output, output_grid.data()):
                            seq = (first_hypothesis, second_hypothesis)
                            ranked_hypotheses[seq] = ranked_hypotheses.get(seq, 0) + 1

    return ranked_hypotheses
