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

from ArcMemory import ArcState, PropertyKwarg
from ArcSet import ArcSet
from ArcHeuristics import HeuristicSummary
from kwarg_engine import check_relevant_kwargs, iter_relevant_kwargs


class BoundTransformation:
    """
    Class that holds a transformation function and its associated keyword arguments.
    """

    def __init__(self, transformation: Callable, kwargs: dict, prior: float):
        self.transformation = transformation
        self.kwargs = kwargs
        self.prior = self.boost_prior_kwargs(kwargs, prior)

    def apply(self, state: ArcState) -> ArcState:
        """
        Apply the transformation to the given state.
        """
        return self.transformation(state, **self.kwargs)

    def boost_prior_kwargs(self, kwargs: dict, prior: float) -> float:
        """
        If kwargs is a property association,
        boost the prior of the transformation based on the support score of the property association.
        """
        boost = 1
        for key, value in kwargs.items():
            if isinstance(value, PropertyKwarg):
                # scale by log of support score to avoid over boosting
                boost *= np.log(value.support_score + 1)
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
        self.diff_error = self.grid_diff_error

        input_states, target_states = [], []
        for train_set in training_data:
            input_array = train_set.get_input_data().data()
            input_states.append(self.state_from_array(input_array))
            target_array = train_set.get_output_data().data()
            target_states.append(self.state_from_array(target_array))

        self._input_state: Tuple[np.ndarray, ...] = tuple(input_states)
        self._target_state: Tuple[np.ndarray, ...] = tuple(target_states)

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
            share_prior = prior / len(combinations) if combinations else 0
            for kwargs in combinations:
                candidates.append(
                    BoundTransformation(transformation, kwargs, share_prior)
                )
        return candidates

    def apply_transformation(
        self, action: BoundTransformation, states: list[ArcState]
    ) -> Tuple:
        new_states = []
        for arc_state in states:
            try:
                new_states.append(action.apply(arc_state))
            except Exception as e:
                print(
                    f"Error applying transformation {action.transformation.__name__}: {e}"
                )
                return None
        return tuple(new_states)

    def predict(
        self, input_array: np.ndarray, program: list[BoundTransformation]
    ) -> np.ndarray:
        """
        Apply a sequence of transformations to the input array and return the predicted output array.
        """
        state = self.state_from_array(input_array)
        for action in program:
            try:
                state = action.apply(state)
            except Exception as e:
                print(f"Error in predict for {action.transformation.__name__}: {e}")
                return None
        return self.state_to_array(state)

    @staticmethod
    def grid_diff_error(predicted: np.ndarray, target: np.ndarray) -> float:
        """
        Calculates the error between the predicted output and the actual output.
        Normalised between 0 and 1, where 0 is a perfect match and 1 is a complete mismatch.

        Penalise shape mismatch not just pixel mismatch
        TODO: Improve the error calculation to consider objects
        """
        if predicted is None:
            return 1.0
        # First check grid shape
        if predicted.shape != target.shape:
            # find difference in shape
            pred, act = predicted.shape, target.shape
            shape_diff = abs(pred[0] - act[0]) / max(pred[0], act[0], 1)
            return min(1.0, 0.5 + shape_diff)
        # pixel-wise comparison
        pixel_mismatch = float(np.count_nonzero(predicted != target))
        total_pixels = predicted.size
        return pixel_mismatch / total_pixels if total_pixels > 0 else 1.0

    def evaluate_cost(
        self, states: list[ArcState], depth: int = 0, complexity_penalty: float = 0.1
    ) -> float:
        """
        Evaluate the cost of the current state based on the difference between predicted output and actual output.
        Account for complexity of the transformation sequence by adding a penalty for depth.
        """
        error = sum(
            self.diff_error(self.state_to_array(state), self.state_to_array(target))
            for state, target in zip(states, self._target_state)
        )
        complexity_penalty = complexity_penalty * depth
        return error + complexity_penalty


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
                        try:
                            second_state = second_hypothesis(
                                result_state, **second_kwargs
                            )
                            second_output = second_state.grid_state.as_array
                        except Exception as e:
                            continue

                        if np.array_equal(second_output, output_grid.data()):
                            seq = (first_hypothesis, second_hypothesis)
                            ranked_hypotheses[seq] = ranked_hypotheses.get(seq, 0) + 1

    return ranked_hypotheses
