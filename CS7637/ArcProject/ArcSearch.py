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
from typing import Callable, Optional

from ArcMemory import ArcState
from ArcSet import ArcSet
from ArcHeuristics import HeuristicSummary
from kwarg_engine import check_relevant_kwargs, iter_relevant_kwargs


def breadth_first_search(
    sorted_primitives: list[Callable],
    kwarg_pool: dict[str, list],
    training_data: list[ArcSet],
):
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

@dataclass
class DefaultSearchWeights:
    """
    The search engine will optimise the cost function when searching the hypothesis space.
    Minimise error (difference between predicted output and actual output)
    Minimise number of transformations (shortest path to the solution)
    """
    error_weight: float = 1.0
    complexity_weight: float = 0.1
    search_depth: int = 5

def error_calculation(predicted_output: np.ndarray, actual_output: np.ndarray) -> float:
    """
    Calculates the error between the predicted output and the actual output.
    Normalised between 0 and 1, where 0 is a perfect match and 1 is a complete mismatch.

    Penalise shape mismatch not just pixel mismatch
    TODO: Improve the error calculation to consider objects
    """
    if predicted_output is None:
        return 1.0
    # First check grid shape
    if predicted_output.shape != actual_output.shape:
        # find difference in shape
        pred, act = predicted_output.shape, actual_output.shape
        shape_diff = abs(pred[0] - act[0]) / max(pred[0], act[0], 1)
        return min(1.0, 0.5 + shape_diff)
    # pixel-wise comparison
    pixel_mismatch = int(np.count_nonzero(predicted_output != actual_output))
    total_pixels = predicted_output.size
    return pixel_mismatch / total_pixels if total_pixels > 0 else 1.0

def cost_calculation(
    error: float, depth: int, weights: DefaultSearchWeights
):
    """
    Calculates the cost of a hypothesis based on the error and depth.
    Lower cost is better.
    """
    return weights.error_weight * error + weights.complexity_weight * depth


@dataclass
class MCTSNode:
    """
    A node in the Monte Carlo Tree Search (MCTS) tree.
    """
    hypothesis: tuple[Callable, ...]
    state: tuple[ArcState, ...]
    prior_weight: float
    depth: int
    mean_error: float
    solved: bool
    parent: Optional["MCTSNode"] = None
    children: list["MCTSNode"] = None
    visits: int = 0

    def is_terminal(self) -> bool:
        """
        Check if the node is terminal (i.e., it has reached the maximum depth or solved the problem).
        """
        return self.solved

DEFAULT_WEIGHTS = DefaultSearchWeights()

class MCTSEngine:
    def __init__(
        self,
        sorted_primitives: list[Callable],
        kwarg_pool: dict[str, list],
        training_data: list[ArcSet],
        max_depth: int = 3,
        exploration_constant: float = 1.0,
        iterations: int = 1000,
        weights: DefaultSearchWeights = DEFAULT_WEIGHTS,
        seed: Optional[int] = 123,
    ):
        self.sorted_primitives = sorted_primitives
        self.kwarg_pool = kwarg_pool
        self.training_data = training_data
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.iterations = iterations
        self.weights = weights
        self.random_state = np.random.RandomState(seed)

    def search(self):
        # Implement the MCTS algorithm here
        pass
