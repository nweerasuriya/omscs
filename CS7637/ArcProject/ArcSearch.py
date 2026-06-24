"""
Search Engine

Initial Implementation: Bredth first search over all primtives in a 3 stage loop
MCTS: Monte Carlo Tree Search implementation of the search engine
"""

__date__ = "2026-06-24"
__author__ = "NedeeshaWeerasuriya"
__version__ = "0.1"


import numpy as np
from typing import Callable, Optional

from ArcMemory import ArcState
from ArcSet import ArcSet
from ArcHeuristics import HeuristicSummary
from kwarg_engine import check_relevant_kwargs


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
            # Check if kwargs are needed
            input_state = ArcState.from_array(input_grid.data(), extract_objects=True)
            relevant_kwargs = check_relevant_kwargs(first_hypothesis, kwarg_pool)
            # Round 1: Test the hypothesis with parameters
            try:
                result_state = first_hypothesis(input_state, **relevant_kwargs)
                predicted_output = result_state.grid_state.as_array
            except Exception as e:
                print(f"Error testing hypothesis {first_hypothesis.__name__}: {e}")
                continue
            # Check if the predicted output matches the expected output
            if np.array_equal(predicted_output, output_grid.data()):
                h = (first_hypothesis,)
                ranked_hypotheses[h] = ranked_hypotheses.get(h, 0) + 1
                continue

            for second_hypothesis in hypotheses:
                if second_hypothesis == first_hypothesis:
                    continue
                second_kwargs = check_relevant_kwargs(second_hypothesis, kwarg_pool)
                try:
                    second_state = second_hypothesis(result_state, **second_kwargs)
                    second_output = second_state.grid_state.as_array
                except Exception as e:
                    print(
                        f"Error testing second hypothesis {second_hypothesis.__name__} after {first_hypothesis.__name__}: {e}"
                    )
                    continue
                # Check
                if np.array_equal(second_output, output_grid.data()):
                    seq = (first_hypothesis, second_hypothesis)
                    ranked_hypotheses[seq] = ranked_hypotheses.get(seq, 0) + 1

                # for third_hypothesis in hypotheses:
                #     if third_hypothesis in (first_hypothesis, second_hypothesis):
                #         continue
                #     third_kwargs = check_relevant_kwargs(third_hypothesis, kwarg_pool)
                #     try:
                #         third_state = third_hypothesis(second_state, **third_kwargs)
                #         third_output = third_state.grid_state.as_array
                #     except Exception as e:
                #         print(
                #             f"Error testing third hypothesis {third_hypothesis.__name__} after {first_hypothesis.__name__} and {second_hypothesis.__name__}: {e}"
                #         )
                #         continue
                #     # Check
                #     if np.array_equal(third_output, output_grid.data()):
                #         seq = (first_hypothesis, second_hypothesis, third_hypothesis)
                #         ranked_hypotheses[seq] = ranked_hypotheses.get(seq, 0) + 1

    return ranked_hypotheses
