import numpy as np
from typing import Callable
from ArcSet import ArcSet
from ArcProblem import ArcProblem
from ArcHeuristics import HeuristicEngine, HeuristicSummary
from ArcPruning import PruningEngine
from ArcSearch import breadth_first_search
from ArcMemory import (
    ArcState,
    GRID_PRIMITIVES,
    OBJECT_PRIMITIVES,
    SPLIT_GRID_PRIMITIVES,
)
from kwarg_engine import build_kwarg_pool, check_relevant_kwargs
from ArcDSL import *


class ArcAgent:
    def __init__(self):
        self.heuristics = HeuristicEngine()
        self.pruning_engine = PruningEngine()
        self.hypotheses = []

    def test_hypotheses(
        self,
        heuristic_summary: HeuristicSummary,
        weighted_primitives: dict[Callable, float],
        training_data: list[ArcSet],
    ):
        """
        Test the generated hypotheses on the training data and rank them based on performance.
        """
        sorted_primitives = sorted(
            weighted_primitives.items(), key=lambda x: x[1], reverse=True
        )
        kwarg_pool = build_kwarg_pool(self.heuristics.state_cache, heuristic_summary)
        ranked_hypotheses = breadth_first_search(
            sorted_primitives, kwarg_pool, training_data
        )

        ranked_final = sorted(
            ranked_hypotheses.items(), key=lambda x: x[1], reverse=True
        )
        return ranked_final

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

        heuristic_summary = self.heuristics.run_analysis(arc_problem.training_set())
        pruned_primitives, weighted_primitives = (
            self.pruning_engine.generate_candidate_primitives(
                heuristic_summary,
                GRID_PRIMITIVES + OBJECT_PRIMITIVES + SPLIT_GRID_PRIMITIVES,
            )
        )
        ranked_hypotheses = self.test_hypotheses(
            heuristic_summary, weighted_primitives, arc_problem.training_set()
        )

        kwarg_pool = build_kwarg_pool(self.heuristics.state_cache, heuristic_summary)

        input_state = ArcState.from_array(input_test.data(), extract_objects=True)
        # Add top 3 hypotheses predictions to the predictions list
        for h, score in ranked_hypotheses[:3]:
            input_state = ArcState.from_array(input_test.data(), extract_objects=True)
            try:
                for fn in h:
                    kwargs = check_relevant_kwargs(fn, kwarg_pool)
                    input_state = fn(input_state, **kwargs)
                predictions.append(input_state.grid_state.as_array)
            except Exception as e:
                print(f"Error applying hypothesis {h}: {e}")
                continue
        return predictions
