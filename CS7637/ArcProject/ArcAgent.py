import numpy as np
from ArcSet import ArcSet
from ArcProblem import ArcProblem
from ArcHeuristics import ArcHeuristics, HeuristicSummary
from ArcMemory import ArcState, GRID_PRIMITIVES, OBJECT_PRIMITIVES
from helpers import build_kwarg_pool, check_relevant_kwargs
from ArcDSL import *


class ArcAgent:
    def __init__(self):
        self.heuristics = ArcHeuristics()
        self.hypotheses = []

    def hypothesis_generation(
        self, heuristic_summary: HeuristicSummary
    ) -> list[callable]:
        """
        Based on heuristic summary, generate hypotheses transformation for testing
        For now uses grid primitivs and object primitives only
        TODO: Add pixel level primitives
        """
        all_primitives = GRID_PRIMITIVES + OBJECT_PRIMITIVES
        candidate_primitives = all_primitives.copy()
        candidate_primitives = [
            primitive
            for primitive in all_primitives
            if primitive.__name__ not in heuristic_summary.pruned_primitives
        ]
        return candidate_primitives

    def test_hypotheses(
        self,
        heuristic_summary: HeuristicSummary,
        hypotheses: list[callable],
        training_data: list[ArcSet],
    ):
        """
        Test the generated hypotheses on the training data and rank them based on performance.
        """
        ranked_hypotheses: dict[tuple[callable, ...], int] = {}
        kwarg_pool = build_kwarg_pool(self.heuristics.state_cache, heuristic_summary)
        for first_hypothesis in hypotheses:
            for train_set in training_data:
                input_grid = train_set.get_input_data()
                output_grid = train_set.get_output_data()
                # Check if kwargs are needed
                input_state = ArcState.from_array(
                    input_grid.data(), extract_objects=True
                )
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
                # Round 2: From result of first hypothesis, test all other hypotheses to see if result matches output
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
        print(f"Pruned Primitives: {heuristic_summary.pruned_primitives}")
        hypotheses = self.hypothesis_generation(heuristic_summary)
        print(f"Generated {len(hypotheses)} hypotheses.")
        ranked_hypotheses = self.test_hypotheses(
            heuristic_summary, hypotheses, arc_problem.training_set()
        )
        kwarg_pool = build_kwarg_pool(self.heuristics.state_cache, heuristic_summary)

        input_state = ArcState.from_array(input_test.data(), extract_objects=True)
        # Add top 3 hypotheses predictions to the predictions list
        for h, score in ranked_hypotheses[:3]:
            try:
                for fn in h:
                    kwargs = check_relevant_kwargs(fn, kwarg_pool)
                    input_state = fn(input_state, **kwargs)
                predictions.append(input_state.grid_state.as_array)
            except Exception as e:
                print(f"Error applying hypothesis {h}: {e}")
                continue
        return predictions
