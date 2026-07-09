import numpy as np
from typing import Callable, Optional, Tuple, List
from ArcSet import ArcSet
from ArcProblem import ArcProblem
from ArcHeuristics import HeuristicEngine, HeuristicSummary
from ArcPruning import PruningEngine
from MCTS_Engine import MCTSResult, MCTSNode, MCTSEngine
from ArcSearch import (
    ArcSearch,
    BoundTransformation,
    breadth_first_search,
)
from ArcMemory import ArcState
from MemoryDecorators import (
    RELATIONAL_PRIMITIVES,
    GRID_PRIMITIVES,
    OBJECT_PRIMITIVES,
    SPLIT_GRID_PRIMITIVES,
)
from kwarg_engine import (
    build_kwarg_pool,
    iter_relevant_kwargs,
    prune_primitives_required_kwargs,
)
from ArcDSL import *


def run_mcts_engine(
    sorted_primitives: list[Tuple[Callable, float]],
    kwarg_pool: dict[str, list],
    training_data: list[ArcSet],
) -> MCTSResult:
    """
    Run the MCTS search engine to find best transformations
    """
    problem = ArcSearch(
        sorted_primitives=sorted_primitives,
        kwarg_pool=kwarg_pool,
        training_data=training_data,
        iter_relevant_kwargs=iter_relevant_kwargs,
        state_from_array=ArcState.from_array,
        state_to_array=lambda state: state.grid_state.as_array,
    )
    root_node = MCTSNode(problem=problem, state=problem._input_state, depth=0)
    mcts_engine = MCTSEngine(root_node=root_node, iterations=1000)
    mcts_engine.search()
    if not root_node.children:
        return MCTSResult(program=[], reward=0)
    best_program, best_node = mcts_engine.best_action()
    ranked_hypotheses: dict[tuple[Callable, ...], int] = {}
    ranked_hypotheses[tuple(best_program)] = best_node.best_reward
    return MCTSResult(
        program=best_program, reward=best_node.best_reward, problem=problem
    )


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
    ) -> MCTSResult:
        """
        Test the generated hypotheses on the training data and rank them based on performance.
        Return a list of hypotheses, their associated kwargs, sorted by their performance score.
        """
        kwarg_pool = build_kwarg_pool(self.heuristics.state_cache, heuristic_summary)
        pruned_primitives = prune_primitives_required_kwargs(
            [primitive for primitive, _ in weighted_primitives.items()], kwarg_pool
        )
        # Keep pruned primitives from weighted_primitives
        weighted_primitives = {
            primitive: weight
            for primitive, weight in weighted_primitives.items()
            if primitive in pruned_primitives
        }
        sorted_primitives = sorted(
            weighted_primitives.items(), key=lambda x: x[1], reverse=True
        )
        # print(
        #     f"All Primitives: {[p.__name__ + ' weight: ' + str(weight) for p, weight in weighted_primitives.items()]}"
        # )
        # ranked_hypotheses = breadth_first_search(
        #     sorted_primitives, kwarg_pool, training_data
        # )
        # print("Ranked Hypotheses: " + str(ranked_hypotheses))
        mcts_result = run_mcts_engine(sorted_primitives, kwarg_pool, training_data)
        print(
            "Final Primitives: "
            + str([p.transformation.__name__ for p in mcts_result.program])
        )

        return mcts_result

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
                GRID_PRIMITIVES
                + OBJECT_PRIMITIVES
                + SPLIT_GRID_PRIMITIVES
                + RELATIONAL_PRIMITIVES,
            )
        )
        mcts_result = self.test_hypotheses(
            heuristic_summary, weighted_primitives, arc_problem.training_set()
        )

        prediction = mcts_result.predict(input_test.data())

        predictions.append(prediction)

        return predictions
