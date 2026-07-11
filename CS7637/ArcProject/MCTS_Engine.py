"""
MCTS Node

1. Selection - Starting at the parent node, select the child node with the UCB1 value until a leaf node is reached.
    UCB1 is the upper confidence bound for trees, which balances exploration and exploitation.
2. Expansion - If the leaf node is not a terminal state, expand the node by adding a new child node when choosing an unvisited action.
3. Evaluation - Evaluate the new child node by applying the sequence of transformations to the input state and calculating the reward based on the cost function.
4. Backpropagation - Using result from the evaluation, update the statistics of all nodes in the path from the new child node to the parent node.
"""

__date__ = "2026-06-28"
__author__ = "NedeeshaWeerasuriya"
__version__ = "0.1"


import numpy as np
import math
from typing import Callable, Optional, Tuple, List
from ArcSearch import ArcSearch, BoundTransformation
from ArcMemory import ArcState
from dataclasses import dataclass


@dataclass
class DefaultSearchParams:
    """
    The search engine will optimise the cost function when searching the hypothesis space.
    Minimise error (difference between predicted output and actual output)
    Minimise number of transformations (shortest path to the solution)
    """

    error_weight: float = 1.0
    complexity_weight: float = 0.1
    exploration_constant: float = np.sqrt(2)
    search_depth: int = 10
    widening_factor: int = 2
    temperature: float = 1.0


DEFAULT_PARAMS = DefaultSearchParams()


@dataclass
class MCTSResult:
    program: List[BoundTransformation]
    reward: float
    problem: ArcSearch

    def predict(self, input_array: np.ndarray) -> np.ndarray:
        """
        Apply the sequence of transformations to the input array and return the predicted output array.
        """
        return self.problem.predict(input_array, self.program)


class MCTSNode:

    def __init__(
        self,
        problem: ArcSearch,
        state: tuple[ArcState],
        action: Optional[Callable] = None,
        prior: Optional[float] = None,
        parent: Optional["MCTSNode"] = None,
        depth: int = 0,
        params: DefaultSearchParams = DEFAULT_PARAMS,
        seed: Optional[int] = 123,
    ):
        self.problem = problem
        self.state = state
        self.prior = prior
        self.parent = parent
        self.action = action
        self.depth = depth
        self.params = params
        self.random_state = np.random.RandomState(seed)

        self.children: list["MCTSNode"] = []
        self.visits = 0
        self.total_reward = 0.0
        self.best_reward = 0.0  # best reward in subtree

        self._reward = None
        self._unvisited_actions = None

        self.skip_log = []

    @property
    def n_visits(self) -> int:
        return self.visits

    @property
    def q_value(self) -> float:
        return self.total_reward / self.visits if self.visits > 0 else 0.0

    @property
    def is_fully_expanded(self) -> bool:
        return (
            len(self.unvisited_actions) == 0
            if self.unvisited_actions is not None
            else True
        )

    @property
    def can_expand(self) -> bool:
        if not self.unvisited_actions:
            return False
        # Widen the search if the node has been visited a lot
        # limit = self.params.widening_factor * ((self.visits + 1) ** 0.5)
        # return len(self.children) < limit
        return True

    @property
    def unvisited_actions(self):
        """
        (action, prior) pairs not yet expanded so pop() takes the highest prior transformation first
        Priors are normalised across this nodes children so the heuristic weight magnitude
        doesn't distort the search.
        """
        if self._unvisited_actions is None:
            candidates = self.problem.candidate_transformations()
            total_prior = sum(max(can.prior, 0) for can in candidates)
            if total_prior > 0:
                # get max prior first
                new_pair = [
                    (can, max(can.prior, 0) / total_prior) for can in candidates
                ]
            elif candidates:
                # No positive priors, so just normalise to uniform distribution
                new_pair = [(can, 1.0 / len(candidates)) for can in candidates]
            else:
                new_pair = []

            # Sort candidates by prior in descending order
            new_pair.sort(key=lambda x: x[1], reverse=True)
            self._unvisited_actions = new_pair
        return self._unvisited_actions

    def is_terminal(self) -> bool:
        return (
            self.depth >= self.params.search_depth
            or self.problem.is_solved(self.state)
            or (len(self.unvisited_actions) == 0 and not self.children)
        )

    def _same_state(self, a, b) -> bool:
        """
        Check if two states are the same by comparing their array representations.
        """
        return all(
            np.array_equal(
                self.problem.state_to_array(sa), self.problem.state_to_array(sb)
            )
            for sa, sb in zip(a, b)
        )

    def puct(self) -> float:
        """
        Calculate the PUCT (Predictor + Upper Confidence Bound for Trees) value for this node.
        PUCT calculation is
        """
        exploration = (
            self.params.exploration_constant
            * self.prior
            * math.sqrt(self.parent.visits)
            / (1 + self.visits)
        )
        return self.q_value + exploration

    def select_child(self) -> "MCTSNode":
        """
        Select the child node with the highest PUCT value.
        """
        return max(self.children, key=lambda child: child.puct())

    def best_child(self) -> "MCTSNode":
        """
        Return the child node with the highest number of visits.
        """
        return max(self.children, key=lambda child: child.visits)

    def program(self) -> list[BoundTransformation]:
        """
        Return the sequence of actions (program) from the root to this node.
        """
        actions = []
        node = self
        while node.parent is not None and node.action is not None:
            actions.append(node.action)
            node = node.parent
        return list(reversed(actions))

    def expand(self) -> Optional["MCTSNode"]:
        """
        Expand the node by creating a new child node for an unvisited action.
        """

        while self.unvisited_actions:
            # TODO: Use temperature to sample actions based on their prior probabilities instead of always taking the highest prior.
            action, prior = self.unvisited_actions.pop(0)
            try:
                new_state = self.problem.apply_transformation(action, self.state)
            except Exception as e:
                print(
                    f"Error applying transformation {action.__name__} at depth {self.depth}: {e}"
                )
                continue
            name = action.transformation.__name__
            if new_state is None:
                self.skip_log.append((name, "No new state generated"))
                continue
            if self._same_state(new_state, self.state):
                self.skip_log.append((name, "State unchanged"))
                continue
            self.skip_log.append((name, "Expanded"))
            child_node = MCTSNode(
                problem=self.problem,
                state=new_state,
                action=action,
                prior=prior,
                parent=self,
                depth=self.depth + 1,
                params=self.params,
            )
            self.children.append(child_node)
            return child_node
        return None

    def evaluate(self) -> float:
        """
        Evaluate the current state, checking its cost and returning a reward value.
        """
        if self._reward is None:
            cost = self.problem.evaluate_cost(
                self.state, self.depth, self.params.complexity_weight
            )
            self._reward = self.cost_to_reward(cost)
        return self._reward

    @staticmethod
    def cost_to_reward(cost: float) -> float:
        """
        Convert a cost value to a reward value using non-linear inverse.
        """
        return 1.0 / (1.0 + cost)

    def backpropagate(self, reward: float) -> None:
        """
        Backpropagate the reward value up the tree, updating the total reward and visit count for each node.
        """
        node = self
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node.best_reward = max(node.best_reward, reward)
            node = node.parent


class MCTSEngine:
    def __init__(
        self,
        root_node: "MCTSNode",
        iterations: int = 1000,
    ):
        self.root_node = root_node
        self.iterations = iterations
        self.solved_node = None

    def search(self):
        """
        Perform MCTS search for a specified number of iterations.
        """
        for _ in range(self.iterations):
            leaf_node = self.tree_policy(self.root_node)
            if leaf_node is None:
                continue
            reward = leaf_node.evaluate()
            leaf_node.backpropagate(reward)
            if reward == 1.0:
                self.solved_node = leaf_node
                break

    def tree_policy(self, node: MCTSNode) -> MCTSNode:
        """
        Selection and Expansion steps.
        """
        current_node = node
        while not current_node.is_terminal():
            if current_node.can_expand:
                child = current_node.expand()
                if child is not None:
                    return child
            if not current_node.children:
                return current_node
            current_node = current_node.select_child()
        return current_node

    def best_action(self) -> Tuple[List[BoundTransformation], MCTSNode]:
        """
        Return the best actions from the root node after MCTS search.
        Find the highest reward node with ties broken by number of visits.
        """
        if self.solved_node is not None:
            # print(
            #     f"Best program found with reward {self.solved_node.best_reward:.4f} and visits {self.solved_node.visits}"
            # )
            return self.solved_node.program(), self.solved_node

        # Find the child node with the highest reward, breaking ties by depth
        best_node = self.root_node
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            if (
                node.best_reward > best_node.best_reward
                or (
                    node.best_reward == best_node.best_reward
                    and node.depth <= best_node.depth
                )
                or best_node == self.root_node
            ):
                best_node = node
            stack.extend(node.children)

        # best_node.problem.candidate_report("fill_overlap_with_original_grid")
        # print(
        #     f"Best program found with reward {best_node.best_reward:.4f} and visits {best_node.visits}"
        # )
        # print("Input State:", best_node.state[0].grid_state.as_array)
        # print("Output State:", best_node.state[-1].grid_state.as_array)
        # print("Log for best node", best_node.skip_log)
        return best_node.program(), best_node
