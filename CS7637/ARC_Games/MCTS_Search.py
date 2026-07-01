"""
MCTS Search
"""

__date__ = "2026-06-28"
__author__ = "NedeeshaWeerasuriya"
__version__ = "0.2"


import math
import numpy as np
from MCTS_Node import MCTSNode
from Connect4Game import drop_piece


class MCTSEngine:
    def __init__(self, root_node: MCTSNode, exploration_constant: float = math.sqrt(2)):
        self.root = root_node
        self.explore_c = exploration_constant

    def best_action(self, n_iterations: int) -> MCTSNode:
        """
        Identify the best action to take from the root node by performing MCTS for a specified number of iterations.
        Returns the child node corresponding to the best action.
        """
        for _ in range(n_iterations):
            try:
                leaf = self.tree_policy(self.root)
                result = leaf.rollout()
                leaf.backpropagate(result)
            except Exception as e:
                print(f"Error during MCTS iteration: {e}")
        chosen_node = self.root.best_child()

        # advance the root to the chosen child node for the next move
        self.update_root(chosen_node)
        return chosen_node

    def update_root(self, new_root: MCTSNode):
        """
        Update the root of the MCTS tree to the specified new root node.
        """
        self.root = new_root
        self.root.parent = None

    def tree_policy(self, node: MCTSNode) -> MCTSNode:
        """
        Explore the tree until a leaf node is reached.
        """
        current_node = node
        while not current_node.is_terminal():
            if not current_node.is_fully_expanded:
                return current_node.expand()
            else:
                current_node = current_node.select_child()
        return current_node
