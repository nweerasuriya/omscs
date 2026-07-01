"""
MCTS Node

1. Selection - Starting at the parent node, select the child node with the UCB1 value until a leaf node is reached.
    UCB1 is the upper confidence bound for trees, which balances exploration and exploitation.
2. Expansion - If the leaf node is not a terminal state, expand the node by adding a new child node when choosing an unvisited action.
3. Simulation - From the newly added child node, simulate a random playout until a terminal state is reached. Random moves are chosen for both players.
4. Backpropagation - Using result from the simulation, update the statistics of all nodes in the path from the new child node to the parent node.
"""

__date__ = "2026-06-28"
__author__ = "NedeeshaWeerasuriya"
__version__ = "0.1"


import numpy as np
import math
from Game import Game
from Connect4Game import is_winning_move, board_is_full, column_has_space, drop_piece

EXPLORATION_CONSTANT = math.sqrt(2)


class MCTSNode:

    def __init__(
        self,
        state: Game,
        board: np.ndarray = None,
        parent: "MCTSNode" = None,
        seq: int = 4,
        token: str = None,
        positional_scores: np.ndarray = None,
    ):
        self.state = state
        self.token = state.player1_token() if token is None else token
        self.board = state.get_board() if board is None else board
        self.parent = parent
        self.action = None
        self.children = []
        self.visits = 0
        self.result: float = 0.0
        self.unvisited_actions = None
        self.seq = seq  # Number of pieces to win (default is 4 for Connect 4 basic)

        # Positional scores for the board
        self.positional_scores = (
            positional_scores
            if positional_scores is not None
            else self._calculate_positional_scores(self.board)
        )

        # player tokens
        self.p1_token = state.player1_token()
        self.p2_token = state.player2_token()
        self.last_move = None

    @property
    def n_visits(self):
        return self.visits

    @property
    def is_fully_expanded(self):
        return len(self.unvisited) == 0

    @property
    def unvisited(self):
        if self.unvisited_actions is None:
            actions = self.get_possible_actions()
            actions.sort(key=lambda col: self._action_score(self.board, col))
            self.unvisited_actions = actions
        return self.unvisited_actions

    def _other_player_token(self, token: str) -> str:
        new_token = token if token is not None else self.token
        return self.p2_token if new_token == self.p1_token else self.p1_token

    def _calculate_positional_scores(self, board: np.ndarray):
        """
        Calculate positional scores for the board.
        Higher scores are given to more favourable positions (centre)
        """
        rows, cols = board.shape
        score = np.zeros((rows, cols))
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for row in range(rows):
            for col in range(cols):
                for dr, dc in directions:
                    end_r = row + dr * (self.seq - 1)
                    end_c = col + dc * (self.seq - 1)
                    if 0 <= end_r < rows and 0 <= end_c < cols:
                        for i in range(self.seq):
                            score[row + dr * i][col + dc * i] += 1
        return score

    def is_terminal(self):
        last_player = self._other_player_token(self.token)
        return is_winning_move(self.board, last_player, self.seq) or board_is_full(
            self.board
        )

    def get_possible_actions(self, board: np.ndarray = None):
        board = self.board if board is None else board
        return [col for col in range(board.shape[1]) if column_has_space(board, col)]

    def _get_row(self, board: np.ndarray, col: int) -> int:
        """
        Find the row a token will land
        """
        for row in range(board.shape[0] - 1, -1, -1):
            if not board[row][col]:
                return row
        return -1  # Column is full

    def _action_score(
        self, board: np.ndarray, col: int, player_token: str = None
    ) -> float:
        """
        Centre preference: Add a preference for the center column.
        Winning/blocking moves: Add a high score for winning moves or blocking opponent's winning moves.
        TODO: More heuristics can be added here later if needed
        """
        positional_score = self.positional_scores[self._get_row(board, col)][col]
        noise = np.random.uniform(0, 0.5)  # Small random noise to break ties
        return positional_score - noise

    def policy(self):
        if self.visits == 0:
            return float("inf")
        exploitation = -self.result / self.visits
        exploration = EXPLORATION_CONSTANT * math.sqrt(
            2 * math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

    def select_child(self) -> "MCTSNode":
        return max(self.children, key=lambda child: child.policy())

    def best_child(self) -> "MCTSNode":
        """
        Final move selection after MCTS iterations
        """
        return max(self.children, key=lambda child: child.visits)

    # EXPANSION to add a new child node
    def expand(self):
        action = self.unvisited.pop()
        next_board = self.board.copy()
        row = self._get_row(next_board, action)
        drop_piece(next_board, action, self.token)
        child_node = MCTSNode(
            self.state,
            next_board,
            parent=self,
            seq=self.seq,
            token=self._other_player_token(self.token),
            positional_scores=self.positional_scores,
        )
        child_node.action = action
        child_node.last_move = (row, action)
        self.children.append(child_node)
        return child_node

    # SIMULATION
    def rollout(self):
        current_board = self.board.copy()
        player_token = self.token

        last_player = self._other_player_token(self.token)
        if is_winning_move(current_board, last_player, self.seq):
            return -1
        if board_is_full(current_board):
            return 0

        while True:
            possible_actions = self.get_possible_actions(current_board)
            if not possible_actions:
                return 0

            action = np.random.choice(possible_actions)
            drop_piece(current_board, action, player_token)
            if is_winning_move(current_board, player_token, self.seq):
                return 1 if player_token == self.token else -1
            if board_is_full(current_board):
                return 0
            # Switch player token
            player_token = self._other_player_token(player_token)

    def backpropagate(self, result):
        self.visits += 1
        self.result += result
        if self.parent:
            self.parent.backpropagate(
                -result
            )  # Invert result for the parent node since it's the opponent's turn
