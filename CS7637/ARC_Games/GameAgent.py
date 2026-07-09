import random
from typing import Tuple
import numpy as np
import math as math
from Game import Game, Type
from Token import Token


class GameAgent:
    def __init__(self, token: Token):
        """
        Initial call to create your agent.
        This only gets called 1 time and then your agent will play multiple games.
        """
        self._token = token

    def token(self):
        return self._token

    def opponent_token(self):
        """
        Returns the opponent's token for this game based on user token
        """
        if self._token.value() == "X":
            return Token("O")
        else:
            return Token("X")

    def check_tic_tac_winning(self, game: Game, token: Token) -> tuple:
        """
        Either prevent opponent from winning or win the game depending on the token provided.
        Check for empty spaces and if the token has 2 in a row in the same row, column, or diagonal.
        Return the coordinates of the winning move if found, otherwise return (-1, -1)
        """
        for row in range(3):
            for col in range(3):
                # First check for empty space
                if game.get_board()[row][col] == "":
                    # Check if opponent has 2 in the same row or column or diagonal
                    # Check row
                    if (
                        game.get_board()[row][(col + 1) % 3] == token.value()
                        and game.get_board()[row][(col + 2) % 3] == token.value()
                    ):
                        return row, col
                    # Check column
                    if (
                        game.get_board()[(row + 1) % 3][col] == token.value()
                        and game.get_board()[(row + 2) % 3][col] == token.value()
                    ):
                        return row, col
                    # Check right diagonal
                    if row == col:
                        if (
                            game.get_board()[(row + 1) % 3][(col + 1) % 3]
                            == token.value()
                            and game.get_board()[(row + 2) % 3][(col + 2) % 3]
                            == token.value()
                        ):
                            return row, col
                    # Check left diagonal
                    if row + col == 2:
                        if (
                            game.get_board()[(row + 1) % 3][(col + 2) % 3]
                            == token.value()
                            and game.get_board()[(row + 2) % 3][(col + 1) % 3]
                            == token.value()
                        ):
                            return row, col
        return -1, -1

    def make_move(self, game: Game) -> Tuple[int, int] | int:
        """
        This is the main driver of the agent. The game controller will call this with an updated game object
        every time the agent is expected to make a move.

        The agent will return a Tuple(int, int) for Tic-tac-toe
        or just an int for all Connect Four games.

        Tic-tac-toe:
        Return a tuple in row, column form. (row, col)
        Returning (-1,-1) will make a random selection of available positions.
        """
        # Tic tac-toe
        if game.get_type() == Type.TIC_TAC_TOE:
            # First move is pick the center if available
            if game.get_board()[1][1] == "":
                return 1, 1
            # If only the center is taken, pick a corner randomly
            if (
                game.get_board()[0][0] == ""
                and game.get_board()[0][2] == ""
                and game.get_board()[2][0] == ""
                and game.get_board()[2][2] == ""
            ):
                corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
                move = corners[np.random.randint(0, 4)]
            # Check if the agent can win or prevent opponent from winning
            move = self.check_tic_tac_winning(game, self.opponent_token())
            move = (
                self.check_tic_tac_winning(game, self.token())
                if move == (-1, -1)
                else move
            )
            # Otherwise, pick a random available position
            return move

        # Connect 4
        if game.get_type() in [Type.CONNECT_4_BASIC, Type.CONNECT_4_EXTENDED]:
            # Determine number of pieces in a row needed to win based on game type
            seq = 4
            if game.get_type() == Type.CONNECT_4_EXTENDED:
                width, height = game.get_board().shape
                seq = int((width + height) // 3)
            player_token = game.player1_token
            int_token = 1 if player_token == "Y" else 2
            root_node = MCTSNode(game, seq=seq, token=int_token)
            mcts_engine = MCTSEngine(root_node)
            best_node = mcts_engine.best_action(n_iterations=300)
            return best_node.action


EXPLORATION_CONSTANT = math.sqrt(2)
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def convert_board_to_int(board: np.ndarray) -> np.ndarray:
    """
    Convert the board to a numpy array with 0 for empty, 1 for player 1, and 2 for player 2.
    This is to improve runtime performance as string comparisons are slower than integer comparisons.
    """
    int_board = np.zeros(board.shape, dtype=int)
    int_board[board == "Y"] = 1
    int_board[board == "R"] = 2
    return int_board


def drop_piece(board: np.ndarray, col: int, token: int) -> None:
    """
    Integer version of drop_piece to improve performance.
    """
    for row in reversed(range(board.shape[0])):
        if board[row][col] == 0:
            board[row][col] = token
            break


def board_is_full(board: np.ndarray) -> bool:
    """
    Check if the board is full.
    """
    return not np.any(board == 0)


def column_has_space(board: np.ndarray, col: int) -> bool:
    """
    Check if the column has space for a new piece.
    """
    return 0 in board[:, col]


class MCTSNode:

    def __init__(
        self,
        state: Game,
        board: np.ndarray = None,
        parent: "MCTSNode" = None,
        seq: int = 4,
        token: int = 1,
        positional_scores: np.ndarray = None,
    ):
        self.state = state
        self.token = token
        self.board = self.get_board_array() if board is None else board
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
        self.p1_token = 1 if state.player1_token().value == "Y" else 2
        self.p2_token = 1 if state.player2_token().value == "Y" else 2
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

    def get_int_token(self):
        """
        Get the integer representation of the current player's token.
        """
        return (
            self.p1_token if self.token == self.state.player1_token() else self.p2_token
        )

    def get_board_array(self):
        """
        Get the current board as a numpy array of integers for performance.
        """
        str_board = self.state.get_board()
        return convert_board_to_int(str_board)

    def check_winning_move(
        self, board: np.ndarray, token: int, last_move: tuple, seq: int
    ) -> bool:
        """
        Check if the last move made by the player with the given token resulted in a win.
        """
        r, c = last_move
        rows, cols = board.shape

        # Check in each direction for a winning sequence
        for dr, dc in DIRECTIONS:
            count = 1

            # Check in the positive direction
            new_r, new_c = r + dr, c + dc
            while (
                0 <= new_r < rows and 0 <= new_c < cols and board[new_r][new_c] == token
            ):
                count += 1
                new_r += dr
                new_c += dc
            # Check in the negative direction
            new_r, new_c = r - dr, c - dc
            while (
                0 <= new_r < rows and 0 <= new_c < cols and board[new_r][new_c] == token
            ):
                count += 1
                new_r -= dr
                new_c -= dc
            # Check if we have a winning sequence
            if count >= seq:
                return True
        return False

    def _other_player_token(self, token: int) -> int:
        new_token = token if token is not None else self.token
        return self.p2_token if new_token == self.p1_token else self.p1_token

    def _calculate_positional_scores(self, board: np.ndarray):
        """
        Calculate positional scores for the board.
        Higher scores are given to more favourable positions (centre)
        Due to timing issues as board size increases, limit scoring to a 5x5 board. Fill in the rest of the board with 0s.
        """
        # max_rows, max_cols = 6, 7
        # board_r, board_c = board.shape

        rows, cols = board.shape
        score = np.zeros((rows, cols), dtype=float)
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

    def is_terminal(self) -> bool:
        last_player = self._other_player_token(self.token)
        check = (
            self.last_move
            and self.check_winning_move(
                self.board, last_player, self.last_move, self.seq
            )
            or board_is_full(self.board)
        )
        return check

    def get_possible_actions(self, board: np.ndarray = None) -> list[int]:
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

    def _action_score(self, board: np.ndarray, col: int) -> float:
        """
        Centre preference: Add a preference for the center column.
        Winning/blocking moves: Add a high score for winning moves or blocking opponent's winning moves.
        TODO: More heuristics can be added here later if needed
        """
        positional_score = self.positional_scores[self._get_row(board, col)][col]
        # noise = random.random() * 0.5  # Small random noise to break ties
        return positional_score

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

        if self.last_move and self.check_winning_move(
            current_board, last_player, self.last_move, self.seq
        ):
            return -1

        valid_cols = [
            c for c in range(current_board.shape[1]) if current_board[0][c] == 0
        ]

        while valid_cols:

            action = random.choice(valid_cols)
            row = self._get_row(current_board, action)
            drop_piece(current_board, action, player_token)

            if self.check_winning_move(
                current_board, player_token, (row, action), self.seq
            ):
                return 1 if player_token == self.token else -1

            if row == 0:
                valid_cols.remove(action)

            # Switch player token
            player_token = self._other_player_token(player_token)

        return 0

    def backpropagate(self, result):
        self.visits += 1
        self.result += result
        if self.parent:
            self.parent.backpropagate(
                -result
            )  # Invert result for the parent node since it's the opponent's turn


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
