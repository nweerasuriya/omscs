from typing import Tuple
import numpy as np

from Game import Game, Type
from Token import Token
from MCTS_Node import MCTSNode
from MCTS_Search import MCTSEngine


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

            root_node = MCTSNode(game, seq=seq)
            mcts_engine = MCTSEngine(root_node)
            best_node = mcts_engine.best_action(n_iterations=100)
            return best_node.action
