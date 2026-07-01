from copy import deepcopy
from enum import Enum
from Token import Token

import numpy as np


class Type(Enum):
    """
    An Enum of the types of Games this object can represent
    """
    TIC_TAC_TOE = 0
    CONNECT_4_BASIC = 1
    CONNECT_4_EXTENDED = 2
    CONNECT_4_MULTIPLAYER = 3
    CONNECT_4_HIDDEN_MULTIPLAYER = 4

    def __eq__(self, other):
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)


class Game:
    """
    A general container that represents a single game specified by the game type and final board
    """
    def __init__(self, game_type: Type, board: np.ndarray, seq_of_tokens_needed: int, number_of_players: int,
                 player1_token: Token, opp_token: Token, alt_opp_token: Token = None):
        """
        :param game_type: the type of game being played; see Type above for a listing
        :param board: the current board state
        :param seq_of_tokens_needed: the number of tokens needed sequentially to win a game
        :param number_of_players: the number of players for this game
        :param player1_token: a players token (not necessarily the agent playing first)
        :param opp_token: a players token (no necessarily the agent playing second)
        :param alt_opp_token: a players token when there are 3 possible agents playing a game
        """
        self._game_type = game_type
        self._board = board
        self._seq_of_tokens = seq_of_tokens_needed
        self._num_of_players = number_of_players
        self._player1_token = player1_token
        self._player2_token = opp_token
        self._player3_token = alt_opp_token

    def update_board(self, board: np.ndarray) -> None:
        """
        Updates this game object with a new board representing the current state of the game.
        :param board: the new game board state to use
        """
        self._board = board

    def get_type(self) -> Type:
        """
        Returns the type of game this represents.
        :return: an Enum copy of the game type
        """
        return deepcopy(self._game_type)

    def get_board(self) -> np.ndarray:
        """
        Returns a copy of the board status for this game
        :return: a numpy ndarray
        """
        return deepcopy(self._board)

    def get_row_count(self) -> int:
        """
        Returns the number of rows present in the current game board.
        :return: int
        """
        return self._board.shape[0]

    def get_column_count(self) -> int:
        """
        Returns the number of columns present in the current game board.
        :return: int
        """
        return self._board.shape[1]

    def number_of_seq_tokens_needed(self) -> int:
        """
        Returns the number of tokens needed in sequential order to win the game.
        :return: int
        """
        return deepcopy(self._seq_of_tokens)

    def number_of_players(self) -> int:
        """
        Returns the number of players involved in the current game.
        :return: an integer of the number of players
        """
        return deepcopy(self._num_of_players)

    def player1_token(self) -> Token:
        """
        Returns player1's game board token.
        :return: Token
        """
        return self._player1_token

    def player2_token(self) -> Token:
        """
        Returns player2's game board token.
        :return: Token
        """
        return self._player2_token

    def player3_token(self) -> Token:
        """
        Returns player3's game board token
        (or None if there are only 2 players).
        :return: Token or None
        """
        return self._player3_token

    def __eq__(self, other):
        """
        Returns True if and only if the Game types match and the board is exactly the same
        (contains the same pieces in the same ordering) as well as the number of players
        and the number of sequenced tokens; otherwise returns False
        :param other: the other game object to compare to
        :return: True if they are the same otherwise False
        """
        if (self.get_type() is other.get_type()
                and np.equal(self.get_board(), other.get_board())
                and self.number_of_players() == other.get_number_of_players()
                and self.number_of_seq_tokens_needed() == other.number_of_seq_tokens_needed()) \
                and self.player1_token() == other.player1_token() \
                and self.player2_token() == other.player2_token() \
                and self.player3_token() == other.player3_token():
            return True
        return False
