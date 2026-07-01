import numpy as np

from Game import Game
from Game import Type
from GameEndingStatus import Status
from typing import List


class GameStats:
    """
    The overall outcome for a game which includes the name of the game, the ending status, the player that went first,
    the final game board and the lists of players moves (1 for each active agent).
    """

    def __init__(self, name: str, status: Status, up_first: str, game_time: float,
                 game: Game, player_moves: List[tuple[int, str]], opp_moves: List[tuple[int, str]], alt_opp_moves=None):
        self._game_name: str = name
        self._status = status
        self._starter = up_first
        self._game_time = game_time
        self._game = game
        self._player_moves = player_moves
        self._opp_moves = opp_moves
        self._alt_opp_moves = alt_opp_moves

    def game_name(self) -> str:
        """
        Returns the name of the game that this stat represents.
        :return: string of the game name
        """
        return self._game_name

    def status(self) -> Status:
        """
        Returns the ending status of the game.
        :return: GameStatus of the  ending state of the game
        """
        return self._status

    def first_player(self) -> str:
        """
        Returns the name of who went first for this game.
        :return: string of the first players name
        """
        return self._starter

    def game_time(self) -> float:
        """
        Returns the time spent playing the game (will be 0.0 for Tic-Tac-Toe)
        :return: a float value representing the time taken to play the game in seconds
        """
        return self._game_time

    def game_type(self) -> Type:
        """
         Returns the type of game this stat represents
        :return: Game.Type
        """
        return self._game.get_type()

    def final_game_board(self) -> np.ndarray:
        """
        Returns the final game object which contains the game type and board
        :return: a game object
        """
        return self._game.get_board()

    def number_of_sequential_tokens(self) -> int:
        """
        Returns the number of consecutive tokens needed to win a game
        :return: an int for the number of tokens needed in sequential order
        """
        return self._game.number_of_seq_tokens_needed()

    def player_1_moves(self) -> List[tuple[int, str]]:
        """
        Returns the moves of the player
        :return: a list of tuple<int, int>(row, column) or a list of tuple<int, str>(col, token)
        """
        return self._player_moves

    def player_2_moves(self) -> List[tuple[int, str]]:
        """
        Returns the moves of the opponent player
        :return: a list of tuple<int, int>(row, column) or a list of tuple<int, str>(col, token)
        """
        return self._opp_moves

    def player_3_moves(self) -> List[tuple[int, str]]:
        """
        Returns the moves of the alternate opponent player (if there is more than just one otherwise returns None
        :return: a list of tuple<int, int>(row, column) or a list of tuple<int, str>(col, token) or None
        """
        return self._alt_opp_moves
