import Connect4Game
import TicTacToeGame
from Game import Game, Type
from Token import Token


class Random_Agent:
    def __init__(self, token: Token):
        self._token = token

    def token(self):
        return self._token

    def make_move(self, game: Game):
        if game.get_type() is Type.TIC_TAC_TOE:
            return TicTacToeGame.make_random_move(game.get_board())
        else:
            return Connect4Game.make_random_move(game.get_board())
