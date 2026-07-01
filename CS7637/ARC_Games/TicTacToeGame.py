import time
from random import randint

import numpy as np

import GameUtil
from Game import Game, Type
from GameAgent import GameAgent
from GameEndingStatus import Status
from GameStats import GameStats
from Token import Token


def create_board() -> np.ndarray:
    """
    Creates a standard TicTacToe board of size 3 x 3
    :return: a numpy 3x3 array
    """
    return np.empty((3, 3), dtype=str)


def insert_piece(board: np.ndarray, row: int, column: int, token: Token) -> None:
    """
    Insert the provided piece into the given board specifically at row, column position
    :param board: the numpy 3x3 array
    :param row: the horizontal location to place the piece
    :param column: the vertical location to place the piece
    :param token: the players game token
    """
    board[row][column] = token.value()


def is_valid_position(board: np.ndarray, row: int, column: int):
    """
    Returns true if there is no player token at the location specified by the row and column positions.
    :param board: the board to check for a valid position
    :param row: the horizontal location to check
    :param column: the vertical location to check
    :return: true if and only if the row and column has no player token associated with it; false otherwise
    """
    try:
        return np.isin(str(''), board[row][column])
    except IndexError:
        return False


def board_is_full(board: np.ndarray):
    """
    Returns True if all the locations on the board are filled with a Token otherwise returns False
    :param board: ndarray
    :returns: bool
    """
    return np.count_nonzero(board) == 9


def print_board(board: np.ndarray):
    """
    Prints the board to standard out
    :param board: the playing surface to print
    """
    print(GameUtil.print_ttt_board(board))


def winning_move(board: np.ndarray, token: Token) -> bool:
    """
    Checks if the provided token has 3 consecutive positions either horizontally, vertically
    or diagonally.
    :param board: the game board to check
    :param token: the player to check
    :return: True if and only if there are 3 consecutive tokens; otherwise False
    """
    # check horizontal
    for r in range(3):
        if board[r][0] == token.value() and board[r][1] == token.value() and board[r][2] == token.value():
            return True

    # check vertical
    for c in range(3):
        if board[0][c] == token.value() and board[1][c] == token.value() and board[2][c] == token.value():
            return True

    # check positive slope diagonals
    if board[0][0] == token.value() and board[1][1] == token.value() and board[2][2] == token.value():
        return True

    # check negative slope diagonals
    if board[2][0] == token.value() and board[1][1] == token.value() and board[0][2] == token.value():
        return True

    return False


def make_random_move(board: np.ndarray):
    """
        Returns a random move from available positions from the provided board
        :param board: the board from which to evaluate the empty coordinates
        :return: the set of coordinates in row, column order
        """
    empty_coord = np.argwhere(board == '')
    idx = np.random.randint(len(empty_coord))
    return empty_coord[idx]


class TicTacToeGame:
    def play_game(self, player1: GameAgent, player2: GameAgent = None) -> GameStats:
        """
        The main game play for Tic Tac Toe.
        The Student Agent is always X.
        Opponent Agent is always O.

        First player is randomly selected.
        Invalid moves automatically end the game.

        returns game stats
        """
        players_token = player1.token()
        opp_token = Token('O') if player2 is None else player2.token()

        game = Game(Type.TIC_TAC_TOE, create_board(), 3, 2, players_token, opp_token)
        game_over = False
        turn = randint(0, 1)  # randomly selects who goes first

        first_player = f'Player({players_token.value()})' if turn == 0 else f'Opponent({opp_token.value()})'
        player1_moves = list()
        player2_moves = list()
        game_status = None

        start_time = time.time()
        while not game_over:
            board = game.get_board()  # get the game board for this round

            if not game_over and board_is_full(board):
                game_status = Status.TIE
                game_over = True
                continue

            if turn == 0:
                row, col = player1.make_move(game)

                if row == -1 and col == -1:
                    row, col = make_random_move(game.get_board())

                if is_valid_position(board, row, col):
                    insert_piece(board, row, col, players_token)
                    player1_moves.append((int(row), int(col)))

                    if winning_move(board, players_token):
                        game_status = Status.PLAYER_1_WINS
                        game_over = True
                else:
                    game_status = Status.PLAYER_1_INVALID_MOVE
                    game_over = True

            else:  # Player 2 input
                row, col = make_random_move(board) if player2 is None else player2.make_move(game)

                if row == -1 and col == -1:
                    row, col = make_random_move(game.get_board())

                if is_valid_position(board, row, col):
                    insert_piece(board, row, col, opp_token)
                    player2_moves.append((int(row), int(col)))

                    if winning_move(board, opp_token):
                        game_status = Status.PLAYER_2_WINS
                        game_over = True
                else:
                    # Player O made an invalid move: THIS SHOULD NEVER HAPPEN!
                    game_status = Status.PLAYER_2_INVALID_MOVE
                    game_over = True

            # print(board)
            game.update_board(board)  # update the game board for this round
            turn += 1
            turn %= 2

        end_time = time.time()
        game_time = end_time - start_time
        return GameStats('Tic-Tac-Toe', game_status, first_player, game_time, game, player1_moves, player2_moves)
