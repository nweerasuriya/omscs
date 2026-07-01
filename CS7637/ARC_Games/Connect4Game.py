import time

import numpy as np
import random

from Token import Token
from Game import Game, Type
from GameStats import GameStats
from GameEndingStatus import Status


def create_board(rows: int, columns: int) -> np.ndarray:
    """
    Returns a new Connect Four board of size rows x columns.

    :return : a numpy ndarray of string types representing the board in row, column order
    """
    return np.empty((rows, columns), dtype=str)


def column_has_space(board: np.ndarray, col: int) -> bool:
    """
    Returns True if and only if the column exists and has space for a new piece; otherwise False
    :param board: a numpy array representing the board
    :param col: a column index
    """
    try:
        return np.isin(str(''), board[:, col]).sum()
    except IndexError:
        return False


def drop_piece(board: np.ndarray, col: int, token: Token) -> None:
    """
    Drops a piece into the chosen column at the first available row position that is empty.
    :param board: a numpy array representing the board
    :param col: int
    :param token: a Token
    :returns None
    """
    for row in reversed(range(board.shape[0])):
        if np.isin(str(''), board[row][col]):
            board[row][col] = token.value()
            break


def is_winning_move(board: np.ndarray, token: Token, seq: int) -> bool:
    """
    Checks to see if the given board and token have the required number of sequential tokens needed to win the game
    :param board: the np.ndarry to check
    :param token: the Token to compare against
    :param seq: the required number of sequential tokens
    :return: True if the board contains the number of sequential tokens; otherwise returns False
    """
    rows = board.shape[0]
    columns = board.shape[1]
    col_end = columns - seq + 1
    row_end = rows - seq + 1

    # Check horizontally
    for row in board:
        for col in range(col_end):
            if row[col] == token.value():
                count = 1
                for idx in range(1, seq):
                    if row[col + idx] == token.value():
                        count += 1
                    else:
                        break
                if count == seq:
                    return True

    # Check vertically
    for col in range(columns):
        for row in range(row_end):
            if board[row, col] == token.value():
                count = 1
                for idx in range(1, seq):
                    if board[row+idx, col] == token.value():
                        count += 1
                    else:
                        break
                if count == seq:
                    return True

    # Check negative diagonals
    for col in range(col_end):
        for row in range(row_end):
            if board[row, col] == token.value():
                count = 1
                for idx in range(1, seq):
                    if board[row+idx, col+idx] == token.value():
                        count += 1
                    else:
                        break
                if count == seq:
                    return True

    # Check positive diagonals
    for col in range(seq - 1, columns):
        for row in range(row_end):
            if board[row, col] == token.value():
                count = 1
                for idx in range(1, seq):
                    if board[row+idx, col-idx] == token.value():
                        count += 1
                    else:
                        break
                if count == seq:
                    return True

    return False


def board_is_full(board) -> bool:
    """
    Returns True if and only if the board is completely full of tokens; otherwise returns False
    """
    for col in range(board.shape[1]):
        if column_has_space(board, col):
            return False
    return True


def make_random_move(board: np.ndarray) -> int:
    """
    Makes a random move given the current board state.
    :param board: the current board state
    :return: a randomly selected move from the available moves
    """
    available_moves = list([x for x in range(board.shape[1]) if column_has_space(board, x)])
    return random.choice(available_moves)


class Connect4Game:
    def __init__(self, play_type: Type):
        self._play_type = play_type

    def play_game(self, name, player1, player2, player3=None):

        no_of_players = 2 if player3 is None else 3
        hide_pieces = False
        match self._play_type:
            case Type.CONNECT_4_EXTENDED:
                row = random.randint(6, 12)
                col = random.randint(7, 14)
                seq = min(row, col) - random.randint(1, 2)
            case Type.CONNECT_4_MULTIPLAYER:
                row, col, seq = 9, 10, 4
            case Type.CONNECT_4_HIDDEN_MULTIPLAYER:
                row, col, seq = 9, 10, 4
                hide_pieces = True
            case _:
                row, col, seq = 6, 7, 4

        board = create_board(row, col)
        hidden_token = Token('H')
        player3_token = None if player3 is None else player3.token()
        game = Game(self._play_type, board, seq, no_of_players, player1.token(), player2.token(), player3_token)
        game_over = False

        turn = random.randint(0, no_of_players - 1)  # randomly selects who goes first
        first_player = f"Player({game.player1_token().value()})" if turn == 0 \
            else f"Opponent({game.player2_token().value()})" if turn == 1 \
            else f"Opponent({game.player3_token().value()})"

        player1_moves = list()
        player2_moves = list()
        player3_moves = list() if player3 else None
        game_status = None

        start_time = time.time()
        while not game_over:
            if board_is_full(board):
                game_over = True
                game_status = Status.TIE
                continue

            if turn == 0:  # player 1's turn
                board = game.get_board()
                temp_board = game.get_board()
                if hide_pieces:
                    temp_board[(temp_board == player2.token().value())
                               | (temp_board == player3.token().value())] = hidden_token.value()

                game.update_board(temp_board)  # replace gameboard before sending to the agent

                try:
                    col = player1.make_move(game)
                except Exception as ex:
                    game_status = Status.PLAYER_1_INVALID_MOVE
                    game_over = True
                    continue

                if col == -1:
                    col = make_random_move(board)

                if col is None:
                    game_status = Status.PLAYER_1_INVALID_MOVE
                    game_over = True
                    continue

                if not isinstance(col, int):
                    game_status = Status.PLAYER_1_INVALID_MOVE
                    game_over = True
                    continue

                if column_has_space(board, col):
                    try:
                        drop_piece(board, col, player1.token())
                        player1_moves.append((col, player1.token().value()))
                    except Exception as ex:
                        game_status = Status.PLAYER_1_INVALID_MOVE
                        game_over = True

                    if is_winning_move(board, player1.token(), seq):
                        game_status = Status.PLAYER_1_WINS
                        game_over = True
                else:
                    game_status = Status.PLAYER_1_INVALID_MOVE
                    game_over = True

            # player 2's turn
            elif turn == 1:
                board = game.get_board()
                temp_board = game.get_board()
                if hide_pieces:
                    temp_board[(temp_board == player1.token().value())
                               | (temp_board == player3.token().value())] = hidden_token.value()

                game.update_board(temp_board)

                try:
                    col = player2.make_move(game)
                except Exception as ex:
                    game_status = Status.PLAYER_2_INVALID_MOVE
                    game_over = True
                    continue

                if col == -1:
                    col = make_random_move(board)

                if col is None:
                    game_status = Status.PLAYER_2_INVALID_MOVE
                    game_over = True
                    continue

                if not isinstance(col, int):
                    game_status = Status.PLAYER_2_INVALID_MOVE
                    game_over = True
                    continue

                if column_has_space(board, col):
                    try:
                        drop_piece(board, col, player2.token())
                        player2_moves.append((col, player2.token().value()))
                    except Exception as ex:
                        game_status = Status.PLAYER_2_INVALID_MOVE
                        game_over = True

                    if is_winning_move(board, player2.token(), seq):
                        game_status = Status.PLAYER_2_WINS
                        game_over = True
                else:
                    game_status = Status.PLAYER_2_INVALID_MOVE
                    game_over = True

            else:  # player3 turn
                board = game.get_board()
                temp_board = game.get_board()
                if hide_pieces:
                    temp_board[(temp_board == player1.token().value())
                               | (temp_board == player2.token().value())] = hidden_token.value()

                game.update_board(temp_board)

                try:
                    col = player3.make_move(game)
                except Exception as ex:
                    game_status = Status.PLAYER_3_INVALID_MOVE
                    game_over = True
                    continue

                if col == -1:
                    col = make_random_move(board)

                game.update_board(board)

                if col is None:
                    game_status = Status.PLAYER_3_INVALID_MOVE
                    game_over = True
                    continue

                if not isinstance(col, int):
                    game_status = Status.PLAYER_3_INVALID_MOVE
                    game_over = True
                    continue

                if column_has_space(board, col):
                    try:
                        drop_piece(board, col, player3.token())
                        player3_moves.append((col, player3.token().value()))
                    except Exception as ex:
                        game_status = Status.PLAYER_3_INVALID_MOVE
                        game_over = True

                    if is_winning_move(board, player3_token, seq):
                        game_status = Status.PLAYER_3_WINS
                        game_over = True
                else:
                    game_status = Status.PLAYER_3_INVALID_MOVE
                    game_over = True

            if board_is_full(board):
                game_over = True
                game_status = Status.TIE

            game.update_board(board)
            turn += 1
            mod = 2 if player3 is None else 3
            turn %= mod

        end_time = time.time()
        game_time = end_time - start_time
        game_stats = GameStats(name, game_status, first_player, game_time, game,
                               player1_moves, player2_moves, player3_moves)
        return game_stats
