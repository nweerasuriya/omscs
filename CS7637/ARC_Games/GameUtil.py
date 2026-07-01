import numpy as np
from Game import Type
from GameStats import GameStats


def print_c4_board(board: np.ndarray) -> str:
    """
    Prints the current state of the Connect4 board.
    :param board: a numpy ndarray representing the current game state
    :return: a string representation of the board suitable for printing to stdout
    """
    msg = str('\n\t')
    for row in board[:]:
        msg += "| "
        for cell in row:
            msg += '  | ' if cell == '' else cell + ' | '
        msg += '\n\t'

    return msg


def print_ttt_board(board: np.ndarray) -> str:
    """
    Prints the current state of a Tic-Tac-Toe board.
    :param board: a numpy ndarray representing the current game state
    :return: a string representation of the board suitable for printing to stdout
    """
    msg = str('\n\t')
    for row in board:
        for col in row:
            msg += '  ' if col == '' else col + ' '    # '\u2509' vert line : 'u2505' horz line
        msg += '\n\t'

    return msg


def print_game_stat(stat: GameStats) -> str:
    board = stat.final_game_board()
    if stat.game_type() == Type.TIC_TAC_TOE:
        game = f'\n\t{print_ttt_board(board)}'
    else:
        game = f'\n\t{print_c4_board(board)}'

    msg = f'\n{stat.game_name()}'
    msg += f'\n{stat.status()}'
    msg += f'\n{stat.first_player()} played first'
    msg += f'\nBoard size: {board.shape} | Token Seq Size: {stat.number_of_sequential_tokens()}'
    msg += f'\n  Player 1 Moves: {stat.player_1_moves()}'
    msg += f'\n  Player 2 Moves: {stat.player_2_moves()}'
    if stat.player_3_moves() is not None:
        msg += f'\n  Player 3 Moves: {stat.player_3_moves()}'
    msg += f'\nGame Time: {stat.game_time()}'
    msg += f'{game}'
    return msg
