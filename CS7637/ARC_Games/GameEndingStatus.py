from enum import Enum


class Status(Enum):
    """
    Represents the states a game could finish in.
    """
    PLAYER_1_WINS = 'Player 1 Wins'                    # Primary Agent
    PLAYER_1_INVALID_MOVE = 'Player 1 Invalid Move'

    PLAYER_2_WINS = 'Player 2 Wins'                    # Secondary Agent
    PLAYER_2_INVALID_MOVE = 'Opponent 1 Invalid Move'

    PLAYER_3_WINS = 'Player 3 Wins'                    # Alternate Agent for 3 player games
    PLAYER_3_INVALID_MOVE = 'Player 3 Invalid Move'

    TIE = 'Tie'                     # Board Filled with No Clear Winner
