from Token import Token
from TicTacToeGame import TicTacToeGame
from Connect4Game import Connect4Game
from Game import Type
from GameAgent import GameAgent
from RandomAgent import Random_Agent
import GameUtil
from GameStats import GameStats

if __name__ == '__main__':
    play_game = True
    playing_random = True
    opponent = 'RandomAgent'
    game_stat: GameStats|None  = None
    while play_game:
        print(' 1: Tic Tac Toe\n'
              ' 2: Connect 4 Basic\n'
              ' 3: Connect 4 Extended\n'
              ' 4: Connect 4 Multiplayer\n'
              ' 5: Connect 4 Hidden Multiplayer\n'
              ' 6: Swap Opponent Agent (Random or GameAgent)\n'
              ' 0: Exit')
        try:
            game_number = int(input('Select Game To Play: (1-5): '))
        except ValueError:
            print('\n\nPlease enter a number from 0-6\n\n')
            continue

        if game_number == 0:
            play_game = False
            continue

        match game_number:
            case 1:
                tic_tac_toe = TicTacToeGame()
                player1 = GameAgent(Token('X'))
                token = Token('O')
                player2 = Random_Agent(token) if playing_random else GameAgent(token)
                print(f'\nPlaying Tic Tac Toe GameAgent vs {opponent}')
                game_stat = tic_tac_toe.play_game(player1, player2)
            case 2:
                connect4 = Connect4Game(Type.CONNECT_4_BASIC)
                player1 = GameAgent(Token('Y'))
                token = Token('R')
                player2 = Random_Agent(token) if playing_random else GameAgent(token)
                game_type = 'Connect 4 Basic'
                print(f'\nPlaying {game_type} GameAgent vs {opponent}')
                game_stat = connect4.play_game(game_type, player1, player2)
            case 3:
                connect4 = Connect4Game(Type.CONNECT_4_EXTENDED)
                player1 = GameAgent(Token('Y'))
                p2_token = Token('R')
                player2 = Random_Agent(p2_token) if playing_random else GameAgent(p2_token)
                game_type = 'Connect 4 Extended'
                print(f'\nPlaying {game_type} GameAgent vs {opponent}')
                game_stat = connect4.play_game(game_type, player1, player2)
            case 4:
                connect4 = Connect4Game(Type.CONNECT_4_MULTIPLAYER)
                player1 = GameAgent(Token('Y'))
                p2_token = Token('R')
                p3_token = Token('W')
                player2 = Random_Agent(p2_token) if playing_random else GameAgent(p2_token)
                player3 = Random_Agent(p3_token) if playing_random else GameAgent(p3_token)
                game_type = 'Connect 4 Multiplayer'
                print(f'\nPlaying {game_type} GameAgent vs {opponent} vs {opponent}')
                game_stat = connect4.play_game(game_type, player1, player2, player3)
            case 5:
                connect4 = Connect4Game(Type.CONNECT_4_HIDDEN_MULTIPLAYER)
                player1 = GameAgent(Token('Y'))
                p2_token = Token('R')
                p3_token = Token('W')
                player2 = Random_Agent(p2_token) if playing_random else GameAgent(p2_token)
                player3 = Random_Agent(p3_token) if playing_random else GameAgent(p3_token)
                game_type = 'Connect 4 Hidden Multiplayer'
                print(f'\nPlaying {game_type} GameAgent vs {opponent} vs {opponent}')
                game_stat = connect4.play_game(game_type, player1, player2, player3)
            case 6:
                game_stat = None
                old_opp = opponent
                playing_random = not playing_random
                opponent = 'RandomAgent' if playing_random else 'GameAgent'
                print(f'\n\nSwapping {old_opp} for {opponent}\n\n')
            case _:
                game_stat = None
                print("\nInvalid Entry. 1 to 6 or 0 to Exit\n")

        if game_stat is not None:
            print(f'{GameUtil.print_game_stat(game_stat)}')
