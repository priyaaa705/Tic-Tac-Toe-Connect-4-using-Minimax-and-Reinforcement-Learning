from random import choice
from math import inf
import numpy as np
import json
import random
import matplotlib.pyplot as plt


XPLAYER = +1
OPLAYER = -1
EMPTY = 0

trained_q_table_path = "/Users/priya/Desktop/AI/Game/"

board = [[EMPTY, EMPTY, EMPTY],
         [EMPTY, EMPTY, EMPTY],
         [EMPTY, EMPTY, EMPTY]]


def printBoard(brd):
    chars = {XPLAYER: 'X', OPLAYER: 'O', EMPTY: ' '}
    for x in brd:
        for y in x:
            ch = chars[y]
            print(f'| {ch} |', end='')
        print('\n' + '---------------')
    print('===============')


def clearBoard(brd):
    for x, row in enumerate(brd):
        for y, col in enumerate(row):
            brd[x][y] = EMPTY


def winningPlayer(brd, player):
    winningStates = [[brd[0][0], brd[0][1], brd[0][2]],
                     [brd[1][0], brd[1][1], brd[1][2]],
                     [brd[2][0], brd[2][1], brd[2][2]],
                     [brd[0][0], brd[1][0], brd[2][0]],
                     [brd[0][1], brd[1][1], brd[2][1]],
                     [brd[0][2], brd[1][2], brd[2][2]],
                     [brd[0][0], brd[1][1], brd[2][2]],
                     [brd[0][2], brd[1][1], brd[2][0]]]

    if [player, player, player] in winningStates:
        return True

    return False


def gameWon(brd):
    return winningPlayer(brd, XPLAYER) or winningPlayer(brd, OPLAYER)


def printResult(brd):
    if winningPlayer(brd, XPLAYER):
        print('X has won! ' + '\n')

    elif winningPlayer(brd, OPLAYER):
        print('O\'s have won! ' + '\n')

    else:
        print('Draw' + '\n')


def emptyCells(brd):
    emptyC = []
    for x, row in enumerate(brd):
        for y, col in enumerate(row):
            if brd[x][y] == EMPTY:
                emptyC.append([x, y])
    return emptyC


def boardFull(brd):
    if len(emptyCells(brd)) == 0:
        return True
    return False


def setMove(brd, x, y, player):
    brd[x][y] = player


def getScore(brd):
    if winningPlayer(brd, XPLAYER):
        return 10
    elif winningPlayer(brd, OPLAYER):
        return -10
    else:
        return 0


def playerMove(brd):
    e = True
    moves = {1: [0, 0], 2: [0, 1], 3: [0, 2],
             4: [1, 0], 5: [1, 1], 6: [1, 2],
             7: [2, 0], 8: [2, 1], 9: [2, 2]}
    while e:
        try:
            move = int(input('Pick a position(1-9)'))
            if move < 1 or move > 9:
                print('Invalid location! ')
            elif not (moves[move] in emptyCells(brd)):
                print('Location filled')
            else:
                setMove(brd, moves[move][0], moves[move][1], XPLAYER)
                printBoard(brd)
                e = False
        except(KeyError, ValueError):
            print('Please pick a number!')


def default_opponent(board):
    # Try to find a winning move for the opponent and block it
    for x in range(3):
        for y in range(3):
            if board[x][y] == EMPTY:
                # Check if placing O here wins the game for O
                board[x][y] = OPLAYER
                if winningPlayer(board, OPLAYER):
                    board[x][y] = EMPTY  # Undo the move
                    return (x, y)  # Block O from winning
                board[x][y] = EMPTY
                
    # If no immediate threat, pick a random move
    return random.choice(emptyCells(board))


def MiniMax(brd, depth, player):
    if gameWon(brd) or depth == 0:
        return [-1, -1, getScore(brd)]
    if player == XPLAYER:
        best = [-1, -1, -inf]
    else:
        best = [-1, -1, inf]
    for cell in emptyCells(brd):
        x, y = cell
        brd[x][y] = player
        score = MiniMax(brd, depth - 1, -player)
        brd[x][y] = EMPTY
        score[0], score[1] = x, y
        if player == XPLAYER:
            if score[2] > best[2]:
                best = score  # Maximize the XPLAYER
        else:
            if score[2] < best[2]:
                best = score  # Minimize the OPLAYER
    return best


def MiniMaxAB(brd, depth, alpha, beta, player):
    row = -1
    col = -1
    if depth == 0 or gameWon(brd):
        return [row, col, getScore(brd)]
    else:
        for cell in emptyCells(brd):
            setMove(brd, cell[0], cell[1], player)
            score = MiniMaxAB(brd, depth - 1, alpha, beta, -player)
            if player == XPLAYER:
                # X is always the max player
                if score[2] > alpha:
                    alpha = score[2]
                    row = cell[0]
                    col = cell[1]
            else:
                if score[2] < beta:
                    beta = score[2]
                    row = cell[0]
                    col = cell[1]
            setMove(brd, cell[0], cell[1], EMPTY)
            if alpha >= beta:
                break
        if player == XPLAYER:
            return [row, col, alpha]
        else:
            return [row, col, beta]
        

def AIMove(brd, use_alpha_beta=True):
    if len(emptyCells(brd)) == 9:
        x, y = choice([0, 1, 2]), choice([0, 1, 2])
    else:
        if use_alpha_beta:
            x, y, _ = MiniMaxAB(brd, len(emptyCells(brd)), -inf, inf, OPLAYER)
        else:
            x, y, _ = MiniMax(brd, len(emptyCells(brd)), OPLAYER)
    if x != -1 and y != -1:  # Ensure the move is valid
        setMove(brd, x, y, OPLAYER)
        printBoard(brd)
        return (x, y)  # Return the move
    return None  # In case no valid move is found
        

# Enhance default opponent logic
def enhancedAIMove(brd):
    # Prioritize winning moves, then blocking moves, then default to MiniMaxAB or MiniMax
    move = findWinningMove(brd, OPLAYER) or findBlockingMove(brd, XPLAYER) or MiniMaxAB(brd, len(emptyCells(brd)), -inf, inf, OPLAYER)[:2]
    if move:
        setMove(brd, move[0], move[1], OPLAYER)
        printBoard(brd)
    else:
        AIMove(brd)  # Fallback to the existing AIMove if no strategic move is found


def findWinningMove(brd, player):
    for cell in emptyCells(brd):
        x, y = cell
        brd[x][y] = player
        if winningPlayer(brd, player):
            brd[x][y] = EMPTY  # Reset to empty before returning
            return [x, y]
        brd[x][y] = EMPTY
    return None


def findBlockingMove(brd, opponent):
    return findWinningMove(brd, -opponent)  # Use winning logic for opponent to find blocking move


def AI2Move(brd, use_alpha_beta=True):
    if len(emptyCells(brd)) == 9:
        x = choice([0, 1, 2])
        y = choice([0, 1, 2])
        setMove(brd, x, y, XPLAYER)
        printBoard(brd)
    else:
        if use_alpha_beta:
            result = MiniMaxAB(brd, len(emptyCells(brd)), -inf, inf, XPLAYER)
        else:
            result = MiniMax(brd, len(emptyCells(brd)), XPLAYER)
        setMove(brd, result[0], result[1], XPLAYER)
        printBoard(brd)
    

def AIvsAI(use_alpha_beta, num_games):
    results = {'X Wins': 0, 'O Wins': 0, 'Draws': 0}
    for _ in range(num_games):
        clearBoard(board)
        currentPlayer = XPLAYER
        while not boardFull(board) and not gameWon(board):
            if use_alpha_beta:
                move = MiniMaxAB(board, len(emptyCells(board)), -inf, inf, currentPlayer)[:2]
            else:
                move = MiniMax(board, len(emptyCells(board)), currentPlayer)[:2]

            if move[0] != -1:  # Ensure move is valid
                setMove(board, move[0], move[1], currentPlayer)
            else:
                print("Invalid move detected, which should never happen in AI vs AI with Minimax")

            if gameWon(board):
                winner = 'O Wins' if currentPlayer == XPLAYER else 'X Wins'
                results[winner] += 1
                break
            elif boardFull(board):
                results['Draws'] += 1
                break

            currentPlayer = OPLAYER if currentPlayer == XPLAYER else XPLAYER
        
        # Debug: Print final board state for each game
        printBoard(board)
        print(f"Game Result: {get_winner(board)}\n")

    return results


def makeMove(brd, player, mode, use_alpha_beta=True):
    if mode == "default":
        move = default_opponent(brd)
        if move:
            setMove(brd, move[0], move[1], player)
    elif mode == "ai":
        if use_alpha_beta:
            move = MiniMaxAB(brd, len(emptyCells(brd)), -inf, inf, player)[:2]
        else:
            move = MiniMax(brd, len(emptyCells(brd)), player)[:2]
        if move:
            setMove(brd, move[0], move[1], player)


def default_vs_ai(num_games, use_alpha_beta=False):
    results = {"Default Wins": 0, "AI Wins": 0, "Draws": 0}
    for _ in range(num_games):
        board = [[EMPTY, EMPTY, EMPTY] for _ in range(3)]
        current_player = choice([XPLAYER, OPLAYER])
        while not boardFull(board) and not gameWon(board):
            if current_player == XPLAYER:
                x, y = default_opponent(board)  # Assuming default_opponent returns a tuple (x, y)
                setMove(board, x, y, current_player)
            else:
                if use_alpha_beta:
                    move = MiniMaxAB(board, len(emptyCells(board)), -inf, inf, current_player)[:2]
                else:
                    move = MiniMax(board, len(emptyCells(board)), current_player)[:2]
                if move[0] != -1:  # Ensure move is valid
                    setMove(board, move[0], move[1], current_player)

            if gameWon(board):
                # Determine the winner before switching players
                winner = "AI Wins" if current_player == OPLAYER else "Default Wins"
                results[winner] += 1
                break  # Exit the loop since the game is won
            elif boardFull(board):
                results["Draws"] += 1
                break  # Exit the loop since the board is full

            current_player = OPLAYER if current_player == XPLAYER else XPLAYER

        printBoard(board)
        print(f"Game Result: {get_winner(board)}\n")

    return results

# To use Alpha-Beta pruning
def default_vs_ai_with_pruning(num_games):
    return default_vs_ai(num_games, use_alpha_beta=True)


class QLearning:

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2, q_table_path=trained_q_table_path):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table_path = q_table_path
        self.q_table = {}
        self.load_q_table(q_table_path)

    def get_state(self, brd):
        return ''.join(str(cell) for row in brd for cell in row)

    def choose_action(self, state, available_actions):
        # Convert state to board for analysis
        board = self.state_to_board(state)

        # Check for a winning move
        for action in available_actions:
            if self.is_winning_move(board, action, OPLAYER):
                return action

        # Check for a blocking move
        for action in available_actions:
            if self.is_winning_move(board, action, XPLAYER):
                return action

        # Proceed with Q-table or exploration
        if np.random.uniform(0, 1) < self.epsilon:
            return choice(available_actions)
        else:
            q_values = [self.q_table.get((state, tuple(a)), 0) for a in available_actions]
            max_q = max(q_values)
            max_q_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]
            return choice(max_q_actions)

    def is_winning_move(self, brd, action, player):
        brd_copy = [row[:] for row in brd]
        brd_copy[action[0]][action[1]] = player
        return winningPlayer(brd_copy, player)

    def state_to_board(self, state):
        # Assuming state is a flat string '111-1-10-10'
        state = state.replace('-', '')
        board = [[int(cell) for cell in state[i:i+3]] for i in range(0, 9, 3)]
        return board

    def update_q_table(self, state, action, reward, next_state, next_available_actions):
        max_next_q = max([self.q_table.get((next_state, a), 0) for a in next_available_actions], default=0)
        self.q_table[(state, action)] = self.q_table.get((state, action), 0) + self.alpha * (reward + self.gamma * max_next_q - self.q_table.get((state, action), 0))


    def save_q_table(self):
        try:
            # Serialize the dictionary with tuple keys to string keys
            serialized_q_table = {f"{state}_{action[0]}_{action[1]}": value for (state, action), value in self.q_table.items()}
            with open(self.q_table_path, 'w') as file:
                json.dump(serialized_q_table, file)
            print(f"Q-table successfully saved to {self.q_table_path}.")
        except Exception as e:
            print(f"Failed to save Q-table to {self.q_table_path}. Error: {e}.")


    def load_q_table(self, filename):
        try:
            with open(filename, 'r') as file:
                serialized_q_table = json.load(file)
            # Deserialize the string keys back into (state, action) tuple format
            self.q_table = {(key.split('_')[0], (int(key.split('_')[1]), int(key.split('_')[2]))): value for key, value in serialized_q_table.items()}
            print(f"Q-table successfully loaded from {filename}.")
        except FileNotFoundError:
            print(f"Q-table file not found at {filename}. Starting with an empty Q-table.")
        except json.JSONDecodeError as e:
            print(f"Error decoding Q-table from {filename}. Error: {e}. Starting with an empty Q-table.")
        except Exception as e:
            print(f"Unexpected error loading Q-table from {filename}. Error: {e}. Starting with an empty Q-table.")


def train_q_learning_against_minimax(q_learning_agent1, episodes=10000, use_alpha_beta=True, save=True):
    for episode in range(episodes):
        print(f"Episode: {episode}")
        clearBoard(board)
        currentPlayer = choice([XPLAYER, OPLAYER])  # Randomize starting player

        while not (boardFull(board) or gameWon(board)):
            state = q_learning_agent1.get_state(board)
            available_actions = [tuple(a) for a in emptyCells(board)]

            if currentPlayer == XPLAYER:
                # Q-learning agent's turn
                action = q_learning_agent1.choose_action(state, available_actions)
            else:
                # Minimax's turn
                if use_alpha_beta:
                    action = MiniMaxAB(board, len(available_actions), -inf, inf, currentPlayer)[:2]
                else:
                    action = MiniMax(board, len(available_actions), currentPlayer)[:2]
                # Convert action to the same format used by Q-learning agent
                action = tuple(action)

            # Execute the chosen action
            setMove(board, action[0], action[1], currentPlayer)

            if currentPlayer == XPLAYER or gameWon(board) or boardFull(board):
                # Update Q-table only after the Q-learning agent's move or at the end of the game
                new_state = q_learning_agent1.get_state(board)
                reward = getScore(board) if currentPlayer == OPLAYER else -getScore(board) if gameWon(board) else 0
                q_learning_agent1.update_q_table(state, action, reward, new_state, [tuple(a) for a in emptyCells(board)])

            currentPlayer = OPLAYER if currentPlayer == XPLAYER else XPLAYER  # Switch turns

        # Adjust epsilon (exploration rate) for the Q-learning agent
        q_learning_agent1.epsilon *= 0.99

    if save:
        q_learning_agent1.save_q_table()


def QLearningVsQLearning(q_learning_agent1, q_learning_agent2, num_games):
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    for _ in range(num_games):
        clearBoard(board)
        currentPlayer = XPLAYER
        while not (boardFull(board) or gameWon(board)):
            state = ''.join(str(cell) for row in board for cell in row)
            available_actions = [tuple(a) for a in emptyCells(board)]
            if currentPlayer == XPLAYER:
                action = q_learning_agent1.choose_action(state, available_actions)
            else:
                action = q_learning_agent2.choose_action(state, available_actions)
            setMove(board, action[0], action[1], currentPlayer)
            if gameWon(board):
                if currentPlayer == XPLAYER:
                    agent1_wins += 1
                else:
                    agent2_wins += 1
            elif boardFull(board):
                draws += 1
            currentPlayer *= -1  # Switch turns

        printBoard(board)
        print(f"Game Result: {get_winner(board)}\n")
    results = {'Agent1 wins': agent1_wins, 'Agent2 wins': agent2_wins, 'Draws': draws}
    return results  # Returning the results dictionary


def default_vs_q_learning(q_learning_agent1, num_games):
    results = {"Default Wins": 0, "Q-Learning Wins": 0, "Draws": 0}
    for _ in range(num_games):
        board = [[EMPTY, EMPTY, EMPTY] for _ in range(3)]
        current_player = random.choice([XPLAYER, OPLAYER])
        while not boardFull(board) and not gameWon(board):
            if current_player == XPLAYER:  # Default opponent's turn
                x, y = default_opponent(board)  # Ensure default_opponent returns coordinates (x, y)
                if x is not None and y is not None:  # Check if a valid move was returned
                    setMove(board, x, y, current_player)
            else:  # Q-Learning Agent's turn
                state = q_learning_agent1.get_state(board)
                available_actions = emptyCells(board)
                action = q_learning_agent1.choose_action(state, available_actions)
                setMove(board, action[0], action[1], current_player)

            if gameWon(board):
                if current_player == XPLAYER:
                    results["Default Wins"] += 1
                else:
                    results["Q-Learning Wins"] += 1
                break  # Exit loop if the game is won
            elif boardFull(board):
                results["Draws"] += 1
                break  # Exit loop if the board is full
            current_player = OPLAYER if current_player == XPLAYER else XPLAYER

        printBoard(board)
        print(f"Game Result: {get_winner(board)}\n")
    return results


def AIvsQAI(q_learning_agent1, num_games, use_alpha_beta=False, display_board=False):

    minimax_wins = 0
    q_learning_wins = 0
    draws = 0
    for _ in range(num_games):
        clearBoard(board)
        currentPlayer = XPLAYER  # Let's assume XPLAYER is the Minimax/AI player for consistency
        while not boardFull(board) and not gameWon(board):
            state = q_learning_agent1.get_state(board)
            available_actions = [tuple(a) for a in emptyCells(board)]
            if currentPlayer == XPLAYER:
                if use_alpha_beta:
                    move = MiniMaxAB(board, len(emptyCells(board)), -inf, inf, currentPlayer)[:2]
                else:
                    move = MiniMax(board, len(emptyCells(board)), currentPlayer)[:2]
                if move[0] != -1:  # valid move
                    setMove(board, move[0], move[1], currentPlayer)
            else:  # Q-learning AI's turn
                action = q_learning_agent1.choose_action(state, available_actions)
                setMove(board, action[0], action[1], currentPlayer)
            if gameWon(board):
                if currentPlayer == XPLAYER:
                    minimax_wins += 1
                else:
                    q_learning_wins += 1
            elif boardFull(board):
                draws += 1
            if display_board:
                printBoard(board)  # Show the board after each move
            currentPlayer *= -1  # Switch turns

        printBoard(board)
        print(f"Game Result: {get_winner(board)}\n")
    return {'AI wins': minimax_wins, 'Q-Learning wins': q_learning_wins, 'Draws': draws}


def simulate_game(player1, player2, game_mode, use_alpha_beta=False, display_board=False):
    board = [[EMPTY, EMPTY, EMPTY] for _ in range(3)]
    current_player = XPLAYER  # Start with XPLAYER by convention
    while not boardFull(board) and not gameWon(board):
        move = None
        if game_mode == 'default_vs_ai':
            if current_player == XPLAYER:
                move = player1(board)
            else:
                # Directly call AIMove with the use_alpha_beta flag
                move = player2(board, use_alpha_beta)
                move = move[:2]  # Assume player2 (AIMove) returns move coordinates as the first two elements
        elif game_mode == 'default_vs_q_learning':
            if current_player == XPLAYER:
                move = player1(board)
            else:
                state = player2.get_state(board)
                move = player2.choose_action(state, emptyCells(board))
                move = (move[0], move[1])       
        if move:
            setMove(board, move[0], move[1], current_player)
            if display_board:
                printBoard(board)
        current_player = OPLAYER if current_player == XPLAYER else XPLAYER  
    return get_winner(board)


def get_winner(board):
    if winningPlayer(board, XPLAYER):
        return 'XPLAYER Wins'
    elif winningPlayer(board, OPLAYER):
        return 'OPLAYER Wins'
    else:
        return 'Draw'


def run_simulations(player1, player2, num_games=10, game_mode='default_vs_ai', use_alpha_beta=False):
    results = {'XPLAYER Wins': 0, 'OPLAYER Wins': 0, 'Draw': 0}  
    for _ in range(num_games):
        winner = simulate_game(player1, player2, game_mode, use_alpha_beta)
        results[winner] += 1 
    return results


def plot_results(results, title):
    labels = results.keys()
    values = results.values()
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel('Outcome')
    plt.ylabel('Number of Games')
    plt.show()


def main():

    # Initialize two QLearning agents
    q_learning_agent1 = QLearning(alpha=0.1, gamma=0.9, epsilon=0.2, q_table_path="/Users/priya/Desktop/AI/Game/q_table1.json")
    q_learning_agent2 = QLearning(alpha=0.1, gamma=0.9, epsilon=0.2, q_table_path="/Users/priya/Desktop/AI/Game/q_table2.json")

    # Prompt the user to train a new Q-learning policy or use the existing one
    if input("Train new Q-learning policy? (Y/N): ").lower() == 'y':
        episodes = int(input("Enter number of training episodes: "))
        # Train the Q-learning agent with the specified number of episodes
        # train_two_q_learning_agents(q_learning_agent1, q_learning_agent2, episodes)
        train_q_learning_against_minimax(q_learning_agent1, episodes, use_alpha_beta=True, save=True)

        # Save the updated Q-table at the end of training
        q_learning_agent1.save_q_table()
        q_learning_agent2.save_q_table()

    while True:
        user_action = input('Play a game (Y), Re-train Q-Learning (T), or Quit (Q)? ').lower()
        if user_action == 'y':
            game_mode = input('Select game mode - AI vs AI (1), Q-Learning vs Q-Learning (2), AI vs Q-Learning AI (3), Default vs AI (4), Default vs Q-Learning (5): ')
            
            num_games = int(input("Enter number of games: "))

            if game_mode == '2':
                results = QLearningVsQLearning(q_learning_agent1, q_learning_agent2, num_games)
                print("Simulation results:", results)
                plot_results(results, "Q-Learning vs Q-Learning Results")
            elif game_mode == '1':
                use_alpha_beta = input("Choose algorithm - Minimax without pruning (1) or with Alpha-Beta pruning (2): ") == '2'
                results = AIvsAI(use_alpha_beta=use_alpha_beta, num_games=num_games)
                print("Simulation results:", results)
                plot_results(results, "AI vs AI Results")
            elif game_mode == '3':
                use_alpha_beta = input("Choose algorithm - Minimax without pruning (1) or with Alpha-Beta pruning (2): ") == '2'
                results = AIvsQAI(q_learning_agent1, use_alpha_beta=bool(use_alpha_beta), num_games=num_games)
                print("Simulation results:", results)
                plot_results(results, "AI vs Q-Learning AI Results")
            elif game_mode == '4':
                use_alpha_beta = input("Play Default vs AI with Alpha-Beta pruning? (Y/N): ").lower() == 'y'
                results = run_simulations(default_opponent, AIMove, num_games=num_games, game_mode='default_vs_ai', use_alpha_beta=use_alpha_beta)
                adjusted_results = {
                    'Default Wins': results['XPLAYER Wins'],
                    'AI Wins': results['OPLAYER Wins'],
                    'Draws': results['Draw']
                }
                print("Simulation results:", adjusted_results)
                plot_results(adjusted_results, "Default vs AI Results")
            elif game_mode == '5':
                results = run_simulations(default_opponent, q_learning_agent1, num_games=num_games, game_mode='default_vs_q_learning')
                adjusted_results = {
                    'Default Wins': results['XPLAYER Wins'],
                    'Q-Learning Wins': results['OPLAYER Wins'],
                    'Draws': results['Draw']
                }
                print("Simulation results:", adjusted_results)
                plot_results(adjusted_results, "Default vs Q-Learning Results")
            else:
                print("Invalid game mode selected.")
        elif user_action == 't':
            episodes = int(input("Enter number of training episodes: "))
            # train_two_q_learning_agents(q_learning_agent1, q_learning_agent2, episodes)
            train_q_learning_against_minimax(q_learning_agent1, episodes, use_alpha_beta=True, save=True)
            q_learning_agent1.save_q_table()
            q_learning_agent2.save_q_table()
            print("Training completed and Q-tables saved.")
        elif user_action == 'q':
            print('Exiting program. Goodbye!')
            break
        else:
            print('Invalid input, please try again.')


if __name__ == '__main__':
    main()
