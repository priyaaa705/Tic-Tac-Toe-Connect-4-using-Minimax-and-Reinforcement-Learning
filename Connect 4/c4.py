import numpy as np
import math
from random import choice
import matplotlib.pyplot as plt
import numpy as np
import copy
import random
import pickle
import os


class Connect4:
    ROW_COUNT = 6
    COLUMN_COUNT = 7

    def __init__(self):
        self.board = np.zeros((self.ROW_COUNT, self.COLUMN_COUNT))
        self.current_player = 1

    def make_move(self, column, player):
        for r in range(self.ROW_COUNT - 1, -1, -1):
            if self.board[r][column] == 0:
                self.board[r][column] = player
                return True
        return False

    def is_valid_location(self, column):
        return self.board[self.ROW_COUNT - 1][column] == 0

    def get_next_open_row(self, column):
        for r in range(self.ROW_COUNT):
            if self.board[r][column] == 0:
                return r
            
    def reset(self):
        """
        Reset the game board to start a new game.
        """
        self.board = np.zeros((self.ROW_COUNT, self.COLUMN_COUNT))
        self.current_player = random.choice([1, 2])  # Randomly select the starting player
        return self.get_state()

    def get_state(self):
        """
        Get the current state of the board.
        """
        return tuple(map(tuple, self.board))

    def step(self, column):
        """
        Apply an action to the board, return the new state, reward, and whether the game is done.
        """
        reward = 0
        done = False
        valid = self.make_move(column, self.current_player)
        if not valid:
            return self.get_state(), -10, True  # Penalize invalid moves
        
        if self.winning_move(self.current_player):
            reward = 1
            done = True
        elif self.is_terminal_node():
            done = True
        
        self.switch_player()
        
        return self.get_state(), reward, done

    def get_valid_actions(self):
        """
        Return a list of valid actions (columns) that can be taken.
        """
        return self.get_valid_locations()

    def print_board(self):
        print(np.flip(self.board, 0))

    def winning_move(self, player):
        # Check horizontal locations
        for c in range(self.COLUMN_COUNT - 3):
            for r in range(self.ROW_COUNT):
                if self.board[r][c] == player and self.board[r][c + 1] == player and \
                   self.board[r][c + 2] == player and self.board[r][c + 3] == player:
                    return True

        # Check vertical locations
        for c in range(self.COLUMN_COUNT):
            for r in range(self.ROW_COUNT - 3):
                if self.board[r][c] == player and self.board[r + 1][c] == player and \
                   self.board[r + 2][c] == player and self.board[r + 3][c] == player:
                    return True

        # Check positively sloped diagonals
        for c in range(self.COLUMN_COUNT - 3):
            for r in range(self.ROW_COUNT - 3):
                if self.board[r][c] == player and self.board[r + 1][c + 1] == player and \
                   self.board[r + 2][c + 2] == player and self.board[r + 3][c + 3] == player:
                    return True

        # Check negatively sloped diagonals
        for c in range(self.COLUMN_COUNT - 3):
            for r in range(3, self.ROW_COUNT):
                if self.board[r][c] == player and self.board[r - 1][c + 1] == player and \
                   self.board[r - 2][c + 2] == player and self.board[r - 3][c + 3] == player:
                    return True
        return False

    def is_terminal_node(self):
        return self.winning_move(1) or self.winning_move(2) or len(self.get_valid_locations()) == 0

    def get_valid_locations(self):
        valid_locations = []
        for col in range(self.COLUMN_COUNT):
            if self.is_valid_location(col):
                valid_locations.append(col)
        return valid_locations

    def score_position(self, player):
        score = 0
        opponent = 1 if player == 2 else 2

        ## Center column preference
        center_array = [int(i) for i in list(self.board[:, self.COLUMN_COUNT//2])]
        center_count = center_array.count(player)
        score += center_count * 6

        ## Score Horizontal
        for r in range(self.ROW_COUNT):
            row_array = [int(i) for i in list(self.board[r,:])]
            for c in range(self.COLUMN_COUNT-3):
                window = row_array[c:c+4]
                score += self.evaluate_window(window, player)

        ## Score Vertical
        for c in range(self.COLUMN_COUNT):
            col_array = [int(i) for i in list(self.board[:,c])]
            for r in range(self.ROW_COUNT-3):
                window = col_array[r:r+4]
                score += self.evaluate_window(window, player)

        ## Score positive sloped diagonal
        for r in range(self.ROW_COUNT-3):
            for c in range(self.COLUMN_COUNT-3):
                window = [self.board[r+i][c+i] for i in range(4)]
                score += self.evaluate_window(window, player)

        ## Score negative sloped diagonal
        for r in range(self.ROW_COUNT-3):
            for c in range(self.COLUMN_COUNT-3):
                window = [self.board[r+3-i][c+i] for i in range(4)]
                score += self.evaluate_window(window, player)

        return score

    def evaluate_window(self, window, player):
        score = 0
        opponent = 1 if player == 2 else 2

        if window.count(player) == 4:
            score += 100
        elif window.count(player) == 3 and window.count(0) == 1:
            score += 5
        elif window.count(player) == 2 and window.count(0) == 2:
            score += 2

        # Discourage opponent's winning move
        if window.count(opponent) == 3 and window.count(0) == 1:
            score -= 80  # Adjusted to make AI more aggressive in blocking

        return score

    def switch_player(self):
        self.current_player = 1 if self.current_player == 2 else 2

    def is_terminal_node(self):
        # Check for win or full board
        return self.winning_move(1) or self.winning_move(2) or len(self.get_valid_locations()) == 0


def smart_default_player(game):
    valid_locations = game.get_valid_locations()
    best_score = -float("inf")
    best_col = random.choice(valid_locations)  # Adds randomness to the default player's choice
    for col in valid_locations:
        row = game.get_next_open_row(col)
        temp_score = game.score_position(game.current_player) + random.randint(0, 5)  # Adjust random range as needed
        if temp_score > best_score:
            best_score = temp_score
            best_col = col
    return best_col


def minimax(game, depth, maximizingPlayer):
    valid_locations = game.get_valid_locations()
    is_terminal = game.is_terminal_node()
    if depth == 0 or is_terminal:
        if is_terminal:
            if game.winning_move(game.current_player):
                return (None, float("inf"))
            elif game.winning_move(1 if game.current_player == 2 else 2):
                return (None, -float("inf"))
            else: # Game is over, no more valid moves
                return (None, 0)
        else: # Depth is zero
            return (None, game.score_position(game.current_player))
    if maximizingPlayer:
        value = -float("inf")
        column = valid_locations[0]
        for col in valid_locations:
            row = game.get_next_open_row(col)
            game.make_move(col, game.current_player)
            game.switch_player()
            new_score = minimax(game, depth - 1, False)[1]
            game.board[row][col] = 0  # Undo the move
            game.switch_player()
            if new_score > value:
                value = new_score
                column = col
        return column, value
    else:
        value = float("inf")
        column = valid_locations[0]
        for col in valid_locations:
            row = game.get_next_open_row(col)
            game.make_move(col, game.current_player)
            game.switch_player()
            new_score = minimax(game, depth - 1, True)[1]
            game.board[row][col] = 0  # Undo the move
            game.switch_player()
            if new_score < value:
                value = new_score
                column = col
        return column, value

def minimax_ab(game, depth, maximizingPlayer, alpha, beta):
    valid_locations = game.get_valid_locations()
    is_terminal = game.is_terminal_node()
    if depth == 0 or is_terminal:
        if is_terminal:
            if game.winning_move(game.current_player):
                return (None, float("inf"))
            elif game.winning_move(1 if game.current_player == 2 else 2):
                return (None, -float("inf"))
            else:  # Game is over, no more valid moves
                return (None, 0)
        else:  # Depth is zero
            return (None, game.score_position(game.current_player))
    if maximizingPlayer:
        value = -float("inf")
        column = valid_locations[0]
        for col in valid_locations:
            row = game.get_next_open_row(col)
            game.make_move(col, game.current_player)
            game.switch_player()
            new_score = minimax_ab(game, depth-1, False, alpha, beta)[1]
            game.board[row][col] = 0  # Undo the move
            game.switch_player()
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value
    else:
        value = float("inf")
        column = valid_locations[0]
        for col in valid_locations:
            row = game.get_next_open_row(col)
            game.make_move(col, game.current_player)
            game.switch_player()
            new_score = minimax_ab(game, depth-1, True, alpha, beta)[1]
            game.board[row][col] = 0  # Undo the move
            game.switch_player()
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value


def ai_vs_ai_with_pruning(game, num_games):
    results = {"Player 1 wins": 0, "Player 2 wins": 0, "Draws": 0}
    for _ in range(num_games):
        game.reset()
        while not game.is_terminal_node():
            column, _ = minimax_ab(game, 5, True, -float("inf"), float("inf"))
            game.make_move(column, game.current_player)
            game.switch_player()
            if game.winning_move(game.current_player):
                game.print_board()
                winner = "Player 1 wins" if game.current_player == 2 else "Player 2 wins"
                results[winner] += 1
                break
        if not game.winning_move(game.current_player):
            results["Draws"] += 1
    return results


def ai_vs_ai_without_pruning(game, num_games):
    results = {"Player 1 Wins": 0, "Player 2 Wins": 0, "Draws": 0}
    for _ in range(num_games):
        game.reset()
        game_over = False
        while not game_over:
            # Use current_player to decide who's turn it is
            current_player = game.current_player
            column, _ = minimax(game, 5, True)

            game.make_move(column, current_player)
            # Check for a win or draw immediately after the move, before switching players
            if game.winning_move(current_player):
                game.print_board()
                winner = f"Player {current_player} Wins"
                results[winner] += 1
                game_over = True
            elif game.is_terminal_node():
                game.print_board()
                results["Draws"] += 1
                game_over = True
            else:
                game.switch_player()  # Only switch players if the game is not over

    return results


def default_vs_ai_with_pruning(game, num_games):
    results = {"Default Player Wins": 0, "AI Wins": 0, "Draws": 0}
    for _ in range(num_games):
        game.reset()
        game_over = False
        while not game_over:
            if game.current_player == 1:
                column = smart_default_player(game)
            else:
                column, _ = minimax_ab(game, 5, True, -float("inf"), float("inf"))
                if column is None:
                    column = choice(game.get_valid_locations())

            game.make_move(column, game.current_player)
            
            if game.winning_move(game.current_player):
                game.print_board()
                winner = "AI Wins" if game.current_player == 2 else "Default Player Wins"
                results[winner] += 1
                game_over = True
            elif game.is_terminal_node():
                results["Draws"] += 1
                game_over = True

            if not game_over:
                game.switch_player()

    return results


def default_vs_ai_without_pruning(game, num_games):
    results = {"Default Player Wins": 0, "AI Wins": 0, "Draws": 0}
    for _ in range(num_games):
        game.reset()
        game_over = False
        while not game_over:
            if game.current_player == 1:
                column = smart_default_player(game)
            else:
                column, _ = minimax(game, 5, True)

            game.make_move(column, game.current_player)
            
            if game.winning_move(game.current_player):
                game.print_board()
                winner = "AI Wins" if game.current_player == 2 else "Default Player Wins"
                results[winner] += 1
                game_over = True
            elif game.is_terminal_node():
                results["Draws"] += 1
                game_over = True

            if not game_over:
                game.switch_player()

    return results


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2, epsilon_decay=0.995, min_epsilon=0.01):
        self.Q_table = {}  # {(state, action): Q-value}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def state_to_key(self, state):
        """
        Convert the game state to a key that can be used in the Q-table.
        """
        return str(state)  # Example conversion, adjust based on your state representation

    def choose_action(self, state, valid_actions):
        """
        Choose an action based on the current state using an epsilon-greedy policy.
        """
        state_key = self.state_to_key(state)
        if random.random() < self.epsilon:  # Explore
            return random.choice(valid_actions)
        else:
            state_key = self.state_to_key(state)
            Q_values = {action: self.Q_table.get((state_key, action), 0) for action in valid_actions}
            return max(Q_values, key=Q_values.get)
        
    def save_q_table(self, filename):
        try:
            with open(filename, 'wb') as file:
                pickle.dump(self.Q_table, file)
            print(f"Q-table saved successfully to {filename}")
        except Exception as e:
            print(f"Error occurred while saving Q-table: {e}")

    def load_q_table(self, filename):
        try:
            with open(filename, 'rb') as file:
                self.Q_table = pickle.load(file)
            print(f"Q-table loaded successfully from {filename}")
        except Exception as e:
            print(f"Error occurred while loading Q-table: {e}")


    def update_Q_table(self, state, action, reward, next_state, done, valid_actions):
        """Update the Q-value based on observed transition."""
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        current_Q = self.Q_table.get((state_key, action), 0)
        future_rewards = [self.Q_table.get((next_state_key, a), 0) for a in valid_actions]
        max_future_Q = max(future_rewards) if future_rewards else 0
        updated_Q = current_Q + self.alpha * (reward + self.gamma * max_future_Q * (1 - int(done)) - current_Q)
        self.Q_table[(state_key, action)] = updated_Q

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


def train_two_q_learning_agents(game, agent1, agent2, episodes=10000):
    for episode in range(episodes):
        state = game.reset()
        done = False
        agent1_turn = random.choice([True, False])  # Randomly choose who starts

        while not done:
            agent = agent1 if agent1_turn else agent2
            valid_actions = game.get_valid_actions()
            action = agent.choose_action(state, valid_actions)
            next_state, reward, done = game.step(action)

            # For the purpose of training, consider reversing the reward when it's agent2's turn
            # This is because a positive outcome for agent1 is a negative outcome for agent2 and vice versa
            if not agent1_turn:
                reward = -reward
            
            agent.update_Q_table(state, action, reward, next_state, done, game.get_valid_actions())

            state = next_state
            agent1_turn = not agent1_turn  # Switch turns

        if episode % 100 == 0:
            print(f"Episode {episode}: Training in progress...")

    print("Training completed.")


def q_learning_vs_q_learning(game, agent1, agent2, num_games):
    results = {"Agent 1 Wins": 0, "Agent 2 Wins": 0, "Draws": 0}
    for _ in range(num_games):
        game.reset()
        agent1_turn = True  # Determines which agent's turn it is; agent1 starts first

        while not game.is_terminal_node():
            if agent1_turn:
                state = game.get_state()
                action = agent1.choose_action(state, game.get_valid_actions())
                game.make_move(action, game.current_player)
                if game.winning_move(game.current_player):
                    results["Agent 1 Wins"] += 1
                    break
            else:
                state = game.get_state()
                action = agent2.choose_action(state, game.get_valid_actions())
                game.make_move(action, game.current_player)
                if game.winning_move(game.current_player):
                    results["Agent 2 Wins"] += 1
                    break

            agent1_turn = not agent1_turn  # Switch turns
            game.switch_player()  # This assumes game.switch_player() switches the current player in the game

            if game.is_terminal_node() and not game.winning_move(game.current_player):
                results["Draws"] += 1

    return results

    
def default_vs_q_learning(game, q_agent, num_games, default_player_starts=False):
    results = {"Default Player Wins": 0, "Q-learning Agent Wins": 0, "Draws": 0}
    for _ in range(num_games):
        game.reset()
        q_agent_turn = not default_player_starts  # Determine who starts based on the flag
        while not game.is_terminal_node():
            if q_agent_turn:
                # Q-learning agent chooses action based on its policy
                action = q_agent.choose_action(game.get_state(), game.get_valid_actions())
                valid = game.make_move(action, game.current_player)
                if not valid: continue  # Skip to next iteration if not valid
            else:
                # Default player's turn
                column = smart_default_player(game)  # Your smart default player logic
                valid = game.make_move(column, game.current_player)
                if not valid: continue  # Skip to next iteration if not valid
            # After making a move, check for win condition
            if game.winning_move(game.current_player):
                game.print_board()
                if q_agent_turn:
                    results["Q-learning Agent Wins"] += 1
                else:
                    results["Default Player Wins"] += 1
                break
            if game.is_terminal_node():
                game.print_board()
                results["Draws"] += 1
                break
            q_agent_turn = not q_agent_turn  # Switch turns
            game.switch_player()
    return results


def q_learning_vs_ai_with_pruning(game, q_agent, num_games, q_agent_starts=True):
    results = {"Q-learning Agent Wins": 0, "AI with Pruning Wins": 0, "Draws": 0}
    for _ in range(num_games):
        game.reset()
        q_agent_turn = q_agent_starts

        while True:
            if q_agent_turn:
                action = q_agent.choose_action(game.get_state(), game.get_valid_actions())
                _, reward, done = game.step(action)
            else:
                column, _ = minimax_ab(game, 5, True, -float("inf"), float("inf"))
                if column is None:  # Fallback for unexpected None
                    column = choice(game.get_valid_locations())
                _, reward, done = game.step(column)

            # Check for win condition immediately after the move
            if game.winning_move(game.current_player):
                game.print_board()
                if q_agent_turn:
                    results["Q-learning Agent Wins"] += 1
                else:
                    results["AI with Pruning Wins"] += 1
                break

            # Check for draw condition
            if done or game.is_terminal_node():
                if not game.winning_move(game.current_player):
                    results["Draws"] += 1
                break

            q_agent_turn = not q_agent_turn  # Switch turns
            game.switch_player()

    return results


def q_learning_vs_ai_without_pruning(game, q_agent, num_games, q_agent_starts=True):
    results = {"Q-learning Agent Wins": 0, "AI without Pruning Wins": 0, "Draws": 0}
    for _ in range(num_games):
        game.reset()
        q_agent_turn = q_agent_starts

        while True:
            if q_agent_turn:
                action = q_agent.choose_action(game.get_state(), game.get_valid_actions())
                game.make_move(action, game.current_player)
            else:
                column, _ = minimax(game, 5, True)  # Assuming minimax returns (column, score)
                if column is not None:
                    game.make_move(column, game.current_player)
                else:
                    # Fallback in case minimax returns None, which should be rare
                    column = choice(game.get_valid_locations())
                    game.make_move(column, game.current_player)

            # Check for win condition immediately after the move
            if game.winning_move(game.current_player):
                game.print_board()
                if q_agent_turn:
                    results["Q-learning Agent Wins"] += 1
                else:
                    results["AI without Pruning Wins"] += 1
                break

            # Check for draw condition
            if game.is_terminal_node():
                game.print_board()
                if not game.winning_move(game.current_player):  # Confirm no player has won
                    results["Draws"] += 1
                break

            q_agent_turn = not q_agent_turn  # Switch turns
            game.switch_player()

    return results


def plot_results(results):
    labels = results.keys()
    wins = results.values()

    plt.bar(labels, wins, color=['blue', 'red', 'green'])
    plt.xlabel('Outcome')
    plt.ylabel('Number of Games')
    plt.title('Simulation Results')
    plt.show()

    
def main():
    game = Connect4()
    agent1 = QLearningAgent()
    agent2 = QLearningAgent()

    q_table_agent1_path = "Connect 4/q_table_agent1.pkl"
    q_table_agent2_path = "Connect 4/q_table_agent2.pkl"

    if os.path.exists(q_table_agent1_path) and os.path.exists(q_table_agent2_path):
        agent1.load_q_table(q_table_agent1_path)
        agent2.load_q_table(q_table_agent2_path)
        print("Q-tables loaded successfully.")
    else:
        print("Q-tables not found. Some options will be disabled until the agents are trained.")

    # Prompt user for action
    while True:
        print("Select a game mode:")
        print("1. Play a game")
        print("2. Train Q-Learning Agent")
        print("3. Quit")
        action = input("Choose an option: ").lower()

        if action == '1':
            print("1. AI vs AI with pruning")
            print("2. AI vs AI without pruning")
            print("3. Default vs AI with pruning")
            print("4. Default vs AI without pruning")
            print("5. Q-Learning (Player X) vs Q-Learning (Player O)")
            print("6. Default vs Q-Learning (Player X)")
            print("7. Q-Learning (Player X) vs AI (With Pruning)")
            print("8. Q-Learning (Player X) vs AI (Without Pruning)")
            
            choice = input("Enter your choice (1-8): ")
            num_games = int(input("Enter the number of games to simulate: "))

            if choice == '1':
                results = ai_vs_ai_with_pruning(game, num_games)
                print("Simulation results:", results)
                plot_results(results)
            elif choice == '2':
                results = ai_vs_ai_without_pruning(game, num_games)
                print("Simulation results:", results)
                plot_results(results)
            elif choice == '3':
                results = default_vs_ai_with_pruning(game, num_games)
                print("Simulation results:", results)
                plot_results(results)
            elif choice == '4':
                results = default_vs_ai_without_pruning(game, num_games)
                print("Simulation results:", results)
                plot_results(results)
            elif choice == '5':
                # Make sure q_learning_vs_q_learning is correctly implemented
                results = q_learning_vs_q_learning(game, agent1, agent2, num_games)
                print("Simulation results:", results)
                plot_results(results)
            elif choice == '6':
                results = default_vs_q_learning(game, agent1, num_games)
                print("Simulation results:", results)
                plot_results(results)
            elif choice == '7':
                results = q_learning_vs_ai_with_pruning(game, agent1, num_games)
                print("Simulation results:", results)
                plot_results(results)
            elif choice == '8':
                results = q_learning_vs_ai_without_pruning(game, agent1, num_games)
                print("Simulation results:", results)
                plot_results(results)
            else:
                print("Invalid choice. Please select a valid option.")
        elif action == '2':
            episodes = int(input("Enter number of episodes for training: "))
            train_two_q_learning_agents(game, agent1, agent2, episodes)
            agent1.save_q_table("Connect 4/q_table_agent1.pkl")  # Specify a filename
            agent2.save_q_table("Connect 4/q_table_agent2.pkl")  # Specify a filename
            print("Training completed and Q-tables saved.")
        elif action == '3':
            print('Exiting program. Goodbye!')
            break
        else:
            print('Invalid input, please try again.')

if __name__ == "__main__":
    main()
