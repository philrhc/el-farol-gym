import numpy as np

# Initialize parameters
alpha = 0.01  # Learning rate
gamma = 0.9  # Discount factor
num_episodes = 1000  # Number of episodes
epsilon = 0.1  # Exploration rate

# Define the grid world
grid_size = (5, 5)
start_state = (0, 0)
goal_state = (0, 4)
obstacles = {(1, 1), (1, 2), (1, 3), (2, 1), (3, 1), (3, 3)}

# Action space: 0=Up, 1=Down, 2=Left, 3=Right
actions = ['Up', 'Down', 'Left', 'Right']

# Initialize weights (features + bias term)
num_features = 7  # Example: position (x, y) + actions (4 one-hot) + bias
theta = np.random.randn(num_features)


def get_features(state, action):
    x, y = state
    features = np.zeros(num_features)
    features[0] = x
    features[1] = y
    features[action + 2] = 1  # One-hot encoding for actions
    return features


def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(len(actions))
    else:
        q_values = [np.dot(theta, get_features(state, a)) for a in range(len(actions))]
        return np.argmax(q_values)


def step(state, action):
    x, y = state
    if action == 0:  # Up
        next_state = (max(0, x - 1), y)
    elif action == 1:  # Down
        next_state = (min(grid_size[0] - 1, x + 1), y)
    elif action == 2:  # Left
        next_state = (x, max(0, y - 1))
    elif action == 3:  # Right
        next_state = (x, min(grid_size[1] - 1, y + 1))

    reward = -1  # Default reward
    if next_state == goal_state:
        reward = 10
    elif next_state in obstacles:
        next_state = state
        reward = -10

    return next_state, reward


# Training loop
for episode in range(num_episodes):
    state = start_state
    while state != goal_state:
        action = choose_action(state, epsilon)
        next_state, reward = step(state, action)
        features = get_features(state, action)
        next_features = get_features(next_state, np.argmax(
            [np.dot(theta, get_features(next_state, a)) for a in range(len(actions))]))

        # Q-learning update
        td_target = reward + gamma * np.dot(theta, next_features)
        td_error = td_target - np.dot(theta, features)
        theta += alpha * td_error * features

        state = next_state

print("Learned weights:", theta)