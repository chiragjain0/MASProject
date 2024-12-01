from RoverDomainCore.rover_domain import RoverDomain
import numpy as np
from parameters import parameters as p
from global_functions import *
from Visualizer.turtle_visualizer import run_rover_visualizer
from Visualizer.visualizer import run_visualizer
import copy

import matplotlib.pyplot as plt
import numpy as np

def rover_difference():
    # Initialize reward tracker
    rd = RoverDomain()
    global_rewards_over_time = []  # Track global rewards for the last episode
    final_paths = {}  # To store paths from the final episode
    rd.load_world()  # Reset the environment for each episode
    all_global_rewards = []
    global_rewards_over_time = []  # Reset rewards for each episode

    # Run multiple episodes
    num_episodes = 100  # Number of episodes
    for episode in range(num_episodes):
        rd.reset_world(0)
        epsilon = p["epsilon_q"]
        decay = p["epsilon_decay_factor"]

        # global_rewards_over_time = []  # Reset rewards for each episode

        for step in range(p["steps"]):
            rover_actions = []
            # Choose actions for all rovers
            for rv in rd.rovers:
                direction, action_bracket = e_greedy(epsilon, rd.rovers[rv])
                rd.rovers[rv].action_quad = action_bracket
                rover_actions.append(direction)

            # Environment Step
            step_rewards = rd.step(rover_actions)
            global_reward = sum(step_rewards)  # Get global reward from the environment
            global_rewards_over_time.append(global_reward)  # Track global reward

            # Calculate Difference Rewards
            diff_rewards = rd.calc_difference(step, global_reward)
            
            # Update Q-values for each rover
            for agent_idx, rv in enumerate(rd.rovers):
                local_reward = diff_rewards[agent_idx] # Modify this if using localized rewards
                rd.rovers[rv].reward = local_reward
                rd.rovers[rv].update_Qvalues()

            epsilon *= decay  # Decay exploration rate


        all_global_rewards.append(global_rewards_over_time)

        # Save paths from the final episode
        if episode == num_episodes - 1:
            final_paths = rd.rover_paths
            for rv in rd.rovers:
                print(rd.rovers[rv].reward)

        print(f"Episode {episode + 1}/{num_episodes} completed.")

     # Plot rewards and paths for the last episode
    plot_rewards(all_global_rewards)
    visualize_last_episode_paths(final_paths)

def rover_global():
    # Initialize reward tracker
    rd = RoverDomain()
    global_rewards_over_time = []  # Track global rewards for the last episode
    final_paths = {}  # To store paths from the final episode
    rd.load_world()  # Reset the environment for each episode
    all_global_rewards = []
    global_rewards_over_time = []  # Reset rewards for each episode

    # Run multiple episodes
    num_episodes = 100  # Number of episodes
    for episode in range(num_episodes):
        rd.reset_world(0)
        epsilon = p["epsilon_q"]
        decay = p["epsilon_decay_factor"]

        # global_rewards_over_time = []  # Reset rewards for each episode

        for step in range(p["steps"]):
            rover_actions = []
            # Choose actions for all rovers
            for rv in rd.rovers:
                direction, action_bracket = e_greedy(epsilon, rd.rovers[rv])
                rd.rovers[rv].action_quad = action_bracket
                rover_actions.append(direction)

            # Environment Step
            global_reward = sum(rd.step(rover_actions))  # Get global reward from the environment
            global_rewards_over_time.append(global_reward)  # Track global reward

            # Update Q-values for each rover
            for rv in rd.rovers:
                local_reward = global_reward  # Modify this if using localized rewards
                rd.rovers[rv].reward = local_reward
                rd.rovers[rv].update_Qvalues()

            epsilon *= decay  # Decay exploration rate


        all_global_rewards.append(global_rewards_over_time)

        # Save paths from the final episode
        if episode == num_episodes - 1:
            final_paths = rd.rover_paths
            for rv in rd.rovers:
                print(rd.rovers[rv].Qtable)

        print(f"Episode {episode + 1}/{num_episodes} completed.")
    
    # Plot rewards and paths for the last episode
    plot_rewards(all_global_rewards)
    visualize_last_episode_paths(final_paths)

def e_greedy(epsilon, rover):
    """
    Epsilon-greedy policy for action selection
    """
    if np.random.rand() < epsilon:
        # Explore: Random action
        action_idx = np.random.choice(len(rover.action_space))
    else:
        # Exploit: Choose best action from Q-table
        q_values = [rover.get_Qvalue(tuple(rover.observations.tolist()), a) for a in range(len(rover.action_space))]
        action_idx = np.argmax(q_values)

    direction = rover.action_space[action_idx]
    return direction, action_idx


def plot_rewards(all_rewards):
    """
    Plot global rewards over time averaged across episodes.
    """
    # Calculate average reward over episodes
    avg_rewards = np.mean(all_rewards, axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(avg_rewards, label='Average Global Reward')
    plt.xlabel('Time Steps')
    plt.ylabel('Reward')
    plt.title('Average Global Reward Over Time')
    plt.legend()
    plt.grid()
    plt.show()

def visualize_paths(rover_paths):
    """
    Plot paths taken by all rovers in the environment (final episode).
    """
    plt.figure(figsize=(10, 10))
    for rover_id, path in rover_paths.items():
        #print(path[-150:])
        path = np.array(path)  # Convert list of tuples to numpy array for easier plotting
        print(np.size(path))
        plt.plot(path[-2500:, 0], path[-2500:, 1], marker='o', label=f'Rover {rover_id}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Paths Taken by Rovers (Final Episode)')
    plt.legend()
    plt.grid()
    plt.show()

def plot_last_episode_rewards(global_rewards_over_time):
    """
    Plot global rewards over time for the last episode.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(global_rewards_over_time, label='Global Reward (Last Episode)')
    plt.xlabel('Time Steps')
    plt.ylabel('Reward')
    plt.title('Global Reward Over Time (Last Episode)')
    plt.legend()
    plt.grid()
    plt.show()

def visualize_last_episode_paths(rover_paths):
    """
    Plot paths taken by all rovers in the last episode.
    """
    plt.figure(figsize=(10, 10))
    for rover_id, path in rover_paths.items():
        path = np.array(path)  # Convert list of tuples to numpy array for easier plotting
        plt.plot(path[:, 0], path[:, 1], marker='o', label=f'Rover {rover_id}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Paths Taken by Rovers (Last Episode)')
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == "__main__":
    for _ in range(1):  # Run for multiple episodes
        rover_global()
        # rover_difference()
