from RoverDomainCore.rover_domain import RoverDomain
import numpy as np
from parameters import parameters as p
from global_functions import *
from Visualizer.turtle_visualizer import run_rover_visualizer
from Visualizer.visualizer import run_visualizer

def rover_global():
    """
    Train rovers in the classic rover domain using the global reward
    """
    # World Setup
    rd = RoverDomain()
    rd.load_world()
    ep = 0
    while ep < p["n_eps"]:
        rd.reset_world(0)
        
        # Q-learning
        # Take the step
        # Get observations(s') and rewards r
        # Update Q values for each agent according to action each took.
        # decay epsilon
        epsilon = p["epsilon_q"]
        decay = p["epsilon_decay_factor"]
        poi_rewards = np.zeros((p["n_poi"], p["steps"]))
        step = 0
        g_reward = 0.0
        
        while step < p["steps"] or rd.done:
            if(step % 1000 == 0):
                print("Episode #: %d \t Step #: %d"% (ep, step))
                print("global_rewards: %f"% g_reward)
                print("***********************************************")
                
            rover_actions = []
            for rv in rd.rovers:
                direction, angle = e_greedy(epsilon, rd.rovers[rv])
                action_bracket = int(angle / rd.rovers[rv].sensor_res)
                if action_bracket > rd.rovers[rv].n_brackets - 1:
                        action_bracket -= n_brackets
                rd.rovers[rv].action_quad = action_bracket
                rover_actions.append(direction)
                
            step_rewards = rd.step(rover_actions)
            g_reward = sum(step_rewards) - 1 # -1 for taking a step.
            
            # Assign the agents with thier local rewards.
            for rv in rd.rovers:
                # Using G(z) for local rewards and Q-value updates.
                rd.rovers[rv].reward = g_reward
                
                # Call Update Qvalues
                rd.rovers[rv].update_Qvalues()
                # print(step, rd.rovers[rv].reward)
            
            epsilon *= decay
            rd.done = rd.goals_done()
            step += 1
        ep += 1

def e_greedy(epsilon, rv):
    if np.random.rand() < epsilon:
        # We explore
        direction = [np.random.rand(), np.random.rand()]
        angle = get_angle(direction[0], direction[1], p["x_dim"]/2, p["y_dim"]/2)
        return direction, angle
    else:
        # We exploit
        q_values = []
        # print(tuple(rv.observations.tolist()))
        for action in range(rv.n_brackets):
            q_values.append(rv.get_Qvalue(tuple(rv.observations.tolist()),action))
        # q_values = [rv.get_Qvalue(rv.observations, action) for action in range(rv.n_brackets)]
        max_Qvalue = np.max(q_values)
        best_action = np.argmax(q_values)
        
        direction = rv.action_space[best_action]        
        angle = get_angle(direction[0], direction[1], p["x_dim"]/2, p["y_dim"]/2)
        return direction, angle

if __name__ == '__main__':
    """
    Run classic or tightly coupled rover domain using either G, D, D++, or CFL
    This main file is for use with rovers learning navigation (not skills)
    """
    rover_global()