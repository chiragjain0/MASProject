from rover_domain_python import *

class Args:
    '''
    env_choice: Task type for the environment (e.g., 'rover_tight', 'rover_loose', or 'rover_trap').
    num_poi: Number of POIs.
    num_agents: Number of rovers (agents).
    dim_x and dim_y: Dimensions of the environment grid.
    angle_res: Angle resolution for sensor discretization.
    obs_radius: Radius within which objects are observable.
    act_dist: Distance within which POIs are considered accessible.
    ep_len: Length of each episode (number of steps).
    harvest_period: Time required to interact with a POI
    is_lsg : local subsume global
    is_proxim_rew: proximity rewards
    '''
    env_choice = 'rover_loose'
    num_poi = 5
    num_agents = 2
    dim_x, dim_y = 100, 100
    angle_res = 10
    obs_radius = 20
    act_dist = 5
    ep_len = 100
    harvest_period = 3
    poi_rand = False
    sensor_model = 'density'
    is_lsg = False
    is_proxim_rew = False

args = Args()
rover_env = RoverDomainVel(args)

state = rover_env.reset()

done = False
while not done:
    joint_action = np.random.uniform(-1, 1, (args.num_agents, 2))  # Random 
    next_state, local_rewards, done, global_reward = rover_env.step(joint_action)
rover_env.render()       # Prints environment state in console
# rover_env.viz(save=False, fname='F')  # Saves visualization as PNG
rover_env.viz(save=False)  # Saves visualization as PNG



