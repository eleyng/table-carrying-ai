"""
This script converts data collected from this table-carrying environment into the data format usable by the diffusion policy repository (https://github.com/real-stanford/diffusion_policy).

"""

import pickle
from os import listdir, mkdir
from os.path import isdir, isfile, join

import numpy as np
import zarr
import sys
sys.path.append(os.path.join(os.getcwd(), "../../diffusion_policy"))
from diffusion_policy.common.replay_buffer import ReplayBuffer
from scipy import signal

from cooperative_transport.gym_table.envs.utils import (
    FPS, get_idx_repeats_of_len_n, load_cfg, make_occupancy_grid)
from cooperative_transport.gym_table.envs.custom_rewards import (
    custom_reward_function)

new_fps = 10
skip = FPS // new_fps

yaml_filepath = join("configs/dataset_processing_params.yml")
cfg = load_cfg(yaml_filepath)
map_cfg = load_cfg(join("cooperative_transport/gym_table/config/maps", cfg["map_config"] + ".yml"))
run_name = cfg["run_name"]
npz_data_base = cfg["npz_data_base"]
pkl_data_base = cfg["pkl_data_base"]
keyboard_actions = cfg["keyboard_actions"]
occupancy_grid = cfg["make_occupancy_grid"]
remove_stopped_motion = cfg["remove_stopped_motion"]
remove_len_of_n = cfg["remove_len_of_n"]
skip_frames = cfg["skip_frames"]
skip_start_idx = cfg["skip_start_idx"]
skip_freq = cfg["skip_freq"]
low_pass_filter = cfg["low_pass_filter"]

# Make dataset directories
if not isdir(npz_data_base):
    mkdir(npz_data_base)
if not isdir(join(npz_data_base, cfg["map_config"])):
    mkdir(join(npz_data_base, cfg["map_config"]))
np_root = join(npz_data_base, cfg["map_config"], run_name)
if not isdir(np_root):
    mkdir(np_root)

pkl_traj_root = join(pkl_data_base, cfg["map_config"], run_name, "trajectories")
pkl_map_root = join(pkl_data_base, cfg["map_config"], run_name, "map_cfg")

# List all episodes collected as pkl files
traj_files = [join(pkl_traj_root, ssd) for ssd in listdir(pkl_traj_root) if isfile(join(pkl_traj_root, ssd))]
# List all map configurations for the corresponsing episodes
map_files = [join(pkl_map_root, ssd) for ssd in listdir(pkl_map_root) if isfile(join(pkl_map_root, ssd))]

print("dataset from:", pkl_traj_root)
print("map corresponding data ", pkl_map_root)
print("dataset to:", np_root)
state_lst = np.array([])
action_lst = np.array([])
past_action_lst = np.array([])
terminal = []
ep_idx = 0

buff = ReplayBuffer.create_empty_numpy()

### Loop through all pkl episodes from demos and save as npz files ###
for f in traj_files:
    game_str = f.split("/")
    game = game_str[-1].split(".")
    match = [match for match in map_files if game[0] in match]
    map_file = match[0]
    run = pickle.load(open(f, "rb"))
    map_run = dict(np.load(match[0], allow_pickle=True))
    map_data = []
    # table initial pose
    for key in map_run["table"].item():
        map_data.append(map_run["table"].item()[key])
    # table goal position
    for key in map_run["goal"].item():
        map_data.append(map_run["goal"].item()[key][0])
        map_data.append(map_run["goal"].item()[key][1])
    map_data = np.array(map_data)
    # table obstacles as grid
    map_dim = map_run["obstacles"].item()["obs_dim"]
    num_obs = map_run["obstacles"].item()["num_obstacles"]
    map_encoding = np.sum(
        np.eye(map_dim)[map_run["obstacles"].item()["obs_lst"]], axis=0
    )

    # store map data separately: get coordinates in the world frame (wf) for the table, obstacles, and goal
    wf_obs_coord_lst = np.asarray(
        map_run["obstacles"].item()["obstacles"], dtype=np.float32
    )  # shape (num_obs, 2)
    wf_goal_coord_lst = np.asarray(
        map_run["goal"].item()["goal"], dtype=np.float32
    )  # shape (2,)
    wf_table_coord_lst = np.asarray(map_data[:2], dtype=np.float32)  # shape (2,)

    # store map data as one vector
    map_data = np.concatenate((map_data, map_encoding))  # shape (14, )

    T = len(run)


    rewards = np.zeros(shape=(T, 1), dtype=np.float32)
    sh = 1  # 1 for cos sin representation
    conditioning_var = np.tile(map_data, (T, 1)).astype(np.float32)

    # For details regarding what is in the trajectory recorded, see 
    # https://github.com/eleyng/table-carrying-ai/blob/5e6f22161d730b095f12e81a49e062e67d1aae66/cooperative_transport/gym_table/envs/table_env.py#L522
    pos_idx = [0, 1]
    vel_idx = [3, 4, 5]
    f_idx = 6
    r_idx = 7
    d_idx = 8
    dt_idx = 9

    state_ep_lst = np.array([])
    action_ep_lst = np.array([])
    past_action_ep_lst = np.array([])

    for t in range(0, T, skip):

        states = np.array([])
        actions = np.array([])
        past_actions = np.array([])

        #### STATE AND ACTIONS DATA ####
        states = np.append(states, np.asarray([run[t][idx] for idx in pos_idx], dtype=np.float32).flatten())
        states = np.append(states, np.asarray([np.cos(run[t][2]), np.sin(run[t][2])], dtype=np.float32))
        states = np.append(states, np.asarray([run[t][idx] for idx in vel_idx], dtype=np.float32).flatten())

        states = np.append(states, np.asarray([map_run["table"].item()[key] for key in map_run["table"].item()], dtype=np.float32).flatten())
        states = np.append(states, wf_goal_coord_lst.flatten())
        
        if occupancy_grid:
            states = np.append(states, make_occupancy_grid(wf_obs_coord_lst.flatten()))
        else:
            states = np.append(states, wf_obs_coord_lst.flatten())
            if wf_obs_coord_lst.size < map_dim * 2:
                states = np.append(states, np.zeros((map_dim * 2 - wf_obs_coord_lst.size), dtype=np.float32))
        

        actions = np.append(actions, np.asarray(run[t][f_idx], dtype=np.float32).flatten())
        r = custom_reward_function(states, wf_goal_coord_lst, wf_obs_coord_lst, vectorized=False, interaction_forces=True, skip=1, u_r=actions[:2], u_h=actions[2:])
        states = np.append(states, np.asarray(r), dtype=np.float32)
        
        if t > 0:
            past_actions = np.append(past_actions, np.asarray(run[t-1][f_idx], dtype=np.float32).flatten())
        else:
            past_actions = np.zeros((1, 4), dtype=np.float32).flatten()

        state_lst = np.vstack([state_lst, states]) if state_lst.size else states
        action_lst = np.vstack([action_lst, actions]) if action_lst.size else actions
        past_action_lst = np.vstack([past_action_lst, past_actions]) if past_action_lst.size else past_actions

        state_ep_lst = np.vstack([state_ep_lst, states]) if state_ep_lst.size else states
        action_ep_lst = np.vstack([action_ep_lst, actions]) if action_ep_lst.size else actions
        past_action_ep_lst = np.vstack([past_action_ep_lst, past_actions]) if past_action_ep_lst.size else past_actions

    ep_idx += t
    terminal.append(ep_idx)

    ep_data = {
        'obs': state_ep_lst,
        'action': action_ep_lst,
        'past_action': past_action_ep_lst,
    }
    buff.add_episode(ep_data)

# terminal = np.asarray(terminal, dtype=np.int64)

# Save the data as a zarr file

out_path = 'datasets/table_10Hz.zarr'

storage = zarr.DirectoryStore(out_path)
out_store = zarr.DirectoryStore(out_path)
buff.save_to_store(out_store)

print("Trajectory saved to ", out_path)