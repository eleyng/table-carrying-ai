import numpy as np
from scipy import signal
from scipy.fftpack import fft, ifft
import pickle
from os import listdir, mkdir
from os.path import join, isfile, isdir
from cooperative_transport.gym_table.envs.utils import (
    load_cfg,
    get_idx_repeats_of_len_n,
    FPS,
)

yaml_filepath = join("configs/dataset_processing_params.yml")
cfg = load_cfg(yaml_filepath)
map_cfg = load_cfg(join("cooperative_transport/gym_table/config/maps", cfg["map_config"] + ".yml"))
run_name = cfg["run_name"]
npz_data_base = cfg["npz_data_base"]
pkl_data_base = cfg["pkl_data_base"]
keyboard_actions = cfg["keyboard_actions"]
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

    if keyboard_actions:
        actions = np.zeros(shape=(T, 1), dtype=np.float32)
    else:
        actions = np.zeros(shape=(T, 4), dtype=np.float32)

    rewards = np.zeros(shape=(T, 1), dtype=np.float32)
    terminal = np.zeros(shape=(T, 1), dtype=np.float32)
    sh = 1  # 1 for cos sin representation
    states = np.zeros(
        shape=(T, 3 + sh + 2 + 2), dtype=np.float32
    )  # x, y, csth, sth, goal, most_relevant_obs
    conditioning_var = np.tile(map_data, (T, 1)).astype(np.float32)

    # For details regarding what is in the trajectory recorded, see 
    # https://github.com/eleyng/table-carrying-ai/blob/5e6f22161d730b095f12e81a49e062e67d1aae66/cooperative_transport/gym_table/envs/table_env.py#L522
    pos_idx = [0, 1]
    vel_idx = [3, 4, 5]
    f_idx = 6
    r_idx = 7
    d_idx = 8
    dt_idx = 9

    for t in range(T):

        #### STATE AND ACTIONS DATA ####
        states[t, :2] = np.asarray([run[t][idx] for idx in pos_idx], dtype=np.float32).flatten()
        states[t, 2:4] = np.asarray([np.cos(run[t][2]), np.sin(run[t][2])], dtype=np.float32)
        states[t, 4:7] = np.asarray([run[t][idx] for idx in vel_idx], dtype=np.float32).flatten()
        actions[t] = np.asarray(run[t][f_idx], dtype=np.float32).flatten()
        rewards[t] = np.asarray(run[t][r_idx], dtype=np.float32).flatten()
        terminal[t] = np.asarray(run[t][d_idx], dtype=np.float32).flatten()

        #### MAP DATA ####
        state_table = np.array([states[t, :2]])

        # goal
        GOAL = map_run["goal"].item()[key]
        direction = GOAL - state_table
        dist2goal = np.linalg.norm(direction)

        # obstacles
        OBSTACLES = map_run["obstacles"].item()["obstacles"]
        avoid = OBSTACLES - state_table
        dist2obs = np.linalg.norm(avoid, axis=1)

        states[t, 3 + sh] = GOAL[0]
        states[t, 4 + sh] = GOAL[1]
        most_relevant_obs_idx = np.argmin(dist2obs)
        most_relevant_obs = OBSTACLES[most_relevant_obs_idx]
        states[t, 5 + sh] = most_relevant_obs[0]
        states[t, 6 + sh] = most_relevant_obs[1]


    ### OPTIONAL: Additional filtering, combine as needed or comment out as desired ###
    
    # Option A: Subsample by taking every {fr} frame, starting from idx {st}
    if skip_frames:
        fr = skip_frames
        st = skip_start_idx
        states = states[st::fr]
        actions = actions[st::fr]
        rewards = rewards[st::fr]
        terminal = terminal[st::fr]
        conditioning_var = conditioning_var[st::fr]

    # Option B: Remove all states where the table is not moving
    if remove_stopped_motion:
        repeat_intersect = get_idx_repeats_of_len_n(states[:, :3], remove_len_of_n)
        if repeat_intersect.size != 0:
            print("game:", game, "repeat intersect", repeat_intersect)
        states = np.delete(states, repeat_intersect, axis=0)
        actions = np.delete(actions, repeat_intersect, axis=0)
        rewards = np.delete(rewards, repeat_intersect, axis=0)
        terminal = np.delete(terminal, repeat_intersect, axis=0)
        conditioning_var = np.delete(conditioning_var, repeat_intersect, axis=0)

    # Option C: Apply low pass filter to actions
    if low_pass_filter:
        sos = signal.butter(1, 3, 'low', fs=FPS, output='sos')
        for i in range(actions.shape[1]):
            filtered = signal.sosfiltfilt(sos, actions[:, i])
            filtered = np.clip(filtered, -1, 1)
            actions[:, i] = filtered

    ### Check data for validity ###
    if np.any(actions) < -1.0 or np.any(actions) > 1.0:
        print("Out of valid range of actions.", f)
        raise ValueError("Out of valid range of actions.")
    if np.any(abs(states[:, :3]) < 0.5 * 10 ** (-6)) or np.any(states[:, 0]) <= 0.0:
        print("Out of valid range of states (x-dim).", f)
        raise ValueError("Out of valid range of states (x-dim).")
    if np.any(states[:, 1]) < 0 or np.any(states[:, 1]) > 600:
        print("Out of valid range of states (y-dim).", f)
        raise ValueError("Out of valid range of states (y-dim).")
    if np.any(states[:, 2:4]) < -1 or np.any(states[:, 2:4]) > 1:
        print("Out of valid range of states (theta dim).", f)
        raise ValueError("Out of valid range of states (theta dim).")

    ### Save data as .npz file for model handling ###
    if not isdir(join(np_root, game[-2])):
        mkdir(join(np_root, game[-2]))
    rollout_f = join(np_root, game[-2], "ep_" + game[-2] + ".npz")

    np.savez(
        rollout_f,
        states=states,
        actions=actions,
        rewards=rewards,
        terminal=terminal,
        conditioning_var=conditioning_var,
        map_file=map_file,
        table_init=wf_table_coord_lst,
        table_goal=wf_goal_coord_lst,
        obstacles=wf_obs_coord_lst,
    )
    print("Trajectory saved to ", rollout_f)