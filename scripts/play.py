import argparse
import time

import gym
import numpy as np
import pygame

import cooperative_transport.gym_table.envs.utils as utils
from cooperative_transport.gym_table.envs.utils import (
    CONST_DT,
    FPS,
    MAX_FRAMESKIP,
    debug_print,
    get_keys_to_action,
    init_joystick,
    set_action_keyboard,
)

VERBOSE = False  # Set to True to print debug info


def play(env, zoom=None, keys_to_action=None):
    """Allows two people to play the game using keyboard or joysticks.

    Player 1 is the BLUE TRIANGLE.
    Player 2 is the ORANGE CIRCLE.

    To play the game use:

        python -m cooperative_transport.gym_table.scripts.play --control [keyboard|joystick] ...

    Arguments
    ---------
    env: gym.Env
        Environment to use for playing.
    zoom: float
        Make screen edge this many times bigger
    keys_to_action: dict: tuple(int) -> int or None
        Mapping from keys pressed to action performed. Useful for keyboard control only.
        If None, default key_to_action mapping for that env is used, if provided.
    """
    obs = env.reset()
    rendered = env.render(mode="rgb_array")

    if keys_to_action is None:
        if hasattr(env, "get_keys_to_action"):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, "get_keys_to_action"):
            keys_to_action = env.unwrapped.get_keys_to_action()
        else:
            keys_to_action = get_keys_to_action()

    relevant_keys = set(sum(map(list, keys_to_action.keys()), []))

    video_size = [rendered.shape[1], rendered.shape[0]]
    if zoom is not None:
        video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

    pressed_keys = []
    running = True
    env_done = False

    screen = pygame.display.set_mode(video_size)
    next_game_tick = time.time()
    clock = pygame.time.Clock()
    cnt = 0
    if env.control_type == "joystick":
        joysticks = init_joystick()

    # GAME LOOP
    while running:
        loops = 0
        cnt += 1
        if env_done:
            env_done = False
            # obs = env.reset()
            time.sleep(1)

            debug_print("Done. Resetting environment.")
        else:
            while time.time() > next_game_tick and loops < MAX_FRAMESKIP:
                if env.control_type == "keyboard":
                    action = keys_to_action.get(tuple(sorted(pressed_keys)), 0)
                    action = set_action_keyboard(action)
                    action = action.reshape(1, -1).flatten()
                elif env.control_type == "joystick":
                    p1_id = 0  # player 1 is blue
                    p2_id = 1  # player 2 is orange

                    assert (
                        len(joysticks) == 2
                    ), "Demonstration collection in joystick control requires 2 joysticks."

                    action = np.array(
                        [
                            joysticks[p1_id].get_axis(0),
                            joysticks[p1_id].get_axis(1),
                            joysticks[p2_id].get_axis(0),
                            joysticks[p2_id].get_axis(1),
                        ]
                    )

                debug_print("Action: ", action)
                print("Action: ", action, action.shape)
                obs, rew, env_done, info = env.step(action)
                debug_print("Loop: ", loops, info, "\n\, obs: ", obs)
                next_game_tick += CONST_DT
                loops += 1
                if env_done:
                    break

            clock.tick(FPS)
            if clock.get_fps() > 0:
                debug_print("Reported dt: ", 1 / clock.get_fps())

        if obs is not None:
            rendered = env.render(mode="rgb_array")

        # process pygame events
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key in relevant_keys:
                    debug_print("REGISTERED KEY PRESS")
                    pressed_keys.append(event.key)
                elif event.key == 27:
                    running = False
            elif event.type == pygame.KEYUP:
                if event.key in relevant_keys:
                    pressed_keys.remove(event.key)
            elif event.type == pygame.QUIT:
                running = False

    pygame.quit()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="cooperative_transport.gym_table:table-v0",
        help="Define Environment",
    )
    parser.add_argument(
        "--obs",
        type=str,
        default="discrete",
        help="Define Observation Space, discrete/rgb (rgb still testing)",
    )
    parser.add_argument(
        "--control",
        type=str,
        default="joystick",
        help="Define Control Input, keyboard/joystick",
    )
    parser.add_argument(
        "--map_config",
        type=str,
        default="cooperative_transport/gym_table/config/maps/rnd_obstacle_v2.yml",
        help="Map Config File Path",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="random_run_name_3",
        help="run_name name for data collection. creates a folder with this name in the repo's base directory to store data.",
    )
    parser.add_argument(
        "--ep",
        type=int,
        default=0,
        help="episode number of trajectory data.",
    )
    parser.add_argument(
        "--max_obs",
        type=int,
        default=3,
        help="maximum number of obstacles to spawn in the environment.",
    )

    args = parser.parse_args()
    env = gym.make(
        args.env,
        obs=args.obs,
        control=args.control,
        map_config=args.map_config,
        run_mode="demo",
        run_name=args.run_name,
        ep=args.ep,
        dt=CONST_DT,
        render_mode="gui",
        max_num_obstacles=args.max_obs,
    )

    play(env, zoom=1)


if __name__ == "__main__":
    main()
