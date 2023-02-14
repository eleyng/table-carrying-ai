# Table Carrying (╯°□°）╯┬─┬ノ( ◕◡◕ ノ)

An environment for table-carrying, a *joint-action* cooperative task.

*[TODO]: Insert gif*

## About

This is a continuous state-action custom environment for a two-agent cooperative carrying task. Possible configurations are human-human, human-robot, and robot-robot. 

The objective is to carry the table from start to goal while avoiding obstacles. Each agent is physically constrained to the table while moving it. Rewards can be customized to achieve task success (reaching goal without hitting obstacles), but also other cooperative objectives (minimal interaction forces, etc. for *fluency*).

The main branch environment is used in the 2023 paper *[It Takes Two: Learning to Plan for Human-Robot Cooperative Carrying](https://arxiv.org/abs/2209.12890)*. 

## Installation

We recommend using any environment manager to keep your dependencies clean. For conda:
1. Create a conda env: 
  `conda create --name table python=3.8`
  `conda activate table`
2. Clone this repo using `git clone git@github.com:eley-ng/table-carrying-ai.git`.
3. Install the dependencies using `pip install -r requirements.txt --use-deprecated=legacy-resolver`.
4. `cd ../; pip install -e .` to install.
5. Test that the install worked by running: `python scripts/cooperative-transport/cooperative_transport/gym_table/test/test.py`.

## Code Structure Overview

`table-carrying-ai` contains:

`cooperative-transport/`:
- `gym-table/`: houses gym env
  - `config/` : stores config files to pre-load map configs. Specify potential obstacle locations, goal locations, and initial table pose
  - `envs/` : main env code
    - `table_env.py` : pygame / gym class
    - `utils.py` : stores necessary table parameters and optional methods for table env, such as keyboard mapping if using discrete actions
  - `test/` : houses scripts for interacting with env
    - `test_joystick.py` : checks if joysticks are configured correctly
    - `play.py` : run this to collect human-human demos

```
└── cooperative-transport
    ├── config
    │   ├── maps
    │   │   ├── rnd_obstacle_v2.yml
    │   │   └── ... (custom map configurations specified here)
    │   ├── game_objects_params.yml : specify physics parameters and video rendering parameters
    │   └── inference_params.yml
    ├── envs
    │   ├── __init__.py
    │   ├── game_objects
    │   │   ├── images
    │   │   │   └── ... (images for the game)
    │   │   └── game_objects.py 
```

    
## TODO:
- add gif

    

