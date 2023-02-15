# Table Carrying

An environment for table-carrying, a *joint-action* cooperative task. (╯°□°）╯┬─┬ノ( ◕◡◕ ノ)

*[TODO]: Insert gif*

## About

This is a continuous state-action custom environment for a two-agent cooperative carrying task. Possible configurations are human-human, human-robot, and robot-robot. 

The objective is to carry the table from start to goal while avoiding obstacles. Each agent is physically constrained to the table while moving it. Rewards can be customized to achieve task success (reaching goal without hitting obstacles), but also other cooperative objectives (minimal interaction forces, etc. for *fluency*).

The main branch environment is used in the 2023 paper *[It Takes Two: Learning to Plan for Human-Robot Cooperative Carrying](https://arxiv.org/abs/2209.12890)*. 

## Installation

We recommend using any environment manager to keep your dependencies clean. For conda:
1. Create a conda env:   
  `conda create --name table python=3.8`  
  `conda activate table-carrying`
2. Clone this repo using `git clone git@github.com:eley-ng/table-carrying-ai.git`.
3. Install the dependencies using `pip install -r requirements.txt --use-deprecated=legacy-resolver`.
4. `cd ../; pip install -e .` to install.
5. Test that the install worked by running: `python scripts/cooperative-transport/cooperative_transport/gym_table/test/test.py`.

## Custom Env Structure Overview

The core custom environment code is under `cooperative-transport`, which contains:

```
└── cooperative-transport/
    ├── config/
    │   ├── maps/
    │   │   ├── rnd_obstacle_v2.yml
    │   │   └── ... (custom map configs specified here: potential obstacle locations, goal locations, and initial table pose which the env will sample from)
    │   ├── game_objects_params.yml : specify physics parameters and video rendering parameters
    │   └── inference_params.yml
    ├── envs/
        ├── __init__.py
        ├── game_objects/
        │   ├── images/
        │   │   └── ... (images for the game)
        │   └── game_objects.py
        ├── table_env.py
        └── utils.py
```

To test if your local install works, run the following test scripts:

```    
└── test/
    ├── test_joystick.py : check if joystick(s) are working if using joystick control
    └── test_gym_table.py : check if env is working
```

There are several things you can do with this environment, by running any of the following scripts:

```
└── scripts/
    ├── data_playback.py : render a saved trajectory with pygame
    ├── play.py : collect demonstrations with two humans (**interactive**)
    ├── test_model.py : load a model in two modes: 1) (**interactive**) one-player (human) w/ robot, 2) robot-robot only
    └── visualize.py : plot a saved trajectory and save as image  
```
    
## TODO:
- add gif
- add trained models
- add model class example
- add tests 
- add scripts
- test download instructions   

