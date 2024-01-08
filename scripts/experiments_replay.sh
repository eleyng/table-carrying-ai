#!/bin/bash
test_set=('unseen_map')
subject=('0') #$1
types=('diffusion_policy' 'bc_lstm_gmm' 'cogail' 'vrnn')
run_mode=('replay_traj')
human_mode=('data')


for type in "${types[@]}"
do
    echo $type

    if [ "$type" == "vrnn" ] ; then 
        python scripts/test_model.py --run-mode ${run_mode} --robot-mode planner --human-mode ${human_mode} --human-control joystick --subject-id ${subject} --render-mode headless --planner-type ${type} --display-pred --lookahead 120 --SEQ-LEN 160 &&
        python scripts/test_model.py --run-mode ${run_mode} --robot-mode planner --human-mode ${human_mode} --human-control joystick --subject-id ${subject} --render-mode headless --planner-type ${type} --data-dir datasets/rnd_obstacle_v3/random_run_name_3 --map-dir demo/rnd_obstacle_v3/random_run_name_3/map_cfg --map-config cooperative_transport/gym_table/config/maps/rnd_obstacle_v3.yml  --lookahead 120 --SEQ-LEN 160
    else
        if [ "$type" == "diffusion_policy" ] ; then
            python scripts/test_model.py --run-mode ${run_mode} --robot- planner --human-mode ${human_mode} --human-control joystick --human-act-as-cond --subject-id ${subject} --render-mode headless --planner-type ${type} &&
            python scripts/test_model.py --run-mode ${run_mode} --robot-mode planner --human-mode ${human_mode} --human-control joystick --human-act-as-cond --subject-id ${subject} --render-mode headless --planner-type ${type} --data-dir datasets/rnd_obstacle_v3/random_run_name_3 --map-dir demo/rnd_obstacle_v3/random_run_name_3/map_cfg --map-config cooperative_transport/gym_table/config/maps/rnd_obstacle_v3.yml 
        fi
        python scripts/test_model.py --run-mode ${run_mode} --robot- planner --human-mode ${human_mode} --human-control joystick --subject-id ${subject} --render-mode headless --planner-type ${type} &&
        python scripts/test_model.py --run-mode ${run_mode} --robot-mode planner --human-mode ${human_mode} --human-control joystick --subject-id ${subject} --render-mode headless --planner-type ${type} --data-dir datasets/rnd_obstacle_v3/random_run_name_3 --map-dir demo/rnd_obstacle_v3/random_run_name_3/map_cfg --map-config cooperative_transport/gym_table/config/maps/rnd_obstacle_v3.yml 
    fi

    echo "Done with ${type}."
    
done
