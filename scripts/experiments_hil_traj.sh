#!/bin/bash

subject=$1
types=('bc_lstm_gmm' 'cogail' 'diffusion_policy' 'vrnn')
run_mode=('hil') 
human_mode=('real') 

for type in "${types[@]}"
do
    echo "Beginning experiments. $type"
    read -t 5 -p "Press enter to continue or wait 5 seconds."

    if [ "$type" == "vrnn" ] ; then 
        python scripts/test_model.py --run-mode ${run_mode} --robot-mode planner --human-mode ${human_mode} --human-control joystick --subject-id ${subject} --render-mode gui --planner-type ${type} --data-dir datasets/rnd_obstacle_v3/random_run_name_3 --map-dir demo/rnd_obstacle_v3/random_run_name_3/map_cfg --map-config cooperative_transport/gym_table/config/maps/rnd_obstacle_v3.yml  --lookahead 120 --SEQ-LEN 160
    else
        if [ "$type" == "diffusion_policy" ] ; then
            python scripts/test_model.py --run-mode ${run_mode} --robot-mode planner --human-mode ${human_mode} --human-control joystick --human-act-as-cond --subject-id ${subject} --render-mode gui --planner-type ${type} --data-dir datasets/rnd_obstacle_v3/random_run_name_3 --map-dir demo/rnd_obstacle_v3/random_run_name_3/map_cfg --map-config cooperative_transport/gym_table/config/maps/rnd_obstacle_v3.yml 
        fi
        python scripts/test_model.py --run-mode ${run_mode} --robot-mode planner --human-mode ${human_mode} --human-control joystick --subject-id ${subject} --render-mode gui --planner-type ${type} --data-dir datasets/rnd_obstacle_v3/random_run_name_3 --map-dir demo/rnd_obstacle_v3/random_run_name_3/map_cfg --map-config cooperative_transport/gym_table/config/maps/rnd_obstacle_v3.yml 
    fi

    echo "Done with ${type}."
    
done
