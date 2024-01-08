#!/bin/bash
test_set=('test_holdout')
subject=('100')
types=('cogail' 'diffusion_policy' 'bc_lstm_gmm' 'vrnn')
run_mode=('coplanning') 
human_mode=('planner')

for type in "${types[@]}"
do
    echo $type

    if [ "$type" == "vrnn" ] ; then 
        python scripts/test_model_${test_set}.py --run-mode ${run_mode} --robot-mode planner --human-mode ${human_mode} --human-control joystick --subject-id ${subject} --render-mode headless --planner-type ${type} --display-pred --lookahead 120 --SEQ-LEN 160 --map-config cooperative_transport/gym_table/config/maps/varied_maps_${test_set}.yml
    else
        
        python scripts/test_model_${test_set}.py --run-mode ${run_mode} --robot-mode planner --human-mode ${human_mode} --human-control joystick --subject-id ${subject} --render-mode headless --planner-type ${type} --map-config cooperative_transport/gym_table/config/maps/varied_maps_${test_set}.yml
        if [ "$type" == "diffusion_policy" ] ; then
            python scripts/test_model_${test_set}.py --run-mode ${run_mode} --robot- planner --human-mode ${human_mode} --human-control joystick --human-act-as-cond --subject-id ${subject} --render-mode headless --planner-type ${type} --map-config cooperative_transport/gym_table/config/maps/varied_maps_${test_set}.yml
        fi
    fi

    echo "Done with ${type}."
    
done
