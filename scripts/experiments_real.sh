#!/bin/bash


subject=$1

types=( 'bc_lstm_gmm' 'cogail' 'vrnn' 'diffusion_policy'  )

for idx in  1 
do

    for type in "${types[@]}"
    do
        echo "Starting experiment with ${type} for subject ${subject}."

        if [ "$type" == "vrnn" ]
        then

            read -p "Press enter to continue"

            python scripts/test_model_real.py --run-mode hil --robot-mode planner --human-mode real --human-control joystick --subject-id $1  --render-mode gui --planner-type ${type} --lookahead 120 --SEQ-LEN 160 --map-config datasets/real_test/real_test.yml --test-idx ${idx}

        
        else

            read -p "Press enter to continue"

            python scripts/test_model_real.py --run-mode hil --robot-mode planner --human-mode real --human-control joystick --subject-id $1  --render-mode gui --planner-type ${type}  --map-config datasets/real_test/real_test.yml --test-idx ${idx}


            if [ "$type" == "diffusion_policy" ]
            then

                read -p "Press enter to continue"

                python scripts/test_model_real.py --run-mode hil --robot-mode planner --human-mode real --human-control joystick --subject-id $1  --human-act-as-cond --render-mode gui --planner-type ${type}  --map-config datasets/real_test/real_test.yml --test-idx ${idx}

            fi

        fi        
        echo "Done with ${type}."

    done

done

echo "Done with all experiments."