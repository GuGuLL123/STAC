#!/bin/bash


while getopts 'm:e:c:t:l:w:s:' OPT; do
    case $OPT in
        m) method=$OPTARG;;
        e) exp=$OPTARG;;
        c) cuda=$OPTARG;;
		t) task=$OPTARG;;
		l) lr=$OPTARG;;
		w) cps_w=$OPTARG;;
		s) use_stac=$OPTARG;;
    esac
done
echo $method
echo $cuda

epoch=300
echo $epoch

labeled_data="labeled_20p"
unlabeled_data="unlabeled_20p"
folder="Task_"${task}"_20p/"
cps="AB"

alpha=1
bata=-1
max_extent=3

echo $folder

python code/train_${method}.py --task ${task} --exp ${folder}${method}${exp}/fold1 --seed 0 --gpu ${cuda} --base_lr ${lr} --cps_w ${cps_w} --max_epoch ${epoch} --split_labeled ${labeled_data} --split_unlabeled ${unlabeled_data} --cps_rampup --use_stac ${use_stac} --alpha ${alpha} --bata ${bata} --max_extent ${max_extent}
python code/test.py --task ${task} --exp ${folder}${method}${exp}/fold1 --gpu ${cuda} --cps ${cps} --use_stac ${use_stac} --alpha ${alpha} --bata ${bata} --max_extent ${max_extent} --seed 0
python code/evaluate_Ntimes.py --task ${task} --exp ${folder}${method}${exp} --folds 1 --cps ${cps} --use_stac ${use_stac} --alpha ${alpha} --bata ${bata} --max_extent ${max_extent} --seed 0

