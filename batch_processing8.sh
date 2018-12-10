#!/bin/bash
CUDA_VISIBLE_DEVICES=$1
name_prefix=$2
gamma=0.3
step=10
max_epoch=50
mode=1
skiptuning=true
for lr in 0.001
do
	for nn_hidden_dim in 512 256
	do
		echo lr=$lr
		echo nn_hidden_dim=$nn_hidden_dim
		echo mode=$mode
		EXPNAME=${name_prefix}_mode${mode}_lr${lr}_hid${nn_hidden_dim}_gamma${gamma}_step${step}
		CMD="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python exp_myFFNN.py --lr $lr --max_epoch $max_epoch --nn_hidden_dim $nn_hidden_dim --expname $EXPNAME --mode $mode"
		if [ "$skiptuning" == "true" ]; then
			CMD="$CMD --skiptuning"
		fi
		CMD="$CMD > logs/$EXPNAME.txt"
		echo Running: $CMD
		eval $CMD
	done
done