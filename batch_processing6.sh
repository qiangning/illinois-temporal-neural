#!/bin/bash
CUDA_VISIBLE_DEVICES=$1
name_prefix=$2
gamma=0.5
step=10
max_epoch=50
mode=0
skiptuning=true
skiptraining=false
for lr in 0.01 0.001
do
	for finetune in 2 4 8
	do
		for weight_decay in 1e-3
		do
			echo lr=$lr
			echo mode=$mode
			EXPNAME=${name_prefix}_mode${mode}_lr${lr}_wdecay${weight_decay}_gamma${gamma}_step${step}_finetune${finetune}
			CMD="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python exp_compose_LSTM_FFNN.py --lr $lr --max_epoch $max_epoch --expname $EXPNAME --weight_decay $weight_decay --finetune $finetune"
			if [ "$skiptuning" == "true" ]; then
				CMD="$CMD --skiptuning"
			fi
			if [ "$skiptraining" == "true" ]; then
				CMD="$CMD --skiptraining"
			fi
			if [ "$skiptraining" == "true" ]; then
				EXPNAME=${EXPNAME}_skiptrain
			fi
			CMD="$CMD > logs/$EXPNAME.txt"
			echo Running: $CMD
			eval $CMD
		done
	done
done