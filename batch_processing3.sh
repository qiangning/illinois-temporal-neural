#!/bin/bash
CUDA_VISIBLE_DEVICES=$1
name_prefix=$2
gamma=0.5
size=2
step=20
max_epoch=80
for w2v_option in 2
do
	for mode in 10 11 12 13
	do
		for lr in 0.01 0.001
		do
			for dropout in false
			do
				for timeline_kb in true
				do
					EXPNAME=${name_prefix}_w2v${w2v_option}_mode${mode}_sz${size}_gm${gamma}_step${step}_lr${lr}
					if [ "$timeline_kb" == "true" ]; then
						EXPNAME=${EXPNAME}_timeline
					fi
					if [ "$dropout" == "true" ]; then
						EXPNAME=${EXPNAME}_dropout
					fi
					echo Experiment Name $EXPNAME

					echo CUDA=$CUDA_VISIBLE_DEVICES
					echo -----------------
					echo $w2v_option, $mode, $size, $gamma, $step, $lr, $max_epoch
					echo lstm_hid_dim=$(($size*128))
					echo nn_hid_dim=$(($size*64))
					echo pos_emb_dim=$(($size*32))
					echo expname=$EXPNAME
					CMD="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python exp_myLSTM.py --w2v_option $w2v_option --lstm_hid_dim $(($size*128)) --nn_hid_dim $(($size*64)) --pos_emb_dim $(($size*32)) --step_size $step --max_epoch $max_epoch --lr $lr --gamma $gamma --expname $EXPNAME --mode $mode --skiptuning"
					if [ "$timeline_kb" == "true" ]; then
						CMD="${CMD} --timeline_kb"
					fi
					if [ "$dropout" == "true" ]; then
						CMD="${CMD} --dropout"
					fi
					CMD="$CMD > logs/$EXPNAME.txt"
					echo Running: $CMD
					eval $CMD
				done
			done
		done
	done
done