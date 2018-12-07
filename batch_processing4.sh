#!/bin/bash
CUDA_VISIBLE_DEVICES=$1
name_prefix=$2
gamma=0.2
size=2
step=10
max_epoch=40
for w2v_option in 2
do
	for bilstm in false
	do
		for mode in -2 -1 0
		do
			for lr in 0.1 0.01 0.001
			do
				for dropout in false
				do
					for common_sense_emb_dim in -1
					do
						EXPNAME=${name_prefix}_w2v${w2v_option}_mode${mode}_sz${size}_gm${gamma}_step${step}_lr${lr}
						if [ "$common_sense_emb_dim" -gt 0 ]; then
							EXPNAME=${EXPNAME}_csed${common_sense_emb_dim}
						fi
						if [ "$bilstm" == "true" ]; then
							EXPNAME=${EXPNAME}_bilstm
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
						CMD="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python exp_myLSTM.py --w2v_option $w2v_option --lstm_hid_dim $(($size*128)) --nn_hid_dim $(($size*64)) --pos_emb_dim $(($size*32)) --step_size $step --max_epoch $max_epoch --lr $lr --gamma $gamma --expname $EXPNAME --mode $mode --gen_output"
						if [ "$common_sense_emb_dim" -gt 0 ]; then
							CMD="${CMD}  --common_sense_emb_dim $common_sense_emb_dim"
						fi
						if [ "$bilstm" == "true" ]; then
							CMD="${CMD} --bilstm"
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
done