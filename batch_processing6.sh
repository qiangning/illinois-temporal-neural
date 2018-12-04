#!/bin/bash
CUDA_VISIBLE_DEVICES=$1
name_prefix=$2
gamma=0.5
size=2
step=20
max_epoch=60
lr=0.001
for w2v_option in 6
do
	for mode in 12 13
	do
		EXPNAME=${name_prefix}_w2v${w2v_option}_mode${mode}_sz${size}_gm${gamma}_step${step}_lr${lr}
		echo Experiment Name $EXPNAME

		echo CUDA=$CUDA_VISIBLE_DEVICES
		echo -----------------
		echo $w2v_option, $mode, $size, $gamma, $step, $lr, $max_epoch
		echo lstm_hid_dim=$(($size*128))
		echo nn_hid_dim=$(($size*64))
		echo pos_emb_dim=$(($size*32))
		echo expname=$EXPNAME
		CMD="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python exp_myLSTM.py --w2v_option $w2v_option --lstm_hid_dim $(($size*128)) --nn_hid_dim $(($size*64)) --pos_emb_dim $(($size*32)) --step_size $step --max_epoch $max_epoch --lr $lr --gamma $gamma --expname $EXPNAME --mode $mode --skiptuning> logs/$EXPNAME.txt"
		echo Running: $CMD
		eval $CMD
	done
done