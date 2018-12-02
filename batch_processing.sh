#!/bin/bash
CUDA_VISIBLE_DEVICES=$1
w2v_option=$2
gamma=0.2
size=2
# gamma=$3
# size=$4
# mode=$5
for mode in -1 0 1 3
do
	if [[ $# -lt 6 ]]; then
		EXPNAME=w2v${w2v_option}_mode${mode}_sz${size}_gm${gamma}
	else
		EXPNAME=$6
	fi
	echo Experiment Name $EXPNAME

	echo CUDA=$CUDA_VISIBLE_DEVICES
	echo -----------------
	echo $w2v_option, $size, $gamma, $mode
	echo lstm_hid_dim=$(($size*128))
	echo nn_hid_dim=$(($size*64))
	echo pos_emb_dim=$(($size*32))
	echo expname=$EXPNAME
	CMD="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python exp_myLSTM.py --w2v_option $w2v_option --lstm_hid_dim $(($size*128)) --nn_hid_dim $(($size*64)) --pos_emb_dim $(($size*32)) --step_size 5 --max_epoch 20 --gamma $gamma --expname $EXPNAME --mode $mode --skiptuning > logs/$EXPNAME.txt"
	echo Running: $CMD
	eval $CMD
done