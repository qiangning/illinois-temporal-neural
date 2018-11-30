#!/bin/bash
CUDA_VISIBLE_DEVICES=$1
elmo_options=$2
gamma=$3
size=$4
if [[ $# -lt 5 ]]; then
	EXPNAME=${elmo_options}_sz${size}_gm${gamma}
else
	EXPNAME=$5
fi
echo Experiment Name $EXPNAME

echo CUDA=$CUDA_VISIBLE_DEVICES
echo -----------------
echo $elmo_options, $size, $gamma
echo lstm_hid_dim=$(($size*128))
echo nn_hid_dim=$(($size*64))
echo pos_emb_dim=$(($size*32))
echo expname=$EXPNAME
CMD="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python exp_myLSTM.py --elmo_option $elmo_options --lstm_hid_dim $(($size*128)) --nn_hid_dim $(($size*64)) --pos_emb_dim $(($size*32)) --step_size 5 --max_epoch 20 --gamma $gamma --expname $EXPNAME> logs/$EXPNAME.txt"
echo Running: $CMD
eval $CMD