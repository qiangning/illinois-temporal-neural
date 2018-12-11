#!/bin/bash
CUDA_VISIBLE_DEVICES=$1
name_prefix=repro
gamma=0.3
step=10
max_epoch=30
skiptuning=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
for w2v_option in 5
do
	for mode in -2 -1
	do
		for lr in 0.001 0.01
		do
			for dropout in false
			do
				for weight_decay in 1e-4
				do
					for lstm_hid_dim in 128 64
					do
						for nn_hid_dim in 64 32
						do
							EXPNAME=${name_prefix}_w2v${w2v_option}_mode${mode}_hidden${lstm_hid_dim}${nn_hid_dim}_gm${gamma}_step${step}_lr${lr}
							if [ "$dropout" == "true" ]; then
								EXPNAME=${EXPNAME}_dropout
							fi
							if [ "$weight_decay" != "1e-4" ]; then
								EXPNAME=${EXPNAME}_wdecay${weight_decay}
							fi
							echo Experiment Name $EXPNAME

							echo CUDA=$CUDA_VISIBLE_DEVICES
							echo -----------------
							echo $w2v_option, $mode, $size, $gamma, $step, $lr, $max_epoch
							echo lstm_hid_dim=$lstm_hid_dim
							echo nn_hid_dim=$nn_hid_dim
							echo pos_emb_dim=32
							echo expname=$EXPNAME
							CMD="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python exp_myLSTM.py --w2v_option $w2v_option --lstm_hid_dim ${lstm_hid_dim} --nn_hid_dim ${nn_hid_dim} --pos_emb_dim 32 --step_size $step --max_epoch $max_epoch --lr $lr --gamma $gamma --expname $EXPNAME --mode $mode --gen_output"
							if [ "$skiptuning" == "true" ]; then
								CMD="${CMD} --skiptuning"
							fi
							if [ "$dropout" == "true" ]; then
								CMD="${CMD} --dropout"
							fi
							if [ "$weight_decay" != "1e-4" ]; then
								CMD="${CMD} --weight_decay ${weight_decay}"
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
done