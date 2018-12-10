#!/bin/bash
CUDA_VISIBLE_DEVICES=0
name_prefix=lstm
gamma=0.3
step=10
max_epoch=30
skiptuning=true
skiptraining=true
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
for w2v_option in 7
do
	for mode in 16
	do
		for lr in 0.001
		do
			for dropout in false # seems uneffective
			do
				for weight_decay in 1e-2
				do
					for lstm_hid_dim in 64
					do
						for nn_hid_dim in 64
						do
							for common_sense_emb_dim in 32
							do
								for granularity in 0.2
								do
									for bigramstats_dim in 2
									do
										for timeline_kb in true
										do
											EXPNAME=${name_prefix}_w2v${w2v_option}_mode${mode}_hidden${lstm_hid_dim}${nn_hid_dim}_gm${gamma}_step${step}_lr${lr}
											EXPNAME=${EXPNAME}_csed${common_sense_emb_dim}_gn${granularity}_csdim${bigramstats_dim}
											if [ "$dropout" == "true" ]; then
												EXPNAME=${EXPNAME}_dropout
											fi
											if [ "$weight_decay" != "1e-4" ]; then
												EXPNAME=${EXPNAME}_wdecay${weight_decay}
											fi
											if [ "$timeline_kb" == 'true' ]; then
												EXPNAME=${EXPNAME}_timeline
											fi
											echo Experiment Name $EXPNAME

											echo CUDA=$CUDA_VISIBLE_DEVICES
											echo -----------------
											echo $w2v_option, $mode, $size, $gamma, $step, $lr, $max_epoch
											echo lstm_hid_dim=$lstm_hid_dim
											echo nn_hid_dim=$nn_hid_dim
											echo expname=$EXPNAME
											CMD="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python exp_myLSTM.py --w2v_option $w2v_option --lstm_hid_dim ${lstm_hid_dim} --nn_hid_dim ${nn_hid_dim} --pos_emb_dim 32 --step_size $step --max_epoch $max_epoch --lr $lr --gamma $gamma --expname $EXPNAME --mode $mode --gen_output"
											CMD="${CMD}  --common_sense_emb_dim $common_sense_emb_dim --granularity $granularity --bigramstats_dim $bigramstats_dim"
											if [ "$skiptuning" == "true" ]; then
												CMD="${CMD} --skiptuning"
											fi
											if [ "$dropout" == "true" ]; then
												CMD="${CMD} --dropout"
											fi
											if [ "$weight_decay" != "1e-4" ]; then
												CMD="${CMD} --weight_decay ${weight_decay}"
											fi
											if [ "$skiptraining" == "true" ]; then
												CMD="$CMD --skiptraining"
											fi
											if [ "$timeline_kb" == 'true' ]; then
												
												CMD="$CMD --timeline_kb"
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
							done
						done
					done
				done
			done
		done
	done
done