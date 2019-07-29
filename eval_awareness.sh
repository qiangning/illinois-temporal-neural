#!/bin/bash

input=lstm_w2v6_mode-1_sz2_gm0.5_step10_lr0.01.output
output=fasttext_mode-1_tuned

cd ~/Research/illinois-temporal
mvn exec:java -Dexec.mainClass=edu.illinois.cs.cogcomp.temporal.explorations.naacl19_neural.allEvaluations -Dexec.args="/home/qning2/Servers/home/Research/illinois-temporal-lstm/output/${input}" > /home/qning2/Servers/home/Research/illinois-temporal-lstm/logs/allMetricEval/${output}
