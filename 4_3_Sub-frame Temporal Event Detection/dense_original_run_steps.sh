#!/bin/bash --login

run=65

for split in 1 2 3 4
do
  for step in 16 8 4 2 1
  do
    python -u dense_original_train.py $step $run $split
    python -u dense_original_eval.py $step $run $split
  done
done


