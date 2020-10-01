#!/bin/bash --login

run=66

for step in 16 8 4 2 1
do
  for split in 1 2 3 4
  do
    python -u multi_offset_train.py $step $run $split
    python -u multi_offset_eval.py $step $run $split
  done
done

