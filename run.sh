#!/bin/bash

dataset=dblp-sub
a=0.25
b=0.25
c=0.25
d=0.25
t=0.6
rel_num=2

python3 src/train.py --dataset $dataset --prefix full_model --a $a --b $b --c $c --d $d --t $t

