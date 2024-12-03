#!/bin/bash
echo ======================================
echo "Tool Start"
date
echo "parameter1(Loop): $1"
echo ======================================

########################
# exo='False'
exo='True'
layers=1


for i in `seq 1 $1`
do 
  # コマンド実行
  echo Repeat: $i
  python3 lstm_split.py "$exo" "$layers"
  sleep 2
done
########################

echo ======================================
echo Tool Finish
date
echo ======================================