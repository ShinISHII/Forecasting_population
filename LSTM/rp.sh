#!/bin/bash
echo ======================================
echo "Tool Start"
date
echo "parameter1(Loop): $1"
echo ======================================

for i in `seq 1 $1`
do 
  # コマンド実行
  echo Repeat: $i
  jupyter execute lstm_aggregate.ipynb
  sleep 2
done

for i in `seq 1 $1`
do 
  # コマンド実行
  echo Repeat: $i
  python3 lstm_split.py
  sleep 2
done

echo ======================================
echo Tool Finish
date
echo ======================================

