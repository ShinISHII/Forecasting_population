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
  jupyter execute --kernel python3.12 SARIMAX_ver6.ipynb
  sleep 2
done

echo ======================================
echo Tool Finish
date
echo ======================================
