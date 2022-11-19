#!/bin/bash
 
export CUDA_VISIBLE_DEVICES=1
# Declare an array of city list
declare -a city_list=("Moscow" "London" "Shanghai" "Cairo" "Delhi" "New_york" "Rio_de_Janeiro" "Sydney" "Roma" "Tokyo")
 
# Iterate the string array using for loop
for city in ${city_list[@]}; do
   python3 verifier_sigmoid_365_batch.py --test_city $city --database_city Moscow
   sleep 1m
done


