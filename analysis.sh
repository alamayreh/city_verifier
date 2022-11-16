#!/bin/bash
 
export CUDA_VISIBLE_DEVICES=4
# Declare an array of city list
declare -a city_list=("Moscow" "London" "Shanghai" "Cairo" "Delhi" "New_york" "Rio_de_Janeiro" "Sydney" "Roma" "Tokyo")
 
# Iterate the string array using for loop
for city in ${city_list[@]}; do
   python3 analysis_results.py --test_city Cairo --database_city $city
   #sleep 1m
done


