#!/bin/bash
 
export CUDA_VISIBLE_DEVICES=5
# Declare an array of city list
#declare -a city_list=("Moscow" "London" "Shanghai" "Cairo" "Delhi" "New_york" "Rio_de_Janeiro" "Sydney" "Roma" "Tokyo")
declare -a city_list=("Amman" "Istanbul" "Mexico_city" "Paris" "Singapore")

# Iterate the string array using for loop
for city in ${city_list[@]}; do
   python3 verifier_sigmoid_filtered.py --test_city Singapore --database_city $city
   sleep 1m
done


