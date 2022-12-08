#!/bin/bash
 
#export CUDA_VISIBLE_DEVICES=4
# Declare an array of city list
declare -a city_list=("Moscow" "London" "Shanghai" "Cairo" "Delhi" "New_york" "Rio_de_Janeiro" "Sydney" "Roma" "Tokyo")
#declare -a city_list=("Amman" "Istanbul" "Mexico_city" "Paris" "Singapore")

# Iterate the string array using for loop
for city in ${city_list[@]}; do
   #python3 analysis_vipp.py --test_city Tokyo --database_city $city
   #python3 analysis_results_rejection.py --test_city London --database_city $city
   #python3 analysis_results.py --test_city Tokyo --database_city $city
   python3 analysis_filtered_vipp.py --test_city Tokyo --database_city $city
   #sleep 1m
done


