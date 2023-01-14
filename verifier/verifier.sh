#!/bin/bash
 
export CUDA_VISIBLE_DEVICES=4
# Declare an array of city list
#declare -a city_list=("Moscow" "St_Petersburg" "London" "Edinburgh" "Shanghai" "Beijing" "Cairo" "Delhi" "New_york" "Los_Angeles" "Rio_de_Janeiro" "Sydney" "Roma" "Milan" "Tokyo")
declare -a city_list=("Amman" "Istanbul" "Mexico_city" "Paris" "Singapore")


#declare -a city_list=("Cairo" "Delhi" "New_york" "Los_Angeles" "Rio_de_Janeiro" "Sydney" "Tokyo")
#declare -a city_list=("New_york" "London" "Moscow" "Tokyo")

# Iterate the string array using for loop
#for city in ${city_list[@]}; do
#   python3 verifier_sigmoid_filtered.py --test_city $city --database_city $city
#done


#declare -a city_list=("Roma" "Milan")

for city1 in ${city_list[@]}; do
   for city2 in ${city_list[@]}; do
      python3 verifier_sigmoid_filtered.py --test_city $city1 --database_city $city2
   done   
done


#python3 verifier_sigmoid_filtered.py --test_city Los_Angeles --database_city New_york 