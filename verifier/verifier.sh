#!/bin/bash
 
export CUDA_VISIBLE_DEVICES=2
# Declare an array of city list
#declare -a city_list=("Moscow" "St_Petersburg" "London" "Edinburgh" "Shanghai" "Beijing" "Cairo" "Delhi" "New_york" "Los_Angeles" "Rio_de_Janeiro" "Sydney" "Roma" "Milan" "Tokyo")

#declare -a city_list=("Moscow" "St_Petersburg" "London" "Edinburgh" "Shanghai" "Beijing" "Cairo" "Delhi" "New_york" "Rio_de_Janeiro" "Berlin" "Munich" "Roma" "Milan" "Tokyo")


#declare -a city_list=("Amman" "Istanbul" "Mexico_city" "Paris" "Singapore")
# Iterate the string array using for loop
#for city in ${city_list[@]}; do
#   python3 verifier_sigmoid_filtered.py --test_city Singapore --database_city $city
#done


#declare -a city_list=("Amman" "Istanbul" "Mexico_city" "Paris" "Singapore")
#declare -a city_list=("Cairo" "Delhi" "Rio_de_Janeiro" "Sydney" "Tokyo")

#declare -a city_list=("Moscow" "St_Petersburg")
#declare -a city_list=("London" "Edinburgh")
#declare -a city_list=("Shanghai" "Beijing")
#declare -a city_list=("New_york" "Los_Angeles")
#declare -a city_list=("Roma" "Milan")
declare -a city_list=("Cairo" "Delhi" "Rio_de_Janeiro" "Sydney" "Tokyo")

for city1 in ${city_list[@]}; do
   for city2 in ${city_list[@]}; do
      python3 verifier_sigmoid_filtered.py --test_city $city1 --database_city $city2
   done   
done


