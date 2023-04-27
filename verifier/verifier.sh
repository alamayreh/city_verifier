#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
# Declare an array of city list
#declare -a city_list=("Moscow" "St_Petersburg" "London" "Edinburgh" "Shanghai" "Beijing" "Cairo" "Delhi" "New_york" "Los_Angeles" "Rio_de_Janeiro" "Sydney" "Roma" "Milan" "Tokyo")

#declare -a city_list=("Moscow" "St_Petersburg" "London" "Edinburgh" "Shanghai" "Beijing" "Cairo" "Delhi" "New_york" "Rio_de_Janeiro" "Berlin" "Munich" "Roma" "Milan" "Tokyo")




# Iterate the string array using for loop
# Closed 
declare -a city_list=("Amsterdam" "Barcelona" "Berlin" "London" "NewYork" "LosAngeles" "Rome" "Milan" "Paris" "Tokyo")
declare -a city_list=("Amman" "Istanbul" "Mexico_city" "Singapore" "Quebec" "Vancouver" "Venice" "Florence" "Rome" "Moscow" "St_Petersburg" "Shanghai" "Beijing" "London" "Edinburgh")

for city in ${city_list[@]}; do
   python3 verifier_vit.py --test_city $city  --database_city $city
done



#for city1 in ${city_list[@]}; do
#   for city2 in ${city_list[@]}; do
#      python3 verifier_sigmoid_filtered.py --test_city $city1 --database_city $city2
#   done   
#done

#python3 verifier_sigmoid_filtered.py --test_city Los_Angeles --database_city Los_Angeles



for city1 in ${city_list[@]}; do
   for city2 in ${city_list[@]}; do
      python3 verifier_temp3.py --test_city $city1 --database_city $city2
   done   
done

