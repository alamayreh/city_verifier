#!/bin/bash
 
#export CUDA_VISIBLE_DEVICES=4
# Declare an array of city list
#declare -a city_list=("Moscow" "London" "Shanghai" "Cairo" "Delhi" "New_york" "Rio_de_Janeiro" "Sydney" "Roma" "Tokyo")
#declare -a city_list=("Moscow" "St_Petersburg" "London" "Edinburgh" "Shanghai" "Beijing" "Cairo" "Delhi" "New_york" "Los_Angeles" "Rio_de_Janeiro" "Sydney" "Roma" "Milan" "Tokyo")
declare -a city_list=("Cairo" "Delhi" "Rio_de_Janeiro" "Sydney" "Tokyo")
#declare -a city_list=("Moscow" "St_Petersburg" "London" "Edinburgh" "Shanghai" "Beijing" "Cairo" "Delhi" "New_york" "Rio_de_Janeiro" "Berlin" "Munich" "Roma" "Milan" "Tokyo")



#"Cairo" "Delhi"  "Rio_de_Janeiro" "Sydney" "Roma" "Milan" "Tokyo"


#declare -a city_list=("Cairo" "Delhi" "New_york"  "Rio_de_Janeiro" "Tokyo")

#declare -a city_list=("Rio_de_Janeiro")
#declare -a city_list=("Amman" "Istanbul" "Mexico_city" "Paris" "Singapore" "Quebec" "Vancouver" )
#declare -a city_list=("Cairo" "Delhi" "Rio_de_Janeiro" "Sydney" "Tokyo")

# Iterate the string array using for loop
#for city in ${city_list[@]}; do
   #python3 analysis_results.py --test_city Tokyo --database_city 
#   python3 analysis_filtered_vipp_similarity.py --test_city $city --database_city $city
#   python3 analysis_filtered_vipp.py --test_city  $city --database_city Vancouver
   #sleep 1m
#done

#declare -a city_list=("Moscow" "St_Petersburg")
#declare -a city_list=("London" "Edinburgh")
#declare -a city_list=("Shanghai" "Beijing")
declare -a city_list=("New_york" "Los_Angeles")
#declare -a city_list=("Roma" "Milan")


#declare -a city_list=("Amman" "Istanbul" "Mexico_city" "Paris" "Singapore")

for city1 in ${city_list[@]}; do
   for city2 in ${city_list[@]}; do
      #python3 analysis_filtered_vipp_similarity.py --test_city $city1 --database_city $city2
      python3 analysis_filtered_vipp.py --test_city $city1 --database_city $city2
   done
   echo "-----------------------------------------------------------------------------------------"
done
