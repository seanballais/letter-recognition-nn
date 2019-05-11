shopt -s nullglob

data_files=( test_results/ga_data/1---*.csv )
output_folder=test_results

for file in "${data_files[@]}"
do
    echo 'Plotting data file:' $file
    python3 plot_ga_average_data.py $file test_results FF7482
done
