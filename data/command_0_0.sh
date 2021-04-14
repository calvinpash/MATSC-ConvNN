

# need to add -o for where you want to output the file

for (( i=3; i<=10; i+=1))
do
    python remap_grid_data_0_0.py $i
done
