

# need to add -o for where you want to output the file

for (( i=11; i<=11; i+=1))
do
    neper -M -loadmesh /tri-s-1/s2/dcp5303/projects/2021_03_16_GNN_Paper/test_$i/simulation.msh -loadpoint grid.txt -statpoint elt -o data/meshes
    python remap_grid_data.py $i
done
