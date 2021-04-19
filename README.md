# MATSC-ConvNN
Generalization of the CountLoops project. Our goal is to create a ConvNN that uses low-length simulations of metals under an applied load to understand their micromechanical response.

### trainer

Train neural network

Args:
arg name  | default val | regex
--- | --- | ---
epoch      | 100       | [0-9]+
batch size | 1000      | b[0-9]+
num_workers| 4         | nw[0-9]+
output     | False     | o[+]? \| [^o]
learn rate | 0.01      | lr0\.[0-9]+
model      | "./loops_counter_net.pth" | \\./.\+\\.pth
iterations | False     | i[0-9]\* \\./.+

* Iterations
    * Number is how many epochs pass between each save
    * Folder is the location of the save

### tester
Test neural network

Args:
arg name  | default val | regex
--- | --- | ---
batch size   | 1000  |  b[0-9]+
num_workers  | 4     |  nw[0-9]+
output       | False |  o | [^o]
model        | "./loops_counter_net.pth" | \./.+\.pth

### grapher
Generates a graph for a given csv file

Args:
arg name  | default val | regex
--- | --- | ---
csv    | "./loss.csv" | \\./.\*\\.csv
trim   | 0            | trim[0-9]+
output | "/graph.png" | \\./.\*\\.png

### definitions
Stores definitions of network, dataset, and transforms
