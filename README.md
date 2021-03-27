# MATSC-ConvNN
Generalization of the CountLoops project. Our goal is to create a ConvNN that uses low-length simulations of metals under an applied load to understand their micromechanical response.

### trainer

Train neural network 

Args:
*    epoch = 100:        [0-9]+
*    batch size = 1000:  b[0-9]+
*    one-hot = FALSE:    hot | [^(hot)]
*    num_workers = 4:    nw[0-9]+
*    output = False:     o[+]? | [^o]
*    learn rate = 0.01:    lr0\.[0-9]+
*    model = "./loops_counter_net.pth": \./.+\.pth

### tester

Test neural network

Args:
*    batch size = 1000:  b[0-9]+
*    one-hot = FALSE:    hot | [^(hot)]
*    num_workers = 4:    nw[0-9]+
*    output = False:     o | [^o]
*    net size = 12:      [0-9]+
*    model = "./loops_counter_net.pth": \./.+\.pth

### data/download
Download images to image folder using data in csv file

To Call:
python ./data/download.py

Args:
*    test = False:       t | [^t]
*    display = False:    d | [^d]
*    append = False:     + | [^+]

### data/generator
Generate a given number of dummy images
Generate corresponding csv file with number of loops

To Call:
python ./data/generator.py

Args:
*    n = 100:            [0-9]+
*    test = False:       t | [^t]
*    display = False:    d | [^d]
*    append = False:     + | [^+]

### definitions
