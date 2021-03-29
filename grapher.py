import pandas as pd
import matplotlib.pyplot as plt
import os
from sys import argv, exit

def main(args):
    title = ""
    csv = "./loss.csv"
    for arg in args:#Takes in command line arguments; less sugar but more rigorous
        if arg[0] == ".":
            csv = arg
        if arg[0] == "t":
            title = arg[1:]
        else:
            print(f"Argument '%s' ignored" % str(arg))


    print("CSV file: %s" % csv)

    graph(csv)

def graph(csv):
    if not os.path.exists(csv):
        print("CSV file %s not found. Must enter whole path name" % csv)
        exit()

    target = csv[:csv.rindex("/")] + "/graph.png"
    print("Outputting graph to %s" % target)

    loss = pd.read_csv(csv)

    fig, ax = plt.subplots()

    ax = loss[['loss','acc','test_acc']].plot(secondary_y=['loss'])

    plt.savefig(target)

if __name__ == '__main__':
    main(argv[1:])
