import pandas as pd
import matplotlib.pyplot as plt
import os
from sys import argv, exit

def main(args):
    title = ""
    csv = "./loss.csv"
    output = "/graph.png"
    trim = 0
    for arg in args:#Takes in command line arguments; less sugar but more rigorous
        if arg[0] == ".":
            csv = arg
        elif arg[:4] == "trim":
            try: trim = int(arg[4:])
            except: pass
        elif arg[0] == "t":
            title = arg[1:]
        elif arg[0] == "o":
            output = arg[2:]
        else:
            print(f"Argument '%s' ignored" % str(arg))

    print(f"CSV file: {csv}")
    filename = (csv[:csv.rindex("/")] + output)
    print(f"Output file: {filename}")
    if trim != 0: print(f"Trimming to: {trim}")
    if title != "": print(f"Title: {title}")

    graph(csv, output, trim)

def graph(csv, output = "/graph.png", trim = 0):
    if not os.path.exists(csv):
        print("CSV file %s not found. Must enter whole path name" % csv)
        exit()

    target = csv[:csv.rindex("/")] + output
    print("Outputting graph to %s" % target)

    loss = pd.read_csv(csv)

    fig, ax = plt.subplots()

    ax = loss[trim:][['loss','acc','test_acc']].plot(secondary_y=['loss'])

    plt.savefig(target)

if __name__ == '__main__':
    main(argv[1:])
