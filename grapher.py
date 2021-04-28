import pandas as pd
import matplotlib.pyplot as plt
import os
from sys import argv, exit

def main(args):
    title = ""
    xlab = ""
    ylab = ""
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
        elif arg[0] == "x":
            xlab = arg[1:]
        elif arg[0] == "y":
            ylab = arg[1:]
        elif arg[0] == "o":
            output = arg[2:]
        else:
            print(f"Argument '%s' ignored" % str(arg))

    print(f"CSV file: {csv}")
    filename = (csv[:csv.rindex("/")] + output)
    print(f"Output file: {filename}")
    if trim: print(f"Trimming to: {trim}")
    if title: print(f"Title: {title}")
    if xlab: print(f"x-label: {xlab}")
    if ylab: print(f"y-label: {ylab}")

    graph(csv, output, trim, title, xlab, ylab)

def graph(csv, output = "/graph.png", trim = 0, title = "", xlab = "", ylab = ""):
    if not os.path.exists(csv):
        print("CSV file %s not found. Must enter whole path name" % csv)
        exit()

    target = csv[:csv.rindex("/")] + output
    print("Outputting graph to %s" % target)

    loss = pd.read_csv(csv)[['loss','test_acc']]

    if trim:
        loss = loss[trim:]

    fig, ax = plt.subplots()

    loss.columns = ['loss','test loss']
    ax = loss[['loss', 'test loss']].plot()
    if title:
        plt.title(title)
    if xlab:
        plt.xlabel(xlab)
    if ylab:
        plt.ylabel(ylab)

    plt.savefig(target)

if __name__ == '__main__':
    main(argv[1:])
