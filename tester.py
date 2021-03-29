import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from definitions import Net, LoopsDataset, ToTensor
from sys import argv, exit
import numpy as np
import pandas as pd

def main(args):
    b = 1000
    nw = 4
    e = 12
    o = False
    hot = False
    target = "./net.pth"
    test_data="data/test_dat.csv"
    test_dir='data/test_inputs'

    for arg in args:#Takes in command line arguments; less sugar but more rigorous
        try: arg = int(arg)
        except: pass
        if type(arg) == int:
            e = arg
        elif arg[0] == "o":
            o = True
        elif arg[0] == "b":
            try: b = int(arg[1:])
            except: pass
        elif arg[:2] == "nw":
            try: nw = int(arg[2:])
            except: pass
        elif arg[0] == ".":
            target = arg
        elif arg == "hot":
            hot = True
        else:
            print(f"Argument '%s' ignored" % str(arg))

    if b != 1000:
        print(f"Batch Size: %d" % b)
    print(f"Output Loss: %r" % o)
    if hot:
        print("One-Hot encoding: True")
    print(f"Model file: %s\n" % target)

    #print("Loading Network")
    net = Net(e)
    net.load_state_dict(torch.load(target))
    #print("Network Loaded\n")

    #print("Loading Dataset")
    test_dataset = LoopsDataset(csv_file=test_data, root_dir=test_dir, transform = transforms.Compose([ToTensor()]))
    testloader = DataLoader(test_dataset, batch_size=b, shuffle=False, num_workers=nw)
    if o:
        correct, total, all_predictions, all_loops, all_scores = test(net, test_dataset, test_loader, o, b, nw)
        open(f"./%s_guesses.csv" % target, 'w')
        guess_output = pd.DataFrame(all_loops, columns = ["loops"]).assign(guess = all_predictions)
        scores_output = pd.DataFrame(all_scores, columns = [f"s%d" % i for i in range(e)])
        guess_output.join(scores_output).to_csv(f"./%s_guesses.csv" % target[:-4], index = False)
    else:
        correct, total, all_predictions = test(net, test_dataset, test_loader, b = b, nw = nw)


    print('Accuracy of the network on the %d test inputs: %.2f%%' % (len(pd.read_csv(test_data)), 100 * correct / total))
    print(f"Average guess: %.2f" % (np.array(all_predictions).mean()))
    print(f"SD of guesses: %.2f" % (np.array(all_predictions).var()**.5))

def test(net, test_dataset, test_loader, o = False, device = 'cpu'):
    #print("Dataset Loaded\n")

    #print("Testing Network")
    correct = 0
    total = 0
    all_predictions, all_loops, all_scores = [],[],[]

    with torch.no_grad():
        for data in test_loader:
            inputs, loops, text = data['inputs'].to(device), data['labels'], data['text']
            outputs = net(inputs)
            predicted = outputs.data#add the _, if one-hot-encoded

            if 'cuda' in  device:
                scores = np.array([i for i in predicted])
            else:
                scores = np.array(predicted)
            predicted = np.array([max([(v,i) for i,v in enumerate(predicted[j])])[1] for j in range(len(predicted))])

            total += loops.size(0)

            loops = np.array(loops)

            correct += (predicted == loops).sum()


            all_predictions += list(predicted)
            if o:
                all_loops += list(loops)
                all_scores += list(scores)

    if o:
        return (correct, total, all_predictions, all_loops, all_scores)
    else:
        return (correct, total, all_predictions)


if __name__ == '__main__':
    main(argv[1:])
