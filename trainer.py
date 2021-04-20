import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from definitions import Net, StressDataset, ToTensor
from tester import test
from grapher import graph
from torchvision import transforms, utils
import numpy as np
from sys import argv, exit
import matplotlib.pyplot as plt
import os
from math import isnan
#Create Dataset

def make_folder(addy):
    #Make file structure if missing
    #print("Makefolder")
    split_addy = addy.split("/")
    for i in range(len(split_addy)-int("." in split_addy[-1])):
        if not os.path.exists("/".join(split_addy[:i+1])):
            os.mkdir("/".join(split_addy[:i+1]))
            print("/".join(split_addy[:i+1]))

def main(args):
    from os import listdir
    e = 100
    b = 1000
    nw = 4
    append = False
    o = False
    o_append = False
    hot = False
    start_target = "./output/net.pth"
    target = "/"
    data_file='data/data/processed/processed_field.npz'
    test_data_file = "data/data/test/processed_field.npz"
    model_num = -1
    incr_size = 25
    lr = 0.01
    for arg in args:#Takes in command line arguments; less sugar but more rigorous
        try: arg = int(arg)
        except: pass
        if type(arg) == int:
            e = arg
        elif arg[0] == "+":
            append = True
            if "." in arg:
                start_target = arg[arg.index(".")+2:]
        elif arg[0] == "o":
            o = True
            if len(arg) > 1 and arg[1] == "+":
                o_append = True
        elif arg == "hot":
            hot = True
        elif arg[0] == "b":
            try: b = int(arg[1:])
            except: pass
        elif arg[:2] == "nw":
            try: nw = int(arg[2:])
            except: pass
        elif arg[0] == ".":
            target = arg[1:]
        elif arg[0] == "i":
            model_num = 0
            if len(arg) > 1:
                for i in range(4):
                    try: incr_size = int(arg[1:1+i])
                    except: continue
            if "." in arg:
                target = arg[arg.index(".")+2:]
        elif len(arg) > 1 and arg[:2] == "lr":
             try: lr = float(arg[3:])
             except: pass
        else:
            print(f"Argument '%s' ignored" % str(arg))

    if append and not os.path.exists(start_target):
        append = False
        print("Couldn't find target starting model")

    if target[-1] == "/":
        target = target[:-1]
    if not os.path.exists(target):
        make_folder(f"./output/%s" % target)
        make_folder(f"./output/{target}/preds")

    if append and model_num >= 0:
        model_num = len(os.listdir(f"./output/{target}"))



    print(f"Epoch Count: %d" % e)
    print(f"Batch Size: %d" % b)
    print(f"Output Loss: %r" % o)
    print(f"Output model file: output/%s/net.pth" % target)
    if append:
        print(f"Starting with model: %r" % start_target)

    stress_dataset = StressDataset(data_dir=data_file, transform = transforms.Compose([ToTensor()]))
    b = len(stress_dataset) if b > len(stress_dataset) else b
    dataloader = DataLoader(stress_dataset, batch_size = b, num_workers = nw)

    if model_num >= 0:
        make_folder(f"./output/%s/models" % target)
        print(f"\nIncremental Model saving ON")
        print(f"Saving to: ./output/%s/models" % target)
        print(f"Every %d epochs\n" % incr_size)
        test_dataset = StressDataset(data_dir=data_file, transform = transforms.Compose([ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=nw)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = "cpu"

    print(f"Using device: %s" % device)
    net = Net()
    if append:
        net.load_state_dict(torch.load(start_target))

    criterion = nn.MSELoss(reduction = 'sum')#weight = weights)

    net.to(device)
    print("Loading Data")

    if model_num >= 0 and incr_size > e:
        print("Incremental save size greater than epoch.")
        exit()

    optimizer = optim.SGD(net.parameters(), lr=lr)#,momentum = 0.9)
    #scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.99**epoch)
    print("Loaded\n")

    if o:
        settings = f"e%db%dnw%d" % (e, b, nw)
        loss_file = f"./output/%s/loss.csv" % (target)

        make_folder(loss_file)

        loss_output = []#epoch, batch, loss
        if not os.path.exists(loss_file):
            o_append = False
            out_file = open(loss_file,'w+')
            out_file.write("epoch,loss,acc,test_acc,loss_sd,acc_sd\n")

        elif not o_append:
            print("These settings will overwrite an existing loss output file.")

            out_file = open(loss_file, 'w')
            out_file.write("epoch,loss,acc,test_acc,loss_sd,acc_sd\n")

        print(f"Outputting loss to: %s\n" % loss_file)

    print("%d Batches per Epoch" % np.floor(len(stress_dataset)/b))
    test_acc = []

    loss_threshold = 0.0005
    min_loss = 2**32
    curr_lr = lr

    for epoch in range(e):  # loop over the dataset multiple times
        print(f"Current Epoch: %d" % epoch)
        running_loss = 0.0
        losses = []

        for i, data in enumerate(dataloader, 0):
            inputs, labels = data['inputs'].to(device, dtype = torch.float32), data['labels'].to(device, dtype = torch.float32)

            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs,labels)

            losses.append(loss.sum().item()/b)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()


        if model_num >= 0 and epoch % incr_size == 0:
            print(f"\nSaving model {model_num}\n")
            torch.save(net.state_dict(), f"./output/%s/models/%d.pth" % (target, model_num))

            if o:
                test_lm, test_sd, all_inputs, all_labels, all_outputs = test(net, test_dataset, test_loader, device = device, o = o)
                make_folder(f"./output/{target}/preds/e{epoch}")
                for i in range(len(all_labels)):
                    plt.imshow(all_labels[i]).write_png(f"./output/{target}/preds/e{epoch}/stress{model_num}_{i}.png")
                    plt.imshow(all_outputs[i]).write_png(f"./output/{target}/preds/e{epoch}/pred{model_num}_{i}.png")
                    plt.imshow(np.moveaxis(all_inputs[i],0,2)+.5).write_png(f"./output/{target}/preds/e{epoch}/ori{model_num}_{i}.png")
                
            else:
                test_lm, test_sd = test(net, test_dataset, test_loader, device = device, o = o)
            print(f"Test loss Mean: {test_lm}")
            print(f"Test loss SD  : {test_sd}")
            print()

            model_num += 1

            test_acc.append(test_lm)
        elif model_num >= 0:
            test_acc.append(test_acc[-1])
        else:
            test_acc.append(0)


        print("\tMean\tSD")
        lm, lsd = (np.array(losses).mean(), np.array(losses).var()**.5)
        am, asd = (0,0)
        print(f"Loss\t%.4f\t%.4f" % (lm, lsd))

        if o:
            out = [str(i) for i in [epoch, lm, am, test_acc[-1], lsd, asd]]
            out_file.write(",".join(out) + "\n")

        if lm < min_loss and lm > min_loss*(1-loss_threshold):
            min_loss = lm
            print(f"Curr_lr: {curr_lr}")

            #if epoch < 200: curr_lr *= 1.5
            curr_lr *= .95

            print(f"Updated to {curr_lr}")

            optimizer = optim.SGD(net.parameters(), lr = curr_lr)

    print('Finished Training')

    torch.save(net.state_dict(), "output/%s/net.pth" % target)
    in_def = open("definitions.py").read()
    out_def = open("output/%s/definitions.py" % target,'w+')
    out_def.write(in_def)
    out_def.close()

    if o:
        out_file.close()
        graph(loss_file)

if __name__ == '__main__':
    main(argv[1:])
