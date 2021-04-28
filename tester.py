import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from definitions import Net, StressDataset, ToTensor
from sys import argv, exit
import numpy as np

def main(args):
    b = 1000
    nw = 4
    e = 12
    o = False
    hot = False
    target = "./net.pth"
    test_data="data/data/test/processed_field.npz"
    
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

    net = Net(e)
    net.load_state_dict(torch.load(target))

    test_dataset = StressDataset(data_dir=test_data, transform = transforms.Compose([ToTensor()]))
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=nw)

    lm, lsd = test(net, test_dataset, testloader, o = o)[:2]
    
    print(f"Average loss: %.2f" % (lm))
    print(f"SD of loss: %.2f" % (lsd))

def test(net, test_dataset, test_loader, o = False, device = 'cpu'):
    losses = []
    all_inputs = []
    all_labels = []
    all_outputs = []
    criterion = nn.MSELoss(reduction = 'sum')
    
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data['inputs'].to(device, dtype = torch.float32), data['labels'].to(device, dtype = torch.float32)
            outputs = net(inputs)

            losses.append(criterion(np.squeeze(outputs), np.squeeze(labels)).item())

            if o:
                all_inputs.append(np.squeeze(inputs.cpu().detach().numpy()))
                all_labels.append(np.squeeze(labels.cpu().detach().numpy()))
                all_outputs.append(np.squeeze(outputs.cpu().detach().numpy()))

    losses = np.array(losses)
    lm, lsd = losses.mean(), losses.std()

    if o:
        loss_percs = np.percentile(np.round(losses,2), [0,10,50,90,100])
        print(f"0th \t% loss: {loss_percs[0]}")
        print(f"10th \t% loss: {loss_percs[1]}")
        print(f"50th \t% loss: {loss_percs[2]}")
        print(f"90th \t% loss: {loss_percs[3]}")
        print(f"100th \t% loss: {loss_percs[4]}")
        print()

        idx_percs = np.array([(np.abs(losses - perc)).argmin() for perc in loss_percs])

        return (lm, lsd, np.array(all_inputs)[idx_percs], np.array(all_labels)[idx_percs], np.array(all_outputs)[idx_percs])
    else:
        return (lm, lsd)


if __name__ == '__main__':
    main(argv[1:])
