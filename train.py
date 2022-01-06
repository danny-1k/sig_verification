import torch
import torch.nn as nn 
from torch.utils.data import DataLoader

from data import SignatureDataset

from model import Network

import matplotlib.pyplot as plt

from tqdm import tqdm

train = SignatureDataset(True)
test = SignatureDataset(False)

len_test = len(test)

train = DataLoader(train,batch_size=32,shuffle=True)
test = DataLoader(test,batch_size=32,shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = Network()
net.to(device)

lossfn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=3e-4)

train_loss_over_time = []
test_loss_over_time = []
accuracy_over_time = []

last_best_accuracy = 0


for epoch in tqdm(range(100)):
    net.train()

    batch_train_loss = []
    batch_test_loss = []

    for x1,x2,y in train:
        optimizer.zero_grad()
        p = net(x1,x2)
        loss = lossfn(p,y)
        loss.backward()
        optimizer.step()
        batch_train_loss.append(loss.item())

    net.eval()
    
    with torch.no_grad():
        acc = 0
        for x1,x2,y in test:
            p = net(x1,x2)
            loss = lossfn(p,y)
            batch_test_loss.append(loss.item())
            acc += sum(p.argmax(dim=1) == y) 


        acc = acc/len_test

    train_loss_over_time.append(sum(batch_train_loss)/len(batch_train_loss))
    test_loss_over_time.append(sum(batch_test_loss)/len(batch_test_loss))
    accuracy_over_time.append(acc)

    plt.plot(train_loss_over_time,label="train loss")
    plt.plot(test_loss_over_time,label="test loss")

    plt.legend()

    plt.savefig('plots/loss.png')

    plt.close('all')

    plt.plot(accuracy_over_time,label='test accuracy')
    plt.legend()


    plt.savefig('plots/accuracy.png')

    plt.close('all')

    if acc > last_best_accuracy:
        net.save_()
        last_best_accuracy = acc