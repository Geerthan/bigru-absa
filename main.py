"""
Trains and tests the AbsaGRU model on one subject/aspect pair of Sentihood.

@author: Geerthan
"""

import torch
import torchtext
import time

import numpy as np

import ABSAData
import ABSAModel

import tqdm

import pandas as pd

import sys

if len(sys.argv) != 3:
    print("Incorrect args. Should be [python main.py subject aspect]")

subj = sys.argv[1]
asp = sys.argv[2]

print(f'Subject {subj}, Aspect {asp}')

batch_size = 80
embedding_dim = 300
hidden_units = 50
output_dim = 3
epochs = 20

device = torch.device("cuda")

print("Device:", device)

print("Downloading tokenizer")

tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

print("Tokenizer downloaded, loading afinn")

afinn = pd.read_csv('data/afinn/AFINN-111.txt', header=None, sep='\t', names=['Words', 'Score'])
afinn_dict = dict([(a,b) for a,b in zip(afinn.Words, afinn.Score)])

print("Afinn loaded, loading train_data")
train_data, train_weights, init_embed, vocab = ABSAData.get_train(subj, asp, tokenizer, afinn_dict)

print("Loading dev_data")
dev_data = ABSAData.get_other('dev', subj, asp, tokenizer, vocab, afinn_dict)

print("Data loaded\n")

train_sampler = torch.utils.data.WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)
train_dl = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

dev_dl = torch.utils.data.DataLoader(dev_data, batch_size=batch_size, shuffle=False)

model = ABSAModel.AbsaGRU(len(vocab), embedding_dim, hidden_units, batch_size, output_dim, device, init_embed)
model.to(device)

loss = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)

def loss_fn(pred, y):
    return loss(pred, y)
    
def acc_fn(pred, y):
    corr = (torch.max(pred, 1)[1].data == y).sum()
    return 100 * corr / len(pred)

best_loss = 999

# Training loop
for epoch in tqdm.trange(epochs, desc="Epoch"):
    start = time.time()
    train_loss = 0
    train_acc = 0
    train_itrs = 0
    
    val_loss = 0
    val_acc = 0
    val_itrs = 0
    
    for itr, batch in enumerate(tqdm.tqdm(train_dl, "Train Iteration")):
        temp_loss = 0
        batch = tuple(t.to(device) for t in batch)
        ids, lbls, subs, sents = batch
        
        pred, _ = model(ids, lbls, subs, sents)
        temp_loss += loss_fn(pred, lbls.to(device))
        batch_loss = temp_loss / pred.shape[1]
        train_loss += batch_loss
        
        optim.zero_grad()
        temp_loss.backward()
        optim.step()
        
        batch_acc = acc_fn(pred, lbls.to(device))
        train_acc += batch_acc
        
        train_itrs+=1
    
    # Evaluating on dev(val) set
    model.eval()
    for itr, batch in enumerate(tqdm.tqdm(dev_dl, "Dev Iteration")):
        temp_loss = 0
        batch = tuple(t.to(device) for t in batch)
        ids, lbls, subs, sents = batch
        
        with torch.no_grad():
            pred, _ = model(ids, lbls, subs, sents)
            temp_loss += loss_fn(pred, lbls.to(device))
            batch_loss = temp_loss / pred.shape[1]
            val_loss += batch_loss
        
        batch_acc = acc_fn(pred, lbls.to(device))
        val_acc += batch_acc
        
        val_itrs+=1
        

    model.train()
        
    train_loss /= train_itrs
    train_acc /= train_itrs
    
    val_loss /= val_itrs
    val_acc /= val_itrs
    
    print(f"\n\nEpoch {epoch+1} Loss: {train_loss:.2f}, Acc: {train_acc:.2f} | Val loss: {val_loss:.2f}, Val acc: {val_acc:.2f}")
    
    if val_loss < best_loss:
        best_loss = val_loss
        print("Saving new model\n")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss}, 'models/best.pth')


print("Loading test_data")
test_data = ABSAData.get_other('test', subj, asp, tokenizer, vocab, afinn_dict)
test_dl = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)


model.eval()

test_loss = 0
test_acc = 0
itrs = 0

cp = torch.load('models/best.pth')
epoch = cp['epoch']
model.load_state_dict(cp['model_state_dict'])
optim.load_state_dict(cp['optimizer_state_dict'])
loss = cp['loss']

# Writing test results to file
file = open("results/sentihood_" + subj + "_" + asp + ".txt", "w")
for itr, batch in enumerate(tqdm.tqdm(test_dl, "Iteration")):
    temp_loss = 0
    batch = tuple(t.to(device) for t in batch)
    ids, lbls, subs, sents = batch
    
    with torch.no_grad():
        pred, _ = model(ids, lbls, subs, sents)
    temp_loss += loss_fn(pred, lbls.to(device))
    batch_loss = temp_loss / pred.shape[1]
    test_loss += batch_loss
    
    batch_acc = acc_fn(pred, lbls.to(device))
    test_acc += batch_acc
    
    pred = torch.nn.functional.softmax(pred, dim=1).detach().cpu().numpy()
    predIDs = np.argmax(pred, axis=1)
    
    for i in range(len(predIDs)):
        file.write(str(predIDs[i]))
        for p in pred[i]:
            file.write(" " + str(p))
        file.write("\n")
    
    itrs += 1

file.close()

print(f"Test Loss: {test_loss/itrs:.2f}, Acc: {test_acc/itrs:.2f}\n")
