#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import functions as f
import torch
from torch import autograd, nn
import torch.optim as optim
import matplotlib.pylab as plt
from tqdm import tqdm
import sys

for ind in range(1, len(sys.argv)):
	# ALL DATASET
	data = pd.read_csv('data_fs.dat', delim_whitespace=True, decimal=",")

	par = sys.argv[ind]

	dataset = np.stack((data["x"].values, data["y"].values, data[par]), axis=1)
	dataset = torch.from_numpy(dataset).float()

	X = dataset[:,:2]
	y = dataset[:,2]

	# Parametri che servono per la normalizzazione
	data_max = data[par].max()
	data_min = data[par].min()


	# ## PARAMETRI
	# Definisco ora tutti i parametri in modo che siano facilmente modificabili

	# In[2]:


	input_size = 2
	hidden_size1 = 3
	hidden_size2 = 3
	hidden_size3 = 3
	activation_fun = nn.Tanh()
	epochs = 10000
	loss_fun = nn.MSELoss()


	# Usando `torch.utils.data.DataLoader` il dataset iniziale viene suddiviso casualmente in modo tale che circa il 75% (126) dei dati finiscano nel training set e il 25% nel validation

	# In[3]:


	loader = torch.utils.data.DataLoader(dataset, batch_size = 126, shuffle=True)
	train, validation = loader
	X_train = train[:,:2]
	y_train = train[:,2]
	y_train = f.normalization2(y_train, data_max, data_min)

	X_val = validation[:,:2]
	y_val = validation[:,2]
	y_val = f.normalization2(y_val, data_max, data_min)


	# In[4]:


	class Net(nn.Module):
	    def __init__(self):
	        super(Net, self).__init__()
	        
	        self.hidden1 = nn.Linear(input_size, hidden_size1)
	        self.activation = activation_fun
	        self.hidden2 = nn.Linear(hidden_size1, hidden_size2)
	        self.hidden3 = nn.Linear(hidden_size2, hidden_size3)
	        self.output = nn.Linear(hidden_size3, 1)
	        
	    def forward(self, input):
	        hidden1 = self.hidden1(input)
	        activated1 = self.activation(hidden1)
	        hidden2 = self.hidden2(activated1)
	        activated2 = self.activation(hidden2)
	        hidden3 = self.hidden3(activated2)
	        activated3 = self.activation(hidden3)
	        output = self.output(activated3)
	        
	        return self.activation(output)

	net = Net()
	opt = optim.Adam(net.parameters(), lr=0.001)


	# Funzione per inizializzare i pesi della rete

	# In[5]:


	def init_weights(m):
	    if type(m) == nn.Linear:
	        m.weight.data.uniform_(0.0, 1.0)
	        m.bias.data.fill_(0.0)


	# ## Controllo bontà della rete
	# Eseguo per 5 volte il training resettando ad ogni ciclo i pesi

	# In[6]:


	net.train()
	colors = ['b', 'g', 'r', 'c', 'm']

	plt.figure(figsize=(18,10))

	for i in range(5):
	    # inizializzo i pesi
	    net.apply(init_weights)
	    training_loss = []
	    validation_loss = []
	    
	    # shuffle del training e validation set
	    loader = torch.utils.data.DataLoader(dataset, batch_size = 126, shuffle=True)
	    train, validation = loader
	    X_train = train[:,:2]
	    y_train = train[:,2]
	    y_train = f.normalization2(y_train, data_max, data_min)
	    X_val = validation[:,:2]
	    y_val = validation[:,2]
	    y_val = f.normalization2(y_val, data_max, data_min)


	    for epoch in tqdm(range(epochs)):
	        opt.zero_grad()
	        y_pred = net(X_train)
	        y_pred = torch.squeeze(y_pred)
	        train_loss = loss_fun(y_train, y_pred)
	        training_loss.append(train_loss)

	        with torch.no_grad():
	            y_test = net(X_val)
	            y_test = torch.squeeze(y_test)
	            val_loss = loss_fun(y_val, y_test)
	            assert val_loss.requires_grad == False
	            validation_loss.append(val_loss)

	        train_loss.backward()
	        opt.step()
	    
	    #plt.figure(figsize=(10,6))
	    plt.plot(validation_loss, label=f'Validation {i+1}', ls='-', color=colors[i])
	    plt.plot(training_loss, label=f'Training {i+1}', ls='dashed', color=colors[i])
	    plt.xlabel('Epoch')
	    plt.ylabel('Loss')
	    plt.title(f'Parameter: {par}')
	    #plt.ylim(0, 0.05)
	    plt.legend(fontsize=12, loc='upper right')
	    plt.savefig(f"./Plots/TestArch/Test_3L_Tanh_{par}.png")


	# ## Testo la rete utilizzando KFold

	# In[7]:


	from sklearn.model_selection import KFold

	kf = KFold(n_splits=5)    
	i = 0

	fig = plt.figure(figsize=(18,10))
	ax = fig.add_subplot(111)

	# carico il dataset e lo mescolo
	dataset = np.stack((data["x"].values, data["y"].values, data[par]), axis=1)
	np.random.shuffle(dataset)
	shuffled = torch.from_numpy(dataset).float()

	net.train()
	colors = ['b', 'g', 'r', 'c', 'm']

	X = shuffled[:,:2]
	y = shuffled[:,2]

	for train_index, val_index in kf.split(shuffled):

	    X_train, X_val = X[train_index], X[val_index]
	    y_train, y_val = y[train_index], y[val_index]
	    y_train = f.normalization2(y_train, data_max, data_min)
	    y_val = f.normalization2(y_val, data_max, data_min)

	    net.apply(init_weights)
	    training_loss = []
	    validation_loss = []

	    for epoch in tqdm(range(epochs)):
	        opt.zero_grad()
	        y_pred = net(X_train)
	        y_pred = torch.squeeze(y_pred)
	        train_loss = loss_fun(y_train, y_pred)
	        training_loss.append(train_loss)

	        with torch.no_grad():
	            y_test = net(X_val)
	            y_test = torch.squeeze(y_test)
	            val_loss = loss_fun(y_val, y_test)
	            assert val_loss.requires_grad == False
	            validation_loss.append(val_loss)

	        train_loss.backward()
	        opt.step()

	    ax.plot(validation_loss, label=f'Validation {i+1}', ls='-', color=colors[i])
	    ax.plot(training_loss, label=f'Training {i+1}', ls='dashed', color=colors[i])
	    ax.set_xlabel('Epoch')
	    ax.set_ylabel('Loss')
	    ax.legend(fontsize=12, loc='upper right')
	    fig.savefig(f"./Plots/TestArch/TestKF_3L_Tanh_{par}.png")
	    i += 1


	# In[ ]:




