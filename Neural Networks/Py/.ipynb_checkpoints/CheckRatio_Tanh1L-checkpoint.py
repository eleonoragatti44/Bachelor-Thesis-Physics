import numpy as np
import pandas as pd
import functions as f
import torch
from torch import autograd, nn
import torch.optim as optim
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from tqdm import tqdm
import sys

# FACCIO TUTTO PER FWHM_X
data = pd.read_csv('data_fs.dat', delim_whitespace=True, decimal=",")


dataset = np.stack((data["x"].values, data["y"].values, data["fwhm_x"], data["fwhm_y"], data["e"]), axis=1)
dataset = torch.from_numpy(dataset).float()
data_max = data["fwhm_x"].max()
data_min = data["fwhm_x"].min()

input_size = 2
hidden_size = 7
activation_fun = nn.Tanh()
epochs = 30000
loss_fun = nn.MSELoss()

loader = torch.utils.data.DataLoader(dataset, batch_size = 126, shuffle=True)
train, validation = loader
X_train = train[:,:2]
y_train = train[:,3]
y_train = f.normalization2(y_train, data_max, data_min)

X_val = validation[:,:2]
y_val = validation[:,3]
y_val = f.normalization2(y_val, data_max, data_min)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.activation = activation_fun
        self.output = nn.Linear(hidden_size, 1)
    def forward(self, input):
        hidden = self.hidden(input)
        activated = self.activation(hidden)
        output = self.output(activated)       
        return self.activation(output)
net = Net()
opt = optim.Adam(net.parameters(), lr=0.001)

def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0.0)

net.eval()
net.apply(init_weights)

y_pred = net(X_val)
y_pred = torch.squeeze(y_pred)
before_train = loss_fun(y_pred, y_val)
print('Test loss before Training' , before_train.item())

net.train()

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
    
    # shuffle del training e validation set ogni 5000 epoche
    if epoch % 5000 == 0:
        loader = torch.utils.data.DataLoader(dataset, batch_size = 126, shuffle=True)
        train, validation = loader
        X_train = train[:,:2]
        y_train = train[:,3]
        y_train = f.normalization2(y_train, data_max, data_min)
        X_val = validation[:,:2]
        y_val = validation[:,3]
        y_val = f.normalization2(y_val, data_max, data_min)       
net.eval()

y_pred = net(X_val)
y_pred = torch.squeeze(y_pred)
after_train = loss_fun(y_pred, y_val)
print('Test loss after Training' , after_train.item())
if before_train < after_train:
    print('OVERFITTING')
else:
    print('Everything is ok')

#####   DECIDO QUI IL dataset SU CUI VALUTARE L'ERRORE IN ENTRAMBI I CASI
loader_def = torch.utils.data.DataLoader(dataset, batch_size = 126, shuffle=True)
train_def, validation_def = loader_def
X_train_def = train_def[:,:2]
y_train_def1 = train_def[:,2]
y_train_def1 = f.normalization2(y_train_def1, data_max, data_min)
X_val_def = validation_def[:,:2]
y_val_def1 = validation_def[:,2]
y_val_def1 = f.normalization2(y_val_def1, data_max, data_min)

predicted_fwhmx = torch.squeeze( net(X_val_def).float() )
true_fwhmx = torch.squeeze( y_val_def1.float() )

# inverto la normalizzazione in modo da tornare al dominio di partenza
predicted_fwhmx = f.inverse_normalization2(predicted_fwhmx, data_max, data_min)
true_fwhmx = f.inverse_normalization2(true_fwhmx, data_max, data_min)

t = [np.array(X_train_def[:,0]), np.array(X_train_def[:,1])]
v = [np.array(X_val_def[:,0]), np.array(X_val_def[:,1])]
popt, pcov = curve_fit( f.func, t, f.inverse_normalization2(np.array(y_train_def1), data_max, data_min ))
function = f.func(v, *popt)

net_fwhmx = predicted_fwhmx.detach().numpy()
int_fwhmx = f.inverse_normalization2(np.array(y_val_def1), data_max, data_min)

################################################################################
################################################################################
################################################################################
################################################################################

# FACCIO LA STESSA COSA PER FWHM_Y RICORDANDOMI CHE HO GIÀ IL DATASET SU CUI CALCOLARE L'ERRORE

data_max = data["fwhm_y"].max()
data_min = data["fwhm_y"].min()

loader = torch.utils.data.DataLoader(dataset, batch_size = 126, shuffle=True)
train, validation = loader
X_train = train[:,:2]
y_train = train[:,3]
y_train = f.normalization2(y_train, data_max, data_min)

X_val = validation[:,:2]
y_val = validation[:,3]
y_val = f.normalization2(y_val, data_max, data_min)

net.eval()
net.apply(init_weights)

y_pred = net(X_val)
y_pred = torch.squeeze(y_pred)
before_train = loss_fun(y_pred, y_val)
print('Test loss before Training' , before_train.item())

net.train()

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
    
    # shuffle del training e validation set ogni 5000 epoche
    if epoch % 5000 == 0:
        loader = torch.utils.data.DataLoader(dataset, batch_size = 126, shuffle=True)
        train, validation = loader
        X_train = train[:,:2]
        y_train = train[:,3]
        y_train = f.normalization2(y_train, data_max, data_min)
        X_val = validation[:,:2]
        y_val = validation[:,3]
        y_val = f.normalization2(y_val, data_max, data_min)       
net.eval()

y_pred = net(X_val)
y_pred = torch.squeeze(y_pred)
after_train = loss_fun(y_pred, y_val)
print('Test loss after Training' , after_train.item())
if before_train < after_train:
    print('OVERFITTING')
else:
    print('Everything is ok')


########################################################################
X_train_def = train_def[:,:2]
y_train_def2 = train_def[:,3]
y_train_def2 = f.normalization2(y_train_def2, data_max, data_min)
X_val_def = validation_def[:,:2]
y_val_def2 = validation_def[:,3]
y_val_def2 = f.normalization2(y_val_def2, data_max, data_min)

predicted_fwhmy = torch.squeeze( net(X_val_def).float() )
true_fwhmy = torch.squeeze( y_val_def2.float() )

# inverto la normalizzazione in modo da tornare al dominio di partenza
predicted_fwhmy = f.inverse_normalization2(predicted_fwhmy, data_max, data_min)
true_fwhmy = f.inverse_normalization2(true_fwhmy, data_max, data_min)

t1 = [np.array(X_train_def[:,0]), np.array(X_train_def[:,1])]
v1 = [np.array(X_val_def[:,0]), np.array(X_val_def[:,1])]
popt1, pcov1 = curve_fit( f.func, t1, f.inverse_normalization2(np.array(y_train_def2), data_max, data_min ))
function1 = f.func(v1, *popt1)

net_fwhmy = predicted_fwhmy.detach().numpy()
int_fwhmy = f.inverse_normalization2(np.array(y_val_def2), data_max, data_min)

########################################################################
########################################################################
########################################################################
########################################################################

# CONFRONTO ELLITTICITA OTTENUTA COME RAPPORTO CON IL VALORE VERO
e_true = validation_def[:,4]

e_net = []
for i in range(len(net_fwhmx)):
    if net_fwhmx[i]>net_fwhmy[i]:
        e = (net_fwhmx[i]/net_fwhmy[i])
        e_net = np.append(e_net, e) 
    else:
        e = (net_fwhmx[i]/net_fwhmy[i])
        e_net = np.append(e_net, e)


e_int = []
for i in range(len(int_fwhmx)):
    if int_fwhmx[i]>int_fwhmy[i]:
        e = (int_fwhmx[i]/int_fwhmy[i])
        e_int = np.append(e_int, e) 
    else:
        e = (int_fwhmx[i]/int_fwhmy[i])
        e_int = np.append(e_int, e)

err_net = abs(e_net - e_true.detach().numpy())
err_int = abs(e_int - e_true.detach().numpy())

err = np.stack((err_net, err_int), axis=1)

np.savetxt(f'./ErrorPlot/Tanh_1L_CheckRatio', err)





