import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d, griddata

####################################################################################################################

def func(X, a, b, c, d, e, f):
    return a*X[0]**2 + b*X[1]**2 + c*X[0]*X[1] + d*X[0] + e*X[1] + f

####################################################################################################################

def add_bool(data):
    i = 0
    while i < len(data):
        if i%2 == 0:
            data.at[i, 'check'] = True
        else:
            data.at[i, 'check'] = False
        i += 1
        
####################################################################################################################

def add_bool2(data):
    i = 0
    while i < len(data):
        if (i%2 == 0) or (i%3 == 0):
            data.at[i, 'check'] = True
        else:
            data.at[i, 'check'] = False
        i += 1

####################################################################################################################

#normalization between 0 and 1
def normalization1(y, data_max, data_min):
    m = 1/(data_max - data_min)
    q = - data_min/(data_max - data_min)
    return y * m + q

#normalization between -1 and 1
def normalization2(y, data_max, data_min):
    m = 2/(data_max - data_min)
    q = (data_min + data_max)/(data_min - data_max)
    return y * m + q

def inverse_normalization1(y, data_max, data_min):
    m = 1/(data_max - data_min)
    q = - data_min/(data_max - data_min)
    return (y - q) / m

def inverse_normalization2(y, data_max, data_min):
    m = 2/(data_max - data_min)
    q = (data_min + data_max)/(data_min - data_max)
    return (y - q) / m
    
####################################################################################################################

def linspace(vec, n):
    return np.linspace(np.min(vec), np.max(vec), n)

def int_lin(data, par):
    return interp2d(data["x"], data["y"], data[par], kind="linear")

def val_int(data, f_int):
    x = np.array(data["x"])
    y = np.array(data["y"])
    par_int = np.zeros(len(data))
    i = 0
    while i < par_int.size:
        par_int[i] = f_int(x[i], y[i])
        i += 1
    return par_int

####################################################################################################################

def grid(data, par, xpoints, ypoints):
    return griddata((data["x"], data["y"]), data[par], (xpoints[None, :], ypoints[:, None]), method="linear")

####################################################################################################################

def delta(data, f_int, f_ver):
    x = np.array(data["x"])
    y = np.array(data["y"])
    par_int = np.zeros(len(data))
    par_ver = np.zeros(len(data))
    i = 0
    #Posso, in qualche modo, evitare di ciclare?
    while i < par_ver.size:
        par_int[i] = f_int(x[i], y[i])
        par_ver[i] = f_ver(x[i], y[i])
        i += 1
    return abs(par_int - par_ver)

####################################################################################################################

def graph(data_ver, data_int, par, name):
    fig = plt.figure(figsize=(14, 6))
    ###    
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(data_ver["x"], data_ver["y"], par)
    ax.set_title(f'Errore assoluto [{name}]')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Δ')
    ###
    ax_log = fig.add_subplot(1, 2, 2, projection='3d')
    ax_log.scatter(data_ver["x"], data_ver["y"], np.log10(par))
    ax_log.set_title(f'Errore assoluto [{name}] (log)')
    ax_log.set_xlabel('X')
    ax_log.set_ylabel('Y')
    ax_log.set_zlabel('Δ')
    ###
    fig.tight_layout()
    ###
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(1, 1, 1)
    x = np.array(data_ver["x"])
    y = np.array(data_ver["y"])
    g = ax2.tricontourf(x, y, np.log10(par))
    pt_int = ax2.scatter(data_int["x"], data_int["y"], color="red")
    pt_ver = ax2.scatter(data_ver["x"], data_ver["y"], color="blue")
    plt.legend((pt_int, pt_ver), ('Interpolation', 'Check'), scatterpoints=1, loc='lower right', fontsize=10)
    fig2.colorbar(g)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(f'Errore assoluto [{name}] (log)')
    fig2.savefig(f'./Plots/Errore assoluto [{name}] (log)', dpi=800)

####################################################################################################################
    
def cgraph(data, xpoints, ypoints, par, par_true):
    fig = plt.figure(figsize=(15, 10))
    ###
    ax1 = fig.add_subplot(221)
    g1 = ax1.contourf(xpoints, ypoints, par)
    ax1.set_title('Griddata su data_int')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    fig.colorbar(g1)
    ###
    ax2 = fig.add_subplot(222)
    g2 = ax2.contourf(xpoints, ypoints, par_true)
    ax2.set_title('Griddata su data_check')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    fig.colorbar(g2)
    ###
    ax3 = fig.add_subplot(223)
    g3 = ax3.contourf(xpoints, ypoints, (par-par_true))
    ax3.set_title('Errore assoluto')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.scatter(data["x"], data["y"], color="red")
    fig.colorbar(g3)
    ###
    fig.tight_layout()

####################################################################################################################   

def check_graph_row(data, f_int, test_func, test_func2, par):
    i = 0
    step = abs(data["y"].max()-data["y"].min())/13
    for i in range(13):
        lim_inf = data["y"] <= data["y"].max() - i*step
        lim_sup = data["y"] >= data["y"].max() - step - i*step
        data_test = data[lim_inf & lim_sup]
        ellit = val_int(data_test, f_int)
        fig = plt.figure(figsize=(24,10))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter(data["x"], data["y"], color="orange")
        ax1.scatter(data_test["x"], data_test["y"], color="blue")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(data_test.index, data_test[par], color="red", label=f"{par} vera")
        ax2.plot(test_func, label="curve_fit", color="blue")
        #ax2.plot(test_func2, label="curve_fit", color="blue")
        ax2.plot(data_test.index, ellit, label="interp2d", color="green")
        plt.legend(loc='lower left')
        plt.xlabel("Index")
        plt.ylabel("Ellitticità")
        fig.savefig(f'./Plot_{par}/Row_{i}', dpi=400)
        plt.close(fig)
        
####################################################################################################################   

def check_graph_col(data, f_int, test_func, test_func2, par):
    i = 0
    step = abs(data["x"].max()-data["x"].min())/13
    for i in range(13):
        lim_inf = data["x"] <= data["x"].max() - i*step
        lim_sup = data["x"] >= data["x"].max() - (step+0.004) - i*step
        data_test = data[lim_inf & lim_sup]
        ellit = val_int(data_test, f_int)
        fig = plt.figure(figsize=(24,10))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter(data["x"], data["y"], color="orange")
        ax1.scatter(data_test["x"], data_test["y"], color="blue")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(data_test.index, data_test[par], color="red", label=f"{par} vera")
        ax2.plot(test_func, label="curve_fit", color="blue")
        #ax2.plot(test_func2, label="Fit su complete data", color="blue")
        ax2.set_xlim([data_test.index.min() - 1, data_test.index.max() + 1])
        ax2.plot(data_test.index, ellit, label="interp2d", color="green")
        plt.legend(loc='upper left')
        plt.xlabel("Index")
        plt.ylabel("Ellitticità")
        fig.savefig(f'./Plot_{par}/Col_{i}', dpi=400)
        plt.close(fig)






