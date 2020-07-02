import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d, griddata
import pandas as pd
import os
import seaborn as sns
import glob
    
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
    plt.legend((pt_int, pt_ver), ('Interpolation', 'Check'), scatterpoints=1, loc='upper left', fontsize=10)
    fig2.colorbar(g)
    ax2.set_title(f'Errore assoluto [{name}] (log)')

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
        fig = plt.figure(figsize=(12,5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter(data["x"], data["y"], color="orange")
        ax1.scatter(data_test["x"], data_test["y"], color="blue")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(data_test.index, data_test["e"], color="red", label=f"{par} vera")
        ax2.plot(test_func, label="Fit su half data", color="red")
        ax2.plot(test_func2, label="Fit su complete data", color="blue")
        ax2.plot(data_test.index, ellit, label="Interp2d", color="green")
        plt.legend(loc='upper left')
        plt.xlabel("Index")
        plt.ylabel("Ellitticità")
        fig.savefig(f'./Plot_{par}/Row_{i}')
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
        fig = plt.figure(figsize=(12,5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter(data["x"], data["y"], color="orange")
        ax1.scatter(data_test["x"], data_test["y"], color="blue")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(data_test.index, data_test["e"], color="red", label=f"{par} vera")
        ax2.plot(test_func, label="Fit su half data", color="red")
        ax2.plot(test_func2, label="Fit su complete data", color="blue")
        ax2.set_xlim([data_test.index.min() - 1, data_test.index.max() + 1])
        ax2.plot(data_test.index, ellit, label="Interp2d", color="green")
        plt.legend(loc='upper left')
        plt.xlabel("Index")
        plt.ylabel("Ellitticità")
        fig.savefig(f'./Plot_{par}/Col_{i}')
        plt.close(fig)

####################################################################################################################   

def expand_title(file):
    tit_elements = str.split(file, sep = "_")    
    print(tit_elements)
    tit_elements[0] = "Activation Function: " +  tit_elements[0] 
    tit_elements[1] = "Layers: " + tit_elements[1].strip("L")
    if tit_elements[2] == "e":
        tit_elements[2] = "Parameter: Ellipticity"
    if tit_elements[2] == "fwhm": 
        if tit_elements[3] == "x":
            tit_elements[2] = "Parameter: Full width half maximum (x)"
        if tit_elements[3] == "y":
            tit_elements[2] = "Parameter: Full width half maximum (y)"
    if tit_elements[2] == "co":
        tit_elements[2] = "Parameter: Co Max"
    if tit_elements[2] == "cx":
        tit_elements[2] = "Parameter: Cx Max"
        
    title_str = "\n".join(tit_elements[:3])
    return title_str

####################################################################################################################   

def plot_from_path(PATH):
    files = os.listdir(PATH)

    col_names = ["Network error", "Interpolation error"]
    n_plots = len(col_names)

    for file in files:
        if file[0] != ".":
            path_file = os.path.join(PATH, file)
            token = pd.read_csv(path_file, 
                                index_col = None, 
                                delimiter = " ", 
                                header = None, 
                                names = col_names)

            Title_str = expand_title(file)

            sns.set(font_scale = 1.7, 
                    style = "whitegrid",
                    palette = "colorblind",
                    rc={"lines.linewidth": 2})

            fig, axs = plt.subplots(1, 2, figsize = (20, 8))
            fig.suptitle(Title_str, ha = "center", va = "baseline")
            for name in col_names:
                plot = sns.lineplot(y = name, 
                                    x = "index", 
                                    markers = True, 
                                    label = name,
                                    data = token.reset_index(),
                                    ax = axs[0])
                hist = sns.distplot(token["Network error"], kde=False, ax=axs[1])
                hist = sns.distplot(token["Interpolation error"], kde=False, ax=axs[1])
                hist.legend(labels=["Network error", "Interpolation error"])
                hist.set(xlabel="Error", ylabel="Count")
            plot.set(xlabel = "Index", ylabel = "Error")
            fig.savefig(f"./Plots/Hist/{Title_str}.png", bbox_inches='tight', dpi=600)
            plt.show()
    
####################################################################################################################   

def violin_from_df(df, par):
    names=df["architecture"].unique()
    names.sort()
    fig, axs = plt.subplots(1, 1, figsize = (18, 8))
    g = sns.violinplot(x="architecture", y="value", hue="error", data=df, split=True, cut=0, order=names)
    g.set_title(f"Error for Network and Interpolation [{par}]")  
    g.set(xlabel="Architecture", ylabel="Error")
    fig.savefig(f"./Plots/ViolinPlots/violin_plot_{par}.png", dpi=600)
    plt.show()
    
####################################################################################################################     
    
def df_from_path(PATH):
    all_files = glob.glob(PATH + "/*")
    l_e = []
    l_fwhmx = []
    l_fwhmy = []
    l_comax = []
    l_cxmax = []
    #l_check = []

    for filename in all_files:
        if "_e" in filename: 
            file = (os.path.basename(filename))
            name = [file+" [Net]", file+" [Int]"]
            df = pd.read_csv(filename, delimiter=" ", header=None, names=name)
            l_e.append(df)

        if "_fwhm_x" in filename:
            file = (os.path.basename(filename))
            name = [file+" [Net]", file+" [Int]"]
            df = pd.read_csv(filename, delimiter=" ", header=None, names=name)
            l_fwhmx.append(df)

        if "_fwhm_y" in filename:
            file = (os.path.basename(filename))
            name = [file+" [Net]", file+" [Int]"]
            df = pd.read_csv(filename, delimiter=" ", header=None, names=name)
            l_fwhmy.append(df)
        
        if "_co_max" in filename:
            file = (os.path.basename(filename))
            name = [file+" [Net]", file+" [Int]"]
            df = pd.read_csv(filename, delimiter=" ", header=None, names=name)
            l_comax.append(df)
        
        if "_cx_max" in filename:
            file = (os.path.basename(filename))
            name = [file+" [Net]", file+" [Int]"]
            df = pd.read_csv(filename, delimiter=" ", header=None, names=name)
            l_cxmax.append(df)
        
        '''
        if "_Check" in filename:
            file = (os.path.basename(filename))
            name = [file+" [Net]", file+" [Int]"]
            df = pd.read_csv(filename, delimiter=" ", header=None, names=name)
            l_check.append(df)
        '''
        
    df_e = pd.concat(l_e, axis=1, sort=False)
    df_fwhmx = pd.concat(l_fwhmx, axis=1, sort=False)
    df_fwhmy = pd.concat(l_fwhmy, axis=1, sort=False)
    df_comax = pd.concat(l_comax, axis=1, sort=False)
    df_cxmax = pd.concat(l_cxmax, axis=1, sort=False)
    #df_check = pd.concat(l_check, axis=1, sort=False)

    df_e = pd.melt(df_e, var_name="description", value_name="value")
    df_fwhmx = pd.melt(df_fwhmx, var_name="description", value_name="value")
    df_fwhmy = pd.melt(df_fwhmy, var_name="description", value_name="value")
    df_comax = pd.melt(df_comax, var_name="description", value_name="value")
    df_cxmax = pd.melt(df_cxmax, var_name="description", value_name="value")
    #df_check = pd.melt(df_check, var_name="description", value_name="value")

    def geterr(descr):
        return descr.split(sep=" ")[-1]

    def getarch(descr):
        arch = descr.split(sep="_")[:2]
        return "_".join(arch)

    df_e["error"] = df_e["description"].apply(geterr)
    df_e["architecture"] = df_e["description"].apply(getarch)
    df_fwhmx["error"] = df_fwhmx["description"].apply(geterr)
    df_fwhmx["architecture"] = df_fwhmx["description"].apply(getarch)
    df_fwhmy["error"] = df_fwhmy["description"].apply(geterr)
    df_fwhmy["architecture"] = df_fwhmy["description"].apply(getarch)
    df_comax["error"] = df_comax["description"].apply(geterr)
    df_comax["architecture"] = df_comax["description"].apply(getarch)
    df_cxmax["error"] = df_cxmax["description"].apply(geterr)
    df_cxmax["architecture"] = df_cxmax["description"].apply(getarch)
    #df_check["error"] = df_check["description"].apply(geterr)
    #df_check["architecture"] = df_check["description"].apply(getarch)
    
    return df_e, df_fwhmx, df_fwhmy, df_comax, df_cxmax#, df_check

####################################################################################################################   

def mod_df(df):
    df0 = df[df["error"]=="[Net]"] 
    df1 = df[df["error"]=="[Int]"]
    df1 = df1.drop(["architecture"], axis=1)
    df1["architecture"] = "Tanh_1L"
    df2 = df1
    df2 = df2.drop(["architecture"], axis=1)
    df2["architecture"] = "Tanh_2L"
    df3 = df1
    df3 = df3.drop(["architecture"], axis=1)
    df3["architecture"] = "Tanh_3L"
    df4 = df1
    df4 = df4.drop(["architecture"], axis=1)
    df4["architecture"] = "Sigmoid_1L"
    df5 = df1
    df5 = df5.drop(["architecture"], axis=1)
    df5["architecture"] = "Sigmoid_2L"
    df6 = df1
    df6 = df6.drop(["architecture"], axis=1)
    df6["architecture"] = "Sigmoid_3L"
    tot = pd.concat([df0, df1, df2, df3, df4, df5, df6])
    tot.reset_index()
    return tot
    
####################################################################################################################   

def getstats(df):
    archs = ["Tanh_1L", "Tanh_2L", "Tanh_3L", "Sigmoid_1L", "Sigmoid_2L", "Sigmoid_3L"]
    for arch in archs:
        print(arch)
        filt = df.loc[df_e["architecture"] == arch]
        filt_net = filt.loc[ df_e["error"] == "[Net]"]
        print("Net\tMin, Med, Max: ", np.percentile(filt_net["value"], [5,50,95]))
        filt_int = filt.loc[ df_e["error"] == "[Int]"]
        print("Int\tMin, Med, Max: ", np.percentile(filt_int["value"], [5,50,95]), "\n")    
    
    
    
