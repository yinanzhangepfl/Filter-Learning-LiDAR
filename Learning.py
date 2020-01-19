# Import basic packages
import pandas as pd
import numpy as np
# import plot packages
import matplotlib.pyplot as plt
import gspplot
# Import graph packages
import gsp
import pygsp
# Import pytorch packages
import torch
import torch.nn as nn
import torch.optim as optim
# Import other packages
import os

def set_ground_truth(patch, threshold, f, figsize=(6, 6)):
    """
    Set ground truth.
    
    Attributes:
        - patch              : A patch cropped from the cloud
        - threshold          : Threshold to extract roof points
        - f                  : Filter
        - figsize            : Figsize of visualization

    Return:
        - df2                : DataFrame added the ground truth column
        - edge_roof          : Estimated points of edge roof
    """
    df2 = patch.copy()
    df2.reset_index(drop=False, inplace=True)
    
    # Prepare the signal
    mask = (df2['label'] == 5)|(df2['label'] == 0)
    df2['is_building'] = mask
    df2['is_building'] = df2['is_building'].apply(lambda x: int(x))
    
    # Filter the signal
    signal_roof = f.filter(df2.is_building, method = 'chebyshev')
    if signal_roof.ndim == 1:
        edge_roof = signal_roof >= threshold
    else:
        edge_roof = signal_roof[:, -1] >= threshold
    
    
    # Remove positive false points
    tmp = df2[edge_roof]
    edge_roof[tmp[tmp['label']!=5].index] = False
    df2['is_building'] = df2['label'] == 5
    df2['is_edge'] = edge_roof
    df2['is_edge'] = df2['is_edge'].apply(lambda x: int(x))
    
    # Visualize 
    fig, ax = plt.subplots(figsize=figsize)
    gspplot.plot_vaihingen_2D(patch, 
                              id_highlight=np.nonzero(edge_roof)[0], 
                              label_high='Edges',
                              ax=ax, 
                              title="Ground Truth")
    return df2, edge_roof

def comp_df(patch, edge_dict, labels, normalize=True, **kwargs):
    """
    Calculate the composition of the highlighted points.
    
    Attributes:
        - patch                 : 
        - edge_dict             : 
        - labels                : 
        - normalize             :
        
    Return:
        - df                    : 
    """
    if kwargs is not None:
        num = kwargs['num'] if 'num' in kwargs else None

    data = []
    for i in edge_dict.keys():
        if num is None:
            tmp =  patch[edge_dict[i]].label.value_counts(normalize=normalize)
        else:
            tmp =  patch[edge_dict[i][num]].label.value_counts(normalize=normalize)
        tmp = tmp.sort_index()
        data.append(tmp)
        
    for i in range(len(data)):
        for j in range(len(patch.label.unique())):
            if j not in data[i].index:
                data[i].loc[j] = 0
        data[i].sort_index(inplace=True)
        
    data = [list(data[i]) for i in range(len(data))]
    df   = pd.DataFrame(data = data, columns=list(labels.values()))
    
    new_index = [i[i.find('_')+1:] for i in list(edge_dict.keys())]
    new_index = dict(zip(range(len(new_index)), new_index))
    df.rename(index=new_index, inplace=True) # Use a dictionary to change index
    return df

def qua_comp_df(df, df2, edge_dict, edge_roof, **kwargs):
    """
    Add another three columns to our dataframe.
    
    Attributes:
        - df                    : 
        - df2                   : 
        - edge_dict             : 
        - edge_roof             :
        
    Return:
        - df                    : 
    """
    if kwargs is not None:
        num = kwargs['num'] if 'num' in kwargs else None

    if num is None:
        total = [np.sum(edge_dict[i]) for i in edge_dict.keys()]
        total_roof = [np.sum(df2.loc[edge_dict[i], 'is_building']) for i in edge_dict.keys()]
        total_edge = [np.sum(df2.loc[edge_dict[i], 'is_edge']) for i in edge_dict.keys()]
    else:
        total = [np.sum(edge_dict[i][num]) for i in edge_dict.keys()]
        total_roof = [np.sum(df2.loc[edge_dict[i][num], 'is_building']) for i in edge_dict.keys()]
        total_edge = [np.sum(df2.loc[edge_dict[i][num], 'is_edge']) for i in edge_dict.keys()]
    df['Total'] = total
    df['Precision'] = [i/j for i,j in zip(total_edge, total)]
    df['Recall_GT'] = [i/np.sum(df2['is_edge']) for i in total_edge]
    df['Recall_roof'] = [i/np.sum(df2['label'] == 5) for i in total_roof]
    return df.sort_values(by='Recall_GT', ascending=False)

def recover_df(edge_dict, df3, labels2, normalize=True, **kwargs):
    """
    Set ground truth.
    
    Attributes:
        - edge_dict          : 
        - df3                : 
        - labels2            : 
        - normalize          : 

    Return:
        - tmp                : 
    """
    if kwargs is not None:
        num = kwargs['num'] if 'num' in kwargs else None

    s = df3.replace({"label": labels2}).label.value_counts()
    if num is None:
        tmp = comp_df(df3, edge_dict, labels2, normalize=False).append(s)
    else:
        tmp = comp_df(df3, edge_dict, labels2, normalize=False, num=num).append(s)
    tmp=tmp.rename(index = {'label':'total'})
    tmp['sum'] = tmp.sum(axis=1)
    tmp = tmp.T
    if normalize:
        tmp['bp_30nn_Binary'] = tmp['bp_30nn_Binary'] / tmp['total']
        tmp['bp_30nn_Local'] = tmp['bp_30nn_Local'] / tmp['total']
    return tmp

def load_G_30nn_Binary(path_e, path_U, path_L, patch):
    """
    Load 30nn binary graph and do eigen decomposition.
    
    Attributes:
        - path_e                : Path to Eigen-value  matrix
        - path_U                : Path to Eigen-vector matrix
        - path_L                : Path to Laplacian matrix
        - patch                 : A patch cropped from the cloud
        
    Return:
        - G_30nn_Binary_e       : Eigen-value  matrix of 30nn binary graph
        - G_30nn_Binary_U       : Eigen-vector matrix of 30nn binary graph
        - G_30nn_Binary_L       : Laplacian matrix of 30nn binary graph
    """
    # Load matrices if they exist
    if os.path.exists(path_e) and os.path.exists(path_U) and os.path.exists(path_L):
        G_30nn_Binary_e = np.load(path_e)
        G_30nn_Binary_U = np.load(path_U)
        G_30nn_Binary_L = np.load(path_L, allow_pickle=True).item()
        
    else:
        G_30nn_Binary, l2 = gsp.knn_graph(cloud=patch[['x', 'y', 'z']], 
                                          k=30, 
                                          dist3D=False, 
                                          mode='connectivity', 
                                          neightype='number', 
                                          lap_type='combinatorial', 
                                          norm=True)
        
        # Compute Eigen-value, Eigen-vector
        if not os.path.exists(path_e) or not os.path.exists(path_U):
            G_30nn_Binary.compute_fourier_basis(recompute=False)
            G_30nn_Binary_e, G_30nn_Binary_U = G_30nn_Binary.e, G_30nn_Binary.U
            np.save(path_U, G_30nn_Binary_U)
            np.save(path_e, G_30nn_Binary_e)
        else:
            G_30nn_Binary_e = np.load(path_e)
            G_30nn_Binary_U = np.load(path_U)
        
        # Compute Laplacian matrices
        if not os.path.exists(path_L):
            G_30nn_Binary.compute_laplacian('combinatorial')
            G_30nn_Binary_L = G_30nn_Binary.L
            np.save(path_L, G_30nn_Binary_L)
        else:
            G_30nn_Binary_L = np.load(path_L, allow_pickle=True).item()
            
    return G_30nn_Binary_e, G_30nn_Binary_U, G_30nn_Binary_L

def generate_data_non_parametric(U, patch):
    """
    Generate input data to learn a non parametric filter.
    
    Attributes:
        - U                  : Eigen-vector matrix
        - patch              : Aimed patch
        
    Return:
        - train_input        : Polynomialized terms in tensor format
    """
    n_dim       = len(patch)
    train_input = np.zeros([n_dim, n_dim])
    for i in range(n_dim):
        train_input[:, i] = U[:, i]*(U[:, i]@patch.z)
    train_input = torch.from_numpy(train_input).float()
    return train_input

def train_non_parametric_filter(nb_epochs, train_input, train_target, e, edge, f, gamma=1e-6, alter_thresh = False):
    """
    Training process to learn a non parametric filter.
    
    Attributes:
        - nb_epochs          : Number of epochs to train
        - train_input        : Initial filtered signal
        - train_target       : Binary targets labeling node class
        - e                  : Eigen values
        - edge               : Edge
        - f                  : Filter to initalize parameters
        - gamma              : Learning rate
        - alter_thresh       : Whether to optimize the threshold
        
    Return:
        - w                  : Optimized Omega coefficients 
        - train_error_list   : List of training error
        - loss_list          : List of training loss
    """
    def compute_error_loss(signal):
        """
        Compute train error and loss.

        Return:
            - error              : Training error
            - loss               : Training loss
        """
        error = 0
        for i,j in zip(signal, train_target):
            if int(i) != int(j):
                error += 1
        error = 100 * error / train_input.size(0)
        loss  = float(l(torch.sigmoid(train_input.mv(w) - t), train_target)) * train_input.shape[0]
        return error, loss
    
    strMargin = lambda s, l=50, sep=' ': int((l-len(s))/2)*'-' + sep + s + sep + (l-int((l-len(s))/2)-len(s))*'-'
    print(strMargin('Train non Parametric Filter'))
    
    # Initialization
    initial_error = 0
    loss_list, train_error_list = [], []
    
    # Initialize polynomial coefficients: Omega
    w = torch.from_numpy(f.evaluate(e)[1, :]).float()
    w.requires_grad = True
    
    # Initialize threshold
    t = torch.tensor([0.2], requires_grad = True)
    
    # Define loss
    l = nn.BCELoss()

    # Compute initial train error and loss
    initial_error, initial_loss = compute_error_loss(edge)
    train_error_list.append(initial_error)
    loss_list.append(initial_loss)
    
    # Start training
    print("Start training...")
    for epoch in range(nb_epochs):
        for n in range(train_input.size(0)):
            x_tr, y_tr = train_input[n], train_target[n]
            output = torch.sigmoid(x_tr.view(1, -1).mv(w) - t)
            loss = l(output, y_tr)   
            loss.backward()                 
            with torch.no_grad():
                w = w - gamma * w.grad
            w = torch.clamp(w, min=0)
            w.requires_grad = True
        # Append new train error rate and new loss
        error, loss = compute_error_loss((train_input.mv(w) - t)>=0)
        train_error_list.append(error)
        loss_list.append(loss)
        print('=> Epoch: {}/{}\tError: {:.4f}\t Loss: {:.4f}\t'.format(epoch+1, nb_epochs, error, loss), end='\r')
    print('\n=> Done.')
    
    print(strMargin('',sep='-'))
    return w, train_error_list, loss_list

def generate_data_mexican_hat(U, patch):
    """
    Generate input data to learn a mexican hat filter.
    
    Attributes:
        - U                  : Eigen-vector matrix
        - patch              : Aimed patch
        
    Return:
        - train_input        : Polynomialized terms in tensor format
    """
    n_dim       = len(patch)
    train_input = np.zeros([n_dim, n_dim])
    for i in range(n_dim):
        train_input[:, i] = U[:, i]*(U[:, i]@patch.z)
    train_input = torch.from_numpy(train_input).float()
    return train_input

def train_mexican_hat(nb_epochs, train_input, train_target, e, signal, edge, alter_thresh = False, initial_tau=1.5, gamma=1e-4, breakDetect=True):
    """
    Training process to learn a mexican hat filter.
    
    Attributes:
        - nb_epochs          : Number of epochs to train
        - train_input        : Initial filtered signal
        - train_target       : Binary targets labeling node class
        - e                  : Eigen values
        - signal             : Signal
        - edge               : Edge
        - alter_thresh       : Whether to optimize the threshold
        - initial_tau        : Initial tau
        - gamma              : Learning rate
        - breakDetect        : Whether to break before looping all epochs
        
    Return:
        - tau                : Optimized tau coefficients 
        - t                  : Optimized threshold
        - train_error_list2  : List of training error
        - loss_list2         : List of training loss
    """
    def compute_error_loss(edge, signal):
        """
        Compute train error and loss.

        Return:
            - error              : Training error
            - loss               : Training loss
        """
        error = 0        
        for i,j in zip(edge, train_target):
            if int(i) != int(j):
                error += 1
        error = 100 * error / train_input.size(0)
        loss  = float(l(torch.sigmoid(signal.float() - t), train_target)) * train_input.shape[0]
        return error, loss
    
    strMargin = lambda s, l=50, sep=' ': int((l-len(s))/2)*'-' + sep + s + sep + (l-int((l-len(s))/2)-len(s))*'-'
    print(strMargin('Train Mexican Hat Kernel'))
    
    # Initialization
    initial_error, e = 0, torch.from_numpy(e).float()
    loss_list2, train_error_list2 = [], []
    
    # Initialize tau
    tau = torch.tensor([initial_tau], requires_grad = True)
    
    # Initialize threshold
    t   = torch.tensor([0.2], requires_grad = True)
    
    # Define optimizer
    lr_t = alter_thresh if alter_thresh else 0
    optimizer = optim.SGD([{'params': tau, 'lr': gamma},
                           {'params': t,   'lr': lr_t}])
    
    # Define loss
    l = nn.BCELoss()
    
    # Calculate initial loss and initial train error
    initial_error, initial_loss = compute_error_loss(edge, torch.from_numpy(signal[:, -1]))
    loss_list2.append(initial_loss)
    train_error_list2.append(initial_error)
    
    # Start training
    print("Start training...")
    for epoch in range(nb_epochs):
        for n in range(train_input.size(0)):
            x_tr, y_tr = train_input[n], train_target[n]
            w2 = e * tau * torch.exp(- e * tau)
            optimizer.zero_grad()
            output = torch.sigmoid(x_tr.view(1, -1).mv(w2) - t)
            loss   = l(output, y_tr)    
            loss.backward()                 
            optimizer.step()

        result = train_input.mv(e*tau*torch.exp(-e*tau))
        error, loss = compute_error_loss(result-t>=0, result)
        loss_list2.append(loss)
        train_error_list2.append(error)
        print('=> Epoch: {}/{}\tError: {:.4f}\t Loss: {:.4f}\t'.format(epoch+1, nb_epochs, error, loss), end='\r')
        
        if breakDetect:
            if loss_list2[-2] - loss_list2[-1]<0.1:
                break
    print('\n=> Done.')
    print(strMargin('',sep='-'))
    return tau, t,  w2, train_error_list2, loss_list2

def build_poly(patch, L, k):
    """
    Build polynomial terms for Laplacian matrix.
    
    Attributes:
        - patch              : A patch cropped from the cloud
        - L                  : Laplacian matrix
        - k                  : The heightest order
        
    Return:
        - train_input        : Polynomialized terms
    """
    train_input = L@patch.z
    feature = L@train_input
    for i in range(k-1):
        train_input = np.c_[train_input, feature]
        feature = L@feature
    return train_input

def poly_e(e, k):
    """
    Build polynomial terms for eigen-value.

    Attributes:
        - e                  : Eigen-value
        - k                  : The heightest order

    Return:
        - result             : Polynomialized terms
    """
    result = e
    for i in range(1, k):
        result = np.c_[result, e**(i+1)]
    return result

def generate_data_poly(patch, L, k):
    """
    Generate input data to learn a polynomial filter.
    
    Return:
        - train_input        : Polynomialized terms in tensor format
    """
    train_input = build_poly(patch, L, k)
    train_input = torch.from_numpy(train_input).float()
    return train_input

def train_polynomial_kernel(nb_epochs, train_input3, train_target, k, e, f, gamma=1e-4, alter_thresh=False, scheduler_flag=False, scheduler_step=10, scheduler_gamma=0.5):
    """
    Training process to learn a Polynomial Kernel.
    
    Attributes:
        - nb_epochs          : Number of epochs to train
        - train_input3       : Initial filtered signal
        - train_target       : Binary targets labeling node class
        - k                  : The heightest order
        - e                  : Eigen values
        - f                  : Filter to initalize parameters
        - gamma              : Learning rate of alpha
        - alter_thresh       : Whether to optimize the threshold
        - scheduler_flag     : Whether to use scheduler
        
    Return:
        - x                  : Polynomialized terms for eigen-value
        - alpha              : Optimized alpha coefficients 
        - t3                 : Optimized threshold
        - train_error_list3  : List of training error
        - loss_list3         : List of training loss
    """
    
    def compute_error_loss():
        """
        Compute train error and loss.

        Return:
            - error              : Training error
            - loss               : Training loss
        """
        error = 0
        f_hat = train_input3.mv(alpha)
        for i,j in zip((f_hat - t3) > 0, train_target):
            if int(i) != int(j):
                error += 1
        error = 100 * error / train_input3.size(0)
        loss  = float(l(torch.sigmoid(f_hat - t3), train_target)) * train_input3.shape[0]
        return error, loss
        
    strMargin = lambda s, l=50, sep=' ': int((l-len(s))/2)*'-' + sep + s + sep + (l-int((l-len(s))/2)-len(s))*'-'
    print(strMargin('Train Polynomial Kernel'))
    print("Entire epochs: {}\tHeightest order k: {}".format(nb_epochs, k) )
    
    # Initialization
    initial_error = 0
    loss_list3, train_error_list3 = [], []
    
    # Initialize polynomial coefficients: alpha
    x, y = poly_e(e, k), f.evaluate(e)[1, :]
    alpha = torch.from_numpy(np.linalg.lstsq(x, y)[0]).float()
    alpha.requires_grad = True
    
    # Initialize threshold
    t3 = torch.tensor([0.2], requires_grad = True)
    
    # Define optimizer
    lr_t = 0 if not alter_thresh else alter_thresh
    optimizer = torch.optim.SGD([{'params': alpha, 'lr': gamma},
                                 {'params': t3,    'lr': lr_t}])
    if scheduler_flag:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    # Define loss
    l = nn.BCELoss()
    
    # Compute original train error and loss
    initial_error, initial_loss = compute_error_loss()
    train_error_list3.append(initial_error)                        
    loss_list3.append(initial_loss)

    # Start training
    print("Start training...")
    for epoch in range(nb_epochs):
        for n in range(train_input3.size(0)):
            x_tr, y_tr = train_input3[n], train_target[n]
            output = torch.sigmoid(x_tr.view(1, -1).mv(alpha) - t3)
            loss   = l(output, y_tr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler_flag:
            scheduler.step()
        # Append train error and loss after each epoch
        error, loss = compute_error_loss()
        train_error_list3.append(error)
        loss_list3.append(loss)
        print('=> Epoch: {}/{}\tError: {:.4f}\t Loss: {:.4f}\t'.format(epoch+1, nb_epochs, error, loss), end='\r')
    print('\n=> Done.')
    
    print(strMargin('',sep='-'))
    return x, alpha, t3, train_error_list3, loss_list3

def type_error_tree(df):
    """
    Calculate type 1 error and type 2 error for tree detection.
    
    Attributes:
        - df                 : DataFrame containing data for computing error
        
    Return:
        - df                 : DataFrame after processing
    """
    df.loc['type_1 error'] = (1 - df.loc['Tree']/df.loc['sum'])*100
    df.loc['type_2 error'] = (df.loc['Tree', 'total']-df.loc['Tree'])/(df.loc['sum', 'total']-df.loc['sum'])*100
    df.loc['type_1 error', 'total'] = '/'
    df.loc['type_2 error', 'total'] = '/'
    df.iloc[:, :-1] = df.iloc[:, :-1].astype(int)
    return df

def type_error_roof(df):
    df.loc['type_1 error'] = (1 - df.loc['Edge']/df.loc['sum'])*100
    df.loc['type_2 error'] = (df.loc['Edge', 'total']-df.loc['Edge'])/(df.loc['sum', 'total']-df.loc['sum'])*100
    df.loc['type_1 error', 'total'] = '/'
    df.loc['type_2 error', 'total'] = '/'
    df.iloc[:, :-1] = df.iloc[:, :-1].astype(int)
    return df

def type_error_tree_roof(df):
    """
    Calculate type 1 error and type 2 error.

    """
    df.loc['type_1 error','roof'] = (1 - df.loc['Edge','roof']/df.loc[ 'sum','roof'])*100
    df.loc['type_2 error','roof'] = (df.loc['Edge','total']-df.loc['Edge','roof'])/ \
                                         (df.loc['sum','total']-df.loc['sum','roof'])*100
    df.loc['type_1 error','tree'] = (1 - df.loc['Tree','tree']/df.loc[ 'sum','tree'])*100
    df.loc['type_2 error','tree'] = (df.loc['Tree','total']-df.loc['Tree','tree'])/ \
                                         (df.loc['sum','total']-df.loc['sum','tree'])*100
    df.loc['type_1 error', 'total'] = '/'
    df.loc['type_2 error','total'] = '/'
    df.iloc[:, :-1] = df.iloc[:, :-1].astype(int)
    return df