import time
import numpy as np
import pandas as pd
import gspplot
import matplotlib.pyplot as plt

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from pygsp import graphs
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC

TIME_DISPLAY = False


def knn_graph(cloud, k=30, r=1, dist3D=True, mode='connectivity', neightype='number', lap_type='combinatorial', norm=False, sigma=None, plot_dist_ditrib=False, V=None):
    """
    Construct graph using PyGSP toolbox from the adjacency matrix.
    
    Return:
        If the graph is normalized by the largest eigenvalue:
        - G                 : Graph constructed by PyGSP package
        - l                 : The largest eignevalue of the normalized graph, which is 1.
        
        Else, only return G
    """
    W, _ = knn_w(cloud, k=k, r=r, dist3D=dist3D, mode=mode, neightype=neightype, sigma=sigma, plot_dist_ditrib=plot_dist_ditrib, V=V)
    G = graphs.Graph(W, lap_type=lap_type)
    G.estimate_lmax()
    if norm == True:
        l = G.lmax
        W = W/l
        G = graphs.Graph(W, lap_type=lap_type)
        G.estimate_lmax()
        return G, l
    else:
        return G
    

def knn_w(cloud, k=30, r=1, dist3D=False, mode='connectivity', neightype='number', sigma=None, plot_dist_ditrib=False, V=None):
    """
    Compute the laplacian matrix. Use kNN to find nearest neig. and link points.

    Attributes:
        - cloud (array, Nx4): Point cloud expressed as [X | Y | Z | label] matrix of size Nx4
        - k (int)           : the number of neig. to consider for kNN
        - r (float)         : the search radius when looking for neighbors of a particular node
        - dist3D            : Use Z coordinate to compute distance between points while computing graph. If set to False, only XY                               values are considered.
        - mode (string)     : Can be either 'connectivity', 'distance' or 'local'. 'connectivity' will set weights to default                                   value 1.           
                             'distance' will set weigths value as a function of w_ij = exp(dist_ij^2) / (2*s^2)), where s is the                               mean value of the distances if sigma is set to be None. 'local' will adjust the scaling parameter                                 as the distance between a node and its farthest neighbor. 
        - neightype (string): Can be 'number' or 'radius'. 'number' will generate k nearest neighbors. 'radius' will generate                                   neighbors within a given scope.
        - sigma             : The scaling parameter when constructing the Gaussian graph, if set to be None, the scaling                                       parameter is the mean value of the distances 
        - plot_dist_ditrib  : Whether to plot the distribution of distances and weights
        - V                 : Additional keyword arguments for the metric function NearestNeighbors
    Return:
        - W (sparse matrix, NxN): The adjacency matrix
        - dists_                : The vector of distances
        
    """
    start_time = time.time()

    N = len(cloud)
    # Copy cloud to avoid modifying base one
    cloud_res = np.array(cloud, copy=True)
    _k = k-1
    
    # Check if 2D or 3D distance used to construct graph
    if not dist3D:
        cloud_res[:,2] = 0
            
    # Compute kNN and fit data
    if V is not None:
        nn = NearestNeighbors(n_neighbors=k, metric='mahalanobis', 
                              metric_params={'V': V })
    else:
        nn = NearestNeighbors(n_neighbors=k, radius=r)
        
    nn.fit(cloud_res)
    if neightype == 'number':
        dists, ids = nn.kneighbors(cloud_res)
    elif neightype == 'radius':
        dists, ids = nn.radius_neighbors(cloud_res)
    dists_ = [j for i in dists for j in i]
    ids_ = [j for i in ids for j in i]#generate column indices
    
    # Generate row indices
    rows = [i for i, j in enumerate(ids) for k in j]
    #number of edges
    M = len(rows)
    
    # Check construction mode
    if mode == 'connectivity':
        w = np.ones(M)
    elif mode == 'distance':
        dists_array = np.array(dists_)
        if sigma==None:
            s = np.mean(dists_array[dists_array!=0])#the distance between a node to itself is zero
        else:
            s = sigma
        w = np.exp(-np.power(dists_,2)/(2*np.power(s,2)))
    elif mode == 'local':
        if neightype == 'number':
            # Check construction mode
            dists_ki = np.zeros(dists.shape) 
            dists_kj = np.zeros(dists.shape) 
            dists_ki = np.repeat([dists[:, -1]], dists.shape[1], axis=0).T + 1e-10
            dists_kj = dists[:, -1][ids] + 1e-10
            w = np.exp(-np.power(dists,2)/(dists_ki*dists_kj))
            w = w.flatten()
        elif neightype == 'radius':
            w = np.exp(-np.power(dists_,2)/(r**2))
            
    else:
        return
    if plot_dist_ditrib:
        plt.figure(figsize=(16, 4))
        plt.subplot(1,2,1); plt.title('Distances distribution')
        plt.hist(dists[:,1:].flatten(), bins=40); plt.xlabel('distance'); plt.ylabel('#points')
        plt.subplot(1,2,2); plt.title('Weights distribution')
        plt.hist(w.flatten(), bins=40); plt.xlabel('weigths'); plt.ylabel('#points')
        # plt.vlines(s, [0], [10000], lw=2, color='b')
    
    # Complete matrix according to positions
    _W = coo_matrix((w, (rows, ids_)), shape=(N, N))
    coo_matrix.setdiag(_W, 0)

    _W = 1/2*(_W + _W.T)
    
    if TIME_DISPLAY:
        print("--- kNN graph: {:.4f} seconds ---".format(time.time() - start_time))
    
    return _W, dists_

def set_ground_truth(patch, threshold, f, figsize=(6, 6)):
    """
    Set ground truth.
    
    Attributes:
        - patch              : A patch cropped from the cloud
        - threshold          : Threshold to extract roof points
        - f                  : Filter
        - figsize            : Figure size for visualization

    Return:
        - df2                : The original 'patch' dataframe plus two columns 'is_building' and 'is_edge'
        - edge_roof          : Constructed ground truth for roof points
    """
    df2 = patch.copy()
    df2.reset_index(drop=False, inplace=True)
    
    # Prepare the signal
    mask = (df2['label'] == 5)|(df2['label'] == 0)
    df2['is_building'] = mask
    df2['is_building'] = df2['is_building'].apply(lambda x: int(x))
    
    # Filter the signal
    signal_roof = f.filter(df2.is_building, method = 'chebyshev')
    edge_roof   = signal_roof[:, -1] >= threshold
    
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
        - patch                 : A patch cropped from the cloud
        - edge_dict             : A dictionary containing points of interest
        - labels                : A dictionary defined in DataProcessing.py where keys are label indice and values are label                                       names
        - normalize             : Parameter of the function pandas.Series.value_counts
        
    Return:
        - df                    : A dataframe containing the ratio of points in each class
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
    Calculate the total number of red points, the precision rate and the recall rate (refer to the ground truth or the label).
    
    Attributes:
        - df                    : A dataframe returned from the self-defined function comp_df
        - df2                   : The original 'patch' dataframe plus two columns 'is_building' and 'is_edge'
        - edge_dict             : A dictionary containing points of interest
        - edge_roof             : Constructed ground truth for roof points
        
    Return:
        - df                    : The original dataframe 'df' returned by the self-defined function 'comp_df' plus four columns                                     'Total', 'Precision', 'Recall_GT', 'Recall_Roof'
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

def sum_df(patch, df, edge_dict, df2):
    """
    Count the total number of red points, the number of detected roof points, the number of detected roof edge points, the recall     rate of roof points.
    
    Attributes:
        - patch         : A patch cropped from the cloud
        - df            : A dataframe returned from the self-defined function comp_df
        - edge_dict     : A dictionary containing points of interest
        - df2           : The original 'patch' dataframe plus two columns 'is_building' and 'is_edge'
    Return: the original dataframe 'df' returned by the self-defined function 'comp_df' plus four columns 'Total', 'Total_roof',             'Total_edge' and 'Ratio_roof'
    """
    total = [np.sum(edge_dict[i]) for i in edge_dict.keys()]
    total_roof = [np.sum(df2.loc[edge_dict[i], 'is_building']) for i in edge_dict.keys()]
    total_edge = [np.sum(df2.loc[edge_dict[i], 'is_edge']) for i in edge_dict.keys()]
    ratio_roof = [np.sum(patch[edge_dict[i]].label == 5)/len(patch[patch.label == 5]) for i in edge_dict.keys()]
    df['Total'] = total
    df['Total_roof'] = total_roof
    df['Total_edge'] = total_edge
    df['Ratio_roof'] = ratio_roof
    return df.sort_values(by='Total_edge', ascending=False)

def SVM_Mahalanobis(class_weight, gamma=None, C=1, X=None):
    """
    Define a nonlinear SVM classifier
    """
    # Compute sigma
    (S, n) = X.shape
    Sigma = (1/S)*X.T.dot(X)
    # mu = 1/(S**2) * x_train_tree.T.dot(np.ones((S,S))).dot(x_train_tree)
        
    def mahalanobis_linear_kernel(X, Y):
        # Compute RBF ; exp(- gamma * ||x-y||^2)
        if gamma is None:
            gamma_ = 1/X.shape[1]
        else:
            gamma_ = gamma
        dist = DistanceMetric.get_metric('mahalanobis', V=Sigma)
        K = dist.pairwise(X, Y)**2
        K *= -gamma_
        np.exp(K, K)    # exponentiate K in-place
        return K

    return SVC(gamma=gamma, C=C, kernel=mahalanobis_linear_kernel,class_weight=class_weight)  


