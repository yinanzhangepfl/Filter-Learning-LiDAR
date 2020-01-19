# Import basic packages
import numpy as np
# import plot packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.collections as coll
from mpl_toolkits.mplot3d import Axes3D


# Set constants
color_list = [(1, 1, 125/225), 
              (0, 1, 1), 
              (1, 1, 1), 
              (1, 1, 0), 
              (0, 1, 125/225),
              (0, 0, 1), 
              (0, 125/225, 1), 
              (125/255, 1, 0), 
              (0, 1, 0)]

color_dict = {0: 'rgb(255, 255, 125)', 
              1: 'rgb(0, 255, 255)', 
              2: 'rgb(255, 255, 255)', 
              3: 'rgb(255, 255, 0)', 
              4: 'rgb(0, 255, 125)', 
              5: 'rgb(0, 0, 255)', 
              6: 'rgb(0, 125, 255)', 
              7: 'rgb(125, 255, 0)', 
              8: 'rgb(0, 255, 0)'}

COLOR_LABELS = {0: [255, 255, 125, 255], 
                1: [0, 255, 255, 255], 
                2: [255, 255, 255, 255], 
                3: [255, 255, 0, 255], 
                4: [0, 255, 125, 255], 
                5: [0, 0, 255, 255], 
                6: [0, 125, 255, 255], 
                7: [125, 255, 0, 255], 
                8: [0, 255, 0, 255]}

legend_list= ['Powerline', 
              'Low Vegetation', 
              'Impervious surfaces', 
              'Car', 
              'Fence/Hedge', 
              'Roof', 
              'Facade', 
              'Shrub', 
              'Tree']

    
def plot_vaihingen_2D(cloud, id_highlight=None, ax=None, label_high='OUTLIERS', **kwargs):
    """
    3d visualization using matplotlib.
    
    Attributes:
        - cloud (array, Nx4) : Point cloud expressed as [X | Y | Z | label] matrix of size Nx4
        - id_highlight       : The id of highlight
        - ax                 : Axes object to plot 
        - label_high         : The label of highlight
    """  
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        _ax = fig.add_subplot(111)
    else:
        _ax = ax
    leg_ = []
    _ax.set_facecolor((0, 0, 0))
    _ax.set_xlabel('x')
    _ax.set_ylabel('y')
    
    if 'label' in cloud.columns:
        for i, tag in enumerate(cloud['label'].unique()):
            cloud_tag = cloud[cloud.label == tag]
            if tag == -1:
                leg_.append(_ax.scatter(cloud_tag.x, cloud_tag.y, c=cloud_tag.z, cmap='terrain', label='data', s=2))
            else:
                leg_.append(_ax.scatter(cloud_tag.x, cloud_tag.y, c=np.array(COLOR_LABELS[tag])/255, label=tag, s=2))
    else:
        leg_.append(_ax.scatter(cloud.x, cloud.y, c=cloud.z, cmap='terrain', label='data', s=2))
    if id_highlight is not None:
        leg_.append(_ax.scatter(cloud.iloc[id_highlight].x, cloud.iloc[id_highlight].y, s=2, c=np.array([1, 0, 0, 1]), label=label_high) )
    if False:
        leg_.append( mpatches.Patch(color='black', label='Background') )
        lgnd = plt.legend(handles=leg_, numpoints=1, fontsize=10, loc=1)
        #change the marker size manually only for scatter points
        for handle in lgnd.legendHandles:
            if isinstance(handle, coll.PathCollection):
                handle.set_sizes([30])
                
    if kwargs is not None and 'title' in kwargs:
        _ax.set_title(kwargs['title'])
        
    if ax is None:
        plt.show()
        
        
def plot_vaihingen_2D_zones(cloud, zones=None, zones_label=None, ax=None, **kwargs):
    """
    Plot vaihingen 2D zones.
    
    Attributes:
        - cloud (array, Nx4) : Point cloud expressed as [X | Y | Z | label] matrix of size Nx4
        - zones              : Vaihingen zones
        - zones_label        : The labels of zones
        - ax                 : Axes object to plot 
    """  
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        _ax = fig.add_subplot(111)
    else:
        _ax = ax
    leg_ = []
    _ax.set_facecolor((0, 0, 0))
    _ax.set_xlabel('x')
    _ax.set_ylabel('y')
    
    if 'label' in cloud.columns:
        for i, tag in enumerate(cloud['label'].unique()):
            cloud_tag = cloud[cloud.label == tag]
            leg_.append( _ax.scatter(cloud_tag.x, cloud_tag.y, c=np.array(COLOR_LABELS[tag])/255, label=tag, s=2))
    else:
        leg_.append(_ax.scatter(cloud.x, cloud.y, c=cloud.z, cmap='terrain', label='data', s=2))
    for i, (zone, zone_label) in enumerate(zip(zones, zones_label)):
        leg_.append( _ax.add_patch(mpatches.Rectangle(
            (int(zone[0]), int(zone[2])), int(zone[1])-int(zone[0]), int(zone[3])-int(zone[2]), 
            edgecolor=zone_label[0], facecolor=zone_label[0], alpha=0.4, linewidth=3, label=zone_label[1])))

    if True:
        leg_.append( mpatches.Patch(color='black', label='Background') )
        lgnd = plt.legend(handles=leg_, numpoints=1, fontsize=10)
        #change the marker size manually only for scatter points
        for handle in lgnd.legendHandles:
            if isinstance(handle, coll.PathCollection):
                handle.set_sizes([30])
                
    if kwargs is not None and 'title' in kwargs:
        _ax.set_title(kwargs['title'],  fontsize=16)
        
    if ax is None:
        plt.show()
        

def plot_graph_3D(patch, figsize=(16, 8), marksize=2, markerscale=2, ax=None, **kwargs):
    """
    3d visualization using matplotlib.
    
    Attributes:
        - patch              : A patch cropped from the cloud
        - figsize            : Size of the figure  
        - marksize           : Size of the Marker 
        - markerscale        : Size of legend marker
        - ax                 : Axes object to plot 
    """  
    # Fetch parameters
    if kwargs is not None:
        xRange = kwargs['xRange'] if 'xRange' in kwargs else None
        yRange = kwargs['yRange'] if 'yRange' in kwargs else None
        
    # Initalize the figure    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        _ax = plt.gca(projection='3d')
    else:
        _ax = ax
    
    # Generate points
    dots = []
    for i in range(0, 9):
        x, y, z = patch.loc[patch['label'] == i].x, patch.loc[patch['label'] == i].y, patch.loc[patch['label'] == i].z
        dots.append(_ax.scatter(x, y, z, s=marksize, color=[color_list[i]], marker='o')) 
        
    # Set the axes
    _ax.set_xlabel('X')
    _ax.set_ylabel('Y')
    if xRange is not None:
        _ax.set_xlim(xRange)
    if yRange is not None:
        _ax.set_ylim(yRange)
        
    # Set title and legend
    _ax.grid()
    _ax.set_title('3d point cloud with labels')
    legend = _ax.legend(dots, legend_list, markerscale=markerscale, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    legend.get_frame().set_facecolor('grey')
    
def plot_graph_2D(patch, figsize=None, marksize=2, markerscale=1, ax=None, mode='point_cloud', signal=None, mask=None, **kwargs):
    """
    2d visualization using matplotlib.
    
    Attributes:
        - patch              : A patch cropped from the cloud
        - figsize            : Size of the figure  
        - marksize           : Size of the marker
        - markerscale        : Size of legend marker
        - ax                 : Axes object to plot 
        - mode               : The mode of graph ploting
        - signal             : Signal
        - mask               : Mask
    """
    dots = []
    
    # Fetch parameters
    if kwargs is not None:
        xRange = kwargs['xRange'] if 'xRange' in kwargs else None
        yRange = kwargs['yRange'] if 'yRange' in kwargs else None
        title  = kwargs['title']  if 'title' in kwargs else None
        blackBack = kwargs['blackBack'] if 'blackBack' in kwargs else True

    # Initalize the figure 
    if ax is None:
        fig = plt.figure(figsize=figsize)
        _ax = plt.gca()
    else:
        _ax = ax
        
    # Generate points
    if mode == 'point_cloud':
        for i in range(8, -1, -1):
            x, y = patch.loc[patch['label'] == i].x, patch.loc[patch['label'] == i].y
            dots.append(_ax.scatter(x, y, s=marksize, color=[color_list[i]], marker='o')) 
        _ax.set_title('2d point cloud with labels')
        if blackBack:
            _ax.patch.set_facecolor((0,0,0))
        legend = _ax.legend(dots, reversed(legend_list), markerscale=markerscale, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        legend.get_frame().set_facecolor('grey')
        
    if mode == 'signal_cmp':
        sc =_ax.scatter(patch[mask].x, patch[mask].y, s=0.2, c=signal, marker='o', cmap='nipy_spectral', norm=mpl.colors.LogNorm())
        cbar = plt.colorbar(sc , ax=_ax)
    
    # Set the axes
    _ax.set_xlabel('X')
    _ax.set_ylabel('Y')
    if xRange is not None:
        _ax.set_xlim(xRange)
    if yRange is not None:
        _ax.set_ylim(yRange)
    
    # Set the title
    if title is not None:
        _ax.set_title(kwargs['title'],  fontsize=16)