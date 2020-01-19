# Import basic packages
import numpy as np
import pandas as pd

labels = {0: 'Powerline', 
          1: 'Low Vegetation', 
          2: 'Impervious surfaces', 
          3: 'Car', 
          4: 'Fence/Hedge', 
          5: 'Roof', 
          6: 'Facade', 
          7: 'Shrub', 
          8: 'Tree' }

def load_cloud(path):
    """
    Load cloud points.
    
    Attributes:
        - path               : Path of cloud object
        
    Return:
        - cloud (array, Nx4) : Point cloud expressed as [X | Y | Z | label] matrix of size Nx4
    """
    strMargin = lambda s, l=50, sep=' ': int((l-len(s))/2)*'-' + sep + s + sep + (l-int((l-len(s))/2)-len(s))*'-'
    print(strMargin('Load Data'))
    print("Data Loading...")
    cloud_orig = pd.read_csv(path, sep = ' ', names = ['x', 'y', 'z', 'Intensity', 'return_number', 'num_returns', 'label'])
    cloud  = cloud_orig[['x', 'y', 'z', 'label']]
    print('=> Done.')
    
    print(strMargin('',sep='-'))
    return cloud

def cloud_preprocess(cloud):
    """
    Preprocess cloud point data.
    
    Attributes:
        - cloud (array, Nx4) : Point cloud expressed as [X | Y | Z | label] matrix of size Nx4

    Return:
        - newCloud           : Cloud object after processing
    """
    strMargin = lambda s, l=50, sep=' ': int((l-len(s))/2)*'-' + sep + s + sep + (l-int((l-len(s))/2)-len(s))*'-'
    print(strMargin('Data Pre-Processing'))
    
    # Check if there are duplicated points
    print("Drop duplicates...")
    newCloud   = cloud.copy()
    duplicated = newCloud.duplicated(keep='first')
    newCloud.drop_duplicates(keep='first', inplace=True)
    print('=> Duplicates #: {}, Ratio: {:.2f}% of the dataset.'.format(np.sum(duplicated), np.mean(duplicated)*100))
    
    # Re-center the data
    print("Re-center the data...")
    newCloud.x = newCloud.x - newCloud.x.min()
    newCloud.y = newCloud.y - newCloud.y.min()
    newCloud.z = newCloud.z - newCloud.z.min()
    print('=> Done.')
    
    print(strMargin('',sep='-'))
    return newCloud

def crop_patch(cloud, xRange, yRange, reIndex=False):
    """
    Crop a patch from cloud.
    
    Attributes:
        - cloud (array, Nx4) : Point cloud expressed as [X | Y | Z | label] matrix of size Nx4
        - xRange             : The range of X coordinate used to crop data
        - yRange             : The range of Y coordinate used to crop data
        - reIndex            : Whether to reset index
        
    Return:
        - patch              : A patch cropped from the cloud
    """
    strMargin = lambda s, l=50, sep=' ': int((l-len(s))/2)*'-' + sep + s + sep + (l-int((l-len(s))/2)-len(s))*'-'
    print(strMargin('Crop Patch'))
    
    # Crop the patch from the point cloud
    print("Crop the patch...")
    patch = cloud[np.logical_and(cloud.x >= xRange[0], cloud.x <= xRange[1])]
    patch = patch[np.logical_and(patch.y >= yRange[0], patch.y <= yRange[1])]
    
    # Reset index of patch
    if reIndex:
        patch = patch.reset_index(drop=True)
    print("=> Crop a patch from: x:[{}, {}], y:[{}, {}].".format(xRange[0], xRange[1], yRange[0], yRange[1]))
    print("=> Points left for patch: {}.".format(len(patch)))
    print(strMargin('',sep='-'))
    return patch