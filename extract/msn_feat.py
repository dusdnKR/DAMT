import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
from nilearn import connectome
from numpy import inf
from scipy.spatial import distance
import warnings
warnings.filterwarnings(action='ignore')

def zscore(df):
    scaler = StandardScaler()
    df = df.T
    df = scaler.fit_transform(df)
    
    return df

def MSN(df, save_path):
    
    CT = pd.DataFrame(zscore(thick_df)) # cortical thickness
    MC = pd.DataFrame(zscore(meancurv_df)) # mean curvature
    Vol = pd.DataFrame(zscore(vol_df)) # gray matter volume
    SD = pd.DataFrame(zscore(sulc_df)) # sulcal depth
    SA = pd.DataFrame(zscore(area_df)) # surface area

    for i in range(0, len(SA.T)):
        
        print(ids[i])
        df = pd.concat([CT[i], MC[i], Vol[i], SD[i], SA[i]], axis=1).T.corr()
        val =  pd.concat([CT[i], MC[i], Vol[i], SD[i], SA[i]], axis=1)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        sio.savemat(os.path.join(save_path, f"{ids[i]}.mat"), {'value' : val.values, 'connectivity' : df.values})