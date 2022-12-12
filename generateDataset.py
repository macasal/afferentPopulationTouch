import numpy as np
import scipy
import pickle
import warnings
warnings.simplefilter
import random 
import os
import touchsim as ts
import copy

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


file = '/home/neutouch/afferentProject/data/dataset/indexes_40t.pkl'
file = open(file, "rb")
index = pickle.load(file)
file.close()

trainIndex = index['train']
testIndex = index['test']

poisson = None

destfolder= '/home/neutouch/afferentProject/data/dataset/' 
# =============================================================================
# initial parameters
# =============================================================================
tmin = 0
tmax = data['duration'] #if you want a different time window, change

trainFrac = 0.25
affclass =['SA1']
tbin = 0.002

# =============================================================================
# change density
# =============================================================================
d = [30,40,50,60,70,80,90,100,120,140]


for den in d:
    file = '/home/neutouch/afferentProject/data/dataset/dataApr21_40t.pkl'
    file = open(file, "rb")
    data = pickle.load(file)
    file.close()
    
    D0 = 140
    D={}
    D['SA1'] = den 
    D['RA'] = den
    D['PC'] =den
    
    n={}
    n['SA1'] = len(np.where(data['affClass'] == 'SA1')[0])
    n['RA'] = len(np.where(data['affClass'] == 'RA')[0])
    n['PC'] = len(np.where(data['affClass'] == 'PC')[0])
    
    index={}
    index['SA1']=np.rint(np.arange(0,n['SA1'],D0/D['SA1'] )).astype('int')
    index['RA']=np.rint(np.arange(0,n['RA'],D0/D['RA'])).astype('int')
    index['PC']=np.rint(np.arange(0,n['PC'],D0/D['PC'])).astype('int')
    
    for a in ['SA1', 'RA', 'PC']:
        if index[a][len(index[a])-1]==n[a]:
            index[a][len(index[a])-1] -=1
    i_SA1 = np.where(data['affClass'] == 'SA1')[0][index['SA1']]
    i_RA = np.where(data['affClass'] == 'RA')[0][index['RA']]
    i_PC = np.where(data['affClass'] == 'PC')[0][index['PC']]
    
    index_D = np.append(np.append(i_PC, i_RA), i_SA1)
    # =============================================================================
    # prepare data
    # =============================================================================
    indexes = np.empty([0])
    
    
    if affclass=='all':
        affclass =['SA1', 'RA', 'PC']
                
    indexesClass = np.empty([0])
    
    if affclass == ['SA1', 'RA', 'PC']:
        indexesClass = np.arange(data['nAfferents'])
    else:
        for ac in affclass: 
            indexesClass = np.append(indexesClass, np.where(data['affClass'] == ac))
        indexesClass = indexesClass.astype(int)
    
    indexes = np.intersect1d(indexesClass, index_D)
    
    
    for tr in range(len(data['r'])):
        data['r'][tr] = [data['r'][tr][i] for i in indexes]
    if poisson:    
        poisson['inh'] = poisson['inh'][:, indexes]
        poisson['exc'] = poisson['exc'][:, indexes]
    data['region'] = data['region'][indexes]
    data['affClass'] = data['affClass'][indexes]
    data['location'] = data['location'][indexes]
    data['nAfferents'] = len(indexes)
    
    
    print('afferent considered' + str(len(indexes)))
    
    [Rtrain, Rtest, labelsTrain, labelsTest] = generateR(data, poisson = poisson, tbin = tbin, trainFrac = trainFrac, 
                                                         affclass = affclass, tmin=tmin, tmax=tmax, trainIndex = trainIndex, testIndex=testIndex)
    T = int(Rtrain.shape[0]/len(indexes))
    
    
    
    results={}
    results['densityAfferents'] = {}
    results['densityAfferents'][affclass[0]] = den
    results['affclass'] = affclass
    results['Rtrain'] = Rtrain
    results['Rtest'] = Rtest
    results['label']={}
    results['label']['labelTest'] = labelsTest
    results['label']['labelTrain'] = labelsTrain
    results['stimPars'] = data['stimPars']
    results['sIndex'] = data['sIndex']
    results['pars'] = ['circleRadius','indentation_ramp','indentation_sin','frequency', 'location','ramp_len', 'initialPad']
    
    
    fileData = destfolder + affclass[0] +'/R_40t_'+ str(affclass[0])+'_D'+str(den)+'.pkl'
    res_file = open(fileData, "wb")
    pickle.dump(results, res_file)
    res_file.close()
        
        
        
        
        
        
        
