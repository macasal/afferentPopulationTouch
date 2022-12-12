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
        
        
   def generateR(data, **args):
    # input arguments: 
    #     data         : dictionary with spike trains and information about spike trains (required keys: 'r', 
    #                       'nTrials', 'affClass', 'sIndex', 'duration')
    #     poisson     : dictionary with inhibitory and excitatory spike train to sum on the spikes
    #     tbin        : length of time bins (default: 0.01 s)
    #     trainFrac   : fraction of training trials (default: 50%)
    #     affclass  : classes of afferents to consider (default: all)
    #     tmin :      istant from which start considering spikes
    #     tmax :      instant from which stop considering spikes
    # output: 
    #     Rtrain      : training data matrix (#neurons * #bins X #training trials)
    #     Rtest       : test data matrix (#neurons * #bins X #test trials)
    #     labelaTrain : labels of the training trails
    #     labelsTest  : labels of the test trials
    poisson = args.get('poisson', None)
    trainFrac = args.get('trainFrac', 0.5)
    tbin = args.get('tbin', 0.01)
    tmin = args.get('tmin', 0)
    tmax = args.get('tmax', data['duration'])
    trainIndex = args.get('trainIndex', [])
    testIndex = args.get('testIndex', [])

    N = data['nAfferents']

    bins = np.arange(tmin,tmax+tbin,tbin)
    T = len(bins)-1
                
    R = np.empty((0,N*T))
    for tr in range(len(data['r'])):
        Rtrial = np.empty((1,0))
        for n in range(N):
            spikes = data['r'][tr][n]
            if poisson:
                spikes = np.append(data['r'][tr][n], poisson['exc'][tr,n])
                inh = poisson['inh'][tr,n]
            Raff = np.zeros((1,T))
            for s in range(len(spikes)):
                if spikes[s]>= tmin and spikes[s]<=tmax:
                    t = int(spikes[s]/tbin-tmin/tbin)
                    if t == T: 
                        t = T-1
                    Raff[:,t]+=1
            if poisson:
                for i in range(len(inh)):
                    if inh[i]>tmin and inh[i]<tmax:
                        t = int(inh[i]/tbin-tmin/tbin)
                        if t == T: 
                            t = T-1
                        Raff[:,t]-=1
                        if Raff[:,t] < 0:
                            Raff[:,t] = 0
            Rtrial = np.append(Rtrial, Raff, axis=1)
        print(tr)
        R = np.append(R,Rtrial, axis=0)   
    
        
    classes = int(len(np.unique(data['sIndex'])))
    labelsTrain = np.empty([0, int(data['nTrials']*trainFrac)])
    labelsTest = np.empty([0, int(data['nTrials']*(1-trainFrac))])
    Rtrain = np.empty((0,N*T))
    Rtest = np.empty((0,N*T))
    if trainIndex ==[] or testIndex==[]:
        for c in range(classes):
            randomIndex = np.arange(c*data['nTrials'],c*data['nTrials']+data['nTrials'])
            random.shuffle(randomIndex)
            trainIndex = randomIndex[0:int(data['nTrials']*trainFrac)]
            testIndex = randomIndex[int(data['nTrials']*trainFrac):]
            Rtrain = np.append(Rtrain, R[trainIndex,:], axis=0)
            Rtest = np.append(Rtest, R[testIndex,:], axis=0)
            labels = data['sIndex'][c*data['nTrials']:c*data['nTrials']+data['nTrials']]
            print(labels)
            labelsTrain = np.append(labelsTrain, labels[0:int(data['nTrials']*trainFrac)])
            labelsTest = np.append(labelsTest, labels[int(data['nTrials']*trainFrac):])
        trainIndex = np.arange(int(len(labelsTrain)))
        random.shuffle(trainIndex)
        testIndex = np.arange(int(len(labelsTest)))
        random.shuffle(testIndex)
        labelsTrain = labelsTrain[trainIndex] 
        labelsTest = labelsTest[testIndex] 
        labelsTrain = labelsTrain.astype(int)
        labelsTest = labelsTest.astype(int)
        Rtrain = Rtrain[trainIndex]
        Rtest = Rtest[testIndex]
    else:
        print('use existing train-test indexes')
        Rtrain = R[trainIndex,:]
        Rtest = R[testIndex, :]
        labels = np.empty([0])
        for c in range(classes):
           labels = np.append(labels,data['sIndex'][c*data['nTrials']:c*data['nTrials']+data['nTrials']],axis=0 )
        labelsTrain = labels[trainIndex]   
        labelsTest = labels[testIndex]
        

    Rtrain = Rtrain.T
    Rtest = Rtest.T
   
   
    return Rtrain, Rtest, labelsTrain, labelsTest

        
        
        
        
        
        
