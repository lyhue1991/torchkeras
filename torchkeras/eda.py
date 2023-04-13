import datetime
import numpy as np 
import pandas as pd
from scipy import stats
from collections import Counter

from tqdm import tqdm
tqdm.pandas(desc="progress")

def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s'%nowtime)
    print(info+'\n')

# 相关性ks检验
def relativity_ks(labels,features):
    assert len(labels) == len(features)
    labels = np.array(labels)
    features = np.array(features)
    # 非数值特征将字符转换成对应序号
    if features.dtype is np.dtype('O'):
        features_notnan = set(features[~pd.isna(features)])
        features_notnan = [str(x) for x in features_notnan]
        dic = dict(zip(sorted(list(features_notnan)),range(0,len(features_notnan))))
        features = np.array([dic.get(x,-1) for x in features])
    else:
        features = features
    if set(labels) == {0,1}:  #二分类问题
        data_1 = features[labels > 0.5]
        data_0 = features[labels < 0.5]
    elif "int" in str(labels.dtype): #多分类问题
        most_label = Counter(labels).most_common(1)[0][0]
        data_0 = features[labels == most_label]
        data_1 = features[labels != most_label]
    else:  #回归问题
        mid = np.median(labels)
        data_1 = features[labels > mid]
        data_0 = features[labels <= mid ]
    result = stats.ks_2samp(data_1,data_0)
    return result[0]

# 同分布性ks检验
def stability_ks(data1,data2):
    data1 = np.array(data1)
    data2 = np.array(data2)
    features = np.concatenate((data1,data2))
    # 非数值特征将字符转换成对应序号
    if features.dtype is np.dtype('O'):
        features_notnan = set(features[~pd.isna(features)])
        features_notnan = [str(x) for x in features_notnan]
        dic = dict(zip(sorted(list(features_notnan)),range(0,len(features_notnan))))
        data1 = np.array([dic.get(x,-1) for x in data1])
        data2 = np.array([dic.get(x,-1) for x in data2])
    result = stats.ks_2samp(data1,data2)
    return result[0]

def pipeline(dftrain,dftest=pd.DataFrame(),label_col="label",language="Chinese"):
    """
    Examples:
    ---------
    >> from sklearn import datasets
    >> from sklearn.model_selection import train_test_split
    >> import pandas as pd 
    >> breast = datasets.load_breast_cancer()
    >> df = pd.DataFrame(breast.data,columns = breast.feature_names)
    >> df["label"] = breast.target
    >> dftrain,dftest = train_test_split(df,test_size = 0.3)
    >> dfeda = pipeline(dftrain,dftest)
    """
    
    print("start exploration data analysis...")
    printlog('step1: count features & samples...')
    
    if len(dftest)==0: 
        dftest = pd.DataFrame(columns = dftrain.columns) 
    assert label_col in dftrain.columns, 'train data should with label column!'
    assert all(dftrain.columns == dftest.columns), 'train data and test data should with the same columns!'
    print('train samples number : %d'%len(dftrain))
    print('test samples number : %d'%len(dftest))
    print('features number : %d\n'%(len(dftrain.columns) - 1))

    n_samples = len(dftrain)
    n_features = len(dftrain.T)

    dfeda = pd.DataFrame( np.zeros((n_features,8)),
        columns = ['not_nan_ratio','not_nan_zero_ratio','not_nan_zero_minus1_ratio',
                   'classes_count','most','relativity','cor','stability'])
    dfeda.index = dftrain.columns


    printlog('step2: evaluate not nan ratio...\n')
    dfeda['not_nan_ratio'] =  dftrain.count()/n_samples


    printlog('step3: evaluate not zero ratio...\n')
    dfeda['not_nan_zero_ratio'] = ((~dftrain.isna())&(~dftrain.isin([0,'0','0.0','0.00']))).sum()/n_samples


    printlog('step4: evaluate not negative ratio...\n')
    dfeda['not_nan_zero_minus1_ratio'] =  ((~dftrain.isna())&(~dftrain.isin(
                                   [0,'0','0.0','0.00',-1,-1.0,'-1','-1.0']))).sum()/n_samples


    printlog('step5: evaluate classes count...\n')
    dfeda['classes_count'] = dftrain.progress_apply(lambda x:len(x.drop_duplicates()))


    printlog('step6: evaluate most value...\n')
    try:
        dfeda['most'] = dftrain.mode(dropna = False).iloc[0,:].T
    except:
        dfeda['most'] = dftrain.mode().iloc[0,:].T


    printlog('step7: evaluate relativity(ks)...\n')
    dfeda['relativity'] = dftrain.progress_apply(lambda x: relativity_ks(dftrain[label_col],x))
    
    printlog('step8: evaluate spearman cor...\n')
    dfeda['cor'] = dftrain.progress_apply(lambda x: dftrain[label_col].corr(x,method = 'spearman'))
    #reference: https://zhuanlan.zhihu.com/p/34717666 
    

    printlog('step9: evaluate stability...\n')
    if len(dftest)==0:
        dfeda['stability'] = np.nan
    else:
        dfeda['stability'] = dftrain.progress_apply(lambda x: 1-stability_ks(x,dftest[x.name]))
        
    
    printlog('tast end...\n\n')
    if language=="Chinese":
        dfeda_zh = dfeda.copy()
        dfeda_zh.columns = [u'非空率',u'非空非零率',u'非空非零非负1率',u'取值类别数',u'众数',
                            u'相关性ks',u'相关性cor',u'同分布性']  
        return dfeda_zh
    else:
        return dfeda