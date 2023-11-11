import pandas as pd 
import numpy as np 
from torchkeras.pbar import is_jupyter 

class VLogCallback:
    def __init__(self, monitor_metric='val_Accuracy', monitor_mode='max', figsize = (8,6), save_path='history.png'):
        import matplotlib.pyplot as plt
        self.in_jupyter = is_jupyter()
        self.plt = plt
        self.figsize = figsize
        self.metric = monitor_metric
        self.mode = monitor_mode
        self.save_path = save_path
        x_bounds = [0, 10]
        title = f'best {self.metric} = ?'
        self.update_graph(None, title=title, x_bounds = x_bounds)
        
    def get_dfhistory(self, info):
        from copy import deepcopy
        if not hasattr(info,'metrics'):
            return pd.DataFrame() 
        
        dic = deepcopy(info.metrics) 
        if 'learn' in dic:
            dic['train'] = dic['learn']
            dic.pop('learn')

        if 'validation' in dic:
            dic['val'] = dic['validation']
            dic.pop('validation')

        dfhis_train = pd.DataFrame(dic['train']) 
        dfhis_train.columns = ['train_'+x for x in dfhis_train.columns]

        dfhis_val = pd.DataFrame(dic['val']) 
        dfhis_val.columns = ['val_'+x for x in dfhis_val.columns]
        dfhistory = dfhis_train.join(dfhis_val)
        dfhistory['iteration'] = range(1,len(dfhistory)+1)
        
        return dfhistory 

    def after_iteration(self, info):
        dfhistory = self.get_dfhistory(info)
        n = len(dfhistory)
        x_bounds = [dfhistory['iteration'].min(), 10+(n//10)*10]
        title = self.get_title(info)
        self.update_graph(info, title = title, x_bounds = x_bounds)
        return True
        
    def get_best_score(self, info):
        dfhistory = self.get_dfhistory(info)
        arr_scores = dfhistory[self.metric]
        best_score = np.max(arr_scores) if self.mode=="max" else np.min(arr_scores)
        best_iteration = dfhistory.loc[arr_scores==best_score,'iteration'].tolist()[0]
        return (best_iteration, best_score)
        
    def get_title(self,  info):
        best_iteration,best_score = self.get_best_score(info)
        title = f'best {self.metric} = {best_score:.4f} (@iteration {best_iteration})'
        return title

    def update_graph(self, info, title=None, x_bounds=None, y_bounds=None):
        from IPython.display  import display 
        if not hasattr(self, 'graph_fig'):
            self.graph_fig, self.graph_ax = self.plt.subplots(1, figsize=self.figsize)
            self.graph_out = display(self.graph_ax.figure, display_id=True)
        self.graph_ax.clear()
        dfhistory = self.get_dfhistory(info)
        iterations = dfhistory['iteration'] if 'iteration' in dfhistory.columns else []
        
        m1 = "train_"+self.metric.replace('val_','')
        if  m1 in dfhistory.columns:
            train_metrics = dfhistory[m1]
            
            self.graph_ax.plot(iterations,train_metrics,'bo--',label= m1)

        m2 = 'val_'+self.metric.replace('val_','')
        if m2 in dfhistory.columns:
            val_metrics = dfhistory[m2]
            self.graph_ax.plot(iterations,val_metrics,'co-',label =m2)


        self.graph_ax.set_xlabel("iteration")
        self.graph_ax.set_ylabel(self.metric.replace('val_',''))  
        if title:
             self.graph_ax.set_title(title)
        if m1 in dfhistory.columns or m2 in dfhistory.columns or self.metric in dfhistory.columns:
            self.graph_ax.legend(loc='best')
            
        if len(iterations)>0:
            best_iteration, best_score = self.get_best_score(info)
            self.graph_ax.plot(best_iteration,best_score,'r*',markersize=15)

        if x_bounds is not None: self.graph_ax.set_xlim(*x_bounds)
        if y_bounds is not None: self.graph_ax.set_ylim(*y_bounds)
        
        if self.in_jupyter:
            self.graph_out.update(self.graph_ax.figure)
        self.graph_fig.savefig(self.save_path)
        
        self.plt.close();
        
if __name__ =='__main__':
    from IPython.display import display 

    import datetime,json
    import numpy as np
    import pandas as pd
    import catboost as cb 
    from catboost.datasets import titanic
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedKFold

    from sklearn.metrics import f1_score,roc_auc_score,accuracy_score
    import plotly.graph_objs as go 
    import plotly.express as px 

    from torchkeras.tools.catboost import VLogCallback 

    def printlog(info):
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n"+"=========="*8 + "%s"%nowtime)
        print(info+'...\n\n')


    #================================================================================
    printlog("step1: preparing data...")

    dfdata,dftest = titanic()

    display(dfdata.head()) 

    label_col = "Survived"

    dfnull = pd.DataFrame(dfdata.isnull().sum(axis=0),columns = ["null_cnt"]).query("null_cnt>0")
    print("null_features:") 
    print(dfnull)

    dfdata.fillna(-9999, inplace=True)
    dftest.fillna(-9999, inplace=True)


    cate_cols = [x for x in dfdata.columns 
                 if dfdata[x].dtype not in [np.float32,np.float64] and x!=label_col]
    for col in cate_cols:
        dfdata[col] = pd.Categorical(dfdata[col]) 
        dftest[col] = pd.Categorical(dftest[col]) 

    dftrain,dfvalid = train_test_split(dfdata, train_size=0.75, random_state=42)
    Xtrain,Ytrain = dftrain.drop(label_col,axis = 1),dftrain[label_col]
    Xvalid,Yvalid = dfvalid.drop(label_col,axis = 1),dfvalid[label_col]
    cate_cols_indexs = np.where(Xtrain.columns.isin(cate_cols))[0]


    pool_train = cb.Pool(data = Xtrain, label = Ytrain, cat_features=cate_cols)
    pool_valid = cb.Pool(data = Xvalid, label = Yvalid, cat_features=cate_cols)



    #================================================================================
    printlog("step2: setting parameters...")

    iterations = 1000
    early_stopping_rounds = 200

    params = {
        'learning_rate': 0.05,
        'loss_function': "Logloss",
        'eval_metric': "Accuracy",
        'depth': 6,
        'min_data_in_leaf': 20,
        'random_seed': 42,
        'logging_level': 'Silent',
        'use_best_model': True,
        'one_hot_max_size': 5,  
        'boosting_type':"Ordered", 
        'max_ctr_complexity': 2, 
        'nan_mode': 'Min' 
    }


    #================================================================================
    printlog("step3: training model...")


    model = cb.CatBoostClassifier(
        iterations = iterations,
        early_stopping_rounds = early_stopping_rounds,
        train_dir='catboost_info/',
        **params
    )

    vlog_cb = VLogCallback(monitor_metric='val_Accuracy',monitor_mode='max')

    model.fit(
        pool_train,
        eval_set=pool_valid,
        plot=False,
        callbacks=[vlog_cb]
    )


    #================================================================================
    printlog("step4: evaluating model ...")


    y_pred_train = model.predict(Xtrain)
    y_pred_valid = model.predict(Xvalid)

    train_score = f1_score(Ytrain,y_pred_train)
    valid_score = f1_score(Yvalid,y_pred_valid)


    print('train f1_score: {:.5} '.format(train_score))
    print('valid f1_score: {:.5} \n'.format(valid_score))   


    #feature importance 
    dfimportance = model.get_feature_importance(prettified=True) 
    dfimportance = dfimportance.sort_values(by = "Importances").iloc[-20:]
    fig_importance = px.bar(dfimportance,x="Importances",y="Feature Id",title="Feature Importance")

    fig_importance.show() 

    #score distribution
    y_test_prob = model.predict_proba(dfvalid.drop(label_col,axis = 1))[:,-1]
    fig_hist = px.histogram(
        x=y_test_prob,color =dfvalid[label_col],  nbins=50,
        title = "Score Distribution",
        labels=dict(color='True Labels', x='Score')
    )
    fig_hist.show() 


    #================================================================================
    printlog("step5: using model ...")

    y_pred_test = model.predict(dftest)
    y_pred_test_prob = model.predict_proba(dftest)


    #================================================================================
    printlog("step6: saving model ...")

    model_dir = 'catboost_model'
    model.save_model(model_dir)
    model_loaded = cb.CatBoostClassifier()
    model.load_model(model_dir)
        