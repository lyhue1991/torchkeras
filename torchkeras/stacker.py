import numpy as np
import pandas as pd 
import datetime 
from tqdm.auto  import tqdm 

def printlog(info: str) -> None:
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 8 + "%s" % now_time)
    print(info + '...\n\n')

def prettydf(df):
    from tabulate import tabulate
    return tabulate(df,headers=df.columns, tablefmt="pretty")

def process_data(df, yhat_col = 'yhat', 
                 y_col = 'y', 
                 id_col = 'id', 
                 model_col = 'model'):
    for col in [yhat_col,id_col,model_col]:
        assert col in df.columns

    dfdata = df.pivot_table(index=  id_col, 
                                 columns = model_col, 
                                 values = yhat_col)
    dfdata.columns.name = None
    if y_col in df.columns:
        dflabels = df.pivot_table(index = id_col, columns = model_col, values = y_col)
        assert np.all(dflabels.max(axis=1) == dflabels.min(axis=1))
        labels = dflabels.mean(axis=1)
        assert np.all(dfdata.index==labels.index)
        dfdata[y_col] = labels 
    return dfdata 

class Stacker:
    def __init__(self, model_names, 
                 score_fn, 
                 greater_is_better = True
                ):
        
        self.model_names = model_names
        self.score_fn = score_fn 
        self.greater_is_better = greater_is_better 
        
    def fit_greedy_stacking(self, dfstack, num_models = None, num_warmup = None):
        
        assert 'y' in dfstack.columns, 'dfstack should with a label column  y'

        for name in self.model_names:
            assert name in dfstack.columns, f'{name} not in dfstack.columns.'
    
        if num_models is None:
            num_models = len(self.model_names)*200 
        if num_warmup is None:
            num_warmup = len(self.model_names)*5

        dfstack = dfstack.copy()
        
        printlog('step1: sort base models by score ...')
        
        scores = [self.score_fn(dfstack['y'],dfstack[name]) for name in self.model_names]
        dfscores = pd.DataFrame({'model': self.model_names, 'score': scores})
        dfscores = dfscores.sort_values('score',
                    ascending= False if self.greater_is_better else True).copy()
        
        dfscores.index = range(len(dfscores))
        print(prettydf(dfscores))
        
        
        dfscores['weights'] = [0 for _ in dfscores['model']]
        sorted_models = dfscores['model'].tolist()
        dfstack = dfstack[sorted_models+['y']]
        
        loop = tqdm(range(num_models),total = num_models)
        
        printlog('step2: start greedy stacking...')
        
        for i in loop:
            
            if i==0:
                cur_yhat = dfstack[sorted_models[0]].tolist()
                cur_score = dfscores['score'].iloc[0]
                dfscores.loc[0,'weights'] = 1
                loop.set_postfix(**{'i':i,'score':cur_score})
        
            elif i+1<=num_warmup:
                k = i%(len(self.model_names))
                dfscores.loc[k,'weights'] = 1 + dfscores['weights'].loc[k]
                cur_yhat = (dfstack[sorted_models].values)@(dfscores['weights'].values)/dfscores['weights'].sum()
                cur_score = self.score_fn(dfstack['y'],cur_yhat)
                loop.set_postfix(**{'i':i,'score':cur_score})
            else:
                for j in range(len(self.model_names)):
                    maybe_weight = dfscores['weights'].tolist()
                    maybe_weight[j] = maybe_weight[j] + 1 
                    maybe_weight = np.array(maybe_weight)
                    maybe_yhat = (dfstack[sorted_models].values)@maybe_weight/maybe_weight.sum()
                    maybe_score = self.score_fn(dfstack['y'],maybe_yhat)
                    
                    if (self.greater_is_better and maybe_score>=cur_score) or (
                        self.greater_is_better==False and maybe_score<=cur_score):
                        cur_yhat = maybe_yhat
                        cur_score = maybe_score
                        dfscores.loc[j,'weights'] = dfscores['weights'][j]+1
                        loop.set_postfix(**{'i':i,'score':cur_score})
                        break
                else:
                    loop.set_postfix(**{'i':i,'score':cur_score})
                    loop.close()
                    print('could not get better score, early stopping...')
                    break 
        
        printlog('step3: save result...')
        dfstack['yhat'] = cur_yhat
        print('best_score = ', cur_score)
        
        total_weights = dfscores['weights'].sum()
        dfensemble = pd.DataFrame({'model':['GreedyStacking'],
                                 'score':[cur_score],'weights':[total_weights]})
        
        dfsummary = pd.concat([dfscores,dfensemble],ignore_index=True)
        dfsummary['weight_ratio'] = np.round(dfsummary['weights']/total_weights,3)
        
        print(prettydf(dfsummary))
        
        self.dfscores = dfscores
        self.dfsummary = dfsummary
        self.sorted_models = sorted_models
        return dfstack 

    def fit_optuna_stacking(self, dfstack, n_trials=500, timeout=1200):
    
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        dfstack = dfstack.copy()
        
        printlog('step1: sort base models by score ...')
        scores = [self.score_fn(dfstack['y'],dfstack[name]) for name in self.model_names]
        dfscores = pd.DataFrame({'model': self.model_names, 'score': scores})
        dfscores = dfscores.sort_values('score',
                    ascending= False if self.greater_is_better else True) 
        dfscores.index = range(len(dfscores))
        sorted_models = dfscores['model'].tolist()
        print(prettydf(dfscores))
        
        
        def objective(trial):
            weights_dict = {name: trial.suggest_int(name, 1, 100) for name in sorted_models}
            weights = np.array([weights_dict[name] for name in sorted_models])
            yhat = (dfstack[sorted_models].values)@weights/(weights.sum())
            score = self.score_fn(dfstack['y'], yhat)
            return score
    
        study = optuna.create_study(
            direction="maximize" if self.greater_is_better else "minimize",
            study_name="optuna_stacking"
        )
    
        printlog('step2: start optuna search ...')
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    
        printlog('step3: save result...')
        best_weights = study.best_params
        best_score = study.best_value
        
        weights = np.array([best_weights[name] for name in sorted_models])
        total_weights = weights.sum()
        stack_yhat = (dfstack[sorted_models].values)@weights/(weights.sum())
    
        dfstack['yhat'] = stack_yhat 
        print('best_score = ', best_score)
    
        dfscores['weights'] = weights
        
        dfensemble = pd.DataFrame({'model':['OptunaStacking'],'score':[best_score],'weights':[total_weights]})
        dfsummary = pd.concat([dfscores,dfensemble],ignore_index=True)
        dfsummary['weight_ratio'] = np.round(dfsummary['weights']/total_weights,3)
        
        print(prettydf(dfsummary))

        self.dfscores = dfscores
        self.dfsummary = dfsummary
        self.sorted_models = sorted_models
        return dfstack 
        
    
    def predict(self,dftest):
        dftest = dftest.copy()
        test_yhat = (dftest[self.sorted_models].values)@(self.dfscores['weights'].values
                    )/self.dfscores['weights'].sum()
        dftest['yhat'] = test_yhat
        if 'y' in dftest.columns:
            test_score = self.score_fn(dftest['y'],test_yhat)
            print('test_score = ', test_score)
        return dftest
