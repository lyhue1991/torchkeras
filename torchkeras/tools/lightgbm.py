from torchkeras import VLog
class VLogCallback:
    def __init__(self, num_boost_round, 
                 monitor_metric='val_loss',
                 monitor_mode='min'):
        self.order = 20
        self.num_boost_round = num_boost_round
        self.vlog = VLog(epochs = num_boost_round, monitor_metric = monitor_metric, 
                         monitor_mode = monitor_mode)

    def __call__(self, env) -> None:
        metrics = {}
        for item in env.evaluation_result_list:
            if len(item) == 4:
                data_name, eval_name, result = item[:3]
                metrics[data_name+'_'+eval_name] = result
            else:
                data_name, eval_name = item[1].split()
                res_mean = item[2]
                res_stdv = item[4]
                metrics[data_name+'_'+eval_name] = res_mean
        self.vlog.log_epoch(metrics)
        
if __name__=='__main__':
    import datetime
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from torchkeras.tools.lightgbm import VLogCallback 

    def printlog(info):
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n"+"=========="*8 + "%s"%nowtime)
        print(info+'...\n\n')


    #================================================================================
    printlog("step1: reading data...")


    breast = datasets.load_breast_cancer()
    df = pd.DataFrame(breast.data,columns = [x.replace(' ','_') for x in breast.feature_names])
    df['label'] = breast.target
    df['mean_radius'] = df['mean_radius'].apply(lambda x:int(x))
    df['mean_texture'] = df['mean_texture'].apply(lambda x:int(x))
    dftrain,dftest = train_test_split(df)

    categorical_features = ['mean_radius','mean_texture']
    lgb_train = lgb.Dataset(dftrain.drop(['label'],axis = 1),label=dftrain['label'],
                            categorical_feature = categorical_features)

    lgb_valid = lgb.Dataset(dftest.drop(['label'],axis = 1),label=dftest['label'],
                            categorical_feature = categorical_features,
                            reference=lgb_train)


    #================================================================================
    printlog("step2: setting parameters...")

    boost_round = 50                   
    early_stop_rounds = 10

    params = {
        'boosting_type': 'gbdt',
        'objective':'binary',
        'metric': ['auc'], #'l2'
        'num_leaves': 15,   
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'early_stopping_round':5
    }


    #================================================================================
    printlog("step3: training model...")

    result = {}

    vlog_cb = VLogCallback(boost_round, monitor_metric = 'val_auc', monitor_mode = 'max')
    vlog_cb.vlog.log_start()

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round= boost_round,
                    valid_sets=(lgb_valid, lgb_train),
                    valid_names=('val','train'),
                    callbacks = [lgb.record_evaluation(result),
                                 vlog_cb]
                   )

    vlog_cb.vlog.log_end()

 
    #================================================================================
    printlog("step4: evaluating model ...")

    y_pred_train = gbm.predict(dftrain.drop('label',axis = 1), num_iteration=gbm.best_iteration)
    y_pred_test = gbm.predict(dftest.drop('label',axis = 1), num_iteration=gbm.best_iteration)

    print('train accuracy: {:.5} '.format(accuracy_score(dftrain['label'],y_pred_train>0.5)))
    print('valid accuracy: {:.5} \n'.format(accuracy_score(dftest['label'],y_pred_test>0.5)))

    lgb.plot_metric(result,metric='auc')
    lgb.plot_importance(gbm,importance_type = "gain")



    #================================================================================
    printlog("step5: saving model ...")

    model_dir = "gbm.model"
    print("model_dir: %s"%model_dir)
    gbm.save_model("gbm.model")
    printlog("task end...")
