import pandas as pd
import torch
import datetime 
from tqdm.auto  import tqdm 

def printlog(info: str) -> None:
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 8 + "%s" % now_time)
    print(info + '...\n\n')

def prettydf(df):
    from tabulate import tabulate
    return tabulate(df,headers=df.columns, tablefmt="pretty")

def merge_state(state_list, weights=None):
    """
    Merges states from multiple checkpoints with optional weights

    Args:
        state_list (list): List of checkpoint paths or state dictionaries
        weights (list): List of weights corresponding to each state. If None, equal weights are assigned

    Returns:
        dict: Merged state dictionary
    """
    if weights is None:
        weights = [1 for _ in state_list]
    total = sum(weights)
    weights = [x / total for x in weights]

    if isinstance(state_list[0], str):
        state = {k: v * weights[0] for k, v in torch.load(state_list[0]).items()}
    else:
        state = {k: v * weights[0] for k, v in state_list[0].items()}

    for st, w in zip(state_list[1:], weights[1:]):
        if isinstance(st, str):
            st = torch.load(st)
        for k, v in st.items():
            state[k] = state[k] + w * v

    return state

def uniform_soup(model, ckpt_path_list, saved_ckpt_path='checkpoint_uniform_soup.pt'):
    """
    Uniformly merges states from multiple checkpoints and evaluates the model

    Args:
        model: The PyTorch model
        ckpt_path_list (list): List of checkpoint paths
        saved_ckpt_path (str): Path to save the merged checkpoint

    Returns:
        float: Evaluation score
    """
    state = merge_state(ckpt_path_list)
    model.net.load_state_dict(state)
    score = model.evaluate(model.val_data, quiet=True)[model.monitor]
    torch.save(model.net.state_dict(), saved_ckpt_path)
    
    print(f'saved uniform soup state_dict at {saved_ckpt_path}')
    
    return score

def greedy_soup(model, ckpt_path_list, 
                num_models=None, num_warmup=None, 
                saved_ckpt_path='checkpoint_greedy_soup.pt'):
    """
    Greedily merges states from multiple checkpoints and evaluates the model

    Args:
        model: The PyTorch model
        ckpt_path_list (list): List of checkpoint paths
        num_models (int): Number of models to merge
        num_warmup (int): Number of warmup models (do not choose greedily)
        saved_ckpt_path (str): Path to save the merged checkpoint

    Returns:
        float: Evaluation score
    """
    if num_models is None:
        num_models = len(ckpt_path_list)*100 
    if num_warmup is None:
        num_warmup = len(ckpt_path_list)*5

    
    dfckpt = pd.DataFrame({'ckpt_path':ckpt_path_list})

    scores = []
    printlog('step1: sort ckpt_path by metric...')
    
    loop = tqdm(dfckpt['ckpt_path'])
    for ckpt_path in loop:
        model.load_ckpt(ckpt_path)
        score = model.evaluate(model.val_data,quiet=True)[model.monitor]
        scores.append(score)
        loop.set_postfix(**{model.monitor:score})

    dfckpt['score'] = scores
    dfckpt = dfckpt.sort_values('score',ascending= True if model.mode=='min' else False) 
    prettydf(dfckpt,str_len=50,show=True)
    
    dfckpt['weights'] = [0 for _ in dfckpt['ckpt_path']]
    
    loop = tqdm(range(num_models),total = num_models)
    
    printlog('step2: start greedy merge...')
    
    for i in loop:
        
        if i==0:
            cur_state = torch.load(dfckpt['ckpt_path'].iloc[0])
            cur_score = dfckpt['score'].iloc[0]
            dfckpt['weights'].iloc[0] = 1
            loop.set_postfix(**{'i':i,'score':cur_score})

        elif i+1<=num_warmup:
            k = i%(len(ckpt_path_list))
            dfckpt.loc[k,'weights'] = 1 + dfckpt['weights'].loc[k]
            
            state_i = torch.load(dfckpt['ckpt_path'].iloc[k])
            cur_state = merge_state(state_list = [cur_state, state_i],
                                    weights =[i,1])
            
            model.net.load_state_dict(cur_state)
            cur_score = model.evaluate(model.val_data,quiet=True)[model.monitor]
            
        else:
            for j in range(len(dfckpt)):
                state_j = torch.load(dfckpt['ckpt_path'].iloc[j])
                maybe_state = merge_state(state_list = [cur_state,state_j],
                                          weights = [i,1]
                                         )
                model.net.load_state_dict(maybe_state)
                maybe_score = model.evaluate(model.val_data,quiet=True)[model.monitor]
                if (model.mode=='max' and maybe_score>cur_score) or (
                    model.mode=='min' and maybe_score<cur_score):
                    
                    cur_state = maybe_state
                    cur_score = maybe_score
                    dfckpt['weights'].iloc[j] = dfckpt['weights'].iloc[j]+1
                    loop.set_postfix(**{'i':i,'score':cur_score})
                    break
            else:
                loop.set_postfix(**{'i':i,'score':cur_score})
                loop.close()
                print('could not get better score, early stopping...')
                break 
 
    printlog('step3: save result...')
    model.net.load_state_dict(cur_state)
    torch.save(model.net.state_dict(),saved_ckpt_path)
    prettydf(dfckpt,str_len=50,show=True)
    print('best_score = ', cur_score)
    print('greedy soup ckpt saved at path: '+saved_ckpt_path)
    
    return cur_score


def optuna_soup(model,
                 ckpt_path_list,
                 n_trials=100,
                 timeout=1200,
                 saved_ckpt_path='checkpoint_optuna_soup.pt'):
    """
    Perform an Optuna search to find optimal weights for checkpoint ensemble

    Args:
        model: The PyTorch model
        ckpt_path_list (list): List of checkpoint paths
        n_trials (int): Number of optimization trials
        timeout (int): Timeout for the optimization in seconds
        saved_ckpt_path (str): Path to save the final checkpoint

    Returns:
        float: Best evaluation score
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        weights_dict = {name: trial.suggest_int(name, 1, 100) for name in ckpt_path_list}
        state = merge_state(ckpt_path_list, weights=[weights_dict[name] for name in ckpt_path_list])
        model.net.load_state_dict(state)
        score = model.evaluate(model.val_data, quiet=True)[model.monitor]
        return score

    study = optuna.create_study(
        direction="maximize" if model.mode == 'max' else "minimize",
        study_name="optuna_ensemble",
        
    )

    printlog('step1: start Optuna search...')
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    printlog('step2: save result...')
    best_weights = study.best_params
    best_score = study.best_value

    state = merge_state(ckpt_path_list, weights=[best_weights[name] for name in ckpt_path_list])
    model.net.load_state_dict(state)
    torch.save(model.net.state_dict(), saved_ckpt_path)

    print(f"best_score = {best_score}")
    print("best_weights:")
    print(best_weights)
    print(f'optuna soup ckpt saved at path: {saved_ckpt_path}')

    return best_score

