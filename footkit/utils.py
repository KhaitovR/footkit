from sklearn.metrics import log_loss, f1_score, roc_auc_score, accuracy_score, roc_curve, auc
import pandas as pd
import numpy as np

def set_to_list(cols, excepted):
    return list(set(cols) - set(excepted))

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])

def accuracy_report(fact, pred, threshold, label='oof'):
    print('')
    print('*'*30)
    print(label+' roc_auc:',format(roc_auc_score(fact, pred), '.2f'))
    print(label+' f1_score:',format(f1_score(fact, np.round(pred)), '.1%'))
    print(label+' accuracy_score:',format(accuracy_score(fact, np.round(pred)), '.1%'))
    # calc accuracy > threshold
    print('Calc threshold roc_curve:',format(threshold, '.3f'))
    print(label+' f1_score:',format(f1_score(fact, np.where(pred>=threshold, 1, 0)), '.1%'))
    print(label+' accuracy_score:',format(accuracy_score(fact, np.where(pred>=threshold, 1, 0)), '.1%'))
    print('*'*30)
    print('')
    

def report_validation(df, df_targets, target, oof_pred, y_pred):
    '''
        отчет после расчета валидации
    '''
    
    df['y_pred'] = oof_pred
    fact = df_targets[(df['y_pred'].isnull()==False)&(df_targets[target].isnull()==False)][[target]]
    pred = df[(df['y_pred'].isnull()==False)&(df_targets[target].isnull()==False)]['y_pred']
    threshold = Find_Optimal_Cutoff(fact, pred)[0]
    accuracy_report(fact, pred, threshold=threshold, label='oof')

    df['y_pred'] = y_pred
    fact = df_targets[(df['y_pred'].isnull()==False)&(df_targets[target].isnull()==False)][[target]]
    pred = df[(df['y_pred'].isnull()==False)&(df_targets[target].isnull()==False)]['y_pred']
    
    if fact.shape[0]>0:
        accuracy_report(fact, pred, threshold=threshold, label='test')
    else:
        print('fact for y_pred shape == 0, cant calc accuracy')
