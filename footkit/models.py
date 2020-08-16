import pandas as pd
import numpy as np
from datetime import datetime as dt, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, f1_score, roc_auc_score, accuracy_score, roc_curve, auc
from sklearn.model_selection import GroupKFold, StratifiedKFold
import lightgbm as lgb
from footkit.utils import Find_Optimal_Cutoff, accuracy_report
from config import Config

class Validation(Config):
    def __init__(self, Config):
        super().__init__()

    # help function
    def GetSplit(
        self,
        df,
        df_targets,
        target,
        date='1900-01-01',
        IsResult = False,
        div = None,
        timedlt = 0,
        is_print = True,
    ):
        df=df.copy()
        df=pd.concat([df, df_targets[target]], axis=1, sort=False)

        if timedlt != 0:
            date_test_to = str(dt.strptime(date,'%Y-%m-%d')+timedelta(days=timedlt))[:10]
        else:
            date_test_to = df[df[target]==0][self.DATETIME].max()

        if is_print:
            print(date_test_to)

        if div != None:
            if IsResult==False:
                #calc_df
                x_train_val = df[(df[self.DATETIME] < date) & (df[self.LEAGUENAME] == div) & (df[target].isnull()==False)]
                x_test_val = df[((df[self.DATETIME] >= date) & (df[self.DATETIME] <= date_test_to)) & (df[self.LEAGUENAME] == div) & (df[target].isnull()==False)]
                y_train_val = df[(df[self.DATETIME] < date)  & (df[self.LEAGUENAME] == div) & (df[target].isnull()==False)][target]
                y_test_val = df[((df[self.DATETIME] >= date) & (df[self.DATETIME] <= date_test_to))  & (df[self.LEAGUENAME] == div) & (df[target].isnull()==False)][target]
            else:
                x_train = df[(df[target].isnull()==False) & (df[self.LEAGUENAME] == div)]
                x_test = df[(df[target].isnull()==True)  & (df[self.LEAGUENAME] == div)]
                y_train = df[(df[target].isnull()==False)  & (df[self.LEAGUENAME] == div)][target]
                y_test = df[(df[target].isnull()==True)  & (df[self.LEAGUENAME] == div)][target]
        else:
            if IsResult==False:
                #calc_df
                x_train_val = df[(df[self.DATETIME] < date) & (df[target].isnull()==False)]
                x_test_val = df[((df[self.DATETIME] >= date) & (df[self.DATETIME] <= date_test_to)) & (df[target].isnull()==False)]
                y_train_val = df[(df[self.DATETIME] < date) & (df[target].isnull()==False)][target]
                y_test_val = df[((df[self.DATETIME] >= date) & (df[self.DATETIME] <= date_test_to)) & (df[target].isnull()==False)][target]
            else:
    #             x_train = df[(df[target].isnull()==False)]
    #             x_test = df[(df[target].isnull()==True)]
    #             y_train = df[(df[target].isnull()==False)][target]
    #             y_test = df[(df[target].isnull()==True)][target]

                x_train = df[(df[self.DATETIME] < date) & (df[target].isnull()==False)]
                x_test = df[((df[self.DATETIME] >= date) & (df[self.DATETIME] <= date_test_to))]
                y_train = df[(df[self.DATETIME] < date) & (df[target].isnull()==False)][target]
                y_test = df[((df[self.DATETIME] >= date) & (df[self.DATETIME] <= date_test_to))][target]


        if IsResult == False:
            if is_print:
                print('Learn set:', x_train_val.shape[0]
                      ,'\nTarget:',target
                      ,'\nmin date:' + str(x_train_val[self.DATETIME].min())[:10]
                      ,'\nmax date:' + str(x_train_val[self.DATETIME].max())[:10]
                      ,'\nWeekend set:',x_test_val.shape[0]
                      ,'\nmin day:',str(x_test_val[self.DATETIME].min())[:10]
                      ,'\nmax day:',str(x_test_val[self.DATETIME].max())[:10]
                 )
            shape_xtest = x_test_val.shape[0]
            return shape_xtest, x_train_val, y_train_val, x_test_val, y_test_val

        else:
            if is_print:
                print('Learn set:', x_train.shape[0]
                    ,'\nTarget:',target
                    ,'\nmin date:' + str(x_train[self.DATETIME].min())[:10]
                    ,'\nmax date:' + str(x_train[self.DATETIME].max())[:10]
                    ,'\nWeekend set:', x_test.shape[0]
                    ,'\nmin day:',str(x_test[self.DATETIME].min())[:10]
                    ,'\nmax day:',str(x_test[self.DATETIME].max())[:10]
                 )

            shape_xtest = x_test.shape[0]
            return shape_xtest, x_train, y_train, x_test, y_test


    def get_split_lastweek_football(
        self,
        df,
        df_targets,
        num_split,
        target,
        start_date = dt.now()
    ):
        df = df.copy()
        cv_splits=[]
        for i in range(num_split, 0, -1):
            shape_xtest, x_train, y_train, x_test, y_test = self.GetSplit(
                df,
                df_targets,
                target=target,
                date=str(start_date + timedelta(days=-7*i))[:10],
                IsResult=True if i <= 0 else False,
                div = None,
                timedlt=7,
                is_print = False
            )
            cv_splits.append(
                [
                    x_train.index, # train val
                    x_test.index  # test val
                ]
            )
        return cv_splits

    def run_model(
        self,
        df,
        df_targets,
        start_date,
        features,
        target,
        val_function,
        timeseries=False,
        num_splits=5,
        plot_importances=True,
        exp_target=False,
        verbose=0,
        cv_kfold='StratifiedKFold',
        model_name='lightgbm'
    ):

        if verbose:
            print('')
            print('Run model, calc', target, 'with cv per', cv_kfold)
            print('*'*40)

        # calc main x_train, x_test
        shape_xtest, X_train, y_train, X_test, y_test = self.GetSplit(
            df,
            df_targets[[target]],
            target=target,
            date=str(start_date)[:10],
            IsResult=True,
            div = None,
            timedlt=7,
            is_print=verbose
        )

        oof_pred = pd.Series(data=0, index=df.index)
        # np.zeros(len(uplift_train))
        y_pred = pd.Series(data=0, index=X_test.index)
        # np.zeros(len(uplift_test))
        # X_test=df.loc[indices_test, features].fillna(0).values

        all_val_test_index = []
        update_ix = False
        if cv_kfold=='StratifiedKFold':
            cv = StratifiedKFold(n_splits=num_splits)
            cv_split = cv.split(df, df[target])
        elif cv_kfold=='GroupKFold':
            cv = GroupKFold(n_splits=df['LeagueName'].nunique())
            cv_split = cv.split(df, df_targets[[target]], df['LeagueName'])
            update_ix = True
        elif cv_kfold=='lastweek':
            cv_split = self.get_split_lastweek_football(df,df_targets[[target]], num_split=num_splits, target=target, start_date=start_date)
            update_ix = False
        fold = 0
        for val_train_index, val_test_index in cv_split:
            if update_ix:
                val_train_index = pd.Index(df.reset_index().loc[val_train_index]['IdMatch'].values, name='IdMatch')
                val_test_index = pd.Index(df.reset_index().loc[val_test_index]['IdMatch'].values, name='IdMatch')

            if model_name == 'lightgbm':
                X_train_val = df.loc[val_train_index, features].values
                # print(X_train_val)
                X_test_val = df.loc[val_test_index, features].values
                
                if verbose:
                    print('val train index from', str(df.loc[val_train_index, ['Date']].min().values[0]), 'to', str(df.loc[val_train_index, ['Date']].max().values[0]), 'and shape', len(val_train_index))
                    print('val test index from', str(df.loc[val_test_index, ['Date']].min().values[0]), 'to', str(df.loc[val_test_index, ['Date']].max().values[0]), 'and shape', len(val_test_index))

                target_train_val = df_targets.loc[val_train_index, target].values        
                target_test_val = df_targets.loc[val_test_index, target].values
                control_train_set = lgb.Dataset(X_train_val, target_train_val)
                control_val_set = lgb.Dataset(X_test_val, target_test_val)
                model_tc = lgb.train(
                    params=self.lgb_params,
                    train_set=control_train_set,
                    num_boost_round = 10000,
                    early_stopping_rounds = 100, 
                    valid_sets=[control_train_set, control_val_set],
                    verbose_eval = 100 if verbose != 0 else 0,
                )
                oof_pred.loc[val_test_index] = model_tc.predict(X_test_val)
                y_pred += (model_tc.predict(X_test[features])/num_splits).astype('float64')
            elif model_name=='catboost':
                X_train_val = df.loc[val_train_index, features].values
                X_test_val = df.loc[val_test_index, features].values
                target_train_val = df_targets.loc[val_train_index, target].values        
                target_test_val = df_targets.loc[val_test_index, target].values
                model_tc = CatBoostClassifier(**self.cat_params)
                model_tc.fit(
                    X_train_val,
                    target_train_val,
                    eval_set=(X_test_val, target_test_val),
                    use_best_model=True,
                    verbose=100 if verbose != 0 else 0
                )
                oof_pred.loc[val_test_index] = model_tc.predict_proba(X_test_val)[:, 1]
                y_pred += (model_tc.predict_proba(X_test[features])[:, 1]/num_splits).astype('float64')

            # print(y_pred)
            try:
                val_crt_fold = val_function(target_test_val, oof_pred.loc[val_test_index])
            except ValueError:
                if verbose:
                    print('Один класс, не можем посчитать score')
            # except Exception:
            #     print('error val_crt_fold')
            if verbose:
                print(f'Fold: {fold+1}  score: {np.round(val_crt_fold,4)}')
            all_val_test_index = all_val_test_index + list(val_test_index)
            fold+=1

        try:
            res = val_function(df_targets.loc[all_val_test_index, target].values, oof_pred.loc[all_val_test_index])
        except Exception:
            res=9
            # print('error res calc')


        if plot_importances:
            (pd.Series(model_tc.feature_importance(importance_type='gain'), index=features)
               .nlargest(20)
               .plot(kind='barh')
            );
            plt.show();

        if verbose:
            print(f'Weighted score: {np.round(res,4)}')

        return y_pred, res, model_tc, oof_pred.loc[all_val_test_index], all_val_test_index
        
    def calc_forecast_on_targets(
        self,
        df,
        df_targets,
        # predict_targets,
        features,
        start_date,
        val_function=roc_auc_score,
        num_splits=5,
        plot_importances=False,
        print_report=False,
        verbose=False,
    ):
        df=df.copy();df_targets=df_targets.copy();
        
        del_shape_xtest, del_X_train, del_y_train, del_X_test, del_y_test = self.GetSplit(
            df,
            df_targets[[self.predict_targets[0]]],
            target=self.predict_targets[0],
            date=str(start_date)[:10],
            IsResult=True,
            div = None,
            timedlt=7,
            is_print=False
        )

        df_pred = pd.DataFrame(index=del_X_test.index)
        df_pred_cutoff = pd.DataFrame(index=del_X_test.index)
        
        for target in self.predict_targets:
            y_pred, res, model_tc, oof_pred, all_val_test_index = self.run_model(
                df=df,
                df_targets=df_targets,
                start_date=start_date,
                features=features,
                target=target,
                val_function=val_function,
                timeseries=False,
                num_splits=num_splits,
                plot_importances=plot_importances,
                exp_target=False,
                verbose=verbose,
                cv_kfold='lastweek',
                model_name='lightgbm'
            )
        
            df_pred[target] = y_pred

            # df['y_pred'] = oof_pred
            fact = df_targets.loc[oof_pred.index][(df_targets[target].isnull()==False)][target]
            pred = oof_pred[(df_targets[target].isnull()==False)]
            # df.loc[oof_pred.index][(oof_pred.isnull()==False)&(df_targets[target].isnull()==False)]['y_pred']
            threshold = Find_Optimal_Cutoff(fact, pred)[0]
            df_pred_cutoff[target] = np.where(y_pred>threshold, 1, 0)

            if print_report:
                df['y_pred'] = oof_pred
                fact = df_targets[(df['y_pred'].isnull()==False)&(df_targets[target].isnull()==False)][[target]]
                pred = df[(df['y_pred'].isnull()==False)&(df_targets[target].isnull()==False)]['y_pred']
                threshold = Find_Optimal_Cutoff(fact, pred)[0]
                accuracy_report(fact, pred, threshold=threshold, label='oof')
        return df_pred, df_pred_cutoff
