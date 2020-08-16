import numpy as np
import pandas as pd
from config import Config

class Calculation(Config):
    def __init__(self, Config):
        super().__init__()
#         Config().__init__()

    def calc_profit(
        self,
        df_odds,
        df_bookmakers,
        df_pred,
        df_pred_round,
        df_targets,
        # predict_targets,
        df,
        # info_features,
        prev_df_results=False,
        rename='Interval',
        concat_with_info_df=True,
    ):
        df_targets=df_targets.copy()
        df_odds=df_odds.copy()
        df_pred=df_pred.copy()
        df_pred_round=df_pred_round.copy()

        ids_index=set(df_pred.index)&set(df_odds.index)
        df_odds=df_odds.loc[ids_index]
        df_pred=df_pred.loc[ids_index]
        df_pred_round=df_pred_round.loc[ids_index]

        df_roi = df_pred_round*((df_pred*100*df_odds[self.predict_targets]-100)/100)
        df_roi['Best Roi'] = df_roi.max(axis=1)
        df_roi['Best Roi Bet'] = df_roi.idxmax(axis=1)
        df_roi = df_roi[df_roi['Best Roi']>0.05]
        df_roi['Coef'] = df_odds.loc[df_roi.index].lookup(df_roi.index, df_roi['Best Roi Bet'].values)
        df_roi['Bookmaker'] = df_bookmakers.loc[df_roi.index].lookup(df_roi.index, df_roi['Best Roi Bet'].values)
        df_roi['Potencial $'] = ((df_odds*100)-100).loc[df_roi.index].lookup(df_roi.index, df_roi['Best Roi Bet'].values)
        df_roi['Proba'] = df_pred.loc[df_roi.index].lookup(df_roi.index, df_roi['Best Roi Bet'].values)
        df_roi['Predict'] = df_pred_round.loc[df_roi.index].lookup(df_roi.index, df_roi['Best Roi Bet'].values)
        df_roi['Targets'] = df_targets.loc[df_roi.index].lookup(df_roi.index, df_roi['Best Roi Bet'].values)
        df_roi['Profit'] = np.where(
            (df_roi['Targets'].isnull()==False)
            , np.where(
                (df_roi['Predict']==1)&(df_roi['Targets']==0)
                , -100
                , df_roi['Potencial $']*df_roi['Predict']
            )
            , 0
        )

        df_results=df_roi[self.set_to_list(df_roi.columns,self.predict_targets)].copy()
        if rename:
            df_results.rename(columns={f:rename+' '+f for f in df_results.columns}, inplace=True)
        if type(prev_df_results)==pd.DataFrame:
            df_results=pd.concat([prev_df_results, df_results], axis=1, sort=False)
            # print('Невозможно объединить датафрейм с ', type(prev_df_results), 'измените параметр prev_df_results')
        if concat_with_info_df:
            df_results=pd.concat([df.loc[df_results.index][self.info_features],df_results],axis=1, sort=False)
        return df_roi, df_results
    
    def set_optimize_proba_interval(self, df_odds, df_pred):
        df_odds_interval=df_odds.where(df_odds>self.params_confidence_interval['odds']).fillna(1)
        df_pred_round_predel=df_pred.where(df_pred>self.params_confidence_interval['probability']).fillna(0).mask(df_pred>self.params_confidence_interval['probability']).fillna(1).astype('uint8')
        return df_odds_interval, df_pred_round_predel
