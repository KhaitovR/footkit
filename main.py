#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from datetime import datetime as dt, timedelta

from config import Config
config=Config()

from footkit import Preprocess, Parser, Validation, Calculation, OddsParser, PlotPredict, InstaBot, FeatureSelector
from footkit.utils import report_validation

from sklearn.metrics import log_loss, f1_score, roc_auc_score, accuracy_score, roc_curve, auc

import matplotlib.pyplot as plt
# making the workspace wider
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))
# more information in the tables
pd.set_option('display.max_columns',150)
pd.set_option('display.max_rows',150)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings("ignore")

import asyncio


# Парсер событий из UnderStat.com, читаем через api
# * сбор данных, агрегация, генерация признаков
# * если нужно обновить update_data = True

parser = Parser(Config)
df, teams_df = await parser.get_parse_data(update_data=False, resaved_data=True)
# asyncio.get_event_loop().run_until_complete(parser.get_parse_data(update_data=False, resaved_data=True))
print('Прогнозируем:', parser.predict_targets)

preprocess=Preprocess(Config, df, teams_df)
teams_df=preprocess.get_teams_df()

df, engineering_features = preprocess.create_features()
df_targets, targets = preprocess.get_target_on_df_eng('FTHG', 'FTAG')

# Парсер OddsPortal.com (selenium+beautySoup)
# * если нужно обновить from_file = False
# df_odds, df_bookmakers = Odds(Config).get_last_update_scrapper(df)
odds = OddsParser(Config, last_update=True)
df_odds, df_bookmakers = odds.get_last_update_scrapper(
    df,
    season=2020,
    url='https://www.oddsportal.com/login/',
    username='user',
    password='pass',
    proxy=False,
    headless=False,
    from_file=True,
)

# пример простой модели, подаем дату и один таргет
run_model = Validation(Config).run_model
start_date = dt.now()
target='2X'

features = list(set(engineering_features + ['HomeTeamId', 'AwayTeamId', 'Season'])-set(['Date']))

df = df[~df.index.duplicated(keep='first')]
df_targets = df_targets[~df_targets.index.duplicated(keep='first')]
y_pred, res, model_tc, oof_pred, all_val_test_index = run_model(
    df=df,
    df_targets=df_targets,
    start_date=start_date,
    target=target,
    features=features,
    val_function=roc_auc_score,
    timeseries=False,
    num_splits=5,
    plot_importances=False,
    exp_target=False,
    verbose=True,
    cv_kfold='lastweek',
    model_name='lightgbm'
)

report_validation(df, df_targets, target, oof_pred, y_pred)

# Прогнозируем все predict_targets из config.py
start_date = dt.now()
# pd.to_datetime('2020-02-21').date()# dt.now()# pd.to_datetime('2018-12-01').date()
features = list(set(engineering_features + ['HomeTeamId', 'AwayTeamId', 'Season'])-set(['Date']))
calc_forecast_on_targets = Validation(Config).calc_forecast_on_targets
df_pred, df_pred_cutoff = calc_forecast_on_targets(
    df=df[df['Date']>pd.to_datetime('2018-08-01')], # df,
    df_targets=df_targets,
    features=features,
    start_date=start_date,
    val_function=roc_auc_score,
    num_splits=5,
    plot_importances=False,
    print_report=False,
    verbose=True
)


# добавляем кэфы, считаем ROI, выбираем лучший вариант по threshold (cutoff), и пороги выставленные в файле config.py
CalcProf = Calculation(Config)

df_roi, df_results = CalcProf.calc_profit(
    df_odds=df_odds,
    df_bookmakers=df_bookmakers,
    df_pred=df_pred,
    df_pred_round=df_pred_cutoff,
    df_targets=df_targets,
    # predict_targets=predict_targets,
    rename=False,#'Interval'
    df=df,
    # info_features=info_features,
    concat_with_info_df=False,
    prev_df_results=False
)

df_odds_interval, df_pred_round_predel = CalcProf.set_optimize_proba_interval(df_odds, df_pred)
df_roi_interval, df_results = CalcProf.calc_profit(
    df_odds=df_odds_interval,
    df_bookmakers=df_bookmakers,
    df_pred=df_pred,
    df_pred_round=df_pred_round_predel,
    df_targets=df_targets,
    # predict_targets=predict_targets,
    rename='Interval',
    df=df,
    # info_features=info_features,
    concat_with_info_df=True,
    prev_df_results=df_results
)

time = str(dt.now().date())
time = time[:4]+'_'+time[5:7]+'_'+time[8:10]
df_results.to_pickle('./Predictions/'+time+' prediction.pkl')
df_results.to_excel('./Predictions/'+time+' prediction.xlsx', index=None)

# Рисуем картинки с прогнозом
PltPred = PlotPredict(Config)
PltPred.get_plot_predict(
    paint_df=df_results,
    hometeam='HomeTeam',
    awayteam='AwayTeam',
    leaguename='LeagueName',
    bookmaker='Interval Bookmaker',
    bet='Interval Best Roi Bet',
    betcoef='Interval Coef',
    datename='DateTime',
    proba='Interval Proba',
    show_pictures=True,
    clear_pic_path=True,
)

# # Выкладываем в instagram
# * генерируем хештег из вывески матча
# * канал: https://www.instagram.com/footballbets.tv/
# 
# 
# 
# - 2020-08-16 21:49:51,387 - INFO - Photo './Pictures/prediction_pics/19_08_20 Ural - Lokomotiv Moscow, FC Ufa - Spartak Moscow, Dinamo Moscow - FC Rostov.jpg' is uploaded.
# - ['#Ural', '#Lokomotiv', '#Moscow', '#FC', '#Ufa', '#Spartak', '#Moscow', '#Dinamo', '#Moscow', '#FC', '#Rostov#Football', '#Bets', '#Recommendations', '#stavki']

# insta=InstaBot(Config)
# insta.login_bot()
# while True:
#     insta.autoposting(insta.upload_photos, path_medias='./Pictures/prediction_pics/', fromCaptions='File')
#     insta.sleep(100)