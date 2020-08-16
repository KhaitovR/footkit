import pandas as pd
import numpy as np

class Config:

    def __init__(self):
        self.seasons = [2016,2017,2018,2019,2020] # ,2020
        self.leagues = ["epl", "la_liga", "bundesliga", "serie_a", "ligue_1", "rfpl"]
        self.fixtures_seasons = [2020]
        self.params_confidence_interval={'odds':1.3, 'probability':0.6, 'cutoff':False}
        self.info_features = ['DateTime', 'LeagueName', 'RoundEvent', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
        self.info_test_features = ['DateTime', 'LeagueName', 'RoundEvent', 'HomeTeam', 'AwayTeam']
        self.cat_params = {}
        self.lgb_params = {
            'objective': 'binary',
            'boosting': 'gbdt',
            'num_iterations': 10000, # 00
            'learning_rate': 0.03,
            'num_leaves': 15,
            'num_threads': 7,
            'seed': 17,
            'max_depth': 3,
            'min_data_in_leaf': 5,
            'subsample': 0.05,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.01,
            'reg_lambda': 0.01,
            'bagging_fraction': 0.75,
            'bagging_freq': 3,
            'bagging_seed': 17,
            'feature_fraction': 0.25,
            'feature_fraction_seed': 17,
            'metric': 'binary_logloss',
            'n_jobs': 2
        }

        # validation parametr
        self.COUNT_LAST_WEEK = 2

        self.SEASON='Season'
        self.TEAMNAME='TeamName'
        self.LEAGUENAME='LeagueName'
        self.DATETIME='DateTime'
        self.PLACEMATCH='h_a'
        self.GOALS='scored'
        self.MISSED='missed'
        self.POINTS='pts'
        self.ROUNDEVENT='RoundEvent'
        self.FEATURES = [
            self.GOALS,
            self.MISSED,
            self.POINTS,
            'xG', 'xGA', 'npxG', 'npxGA', 'ppda_att', 'ppda_def', 'ppda_allowed_att',
            'ppda_allowed_def', 'deep', 'deep_allowed', 'xpts', 'wins', 'draws',
            'loses', 'npxGD', 'ChillDays', 
        ]

        self.predict_targets = ['W', 'L', '1X', '2X','Total Less 1.5','Total Less 2.5','Total Less 3.5','Total More 1.5','Total More 2.5','Total More 3.5']
        # self.predict_targets = ['1X']

        self.all_targets = [
            'W',
            'D',
            'L',
            # 'Winner',
            '12','1X','2X',
            'Goals 0-1','Goals 2-3','Goals 4-5','Goals 6>',
            'HT Score','AT Score','Both teams to score',
            # 'Handicap AT',
            'Handicap AT -0.5','Handicap AT -1','Handicap AT -1.5','Handicap AT -2','Handicap AT -2.5','Handicap AT 0.5','Handicap AT 1','Handicap AT 1.5','Handicap AT 2','Handicap AT 2.5','Handicap AT 3','Handicap AT 3.5',
            # 'Handicap HT',
            'Handicap HT -0.5','Handicap HT -1','Handicap HT -1.5','Handicap HT -2','Handicap HT -2.5','Handicap HT -3','Handicap HT -3.5','Handicap HT 0.5','Handicap HT 1','Handicap HT 1.5','Handicap HT 2','Handicap HT 2.5',
            'Ind. Total AT Total Less 0.5','Ind. Total AT Total Less 1','Ind. Total AT Total Less 1.5','Ind. Total AT Total Less 2','Ind. Total AT Total Less 2.5','Ind. Total AT Total Less 3.5','Ind. Total AT Total More 0.5','Ind. Total AT Total More 1','Ind. Total AT Total More 1.5','Ind. Total AT Total More 2','Ind. Total AT Total More 2.5','Ind. Total AT Total More 3.5',
            'Ind. Total HT Total Less 0.5','Ind. Total HT Total Less 1','Ind. Total HT Total Less 1.5','Ind. Total HT Total Less 2','Ind. Total HT Total Less 2.5','Ind. Total HT Total Less 3.5','Ind. Total HT Total More 0.5','Ind. Total HT Total More 1','Ind. Total HT Total More 1.5','Ind. Total HT Total More 2','Ind. Total HT Total More 2.5','Ind. Total HT Total More 3.5',
            'Score 0:1','Score 0:2','Score 0:3','Score 1:0','Score 1:1','Score 1:2','Score 1:3','Score 2:0','Score 2:1','Score 2:2','Score 2:3','Score 3:0','Score 3:1','Score 3:2','Score 3:3',
            'Total Less 1','Total Less 1.5','Total Less 2','Total Less 2.5','Total Less 3','Total Less 3.5','Total Less 4','Total Less 4.5','Total Less 5','Total Less 5.5',
            'Total More 1','Total More 1.5','Total More 2','Total More 2.5','Total More 3','Total More 3.5','Total More 4','Total More 4.5','Total More 5','Total More 5.5',
        ]
        
        self.trans_ru = {
            'DateTime':'Начало матча',
            'LeagueName':'Лига',
            'RoundEvent':'Тур',
            'HomeTeam':'Хозяева',
            'AwayTeam':'Гости',
            'FTHG':'Голы, хозяева',
            'FTAG':'Голы, гости',
        }
        
        self.path_pictures = './Pictures/photos/Base.jpg'
        self.path_insta_logo = './Pictures/photos/sg_logo_inst.jpg'
        self.path_fonts = './Pictures/fonts/'
        self.path_pictures_pred='./Pictures/prediction_pics/'

        self.instagram_username='LogInstagram'
        self.instagram_password = 'passInst'

    def set_to_list(self, cols, excepted):
        return list(set(cols) - set(excepted))
