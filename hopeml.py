# отключим предупреждения Anaconda
import warnings
warnings.simplefilter('ignore')

# будем отображать графики прямо в jupyter'e
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
#графики в svg выглядят более четкими
# %config InlineBackend.figure_format = 'svg'

#увеличим дефолтный размер графиков
from pylab import rcParams
rcParams['figure.figsize'] = 8, 5

import pandas as pd
# pd.options.display.float_format = '{:.2f}'

import pandas_profiling
import numpy as np

# from sklearn import metrics
# from mlxtend.regressor import StackingRegressor
from sklearn.metrics import accuracy_score, auc, classification_report, f1_score, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier, ExtraTreesRegressor, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV, train_test_split, TimeSeriesSplit, learning_curve, KFold

from sklearn.linear_model import BayesianRidge
from sklearn.feature_selection import SelectFromModel, chi2, SelectKBest, RFE
# from sklearn.model_selection import 
from sklearn.metrics.scorer import make_scorer, roc_auc_score

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# from tensorflow.keras import models
# from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.svm import SVC
# from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb
import catboost as cbst
from numba import jit

import gc
import copy
import datetime
from math import ceil

def train_test_split_sorted(X, y, test_size = 0.25, dates = None):
    n_test = ceil(test_size * len(X))

    if dates!=None:
        sorted_index = [x for _, x in sorted(zip(np.array(dates), np.arange(0, len(dates))), key=lambda pair: pair[0])]
    else:
        sorted_index = X.index
        increment_index = [n for n, i in enumerate(X.index)]

    train_idx = sorted_index[:-n_test]
    test_idx = sorted_index[-n_test:]

    inc_train_idx = increment_index[:-n_test]
    inc_test_idx = increment_index[-n_test:]

    if isinstance(X, (pd.Series, pd.DataFrame)):
        X_train = X.loc[train_idx]
        X_test = X.loc[test_idx]
    else:
        X_train = X[inc_train_idx]
        X_test = X[inc_test_idx]

    if isinstance(y, (pd.Series, pd.DataFrame)):
        y_train = y.loc[train_idx]
        y_test = y.loc[test_idx]
    else:
        y_train = y[inc_train_idx]
        y_test = y[inc_test_idx]

    return X_train, X_test, y_train, y_test

# reduce memory usage
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float32', 'float64'] # , 'float16'
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# khaitov / 10092019 / 
# fast optimize data for large data
def fast_reduce_mem_usage(df, verbose=True):
    '''
        1. Определяем через describe всю статистику
        2. Собираем count null значений в таблице
        3. Определяем типы данных и форматируем все столбцы с этими типами.
    '''
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    describe_df = df.describe()
    describe_df = describe_df.T
    
    describe_df = pd.concat(
        [
            describe_df
            , pd.DataFrame(df.select_dtypes(exclude=['object', 'datetime']).isnull().sum(), columns = ['CntNull'])
        ], axis = 1
    )

    describe_df['Type_float'] = np.where(
        (describe_df['min'] > np.finfo(np.float16).min) & (describe_df['max'] < np.finfo(np.float16).max)
        , 'float16'
        , np.where(
            (describe_df['min'] > np.finfo(np.float32).min) & (describe_df['max'] < np.finfo(np.float32).max)
            , 'float32'
            , 'float64'
        )
    )

    describe_df['Type'] = np.where(
        (describe_df['min'].round() == describe_df['min']) & (describe_df['max'].round() == describe_df['max']) & (describe_df['CntNull'] == 0)
        , np.where(
            (describe_df['min'] > np.iinfo(np.int8).min) & (describe_df['max'] < np.iinfo(np.int8).max)
            , 'int8'
            , np.where(
                (describe_df['min'] > np.iinfo(np.int16).min) & (describe_df['max'] < np.iinfo(np.int16).max)
                , 'int16'
                , np.where(
                    (describe_df['min'] > np.iinfo(np.int32).min) & (describe_df['max'] < np.iinfo(np.int32).max)
                    , 'int32'
                    , 'int64'
                )
            )
        )
        , describe_df['Type_float']
    )
    
    for t in describe_df.Type.unique():
        df[describe_df[describe_df['Type'] == t].index] = df[describe_df[describe_df['Type'] == t].index].astype(t)
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        print('Change columns:', ', '.join(list(describe_df[describe_df['Type'] == t].index)),' IN ', t)
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

@jit
def fast_auc(y_true, y_prob):
    """
    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013
    """
    import numpy as np
    
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc

castom_auc = make_scorer(fast_auc, greater_is_better=True)

def set_to_list(cols, excepted):
    return list(set(cols) - set(excepted))

def df_from_path(path, encoding = None):
    if path.split('.')[-1]=='csv':
        df = pd.read_csv(path, encoding=encoding)
    elif path.split('.')[-1] in ['xls', 'xlsx', 'xlsb']:
        df = pd.read_excel(path)
    elif path.split('.')[-1] in ['fth', 'feather']:
        import feather
        df = feather.read_dataframe(path)
    elif path.split('.')[-1] in ['pkl', 'pickle']:
        df = pd.read_pickle(path)
    return df


def smothed_aggregate(
    data
    , null_field
    , agg_field
    , alpha = 10
    , fillna = False
):
    data = data.copy()
    
    if fillna:
        data.fillna(0, inplace=True)
        
    result_series = (
        data[[null_field, agg_field]].groupby(
            [
                null_field
            ]
        )[
            agg_field
        ].transform('mean') * data[[null_field, agg_field]].groupby(
            null_field
        )[
            null_field
        # count строк
        ].transform('count') + data[
            agg_field
        # global mean
        ].astype(np.float32).mean() * alpha
    ) / (
        # count строк
        data[[null_field, agg_field]].groupby(
            null_field
        )[
            null_field
        ].transform('count') + alpha
    )
    
    return result_series


def declare_models(value, seed):

    # Create function returning a compiled network
    def CNNClassifier(optimizer='rmsprop', units = 16):
        from tensorflow.keras import models
        from tensorflow.keras import layers
        from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

        # Start neural network
        network = models.Sequential()
        # Add fully connected layer with a ReLU activation function
        network.add(layers.Dense(units=units, activation='relu', input_shape=(number_input_shape_for_cnn,)))
        # Add fully connected layer with a ReLU activation function
        network.add(layers.Dense(units=np.round(units/2), activation='relu'))
        # Add fully connected layer with a sigmoid activation function
        network.add(layers.Dense(units=1, activation='sigmoid'))
        # Compile neural network
        network.compile(loss='binary_crossentropy', # Cross-entropy
                        optimizer=optimizer, # Optimizer
                        metrics=['accuracy']) # Accuracy performance metric
        # Return compiled network
        return network

    
    if value == 'Classification':
        models = {
            'ExtraTreesClassifier': ExtraTreesClassifier(random_state = seed),
            'RandomForestClassifier': RandomForestClassifier(random_state = seed),
            # 'AdaBoostClassifier': AdaBoostClassifier(random_state = seed),
            # 'GradientBoostingClassifier': GradientBoostingClassifier(random_state = seed),
            'LGBMClassifier' : lgb.LGBMClassifier(random_state = seed, min_child_samples=1, importance_type='gain'),
            # '3DenseNN_Classifier' : KerasClassifier(build_fn=CNNClassifier, verbose=0) ,# 'Declare on class', # KerasClassifier(build_fn=CNNClassifier, verbose=0),
            # 'CatBoostClassifier' : cbst.CatBoostClassifier(random_state = seed),
            # 'XGBRFClassifier':xgb.XGBRFClassifier(random_state = seed)
            # 'SVC': SVC(random_state = seed)
        }

        params_cv = {
            # , 'max_features': range(4,19)
            'ExtraTreesClassifier': {
                "max_depth": [i if i ==-1 else i + 1 for i in range(2, 12, 2)],
                # "max_features": [1, 3, 10],
                "min_samples_split": range(2, 24, 2),
                "min_samples_leaf": range(2, 24, 2),
                "bootstrap": [False],
                "n_estimators" :range( 20,400, 10),
                "criterion": ["gini"]
            },
            'RandomForestClassifier': {
                "max_depth": [i if i ==-1 else i + 1 for i in range(2, 12, 2)],
                # "max_features": [1, 3, 10],
                "min_samples_split": range(2, 24, 2),
                "min_samples_leaf": range(2, 24, 2),
                "bootstrap": [False],
                "n_estimators" :range(20,400, 10),
                "criterion": ["gini"]
            },
            'AdaBoostClassifier':  {
                # "base_estimator__criterion" : ["gini", "entropy"],
                # "base_estimator__splitter" :   ["best", "random"],
                "algorithm" : ["SAMME","SAMME.R"],
                "n_estimators" : range(20,500, 20),
                "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]
            },
            'GradientBoostingClassifier': {
                'loss' : ["deviance"],
                "n_estimators" : range(20,500, 20),
                "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5],
                'max_depth': range(2, 24, 2),
                'min_samples_leaf': range(2, 24, 2),
                'max_features': [None, 0.3, 0.1, 0.6, 0.9]
              },
            'LGBMClassifier': {
                'boosting_type': ['gbdt'],
                'objective': ['binary', 'cross_entropy'],
                'metric': ['auc'], # ['f1']
                'tree_learner': ['serial'],
                'colsample_bytree': [0.8, 0.9, 1],
                'subsample_freq': [1, 2, 3, 4],
                'subsample': [0.8, 0.9, 1],
                'n_estimators': range(10, 10000, 100),
                'num_leaves': range(2, 256, 4),
                'learning_rate': [0.01, 0.1, 0.015, 0.001, 0.2, 0.4],
                'max_depth': [-1, 2, 4, 6, 8, 10, 12],
                'reg_alpha': [0.15, 0.2, 0.25],
                'reg_lambda': [0.15, 0.2, 0.25],
                # 'early_stopping_round':[100],
            },
            '3DenseNN_Classifier' : dict(
                units = [4,8,12,16,64,128,512],
                optimizer=['rmsprop', 'adam'], 
                epochs=[5, 10], 
                batch_size=[5, 10, 100]
            ),
            'SVC': {
                # 'kernel': ['rbf'], 
                'gamma': [ 0.001, 0.01, 0.1, 1],
                'C': [1, 10, 50, 100,200,300, 1000]
            },
            'CatBoostClassifier' : {
                'depth':[2, 3, 4],
                'loss_function': ['Logloss', 'CrossEntropy'],
                'l2_leaf_reg':np.logspace(-20, -19, 3),
                'iterations':[2500],
                'eval_metric':['Accuracy'],
                'leaf_estimation_iterations' : [10],
                # 'use_best_model' : [True]             
            },
            'XGBClassifier': {    
                'max_depth':[3],
                'learning_rate':[1],
                'n_estimators':[100],
                'verbosity':[1],
                'silent':[None],
                'objective':['binary'], # :logistic',
                'n_jobs':[1],
                'nthread':[None],
                'gamma':[0],
                'min_child_weight':[1],
                'max_delta_step':[0],
                'subsample':[0.8],
                'colsample_bytree':[1],
                'colsample_bylevel':[1],
                'colsample_bynode':[0.8],
                'reg_alpha':[0],
                'reg_lambda':[1],
                'scale_pos_weight':[1],
                'base_score':[0.5],
                # 'random_state':[seed],
                # 'seed':[None],
                'missing':[None],
            },

        }

        metrics = [accuracy_score, f1_score]

    elif value == 'Regression':
        models = {
            'ExtraTreesRegressor': ExtraTreesRegressor(random_state = seed),
            # 'RandomForestRegressor': RandomForestRegressor(random_state = seed),
            # 'AdaBoostRegressor': AdaBoostRegressor(random_state = seed),
            # 'GradientBoostingRegressor': GradientBoostingRegressor(random_state = seed),
            'LGBMRegressor' : lgb.LGBMRegressor(random_state = seed, min_child_samples=1),
            # 'XGBRegressor': XGBRegressor(random_state=seed),
        }

        params_cv = {
            # , 'max_features': range(4,19)
            'ExtraTreesRegressor': {
                "max_depth": [i if i ==-1 else i + 1 for i in range(2, 12, 2)],
                # "max_features": [1, 3, 10],
                "min_samples_split": range(2, 24, 2),
                "min_samples_leaf": range(2, 24, 2),
                "bootstrap": [False],
                "n_estimators" :range( 20,400, 10),
                # "criterion": ["gini"]
            },
            'RandomForestRegressor': {
                "max_depth": [i if i ==-1 else i + 1 for i in range(2, 12, 2)],
                # "max_features": [1, 3, 10],
                "min_samples_split": range(2, 24, 2),
                "min_samples_leaf": range(2, 24, 2),
                "bootstrap": [False],
                "n_estimators" :range(20,400, 10),
                # "criterion": ["gini"]
            },
            'AdaBoostRegressor':  {
                # "base_estimator__criterion" : ["gini", "entropy"],
                # "base_estimator__splitter" :   ["best", "random"],
                # "algorithm" : ["SAMME","SAMME.R"],
                "n_estimators" : range(20,500, 20),
                "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]
            },
            'GradientBoostingRegressor': {
                # 'loss' : ["deviance"],
                "n_estimators" : range(20,500, 20),
                "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5],
                'max_depth': range(2, 12, 2),
                'min_samples_leaf': range(2, 24, 2),
                'max_features': [None, 0.3, 0.1, 0.6, 0.9]
              },
            'LGBMRegressor' : {
                'boosting_type':['gbdt', 'dart', 'rf'], # , 'goss'
                # 'objective':['binary'],
                # 'metric':['auc'],
                'tree_learner':['serial'],
                'colsample_bytree': [1],
                'subsample_freq':[1,2,8],
                'subsample':[float(x) for x in np.linspace(start = .1, stop = 0.99, num = 10)],
                # 'min_child_weight':[float(x) for x in np.linspace(start = .1, stop = 0.99, num = 5)],
                # [i/100 for i in range(1, 110, 10)],
                # 'min_split_gain':[float(x) for x in np.linspace(start = .1, stop = 0.99, num = 5)],
                'n_estimators':[int(x) for x in np.linspace(start = 10, stop = 400, num = 20)],
                'num_leaves': [int(x) for x in np.linspace(start = 2, stop = 256, num = 20)],
                'learning_rate': [float(x) for x in np.linspace(start = .001, stop = 0.2, num = 20)],
                'max_depth': [int(x) for x in np.linspace(start = -1, stop = 16, num = 4)],
                # 'reg_alpha':[float(x) for x in np.linspace(start = .1, stop = 0.99, num = 5)],
                # 'reg_lambda':[float(x) for x in np.linspace(start = .1, stop = 0.99, num = 25)],
            },
            'XGBRegressor':{
                'min_child_weight': [1, 5, 10],
                'gamma': [0.5, 1, 1.5, 2, 5],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'max_depth': [3, 4, 5]
            }
        }

        metrics = [mean_absolute_error, r2_score]

        from sklearn import metrics
        all_metrics = {
            'Classification':
            {
                'accuracy':metrics.accuracy_score,                      
                'balanced_accuracy':metrics.balanced_accuracy_score,    
                'average_precision':metrics.average_precision_score,    
                'brier_score_loss':metrics.brier_score_loss,            
                'f1':metrics.f1_score,                                  # for binary targets
                'f1_micro':metrics.f1_score,                            # micro-averaged
                'f1_macro':metrics.f1_score,                            # macro-averaged
                'f1_weighted':metrics.f1_score,                         # weighted average
                'f1_samples':metrics.f1_score,                          # by multilabel sample
                'neg_log_loss':metrics.log_loss,                        # requires predict_proba support
                'precision':metrics.precision_score,                    # suffixes apply as with ‘f1’
                'recall':metrics.recall_score,                          # suffixes apply as with ‘f1’
        #         'jaccard':metrics.jaccard_score,                        # suffixes apply as with ‘f1’
                'roc_auc':metrics.roc_auc_score,
            },
            'Clustering':
            {
                'adjusted_mutual_info_score':metrics.adjusted_mutual_info_score,
                'adjusted_rand_score':metrics.adjusted_rand_score,
                'completeness_score':metrics.completeness_score,
                'fowlkes_mallows_score':metrics.fowlkes_mallows_score,
                'homogeneity_score':metrics.homogeneity_score,
                'mutual_info_score':metrics.mutual_info_score,
                'normalized_mutual_info_score':metrics.normalized_mutual_info_score,
                'v_measure_score':metrics.v_measure_score,
            },
            'Regression':
            {
                'explained_variance':metrics.explained_variance_score,
        #         'max_error':metrics.max_error,
                'neg_mean_absolute_error':metrics.mean_absolute_error,
                'neg_mean_squared_error':metrics.mean_squared_error,
                'neg_mean_squared_log_error':metrics.mean_squared_log_error,
                'neg_median_absolute_error':metrics.median_absolute_error,
                'r2':metrics.r2_score,
            }
        }

    return models, params_cv, metrics


# import numpy as np
# import csv
# from sklearn.datasets.base import Bunch

# def load_my_fancy_dataset():
#     with open('my_fancy_dataset.csv') as csv_file:
#         data_file = csv.reader(csv_file)
#         temp = next(data_file)
#         n_samples = int(temp[0])
#         n_features = int(temp[1])
#         data = np.empty((n_samples, n_features))
#         target = np.empty((n_samples,), dtype=np.int)

#         for i, sample in enumerate(data_file):
#             data[i] = np.asarray(sample[:-1], dtype=np.float64)
#             target[i] = np.asarray(sample[-1], dtype=np.int)

#     return Bunch(data=data, target=target)

# mfd = load_my_fancy_dataset()
# X = mfd.data
# y = mfd.target

class HopeML:
    '''
        Declare class with type issues:
            Classification / Regression / ...
    '''
    def __init__(self, value):
        
        support_issue = ['Classification', 'Regression']
        
        if value not in support_issue:
            print('Not support type', value)
            print('select type in:', ', '.join(support_issue))
        
        self.target = np.nan
        self.features = []
        self.with_null_fields = []
        self.string_variables = []
        self.type_issues = value
        self.transRu = []
        self.cat_features = []
        self.SEED = 17
        # encode
        self._is_le_encode = None
        self._is_dummies = None
        
        
        # grid searches and select models
        self.grid_searches = {}
        self.best_models = {}
        
        self.models, self.params_cv, self.metrics = declare_models(value, self.SEED)
        

    def load_df(
        self,
        path_df = None,
        path_xtrain = None,
        path_xtest = None,
        target = None,
        is_print = True,
        reduce_mem = True,
        exclude_features = []
    ):
        if target == None:
            print('select target')
            pass
        
        if path_df != None:
            self.df = df_from_path(path_df)
            print('Success load df')
        elif path_xtrain != None and path_xtest != None:
            train = df_from_path(path_xtrain)
            print('Success load train')
            test = df_from_path(path_xtest)
            print('Success load test')
            self.df = pd.concat([train, test], sort= False)
            self.df.reset_index(inplace=True, drop=True)
            print('Success concat train & test')
            import gc;
            del train, test; gc.collect();
		
		

        if target not in self.df.columns:
            return print(target,'not in',self.df.columns)
        
        self.exclude_features = exclude_features
        self.columns = set_to_list(self.df.columns, self.exclude_features)
        self.target = target
        self.features = set_to_list(self.columns, [target])
        self.string_variables = set_to_list(list(self.df.select_dtypes('object').columns), self.exclude_features)
        self.features = set_to_list(self.features, self.string_variables)
        self.with_null_fields = set_to_list(
            set_to_list(
                self.df.columns[self.df.isnull().any()]
                , [target]
            )
            , self.exclude_features
        )
        
        self.numeric_features = self.features
        
        
        if reduce_mem:
            print('Reduce memory usage:')
            get_ipython().run_line_magic('time', 'self.df = fast_reduce_mem_usage(self.df, verbose=True)')
            print('Success reduce.')
        if is_print:
            print('*'*40)
            print('Info df:')
            self.get_df_info()

    def update_xtrain_xtest(
        self,
        use_best_features = True
    ):

        '''
            read two files and concat to df with Null target for x_test
        '''

        self.x_train = self.df[self.df[self.target].isnull()==False][self.features].copy()
        self.y_train = self.df[self.df[self.target].isnull()==False][self.target].values.copy()
        self.x_test = self.df[self.df[self.target].isnull()==True][self.features].copy()
                

    def get_df_info(
        self
    ):
        print(
            'features - ', ', '.join(self.features)
            , '\ntarget - ', self.target
            , '\nhave null - ', ', '.join(self.with_null_fields)
            , '\nVarchar - ', ', '.join(self.string_variables)

        )
        
    
    def get_train_test_val(
        self,
        test_size = 0.25,
        # fillna = None,
        not_nan_target = True,
        isTimeSeries = False,
        is_pca_params = {
            'Calc':False,
            'n_components' : 0, 
            'plot':False,
            'PCA_Features':'add',
            'best_count' : False
        },
        is_print = False
# [False, 0]
    ):
        
        '''
            if add prepare with pca, add in list n_components_values
            'PCA_Features':'add' / 'PCA_Features':'replace'
            n_components : count / 'best'
        '''
        
        if is_pca_params['Calc'] == True:
            # replace temp variable
            self.features = self.get_pca_components(is_pca_params)
        
        if isTimeSeries:
            # print('TimeSeries split data')
            self.x_train_val, self.x_test_val, self.y_train_val, self.y_test_val = train_test_split_sorted(
                X = self.df[self.df[self.target].isnull() == False][self.features]
                , y = self.df[self.df[self.target].isnull() == False][self.target]
                , test_size = test_size
            )
        else:
            self.x_train_val, self.x_test_val, self.y_train_val, self.y_test_val = train_test_split(
                self.df[self.df[self.target].isnull() == False][self.features]
                , self.df[self.df[self.target].isnull() == False][self.target]
                , test_size=test_size
                , random_state = self.SEED
            )
        if is_print:
            print('TimeSeriesSplit' if isTimeSeries else 'TrainTestSplit','data with '+str(len(self.x_train_val.columns))+' columns')
        
        

    ### 2. work with features, pandas profiling, ohe, le, dummies
    
    def get_pca_components(self, is_pca_params):

        if is_pca_params['Calc'] == True and is_pca_params['n_components'] < 2:
            print('Check PCA params, True but n_comps < 2')
        
        if is_pca_params['Calc'] == True and is_pca_params['n_components'] > 1:

            # get only have target data
            X = self.df[self.features].copy()
            # centroid data for get elipsoid
            X_centered = X - X.mean(axis=0)
            
            # Search best n-components
            pca = PCA().fit(X)
            pca_search_ncomp = pd.DataFrame(
                np.cumsum(pca.explained_variance_ratio_)
                , columns = ['Cumsum importance of components']
            )

            pca_search_ncomp = pca_search_ncomp
            best_count = pca_search_ncomp[pca_search_ncomp['Cumsum importance of components'] < 0.9].shape[0]+1
            best_count = np.max(
                [np.min([best_count, pca_search_ncomp.shape[0]]), 2]
            )
            
            best_variance = pca_search_ncomp['Cumsum importance of components'].values[best_count]
            # .max()

            plt.figure(figsize=(10,7))
            plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
            plt.xlabel('Number of components')
            plt.ylabel('Total explained variance')
            plt.xlim(0, 63)
            plt.title('Count components for ' +str(format(best_variance, '.0%'))+ ' variance:' + str(best_count) )
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.axvline(best_count, c='b')
            plt.axhline(0.9, c='r')
            plt.show();
            
            
            # declare # create temp var for fix features # PCA_Features: 'add'/'replace' 
            n_comp = is_pca_params['n_components'] if is_pca_params['best_count'] == False else best_count
            x_pca_columns = [str(c+1) + ' PCA component' for c in range(n_comp)]
            if is_pca_params['PCA_Features'] == 'add':
                features_val = self.features + x_pca_columns
            elif is_pca_params['PCA_Features'] == 'replace':
                features_val = x_pca_columns
            else:
                features_val = self.features
            
            pca = PCA(
                n_components = n_comp
            )
            
            # fit centered pca data
            pca.fit(X_centered)
            # Transform pca data with n_components
            X_pca = pca.transform(X_centered)
            
            print('PCA Report.\nExplained variance:', ', '.join([format(c,'.1%') for c in pca.explained_variance_ratio_]))
            print('Sum variance:', format(pca.explained_variance_ratio_.sum(),'.1%'))

            X_pca_df = pd.DataFrame(X_pca, columns = x_pca_columns)
            for c in x_pca_columns:
                self.df[c] = X_pca_df[c]

            # get df weights for each columns
            pca_comp_df = pd.DataFrame(pca.components_, columns = self.features).T*100
            pca_comp_df.columns = [str(c) + ' PCA component' for c in pca_comp_df.columns]
            pca_comp_df['Mean_Weight'] = pd.DataFrame(pd.DataFrame(pca.components_, columns = self.features).T.mean(axis=1)*100, columns = ['Mean Weigth'])

            self.pca_comp_df = pca_comp_df
            
            if is_pca_params['plot'] == True:
                # И нарисуем получившиеся точки в нашем новом пространстве
                plt.plot(
                    X_pca.T[:1].T, # 1 components
                    X_pca.T[1:2].T, # 2 components
                    c='b',
                    label=str('PCA')
                )
                plt.legend(loc=0);
                plt.show();
        return features_val
    
    def df_replace_nan_values(
        self
        , df
        , type_replace_in_columns
        , all_replace_type = None
        , is_print = True
    ):
        '''
            Working with null values
            1. change null to value ( 0, -1, -9999)
            2. min/max in field ( min, max )
            3. aggregate_groupby_field (field - groupby_field - function transform)
            
            type_fillna = [
                ['Age','replace_to_value', 1],
                ['Age','function_field','mean'],
                ['Fare','function_field','std'],
                ['Fare','aggregate_groupby_field',['Sex', 'Pclass'], 'mean'],
            ]
            
        '''

        df = df.copy()
        
        for row in type_replace_in_columns:
            
            key = row[0] # field
            metod = row[1] # metod fillna
            fill_value = row[2] # value fillna / function / groupby columns

            if metod == 'replace_to_value':
                if is_print:
                    print('*'*30)
                    print('field:', key)
                    print('metod:', metod)
                    print('value:', fill_value)
                df[key+'_'+metod+'_'+str(fill_value)] = df[key].fillna(fill_value)
                self.features.append(key+'_'+metod+'_'+str(fill_value))
                self.numeric_features.append(key+'_'+metod+'_'+str(fill_value))

            elif metod == 'function_field':
                if is_print:
                    print('*'*30)
                    print('field:', key)
                    print('metod:', metod)
                    print('function:', fill_value)
                df[key+'_'+metod+'_'+str(fill_value)] = df[key].fillna(
                    df[key].astype('float32').agg(fill_value)
                ).astype(df[key].dtype)
                self.features.append(key+'_'+metod+'_'+str(fill_value))
                self.numeric_features.append(key+'_'+metod+'_'+str(fill_value))

            elif metod == 'aggregate_groupby_field':
                groupby_columns = row[2]
                function_agg = row[3]
                
                if is_print:
                    print('*'*30)
                    print('field:',key)
                    print('metod:',metod)
                    print('groupby:',groupby_columns)
                    print('function:',function_agg)
                    
                df[key+'_gb_'+'_'.join(groupby_columns)] = df[key].fillna(
                    df.groupby(groupby_columns)[key].astype('float32').transform(function_agg)
                ).astype(df[key].dtype)
                
                self.features.append(key+'_'+metod+'_'+str(fill_value))
                # add for binn discr..
                self.numeric_features.append(key+'_'+metod+'_'+str(fill_value))
                
                
        return df
    
    
    def automated_fillna_df(
        self
        , drop_null_fields = True
        , replace_to_value = [0] # -1,
        , function_field = ['mean', 'std']
        , aggregate_groupby_field = []
        , create_sum_null_feature = True
    ):
        
        # self.with_null_fields = set_to_list(self.with_null_fields,self.string_variables)
        
        type_fillna = []
        
        if create_sum_null_feature:
            self.df['CountColumnsNull'] = self.df[self.with_null_fields].isnull().sum(axis=1)
            self.features = self.features + ['CountColumnsNull']

        if len(replace_to_value) > 0:
            for f in set_to_list(self.with_null_fields,self.string_variables):
                for value in replace_to_value: # , -1, -999
                    type_fillna.append(
                        [f,'replace_to_value', value]
                    )
#             for value in replace_to_value: # , -1, -999
#                 self.df[
#                     set_to_list(self.with_null_fields,self.string_variables)
#                 ]
            
                    
        if len(function_field) > 0:
            for f in set_to_list(self.with_null_fields,self.string_variables):
                for value in function_field: # , 'min'
                    type_fillna.append(
                        [f,'function_field', value]
                    )
        
        if len(aggregate_groupby_field) > 0:
            for f in ML.with_null_fields:
                for value in ['mean', 'std', 'min']:
                    type_fillna.append(
                        [f,'aggregate_groupby_field', value]
                    )
        
        self.df = self.df_replace_nan_values(self.df, type_fillna)
        

#         null_field = 'Survived'
#         agg_field = 'Sex'
#         # calc
#         smothed_aggregate(
#             temp_df, 
#             null_field = null_field,
#             agg_field = agg_field,
#             alpha = 10
#         )

        if drop_null_fields:
            self.features = set_to_list(
                self.features
                ,set_to_list(self.with_null_fields,self.string_variables)
            )
            
            self.numeric_features = set_to_list(
                self.numeric_features
                ,set_to_list(self.with_null_fields,self.string_variables)
            )
            
            self.df.drop(set_to_list(self.with_null_fields,self.string_variables), axis=1,inplace=True)
            
            

    
    def pandas_profiling(self, count_row = None, clear_corr_feat = True):
        # def clear_rejected_variables(self):
        #     self.features = list( set(self.features) - set(self.pp_rejected_variables) )

        if count_row == None: 
            count_row = self.df.shape[0] - 1

        profile = pandas_profiling.ProfileReport(self.df[:count_row][self.features])
        self.pp_rejected_variables = profile.get_rejected_variables(threshold=0.9)
        
        if clear_corr_feat:
            print('Clear corr features:', ', '.join(self.pp_rejected_variables))
            self.features = list( set(self.features) - set(self.pp_rejected_variables) )
        return profile

    
    # Outlier detection
    def detect_outliers(
        self
        , df
        , n
        , features
    ):
        """
        Для обнаружения выбросов, которые определяют межквартильный диапазон, заключенный между 1-м и 3-м квартилями значений распределения (IQR). 
        Выброс-это строка, которая имеет значение объекта за пределами (IQR +- шаг выброса).
        Принимает фрейм данных df и возвращает список индексов
        соответствующие наблюдениям, содержащим более n выбросов согласно к методу Тьюки.
        """
        
        df = df.copy()
        
        outlier_indices = []
        from collections import Counter
        # iterate over features(columns)
        for col in features:
            # 1st quartile (25%)
            Q1 = np.percentile(df[col], 25)
            # 3rd quartile (75%)
            Q3 = np.percentile(df[col],75)
            # Interquartile range (IQR)
            IQR = Q3 - Q1

            # outlier step
            outlier_step = 1.5 * IQR

            # Determine a list of indices of outliers for feature col
            outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

            # append the found outlier indices for col to the list of outlier indices 
            outlier_indices.extend(outlier_list_col)

        # select observations containing more than 2 outliers
        outlier_indices = Counter(outlier_indices)
        multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

        return multiple_outliers
    

    def get_update_string_variables(self):
        self.string_variables = set_to_list(list(self.df.select_dtypes('object').columns), self.exclude_features)

    def get_update_null_variables(self):
        self.with_null_fields = set_to_list(
            set_to_list(
                self.df.columns[self.df.isnull().any()]
                , [self.target]
            )
            , self.exclude_features
        )
    
    def set_le_encode_object_var(self, fillna = False):
        self.get_update_string_variables()
        if self._is_le_encode == None and len(self.string_variables)>0:
            print('LE preprocess for:', ', '.join(self.string_variables))
            from sklearn.preprocessing import LabelEncoder
            for sv in self.string_variables:
                # print(sv)
                if fillna:
                    self.df[sv] = self.df[sv].fillna('nan')
                    self.df[sv] = LabelEncoder().fit_transform(self.df[sv])
                else:
                    self.df[[sv]] = self.df[[sv]].apply(lambda series: pd.Series(
                        LabelEncoder().fit_transform(series[series.notnull()]),
                        index=series[series.notnull()].index
                    ))

                self.features.append(sv)
                self.cat_features.append(sv)

            self._is_le_encode = True

    def set_dummies_object_var(self):
        self.get_update_string_variables()
        if self._is_dummies == None and len(self.string_variables)>0:
            print('Dummies preprocess for:', ', '.join(self.string_variables))
            
            # save change
            self.features = list(
                self.features + list(
                    pd.get_dummies(self.df[self.string_variables]).columns
                )
            )
            self.cat_features = list(
                self.cat_features + list(
                    pd.get_dummies(self.df[self.string_variables]).columns
                )
            )
            
            self.df = pd.get_dummies(self.df, columns = self.string_variables)
            # update flag
            self._is_dummies = True


    def set_dummies(self, variables):
        if len(variables)>0:
            print('Dummies preprocess for:', ', '.join(variables))
            # save change
            dummies_columns = pd.get_dummies(self.df[variables]).columns

            self.features = list(
                self.features + list(
                    dummies_columns
                )
            )
            
            self.cat_features = list(
                self.cat_features + list(
                    dummies_columns
                )
            )

            self.df = pd.get_dummies(self.df, columns = variables)
            self.features = set_to_list(self.features, variables)
            self.cat_features = set_to_list(self.cat_features, variables)

    # column Transformer df 
    def set_columnTransformer_df(
        self
        , properties = None
        # , features = None
    ):
        '''
            pipeline transformer for features
            [
                ("scaling", StandardScaler(), ['Parch', 'SibSp']),
            ]
        '''

        alg = {
            "scaling" : StandardScaler()
        }

        if properties!=None:
            pipeline = [(p, alg[p], properties[p]) for p in properties.keys()]        

            features = []
            for i in pipeline:
                for c,j in enumerate(i[2]):
                    features.append(i[2][c])

            ct = ColumnTransformer(pipeline)

            ct.fit(self.df)
            self.df[features] = ct.transform(self.df)

    # bins for features   
    def set_KBinsDiscretizer_features(
        self
        , numeric_features = None
        , drop_base_columns_from_df = False
        , params = {'n_bins': 3, 'strategy': 'quantile', 'encode': 'onehot-dense'}
    ):

        if numeric_features == None:
            numeric_features = self.numeric_features    

        # reset index на всякий случай
        self.df.reset_index(drop=True, inplace=True)

        kb = KBinsDiscretizer(
            n_bins=params['n_bins']
            , strategy=params['strategy'] # 'uniform', 'quantile', 'kmeans' # 
            , encode=params['encode'] # 'onehot', 'onehot-dense', 'ordinal'
        )

        for feature in numeric_features:
            X_bin = kb.fit(self.df[[feature]]).transform(self.df[[feature]])
            X_bin_columns = [
                feature + ': min to ' + str(
                    round(i, 3)
                ) if n == 0 else feature + ': ' + str(
                    round(kb.bin_edges_[0][1:][n-1], 3)
                ) + ' to ' + str(
                    round(i, 3)
                ) for n, i in enumerate(list(kb.bin_edges_[0][1:]))
            ]

            X_bin_df = pd.DataFrame(
                X_bin
                , columns = X_bin_columns
            ).astype('uint8')


            # drop columns without information
            count0_df = X_bin_df.sum().reset_index().rename(
                columns = {
                    'index':'Features', 0: 'Count0'
                }
            )
            zero_features = list(count0_df[count0_df['Count0'] < 1]['Features'].values)
            # X_bin_columns = set_to_list(, zero_features)        

            for zc in list(count0_df[count0_df['Count0'] < 1]['Features'].values):
                X_bin_df.drop(zc, axis=1, inplace=True)
                X_bin_columns.remove(zc)

            print('For',feature,'create:',', '.join(X_bin_columns))

            if len(zero_features) > 0:
                print('For',feature,'drop zero:',', '.join(zero_features))

            self.df[X_bin_df.columns] = X_bin_df[X_bin_df.columns]
    #         self.df = pd.concat(
    #             [
    #                 self.df, X_bin_df
    #             ]
    #             , axis=1
    #             , sort=False
    #         ) # [[feature] + X_bin_columns]

            # add to features
            self.features = self.features + X_bin_columns

            if drop_base_columns_from_df == True:
                # del main features (without bin discretizer) from all features
                self.features = set_to_list(self.features, [feature])


    # month, week, day, dayofweek
    def get_cyclical_encode(
        self,
        cols_maxval = {},
        is_drop = False
    ):
        for col in cols_maxval.keys():
            print('Start ', col)
            self.df[col + '_sin'] = np.sin(2 * np.pi * self.df[col]/cols_maxval[col])
            self.df[col + '_cos'] = np.cos(2 * np.pi * self.df[col]/cols_maxval[col])
            print('Add', col + '_sin',col + '_cos')

            # добавляем в фитчи
            self.features = self.features + [col + '_sin', col + '_cos']
            print('Add in features')

            if is_drop:
                # удаляем старые
                self.df.drop(col, axis=1, inplace=True)
                self.features = set_to_list(self.features, [col])
                print('Drop in features')

    def drop_corr_features(self, perc = 0.95, is_print = True):
        # Identify Highly Correlated Features
        # Create correlation matrix
        corr_matrix = self.df[self.features].corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find index of feature columns with correlation greater than 0.95/perc var
        self.to_drop_corr_feat = [column for column in upper.columns if any(upper[column] > perc)]
        print(self.to_drop_corr_feat)
        # Drop Marked Features
        self.df.drop(self.to_drop_corr_feat, axis=1, inplace = True)
        self.features = set_to_list(self.features, self.to_drop_corr_feat)
        self.cat_features = set_to_list(self.cat_features, self.to_drop_corr_feat)

    ### 3. Cross-validation, select model, select features, score summary, correlation relationship models
    def fit_cv(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False, is_RandomizedSCV = False, is_timeseries_split = False):
        
       
        if is_timeseries_split:
            if verbose > 0:
                print('Get TimeSeriesSplit')
            self.Kf = TimeSeriesSplit(n_splits=cv)
            # cv = TimeSeriesSplit(n_splits=cv)
        else:
            ### 1. df and X_test/X_train load/prepare/features
            if verbose > 0:
                print('Get StratifiedKFold') 
            self.Kf = StratifiedKFold(n_splits=cv, shuffle=False, random_state=0)
            # cv = StratifiedKFold(n_splits=cv, shuffle=False, random_state=0)
            

        self.grid_searches = {}
        self.best_models = {}
        for key in self.models.keys():
            if verbose > 0:
                print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params_cv[key]

            # For Neural Network, declare input shape
            if key in ['3DenseNN_Classifier']: 
                is_cnn = True
                global number_input_shape_for_cnn
                number_input_shape_for_cnn = len(self.features)
            else:
                is_cnn = False

            # test, в LGBM хочу определять кат. фитчи
            if key == 'LGBMClassifier':
                # fit_params={'categorical_feature' : list(self.cat_features)}
                fit_params={'categorical_feature': 'auto'}
            else:
                fit_params=None

            if is_RandomizedSCV:#  and is_cnn == False:
                gs = RandomizedSearchCV(
                    estimator = model, 
                    param_distributions = params, 
                    cv=cv,# if is_cnn == False else None, 
                    n_jobs=None if is_cnn else n_jobs,
                    verbose=verbose, 
                    scoring=scoring, 
                    refit=refit,
                    return_train_score=True, # , fit_params=fit_params
                    random_state = self.SEED,
                )
            else:
                gs = GridSearchCV(
                    estimator = model, 
                    param_grid = params, 
                    cv=cv, 
                    n_jobs=n_jobs,
                    verbose=verbose, 
                    scoring=scoring, 
                    refit=refit,
                    return_train_score=True,
                    # random_state = self.SEED,
                )

            if model.__class__.__name__ == 'LGBMRegressor':
                # print('use kw_params')
    #             print(self.x_test_val)
    #             print(self.y_test_val)
                kw_params = {
                    'eval_set':[(self.x_train_val, self.y_train_val), (self.x_test_val, self.y_test_val)],
                    'verbose':100
                }
            else:
                kw_params = {}
                
            gs.fit(
                X[self.features].as_matrix()
                , y.values
                , **kw_params
            )
            
            self.grid_searches[key] = gs

            # print(gs.best_estimator_.get_params())
            self.best_models[key] = model.set_params(**self.grid_searches[key].best_params_)
            # self.grid_searches[key]['best_params']

        # update stat for best features on df
        self.get_best_features()
            
    def get_best_features(
        self
        , fe_selector = SelectFromModel# RFE # ()
        , use_best_features = False
    ):
        SFM = {}
        feature_selector = {}
        # True False for each variable
        feature_support = {}
        dont_support = []
        for key in self.best_models.keys():

            # key = 'ExtraTreesClassifier'
            # SFM[key] = SelectFromModel(self.best_models[key], threshold='1.25*median')
            SFM[key] = fe_selector(self.best_models[key], threshold='1.25*median')
            SFM[key].fit(self.x_train_val[self.features], self.y_train_val)
            try:
                feature_support[key] = SFM[key].get_support()
                feature_selector[key] = self.x_train_val[self.features].loc[:,feature_support[key]].columns.tolist()
            except Exception:
                print('For',key,'dont support feature selection get_support')
                dont_support.append(key)
                pass
    
        dataframe = {}
        dataframe['Features'] = self.features

        for key in SFM.keys():
            if key not in dont_support:
                dataframe[key] = feature_support[key]
        
        # print(dataframe)
        self.feature_selection_df = pd.DataFrame(dataframe)   
        # count the selected times for each feature
        self.feature_selection_df['Total'] = np.sum(self.feature_selection_df, axis=1)
        # display the top 100
        self.feature_selection_df.sort_values(['Total','Features'] , ascending=False, inplace = True)
        self.feature_selection_df.index = range(1, len(self.feature_selection_df)+1)
        if use_best_features:
            old_features = self.features
            self.features = list(self.feature_selection_df[self.feature_selection_df['Total'] > 0]['Features'].values)
            print('Use best features from all model', ', '.join(old_features))
            
        # return feature_selection_df
    
    # Model-driven Imputation w RF
    # exclude object and datetime
    def set_predict_null_values(
        self,
        features_for_predict,
        # ensemble = []
        type_predict, # Regressor, Classifier
        n_estimators = 30,
        n_jobs = 20,
    ):
        
        from tqdm import tqdm_notebook
        if type_predict == 'Regressor':
            model = RandomForestRegressor(
                n_estimators=n_estimators, n_jobs = n_jobs, random_state = self.SEED
            )
        elif type_predict == 'Classifier':
            model = RandomForestClassifier(
                n_estimators=n_estimators, n_jobs = n_jobs, random_state = self.SEED
            )

        df_null_stat = pd.DataFrame(self.df[features_for_predict].isnull().sum(), columns = ['CountNull'])
        df_null_stat.sort_values('CountNull', inplace = True)
        predict_null_fields = list(df_null_stat[df_null_stat['CountNull'] > 0].index)
        
        print('Apply encoder for this columns, they have object or datetime variables',', '.join(self.df[features_for_predict].select_dtypes(include=['object', 'datetime']).columns))
        print(list(self.df[features_for_predict].select_dtypes(include=['object', 'datetime']).columns))

        # predict_null_fields = set_to_list(
        #     predict_null_fields
        #     , self.df[features_for_predict].select_dtypes(include=['object', 'datetime']).columns
        # )
        

        for c in self.df[features_for_predict].select_dtypes(include=['object', 'datetime']).columns:
            if c in predict_null_fields:
                predict_null_fields.remove(c)
                print('Remove obj/datetime var:', c)

        print('\nFeatures changes:',', '.join(predict_null_fields))
        features_wo_null = set_to_list(
            # set_to_list(self.features,predict_null_fields)
            self.df[set_to_list(self.features, self.with_null_fields)].columns
            , self.df.select_dtypes(include=['object', 'datetime']).columns
        )
        
        for c in tqdm_notebook(predict_null_fields):
            x_train = self.df[self.df[c].isnull()==False][features_wo_null]
            x_test = self.df[self.df[c].isnull()==True][features_wo_null]
            y_train = self.df[self.df[c].isnull()==False][c]

            print(
                'Predict:',c,'\nCount Null string:', x_test.shape[0],
                '\nFeatures wo null in train:', ', '.join(features_wo_null),
            )
            
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            x_test[c] = y_pred

            df_without_null_c = pd.DataFrame(
                pd.concat([self.df[self.df[c].isnull()==False][c], x_test[c]])
                , columns = [c]
            )

            self.df[c] = df_without_null_c.sort_index()[c]

            features_wo_null.append(c)
    
    def score_summary(self, sort_by='mean_score'):
        
            def row(key, scores, params):
                d = {
                     'estimator': key,
                     'min_score': min(scores),
                     'max_score': max(scores),
                     'mean_score': np.mean(scores),
                     'std_score': np.std(scores),
                }
                return pd.Series({**params,**d})

            rows = []
            for k in self.grid_searches:
                # print(k)
                params = self.grid_searches[k].cv_results_['params']
                scores = []
                for i in range(self.grid_searches[k].cv):
                    key = "split{}_test_score".format(i)
                    r = self.grid_searches[k].cv_results_[key]        
                    scores.append(r.reshape(len(params),1))

                all_scores = np.hstack(scores)
                for p, s in zip(params,all_scores):
                    rows.append((row(k, s, p)))

            self.df_score_cv = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)
            self.df_score_cv.reset_index(drop=True, inplace=True)
            columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
            columns = columns + [c for c in self.df_score_cv.columns if c not in columns]

            return self.df_score_cv[columns]

    def _get_score_val_test(self, ensemble_df, y_true, keys):
        accuracy_val = {}
        for key in keys:
            accuracy_val[key] = {
                metric.__name__ : metric(ensemble_df[y_true], ensemble_df[key]) for metric in self.metrics
            }

        self.accuracy_ensemble_results = pd.DataFrame(accuracy_val)

       
    def get_corr_models(self, is_plot = False):
        accuracy_val = {}
        ensemble_results = {}
        ensemble_results['y_test'] = self.y_test_val.values

        for key in self.best_models.keys():
            ensemble_results[key] = self.best_models[key].fit(self.x_train_val, self.y_train_val).predict(self.x_test_val)
        self.ensemble_results = pd.DataFrame(ensemble_results)
        
        self._get_score_val_test(self.ensemble_results, 'y_test', list(self.best_models.keys()))
        
        if is_plot:
            g = sns.heatmap(self.ensemble_results.corr(),annot=True)


    def get_clf_voiting(self, is_val = True, n_jobs=3):
        '''
            Select best model with help VoitingClassifier
        '''
        # maybe add to params x_train, y_train, x_test
        if is_val:
            # train
            x_train = self.x_train_val
            y_train = self.y_train_val
            # test
            x_test = self.x_test_val
            y_test = self.y_test_val

        else:
            # predict and 
            x_train = self.x_train
            y_train = self.y_train
            x_test = self.x_test

        estimators = []
        for key in self.best_models.keys():
            estimators.append((key, self.best_models[key]))

        VC = VotingClassifier(estimators=estimators, voting='soft', n_jobs=n_jobs)
        VC.fit(x_train, y_train)
        self.y_vc_pred = pd.Series(VC.predict(x_test), name=self.target)

        if is_val:
            ensemble_results = {}
            ensemble_results['y_test'] = self.y_test_val.values
            ensemble_results['VoiceClfPred'] = self.y_vc_pred.values

            for key in self.best_models.keys():
                ensemble_results[key] = self.best_models[key].fit(self.x_train_val, self.y_train_val).predict(self.x_test_val)

            self.ensemble_results = pd.DataFrame(ensemble_results)
            
            # get score on validation test
            self._get_score_val_test(self.ensemble_results, 'y_test', list(self.best_models.keys())+['VoiceClfPred'])

            
    # just for classification eda
    def fast_classification_eda_object(self, features = False, max_nunique_val = 20):
        '''
            Categorial EDA
        '''

        if features == False or type(features) != list:
            features = self.string_variables
            
        if self.type_issues == 'Classification':
            for feature in features:
                if len(self.df[feature].unique())>2:
                    a = len(self.df[feature].unique())
                    if a <= max_nunique_val:
                        plt.figure(figsize = [15,min(max(8,a),5)])

                        plt.subplot(1,2,1)
                        x_ = self.df.groupby([feature])[feature].count()
                        x_.plot(kind='pie')
                        plt.title(feature)

                        plt.subplot(1,2,2)
                        cross_tab = pd.crosstab(self.df[self.target],self.df[feature],normalize=0).reset_index()
                        x_ = cross_tab.melt(id_vars=[self.target])
                        x_['value'] = x_['value']

                        sns.barplot(
                            x=feature,
                            y='value', 
                            hue = self.target if len(self.df[self.target].unique()) < 10 else None,
                            data=x_,
                            palette = ['b','r','g'],alpha =0.7
                        )

                        plt.xticks(rotation='vertical')
                        plt.title(feature + " - " + self.target)

                        plt.tight_layout()
                        plt.legend()
                        plt.show()
                    else:
                        print('Unique values for',feature,'so much..')

    # just for classification eda NUMERIC
    def fast_classification_eda_numeric(self, features = False):
        '''
            Numeric Eda for Classification
        '''
        if features == False:
            features = self.features

        if self.type_issues == 'Classification':

            for feature in features:
                if len(self.df[feature].unique())>2:
                    x_ = self.df[feature]
                    y_ = self.df[self.target]

                    data = pd.concat([x_,y_],1)
                    plt.figure(figsize=[15,5])

                    ax1 = plt.subplot(1,2,1)
                    sns.boxplot(x=self.target,y=feature,data=data)
                    plt.title(feature+ " - Boxplot")
                    upper_0 = data[data[self.target]==0][feature].quantile(q=0.75)
                    upper_1 = data[data[self.target]==1][feature].quantile(q=0.75)
                    lower_0 = data[data[self.target]==0][feature].quantile(q=0.25)
                    lower_1 = data[data[self.target]==1][feature].quantile(q=0.25)

                    ax1.set(ylim=(min(lower_0,lower_1),max(upper_0,upper_1)))

                    ax2 = plt.subplot(1,2,2)
                    plt.title(feature+ " - Density with Log")

                    p1=sns.kdeplot(data[data[self.target]==0][feature].apply(np.log), color="b",legend=False)
                    p2=sns.kdeplot(data[data[self.target]==1][feature].apply(np.log), color="r",legend=False)
                    plt.legend(loc='upper right', labels=['0', '1'])

                    plt.tight_layout()
                    plt.show() 

    # plot learning curve for best estimator
    def plot_learning_curve(self, estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
        """Generate a simple plot of the test and training learning curve"""
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        return plt

    def plot_score_summary(self, cv = 5):
        for key in self.models.keys():
            self.plot_learning_curve(self.best_models[key],key,self.x_train_val,self.y_train_val,cv=cv, n_jobs=None)

    def get_final_stack_predict(self):
        '''
            Using Stacking Regressor for blend models
        '''
        stregr = StackingRegressor(
            regressors=[self.best_models[key] for key in self.best_models.keys()],
            meta_regressor=self.best_models[
                # best on cv
                self.df_score_cv['estimator'].values[0]
            ]
        )

        stregr.fit(self.x_train[self.features].fillna(0), self.y_train)
        self.y_stack_pred = stregr.predict(self.x_test[self.features].fillna(0))
        return self.y_stack_pred
    
    def get_predict_best_model(
        self
        , priority = None
        , top_n_models = 5
        , is_predict_proba = False
        , is_val_test = False
        , metrics = None
        , print_result = False
    ):

        '''
            Using best model for predict
        '''


        from copy import deepcopy

        if is_val_test:
            x_train = self.x_train_val
            y_train = self.y_train_val
            x_test = self.x_test_val
            y_test = self.y_test_val
        else:
            x_train = self.x_train
            y_train = self.y_train
            x_test = self.x_test
            try:
                y_test = self.y_test
            except:
                if print_result:
                    print('self.y_test is empty')

        # create dict with n-top models and best params
        self.df_score_cv.reset_index(drop=True, inplace=True)
        top_models = {}
        feat = set_to_list(self.df_score_cv.columns, ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score'])
        for i in range(top_n_models):
            params = {}
            for f in feat:
                value_f = self.df_score_cv.loc[i][f]
                type_f = type(value_f)
                if ( type_f != str and value_f != np.nan and np.isnan(value_f) == False ) or type_f == str:
                    params[f] = value_f
            top_models[
                self.df_score_cv.loc[i]['estimator'] + ' top ' + str(i+1)
            ] = deepcopy(
                self.models[
                    self.df_score_cv.loc[i]['estimator']
                ].set_params(**params)
            )

        if priority != None:
            params = {}
            top_models = {}
            i = 0

            priority_df_score_cv = self.df_score_cv[self.df_score_cv['estimator']==priority].copy()
            priority_df_score_cv.reset_index(drop=True, inplace=True)
            for f in feat:
                value_f = priority_df_score_cv.loc[i][f]
                type_f = type(value_f)
                if ( type_f != str and value_f != np.nan and np.isnan(value_f) == False ) or type_f == str:
                    params[f] = value_f

            top_models[
                priority_df_score_cv.loc[i]['estimator'] + ' top ' + str(i+1)
            ] = deepcopy(
                self.models[
                    priority_df_score_cv.loc[i]['estimator']
                ].set_params(**params)
            )

        predictions = pd.DataFrame()
        for best_model in top_models.keys():
            name_model = best_model
            # best_model = self.best_models[name_model]
            if print_result:
                print('calc best model ' + name_model + ' with params:\n',top_models[best_model].get_params())
            top_models[best_model].fit(x_train[self.features].fillna(0), y_train)

    #         if is_predict_proba:
    #             self.y_pred_bm = best_model.predict_proba(
    #                 x_test[self.features].fillna(0)
    #             )
    #         else:
    #             self.y_pred_bm = best_model.predict(
    #                 x_test[self.features].fillna(0)
    #             )

            predictions[name_model] = top_models[best_model].predict_proba(x_test[self.features].fillna(0)).T[1]

        self.y_pred_bm = predictions.mean(axis=1).values

        if is_val_test:
            if metrics == None:
                if self.type_issues == 'Classification':
                    metrics = accuracy
                elif self.type_issues == 'Regression':
                    metrics = r2_score
            if print_result:
                print('Test val with metric:',metrics.__name__)
                print('Accuracy:',format(metrics(y_test, self.y_pred_bm),'.1%'))

        return self.y_pred_bm
    
    # get predict on train and test for linear models
    def get_oof_pred_models(
        self,
        ntop_models=3,
        n_jobs=-1,
        n_splits=4,
        shuffle_kfold=False,
        is_proba=True,
        oof_result_print=True,
        metric_print=roc_auc_score
    ):
        '''
            HML.get_oof_pred_models(is_proba=False, n_jobs = 25)
        '''
        from tqdm import tqdm_notebook
        from copy import deepcopy

        top_models = {}

        df_score = self.score_summary()
        df_score.reset_index(drop=True, inplace=True)
        display(df_score)
        
        feat = set_to_list(df_score.columns, ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score'])

        # create dict with n-top models and best params
        for i in range(0,ntop_models):
            params = {}
            for f in feat:
                if np.isnan(df_score.loc[i][f]) == False and df_score.loc[i][f] != None:
                    params[f] = df_score.loc[i][f]
            print(df_score.loc[i]['estimator'], params)
            top_models[
                df_score.loc[i]['estimator'] + ' top ' + str(i+1)
            ] = deepcopy(
                self.models[
                    df_score.loc[i]['estimator']
                ].set_params(**params)
            )

        gc.collect();
        
        # update x_train and x_test for submission
        self.update_xtrain_xtest()
        
        # create df for cv train and test predict
        self.df_oof = pd.DataFrame()
        self.df_pred = pd.DataFrame()

        # for unique top_models calc kfolds and pred val data in train and test
        for m in tqdm_notebook(top_models.keys()):
            model = top_models[m]
            model.n_jobs = n_jobs

            print(m,'with params:', model.get_params())
            folds = KFold(
                n_splits=n_splits,
                shuffle=shuffle_kfold,
                random_state=17
            )

            oof_stack = np.zeros(self.x_train.shape[0])

            for fold_, (trn_idx, val_idx) in enumerate(folds.split(self.x_train, self.y_train)):
                print("fold #{}".format(fold_+1))
                trn_data = self.x_train.loc[trn_idx]
                y_trn_data = pd.Series(self.y_train)[trn_idx].values
                val_data = self.x_train.loc[val_idx]
                model.fit(
                    trn_data,
                    y_trn_data,
                    verbose=1,
                )
                if is_proba:
                    oof_stack[val_idx] = model.predict_proba(val_data).T[1]
                else:
                    oof_stack[val_idx] = model.predict(val_data)

            model.fit(self.x_train, self.y_train)
            
            if is_proba:
                predictions_stack = model.predict_proba(self.x_test).T[1]
            else:
                predictions_stack = model.predict(self.x_test)

            self.df_oof['Top model '+str(m)] = oof_stack
            self.df_pred['Top model '+str(m)] = predictions_stack

        self.df_oof['y'] = self.y_train
        
        if oof_result_print:
            print('OOF RESULT:')
            for f in set_to_list(self.df_oof.columns, ['y']):
                print(
                    f, format(
                        metric_print(self.df_oof['y'], self.df_oof[f].values)
                        , '.1%'
                    )
                )

            print(
                'Mean:', format(metric_print(self.df_oof['y'], self.df_oof[set_to_list(self.df_oof.columns, ['y'])].mean(axis=1)), '.1%')
            )    
    
    
    def get_BayesianStack_models(
        self,
        n_splits = 10,
        score_funtion = roc_auc_score# auc
    ):

        oof = []
        pred = []
        for key in self.best_models.keys():
            oof.append(
                self.best_models[key].fit(self.x_train[self.features], self.y_train).predict(self.x_train[self.features])
            )
            pred.append(
                self.best_models[key].fit(self.x_train[self.features], self.y_train).predict(self.x_test[self.features])
            )

        # ? why vstack
        train_stack = np.vstack(oof).transpose()
        test_stack = np.vstack(pred).transpose()

        folds = KFold(
            n_splits=n_splits,
            shuffle=False,
            random_state=17
        )

        oof_stack = np.zeros(train_stack.shape[0])
        predictions_stack = np.zeros(test_stack.shape[0])

        for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_stack, self.y_train)):
            # print("fold n°{}".format(fold_+1))
            trn_data, trn_y = train_stack[trn_idx], pd.Series(self.y_train).iloc[trn_idx].values
            val_data, val_y = train_stack[val_idx], pd.Series(self.y_train).iloc[val_idx].values

            clf = BayesianRidge()
            clf.fit(trn_data, trn_y)

            oof_stack[val_idx] = clf.predict(val_data)
            predictions_stack += clf.predict(test_stack) / folds.n_splits
        
        
        # oof_stack,
        return predictions_stack
        
    def get_linear_stack_models(
        self,
        model_name='Bayesian', # Bayesian, LassoCV
        n_splits=10,
        shuffle_kfold=False,
        metric_print=roc_auc_score,
        return_stack_pred=True
    ):

        from sklearn.linear_model import LassoCV
        oof = self.df_oof[set_to_list(self.df_oof.columns, ['y'])].T.values
        pred = self.df_pred[set_to_list(self.df_pred.columns, ['y'])].T.values
        train_stack = np.vstack(oof).transpose()
        test_stack = np.vstack(pred).transpose()

        folds = KFold(
            n_splits=n_splits,
            shuffle=shuffle_kfold,
            random_state=17
        )

        oof_stack = np.zeros(train_stack.shape[0])
        self.predictions_stack = np.zeros(test_stack.shape[0])

        for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_stack, self.y_train)):
            print("fold n°{}".format(fold_+1))
            trn_data, trn_y = train_stack[trn_idx], pd.Series(self.y_train).iloc[trn_idx].values
            val_data, val_y = train_stack[val_idx], pd.Series(self.y_train).iloc[val_idx].values
            
            if model_name == 'LassoCV':
                clf = LassoCV(cv=3, alphas=[i/100 for i in range(1000)])
            elif model_name == 'Bayesian':
                clf = BayesianRidge()
            
            clf.fit(trn_data, trn_y)

            oof_stack[val_idx] = clf.predict(val_data)
            self.predictions_stack += clf.predict(test_stack) / folds.n_splits

        metric_print(self.y_train, oof_stack)
        
        if return_stack_pred:
            return self.predictions_stack
        
        
    def get_submission(
        self,
        data,
        name_sample_submission = None,
        save_submission=True,
        return_submission=True,
    ):
        '''
            HML.get_submission(HML.df_pred.mean(axis=1), save_submission=True, return_submission=True).sum()
        '''

        if name_sample_submission == None:
            try:
                submission = pd.read_csv(self.path_data+'sample_submission.csv')
            except:
                submission = pd.read_csv(self.path_data+'gender_submission.csv')
        else:
            submission = df_from_path(self.path_data+name_sample_submission)
            # pd.read_csv()

        submission[self.target] = np.array(data).astype(submission[self.target].dtype)

        from datetime import datetime as dt
        now = str(dt.now().strftime("%Y-%m-%d %H:%M:%S").replace('-', '').replace(' ', '').replace(':', ''))
        submission.to_csv(self.path_data+'submission_'+now+'.csv', index=None)
        
        if return_submission:
            return submission
