from config import Config
import pandas as pd
import numpy as np
from footkit.utils import set_to_list

class Preprocess(Config):
    def __init__(self, Config, df, teams_df):
        # наследование параметров
        super().__init__()
        self.df=df
        self.teams_df=teams_df        
    
    def get_teams_df(self):
        '''
            переводим разрез команда1-команда2, в:
               команда1-матч
               команда2-матч
        '''
        df=self.df.copy()
        self.teams_df.sort_values('DateTime',inplace=True)
        self.teams_df.reset_index(drop = True, inplace = True)
        print('до обработки Shape==1: ',self.teams_df[(self.teams_df['TeamId'] == 113) & (self.teams_df['DateTime'] == '2019-09-16 18:45:00')].shape[0]==1)
        # справочник команды (команда-матч)
        trans = {
            'HomeTeamId' : 'TeamId',
            'HomeTeam' : 'TeamName',
            'HomeTeam_xG' : 'xG',
            'AwayTeamId' : 'TeamId',
            'AwayTeam' : 'TeamName',
            'AwayTeam_xG' : 'xG',
        }

        home_df = pd.merge(
            df[['HomeTeamId', 'HomeTeam', 'LeagueName', 'Season', 'DateTime', 'WinProba', 'DrawProba', 'LoseProba']].rename(columns=trans)
            , self.teams_df
            , on = ['TeamId', 'TeamName', 'LeagueName', 'Season', 'DateTime']
            , how = 'left'
        )
        home_df['h_a'] = 'h'

        away_df = pd.merge(
            df[['AwayTeamId', 'AwayTeam', 'LeagueName', 'Season', 'DateTime', 'WinProba', 'DrawProba', 'LoseProba']].rename(columns=trans)
            , self.teams_df
            , on = ['TeamId', 'TeamName', 'LeagueName', 'Season', 'DateTime']
            , how = 'left'
        )
        away_df['h_a'] = 'a'

        self.teams_df = pd.concat([home_df, away_df], sort=False, axis = 0).copy()
        print('после обработки Shape==1: ',self.teams_df[(self.teams_df['TeamId'] == 113) & (self.teams_df['DateTime'] == '2019-09-16 18:45:00')].shape[0]==1)
        
        return self.teams_df


    def fillnull_on_test(self,data, targets):
        data=data.copy()
        for i in targets:
            data[i] = np.where(
                data.IsResult == False
                , np.nan
                , data[i]
            )
        return data

    def get_target_on_df(
        self,
        data,
        HomeTeamGoal,
        AwayTeamGoal
    ):

        '''
            example:
            get_target_on_df(
                df,
                HomeTeamGoal = 'FTHG',
                AwayTeamGoal = 'FTAG',
            )
        '''

        data = data.copy()

        data_columns = list(data.columns)

        # Виды ставок
        data['П1'] = np.where(data[HomeTeamGoal] > data[AwayTeamGoal],1,0)
        data['Х']  = np.where(data[HomeTeamGoal] == data[AwayTeamGoal],1,0)
        data['П2'] = np.where(data[HomeTeamGoal] < data[AwayTeamGoal],1,0)
        data['Победитель'] = np.where(data[HomeTeamGoal] > data[AwayTeamGoal],1, np.where(data[HomeTeamGoal] == data[AwayTeamGoal],0,2))

        data['1Х'] = np.where(data[HomeTeamGoal] >= data[AwayTeamGoal],1,0)
        data['12'] = np.where(data[HomeTeamGoal] != data[AwayTeamGoal],1,0)
        data['2Х'] = np.where(data[HomeTeamGoal] <= data[AwayTeamGoal],1,0)

        data['ТМ1'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] <= 1 , 1 , 0)
        data['ТБ1'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] >= 1 , 1 , 0)
        data['ТМ1.5'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] < 1.5 , 1 , 0)
        data['ТБ1.5'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] > 1.5 , 1 , 0)

        data['ТМ2'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] <= 2 , 1 , 0)
        data['ТБ2'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] >= 2 , 1 , 0)
        data['ТМ2.5'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] < 2.5 , 1 , 0)
        data['ТБ2.5'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] > 2.5 , 1 , 0)

        data['ТМ3'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] <= 3 , 1 , 0)
        data['ТБ3'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] >= 3 , 1 , 0)
        data['ТМ3.5'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] < 3.5 , 1 , 0)
        data['ТБ3.5'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] > 3.5 , 1 , 0)

        data['ТМ4'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] <= 4 , 1 , 0)
        data['ТБ4'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] >= 4 , 1 , 0)
        data['ТМ4.5'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] < 4.5 , 1 , 0)
        data['ТБ4.5'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] > 4.5 , 1 , 0)

        data['ТМ5'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] <= 5 , 1 , 0)
        data['ТБ5'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] >= 5 , 1 , 0)
        data['ТМ5.5'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] < 5.5 , 1 , 0)
        data['ТБ5.5'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] > 5.5 , 1 , 0)


        data['Голов 0-1'] = np.where((data[HomeTeamGoal] + data[AwayTeamGoal] >=0) 
                                     & (data[HomeTeamGoal] + data[AwayTeamGoal] <=1)
                                    ,1,0)
        data['Голов 2-3'] = np.where((data[HomeTeamGoal] + data[AwayTeamGoal] >=2) 
                                     & (data[HomeTeamGoal] + data[AwayTeamGoal] <=3)
                                    ,1,0)
        data['Голов 4-5'] = np.where((data[HomeTeamGoal] + data[AwayTeamGoal] >=4) 
                                     & (data[HomeTeamGoal] + data[AwayTeamGoal] <=5)
                                    ,1,0)
        data['Голов 6>' ] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] >= 6,1,0)

        data['Ком1 забьет'] = np.where(data[HomeTeamGoal] > 0,1,0)
        data['Ком2 забьет'] = np.where(data[AwayTeamGoal] > 0,1,0)
        data['Обе забьют'] = np.where((data[HomeTeamGoal] > 0) & (data[AwayTeamGoal] > 0),1,0)

        data['Колво голов КОМ1(0)'] = np.where( data[HomeTeamGoal] == 0 , 1 , 0)
        data['Колво голов КОМ1(1)'] = np.where( data[HomeTeamGoal] == 1 , 1 , 0)
        data['Колво голов КОМ1(2)'] = np.where( data[HomeTeamGoal] == 2 , 1 , 0)
        data['Колво голов КОМ1(3>)'] = np.where( data[HomeTeamGoal] >= 3 , 1 , 0)
        data['Колво голов КОМ2(0)'] = np.where( data[AwayTeamGoal] == 0 , 1 , 0)
        data['Колво голов КОМ2(1)'] = np.where( data[AwayTeamGoal] == 1 , 1 , 0)
        data['Колво голов КОМ2(2)'] = np.where( data[AwayTeamGoal] == 2 , 1 , 0)
        data['Колво голов КОМ2(3>)'] = np.where( data[AwayTeamGoal] == 3 , 1 , 0)

        data['Инд.Тотал К1 ТМ0.5'] = np.where( data[HomeTeamGoal] < 0.5 , 1 , 0)
        data['Инд.Тотал К1 ТБ0.5'] = np.where( data[HomeTeamGoal] > 0.5 , 1 , 0)
        data['Инд.Тотал К2 ТМ0.5'] = np.where( data[AwayTeamGoal] < 0.5 , 1 , 0)
        data['Инд.Тотал К2 ТБ0.5'] = np.where( data[AwayTeamGoal] > 0.5 , 1 , 0)

        data['Инд.Тотал К1 ТМ1'] = np.where( data[HomeTeamGoal] <= 1 , 1 , 0)
        data['Инд.Тотал К1 ТБ1'] = np.where( data[HomeTeamGoal] >= 1 , 1 , 0)
        data['Инд.Тотал К2 ТМ1'] = np.where( data[AwayTeamGoal] <= 1 , 1 , 0)
        data['Инд.Тотал К2 ТБ1'] = np.where( data[AwayTeamGoal] >= 1 , 1 , 0)

        data['Инд.Тотал К1 ТМ1.5'] = np.where( data[HomeTeamGoal] < 1.5 , 1 , 0)
        data['Инд.Тотал К1 ТБ1.5'] = np.where( data[HomeTeamGoal] > 1.5 , 1 , 0)
        data['Инд.Тотал К2 ТМ1.5'] = np.where( data[AwayTeamGoal] < 1.5 , 1 , 0)
        data['Инд.Тотал К2 ТБ1.5'] = np.where( data[AwayTeamGoal] > 1.5 , 1 , 0)

        data['Инд.Тотал К1 ТМ2'] = np.where( data[HomeTeamGoal] <= 2 , 1 , 0)
        data['Инд.Тотал К1 ТБ2'] = np.where( data[HomeTeamGoal] >= 2 , 1 , 0)
        data['Инд.Тотал К2 ТМ2'] = np.where( data[AwayTeamGoal] <= 2 , 1 , 0)
        data['Инд.Тотал К2 ТБ2'] = np.where( data[AwayTeamGoal] >= 2 , 1 , 0)

        data['Инд.Тотал К1 ТМ2.5'] = np.where( data[HomeTeamGoal] < 2.5 , 1 , 0)
        data['Инд.Тотал К1 ТБ2.5'] = np.where( data[HomeTeamGoal] > 2.5 , 1 , 0)
        data['Инд.Тотал К2 ТМ2.5'] = np.where( data[AwayTeamGoal] < 2.5 , 1 , 0)
        data['Инд.Тотал К2 ТБ2.5'] = np.where( data[AwayTeamGoal] > 2.5 , 1 , 0)

        data['Инд.Тотал К1 ТМ3.5'] = np.where( data[HomeTeamGoal] < 3.5 , 1 , 0)
        data['Инд.Тотал К1 ТБ3.5'] = np.where( data[HomeTeamGoal] > 3.5 , 1 , 0)
        data['Инд.Тотал К2 ТМ3.5'] = np.where( data[AwayTeamGoal] < 3.5 , 1 , 0)
        data['Инд.Тотал К2 ТБ3.5'] = np.where( data[AwayTeamGoal] > 3.5 , 1 , 0)

        data['Точный счет 0:1'] = np.where( (data[HomeTeamGoal] == 0) & (data[AwayTeamGoal] == 1) , 1, 0)
        data['Точный счет 0:2'] = np.where( (data[HomeTeamGoal] == 0) & (data[AwayTeamGoal] == 2) , 1, 0)
        data['Точный счет 0:3'] = np.where( (data[HomeTeamGoal] == 0) & (data[AwayTeamGoal] == 3) , 1, 0)
        data['Точный счет 1:0'] = np.where( (data[HomeTeamGoal] == 1) & (data[AwayTeamGoal] == 0) , 1, 0)
        data['Точный счет 2:0'] = np.where( (data[HomeTeamGoal] == 2) & (data[AwayTeamGoal] == 0) , 1, 0)
        data['Точный счет 3:0'] = np.where( (data[HomeTeamGoal] == 3) & (data[AwayTeamGoal] == 0) , 1, 0)
        data['Точный счет 1:1'] = np.where( (data[HomeTeamGoal] == 1) & (data[AwayTeamGoal] == 1) , 1, 0)
        data['Точный счет 1:2'] = np.where( (data[HomeTeamGoal] == 1) & (data[AwayTeamGoal] == 2) , 1, 0)
        data['Точный счет 1:3'] = np.where( (data[HomeTeamGoal] == 1) & (data[AwayTeamGoal] == 3) , 1, 0)
        data['Точный счет 2:1'] = np.where( (data[HomeTeamGoal] == 2) & (data[AwayTeamGoal] == 1) , 1, 0)
        data['Точный счет 2:2'] = np.where( (data[HomeTeamGoal] == 2) & (data[AwayTeamGoal] == 2) , 1, 0)
        data['Точный счет 2:3'] = np.where( (data[HomeTeamGoal] == 2) & (data[AwayTeamGoal] == 3) , 1, 0)
        data['Точный счет 3:1'] = np.where( (data[HomeTeamGoal] == 3) & (data[AwayTeamGoal] == 1) , 1, 0)
        data['Точный счет 3:2'] = np.where( (data[HomeTeamGoal] == 3) & (data[AwayTeamGoal] == 2) , 1, 0)
        data['Точный счет 3:3'] = np.where( (data[HomeTeamGoal] == 3) & (data[AwayTeamGoal] == 3) , 1, 0)

        data["Победитель + обе забьют 1/да" ] = np.where(

            (data['П1'] == 1) & (data['Обе забьют'] == 1),
        1,0
        )


        data["Победитель + обе забьют 1/нет"] = np.where(

            (data['П1'] == 1) & (data['Обе забьют'] == 0),
        1,0
        )

        data["Победитель + обе забьют Х/да" ] = np.where(

            (data['Х'] == 1) & (data['Обе забьют'] == 1),
        1,0
        )

        data["Победитель + обе забьют Х/нет"] = np.where(

            (data['Х'] == 1) & (data['Обе забьют'] == 0),
        1,0
        )

        data["Победитель + обе забьют 2/да" ] = np.where(

            (data['П2'] == 1) & (data['Обе забьют'] == 1),
        1,0
        )

        data["Победитель + обе забьют 2/нет"] = np.where(

            (data['П2'] == 1) & (data['Обе забьют'] == 0),
        1,0
        )


        data['Сухая победа К1'] = np.where(
                (data[HomeTeamGoal]>data[AwayTeamGoal]) & (data[AwayTeamGoal] == 0),
        1,0
        )

        data['Сухая победа К2'] = np.where(
            (data[AwayTeamGoal]>data[HomeTeamGoal]) & (data[HomeTeamGoal] == 0),
        1,0
        )


        data['Обе забьют + ТБ2.5 Да/Бол'] =  np.where( 
            (data['Обе забьют'] ==1) & (data['ТБ2.5'] == 1) 
            , 1 
            , 0 
        )

        data['Обе забьют + ТБ2.5 Да/Мен'] =  np.where( 
            (data['Обе забьют'] ==1) & (data['ТБ2.5'] == 0) 
            , 1 
            , 0 
        )

        data['Обе забьют + ТБ2.5 Нет/Бол'] = np.where( 
            (data['Обе забьют'] ==0) & (data['ТБ2.5'] == 1) 
            , 1 
            , 0 
        )

        data['Обе забьют + ТБ2.5 Нет/Мен'] = np.where( 
            (data['Обе забьют'] ==0) & (data['ТБ2.5'] == 0) 
            , 1 
            , 0 
        )

        data['Ф1'] = np.where(
            data[HomeTeamGoal] - data[AwayTeamGoal] > 0
            , 1
            , 0
        )

        data['Ф1 0.5'] = np.where(
            ( data[HomeTeamGoal] + 0.5 ) - data[AwayTeamGoal] > 0
            , 1
            , 0
        )

        data['Ф1 -0.5'] = np.where(
            ( data[HomeTeamGoal] - 0.5 ) - data[AwayTeamGoal] > 0
            , 1
            , 0
        )

        data['Ф1 1'] = np.where(
            ( data[HomeTeamGoal] + 1 ) - data[AwayTeamGoal] > 0
            , 1
            , 0
        )

        data['Ф1 -1'] = np.where(
            ( data[HomeTeamGoal] - 1 ) - data[AwayTeamGoal] > 0
            , 1
            , 0
        )

        data['Ф1 1.5'] = np.where(
            ( data[HomeTeamGoal] + 1.5 ) - data[AwayTeamGoal] > 0
            , 1
            , 0
        )

        data['Ф1 -1.5'] = np.where(
            ( data[HomeTeamGoal] - 1.5 ) - data[AwayTeamGoal] > 0
            , 1
            , 0
        )

        data['Ф1 2'] = np.where(
            ( data[HomeTeamGoal] + 2 ) - data[AwayTeamGoal] > 0
            , 1
            , 0
        )

        data['Ф1 -2'] = np.where(
            ( data[HomeTeamGoal] - 2 ) - data[AwayTeamGoal] > 0
            , 1
            , 0
        )

        data['Ф1 2.5'] = np.where(
            ( data[HomeTeamGoal] + 2.5 ) - data[AwayTeamGoal] > 0
            , 1
            , 0
        )

        data['Ф1 -2.5'] = np.where(
            ( data[HomeTeamGoal] - 2.5 ) - data[AwayTeamGoal] > 0
            , 1
            , 0
        )

        data['Ф1 -3'] = np.where(
            ( data[HomeTeamGoal] - 3 ) - data[AwayTeamGoal] > 0
            , 1
            , 0
        )

        data['Ф1 -3.5'] = np.where(
            ( data[HomeTeamGoal] - 3.5 ) - data[AwayTeamGoal] > 0
            , 1
            , 0
        )

        data['Ф2'] = np.where(
            data[AwayTeamGoal] - data[HomeTeamGoal] > 0
            , 1
            , 0
        )

        data['Ф2 0.5'] = np.where(
            ( data[AwayTeamGoal] + 0.5 ) - data[HomeTeamGoal] > 0
            , 1
            , 0
        )

        data['Ф2 -0.5'] = np.where(
            ( data[AwayTeamGoal] - 0.5 ) - data[HomeTeamGoal] > 0
            , 1
            , 0
        )

        data['Ф2 1'] = np.where(
            ( data[AwayTeamGoal] + 1 ) - data[HomeTeamGoal] > 0
            , 1
            , 0
        )

        data['Ф2 -1'] = np.where(
            ( data[AwayTeamGoal] - 1 ) - data[HomeTeamGoal] > 0
            , 1
            , 0
        )

        data['Ф2 1.5'] = np.where(
            ( data[AwayTeamGoal] + 1.5 ) - data[HomeTeamGoal] > 0
            , 1
            , 0
        )

        data['Ф2 -1.5'] = np.where(
            ( data[AwayTeamGoal] - 1.5 ) - data[HomeTeamGoal] > 0
            , 1
            , 0
        )

        data['Ф2 2'] = np.where(
            ( data[AwayTeamGoal] + 2 ) - data[HomeTeamGoal] > 0
            , 1
            , 0
        )

        data['Ф2 -2'] = np.where(
            ( data[AwayTeamGoal] - 2 ) - data[HomeTeamGoal] > 0
            , 1
            , 0
        )

        data['Ф2 2.5'] = np.where(
            ( data[AwayTeamGoal] + 2.5 ) - data[HomeTeamGoal] > 0
            , 1
            , 0
        )

        data['Ф2 -2.5'] = np.where(
            ( data[AwayTeamGoal] - 2.5 ) - data[HomeTeamGoal] > 0
            , 1
            , 0
        )

        data['Ф2 3'] = np.where(
            ( data[AwayTeamGoal] + 3 ) - data[HomeTeamGoal] > 0
            , 1
            , 0
        )

        data['Ф2 3.5'] = np.where(
            ( data[AwayTeamGoal] + 3.5 ) - data[HomeTeamGoal] > 0
            , 1
            , 0
        )

        targets = list(
            set(data.columns) - set(data_columns)
        )

        data = self.fillnull_on_test(data, targets)
        
        self.df_targets=data[targets].copy()
        self.targets=targets.copy()
        return data[targets], targets



    def get_target_on_df_eng(
        self,
        HomeTeamGoal,
        AwayTeamGoal
    ):

        '''
            example:
            get_target_on_df(
                df,
                HomeTeamGoal = 'FTHG',
                AwayTeamGoal = 'FTAG',
            )
        '''

        data = self.df.copy()

        data_columns = list(data.columns)

        # Виды ставок
        data['W'] = np.where(data[HomeTeamGoal] > data[AwayTeamGoal],1,0)
        data['D']  = np.where(data[HomeTeamGoal] == data[AwayTeamGoal],1,0)
        data['L'] = np.where(data[HomeTeamGoal] < data[AwayTeamGoal],1,0)
        data['Winner'] = np.where(data[HomeTeamGoal] > data[AwayTeamGoal],1, np.where(data[HomeTeamGoal] == data[AwayTeamGoal],0,2))

        data['1X'] = np.where(data[HomeTeamGoal] >= data[AwayTeamGoal],1,0)
        data['12'] = np.where(data[HomeTeamGoal] != data[AwayTeamGoal],1,0)
        data['2X'] = np.where(data[HomeTeamGoal] <= data[AwayTeamGoal],1,0)

        data['Total Less 1'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] <= 1 , 1 , 0)
        data['Total More 1'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] >= 1 , 1 , 0)
        data['Total Less 1.5'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] < 1.5 , 1 , 0)
        data['Total More 1.5'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] > 1.5 , 1 , 0)

        data['Total Less 2'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] <= 2 , 1 , 0)
        data['Total More 2'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] >= 2 , 1 , 0)
        data['Total Less 2.5'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] < 2.5 , 1 , 0)
        data['Total More 2.5'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] > 2.5 , 1 , 0)

        data['Total Less 3'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] <= 3 , 1 , 0)
        data['Total More 3'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] >= 3 , 1 , 0)
        data['Total Less 3.5'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] < 3.5 , 1 , 0)
        data['Total More 3.5'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] > 3.5 , 1 , 0)

        data['Total Less 4'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] <= 4 , 1 , 0)
        data['Total More 4'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] >= 4 , 1 , 0)
        data['Total Less 4.5'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] < 4.5 , 1 , 0)
        data['Total More 4.5'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] > 4.5 , 1 , 0)

        data['Total Less 5'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] <= 5 , 1 , 0)
        data['Total More 5'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] >= 5 , 1 , 0)
        data['Total Less 5.5'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] < 5.5 , 1 , 0)
        data['Total More 5.5'] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] > 5.5 , 1 , 0)

        data['Goals 0-1'] = np.where((data[HomeTeamGoal] + data[AwayTeamGoal] >=0) 
                                     & (data[HomeTeamGoal] + data[AwayTeamGoal] <=1)
                                    ,1,0)
        data['Goals 2-3'] = np.where((data[HomeTeamGoal] + data[AwayTeamGoal] >=2) 
                                     & (data[HomeTeamGoal] + data[AwayTeamGoal] <=3)
                                    ,1,0)
        data['Goals 4-5'] = np.where((data[HomeTeamGoal] + data[AwayTeamGoal] >=4) 
                                     & (data[HomeTeamGoal] + data[AwayTeamGoal] <=5)
                                    ,1,0)
        data['Goals 6>' ] = np.where(data[HomeTeamGoal] + data[AwayTeamGoal] >= 6,1,0)

        data['HT Score'] = np.where(data[HomeTeamGoal] > 0,1,0)
        data['AT Score'] = np.where(data[AwayTeamGoal] > 0,1,0)
        data['Both teams to score'] = np.where((data[HomeTeamGoal] > 0) & (data[AwayTeamGoal] > 0),1,0)

        data['Ind. Total HT Total Less 0.5'] = np.where( data[HomeTeamGoal] < 0.5 , 1 , 0)
        data['Ind. Total HT Total More 0.5'] = np.where( data[HomeTeamGoal] > 0.5 , 1 , 0)
        data['Ind. Total AT Total Less 0.5'] = np.where( data[AwayTeamGoal] < 0.5 , 1 , 0)
        data['Ind. Total AT Total More 0.5'] = np.where( data[AwayTeamGoal] > 0.5 , 1 , 0)

        data['Ind. Total HT Total Less 1'] = np.where( data[HomeTeamGoal] <= 1 , 1 , 0)
        data['Ind. Total HT Total More 1'] = np.where( data[HomeTeamGoal] >= 1 , 1 , 0)
        data['Ind. Total AT Total Less 1'] = np.where( data[AwayTeamGoal] <= 1 , 1 , 0)
        data['Ind. Total AT Total More 1'] = np.where( data[AwayTeamGoal] >= 1 , 1 , 0)

        data['Ind. Total HT Total Less 1.5'] = np.where( data[HomeTeamGoal] < 1.5 , 1 , 0)
        data['Ind. Total HT Total More 1.5'] = np.where( data[HomeTeamGoal] > 1.5 , 1 , 0)
        data['Ind. Total AT Total Less 1.5'] = np.where( data[AwayTeamGoal] < 1.5 , 1 , 0)
        data['Ind. Total AT Total More 1.5'] = np.where( data[AwayTeamGoal] > 1.5 , 1 , 0)

        data['Ind. Total HT Total Less 2'] = np.where( data[HomeTeamGoal] <= 2 , 1 , 0)
        data['Ind. Total HT Total More 2'] = np.where( data[HomeTeamGoal] >= 2 , 1 , 0)
        data['Ind. Total AT Total Less 2'] = np.where( data[AwayTeamGoal] <= 2 , 1 , 0)
        data['Ind. Total AT Total More 2'] = np.where( data[AwayTeamGoal] >= 2 , 1 , 0)

        data['Ind. Total HT Total Less 2.5'] = np.where( data[HomeTeamGoal] < 2.5 , 1 , 0)
        data['Ind. Total HT Total More 2.5'] = np.where( data[HomeTeamGoal] > 2.5 , 1 , 0)
        data['Ind. Total AT Total Less 2.5'] = np.where( data[AwayTeamGoal] < 2.5 , 1 , 0)
        data['Ind. Total AT Total More 2.5'] = np.where( data[AwayTeamGoal] > 2.5 , 1 , 0)

        data['Ind. Total HT Total Less 3.5'] = np.where( data[HomeTeamGoal] < 3.5 , 1 , 0)
        data['Ind. Total HT Total More 3.5'] = np.where( data[HomeTeamGoal] > 3.5 , 1 , 0)
        data['Ind. Total AT Total Less 3.5'] = np.where( data[AwayTeamGoal] < 3.5 , 1 , 0)
        data['Ind. Total AT Total More 3.5'] = np.where( data[AwayTeamGoal] > 3.5 , 1 , 0)

        data['Score 0:1'] = np.where( (data[HomeTeamGoal] == 0) & (data[AwayTeamGoal] == 1) , 1, 0)
        data['Score 0:2'] = np.where( (data[HomeTeamGoal] == 0) & (data[AwayTeamGoal] == 2) , 1, 0)
        data['Score 0:3'] = np.where( (data[HomeTeamGoal] == 0) & (data[AwayTeamGoal] == 3) , 1, 0)
        data['Score 1:0'] = np.where( (data[HomeTeamGoal] == 1) & (data[AwayTeamGoal] == 0) , 1, 0)
        data['Score 2:0'] = np.where( (data[HomeTeamGoal] == 2) & (data[AwayTeamGoal] == 0) , 1, 0)
        data['Score 3:0'] = np.where( (data[HomeTeamGoal] == 3) & (data[AwayTeamGoal] == 0) , 1, 0)
        data['Score 1:1'] = np.where( (data[HomeTeamGoal] == 1) & (data[AwayTeamGoal] == 1) , 1, 0)
        data['Score 1:2'] = np.where( (data[HomeTeamGoal] == 1) & (data[AwayTeamGoal] == 2) , 1, 0)
        data['Score 1:3'] = np.where( (data[HomeTeamGoal] == 1) & (data[AwayTeamGoal] == 3) , 1, 0)
        data['Score 2:1'] = np.where( (data[HomeTeamGoal] == 2) & (data[AwayTeamGoal] == 1) , 1, 0)
        data['Score 2:2'] = np.where( (data[HomeTeamGoal] == 2) & (data[AwayTeamGoal] == 2) , 1, 0)
        data['Score 2:3'] = np.where( (data[HomeTeamGoal] == 2) & (data[AwayTeamGoal] == 3) , 1, 0)
        data['Score 3:1'] = np.where( (data[HomeTeamGoal] == 3) & (data[AwayTeamGoal] == 1) , 1, 0)
        data['Score 3:2'] = np.where( (data[HomeTeamGoal] == 3) & (data[AwayTeamGoal] == 2) , 1, 0)
        data['Score 3:3'] = np.where( (data[HomeTeamGoal] == 3) & (data[AwayTeamGoal] == 3) , 1, 0)

        data['Handicap HT'] = np.where(
            data[HomeTeamGoal] - data[AwayTeamGoal] > 0
            , 1
            , 0
        )

        data['Handicap HT 0.5'] = np.where(
            ( data[HomeTeamGoal] + 0.5 ) - data[AwayTeamGoal] > 0
            , 1
            , 0
        )

        data['Handicap HT -0.5'] = np.where(
            ( data[HomeTeamGoal] - 0.5 ) - data[AwayTeamGoal] > 0
            , 1
            , 0
        )

        data['Handicap HT 1'] = np.where(
            ( data[HomeTeamGoal] + 1 ) - data[AwayTeamGoal] > 0
            , 1
            , 0
        )

        data['Handicap HT -1'] = np.where(
            ( data[HomeTeamGoal] - 1 ) - data[AwayTeamGoal] > 0
            , 1
            , 0
        )

        data['Handicap HT 1.5'] = np.where(
            ( data[HomeTeamGoal] + 1.5 ) - data[AwayTeamGoal] > 0
            , 1
            , 0
        )

        data['Handicap HT -1.5'] = np.where(
            ( data[HomeTeamGoal] - 1.5 ) - data[AwayTeamGoal] > 0
            , 1
            , 0
        )

        data['Handicap HT 2'] = np.where(
            ( data[HomeTeamGoal] + 2 ) - data[AwayTeamGoal] > 0
            , 1
            , 0
        )

        data['Handicap HT -2'] = np.where(
            ( data[HomeTeamGoal] - 2 ) - data[AwayTeamGoal] > 0
            , 1
            , 0
        )

        data['Handicap HT 2.5'] = np.where(
            ( data[HomeTeamGoal] + 2.5 ) - data[AwayTeamGoal] > 0
            , 1
            , 0
        )

        data['Handicap HT -2.5'] = np.where(
            ( data[HomeTeamGoal] - 2.5 ) - data[AwayTeamGoal] > 0
            , 1
            , 0
        )

        data['Handicap HT -3'] = np.where(
            ( data[HomeTeamGoal] - 3 ) - data[AwayTeamGoal] > 0
            , 1
            , 0
        )

        data['Handicap HT -3.5'] = np.where(
            ( data[HomeTeamGoal] - 3.5 ) - data[AwayTeamGoal] > 0
            , 1
            , 0
        )

        data['Handicap AT'] = np.where(
            data[AwayTeamGoal] - data[HomeTeamGoal] > 0
            , 1
            , 0
        )

        data['Handicap AT 0.5'] = np.where(
            ( data[AwayTeamGoal] + 0.5 ) - data[HomeTeamGoal] > 0
            , 1
            , 0
        )

        data['Handicap AT -0.5'] = np.where(
            ( data[AwayTeamGoal] - 0.5 ) - data[HomeTeamGoal] > 0
            , 1
            , 0
        )

        data['Handicap AT 1'] = np.where(
            ( data[AwayTeamGoal] + 1 ) - data[HomeTeamGoal] > 0
            , 1
            , 0
        )

        data['Handicap AT -1'] = np.where(
            ( data[AwayTeamGoal] - 1 ) - data[HomeTeamGoal] > 0
            , 1
            , 0
        )

        data['Handicap AT 1.5'] = np.where(
            ( data[AwayTeamGoal] + 1.5 ) - data[HomeTeamGoal] > 0
            , 1
            , 0
        )

        data['Handicap AT -1.5'] = np.where(
            ( data[AwayTeamGoal] - 1.5 ) - data[HomeTeamGoal] > 0
            , 1
            , 0
        )

        data['Handicap AT 2'] = np.where(
            ( data[AwayTeamGoal] + 2 ) - data[HomeTeamGoal] > 0
            , 1
            , 0
        )

        data['Handicap AT -2'] = np.where(
            ( data[AwayTeamGoal] - 2 ) - data[HomeTeamGoal] > 0
            , 1
            , 0
        )

        data['Handicap AT 2.5'] = np.where(
            ( data[AwayTeamGoal] + 2.5 ) - data[HomeTeamGoal] > 0
            , 1
            , 0
        )

        data['Handicap AT -2.5'] = np.where(
            ( data[AwayTeamGoal] - 2.5 ) - data[HomeTeamGoal] > 0
            , 1
            , 0
        )

        data['Handicap AT 3'] = np.where(
            ( data[AwayTeamGoal] + 3 ) - data[HomeTeamGoal] > 0
            , 1
            , 0
        )

        data['Handicap AT 3.5'] = np.where(
            ( data[AwayTeamGoal] + 3.5 ) - data[HomeTeamGoal] > 0
            , 1
            , 0
        )

        targets = list(
            set(data.columns) - set(data_columns)
        )

        data = self.fillnull_on_test(data, targets)

        self.df_targets=data[targets].copy()
        self.targets = targets

        return data[targets], targets

    def create_features(
        self
    ):
        
        df=self.df.copy()
        StatTeam_df = self.teams_df.copy()
        StatTeam_df.sort_values([self.SEASON,self.TEAMNAME,self.LEAGUENAME,self.DATETIME],inplace=True)
        features_base = list(StatTeam_df.columns)
        StatTeam_df[self.ROUNDEVENT] = StatTeam_df.groupby([self.SEASON,self.LEAGUENAME,self.TEAMNAME]).cumcount()+1
        StatTeam_df['ChillDays'] = (StatTeam_df['DateTime'] - StatTeam_df.groupby(['Season','TeamName'])['DateTime'].shift(1)).dt.days

        # результаты предыдущего матча
        for target in self.FEATURES:
            for sh in [1, 2, 3]:
                StatTeam_df['Shift'+str(sh)+target] = StatTeam_df.groupby([self.SEASON,self.TEAMNAME,self.LEAGUENAME])[target].shift(sh).fillna(0)
                StatTeam_df['Cum'+str(sh)+target] = StatTeam_df.groupby([self.SEASON,self.TEAMNAME,self.LEAGUENAME])['Shift'+str(sh)+target].cumsum()
            StatTeam_df['DeltaShift'+target] = ((StatTeam_df['Shift'+str(1)+target] - StatTeam_df['Shift'+str(2)+target]) + (StatTeam_df['Shift'+str(2)+target]-StatTeam_df['Shift'+str(3)+target]))/2
            StatTeam_df['DeltaCum'+target] = ((StatTeam_df['Cum'+str(1)+target] - StatTeam_df['Cum'+str(2)+target]) + (StatTeam_df['Cum'+str(2)+target]-StatTeam_df['Cum'+str(3)+target]))/2
            # Накопительные результаты ( исключая текущий )
            StatTeam_df['Cum'+target] = StatTeam_df.groupby([self.SEASON,self.TEAMNAME,self.LEAGUENAME])['Shift'+str(1)+target].cumsum()

        # место в турнирной таблице перед матчем
        StatTeam_df['Place'] = StatTeam_df.groupby([self.SEASON,self.LEAGUENAME,self.ROUNDEVENT])['Cum1'+self.POINTS].rank(ascending=False,method='average')

        # Накопительные результаты ( исключая текущий зависимо, от места игры )
        # чтобы не создавать лишние поля, тут же шифтуем, чтобы забрать все до "сегодняшнего" матча, сразу же считаем нак. сумму
        for target in self.FEATURES:    
            StatTeam_df['Cum'+target+'InPlace'] = StatTeam_df.groupby([self.SEASON,self.TEAMNAME,self.LEAGUENAME,self.PLACEMATCH])[target].shift(1).fillna(0)
            StatTeam_df['Cum'+target+'InPlace'] = StatTeam_df.groupby([self.SEASON,self.TEAMNAME,self.LEAGUENAME,self.PLACEMATCH])['Cum'+target+'InPlace'].cumsum()
            StatTeam_df['Perc'+target+'InPlace'] = StatTeam_df['Cum'+target+'InPlace'] / StatTeam_df['Cum'+target]
            StatTeam_df['Avg'+target+'Season'] = StatTeam_df['Cum'+target] / StatTeam_df[self.ROUNDEVENT]

        # шифт+дифф
        StatTeam_df['ShiftPlace'+self.GOALS+'Diff'] = StatTeam_df.groupby(
            [
                self.SEASON,self.TEAMNAME,self.LEAGUENAME,self.PLACEMATCH
            ]
        )[self.GOALS].shift(1).fillna(0) - \
        StatTeam_df.groupby(
            [
                self.SEASON,self.TEAMNAME,self.LEAGUENAME,self.PLACEMATCH
            ]
        )[self.MISSED].shift(1).fillna(0)

        StatTeam_df['Cum'+self.GOALS+'DiffInPlace'] = StatTeam_df.groupby([self.SEASON,self.TEAMNAME,self.LEAGUENAME,self.PLACEMATCH])['ShiftPlace'+self.GOALS+'Diff'].shift(1).fillna(0)
        StatTeam_df['Cum'+self.GOALS+'DiffInPlace'] = StatTeam_df.groupby([self.SEASON,self.TEAMNAME,self.LEAGUENAME,self.PLACEMATCH])['Cum'+self.GOALS+'DiffInPlace'].cumsum()
        StatTeam_df.drop('ShiftPlace'+self.GOALS+'Diff',axis=1,inplace=True)

        # место в турнирной таблице перед матчем
        # StatTeam_df['PlaceInPlace'] = StatTeam_df.groupby([self.SEASON,self.LEAGUENAME,self.ROUNDEVENT,self.PLACEMATCH])['Cum'+self.POINTS+'InPlace'].rank(ascending=False,method='average')

        # 3 days
        for target in self.FEATURES:
            # Дивизион - команда
            agg_group = [self.LEAGUENAME,self.TEAMNAME] #,self.SEASON
            StatTeam_df['Avg'+target+'3'] = np.mean([StatTeam_df.groupby(agg_group)[target].shift(i) for i in [1,2,3]], axis=0)
            StatTeam_df['Avg'+target+'5'] = np.mean([StatTeam_df.groupby(agg_group)[target].shift(i) for i in [1,2,3,4,5]], axis=0)

            # Среднее колво забитых/пропущенных голов команды (за 5 и 3 матча)
            # Матчи учитываются дома и в гостях раздельно(на каждый по 5/3 матча)    
            agg_group = [self.LEAGUENAME,self.TEAMNAME,self.PLACEMATCH]
            StatTeam_df['Avg'+target+'3_PLACE'] = np.mean([StatTeam_df.groupby(agg_group)[target].shift(i) for i in [1,2,3]], axis=0)
            StatTeam_df['Avg'+target+'5_PLACE'] = np.mean([StatTeam_df.groupby(agg_group)[target].shift(i) for i in [1,2,3,4,5]], axis=0)

        # {i : 'Home'+i for i in }
        rename_cols_home = {}
        rename_cols_away = {}
        key_cols = [self.DATETIME,self.LEAGUENAME,'Team',self.SEASON]

        rename_base_cols = list( 
            set(StatTeam_df.columns) - set([self.SEASON, self.LEAGUENAME, self.DATETIME,self.ROUNDEVENT]) - set(self.FEATURES) - set(['h_a', 'result'])
        )

        rename_targets = self.FEATURES

        for i in rename_targets:
            rename_cols_home[i] = 'Home'+i
            rename_cols_away[i] = 'Away'+i

        for i in rename_base_cols:
            rename_cols_home[i] = 'Home'+i
            rename_cols_away[i] = 'Away'+i

        df = pd.merge(
                pd.merge(
                    df
                    , StatTeam_df.rename(columns=rename_cols_home)
                    , on=['DateTime','LeagueName','HomeTeamId','Season']
                    , suffixes = (False, False)
                )
                , StatTeam_df.rename(columns=rename_cols_away).drop([self.ROUNDEVENT, 'h_a', 'result'], axis = 1)
                ,on = ['DateTime','LeagueName','AwayTeamId','Season']
        )

        df['Month'] = df['DateTime'].dt.month
        df['Year'] = df['DateTime'].dt.year
        df['Week'] = df['DateTime'].dt.week
        df['Day'] = df['DateTime'].dt.day
        df['Dayofweek'] = df['DateTime'].dt.dayofweek

        leaky_features = rename_targets
        leaky_features = ['Home'+i for i in leaky_features] + ['Away'+i for i in leaky_features] + set_to_list(leaky_features, ['ChillDays'])
        leaky_features = leaky_features+[        
            'IdMatch',
            # 'DateTime',
            # 'LeagueName',
            # 'Season',
            'HomeTeam',
            'HomeShortTeam',
            # 'HomeTeamId',
            'HomeTeam_xG',
            'AwayTeam',
            'AwayShortTeam',
            # 'AwayTeamId',
            'AwayTeam_xG',
            'FTHG',
            'FTAG',
            'IsResult',
            'h_a',
            'result',
            'HomeTeamName',
            'AwayTeamName',
            'HomePlace',
            'AwayPlace',
            'HomeWinProba', 'HomeDrawProba', 'HomeLoseProba',
            'AwayWinProba', 'AwayDrawProba', 'AwayLoseProba',
        ]

        self.engineering_features = list(
            set(df.columns) - set(features_base) - set(leaky_features)
        )

        df[self.engineering_features][:1].to_pickle('./data/example_features.pkl')

        for i in ['Awayscored', 'Awaymissed', 'Awaydraws', 'Homescored', 'Homemissed', 'Awayloses', 'Homeloses', 'Awaywins', 'AwayxGA', 'Homedraws', 'AwayxG', 'HomenpxG']:
            if i in self.engineering_features:
                print('loh, drop this feature', i)

        df.set_index(['IdMatch'], inplace=True)
        
        self.df=df.copy()
        
        return self.df, self.engineering_features
