import asyncio
import json
import pandas as pd
import numpy as np
import aiohttp
from understat import Understat
from config import Config

class Parser(Config):
    def __init__(self, Config):
        # наследование параметров
        super().__init__()

    def json_to_series(self, i):
        row = []
        # берем каждый столбец
        for j in range(len(i)):
            # если это словарь, размножаем на кол-во элементов
            if type(i[j]) == dict:
                # keys = i[j].keys()
                [row.append(i[j][k]) for k in i[j].keys()]
    #             print(cast)
            else:
                row.append(i[j])
    #             print(i[j])
        return row

    # Modules Understat
    # get_league_results(self, league_name, season, options=None, **kwargs)
    async def get_league_results(self):
        async with aiohttp.ClientSession() as session:
            understat = Understat(session)

            main_df = []
            for league_name in self.leagues:
                for season in self.seasons:

                    league_stats = await understat.get_league_results(
                        league_name=league_name,
                        season=season
                    )

                    for r in league_stats:
                        main_df.append(
                            [
                                r['id'],
                                r['datetime'],
                                league_name,
                                season,
                                r['h']['title'],
                                r['h']['short_title'],
                                r['h']['id'],
                                r['xG']['h'],
                                r['a']['title'],
                                r['a']['short_title'],
                                r['a']['id'],
                                r['xG']['a'],
                                r['forecast']['w'],
                                r['forecast']['d'],
                                r['forecast']['l'],
                                r['goals']['h'],
                                r['goals']['a'],
                            ]
                        )

            main_df = pd.DataFrame(
                main_df,
                columns=[
                    'IdMatch',
                    'DateTime',
                    'LeagueName',
                    'Season',
                    'HomeTeam',
                    'HomeShortTeam',
                    'HomeTeamId',
                    'HomeTeam_xG',
                    'AwayTeam',
                    'AwayShortTeam',
                    'AwayTeamId',
                    'AwayTeam_xG',
                    'WinProba',
                    'DrawProba',
                    'LoseProba',
                    'FTHG',
                    'FTAG'
                ]
            )
            # print(league_stats)
        return main_df



    # get_league_fixtures(self, league_name, season,  options=None, **kwargs)
    async def fixtures(self):
        async with aiohttp.ClientSession() as session:
            understat = Understat(session)

            fixtures_df = []

            for league_name in self.leagues:
                for season in self.fixtures_seasons:

                    league_stats = await understat.get_league_fixtures(
                        league_name=league_name,
                        season=season
                    )

                    for r in league_stats:
                        fixtures_df.append(
                            [
                                r['id'],
                                r['datetime'],
                                league_name,
                                season,
                                r['h']['title'],
                                r['h']['short_title'],
                                r['h']['id'],
                                r['xG']['h'],
                                r['a']['title'],
                                r['a']['short_title'],
                                r['a']['id'],
                                r['xG']['a'],
                                None,
                                None,
                                None,
                                None,
                                None,
                            ]
                        )

            fixtures_df = pd.DataFrame(
                fixtures_df,
                columns=[
                    'IdMatch',
                    'DateTime',
                    'LeagueName',
                    'Season',
                    'HomeTeam',
                    'HomeShortTeam',
                    'HomeTeamId',
                    'HomeTeam_xG',
                    'AwayTeam',
                    'AwayShortTeam',
                    'AwayTeamId',
                    'AwayTeam_xG',
                    'WinProba',
                    'DrawProba',
                    'LoseProba',
                    'FTHG',
                    'FTAG'
                ]
            )

        return fixtures_df # ,league_stats team_stats,league_stats,all_stats

    # get_teams(self, league_name, season, options=None, **kwargs)
    async def teams(self):
        async with aiohttp.ClientSession() as session:
            understat = Understat(session)
            # leagues = ["epl", "la_liga", "bundesliga", "serie_a", "ligue_1", "rfpl"]
            # seasons = [2019]

            teams = []

            for league_name in self.leagues:
                for season in self.seasons:

                    league_stats = await understat.get_teams(
                        league_name=league_name,
                        season=season
                    )

                    for r in league_stats:
                        for r_history in r['history']:
                            teams.append(
                                [
                                    r['id'],
                                    r['title'],
                                    league_name,
                                    season,
                                    r_history['date'],
                                    r_history['h_a'],
                                    r_history['xG'],
                                    r_history['xGA'],
                                    r_history['npxG'],
                                    r_history['npxGA'],
                                    r_history['ppda']['att'],
                                    r_history['ppda']['def'],
                                    r_history['ppda_allowed']['att'],
                                    r_history['ppda_allowed']['def'],
                                    r_history['deep'],
                                    r_history['deep_allowed'],
                                    r_history['scored'],
                                    r_history['missed'],
                                    r_history['xpts'],
                                    r_history['result'],
                                    r_history['wins'],
                                    r_history['draws'],
                                    r_history['loses'],
                                    r_history['pts'],
                                    r_history['npxGD'],
                                ]
                            )

        teams = pd.DataFrame(
            teams
            , columns = [
                'TeamId',
                'TeamName',
                'LeagueName',
                'Season',
                'DateTime',
                'h_a',
                'xG',
                'xGA',
                'npxG',
                'npxGA',
                'ppda_att',
                'ppda_def',
                'ppda_allowed_att',
                'ppda_allowed_def',
                'deep',
                'deep_allowed',
                'scored',
                'missed',
                'xpts',
                'result',
                'wins',
                'draws',
                'loses',
                'pts',
                'npxGD',
            ]
        )

        # format data values
        for c in ['TeamId', 'Season']:
            teams[c] = teams[c].astype(np.int32)

        for c in ['xG', 'xGA','npxG', 'npxGA', 'ppda_att', 'ppda_def', 'ppda_allowed_att',
                 'ppda_allowed_def', 'deep', 'deep_allowed', 'scored', 'missed', 'xpts',
                   'wins', 'draws', 'loses', 'pts', 'npxGD'
                 ]: # 'result',
            teams[c] = teams[c].astype(np.float32)

        teams['DateTime'] = pd.to_datetime(teams['DateTime'])# .astype('datetime[64]')

        return teams


    # get_stats(self, options=None, **kwargs)
    async def stats(self):
        async with aiohttp.ClientSession() as session:
            understat = Understat(session)
            # leagues = ["epl", "la_liga", "bundesliga", "serie_a", "ligue_1", "rfpl"]


    #         example
    #         {'league_id': '6',
    #           'league': 'RFPL',
    #           'h': '1.4583',
    #           'a': '1.1250',
    #           'hxg': '1.450178946678837',
    #           'axg': '1.016120968464141',
    #           'year': '2014',
    #           'month': '8',
    #           'matches': '48'}

            list_data = []
            get_data = await understat.get_stats()

            for r in get_data:
                list_data.append(
                    [
                        r['league_id'],
                        r['league'],
                        r['h'],
                        r['a'],
                        r['hxg'],
                        r['axg'],
                        r['year'],
                        r['month'],
                        r['matches'],
                    ]
                )

            list_data = pd.DataFrame(
                list_data
                , columns = [
                    'LeagueId',
                    'League',
                    'Home',
                    'Away',
                    'Home_xG',
                    'Away_xG',
                    'Year',
                    'Month',
                    'Matches',
                ]
            )


            # format data values
            for c in ['LeagueId','Year','Month','Matches',]:
                list_data[c] = list_data[c].astype(np.int32)

            for c in ['Home','Away','Home_xG','Away_xG',]:
                list_data[c] = list_data[c].astype(np.float32)

            # list_data['DateTime'] = pd.to_datetime(list_data['DateTime'])

        return list_data

    # get_team_players(self, team_name, season, options=None, **kwargs)
    async def player_stats(self):
        async with aiohttp.ClientSession() as session:
            understat = Understat(session)

    #         example
    # {'id': '594', 'player_name': 'Romelu Lukaku', 'games': '37', 'time': '3271',
    # 'goals': '25', 'xG': '16.665452419780195', 'assists': '6', 'xA': '5.440816408023238',
    # 'shots': '110', 'key_passes': '47', 'yellow_cards': '3', 'red_cards': '0', 'position': 'F S', 
    # 'team_title': 'Everton', 'npg': '24', 'npxG': '15.904283582232893', 'xGChain': '21.251998490653932',
    # 'xGBuildup': '3.9702013842761517'}


            list_data = []

            for i in teams_df[['TeamName','Season']].drop_duplicates().values:
                get_data = await understat.get_team_players(team_name = i[0], season = i[1])

                # columns keys from json
                keys = list(get_data[0].keys())

                for r in get_data:
                    list_data.append(
                        [i[1]] + [
                            r[k] for k in keys
                        ]
                    )

            # create dataframe
            list_data = pd.DataFrame(
                list_data
                , columns = ['Season'] + keys
            )


            # format data values
            for c in [
                'id', 'games', 'time', 'goals', 'assists', 'shots', 'key_passes', 'yellow_cards', 'red_cards', 
                'npg', 'Season'
            ]:
                list_data[c] = list_data[c].astype(np.int32)

            for c in ['xG', 'xA', 'npxG', 'xGChain', 'xGBuildup']:
                list_data[c] = list_data[c].astype(np.float32)

            # list_data['DateTime'] = pd.to_datetime(list_data['DateTime'])

            print(get_data[0])
        return list_data


    # understat.get_league_players(league_name=l, season=s)
    async def league_players(self):
        async with aiohttp.ClientSession() as session:
            understat = Understat(session)

            list_data = []

            for l in self.leagues:
                for s in self.seasons:

                    get_data = await understat.get_league_players(league_name=l, season=s)

                    if len(get_data) > 0:
                        # columns keys from json
                        keys = list(get_data[0].keys())
                        for r in get_data:
                            list_data.append(
                                [l] + [s] + [
                                    r[k] for k in keys
                                ]
                            )

            # create dataframe
            list_data = pd.DataFrame(
                list_data
                , columns = ['League'] + ['Season'] + keys
            )


            # format data values
            for c in ['id', 'games', 'time', 'goals', 'assists', 'shots', 'key_passes', 'yellow_cards', 'red_cards', 'npg']:
                list_data[c] = list_data[c].astype(np.int32)

            for c in ['xG', 'xA', 'npxG', 'xGChain', 'xGBuildup']:
                list_data[c] = list_data[c].astype(np.float32)


        return list_data


    # understat.get_player_matches(league_name=l, season=s)
    async def player_matches(self):
        async with aiohttp.ClientSession() as session:
            understat = Understat(session)

            list_data = []

    #         for l in leagues:
    #             for s in seasons:
            for i in df_league_players['id'].unique():
                # print(i, df_league_players[df_league_players['id']==i]['player_name'].values[0])
                player_id = i
                player_name = df_league_players[df_league_players['id']==i]['player_name'].values[0]

                get_data = await understat.get_player_matches(player_id = i)

                if len(get_data) > 0:
                    # columns keys from json
                    keys = list(get_data[0].keys())
                    for r in get_data:
                        list_data.append(
                            [i] + [player_name] + [
                                r[k] for k in keys
                            ]
                        )

            # create dataframe
            list_data = pd.DataFrame(
                list_data
                , columns = ['player_id'] + ['player_name'] + keys
            )

            list_data['date'] = pd.to_datetime(list_data['date'])


            # format data values
            for c in ['player_id', 'goals', 'shots', 'time', 'h_goals', 'a_goals', 'id', 'season','roster_id', 'assists', 'key_passes', 'npg']:
                list_data[c] = list_data[c].astype(np.int32)

            for c in ['xA', 'xG','npxG', 'xGChain','xGBuildup']:
                list_data[c] = list_data[c].astype(np.float32)

            list_data.rename(columns={'id':'IdMatch', 'date':'DateTime'}, inplace=True)

        return list_data


    # get_team_results(self, team_name, season, options=None, **kwargs)
    async def team_results(self):
        async with aiohttp.ClientSession() as session:
            understat = Understat(session)

            list_data = []

            for i in teams_df[['TeamName','Season']].drop_duplicates().values:
                get_data = await understat.get_team_results(team_name = i[0], season = i[1])

                if len(get_data) > 0:
                    # columns keys from json
                    keys = list(get_data[0].keys())
                    for r in get_data:
                        list_data.append(
                            [i[0]] + [i[1]] + [
                                r[k] for k in keys
                            ]
                        )

            cols = [
                'TeamName',
                'Season',
                'IdMatch',
                'Flag',
                'h_a',
                'HomeTeamId',
                'HomeTeam',
                'ShortHomeTeamName',
                'AwayTeamId',
                'AwayTeam',
                'ShortAwayTeamName',
                'FTHG',
                'FTAG',
                'HomeTeam_xG',
                'AwayTeam_xG',
                'DateTime',
                'Frcst_proba_w',
                'Frcst_proba_d',
                'Frcst_proba_l',
                'Result'
            ]
            series_list_data = []

            for i in list_data:
                series_list_data.append(self.json_to_series(i))


            df_team_results = pd.DataFrame(series_list_data, columns= cols)


            df_team_results['DateTime'] = pd.to_datetime(df_team_results['DateTime'])


            # format data values
            for c in ['Season','IdMatch','HomeTeamId','AwayTeamId','FTHG','FTAG']:
                df_team_results[c] = df_team_results[c].astype(np.int32)

            for c in ['HomeTeam_xG','AwayTeam_xG','Frcst_proba_w','Frcst_proba_d', 'Frcst_proba_l']:
                df_team_results[c] = df_team_results[c].astype(np.float32)


        return df_team_results

    async def get_parse_data(self, update_data=True, resaved_data=True):
        if update_data:
            df = await self.get_league_results()
            fixtures_df = await self.fixtures()
            df['IsResult'] = True
            fixtures_df['IsResult'] = False
            df = pd.concat([df,fixtures_df],sort=False)

            for c in ['IdMatch', 'Season','HomeTeamId','AwayTeamId']:
                df[c] = df[c].astype(np.int32)

            for c in ['HomeTeam_xG','AwayTeam_xG','WinProba','DrawProba','LoseProba','FTHG','FTAG']:
                df[c] = df[c].astype(np.float32)

            df['DateTime'] = pd.to_datetime(df['DateTime'])
            df['Date'] = df['DateTime'].dt.date
            teams_df = await self.teams()

            if resaved_data:
                df.to_pickle('./data/df.pkl')
                teams_df.to_pickle('./data/teams_df.pkl')
        else:
            try:
                df = pd.read_pickle('./data/df.pkl')
                teams_df = pd.read_pickle('./data/teams_df.pkl')
            except Exception:
                print('Нет данных, необходим интернет и параметр update_data=True')
                return pd.DataFrame(), pd.DataFrame()
        return df, teams_df
