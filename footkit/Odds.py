from os import listdir, sep
from os.path import isfile, join
import re
from bs4 import BeautifulSoup
# from DbManager import DatabaseManager
import json
from selenium import webdriver
# from SoccerMatch import SoccerMatch
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time
import pandas as pd
import numpy as np
import sys
# url = 'https://www.oddsportal.com/soccer/ecuador/liga-pro/results/'
TIMEOUT = 100

import pandas as pd
from datetime import datetime as dt
from config import Config
import numpy as np
from tqdm import tqdm

class OddsParser(Config):
    def __init__(self, Config, last_update=True):
        super().__init__()
        self.last_update=last_update
        if self.last_update:
            self.EVENT_PAGE = [
                'https://www.oddsportal.com/soccer/england/premier-league/',
                'https://www.oddsportal.com/soccer/germany/bundesliga/',
                'https://www.oddsportal.com/soccer/france/ligue-1/',
                'https://www.oddsportal.com/soccer/russia/premier-league/',
                'https://www.oddsportal.com/soccer/spain/laliga/',
                'https://www.oddsportal.com/soccer/italy/serie-a/',
            ]
        else:
            self.EVENT_PAGE = [
                'https://www.oddsportal.com/soccer/england/premier-league/results/',
                'https://www.oddsportal.com/soccer/england/premier-league/',
                'https://www.oddsportal.com/soccer/germany/bundesliga/results/',
                'https://www.oddsportal.com/soccer/germany/bundesliga/',
                'https://www.oddsportal.com/soccer/france/ligue-1/results/',
                'https://www.oddsportal.com/soccer/france/ligue-1/',
                'https://www.oddsportal.com/soccer/russia/premier-league/results/',
                'https://www.oddsportal.com/soccer/russia/premier-league/',
                'https://www.oddsportal.com/soccer/spain/laliga/',
                'https://www.oddsportal.com/soccer/spain/laliga/results/',
                'https://www.oddsportal.com/soccer/italy/serie-a/',
                'https://www.oddsportal.com/soccer/italy/serie-a/results/',
            ]
    def get_all_odds_2018(self, df):
        
        df_odds = pd.read_pickle('./Debug/parser_20182019_all.pkl')
        df_odds = df_odds[df_odds['bet_tab']!='European Handicap']
        # combined measures for concat understat
        df_odds['Targets_oddsportal'] = np.where(
            df_odds['bet_tab'].isin(['1X2','Double Chance'])
            , df_odds['bet_tab'] + ' ' + df_odds['bet_type']
            , np.where(
                df_odds['bet_tab'].isin(['Over/Under', 'European Handicap'])
                , df_odds['bet_tab'] + ' ' + df_odds['bet_type'] + ' ' + df_odds['bet_set']
                , ''
            )
        )

        df_odds = pd.merge(
            df_odds, 
            pd.read_excel('./data/info/targets.xlsx'),
            on = ['Targets_oddsportal']
        )


        df_odds = pd.merge(
            df_odds,
            pd.read_excel('./data/info/v_teams.xlsx')[
                ['Und_TeamName', 'OddsPortalTeamName']
            ].rename(
                columns={
                    'Und_TeamName':'HomeTeam',
                    'OddsPortalTeamName':'home_team'
                }
            ),
            on = 'home_team'
        )

        df_odds = pd.merge(
            df_odds,
            pd.read_excel('./data/info/v_teams.xlsx')[
                ['Und_TeamName', 'OddsPortalTeamName']
            ].rename(
                columns={
                    'Und_TeamName':'AwayTeam',
                    'OddsPortalTeamName':'away_team'
                }
            ),
            on = 'away_team'
        )

        # max bookmaker coefficient
        df_odds = pd.merge(
            df_odds,
            df_odds.groupby(
                ['HomeTeam', 'AwayTeam', 'match_date', 'Target']
            )['bet_value'].max().reset_index(),
            on = ['HomeTeam', 'AwayTeam', 'match_date', 'Target', 'bet_value']
        ).groupby(
                ['HomeTeam', 'AwayTeam', 'match_date', 'Target','bet_value']
            )['bookmaker'].max().reset_index().rename(columns={'match_date':'DateTime'})

        df_odds['Date'] = pd.to_datetime(df_odds['DateTime']).dt.date
        df_odds = df_odds.pivot_table(index=['HomeTeam','AwayTeam','Date'], columns=['Target'], values='bet_value', aggfunc='max')
        df_odds = df_odds.reset_index().reset_index(drop=True)
        # df_check = df.copy()
        df_odds = pd.merge(
            df.reset_index()[['HomeTeam', 'AwayTeam', 'Date', 'IdMatch']],
            df_odds, # .rename(columns={'bet_value':'Bet coef value'}),
            on=['HomeTeam', 'AwayTeam', 'Date'],
            how='inner'
        ).set_index('IdMatch').drop(['HomeTeam', 'AwayTeam', 'Date'], axis=1)

        return df_odds
        
    def get_odds_from_last_excel(self, df):
        import glob
        import os
        list_of_files = glob.glob(r'./Parser/data/*.xlsx')
        latest_file = max(list_of_files, key=os.path.getctime)
        # print(latest_file)
        df_odds = pd.read_excel(latest_file)
        df_odds['Date'] = pd.to_datetime(df_odds['DateTime']).dt.date

        df_bookmakers = df_odds.pivot_table(index=['HomeTeam','AwayTeam','Date'], columns=['Target'], values='bookmaker', aggfunc='max')
        df_bookmakers = df_bookmakers.reset_index().reset_index(drop=True)

        df_odds = df_odds.pivot_table(index=['HomeTeam','AwayTeam','Date'], columns=['Target'], values='bet_value', aggfunc='max')
        df_odds = df_odds.reset_index().reset_index(drop=True)

        # df_check = df.copy()
        df_odds = pd.merge(
            df.reset_index()[['HomeTeam', 'AwayTeam', 'Date', 'IdMatch']],
            df_odds, # .rename(columns={'bet_value':'Bet coef value'}),
            on=['HomeTeam', 'AwayTeam', 'Date'],
            how='inner'
        ).set_index('IdMatch').drop(['HomeTeam', 'AwayTeam', 'Date'], axis=1)

        # df_check = df.copy()
        df_bookmakers = pd.merge(
            df.reset_index()[['HomeTeam', 'AwayTeam', 'Date', 'IdMatch']],
            df_bookmakers, # .rename(columns={'bet_value':'Bet coef value'}),
            on=['HomeTeam', 'AwayTeam', 'Date'],
            how='inner'
        ).set_index('IdMatch').drop(['HomeTeam', 'AwayTeam', 'Date'], axis=1)
        return df_odds, df_bookmakers

    #### ONLINE SCRAPPER ####
    def get_browser(self, url, username = False, password = False, proxy = False, headless=True):
        # PROXY = "74.208.83.188:80" # IP:PORT or HOST:PORT # Сюда пишем прокси
        options = webdriver.ChromeOptions()
        
        if headless:
            options.add_experimental_option("excludeSwitches",["ignore-certificate-errors"])
            options.add_argument('headless')
            options.add_argument('window-size=0x0')
        if proxy:
            # options.add_argument('--proxy-server=%s' % proxy)
            options.add_extension(r'./Parser/chromedriver/bihmplhobchoageeokmgbdihknkjbknd.crx')
        self.browser = webdriver.Chrome(executable_path=r"./Parser/chromedriver/chromedriver.exe",chrome_options=options)
        self.browser.get(url)
        
        time.sleep(10)

        if username:
            inputElement = self.browser.find_element_by_xpath('//*[@id="login-username1"]')
            inputElement.send_keys(username)
            inputElement = self.browser.find_element_by_xpath('//*[@id="login-password1"]')
            inputElement.send_keys(password)
            inputElement.send_keys(Keys.ENTER)
        return self.browser

    def get_hide_elements(self): # , browser
        try :
            self.browser.find_element_by_xpath("//div[@id='show-all-link']/div/div/div/div/p[@class='all']/a").click()
            #print("have hide matches")
        except:
            pass

    def find_element_to_BS(self, element): # , browser
        # получаем нужный кусок таблицы
        content = self.browser.find_element_by_id(element)
        # Собираем все в html
        soup = BeautifulSoup(content.get_attribute("innerHTML"), "html.parser")
        return soup

    def get_urls_events_on_page(self, TableElement): # , browser
        try:
            # скрываем рекламу / или еще что-либо
            self.get_hide_elements()

            tag_list = ["odd deactivate","odd"," deactivate","odd hidden" ]
            soup = self.find_element_to_BS(TableElement)# "tournamentTable")
            links = [
                "https://www.oddsportal.com" + td.find(
                    'a', class_= lambda x: x != 'ico-tv-tournament'
                ).get(
                    'href'
                ) for td in soup.find_all(
                    'td'
                    ,class_="name table-participant"
                )
            ]        
            links = [l.split('inplay-odds/')[0] for l in links]

        except Exception:
            links = []

        return links

    def get_max_page(self):
        # находим максимальное кол-во страниц со ставками
        max_page = 1
        try:
            pages_list = self.find_element_to_BS("pagination").find_all('a')
        except:
            max_page = 1
        else:
            for a in pages_list:
                t = max_page
                try :
                    t = int(re.search(r'\d+', a.get('href')).group(0))
                except:
                    err = 1
                else:
                    if max_page < t :
                        max_page = t

        return max_page

    def get_number_season_from_site(self):
        try:
            # получили сезон, который парсим
            soup = self.find_element_to_BS("col-content")
            season = str(soup.find_all("ul", {"class" : "main-filter"})[1].find_all("span", {"class" : "active"})[0].find_all('a')[0].get_text())
        except:
            season = 'fixtures'
            # print('get_number_season_from_site - не удалось получить название сезона')
        return season

    def get_seasons_url_in_league(self, url, count_season = 3):
        # Это нужно, если парсим лиги и считываем пару последних сезонов
        # browser = get_browser(url)
        url_list = []

        try:
            soup = self.find_element_to_BS("col-content")

            # собрали основной фильтер - это номера сезонов в историю
            season = soup.find_all("ul", {"class" : "main-filter"})

            for a in season[1].find_all('a', href=True)[0:count_season]:
                # print ('https://www.oddsportal.com'+a['href'])
                url_list.append('https://www.oddsportal.com'+a['href'])
                # print(url_list)
        except Exception:
            # print('get_seasons_url_in_league - не удалось собрать ссылки сезонов в лиге')
            url_list = [url]

        return url_list

    def get_urls_in_leagues(self, TableElement, url, last_update_ = True):

        '''
        TableElement = "tournamentTable"
        url = 'https://www.oddsportal.com/soccer/france/ligue-1/'
        '''

        # Собираем все ссылки на все матчи
        urls_events = []
        self.browser.get(url)
        max_page = self.get_max_page()
        # только последнюю, если LAST_UPDATE = True
        if max_page == 1 :

            # ждем пока загрузится страница
            self.browser.get(url)
            # try:
            element = WebDriverWait(self.browser, TIMEOUT).until(EC.visibility_of_element_located((By.ID, TableElement)))
            # except:
            #     return urls_events
            urls_events = self.get_urls_events_on_page(TableElement)
        else:
            pages_data = []
            for i in range(1, 3 if last_update_ else max_page + 1):
                self.browser.get(url + "#/page/" + str(i) + "/")
                # add wait
                WebDriverWait(self.browser, TIMEOUT).until(EC.visibility_of_element_located((By.ID, TableElement)))
                pages_data.append(self.get_urls_events_on_page(TableElement ))

            for pages in pages_data:
                for u in pages:
                    # if u.split('/')[-2] == 'inplay-odds':
                    #     # print('очищаем из ссылки /inplay-odds/ - коэф. во время матча')
                    #     u = '/'.join(u.split('/')[:-2])+'/'
                    urls_events.append(u.split('inplay-odds/')[0])

        return urls_events

    def get_header_event(self, link):

        # open link
        try:
            self.browser.get(link)
            #ждем пока загрузится страница
            element = WebDriverWait(self.browser, TIMEOUT).until(EC.visibility_of_element_located((By.ID, "odds-data-table")))
        except Exception:
            try:
                self.browser.get(link)
                #ждем пока загрузится страница
                element = WebDriverWait(self.browser, TIMEOUT).until(EC.visibility_of_element_located((By.ID, "odds-data-table")))
            except Exception:
                pass
        # finally:
        #     print(link)


        soup = self.find_element_to_BS("col-content")
        match = soup.find('h1').text
        # Наименование команд
        home_team = match.split('-')[0].strip()
        away_team = match.split('-')[1].strip()

        # Парсим дату
        regex = re.compile('.*date datet .*')
        dates_ = soup.find_all("p", {"class" : regex})[0].get_text()
        match_date = dt.strptime((dates_[dates_.find(',')+1:]).strip(),'%d %b %Y, %H:%M')

        # Парсим счёт
        soup = self.find_element_to_BS("event-status")
        home_point, away_point = -1, -1

        try:
            result = soup.find_all("p", {"class" : 'result'})[0]
            final_point = result.find('strong').get_text()
            home_point = result.find('strong').get_text().split(' ')[0].split(':')[0].strip()
            away_point = result.find('strong').get_text().split(' ')[0].split(':')[1].strip()
        except Exception:
            # на текущий момент результата - нет
            # final_point = result.find('strong').get_text()
            # home_point = 1
            # away_point = result.find('strong').get_text().split(' ')[0].split(':')[1].strip()
            pass

        # Парсим наименование лиги
        breadcrumb = self.find_element_to_BS("breadcrumb")
        Div = breadcrumb.find_all('a')[-2].get_text() + ' | ' + breadcrumb.find_all('a')[-1].get_text()
        Div_main = breadcrumb.find_all('a')[-2].get_text()
        return match_date, home_team, away_team, home_point, away_point, Div, Div_main

    def get_info_from_tab(self, df, link, season):

        # columns = ["Div", "Season", "match_date","home_team","away_team", "home_point", "away_point", "bookmaker","bet_tab","bet_type", "bet_set","bet_value"]
        # df = pd.DataFrame(columns=columns)
        span_array = {
            "1X2": 0,
            # "Home/Away" : 0,
            # "Asian Handicap": 5,
            "Over/Under": 5,
            # "Draw No Bet": 0,
            "European Handicap": 6,
            "Double Chance": 0,
            # "Correct Score": 3,
            # "Both Teams to Score": 0,
            # "Odd or Even": 0,
        }

        tabs_xpath = {
            "1X2": '//*[@id="bettype-tabs"]/ul/li[2]',
            # "Home/Away" : 0,
            "Asian Handicap": '//*[@id="bettype-tabs"]/ul/li[4]',
            "Over/Under": '//*[@id="bettype-tabs"]/ul/li[5]',
            "Draw No Bet": '//*[@id="bettype-tabs"]/ul/li[6]',
            "European Handicap": '//*[@id="bettype-tabs"]/ul/li[7]',
            "Double Chance": '//*[@id="bettype-tabs"]/ul/li[8]',
            # "Correct Score": '//*[@id="bettype-tabs"]/ul/li[10]',
            # "Both Teams to Score": '//*[@id="bettype-tabs"]/ul/li[14]',
            # "Odd or Even": '//*[@id="tab-sport-others"]',
        }

        tabs_array = list(span_array.keys())
        # print(tabs_array)

        match_date, home_team, away_team, home_point, away_point, Div, Div_main = self.get_header_event(link)
        for tab in tabs_array:
            span = span_array[tab]

            try:
                self.browser.find_element_by_xpath("//ul[@class='ul-nav']/li/a[@title='"+tab+"']").click()
            except:
                try:
                    # hide elements
                    element = self.browser.find_element_by_xpath("//ul[@class='ul-nav']/li[@class='r more ']/div[@class='othersListParent']/div/p/a[text()='%s']" % tab)
                    self.browser.execute_script("$(arguments[0]).click();", element)
                except:
                    try:
                        self.browser.find_element_by_xpath(tabs_xpath[tab]).click()
                    except:
                        pass
                        # print('Error, dont click on xpath tab')

            soup = self.find_element_to_BS('odds-data-table')
            containers = soup.find_all('div', class_ = "table-container")
            try:
                WebDriverWait(self.browser, TIMEOUT).until(EC.visibility_of_element_located((By.ID, "odds-data-table")))
            except:
                try:
                    WebDriverWait(self.browser, TIMEOUT).until(EC.visibility_of_element_located((By.ID, "odds-data-table")))
                except:
                    WebDriverWait(self.browser, TIMEOUT).until(EC.visibility_of_element_located((By.ID, "odds-data-table")))
            # расскрываем скрытые вкладки, где это необходимо (span != 0)
            # //*[@id="odds-data-table"]/div[3]/div/span[6]/a
            if span !=0:
                for i in range(1,60):
                    try:
                        if self.browser.find_element_by_xpath("//*[@id='odds-data-table']/div[" + str(i) + "]/div/span[" + str(span) + "]/a").text != 'Hide odds':
                            self.browser.find_element_by_xpath("//*[@id='odds-data-table']/div[" + str(i) + "]/div/span[" + str(span) + "]/a").click()
                            WebDriverWait(self.browser, TIMEOUT).until(EC.visibility_of_element_located((By.CLASS_NAME, "more")))

                    # close 
                        if self.browser.find_element_by_xpath("//*[@id='odds-data-table']/div[" + str(i) + "]/div/span[" + str(span) + "]/a").text == 'Hide odds':
                            self.browser.find_element_by_xpath("//*[@id='odds-data-table']/div[" + str(i) + "]/div/span[" + str(span) + "]/a").click()
                            WebDriverWait(self.browser, TIMEOUT).until(EC.visibility_of_element_located((By.CLASS_NAME, "more")))
                    except Exception:
                        # print('error 1. wait browser')
                        pass
            soup = self.find_element_to_BS('odds-data-table')
            containers = soup.find_all('div', class_ = "table-container")
            try:
                WebDriverWait(self.browser, TIMEOUT).until(EC.visibility_of_element_located((By.ID, "odds-data-table")))
            except:
                try:
                    WebDriverWait(self.browser, TIMEOUT).until(EC.visibility_of_element_located((By.ID, "odds-data-table")))
                except:
                    WebDriverWait(self.browser, TIMEOUT).until(EC.visibility_of_element_located((By.ID, "odds-data-table")))

            for tContainer in containers:

                col_tabs = []
                for el in tContainer.find_all('th', class_ = "center odds-odds") :
                    if el.find('a') != None:
                        col_tabs.append(el.find('a').text)
                try:
                    bet_type = tContainer.find_all('a')[0].text
                except:
                    # print('error 2. center odds-odds')
                    pass

                # print(col_tabs)
                # цикл для букмекеров

                for el in tContainer.find_all('tr', class_= re.compile('lo *') ) :

                    i = 0

                    for td in el.find_all('td'):
                        # print(td)
                        # тип ставки
                        # print(el.find_all('td', class_ = 'center'))
                        if el.find_all('td', class_ = 'center')[0].text:
                            bet_set = el.find_all('td', class_ = 'center')[0].text
                        # print('букмекер ',td)

                        # Определяем столбец с букмекером
                        if td.find('a', class_ = 'name') != None :
                            bookmaker = td.find('a', class_ = 'name').get_text()
                            # print(td.find('a', class_ = 'name').get_text())

                        # Определяем столбец с типом ставки    
                        # print(td)
                        # print('*******************')

    #                     if tab != "1X2":
    #                         print(td)

                        if td.has_attr('class') and 'right' in td['class']:
                            # 
                            if len(td.find_all('div'))>0 and td.find('div').text != 'Log in to display the OddsAlert!':
                                # print(bookmaker,bet_set, td.find('div').text)
                                try:
                                    bet_value = float(td.find('div').text)
                                except:
                                    # print('error 3. bet_value null')
                                    bet_value = np.nan

                                # print(Div, season, match_date,home_team,away_team, home_point, away_point,bookmaker,tab, col_tabs[i], bet_set,bet_value)
                                #Добавляем запись в DataFrame
                                df.loc[len(df)] = [Div, season, match_date, home_team, away_team, home_point, away_point,bookmaker,tab, col_tabs[i], bet_set,bet_value]
                                i += 1

                            if len(td.find_all('a'))>0 and td.find('a').text != 'Log in to display the OddsAlert!':
                                # print(bookmaker,bet_set, td.find('a').text)
                                try:
                                    bet_value = float(td.find('a').text)
                                    # print(Div, season, match_date,home_team,away_team, home_point, away_point,bookmaker,tab, col_tabs[i], bet_set,bet_value)
                                    #Добавляем запись в DataFrame
                                    df.loc[len(df)] = [Div, season, match_date,home_team,away_team, home_point, away_point,bookmaker,tab, col_tabs[i], bet_set,bet_value]
                                    i += 1
                                except:
                                    # print('error 3. df not added')
                                    pass
        return df
    
    def get_url_events(self):
        # # 2018-2019 Season
        # EVENT_PAGE = [
        #     'https://www.oddsportal.com/soccer/england/premier-league-2018-2019/results/',
        #     'https://www.oddsportal.com/soccer/germany/bundesliga-2018-2019/results/',
        #     'https://www.oddsportal.com/soccer/france/ligue-1-2018-2019/results/',
        #     'https://www.oddsportal.com/soccer/russia/premier-league-2018-2019/results/',
        #     'https://www.oddsportal.com/soccer/spain/laliga-2018-2019/results/',
        #     'https://www.oddsportal.com/soccer/italy/serie-a-2018-2019/results/',
        # ]

        self.URLS_EVENTS = []
        # EVENT_PAGE = prev_7_days
        # max_page = 1

        # browser.get(EVENT_PAGE[0])
        # print(season)
        # season = '2018/2019'
        # get_number_season_from_site(browser)
        # собираем все ссылки на матчи в event page

        for u in tqdm(self.EVENT_PAGE):
            self.URLS_EVENTS = self.URLS_EVENTS + self.get_urls_in_leagues(
                    # browser,
                    TableElement = "table-matches" if u.split('/')[3] == 'matches' else "tournamentTable",
                    url = u,
                    last_update_ = self.last_update
            )
        # print('In', len(self.EVENT_PAGE), 'league search', len(self.URLS_EVENTS), 'events')
        # return URLS_EVENTS


    def get_data_from_url_events(
        self,
        df_predict,
        season = 2019,
        columns = ["Div", "Season", "match_date","home_team","away_team", "home_point", "away_point", "bookmaker","bet_tab","bet_type", "bet_set","bet_value"],
        save_to_file=True
    ):

        df_parser = pd.DataFrame(columns=columns)

        for url in self.URLS_EVENTS: # [:2]
            df_parser = self.get_info_from_tab(df_parser, url, season)

        # combined measures for concat understat
        df_parser['Targets_oddsportal'] = np.where(
            df_parser['bet_tab'].isin(['1X2','Double Chance'])
            , df_parser['bet_tab'] + ' ' + df_parser['bet_type']
            , np.where(
                df_parser['bet_tab'].isin(['Over/Under', 'European Handicap'])
                , df_parser['bet_tab'] + ' ' + df_parser['bet_type'] + ' ' + df_parser['bet_set']
                , ''
            )
        )

        df_parser = pd.merge(
            df_parser, 
            pd.read_excel('./data/info/targets.xlsx'),
            on = ['Targets_oddsportal']
        )

        df_parser = pd.merge(
            df_parser,
            pd.read_excel('./data/info/v_teams.xlsx')[
                ['Und_TeamName', 'OddsPortalTeamName']
            ].rename(
                columns={
                    'Und_TeamName':'HomeTeam',
                    'OddsPortalTeamName':'home_team'
                }
            ),
            on = 'home_team'
        )

        df_parser = pd.merge(
            df_parser,
            pd.read_excel('./data/info/v_teams.xlsx')[
                ['Und_TeamName', 'OddsPortalTeamName']
            ].rename(
                columns={
                    'Und_TeamName':'AwayTeam',
                    'OddsPortalTeamName':'away_team'
                }
            ),
            on = 'away_team'
        )


        # df.rename(columns={'Targets':'Target'},inplace=True)
        self.final_parse_df = pd.merge(
            df_parser,
            df_parser.groupby(
                ['HomeTeam', 'AwayTeam', 'match_date', 'Target']
            )['bet_value'].max().reset_index(),
            on = ['HomeTeam', 'AwayTeam', 'match_date', 'Target', 'bet_value']
        ).groupby(
                ['HomeTeam', 'AwayTeam', 'match_date', 'Target','bet_value']
            )['bookmaker'].max().reset_index().rename(columns={'match_date':'DateTime'})
        
        if save_to_file:
            time = str(dt.now().date())
            time = time[:4]+'_'+time[5:7]+'_'+time[8:10]
            self.final_parse_df.to_excel('./Parser/data/'+time+' parser_last_update.xlsx', index=None)

        
        self.df_odds = self.final_parse_df.copy()
        self.df_odds['Date'] = pd.to_datetime(self.df_odds['DateTime']).dt.date

        self.df_bookmakers = self.df_odds.pivot_table(index=['HomeTeam','AwayTeam','Date'], columns=['Target'], values='bookmaker', aggfunc='max')
        self.df_bookmakers = self.df_bookmakers.reset_index().reset_index(drop=True)

        self.df_odds = self.df_odds.pivot_table(index=['HomeTeam','AwayTeam','Date'], columns=['Target'], values='bet_value', aggfunc='max')
        self.df_odds = self.df_odds.reset_index().reset_index(drop=True)

        # df_check = df.copy()
        self.df_odds = pd.merge(
            df_predict.reset_index()[['HomeTeam', 'AwayTeam', 'Date', 'IdMatch']],
            self.df_odds, # .rename(columns={'bet_value':'Bet coef value'}),
            on=['HomeTeam', 'AwayTeam', 'Date'],
            how='inner'
        ).set_index('IdMatch').drop(['HomeTeam', 'AwayTeam', 'Date'], axis=1)

        # df_check = df.copy()
        self.df_bookmakers = pd.merge(
            df_predict.reset_index()[['HomeTeam', 'AwayTeam', 'Date', 'IdMatch']],
            self.df_bookmakers, # .rename(columns={'bet_value':'Bet coef value'}),
            on=['HomeTeam', 'AwayTeam', 'Date'],
            how='inner'
        ).set_index('IdMatch').drop(['HomeTeam', 'AwayTeam', 'Date'], axis=1)
        return self.df_odds, self.df_bookmakers
    
    def get_last_update_scrapper(
        self,
        df,
        season=2019,
        url='https://www.oddsportal.com/login/',
        username = 'khaitov',
        password = '*******',
        proxy=False,
        headless=True,
        from_file=False
    ):
        if from_file:
            df_odds, df_bookmakers = self.get_odds_from_last_excel(df)
        else:
            browser = self.get_browser(
                url=url
                , username=username
                , password=password
                , proxy=proxy
                , headless=headless
            )

            self.get_url_events()
            df_odds, df_bookmakers = self.get_data_from_url_events(df, season=season)
            self.browser.close()
        return df_odds, df_bookmakers