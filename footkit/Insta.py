from instabot import Bot, utils
import re
# import sqlalchemy as sa
import pandas as pd
import numpy as np
import time
import threading
import glob
import sys
import os
sys.path.append(os.path.join(sys.path[0], '../Instagram/instabot_txt/'))
import schedule
from config import Config
# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
# import numpy as np
# # Создаем модель с архитектурой VGG16 и загружаем веса, обученные
# # на наборе данных ImageNet
# model = VGG16(weights='imagenet')

class InstaBot(Config):
    def __init__(self, Config):
        super().__init__()

    def login_bot(self):
        try:
            self.bot.logout()
            # print('log')
        except Exception:
            pass

        self.bot = Bot(
            whitelist_file='../Instagram/instabot_txt/whitelist.txt',
            blacklist_file='../Instagram/instabot_txt/blacklist.txt', 
            comments_file='../Instagram/instabot_txt/comments.txt', 
            followed_file='../Instagram/instabot_txt/followed.txt', 
            unfollowed_file='../Instagram/instabot_txt/unfollowed.txt', 
            skipped_file='../Instagram/instabot_txt/skipped.txt', 
            friends_file='../Instagram/instabot_txt/friends.txt', 
            proxy=None, 
            max_likes_per_day=1000, 
            max_unlikes_per_day=1000, 
            max_follows_per_day=1000, 
            max_unfollows_per_day=1000,
            max_comments_per_day=100, 
            max_blocks_per_day=100, 
            max_unblocks_per_day=100, 
            max_likes_to_like=100, 
            min_likes_to_like=20, 
            max_messages_per_day=500, 
            filter_users=True, 
            filter_private_users=True, 
            filter_users_without_profile_photo=True, 
            filter_previously_followed=False, 
            filter_business_accounts=True, 
            filter_verified_accounts=True, 
            max_followers_to_follow=2000, 
            min_followers_to_follow=10, 
            max_following_to_follow=2000, 
            min_following_to_follow=10, 
            max_followers_to_following_ratio=10, 
            max_following_to_followers_ratio=2, 
            min_media_count_to_follow=3,
            max_following_to_block=2000, 
            like_delay=10, 
            unlike_delay=10,
            follow_delay=60, 
            unfollow_delay=30, 
            comment_delay=60,
            block_delay=30,
            unblock_delay=30,
            message_delay=2,
            stop_words=('shop', 'store', 'free'),
            blacklist_hashtags=['#shop', '#store', '#free'],
            blocked_actions_protection=True,
            verbosity=True,
            device=None,
            # save_logfile=False,
        )

        self.bot.login(
            username=self.instagram_username,
            password=self.instagram_password,
        )
        
    def create_dir(self, path):
        import os
        # define the name of the directory to be created
        try:
            os.mkdir(path)
        except OSError:
            # print ("Creation of the directory %s failed" % path)
            pass
        else:
            print ("Successfully created the directory %s " % path)

    def sleep(self, n_sec):
        import time
        
        n_sec = int(n_sec)+1
        for i in range(n_sec):
            time.sleep(1)

    def update_follow_flag(self):
        import ast
        try:
            with open('../Instagram/instabot_txt/flag_follow.txt') as f:
                flag_follow = f.read().splitlines()[0]
        except:
            flag_follow = 'True'
        return ast.literal_eval(flag_follow)

    def clear_old_followers(self,follow_list):
        with open('../Instagram/instabot_txt/followed.txt') as f:
            followed_txt = f.read().splitlines()

        with open('../Instagram/instabot_txt/skipped.txt') as f:
            skipped_txt = f.read().splitlines()

        followed_txt = np.unique(np.concatenate([followed_txt,skipped_txt]))
        return list(set(follow_list) - set(followed_txt))

    # массфоловинг
    def massfollow(self,bot, count_actions):
        from random import random
        print('Start...')
        # получаем список ЦА
        with open('../Instagram/instabot_txt/userdb.txt') as f:
            follow_list = f.read().splitlines()

        # Колво подписчиков
        CNT_FOLLOWINGS = bot.get_user_info(bot.user_id)['following_count']
        
        # счетчик подписок/отписок
        cnt_follow = 0 
        
        # счетчик
        number_follow_user = 0
        
        # чистим от старых пользователей
        follow_list = clear_old_followers(follow_list)
        
        # проверяем флаг
        flag_follow = update_follow_flag()
           
        print('Колво подписчиков:',CNT_FOLLOWINGS)
        
        
        # ПОДПИСКА
        if flag_follow: 
            print('Начинаем подписку')
            
        while cnt_follow < count_actions and flag_follow == True:
            if bot.follow(follow_list[number_follow_user]):
                sleep(random()*30)
                cnt_follow += 1

            number_follow_user += 1

            if CNT_FOLLOWINGS + cnt_follow >= 6999:
                utils.file('../Instagram/instabot_txt/flag_follow.txt').save_list([False])
                flag_follow = update_follow_flag()
                # Сутки
                print('Ждем сутки, перед тем как отписываться')
                sleep(86400)
        
        # ОТПИСКА
        # обнуляем счетчики
        cnt_unfollow = 0
        number_unfollow_user = 0
        CNT_FOLLOWINGS = bot.get_user_info(bot.user_id)['following_count']

        # Получаем список подписчиков
        unfollow_list = bot.following
        unfollow_list.reverse()
        
        if flag_follow == False: 
            print('Начинаем отписку')
        while cnt_unfollow < count_actions and flag_follow == False:
            
            sleep(15+random()*30)
            bot.unfollow(unfollow_list[number_unfollow_user])
            number_unfollow_user += 1

            if CNT_FOLLOWINGS - cnt_unfollow == 115:
                utils.file('../Instagram/instabot_txt/flag_follow.txt').save_list([True])
                flag_follow = update_follow_flag()
                # Сутки
                print('Ждем сутки, перед тем как подписываться')
                sleep(86400)


    def get_followings_user_indb(self,bot,username,nfollows, clear_base = False):
        try:
            with open('../Instagram/instabot_txt/userdb.txt') as f:
                follow_list = f.read().splitlines()
        except:
            follow_list = []
        
        if clear_base:
            follow_list = []
        else:
            print('Было в базе:', len(follow_list))
        
        id_comp = bot.get_user_id_from_username(username)
        followers_comp = bot.get_user_followers(id_comp, nfollows=nfollows)
        
        follow_list = np.unique(np.concatenate([follow_list,followers_comp]))
        utils.file('../Instagram/instabot_txt/userdb.txt').save_list(follow_list)
        print('Стало в базе:', len(follow_list))

        
    def download_videos_from_username(
        self,
        bot, 
        path = '../Instagram/videos/',
        user_competitor_name = 'pnh',
        count_medias = 100
    ):
        
        '''
            # example:
            download_videos_from_username('mountains_geek')
        '''
        cnt_download = 0
        total_medias = bot.get_total_user_medias(bot.get_user_id_from_username(user_competitor_name))
        
        if count_medias != False:
            if count_medias > len(total_medias):
                print('Count > total medias, count -', len(total_medias))
                count_medias = len(total_medias)
            else:
                # ничего не меняем
                count_medias = count_medias
        else:
            count_medias = len(total_medias)
        
        
        create_dir(path+user_competitor_name)
        for m in total_medias[:count_medias]:
            if cnt_download <= count_medias:
                bot.download_video(m,folder=path+user_competitor_name,save_description=True)
                cnt_download += 1

    def download_photos_from_username(
        self,
        bot, 
        path = '../Instagram/photos/', 
        user_competitor_name = 'pnh',
        count_medias = 100
    ):
        
        '''
            # example:
            download_photos_from_username('mountains_geek')
        '''
        cnt_download = 0
        total_medias = bot.get_total_user_medias(bot.get_user_id_from_username(user_competitor_name))
        
        if count_medias != False:
            if count_medias > len(total_medias):
                print('Count > total medias, count -', len(total_medias))
                count_medias = len(total_medias)
            else:
                # ничего не меняем
                count_medias = count_medias
        else:
            count_medias = len(total_medias)
        create_dir(path+user_competitor_name)
        for m in total_medias:
            if cnt_download <= count_medias:
                bot.download_photo(m,folder=path+user_competitor_name,save_description=True)
                cnt_download += 1


    # autoposting
    def upload_videos(self, bot, cnt_media, path_medias, fromCaptions = 'helped_text', top10_hashtag_pred = ['']):
        # Get the filenames of the medias in the path ->
        posted_medias_file = "../Instagram/instabot_txt/videos.txt"
        posted_medias_list = []

        if not os.path.isfile(posted_medias_file):
            with open(posted_medias_file, 'w'):
                pass
        else:
            with open(posted_medias_file, 'r') as f:
                posted_medias_list = f.read().splitlines()

        caption = ''
        # fromCaptions = 'File'
        # fromCaptions = 'helped_text'
        # path_medias = './videos/all/'
        medias = []
        exts = ['mp4', 'MP4', 'mov', 'MOV']
        for ext in exts:
            medias += [os.path.basename(x) for x in glob.glob(path_medias+'*.{}'.format(ext))]
        from random import shuffle
        shuffle(medias)

        medias = list(set(medias) - set(posted_medias_list))
        media = medias[cnt_media]

        helped_text = '''
                \n \nОтмечай друзей 👌 
                \n👇\n➖➖➖➖➖➖➖➖➖
                \n➖\nБольше смешных видео футбольных
                \n🤣\n@tv.target
                \n👌\n➖➖➖➖➖➖➖➖➖
                \n➖\n😈Отметь своих друзей
                \n😈\n🤯Каждый день новые публикации
                \n🤯\n❌Я не являюсь владельцем медиа
                \n❌\n➖➖➖➖➖➖➖➖➖
                \n➖\nФутбольные ставки/#Football #Bets #Recommendations
                \n:\n@footballbets.tv
                '''

        if fromCaptions == 'Comment':
            try:
                caption = pd.DataFrame(
                    bot.get_media_comments_all(
                        media.split('.')[0].split('_')[len(media.split('.')[0].split('_'))-1]
                    )
                ).sort_values(
                    [
                        'comment_like_count'
                    ],
                    ascending=[False]
                )['text'][0] + helped_text
            except:
                caption = helped_text
        
        elif fromCaptions == 'File':
            try:
                with open(path_medias+media.split('.')[0]+'.txt', 'r', encoding='utf-8') as f:
                    caption = f.read().splitlines() + helped_text
            except:
                try:
                    caption = bot.get_media_info(
                        media.split('.')[0].split('_')[len(media.split('.')[0].split('_'))-1]
                    )[0]['caption']['text'] + helped_text
                except:
                     caption = helped_text
                        
        elif fromCaptions == 'helped_text':
            caption = helped_text
            
        else:
            try:
                caption = bot.get_media_info(
                    media.split('.')[0].split('_')[len(media.split('.')[0].split('_'))-1]
                )[0]['caption']['text'] + helped_text
            except:
                 caption = helped_text

        if bot.upload_video(path_medias + media, caption=caption):

            # top10_hashtag_pred = '#apexlegendsxbox #apexlegendsclips #apexlegendsmeme #apexlegendsbr #apexbattleroyale #apexlegendsps4 #apexlegends2019 #apexlegendsmemes #apex #legends #gaming #easports #apexlegendsdaily #apexlegendspc #apexlegendswin #apexlegends #рapexlegendsvideos #apexlegendsclip #explore #fortnite #blackops4'
            # Пробуем собрать хэштэги из файла
            try:
                with open(path_medias+media.split('.')[0]+'.txt', 'r', encoding='utf-8') as f:
                    s = f.read().splitlines()
                    s = [i  for i in ' '.join(s).split() if i.startswith("#") ]
                    if len(s)==0:
                        s= [' ']
            except:
                s = [' ']
            
    #         print(top10_hashtag_pred,s)
            
            bot.comment(
                bot.get_user_medias(
                    bot.user_id,filtration=False
                )[0]
                , '''
            Tags:
            '''+' '.join(list(np.unique(np.concatenate([top10_hashtag_pred.split(' '),s])))[:25]) # Не больше 25 тэгов
            )
            
            with open(posted_medias_file, 'a') as f:
                f.write(media + "\n")
                bot.logger.info("Succesfully uploaded: " + media)
            
            clear_media = media[:-''.join(reversed(media)).find('.')-1]
            for delete_medias in glob.glob(path_medias+clear_media+'.*'):
                try:
                    os.remove(delete_medias)
                except:
                    print('File', delete_medias,'does not exist')

            upload_flag = True
        else:
            bot.logger.info("Не смогли загрузить: " + media)
            upload_flag = False
        return upload_flag


    def upload_photos(self, cnt_media, path_medias, fromCaptions = 'helped_text', top10_hashtag_pred = ['']):
        # Get the filenames of the medias in the path ->
        posted_medias_file = "./Pictures/photos.txt"

        posted_medias_list = []

        if not os.path.isfile(posted_medias_file):
            with open(posted_medias_file, 'w'):
                pass
        else:
            with open(posted_medias_file, 'r') as f:
                posted_medias_list = f.read().splitlines()

        caption = ''
        # fromCaptions = 'File'
        # fromCaptions = 'helped_text'
        # path_medias = './photos/all/'
        medias = []
        exts = ['png', 'jpeg', 'jpg', 'bmp']
        for ext in exts:
            medias += [os.path.basename(x) for x in glob.glob(path_medias+'*.{}'.format(ext))]
        from random import shuffle
        shuffle(medias)

        medias = list(set(medias) - set(posted_medias_list))
        media = medias[cnt_media]

    #     # predict keywords
    #     img_path = path_medias+media
    #     img = image.load_img(img_path, target_size=(224, 224))
    #     x = image.img_to_array(img)
    #     x = np.expand_dims(x, axis=0)
    #     x = preprocess_input(x)

    #     # Запускаем распознавание объекта на изображении
    #     preds = model.predict(x)

    #     # Печатаем три класса объекта с самой высокой вероятностью
    #     print('Результаты распознавания:', decode_predictions(preds, top=10)[0])
    #     top_pred = ' '.join(decode_predictions(preds, top=1)[0][0][1].split('_'))
    #     top10_hashtag_pred = ['#'+i[1] for i in decode_predictions(preds, top=10)[0]]

        helped_text = top10_hashtag_pred + '@' + self.bot.username

        if fromCaptions == 'Comment':
            try:
                caption = pd.DataFrame(
                    self.bot.get_media_comments_all(
                        media.split('.')[0].split('_')[len(media.split('.')[0].split('_'))-1]
                    )
                ).sort_values(
                    [
                        'comment_like_count'
                    ],
                    ascending=[False]
                )['text'][0] + helped_text
            except:
                caption = helped_text
        
        elif fromCaptions == 'File':
            # try:
            with open(path_medias+media[:-''.join(reversed(media)).find('.')-1]+'.txt', 'r', encoding='utf-8') as f:
                caption = f.read().splitlines() # + helped_text
                caption = ''.join(caption)
           # except:
           #     try:
           #         caption = self.bot.get_media_info(
           #             media.split('.')[0].split('_')[len(media.split('.')[0].split('_'))-1]
           #         )[0]['caption']['text'] + helped_text
           #     except:
           #          caption = helped_text
                        
        elif fromCaptions == 'helped_text':
            caption = helped_text
            
        else:
            try:
                caption = self.bot.get_media_info(
                    media.split('.')[0].split('_')[len(media.split('.')[0].split('_'))-1]
                )[0]['caption']['text'] + helped_text
            except:
                 caption = helped_text

        if self.bot.upload_photo(path_medias + media, caption=caption):
            
            # Пробуем собрать хэштэги из файла
            try:
                with open(path_medias+media.split('.')[0]+'.txt', 'r', encoding='utf-8') as f:
                    s = f.read().splitlines()
                    s = [i  for i in ' '.join(s).split() if i.startswith("#") ]
                    if len(s)==0:
                        s= [' ']
            except:
                s = [' ']
                
            print(top10_hashtag_pred,s)
            self.bot.comment(
                self.bot.get_user_medias(
                    self.bot.user_id,filtration=False
                )[0]
                , '''
            Tags:
            '''+' '.join(list(np.unique(np.concatenate([top10_hashtag_pred.split(' '),s])))[:25]) # Не больше 25 тэгов
            )

            with open(posted_medias_file, 'a') as f:
                f.write(media + "\n")
                self.bot.logger.info("Succesfully uploaded: " + media)
            
            clear_media = media[:-''.join(reversed(media)).find('.')-1]
            for delete_medias in glob.glob(path_medias+clear_media+'.*'):
                try:
                    os.remove(delete_medias)
                except:
                    print('File', delete_medias,'does not exist')

            upload_flag = True
        else:
            self.bot.logger.info("Не смогли загрузить: " + media)
            upload_flag = False
        return upload_flag


    def autoposting(
        self,
        # bot,
        method,
        path_medias,
        fromCaptions = 'helped_text', # описание под медиа
        top10_hashtag_pred = '', # можно добавить топ хэштегов в комментарий
    ):
        '''
        # example:
        while True:
            autoposting(upload_videos, path_medias) # upload_photos
            sleep(3600)

        '''
        cnt_media = 0
        flag_update = False
        
        while flag_update!=True:
            if cnt_media>0:
                print('Пробуем снова.')
            flag_update = method(cnt_media=cnt_media, path_medias = path_medias, fromCaptions = fromCaptions, top10_hashtag_pred = top10_hashtag_pred)
            cnt_media += 1
            if cnt_media > 10:
                pass