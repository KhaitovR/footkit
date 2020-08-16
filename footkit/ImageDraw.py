from PIL import Image, ImageDraw, ImageFont
import textwrap
import glob
import os
import pandas as pd
import numpy as np
from datetime import datetime as dt, timedelta
from config import Config

class PlotPredict(Config):
    def __init__(self, Config):
        super().__init__()
#         Config().__init__()

    # Open an Image
    def open_image(self, path):
        newImage = Image.open(path)
        return newImage

    # Save Image
    def save_image(self, image, path):
        image.save(path, 'jpeg')

    # Create a new image with the given size
    def create_image(self, i, j):
        image = Image.new("RGB", (i, j), "white")
        return image

    # Get the pixel from the given image
    def get_pixel(self, image, i, j):
        # Inside image bounds?
        width, height = image.size
        if i > width or j > height:
            return None

        # Get Pixel
        pixel = image.getpixel((i, j))
        return pixel

    # Create a Grayscale version of the image
    def convert_grayscale(self, image):
        # Get size
        width, height = image.size

        # Create new Image and a Pixel Map
        new = self.create_image(width, height)
        pixels = new.load()

        # Transform to grayscale
        for i in range(width):
            for j in range(height):
                # Get Pixel
                pixel = self.get_pixel(image, i, j)

                # Get R, G, B values (This are int from 0 to 255)
                red =   pixel[0]
                green = pixel[1]
                blue =  pixel[2]

                # Transform to grayscale
                gray = (red * 0.299) + (green * 0.587) + (blue * 0.114)

                # Set Pixel in new image
                pixels[i, j] = (int(gray), int(gray), int(gray))

        # Return new image
        return new

    # Create a Half-tone version of the image
    def convert_halftoning(self, image):
        # Get size
        width, height = image.size
      
        # Create new Image and a Pixel Map
        new = self.create_image(width, height)
        pixels = new.load()
      
        # Transform to half tones
        for i in range(0, width, 2):
            for j in range(0, height, 2):
                # Get Pixels
                p1 = self.get_pixel(image, i, j)
                p2 = self.get_pixel(image, i, j + 1)
                p3 = self.get_pixel(image, i + 1, j)
                p4 = self.get_pixel(image, i + 1, j + 1)
          
                # Transform to grayscale
                gray1 = (p1[0] * 0.299) + (p1[1] * 0.587) + (p1[2] * 0.114)
                gray2 = (p2[0] * 0.299) + (p2[1] * 0.587) + (p2[2] * 0.114)
                gray3 = (p3[0] * 0.299) + (p3[1] * 0.587) + (p3[2] * 0.114)
                gray4 = (p4[0] * 0.299) + (p4[1] * 0.587) + (p4[2] * 0.114)
          
                # Saturation Percentage
                sat = (gray1 + gray2 + gray3 + gray4) / 4
          
                # Draw white/black depending on saturation
                if sat > 223:
                    pixels[i, j]         = (255, 255, 255) # White
                    pixels[i, j + 1]     = (255, 255, 255) # White
                    pixels[i + 1, j]     = (255, 255, 255) # White
                    pixels[i + 1, j + 1] = (255, 255, 255) # White
                elif sat > 159:
                    pixels[i, j]         = (255, 255, 255) # White
                    pixels[i, j + 1]     = (0, 0, 0)       # Black
                    pixels[i + 1, j]     = (255, 255, 255) # White
                    pixels[i + 1, j + 1] = (255, 255, 255) # White
                elif sat > 95:
                    pixels[i, j]         = (255, 255, 255) # White
                    pixels[i, j + 1]     = (0, 0, 0)       # Black
                    pixels[i + 1, j]     = (0, 0, 0)       # Black
                    pixels[i + 1, j + 1] = (255, 255, 255) # White
                elif sat > 32:
                    pixels[i, j]         = (0, 0, 0)       # Black
                    pixels[i, j + 1]     = (255, 255, 255) # White
                    pixels[i + 1, j]     = (0, 0, 0)       # Black
                    pixels[i + 1, j + 1] = (0, 0, 0)       # Black
                else:
                    pixels[i, j]         = (0, 0, 0)       # Black
                    pixels[i, j + 1]     = (0, 0, 0)       # Black
                    pixels[i + 1, j]     = (0, 0, 0)       # Black
                    pixels[i + 1, j + 1] = (0, 0, 0)       # Black
        
        # Return new image
        return new
        
    def get_plot_predict(
        self,
        paint_df,
        hometeam,
        awayteam,
        leaguename,
        bookmaker,
        bet,
        betcoef,
        datename,
        proba,
        show_pictures=False,
        clear_pic_path=True,
        
    ):
    
        if clear_pic_path:

            import shutil, os
            try:
                shutil.rmtree(self.path_pictures_pred)
            except:
                pass

            try:
                os.mkdir(self.path_pictures_pred)
            except:
                pass

        paint_df=paint_df.copy()
        paint_df=paint_df[paint_df[datename] < paint_df[datename].min()+timedelta(5)]
        paint_df=paint_df[paint_df[bookmaker].isnull()==False]
        paint_df.reset_index(inplace=True)
        paint_df['Time'] = paint_df[datename].dt.strftime('%d/%m/%y %H:%M')
        paint_df['Title1'] = paint_df['Time'] + '  ' + paint_df[leaguename].replace('_', ' ', regex=True)

        for r in range(0,paint_df.index.max(),3):
            image = self.open_image(self.path_pictures)
            image = self.convert_grayscale(image)
            ImageDraw.Draw(image).rectangle([30,30,500,500], outline = 'rgb(235, 188, 78)')

            draw = ImageDraw.Draw(image)
            start_w = 120
            start_h = 90
            font = ImageFont.truetype(self.path_fonts+'Roboto-Bold.ttf', size=40)
            (x, y) = (170, start_h)
            message = 'FREE BET'
            color = 'rgb(255, 255, 255)'
            draw.text((x, y), message, fill=color, font=font)

            font = ImageFont.truetype(self.path_fonts+'17643.otf', size=9)
            (x, y) = (160, start_h+50)
            message = '@footballbets.tv'
            color = 'rgb(255, 255, 255)'
            draw.text((x, y), message, fill=color, font=font)

            inc = 0
            save_event = []
            teams = []
            while inc <=2:
                step = 90
                size = 20
                try:
                    title = paint_df.loc[r]['Title1']
                    times = paint_df.loc[r]['Time'][:2]+'_'+paint_df.loc[r]['Time'][3:5]+'_'+paint_df.loc[r]['Time'][6:8]
                    w = font.getsize(title[:15])[0]
                    font = ImageFont.truetype(self.path_fonts+'17425.otf', size=size)
                    (x, y) = (start_w, start_h+100+step*inc)
                    color = 'rgb(235, 188, 78)'
                    draw.text((x, y), title, fill=color, font=font)

                    event = paint_df.loc[r][hometeam] + ' - ' + paint_df.loc[r][awayteam]
                    teams.append(paint_df.loc[r][hometeam])
                    teams.append(paint_df.loc[r][awayteam])
                    save_event.append(event)
                    target = paint_df.loc[r][bet]
                    coeff = ' x'+str(paint_df.loc[r][betcoef]) + ' ' + str(paint_df.loc[r][bookmaker])
                    font = ImageFont.truetype(self.path_fonts+'17425.otf', size=size)
                    # (x, y) = ((image.size[0]/2)-(font.getsize(event)[0]/2)-10, start_h+120+step*inc)
                    (x, y) = (start_w, start_h+122+step*inc)

                    draw.text((x, y), event, fill='rgb(255, 255, 255)', font=font)
                    draw.text((start_w, start_h+120+step*inc+25), target, fill='rgb(177, 228, 227)', font=font)
                    draw.text(
                        (
                            start_w + font.getsize(target)[0],
                            start_h+120+step*inc+25
                        ), 
                        coeff,
                        fill='rgb(235, 188, 78)',
                        font=font
                    )

                    inc+=1
                    r+=1
                except:
                    inc+=1
                    r+=1
                    pass
            if show_pictures:
                display(image)
            if inc>0:
                # save
                # save_image(image, self.path = './Pictures/prediction_pics/'+times+' '+', '.join(save_event)+'.jpeg')
                image.save(self.path_pictures_pred+times+' '+', '.join(save_event)+'.jpg')
                with open(self.path_pictures_pred+times+' '+', '.join(save_event)+".txt","w") as a:#write mode 
                    a.write('Free bets with @footballbets.tv ''\n#'+' #'.join([x for t in teams for x in t.split(' ')]) + '#Football #Bets #Recommendations #stavki')