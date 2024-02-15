import os
import time
import csv
from instabot import Bot
import re
import random

# Instagram credentials
username = 'xsquared.lab'
password = 'Sangdeptraivl123456789*'

# Input files and directories
image_dir = "/mnt/tqsang/x2_lab/insta"

def extract_caption_and_celebrity(filename):
    caption = re.sub(r'^\d+-', '', filename.split('.')[0])
    celebrity = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', caption)
    celebrity_hashtag = '#' + celebrity[0].replace(' ', '') if celebrity else ''
    
    return caption, celebrity_hashtag

def post_image(bot, image_dir, hashtags):
    images = [filename for filename in os.listdir(image_dir) if filename.endswith(".jpg")]
    random.shuffle(images)

    for filename in images:
        caption, celebrity_hashtag = extract_caption_and_celebrity(filename)
        final_caption = f"{caption}\n\n\n\n\n\n{hashtags} {celebrity_hashtag}"
        image_path = os.path.join(image_dir, filename)
        upload_result = bot.upload_photo(image_path, caption=final_caption)
        
        if upload_result:
            # Randomly decide whether to share the post as a story (50% chance)
            if random.random() < 0.3:
                bot.story_share(upload_result['media']['pk'])

            return # Post only one image and return

def main():
    # Initialize bot and login
    bot = Bot()
    bot.login(username=username, password=password)

    while True:
        hashtags = "#AI #artificialintelligence #machinelearning #deeplearning #AIimage #AIart #AIphotography #AIportrait #AIchallenge #AImeme #AIquotes #AIinfluencer #explorepage #explore #meme #trending"

        post_image(bot, image_dir, hashtags)

        # Wait 1 hour before posting the next image
        time.sleep(2 * 60 * 60)

if __name__ == "__main__":
    main()
