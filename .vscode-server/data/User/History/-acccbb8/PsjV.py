from instabot import Bot
import os

# Create a new instance of the InstaBot
bot = Bot()
# Instagram credentials
username = 'xsquared.lab'
password = 'Sangdeptraivl123456789*'
# Login to Instagram
bot.login(username=username, password=password)

# Get the lists of your followers and people you follow
followers = bot.get_user_followers('ngoknghekvailiz')
following = bot.get_user_following('ngoknghekvailiz')

# Find the users who don't follow you back
non_followers = [user for user in following if user not in followers]

# Print the usernames of the non-followers
# Print the usernames of the non-followers
for user in non_followers:
    user_info = bot.get_user_info(user)
    print(user_info['username'])

# Delete the config file
os.remove('/home/tqsang/x2_lab/abnormal/config')

# Logout from Instagram
bot.logout()
