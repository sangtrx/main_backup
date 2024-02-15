from instabot import Bot
bot = Bot()
# Instagram credentials
username = 'xsquared.lab'
password = 'Sangdeptraivl123456789*'
bot.login(username=username, password=password)
followers = bot.get_user_followers("ngoknghekvailiz")
following = bot.get_user_following("ngoknghekvailiz")
not_following_back = [user for user in following if user not in followers]
print(not_following_back)