from instabot import Bot

# Create a new instance of the InstaBot
bot = Bot()
# Instagram credentials
username = 'xsquared.lab'
password = 'Sangdeptraivl123456789*'
# Login to Instagram
bot.login(username=username, password=password)

# Get a list of users who don't follow you back
non_followers = bot.get_non_followers()

# Print the usernames of the non-followers
for user in non_followers:
    print(user)

# Logout from Instagram
bot.logout()
