import telebot
import gspread
import importlib

# Import the telebot module dynamically
telebot = importlib.import_module("telebot")


# Replace the following values with your own
TELEGRAM_BOT_TOKEN = "6945926819:AAG6fTkfxVIzwffeU-WItM9QDr-8LAtb78o"
SPREADSHEET_ID = "11gpBY3FGT3Ugu4VtUDwce4EMwjbvcxhxxqqq_S_i7So"

# Create a Telegram bot object
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# Create a Google Sheets service object
gc = gspread.service_account()
sh = gc.open_by_key(SPREADSHEET_ID)
worksheet = sh.get_worksheet(0)

# Define a function to handle incoming messages
def handle_message(message):
    # Get the text message
    text = message.text

    # Split the text message into a list of cells
    cells = text.split(",")

    # Create a new row in the Google Sheet
    row = worksheet.row_count + 1
    worksheet.update_row(row, cells)

    # Send a confirmation message to the user
    bot.send_message(message.chat.id, "Your message has been added to the Google Sheet.")

# Register the message handler
bot.message_handler(func=handle_message)

# Start the bot
bot.start_polling()