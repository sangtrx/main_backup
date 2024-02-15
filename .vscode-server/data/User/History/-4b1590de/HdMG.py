import telegram
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import trio
import asyncio
# Set up Telegram bot
bot = telegram.Bot(token='6945926819:AAG6fTkfxVIzwffeU-WItM9QDr-8LAtb78o')

update_queue = asyncio.Queue()

# Set up Google Sheets API credentials
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('/home/tqsang/nifty-vault-293616-ba427e12aec4.json', scope)
client = gspread.authorize(creds)

# Open the Google Sheet
sheet = client.open_by_key('11gpBY3FGT3Ugu4VtUDwce4EMwjbvcxhxxqqq_S_i7So').sheet1

# Define a function to handle incoming Telegram messages
def handle_message(update, context):
    # Get the text of the incoming message
    text = update.message.text
    
    # Split the text into separate cells
    cells = text.split(',')
    
    # Add the cells to a new row in the Google Sheet
    sheet.append_row(cells)
    
    # Send a confirmation message back to the user
    update.message.reply_text('Added to Google Sheet!')

# Set up the Telegram bot to listen for incoming messages
updater = telegram.ext.Updater(bot=bot, update_queue=update_queue)
dispatcher = updater.dispatcher
dispatcher.add_handler(telegram.ext.MessageHandler(telegram.ext.Filters.text, handle_message))

# Start the Telegram bot
updater.start_polling()