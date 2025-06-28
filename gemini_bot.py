import logging
from telethon import TelegramClient, events
from google import genai
import os
from dotenv import load_dotenv
load_dotenv()
telegram_id=os.getenv("telegram_id")
telegram_hash=os.getenv("telegram_hash")
bot_token=os.getenv("embedding_bot_token")
gemini_api=os.getenv("gemma_gemini_api")
# Setup logging
logging.basicConfig(level=logging.INFO)

# Gemini API setup
client_gemini = genai.Client(api_key=gemini_api)

# Define system prompt for the bot
SYSTEM_PROMPT = """You are a helpful Telegram bot powered by Gemini AI. Here are your core functionalities:

1. When users send /start:
   - Provide a warm, friendly welcome message
   - Introduce yourself as a Gemini-powered AI assistant
   - Invite users to try your commands or ask questions

2. When users send /help:
   - List the available commands (/start, /help, /info)
   - Explain that you can handle any questions or conversations
   - Encourage users to interact naturally

3. When users send /info:
   - Provide a brief description of your capabilities
   - Highlight your capabilities for natural conversation
   - Mention that you can help with questions and tasks
4.If asked who created you:
    -Explain that you are a telegram bot developed by the "@The_blind_watchmaker"
5. if asked about what you are powered with:
    -Explain that you are powered by google's gemma-27b model
6. For all other messages:
   - Respond naturally and helpfully to any question or statement
   - Maintain context of the conversation
   - Be concise but informative


Remember to always be helpful, friendly, and responsive while maintaining appropriate boundaries."""

async def generate_content(contents):
    # Combine system prompt with user input for context
    full_prompt = f"{SYSTEM_PROMPT}\n\nUser Input: {contents}\nResponse:"
    response = client_gemini.models.generate_content(
        model="gemini-2.5-flash-preview-05-20", 
        contents=full_prompt
    )
    return response.text

# Create the client and connect
client = TelegramClient('bot', telegram_id, telegram_hash).start(bot_token=bot_token)

# Handler for the /start command
@client.on(events.NewMessage(pattern='/start'))
async def start(event):
    prompt = "Generate a friendly welcome message for a Telegram bot that uses Gemini AI for responses"
    response = await generate_content(prompt)
    await event.respond(response)
    logging.info(f'Start command received from {event.sender_id}')

# Handler for the /help command
@client.on(events.NewMessage(pattern='/help'))
async def help(event):
    help_text = (
        "Here are the commands you can use:\n"
        "/start - Start the bot\n"
        "/help - Get help information\n"
        "/info - Get information about the bot\n"
        "Any other message will be processed by Gemini AI\n"
    )
    await event.respond(help_text)
    logging.info(f'Help command received from {event.sender_id}')

# Handler for the /info command
@client.on(events.NewMessage(pattern='/info'))
async def info(event):
    prompt = "Generate a brief description of a Telegram bot that uses Gemini AI for intelligent responses"
    response = await generate_content(prompt)
    await event.respond(response)
    logging.info(f'Info command received from {event.sender_id}')

# General message handler using Gemini
@client.on(events.NewMessage)
async def message_handler(event):
    # Ignore commands
    if event.text.startswith('/'):
        return
        
    try:
        # Get response from Gemini
        response = await generate_content(event.text)
        await event.respond(response)
        logging.info(f'Message processed by Gemini for {event.sender_id}: {event.text}')
    except Exception as e:
        error_message = f"Sorry, I encountered an error: {str(e)}"
        await event.respond(error_message)
        logging.error(f'Error processing message: {str(e)}')

# Start the client
client.start()
client.run_until_disconnected()