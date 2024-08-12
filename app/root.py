import asyncio
from aiogram import F,Router,Dispatcher
from aiogram.filters import CommandStart,Command
from aiogram.types import Message
# from aiogram import Bot, Dispatcher
from datetime import datetime

#import models
import app.components.keyboards as kb 
from app.lstm.price_forecast import price_forecast


router = Router()

assets = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc. (Class A)',
    'AMZN': 'Amazon.com Inc.',
    'TSLA': 'Tesla Inc.',
    'META': 'Meta Platforms Inc. (formerly Facebook)',
    'NFLX': 'Netflix Inc.',
    'NKE': 'Nike Inc.',
    'NVDA': 'NVIDIA Corporation',
    'BABA': 'Alibaba Group Holding Ltd.',
    '^GSPC': 'S&P 500',
    '^DJI': 'Dow Jones Industrial Average',
    '^IXIC': 'NASDAQ Composite',
    '^RUT': 'Russell 2000',
    '^FTSE': 'FTSE 100 (London Stock Exchange Index)',
    '^DAX': 'DAX (German Stock Index)',
    'BTC-USD': 'Bitcoin (BTC) to USD',
    'ETH-USD': 'Ethereum (ETH) to USD',
    'ADA-USD': 'Cardano (ADA) to USD',
    'SOL-USD': 'Solana (SOL) to USD',
    'DOGE-USD': 'Dogecoin (DOGE) to USD',
    'XRP-USD': 'Ripple (XRP) to USD',
    'GC=F': 'Gold Futures',
    'CL=F': 'Crude Oil Futures',
    'SI=F': 'Silver Futures',
    '^IRX': '3-Month Treasury Bill',
    '^TNX': '10-Year Treasury Note Yield',
}


HELP_info = """
*Акции:*
- AAPL — Apple Inc.
- MSFT — Microsoft Corporation
- GOOGL — Alphabet Inc. (Class A)
- AMZN — Amazon.com Inc.
- TSLA — Tesla Inc.
- META — Meta Platforms Inc. (ранее Facebook)
- NFLX — Netflix Inc.
- NKE — Nike Inc.
- NVDA — NVIDIA Corporation
- BABA — Alibaba Group Holding Ltd.

*Индексы:*
- ^GSPC — S&P 500
- ^DJI — Dow Jones Industrial Average
- ^IXIC — NASDAQ Composite
- ^RUT — Russell 2000
- ^FTSE — FTSE 100 (Лондонский фондовый индекс)
- ^DAX — DAX (Немецкий фондовый индекс)

*Криптовалюты:*
- BTC-USD — Bitcoin (BTC) к USD
- ETH-USD — Ethereum (ETH) к USD
- ADA-USD — Cardano (ADA) к USD
- SOL-USD — Solana (SOL) к USD
- DOGE-USD — Dogecoin (DOGE) к USD
- XRP-USD — Ripple (XRP) к USD

*Товары:*
- GC=F — Gold Futures
- CL=F — Crude Oil Futures
- SI=F — Silver Futures

*Облигации и другие активы:*
- ^IRX — 3-Month Treasury Bill
- ^TNX — 10-Year Treasury Note Yield
"""

HELP_COMMANDS = """
Я могу предсказать цены на акции, индексы, товары, криптовалюту, облигации.
*набор команд*
<b>/start</b> - <em>старт бот</em> 
"<b>/forecast \"ваш выбор\"</b> - <em>укажите актив после команды. Например:  <code>/forecast TSLA</code></em>"
<b>/info</b> - <em>список что можно запросить</em> 
<b>/help</b> - <em>набор команд в бот </em>
"""

# command START !
@router.message(CommandStart())
async def start(message:Message):
    await message.answer(f"Привет! мой друг {message.from_user.first_name.capitalize()}")
    await message.answer_sticker("CAACAgIAAxkBAAEtE2Bmt1mNl2lmUWGBN3EsEfcaKWgmgwACAQEAAladvQoivp8OuMLmNDUE") 
    await message.answer(text=HELP_COMMANDS, parse_mode='HTML')
    
# work with PHOTO
@router.message(F.photo)
async def get_photo(message:Message):
    #  тут получаю id photo  await message.answer(f"ID photo {message.photo[-1].file_id}")
    await message.answer_photo(photo = message.photo[-1].file_id , caption="я не умею работать с фото")
    await message.answer_sticker("CAACAgIAAxkBAAEtE35mt14lwa_JN0vcH5jReyRhAt4uUAACAgEAAladvQpO4myBy0Dk_zUE") 

# command HELP !
@router.message(Command('help'))
async def get_help(message:Message):
    await message.answer("команда /help ")
    await message.answer(text=HELP_COMMANDS, parse_mode='HTML')
    
# command INFO !
@router.message(Command('info'))
async def get_help(message:Message):
    await message.answer(text=HELP_info, parse_mode='HTML')
# Идентификатор стикера

# command FORECAST !
@router.message(Command('forecast'))
async def get_forecast(message: Message):
    # await message.answer("Привет!")
    # command_text = message.text
    
    command_text = message.text.strip()
    # Разделяем команду и аргументы
    parts = command_text.split(maxsplit=1)

    if len(parts) == 1:
        await message.answer("Пожалуйста, укажите актив после команды. Например: /forecast TSLA")
        await message.answer("Узнать подробнее про активы поможет команда: /info")
    
    elif len(parts) > 2:
        await message.answer("Пожалуйста, укажите только один актив после команды. Например: /forecast TSLA")
        await message.answer_sticker("CAACAgIAAxkBAAEtG3VmuchGeLypB21QdA2u4GFjGvNo6QACCwEAAladvQpOseemCPvtSTUE") 
        await message.answer("Узнать подробнее про активы поможет команда: /info")
    
    elif len(parts) == 2:
        # Пользователь ввёл команду и один аргумент
        _, asset = parts
        now = datetime.now()
        formatted_date = now.strftime('%Y-%m-%d')

        def search(text):
            # Поиск описания актива в словаре
            return assets.get(text, False)

        descriptions = search(asset)
        
        if descriptions:
            await message.answer(f"Вы указали: {asset.upper()}: {descriptions}")    
            await message.answer_sticker("CAACAgIAAxkBAAEtG21mucdTbNDXp0ZlgwZSQmDGoOqUOQACIQMAApzW5wofM3WHdLVAUzUE") 
        
            df, results_df = price_forecast('crypto', asset, formatted_date)
            # print(results_df)
            results_str = results_df[['Date', 'Predicted_Close']].to_string(index=False, header=True)
        
            await message.answer(f'Прогноз цен на следующие 7 дней:\n\n{results_str}')
        else:
            await message.answer(f"Пока нет такого актива в списках: {asset}")
            await message.answer_sticker("CAACAgIAAxkBAAEtG3FmucgFQiuU5BOhwee44XU3ObX-yAACAgEAAladvQpO4myBy0Dk_zUE") 
            await message.answer("Узнать подробнее про активы поможет команда: /info")