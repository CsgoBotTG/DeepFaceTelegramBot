import sys
import TelegramBotStart


def main():
    token = None
    try:
        index = sys.argv.index('-token') + 1
        token = sys.argv[index]
    except:
        token = '6637485467:AAFmS9mSSgTQDf8ZrbQQPapJ4neoCAPzBoo'
    
    info_bot = TelegramBotStart.get_address_bot(token=token)

    print(f"Starting bot {info_bot['first_name']} with token {token}. https://t.me/{info_bot['username']} | @{info_bot['username']}")
    print(f"Bot starting")
    
    TelegramBotStart.TelegramBotStart(token=token)


if __name__ == '__main__':
    main()