TOKEN = ''

with open('.env', 'r') as tokens:
    TOKENS = tokens.readlines()
    TOKEN = TOKENS[0].rsplit()

