import nltk
import string
import re

#Tweet List
tweets = []

#Get Trump Tweets
trump = open('realdonaldtrump.csv', 'r')

for line in trump:
    line = line.lower()
    #remove all links
    line = re.sub(r'https?:\/\/[^\s]*([\b\s]+|$)', '', line)
    line = re.sub(r'pic\.twitter\.com[^\s]*[\b\s]+', '', line)
    #remove coninuation
    line = line.replace('(cont)', '')
    words = nltk.word_tokenize(line)
    mention = False
    hashtag = False
    for word in list(words):
        #if it's a mention, we add the @ so we know
        if mention:
            words[words.index(word)] = '@' + word
            mention = False
        #if it's a hashtag, we remove it
        elif hashtag:
            words.remove(word)
            hashtag = False
        #check if the next word in the list is a mention
        elif word[0] == '@':
            mention = True
            words.remove(word)
        #check if the next word in the list is a hastag
        elif word[0] == '#':
            hashtag = True
            words.remove(word)
        #convert & to and
        elif word == '&':
            words[words.index(word)] = 'and'
        #remove any other punctuation unless it's a contraction
        else:
            noPunc = word.strip(string.punctuation)
            if noPunc == '':
                words.remove(word)
            elif noPunc == 's' or noPunc == 'm' or noPunc == 're' or noPunc == 've' or noPunc == 'd' or noPunc =='ll':
                break
            else:
                words[words.index(word)] = noPunc
    if len(line) > 0:
        tweets.append((line, 'Trump'))

#Get Biden Tweets
biden = open("JoeBidenTweets.csv", "r")

for line in biden:
    line = line.lower()
    #remove all links
    line = re.sub(r'https?:\/\/[^\s]*([\b\s]+|$)', '', line)
    line = re.sub(r'pic\.twitter\.com[^\s]*[\b\s]', '', line)
    #remove coninuation
    line = line.replace('(cont)', '')
    words = nltk.word_tokenize(line)
    mention = False
    hashtag = False
    for word in list(words):
        #if it's a mention, we add the @ so we know
        if mention:
            words[words.index(word)] = '@' + word
            mention = False
        #if it's a hashtag, we remove it
        elif hashtag:
            words.remove(word)
            hashtag = False
        #check if the next word in the list is a mention
        elif word[0] == '@':
            mention = True
            words.remove(word)
        #check if the next word in the list is a hastag
        elif word[0] == '#':
            hashtag = True
            words.remove(word)
        #convert & to and
        elif word == '&':
            words[words.index(word)] = 'and'
        #remove any other punctuation unless it's a contraction
        else:
            noPunc = word.strip(string.punctuation)
            if noPunc == '':
                words.remove(word)
            elif noPunc == 's' or noPunc == 'm' or noPunc == 're' or noPunc == 've' or noPunc == 'd' or noPunc =='ll':
                break
            else:
                words[words.index(word)] = noPunc
    if len(line) > 0:
        tweets.append((line, 'Biden'))
print(tweets)
