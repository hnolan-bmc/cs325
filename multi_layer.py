"""
File name: multi_layer.py
Project by: Haley Nolan and Umme Tanjuma Haque
Purpose: Implements Multi-Layer Perceptron
Date: 12/18/2020
"""
import nltk
import string
import re
import random
from collections import Counter
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Embedding, Flatten, Dropout, LSTM
from datetime import datetime

#------preprocessing starts-----#
#intializing the default dictionary of parts of speech with frequencies set as 0
pos_dict = {'$': 0, 'CC': 0, 'CD': 0, 'DT': 0, 'EX': 0, 'FW': 0, 'IN': 0,
            'JJ': 0, 'JJR': 0, 'JJS': 0, 'LS': 0, 'MD': 0, 'NN': 0,
            'NNP': 0, 'NNPS': 0, 'NNS': 0, 'PDT': 0, 'POS': 0, 'PRP': 0,
            'PRP$': 0, 'RB': 0, 'RBR': 0, 'RBS': 0, 'RP': 0, 'SYM': 0,
            'TO': 0, 'UH': 0, 'VB': 0, 'VBD': 0, 'VBG': 0, 'VBN': 0,
            'VBP': 0, 'VBZ': 0, 'WDT': 0, 'WP': 0, 'WP$': 0, 'WRB': 0}

#Method to convert POS counts to string
def convert_pos(counts):
    tup = (counts['$'], counts['CC'], counts['CD'], counts['DT'], counts['EX'],
           counts['FW'], counts['IN'], counts['JJ'], counts['JJR'],
           counts['JJS'], counts['LS'], counts['MD'], counts['NN'],
           counts['NNP'], counts['NNPS'], counts['NNS'], counts['PDT'],
           counts['POS'], counts['PRP'], counts['PRP$'], counts['RB'],
           counts['RBR'], counts['RBS'], counts['RP'], counts['SYM'],
           counts['TO'], counts['UH'], counts['VB'], counts['VBD'],
           counts['VBG'], counts['VBN'], counts['VBP'], counts['VBZ'],
           counts['WDT'], counts['WP'], counts['WP$'], counts['WRB'])
    return np.asarray(tup)

#Tweet List
tweets = []

#Get Biden Tweets
biden = open("JoeBidenTweets.csv", "r")

#iterates through the file
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
        #pos tagger is run to attach part-of-speech tag to each word
        tagged = nltk.pos_tag(words)
        #counts is a list of all frequencies of pos in the line
        counts = (Counter(tag for word,tag in tagged)).most_common()
        count_dict = pos_dict
        for count in counts:
            count_dict[count[0]] = count[1]
        tweets.append((line, 0, convert_pos(count_dict)))

num_biden = len(tweets)
num_trump = 0

#Get Trump Tweets
trump = open('realdonaldtrump.csv', 'r')

#iterates through the file
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
        #pos tagger is run to attach part-of-speech tag to each word
        tagged = nltk.pos_tag(words)
        #counts is a list of all frequencies of pos in the line
        counts = (Counter(tag for word,tag in tagged)).most_common()
        count_dict = pos_dict
        for count in counts:
            count_dict[count[0]] = count[1]
        tweets.append((line, 1, convert_pos(count_dict)))
    num_trump += 1
    if num_trump >= num_biden:
        break
#------preprocessing ends-----#

#Input Setup
random.seed(datetime.now())
random.shuffle(tweets, ) #mix them up
split = (int)(round((len(tweets)*0.7)))
x_train = []
y_train = []
x_test = []
y_test = []

for i in range(len(tweets)):
    if i < split:
        x_train.append(tweets[i][2])
        y_train.append(tweets[i][1])
    else:
        x_test.append(tweets[i][2])
        y_test.append(tweets[i][1])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
#Model Creation
epochs = 20 #or 30

model = Sequential()

# model.add(Dense(5, activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(3, activation='relu'))
# model.add(Dense(2, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

for x in range(100, 0, -10):
    model.add(Dense(x, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


history = model.compile(loss='binary_crossentropy',
    optimizer = SGD(lr = 0.01),
    metrics=['accuracy'])
print('Train...')
history = model.fit(x_train, y_train,
    batch_size=128,
    epochs=epochs,
    validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test, batch_size=128)
print('Test score:', score)
print('Test accuracy:', acc)

#code for plotting graphs
val_loss = history.history['val_loss']
loss = history.history['loss']
dic = {}
for i in range(1, epochs + 1):
    dic[i] = [loss[i - 1], val_loss[i - 1]]
data = pd.DataFrame.from_dict(dic, orient='index')
g = sns.relplot(data=data, kind="line", palette=['#a6cee3','#1f78b4'])
g.set_axis_labels('Epochs', 'Loss')
g.fig.suptitle('Model Loss')
new_labels = ['loss', 'val_loss']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)
plt.tight_layout()
plt.savefig('multilayer_loss.png')

val_acc = history.history['val_accuracy']
acc = history.history['accuracy']
dic = {}
for i in range(1, epochs + 1):
    dic[i] = [acc[i - 1], val_acc[i - 1]]
data = pd.DataFrame.from_dict(dic, orient='index')
g = sns.relplot(data=data, kind="line", palette=['#b2df8a','#33a02c'])
g.set_axis_labels('Epochs', 'Accuracy')
g.fig.suptitle('Model Accuracy')
new_labels = ['accuracy', 'val_accuracy']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)
plt.tight_layout()
plt.savefig('multilayer_accuracy.png')
