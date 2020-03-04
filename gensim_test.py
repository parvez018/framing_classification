import gensim.downloader as api
import csv
from random import randrange
from utils import author2vec,issue2vec,text2vec,combine_vec,get_vocab
import numpy as np
from SimpleLogRegression import LogRegression
from SimpleNeuralNet import NeuralNet
import pickle




def ReadFile(input_csv_file):
    # Reads input_csv_file and returns four dictionaries tweet_id2text, tweet_id2issue, tweet_id2author_label, tweet_id2label

    tweet_id2text = {}
    tweet_id2issue = {}
    tweet_id2author_label = {}
    tweet_id2label = {}
    f = open(input_csv_file, "r")
    csv_reader = csv.reader(f)
    row_count=-1
    for row in csv_reader:
        row_count+=1
        if row_count==0:
            continue

        tweet_id = int(row[0])
        issue = str(row[1])
        text = str(row[2])
        author_label = str(row[3])
        label = row[4]
        tweet_id2text[tweet_id] = text
        tweet_id2issue[tweet_id] = issue
        tweet_id2author_label[tweet_id] = author_label
        tweet_id2label[tweet_id] = label

    #print("Read", row_count, "data points...")
    return tweet_id2text, tweet_id2issue, tweet_id2author_label, tweet_id2label
# info = api.info()  # show info about available models/datasets
# print(info)
# glove_embedding = api.load("glove-twitter-25")


def save_dict_to_file(dic,fname):
    f = open(fname,'w')
    f.write(str(dic))
    f.close()

def load_dict_from_file(fname):
    f = open(fname,'r')
    data=f.read()
    f.close()
    return eval(data)

# words = ['obama','trump']
# wdic = {}
# for w in words:
#     wdic[w] = glove_embedding[w]
# print(wdic)
train_tweet_id2text, train_tweet_id2issue, train_tweet_id2author_label, train_tweet_id2label = ReadFile('train.csv')

word2index = get_vocab(train_tweet_id2text)
# word2glove = {}
# for w in word2index:
#     if w in glove_embedding:
#         word2glove[w] = glove_embedding[w]

em_file = 'glove25_embeddings.txt'
# save_dict_to_file(word2glove,em_file)
# loaded_glove = load_dict_from_file(em_file)


# with open(em_file, 'wb') as handle:
#   pickle.dump(word2glove, handle)

with open(em_file, 'rb') as handle:
  loaded_glove = pickle.loads(handle.read())

for w in word2glove:
    print(word2glove[w])
    print(loaded_glove[w])
    break
print(len(loaded_glove))
print(word2glove == loaded_glove) # True