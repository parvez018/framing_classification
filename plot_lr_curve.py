import csv
from random import randrange
from utils import author2vec,issue2vec,text2vec,combine_vec,get_vocab
import numpy as np
import matplotlib.pyplot as plt

from SimpleLogRegression import LogRegression
from SimpleNeuralNet import NeuralNet


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


def SaveFile(tweet_id2text, tweet_id2issue, tweet_id2author_label, tweet_id2label, output_csv_file):

    with open(output_csv_file, mode='w') as out_csv:
        writer = csv.writer(out_csv, delimiter=',', quoting=csv.QUOTE_NONE)
        writer.writerow(["tweet_id", "issue", "text", "author", "label"])
        for tweet_id in tweet_id2text:
            writer.writerow([tweet_id, tweet_id2issue[tweet_id], tweet_id2text[tweet_id], tweet_id2author_label[tweet_id], tweet_id2label[tweet_id]])

def plot_lr(labels,y,fname):
    line_prop = ['-or','-xb','-vg','-ok','-oc']
    labels = ['Learning rate: '+str(l) for l in labels]
    for i in range(0,len(labels)):
        plt.plot(y[i], line_prop[i],label=labels[i],markersize=3)
    plt.grid()
    plt.legend()
    plt.xlabel("#epochs")
    plt.ylabel("% Wrong classification")
    pdfname = fname
    plt.savefig(pdfname, format = "pdf")


def LR():

    # Read training data
    train_tweet_id2text, train_tweet_id2issue, train_tweet_id2author_label, train_tweet_id2label = ReadFile('train.csv')

    '''
    Implement your Logistic Regression classifier here
    '''
    BOW = True
    GLOVE = False
    word2index = get_vocab(train_tweet_id2text)
    data_dict = combine_vec(word2index,train_tweet_id2text,train_tweet_id2author_label,train_tweet_id2issue,bow=True)

    data = []
    labels = []
    for k in data_dict:
        data.append(data_dict[k])
        labels.append(int(train_tweet_id2label[k]))
    n_class = len(set(labels))
    print(np.shape(data))
    # print(np.shape(labels))
    
    # print(len(word2index))
    n_sample,n_feature = np.shape(data)
    lrates = [0.2,0.5,0.8]
    all_loss = []
    for r in lrates:
        model = LogRegression(n_feature,n_class,lrate=r,verbose=True)
        train_loss = model.fit(data,labels,max_iter=200)
        print(len(train_loss))
        all_loss.append(train_loss)
    file_name = 'train_loss_lr.pdf'
    plot_lr(lrates,all_loss,file_name)

def NN():

    # Read training data
    train_tweet_id2text, train_tweet_id2issue, train_tweet_id2author_label, train_tweet_id2label = ReadFile('train.csv')

    '''
    Implement your Neural Network classifier here
    '''
    word2index = get_vocab(train_tweet_id2text)
    data_dict = combine_vec(word2index,train_tweet_id2text,train_tweet_id2author_label,train_tweet_id2issue, bow = False)

    data = []
    labels = []
    for k in data_dict:
        data.append(data_dict[k])
        labels.append(int(train_tweet_id2label[k]))
    data = np.array(data)
    labels = np.array(labels)
    n_class = len(set(labels))
    n_sample,n_feature = np.shape(data)
    print(np.shape(data))

    lrates = [0.2,0.4,0.9]
    all_loss = []
    for r in lrates:
        model = LogRegression(n_feature,n_class,lrate=r,verbose=True)
        train_loss = model.fit(data,labels,max_iter=500)
        print(len(train_loss))
        all_loss.append(train_loss)
    file_name = 'train_loss_nn.pdf'
    plot_lr(lrates,all_loss,file_name)

if __name__ == '__main__':
    np.random.seed(0)
    LR()
    NN()