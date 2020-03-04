import csv
from random import randrange
from utils import author2vec,issue2vec,text2vec,combine_vec,get_vocab
import numpy as np
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

def cv_LR(kfold):
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
    print('Cross validation for Logistic Regression')
    n_sample,n_feature = np.shape(data)
    fold_size = int(np.ceil(n_sample/kfold))
    print('Fold size:',fold_size)
    accuracy = []
    for k in range(kfold):
        tstart = k*fold_size
        tend = min(n_sample,tstart+fold_size)
        training_x = np.array([x for i,x in enumerate(data) if not (tstart<=i and i<tend)])
        test_x = np.array([x for i,x in enumerate(data) if (tstart<=i and i<tend)])
        training_y = np.array([x for i,x in enumerate(labels) if not (tstart<=i and i<tend)])
        test_y = np.array([x for i,x in enumerate(labels) if (tstart<=i and i<tend)])
        model = LogRegression(n_feature,n_class,lrate=0.8,verbose=False)
        model.fit(training_x,training_y,max_iter=500)
        accuracy.append(model.score(test_x,test_y))
        print('Fold',k,'accuracy',accuracy[-1])
    print('Mean accuracy',np.mean(accuracy))


def cv_NN(kfold):
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
    print('Cross validation for Neural network')
    n_sample,n_feature = np.shape(data)
    fold_size = int(np.ceil(n_sample/kfold))
    print('Fold size:',fold_size)
    accuracy = []
    for k in range(kfold):
        tstart = k*fold_size
        tend = min(n_sample,tstart+fold_size)
        training_x = np.array([x for i,x in enumerate(data) if not (tstart<=i and i<tend)])
        test_x = np.array([x for i,x in enumerate(data) if (tstart<=i and i<tend)])
        training_y = np.array([x for i,x in enumerate(labels) if not (tstart<=i and i<tend)])
        test_y = np.array([x for i,x in enumerate(labels) if (tstart<=i and i<tend)])
        model = NeuralNet(n_feature,n_class,lrate=0.9,verbose=False)
        model.fit(training_x,training_y,max_iter=500)
        accuracy.append(model.score(test_x,test_y))
        print('Fold',k,'accuracy',accuracy[-1])
    print('Mean accuracy',np.mean(accuracy))
if __name__ == '__main__':
    np.random.seed(0)
    k = 5
    # cv_LR(k)
    np.random.seed(0)
    cv_NN(k)