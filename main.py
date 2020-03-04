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


def SaveFile(tweet_id2text, tweet_id2issue, tweet_id2author_label, tweet_id2label, output_csv_file):

    with open(output_csv_file, mode='w') as out_csv:
        writer = csv.writer(out_csv, delimiter=',', quoting=csv.QUOTE_NONE)
        writer.writerow(["tweet_id", "issue", "text", "author", "label"])
        for tweet_id in tweet_id2text:
            writer.writerow([tweet_id, tweet_id2issue[tweet_id], tweet_id2text[tweet_id], tweet_id2author_label[tweet_id], tweet_id2label[tweet_id]])

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
    print('dataset shape:',np.shape(data))
    # print(np.shape(labels))
    
    # print(len(word2index))
    n_sample,n_feature = np.shape(data)

    model = LogRegression(n_feature,n_class,lrate=0.8,verbose=True)
    model.fit(data,labels,max_iter=500)
    # y_pred = [model.predict(x) for x in data]

    # Read test data
    test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label = ReadFile('test.csv')
    
    # Predict test data by learned model

    '''
    Replace the following random predictor by your prediction function.
    '''
    test_data_dict = combine_vec(word2index,test_tweet_id2text,test_tweet_id2author_label,test_tweet_id2issue,bow = True)

    for tweet_id in test_tweet_id2text:
        # Get the text
        # text=test_tweet_id2text[tweet_id]
        
        # Predict the label
        test_x = test_data_dict[tweet_id]
        label = model.predict(test_x)

        # Store it in the dictionary
        test_tweet_id2label[tweet_id] = label

    # Save predicted labels in 'test_lr.csv'
    SaveFile(test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label, 'test_lr.csv')


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

    model = NeuralNet(n_feature,n_class,lrate=0.9,verbose=True)
    model.fit(data,labels,max_iter=800)

    # Read test data
    test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label = ReadFile('test.csv')
    '''
    Replace the following random predictor by your prediction function.
    '''
    test_data_dict = combine_vec(word2index,test_tweet_id2text,test_tweet_id2author_label,test_tweet_id2issue,bow = False)

    for tweet_id in test_tweet_id2text:
        # Predict the label
        test_x = test_data_dict[tweet_id]
        label = model.predict(test_x)

        # Store it in the dictionary
        test_tweet_id2label[tweet_id] = label

    # Save predicted labels in 'test_lr.csv'
    SaveFile(test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label, 'test_nn.csv')

if __name__ == '__main__':
    np.random.seed(0)
    LR()
    np.random.seed(0)
    NN()