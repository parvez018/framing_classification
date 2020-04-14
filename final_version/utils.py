import numpy  as np
import nltk
from nltk.tokenize import TweetTokenizer
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
set(stopwords.words('english'))
import pickle

def tokenize(sentences):
    words = []
    for sentence in sentences:
        w = word_extraction(sentence)
        words.extend(w)
        words = sorted(list(set(words)))
    return words

def text2vec(word2index,text_dict,bow=True):
    ######################################################
    em_file = 'glove25_embeddings.txt'
    glove_embedding = None
    with open(em_file, 'rb') as handle:
        glove_embedding = pickle.loads(handle.read())
    #######################################################
    retval = {}
    ret_gem = {}
    tw2tok = TweetTokenizer(preserve_case=False, reduce_len=False, strip_handles=False)
    total_word = len(word2index)
    pstemmer = PorterStemmer()
    for t in text_dict:
        content = text_dict[t]
        words = tw2tok.tokenize(content)
        words = [pstemmer.stem(w) for w in words]
        bow_vector = np.zeros(total_word)
        glove_vector = []
        for w in words:
            if w in word2index:
                bow_vector[word2index[w]] += 1
            else:
                pass
            if w in glove_embedding:
                glove_vector.append(glove_embedding[w])
        ret_gem[t] = np.mean(np.array(glove_vector),axis=0)
        # print('ret_gem',len(ret_gem))
        # total_word_in_sent = len(words)*1.0
        # bow_vector = np.array([w/total_word_in_sent for w in bow_vector])
        # print(np.sum(bow_vector))
        retval[t] = bow_vector
    if bow:
        return retval
    else:
        return ret_gem


def author2vec(author_dict):
    retval = {}
    possible_values = set([author_dict[k] for k in author_dict])
    # print(possible_values)
    num_vals = len(possible_values)
    val2num = {}
    for i,k in enumerate(possible_values):
        val2num[k]=i
    # print(val2num)
    for k in author_dict:
        retval[k] = [val2num[author_dict[k]]]
    return retval

def issue2vec(issue_dict):
    retval = {}
    possible_values = set([issue_dict[k] for k in issue_dict])
    # print(possible_values)
    num_vals = len(possible_values)
    val2num = {}
    for i,k in enumerate(possible_values):
        val2num[k]=i
    # print(val2num)
    for k in issue_dict:
        one_hot = np.zeros(num_vals)
        one_hot[val2num[issue_dict[k]]-1] = 1
        retval[k] = one_hot
    return retval

def get_vocab(text_dict):
    all_words = []
    stop_words = set(stopwords.words('english'))
    tw2tok = TweetTokenizer(preserve_case=False, reduce_len=False, strip_handles=False)
    for t in text_dict:
        content = text_dict[t]
        words = tw2tok.tokenize(content)
        all_words.extend(words)
    pstemmer = PorterStemmer()
    all_words  = [pstemmer.stem(w) for w in all_words if w not in stop_words]
    all_words = sorted(set(all_words))
    word2index = {}
    for i,w in enumerate(all_words):
        word2index[w] = i
    return word2index

def combine_vec(word2index,text,author,issue,bow = False):
    avec = author2vec(author)
    ivec = issue2vec(issue)
    tvec = text2vec(word2index,text,bow=bow)
    retval = {}
    for k in avec:
        retval[k] = np.concatenate([avec[k],ivec[k],tvec[k]])
        # print(retval[k].shape)
    return retval