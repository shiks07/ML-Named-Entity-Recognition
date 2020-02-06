import numpy as np
import sys 

# unpacking the command line arguments

input_data,word_index,tag_index,hmmprior,hmmemit,hmmtrans,predicted_file,metric_file = sys.argv[1:]

def read_data(filename,word_to_index,tag_to_index):
    f = open(filename,'r')
    data,data_tags,data_words = [],[],[]
    for line in f:
        seq,tags,words = [],[],[]
        for x in line.strip().split(" "):
            word,tag = x.split("_")
            seq.append((tag_to_index[tag],word_to_index[word]))
            tags.append(tag_to_index[tag])
            words.append(word_to_index[word])
        data.append(seq)
        data_tags.append(tags)
        data_words.append(words)
    return data,data_tags,data_words

def read_index(filename):
    str_to_index = tuple((x.strip(),i) for i,x in enumerate(open(filename,'r'),0))
    index_to_str = tuple((i,x.strip()) for i,x in enumerate(open(filename,'r'),0))
    return dict(str_to_index), dict(index_to_str)

def viterbi(sentence,pi,A,B):
    lw = np.zeros((A.shape[0],len(sentence)))
    p = np.zeros((A.shape[0],len(sentence)),dtype = int)
    for t in range(len(sentence)):
        if t == 0:
            lw[:,t] = np.log(pi) + np.log(B[:,sentence[t]])
            p[:,t] = range(A.shape[0])
        else:
            for j in range(lw.shape[0]):
                lw[j,t] = np.max(np.log(B[j,sentence[t]])+ np.log(A[:,j]) + lw[:,t-1])
                p[j,t] = np.argmax(np.log(B[j,sentence[t]])+ np.log(A[:,j]) + lw[:,t-1])
    return lw,p
                
def backtrack(lw,p):
    yhat = np.zeros((lw.shape[1]),dtype = int)
    yhat[-1] = np.argmax(lw[:,-1])
    for t in range(len(yhat)-1,-1,-1):
        yhat[t-1] = p[yhat[t],t]
    return yhat
            

def prediction(input_data,word_index,tag_index,pi,A,B):
    tag_to_index,index_to_tag = read_index(tag_index)
    word_to_index,index_to_word = read_index(word_index)
    all_data, data_tags,data_words =read_data(input_data,word_to_index,tag_to_index)
    predicted_tags = []
    for sentence in data_words:
        lw,p = viterbi(sentence,pi,A,B)
        yhat = backtrack(lw,p)
        predicted_tags.append(yhat)
    return data_words,data_tags,predicted_tags,index_to_tag,index_to_word
        
def format_out(data,predicted_tags,index_to_tag,index_to_word):
    out = []
    for i in range(len(data)):
        line = []
        for j in range(len(data[i])):
            line.append(index_to_word[data[i][j]]+'_'+index_to_tag[predicted_tags[i][j]])
        out.append(line)
    return out


def write_data(output,filename):
    f = open(filename,'a+')
    for i in range(len(output)):
        if i != 0:
            f.write('\n')
        for j in range(len(output[i])):
            if j < len(output[i])-1:     
                f.write(output[i][j]+' ')
            else:
                f.write(output[i][j])


def read_param(filename):
    f = open(filename,'r')
    lst = []
    for line in f:
        lst.append(line.strip().split())
    return np.array(lst,dtype = np.float64)

def read_prior(filename):
    f = open(filename,'r')
    lst = []
    for line in f:
        lst.append(line.strip())
    return np.array(lst,dtype = np.float64)

def accuracy(data_tags,predicted_tags):
    total_tags,correct = 0,0
    for i in range(len(data_tags)):
        total_tags += len(data_tags[i])
        for j in range(len(data_tags[i])):
            if data_tags[i][j] == predicted_tags[i][j]:
                correct += 1
    return correct/total_tags

        
               
B = read_param(hmmemit)
A = read_param(hmmtrans)
pi = read_prior(hmmprior)

data_words,data_tags,predicted_tags,index_to_tag,index_to_word = prediction(input_data,word_index,tag_index,pi,A,B)
formatted_out = format_out(data_words,predicted_tags,index_to_tag,index_to_word)
accuracy = accuracy(data_tags,predicted_tags)
write_data(formatted_out,predicted_file)
open(metric_file,'w').write('Accuracy: '+str(accuracy))


