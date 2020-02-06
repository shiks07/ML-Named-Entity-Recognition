import numpy as np
import sys

#unpacking the command line arguments
train_input,word_index,tag_index,hmmprior,hmmemit,hmmtrans = sys.argv[1:]

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

def init_probs(train_tags,index_tag):
    init_probs = np.zeros(len(index_tag))
    for tag in index_tag.values():
        for seq in train_tags:
            if seq[0] == tag:
                init_probs[tag] += 1
        init_probs[tag] += 1
    denom = sum(init_probs)
    pi = np.log(init_probs/denom)
    return np.exp(pi)

def trans_probs(train_tags,total_tags):
    trans_probs = np.zeros((total_tags,total_tags))
    for seq in train_tags:
        for i in range(len(seq)):
            if i < len(seq)-1:
                from_tag = seq[i]
                to_tag = seq[i+1]
                trans_probs[from_tag,to_tag] += 1
    trans_probs += 1
    denom = np.sum(trans_probs,axis = 1).reshape((-1,1))
    return trans_probs/denom

def emit_probs(train_data,total_tags,total_words):
    emit_probs = np.zeros((total_tags,total_words))
    for seq in train_data:
        for tag,word in seq:
            emit_probs[tag,word] += 1
    emit_probs += 1
    total_count = np.sum(emit_probs,axis = 1).reshape((-1,1))
    return emit_probs/total_count
        

def training(train_input,word_index,tag_index):
    tag_to_index,index_to_tag = read_index(tag_index)
    word_to_index,index_to_word = read_index(word_index)
    train_data, train_tags,train_words =read_data(train_input,word_to_index,tag_to_index)
    total_tags = len(tag_to_index)
    total_words = len(word_to_index)
    pi = init_probs(train_tags,tag_to_index)
    A = trans_probs(train_tags,total_tags)
    B = emit_probs(train_data,total_tags,total_words)
    return pi,A,B


pi,A,B = training(train_input,word_index,tag_index)
np.savetxt(hmmtrans,A)
np.savetxt(hmmemit,B)
np.savetxt(hmmprior,pi)


# def write_params(param,filename):
#     f = open(filename,"a+")
#     for i in range(param.shape[0]):
#         if i != 0:
#             f.write('\n')
#         for j in range(len(param[i])):
# #             if j < len(param[i])-1:
# #                 output = str(param[i][j])+' '
# #             else:
#             output = str(param[i][j])+' '
#             f.write(output)   
#     f.close()

    



# for i in range(len(pi)):
#     f = open(hmmprior,"a+")
#     if i == 0:
#         f.write(str(pi[i]))
#     else:
#         f.write('\n')
#         f.write(str(pi[i]))
#     f.close()


# write_params(B,hmmemit)
# write_params(A,hmmtrans)

    