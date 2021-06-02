#encoding: utf-8
import sys
import importlib
import numpy
from pandas import Series, DataFrame
import pandas as pd
import torch
import re
import nltk
import nltk.tokenize as tk
import nltk.stem as ns
from nltk.corpus import stopwords
importlib.reload(sys)
sys.path.append("../")
import config
def get_origin_word_from_sentence(sentence):
    pattren = re.compile("[^a-zA-Z0-9\n]") #数字字母的正则匹配

    #分词    
    Tokenization = tk.word_tokenize(sentence) 

    #标准化
    Normallization = []
    for word in Tokenization:
        # 将所有非数字字母字符全部转换为空格,并统一小写化
        word = re.sub(pattren," ",word).lower()
        word = tk.word_tokenize(word) #再次分词
        #去停用词
        word = [w for w in word if (w not in stopwords.words("english"))]
        Normallization += word

    #词干提取
    Stemming = []
    for word in Normallization:
        pt_stem = nltk.stem.porter.PorterStemmer().stem(word)
        Stemming.append(word)

    #词性标注
    word_pos = nltk.pos_tag(Stemming)
    
    #词性还原
    Lemmatization = []
    lemmatizer = ns.WordNetLemmatizer()
    for item in word_pos:
        word = item[0]
        tag = item[1][0].lower()
        if tag == 'j':
            tag = 'a'
        elif tag == 'r' or tag == 'v':
            tag = tag
        else:
            tag = 'n'
        Lemmatization.append(lemmatizer.lemmatize(word,tag))
    return Lemmatization

def Make_dictionary(source_file,to_file):
    source = pd.read_csv(source_file)
    lines = len(source)
    word_set = set()  #集合去重

    for i in range(lines):
        name = source["FullName"][i]
        if name == "nan": #数据缺失
            name = ""
        words = get_origin_word_from_sentence(str(name))
        for word in words:
            word_set.add(word)

    with open(to_file, 'w', encoding = "utf-8") as fout:
        for word in word_set:
            fout.write(word+"\n")

def Read_dict_from_txt(source_file):
    dict = {}
    index = 0
    with open(source_file, 'r', encoding = "utf-8") as fin:
        content = fin.readlines()
        for i in range(len(content)):
            dict[content[i][:-1]] = i#去除换行符
    return dict

def cosine_similarity(vector_a,vector_b):
    vector_a = vector_a.detach().numpy()[0]
    vector_b = vector_b.detach().numpy()[0]
    dimension = len(vector_a)
    sum = 0
    mod_a = 0
    mod_b = 0
    for i in range(dimension):
        sum += vector_a[i]*vector_b[i]
        mod_a += vector_a[i]**2
        mod_b += vector_b[i]**2
    return sum/(numpy.sqrt(mod_a*mod_b))

if __name__ == '__main__':
    # 生成 Journal_dict.txt 词汇表
    #Make_dictionary(config.JOURNAL_FILE,config.JOURNAL_DICT)
    # 生成 Conference_dict.txt 词汇表
    #Make_dictionary(config.CONFERENCE_FILE,config.CONFERENCE_DICT)
    x = "IEEE Transactions on Dependable and Secure Computing"
    y = "IEEE Transactions on Evolutionary Computation"
    z = "Journal of The Ais"
    dimension = 10000
    x = get_origin_word_from_sentence(x)
    y = get_origin_word_from_sentence(y)
    z = get_origin_word_from_sentence(z)
    journal_dict = Read_dict_from_txt(config.JOURNAL_DICT)
    embeds = torch.nn.Embedding(len(journal_dict),dimension)

    word_vector_x = torch.zeros(1,dimension)
    for word in x:
        tmp = embeds(torch.LongTensor([journal_dict[word]]))
        word_vector_x.add_(tmp)
    word_vector_x /= len(x)


    word_vector_y = torch.zeros(1,dimension)
    for word in y:
        tmp = embeds(torch.LongTensor([journal_dict[word]]))
        word_vector_y.add_(tmp)
    word_vector_y /= len(y)


    word_vector_z = torch.zeros(1,dimension)
    for word in z:
        tmp = embeds(torch.LongTensor([journal_dict[word]]))
        word_vector_z.add_(tmp)
    word_vector_z /= len(z)

    print(cosine_similarity(word_vector_x,word_vector_y))
    print(cosine_similarity(word_vector_x,word_vector_z))
    print(cosine_similarity(word_vector_y,word_vector_z))

