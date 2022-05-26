#!/usr/bin/env python
# coding: utf-8

# ## 文本分类器

# In[1]:


import pandas  as pd
from pandas import DataFrame
import re
import math
import json
import jieba
from gensim import corpora,models,similarities
from sklearn.utils import shuffle as reset
import numpy


# In[2]:


df_white = pd.read_excel('./域名白名单汇总.xlsx',usecols=['dn'])
df_black = pd.read_excel('./域名黑名单汇总.xlsx',usecols=['dn'])
whitednlist = df_white['dn'].values.tolist()
blackdnlist = df_black['dn'].values.tolist()


# In[3]:


with open("./岩浆地球化学/google_search_py_yj.json",'rb') as load_f:
     load_dict = json.load(load_f)
datayj = load_dict['RECORDS']
with open("./沉积学黑白名单/google_search_py_cj.json",'rb') as load_f:
     load_dict = json.load(load_f)
datacj = load_dict['RECORDS']
with open("./地貌学黑白名单/google_search_py_dm.json",'rb') as load_f:
     load_dict = json.load(load_f)
datadm = load_dict['RECORDS']
data = datayj + datacj + datadm


# In[4]:


whiteurllist,blackurllist = [],[]
blackword=['.pdf','/paper','/article','baidu','bilibili','sohu','weibo','sina','youtube','google','taobao','douban','youdao','baike','wiki','pedia','researchgate',
             'amazon','twitter','qq','tecent','patent','worldwidescience','zhuanli','163','rhhz','d-nb','zhihu','zhidao','csdn','jianshu',
             'baijiahao','zhuanlan','blog','statesurveys','bartoc','jstor','Cited by','被引用','数据库',' OR ','…']
for i in data:
    matchObj = i['url_path'].split(' › ')
    if matchObj[0] in whitednlist:
        cnt = 0
        for j in blackword:
            if re.search(j, i['url'], re.I) is None:
                cnt = cnt + 1
        if cnt == len(blackword):
            whiteurllist.append(i['url'])
        else:
            blackurllist.append(i['url'])
    if matchObj[0] in blackdnlist:
        blackurllist.append(i['url'])
whiteurllist=list(set(whiteurllist))
blackurllist=list(set(blackurllist))


# In[5]:


datadic_test = {'url':[], 'title':[], 'desc':[], 'label':[]}
for i in data:
    if i['url'] in whiteurllist:
        datadic_test['url'].append(i['url'])
        datadic_test['title'].append(i['title'])
        datadic_test['desc'].append(i['desc'])
        datadic_test['label'].append(1)
    elif i['url'] in blackurllist:
        datadic_test['url'].append(i['url'])
        datadic_test['title'].append(i['title'])
        datadic_test['desc'].append(i['desc'])
        datadic_test['label'].append(0)
data_test = pd.DataFrame(datadic_test)


# In[6]:


def drop_stopwords(contents, stopwords):
    contents_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(str(word))
        contents_clean.append(line_clean)
    return contents_clean,all_words


# In[7]:


content1 = data_test.desc.values.tolist()
content2 = data_test.title.values.tolist()
label = data_test.label.values.tolist()
content = []
for i in range(len(content1)):
    content.append(content1[i].strip()+content2[i])

content_S = []
for line in content:
    line = line.strip()
    line = re.sub(r'[^\w\s]','',line)
    current_seg = jieba.lcut(line)
    for i in range(current_seg.count(' ')):
        current_seg.remove(' ')
    for i in current_seg:
        if i == '\xa0':
            current_seg.remove(i)
    content_S.append(current_seg)
df_content=pd.DataFrame({'content_S':content_S})
stopwords=pd.read_csv("CNENstopwords.txt", index_col=False,sep='\t',quoting=3,names=['stopwords'],encoding='utf-8')
contents0 = df_content.content_S.values.tolist()
stopwords0 = stopwords.stopwords.values.tolist()
contents_clean,all_words = drop_stopwords(contents0,stopwords0)
df_content_test = pd.DataFrame({'contents_clean':contents_clean,'label':label})
df_test_x = df_content_test['contents_clean'].values
df_test_y = df_content_test['label'].values
df_test_words = []
for line_index in range(len(df_test_x)):
    try:
        df_test_words.append(' '.join(df_test_x[line_index]))
    except:
        print(line_index,word_index)


# In[8]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_content_test['contents_clean'].values, df_content_test['label'].values, test_size=0.3, random_state=1)


# In[9]:


y_train[:20]


# In[10]:


words = []
for line_index in range(len(x_train)):
    try:
        words.append(' '.join(x_train[line_index]))
    except:
        print(line_index,word_index)
test_words = []
for line_index in range(len(x_test)):
    try:
        test_words.append(' '.join(x_test[line_index]))
    except:
        print(line_index,word_index)


# In[13]:


from sklearn.feature_extraction.text import CountVectorizer
vec_count = CountVectorizer(analyzer='word', max_features=3000, lowercase=False)
vec_count.fit(words)
from sklearn.naive_bayes import MultinomialNB
classifier_count = MultinomialNB()
classifier_count.fit(vec_count.transform(words), y_train)
classifier_count.score(vec_count.transform(test_words), y_test)


# In[15]:

from sklearn.feature_extraction.text import TfidfVectorizer
vec_tfidf = TfidfVectorizer(analyzer='word', max_features=3000, lowercase=False)
vec_tfidf.fit(words)
from sklearn.naive_bayes import MultinomialNB
classifier_tfidf = MultinomialNB()
classifier_tfidf.fit(vec_tfidf.transform(words), y_train)
classifier_tfidf.score(vec_tfidf.transform(test_words), y_test)


# 逻辑回归
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
logitmodel=LogisticRegression(max_iter=10000)#定义回归模型
logitmodel.fit(vec_count.transform(words),y_train)#训练模型
logitmodel.score(vec_count.transform(test_words), y_test)


