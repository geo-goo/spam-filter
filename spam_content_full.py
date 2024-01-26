#importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import string , re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

nltk.download('stopwords' , download_dir="/home/geo/Downloads/geo/text_mining_bot/ntlk_data")
nltk.download('wordnet' ,   download_dir="/home/geo/Downloads/geo/text_mining_bot/ntlk_data")
file_path = '/home/geo/Downloads/geo/text_mining_bot/case-spam-and-data-visualtion/spam.csv'

class spam_collection :
    '''
    spam collection.
    '''
    data = []
    label = []

if __name__ == '__main__':
    
    df = pd.read_csv(file_path,encoding='latin-1')
    df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
    df.rename(columns={'v1':'label','v2':'text'},inplace=True)
    df['label'] = np.where(df['label']=='ham',0,1)

    # 分割数据为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=1)

    # 文本向量化
    vectorizer = CountVectorizer()
    X_train_counts = vectorizer.fit_transform(X_train)

    # 使用朴素贝叶斯分类器
    clf = MultinomialNB()
    clf.fit(X_train_counts, y_train)

    # 测试模型
    X_test_counts = vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_counts)

    # 输出准确率
    print(df.head()) 
    print(f"accuracy score : {accuracy_score(y_test, y_pred)}")
       

