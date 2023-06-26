import os
import io
import joblib
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.data.path.append("/home/geo/Downloads/Trade/text-mining-bot/ntlk_data")
# nltk.download('stopwords',download_dir="/home/geo/Downloads/Trade/text-mining-bot/ntlk_data")

span_path = '/home/geo/Downloads/Trade/text-mining-bot/ntlk_data/spam'
ham_path = '/home/geo/Downloads/Trade/text-mining-bot/ntlk_data/easy_ham'

spam_files = [os.path.join(span_path, file) for file in os.listdir(span_path)]
ham_files = [os.path.join(ham_path, file) for file in os.listdir(ham_path)]

data = []
label = []

# 读取垃圾邮件
for file_path in spam_files:
    with io.open(file_path, 'r', encoding='latin-1') as file:
        data.append(file.read())
        label.append(1)

# 读取非垃圾邮件
for file_path in ham_files:
    with io.open(file_path, 'r', encoding='latin-1') as file:
        data.append(file.read())
        label.append(0)

# df = pd.DataFrame({'text':data,'label':label})

# 特征提取器
vectorizer = CountVectorizer(stop_words='english')

# 将文本转换为特征向量
X = vectorizer.fit_transform(data)

model_file = 'spam_classifier_model.pkl'
try :
    model = joblib.load(model_file)
    print(f'Pretrained model loaded.')

except FileNotFoundError:
    print(f'Pretrained model not found. Initializing with None.')
    # 创建并训练朴素贝叶斯分类器
    model = MultinomialNB()
    model.fit(X, label)
    joblib.dump(model, model_file)

except Exception as e :
    print(f'{e}')

# 测试数据
test_data = [ 
    "Congratulations! You've won a free vacation.",
    "Hi John, can you send me the report?",
    "Get cheap watches and handbags in our online store!",
    "Don't miss the investment chance!",
    "请把握这次的投资机会!",
    "我是诈骗"
]

# 将测试数据转换为特征向量
X_test = vectorizer.transform(test_data)

if __name__ == '__main__':
    predict_results = model.predict(X_test)

    # 输出预测结果
    for text, prediction in zip(test_data, predict_results):
        if prediction == 1:
            print(f'{text} : 是垃圾邮件。')
        else:
            print(f'{text} : 不是垃圾邮件。')
