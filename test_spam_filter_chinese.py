import os
import io
import joblib
import jieba
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.data.path.append("/home/geo/Downloads/Trade/text-mining-bot/ntlk_data")
# nltk.download('stopwords',download_dir="/home/geo/Downloads/Trade/text-mining-bot/ntlk_data")

paths = '/home/geo/Downloads/Trade/text-mining-bot/ntlk_data/trec06p'

# 定义读取垃圾邮件和非垃圾邮件的函数

def prepare(str):
    str1 , str2 = str.split(" ")
    return str1 , str2.replace("..",paths)

str = []
data = []
label = []
path = os.path.join(paths,'full')
for filename in os.listdir(path):
    file_path = os.path.join(path, filename)
    with open(file_path, "r", encoding="latin-1") as file:
        str.append(file.readlines())

str = pd.DataFrame(str[0],columns=['text'])

df = str['text'].apply(lambda x : prepare(x))

for file_path in df :
    with io.open(file_path[1].strip(),'r',encoding='latin-1') as file :
        data.append(file.read())
        label.append(file_path[0])

df = pd.DataFrame({'text':data,'label':label})
print(df.head())

seg_data = []
for text in data:
    seg_text = ' '.join(jieba.cut(text))  # 使用jieba进行分词处理
    seg_data.append(seg_text)

# 特征提取器
vectorizer = CountVectorizer()

# 将文本转换为特征向量
X = vectorizer.fit_transform(seg_data)

model_file = 'model/spam_classifier_chinese_model.pkl'
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



if __name__ == '__main__':

    # 测试数据
    test_data = [ 
        "Congratulations! You've won a free vacation.",
        "Hi John, can you send me the report?",
        "Get cheap watches and handbags in our online store!",
        "Don't miss the investment chance!",
        "请把握这次的投资机会!",
        "我是诈骗",
        "恭喜您！您赢得了免费假期。",
        "嗨约翰，能给我发报告吗？",
        "在我们的在线商店中购买廉价手表和手提包！"
    ]

    # 分词处理测试数据
    seg_test_data = []
    for text in test_data:
        seg_text = ' '.join(jieba.cut(text))  # 使用jieba进行分词处理
        seg_test_data.append(seg_text)

    # 将测试数据转换为特征向量
    X_test = vectorizer.transform(seg_test_data)
    predict_results = model.predict(X_test)

    # 输出预测结果
    for text, prediction in zip(test_data, predict_results):
        if prediction == 1:
            print(f'{text} : 是垃圾邮件。')
        else:
            print(f'{text} : 不是垃圾邮件。')
