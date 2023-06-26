import os
import io
import joblib
import jieba
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from loguru import logger 
logger.add('log/error.log',level="INFO", rotation="10 MB", format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}")

class spam_filter:
    '''
    for English or Chinese only
    '''
    def __init__(self) :
        nltk.data.path.append("/home/geo/Downloads/Trade/text-mining-bot/ntlk_data")
        # nltk.download('stopwords',download_dir="/home/geo/Downloads/Trade/text-mining-bot/ntlk_data")
        self.model_file = 'model/spam_classifier_model.pkl'
        self.model_file_chinese = 'model/spam_classifier_chinese_model.pkl'
        self.data = []
        self.label = []

    def load_data(self):
        span_path = '/home/geo/Downloads/Trade/text-mining-bot/ntlk_data/spam'
        ham_path = '/home/geo/Downloads/Trade/text-mining-bot/ntlk_data/easy_ham'

        spam_files = [os.path.join(span_path, file) for file in os.listdir(span_path)]
        ham_files = [os.path.join(ham_path, file) for file in os.listdir(ham_path)]
    
        # 读取垃圾邮件
        for file_path in spam_files:
            with io.open(file_path, 'r', encoding='latin-1') as file:
                self.data.append(file.read())
                self.label.append(1)

        # 读取非垃圾邮件
        for file_path in ham_files:
            with io.open(file_path, 'r', encoding='latin-1') as file:
                self.data.append(file.read())
                self.label.append(0)

        #return pd.DataFrame({'text':data,'label':label})

    def prepare(self,str):
        str1 , str2 = str.split(" ")
        return str1 , str2.replace("..", self.paths)

    def load_data_chinese(self):
        self.paths = '/home/geo/Downloads/Trade/text-mining-bot/ntlk_data/trec06p'
        str = []
        path = os.path.join(self.paths,'full')
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            with open(file_path, "r", encoding="latin-1") as file:
                str.append(file.readlines())

        str = pd.DataFrame(str[0],columns=['text'])
        df = str['text'].apply(lambda x : self.prepare(x))

        for file_path in df :
            with io.open(file_path[1].strip(),'r',encoding='latin-1') as file :
                self.data.append(file.read())
                self.label.append(file_path[0])
        
        # df = pd.DataFrame({'text':self.data,'label':self.label})
        # print(df.head())

    def load_model_chinese(self,X=None):
        try :
            model = joblib.load(self.model_file_chinese)
            print(f'Pretrained model loaded.')
            return model

        except FileNotFoundError:
            print(f'Pretrained model not found. Initializing with None.')
            # 创建并训练朴素贝叶斯分类器
            model = MultinomialNB()
            model.fit(X, self.label)
            joblib.dump(model, self.model_file_chinese)
            return model
        
        except Exception as e :
            logger.error(f'{e}')
    
    def load_model(self,X=None):
        try :
            model = joblib.load(self.model_file)
            print(f'Pretrained model loaded.')
            return model

        except FileNotFoundError:
            print(f'Pretrained model not found. Initializing with None.')
            # 创建并训练朴素贝叶斯分类器
            model = MultinomialNB()
            model.fit(X, self.label)
            joblib.dump(model, self.model_file)
            print(f'Model saved.')
            return model
        
        except Exception as e :
            logger.error(f'{e}')

    def main(self,test_data):
        self.load_data()
        
        # 将测试数据转换为特征向量
        vectorizer = CountVectorizer(stop_words='english')

        # 将文本转换为特征向量
        X = vectorizer.fit_transform(self.data)
        self.model = self.load_model(X)

        X_test = vectorizer.transform(test_data)
        predict_results = self.model.predict(X_test)

        # 输出预测结果
        for text, prediction in zip(test_data, predict_results):
            if prediction == 1:
                print(f'{text} : 是垃圾。')
            else:
                print(f'{text} : 不是垃圾。')

    def main_chinese(self,test_data,seg_test_data):
        self.load_data_chinese()

        seg_data = []
        for text in self.data:
            seg_text = ' '.join(jieba.cut(text))  # 使用jieba进行分词处理
            seg_data.append(seg_text)
        
        # 特征提取器
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(seg_data)
        self.model = self.load_model_chinese(X)
        
        X_test = vectorizer.transform(seg_test_data)
        predict_results = self.model.predict(X_test)

        # 输出预测结果
        for text, prediction in zip(test_data, predict_results):
            if prediction == 1:
                print(f'{text} : 是垃圾。')
            else:
                print(f'{text} : 不是垃圾。')

if __name__ == '__main__':

    check = spam_filter()
    # 测试数据
    test_data = [ 
        "Congratulations! You've won a free vacation.",
        "Hi John, can you send me the report?",
        "Get cheap watches and handbags in our online store!",
        "Don't miss the investment chance!",
        "Do you know Binance ? Come to join us!",
        "Do you know Binance ? Come to join us for gain the good profit!"
    ]

    check.main(test_data)

    # # 测试数据
    # test_data = [ 
    #     # "Congratulations! You've won a free vacation.",
    #     # "Hi John, can you send me the report?",
    #     # "Get cheap watches and handbags in our online store!",
    #     # "Don't miss the investment chance!",
    #     "请把握这次的投资机会!",
    #     "我是诈骗",
    #     "恭喜您！您赢得了免费假期。",
    #     "嗨约翰，能给我发报告吗？",
    #     "在我们的在线商店中购买廉价手表和手提包！"
    # ]

    # # 分词处理测试数据
    # seg_test_data = []
    # for text in test_data:
    #     seg_text = ' '.join(jieba.cut(text))  # 使用jieba进行分词处理
    #     seg_test_data.append(seg_text)

    # check.main_chinese(test_data,seg_test_data)
    
