import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_datasets as tfds
import pickle

##      LSTM 관련 데이터 로드
lstm_model = load_model('C:/MachineLearning/Financial_chatbot/Model/bidirectional_LSTM.h5')
tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('C:/MachineLearning/Financial_chatbot/LSTM/Data/tokenizer')

##      사전 데이터 로드
with open('C:/MachineLearning/Financial_chatbot/LSTM/Data/keyword_dict.pkl', 'rb') as file:
    keyword_dict = pickle.load(file)
keyword_list = list(keyword_dict.keys())

while True:
    question = input('나 : ')
    if question == "끝!":
        break
    tokenied_question=[]
    tokenied_question.append(tokenizer.encode(question))
    final_question = tf.keras.preprocessing.sequence.pad_sequences(
        tokenied_question, maxlen=77, padding='post'
    )
    score = float(lstm_model.predict(final_question))
    print(f'나 : {question}')
    
    if(score > 0.5):        ##      사전 질문이 들어왔을때
        try :
            question = question.replace(" ", "")
            included_keywords = [keyword for keyword in keyword_list if keyword in question]
            print(included_keywords)
            if len(included_keywords) > 1:
                keyword = max(included_keywords, key=len)
            else:
                keyword = included_keywords[0]
            print(f"봇 : '{keyword}'의 뜻을 알려드릴께요\n{keyword}(이)란 '{keyword_dict.get(keyword)}'을(를) 뜻합니다.")
        except ValueError:
            print('몰라')
    else:                   ##      지식인 질문이 들어왔을때
        print('<JSK>')
