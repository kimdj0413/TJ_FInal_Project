import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_datasets as tfds
import pickle
import numpy as np
import string

##      질문 구분 데이터 로드
lstm_model = load_model('C:/MachineLearning/Financial_chatbot/Model/bidirectional_LSTM.h5')
lstm_tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('C:/MachineLearning/Financial_chatbot/LSTM/Data/tokenizer')

##      사전 데이터 로드
with open('C:/MachineLearning/Financial_chatbot/LSTM/Data/keyword_dict.pkl', 'rb') as file:
    keyword_dict = pickle.load(file)
keyword_list = list(keyword_dict.keys())

##      챗봇 데이터 로드
seq2seq_model = load_model('D:/TJ_FInal_Project/KDJ/Financial_Chatbot/Model/chatbot.h5')
seq2seq_tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('D:/TJ_FInal_Project/KDJ/Financial_Chatbot/Sequence2Sequence/Data/Final_Tokenizer')
START_TOKEN, END_TOKEN = [seq2seq_tokenizer.vocab_size], [seq2seq_tokenizer.vocab_size+1]
pre_token = -1
prepre_token = -1

while True:
    question = input('나 : ')
    if question == "끝!":
        break
    if len(question) > 50:
        print('질문이 너무 깁니다! 조금만 줄여주세요!')
    else:
        tokenied_question=[]
        tokenied_question.append(lstm_tokenizer.encode(question))
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
                print('죄송합니다. 잘 모르겠습니다.')
        else:                   ##      지식인 질문이 들어왔을때
            question_list = []
            result_list = []
            final_sentence = ''

            question_list.append(seq2seq_tokenizer.encode(question))
            tokenized_input_sentence = tf.keras.preprocessing.sequence.pad_sequences(
                question_list, maxlen=59, padding='post'
            )

            temparary_decoder_input = [START_TOKEN[0],]

            for i in range(0,142):
                pred = seq2seq_model.predict([tokenized_input_sentence, np.array(temparary_decoder_input).reshape(1, -1)],verbose=0)
                last_pred = pred[:, -1, :]
                sorted_indices = np.argsort(last_pred, axis=-1)[:, ::-1]
                second_max_index = sorted_indices[:, 0]
                next_token = second_max_index[0]

                if next_token >= 16260:
                    break

                temparary_decoder_input.append(next_token)
                
                if pre_token != next_token and prepre_token != next_token:
                    result_list.append(seq2seq_tokenizer.decode([next_token]))

                prepre_token = pre_token
                pre_token = next_token
            final_sentence = ''.join(result_list)

            special_characters = string.punctuation
            final_sentence = final_sentence.lstrip(special_characters)

            if not final_sentence.endswith(('.', '!')):
                final_sentence += '.'

            print(f'봇 : {final_sentence}')
