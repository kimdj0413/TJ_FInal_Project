import pandas as pd
import string
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Masking, Bidirectional, Add, BatchNormalization, Layer, Concatenate, Attention, LayerNormalization

from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint

model_path = 'D:/TJ_FInal_Project/KDJ/Financial_Chatbot/Model/chatbot.h5'

tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('D:/TJ_FInal_Project/KDJ/Financial_Chatbot/Sequence2Sequence/Data/Final_Tokenizer')
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size+1]

QUE_MAX_LENGTH = 48
ANS_MAX_LENGTH = 138

model = load_model(model_path)
pre_token = -1
prepre_token = -1

print('################################################')
print('## 채팅을 끝내시려면 "끝!" 이라고 입력해주세요! ##')
print('################################################')

while True:
    question = []
    result_list = []
    final_sentence = ''
    input_sentence = input('나 : ')
    if input_sentence == '끝!':
        break
    if len(input_sentence) > 50:
        print('질문이 너무 깁니다! 조금만 줄여주세요!')
    else:
        question.append(tokenizer.encode(input_sentence))
        tokenized_input_sentence = tf.keras.preprocessing.sequence.pad_sequences(
            question, maxlen=QUE_MAX_LENGTH, padding='post'
        )

        temparary_decoder_input = [START_TOKEN[0],]

        for i in range(0,ANS_MAX_LENGTH):
            pred = model.predict([tokenized_input_sentence, np.array(temparary_decoder_input).reshape(1, -1)],verbose=0)
            last_pred = pred[:, -1, :]
            sorted_indices = np.argsort(last_pred, axis=-1)[:, ::-1]
            second_max_index = sorted_indices[:, 0]
            next_token = second_max_index[0]

            if next_token >= 16260:
                break

            temparary_decoder_input.append(next_token)
            
            if pre_token != next_token and prepre_token != next_token:
                result_list.append(tokenizer.decode([next_token]))

            prepre_token = pre_token
            pre_token = next_token
        final_sentence = ''.join(result_list)

        special_characters = string.punctuation
        final_sentence = final_sentence.lstrip(special_characters)

        if not final_sentence.endswith(('.', '!')):
            final_sentence += '.'

        print(f'봇 : {final_sentence}')