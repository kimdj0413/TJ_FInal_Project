# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from transformers import AutoTokenizer, TFGPT2LMHeadModel
import re
from tqdm import tqdm

df = pd.read_csv("D:/TJ_FInal_Project/KDJ/News_Summarization/Data/문서요약 텍스트/Preprocess/train_preprocess_test.csv")

sentence_max_len = 160
abs_max_len = 30

df = df[df['sentence'].apply(lambda x: len(x.split()) <= sentence_max_len)]
df = df[df['abs'].apply(lambda x: len(x.split()) <= abs_max_len)]

tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', pad_token='<pad>')
model = TFGPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', from_pt=True)

def tokenize_df():
    # 질문, 레이블 응답 문장을 불러옵니다.
    for question, response in zip(df['sentence'].to_list(), df['abs'].to_list()):
        # 문장의 BOS token : </s>
        bos_token = [tokenizer.bos_token_id]
        # 문장의 EOS token : </s>
        eos_token = [tokenizer.eos_token_id]
        
        #문장 구조 : BOS + 질문 + (토큰으로 구분) + 레이블 + (토큰으로 구분) + 응답 + EOS
        sentence = tokenizer.encode('<unused0>' + question + '<unused1>' + response)
        
        yield bos_token + sentence + eos_token

batch_size = 4

# def 함수를 적용한 tf dataset을 만듭니다.
dataset = tf.data.Dataset.from_generator(tokenize_df, output_types = tf.int32)

# batch에서 가장 긴 문장을 기준으로 zero-padding을 진행합니다.
dataset = dataset.padded_batch(batch_size = batch_size, padded_shapes=(None,),
                               padding_values = tokenizer.pad_token_id)

for batch in dataset:
    break
# print(batch[0])
# print(tokenizer.decode(batch[0]))

EPOCHS = 20
adam = tf.keras.optimizers.Adam(learning_rate = 3e-5, epsilon=1e-08)
steps = len(df) // batch_size + 1

for epoch in range(EPOCHS):
    train_loss = 0

    try:
        for batch in tqdm(dataset, total = steps):
            try:
                with tf.GradientTape() as tape:
                    result = model(batch, labels = batch)
                    loss = result[0]
                    batch_loss = tf.reduce_mean(loss, -1)
      
                grads = tape.gradient(batch_loss, model.trainable_variables)
                adam.apply_gradients(zip(grads, model.trainable_variables))
                train_loss += batch_loss / steps
                
            except:
                pass
            
    except:
        pass

tokenizer.save_pretrained('D:/TJ_FInal_Project/KDJ/News_Summarization/Model/kogptToken2')
model.save_pretrained('D:/TJ_FInal_Project/KDJ/News_Summarization/Model/kogpt2Model')

tokenizer = AutoTokenizer.from_pretrained('D:/TJ_FInal_Project/KDJ/News_Summarization/Model/kogptToken2', bos_token='</s>', eos_token='</s>', pad_token='<pad>')
model = TFGPT2LMHeadModel.from_pretrained('D:/TJ_FInal_Project/KDJ/News_Summarization/Model/kogpt2Model')

def chatbot(text):
    # input sentence : "질문" / 레이블 + 응답
    sentence = '<unused0>' + text + '<unused1>'
    tokenized = [tokenizer.bos_token_id] + tokenizer.encode(sentence)
    tokenized = tf.convert_to_tensor([tokenized])
    
    # 질문 문장으로 "레이블 + 응답" 토큰 생성
    result = model.generate(tokenized, min_length = len(text)+32, max_length = 190, repetition_penalty = 0.8,
                            do_sample = True, no_repeat_ngram_size = 3, temperature = 0.01,
                            top_k = 5)
    
    output = tokenizer.decode(result[0].numpy().tolist())
    print(output)
    response = output.split('<unused1> ')[1]
    # 응답 토큰 생성
    # response = response.split('<unused> ')[1].replace('</s>', '')
    
    return response

label = chatbot("박원순 서울시장 사진 이 8일 고층 재개발 재건축 관련 요구에 작심한 듯 쓴소리를 쏟아냈다 박 시장의 발언은 서울 내 노후 아파트 주민들이 서울시를 상대로 재건축 인허가를 요구하며 집단행동을 시작한 와중에 나온 것이다")
print('원본 : 박원순 서울시장 사진 이 8일 고층 재개발 재건축 관련 요구에 작심한 듯 쓴소리를 쏟아냈다 박 시장의 발언은 서울 내 노후 아파트 주민들이 서울시를 상대로 재건축 인허가를 요구하며 집단행동을 시작한 와중에 나온 것이다')
print(f'결과 : {label}')
print("정답 : 박원순 서울시장은 8일 서울시청에서 열린 '골목길 재생 시민 정책 대화'에 참석하여 고층 재개발, 재건축 관련 요구에 '옆집 사람이 누구인지도 모르는 이것이 서울의 미래이고 행복한 삶을 보장하는 것이냐' 며 작심한 듯 쓴소리를 했다.")