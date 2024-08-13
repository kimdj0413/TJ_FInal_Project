 # -*- coding: utf-8 -*-
import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel
import pandas as pd
from tqdm import tqdm
import urllib.request

tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='', eos_token='', pad_token='')
model = TFGPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', from_pt=True)

train_data = pd.read_csv('D:/TJ_FInal_Project/KDJ/News_Summarization/Data/문서요약 텍스트/Preprocess/train_preprocess_test.csv')

sentence_max_len = 300
abs_max_len = 40

train_data = train_data[train_data['sentence'].apply(lambda x: len(x.split()) <= sentence_max_len)]
train_data = train_data[train_data['abs'].apply(lambda x: len(x.split()) <= abs_max_len)]

batch_size = 32

def get_article():
  for sentence, abs in zip(train_data['sentence'].to_list(), train_data['abs'].to_list()):
    bos_token = [tokenizer.bos_token_id]
    eos_token = [tokenizer.eos_token_id]
    sent = tokenizer.encode('' + sentence + '' + abs) 
    yield bos_token + sent + eos_token

dataset = tf.data.Dataset.from_generator(get_article, output_types=tf.int32)
dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=(None,), padding_values=tokenizer.pad_token_id)

for batch in dataset:
    print(batch)
    break
# print(tokenizer.decode(batch[2]))
# print(tokenizer.encode('올해 세비 인상분 기부하기로 민주당은 이날 국회에서 정책 의원총회를 열어 이런 내용의 당론을 채택했다고 권미혁 원내대변인이 브리핑을 통해 밝혔다. 권 원내대변인은 5 18 운동의 정의와 규정을 좀 더 명확히 하고, 5 18에 대한 비방과 왜곡, 날조, 허위사실 유포 등에 대한 처벌을 강화하는 방식으로 갈 것 이라고 말했다. 권 원내대변인은 박광온 의원이 발의한 법안이 있는데, 여기에 민주평화당과 정의당, 바른미래당에서 개별적으로 참여할 분들, 무소속 의원이 함께해 개정안을 공동 발의할 계획 이라고 설명했다. 자유한국당 김진태 이종명 김순례 의원의 5 18 망언 논란이 불거진 상황에서 5 18 왜곡 처벌법 추진을 통해 한국당을 향한 공세를 강화하려는 의도로도 보인다. 민주당은 또 올해 국회의원 세비 인상분을 기부하기로 하고, 방식과 기부단체 선정 등은 홍영표 원내대표에게 위임하기로 정했다. 의총에선 전날 경제사회노동위원회 경사노위 가 탄력근로제 단위 기간을 최장 3개월에서 6 개월로 합의한 것과 관련한 후속 입법 문제와 법관 탄핵 사안도 논의됐다. 민주당은 특히 양승태 전 대법원장 시절 사법농단 사건에 연루된 현직 판사들의 탄핵소추를 실제로 추진할지 문제와 추진 시 방식과 범위 등에 대해선 다시 의총을 열어 결정하기로 했다. 임채만 기자더불어민주당은 한국당에 대한 공세를 강화할 목적으로 20일 민주평화당, 정의당, 바른미래당 일부 의원, 무소속 의원이 함께하는 5.18 민주화운동에 대한 왜곡·날조·비방행위를 처벌하기 위한 특별법 개정안을 공동 발의하기로 했다.'))

adam = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)
steps = len(train_data) // batch_size + 1

EPOCHS = 3

for epoch in range(EPOCHS):
  epoch_loss = 0

  for batch in tqdm(dataset, total=steps):
      with tf.GradientTape() as tape:
          result = model(batch, labels=batch)
          loss = result[0]
          batch_loss = tf.reduce_mean(loss)
          
      grads = tape.gradient(batch_loss, model.trainable_variables)
      adam.apply_gradients(zip(grads, model.trainable_variables))
      epoch_loss += batch_loss / steps

  print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, epoch_loss))

  def return_answer_by_chatbot(user_text):
    sent = '' + user_text + ''
    input_ids = [tokenizer.bos_token_id] + tokenizer.encode(sent)
    input_ids = tf.convert_to_tensor([input_ids])
    output = model.generate(input_ids, max_length=50, do_sample=True, top_k=20)
    sentence = tokenizer.decode(output[0].numpy().tolist())
    chatbot_response = sentence.split(' ')[1].replace('', '')
    return chatbot_response
  
print(return_answer_by_chatbot('올해 세비 인상분 기부하기로 민주당은 이날 국회에서 정책 의원총회를 열어 이런 내용의 당론을 채택했다고 권미혁 원내대변인이 브리핑을 통해 밝혔다. 권 원내대변인은 5 18 운동의 정의와 규정을 좀 더 명확히 하고, 5 18에 대한  비방과 왜곡, 날조, 허위사실 유포 등에 대한 처벌을 강화하는 방식으로 갈 것 이라고 말했다. 권 원내대변인은 박광온 의원이 발의한 법안이 있는데, 여기에 민주평화당과 정의당, 바른미래당에서 개별적으로 참여할 분들, 무소속 의원이 함께해 개 정안을 공동 발의할 계획 이라고 설명했다. 자유한국당 김진태 이종명 김순례 의원의 5 18 망언 논란이 불거진 상황에서 5 18 왜곡 처벌법 추진을 통해 한국당을 향한 공세를 강화하려는 의도로도 보인다. 민주당은 또 올해 국회의원 세비 인상분을  기부하기로 하고, 방식과 기부단체 선정 등은 홍영표 원내대표에게 위임하기로 정했다. 의총에선 전날 경제사회노동위원회  경사노위 가 탄력근로제 단위 기간을 최장 3개월에서 6개월로 합의한 것과 관련한 후속 입법 문제와 법관 탄핵 사안도 논의 됐다. 민주당은 특히 양승태 전 대법원장 시절 사법농단 사건에 연루된 현직 판사들의 탄핵소추를 실제로 추진할지 문제와  추진 시 방식과 범위 등에 대해선 다시 의총을 열어 결정하기로 했다. 임채만 기자더불어민주당은 한국당에 대한 공세를 강 화할 목적으로 20일 민주평화당, 정의당, 바른미래당 일부 의원, 무소속 의원이 함께하는 5.18 민주화운동에 대한 왜곡·날조·비방행위를 처벌하기 위한 특별법 개정안을 공동 발의하기로 했다.'))