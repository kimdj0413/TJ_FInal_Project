import pandas as pd
import re

valid_data = pd.read_csv('D:/TJ_FInal_Project/KDJ/News_Summarization/Data/문서요약 텍스트/Preprocess/valid_test.csv')
val_sentence = valid_data['sentence']
val_abs = valid_data['abs']

def replace_special_characters(series):     # 영어,한자, 숫자, 한글, 공백, ,!?'"~ = 공백 으로 대치
    return series.apply(lambda x: re.sub(r'[^가-힣a-zA-Z0-9\u4e00-\u9fff\s.,!?\'\"~]', ' ', x))

def preprocess_sentence(sentence):
    # 이메일 패턴 제거
    sentence = re.sub(r'\S+@\S+\.\S+', '', sentence)  # 이메일 제거
    # 연속된 공백을 하나로, 양쪽 공백 제거
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    # 개행 삭제
    sentence = sentence.replace('\n', '')
    # 특수 문자 제거 (한글, 영어, 숫자, 공백만 남기기)
    sentence = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', sentence)
    # 다시 연속된 공백을 하나로, 양쪽 공백 제거
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    return sentence
val_sentence = val_sentence.apply(preprocess_sentence)

df = pd.DataFrame({
    'sentence':val_sentence,
    'abs':val_abs
})
df.to_csv('D:/TJ_FInal_Project/KDJ/News_Summarization/Data/문서요약 텍스트/Preprocess/valid_test.csv',index=False)

print(df)