import json
import pandas as pd
import re

# JSON 파일 경로
file_path = r'D:/TJ_FInal_Project/KDJ/News_Summarization/Data/문서요약 텍스트/Validation/news_valid_original.json'

# JSON 파일 열기
with open(file_path, 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# 문서 데이터 추출
documents = json_data['documents']

# 데이터프레임 생성
data = []
for doc in documents:
    sentences = []
    for sentence_list in doc['text']:  # 각 문장 리스트에 접근
        if sentence_list:  # 리스트가 비어 있지 않은 경우
            # 첫 번째 문장 추출 후 ""를 "로 변경
            sentence = sentence_list[0]['sentence']
            sentences.append(sentence)  # 수정된 문장 추가
    abstract = doc['abstractive'][0] if doc['abstractive'] else ''  # abstractive 내용 추출 (비어있을 경우 처리)
    
    data.append({
        'title': doc['title'],
        'sentence': ' '.join(sentences),  # sentence 열에 문장 결합
        'abs': abstract  # abs 열에 abstractive 내용
    })

df = pd.DataFrame(data)
df.to_csv('D:/TJ_FInal_Project/KDJ/News_Summarization/Data/문서요약 텍스트/Preprocess/valid.csv', index=False)

# 데이터프레임 출력
print(df)
