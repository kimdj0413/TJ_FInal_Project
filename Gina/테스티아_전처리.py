from transformers import BartTokenizerFast, BartForConditionalGeneration
import pandas as pd
import re

# KoBART 모델과 Fast 토크나이저 로드
tokenizer = BartTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v1')

def summarize_question(question):
    inputs = tokenizer([question], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return summary

def clean_answer(answer):
    # 답변 추출하는 함수
    match = re.search(r"본인 입력 포함 정보(.+?)(\d{1,2}(?:분|시간) 전)", answer, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return answer  # 만약 패턴을 찾지 못하면 원본 반환

# 데이터 읽기
df = pd.read_csv('./Gina/jisik_in_테스티아.csv')

# Question 컬럼 요약
df["Question"] = df["Question"].apply(summarize_question)

# Answer 컬럼 정리
df["Answer"] = df["Answer"].apply(clean_answer)

# 결과 확인
print(df.head())

# 전처리된 데이터를 CSV로 저장
df.to_csv("./Gina/processed_jisik_in_테스티아.csv", index=False, encoding="utf-8-sig")