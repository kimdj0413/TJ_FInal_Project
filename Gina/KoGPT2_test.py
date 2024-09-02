import pandas as pd
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

# 데이터셋 경로 설정
input_csv_path = "C:\\TJ_FInal_Project\\Gina\\processed_jisikin.csv"
output_csv_path = "C:\\TJ_FInal_Project\\Gina\\Summarized_Results.csv"

# 데이터 로드
try:
    df = pd.read_csv(input_csv_path)
    print(f"데이터를 성공적으로 읽었습니다. 데이터 프레임의 상위 5개 행:\n{df.head()}")
except FileNotFoundError:
    print(f"파일을 찾을 수 없습니다: {input_csv_path}")
    raise
except pd.errors.EmptyDataError:
    print(f"파일이 비어 있습니다: {input_csv_path}")
    raise
except Exception as e:
    print(f"데이터를 읽는 중 오류가 발생했습니다: {e}")
    raise

# 문장의 최대 길이에 맞게 데이터 필터링
sentence_max_len = 200
abs_max_len = 800

df = df[df['Question'].apply(lambda x: len(str(x).split()) <= sentence_max_len)]
df = df[df['Answer'].apply(lambda x: len(str(x).split()) <= abs_max_len)]

# 토크나이저 및 모델 로드
try:
    tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2')
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
    model.eval()  # 평가 모드로 전환
    print("토크나이저와 모델을 성공적으로 로드했습니다.")
except Exception as e:
    print(f"모델을 로드하는 중 오류가 발생했습니다: {e}")
    raise

def summarize_text(text):
    if not text:
        return text  # 비어 있거나 문자열이 아닌 경우 원래의 값을 반환
    
    # input sentence : "질문" / 레이블 + 응답
    sentence = '<unused0>' + text + '<unused1>'
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512)  # PyTorch 텐서로 변환
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'], 
            min_length=30,  # 적절한 min_length 설정
            # max_length=512,
            max_new_tokens=50,  # 생성할 새로운 토큰의 최대 개수를 50으로 설정
            repetition_penalty=1.0,  # 반복 패널티 조정
            do_sample=True, 
            no_repeat_ngram_size=3, 
            temperature=0.7,  # 온도 조정
            top_k=50
        )
    
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = output.split('<unused1> ')[1] if '<unused1>' in output else output
    
    return response

# 요약 결과를 저장할 리스트
summarized_data = []

# 데이터프레임의 각 행을 처리하여 요약을 생성
for idx, row in df.iterrows():
    question = row['Question']
    answer = row['Answer']
    
    # 질문과 답변을 각각 요약
    try:
        summarized_question = summarize_text(question)
        summarized_answer = summarize_text(answer)
    except Exception as e:
        print(f"요약 생성 중 오류가 발생했습니다: {e}")
        summarized_question = "요약 오류"
        summarized_answer = "요약 오류"
    
    # 원본 질문과 요약된 답변을 저장
    summarized_data.append({
        'Question': summarized_question,
        'Summarized_Answer': summarized_answer
    })

# 요약 결과를 데이터프레임으로 변환
summarized_df = pd.DataFrame(summarized_data)

# 데이터프레임을 CSV 파일로 저장
try:
    summarized_df.to_csv(output_csv_path, index=False)
    print(f"요약 결과가 '{output_csv_path}'에 저장되었습니다.")
    print(summarized_df.head(5))
except Exception as e:
    print(f"CSV 파일을 저장하는 중 오류가 발생했습니다: {e}")
    raise
