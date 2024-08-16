import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

# Load the tokenizer and model
tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')

# Input text
text = """
지역균형선발제 대학 신입생 선발시 지역간 불균형 현상을 바로잡기 위해 특정 지역에 혜택을 주는 제도. 교과성적 우수자를 대상으로 모집정원의 20% 내외를 선발한다. 내신 위주로 선발함으로써 낙후지역 학생들이 학업 여건이 좋은 대도시 수험생들과 동일한 조건에서 경쟁할 수 있게 된다. 사회적 약자를 우대하는 정책으로 역차별문제 등 논란을 불러오기도 한다.
"""

# Prepare the input for the model
text = text.replace('\n', ' ')
raw_input_ids = tokenizer.encode(text)
input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

# Generate the summary
summary_ids = model.generate(torch.tensor([input_ids]), num_beams=4, max_length=512, eos_token_id=1)

# Decode and print the summary
summary = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
print(summary)

'''
올해 6월 남편에게 주식을 양도하고 1000만원 가량 수익이 났는데 신고기간이 어떻게 되나요? 
주식을 양도하고 매도한 날로부터 2개월 안에 해야하는지 내년 5월 종합소득세신고 기간에 증권사 대행으로 해도되는지 궁금합니다.
'''

'''지역간선발제 대학 신입생 선발시 지역간 불균형 현상을 바로잡기 위해 특정 지역에 혜택을 주는 제도인 
지역균형선발제 대학 신입생 선발시 지역간 불균형 현상을 바로잡기 위해 특정 지역에 혜택을 주는 제도인
 지역간 불균형 현상을 바로잡기 위해 특정 지역에 혜택을 주는 제도인 지역간 불균형 현상을 바로잡기 위
해 특정 지역에 혜택을 주는 제도이다.'''