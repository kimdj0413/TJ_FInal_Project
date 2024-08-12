import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

# Load the tokenizer and model
tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')

# Input text
text = """
올해 6월 남편에게 주식을 양도하고 1000만원 가량 수익이 났습니다.
250만원 제하고 나머지 수익금에 대한 세금을 내야하는걸로 알고있는데 신고기간이 어떻게 되나요?
주식을 양도하고 매도한 날로부터 2개월 안에 해야하는지 내년 5월 종합소득세신고 기간에 증권사 대행으로 해도되는지 궁금합니다.
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