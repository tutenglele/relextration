import os

from transformers import RobertaTokenizer, RobertaModel

import utils

params = utils.Params()
tokenizer = RobertaTokenizer.from_pretrained(os.path.join(params.bert_model_dir/'roberta'))
model = RobertaModel.from_pretrained(os.path.join(params.bert_model_dir/'roberta'))
text = "Martha Whitney Bagnall , the daughter of Whitney S. Bagnall and Roger S. Bagnall of New York , is to be married today in Rhinebeck , N.Y. , to Edward Bing Han , the son of Chou-Yeen Han and Bing-Hou Han of Hopewell Junction , N.Y. Lynn Tomalas , a Universal Life minister , is to officiate at the Wilderstein Preservation , a historic house on the Hudson River ."
encoded_input = tokenizer(text, return_tensors='pt')

print(encoded_input, "*")
print("input_ids", encoded_input['input_ids'], encoded_input['input_ids'].shape)
output = model(**encoded_input)  # 字典 kwargs 变成关键字参数传递

cls, out = output
print(cls.shape)
print(out.shape)
text_token = tokenizer.tokenize(text)
print("text_token", text_token)
token_id = tokenizer.convert_tokens_to_ids(text_token)
print("text_token_id", token_id)
sentence = tokenizer.convert_tokens_to_string(text_token[0:6])
print("sentence: ", sentence)
model.bert(token_id)