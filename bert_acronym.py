import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import sys

bert='bert-large-uncased-whole-word-masking-finetuned-squad'
model = BertForQuestionAnswering.from_pretrained(bert)
tokenizer = BertTokenizer.from_pretrained(bert)

def untokenize(tokens):
    return ' '.join(tokens).replace(' ##', '')

def make_segs(input_ids):
    a = 1+input_ids.index(tokenizer.sep_token_id) # a is the position after [SEP] token
    b = len(input_ids) - a      # b is everything else
    return [0]*a + [1]*b

def answer_question(question, doc):
    input_ids = tokenizer.encode(question, doc)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    res = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([make_segs(input_ids)]))
    return untokenize(tokens[torch.argmax(res.start_logits):torch.argmax(res.end_logits)])

for line in sys.stdin:
    fields = line.rstrip().split('\t')
    if len(fields) >= 2:
        SF,doc = fields[0:2]
        try:
            LF = answer_question('What does %s stand for?' % SF, doc)
            print('\t'.join([LF,SF,doc]))
        except:
            print("\t".join(["NA",SF,doc]))
