from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
import sys

model_name = "allenai/t5-small-squad11"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def answer_question(question, doc):
    input_string = "{} \n {}".format(question, doc)
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    return output

for line in sys.stdin:
    fields = line.rstrip().split("\t")
    if len(fields) >= 2:
        SF, doc = fields[0:2]
        try:
            LF = answer_question("What does {} stand for?".format(SF), doc)[0]
            print("\t".join([LF, SF, doc]))
        except:
            print("\t".join(["NA",SF,doc]))