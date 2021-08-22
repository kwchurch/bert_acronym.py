from transformers.pipelines import pipeline
import sys

model_name = "phiyodr/bart-large-finetuned-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

def answer_question(question, doc):
    inputs = {"question": question, 
              "context": doc}
    res = nlp(inputs)
    return res["answer"]

for line in sys.stdin:
    fields = line.rstrip().split("\t")
    if len(fields) >= 2:
        SF, doc = fields[0:2]
        try:
            LF = answer_question("What does {} stand for?".format(SF), doc)
            print("\t".join([LF, SF, doc]))
        except:
            print("\t".join(["NA",SF,doc]))
