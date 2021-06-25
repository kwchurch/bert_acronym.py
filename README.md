# Multiword Expressions, Acronyms and Opportunities for Improving Deep Nets


## Dependencies:
We have tested the pipeline with the following dependencies. 
```
    paddlepaddle >= 2.1.0
    paddlenlp >= 2.0.4
    torch >= 1.7.0
    transformers >= 4.5.0
```

Install the above dependencies with: 
```shell
    pip install -r requirements.txt
```

## BERT-SQuAD and ERNIE-SQuAD for acronym identification

Very simple program to use BERT-SQuAD and ERNIE-SQuAD to find long forms (LFs) for short forms (SFs)
 ```shell
   python ernie_acronym.py < sample_input.txt
   python bert_acronym.py < sample_input.txt
 ```
