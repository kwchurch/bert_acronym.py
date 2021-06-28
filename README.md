# Multiword Expressions, Acronyms and Opportunities for Improving Deep Nets


Very simple program to use BERT-SQuAD and ERNIE-SQuAD to find long forms (LFs) for short forms (SFs)
```shell 
  pip install -r requirements.txt
  python ernie_acronym.py < sample_input.txt
  python bert_acronym.py < sample_input.txt
```

We have tested on python 3.6.9.  Other versions may work but not tested.

If this does not work, you may try to upgrade pip.  Also, this may help:
```shell
  pip install --upgrade paddlenlp -i https://pypi.org/simple
  ```
