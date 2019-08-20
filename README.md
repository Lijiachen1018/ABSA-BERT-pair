# A modification from paper

### Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence

for AI challenge 2018 Meituan Dianping ABSA task
## Dataset


|guid|train-0|
|----|------|
label| '-2', '-2', '-2', '-2', '1', '-2', '-2', '-2', '-2', '1', '-2', '-2', '-2', '-2', '-2', '-2', '1', '-2', '1', '-2'|
text_a|"吼吼吼,萌死人的棒棒糖,中了大众点评的霸王餐,太可爱了. 一直就好奇这个棒棒糖是怎么个东西, 大众点评给了我这个土老冒一个见识的机会. 看介绍棒棒糖是用德国糖做的, 不会很甜,中间的照片是糯米的,能食用, 真是太高端大气上档次了,还可以买蝴蝶结扎口,送人可以买礼盒. 我是先打的卖家电话,加了微信,给卖家传的照片.等了几天,卖家就告诉我可以取货了, 去大官屯那取的.虽然连卖家的面都没见到,但是还是谢谢卖家送我这么可爱的东西, 太喜欢了,这哪舍得吃啊.".|
text_b|None|

## Data Preprocess

```
python data_preprocess.py
```
## Train

```
bash ch_run_QA_B.bash
```

## Evaluation

```
python evaluation.py
```

# ABSA as a Sentence Pair Classification Task

Codes and corpora for paper "Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence" (NAACL 2019)

## Requirement

* pytorch: 1.0.0
* python: 3.7.1
* tensorflow: 1.13.1 (only needed for converting BERT-tensorflow-model to pytorch-model)
* numpy: 1.15.4
* nltk
* sklearn


## Citation

```
@inproceedings{sun-etal-2019-utilizing,
    title = "Utilizing {BERT} for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence",
    author = "Sun, Chi  and
      Huang, Luyao  and
      Qiu, Xipeng",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1035",
    pages = "380--385"
}
```
