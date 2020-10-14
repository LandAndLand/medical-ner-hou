## **Introduction**

This project is an implementation of Named Entity Recognition (NER) using a pretrained bert on chinese text finetuned on a dataset of chinese medicine usage/manual/synopsis. An example of such thing is shown below:

> 湖北御金丹药业有限公司 非处方药物（乙类）,国家基本药物目录（2012） 密封(10 ～ 30℃)。 每袋装 15 克 孕妇禁用。 尚不明确。 孕妇禁用。 活血调经。用于血瘀所致的月经不调，症见经水量少。用于月经量少，月经后错，经来腹痛月经不调 开水冲服。一次 15 克，一日 2 次。 祛瘀生新。

This project aims to detect named entities (kind, start index, end index, actual value) such as the following:

> DRUG_EFFICACY 145 149 祛瘀生新 \
> PERSON_GROUP 61 63 孕妇 \
> PERSON_GROUP 75 77 孕妇 \
> DRUG_EFFICACY 82 86 活血调经 \
> SYMPTOM 101 105 经水量少 \
> SYMPTOM 108 112 月经量少 \
> SYMPTOM 113 117 月经后错 \
> SYMPTOM 118 122 经来腹痛 \
> SYMPTOM 122 126 月经不调

## **Prerequisites**

Third-party libraries are needed for this project to work. To install these, simply run:

```bash
pip install -r requirements.txt
```

## **Finetuning BERT**

Before finetuning, the datasets should be downloaded locally. To download and extract these files, run the following:

```bash
wget https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531824/round1_test.zip
wget https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531824/round1_train.zip
unzip round1_test.zip
unzip round1_train.zip
```

To start finetuning BERT, run the following (only available on machines with discrete GPUs):

```bash
python finetune.py
```

## **Using the model**

To try the model using your own text, run the following:

```bash
python classify.py <your-sample-text-here>
```
