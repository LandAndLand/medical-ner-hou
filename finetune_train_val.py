from constants import BERT_VARIANT, LABELS
from dataset import MedicalNERDataset, PreprocessForBERT
from tqdm import tqdm
from transformers import BertForTokenClassification, BertTokenizerFast, AdamW
from pathlib import Path
from modeling import classify
from is_save_model_to_file import is_save_model_to_file

import fire
import logging
import torch
import numpy as np

# 设置运行日志文件
# 通过调用 logger 类的实例来执行日志记录
logging.basicConfig(
    filename='finetune.logs',
    filemode='a',
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG,
)
logger = logging.getLogger()

is_save_model = True

def finetune(dataset_dir: str = 'split_train'):
    # load pretrained bert model
    tokenizer = BertTokenizerFast.from_pretrained(BERT_VARIANT)
    model = BertForTokenClassification.from_pretrained(
        BERT_VARIANT,
        return_dict=True,
        num_labels=len(LABELS),
    ).train()

    # move model to GPU (if available)
    if torch.cuda.is_available():
        model.cuda()

    # init adam optimizer
    optimizer = AdamW(model.parameters(), lr=3e-5)

    # define training path and progress bar
    # 获得训练集

    dataset = MedicalNERDataset(
        dataset_dir=dataset_dir,
        transform=PreprocessForBERT(
            config=model.config,
            tokenizer=tokenizer,
        ),
    )

    # finetune bert to each of the training sample.
    # each sample is seen exactly once.
    num_epoches = 20
    f1_scores = []
    f1_higgest = 0
    for epoch in range(num_epoches):
        print(f'the {epoch} th epoch training process:')
        for f, sequences_and_labels in tqdm(dataset):
            logger.info(f'Finetuning start on {f}')
            for i, (sequence, labels) in enumerate(sequences_and_labels):
                # train model on this sample
                inputs = tokenizer(sequence, return_tensors='pt')
                labels = torch.tensor(labels)

                # put sample on the gpu (if available)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                    labels = labels.cuda()

                # compute loss on this sample
                loss = model(**inputs, labels=labels).loss

                # update model parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                logger.info(f'Block {i} done with loss={loss}')

        is_save_model,f1, f1_higgest = is_save_model_to_file(f1_higgest)
        f1_scores.append((f1))
        if is_save_model:
        # save finetuned weights
            ckpt_path = f'finetuned-{BERT_VARIANT}-{epoch}'.replace('/', '-')
            model.save_pretrained(ckpt_path)


if __name__ == '__main__':
    fire.Fire(finetune)
