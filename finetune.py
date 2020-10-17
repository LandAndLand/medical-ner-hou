from constants import BERT_VARIANT, LABELS
from dataset import MedicalNERDataset, PreprocessForBERT
from tqdm import tqdm
from transformers import BertForTokenClassification, BertTokenizerFast, AdamW

import fire
import logging
import torch

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
    for epoch in range(num_epoches):
        print(f'the {epoch} th epoch training process:')
        for f, sequences_and_labels in tqdm(dataset):
            logger.info(f'Finetuning start on {f}')

            # each sample may contain tokens of length greater than 512
            # which is bigger than bert can handle. PreprocessForBERT
            # splits these sample tokens into blocks of size < 512
            # 由于 bert 长度的 512 限制，训练样本必须保证小于等于 512，
            # 所以有些训练样本可能被拆分为若干个子样本，遍历训练样本的子样本
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

        # save finetuned weights
        ckpt_path = f'finetuned-{BERT_VARIANT}-{epoch}'.replace('/', '-')
        model.save_pretrained(ckpt_path)


if __name__ == '__main__':
    fire.Fire(finetune)
