from constants import BERT_VARIANT
from modeling import classify
from pathlib import Path
from transformers import BertForTokenClassification, BertTokenizerFast

import fire


def do_classify(sequence: str):
    # initialize checkpoint path
    ckpt_path = Path(f'finetuned-{BERT_VARIANT}'.replace('/', '-'))
    assert ckpt_path.is_dir(), 'Finetune model first using finetune.py'

    # load finetuned model
    tokenizer = BertTokenizerFast.from_pretrained(BERT_VARIANT)
    model = BertForTokenClassification.from_pretrained(ckpt_path).eval()

    # print output
    entities = classify(model, tokenizer, sequence)
    for entity in entities:
        print(entity)


if __name__ == '__main__':
    fire.Fire(do_classify)
