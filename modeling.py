from transformers import BertForTokenClassification, BertTokenizerFast
from typing import List
from utils import (
    Entity,
    split_into_blocks,
    extract_entities,
    realign_extracted_entities,
)

import torch


def classify(
    model: BertForTokenClassification,
    tokenizer: BertTokenizerFast,
    sequence: str,
) -> List[Entity]:
    # tokenize input sequence
    sequence = sequence.lower()
    inputs = tokenizer(sequence, return_tensors="pt")
    # tokenizer返回的inputs是一个字典，主要的键分别是:attention_mask、input_ids
    # iput_ids主要是sequence的内容分词之后，每个词的id
    # attention_mask主要是多个sequence样本长短不一，有些句子有填充，该键对应的值是为了标记填充（标记为0）
    # 详细见transformer文档 https://huggingface.co/transformers/glossary.html
    mask = inputs["attention_mask"][:, 1:-1].bool()

    # split the input sequence into blocks
    # if its token length is greater than `max_position_embeddings`
    # 判断样本长度是否超过bert模型能够处理的长度
    if inputs["input_ids"].shape[1] > model.config.max_position_embeddings:
        token_ids = inputs["input_ids"][0, 1:-1].tolist()
        cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id

        # the token IDs of the spliced sequence is collected in an array
        # in preparation for the creation of the needed matrices ahead
        blocks = [
            [cls] + token_ids[start:end] + [sep]
            for start, end in split_into_blocks(
                token_ids=token_ids,
                separator_token_id=tokenizer.get_vocab().get("。"),
                block_size=model.config.max_position_embeddings - 2,
            )
        ]

        # create a matrix vertically stacking the token IDs.
        # the width of this matrix depends on the longest token block.
        # also, each row of the matrix contains [CLS] and [SEP] tokens.
        max_block_len = max([len(block) for block in blocks])
        input_ids = torch.tensor(
            [block + [tokenizer.pad_token_id] * (max_block_len - len(block)) for block in blocks]
        )
        attention_mask = torch.tensor(
            [[1] * len(block) + [0] * (max_block_len - len(block)) for block in blocks]
        )

        # basically the same with `attention_mask` except that it doesn't
        # take into account the [CLS] and [SEP] positions.
        # this is created so that the final logits can be indexed conveniently
        mask = torch.tensor(
            [[1] * (len(block) - 2) + [0] * (max_block_len - len(block)) for block in blocks],
            dtype=torch.bool,
        )

        # combine inputs as one batch to be processed at once
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": torch.zeros_like(input_ids),
        }

    # put data on the gpu (if available)
    if torch.cuda.is_available():
        model.cuda()
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # get model's output
    with torch.no_grad():
        logits = model(**inputs).logits.cpu()

    # decode model's output
    entities = extract_entities(
        sequence=sequence,
        logits=logits[:, 1:-1][mask],
        encode=tokenizer.encode,
        decode=tokenizer.decode,
    )
    entities = realign_extracted_entities(
        sequence=sequence,
        tokens=tokenizer.tokenize(sequence),
        entities=entities,
        vocab=tokenizer.get_vocab(),
    )

    return entities
