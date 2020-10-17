from collections import namedtuple
from constants import LABELS
from io import TextIOWrapper
from typing import Callable, Dict, Generator, List, Tuple
from pathlib import Path

import logging
import numpy as np
import regex as re
import random
import shutil

UNIQUE_LABELS = set(x[2:] for x in LABELS[1:])
Entity = namedtuple('Entity', 'kind start end value')
ENTITY_PATTERN = re.compile(
    r'^T\d+\s+([A-Z\_]+)\s+(\d+)\s+(\d+)\s+([\p{Han}A-Za-z\d]+)$')

logger = logging.getLogger()


def parse_entities(fd: TextIOWrapper) -> Generator[Entity, None, None]:
    for line in fd:
        match = re.match(ENTITY_PATTERN, line)
        assert match is not None, f'Unknown Pattern: {line}'

        # extract matched groups
        label = match.group(1)
        start_index = int(match.group(2))
        end_index = int(match.group(3))
        value = match.group(4)
        assert label in UNIQUE_LABELS, f'Unknown label: {label}'

        # return entity object
        yield Entity(label, start_index, end_index, value)


def align_labels(
    sequence: str,
    tokens: List[str],
    entities: List[Entity],
    vocab: Dict[str, int] = None,
) -> List[int]:
    # initialize labels as all values having no entity label
    labels = [0] * len(tokens)

    # convert sequence IDs to token IDs
    tokeni = seqi_to_tokeni(sequence, tokens, vocab=vocab)

    # reposition each entity start and end indices
    # according to the new indices with respect to the tokens list
    for entity in entities:
        # there are some chinese characters in the training dataset
        # in which the pretrained tokenizer haven't seen (becomes UNK)
        # in this case, we just exclude the entity
        if any([tokeni[x] is None for x in range(entity.start, entity.end)]):
            logger.warn(f'{entity} excluded since some of the tokens is None')
            continue

        new_start = tokeni[entity.start]
        new_end = tokeni[entity.end - 1] + 1

        # assign label ID
        b_label, i_label = f'B-{entity.kind}', f'I-{entity.kind}'
        assert b_label in LABELS and i_label in LABELS, \
            f'Kind "{entity.kind}" not in LABELS'
        n_tokens = new_end - new_start
        labels[new_start: new_end] = [LABELS.index(i_label)] * n_tokens
        labels[new_start] = LABELS.index(b_label)

    return labels


def split_into_blocks(
    token_ids: List[int],
    separator_token_id: int,
    block_size: int,
) -> Generator[Tuple[List[str], List[int]], None, None]:
    n_token_ids = len(token_ids)
    start_index = 0
    remainders = []
    indices = np.array([
        i for i, c in enumerate(token_ids)
        if c == separator_token_id
    ])

    while start_index < n_token_ids:
        # if the remaining tokens are enough
        # it can be returned immediately without further calculations
        # it is also a signal that it is the last block
        if len(token_ids[start_index:]) <= block_size:
            yield start_index, n_token_ids
            break

        # shift already seen positions into the negative side
        # so that only the unseen positions are considered
        block = indices - sum([block_size - x for x in remainders])

        # compute the next biggest index of the separator token
        index = np.argmax(block[block < block_size])
        end_index = indices[index]

        yield start_index, end_index + 1

        # update tracker indices
        remainders.append(block_size - (end_index + 1 - start_index))
        start_index = end_index + 1


def extract_entities(
    sequence: str,
    logits: np.ndarray,
    encode: Callable[[str], List[int]],
    decode: Callable[[List[int]], str]
) -> List[Entity]:
    predictions = np.argmax(logits, axis=1)
    token_ids = encode(sequence, add_special_tokens=False)
    assert len(predictions) == len(token_ids)

    entities = []
    label, index, acc = '', -1, []
    for i, (token_id, prediction) in enumerate(zip(token_ids, predictions)):
        # convert id to label
        prediction = LABELS[prediction]

        # start labels
        if prediction.startswith('B'):
            if label != '':
                # add accumulated stuff as an entity
                entity = Entity(label[2:], index, i, decode(acc))
                entities.append(entity)

            # update tracker
            label, index, acc = prediction, i, [token_id]

        # intermediate labels
        elif prediction.startswith('I'):
            if label == '':
                label, index, acc = prediction, i, [token_id]
            elif label[2:] == prediction[2:]:
                acc.append(token_id)
            else:
                entity = Entity(label[2:], index, i, decode(acc))
                entities.append(entity)
                label, index, acc = prediction, i, [token_id]

        # other labels
        else:
            if label != '':
                # add accumulated stuff as an entity
                entity = Entity(label[2:], index, i, decode(acc))
                entities.append(entity)

                # reset trackers
                label, index, acc = '', -1, []

    # process remaining stuff (if there's any)
    if label != '':
        # add accumulated stuff as an entity
        entity = Entity(label[2:], index, len(predictions), decode(acc))
        entities.append(entity)

    return entities


def realign_extracted_entities(
    sequence: str,
    tokens: List[str],
    entities: List[Entity],
    vocab: Dict[str, int] = None,
) -> List[Entity]:
    seqi = tokeni_to_seqi(tokens, sequence, vocab=vocab)

    for entity in entities:
        # there are some instances where the entity's end index is
        # equal to the length of the tokens in which accessing it
        # directly causes an IndexError (needs to be checked)
        starti = seqi[entity.start]
        endi = seqi[entity.end] if entity.end < len(
            tokens) else [seqi[-1][-1] + 1]

        if isinstance(starti, list) and isinstance(endi, list):
            value = re.sub(r'\s+', '', entity.value)
            yield Entity(entity.kind, starti[0], endi[-1], value)
        else:
            logger.warn(f'Entity ignored: {entity}')


def seqi_to_tokeni(
    sequence: str,
    tokens: List[str],
    vocab: Dict[str, int] = None,
) -> List[int]:
    result = [None] * len(sequence)
    token_i = 0
    seq_i = 0

    while seq_i < len(sequence):
        current_character = sequence[seq_i]

        try:
            # attempt to get the token at position `token_i`
            # this throws IndexError if the tokens is already empty
            # in this case, we just return None for the index
            current_token = tokens[token_i]

            # sometimes the current token is an unknown token.
            # in this case, we proceed to the next known token
            while current_token == '[UNK]':
                token_i += 1
                current_token = tokens[token_i]
        except IndexError:
            seq_i += 1
            continue

        # whitespaces are ignored by the tokenizer,
        # so they don't take up space after a sequence is tokenized.
        # so, we leave their mapping to None
        if vocab is not None and current_character not in vocab:
            seq_i += 1
            continue

        # most of the time, the tokenizer tokenizes the sequence
        # by character. so, we can just map the indices accordingly.
        if current_character == current_token:
            result[seq_i] = token_i
            seq_i, token_i = seq_i + 1, token_i + 1
            continue

        # there are also instances where an unknown word is broken down into
        # multiple tokens (e.g., miguel -> ['mi', '##g', '##uel']).
        # in this case, we can process it the same way as a multiple-char token
        if current_token.startswith('##'):
            current_token = current_token[2:]

        # there are instances where a token can consist of multiple characters.
        # in this case, we point all char positions to the same token position.
        n_chars = len(current_token)
        if sequence[seq_i:seq_i+n_chars] == current_token:
            result[seq_i:seq_i+n_chars] = [token_i] * n_chars
            seq_i, token_i = seq_i + n_chars, token_i + 1
            continue

        # there are instances where the current character is not present
        # in the tokenized sequence. In this case, we map these character
        # indices to None.
        if current_character not in ''.join(tokens):
            seq_i += 1
            continue

        # as in 886.txt, some characters may be tokenized into a totally different
        # character that doesn't exist in the original sequence. In this case,
        # we just treat these tokens the same as [UNK] tokens.
        if current_token not in sequence:
            token_i += 1
            continue

        raise AssertionError(
            f'Unable to process char="{sequence[seq_i]}"'
            f' and token="{tokens[token_i]}"'
        )

    return result


def tokeni_to_seqi(tokens: List[str], sequence: str, vocab: Dict[str, int] = None) -> List[int]:
    # reversing the output of `seqi_to_tokeni` yields a mapping
    # from token IDs to sequence IDs.
    # however, different sequence IDs may point to the same token IDs
    # and this needs to processed further
    tokeni = seqi_to_tokeni(sequence, tokens, vocab=vocab)

    result = [None] * len(tokens)
    for i, x in enumerate(tokeni):
        # whitespace characters are mapped to None indices
        # can be safely skipped
        if x is None:
            continue

        # many sequence IDs can map to the same token IDs
        # so we store these sequence IDs and let the caller decide
        # what to do with it
        if result[x] is None:
            result[x] = [i]
        else:
            result[x].append(i)

    # sanity check: only [UNK] tokens should be mapped to None
    for i, x in enumerate(result):
        if x is None and tokens[i] != '[UNK]':
            raise AssertionError(
                f'Token={tokens[i]} at i={i} should not be None!')

    return result


def split_train_set(
    source_dir='train',
    split_train_dir='split_train',
    split_val_dir='split_val',
    train_raio=0.8,
    shuffle=True,
):
    source_path = Path(source_dir)
    split_train_path = Path(split_train_dir)
    split_val_path = Path(split_val_dir)
    split_train_path.mkdir(parents=False, exist_ok=True)
    split_val_path.mkdir(parents=False, exist_ok=True)

    samples = list(source_path.glob('*.txt'))
    if shuffle:
        random.shuffle(samples)

    for i, sample in enumerate(samples):
        dest_dir = split_train_path if i < len(samples)*train_raio
        samples_ann_name = sample.stem+".ann"
        sample_ann = source_path / sample_ann_name

        shutil.copyfile(sample, dest_dir/sample.name)
        shutil.copyfile(sample_ann, dest_dir/sample_ann_name)
