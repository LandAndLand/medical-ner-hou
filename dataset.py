from pathlib import Path
from torch.utils.data import Dataset
from transformers import BertConfig, BertTokenizerFast
from utils import Entity, parse_entities, align_labels, split_into_blocks
from typing import Tuple, List


class MedicalNERDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_path = Path(dataset_dir)
        assert self.dataset_path.is_dir()
        self.files = list(self.dataset_path.rglob('*.txt'))
        #self.files = train_files
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[str, List[Entity]]:
        f = self.files[index]
        label_path = self.dataset_path / (f.name[:-4] + '.ann')
        assert label_path.is_file()

        # load text sample
        with f.open('r', encoding='utf-8') as fd:
            sequence = fd.read()

        # load text sample entity labels and parse it
        with label_path.open('r', encoding='utf-8') as fd:
            entities = sorted(parse_entities(fd), key=lambda x: x.start)
            outputs = sequence, entities

        # apply transforms if present
        if self.transform:
            outputs = self.transform(sequence, entities)

        return f, outputs

    def __len__(self) -> int:
        return len(self.files)


class PreprocessForBERT:
    def __init__(self, config: BertConfig, tokenizer: BertTokenizerFast):
        self.max_n_positions = config.max_position_embeddings
        self.tokenizer = tokenizer

    def __call__(
        self,
        sequence: str,
        entities: List[Entity],
    ) -> List[Tuple[str, List[Entity]]]:
        # tokenize the input sequence (without special tokens, e.g. [CLS], [SEP])
        tokens = self.tokenizer.tokenize(sequence, add_special_tokens=False)

        # reposition entity labels' start and end indices
        # according to the new indices of the tokens list.
        # Also, the chosen BERT variant is uncased (capitalization is ignored)
        labels = align_labels(
            sequence=sequence.lower(),
            tokens=tokens,
            entities=entities,
            vocab=self.tokenizer.get_vocab(),
        )

        # no need to split the tokens if its length is `max_n_positions` - 2
        # 2 is for the special tokens to be added during the tokenization process
        if len(tokens) <= self.max_n_positions - 2:
            # add labels for the [CLS] and [SEP] token
            labels = [0] + labels + [0]
            return [(sequence, labels)]

        pairs = []
        token_ids = self.tokenizer.encode(sequence)[1:-1]
        assert len(token_ids) == len(tokens)

        # split long sequences into blocks (token length < `max_n_positions`)
        # these blocks will be returned as a list for convenience
        for start, end in split_into_blocks(
            token_ids=token_ids,
            separator_token_id=self.tokenizer.get_vocab().get('。'),
            block_size=self.max_n_positions - 2,
        ):
            # sanity check: block size should be the same for both the tokens and labels
            # also, block size should be less than or equal to `max_n_positions` - 2
            assert len(token_ids[start:end]) == len(labels[start:end])
            assert end - start <= self.max_n_positions - 2

            sequence = self.tokenizer.decode(token_ids[start:end])
            block_labels = [0] + labels[start:end] + [0]
            pairs.append((sequence, block_labels))

        return pairs
