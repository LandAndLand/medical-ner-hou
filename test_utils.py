from constants import LABELS
from io import StringIO
from utils import (
    Entity,
    extract_entities,
    parse_entities,
    align_labels,
    split_into_blocks,
    seqi_to_tokeni, tokeni_to_seqi,
)

import unittest
import numpy as np


class TestUtils(unittest.TestCase):
    def test_parse_entities(self):
        # generate fake entities
        fake_entities = [
            'T1	DRUG_DOSAGE 556 559	糖衣片',
            'T2	DRUG_TASTE 570 572	气香',
            'T13	SYMPTOM 70 74	脾胃虚弱',
        ]
        fake_fd = StringIO('\n'.join(fake_entities))

        # get parsed entities
        entities = list(parse_entities(fake_fd))

        # assert output values is correct
        self.assertEqual(len(entities), 3)
        self.assertEqual(entities[0], Entity('DRUG_DOSAGE', 556, 559, '糖衣片'))
        self.assertEqual(entities[1], Entity('DRUG_TASTE', 570, 572, '气香'))
        self.assertEqual(entities[2], Entity('SYMPTOM', 70, 74, '脾胃虚弱'))

    def test_align_labels(self):
        sequence = '凡脾胃虚弱,食入难化,呕吐泄泻,腹胀便溏,咳嗽痰多者'
        tokens = tuple(sequence)
        entities = [
            Entity('SYMPTOM', 1, 5, '脾胃虚弱'),
            Entity('FOOD', 11, 13, '呕吐'),
            Entity('DRUG', 13, 15, '泄泻'),
            Entity('SYNDROME', 16, 18, '腹胀'),
            Entity('DISEASE', 18, 20, '便溏'),
            Entity('DRUG_GROUP', 21, 23, '咳嗽'),
            Entity('DRUG_EFFICACY', 23, 25, '痰多'),
        ]

        # get assigned labels
        labels = align_labels(sequence, tokens, entities)

        # assert output values are correct
        self.assertEqual(len(labels), len(tokens))

        self.assertEqual(labels[1], LABELS.index('B-SYMPTOM'))
        self.assertEqual(labels[2:5], [LABELS.index('I-SYMPTOM')] * 3)
        self.assertEqual(labels[11], LABELS.index('B-FOOD'))
        self.assertEqual(labels[12], LABELS.index('I-FOOD'))
        self.assertEqual(labels[13], LABELS.index('B-DRUG'))
        self.assertEqual(labels[14], LABELS.index('I-DRUG'))
        self.assertEqual(labels[16], LABELS.index('B-SYNDROME'))
        self.assertEqual(labels[17], LABELS.index('I-SYNDROME'))
        self.assertEqual(labels[18], LABELS.index('B-DISEASE'))
        self.assertEqual(labels[19], LABELS.index('I-DISEASE'))
        self.assertEqual(labels[21], LABELS.index('B-DRUG_GROUP'))
        self.assertEqual(labels[22], LABELS.index('I-DRUG_GROUP'))
        self.assertEqual(labels[23], LABELS.index('B-DRUG_EFFICACY'))
        self.assertEqual(labels[24], LABELS.index('I-DRUG_EFFICACY'))

    def test_split_into_blocks(self):
        token_ids = [1, 2, 3, 99, 4, 5, 6, 99, 7, 8, 99]

        blocks = list(split_into_blocks(
            token_ids=token_ids,
            separator_token_id=99,
            block_size=4,
        ))

        self.assertEqual(len(blocks), 3)

        start, end = blocks[0]
        self.assertEqual(token_ids[start: end], [1, 2, 3, 99])

        start, end = blocks[1]
        self.assertEqual(token_ids[start: end], [4, 5, 6, 99])

        start, end = blocks[2]
        self.assertEqual(token_ids[start: end], [7, 8, 99])

    def test_split_into_blocks2(self):
        token_ids = [1, 2, 3, 99, 4]

        blocks = list(split_into_blocks(
            token_ids=token_ids,
            separator_token_id=99,
            block_size=4,
        ))

        self.assertEqual(len(blocks), 2)

        start, end = blocks[0]
        self.assertEqual(token_ids[start: end], [1, 2, 3, 99])

        start, end = blocks[1]
        self.assertEqual(token_ids[start: end], [4])

    def test_extract_entities(self):
        label2id = dict(zip(LABELS, range(len(LABELS))))
        sequence = '凡脾胃虚弱,食入难化,呕吐泄泻,嘻嘻'
        entities = [0] * len(sequence)

        entities[1] = label2id['B-SYMPTOM']
        entities[2:5] = [label2id['I-SYMPTOM']] * (5 - 2)

        entities[11] = entities[13] = label2id['B-SYMPTOM']
        entities[12] = entities[14] = label2id['I-SYMPTOM']

        logits = np.eye(len(LABELS))[entities]
        result = extract_entities(
            sequence=sequence,
            logits=logits,
            encode=lambda x, **_: [ord(c) for c in x],
            decode=lambda x: ''.join(chr(d) for d in x),
        )

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], Entity('SYMPTOM', 1, 5, '脾胃虚弱'))
        self.assertEqual(result[1], Entity('SYMPTOM', 11, 13, '呕吐'))
        self.assertEqual(result[2], Entity('SYMPTOM', 13, 15, '泄泻'))

    def test_seqi_to_tokeni(self):
        sequence = ' 灯心草汤或温开水送服，两回6g，一日2～3次。  7g*5袋*3盒 miguel  '
        tokens = [
            '灯', '心', '草', '汤', '或', '温', '开', '水', '送', '服',
            '，', '两', '回', '6g', '，', '一', '日', '2～3', '次', '。',
            '7g', '*', '5', '袋', '*', '3', '盒', 'mi', '##g', '##uel'
        ]

        results = seqi_to_tokeni(sequence, tokens)

        self.assertEqual(len(results), len(sequence))
        self.assertIsNone(results[0])  # [1st space]
        self.assertEqual(tokens[results[1]], '灯')
        self.assertEqual(tokens[results[14]], '6g')
        self.assertEqual(tokens[results[15]], '6g')
        self.assertEqual(tokens[results[19]], '2～3')
        self.assertEqual(tokens[results[20]], '2～3')
        self.assertEqual(tokens[results[21]], '2～3')
        self.assertIsNone(results[24])  # [space in between]
        self.assertIsNone(results[25])  # [space in between]
        self.assertEqual(tokens[results[26]], '7g')
        self.assertEqual(tokens[results[27]], '7g')
        self.assertEqual(tokens[results[35]], 'mi')
        self.assertEqual(tokens[results[36]], 'mi')
        self.assertEqual(tokens[results[37]], '##g')
        self.assertEqual(tokens[results[38]], '##uel')
        self.assertEqual(tokens[results[39]], '##uel')
        self.assertEqual(tokens[results[40]], '##uel')

    def test_tokeni_to_seqi(self):
        sequence = ' 灯心草汤或温开水送服，两回6g，一日2～3次。  7g*5袋*3盒 miguel  '
        tokens = [
            '灯', '心', '草', '汤', '或', '温', '开', '水', '送', '服',
            '，', '两', '回', '6g', '，', '一', '日', '2～3', '次', '。',
            '7g', '*', '5', '袋', '*', '3', '盒', 'mi', '##g', '##uel'
        ]

        results = tokeni_to_seqi(tokens, sequence)

        self.assertEqual(len(results), len(tokens))
        self.assertEqual(
            [sequence[j] for j in results[0]],
            ['灯'],
        )
        self.assertEqual(
            [sequence[j] for j in results[1]],
            ['心'],
        )
        self.assertEqual(
            [sequence[j] for j in results[13]],
            ['6', 'g'],
        )
        self.assertEqual(
            [sequence[j] for j in results[17]],
            ['2', '～', '3'],
        )
        self.assertEqual(
            [sequence[j] for j in results[20]],
            ['7', 'g'],
        )
        self.assertEqual(
            [sequence[j] for j in results[27]],
            ['m', 'i'],
        )
        self.assertEqual(
            [sequence[j] for j in results[28]],
            ['g'],
        )
        self.assertEqual(
            [sequence[j] for j in results[29]],
            ['u', 'e', 'l'],
        )


if __name__ == '__main__':
    unittest.main()
