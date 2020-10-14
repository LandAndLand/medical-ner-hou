from dataset import MedicalNERDataset
from pathlib import Path
from utils import Entity

import shutil
import unittest


class TestDataset(unittest.TestCase):
    def setUp(self):
        # create sample dataset directory
        self.test_dir = Path('_test_dataset')
        self.test_dir.mkdir(parents=False, exist_ok=True)

    def tearDown(self):
        # delete sample dataset directory
        shutil.rmtree(self.test_dir)

    def test_iter_dataset(self):
        # write sample text
        txt_value = ' 清热，通淋。用于膀胱湿热，小便浑浊，淋沥作痛膀胱湿热' \
            ' 通化振霖药业有限责任公司  灯心草汤或温开水送服，一次6g，一日2～3次。' \
            '  6g*6袋*3盒 '
        txt_path = self.test_dir / '1.txt'
        with txt_path.open('w', encoding='utf-8') as fd:
            fd.write(txt_value)

        # write sample entity labels
        labels_value = '\n'.join([
            'T1	DRUG_EFFICACY 1 3	清热',
            'T2	DRUG_EFFICACY 4 6	通淋',
            'T3	SYMPTOM 14 18	小便浑浊',
            'T4	SYMPTOM 19 23	淋沥作痛',
        ])
        labels_path = self.test_dir / '1.ann'
        with labels_path.open('w', encoding='utf-8') as fd:
            fd.write(labels_value.strip())

        # create instance of Dataset class
        dataset = MedicalNERDataset(self.test_dir)

        # test returned results
        self.assertEqual(len(dataset), 1)

        fname, (seq, entities) = dataset[0]
        self.assertEqual(fname, self.test_dir / '1.txt')
        self.assertEqual(seq, txt_value)
        self.assertEqual(entities, [
            Entity('DRUG_EFFICACY', 1, 3, '清热'),
            Entity('DRUG_EFFICACY', 4, 6, '通淋'),
            Entity('SYMPTOM', 14, 18, '小便浑浊'),
            Entity('SYMPTOM', 19, 23, '淋沥作痛'),
        ])


if __name__ == '__main__':
    unittest.main()
