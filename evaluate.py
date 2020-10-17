from constants import BERT_VARIANT
from modeling import classify
from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForTokenClassification

import fire
import logging

logging.basicConfig(
    filename='evaluate.logs',
    filemode='a',
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG,
)
logger = logging.getLogger()


def evaluate(dataset_dir: str = 'chusai_xuanshou', output_dir: str = 'chusai_xuanshou_results'):
    # initialize testing set and output directory path
    test_path = Path(dataset_dir)

    # 创建输出路径的文件夹，若该文件夹已经存在，不会报错
    output_path = Path(output_dir)
    assert test_path.is_dir()
    output_path.mkdir(parents=False, exist_ok=True)

    # initialize checkpoint path
    ckpt_path = Path(f'finetuned-{BERT_VARIANT}'.replace('/', '-'))
    assert ckpt_path.is_dir(), 'Finetune model first using finetune.py'

    # load finetuned model
    tokenizer = BertTokenizerFast.from_pretrained(BERT_VARIANT)
    model = BertForTokenClassification.from_pretrained(ckpt_path).eval()

    # gather all txt files inside the test dataset directory
    # in preparation for their evaluation
    # rglob 函数遍历 test_path 下的所有 .txt 文件
    # 相当于 test_path.glob('**/*.txt')
    files = list(test_path.rglob('*.txt'))

    # 遍历测试集的所有样本
    for f in tqdm(files):
        logger.info(f'Evaluation start on {f}')

        # load sample sequence
        # 读取测试集的样本，所以 sequence 中保存的是测试集的某单个样本内容
        # (此时还没有考虑测试集长度超过 512 的问题)
        with f.open('r', encoding='utf-8') as fd:
            sequence = fd.read()

        # get predicted entities
        entities = classify(model, tokenizer, sequence)
        
        # write predicted entities into a file
        # in preparation for submission
        # f.stem 是路径最后一个组件的去掉后缀的内容 ；
        # 例：f=Path(**/**/1056.txt), f.stem=1056
        result_path = output_path / (f.stem + '.ann')
        with result_path.open('w', encoding='utf-8') as fd:
            first = True
            for i, entity in enumerate(entities):
                # 第一行输出时不应该有空格
                t = f'T{i+1}' if first else f'\nT{i+1}'
                kind = entity.kind
                start = entity.start
                end = entity.end
                value = entity.value
                first = False
                fd.write(f'{t}\t{kind} {start} {end}\t{value}')


if __name__ == '__main__':
    fire.Fire(evaluate)
