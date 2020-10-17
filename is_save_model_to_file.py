from pathlib import Path
from constants import BERT_VARIANT
from modeling import classify


def is_save_model_to_file(f1_higgest, dataset_dir: str = 'split_val'):
    val_path = Path(dataset_dir)
    files = list(val_path.rglob('*.txt'))
    f1_higgest = f1_higgest
    # 遍历验证集的所有样本

    # 预测指标：acc, 预测正确的样本; num_labels, 标签数; num_pres, 预测的标签个数
    acc = 0
    num_labels = 0
    num_pres = 0

    for f in tqdm(files):
        logger.info(f'Evaluation start on {f}')
        # load sample sequence
        # 读取测试集的样本，所以 sequence 中保存的是测试集的某单个样本内容
        # (此时还没有考虑测试集长度超过 512 的问题)
        with f.open('r', encoding='utf-8') as fd:
            sequence = fd.read()
        # get predicted entities
        entities = classify(model, tokenizer, sequence)
        result_path = val_path / (f.stem + '.ann')
        labels = []
        with result_path.open('r', encoding='utf-8') as fd:
            while (fd.readline()):
                line = list(fd.readline().split())
                # print(line)
                labels.append(line)
        print(labels)
        num_pres = len(entities)
        num_labels += len(labels)
        print(f'目前预测总数:{num_pres}')
        print(f'目前标签总数:{num_labels}')
        for i, entity in enumerate(entities):
            pre_kind = entity.kind
            pre_start = entity.start
            pre_end = entity.end
            pre_value = entity.value

            for label in labels:
                if pre_kind == label[0] and pre_start == label[1] and pre_end == label[2] and pre_value == label[3]:
                    acc += 1
    precision = acc / num_pres
    recall = acc / num_labels
    f1 = 2(precision*recall) / (precision + recall)

    if f1 > f1_higgest:
        #新的epoch训练得到的model的f1score有提高
        return True, f1, f1
    else:
        return False, f1 ,f1_higgest


