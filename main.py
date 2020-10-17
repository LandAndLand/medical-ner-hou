from utils import split_train_set
from finetune_train_val import finetune
from evaluate import evaluate

def main():
    # split train dataset
    split_train_set()
    # finetune model
    finetune()
    #evaluate
    evaluate()


if __name__ == '__main__':
    fire.Fire(main)
