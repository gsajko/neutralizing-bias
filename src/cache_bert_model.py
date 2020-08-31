from pytorch_pretrained_bert.tokenization import BertTokenizer
from shared.args import ARGS

tokenizer = BertTokenizer.from_pretrained(ARGS.bert_model)
