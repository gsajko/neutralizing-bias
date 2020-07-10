import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer

import os
import numpy as np

import spacy

import sys; sys.path.append('.')
from shared.data import get_dataloader
from shared.args import ARGS
from shared.constants import CUDA

import seq2seq.model as seq2seq_model
import seq2seq.utils as seq2seq_utils

import tagging.model as tagging_model
import tagging.utils as tagging_utils

import joint.model as joint_model
import joint.utils as joint_utils

from flask import Flask, jsonify, request

# # # # # # # # # # SETTINGS # # # # # # # # # # # # 

nlp = spacy.load("en_core_web_sm")

working_dir = 'INFERENCE'
test_file = 'biased.test'
checkpoint = 'model.ckpt'
inference_output = 'INFERENCE/output.txt'

# # # # # # # # ## # # # ## # # DATA # # # # # # # # ## # # # ## # #
tokenizer = BertTokenizer.from_pretrained(ARGS.bert_model, cache_dir= working_dir +'/cache')
tok2id = tokenizer.vocab
tok2id['<del>'] = len(tok2id)


# # # # # # # # ## # # # ## # # MODEL # # # # # # # # ## # # # ## # #

debias_model = seq2seq_model.PointerSeq2Seq(
        vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
        emb_dim=768, dropout=0.2, tok2id=tok2id) # 768 = bert hidden size

tagging_model = tagging_model.BertForMultitaskWithFeaturesOnTop.from_pretrained(
            ARGS.bert_model,
            cls_num_labels=ARGS.num_categories,
            tok_num_labels=ARGS.num_tok_labels,
            cache_dir=working_dir + '/cache',
            tok2id=tok2id)

joint_model = joint_model.JointModel(
    debias_model=debias_model, tagging_model=tagging_model)

if CUDA:
  joint_model = joint_model.cuda()

if checkpoint is not None and os.path.exists(checkpoint):
  print('LOADING FROM ' + checkpoint)
  # TODO(rpryzant): is there a way to do this more elegantly? 
  # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices
  if CUDA:
    joint_model.load_state_dict(torch.load(checkpoint))
    joint_model = joint_model.cuda()
  else:
    joint_model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
  print('...DONE')

# # # # # # # # # # # # EVAL MODE & UTIL METHODS # # # # # # # # # # # # # #
joint_model.eval()

def transform_input(url, headline):
  tokenized = tokenizer.tokenize(headline)
  tokens = nlp(' '.join(tokenized))
  print(tokens, flush=True)
  pos = map(lambda t: t.pos_, tokens)
  deps = map(lambda t: t.dep_, tokens)
  print(deps, flush=True)
  final = url + '\t' + (' '.join(tokenized) + '\t') * 2
  final += headline + '\t' + headline + '\t'
  final += ' '.join(pos) + '\t'
  final += ' '.join(deps)
  print(final, flush=True)
  with open(test_file, 'w') as filetowrite:
    filetowrite.write(final)


def load_data():
  eval_dataloader, num_eval_examples = get_dataloader(
    test_file,
    tok2id, ARGS.test_batch_size, working_dir + '/test_data.pkl',
    test=True, add_del_tok=ARGS.add_del_tok)
  print(eval_dataloader)
  print(num_eval_examples, flush=True)
  return eval_dataloader

def predict(dataloader):
  hits, preds, golds, srcs = joint_utils.run_eval(
    joint_model, dataloader, tok2id, inference_output,
    ARGS.max_seq_len, ARGS.beam_width)

  print(hits, flush=True)
  print(preds, flush=True)
  print(golds, flush=True)
  return preds

# # # # # # # # # Server # # # # # # # # # # 

app = Flask(__name__)

@app.route('/', methods=['GET'])
def root_route():
  return jsonify({'msg' : 'Try POSTing to the /predict endpoint with a url and headline text'})

@app.route('/test', methods=['GET'])
def test_route():
  transform_input('https://google.com/', "tokenizer tries to tell trump to back off of dhruv")
  dataloader = load_data()
  prediction = predict(dataloader)
  print(prediction, flush=True)
  
  return jsonify({'unbiased': prediction})

@app.route('/predict', methods=['POST'])
def predict_route():
  if request.method == 'POST':
    req_data = request.get_json()
    
    url = req_data['url']
    headline = req_data['headline']

    transform_input(url, headline)
    dataloader = load_data()
    prediction = predict(dataloader)
    
    return jsonify({'unbiased': prediction})

if __name__ == '__main__':
  app.run(debug=True,host='0.0.0.0', port=5000)
  print("INFO: Server started")
