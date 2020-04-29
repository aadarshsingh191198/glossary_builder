import csv
import pandas as pd
from pathlib import Path
import matplotlib.cm as cm
import numpy as np

from typing import *

import torch
import torch.optim as optim

import spacy
nlp = spacy.load("en_core_web_sm")

import warnings
warnings.filterwarnings('ignore')

seed = 42

import random
random.seed(seed)

import torch
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(seed)

import numpy as np
np.random.seed(seed)

import os

model_name="bert-base-cased"

# %tensorflow_version 1.x 
from transformers import *
tokenizer = BertTokenizer.from_pretrained(model_name, output_attentions=True)

auto_model = BertForSequenceClassification.from_pretrained(model_name, output_attentions=True, num_labels=2)

auto_model.load_state_dict(torch.load('models/bert_cased_symbols_attn_97_4epochs.pth'))
auto_model.eval()
print(f'Done')

from flair.models import SequenceTagger
from flair.data import Sentence
# model = SequenceTagger.load('models/wcl_best-model.pt')
model = SequenceTagger.load('models/best-model.pt')

from keras.preprocessing.sequence import pad_sequences
from sklearn.utils.extmath import softmax

def tag_collector(tagged_string, tag):
  tagged_string = tagged_string.replace('B-','')
  tagged_string = tagged_string.replace('I-','')
  # print(tagged_string)
  tokenised_sent = tagged_string.split()
  i=0
  tag_word = []
  tagging = False
  while i < len(tokenised_sent):
    if not (tokenised_sent[i][0] == '<' and tokenised_sent[i][-1]=='>'):
      i+=1
      continue
    # print(tokenised_sent[i])
    if ('<' +  tag  + '>') == tokenised_sent[i]:
      # print('Hello3')
      tagging = True
      tag_word.append(tokenised_sent[i-1])
    elif tagging:
      # print('Hello')
      tagging = False
      return ' '.join(tag_word)
      # print('Hello2')
    i+=1
  return ' '.join(tag_word)

def definition_extraction(sentence, override_classification = False):
  input_ids = tokenizer.encode(sentence, return_tensors='pt', add_special_tokens=True)
  input_ids = pad_sequences(input_ids, maxlen=64, dtype="long", truncating="post", padding="post")

  attention_masks = []
  for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)
  input_ids = torch.tensor(input_ids).unsqueeze(0)
  attention_masks = torch.tensor(attention_masks).unsqueeze(0)
  outputs = auto_model(input_ids[0], attention_mask=attention_masks[0], token_type_ids=None)
  p = softmax(outputs[0].detach().numpy())
  if p[0][1]>0.5:
    # print('---------------------------------------------------------------------------')
    # print('Definition')
    sentence  = ' '.join([tok.text for tok in nlp(sentence)])
    sentence = Sentence(sentence)
    model.predict(sentence)
    tagged_sentence = sentence.to_tagged_string('ner')
    
    terms = tag_collector(tagged_sentence, 'Term')
    definitions = tag_collector(tagged_sentence, 'Definition')

    # print(tagged_sentence)
    # print('Term:' , terms, '\nDefinitions:', definitions)
    return terms, definitions

  elif override_classification:
    # print('---------------------------------------------------------------------------')
    # print('Tagged not Definition')
    sentence  = ' '.join([tok.text for tok in nlp(sentence)])
    sentence = Sentence(sentence)
    model.predict(sentence)
    tagged_sentence = sentence.to_tagged_string('ner')
    if 'Term' in tagged_sentence:
      terms = tag_collector(tagged_sentence, 'Term')
      definitions = tag_collector(tagged_sentence, 'Definition')
      # print(tagged_sentence)
      return terms, definitions
    else:
      return None, None
  else:
    return None, None
      
text = 'In chemistry, catenation is the bonding of atoms of the same element into a series, called a chain. A chain or a ring shape may be open if its ends are not bonded to each other (an open-chain compound), or closed if they are bonded in a ring (a cyclic compound). \
Catenation occurs most readily with carbon, which forms covalent bonds with other carbon atoms to form longer chains and structures. This is the reason for the presence of the vast number of organic compounds in nature. \
Carbon is most well known for its properties of catenation, with organic chemistry essentially being the study of catenated carbon structures (and known as catenae).\
 Carbon chains in biochemistry combine any of various other elements, such as hydrogen, oxygen, and biometals, onto the backbone of carbon, and proteins can combine multiple chains encoded by multiple genes (such as light chains and heavy chains making up antibodies). '

tokens = nlp(text)
text = [sent.string.strip() for sent in tokens.sents]
print(text)

print('INPUT PARAGRAPH')
print('---------------------------------------------------------------------------')
for sent in text:
  print(sent)

print('GLOSSARY')
print('---------------------------------------------------------------------------')

df= pd.DataFrame()
terms = []
for sent in text:
  term, definition = definition_extraction(sent, override_classification=True)
  if term and definition :
  # print(terms, definitions)
    df = df.append({'Term': term, 'Definition': definition}, ignore_index=True)
  elif term:
    terms.append(term)
with pd.option_context('max_colwidth',100):
  print(df[['Term','Definition']])
print('Terms without definitions', terms)
  # print('++++++++++++++++++++++++++++++++++++++++++++')


