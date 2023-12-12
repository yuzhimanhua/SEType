import json
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train', default='train_stackoverflow_pseudo', type=str)
parser.add_argument('--valid', default='valid_stackoverflow_pseudo', type=str)
parser.add_argument('--test', default='test_stackoverflow', type=str)
args = parser.parse_args()

# training and validation
train_labels = set()
with open(f'../tmp/{args.train}_pair.json') as fin:
	for line in fin:
		data = json.loads(line)
		y = data['type'].replace('_', ' ')
		train_labels.add(y)
train_labels = list(train_labels)
train_labels.sort()
print('Training types:', train_labels)

with open(f'../tmp/{args.train}_pair.json') as fin, open(f'../tmp/train.txt', 'w') as fout:
	for line in tqdm(fin):
		data = json.loads(line)
		entity = data['entity']
		text = data['text']
		y = data['type'].replace('_', ' ')
		for label in train_labels:
			if label == y:
				continue
			hp = f'In this context, {entity} is referring to {y}.'
			fout.write(f'1\t{text}\t{hp}\n')
			hn = f'In this context, {entity} is referring to {label}.'	
			fout.write(f'0\t{text}\t{hn}\n')
with open(f'../tmp/{args.valid}_pair.json') as fin, open(f'../tmp/valid.txt', 'w') as fout:
	for line in tqdm(fin):
		data = json.loads(line)
		entity = data['entity']
		text = data['text']
		y = data['type'].replace('_', ' ')
		for label in train_labels:
			if label == y:
				continue
			hp = f'In this context, {entity} is referring to {y}.'
			fout.write(f'1\t{text}\t{hp}\n')
			hn = f'In this context, {entity} is referring to {label}.'	
			fout.write(f'0\t{text}\t{hn}\n')

# testing
test_labels = set()
with open(f'../tmp/{args.test}_pair.json') as fin:
	for line in fin:
		data = json.loads(line)
		y = data['type'].replace('_', ' ')
		test_labels.add(y)
test_labels = list(test_labels)
test_labels.sort()
print('Testing types:', test_labels)

with open(f'../tmp/{args.test}_pair.json') as fin, open(f'../tmp/test.txt', 'w') as fout:
	for line in tqdm(fin):
		data = json.loads(line)
		entity = data['entity']
		text = data['text']
		for label in test_labels:
			hypothesis = f'in this context, {entity} is referring to {label}.'
			fout.write(f'1\t{text}\t{hypothesis}\n')
