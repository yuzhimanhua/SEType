import json
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--linking', default=0, type=int)
args = parser.parse_args()

if args.linking == 0:
	suffix = ''
else:
	suffix = '_linked'

dataset = 'overflow'
if not os.path.exists(f'{dataset}_input/'):
	os.mkdir(f'{dataset}_input/')

# training
labels = ['application', 'data type', 'device', 'library', 'operating system', 'programming language']
with open('/shared/data3/yuz9/StackOverflowNER/MNLI/train_pair'+suffix+'.json') as fin, \
	 open(f'{dataset}_input/train.txt', 'w') as fout:
	for line in tqdm(fin):
		data = json.loads(line)
		entity = data['entity']
		
		if args.linking == 1 and len(data['linked']) > 0:
			des = data['linked']
			# text = f'{entity} is referring to {des}.'
			text = data['text'] + ' ' + f'{entity} is referring to {des}.'
		else:
			text = data['text']

		y = data['type'].replace('_', ' ')
		for label in labels:
			if label == y:
				continue
			hp = f'In this context, {entity} is referring to {y}.'
			fout.write(f'1\t{text}\t{hp}\n')
			hn = f'In this context, {entity} is referring to {label}.'	
			fout.write(f'0\t{text}\t{hn}\n')

# validation
with open('/shared/data3/yuz9/StackOverflowNER/MNLI/valid_pair'+suffix+'.json') as fin, \
	 open(f'{dataset}_input/dev.txt', 'w') as fout:
	for line in tqdm(fin):
		data = json.loads(line)
		entity = data['entity']
		
		if args.linking == 1 and len(data['linked']) > 0:
			des = data['linked']
			# text = f'{entity} is referring to {des}.'
			text = data['text'] + ' ' + f'{entity} is referring to {des}.'
		else:
			text = data['text']

		y = data['type'].replace('_', ' ')
		for label in labels:
			if label == y:
				continue
			hp = f'In this context, {entity} is referring to {y}.'
			fout.write(f'1\t{text}\t{hp}\n')
			hn = f'In this context, {entity} is referring to {label}.'	
			fout.write(f'0\t{text}\t{hn}\n')

# testing
with open('/shared/data3/yuz9/StackOverflowNER/MNLI/test_pair'+suffix+'.json') as fin, \
	 open(f'{dataset}_input/test.txt', 'w') as fout:
	for line in tqdm(fin):
		data = json.loads(line)
		entity = data['entity']

		if args.linking == 1 and len(data['linked']) > 0:
			des = data['linked']
			# text = f'{entity} is referring to {des}.'
			text = data['text'] + ' ' + f'{entity} is referring to {des}.'
		else:
			text = data['text']

		for label in labels:
			hypo = f'in this context, {entity} is referring to {label}.'
			fout.write(f'1\t{text}\t{hypo}\n')