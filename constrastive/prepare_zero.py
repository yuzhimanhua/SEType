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

# testing
labels = ['application', 'data type', 'device', 'library', 'operating system', 'programming language', \
		  'value', 'algorithm', 'file type', 'version', 'html xml tag']
with open('/shared/data3/yuz9/StackOverflowNER/MNLI/test_pair_zero'+suffix+'.json') as fin, \
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
