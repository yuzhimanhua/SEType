import json
import argparse

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--seed', default='stack', type=str)
args = parser.parse_args()

def change(x):
	if x == 'data_structure' or x == 'user_interface_element':
		x = 'data_type'
	if x == 'library_class':
		x = 'library'
	if x == 'website':
		x = 'application'
	if x == 'language':
		x = 'programming_language'
	return x

sentence = []
entities = []
entity = []
etypes = []
etype = None
with open(f'data/{args.mode}_{args.seed}.txt') as fin, open(f'{args.mode}_sentence.json', 'w') as fout:
	for line in fin:
		data = line.strip().split()
		if len(data) == 0:
			if len(sentence) > 0:
				sentence_text = ' '.join(sentence)
				sentence = []
			if len(entity) > 0:
				entities.append(' '.join(entity))
				etypes.append(change(etype))
				entity = []
				etype = None
			if len(sentence_text) > 0:
				out = {}
				out['sentence'] = sentence_text
				out['entities'] = []
				for x, y in zip(entities, etypes):	
					out['entities'].append([x, y])
				fout.write(json.dumps(out)+'\n')
				sentence_text = ''
				entities = []
				etypes = []
		else:
			sentence.append(data[0])
			if data[1][0] in ['O', 'B']:
				if len(entity) > 0:
					entities.append(' '.join(entity))
					etypes.append(change(etype))
					entity = []
					etype = None
			if data[1][0] == 'B':
				entity.append(data[0])
				etype = data[1][2:].lower()
			if data[1][0] == 'I':
				entity.append(data[0])
