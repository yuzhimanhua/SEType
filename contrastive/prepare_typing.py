import json
import argparse

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='train_stackoverflow_pseudo', type=str)
parser.add_argument('--window_size', default=1, type=int)
args = parser.parse_args()

# Process the CoNLL-2003 format to sentences
sentence = []
entities = []
entity = []
etypes = []
etype = None
with open(f'../data/{args.dataset}.txt') as fin, open(f'../tmp/{args.dataset}_sentence.json', 'w') as fout:
	for line in fin:
		data = line.strip().split()
		if len(data) == 0:
			if len(sentence) > 0:
				sentence_text = ' '.join(sentence)
				sentence = []
			if len(entity) > 0:
				entities.append(' '.join(entity))
				etypes.append(etype)
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
					etypes.append(etype)
					entity = []
					etype = None
			if data[1][0] == 'B':
				entity.append(data[0])
				etype = data[1][2:].lower()
			if data[1][0] == 'I':
				entity.append(data[0])


# Merge sentences to documents
sentences = []
entities = []
with open(f'../tmp/{args.dataset}_sentence.json') as fin, open(f'../tmp/{args.dataset}_document.json', 'w') as fout:
	for line in fin:
		data = json.loads(line)
		if data['sentence'].startswith('CODE_BLOCK :') or \
		   data['sentence'].startswith('Question_URL :') or \
		   data['sentence'].startswith('Issue_Event_Link :'):
			continue
		if data['sentence'].startswith('Question_ID :') or \
		   data['sentence'].startswith('Answer_to_Question_ID :') or \
		   data['sentence'].startswith('Repository_Name :') or \
		   data['sentence'].startswith('Doc_ID :'):
			if len(sentences) != 0:
				out = {}
				out['sentences'] = sentences
				out['entities'] = entities
				sentences = []
				entities = []
				fout.write(json.dumps(out)+'\n')
		else:
			sentences.append(data['sentence'])
			entities.append(data['entities'])
			
	if len(sentences) != 0:
		out = {}
		out['sentences'] = sentences
		out['entities'] = entities
		sentences = []
		entities = []
		fout.write(json.dumps(out)+'\n')


# Prepare the entity typing format
win_sz = args.window_size
with open(f'../tmp/{args.dataset}_document.json') as fin, open(f'../tmp/{args.dataset}_pair.json', 'w') as fout:
	for line in fin:
		data = json.loads(line)
		for idx in range(len(data['sentences'])):
			L = max(0, idx - win_sz)
			R = min(len(data['sentences']), idx + win_sz + 1)
			text = ' '.join(data['sentences'][L:R])
			for tup in data['entities'][idx]:
				if tup[1] == None:
					continue
				out = {}
				out['text'] = text
				out['entity'] = tup[0]
				out['type'] = tup[1]
				fout.write(json.dumps(out)+'\n')
