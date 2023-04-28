import json
import argparse

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mode', default='train', type=str)
args = parser.parse_args()

sentences = []
entities = []
text_type = None
with open(f'{args.mode}_sentence.json') as fin, open(f'{args.mode}_document.json', 'w') as fout:
	for line in fin:
		data = json.loads(line)
		if data['sentence'].startswith('CODE_BLOCK :'):
			continue
		if data['sentence'].startswith('Question_URL :'):
			continue
		if data['sentence'].startswith('Question_ID :'):
			if len(sentences) != 0:
				out = {}
				out['type'] = text_type
				out['sentences'] = sentences
				out['entities'] = entities
				sentences = []
				entities = []
				fout.write(json.dumps(out)+'\n')
			text_type = 'Question'
		elif data['sentence'].startswith('Answer_to_Question_ID :'):
			if len(sentences) != 0:
				out = {}
				out['type'] = text_type
				out['sentences'] = sentences
				out['entities'] = entities
				sentences = []
				entities = []
				fout.write(json.dumps(out)+'\n')
			text_type = 'Answer'
		else:
			sentences.append(data['sentence'])
			entities.append(data['entities'])
			
	if len(sentences) != 0:
		out = {}
		out['type'] = text_type
		out['sentences'] = sentences
		out['entities'] = entities
		sentences = []
		entities = []
		fout.write(json.dumps(out)+'\n')