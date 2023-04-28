import json
import argparse

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--win', default=1, type=int)
parser.add_argument('--mode', default='train', type=str)
args = parser.parse_args()

win = args.win
with open(f'{args.mode}_document.json') as fin, open(f'{args.mode}_pair.json', 'w') as fout:
	for line in fin:
		data = json.loads(line)
		for idx in range(len(data['sentences'])):
			L = max(0, idx-win)
			R = min(len(data['sentences']), idx+win+1)
			text = ' '.join(data['sentences'][L:R])
			for tup in data['entities'][idx]:
				if tup[1] == None:
					continue
				out = {}
				out['text'] = text
				out['entity'] = tup[0]
				out['type'] = tup[1]
				fout.write(json.dumps(out)+'\n')