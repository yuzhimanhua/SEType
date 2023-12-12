import json
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import argparse

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--test', default='test_stackoverflow', type=str)
args = parser.parse_args()

y = []
test_labels = set()
with open(f'../tmp/{args.test}_pair.json') as fin:
	for line in fin:
		data = json.loads(line)
		y.append(data['type'])
		test_labels.add(data['type'])
test_labels = list(test_labels)
test_labels.sort()

y_pred = []
scores = []
with open(f'../output/prediction_cross.txt') as fin:
	for line in fin:
		data = float(line.strip())
		scores.append(data)
		if len(scores) % len(test_labels) == 0:
			max_score = max(scores)
			max_idx = scores.index(max_score)
			y_pred.append(test_labels[max_idx])
			scores = []

with open(f'../tmp/{args.test}_pair.json') as fin, open('../output/prediction.json', 'w') as fout:
	for line, pred in zip(fin, y_pred):
		data = json.loads(line)
		data['prediction'] = pred
		fout.write(json.dumps(data)+'\n')

report = classification_report(y, y_pred, output_dict=True)
with open('../output/report.txt', 'a') as fout:
	for x in report:
		if x == 'accuracy':
			f1 = report[x]
			out = f'Micro-F1\t\t\t{f1}\n'
		elif x == 'macro avg':
			f1 = report[x]['f1-score']
			out = f'Macro-F1\t\t\t{f1}\n'
		else:
			prec = report[x]['precision']
			rec = report[x]['recall']
			f1 = report[x]['f1-score']
			supp = report[x]['support']
			out = f'{x}\t{prec}\t{rec}\t{f1}\t{supp}\n'
		fout.write(out)
	fout.write('\n')
	
mif1 = f1_score(y, y_pred, average='micro')
maf1 = f1_score(y, y_pred, average='macro')
print(mif1, maf1)
with open('../output/score.txt', 'a') as fout:
	fout.write(str(mif1)+'\t'+str(maf1)+'\n')
