from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import json
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import argparse

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--test', default='test_stackoverflow', type=str)
args = parser.parse_args()

device = 0
classifier = pipeline(task='sentiment-analysis', model='facebook/bart-large-mnli', device=device, return_all_scores=True)
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')

test_labels = set()
with open(f'../tmp/{args.test}_pair.json') as fin:
	for line in fin:
		data = json.loads(line)
		test_labels.add(data['type'])
test_labels = list(test_labels)
test_labels.sort()

max_paper_len = 460
max_label_len = 48
y = []
y_pred = []
with open(f'../tmp/{args.test}_pair.json') as fin, open('bart_prediction.json', 'w') as fout:
	for line in tqdm(fin):
		data = json.loads(line)

		text = data['text']
		tokens = tokenizer(text, truncation=True, max_length=max_paper_len)
		text = tokenizer.decode(tokens["input_ids"][1:-1])

		entity = data['entity']
		y.append(data['type'])
		score = {}
		for label in test_labels:
			label_text = label.replace('_', ' ')
			hypothesis = f'In this context, {entity} is referring to {label_text}.'
			tokens = tokenizer(hypothesis, truncation=True, max_length=max_label_len)
			hypothesis = tokenizer.decode(tokens["input_ids"][1:-1])
			input = f'{text} </s></s> {hypothesis}'
			output = classifier(input)
			score[label] = output[0][-1]['score']
		score_sorted = sorted(score.items(), key=lambda x:x[1], reverse=True)
		y_pred.append(score_sorted[0][0])

		data['prediction'] = score_sorted[0][0]
		fout.write(json.dumps(data)+'\n')

report = classification_report(y, y_pred, output_dict=True)
with open('bart_report.txt', 'a') as fout:
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
with open('score.txt', 'a') as fout:
	fout.write(str(mif1)+'\t'+str(maf1)+'\n')
