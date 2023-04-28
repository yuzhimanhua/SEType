from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import json
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

# huggingface==4.0.0
device = 0
classifier = pipeline(task='sentiment-analysis', model='roberta-large-mnli', device=device, return_all_scores=True)
tokenizer = AutoTokenizer.from_pretrained('roberta-large-mnli')

labels = ['application', 'data_type', 'device', 'library', 'operating_system', 'programming_language']
articles = ['an', 'a', 'a', 'a', 'an', 'a']

max_paper_len = 450
max_label_len = 50
# max_paper_len = 250
# max_label_len = 250
y = []
y_pred = []
with open('test_pair.json') as fin, open('prediction.json', 'w') as fout:
	for line in tqdm(fin):
		data = json.loads(line)

		text = data['text']
		tokens = tokenizer(text, truncation=True, max_length=max_paper_len)
		text = tokenizer.decode(tokens["input_ids"][1:-1])

		entity = data['entity']
		y.append(data['type'])
		score = {}
		for article, label in zip(articles, labels):
			label_text = label.replace('_', ' ')
			# hypothesis = f'{entity} is {article} {label_text}.'
			hypothesis = f'In this context, {entity} is referring to {label_text}.'
			# hypothesis = data['text'].replace(entity, label_text)
			tokens = tokenizer(hypothesis, truncation=True, max_length=max_label_len)
			hypothesis = tokenizer.decode(tokens["input_ids"][1:-1])
			input = f'{text} </s></s> {hypothesis}'
			output = classifier(input)
			score[label] = output[0][-1]['score']
		score_sorted = sorted(score.items(), key=lambda x:x[1], reverse=True)
		y_pred.append(score_sorted[0][0])

		data['prediction'] = score_sorted[0][0]
		fout.write(json.dumps(data)+'\n')

report = classification_report(y, y_pred, target_names=labels, output_dict=True)
with open('report.txt', 'a') as fout:
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
	
print(f1_score(y, y_pred, average='micro'))
print(f1_score(y, y_pred, average='macro'))
