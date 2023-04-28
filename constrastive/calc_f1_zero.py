import json
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

dataset = 'overflow'
labels = ['application', 'data_type', 'device', 'library', 'operating_system', 'programming_language', \
		  'value', 'algorithm', 'file_type', 'version', 'html_xml_tag']
y = []
with open('/shared/data3/yuz9/StackOverflowNER/MNLI/test_pair_zero.json') as fin:
	for line in fin:
		data = json.loads(line)
		y.append(data['type'])

y_pred = []
pred = []
with open(f'{dataset}_output/prediction_cross.txt') as fin, open('result.json', 'w') as fout:
	for line in fin:
		data = float(line.strip())
		pred.append(data)
		if len(pred) % 11 == 0:
			fout.write(json.dumps(pred)+'\n')
			max_sim = max(pred)
			max_idx = pred.index(max_sim)
			y_pred.append(labels[max_idx])
			pred = []

report = classification_report(y, y_pred, output_dict=True)
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
