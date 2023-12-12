import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos
from collections import defaultdict as ddict
import pickle
from utils import *
import time
import argparse
from tqdm import tqdm
import json

def mean_co_expan(query_sets, target_size):
	print(f'start expanding {len(query_sets)} sets: ')
	for qi, s in enumerate(query_sets):
		print(f'Set {qi}: {[eid2name[eid] for eid in s]}')
	print()
	start_time = time.time()
	def g(query_sets):
		for q in query_sets:
			for eid in q:
				yield eid
	current_entities = set([eid for eid in g(query_sets)])
	expanded_terms = [[] for _ in query_sets]
	
	while np.min([len(s) for s in expanded_terms]) < target_size:
		scores = np.array([np.mean(cos(means[[eid2idx[eid] for eid in q+e]], means), axis=0)
						   for q,e in zip(query_sets,expanded_terms)])
		max_scores = np.max(scores, axis=0)
		num_new = 0
		for i in np.argsort(-max_scores):
			if keywords[i] in current_entities:
				continue
			to_add = np.argmax(scores[:, i])
			if len(expanded_terms[to_add]) >= target_size:
				continue
			expanded_terms[to_add].append(keywords[i])
			current_entities.add(keywords[i])
			num_new += 1
			if num_new == 10:
				break
		print('Current Sets')
		for qi, e in enumerate(expanded_terms):
			print(f'Set {qi} (size {len(e)}): {[eid2name[eid] for eid in e]}')
		print(f'time: {int((time.time() - start_time)/60)} min {int(time.time() - start_time)%60} sec')
		print()
	return expanded_terms

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', required=True, help='path to dataset folder')
	parser.add_argument('--seeds_dir', required=True, help='path to seeds folder')
	parser.add_argument('--target', default=50, type=int, help='target set size')
	parser.add_argument('--vocab', default='entity2id.txt', help='vocab file')
	parser.add_argument('--sent', default='sentences.json', help='sent file')
	parser.add_argument('--seeds', default='seeds_stackoverflow.txt', help='seeds file')
	parser.add_argument('--emb_file', default='pretrained_emb.npy_means.pkl', help='name of PLM MEAN embeddings file')
	parser.add_argument('--entity_pos_out', default='entity_pos.pkl', help='name of output entity index file')
	parser.add_argument('--dim', default=768, help='dimension of PLM embeddings')
	args = parser.parse_args()


	eid2name, name2eid, keywords, eid2idx = load_vocab(os.path.join(args.dataset, args.vocab))

	entity_pos = pickle.load(open(os.path.join(args.dataset, args.entity_pos_out), 'rb'))

	alter_name = ddict(set)
	with open(os.path.join(args.dataset, args.sent), 'r', encoding='utf-8') as f:
		for line in tqdm(f, total=get_num_lines(os.path.join(args.dataset, args.sent))):
			obj = json.loads(line)
			for entity in obj['entityMentions']:
				eid = entity['entityId']
				raw = '_'.join(obj['tokens'][entity['start']:entity['end']+1])
				alter_name[eid].add(raw)
			
	means = pickle.load(open(os.path.join(args.dataset, args.emb_file), 'rb'))
	print(means.shape)

	class_names, query_sets = load_queries(os.path.join(args.seeds_dir, args.seeds), name2eid)

	res = mean_co_expan(query_sets, args.target)

	final_res = [q+e for q, e in zip(query_sets, res)]
	with open(os.path.join(args.seeds_dir, f'secoexpan_{args.target}_{args.seeds}'), 'w') as f:
		t2names = {}
		for type_name, class_set in zip(class_names, final_res):
			names = set()
			for eid in class_set:
				names.add(eid2name[eid])
				if eid in alter_name:
					names.update(alter_name[eid])
			t2names[type_name] = names
		for type_name, names in t2names.items():
			names = names - set.union(*[n for t,n in t2names.items() if t != type_name])
			for n in names:
				print(f'{type_name}\t{n}', file=f)
