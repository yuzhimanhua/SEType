import argparse
import os

def update_dict(d, toks, typ):
	if len(toks) > 1:
		if toks[0] not in d:
			d[toks[0]] = {}
		d[toks[0]] = update_dict(d[toks[0]], toks[1:], typ)
	else:
		d[toks[0]] = {'TYPE':typ}
	return d


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--seeds_dir', required=True, help='path to seeds folder')
	parser.add_argument('--target', default=50, type=int, help='target set size')
	parser.add_argument('--seeds', default='seeds_stackoverflow.txt', help='seeds file')
	parser.add_argument('--corpus', required=True, help='corpus for entity matching')
	args = parser.parse_args()

	if args.seeds == 'seeds_stackoverflow.txt':
		type_map = {
			'application': 'Application',
			'data_structure': 'Data_Structure',
			'data_type': 'Data_Type',
			'device': 'Device',
			'library': 'Library',
			'library_class': 'Library_Class',
			'operating_system': 'Operating_System',
			'programming_language': 'Programming_Language',
			'user_interface_element': 'User_Interface_Element',
			'website': 'Website'
			}
	elif args.seeds == 'seeds_nvd.txt':
		type_map = {
			'application': 'application',
			'edition': 'edition',
			'operating_system': 'operating_system',
			'relevant_term': 'relevant_term',
			'vendor': 'vendor'
			}

	dictionary = {}
	with open(os.path.join(args.seeds_dir, f'secoexpan_{args.target}_{args.seeds}')) as f:
		for line in f:
			typ, ent = line.strip().split('\t')
			tok_ent = ent.split('_')
			dictionary = update_dict(dictionary, tok_ent, typ)

	all_sentences = []
	with open(os.path.join(args.seeds_dir, f'{args.corpus}.txt')) as f:
		sent = []
		for line in f:
			if line.strip() == '':
				all_sentences.append(sent)
				sent = []
			else:
				sent.append(line.strip())


	output = []
	total_matched = 0
	for sent in all_sentences:
		i = 0
		out_sent = []
		lower_sent = [tok.lower() for tok in sent]
		while i < len(sent):
			tok = lower_sent[i]
			if tok not in dictionary:
				out_sent.append((sent[i], 'O'))
				i += 1
			else:
				d = dictionary[tok]
				found_types = []
				for j in range(1, len(sent) - i):
					if lower_sent[i+j] in d:
						d = d[lower_sent[i+j]]
						if j == len(sent) - i - 1:
							if 'TYPE' in d:
								typ = type_map[d['TYPE']]
								found_types.append((j+1, typ))
							break
						continue
					if 'TYPE' in d:
						typ = type_map[d['TYPE']]
						found_types.append((j, typ))
					break
				if len(found_types) == 0:
					out_sent.append((sent[i], 'O'))
					i += 1
					continue
				total_matched += 1
				j, typ = found_types[-1]
				for ii in range(j):
					if ii == 0:
						out_sent.append((sent[i], f'B-{typ}'))
					else:
						out_sent.append((sent[i+ii], f'I-{typ}'))
				i += j
		output.append(out_sent)
	# print(total_matched)

	with open(os.path.join(args.seeds_dir, f'{args.corpus}_pseudo.txt'), 'w') as f:
		for sent, ori_sent in zip(output, all_sentences):
			assert len(sent) == len(ori_sent)
			for tok, typ in sent:
				print(f'{tok} {typ}', file=f)
			print('\t', file=f)
