import mmap


def get_num_lines(file_path):
	fp = open(file_path, "r+")
	buf = mmap.mmap(fp.fileno(), 0)
	lines = 0
	while buf.readline():
		lines += 1
	return lines


def load_vocab(filename):
	eid2name = {}
	name2eid = {}
	keywords = []
	with open(filename, encoding='utf-8') as f:
		for line in f:
			temp = line.strip().split('\t')
			eid = int(temp[1])
			name = temp[0]
			eid2name[eid] = name
			name2eid[name] = eid
			keywords.append(eid)
	eid2idx = {w:i for i, w in enumerate(keywords)}
	print(f'Vocabulary: {len(keywords)} keywords loaded')
	return eid2name, name2eid, keywords, eid2idx

def load_queries(filename, name2eid):
	class_names = []
	query_sets = []
	with open(filename) as f:
		for line in f:
			temp = line.strip().split(' ')
			class_names.append(temp[0])
			query_set = [name2eid[x] for x in temp[1:]]
			query_sets.append(query_set)
	return class_names, query_sets
