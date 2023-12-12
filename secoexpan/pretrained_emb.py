import os
import torch
from transformers import BertTokenizerFast, BertModel
import numpy as np
import argparse
import pickle
import mmap
from tqdm import tqdm
import json

def get_num_lines(file_path):
	fp = open(file_path, "r+")
	buf = mmap.mmap(fp.fileno(), 0)
	lines = 0
	while buf.readline():
		lines += 1
	return lines


def load_vocab(filename):
	eid2name = {}
	keywords = []
	with open(filename, encoding='utf-8') as f:
		for line in f:
			temp = line.strip().split('\t')
			eid = int(temp[1])
			eid2name[eid] = temp[0]
			keywords.append(eid)
	eid2idx = {w:i for i, w in enumerate(keywords)}
	print(f'Vocabulary: {len(keywords)} keywords loaded')
	return eid2name, keywords, eid2idx

def get_masked_sentences(filename, tokenizer, eid2idx):
	masked_sentences = []
	sentences = []
	total_line = get_num_lines(filename)
	entity_num = [0 for _ in eid2idx]
	with open(filename, 'r', encoding='utf-8') as f:
		for line in tqdm(f, total=total_line):
			obj = json.loads(line)
			if len(obj['entityMentions']) == 0 or len(obj['tokens']) > 509:
				continue
			raw_sent = tokenizer([token.lower() for token in obj['tokens']], add_special_tokens=False)['input_ids']
			if sum([len(t) for t in raw_sent]) > 509:
				continue
			for entity in obj['entityMentions']:
				eid = entity['entityId']
				if eid not in eid2idx:
					continue
				entity_num[eid2idx[eid]] += 1
				center_ids = [i for ids in raw_sent[entity['start']:entity['end']+1] for i in ids]
				left_ids = [i for ids in raw_sent[:entity['start']] for i in ids]
				right_ids = [i for ids in raw_sent[entity['end']+1:] for i in ids]
				sentences.append((eid, [tokenizer.cls_token_id] + left_ids + center_ids + right_ids + [tokenizer.sep_token_id], len(left_ids)+1, len(left_ids)+len(center_ids)+1))
				masked_sentences.append((eid, [tokenizer.cls_token_id] + left_ids + [tokenizer.mask_token_id] + right_ids + [tokenizer.sep_token_id], len(left_ids)+1, len(left_ids)+2))
	print(f'Sentences: {len(sentences)} sentences constructed')
	entity_pos = np.cumsum(entity_num)
	entity_pos = [0] + list(entity_pos)
	return sentences, masked_sentences, entity_pos

def get_pretrained_emb(model, tokenizer, sentences, entity_pos, eid2idx, np_file, dim=768, batch_size=128, device='cuda'):
	fp = np.memmap(np_file, dtype='float32', mode='w+', shape=(entity_pos[-1], dim))
	ptr_list = [0 for _ in entity_pos[:-1]]
	iterations = int(len(sentences)/batch_size) + (0 if len(sentences) % batch_size == 0 else 1)
	for i in tqdm(range(iterations)):
		start = i * batch_size
		end = min((i+1)*batch_size, len(sentences))
		batch_ids = [ids for _,ids,_,_ in sentences[start:end]]
		batch_max_length = max(len(ids) for ids in batch_ids)
		ids = torch.tensor([ids + [0 for _ in range(batch_max_length - len(ids))] for ids in batch_ids]).long()
		masks = (ids != 0).long()
		ids = ids.to(device)
		masks = masks.to(device)
		with torch.no_grad():
			batch_final_layer = model(ids, masks)[0]
		for final_layer, (eid,_,s,e) in zip(batch_final_layer, sentences[start:end]):
			rep = np.mean(final_layer[s:e].cpu().numpy(), axis=0)
			this_idx = entity_pos[eid2idx[eid]] + ptr_list[eid2idx[eid]]
			ptr_list[eid2idx[eid]] += 1
			fp[this_idx] = rep.astype(np.float32)
	del fp


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', required=True, help='path to dataset folder')
	parser.add_argument('--vocab', default='entity2id.txt', help='vocab file')
	parser.add_argument('--sent', default='sentences.json', help='sent file')
	parser.add_argument('--npy_out', default='pretrained_emb.npy', help='name of output npy file')
	parser.add_argument('--entity_pos_out', default='entity_pos.pkl', help='name of output entity index file')
	parser.add_argument('--model', default='bertoverflow', help='PLM checkpoint')
	parser.add_argument('--dim', default=768, help='PLM embedding dimension')
	args = parser.parse_args()
		
	model_path = args.model
	device = torch.device(0)

	tokenizer = BertTokenizerFast.from_pretrained(model_path, do_lower_case = False)
	model = BertModel.from_pretrained(model_path)
	model.to(device)
	model.eval()

	_, keywords, eid2idx = load_vocab(os.path.join(args.dataset, args.vocab))
	sentences, masked_sentences, entity_pos = get_masked_sentences(os.path.join(args.dataset, args.sent), tokenizer, eid2idx)

	pickle.dump(entity_pos, open(os.path.join(args.dataset, args.entity_pos_out), 'wb'))
	# pickle.dump(sentences, open(os.path.join(args.dataset, 'sentences.pkl'), 'wb'))
	# pickle.dump(masked_sentences, open(os.path.join(args.dataset, 'masked_sentences.pkl'), 'wb'))
	get_pretrained_emb(model, tokenizer, masked_sentences, entity_pos, eid2idx, np_file=os.path.join(args.dataset, args.npy_out), device=device, dim=args.dim)

	pretrained_emb = np.memmap(os.path.join(args.dataset, args.npy_out), dtype='float32', mode='r', shape=(entity_pos[-1], args.dim))

	def get_emb_iter():
		for i in range(len(keywords)):
			yield pretrained_emb[entity_pos[i]:entity_pos[i+1]]

	means = np.array([np.mean(emb, axis=0) for emb in tqdm(get_emb_iter(), total=len(keywords))])
	pickle.dump(means, open(os.path.join(args.dataset, args.npy_out+'_means.pkl'), 'wb'))
