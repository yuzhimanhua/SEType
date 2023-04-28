dataset=overflow
architecture=cross

python prepare.py --linking 0

python3 main.py --bert_model /shared/data2/yuz9/BERT_models/bertoverflow/ \
				--train_dir ${dataset}_input/ --test_file ${dataset}_input/test.txt \
				--output_dir ${dataset}_output/ --architecture ${architecture} --use_pretrain \
				--train_batch_size 8 --eval_batch_size 128 --num_train_epochs 2 \
				--max_contexts_length 462 --max_response_length 50 \
				--print_freq 500 --eval_freq 500
python3 main.py --bert_model /shared/data2/yuz9/BERT_models/bertoverflow/ \
				--train_dir ${dataset}_input/ --test_file ${dataset}_input/test.txt \
				--output_dir ${dataset}_output/ --architecture ${architecture} --use_pretrain --eval \
				--train_batch_size 8 --eval_batch_size 128 --num_train_epochs 2 \
				--max_contexts_length 462 --max_response_length 50 \
				--print_freq 500 --eval_freq 500

# python calc_f1.py


# python prepare_zero.py --linking 0

# python3 main.py --bert_model /shared/data2/yuz9/BERT_models/bertoverflow/ \
# 				--train_dir ${dataset}_input/ --test_file ${dataset}_input/test.txt \
# 				--output_dir ${dataset}_output/ --architecture ${architecture} --use_pretrain --eval \
# 				--train_batch_size 8 --eval_batch_size 128 --num_train_epochs 2 \
# 				--max_contexts_length 462 --max_response_length 50 \
# 				--print_freq 500 --eval_freq 500

# python calc_f1_zero.py
