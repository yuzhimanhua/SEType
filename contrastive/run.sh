train=train_stackoverflow_pseudo
valid=valid_stackoverflow_pseudo
test=test_stackoverflow
window_size=$1
# window_size=1

python3 prepare_typing.py --dataset ${train} --window_size ${window_size}
python3 prepare_typing.py --dataset ${valid} --window_size ${window_size}
python3 prepare_typing.py --dataset ${test} --window_size ${window_size}

python3 prepare_contrastive.py --train ${train} --valid ${valid} --test ${test}
python3 main.py --bert_model /shared/data2/yuz9/BERT_models/bertoverflow/ \
				--train_dir ../tmp/ --test_file ../tmp/test.txt \
				--output_dir ../output/ --architecture cross --use_pretrain \
				--train_batch_size 4 --eval_batch_size 64 --num_train_epochs 1 \
				--max_contexts_length 462 --max_response_length 50 \
				--print_freq 500 --eval_freq 500
python3 main.py --bert_model /shared/data2/yuz9/BERT_models/bertoverflow/ \
				--train_dir ../tmp/ --test_file ../tmp/test.txt \
				--output_dir ../output/ --architecture cross --use_pretrain --eval \
				--train_batch_size 4 --eval_batch_size 64 --num_train_epochs 1 \
				--max_contexts_length 462 --max_response_length 50 \
				--print_freq 500 --eval_freq 500
python3 calc_f1.py --test ${test}


test=test_stackoverflow_newtype
python3 prepare_typing.py --dataset ${test} --window_size ${window_size}
python3 prepare_contrastive.py --train ${train} --valid ${valid} --test ${test}
python3 main.py --bert_model /shared/data2/yuz9/BERT_models/bertoverflow/ \
				--train_dir ../tmp/ --test_file ../tmp/test.txt \
				--output_dir ../output/ --architecture cross --use_pretrain --eval \
				--train_batch_size 4 --eval_batch_size 64 --num_train_epochs 1 \
				--max_contexts_length 462 --max_response_length 50 \
				--print_freq 500 --eval_freq 500
python3 calc_f1.py --test ${test}


test=test_github
python3 prepare_typing.py --dataset ${test} --window_size ${window_size}
python3 prepare_contrastive.py --train ${train} --valid ${valid} --test ${test}
python3 main.py --bert_model /shared/data2/yuz9/BERT_models/bertoverflow/ \
				--train_dir ../tmp/ --test_file ../tmp/test.txt \
				--output_dir ../output/ --architecture cross --use_pretrain --eval \
				--train_batch_size 4 --eval_batch_size 64 --num_train_epochs 1 \
				--max_contexts_length 462 --max_response_length 50 \
				--print_freq 500 --eval_freq 500
python3 calc_f1.py --test ${test}


test=test_github_newtype
python3 prepare_typing.py --dataset ${test} --window_size ${window_size}
python3 prepare_contrastive.py --train ${train} --valid ${valid} --test ${test}
python3 main.py --bert_model /shared/data2/yuz9/BERT_models/bertoverflow/ \
				--train_dir ../tmp/ --test_file ../tmp/test.txt \
				--output_dir ../output/ --architecture cross --use_pretrain --eval \
				--train_batch_size 4 --eval_batch_size 64 --num_train_epochs 1 \
				--max_contexts_length 462 --max_response_length 50 \
				--print_freq 500 --eval_freq 500
python3 calc_f1.py --test ${test}
