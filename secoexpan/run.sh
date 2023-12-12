target=$1
seeds=seeds_stackoverflow.txt

# python3 pretrained_emb.py --dataset ../corpus/ --model /shared/data2/yuz9/BERT_models/bertoverflow/
python3 secoexpan.py --dataset ../corpus/ --seeds_dir ../data/ --target ${target} --seeds ${seeds}
python3 matching.py --seeds_dir ../data/ --target ${target} --seeds ${seeds} --corpus train_stackoverflow
python3 matching.py --seeds_dir ../data/ --target ${target} --seeds ${seeds} --corpus valid_stackoverflow
