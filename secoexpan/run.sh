python3 pretrained_emb.py --dataset ../corpus/ --model /shared/data2/yuz9/BERT_models/bertoverflow/
python3 secoexpan.py --dataset ../corpus/ --seeds_dir ../data/ --target 50
python3 matching.py --seeds_dir ../data/ --target 50 --corpus train_stackoverflow
python3 matching.py --seeds_dir ../data/ --target 50 --corpus valid_stackoverflow
