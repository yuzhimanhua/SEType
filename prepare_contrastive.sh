win=1
mode='train'
seed='seco_50'
python 0_preprocess_bio.py --mode ${mode} --seed ${seed}
python 1_preprocess_sent.py --mode ${mode}
python 2_prepare_test.py --mode ${mode} --win ${win}


mode='valid'
seed='seco_50'
python 0_preprocess_bio.py --mode ${mode} --seed ${seed}
python 1_preprocess_sent.py --mode ${mode}
python 2_prepare_test.py --mode ${mode} --win ${win}


mode='test'
seed='github'
python 0_preprocess_bio.py --mode ${mode} --seed ${seed}
python 1_preprocess_sent.py --mode ${mode}
python 2_prepare_test.py --mode ${mode} --win ${win}
# python 4_prompt_chatgpt.py