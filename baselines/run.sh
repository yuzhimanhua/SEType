test=test_stackoverflow
python3 roberta_large_mnli.py --test ${test}
python3 bart_large_mnli.py --test ${test}

test=test_stackoverflow_newtype
python3 roberta_large_mnli.py --test ${test}
python3 bart_large_mnli.py --test ${test}

test=test_github
python3 roberta_large_mnli.py --test ${test}
python3 bart_large_mnli.py --test ${test}

test=test_github_newtype
python3 roberta_large_mnli.py --test ${test}
python3 bart_large_mnli.py --test ${test}
