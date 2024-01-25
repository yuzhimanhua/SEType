# Seed-Guided Fine-Grained Entity Typing in Science and Engineering Domains

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

In this repository, we implement an entity typing framework in specific science and engineering domains (e.g., software engineering and security), which takes only a few (e.g., 5) seed entities of each type as supervision and can be applied to classify entities in a sentence to both seen and unseen types.

The framework has two modules: **entity set co-expansion** and **contrastive learning**.

## Links

- [Installation](#installation)
- [Entity Set Co-Expansion](#entity-set-co-expansion)
- [Contrastive Learning](#contrastive-learning)
- [Data](#data)
- [Citation](#citation)

## Installation

A GPU is required to run both modules. We use one NVIDIA RTX A6000 GPU in our experiments, where the entity set co-expansion module takes ~8.5 hours, and the contrastive learning module takes ~1 hour.

### Dependency
The code is written in Python 3.6. You can install the dependencies as follows:
```
cd contrastive/
./setup.sh
```

## Entity Set Co-Expansion

### Corpus Preprocessing
You need a large corpus to run entity set co-expansion. In our experiments, we use the training and validation sets (text only, no labels) of the [**StackOverflowNER**](https://github.com/jeniyat/StackOverflowNER) dataset and a sampled subcorpus of the [**StackExchange**](https://data.stackexchange.com/) data dump. You can download them [**here**](https://drive.google.com/file/d/1CUaT91fpTzPrZ4ixwmWrcv0_U4u246C9/view?usp=share_link) (StackOverflowNER, ~1.1MB) and [**here**](https://drive.google.com/file/d/1vAp-8BdsrJc2bWGeZCeLjd2bEZpqwUrj/view?usp=share_link) (StackExchange, ~500MB), respectively.

After merging the two downloaded corpora, please refer to [**HiExpan**](https://github.com/mickeystroller/HiExpan) for the preprocessing code to get ```entity2id.txt``` and ```sentences.json```. 

**NOTE: You need to lowercase the merged corpus before preprocessing.**

You can also used our preprocessed files ```entity2id.txt``` and ```sentences.json```, which can be downloaded [**here**](https://drive.google.com/file/d/1Q_K_LsbP0472CRXTop8NrhYKSBIwFbex/view?usp=share_link) (```entity2id.txt```, ~0.8MB) and [**here**](https://drive.google.com/file/d/1xmeTNtaTBjv5YacrTP5ZjNwQR_0nZ5Eh/view?usp=share_link) (```sentences.json```, ~3.7GB), respectively.

Please create a folder ```./corpus``` and put the two preprocessed files under it.


### Running
You can run the entity set co-expansion module using the following script:
```
cd secoexpan/
./run.sh
```

**NOTE: We use [BERTOverflow](https://huggingface.co/jeniya/BERTOverflow) as the pre-trained language model in this module. You can follow our choice or use any other BERT-based language model. Either way, please change the ```--model``` argument in ```secoexpan/run.sh``` accordingly.**

After running the code, you will find the entity set co-expansion result in ```data/seeds_secoexpan_50.txt``` in the following format:
```
application	nvda
application	opera_mini
application	microsoft_edge
...
data_structure	lists
data_structure	queues
data_structure	multimap
...
data_type	strings
data_type	scalars
data_type	int
...
```

You will also find the pseudo training and validation sets in ```data/train_stackoverflow_pseudo.txt``` and ```data/valid_stackoverflow_pseudo.txt```, respectively, in the following CoNLL-2003 BIO format:
```
I O
've O
been O
trying O
to O
learn O
Sencha B-Library
Touch I-Library
and O
...
```
The pseudo labels are obtained by matching expanded entities with unlabeled text. 


## Contrastive Learning
You can run the contrastive learning module using the following script:
```
cd contrastive/
./run.sh
```

**NOTE: We use [BERTOverflow](https://huggingface.co/jeniya/BERTOverflow) as the pre-trained language model in this module. You can follow our choice or use any other BERT-based language model. Either way, please change the ```--bert_model``` argument in ```contrastive/run.sh``` accordingly.**

After running the code, you will find the entity typing result in ```output/prediction.json```, where each line is a json record:
```
{
  "text": "I am using custom adapter which I use for my ListView . After creating ArrayList",
  "entity": "ListView",
  "type": "library_class",
  "prediction": "library_class"
}
```

You will also find the precision, recall, F1, and the number of testing samples of each entity type, as well as the overall Micro-F1 and Macro-F1 scores in ```output/report.txt```:
```
application	0.8827160493827161	0.35135135135135137	0.5026362038664323	407
data_structure	0.7692307692307693	0.728744939271255	0.7484407484407485	247
data_type	0.6549295774647887	0.8378378378378378	0.7351778656126482	111
device	0.19844357976653695	0.9622641509433962	0.32903225806451614	53
library	0.7110091743119266	0.603112840466926	0.6526315789473685	257
library_class	0.797979797979798	0.4146981627296588	0.5457685664939551	381
operating_system	0.6265060240963856	0.7878787878787878	0.697986577181208	66
programming_language	0.7836538461538461	0.9157303370786517	0.844559585492228	178
user_interface_element	0.641851106639839	0.8985915492957747	0.7488262910798122	355
website	0.32941176470588235	0.9655172413793104	0.49122807017543857	29
Micro-F1			0.6439539347408829
Macro-F1			0.6296287745354355
```

## Data
In the ```data/``` folder, there are several input files. If you need to run our code on your own dataset, you need to prepare these files.

(1) ```seeds.txt```: Each line has a type name followed by the seed entities of that type. **Make sure the provided seeds appear in ```corpus/entity2id.txt```.**
```
application browser mysql git chrome excel visual_studio
data_structure array list table queue
data_type string strings integer pointer character char
device mac port mouse cpu disk
library jquery api angular django spring
library_class exception datagrid asynctask arraylist set
operating_system windows android linux ios ubuntu
programming_language javascript java php html c++ sql python
user_interface_element button image page cell menu window
website google github microsoft facebook stackoverflow
```

(2) ```train_stackoverflow.txt``` and ```valid_stackoverflow.txt```: The unlabeled training and validation sets from the [**StackOverflowNER**](https://github.com/jeniyat/StackOverflowNER) dataset. Each line is a token. Sentences are separated by an empty line. 
```
Question_ID
:
37985879

Question_URL
:
https://stackoverflow.com/questions/37985879/

If
I
would
have
2
tables

...
```
You can rename your own training and validation sets. Make sure you change the ```--corpus``` argument in ```secoexpan/run.sh``` and the ```train``` and ```valid``` arguments in ```contrastive/run.sh``` accordingly.

(3) ```test_stackoverflow.txt```, ```test_stackoverflow_newtype.txt```, ```test_github.txt```, and ```test_github_newtype.txt```: Labeled testing sets from the [**StackOverflowNER**](https://github.com/jeniyat/StackOverflowNER) dataset. 

The two files with "stackoverflow" are texts from StackOverflow QA threads; the two files with "github" are texts from GitHub issues. 

The two files without "newtype" contain ground-truth labels of 10 seen types (i.e., for which 5-7 seed entities are provided in ```seeds.txt```, including **application**, **data structure**, **data type**, **device**, **library**, **library class**, **operating system**, **programming language**, **user interface element**, and **website**); the two files with "newtype" contain ground-truth labels of 10 seen types and 5 unseen types (i.e., without any seed entities provided, including **algorithm**, **file type**, **html xml tag**, **value**, and **version**). The files are in the following CoNLL-2003 BIO format:
```
I O
am O
using O
custom O
adapter O
which O
I O
use O
for O
my O
ListView B-Library_Class
. O

After O
creating O
ArrayList B-Library_Class
...
```
You can rename your own testing sets. Make sure you change the ```test``` argument in ```contrastive/run.sh``` accordingly.

## Citation
```
@article{zhang2024seed,
  title={Seed-Guided Fine-Grained Entity Typing in Science and Engineering Domains},
  author={Zhang, Yu and Zhang, Yunyi and Shen, Yanzhen and Deng, Yu and Popa, Lucian and Shwartz, Larisa and Zhai, ChengXiang and Han, Jiawei},
  journal={arXiv preprint arXiv:2401.13129},
  year={2024}
}
```
```
@article{zhang2022entity,
  title={Entity Set Co-Expansion in StackOverflow},
  author={Zhang, Yu and Zhang, Yunyi and Jiang, Yucheng and Michalski, Martin and Deng, Yu and Popa, Lucian and Zhai, ChengXiang and Han, Jiawei},
  journal={arXiv preprint arXiv:2212.02271},
  year={2022}
}
```
