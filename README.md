# Aggregation is Not All You Need
## Generalizing improvements in Aspect-Based Sentiment Analysis
### W266 / UC Berkeley - MIDS - Fall 2021

Project work for UC Berkeley Masters in Data Science program's ["Natural Language Processing with Deep Learning coursework"](https://www.ischool.berkeley.edu/courses/datasci/266).

Codebase enhanced and built on Karimi et al's "Adversarial Training for Aspect-Based Sentiment Analysis with BERT" work ("[Adversarial Training for Aspect-Based Sentiment Analysis with BERT](https://arxiv.org/pdf/2001.11316)") which consequently improves upon the results from Hu et al ("[BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis](https://www.aclweb.org/anthology/N19-1242.pdf)").

In this project we explore recent improvements in Aspect Based Sentiment Analysis (ABSA), which is a subfield of sentiment analysis. We examine a recent paper by Karimi et al. and find that using newer, more general models in the place of more domain-specific models can improve performance on sentiment classification while impairing performance on sentiment target identification.

## ABSA Tasks
We focus on two major tasks in Aspect-Based Sentiment Analysis (ABSA).

Aspect Extraction (AE): given a review sentence ("The retina display is great."), find aspects("retina display");

Aspect Sentiment Classification (ASC): given an aspect ("retina display") and a review sentence ("The retina display is great."), detect the polarity of that aspect (positive).

## Running
#### In order to run code from Karimi et al's prior work: 
Place laptop and restaurant post-trained BERTs into ```pt_model/laptop_pt``` and ```pt_model/rest_pt```, respectively. The post-trained Laptop weights can be download [here](https://drive.google.com/file/d/1io-_zVW3sE6AbKgHZND4Snwh-wi32L4K/view?usp=sharing) and restaurant [here](https://drive.google.com/file/d/1TYk7zOoVEO8Isa6iP0cNtdDFAUlpnTyz/view?usp=sharing).

#### In order to run our new models: 
Switch to branches `spanbert_absa`, `roberta_absa` or `base_bert_absa`.


Execute the following command to run the model for Aspect Extraction task:

```bash run_absa.sh ae laptop_pt laptop pt_ae 9 0```

Here, ```laptop_pt``` is the post-trained weights for laptop, ```laptop``` is the domain, ```pt_ae``` is the fine-tuned folder in ```run/```, ```9``` means run 9 times and ```0``` means use gpu-0.

Similarly,
```
bash run_absa.sh ae rest_pt rest pt_ae 9 0
bash run_absa.sh asc laptop_pt laptop pt_asc 9 0
bash run_absa.sh asc rest_pt rest pt_asc 9 0
```
### Evaluation
Evaluation wrapper code has been written in ipython notebook ```eval/eval.ipynb```. 
AE ```eval/evaluate_ae.py``` additionally needs Java JRE/JDK to be installed.

Open ```result.ipynb``` and check the results.

