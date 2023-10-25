# SAMRank
This code is for EMNLP 2023 paper "SAMRank: Unsupervised Keyphrase Extraction using Self-Attention Map in BERT and GPT-2" 

## Requirements
- [stanford-corenlp-full-2018-02-27](https://drive.google.com/file/d/1K4Ll54ypTf_tF83Mkkar2QKOcZ4Uskl5/view?usp=sharing)  (please download the .zip file and extract it)

Run 'stanford-corenlp-full-2018-02-27' on your computer's terminal using the following command:

    (1) cd stanford-corenlp-full-2018-02-27/
    
    (2) java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos -status_port 9000 -port 9000 -timeout 15000 &
    
- transformers
- pytorch
- nltk
- pandas
- tqdm


## Runing
```shell
python samrank_eval.py --dataset [Inspec/SemEval2010/SemEval2017] --plm [BERT/GPT2]
```
The experiment results are saved as data frames (.csv) in the 'experiment_results' folder.
