# SAMRank
This code is for paper "[SAMRank: Unsupervised Keyphrase Extraction using Self-Attention Map in BERT and GPT-2](https://aclanthology.org/2023.emnlp-main.630)" 


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
python samrank.py --dataset [Inspec/SemEval2010/SemEval2017] --plm [BERT/GPT2]
```
**The performances of all 144 heads** will be saved as data frames (.csv) in the 'experiment_results' folder.

## Citation
If you use this code, please cite our paper:

```
@inproceedings{kang2023samrank,
  title={SAMRank: Unsupervised Keyphrase Extraction using Self-Attention Map in BERT and GPT-2},
  author={Kang, Byungha and Shin, Youhyun},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages={10188--10201},
  year={2023}
}
```
