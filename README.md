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
@inproceedings{kang-shin-2023-samrank,
    title = "{SAMR}ank: Unsupervised Keyphrase Extraction using Self-Attention Map in {BERT} and {GPT}-2",
    author = "Kang, Byungha  and
      Shin, Youhyun",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.630",
    doi = "10.18653/v1/2023.emnlp-main.630",
    pages = "10188--10201",
    abstract = "We propose a novel unsupervised keyphrase extraction approach, called SAMRank, which uses only a self-attention map in a pre-trained language model (PLM) to determine the importance of phrases. Most recent approaches for unsupervised keyphrase extraction mainly utilize contextualized embeddings to capture semantic relevance between words, sentences, and documents. However, due to the anisotropic nature of contextual embeddings, these approaches may not be optimal for semantic similarity measurements. SAMRank as proposed here computes the importance of phrases solely leveraging a self-attention map in a PLM, in this case BERT and GPT-2, eliminating the need to measure embedding similarities. To assess the level of importance, SAMRank combines both global and proportional attention scores through calculations using a self-attention map. We evaluate the SAMRank on three keyphrase extraction datasets: Inspec, SemEval2010, and SemEval2017. The experimental results show that SAMRank outperforms most embedding-based models on both long and short documents and demonstrating that it is possible to use only a self-attention map for keyphrase extraction without relying on embeddings. Source code is available at https://github.com/kangnlp/SAMRank.",
}
```
