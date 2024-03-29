# NLP4MatSci-ACL23
This repository contains the dataset and code for our ACL'23 publication: "MatSci-NLP: Evaluating Scientific Language Models on Materials Science Language Tasks Using Text-to-Schema Modeling".  

## Abstract
We present MatSci-NLP, a natural language benchmark for evaluating the performance of natural language processing (NLP) models on materials science text. We construct the benchmark from publicly available materials science text data to encompass seven different NLP tasks, including conventional NLP tasks like named entity recognition and relation classification, as well as NLP tasks specific to materials science, such as synthesis action retrieval which relates to creating synthesis procedures for materials.    

We study various BERT-based models pretrained on different scientific text corpora on MatSci-NLP to understand the impact of pretraining strategies on understanding materials science text. 
Given the scarcity of high-quality annotated data in the materials science domain, we perform our fine-tuning experiments with limited training data to encourage the generalize across MatSci-NLP tasks.
Our experiments in this low-resource training setting show that language models pretrained on scientific text outperform BERT trained on general text. 
MatBERT, a model pretrained specifically on materials science journals, generally performs best for most tasks.    

Moreover, we propose a unified text-to-schema for multitask learning on MatSci-NLP and compare its performance with traditional fine-tuning methods. In our analysis of different training methods, we find that our proposed text-to-schema methods inspired by question-answering consistently outperform single and multitask NLP fine-tuning methods.    

## Citation
If you use our code or data in your research, please cite our paper:
```
@article{song2023matsci,
  title={MatSci-NLP: Evaluating Scientific Language Models on Materials Science Language Tasks Using Text-to-Schema Modeling},
  author={Song, Yu and Miret, Santiago and Liu, Bang},
  journal={arXiv preprint arXiv:2305.08264},
  year={2023}
}
```

## QA   
If you have any questions about this code, feel free to email yu.song@umontreal.ca. I will response as soon as possible.
