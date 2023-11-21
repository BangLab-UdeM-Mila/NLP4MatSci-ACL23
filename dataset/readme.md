# Dataset   

## The tasks for each dataset
<img src="https://github.com/BangLab-UdeM-Mila/NLP4MatSci-ACL23/blob/main/dataset/dataset.png" alt="图片加载失败时，显示这段字"/>  

## Introduction of each task
- **Named Entity Recognition~(NER):** The NER task requires models to extract summary-level information from materials science text and recognize entities including materials, descriptors, material properties, and applications amongst others. The NER task predicts the best entity type label for a given text span $s_i$ with a non-entity span containing a “null” label. 
- **Relation Classification:** In the relation classification task, the model predicts the most relevant relation type for a given span pair $(s_i, s_j)$. 
- **Event Argument Extraction:** The event argument extraction task involves extracting event arguments and relevant argument roles. As there may be more than a single event for a given text, we specify event triggers and require the language model to extract corresponding arguments and their roles. 
- **Paragraph Classification:** In the paragraph classification task, the model determines whether a given paragraph pertains to glass science.
- **Synthesis Action Retrieval~(SAR):** SAR is a materials science domain-specific task that defines eight action terms that unambiguously identify a type of synthesis action to describe a synthesis procedure. MatSci-NLP adapts SAR data to ask language models to classify word tokens into pre-defined action categories.
- **Sentence Classification:** In the sentence classification task, models identify sentences that describe relevant experimental facts.
- **Slot Filling:** In the slot-filling task, models extract slot fillers from particular sentences based on a predefined set of semantically meaningful entities. In this task, each sentence describes a single experiment frame for which the model predicts the slots in that frame.

## Data source of each dataset
The sub-datasets of MATSCI-NLP benchmark are collected from internet. If you want to know further details about each sub-dataset, please refer to the original paper.
