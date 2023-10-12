# FER

This is the source code of EMNLP'23 paper "From Relevance to Utility: Evidence Retrieval with Feedback for Fact Verification".

Retrieval-enhanced methods have become a primary approach in fact verification (FV); it requires reasoning over multiple retrieved pieces of evidence to verify the integrity of a claim. To retrieve evidence, existing work often employs off-the-shelf retrieval models whose design is based on the probability ranking principle. We argue that, rather than relevance, for FV we need to focus on the utility that a claim verifier derives from the retrieved evidence. We introduce the feedback-based evidence retriever(FER)
 that optimizes the evidence retrieval process by incorporating feedback from the claim verifier. As a feedback signal we use the divergence in utility between how effectively the verifier utilizes the retrieved evidence and the ground-truth evidence to produce the final claim label. Empirical studies demonstrate the superiority of FER over prevailing baselines.


## Table of Contents
- [Setup](#Setup)
- [Usage](#Usage)
- [Citation](#Citation)

## Setup
Clone as follows:

```
git clone https://github.com/hengran/FER.git
cd FER
pip install -r requirements.txt
```

Download model parameter and data from  [[model_path](https://drive.google.com/drive/folders/1iiB9Hc3KLvgog3Bv-Wq8UovXv7Db7qFK?usp=sharing)] to to folders `save_model/` and `data/`


## Usage
### Train FER models
python FER.py


### Reproduce our results
1. Download the [trained model](https://drive.google.com/drive/folders/1iiB9Hc3KLvgog3Bv-Wq8UovXv7Db7qFK?usp=sharing).
2. Run the code
```
python evaluated.py
```


## Citation

Please cite our paper if you use this code in your work:


