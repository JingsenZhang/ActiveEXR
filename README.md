# ActiveEXR
This is the implementation for paper:
> "Active Explainable Recommendation with Limited Labeling Budgets." In ICASSP 2024. 

## Overview
In this paper, we study a novel problem in the field of explainable recommendation, that is, “given a limited budget to incentivize users to provide behavior explanations, how to effectively collect data such that the downstream models can be better optimized?” To solve this problem, we propose an active learning framework for recommender system, which consists of an acquisition function for sample collection and an explainable recommendation model to provide the final results. We consider both uncertainty and influence based strategies to design the acquisition function, which can determine the sample effectiveness from complementary perspectives.

<img src="https://github.com/JingsenZhang/ActiveEXR/blob/master/asset/graph.png" width="500px"/>

## Requirements
- Python 3.7
- Pytorch >=1.10.1

## Datasets
We use two real-world datasets, including *TripAdvisor-HongKong* and *Yelp Challenge 2019*. All the datasets are available at this [link](https://github.com/lileipisces/NLG4RS).

## Usage
+ **Download the codes and datasets.**
+ **Run** run_active.py

```
python run_active.py --model [model_name] --dataset [dataset_name] --config [config_file]
```

## Acknowledgement
Any scientific publications that use our codes and datasets should cite the following paper as the reference:
````
@inproceedings{Zhang-ICASSP-2024,
    title = "Active Explainable Recommendation with Limited Labeling Budgets",
    author = {Jingsen Zhang, Xiaohe Bo, Chenxi Wang, Quanyu Dai, Zhenhua Dong, Ruiming Tang and Xu Chen},
    booktitle = {{ICASSP}},
    year = {2024},
}
````
If you have any questions for our paper or codes, please send an email to zhangjingsen@ruc.edu.cn.
