# help_a_hermatologist_out
help_a_hermatologist_out challenge by the Helmholtz incubator program
![logo](https://github.com/christinab12/Data-challenge-logo/blob/main/logo.jpg?raw=true)

## Getting started


This notebook is a short summary for getting started with the challenge ( found [here](https://helmholtz-data-challenges.de/web/challenges/challenge-page/93/overview)  ). Below you can find how to download the dataset and also the different labels along with exploring and analyzing the input and output data of the challenge, running a baseline model and creating a submission file to upload to the leaderboard.

***

<b>dataset:</b>

Three datasets, each constituting a different domain, will be used for this challenge:
> 1. The Acevedo_20 dataset with labels
> 2. The Matek_19 dataset with labels
> 3. The WBC dataset <b> without labels </b> (Used for domain adaptation and performance measurement)

The Acevedo_20 and Matek_19 datasets are labeled and should be used to train the model for the domain generalization task.
A small subpart of the WBC dataset, WBC1, will be downloadable from the beginning of the challenge. It is unlabeled and should be used for evaluation and domain adaptation techniques.

A second similar subpart of the WBC dataset, WBC2, will become available for download during phase 2 of the challenge, i.e. on the last day, 24 hours before submissions close.

***
<b>Goal: </b> 

The challenge here is in transfer learning, <b> precisely domain generalization (DG) and domain adaptation (DA) </b> techniques. The focus lies on using deep neural networks to classify single white blood cell images obtained from peripheral blood smears.
<b> Tthe goal of this challenge is to achieve a high performance, especially a high f1 macro score, on the WBC2 dataset. </b>

***
<b>Notes: </b>

This challenge wants to motivate research in domain generalization and adaptation techniques:

To make actual use of deep learning in the medical routine, it is important that the techniques can be used in realistic cases. If a peripheral blood smear is acquired from a patient and classified by a neural network, it is important that this works reliably. But the patient’s blood smear might very likely vary compared to the image domains used as training data of the network, resulting in not trustable results. To overcome this obstacle and build robust domain-invariant classifiers research in domain generalization and adaptation is needed.

***
<b>f1_score: </b>
[wikepedia](https://en.wikipedia.org/wiki/F-score)

> sklearn.metrics.f1_score(y_true, y_pred, *, labels=None, pos_label=1,<b> average='macro' </b>, sample_weight=None, zero_division='warn')

The formula can be see in [click here for the code](https://github.com/scikit-learn/scikit-learn/blob/36958fb24/sklearn/metrics/_classification.py#L1001) and is given as

> <g> F1 = 2 * (precision * recall) / (precision + recall) </g>
***
