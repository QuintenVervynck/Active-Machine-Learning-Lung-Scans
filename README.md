# Machine Learning: Active Learning

## 1. getting familiar with the active learning domain

[this blog](https://dsgissin.github.io/DiscriminativeActiveLearning/) gave a nice introduction to 
what Active Learning really is\
Read what we've learned from it over [here](./lit_study/blog.md)

After reading the blog fully, we decided it was time to get our hands dirty and 
play around a bit with the existing frameworks. 
We found out [scikit-activeml](https://github.com/scikit-activeml/scikit-activeml) is a great framework to do this,
it offers a high level library for creating a Active Learning model from a
wide range of Machine Learning models, as it is build on top of the scikit-learn library. 


## 2. Formulating a research question:

Optimizing Active learning for medical images

We both wanted to do something in the medical field with active learning. This because the primary advantage of
active learning is that it can reduce the amount of data needed to achieve the same accuracy as a model trained
on a randomly acquired data set.\
We thought this applied very well to the medical domain, because the doctors there are under a lot of pressure
already, and asking them to annotate **all** of the images they have, is just a very time consuming task for them.
So if we would only have the ask them to annotate lets say 100 images instead of all 1000 images they had available,
that would be a great improvement.

One key difference with the standard way of training models with scikit-learn is that we'll use `partial_fit`
for incrementally updating the parameters. Instead of just using `fit` which trains the model from start (so overwrites anything that was already there.
So if you trained the model on `D_0`, and then `partial_fit` on `D_1`, this would be conceptually similar 
to training a fresh model with `fit` on a combined dataset.
(see [docs](https://scikit-learn.org/0.15/modules/scaling_strategies.html#incremental-learning))

## 3. designing and executing an experiment
1. Gathering data
2. Preparing that data
3. Choosing a Model
4. Training & Hyperparameter runing
5. Prediction
6. Evaluation


## 4. writing a scientific report
both negative or positive results are fine
2. scientific report:
    1. introduction
    2. methods
    3. experiment setup
    4. results
    5. conclusion


