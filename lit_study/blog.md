[this blog](https://dsgissin.github.io/DiscriminativeActiveLearning/) gave a nice introduction to Active Learning

[github](https://github.com/dsgissin/DiscriminativeActiveLearning)

<details><summary><h1>
Introduction to Active Learning
</h1></summary>

[(Read the post itself)](https://dsgissin.github.io/DiscriminativeActiveLearning/2018/07/05/AL-Intro.html)

### Basic setup:
```
allow the learner/model to ask questions, so it can choose points
which are very informative and quickly gets to the correct
decision boundary.

reduce the number of samples needed to get a strong classifier
by choosing the right examples to label.
```

### Possible scenarios:
```
In all scenarios, at each iteration a model is fitted to the current
labeled set and that model is used to decide which unlabeled example
we should label next.

1) membership query synthesis
     active learner is expected to produce an example that it would
     like us to label.
     requires that the model will be able to capture the data
     distribution well enough to create examples which are reasonable
     and that would have a clear label (can be hard for ex: images).
2) stream based
     learner gets a stream of examples from the data distribution and
     decides if a given instance should be labeled or not
3) pool based
     learner has access to a large pool of unlabeled examples and
     chooses an example to be labeled from that pool
```


### Query strategies
```
query strategy = how they choose the example to query, 
                 given the trained model

most common methods:
1) Uncertainty Sampling:
     make the learner query the example which it is least certain about
     "choosing the sample closest to the decision boundary"
2) Query By Committee (QBC):
     train several different models on the labeled set and look at
     their disagreement on examples in the unlabeled set
     -> there could be many possible models that fit the data, but some
        of them might not generalize well. If we look at our ensemble
        as a set of models from different viable parts of the
        hypothesis class (also called the “version space”), then
        choosing the example with the biggest disagreement corresponds
        to making the current version space as small as possible
3) Expected Model Change:
     query the examples which we expect to cause the greatest change to
     our model
4) Density Weighted Methods:
     query examples which have both a high uncertainty and which are
     “representative”, meaning they are in dense regions of the data.
     -> avoids choosing outliers
     -> assumes outliers give less information on the data distribution
```


### Summary
```
If we extend the supervised learning model to allow the learner to 
query specific examples to be labeled, we can significantly reduce 
the number of samples needed to get the desired accuracy.
```

</details>







<details><summary><h1>
Batch Active Learning
</h1></summary>

[(Read the post itself)](https://dsgissin.github.io/DiscriminativeActiveLearning/2018/07/05/Batch-AL.html)

### Making Active Learning Practical - Batch Queries
```
In each iteration, the learner will train a model on the labeled
examples like before, only now he will query a batch of examples
```

##### The effects of the batch size

```
how to choose the batch?
batch to big? bad
batch to small? no real advantage
```

### Querty strategies for the batch setting

##### Greedy Query Strategies
```
choose the top-K examples under the relevant score

) Uncertainty-Based Strategy

) Margin-Based Strategy - Adversarial Active Learning

) Model Change Strategy
```

##### Batch-Aware Query Strategies


</details>



- batch size
- the initial random batch strongly affects the first batch query
- experimenting with many forms of early stopping and running several evaluations and taking the best one, but these didn’t seem to work very well (mostly due to the difficulty of stopping the training at the right time).


