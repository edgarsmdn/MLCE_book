# An overview of the course 🔭

This course covers the foundations of machine learning and shows some of the common
applications to chemical engineering systems. Machine learning can be broadly classified into **supervised learning**, **unsupervised learning** 
and **reinforcement learning**. Additionally, it covers **hybrid modeling**, a very important aspect that deals with the 
combination of mechanistic knowledge with data-driven tools.

## Machine learning

Machine learning (ML) is the field of study that gives computers the ability to learn without being explicitly programmed {cite}`samuel1959some`. 
This is in contrast to the "traditional" computer science on which exact instructions need to be specified in order to do a specific task.
ML heavily relies on linear algebra, statistics and optimization. Therefore, expect to encounter such topics while studying this course. 
As mentioned before, ML can be broadly classified into 3 areas: supervised, unsupervised and reinforcement learning.

### 1. Supervised learning

This refers to obtaining an **input-output mapping** where the learning agent is fed with examples in order to generalize to new instances.
For instance, assume there exist an unkown fuction $f(\textbf{x})$ from which you only have a collection of input values 
$\textbf{X}$ with their respective output values $\textbf{y}$.
Then, the objective of supervised learning is to get an approximation $h(\textbf{x})$ that minimizes the difference with respect to the true function $f(\textbf{x})$.
The function $h$ is refer to as hypothesis function.

Depending on the form of the output $\textbf{y}$ the problem can either be a:

* Classification: when $\textbf{y}$ is categorical (e.g., a molecule is toxic or not).
* Regression: when $\textbf{y}$ is continuous (e.g., the temperature profile of a reactor).

```{note}
In supervised learning we are not particularly interested in fitting the observed data very well, but rather in generalizing well to unseen data! Therefore, the 
concepts of **overfitting** and **underfitting** become really important.
```

```{figure} media/overview/supervised_learning.png
:alt: supervised_learning
:width: 75%
:align: center

A collection of hypothesis functions (in blue) that could be fitted to the observed data (in red). Which hypothesis is the best? How can we determined the 
best hypothesis function?
```

### 2. Unsupervised learning

In the case of unsupervised learning the output values are not available, only the input data $\textbf{X}$. Therefore, the goal of unsupervised learning is to **identify 
patterns** in the data. For instance, finding clusters or reducing the dimensionality of the data.

```{figure} media/overview/unsupervised_learning.png
:alt: unsupervised_learning
:width: 50%
:align: center

Clustering of data. How can we detect groups of data that are similar to each other? Why is this useful?
```


### 3. Reinforcement learning

The name comes from animal psychology, where we train animals/pets by reinforcing good behaviour, and discouraging bad behaviours. Here, the agent has to learn 
how to interact with its environment in order to maximize the reward or minimize the punishment.

```{figure} media/overview/reinforcement_learning.png
:alt: reinforcement_learning
:width: 50%
:align: center

Reinforcement learning: incentivate actions that maximize reward and/or discourage actions that lead to punishment.  
```

## Hybrid modeling

The central questions about hybrid modeling are: should we discard all the physical knowledge acquired for centuries and replace it by data-driven 
models? Is there a way to combine both? Is it beneficial to do so?

In general, the term hybrid modeling refers to the combination of mechanistic and data-driven models and is also called "grey-box modeling". For example, 
mass and energy balances, thermodynamic laws and kinetics should be respected in our models. Introducing this physical knwoledge reduces the amount of data
that is needed for the ML part and improves the capacity of the models to generalize to unseen conditions.   

```{figure} media/overview/hybrid_modeling.png
:alt: hybrid_modeling
:width: 75%
:align: center

Hybrid modeling is also refer in the literature as grey-box modeling.  
```

## References

```{bibliography}
:filter: docname in docnames
```
