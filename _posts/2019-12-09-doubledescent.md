---
excerpt: "A new approach to bias-variance"
header:
  overlay_image: /assets/images/boat1.jpg
title: "Lit Review: The Double Descent Curve"
last_modified_at: 2019-10-15T15:19:02-05:00
categories:
  - Blog
tags:
  - DL
  - LitReview
  - magic?
---




# 1. Introduction
I love a unique approach to an old problem. The most exciting papers deconstruct a current standard, spin it around, and put it back together in a totally unique way. A team from [Ohio State](http://web.cse.ohio-state.edu/~belkin.8/#ref1) and [Columbia](http://www.cs.columbia.edu/~djhsu/) have this new idea for the bias variance trade-off that is **wild**. [Dr. Hanin](https://www.math.tamu.edu/~bhanin/) introduced me to [the](https://arxiv.org/abs/1903.07571) [papers](https://www.pnas.org/content/116/32/15849) here at TAMU, and I can’t stop thinking about them. 

# 2. The Bias-Variance Trade-Off
Overfitting with too complex a hypothesis is a well studied problem. My most recent machine learning textbook mentions “overfit” 54 times. Perhaps you can recall your freshman chemistry lab, when your TA was very impressed with a 9<sup>th</sup> order polynomial fit to 9 data points. (too specific? :flushed:) The R<sup>2</sup> value is great, but the hypothesis doesn’t generalize to unseen data.

In machine learning, we talk about the bias-variance trade-off. Restricting the complexity of the hypothesis class helps prevent the final predictions from being too dependent on the training data. 
<figure>
	<img src="/assets/images/bias-var.png" style="width:400px;">
	<figcaption>The Bias-Variance Risk Curve</figcaption>
</figure>
Still, many modern frameworks have a huge number of parameters. ResNet50 has a little over 23 million trainable parameters. Generally, the “modern” approach is to have more trainable parameters than there are data points. Unlike my 9th order fit from chemistry, these high parameter models generalize well to new data, without the need for explicit regularization. This is contradictory. How can there be a U-shaped bias-variance curve if modern practitioners are using highly parameterized models?

# 3. The Double Descent Curve
Because we have observed that the test risk on a model is asymptotic to infinity as the complexity of the hypothesis class increases, no one ever bothers trying to come over to the other side of the asymptote. This is the standard. But what’s on the other side of that curve? The original authors are like machine learning Magellans, sailing off the edge of the known world, hoping the risk curve extends after the asymptote.
<figure>
	<img src="/assets/images/boat1.jpg" style="width:400px;">
	<figcaption>Machine Learning Magellans</figcaption>
</figure>
And it does! The test risk comes back down from infinity as you add an extreme number of parameters. Take a look at these test results from a random Fourier feature model on the MNIST dataset. 
<figure>
	<img src="/assets/images/double_dec_four.png" style="width:800px;">
	<figcaption>U-Shaped Double Descent Curve</figcaption>
</figure>
:open_mouth:
Notice that not only does the test risk come back down, it actually exceeds the performance of the initial sweet spot. :open_mouth: Does this apply to other learning methods? Take a look at the results for a random forest. 
<figure>
	<img src="/assets/images/double_dec_rf.png" style="width:400px;">
	<figcaption>Double Descent Curve for RFs</figcaption>
</figure>
The shape of the curve is a little different, but the idea is the same. Both test and training loss improve as the model has an increasing number of parameters. Finally, take a look at the results for a fully connected ReLU network.
<figure>
	<img src="/assets/images/double_dec_ann.png" style="width:400px;">
	<figcaption>Double Descent Curve for ANNs</figcaption>
</figure>
Here we clearly see the double descent behaviour on the test error. The authors naturally put it best. 
> The double-descent risk curve introduced in this paper reconciles the U-shaped curve predicted by the bias–variance trade-off and the observed behavior of rich models used in modern machine-learning practice.

It looks like we need a new description for the bias-variance trade-off. 
<figure>
	<img src="/assets/images/double_dec_2.png" style="width:600px;">
	<figcaption>New Bias-Variance Trade-Off Curve</figcaption>
</figure>
At present, we can only postulate the mechanism that generates these surprising results. Inductive bias in the model is a primary suspect, however. In neural networks, for example, the batch gradient is orthogonal to the parameter space which fits the data. This leads to a minimal norm solution from small initialization. With increasingly many features, the features approach a solution which maximizes smoothness. However, the proof for this idea isn't avaliable yet, so we are left to speculate a little. 


![Alt Text](http://i.imgur.com/aYlNumW.gif)

# 5. Conclusions 
These papers questioned the discrepancy between bias-variance trade-off and practically accurate, highly complex models. While a mechanism for the double descent curve has not been formalized, this new concept will serve as a map for other researchers. Developing a rigorous theoretical basis for neural networks is crucial to long term development, and these publications are a step in that direction. We should watch for the next paper to offer a more complete mechanism for the double descent curve. 