---
title:  "Deriving the Sigmoid Derivative for Neural Networks"
date:   2017-08-06
tags: [neural networks, machine learning, mathematics]

header:
  image: "sigmoid_derivative/bryce_hoodoos.jpg"
  caption: "Photo Credit: Ginny Lehman"

excerpt: "Neural Networks, Hidden Layers, Backpropagation, TensorFlow"
---

Though many state of the art results from neural networks use linear rectifiers as activation functions, the sigmoid is the bread and butter activation function. To really understand a network, it's important to know where each component comes from. The computationally efficient derivative of the sigmoid function is one of the less obvious components. Though it's usually taken care of under the hood in the higher level libraries like Tensorflow and others, it's worth taking the time to understand where it comes from.

The sigmoid function, $$S(x) = \frac{1}{1+e^{-x}}$$ is a special case of the more general [logistic function](https://en.wikipedia.org/wiki/Logistic_function), and it essentially squashes input to be between zero and one. Its derivative has advantageous properties, which partially explains its widespread use as an activation function in neural networks.

But it's not obvious from looking at the function how the derivative arises. In this post, I'll walk through each step of the derivation and discuss why people use $$\frac{d}{dx}S(x) = S(x)(1 - S(x))$$ instead of any other version.


## Derivation

Enough writing. Time for the math. Let's begin by defining the sigmoid function, $$S(x)$$

$$Sigmoid(x) = \large \frac{1}{1+e^{-x}}$$

With the function defined, we can take the derivative with respect to the input, $$x$$

$$\frac{d}{dx}S(x) = \frac{d}{dx} \frac{1}{1+e^{-x}}$$

To compute this derivative, we can use the [quotient rule](https://en.wikipedia.org/wiki/Quotient_rule). The quotient rule is a way to take the derivative a function when the numerator and denominator are both differentiable. By the quotient rule, the derivative of a function $$f(x)$$ with a $$numerator$$ and $$denominator$$ can be expressed as:

$$\frac{d}{dx}f = \frac{(denominator*\frac{d}{dx}numerator) - (numerator*\frac{d}{dx}denominator)}{denominator^{2}}$$

With this, we can come back to the sigmoid derivative. Since the numerator is a constant, its derivative is zero. To take the derivative of the denominator, $$1+e^{-x}$$, we need to use the [chain rule](https://en.wikipedia.org/wiki/Chain_rule). According to the chain rule, the derivative of $$f(a^{x})$$ is $$\frac{df}{dx} = \frac{df}{da} \frac{da}{dx}$$. Using the chain rule on the denominator, we get $$\frac{d(1+e^{-x})}{dx} = -e^{-x}$$

Now, we can use the quotient rule to take the derivative:

$$\frac{d}{dx}S(x) = \frac{(1+e^{-x})(0) - (1)(-e^{-x})}{(1+e^{-x})^2}$$

We can simplify by removing the term that gets multiplied by zero and multiplying the parentheses.

$$\frac{d}{dx}S(x) = \frac{e^{-x}}{(1+e^{-x})^2}$$


At this point, we're done. But it doesn't look like the friendly, computationally useful derivative commonly used in neural networks to backpropagate through sigmoid activation functions. To get to that form, we can use a **simple technique** that is actually also used to derive the quotient and product rules in calculus: **adding and subtracting the same thing (which changes nothing) to create a more useful representation.**


In this case, we can add and subtract the value 1 in the numerator.

$$\frac{d}{dx}S(x) = \frac{1 - 1 + e^{-x}}{(1+e^{-x})^2}$$

Suddenly, it's fairly clear that we might be able to cancel some things if we split this up into separate terms.

$$\frac{d}{dx}S(x) = \frac{1 + e^{-x}}{(1+e^{-x})^2} - \frac{1}{(1+e^{-x})^2}$$

Now we can do the canceling in the first term.

$$\frac{d}{dx}S(x) = \frac{1}{(1+e^{-x})} - \frac{1}{(1+e^{-x})^2}$$

This is starting to look good. Both terms have $$\frac{1}{(1+e^{-x})}$$, so we can take that out and distribute it.

$$\frac{d}{dx}S(x) = \frac{1}{(1+e^{-x})} (1 - \frac{1}{1+e^{-x}})$$

One last step. The definition of sigmoid, $$S(x) = \frac{1}{1+e^{-x}}$$,  is inside this equation, so we can substitute it back in.

$$\frac{d}{dx}S(x) = S(x)(1 - S(x))$$

And there it is. The derivative of the sigmoid function is the sigmoid function times one minus itself.

## Why is this formula Superior?

$$\frac{d}{dx}S(x) = S(x)(1 - S(x))$$ is better than $$\frac{d}{dx}S(x) = \frac{e^{-x}}{(1+e^{-x})^2}$$ primarily for one reason.

When we're backpropagating the errors in a network through a sigmoid activation function, $$S(x)$$ has already been computed. During the forward pass, we computed $$S(x)$$ when we multiplied the inputs by the weights and applied the sigmoid function. If we cache that matrix we can calculate the derivative now with just a few simple matrix operations. This computational speedup is extremely useful when we're doing computations on massive matrices and across multiple layers in a network.



## Conclusion

Understanding the individual components of a model is crucial to thinking deeply about extensions and appropriate applications. For neural networks, that includes both the choice of and the math behind activation functions that underpin backpropagation and the forward pass. Internalizing these derivations lays the foundation for effective imagination and evaluation of new types of network structures and topologies.
