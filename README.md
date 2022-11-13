# Superpolyak.py

A pytorch implementation of the SuperPolyak subgradient method[^superpolyak_reference].

**Quick demo:** [SuperPolyakDemo.ipynb](SuperPolyakDemo.ipynb).

# What is SuperPolyak?

## Problem formulation 
 SuperPolyak is a **first-order** method for solving (possibly) nonsmooth equations/optimization problems of the form:

$$
f(\bar x) = 0 \qquad \iff \qquad  \min f(x) 
$$

where $f \colon \mathbb{R}^d \rightarrow \mathbb{R_{\geq 0}}$ is a _nonnegative function_ with minimal value $0$. When certain **minimal assumptions** are met, SuperPolyak locally converges **superlinearly** (i.e., "double exponentially").


### Example 1: Fitting a 1-hidden layer neural network with max-pooling


Let's try to fit a simple neural network with max-pooling to data

$$
(a_1, y_1), \ldots, (a_m, y_m)
$$

The network has $d$ parameters, $r$ hidden units, and $m$ data samples. Concretely, we want to solve the $m$ equations for weights $\beta_j$:

$$
\max_{j \in [r]} \langle a_i, \beta_j\rangle = y_i \qquad i \in [m]  
$$

We reformulate this as a root finding problem: 

$$
\text{Find a root of } f(\beta_1, \ldots, \beta_r) = \frac{1}{m} \sum_{i=1}^m |y_i - \max_{j \in [r]} \langle a_i, \beta_j\rangle|.
$$

Now let's do a quick experiment:
- **Setup:** 
  - We use Gaussian data.
  - We set $d = 500$, vary $r$, and set $m = 3dr$.
  - We run SuperPolyak and a standard first-order method (Polyak). 
- **Conclusion:** 
  - Superpolyak outperforms Polyak both in terms of time and oracle calls (evals of $f$ and its gradient).
![Performance Plot](figures/max_linear_regression.png)

### Example 2: Solving a smooth and strongly convex optimization  

Let's try to solve the following optimization problem:

$$
\min l(x) 
$$

where $l$ is a smooth and strongly convex function. If $l$ is nonnegative, we can apply SuperPolyak to $l$. Otherwise, we reformulate it via first-order optimality conditions (due to convexity):

$$
\text{Find a root of } \nabla l(x).
$$

The gradient $\nabla l$ is a mapping, not a function. No problem, just reformulate via the norm:

$$
\text{Find a root of } f(x) = \|\nabla l(x)\|.
$$

**TODO:** Vas please make a pretty plot like the above for logistic regression.
**TODO** Need newton's method
- **Setup:**
  - We fit a logistic regression model with l2 regularization to Gaussian data.
  - We vary the dimension $d$ and the number of parameters $m$.
  - We run SuperPolyak, gradient descent, and Newton's method.
- **Conclusion:**
  - Superpolyak outperforms both methods both in terms of time and oracle calls (evals of $f$ and its gradient)?


# What is SuperPolyak doing? 

SuperPolyak is inspired by _Newton's method_, which is taught in first-year calculus. Newton's method attempts to find the root of a single-variable function by repeatedly applying the following two steps:

- Construct the tangent approximation of $f$ at the current iterate
- Declare the next iterate to be the root of the tangent approximation.

The following gif shows newton's method in action.

![Newton's Method](figures/newton_movie_slow.gif)

If we start close enough to the root $\overline x$ and reasonable assumptions hold, Newton's method is superlinearly convergent, meaning

$$
\|x_{k+1} - \overline x\| \leq \text{const} \cdot 2^{-2^k}
$$

This super fast. To put this in perspective, around 10 steps gives ~300 digits of accuracy.
```
k		2^{-2^{k}}
-------------------------------------------------
1		2.50e-01
2		6.25e-02
3		3.91e-03
4		1.53e-05
5		2.33e-10
6		5.42e-20
7		2.94e-39
8		8.64e-78
9		7.46e-155
10		5.56e-309
```


## Slow-down of Newton's method in higher dimensions

Now let's consider an example where the dimension fo the parameter $d$ is greater than 1. In this setting, Newton's method is defined similarly, but with a key difference, highlighted in bold:

- Construct the tangent approximation of $f$ at the current iterate
- Declare the next iterate to be the **nearest root** of the tangent approximation.

Here, a qualifier such as **nearest root** is necessary since there are infinitely many roots of the tangent approximation, as shown in the following contour plot, corresponding to the function $f(x,y) = \|(x, 2y)\|$.

![Performance Plot](figures/newton_2d_onestep.png)

How well does the above Newton's method work? The following animation suggests that it works quite well:

![Newton's Method](figures/newton_2d.gif)

While fast, the following plot shows the method converges only linearly, not superlinearly.

![Newton's method 2d convergence plot](figures/newton_function_values.png)

Is it possible to recover the superlinear rate?

## The SuperPolyak step: a method for repairing Newton's method in higher dimensions

The issue with Newton's method in higher dimensions is that there are infinitely many roots to the tangent approximation: why should we expect that the nearest root is very close to the solution?

In the interest of fixing this, we ask the following simple question:

> What if we choose the next iterate to be a root of several distinct tangent approximations?

For example, suppose we could find $d$ "linearly independent" tangent approximations. Then the choice of the next iterate could simply be the intersection, which is unique. 

There are infinitely many ways to choose the "several distinct tangent approximations." For example, we could choose the locations randomly or sampled from the past history of the algorithm. 

Instead of random, we suggest the following iterative scheme (here $g$ is a "gradient of $f$", e.g., the output of autodifferentiation):

![SuperPolyak explnanation](figures/SuperPolyak_alg_explanation.gif)

Let's take a look at the algorithm in action:

![SuperPolyak](figures/superpolyak_contour.gif)

From the above animation, we see the approach works well, essentially finding the solution in two evaluations of $f$ and its gradient:

![SuperPolyak function values](figures/superpolyak_subgradient_method_function_values.png)

## When does SuperPolyak work?

Two broad families of examples where SuperPolyak works well:
- Piecewise linear functions, e.g., for $f(\beta_1, \ldots, \beta_r) = \frac{1}{m} \sum_{i=1}^m |y_i - \max_{j \in [r]} \langle a_i, \beta_j\rangle|.$
- 

SuperPolyak works under minimal assumptions known as "sharpness" and "semismoothness."


## Practical improvements: early termination of the SuperPolyak step

In [0], we show that SuperPolyak converges superlinearly. However, its Naïve implementation could be prohibitively expensive, since it requires $d$ evaluations of $f$ and its gradient. We've found that this number can be substantially reduced in practice. For instance, in example 1, the total number of iterations is much less than $d = 500$. To achieve this, we implement two early termination strategies, in SuperPolyak.py, both of which are described in [Section 5.1.1, 0]:

- Fix a maximum per-step budget, called ```max_elt```. Then declare the next iterate to be the best among the first ```max_elt``` points $y_i$.
- Fix a "superlinear improvement" exponent ```eta_est``` and exist as soon as one finds a point $y_i$ such that $f(y_i) \leq f(y_0)^{1+ \eta}$. 


## What about semismooth newton?

Semismooth Newton's method is the direct generalization of Newton's method to systems of nonsmooth systems of equations: 

$$
F(x) = 0
$$

where $F \colon \mathbb{R}^d \rightarrow \mathbb{R}^m$. The algorithm iterates

$$
x_{k+1} = x_k - G(x_k)^{\dag} F(x_k), 
$$

where $G(x_k)$ denotes a "generalized Jacobian" of $F$ at $x_k and $G(x_k)^{\dag}$ denotes its Moore-Penrose pseudoinverse.

Semismooth newton is known to converge superlinearly in several circumstances outlined in [0, 1]. However, for the problems we consider in [0], it converges at most linearly, as we saw for the function $f(x,y) = \|(x, 2y)\|$.[^semismooth].




# How to use

SuperPolyak can be run in two ways: 
- Method 1: standalone pytorch optimizer; 
- Method 2: coupled with another pytorch optimizer.

## Standalone optimizer class

SuperPolyak inherits from the pytorch optimizer class. It has several options. 
- max_elts: The size of the bundle.
- ...

### What a single step does (and what are the options)

Animation of algorithm from paper.

## Coupling with a fallback algorithm (e.g. SGD)

The figure from Vasilis presentation with fallback. 

An example code.


# References

[0] V. Charisopoulos, D. Davis. A superlinearly convergent subgradient method for sharp semismooth problems, 2022. URL: https://arxiv.org/abs/2201.04611.

[1] Qi and Sun

[^semismooth]: This story is somewhat subtle. One could of course reformulate the problem to finding a root of the **smooth** mapping $F(x,y) = (x,2y)$ and apply the standard Newton method, which would converge superlinearly. However, our goal is to treat the loss function $f(x) = \|(x, 2y)\|$ as a blackbox, accessible only through gradient and function evaluations. Under this setting, Newton's method only converges linearly.