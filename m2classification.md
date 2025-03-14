 Models  

## (1) Which general model classes do you know?  
- General model classes include linear models (e.g., simple linear regression, weighted regression), local models (e.g., locally weighted regression, RBF networks) and nonlinear models (e.g., neural networks, deep learning models).

## (2) What makes a model linear?  
- A model is called linear if it is linear with respect to its parameters—even if it employs nonlinear basis functions, the combination of parameters remains linear.

## (3) Give examples!  
- Examples include simple linear regression, weighted least squares regression, locally weighted regression (LWR) and radial basis function networks (RBFNs where the output layer is linear).

## (4) How to get the solution for optimal parameters of the linear model?  
- For linear models, we minimize the quadratic (squared) error function, compute its gradient, set it to zero and solve the resulting normal equations.

## (5) Write down the result!  
- The analytic solution is given by $$w^* = (\Phi^T \Phi)^{-1} \Phi^T T.$$

## (6) How to optimize parameters for nonlinear models?  
- For nonlinear models, no closed-form solution exists. Therefore, one must use iterative optimization methods—such as gradient descent or stochastic gradient descent—to minimize the error function.


## (7) Write down the learning rule for gradient descent?  
- The gradient descent update rule is
$$
w \gets w - \eta \nabla E(w),
$$
where $\eta$ is the learning rate.


## (8) Sketch gradient descent! What problems can occur? Sketch them.  
## (9) Sketch the structure of a neural network. What is computed at each local node (= neuron)?  
## (10) Describe an example of “deep” learning.  
- An example of deep learning is a deep multilayer perceptron with many hidden layers that learns hierarchical feature representations, such as in image recognition tasks.


## (11) What is model selection in this context?  
- Model selection refers to choosing the best model among a set of candidates based on criteria such as generalization performance, often via cross-validation or a separate validation set.


## (12) Write down the bias-variance decomposition.  
- The bias-variance decomposition is typically written as
$$
E\left[(t - \hat{t})^2\right] = \text{Bias}^2 + \text{Variance} + \text{Noise}.
$$



## (13) Why do we speak of the bias-variance dilemma?  
- We speak of the bias-variance dilemma because reducing bias often increases variance and vice versa, necessitating a trade-off for optimal generalization.

## (14) When does underfitting/overfitting occur?  
- Underfitting occurs when the model is too simple to capture the underlying structure of the data, while overfitting happens when the model is excessively complex and fits the noise in the data.

## (15) What is “double descent”? Sketch the respective error plot. 
## (16) Why do super-large networks train so well and no bad local minima seem to occur?  
- Super-large networks train well because in high-dimensional parameter spaces most local minima are nearly as good as the global minimum, and the optimization landscape is dominated by saddle points rather than problematic local minima.

## (17) Which local models do you know? What makes them local?  
- Local models include locally weighted regression (LWR), weighted linear regression, and RBF networks. They are “local” because they employ weighting functions that assign higher weights to data points near the query point.

## (18) Sketch an RBF. What is model selection here?  
## (19) Sketch weighted linear regression (with Gaussian weights).  
## (20) How to compute the solution?  
- The optimal solution is computed by solving the weighted normal equation:
$$
w^* = (X^T D X)^{-1} X^T D T.
$$

## (21) Describe linear weighted regression, sketch it!  
## (22) Write down the “unified model”.  
- The unified model can be expressed as
$$
t(x) = \sum_{e=1}^{E} \phi_e(x, \theta_e) \left( w_e^T x + b_e \right).
$$



## (23) How do RBF, linear regression, LWR, and weighted regression relate to the unified model?  
- They can all be seen as special cases of the unified model. For instance, linear regression corresponds to using an identity basis function; RBF networks use constant functions weighted by Gaussian functions; and LWR/weighted regression extend this idea by incorporating local linear models with appropriate weighting.


---  
--------------------------

# **Classification Summary**
## **(1) What is the aim of classification?**
**Goal:** The aim of classification is to assign data points to different categories so that future predictions can be made accurately. The core task is to find a **decision boundary** that maximizes classification accuracy [oai_citation:0‡lecture_07_classification_27_05_2024v2.pdf](file-service://file-DJcfD6QznTwXaTVfXiMwjR).

## **(2) How to model labels for multi-class problems?**
**Multi-class label modeling:**  
- **One-hot encoding:** Assigns each class a unique vector, e.g., for a three-class problem: A=(1,0,0), B=(0,1,0), C=(0,0,1).  
- **Probability distribution:** Uses the softmax function to ensure that the sum of class probabilities is 1 [oai_citation:1‡lecture_07_classification_27_05_2024v2.pdf](file-service://file-DJcfD6QznTwXaTVfXiMwjR).

## **(3) What is “linear separability”?**
**Linear separability:** A dataset is **linearly separable** if a **linear hyperplane** (e.g., a line or a plane) can completely separate different class points. For example, perceptron and logistic regression work well only on linearly separable data [oai_citation:2‡lecture_07_classification_27_05_2024v2.pdf](file-service://file-DJcfD6QznTwXaTVfXiMwjR).

## **(4) Sketch how to proceed for the Fisher discriminant.**
**Fisher Discriminant Analysis (FDA) Steps:**
# **Fisher’s Linear Discriminant (FLD) Steps**
1. **Compute the mean vectors** for each class:  
   \[
   \mathbf{m}_1 = \frac{1}{N_1} \sum_{n \in C_1} \mathbf{x}_n, \quad
   \mathbf{m}_2 = \frac{1}{N_2} \sum_{n \in C_2} \mathbf{x}_n
   \]

2. **Project the data from 2D to 1D** using a linear model:  
   \[
   y(\mathbf{x}) = \mathbf{w}^T \mathbf{x}
   \]

3. **Ensure projected points** lie on a straight line parallel to \( \mathbf{w} \).

4. **Perform projection** along the line orthogonal to \( \mathbf{w} \).[oai_citation:3‡lecture_07_classification_27_05_2024v2.pdf](file-service://file-DJcfD6QznTwXaTVfXiMwjR).

## **(5) Which criterion is optimized (write it down!)? Why is this reasonable?**
**Optimization Criterion:**
(*"maximize inter-class variance"*)  
\[
\max_{w} | m'_1 - m'_2 |
\]

\[
\Leftrightarrow \max_{w} | w^T m_1 - w^T m_2 |
\]

### **Hence:**  
\[
w \propto (m_2 - m_1)
\]
(*"w points in the same direction as \( (m_2 - m_1) \)"*)  
- **Direction and length of \( w \) are not irrelevant.**  
- **Projections may overlap heavily.**  

## **(6) What is the result?**
**Result:** Fisher Discriminant Analysis finds the **optimal projection direction** that maximizes the separation between different classes in the projected space [oai_citation:5‡lecture_07_classification_27_05_2024v2.pdf](file-service://file-DJcfD6QznTwXaTVfXiMwjR).

## **(7) Write down the Bayesian Approach for 2 classes.**
**Bayesian Classifier (Binary Classification):**
\[
P(C_k | x) = \frac{P(x | C_k) P(C_k)}{P(x)}
\]
where:
- \( P(C_k | x) \) is the **posterior probability** of class \( C_k \) given \( x \).
- \( P(x | C_k) \) is the **class-conditional probability** (likelihood).
- \( P(C_k) \) is the **prior probability** of class \( C_k \).
- \( P(x) \) is a **normalization factor** 
- \( P(C_k | x) \) = 1 / (1 + $e^{-a})$
[oai_citation:6‡lecture_04_seq_Bayes_lin_models_30_04_2024v1.pdf](file-service://file-7gSFXdF3sqEAGwK4DdVaTE).

## **(8) Why is this named “generalized linear model”, under which condition?**
**Generalized Linear Model (GLM):**
- A linear model assumes the target variable is a linear combination of inputs.
- **GLM extends this** by applying **non-linear transformations** (e.g., sigmoid, softmax) for classification problems.
- **Condition:** When using **logistic regression** or **softmax regression**, it is called a GLM because it maps a **linear combination** of inputs into a probability range (0,1) [oai_citation:7‡lecture_04_seq_Bayes_lin_models_30_04_2024v1.pdf](file-service://file-7gSFXdF3sqEAGwK4DdVaTE).

## **(9) If the classes are modeled with a Gaussian function, what is the result?**
**Gaussian Model-Based Classification:**
- If each class follows a **Gaussian probability density function (PDF)**:
\[
p(C_1 | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + w_0)
\]
1. **Weight Vector \( \mathbf{w} \):**
   \[
   \mathbf{w} = \Sigma^{-1} (\mu_1 - \mu_2)
   \]

2. **Bias Term \( w_0 \):**
   \[
   w_0 = -\frac{1}{2} \mu_1^T \Sigma^{-1} \mu_1 + \frac{1}{2} \mu_2^T \Sigma^{-1} \mu_2 + \ln \frac{p(C_1)}{p(C_2)}
   \]
- The **decision boundary** becomes a **Quadratic Discriminant Analysis (QDA)** or **Linear Discriminant Analysis (LDA)** depending on whether the covariance matrices \( \Sigma_k \) are assumed equal [oai_citation:8‡lecture_11_GMM_GMR_25_06_2024.pdf](file-service://file-Fyhrm8XccUhtWpPBepmkdy).

## **(10) Describe max-likelihood for labels. Which trick is used for the computation?**
**Maximum Likelihood Estimation (MLE) for Labels:**
- Maximizes the **log-likelihood function**:
  \[
  L(\theta) = \sum_{i=1}^{N} \log P(y_i | x_i, \theta)
  \]
- **Computational Tricks:**
  - **Log Transformation:** Converts product operations into summation to avoid numerical underflow.
  - **Gradient Descent:** Optimizes parameters using iterative methods like SGD or Newton's method.
  - **Softmax Activation:** Ensures valid probability outputs in multi-class classification [oai_citation:9‡lecture_12_Kernels_01_07_2024.pdf](file-service://file-QzTzXaBMP6qhZjHtEcNz8D).

## **(11) How to directly model the posterior?**
**Direct Posterior Modeling:**
- Use **logistic regression** or **softmax regression** to directly estimate the posterior probability:
  \[
  P(C_k | x) = \frac{\exp(w_k^T x)}{\sum_{j} \exp(w_j^T x)}
  \]
- This is a **discriminative model**, which differs from **generative models** like Naïve Bayes [oai_citation:10‡lecture_12_Kernels_01_07_2024.pdf](file-service://file-QzTzXaBMP6qhZjHtEcNz8D).

## **(12) What is the cross-entropy, and how to arrive at it?**
**Cross-Entropy Formula:**

p6 last
[oai_citation:11‡lecture_12_Kernels_01_07_2024.pdf](file-service://file-QzTzXaBMP6qhZjHtEcNz8D).
---
---------------