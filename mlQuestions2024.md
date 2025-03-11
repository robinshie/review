# Definitions and (some) fundamental notions

## (1) What is denoted by Deduction, Abduction, and Induction?  
- Deduction:starts with general axioms and derives specific theorems from there
- Abduction: starts from a special example and attempts to derive facts
- Induction:starts with data and derives general rules
  
## (2) What is special to induction, how to proceed?  
- Induction “starts with data to derive general rules (models)” and relies on limited samples plus prior assumptions (inductive bias), with its generalization verified on new data.
  
## (3) From the point of view of verification, what can be expected from induction?  
- Induction cannot prove a theory absolutely; it only supports or falsifies a theory through data. The expectation is that the model will predict new data accurately and generalize well.
  
## (4) What is the meaning of “Inductive Learning Hypothesis” and “Learning Bias”?  
- Inductive Learning Hypothesis: The assumption that a “finite training set” represents the overall problem, allowing the learning of a model that generalizes to unseen data.
- Learning Bias: The prior knowledge or assumptions that constrain the search space during learning and determine the model’s applicability.
  
## (5) Which fundamental equivalence is valid between deduction and induction?  
*(The answer needs knowledge of the definition of inductive bias).*  
-When prior assumptions (inductive bias) are incorporated, inductive learning can be regarded as an “equivalent deductive system.” 

## (6) What is supervised learning? (Or unsupervised? Or lazy?)  
- Supervised Learning: Learning the mapping between inputs and outputs using labeled data;
- Unsupervised Learning: Discovering underlying structures from unlabeled data;
- Lazy Learning: Not building a model in advance but using training data directly at prediction time (e.g., K-Nearest Neighbors). 

## (7) What is the goal?  
- The goal is to learn a model that can accurately predict new data, thereby achieving good generalization. 

## (8) How do probabilistic and deterministic approaches differ in this?  
- Probabilistic methods: Describe uncertainty using probability distributions and provide confidence measures with predictions;
-Deterministic methods: Provide fixed outputs without explicitly representing uncertainty. 

## (9) How do model selection and parameter optimization differ? Give an example.  
- Model selection: Deciding which model to use (for example, choosing between a linear, polynomial, or other nonlinear model);
- Parameter optimization: Tuning the parameters within the chosen model to minimize error (e.g., using least squares to determine polynomial coefficients).
- For instance, in polynomial curve fitting, one first selects the degree of the polynomial (model selection) and then computes the coefficients using the least squares method (parameter optimization). 

---

# Basics of Supervised Learning: Regression  

## (1) Sketch the basic example from the lecture for supervised learning! *(Draw)*  
## (2) Which concrete data model is used? *(Write down)*  
- $$y(x, w) = w_0 + w_1 \cdot x + w_2 \cdot x^2 + \dots + w_{m-1} \cdot x^{(m-1)}$$

## (3) How is learning organized here?  
- Learning is organized into two main parts:
-Model selection – choosing the form and complexity of the model (e.g. selecting the degree of the polynomial);
- Parameter optimization – learning the model parameters by minimizing the error function (the empirical error).
The lecture highlights the distinction between “create model” and “learn parameters.”

## (4) Write down the error function!  
- $$E(w) = \frac{1}{2} \sum_{n=1}^{N} \left[ y(x_n, w) - t_n \right]^2$$where N is the number of data points and tₙ the target value.

## (5) What is model selection in the example?  
- Model selection refers to choosing the model’s “form” and “complexity” based on the data.In this example, it means selecting a polynomial model and its degree, which determines the number of parameters and the model’s capacity.

## (6) What is regularization and why is it needed? *(General question, requires general answer!)*  
- Regularization is a technique to prevent overfitting by adding a penalty term to the error function to restrict the size of the model parameters, thereby controlling model complexity.Its purpose is to avoid the model from “memorizing” the noise in the training data and to improve generalization to new data.

## (7) How is regularization done in the example? *(Concrete question, requires concrete formula)*  
- In the example, regularization is performed by adding a penalty term to the sum-of-squares error function, resulting in the regularized error function: $$E_{\text{reg}}(w) = E(w) + \frac{\lambda}{2} \cdot \|w\|^2$$ where λ is the regularization parameter controlling the strength of the penalty.

## (8) Sketch the typical graph that indicates overfitting! *(BTW: What is overfitting?)*  

---

# Basics of Supervised Learning: Probabilistic Approach  

## (1) Write down the stochastic data model. Which distribution do we use for the noise?  
- The stochastic data model is $$t = y(x,w) + N(0,\beta^{-1})$$ where the noise is modeled as a Gaussian (normal) distribution, i.e., $\nu \sim \mathcal{N}(0, \beta^{-1})$.   

## (2) Sketch (i.e., draw!) the picture for the example from the lecture. Where to draw the noise?  
## (3) How to arrive at the likelihood for one point from the stochastic data model? How at the data likelihood?  
-For a single data point, from $$t = y(x,w) + N(0,\beta^{-1})$$we have $$p\left(t\mid x,w,\beta\right)=N\left(t\mid y\left(x,w\right),\beta^{-1}\right).$$Assuming that the data points are independent, the likelihood for the whole dataset is $$L(w) = P(T \mid X, w, \beta)
= \prod_{n=1}^{N} \mathcal{N}\bigl(t_n \mid y(x_n, w), \beta^{-1}\bigr) $$

## (4) Which fundamental assumption underlies the formulation of the data likelihood in this form?  
- The fundamental assumption is that the data points are independent and identically distributed (i.i.d.) and that the noise is Gaussian.

## (5) How to find the parameters from this approach? *(Write down the essential steps of the derivation of the best parameters)*  
- Write down the likelihood for the entire dataset: $$L(w) = \prod_{n=1}^{N} \mathcal{N}\bigl(t_n \mid y(x_n, w), \beta^{-1}\bigr)$$
- Take the logarithm to obtain the log-likelihood:$$ log L(w) = \sum_{n=1}^{N} log\mathcal{N}\bigl(t_n \mid y(x_n, w), \beta^{-1}\bigr)$$
- Simplify the log-likelihood using the properties of the Gaussian distribution, which leads to an expression proportional to the sum-of-squares error (plus constant terms).
- Optimize by setting the derivative with respect to 𝑤 to zero, thereby obtaining the maximum likelihood estimator $w_{ML}$.

## (6) How to generalize from the maximum likelihood?  
-Generalization is achieved by using the predictive distribution. With the obtained maximum likelihood parameters $w_{ML}$ and $\beta_{ML}$, the predictive distribution for a new input 𝑥 is $$p\left(t\mid x,w_{ML},\beta_{ML}\right)=N\left(t\mid y\left(x,w_{ML}\right),\beta_{ML}^{-1}\right),$$ which provides both the prediction and its uncertainty.

## (7) What is the relation to the deterministic approach (with the same model)?  
- Under the Gaussian noise assumption, the maximum likelihood approach is equivalent to the deterministic approach that minimizes the sum-of-squares error; both lead to the same optimal parameters and predictions.

## (8) Write down the Bayes formula and name the terms.  
- Bayes' formula is $$P(w \mid D) = \frac{P(D \mid w)\,P(w)}{P(D)}$$
where:<br>
$P(w)$ is the prior,<br>
$P(D \mid w)$ is the likelihood,<br>
$P(D)$ is the evidence (or marginal likelihood),<br>
$P(w \mid D)$ is the posterior.

## (9) Interpret the terms, what do they express?  
- Prior $P(w)$: Represents our initial belief or uncertainty about the parameters before observing any data.<br>
- Likelihood $P(D \mid w)$: Expresses the probability of observing the data 𝐷 given the parameters 𝑤.<br>
- Evidence $P(D)$: Acts as a normalization constant ensuring the posterior is a proper probability distribution.<br>
- Posterior $P(w \mid D)$: Represents our updated belief about the parameters after observing the data.

## (10) How to get the MAP parameters?  
- The MAP parameters are obtained by maximizing the posterior probability: $w_{MAP} = \arg\max_{w} P(w \mid D)$<br>In practice, this is equivalent to minimizing the negative log-posterior, which typically leads to a regularized error function.

## (11) In which sense is this equivalent to the deterministic approach? 
- The MAP estimation, when using a Gaussian prior, is equivalent to solving a regularized least-squares problem. In other words, maximizing the posterior (or minimizing the negative log-posterior) yields the same solution as the deterministic approach with regularization, thus preventing overfitting.



## (12) What is expressed by the predictive distribution and how to generalize with it?  
-The predictive distribution expresses the probability of a new output 𝑡 given a new input 𝑥 and the training data (𝑋,𝑇).<br>
In the Bayesian framework, it is obtained by integrating over the uncertainty in the parameters:$$p(t \mid x, X, T) = \int p(t \mid x, w) p(w \mid D) \, dw $$<br>
This provides not only the predicted value (mean) but also quantifies the uncertainty (variance) in the prediction.

## (13) Sketch the procedure for incremental Bayesian learning! *(General question, general answer required).*  

---

# Models  

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

# Classification  

## (1) What is the aim of classification?  
## (2) How to model labels for multi-class problems?  
## (3) What is “linear separability”?  
## (4) Sketch how to proceed for the Fisher discriminant.  
## (5) Which criterion is optimized (write it down!)? Why is this reasonable?  
## (6) What is the result?  
## (7) Write down the Bayesian Approach for 2 classes.  
## (8) Why is this named “generalized linear model”, under which condition?  
## (9) If the classes are modeled with a Gaussian function, what is the result?  
## (10) Describe max-likelihood for labels. Which trick is used for the computation?  
## (11) How to directly model the posterior?  
## (12) What is the cross-entropy, and how to arrive at it?  

---

# Concept Learning  

## (1) What is the goal of concept learning? Formalize the learning scenario.  
## (2) What is a concept? *(Give different equivalent descriptions).*  
## (3) How many concepts exist for a given space of instances?  
## (4) Why is the application of hypotheses useful? Why is direct search in the space of concepts a bad idea?  
## (5) Explain the “more general” relation and give examples.  
## (6) Special case for a hypothesis space for learning based on attributes. *(Give precise definition in mathematical terms.)*  
## (7) Give an example for what cannot be represented with this hypothesis space.  
## (8) How does search through generalization work? State problems.  
## (9) What is a consistent hypothesis? What is the version space?  
## (10) How to generalize with the version space?  
## (11) What is the inductive bias?  
## (12) What about bias-free learning? If it exists, is it useful?  
## (13) Describe the procedure for decision trees.  
## (14) Are there any issues? How to solve them?  
## (15) Define information gain. How is it computed? *(Write down in math terms.)*  
## (16) What is the hypothesis space of ID3?  
## (17) How is the search organized for ID3, and what is the inductive bias?  
## (18) What different types of inductive biases exist?  
## (19) Compare decision trees and learning by generalization/specialization, specifically regarding their biases.  

---

# Lazy Learning  

## (1) What is the principle of “lazy learning”?  
## (2) Give some simple approaches. Which variations of k-nearest neighbor do you know?  
## (3) How to choose reasonable weightings?  
## (4) What is denoted by “curse of dimensionality”? *(Give an example!)*  
## (5) Give the formula for the Nadaraya-Watson regressor. Which parameter has to be chosen?  
## (6) What is the inductive bias?  

---

# Prototypes, k-Means, GMM, EM  

## (1) Sketch the k-means approach for two clusters.  
## (2) What criterion is optimized?  
## (3) What is the E-step, what is the M-step?  
## (4) What is LBG?  
## (5) Write down the general Gaussian mixture model.  
## (6) Which parameters exist, and how to interpret them?  
## (7) What are the “responsibilities” and how to optimize them? What is the result?  

---

# Theoretical Foundations  

## (1) Describe the following pairs of notions:  
- supervised vs. unsupervised learning  
- batch vs. incremental  
- offline vs. online  
- error/cost vs. likelihood  
- empirical vs. true error  

## (2) Give an example of a cost function.  
## (3) What is generalization ability?  
## (4) How do model complexity and overfitting relate?  
## (5) What is cross-validation? How to perform it?  
## (6) How can noise help/disturb learning?  
