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
---------------
# **Concept Learning Summary**
## **(1) What is the goal of concept learning? Formalize the learning scenario.**
**Goal:**  
Concept learning aims to **extract a Boolean function** from instances and their labeled training data. It is equivalent to **identifying a particular subset** from training data [oai_citation:0‡lecture_08_concept-learning_04_06_2024v1.pdf](file-service://file-Mm1G4sRNyePx2trxiP1wcC).

**Formalization:**  
- Let **\(X\)** be the **set of instances**.
- A **concept** is a subset **\(C \subset X\)**.
- A **concept function** maps each instance to a label **\( c: X \to \{0,1\} \)** (indicator function).
- The **concept space** is the set of all possible functions **\( {c : X \to \{0,1\}} \)** [oai_citation:1‡lecture_08_concept-learning_04_06_2024v1.pdf](file-service://file-Mm1G4sRNyePx2trxiP1wcC).

## **(2) What is a concept? (Give different equivalent descriptions.)**
A **concept** can be defined as:
- A **Boolean classification function** over instances.
- A **subset** of the instance space **\(X\)**.
- A **mapping function** that assigns labels **\( \{0,1\} \)** to instances.
- A **hypothesis** describing which instances belong to a specific class [oai_citation:2‡lecture_08_concept-learning_04_06_2024v1.pdf](file-service://file-Mm1G4sRNyePx2trxiP1wcC).

## **(3) How many concepts exist for a given space of instances?**
The number of possible concepts in a given space of instances **\(X\)** is:
\[
\text{\# of concepts} = 2^{|X|}
\]
where **\(|X|\)** is the number of instances [oai_citation:3‡lecture_08_concept-learning_04_06_2024v1.pdf](file-service://file-Mm1G4sRNyePx2trxiP1wcC).

## **(4) Why is the application of hypotheses useful? Why is direct search in the space of concepts a bad idea?**
- **Hypotheses enable generalization** by restricting search to a smaller space.
- **Direct search is computationally infeasible**, as the number of concepts grows exponentially **(\(2^{|X|}\))**.
- **Inductive bias** helps limit the search space and improves learning efficiency [oai_citation:4‡lecture_08_concept-learning_04_06_2024v1.pdf](file-service://file-Mm1G4sRNyePx2trxiP1wcC).

## **(5) Explain the “more general” relation and give examples.**
A hypothesis **\(h_i\)** is **more general** than **\(h_j\)** if:
\[
\forall x \in X : (h_j(x) = 1 \Rightarrow h_i(x) = 1)
\]
This means **\(h_i\)** includes all instances of **\(h_j\)** and possibly more [oai_citation:5‡lecture_08_concept-learning_04_06_2024v1.pdf](file-service://file-Mm1G4sRNyePx2trxiP1wcC).

**Example:**  
- \( h_1 = (\text{All red objects}) \) is more general than \( h_2 = (\text{All red circles}) \).

## **(6) Special case for a hypothesis space for learning based on attributes. (Give a precise mathematical definition.)**
For attribute-based learning, hypotheses are **conjunctions of attribute constraints**:
\[
h = \langle a_1, a_2, ..., a_n \rangle
\]
where:
- \(a_i\) can be **specific values**, **wildcards (\(?\))**, or **empty (\(\emptyset\))**.
- Example hypothesis: \( \langle \text{red, ?} \rangle \) means "all red objects" [oai_citation:6‡lecture_08_concept-learning_04_06_2024v1.pdf](file-service://file-Mm1G4sRNyePx2trxiP1wcC).

## **(7) Give an example for what cannot be represented with this hypothesis space.**
- **Non-conjunctive rules**, such as **disjunctions**:  
  Example: "Red **or** Circular" cannot be represented as it is not a strict conjunction.
- **Continuous decision boundaries**, like a **linear separator in feature space** [oai_citation:7‡lecture_08_concept-learning_04_06_2024v1.pdf](file-service://file-Mm1G4sRNyePx2trxiP1wcC).

## **(8) How does search through generalization work? State problems.**
- **Start with the most specific hypothesis** (e.g., \( \emptyset \)).
- **Gradually generalize** by replacing attribute constraints with wildcards (\(?\)).
- **Problem:** Risk of overgeneralization (leading to false positives) [oai_citation:8‡lecture_08_concept-learning_04_06_2024v1.pdf](file-service://file-Mm1G4sRNyePx2trxiP1wcC).

## **(9) What is a consistent hypothesis? What is the version space?**
- A **consistent hypothesis** correctly classifies all training examples.
- The **version space** is the set of all consistent hypotheses [oai_citation:9‡lecture_08_concept-learning_04_06_2024v1.pdf](file-service://file-Mm1G4sRNyePx2trxiP1wcC).

## **(10) How to generalize with the version space?**
- Maintain **\(G\)-set (most general hypotheses)**.
- Maintain **\(S\)-set (most specific hypotheses)**.
- Any hypothesis lying between **\(S\) and \(G\)** is a valid generalization [oai_citation:10‡lecture_08_concept-learning_04_06_2024v1.pdf](file-service://file-Mm1G4sRNyePx2trxiP1wcC).

## **(11) What is the inductive bias?**
- **Inductive bias** is the **set of assumptions** a learning algorithm makes to generalize beyond training data.
- Example: Decision trees favor **simpler trees** (Occam’s razor) [oai_citation:11‡lecture_08_concept-learning_04_06_2024v1.pdf](file-service://file-Mm1G4sRNyePx2trxiP1wcC).

## **(12) What about bias-free learning? If it exists, is it useful?**
- **Bias-free learning** would require searching the entire concept space.
- **Not useful**, as it leads to an **exponential search problem** (no generalization) [oai_citation:12‡lecture_08_concept-learning_04_06_2024v1.pdf](file-service://file-Mm1G4sRNyePx2trxiP1wcC).

## **(13) Describe the procedure for decision trees.**
1. Select **best attribute** using **Information Gain**.
2. Partition data **recursively**.
3. Stop when **all instances belong to the same class** or **no attributes remain** [oai_citation:13‡lecture_09_Decision-Trees+Bayesian-Concept-11_06_2024v1.pdf](file-service://file-T1j5eHdREayqkvNyrxJSRy).

## **(14) Are there any issues? How to solve them?**
**Issues:**
- **Overfitting** (tree too complex).
- **Irrelevant attributes** may mislead the tree.
- **Handling missing values** is difficult.

**Solutions:**
- **Pruning** (cut back the tree).
- **Feature selection**.
- **Handling missing values with probabilities** [oai_citation:14‡lecture_09_Decision-Trees+Bayesian-Concept-11_06_2024v1.pdf](file-service://file-T1j5eHdREayqkvNyrxJSRy).

## **(15) Define information gain. How is it computed? (Write down in math terms.)**
\[
IG(A) = H(D) - \sum_{v \in A} \frac{|D_v|}{|D|} H(D_v)
\]
where:
- \(H(D)\) is **entropy before the split**.
- \(H(D_v)\) is **entropy after splitting on attribute \(A\)** [oai_citation:15‡lecture_09_Decision-Trees+Bayesian-Concept-11_06_2024v1.pdf](file-service://file-T1j5eHdREayqkvNyrxJSRy).

## **(16) What is the hypothesis space of ID3?**
- **Set of all possible decision trees**.
- **Includes all finite discrete classifications**.
- **No bias from a-priori restrictions** [oai_citation:16‡lecture_09_Decision-Trees+Bayesian-Concept-11_06_2024v1.pdf](file-service://file-T1j5eHdREayqkvNyrxJSRy).

## **(17) How is the search organized for ID3, and what is the inductive bias?**
- **Search:** Uses **greedy top-down search** with **Information Gain**.
- **Inductive Bias:**
  - Prefers **shorter trees**.
  - Assumes **attributes are conditionally independent** given the class [oai_citation:17‡lecture_09_Decision-Trees+Bayesian-Concept-11_06_2024v1.pdf](file-service://file-T1j5eHdREayqkvNyrxJSRy).

## **(18) What different types of inductive biases exist?**
1. **Restriction Bias:** Limits hypothesis space (e.g., linear models).
2. **Preference Bias:** Prefers certain hypotheses (e.g., Occam’s razor).
3. **Algorithmic Bias:** Comes from how hypotheses are updated [oai_citation:18‡lecture_08_concept-learning_04_06_2024v1.pdf](file-service://file-Mm1G4sRNyePx2trxiP1wcC).

## **(19) Compare decision trees and learning by generalization/specialization, specifically regarding their biases.**
| **Aspect**          | **Decision Trees (ID3)**       | **Generalization/Specialization** |
|--------------------|--------------------------------|--------------------------------|
| **Search**         | Greedy search using entropy    | Incremental update |
| **Inductive Bias** | Prefers shorter trees (Occam)  | Structure of hypothesis space |
| **Efficiency**     | Fast for categorical data      | Slower, depends on constraints |
| **Flexibility**    | Limited to discrete attributes | Works with structured constraints [oai_citation:19‡lecture_08_concept-learning_04_06_2024v1.pdf](file-service://file-Mm1G4sRNyePx2trxiP1wcC). |

-------------------
-------------------
# **Lazy Learning Summary**
## **(1) What is the principle of “lazy learning”?**
**Principle:**
- **Lazy learning** is a **memory-based approach** where learning happens at query time.
- Instead of building an explicit model, it **stores training data** and **performs computations during prediction**.
- **Generalization** is achieved by **recombining stored samples** dynamically at runtime [oai_citation:0‡lecture_10_unsupervised_18_06_2024v1.pdf](file-service://file-949NPaqCH1RQryBXCfTwBZ).

## **(2) Give some simple approaches. Which variations of k-nearest neighbor do you know?**
### **Simple approaches:**
- **Nearest Neighbor (NN):** Assigns the label of the nearest training sample.
- **k-Nearest Neighbors (k-NN):** Takes the average (or majority vote) of the \(k\) nearest samples.

### **Variations of k-NN:**
1. **Weighted k-Nearest Neighbors (Wk-NN):**
   - Weights neighbors based on their distance to the query point.
   - Example: Using inverse Euclidean distance as weight:
     \[
     w_i = \frac{1}{d(x_i, x_q)}
     \]
2. **Locally Adaptive k-NN:** Dynamically selects the value of \(k\) based on local density [oai_citation:1‡lecture_10_unsupervised_18_06_2024v1.pdf](file-service://file-949NPaqCH1RQryBXCfTwBZ).

## **(3) How to choose reasonable weightings?**
- **Common weighting strategies:**
  - **Inverse Distance Weighting:** \( w_i = \frac{1}{d(x_i, x_q)} \).
  - **Gaussian Kernel:** Assigns weights using a Gaussian function:
    \[
    w_i = \exp \left( -\frac{d(x_i, x_q)^2}{2\sigma^2} \right)
    \]
- **Normalization**: Ensures all weights sum to 1:
  \[
  f(x_q) = \frac{\sum_{i=1}^{k} w_i f(x_i)}{\sum_{i=1}^{k} w_i}
  \]
- **Choice of weights depends on:** Noise level, data sparsity, and computational efficiency [oai_citation:2‡lecture_10_unsupervised_18_06_2024v1.pdf](file-service://file-949NPaqCH1RQryBXCfTwBZ).

## **(4) What is denoted by “curse of dimensionality”? (Give an example!)**
- **Curse of dimensionality:** In high-dimensional spaces:
  - Data points become **sparsely distributed**.
  - **Distances lose meaning**, making nearest neighbor approaches ineffective.

### **Example:**
- Consider a **unit sphere** with radius **\(r=1\)** in **\(D\)-dimensional space**.
- The volume of an inner sphere with radius \( (1-\epsilon) \) is:
  \[
  V_D(1 - \epsilon) = K_D (1 - \epsilon)^D
  \]
- As **\(D \to \infty\)**, the relative volume in the outer shell approaches **1**, meaning nearly all the volume is in the shell, making it hard to find meaningful neighbors [oai_citation:3‡lecture_10_unsupervised_18_06_2024v1.pdf](file-service://file-949NPaqCH1RQryBXCfTwBZ).

## **(5) Give the formula for the Nadaraya-Watson regressor. Which parameter has to be chosen?**
- **Nadaraya-Watson estimator (Kernel Regression):**
  \[
  f(x_q) = \frac{\sum_{i} y_i K_{\sigma}(x_i - x_q)}{\sum_{j} K_{\sigma}(x_j - x_q)}
  \]
  where:
  - \( K_{\sigma}(x_i - x_q) \) is a **kernel function**, typically Gaussian:
    \[
    K_{\sigma}(x) = \frac{1}{\sqrt{2\pi} \sigma} e^{-\frac{x^2}{2\sigma^2}}
    \]
  - **Parameter to choose**: The **kernel bandwidth \(\sigma\)**, which controls the smoothness of the predictions [oai_citation:4‡lecture_10_unsupervised_18_06_2024v1.pdf](file-service://file-949NPaqCH1RQryBXCfTwBZ).

## **(6) What is the inductive bias?**
- **Inductive bias in lazy learning**:  
  - Assumes **“close” instances should have similar outputs**.
  - Imposed **through the choice of distance metric** (e.g., Euclidean, Mahalanobis).
  - k-NN assumes **locally linear decision boundaries** [oai_citation:5‡lecture_13_on_the_learning_09_07_2024.pdf](file-service://file-DViwf5dAJGMuJZBejCSkEt).
---

# **Prototypes, k-Means, GMM, EM Summary**

## **(1) Sketch the k-means approach for two clusters.**
1. **Initialization**: Select **K = 2** cluster centroids randomly.
2. **Assignment Step (E-Step)**:
   - Assign each data point **\( x_i \)** to the nearest cluster centroid **\( \mu_k \)** based on Euclidean distance.
3. **Update Step (M-Step)**:
   - Recalculate cluster centroids as the **mean** of all assigned points.
4. **Repeat** steps 2 and 3 until convergence (centroids stop changing) [oai_citation:0‡lecture_10_unsupervised_18_06_2024v1.pdf](file-service://file-949NPaqCH1RQryBXCfTwBZ).

## **(2) What criterion is optimized?**
- **Objective function:** Sum of squared Euclidean distances from each point to its assigned cluster centroid:
  \[
  J = \sum_{i=1}^{N} \sum_{k=1}^{K} r_{ik} || x_i - \mu_k ||^2
  \]
  where:
  - \( r_{ik} = 1 \) if \( x_i \) belongs to cluster \( k \), otherwise 0.
  - \( \mu_k \) is the cluster center [oai_citation:1‡lecture_10_unsupervised_18_06_2024v1.pdf](file-service://file-949NPaqCH1RQryBXCfTwBZ).

## **(3) What is the E-step, what is the M-step?**
- **Expectation Step (E-Step)**:
  - Assign data points to the cluster with the highest probability or minimum distance.
- **Maximization Step (M-Step)**:
  - Update cluster centroids (in k-means) or estimate new parameters (in GMM) [oai_citation:2‡lecture_10_unsupervised_18_06_2024v1.pdf](file-service://file-949NPaqCH1RQryBXCfTwBZ).

## **(4) What is LBG?**
- **Linde-Buzo-Gray (LBG) Algorithm**:
  - A **vector quantization** method used for clustering.
  - Works by **splitting centroids** iteratively and refining them using **k-means**.
  - Used in **speech processing** and **codebook generation** [oai_citation:3‡lecture_10_unsupervised_18_06_2024v1.pdf](file-service://file-949NPaqCH1RQryBXCfTwBZ).

## **(5) Write down the general Gaussian mixture model.**
\[
p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)
\]
where:
- \( \pi_k \) is the **mixing coefficient** (\(\sum_k \pi_k = 1\)).
- \( \mathcal{N}(x | \mu_k, \Sigma_k) \) is a **multivariate Gaussian distribution** [oai_citation:4‡lecture_10_unsupervised_18_06_2024v1.pdf](file-service://file-949NPaqCH1RQryBXCfTwBZ).

## **(6) Which parameters exist, and how to interpret them?**
- **\( \pi_k \) (Mixing Coefficients):** Probability of each component (weight of each Gaussian).
- **\( \mu_k \) (Mean Vectors):** Center of each Gaussian cluster.
- **\( \Sigma_k \) (Covariance Matrices):** Spread/shape of the clusters.
- **\( K \) (Number of Gaussians):** Defines the number of mixture components [oai_citation:5‡lecture_11_GMM_GMR_25_06_2024.pdf](file-service://file-Fyhrm8XccUhtWpPBepmkdy).

## **(7) What are the “responsibilities” and how to optimize them? What is the result?**
- **Responsibilities \( \gamma_{ik} \)**:  
  - Probability that data point \( x_i \) belongs to Gaussian \( k \).
  - Given by Bayes' rule:
    \[
    \gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}
    \]

- **Optimization via Expectation-Maximization (EM) Algorithm**:
  1. **E-Step:** Compute responsibilities \( \gamma_{ik} \).
  2. **M-Step:** Update \( \pi_k, \mu_k, \Sigma_k \) using weighted means.
  3. **Repeat** until convergence [oai_citation:6‡lecture_10_unsupervised_18_06_2024v1.pdf](file-service://file-949NPaqCH1RQryBXCfTwBZ).

- **Result:** A **soft clustering** model where each data point has a probability of belonging to multiple clusters, unlike k-means which provides **hard assignments** [oai_citation:7‡lecture_10_unsupervised_18_06_2024v1.pdf](file-service://file-949NPaqCH1RQryBXCfTwBZ).

---

# **Theoretical Foundations Summary**

## **(1) Describe the following pairs of notions:**
### **Supervised vs. Unsupervised Learning**
- **Supervised Learning**: The model is trained on labeled data \( (X, Y) \), where each input has a known output (e.g., classification, regression).
- **Unsupervised Learning**: The model finds patterns in **unlabeled data** \( X \) (e.g., clustering, PCA) [oai_citation:0‡lecture_02_regression_and_notions_22_04_2024v1.pdf](file-service://file-XPHndy64cK8xsipCi7KtYD).

### **Batch vs. Incremental Learning**
- **Batch Learning**: The model is trained using the entire dataset at once.
- **Incremental Learning**: The model updates **sequentially** as new data arrives (e.g., online learning) [oai_citation:1‡lecture_13_on_the_learning_09_07_2024.pdf](file-service://file-DViwf5dAJGMuJZBejCSkEt).

### **Offline vs. Online Learning**
- **Offline Learning**: Training occurs **before deployment** using a fixed dataset.
- **Online Learning**: The model **updates continuously** with new incoming data (e.g., stock prediction) [oai_citation:2‡lecture_13_on_the_learning_09_07_2024.pdf](file-service://file-DViwf5dAJGMuJZBejCSkEt).

### **Error/Cost vs. Likelihood**
- **Error/Cost Function**: Measures **how far** predictions are from actual values.
  \[
  E(w) = \frac{1}{2} \sum_{n=1}^{N} (y(x_n, w) - t_n)^2
  \]
- **Likelihood**: The probability of observing data given model parameters.
  \[
  P(D | w) = \prod_{\alpha=1}^{M} P(z_{\alpha} | w) = e^{-E_D(w)}
  \]
  The two are **inversely related** [oai_citation:3‡lecture_13_on_the_learning_09_07_2024.pdf](file-service://file-DViwf5dAJGMuJZBejCSkEt).

### **Empirical vs. True Error**
- **Empirical Error**: The error measured on the **training set**:
  \[
  E_{\text{emp}}(w) = \frac{1}{M} \sum_{\alpha=1}^{M} E(z_{\alpha}, w)
  \]
- **True Error**: The expected error over **all possible data points**:
  \[
  E_{\infty}(w) = \int E(z, w) P(z) dz
  \]
  Minimizing empirical error **does not guarantee** minimal true error (overfitting risk) [oai_citation:4‡lecture_13_on_the_learning_09_07_2024.pdf](file-service://file-DViwf5dAJGMuJZBejCSkEt).

## **(2) Give an example of a cost function.**
**Quadratic Loss (Mean Squared Error - MSE)**:
\[
E(w) = \frac{1}{2N} \sum_{n=1}^{N} (y(x_n, w) - t_n)^2
\]
Used in regression problems to **minimize squared differences** [oai_citation:5‡lecture_02_regression_and_notions_22_04_2024v1.pdf](file-service://file-XPHndy64cK8xsipCi7KtYD).

## **(3) What is generalization ability?**
- The ability of a model to **perform well on unseen data**.
- A well-generalized model **minimizes the gap** between training error and test error.
- **Key factors** affecting generalization:
  - Amount of training data.
  - Complexity of the model.
  - Proper use of regularization [oai_citation:6‡lecture_02_regression_and_notions_22_04_2024v1.pdf](file-service://file-XPHndy64cK8xsipCi7KtYD).

## **(4) How do model complexity and overfitting relate?**
- **Low complexity**: Underfits data, poor accuracy.
- **High complexity**: Overfits training data, poor generalization.
- **Trade-off**:
  - Simple models (e.g., linear regression) generalize well but may underfit.
  - Complex models (e.g., deep networks) may overfit without regularization.
  - Solution: Use **cross-validation and regularization** [oai_citation:7‡lecture_02_regression_and_notions_22_04_2024v1.pdf](file-service://file-XPHndy64cK8xsipCi7KtYD).

## **(5) What is cross-validation? How to perform it?**
- **Cross-validation** estimates model performance by **dividing data into training and validation sets**.
- **Steps for k-fold cross-validation**:
  1. Split data into **\( k \) equal folds**.
  2. Train the model on **\( k-1 \) folds**, test on the remaining fold.
  3. Repeat for all \( k \) folds.
  4. Compute the **average validation error**.
- Common types:
  - **k-fold Cross-Validation**: \( k = 5 \) or \( 10 \) is common.
  - **Leave-One-Out (LOO)**: Each data point is used as a test set **once** [oai_citation:8‡lecture_13_on_the_learning_09_07_2024.pdf](file-service://file-DViwf5dAJGMuJZBejCSkEt).

## **(6) How can noise help/disturb learning?**
- **Disturbance (Negative Impact)**:
  - **High noise** increases variance → Overfitting.
  - **Noisy labels** mislead model learning.
  - Causes **higher generalization error**.

- **Helpful (Positive Impact)**:
  - **Adding noise** can help prevent overfitting (e.g., data augmentation).
  - **Dropout** in neural networks acts as **regularization**.
  - **Stochastic Gradient Descent (SGD)** benefits from **noise-induced escape from local minima** [oai_citation:9‡lecture_05_error_minimization_06_05_2024v1.pdf](file-service://file-3qtCmAREhyo7KL8j3WNqVo).