
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