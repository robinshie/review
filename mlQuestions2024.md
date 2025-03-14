
# **Lazy Learning Summary**
## **(1) What is the principle of “lazy learning”?**  

### **Principle:**  
1. **Minimal Effort During Training（训练时计算量小）**  
   - Unlike traditional machine learning methods, lazy learning requires almost no computation during training.  
   - It directly stores training samples without constructing an explicit model.  

2. **High Computational Cost at Query Time（推理时计算量大）**  
   - During the prediction phase, lazy learning methods must traverse stored training samples to find the closest match.  
   - As a result, query time is usually long, and computational cost is high.  

3. **Generalization by Recombination（通过样本重组进行泛化）**  
   - Since no global model is pre-built, lazy learning methods typically make predictions by combining or interpolating stored training samples.  

[oai_citation:0‡lecture_10_unsupervised_18_06_2024v1.pdf](file-service://file-DHExcBhzm4NRuBxotU4GXZ)

## **(2) Give some simple approaches. Which variations of k-nearest neighbor do you know?**

2. **Locally Adaptive k-NN:** ## **(2) Give some simple approaches. Which variations of k-nearest neighbor do you know?**  

### **Simple Approaches（简单方法）**  

1. **Nearest Neighbor (NN) 最近邻**  
   - 输入一个查询点 **xq**，找到最近的训练样本 **x**，并直接使用该样本的输出值作为预测值。  
   - 公式表示如下：  
     \[
     \hat{x} = \arg\min_x \| x - x_q \|
     \]
   - 计算最近邻的过程开销高，对噪声不够鲁棒。  

2. **K-Nearest Neighbors (KNN) K 近邻**  
   - 查找 **k** 个最接近查询点 **xq** 的训练样本。  
   - 计算它们的平均值（回归）或进行多数投票（分类）。  
   - 公式表示如下：
     \[
     f(x_q) = \frac{1}{k} \sum_{i=1}^{k} f(\hat{x}_i)
     \]
   - 仍然具有较高的计算成本，但比单一最近邻更鲁棒。  

3. **Weighted K-Nearest Neighbors (Weighted KNN) 加权 K 近邻**  
   - 赋予每个邻居不同的权重，使距离较近的点对最终预测贡献更大。  
   - 采用欧几里得距离作为度量：  
     \[
     d(x_i, x_q) = \| x_i - x_q \|^2
     \]
   - 计算权重 **wi**：
     \[
     w_i = d(x_i, x_q)^{-1}
     \]
   - 预测值：
     \[
     f(x_q) = \frac{\sum_{i=1}^{k} w_i f(x_i)}{\sum_{i=1}^{k} w_i}
     \]
   - 该方法可以平衡不同邻居的贡献，提高预测精度。  

### **Variations of K-Nearest Neighbor（K 近邻的变体）**  

1. **Standard KNN（标准 KNN）**  
   - 仅考虑 **k** 个最近的邻居，不对其进行加权。  

2. **Weighted KNN（加权 KNN）**  
   - 使用基于距离的加权方式，使得更近的点对预测影响更大。  

3. **Inductive Bias in KNN（KNN 的归纳偏差）**  
   - KNN 依赖于“接近的输入对应相似的输出”的假设，度量距离的方法会影响模型的偏差。  

4. **Curse of Dimensionality in KNN（KNN 受维度灾难影响）**  
   - 在高维空间中，数据点变得稀疏，最近邻搜索的效果会变差。  
   - 需要额外的技巧，如降维方法或核回归。  

[oai_citation:5‡lecture_10_unsupervised_18_06_2024v1.pdf](file-service://file-DHExcBhzm4NRuBxotU4GXZ)

## **(3) How to choose reasonable weightings?**
## **How to choose reasonable weightings?**  
## **如何选择合理的权重？**  

### **Common Weighting Methods（常见的权重计算方法）**  

1. **Inverse Distance Weighting（逆距离加权）**  
   - 公式：  
     \[
     w_i = d(x_i, x_q)^{-1}
     \]
   - 距离越近，权重越高，远离的点影响较小。  
   - 可避免等权重带来的误差，但在高维数据中计算量较大。  

2. **Gaussian Kernel Weighting（高斯核加权）**  
   - 公式（Nadaraya-Watson 核回归）：  
     \[
     K_{\sigma}(x_i - x_q) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{\|x_i - x_q\|^2}{2\sigma^2}}
     \]
   - 赋予较远样本指数衰减的权重，更平滑但对超参数 \( \sigma \) 敏感。  

3. **Adaptive Weighting（自适应加权）**  
   - 根据局部密度调整权重，如自适应 KNN 动态调整 **k** 或基于局部分布优化权重策略。  
[oai_citation:5‡lecture_10_unsupervised_18_06_2024v1.pdf](file-service://file-DHExcBhzm4NRuBxotU4GXZ)
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
  f(x_q) = \sum_{i} y_i \frac {K_{\sigma}(x_i - x_q)}{\sum_{j} K_{\sigma}(x_j - x_q)}
  \]
  where:
  - \( K_{\sigma}(x_i - x_q) \) is a **kernel function**, typically Gaussian:
    \[
    K_{\sigma}(x) = \frac{1}{\sqrt{2\pi} \sigma} e^{-\frac{||x_{i}-x_{q}||^2}{2\sigma^2}}
    \]
  - **Parameter to choose**: The **kernel bandwidth \(\sigma\)**, which controls the smoothness of the predictions [oai_citation:4‡lecture_10_unsupervised_18_06_2024v1.pdf](file-service://file-949NPaqCH1RQryBXCfTwBZ).

## **(6) What is the inductive bias?**

### **Definition（定义）**  
- **Inductive bias** refers to the assumptions a learning algorithm makes to generalize from limited data.  
- **归纳偏差** 是指机器学习算法在有限数据上进行泛化时所依赖的假设。  

### **Inductive Bias in KNN（KNN 的归纳偏差）**  
1. **Smoothness Assumption（平滑性假设）**  
   - **Assumption:** "Close" inputs lead to "similar" outputs.  
   - **假设：** “相近的输入会产生相似的输出”。  

2. **Role of Distance Metric（距离度量的作用）**  
   - The definition of "closeness" depends on the **chosen distance metric**.  
   - 不同的**距离度量方法**决定了什么是“相近”。  

3. **Effect of Inductive Bias（归纳偏差的影响）**  
   - **Strong bias** (e.g., Euclidean distance) → Better generalization but may fail in non-Euclidean spaces.  
   - **弱偏差**（如自适应度量）→ 灵活但可能更容易过拟合。  

### **Key Takeaway（总结）**  
### **Definition（定义）**  
- **Inductive bias** refers to the assumptions a learning algorithm makes to generalize from limited data.  
- **归纳偏差** 是指机器学习算法在有限数据上进行泛化时所依赖的假设。  

### **Inductive Bias in Vector Quantization（向量量化中的归纳偏差）**  

1. **Prototype-based Learning（基于原型的学习）**  
   - Data points **x_n** are assigned to the **nearest prototype w_k** based on a chosen distance metric.  
   - 数据点 **x_n** 根据选定的距离度量方式被分配到**最近的原型 w_k**。  
   - **Inductive Bias:** The model assumes that proximity in the chosen metric reflects similarity.  

2. **Role of Distance Metric（距离度量的作用）**  
   - **Common metric:** Euclidean Distance  
   - 选择不同的距离度量方式（如欧几里得距离、马哈拉诺比斯距离）会影响模型的泛化能力。  
   - 归纳偏差由此引入，决定了如何定义“相似性”。  

3. **Quantization Error Minimization（量化误差最小化）**  
   - The objective function minimizes the distance between data points and their assigned prototypes:  
     \[
     J_{VQ} = \frac{1}{N} \sum_{n=1}^{N} d(x_n, w_c(x_n))
     \]
   - The assumption here is that minimizing this error leads to an optimal clustering structure.  
   - 归纳偏差影响模型如何优化原型位置以减少量化误差。  

- 归纳偏差**不可避免**，但可以通过**合适的距离度量选择**来优化。  

[oai_citation:5‡lecture_10_unsupervised_18_06_2024v1.pdf](file-service://file-DHExcBhzm4NRuBxotU4GXZ)
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
  J = \sum_{i=1}^{N} \sum_{k=1}^{K} r_{ik} \| x_i - w_k \|^2
  \]
  where:
  - \( r_{ik} = 1 \) if \( x_i \) belongs to cluster \( k \), otherwise 0.
  - \( \mu_k \) is the cluster center [oai_citation:1‡lecture_10_unsupervised_18_06_2024v1.pdf](file-service://file-949NPaqCH1RQryBXCfTwBZ).

## **(3) What is the E-step, what is the M-step?**
## E-step: Assign Data Points to Clusters for Fixed Prototypes

Each data point \( x_n \) is assigned to the closest cluster center \( w_k \):

$$
r_{nk} =
\begin{cases} 
1, & \text{if } k = \arg\min_j \| x_n - w_j \|^2 \\
0, & \text{otherwise}
\end{cases}
$$

#### M-step: Update the Cluster Center \( w_k \) for a Fixed Assignment

The new cluster center is computed as the mean of all points assigned to the cluster:

$$
w_k = \frac{\sum_{n=1}^{N} r_{nk} x_n}{\sum_{n=1}^{N} r_{nk}}
$$

#### K-means Objective Function

The goal is to minimize the following cost function:

$$
J = \sum_{i=1}^{N} \sum_{k=1}^{K} r_{ik} \| x_i - w_k \|^2
$$

- The new prototype \( w_k \) is the mean (center) of the assigned points.
- **K-means converges!**[oai_citation:2‡lecture_10_unsupervised_18_06_2024v1.pdf](file-service://file-949NPaqCH1RQryBXCfTwBZ).

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