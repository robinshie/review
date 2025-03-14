

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