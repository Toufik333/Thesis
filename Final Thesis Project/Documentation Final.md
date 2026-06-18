# Documentation: Integrative Explainable AI and Federated Learning Framework for Enhanced Early Breast Cancer Detection

This document provides a comprehensive technical breakdown of the multi-modal, privacy-preserving breast cancer detection framework developed for your thesis: **"Integrative Explainable AI and Federated Learning Framework for Enhanced Early Breast Cancer Detection via Multi-Omic Liquid Biopsy Profiling"**.

---

## 1. System Architecture & Multi-Omic Heterogeneous Clients

The framework addresses the challenge of heterogeneous data distributions across distinct medical institutions (hospitals) by utilizing a **Split-Learning Inspired Federated Learning (FL) Architecture**. Instead of requiring clients to share raw text or image records, each hospital possesses a private local feature encoder suited for its specific data modality. These encoders map local inputs into a unified latent space ($\text{Embedding Dim} = 64$), which is then fed into a globally federated classification head (`shared_head`).

```
[Hospital 1: Tabular Cell Features]  --> [Local MLP Encoder]  ──┐
[Hospital 2: Tabular Biomarkers]     --> [Local MLP Encoder]  ──┼─> [Global Shared Classification Head]
[Hospital 3: Histopathology Images]  --> [Local CNN Encoder]  ──┘

```

### Modality & Network Outline

1. **Hospital 1 (WDBC Dataset)**: 23 tabular cell nuclear features $\rightarrow$ `Hospital1_MLP` encoder with a `dropout_rate=0.3`.
2. **Hospital 2 (Coimbra Dataset)**: 9 tabular blood biomarker features $\rightarrow$ `Hospital2_MLP` encoder with a `dropout_rate=0.3`.
3. **Hospital 3 (BreakHis Dataset)**: 3-channel histopathology images ($224 \times 224$) $\rightarrow$ `Hospital3_CNN` encoder built upon a pre-trained **MobileNetV2** backbone with a `dropout_rate=0.2`.

---

## 2. Phase-by-Phase Technical Breakdown

### Phase 1: Data Preparation & Preprocessing

**Notebook Reference**: `01_data_prep(1).ipynb`

* **Hospital 1 (WDBC)**:
* Total rows: 569 samples (Benign: 357, Malignant: 212).
* Non-predictive identification markers (`id`) and trailing null parsing blocks (`Unnamed: 32`) are programmatically dropped.
* Labels are converted using categorical binary encoding (`Malignant = 1`, `Benign = 0`).


* **Hospital 2 (Coimbra)**:
* Total rows: 116 samples.
* Tabular parameters represent specialized clinical blood analytics: Age, BMI, Glucose, Insulin, HOMA, Leptin, Adiponectin, Resistin, and MCP-1.


* **Hospital 3 (BreakHis)**:
* Histopathology visual structures are statically downsampled to $224 \times 224$ pixels.
* Pixel intensity values are tensor-scaled and standardized using standard ImageNet normalization vectors:

$$\text{Mean} = [0.485, 0.456, 0.406], \quad \text{STD} = [0.229, 0.224, 0.225]$$





---

### Phase 2: Federated Optimization & Centralized Baselines

**Notebook References**: `04_federated_learning_colab.ipynb`, `04_federated_learning_weighted_colab.ipynb`, `07_centralized_baseline_colab.ipynb`

This stage executes the primary cross-institutional model weight federation using variants of the **Federated Averaging (FedAvg)** optimization primitive. Each learning transaction strictly isolates local network layers, transmitting only the output parameters of the classification layer to the server.

* **FedAvg Equal vs. Weighted Execution**:
* **FedAvg Equal**: Aggregates models by computing an unweighted parameters mean across active clients:

$$\theta_{\text{global}}^{t+1} = \frac{1}{N}\sum_{i=1}^{N}\theta_{i}^{t+1}$$


* **FedAvg Weighted**: Weights update tensors proportionally according to the local participant repository magnitude:

$$\theta_{\text{global}}^{t+1} = \sum_{i=1}^{N}\frac{n_i}{n_\text{total}}\theta_{i}^{t+1}$$





#### Centralized Baseline Analysis (Upper Bound)

To compute the true cost of structural data isolation, a **Centralized Non-Private Baseline** was established by pooling all unencrypted representations into a single processing node.

* **The Privacy Paradox Observation**: Surprisingly, **FedAvg Equal outperformed the Centralized Model** across all client domains. In highly skewed non-IID conditions, centralized optimization gets pulled toward dominating data modes. The structural architectural boundaries of Federated Averaging act as a regularizer, preventing larger datasets from washing out the signals of smaller, clinical-biomarker sets.

---

### Phase 3: Tabular Small-Sample Augmentation via GMM

**Notebook Reference**: `05_coimbra_augmentaion_colab.ipynb`

Hospital 2 (Coimbra) presents a standard medical "small-sample" limitation with only 116 instances. Traditional deep generative models like Generative Adversarial Networks (GANs) are highly unstable on tiny tabular matrices.

* **The GMM Strategy**: To expand the local sample population safely, a class-conditioned **Gaussian Mixture Model (GMM)** with two sub-components was fitted to the standard scaled features.
* **Benefits Over GANs**:
1. **Convergence Stability**: GMMs optimize reliably on small sets using the Expectation-Maximization algorithm, avoiding GAN mode collapse.
2. **Statistical Quality Guarantee**: Synthetic data arrays are filtered through the pre-trained global federated embedding vectors, dropping points that deviate from authentic physiological feature bounds.
3. **Privacy Defenses**: Samples represent continuous density mixtures rather than direct points, preventing individual patient replication.



---

### Phase 4: Advanced Architectural Learning Paradigms

**Notebook References**: `05_1_coimbra_domain_invariant_colab.ipynb`, `04_4_federated_learning_fedproto_ft_colab.ipynb`

To regularize variations across target domains, two specialized decentralized algorithmic variants were deployed:

1. **Domain-Invariant Feature Alignment**: Utilizes a Maximum Mean Discrepancy (MMD) or Domain Adversarial penalty to minimize the divergence between feature distributions across clients, forcing the network to build generalized representations.
2. **Federated Prototype Learning (FedProto)**: Instead of passing neural network weight layers, clients exchange local class-mean semantic clusters (prototypes). Local loss is regularized by forcing embeddings to move toward their matching global target prototype:

$$\mathcal{L}_{\text{local}} = \mathcal{L}_{\text{CE}} + \mu \, \Big\| f(x_i) - \overline{\mathbf{C}}_{\text{global}}^{(y_i)} \Big\|_2^2$$


3. **Metrics Annealing Fine-Tuning**: Applies step-down scalar constraints on fine-tuning learning parameters, preventing catastrophic forgetting of the globally learned features while allowing local optimization.

---

### Phase 5: Multi-Layer Robust Validation

**Notebook Reference**: `06_coimbra_validation_kaggle.ipynb`

To prevent model inflation and ensure clinical validity, the framework is tested against a strict **3-Layer Performance Verification Framework**:

* **Layer 1: Empirical Cross-Transfer Verification**: Evaluates performance gains step-by-step by combining subsets (e.g., Coimbra + WDBC vs. Full Integration) against a random label control model to prove authentic cross-modality learning.
* **Layer 2: Statistical Stability Verification**: Runs a comprehensive 25-run, $5 \times 5$ Cross-Validation pipeline to extract exact variance ranges ($\pm \sigma$) for accuracy, precision, recall, and F1-score.
* **Layer 3: Latent Representation Geometry Verification**: Calculates the structural shift in embedding space by checking the cosine similarity of latent vectors before and after federation. This tracks how well the collaborative global head refines class boundaries for local clients.

---

### Phase 6: Privacy-Preserving Differential Privacy (DP)

**Notebook Reference**: `08_diff_privacy_on_best_method_kaggle.ipynb`

To guarantee mathematically formal user confidentiality against model inversion attacks, bounded **Gaussian Noise** ($\sigma$) is injected directly into the aggregated global shared head parameters after each update cycle.

* **Mathematical Definition**: The noise coefficient tracks client sample size properties through standard composition models:

$$\sigma = \frac{\Delta S \sqrt{2\ln(1.25/\delta)}}{\varepsilon}$$



Where $\Delta S = 1.0$ defines the strict local parameter gradient clipping constraint, and the failure probability parameter is locked at $\delta = 1\times 10^{-5}$.
* **Privacy Budget $\varepsilon$ Evaluation Matrix**:
* **Low DP Privacy Protection ($\varepsilon = 10$)**: $\sigma = 0.4845$
* **Medium DP Privacy Protection ($\varepsilon = 1$)**: $\sigma = 4.8448$
* **High DP Privacy Protection ($\varepsilon = 0.1$)**: $\sigma = 48.4481$



---

### Phase 7: Explainable AI (XAI) Interpretation via SHAP

**Notebook Reference**: `09_shap_onto_the_best_model_kaggle.ipynb`

To transition the deep neural network away from a "black box" architecture toward clinical utility, a **SHAP (SHapley Additive exPlanations)** framework layer is applied to the tabular inference pipelines.

* **Mechanics**: By tracking feature contributions across game-theoretic permutations, the model extracts exact local and global importance scores. This provides medical professionals with interpretable, case-by-case breakdowns of which biometric values or biomarker concentrations triggered an elevated risk prediction.

---

## 3. Consolidated Experimental Results Matrix

The table below tracks performance metrics across the entire algorithmic progression:

| Algorithm / Configuration Phase | Hospital 1 (WDBC) Acc | Hospital 2 (Coimbra) Acc | Hospital 3 (BreakHis) Acc |
| --- | --- | --- | --- |
| **Local-Only Model Baseline** | 0.9474 | 0.7083 | 0.9312 |
| **Centralized Baseline (No Privacy Upper Bound)** | 0.9737 | 0.6667 | 0.9394 |
| **FedAvg Weighted Optimization** | 0.9561 | 0.7500 | 0.9635 |
| **FedAvg Equal Optimization (Top Performer)** | **0.9825** | **0.7917** | **0.9919** |
| **FedProx Optimization Regularized ($\mu=0.01$)** | 0.9825 | 0.7917 | 0.9879 |
| **FedProto Representation Setup** | 0.9474 | 0.7500 | 0.9514 |
| **FedProto + Metrics Annealing Fine-Tuning** | 0.9298 | 0.7083 | 0.9393 |
| **Domain-Invariant Optimization Model** | 0.9649 | 0.7917 | 0.9798 |
| **Differential Privacy Low Noise Constraints ($\varepsilon=10$)** | 0.9649 | 0.7500 | 0.9757 |
| **Differential Privacy Medium Noise Constraints ($\varepsilon=1$)** | 0.9123 | 0.6250 | 0.8947 |
| **Differential Privacy High Noise Constraints ($\varepsilon=0.1$)** | 0.5439 | 0.4583 | 0.5101 |

---

## 4. Key Thesis Interpretations & Conclusions

1. **The Privacy-Performance Synergy**: The empirical results demonstrate that privacy-preserving federated frameworks do not inherently compromise model accuracy. The **FedAvg Equal** model consistently outperformed the non-private centralized baseline across all three hospitals, highlighting the self-regularizing advantages of decentralized architectures on heterogeneous datasets.
2. **Mitigating Modality Scarcity**: Incorporating class-conditioned GMM augmentation for small-sample tabular datasets (Coimbra) provides an effective performance boost without risking the mode collapse or training instability typical of GANs.
3. **The Cost of Privacy**: Evaluating the Differential Privacy curve reveals a clear trade-off between privacy and accuracy. Bounding updates with a low-noise threshold ($\varepsilon=10$) preserves high clinical accuracy while adding robust protection. However, strict noise bounds ($\varepsilon=0.1$) introduce significant disruption, signaling clear boundaries for future research in private optimization.