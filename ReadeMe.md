# Graph Neural Network Fraud Detection System

A Graph Neural Network (GNN)-based fraud detection system for identifying illicit cryptocurrency transactions using the **Elliptic Bitcoin Dataset**. This project benchmarks **GCN**, **GraphSAGE**, and **Graph Attention Networks (GAT)** for graph-based node classification and addresses the severe class imbalance present in real-world financial transaction networks.

---

## Overview

Financial fraud is inherently a graph problem, where transactions are interconnected rather than independent. Traditional machine learning models process each transaction individually, often failing to capture suspicious transaction patterns spread across multiple connected wallets.

This project models the Bitcoin transaction network as a graph and applies Graph Neural Networks to classify each transaction as either:

* **Licit (Legitimate)**
* **Illicit (Fraudulent)**

The implementation compares three popular GNN architectures:

* Graph Convolutional Network (GCN)
* GraphSAGE
* Graph Attention Network (GAT)

---

## Features

* Automatic dataset download using KaggleHub
* Complete preprocessing pipeline
* Graph construction using PyTorch Geometric
* Feature normalization using StandardScaler
* Temporal train/validation/test split
* Class-weighted CrossEntropy Loss for imbalance handling
* Benchmarking of multiple GNN architectures
* Performance evaluation using Precision, Recall, F1-Score and Accuracy
* Automatic generation of comparison results

---

## Dataset

**Dataset:** Elliptic Bitcoin Dataset

The dataset represents Bitcoin transactions as a directed graph.

### Dataset Statistics

| Property                 |   Value |
| ------------------------ | ------: |
| Transactions (Nodes)     | 203,769 |
| Payment Flows (Edges)    | 234,355 |
| Features per Transaction |     165 |
| Time Steps               |      49 |
| Fraudulent Transactions  |     ~2% |
| Legitimate Transactions  |    ~21% |
| Unlabeled Transactions   |    ~77% |

Each transaction is represented as:

* Node features
* Directed transaction edges
* Binary fraud label (licit / illicit)

---

## Technologies Used

* Python
* PyTorch
* PyTorch Geometric
* Pandas
* NumPy
* Scikit-learn
* KaggleHub

---

## Project Pipeline

```
Dataset
      │
      ▼
Load CSV Files
      │
      ▼
Preprocessing
 • Label Mapping
 • Feature Scaling
 • Node Mapping
      │
      ▼
Graph Construction
(PyTorch Geometric)
      │
      ▼
Train GCN
Train GraphSAGE
Train GAT
      │
      ▼
Performance Evaluation
      │
      ▼
Model Comparison
```

---

## Model Architectures

### 1. Graph Convolutional Network (GCN)

Uses neighborhood aggregation to learn node representations by averaging information from adjacent transactions.

**Architecture**

```
Input Features
      │
GCN Layer
      │
ReLU
      │
Dropout
      │
GCN Layer
      │
Output
```

---

### 2. GraphSAGE

Learns node embeddings by aggregating neighboring information while supporting inductive learning for unseen nodes.

**Architecture**

```
Input
   │
SAGEConv
   │
ReLU
   │
Dropout
   │
SAGEConv
   │
Output
```

---

### 3. Graph Attention Network (GAT)

Uses attention mechanisms to assign different importance to neighboring transactions.

**Architecture**

```
Input
   │
Multi-Head Attention
   │
ELU
   │
Dropout
   │
Attention Layer
   │
Output
```

---

## Handling Class Imbalance

The Elliptic dataset contains only approximately **2% fraudulent transactions**, making it highly imbalanced.

To improve fraud detection, the project uses **class-weighted CrossEntropy Loss**, assigning a larger penalty to misclassified fraudulent transactions.

This significantly improves recall on the minority class compared to standard cross-entropy.

---

## Results

| Model     | Accuracy   | Precision | Recall    | F1-Score  |
| --------- | ---------- | --------- | --------- | --------- |
| GCN       | **64.62%** | 0.036     | 0.503     | 0.067     |
| GraphSAGE | **73.01%** | 0.040     | 0.420     | 0.073     |
| GAT       | **55.29%** | **0.044** | **0.811** | **0.084** |

---

## Key Observations

* Graph Attention Networks (GAT) achieved the highest fraud detection capability.
* GAT successfully identified over **81% of fraudulent transactions**, significantly outperforming GCN and GraphSAGE in recall.
* GraphSAGE achieved the highest overall accuracy but detected fewer fraudulent transactions.
* For highly imbalanced fraud detection tasks, **Recall** and **F1-score** are more meaningful evaluation metrics than overall accuracy.

---

## Repository Structure

```
.
├── gnn_project.py
├── model_comparison.csv
├── README.md
```

---

---

## Future Improvements

* Incorporate Graph Transformers for improved representation learning.
* Apply focal loss and advanced sampling techniques to further address class imbalance.
* Perform hyperparameter tuning using Optuna.
* Extend the pipeline for real-time fraud detection on streaming transaction data.
* Explore explainability techniques such as GNNExplainer.

---

## References

1. Weber et al., *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks*, 2019.
2. Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks*, ICLR 2017.
3. Hamilton et al., *Inductive Representation Learning on Large Graphs*, NeurIPS 2017.
4. Veličković et al., *Graph Attention Networks*, ICLR 2018.

---
