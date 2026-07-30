# Link Prediction Evaluation

## Classification Metrics

| Split      | Method              |   Samples |   Positive |   Accuracy |   Precision |   Recall |       F1 |   ROC-AUC |   Average Precision |
|:-----------|:--------------------|----------:|-----------:|-----------:|------------:|---------:|---------:|----------:|--------------------:|
| Validation | Logistic Regression |       694 |        347 |   0.564841 |    0.535658 | 0.974063 | 0.691207 |  0.626473 |            0.611835 |
| Test       | Logistic Regression |       450 |        225 |   0.531111 |    0.516279 | 0.986667 | 0.677863 |  0.60237  |            0.560076 |
| Validation | Cosine Similarity   |       694 |        347 |   0.507205 |    0.50366  | 0.991354 | 0.667961 |  0.533897 |            0.534272 |
| Test       | Cosine Similarity   |       450 |        225 |   0.5      |    0.5      | 0.995556 | 0.665676 |  0.535131 |            0.506609 |

## Precision@K (Test)

| k | Logistic Regression | Cosine Baseline |
|---|---|---|
| 10 | 0.300 | 0.300 |
| 20 | 0.500 | 0.300 |
| 50 | 0.560 | 0.440 |

Validation-selected threshold (model): 0.07
Validation-selected threshold (baseline): 0.16