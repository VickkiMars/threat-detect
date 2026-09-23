# Table 12: Comparative Intrusion Detection Performance Across Baselines

| Dataset / Model | Accuracy | Precision | Recall | F1 score | False-positive rate | Inference Latency (ms) | Test samples |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UNSW-NB15 Decision Tree | 0.876716 | 0.877910 | 0.872708 | 0.874661 | 0.166802 | 0.375 | 8,233 |
| UNSW-NB15 Random Forest | 0.927608 | 0.939127 | 0.920284 | 0.925568 | 0.151933 | 48.359 | 8,233 |
| UNSW-NB15 SVM | 0.881574 | 0.902368 | 0.870224 | 0.876631 | 0.241687 | 1.222 | 8,233 |
| UNSW-NB15 CNN-only | 0.936961 | 0.940908 | 0.932883 | 0.935794 | 0.107326 | 0.656 | 8,233 |
| UNSW-NB15 LSTM-only | 0.936232 | 0.941178 | 0.931649 | 0.934963 | 0.113544 | 1.021 | 8,233 |
| UNSW-NB15 Hybrid CNN–LSTM | 0.926394 | 0.933233 | 0.920725 | 0.924716 | 0.135172 | 1.175 | 8,233 |
| UNSW-NB15 Compressed TensorFlow Lite Hybrid | 0.924572 | 0.928883 | 0.919992 | 0.923097 | 0.125169 | 0.007 | 8,233 |
| CICIDS2017 Decision Tree | 0.646453 | 0.647193 | 0.648669 | 0.647930 | 0.355777 | 0.385 | 2,622 |
| CICIDS2017 Random Forest | 0.797483 | 0.794737 | 0.803802 | 0.799244 | 0.208875 | 46.820 | 2,622 |
| CICIDS2017 SVM | 0.812357 | 0.818745 | 0.803802 | 0.811205 | 0.179036 | 1.450 | 2,622 |
| CICIDS2017 CNN-only | 0.874142 | 0.874525 | 0.874525 | 0.874141 | 0.126243 | 0.680 | 2,622 |
| CICIDS2017 LSTM-only | 0.899314 | 0.897805 | 0.901901 | 0.899311 | 0.103290 | 1.050 | 2,622 |
| CICIDS2017 Hybrid CNN–LSTM | 0.868040 | 0.854426 | 0.888213 | 0.867971 | 0.152257 | 1.185 | 2,622 |
| CICIDS2017 Compressed TensorFlow Lite Hybrid | 0.868040 | 0.854426 | 0.888213 | 0.867971 | 0.152257 | 0.024 | 2,622 |
