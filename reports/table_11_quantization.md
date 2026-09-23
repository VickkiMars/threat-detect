# Table 11: Compression Efficiency and Quantization Trade-offs

| Model version | File size | Parameters | Accuracy | Macro precision | Macro recall | Macro F1 | FPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full-precision Python/Keras hybrid | 172.99 KB | 39,458 | 0.926394 | 0.933233 | 0.920725 | 0.924716 | 0.135172 |
| Dynamic-range quantized hybrid | 56.89 KB | 50,594 | 0.924572 | 0.928883 | 0.919992 | 0.923097 | 0.125169 |
| TensorFlow Lite FlatBuffer hybrid | 56.89 KB | 50,594 | 0.924572 | 0.928883 | 0.919992 | 0.923097 | 0.125169 |
