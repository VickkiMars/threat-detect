# Table 11: Compression Efficiency and Quantization Trade-offs

| Model version | File size | Parameters | Accuracy | Macro precision | Macro recall | Macro F1 | FPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full-precision Python/Keras hybrid | 208.66 KB | 50,210 | 0.943034 | 0.947410 | 0.938895 | 0.941960 | 0.101919 |
| Dynamic-range quantized hybrid | 99.59 KB | 50,210 | 0.942548 | 0.946859 | 0.938429 | 0.941469 | 0.102190 |
| TensorFlow Lite FlatBuffer hybrid | 99.59 KB | 50,210 | 0.942548 | 0.946859 | 0.938429 | 0.941469 | 0.102190 |
