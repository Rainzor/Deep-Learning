# Deep Learning

## 1. MLP

<img src="assets/MLP.png" alt="MLP" style="zoom:67%;" />

## 2. CNN

**Tiny ImageNet**

![cnn](assets/cnn.png)

## 3. RNN

Word Embedding + RNN+ FC

| Model       | Pool | Text Len | Bidirectional | Accuracy |
| ----------- | ---- | -------- | ------------- | -------- |
| lstm        | last | 512      | 1             | 0.661    |
| lstm        | max  | 512      | 1             | 0.660    |
| lstm        | attn | 512      | 1             | 0.659    |
| lstm        | mean | 512      | 1             | 0.659    |
| lstm        | mean | 256      | 1             | 0.653    |
| lstm        | last | 256      | 1             | 0.652    |
| lstm        | attn | 256      | 1             | 0.652    |
| lstm        | max  | 256      | 1             | 0.649    |
| gru         | last | 256      | 1             | 0.645    |
| lstm        | last | 256      | 0             | 0.637    |
| gru         | last | 256      | 0             | 0.625    |
| transformer | cls  | 512      | 0             | 0.610    |
| transformer | cls  | 256      | 0             | 0.596    |
| rnn         | last | 256      | 0             | 0.569    |
| rnn         | last | 256      | 1             | 0.544    |
