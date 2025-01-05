# Deep Learning

## 1. [MLP](./Lab1_MLP)

<center>
    <img style = "
        border-radius: 0.3125em;
        box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
        src = "assets/MLP.png" 
        width = "50%">
    <br>
    <div style = "
        color: orange;
        border-bottom: 1px solid #d9d9d9;
        display: inline-block;
        color: #999;
        padding: 2px;">
        MLP
    </div>
    <p> </p>
</center>

## 2. [CNN](Lab2_CNN)

**Tiny ImageNet**

<center>
    <img style = "
        border-radius: 0.3125em;
        box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
        src = "assets/CNN.png" 
        width = "80%">
    <br>
    <div style = "
        color: orange;
        border-bottom: 1px solid #d9d9d9;
        display: inline-block;
        color: #999;
        padding: 2px;">
        CNN
    </div>
    <p> </p>
</center>

## 3. [RNN](Lab3_RNN)

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

<center>
    <img style = "
        border-radius: 0.3125em;
        box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
        src = "assets/RNN.png" 
        width = "80%">
    <br>
    <div style = "
        color: orange;
        border-bottom: 1px solid #d9d9d9;
        display: inline-block;
        color: #999;
        padding: 2px;">
        RNN
    </div>
    <p> </p>
</center>

## 4. [GNN](./Lab4_GNN)

##### Node Classification

<center>
    <img style = "
        border-radius: 0.3125em;
        box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
        src = "assets/GNN-NodeCLS.png" 
        width = "80%">
    <br>
    <div style = "
        color: orange;
        border-bottom: 1px solid #d9d9d9;
        display: inline-block;
        color: #999;
        padding: 2px;">
        GNN: Node Classification
    </div>
    <p> </p>
</center>

##### Link Prediction

<center>
    <img style = "
        border-radius: 0.3125em;
        box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
        src = "assets/GNN-LinkPred.png" 
        width = "80%">
    <br>
    <div style = "
        color: orange;
        border-bottom: 1px solid #d9d9d9;
        display: inline-block;
        color: #999;
        padding: 2px;">
        GNN: Link Prediction
    </div>
    <p> </p>
</center>