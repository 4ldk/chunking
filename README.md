# Chunking Shared Task in Conll2000

Dataset: https://www.clips.uantwerpen.be/conll2000/chunking/  
connlleval.py: https://github.com/sighsmile/conlleval/blob/master/conlleval.py

1. Download and unzip dataset.
2. Set the dataset path in `preprocessing.py`.
3. Set the hyperparameter in `train.py`.
4. Change other setting you want to change.
5. Do `python train.py` if you want to train with BERT, do `python bert_train.py`. 
7. Set model path and do `python pred.py` or `python bert_train.py`.
8. Do `python connlleval.py`. 
9. You can monitor or confirm　loss curve by tensorboard.

trained models: https://drive.google.com/drive/folders/11uyVskbp9oLQVsj7A5lfIsKhhPJOoFKr?usp=sharing

    current score  
    LSTM-crf: 92.6,  
    LSTM-w2v-crf: 92.49,  
    BERT-crf: 96.38: current output.txt,
