# NLP_a4_Do-you-AGREE
# BERT and Sentence-BERT Model Implementation for NLI

## Task 1: BERT Model Training from Scratch

### Objective:
In this task, we implemented a Bidirectional Encoder Representations from Transformers (BERT) model from scratch and trained it on the BookCorpus dataset. The primary objective was to train the model using the Masked Language Modeling (MLM) approach. In this setup, 15% of the tokens in a sequence are randomly masked, and the model learns to predict these masked tokens based on their surrounding context. This task serves as the foundation for further downstream tasks such as text classification or semantic similarity.

### Project Overview:
#### 1. Dataset:
- **Source**: BookCorpus dataset
- **Subset**: 10,000 samples were selected from the BookCorpus dataset for training.
- **Training Size**: 10,000 samples to maintain computational feasibility.

#### 2. Data Preprocessing:
The following preprocessing steps were applied to the dataset:
- **Token Embedding**: Special tokens were added to the text to indicate the start ([CLS]) and separation between sentences ([SEP]). Additionally, random tokens were masked during training.
- **Segment Embedding**: Sentence A and Sentence B were assigned segment IDs (0 for Sentence A and 1 for Sentence B).
- **Masking**: 15% of the tokens were randomly masked during training:
  - 80% of the time, tokens were replaced with the [MASK] token.
  - 10% of the time, a random token was inserted.
  - 10% of the time, the token remained unchanged.
- **Padding**: Sentences were padded to a maximum length of 1000 tokens to ensure uniformity for batch processing.

#### 3. Model Architecture:
The BERT model was implemented with the following specifications:
- **Number of Encoder Layers**: 12
- **Number of Attention Heads**: 12
- **Hidden Embedding Size**: 768
- **Feed-Forward Layer Size**: 3072
- **Vocabulary Size**: As per BookCorpus dataset
- **Maximum Sequence Length**: 1000 tokens
- **Device**: Model was trained on a GPU for enhanced computational efficiency.

#### 4. Training Setup:
The training process utilized the Masked Language Modeling (MLM) objective, where the model learns to predict the masked tokens based on surrounding context. The model was trained for 1,000 epochs with a batch size of 6. Training loss was monitored at intervals of 100 epochs.
- **Final Loss after 1000 epochs**: 3.725234

#### 5. Model Saving:
After training, the model's weights were saved to a file (`bert_model.pth`) for future use in fine-tuning or downstream tasks.

---

## Task 2: Sentence Embedding with Sentence-BERT (SBERT)

### Overview:
In this task, we adapted the pre-trained BERT model from Task 1 and implemented Sentence-BERT (SBERT) using a Siamese network architecture. The goal was to generate semantically meaningful sentence embeddings that could be used for tasks like semantic similarity and Natural Language Inference (NLI). The Sentence-BERT model was fine-tuned to classify sentence pairs into categories like Entailment, Neutral, or Contradiction.

### Objectives:
- Implement SBERT with a Siamese network structure.
- Fine-tune the pre-trained BERT model from Task 1.
- Use Softmax Loss to classify sentence pairs.
- Evaluate sentence embeddings using cosine similarity.

### Datasets:
The following datasets were used for training and evaluation:
- **SNLI** (Stanford Natural Language Inference)
- **MNLI** (Multi-Genre Natural Language Inference)

Both datasets consist of premise-hypothesis pairs with labels indicating whether the relationship between the sentences is entailment, neutral, or contradiction.

### Data Preprocessing:
1. **Loading Datasets**: We used the Hugging Face datasets library to load SNLI and MNLI.
2. **Merging and Shuffling**: The training, validation, and test sets were merged, shuffled, and a subset was selected for faster experimentation.
3. **Tokenization**: Premise and hypothesis sentences were tokenized using the BERT tokenizer.
4. **Dataloader Creation**: Dataloader objects were created for efficient batching and shuffling of the datasets.

### Model Architecture:
1. **Siamese Network**: A Siamese network structure was used, where two identical BERT models shared weights. The premise and hypothesis sentences were processed in parallel by the two BERT models. The embeddings of the sentences were compared using the mean-pooling technique.
2. **Loss Function**: Softmax loss was used to classify the relationships between the sentence pairs into three categories: Entailment, Neutral, and Contradiction.
3. **Training Details**:
  - The model was trained using the Adam optimizer and Softmax loss.
  - Accuracy and cosine similarity of sentence embeddings were monitored during training.

### Training Results:
Sample results from training after multiple epochs:
- **Epoch 5**: Loss = 1.503545
- **Average Cosine Similarity**: 0.9933

---

## Task 3: Evaluation and Analysis

### Performance Metrics:
We evaluated the Sentence-BERT model using the SNLI or MNLI datasets, which provide sentence pairs labeled with Entailment, Neutral, and Contradiction.

#### Evaluation Summary:
- | Model Type   | SNLI or MNLI Accuracy |
  |--------------|---------------|
  | Our Model    | 35.3%         |

- **Cosine Similarity**: 0.9933

### Challenges Encountered:
- **Tensor Dimension Mismatches**: One of the key challenges we faced was adapting the BERT model to the Siamese network for Sentence-BERT (SBERT). This required ensuring that the tensors from both the premise and hypothesis sentence inputs were aligned in terms of shape. Issues arose when tensor dimensions (such as those from attention_mask or token_type_ids) did not match, leading to errors during training. Ensuring that both sentence embeddings were processed in parallel with identical dimensions was critical for successful model execution.
- **Mismatched Variables**: During the adaptation of BERT into the Siamese network architecture, another challenge was dealing with mismatched variables between the two inputs (premise and hypothesis). Variables such as input_ids, attention_mask, and token_type_ids had to be carefully aligned across the two input sentences. Any inconsistencies in how these variables were defined or processed for one sentence could lead to errors. Properly managing these variables and ensuring they had compatible values and shapes was key to preventing training issues.
- **Overfitting**: Another challenge was overfitting, which became apparent after evaluating the model on the test data. The model performed well during training but showed signs of reduced generalization on unseen data. This overfitting issue was likely due to the small size of the training dataset (only 1000 samples). Expanding the dataset or applying regularization techniques such as dropout or data augmentation could help mitigate this issue and improve the model's ability to generalize to new, unseen data.

---

## Task 4: Text Similarity Web Application

### Overview:
A Flask-based web application was developed to showcase the functionality of the Sentence-BERT model. The web app allows users to input a premise and a hypothesis, and the model predicts whether the relationship between them is Entailment, Neutral, or Contradiction.

### Features:
- Two input fields for entering the Premise and Hypothesis sentences.
- The application uses the Sentence-BERT model to predict the relationship between the sentences:
  - **Entailment**
  - **Neutral**
  - **Contradiction**

#### Example:
- **Premise**: A man is playing a guitar on stage.
- **Hypothesis**: The man is performing music.
- **Prediction**: Entailment

  ![web1](https://github.com/user-attachments/assets/7bd7cabd-4b2d-4f04-8cdb-0520b2559409)
  ![web2](https://github.com/user-attachments/assets/bca0a696-18f0-49de-b9ed-af760c77aae4)


