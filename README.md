# **Task 1: Ideology Identification in Parliamentary Speeches**
# 1. **Overview**
This task is to perform binary classification in order to identify the political orientation of the parties of parliamentary speakers. The task is to classify the party of a speaker as left leaning (0) or right leaning (1) using the parliamentary speech data in different languages.

This task  including the following:
1.	Fine-tuning a **multilingual masked language model** for binary classification using:

        o	English text (text_en)

        o	Original language text (text)

2.	Experimenting with a **multilingual causal language model** (zero-shot inference) to evaluate cross-lingual capability.
   
3.	Doing a analysis in comparative way for the performances of the models in different languages and in different methods.

# **2. Dataset**
The dataset contains parliamentary speeches from the ParlaMint Dataset. 

Each record includes:
*	text_en: Speech translated into English.
*	text: Original speech in the respective language.
*	label: Binary classification label for the ideology of the speaker:
  
        o	0: Left
 	
        o	1: Right

## 2.1. Dataset Details
 To maintain label proportions, the dataset is split 90:10 stratified for training and testing purposes respectively.

# **3. Methodology**
## 3.1. Fine-tuned Multilingual Masked Language Model

I fine-tuned two models by using the Hugging Face library:

1.	**English text model:** Fine-tuned on text_en.
2.	**Original language text model:** Fine-tuned on text.
   
### 3.1.1 Metrics Evaluated:
*	Loss: Indicates the error of the model.
*	Accuracy: Calculates the accuracy of predictions in the present model.
*	F1 Score: Optimises both precision and recall for a binary classification problem.
*	Precision: The ability to measure the proportion of positive cases that has been correctly predicted.
*	Recall: Measures the level of actual positive cases that has been recognized appropriately.
## 3.2 Zero-Shot Inference Using a Multilingual Causal Language Model
A causal language model was used zero-shot to predict ideology. Inference was performed on:

*	English text: text_en
*	Original language text: text
### 3.2.1 Metrics Evaluated:
*	Accuracy: Calculates the accuracy of predictions in the present model.
*	F1 Score: Optimises both precision and recall for a binary classification problem.
*	Precision: The ability to measure the proportion of positive cases that has been correctly predicted.
*	Recall: Measures the level of actual positive cases that has been recognized appropriately.

# **4. Results**
## 4.1. Fine-tuned Models
![image](https://github.com/user-attachments/assets/fb557097-813e-4688-a247-4c0b659c1bf7)

## 4.2. Zero-Shot Model
* English Text: Predicted ideology from sample parliamentary speeches.
*	Original Language Text: Predicted ideology from sample speeches in the original language.
  
![image](https://github.com/user-attachments/assets/8b52e329-b5f2-4ce8-a48d-dc34bf73df64)


# **5. Comparative Analysis**
## 5.1. Observations
 * **Performance Metrics:** The fine-tuned models provided an accuracy and F1 score of around 58.17% and 73.56% respectively on both English and original language sets. They well recalled but had moderate accuracy which could be an indication of high bias towards the positive class.

* **Zero-Shot Model Robustness:** While the zero-shot models demonstrated robustness in cross-lingual capabilities, they fell short in precision and recall compared to fine-tuned models. This underscores the advantages of model training specific to the task and dataset.

* **Handling Class Imbalance:** Although the zero-shot models were highly reliable in terms of the cross-lingual evaluation, they were outperformed in terms of precision and recall by the fine-tuned models. This goes to emphasize the benefits of training a model from scratch suitable to the task and data set.
## 5.2. Discussion
*	**Balanced Data:** For this reason, the stratified dataset was used to make sure that the proportions of the labels remained the same after the split.
*	**Performance:** The models fine-tuned were accurate, while the zero-shot model was versatile in dealing with multiple languages.
*	**Improvements:** The ideas for possible improvements may be incorporated in future work such as additional training iterations and data collection, data diversification as well as utilizing other large pre-trained DL models.

# **6. How to Run the Code**
## 6.1. Prerequisites
Install the required libraries:

    !pip install transformers datasets torch sklearn
    !pip install datasets
## 6.2. Steps
1.	**Upload Dataset:** Upload your dataset in Google Drive and set the file path.
2.	**Run the Notebook:** The code must be run through in steps using a Python/Colab notebook.
## 6.3 File Structure
*	dataset/: Contains the parliamentary speeches dataset.
*	scripts/: Python scripts for training, validation and test.
*	results/: Generated metrics and evaluation logs.

# **7. Future Work**
## 7.1.	Zero-Shot Quantitative Evaluation:

o	I suggest to use the zero shot model for test data that is labeled to calculate accuracy and F1 score.
## 7.2.	Model Enhancements:

o	Try out the variety of architectures, features of models, hyperparameters, and start using the models with higher sizes like XLM-Roberta or domain-specific one.
## 7.3.	Class Imbalance:

o	Address imbalance by oversampling, class weighting, or threshold tuning.

# **8. Contact**

Haiqua Meerub: haiquameerub972@gmail.com


