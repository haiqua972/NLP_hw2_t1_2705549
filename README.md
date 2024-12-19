# **Task 1: Ideology Identification in Parliamentary Speeches**
# 1. **Overview**
This task focuses on binary classification to determine the political ideology of parliamentary speakers’ parties. The task involves predicting whether a speaker’s party leans left (0) or right (1) based on parliamentary speech data in multiple languages.

This task  including the following:
1.	Fine-tuning a **multilingual masked language model** for binary classification using:

        o	English text (text_en)

        o	Original language text (text)

2.	Experimenting with a **multilingual causal language model** (zero-shot inference) to evaluate cross-lingual capability.
   
3.	Conducting a comparative analysis of the models' performances across languages and methodologies.

# **2. Dataset**
The dataset consists of parliamentary speeches from the ParlaMint corpus. 

Each record includes:
*	text_en: Speech translated into English.
*	text: Original speech in the respective language.
*	label: Binary classification label for ideology:
  
        o	0: Left
 	
        o	1: Right

## 2.1. Dataset Details
 The dataset is split into 90% training and 10% testing in a stratified manner to maintain label proportions.

# **3. Methodology**
## 3.1. Fine-tuned Multilingual Masked Language Model

Two models were fine-tuned using the Hugging Face library:

1.	**English text model:** Fine-tuned on text_en.
2.	**Original language text model:** Fine-tuned on text.
   
### 3.1.1 Metrics Evaluated:
*	Loss: Indicates the model's error.
*	Accuracy: Measures the proportion of correct predictions.
*	F1 Score: Balances precision and recall for binary classification.
*	Precision: Measures the proportion of correctly predicted positive cases.
*	Recall: Measures the proportion of actual positive cases correctly predicted.
## 3.2 Zero-Shot Inference Using a Multilingual Causal Language Model
A causal language model was applied in a zero-shot manner to predict ideology. Inference was performed on:

*	English text: text_en
*	Original language text: text
### 3.2.1 Metrics Evaluated:
*	Accuracy: Measures the proportion of correct predictions.
*	F1 Score: Balances precision and recall for binary classification.
*	Precision: Measures the proportion of correctly predicted positive cases.
*	Recall: Measures the proportion of actual positive cases correctly predicted.

# **4. Results**
## 4.1. Fine-tuned Models
![image](https://github.com/user-attachments/assets/fb557097-813e-4688-a247-4c0b659c1bf7)

## 4.2. Zero-Shot Model
* English Text: Predicted ideology from sample parliamentary speeches.
*	Original Language Text: Predicted ideology from sample speeches in the original language.
  
![image](https://github.com/user-attachments/assets/8b52e329-b5f2-4ce8-a48d-dc34bf73df64)


# **5. Comparative Analysis**
## 5.1. Observations
 * **Performance Metrics:** The fine-tuned models achieved an accuracy and F1 score of approximately 58.17% and 73.56%, respectively, across both English and original language datasets. They exhibited perfect recall but moderate precision, which might indicate a bias towards the positive class.

* **Zero-Shot Model Robustness:** While the zero-shot models demonstrated robustness in cross-lingual capabilities, they fell short in precision and recall compared to fine-tuned models. This underscores the advantages of model training specific to the task and dataset.

* **Handling Class Imbalance:** Stratification was used during dataset splitting to manage class imbalance, ensuring equitable class representation in both training and testing sets. However, the disparity in recall and precision suggests the need for further strategies to refine this balance.
## 5.2. Discussion
*	Balanced Data: The stratified dataset ensured label proportions were maintained during the split.
*	Performance: Fine-tuned models provided reliable results, while the zero-shot model demonstrated flexibility in handling multilingual data.
*	Improvements: Future enhancements could include more training epochs, data augmentation, or experimenting with larger pre-trained models for improved accuracy.

# **6. How to Run the Code**
## 6.1. Prerequisites
Install the required libraries:

    !pip install transformers datasets torch sklearn
    !pip install datasets
## 6.2. Steps
1.	Upload Dataset: Place your dataset in Google Drive and set the file path.
2.	Run the Notebook: Execute the script step-by-step in a Python/Colab notebook.
## 6.3 File Structure
*	dataset/: Contains the parliamentary speeches dataset (not included in this repo for privacy).
*	scripts/: Python scripts for fine-tuning, evaluation, and inference.
*	results/: Generated metrics and evaluation logs.

# **7. Future Work**
## 7.1.	Zero-Shot Quantitative Evaluation:

o	Apply the zero-shot model to labeled test data to compute metrics like accuracy and F1 score.
## 7.2.	Model Enhancements:

o	Experiment with different model architectures, hyperparameters, and larger pre-trained models like XLM-Roberta or domain-specific models.
## 7.3.	Class Imbalance:

o	Address imbalance by oversampling, class weighting, or threshold tuning.

# **8. Contact**

Haiqua Meerub: haiquameerub972@gmail.com


