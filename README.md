# **CS6120 NLP Final Project:<br>Exploring Sentiment Classification on the <br>"Amazon Fine Food Reviews" Dataset**

## **Abstract**

This project explores the impact of different preprocessing strategies and classification models on sentiment analysis of the “Amazon Fine Food Reviews” dataset. By comparing stemming vs lemmatization, and TF vs TF-IDF vectorization, we evaluate the performance of three supervised learning models (Naive Bayes, Logistic Regression, and Multi-layer Perceptron), alongside K-Means clustering. The results are visualized using confusion matrices and t-SNE plots to highlight differences in model behavior and prediction accuracy.

## ** Dataset and Outside Tools**

The dataset used for this project is the Amazon Fine Food Reviews dataset, publicly available on Kaggle:
> https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

Or, it can be downloaded from Google Drive (Reviews.csv, 287MB):
> https://drive.google.com/file/d/19Cb4Uc5zak7udsetWS_hKwDFxO9M41BF/view?usp=sharing

For computational efficiency and to ensure a balanced dataset for classification, a subset of 20,000 reviews was sampled. This subset was carefully constructed to contain 10,000 positive reviews (4 or 5-star ratings) and 10,000 negative reviews (1 or 2-star ratings). Neutral reviews (3-star ratings) were excluded from the analysis.

The outside tools and packages used in this project include:
- Text preprocessing: nltk (Natural Language Toolkit), re (for regular expressions)
- Machine learning: sklearn (Scikit-learn) with various tools
- Data manipulation: numpy, pandas
- Plotting and visualization: matplotlib, seaborn

## **Source Code and Program Execution**

- To run this program in Colab (Google Colaboratory), please follow these steps:
  1. Download the dataset and put it in the root directory of your Google Drive.
  2. Click on the link (you may be required to authorize the access to your Google Drive):
https://githubtocolab.com/HPW17/public/blob/main/group18_project.ipynb

- To run this program on your local machine, please follow these steps:
  1. Download the source code (group18_project.py) to your Python environment with necessary libraries and dependencies. 
  2. Download the dataset and put it in the same directory as the source code.
  3. Run with command: 
  > python group18_project.py

## **Data Preprocessing and Feature Extraction**

The preprocessing pipeline involved the following steps:
- HTML tag removal
- Lowercasing
- Stopword and punctuation removal
- Tokenization
- Normalization: stemming (Porter Stemmer) or lemmatization (WordNet Lemmatizer)

After preprocessing, the text data needed to be converted into numerical feature vectors. Two vectorization schemes were experimented with: 
- Term Frequency (TF)
- Term Frequency - Inverse Document Frequency (TF-IDF)

## **Model Architectures**

We evaluated the following classifiers:
- Naive Bayes (MultinomialNB): Fast and effective for text data.
- Logistic Regression (LR): Linear classifier with regularization.
- Multi-layer Perceptron (MLP): A simple MLP with 1 hidden layer of 100 neurons was used.

For unsupervised learning, we used:
- K-Means Clustering with k=2 to discover sentiment-based clusters.

## **Training and Evaluation**

- Data Splitting: The preprocessed dataset (20k reviews) was split into an 80% training set (16k) and a 20% testing set (4k).
- Classification Training: Each classification model (Naive Bayes, Logistic Regression, MLP) was trained independently on both TF and TF-IDF feature representations for both stemming and lemmatization preprocessing.
- Evaluation: The metrics included accuracy, precision, recall, F1-score, ROC curve, AUC, and confusion matrix. Results were also visualized using t-SNE.

## **Discussion**

- **Stemming and lemmatization:** The ablation study between two normalization schemes revealed that for this specific sentiment classification task, the choice had a minimal impact on final model performance. Lemmatization produced a slightly larger, more linguistically accurate vocabulary, but this did not translate to a significant performance gain over the more aggressive stemming. This suggests that for many practical text classification problems, the simpler stemming approach can be sufficient, offering faster preprocessing times.

- **Feature representation:** The comparison clearly favored TF-IDF over TF for supervised classification. TF-IDF's ability to weight words by their importance and rarity across the corpus proved beneficial in enhancing model performance, leading to consistently higher accuracy. 

- **Classification model performance:** Multilayer Perceptron (MLP) generally achieved the highest scores across all metrics, indicating its capacity to learn more complex, non-linear patterns in the data. However, Logistic Regression performed remarkably close to MLP, not to mention its significantly faster training time. Naive Bayes, while the fastest, lagged slightly behind. This highlights a trade-off: more complex models (like MLPs) can offer marginal performance gains but come with increased computational cost and training time. For applications where speed is critical, Logistic Regression presents an excellent balance of performance and efficiency.

- **Clustering analysis:** The confusion matrices and t-SNE plots visually confirmed that K-Means clustering failed to effectively separate the reviews into meaningful sentiment-based clusters in our case. This emphasizes the difference that supervised models learn directly from labeled examples, whereas unsupervised methods rely solely on the intrinsic structure of the data. Sentiment, being a nuanced human construct, often requires explicit labels for effective automated detection, especially with simpler feature representations.

## **Conclusion**

This project successfully implemented and evaluated various NLP techniques for sentiment analysis and text clustering on the Amazon Fine Food Reviews dataset.
- The impact of lemmatization over stemming was minimal in our case.
- TF-IDF consistently led to better performance.
- Logistic Regression and MLP outperform Naive Bayes.
- Logistic Regression offers a strong trade-off between performance and efficiency.
- K-Means is limited for sentiment clustering without supervision.

