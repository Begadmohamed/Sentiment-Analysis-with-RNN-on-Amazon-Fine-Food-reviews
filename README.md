## Sentiment Analysis with RNN on Amazon Fine Food reviews

This repo is all about training a neural net to catch the vibe from Amazon food reviews. We're taking raw text from over 500,000 reviews and building a model that can predict the star rating (from 1 to 5) based on what the user wrote.

The whole thing is built in a Kaggle notebook to take advantage of their free GPUs, 'cause training these models on a regular laptop is a big yikes.

---
## The Dataset 📦

- **Name:** [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- **Content:** Over 568,000 reviews with user-submitted text and a star rating (1-5).
- **The Catch:** The dataset is super **imbalanced**. There are way more 5-star reviews than anything else, which is a big challenge we had to tackle.

---
## The Playbook 🚀

- **Data Cleaning 🧹**
  First, we cleaned up the raw text. Made everything lowercase, stripped out all the punctuation, and removed common "stopwords" (like 'the', 'a', 'is') so the model could focus on the words that actually matter.

- **Preprocessing & Tokenizing**
  Turned all the cleaned text into numbers (`tokenization`) so the neural net could understand it. Then we made sure every review was the same length by padding the shorter ones and truncating the longer ones.

- **Model Building 🧠**
  We built and compared two different types of Recurrent Neural Networks (RNNs):
  - **SimpleRNN:** The basic, entry-level model to get a baseline.
  - **LSTM (Bidirectional):** The upgraded model. LSTMs have better "memory" for long reviews, and making it bidirectional means it reads the review forwards *and* backwards for even better context.

- **Training & Regularization**
  We trained the models to predict the star rating. To deal with the busted dataset, we used `class_weight` to force the model to pay more attention to the rare 1 and 2-star reviews. We also used `Dropout` and `L2 regularization` to stop the model from just memorizing the training data (a.k.a. overfitting).

- **Evaluation 📊**
  Checked how the models did using accuracy scores and confusion matrices to see exactly where they were getting confused.

- **Live Prediction**
  Built a final function so you can drop in any review as a string and get a live sentiment prediction on the spot. Straight up.

---
## Tech Stack 🛠️

- TensorFlow / Keras
- Pandas
- NLTK
- Scikit-learn
- Seaborn / Matplotlib

---
## How to Run It 🤙

- Clone the repo: `git clone ...`
- Get the data: Download the dataset from the Kaggle link above and put it in your project directory.
- Install the goods: Make sure you have all the libraries from the Tech Stack installed.
- Run the notebook: The notebook is set up to run from top to bottom. It's best to run it on a platform with a GPU (like Kaggle or Google Colab) to speed up the training.

---
## Key Takeaways 💡

- **Class imbalance is the main boss fight.** Without using `class_weight` or some other strategy, any model will just get lazy and predict "5 stars" for everything.
- **LSTMs > SimpleRNNs.** For text, the better memory of an LSTM makes a huge difference. The bidirectional LSTM was the clear winner.
- **Regularization is non-negotiable.** Without `Dropout`, the model overfits almost immediately.
