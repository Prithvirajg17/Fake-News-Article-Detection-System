# Fake News Detection

# App: `https://fake-news-article-detection-system-p5u2wbztfwkhcjrvy743m6.streamlit.app/#fake-news-detection`

## Project Description:
This project is a web application built using Streamlit that detects fake news articles. The application utilizes a machine learning model trained on a dataset of true and fake news articles to classify whether a given news article is likely to be true or fake.

## Features:
- Load and preprocess true and fake news datasets.
- Train a machine learning model using a TF-IDF vectorizer and Multinomial Naive Bayes classifier.
- Evaluate the model's performance.
- Provide an interface for users to input a news article and get a prediction on its authenticity.
- Custom CSS to hide Streamlit's branding elements for a cleaner interface.

## Requirements:
- Python 3.x
- Streamlit
- pandas
- scikit-learn

## Installation:
1. Clone the repository or download the project files.
2. Install the required packages using pip:
    ```bash
    pip install streamlit pandas scikit-learn
    ```
3. Ensure you have the datasets (`True.csv` and `Fake.csv`) in the project directory.

## Usage:
1. Run the Streamlit app:
    `https://fake-news-article-detection-system-p5u2wbztfwkhcjrvy743m6.streamlit.app/#fake-news-detection`
2. Open the provided local URL in your web browser.
3. Enter a news article in the text area and click the "Submit" button to get a prediction.

## Project Structure:
```bash
.
├── app.py
├── True.csv
├── Fake.csv
├── README.md
```

### Files Description:
- **True.csv**: Dataset of true news articles.
- **Fake.csv**: Dataset of fake news articles.

- ![Screenshot 2024-06-17 210737](https://github.com/Prithvirajg17/Fake-News-Article-Detection-System/assets/148732155/f8bd0a9a-d9a9-4fe3-abeb-3ea84ecba5ba)
- ![Screenshot 2024-06-17 210455](https://github.com/Prithvirajg17/Fake-News-Article-Detection-System/assets/148732155/03d9d7ab-9078-4d93-a034-c56b3f638137)




This README provides a comprehensive guide to understanding, setting up, and running the Fake News Detection project using Streamlit.
