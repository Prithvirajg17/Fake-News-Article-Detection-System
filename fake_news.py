import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load true news dataset
true_df = pd.read_csv('True.csv')
true_df['label'] = 0  # Add a label column for true news

# Load fake news dataset
fake_df = pd.read_csv('Fake.csv')
fake_df['label'] = 1  # Add a label column for fake news

# Concatenate the datasets
df = pd.concat([true_df, fake_df], ignore_index=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF vectorizer and Multinomial Naive Bayes classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
classification_report_result = classification_report(y_test, predictions)

# Display the evaluation results
#print(f'Accuracy: {accuracy}')
#print('Classification Report:\n', classification_report_result)

# Take input from the user
user_input = input("Enter a news article: ")

# Make a prediction
prediction = model.predict([user_input])


# Streamlit UI
st.title('News Classification')

# Input block for user to enter news article
user_input = st.text_area("Enter a news article:")

# Submit button
if st.button('Submit'):
    # Make prediction
    prediction = predict_news(user_input)
    
    # Display the prediction result
    if prediction == 0:
        st.write("The news is likely to be true.")
    else:
        st.write("The news is likely to be fake.")

# Display the prediction result
if prediction[0] == 0:
    print("The news is likely to be true.")
else:
    print("The news is likely to be fake.")
