import numpy as np
import pandas as pd
from tldextract import extract
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler

# Load dataset
url = 'phishing_site_urls.csv'
data = pd.read_csv(url, header='infer')

# Explore the dataset
def explore(dataframe):
    print("Total Records: ", dataframe.shape[0])
    x = dataframe.columns[dataframe.isnull().any()].tolist()
    if not x:
        print("No Missing/Null Records")
    else:
        print("Found Missing Records")

explore(data)

# Custom function to extract domains
def extract_domain(url):
    extracted = extract(url)
    return extracted.domain + '.' + extracted.suffix

# Extract Domain
data['Domain'] = data['URL'].apply(extract_domain)
data.drop('URL', axis=1, inplace=True)
data = data[['Domain', 'Label']]

# Encode the Label
lab_enc = LabelEncoder()
data['Label'] = lab_enc.fit_transform(data['Label'])

# Split data into train and test
x_train, x_test, y_train, y_test = train_test_split(data['Domain'], data['Label'], test_size=0.1, random_state=0)

# Over-sampling
ros = RandomOverSampler(random_state=0)
X_train_2d = x_train.values.reshape(-1, 1)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_2d, y_train)
x_train_resampled = X_train_resampled.ravel()  # Flatten back to 1D array

# Construct Pipeline
pipe = Pipeline([
    ('vect', CountVectorizer(min_df=1, max_df=1.0, binary=False, ngram_range=(1, 3))),
    ('tfidf', TfidfTransformer()),
    ('model', MultinomialNB())
])

# Train the model
mnb_model = pipe.fit(x_train_resampled, y_train_resampled)

# Making Prediction on Test Data
mnb_pred = mnb_model.predict(x_test)
print("Multinomial Naive Bayes Model Accuracy:", '{:.2%}'.format(accuracy_score(y_test, mnb_pred)))

# Function to preprocess single URL and make prediction
def extract_features(domain):
    return domain  # This function can be expanded to extract more features

def predict_phishing(domain):
    prediction = mnb_model.predict([domain])
    return int(prediction[0])  # Convert to integer for JSON response

# Example usage
url_input = input("Enter the URL: ")
domain_input = extract_domain(url_input)
prediction = predict_phishing(domain_input)
print("Prediction:", prediction)
