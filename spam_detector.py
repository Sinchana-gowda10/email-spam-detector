import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
import string   
import pickle

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix


# Download stopwords
nltk.download('stopwords')

# Load dataset from URL
df = pd.read_csv(
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
    sep="\t",  # Columns are separated by tab, not comma
    header=None,  # No column names in the file
    names=['label', 'message']  # Give our own column names
)

# Show first 5 messages
print(df.head())

print("Total messages:", df.shape[0])  # total rows
print("Columns:", df.shape[1])         # total columns

# Count how many spam and ham
print("\nHow many spam and ham?")
print(df['label'].value_counts())

#Checking is there any missing data
print("\nAre there any missing values?")
print(df.isnull().sum())

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Remove stopwords
    stop_words = stopwords.words('english')
    words = text.split()
    cleaned = [word for word in words if word not in stop_words]
    
    return ' '.join(cleaned)

# Apply cleaning to the 'message' column
df['cleaned_message'] = df['message'].apply(clean_text)

encoder = LabelEncoder()
df['label_num'] = encoder.fit_transform(df['label'])

# Initialize vectorizer
tfidf = TfidfVectorizer(max_features=3000)  # limit to 3000 words
X = tfidf.fit_transform(df['cleaned_message']).toarray()  # independent features
y = df['label_num'].values  # target variable (0 or 1)

#Train-Test dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Predict using test data
y_pred = nb_model.predict(X_test)

# Evaluate performance
print("🔍 Accuracy:", accuracy_score(y_test, y_pred))
print("🎯 Precision:", precision_score(y_test, y_pred))
print("📢 Recall:", recall_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Uncomment to test SVM (optional)
#svm_model = SVC()
#svm_model.fit(X_train, y_train)
#y_pred_svm = svm_model.predict(X_test)
#print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

# Save the trained model
with open("spam_classifier_model.pkl", "wb") as model_file:
    pickle.dump(nb_model, model_file)

# Save the vectorizer
with open("tfidf_vectorizer.pkl", "wb") as vec_file:
    pickle.dump(tfidf, vec_file)

print("\n✅ Model and vectorizer saved successfully!")

