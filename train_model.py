import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data=pd.read_csv("dataset.csv")
X=data.drop("label",axis=1)
y=data["label"]

#split data
X_train, X_test, y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model=RandomForestClassifier(  #created the model
    n_estimators=100,
    random_state=42
)

#training the model
model.fit(X_train,y_train)

#predicitions:
predictions=model.predict(X_test)
#accuracy
accuracy=accuracy_score(y_test,predictions)

print("Accuracy: ", accuracy*100)
print(confusion_matrix(
    y_test,
    predictions
))

print(classification_report(
    y_test,
    predictions
))

joblib.dump(
    model,
    "gesture_model.pkl"
)
print("Model saved")