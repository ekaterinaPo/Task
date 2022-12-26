from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
#from google.colab import drive
import pandas as pd
import numpy as np

class AgeGroupEncoder(BaseEstimator):
  def __init__(self) -> None:
    super().__init__
  
  def fit(self, df, y = None):
    return self

  def transform(self, df):
    bins=[0,25,65,110]
    labels=["youth", "adults", "seniors"]
    mask = {
        'youth':1,
        'adults':2,
        'seniors':3,
    }
    AgeGroup_ = pd.cut(df["age"], bins=bins, labels=labels)
    df['AgeGroup'] = AgeGroup_
    df['AgeGroup'].replace(mask,inplace=True)
    df = df.drop(columns = ['age'])
    return df

# Number of trees in random forest
n_estimators = [50, 100, 300]
# Maximum number of levels in tree
max_depth = [None, 5, 10, 20]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Criterion
criterion=['gini', 'entropy']
random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'bootstrap': bootstrap,
}

model_pipeline = Pipeline(steps=[("age_group_encoder", AgeGroupEncoder()),
                                 ("nan_replacer", SimpleImputer(missing_values = np.nan, strategy = "median")),
                                 ("robust_scaler", RobustScaler()),
                                 #('random_forest', RandomForestClassifier())
                                 ("random_search_cv", RandomizedSearchCV(RandomForestClassifier(), param_distributions = random_grid, verbose = 2))
                                 ])

if __name__ == "__main__":
    drive.mount('/content/gdrive', force_remount=True)
    #root_path = 'gdrive/My Drive/WeCloudData/'

    df_test = pd.read_csv("/content/gdrive/My Drive/WeCloudData/CreditScore/cs-test.csv")
    df = pd.read_csv("/content/gdrive/My Drive/WeCloudData/CreditScore/cs-training.csv")

    df = df.drop(['Unnamed: 0'], axis=1)

    # split data into training(70%) and testing(30%); set a random seed for the sake of reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('SeriousDlqin2yrs',axis=1), df['SeriousDlqin2yrs'], test_size=0.2, random_state=20210210)

    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    print("ROC AUC score:", roc_auc_score(y_test, y_pred))