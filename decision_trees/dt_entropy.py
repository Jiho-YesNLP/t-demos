import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import code

col = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'y']
df = pd.read_csv('car_evaluation.csv', names=col, sep=',')
df.head()
df.info()

# preprocessing
target = df['y'].unique()
mapping = {}
enc = preprocessing.LabelEncoder()
for attr in col:
    df[attr] = enc.fit_transform(getattr(df, attr))
    #enc.fit_transform(df[attr])
    mapping[attr] = dict(zip(enc.classes_, enc.fit_transform(enc.classes_)))

X = df.drop('y', axis=1)
y = df[['y']]

X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=42)
cls = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=8)
cls.fit(X_tr, y_tr)

accuracy_score(y_ts, cls.predict(X_ts))


# Plotting Decision tree
import graphviz
vis_data = export_graphviz(cls, feature_names=list(X.columns),
                           class_names=list(target),
                           filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(vis_data)
graph.save(filename='vis.dot')
graph.render(filename='vis', format='jpg')

code.interact(local=dict(globals(), **locals()))
