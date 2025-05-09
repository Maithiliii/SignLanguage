import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data_dict = pickle.load(open('./datatwo.pickle', 'rb'))
# Assuming you have your data stored in data_aux
# Reshape data_aux to have two dimensions
data = np.array(data_dict['data']).reshape(-1, 84)  # Assuming each hand has 21 landmarks with x, y, z coordinates

# Assuming you have labels stored in y_
labels = np.array(data_dict['labels'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=45)

# Initialize and train RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate the classifier
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)


# print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': clf}, f)
f.close()
