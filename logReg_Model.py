import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print('x_train:', x_train.shape)
print('x_test:', x_test.shape)
print('y_train:', y_train.shape)
print('y_test:', y_test.shape)

logReg = LogisticRegression(class_weight='balanced', random_state=42)
logReg.fit(x_train, y_train)

with open('logReg_model.pkl', 'wb') as f:
    pickle.dump(logReg, f)



