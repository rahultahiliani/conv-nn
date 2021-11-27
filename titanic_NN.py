
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier

df_train = pd.read_csv('train.csv')


def title(name):
	if '.' in name:
		return name.split(',')[1].split('.')[0].strip()
	else:
		return 'No title Exists'

def compact_titles(x):
	title_new = x['Title']
	if title_new in ['Capt', 'Col', 'Major']:
		return 'Officer'
	elif title_new in ['Mr','Master','Don','the Countess','Sir', 'Ms','Jonkheer', 'Dona', 'Lady']:
		return 'Royale People'
	elif title_new in ['Mrs','Master','Miss''the Countess','Mme','Lady']:
		return 'Mrs'
	elif title_new in ['Miss','Mrs','Mlle', 'Ms']:
		return 'Miss'
	else:
		return title_new

df_train['Title'] = df_train['Name'].map(lambda x: title(x))
df_train['Title'] = df_train.apply(compact_titles, axis=1)

df_train['Age'].fillna(df_train['Age'].median(), inplace =True)
df_train['Fare'].fillna(df_train['Fare'].median(), inplace =True)
df_train['Embarked'].fillna("S" , inplace =True)
df_train.drop('Name', axis=1, inplace=True)
df_train.drop('Ticket', axis=1, inplace=True)
df_train.drop('Cabin', axis=1, inplace=True)
df_train.Sex.replace(('male', 'female'), (0, 1), inplace=True)
df_train.Embarked.replace(('S', 'Q', 'C'), (0, 1, 2), inplace=True)
df_train.Title.replace(('Mrs', 'Miss', 'Royale People', 'Dr', 'Rev', 'Officer','Royale People'), (0, 1, 2, 3, 4, 5, 6), inplace=True)
y = df_train['Survived']
x = df_train.drop(['Survived', 'PassengerId'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

rf = RandomForestClassifier()
rf.fit(x_train,y_train)
pickle.dump(rf,open('ML_titanic_Model.sav','wb'))


def predict_Model(pclass,sex,age,sibsp,parch,fare,embarked,title):
	import pickle
	x = [[pclass,sex,age,sibsp,parch,fare,embarked,title]]
	rf = pickle.load(open('ML_titanic_Model.sav','rb'))
	prediction = rf.predict(x)
	if prediction == 1:
		print('Survived')
	elif prediction == 0:
		print('Not Survived')



predict_Model(1,1,10,1,1,15,1,1)