
import csv
from sklearn import svm, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def learn(X,y,test_data,ids):
	
	print 'Training...'
	#Random Forest
	forest = RandomForestClassifier(n_estimators=100)
	forest = forest.fit(X,y)
	output_forest = forest.predict(test_data).astype(int)

	#SVM
	support_vec = svm.SVC()
	support_vec = support_vec.fit(X,y)
	output_svm = support_vec.predict(test_data).astype(int)

	#KNN
	knn = KNeighborsClassifier() 
	knn =  knn.fit(X,y) 
	output_knn = knn.predict(test_data).astype(int)

	#Logistic Regression
	logistic = linear_model.LogisticRegression()
	logistic = logistic.fit(X,y)
	output_logistic = logistic.predict(test_data).astype(int)



	print 'Predicting...'
	#Random Forest
	predictions_file = open("random_forest.csv", "wb")
	open_file_object = csv.writer(predictions_file)
	open_file_object.writerow(["PassengerId","Survived"])
	open_file_object.writerows(zip(ids, output_forest))
	predictions_file.close()

	#SVM
	predictions_file = open("svm.csv", "wb")
	open_file_object = csv.writer(predictions_file)
	open_file_object.writerow(["PassengerId","Survived"])
	open_file_object.writerows(zip(ids, output_svm))
	predictions_file.close()

	#KNN
	predictions_file = open("knn.csv", "wb")
	open_file_object = csv.writer(predictions_file)
	open_file_object.writerow(["PassengerId","Survived"])
	open_file_object.writerows(zip(ids, output_knn))
	predictions_file.close()

	#Logistic Regression
	predictions_file = open("logistic3.csv", "wb")
	open_file_object = csv.writer(predictions_file)
	open_file_object.writerow(["PassengerId","Survived"])
	open_file_object.writerows(zip(ids, output_logistic))
	predictions_file.close()

	print 'Done.'