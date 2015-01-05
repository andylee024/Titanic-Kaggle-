#Import libraries
import clean_data as clean #script to clean data
import ml #script to apply machine learning algorithms


def main():
	#build dataframe for analysis
	train_data = '/home/andy/Projects/Kaggle/Titanic/Data/train.csv'
	test_data = '/home/andy/Projects/Kaggle/Titanic/Data/test.csv'
	df_train, ids_train = clean.build_df(train_data)
	df_test, ids_test = clean.build_df(test_data)


	#build training sets
	train_X = df_train[0::,1::] #predictor features span all rows and 1st column on
	train_y = df_train[0::,0] #response variables span all rows and 0th column only

	#train and predict
	ml.learn(train_X,train_y,df_test,ids_test)
	

	
if __name__ == '__main__':
  main()