# Original data from https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

do_describe = 0				# Determine whether or not the dataset main stats should be displayed (see below)
features='worst'				# Can be equal to 'all', 'mean', 'worst', 'mean_worst' or 'se'


## ==================== 1. READING AND CLEANING THE DATA ==================== ##
raw_data = pd.read_csv('data.csv')

# Remove the first and last columns since they do not contain relevant information
raw_data = raw_data.drop(['id'], axis=1)
raw_data = raw_data.drop(['Unnamed: 32'], axis=1)


## ====================  2. GETTING SOME INFORMATION ABOUT THE DATASET ==================== ##
if do_describe == 1:
	print("Overview of the data:")
	print('')
	print(raw_data.head())
	print('')
	print("--------------------------------------------------------")
	print(raw_data.describe())
	print("--------------------------------------------------------")

# Count the number of positive and negative examples
malign = raw_data['diagnosis'] == 'M'
benign = raw_data['diagnosis'] == 'B'
print('')
print("--------------------------------------------------------")
print("Number of positive (malign) examples in the sample: %d" % raw_data[malign].shape[0])		# = 212
print("Number of negative (benign) examples in the sample: %d" % raw_data[benign].shape[0])		# = 357
print("--------------------------------------------------------")
print('')


## ==================== 3. DATA SELECTION AND FEATURES SCALING ==================== ##
def main():
	X = select_features(raw_data, features)
	y = raw_data['diagnosis'].copy()

	if features != 'all':
		print('')
		print("--------------------------------------------------------")
		print("Displaying correlation matrix for the selected features")
		print("--------------------------------------------------------")
		corr_mtrx(X)
		print('')
		print("-------------------------------------------------------")
		print("Displaying dispersion matrix for the selected features")
		print("-------------------------------------------------------")
		X_plot = X.copy()
		X_plot['diagnosis'] = raw_data['diagnosis'].copy()
		corr_plot(X_plot)
		print('')

	# Scaling the X features so they range between -1 and 1 with an average value of 0
	# NOTE: applying StandardScaler() transforms the pandas dataframe into a numpy array,
	# which is problematic when we want to use pandas specific  attributes like .columns
	# from sklearn.preprocessing import StandardScaler
	# sc = StandardScaler()
	# X_scaled = sc.fit_transform(X)
	X_scaled = (X - X.mean()) / (X.std())
	# Transform the 'diagnosis' column so the values are numerical (1=Malignant, 0=Benign)
	y = y.map({'M': 1, 'B': 0})


	## ==================== 4. RANDOMLY SPLITTING THE DATA INTO TRAINING AND TEST SETS ==================== ##
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=0)


	## ==================== 5. CONSTRUCTING THE MODELS ==================== ##
	print("==============================================================")
	print("= STEP 1: CONSTRUCTING MODELS WITH ALL THE SELECTED FEATURES =")
	print("==============================================================")
	from sklearn import linear_model	# Logistic regression
	logreg = linear_model.LogisticRegression(random_state=0)
	logreg_params = {'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]}

	from sklearn import tree	# Decision tree
	dtree = tree.DecisionTreeClassifier(criterion='entropy', max_features='sqrt', random_state=0)
	dtree_params = {'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
					'min_samples_leaf': [2, 3, 4, 5, 6, 7, 8, 9, 10]}

	from sklearn.model_selection import GridSearchCV	# Grid search (parameters optimization)
	from sklearn import metrics
	models_dict = {logreg: logreg_params, dtree: dtree_params}

	for model, params in models_dict.items():	# Loop over the classifiers
		clf = GridSearchCV(model, params, cv=5, scoring='precision')
		clf.fit(X_train, y_train)
		print("Classifier:")
		print(clf.best_estimator_)
		print('')
		print("Cross-validation: searching for the best parameters...\n")
		print("Best fit parameters: ", clf.best_params_)
		print("Precision score obtained on the training set: %.2f" % clf.best_score_)
		prediction = clf.predict(X_test)
		print("Precision score obtained on the test set: %.2f" % metrics.precision_score(prediction, y_test))
		print('')
		print("--------------------------------------------------------")
	input("Program paused, press Enter to continue...\n")


	## ==================== 6. FEATURES SELECTION ==================== ##
	print("=================================================================")
	print("= STEP 2: CONSTRUCTING MODELS WITH NON-CORRELATED FEATURES ONLY =")
	print("=================================================================")
	# Here, we shall perform the same computations than in Step 1 with a reduced number
	# of features, eliminating features that are correlated to other ones.
	X_filt = X_scaled.filter(regex='^(radius_|concave points_|texture_|smoothness_|symmetry_)\D*$', axis=1)
	X_train2, X_test2, y_train2, y_test2 = train_test_split(X_filt, y, test_size=0.30, random_state=0)
	print("The features retained for this step are: ")
	print(X_filt.columns)
	print('')
	print("--------------------------------------------------------")
	for model, params in models_dict.items():	# Loop over the classifiers
		clf = GridSearchCV(model, params, cv=5, scoring='precision')
		clf.fit(X_train2, y_train2)
		print("Classifier: ")
		print(clf.best_estimator_)
		print('')
		print("Cross-validation: searching for the best parameters...\n")
		print("Best fit parameters: ", clf.best_params_)
		print("Precision score obtained on the training set: %.2f" % clf.best_score_)
		prediction = clf.predict(X_test2)
		print("Precision score obtained on the test set: %.2f" % metrics.precision_score(prediction, y_test2))
		print('')
		print("--------------------------------------------------------")
	input("Program paused, press Enter to continue...\n")

	# Step 3 only works on the whole dataset for now
	if features == 'all'
		print("====================================================================")
		print("= STEP 3: CONSTRUCTING MODELS WITH THE MOST 'SIGNIFICANT' FEATURES =")
		print("====================================================================")
		# Applying RFE method with cross validation to i) classify features from best to worst and
		# ii) find the optimal number of features to use
		from sklearn.feature_selection import RFECV
		clf1 = linear_model.LogisticRegression(random_state=0)	# Creating a new regressor with default parameters
		rfecv = RFECV(estimator=clf1, step=1, cv=5, scoring='precision')
		rfecv = rfecv.fit(X_train, y_train)
		print("Classifier:")
		print(rfecv.estimator_)
		print('')
		print("Optimal number of features : %d" % rfecv.n_features_)	# With C=1.0, the optimal number of features is 20
		
		top_feat1 = pd.Series(rfecv.grid_scores_[:rfecv.n_features_], index=X_train.columns[rfecv.support_]).sort_values(ascending=False)
		print("Best feature rankings and precision scores:")
		print(top_feat1)
		print('')

		X_train_best = rfecv.transform(X_train)	# Reshaping X_train in order to keep the top 20 features
		X_test_best = rfecv.transform(X_test)	# Reshaping X_test in order to keep the top 20 features
		clf1.fit(X_train_best, y_train)
		predictrain1 = clf1.predict(X_train_best)	# Computing predicted values on the training set
		predictest1 = clf1.predict(X_test_best)		# Computing predicted values on the test set
		print("Precision score obtained on the training set: %.2f" % metrics.precision_score(predictrain1, y_train))
		print("Precision score obtained on the test set: %.2f" % metrics.precision_score(predictest1, y_test))
		print('')
		print("--------------------------------------------------------")

		# The decision tree classifier has inherently an attribute estimating feature importances; we shall try it
		clf2 = tree.DecisionTreeClassifier(max_features='sqrt', random_state=0)	# Creating a new tree with default parameters
		clf2.fit(X_train, y_train)
		top_feat2 = pd.Series(clf2.feature_importances_, index=X_train.columns).sort_values(ascending=False)
		print("Classifier:")
		print(rfecv.estimator_)
		print('')
		print("Feature rankings and Gini scores:")
		print(top_feat2)
		# List of all the features having a Gini score of 0.0
		droplist = ['area_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean', 'fractal_dimension_worst',
					'fractal_dimension_mean', 'radius_se', 'perimeter_se', 'symmetry_worst', 'compactness_se',
					'concave points_se', 'smoothness_worst', 'smoothness_se']

		print('')
		print("Computing a new decision tree without the following features:")
		print(droplist)
		print('')
		X_filtrain = X_train.drop(droplist, axis=1)	# Removing all the features in dropllist from the training set
		X_filtest = X_test.drop(droplist, axis=1)	# Removing all the features in dropllist from the test set
		clf2.fit(X_filtrain, y_train)
		predictrain2 = clf2.predict(X_filtrain)	# Computing predicted values on the training set
		predictest2 = clf2.predict(X_filtest)	# Computing predicted values on the test set
		print("Precision score obtained on the training set: %.2f" % metrics.precision_score(predictrain2, y_train))
		print("Precision score obtained on the test set: %.2f" % metrics.precision_score(predictest2, y_test))



## ==================== FUNCTIONS/SUBROUTINES ==================== ##
# Function selecting some or all the features of a pandas dataframe and return them as a numpy array
def select_features(df, features):
	if features == 'mean':
		X = df.filter(regex='(^.+_mean$)', axis=1)
	elif features == 'worst':
		X = df.filter(regex='^.+_worst$', axis=1)
	elif features == 'mean_worst':
		X = df.filter(regex='^.+(_mean|_worst)$', axis=1)
	elif features == 'se':
		X = df.filter(regex='^.+_se$', axis=1)
	elif features == 'all':
		X = df.drop(['diagnosis'], axis=1)
	else:
		print("This is not a valid suffix.")

	if features != 'all':
		print("The following features have been conserved:")
		print(X.columns)
	return X

# Function taking the features X (pandas dataframe) in input and displaying the correlation matrix in output
def corr_mtrx(X):
    corr = X.corr()
    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(corr, cbar=True,  square=True, fmt='.2f', cmap='YlGnBu', annot=True,
    				annot_kws={'size': 15}, xticklabels=X.columns, yticklabels=X.columns, linewidths=.6)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()

    save_plot = input("Do you want to save the current figure (y/n)?")
    if (save_plot == 'y' or save_plot == 'Y'):
    	fig.savefig('Correlation_Matrix.png')
    	print("Figure 'Correlation_Matrix.png' has been saved. Proceeding to the next step...")
    else:
    	print("Figure has not been saved. Proceeding to the next step...")


# Function taking the features X (pandas dataframe) in input and displaying the dispersion matrix in output
def corr_plot(X):
	fig = sns.pairplot(X, palette='husl', markers=['o', 'o'], hue='diagnosis')
	plt.xticks(rotation=90)
	plt.yticks(rotation=0)
	plt.show()
	
	save_plot = input("Do you want to save the current figure (y/n)?")
	if (save_plot == 'y' or save_plot == 'Y'):
		fig.savefig('Scatter_Matrix.png')
		print("Figure 'Scatter_Matrix.png' has been saved. Proceeding to the next step...")
	else:
		print("Figure has not been saved. Proceeding to the next step...")


if __name__ == '__main__':
	main()
