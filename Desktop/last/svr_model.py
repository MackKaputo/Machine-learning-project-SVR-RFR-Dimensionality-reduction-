import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load the excel sheet as a pd dataframe called df: 
df = pd.read_excel("rafmdata.xlsx")

#Have a look at the first 5 rows of the table:
df.head()
# Looking at the features' names: 
df.columns
# Replace the NaN (missing values) with zero (0) :
df = df.fillna(value = 0)
# Removing non useful features for stated reasons in report: 
#This include Mo,Nb,P,S,Al,O,Zr,Y,NT,Nt,Ni

df.drop('Mo', axis = 1, inplace = True)
df.drop('Nb', axis = 1, inplace = True)
df.drop('P', axis = 1, inplace = True)
df.drop('S', axis = 1, inplace = True)
df.drop('Al', axis = 1, inplace = True)
df.drop('O', axis = 1, inplace = True)
df.drop('Zr', axis = 1, inplace = True)
df.drop('Y', axis = 1, inplace = True)
df.drop('NT', axis = 1, inplace = True)
df.drop('Nt(min)', axis = 1, inplace = True)
df.drop('Ni', axis = 1, inplace = True)

# For the excel file: 

writer = pd.ExcelWriter('rafm_data_preprocessed.xlsx', engine = 'xlsxwriter')
df.to_excel(writer, sheet_name = 'rafm_data_preprocessed', index = False)
writer.save()

#Imports
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.svm import SVR
from sklearn import metrics

df = pd.read_csv("rafm_data_preprocessed.csv")

# Split test and training
X = df[['C','Cr','Mn','Si','W','V','Ta','Ti','N','B','TT','Tt(min)']]
y = df['YS']
#Plotting the correlation Headmap

heatmap_figure = plt.figure(figsize = (11,7))
plt.title("Correlation Matrix Heatmap", fontsize = 25)
sns.heatmap(correlation_matrix,cmap='Reds',annot=True)
heatmap_figure.savefig("correlation_matrix.png", dpi=300)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=55)

#standardize the features 

sc = StandardScaler()

#For X training set, we do "fit_transform" because we need to compute mean and std,
#and then use it to autoscale the data. For X test set, well, we already have the mean and std,
#so we only do the "transform" part.

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
X_std = sc.fit_transform(X)

#initialize PCA and svm regression model
pca = PCA(n_components = 3)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
X_pca = pca.transform(X_std)


svr_mod = SVR(gamma = 'auto', kernel="rbf")
svr_mod.fit(X_train_pca, y_train)
predictions_svr = svr_mod.predict(X_test_pca)

MAE_svr = metrics.mean_absolute_error(y_test, predictions_svr)
MSE_svr = metrics.mean_squared_error(y_test, predictions_svr)
RMSE_svr = np.sqrt(metrics.mean_squared_error(y_test, predictions_svr))

my_scores = []
my_errors = []
for slack in range(1,2500):
    svr_mod = SVR(gamma = 'auto', kernel="rbf", epsilon = 25, C = slack)
    svr_mod.fit(X_train_pca, y_train)
    predictions_svr = svr_mod.predict(X_test_pca)

    MAE_svr = metrics.mean_absolute_error(y_test, predictions_svr)
    MSE_svr = metrics.mean_squared_error(y_test, predictions_svr)
    RMSE_svr = np.sqrt(metrics.mean_squared_error(y_test, predictions_svr))
    score = svr_mod.score(X_test_pca, y_test)
    my_scores.append(score)
    my_errors.append(MAE_svr) 

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0,5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    #plt.ylim(-75000,100)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

title = "Learning Curve"
cv = 3
plot_learning_curve(estimator=svr_mod, title=title, X=X, y=y, cv=cv, n_jobs=1);

train_sz, train_errs, cv_errs = learning_curve(estimator=svr_mod, X=X_pca, y=y,
                                              train_sizes=np.linspace(0.1, 1, 10),
                                              scoring="neg_mean_squared_error")

# For each training subset, compute average error over the 3-fold cross val
tr_err = np.mean(train_errs, axis=1)
cv_err = np.mean(cv_errs, axis=1)

# Plot the errors to make a learning curve!
fig, ax = plt.subplots(figsize=(10,7))
ax.plot(train_sz, tr_err, linestyle="--", color="r", label="training error")
ax.plot(train_sz, cv_err, linestyle="-", color="b", label="cv error")
ax.legend(loc="lower right")
#snp.labs("Training Set Size", "Score (4-Fold CV avg)", "LC with High Bias")
#ax.legend(loc="lower right")
plt.title("SVM Regressor Loss function (Negative mean squared error)", fontsize=20)
plt.xlabel("Training Set Size", fontsize=15)
plt.ylabel("Negative Mean Squared Error",fontsize=15)
#plt.savefig("SVR_PCA_loss_function_RMSE_negative",dpi=300)

plt.figure(figsize = (10,7))
plt.plot(range(1,2500),my_errors)
plt.xlabel("Hyperparameter (C)", fontsize = 15)
plt.ylabel("Mean absolute error", fontsize = 15)
plt.title("Hyperparameter (C) Search with Mean Absolute Error", fontsize = 20)
plt.savefig("svm_PCA_Hyperparameter.png", dpi=300)



X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
X_std = sc.fit_transform(X)

#initialize PCA and svm new regression model
pca = PCA(n_components = 12)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
X_pca = pca.transform(X_std)


svr_mod = SVR(gamma = 'auto', kernel="rbf", epsilon=25, C=391)
svr_mod.fit(X_train_pca, y_train)
predictions_svr = svr_mod.predict(X_test_pca)

MAE_svr = metrics.mean_absolute_error(y_test, predictions_svr)
MSE_svr = metrics.mean_squared_error(y_test, predictions_svr)
RMSE_svr = np.sqrt(metrics.mean_squared_error(y_test, predictions_svr))
