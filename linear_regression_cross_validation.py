import numpy as np
import pandas as pd
import statsmodels.api as sm

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

#added for Logistic regressions lesson 2.4.2
loansData.to_csv('loansData_clean.csv', header=True, index=False)

cleanInterestRate = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%'))/100, 4))
loansData['Interest.Rate'] = cleanInterestRate
#print loansData['Interest.Rate'][0:5]

cleanLoansLength = loansData['Loan.Length'][0:5].map(lambda x: int(x.rstrip(' months')))
loansData['Loan.Length'] = cleanLoansLength
#print loansData['Loan.Length'][0:5]

#cleanFICORange = loansData['FICO.Range'].map(lambda x: x.split('-'))
#cleanFICORange = cleanFICORange.map(lambda x: [int(n) for n in x])
#loansData['FICO.Range'] = cleanFICORange
#print loansData['FICO.Range'][0:5]

loansData['FICO.Score'] = [int(val.split('-')[0]) for val in loansData['FICO.Range']]

intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']


#transpose data
#The dependent variable
y = np.matrix(intrate).transpose()
# The independent variables shaped as columns
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()

#print intrate
#print loanamt
#print fico
x = np.column_stack([x1,x2])
X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()
print f.summary()

#new cross_validation stuff in 4.1.5
from sklearn.cross_validation import KFold

kf = KFold(2500, n_folds=10) #defining a list where each list has (1. list of row numbers for Training set, 2. list of row numbers for testing set)


#Calculate average R^2
r2 = []
for train, test in kf:
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    model = sm.OLS(y_train, X_train)
    f = model.fit()
    r2.append(f.rsquared)
#this loop runs the model n-1 = 9 times

r2average = sum(r2) / float(len(r2))
print ("The average R2 is "+ str(r2average) + ".")
print ("The average R2 is fairly high around 65%. This means we can explain 65% of the variation in the model.")


#Calculate MSE
y_hat = f.predict(X_test) # predict y based on testing data
total_square_error = 0

for prediction, actual in zip(y_hat, y_test): #wth is zip?

     total_square_error += (prediction-actual)**2

mse = total_square_error / len(y_hat)
print ("The average MSE is "+ str(mse) + ".")
print ("The average MSE is around .0005. This is each observation's deviation from the predicted value, squared. MSE tends to heavily weight statistical outliers, so a smaller MSE is good to see.")


#Calculate MAE
total_absolute_error = 0

for prediction, actual in zip(y_hat, y_test):

     total_absolute_error += abs(prediction-actual)

mae = total_absolute_error / len(y_hat)
print ("The average MAE is "+ str(mae) + ".")
print ("The average MAE is around .019. This is the absolute values in each observation's deviation from the predicted value. It averages magnitude of the errors in a set of forecasts, without considering their direction. It measures accuracy for continuous variables.")

