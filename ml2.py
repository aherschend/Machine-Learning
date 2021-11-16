import pandas as pd 
from sklearn.model_selection import train_test_split

nyc = pd.read_csv("ave_hi_nyc_jan_1895-2018.csv")

#prints the first 3 rows
# date will be x variable and temperature will be y

print(nyc.head(3))

# to use train-test-split we need a 2D model so we need to convert it from a series

print(nyc.Date.values)

print(nyc.Date.values.reshape(-1,1))
#the -1 is telling it to convert as many columns as we have to rows and the 1 tells it to create 1 column

X_train, X_test, y_train, y_test = train_test_split(
    nyc.Date.values.reshape(-1,1), nyc.Temperature.values, random_state=11)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X=X_train, y=y_train)

coef = lr.coef_
intercept = lr.intercept_

predicted = lr.predict(X_test) # only giving it test not target 
expected = y_test 

#the reason this prints so weird is there really isn't a corellation b/w a year and a temperature
print(predicted[:20])
print(expected[:20])

predict = lambda x: coef * x + intercept

print(predict(2025))    

#print a scatterplot with a regression line to see where they show up

import seaborn as sns

axes = sns.scatterplot(
    data=nyc,
    x="Date",
    y="Temperature",
    hue="Temperature",
    palette="winter",
    legend=False
)

axes.set_ylim(10,70)

import numpy as np
x = np.array([min(nyc.Date.values),max(nyc.Date.values)])
print(x)
y = predict(x)
print(y)

import matplotlib.pyplot as plt
line = plt.plot(x,y)

plt.show()