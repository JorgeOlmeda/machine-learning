# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)
results[2][2]
r = sorted([(results[i][2][0][3], results[i][2][0][2], results[i][1], results[i]) for i in range(len(results))],
           reverse=True)
 
for item in r:
    print("| Lift:", f"{item[0]:.2f}",
          "| Conf:", f"{item[1]:.2f}",
          "| Supp:", f"{item[2]:.4f}",
          "| Item:", list(item[3][0]))

def inspect(results):
    rh          = [tuple(result[2][0][0]) for result in results]
    lh          = [tuple(result[2][0][1]) for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(rh, lh, supports, confidences, lifts))
#https://docs.python.org/3.3/library/functions.html#zip
#zip combina elprimer elemento de cada variable en una lista, el segundo igual    
# this command creates a data frame to view
resultDataFrame=pd.DataFrame(inspect(results),
                columns=['rhs','lhs','support','confidence','lift'])

