import pandas as pd
import numpy as np
from scipy.stats import chisquare

def analyzeCSV(extra_1):
    df = pd.read_csv(extra_1)
    df1 = df['profit']
    avgProfit = df1.mean()
    

    ## new function, using quantiles in pandas!!!
    profits = df['profit'].quantile([0.05, 0.1, 0.9, 0.95])
    resultStr = []
    resultStr.append("5%: " + str(df1.quantile(0.05)))
    resultStr.append("10%: " + str(df1.quantile(0.1)))
    resultStr.append("50%: " + str(df1.quantile(0.5)))
    resultStr.append("90%: " + str(df1.quantile(0.9)))
    resultStr.append("95%: " + str(df1.quantile(0.95)))

    
    print(resultStr)

    ## --------add matplotlib if theres time!!!!!---------
    ## --------add chi squared test if theres time!!! ------


analyzeCSV('noisy_results.csv') ## replace with whichever csv name

##old code, may still be useful

#averageProfit = df.groupby('Iteration').agg({'profit': 'mean'})

    ##finding the expected value/mean of times product i bought
    # df2 = df['x1']
    # avgX1 = df2.mean()
    # df3 = df['x2']
    # avgX2 = df3.mean()
    # df4 = df['x3']
    # avgX3 = df4.mean()
    # df5 = df['x4']
    # avgX4 = df5.mean()
    # df6 = df['x5']
    # avgX5 = df6.mean()
    # df7 = df['x6']
    # avgX6 = df7.mean()
    # df8 = df['x7']
    # avgX7 = df8.mean()
    # df9 = df['x8']
    # avgX8 = df9.mean()
    # df10 = df['x9']
    # avgX9 = df10.mean()
    # df11 = df['x10']
    # avgX10 = df11.mean()

#     finalDict = {
#     "X1": avgX1,
#     "X2": avgX2,
#     "X3": avgX3,
#     "X4": avgX4,
#     "X5": avgX5,
#     "X6": avgX6,
#     "X7": avgX7,
#     "X8": avgX8,
#     "X9": avgX9,
#     "X10": avgX10
# }

#     totalVAl = avgX1 + avgX2 + avgX3 + avgX4 + avgX5 + avgX6 + avgX7 + avgX8 + avgX9 + avgX10

    # print(finalDict)
    # print("This is " + str(totalVAl))
    # ## Total Value should be around the number of customers. Otherwise something went wrong
    # print("This is " + str(avgProfit))

