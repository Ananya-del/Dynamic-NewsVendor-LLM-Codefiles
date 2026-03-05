import pandas as pd
import numpy as np

def analyzeCSV(extra_1):
    df = pd.read_csv(extra_1)
    df1 = df['profit']
    avgProfit = df1.mean()
    #averageProfit = df.groupby('Iteration').agg({'profit': 'mean'})

    ##finding the percentage of times product i bought
    df2 = df['x1']
    avgX1 = df2.mean()
    df3 = df['x2']
    avgX2 = df3.mean()
    df4 = df['x3']
    avgX3 = df4.mean()
    df5 = df['x4']
    avgX4 = df5.mean()
    df6 = df['x5']
    avgX5 = df6.mean()
    df7 = df['x6']
    avgX6 = df7.mean()
    df8 = df['x7']
    avgX7 = df8.mean()
    df9 = df['x8']
    avgX8 = df9.mean()
    df10 = df['x9']
    avgX9 = df10.mean()
    df11 = df['x10']
    avgX10 = df11.mean()

    finalDict = {
    "X1": avgX1,
    "X2": avgX2,
    "X3": avgX3,
    "X4": avgX4,
    "X5": avgX5,
    "X6": avgX6,
    "X7": avgX7,
    "X8": avgX8,
    "X9": avgX9,
    "X10": avgX10
}

    totalVAl = avgX1 + avgX2 + avgX3 + avgX4 + avgX5 + avgX6 + avgX7 + avgX8 + avgX9 + avgX10

    print(finalDict)
    print("This is " + str(totalVAl))
    print("This is " + str(avgProfit))

analyzeCSV('dynamnews_resultsStoch.csv')