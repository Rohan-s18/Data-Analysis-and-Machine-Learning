"""
Author: Rohan Singh
Module to create similarity matrix using the Longest smiles subsequence
"""


#  Imports
import matplotlib.pyplot as plt
import random as rand
import pandas as pd
import numpy as np


#  Function to find the smiles similarity using LCS on smile data
def lcs(X, Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)
 
    # declaring the array for storing the dp values
    L = [[None]*(n + 1) for i in range(m + 1)]
 
    """Following steps build L[m + 1][n + 1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0 :
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
 
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]/max(m,n)


#  Function to create a 2-D matrix of lcs values of trp
def create_similarity_matrix(x, y):

    #Matrix list
    matrix = []
    for i in range(0,len(x),1):
        row = []
        for j in range(0,len(y),1):

            #Calculating the similairty using the longest common subsequence 
            val = lcs(x[i],x[j])
            row.append(val)
        row = np.array(row)
        matrix.append(row)

    #Returning the similarity matrix
    return np.array(matrix)

 
 
#  main function to test the code
def main():

    X = "ABCD"
    Y = "ABCD"
    print("Length of LCS is ", lcs(X, Y))

    print("\n\n")

    df = pd.read_csv("/Users/rohansingh/github_repos/Data-Analysis-and-Machine-Learning/Tensor Decomposition/Datasets/intersections/intersections_ic50.csv")

    arr = df["smile"].to_numpy()

    sm = create_similarity_matrix(arr, arr)

    print(sm)

    df_w = pd.DataFrame(sm, columns=df["name"].to_list())

    df_w.to_csv("/Users/rohansingh/github_repos/Data-Analysis-and-Machine-Learning/Tensor Decomposition/Datasets/Side_Information/si_5.csv")



if __name__ == "__main__":
    main()