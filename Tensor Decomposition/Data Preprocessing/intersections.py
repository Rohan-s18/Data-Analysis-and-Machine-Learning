"""
Author: Rohan Singh
Python Module to find the intersectio between different drugs across different datasets
"""

#  Imports
import pandas as pd
import numpy as np
from pubchempy import *
import random


#  Getting the attributes for ic50
def get_ic(filepath, attr):
    df = pd.read_csv(filepath)
    temp = df[attr].to_numpy()
    return temp


#  Getting the attributes for dd disease
def get_ddd(filepath, attr):
    df = pd.read_csv(filepath)
    temp = df[attr].to_numpy()
    return temp


#  helper function to get the intersection between the kinome names and the ddi names for drugs
def get_intersection_drugs(db1, db2):
    db1_set  = set(db1)
    db2_set = set(db2)
    intersections = db1_set.intersection(db2_set)
    print(len(intersections))
    return np.array(intersections)


#  Main Function
def main():

    filepath_ic50 = "/Users/rohansingh/github_repos/Data-Analysis-and-Machine-Learning/Tensor Decomposition/Datasets/IC50.csv"
    filepath_ddd = "/Users/rohansingh/github_repos/Data-Analysis-and-Machine-Learning/Tensor Decomposition/Datasets/intersection_pairs_dd_disease.csv"

    ic = get_ic(filepath_ic50,"CELL_NAME")
    ddd = get_ddd(filepath_ddd,"cell_line")

    intersection = get_intersection_drugs(ic,ddd)

    print("The Intersection is:")
    print(intersection)

    print("\n\n")

    print(ic)
    print("\n\n")
    print(ddd)



if __name__ == "__main__":
    main()
