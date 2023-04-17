"""
Author: Rohan Singh
Python Module to find the intersectio between drugs across transcriptionsal response profile and dd disease datasets
"""

#  Imports
import pandas as pd
import numpy as np
from pubchempy import *
import random


#  Getting the attributes for trp
def get_trp(filepath, attr):
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


#  helper function to get the intersection names
def get_intersection_names(intersection_set, filepath):

    temp = set(intersection_set.tolist())

    names = []

    df = pd.read_csv(filepath)

    names_df = df["Drug1_name"].to_numpy()
    cell_lines = df["cell_line"].to_numpy()

    for i in range(0,len(names_df),1):
        if cell_lines[i] in  temp:
            names.append(names_df[i])

    return set(names)

#  helper function to write the intersection dataset
def write_intersection(intersection_set, ddd, filepath):
    names = []
    drugbank_id =[]
    inchi =[]
    inchikey = []
    smiles = []

    df = pd.read_csv(filepath)
    names_df = df["name"].to_numpy()
    drugbank_id_df = df["drugbank_id"].to_numpy()
    inchi_df = df["inchi"].to_numpy()
    inchikey_df = df["inchikey"].to_numpy()
    smile_df = df["smile"].to_numpy()


    for i in range(0,len(names_df),1):
        if names_df[i] in intersection_set:
            names.append(names_df[i])
            drugbank_id.append(drugbank_id_df[i])
            inchi.append(inchi_df[i])
            inchikey.append(inchikey_df[i])
            smiles.append(smile_df[i])
            


    result = pd.DataFrame(list(zip(names, drugbank_id, inchi, inchikey, smiles)),columns =['name', 'drugbank_id','inchi','inchikey','smile'])
    return result


#  Main Function
def main():

    filepath_trp = "/Users/rohansingh/github_repos/Data-Analysis-and-Machine-Learning/Tensor Decomposition/Datasets/IC50.csv"
    filepath_ddd = "/Users/rohansingh/github_repos/Data-Analysis-and-Machine-Learning/Tensor Decomposition/Datasets/intersection_pairs_dd_disease.csv"

    trp = get_trp(filepath_trp,"CELL_NAME")
    ddd = get_ddd(filepath_ddd,"cell_line")

    intersection = get_intersection_drugs(trp,ddd)

    

    print("The Intersection is:")
    print(intersection)

    """
    print("\n\n")

    print(trp)
    print("\n\n")
    print(ddd)

    """

    #print(write_intersection(pd.read_csv("/Users/rohansingh/github_repos/Data-Analysis-and-Machine-Learning/Tensor Decomposition/Datasets/intersection_pairs_dd_disease.csv"),pd.read_csv("/Users/rohansingh/github_repos/Data-Analysis-and-Machine-Learning/Tensor Decomposition/Datasets/intersection_pairs_dd_disease.csv"),intersection))


    intersecting_names = get_intersection_names(intersection, filepath_ddd)

    df = write_intersection(intersecting_names, ddd, "/Users/rohansingh/github_repos/Data-Analysis-and-Machine-Learning/Tensor Decomposition/Datasets/intersections.csv")

    print("\n\nThese are the intersecting Drugs")

    temp = pd.read_csv(filepath_ddd)

    print(intersecting_names)

    #print(intersecting_names.intersection(set(temp["Drug1_name"].to_list())))

    print("\n\nThis is the intersecting dataframe")

    print(df)

    print("\n\n")

    df.to_csv("/Users/rohansingh/github_repos/Data-Analysis-and-Machine-Learning/Tensor Decomposition/Datasets/intersections/intersections_ic50.csv")



if __name__ == "__main__":
    main()
