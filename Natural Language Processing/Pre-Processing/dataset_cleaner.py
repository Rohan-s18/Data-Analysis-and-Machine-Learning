"""
Author: Rohan Singh
Python module to clean html datasets
Date: 28 May 2023
"""


#  Imports
from text_cleaner import remove_html
import pandas as pd


#  Helper function to obtain the messy text from the dataset
def get_messy(filepath, title):
    df = pd.read_csv(filepath)

    return df[title].to_list()
    


#  Helper function to clean all of the data
def clean_all(data):
    cleaned = []

    for i in range(0,len(data), 1):
        cleaned.append(remove_html(data[i]))

    return cleaned


#  Helper function to make the output dataset
def make_dataset(messy, clean, write, output_filepath):
    df = pd.DataFrame(list(zip(messy, clean)), columns =['messy', 'clean'])

    if write:
        df.to_csv(output_filepath)

    return df


#  Main function to test functionality
def main():
    
    input_filepath = ""
    output_filepath = ""

    messy = get_messy(input_filepath,"")

    clean = clean_all(messy)

    make_dataset(messy, clean, True, output_filepath)

    

    
    


if __name__ == "__main__":
    main()
