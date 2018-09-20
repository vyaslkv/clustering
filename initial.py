import numpy as np
import pandas as pd
import os
from heapq import nsmallest

class Algorithm_a():
    """

    """

    def __init__(self, file_path):
        self.df = pd.DataFrame()
        self.file_path = file_path


    def read_excel_file(self):
        """
        This method will be used to read csv file, and convert data of csv file into the dictionary form.
        :param excel_file_name:
        :return: data in matrix form
        """
        self.df = pd.read_excel(str(self.file_path))
        self.df1 = self.df.astype(bool).astype(int)
        print("dataframe", self.df1)

    def batch_splitting(self):
        self.robotdict = {}
        for i in range(2):
            self.robotdict[i] = i+1
        print("robot dict", self.robotdict)
        temp_list = []
        for i in self.robotdict:
            temp_list.append(self.robotdict[i])
        self.capacity = np.sum(temp_list)
        packdict = {}
        for i in range(len(self.df)):
            packdict[i] = self.df.iloc[i].to_dict()
        print("pack dict", packdict)
        packdictnon_0 = {}
        for i in packdict:
            packdictnon_0[i] = {}
        for i in packdict:
            packdictnon_0[i] = {x: y for x, y in packdict[i].items() if y != 0}
        print("packdictnon_0", packdictnon_0)
        drugslist = list(self.df.columns.values)
        print("drug list", drugslist)
        drugdict = {}
        for drug in drugslist:
            temp_list = []
            for i in packdict:
                temp_list.append(packdict[i][drug])
            drugdict[drug] = np.sum(temp_list)
        print("drug frequency dict", drugdict)
        temp_list1  = []
        for i in drugdict:
            temp_list1.append(drugdict[i])
        print("self.capacity", self.capacity)
        self.difference = len(drugslist) - self.capacity
        print("self.difference",self.difference)
        n_smallest = nsmallest(self.difference,temp_list1)
        temp_list1_drugs = []
        print("n smallest", n_smallest)
        for num in n_smallest:
            for key, value in drugdict.items():
                if value == num:
                    temp_list1_drugs.append(key)
        print("n smallest drugs",temp_list1_drugs)
        for drug in temp_list1_drugs:
            for i in packdictnon_0.keys():
                for key, value in packdictnon_0[i].items():
                    if key == drug:
                        packdictnon_0.pop(i, None)
        print("final packdict", packdictnon_0)

    def run_algorithm(self):
        self.read_excel_file()
        self.batch_splitting()



cwd = os.getcwd()
file_path = cwd + '/test4.xlsx'
print ("file_path", file_path)

c =Algorithm_a(file_path)
c.run_algorithm()