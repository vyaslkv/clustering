import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import os

class Clusters_PacksAs_Samples():
    """
    This class will be used to make the drug clusters.
    """

    def __init__(self,file_path):
        self.pack_drug_dict = {}
        self.data_mat = []
        self.file_path = file_path
        self.z = []
        self.cluster1 = []
        self.cluster1 = []
        self.numberOfRobots = 2
        self.packs_list = []
        self.df = pd.DataFrame()

    def divide_linkage_matrix_in_clusters(self, linkage_matrix, number_of_clusters):
        """
        This method will devide our samples in number_of_clusters specified based on the given linkage matrix.
        :param linkage_matrix:
        :param number_of_clusters:
        :return:
        """
        print ("linkage matrix", linkage_matrix, len(linkage_matrix))
        input("enter0")
        self.linkage_dict = {}
        self.cluster_dict = {}
        initial_key = len(linkage_matrix)+1
        for row in linkage_matrix:
            # converting floats into intigers by the below method
            list_needed =  map(int, row[:2])
            print ("row", list_needed)
            # we are making a dict and mapping at which index which cluster index will merge
            self.linkage_dict[initial_key] = list_needed
            initial_key += 1
        end_key_id = initial_key - 1
        input("enter1")
        print (self.linkage_dict)
        # now we have made a empty dict for number of clusters
        for i in range(number_of_clusters):
            self.cluster_dict[i] = []
        print (self.cluster_dict, "cluster_dict", end_key_id)
        input("enter2")
        # initialising cluster dict
        # storing the values of end key from linkage dict to cluster_dict
        #for i in self.cluster_dict:
        for i in list(self.cluster_dict.keys())[:2]:
            print (self.linkage_dict[end_key_id-1][i])
            self.cluster_dict[i].append(self.linkage_dict[end_key_id-1][i])
            self.cluster_dict[i].append(self.linkage_dict[end_key_id - 2][i])

        for i in self.cluster_dict:
            if i == 2 or i == 3:
                    continue
            else:
            self.cluster_dict[i].remove(self.cluster_dict[i][1])
            self.cluster_dict[i+2].append(self.cluster_dict[i][1])

        print (self.cluster_dict)
        input("enter3")
        # process cluster dict
        for i in self.cluster_dict.keys():
            print (i)
            # here the processing array is the values which we have recently assingned to cluster dict like 37,39
            processing_array = self.cluster_dict[i]
            # 37
            print (processing_array)
            # Process processing array
            while True:
                element_changed_flag = 0
                for element in processing_array:
                    if element in self.linkage_dict.keys():
                        processing_array.remove(element)
                        processing_array.extend(self.linkage_dict[element])
                        element_changed_flag = 1
                        break
                if element_changed_flag == 0:
                    break
            print (processing_array)
        input("enter3")

        pass

    def read_excel_file(self):
        """
        This method will be used to read csv file, and convert data of csv file into the dictionary form.
        :param csv_file_name:
        :return: data in matrix form
        """
        self.df = pd.read_excel(str(self.file_path))
        self.data_mat=np.array(self.df)


    def clustering(self):
        """
        this method will make the clusters of the drugs
        :param data_mat:
        :return: clusters of the drugs
        """
        self.z = linkage(self.data_mat, method='ward', metric='euclidean', optimal_ordering=False)
        self.divide_linkage_matrix_in_clusters(self.z, 4)

    def cluster_visulization(self):
        """
        this method will help to visulize the clusters
        :param z:
        :return: dendrogram
        """
        dn = dendrogram(self.z)
        plt.axhline(y=2)
        plt.show()
        # self.cluster1 = self.z[len(self.z) - 1]
        # self.cluster2 = self.z[len(self.z) - 2]
        #
        # plt.scatter(self.cluster1, self.cluster2)
        # self.clusterdict ={}
        # for i in range(1,int(self.cluster1[3]+1)):
        #
        #     self.clusterdict[i]= self.z[len(self.z) - i][3]
        #
        print('\n')
        print(self.z)
        print('\n')
        self.clustes = []
        self.cluster_list = []
        self.packs = []
        print(len(self.z))
        # this is the linkage matrix of the last cluster
        c1=self.z[len(self.z)-1]
        print(c1)
        c21 = c1[0]
        print(c21)
        c22 = c1[1]
        print(c22)
        c31 = []
        c32 = []

        if c21 > len(self.data_mat):
            c21 = c21 - len(self.data_mat)
            c31.append(c21)

        if c22 > len(self.data_mat):
            c22 = c22 - len(self.data_mat)
            c32.append(c22)
        print(c31)
        print(c32)
        c41 = []
        c42 = []
        c41.append(self.z[int(c31[0])][:2])
        c42.append(self.z[int(c32[0])][:2])
        print(c41)
        print(c42)
        c51 = []
        c52 = []



        if c41[0][0] > len(self.data_mat):
                 c41[0][0] = c41[0][0] - len(self.data_mat)
                 c51.append(int(c41[0][0]))

        if c41[0][1] > len(self.data_mat):
                c41[0][1] = c41[0][1] - len(self.data_mat)
                c51.append(int(c41[0][1]))

        if c42[0][0] > len(self.data_mat):
            c42[0][0] = c42[0][0] - len(self.data_mat)
            c52.append(int(c42[0][0]))

        if c42[0][1] > len(self.data_mat):
            c42[0][1] = c42[0][1] - len(self.data_mat)
            c52.append(int(c42[0][1]))
        print(c51)
        print(c52)
        c61 =[]
        c62 = []


        c61.append(self.z[int(c51[0])][:2])
        c61.append(self.z[int(c51[1])][:2])
        c62.append(self.z[int(c52[0])][:2])
        c62.append(self.z[int(c52[1])][:2])
        print(c61)
        print(c62)
        c71 = []
        c72 = []


        if c61[0][0] >= len(self.data_mat):
            c61[0][0] = c61[0][0] - len(self.data_mat)
            c71.append(int(c61[0][0]))
        if c61[0][1] >= len(self.data_mat):
            c61[0][1] = c61[0][1] - len(self.data_mat)
            c71.append(int(c61[0][1]))
        if c61[1][0] >= len(self.data_mat):
            c61[1][0] = c61[1][0] - len(self.data_mat)
            c71.append(int(c61[1][0]))
        if c61[1][1] >= len(self.data_mat):
            c61[1][1] = c61[1][1] - len(self.data_mat)
            c71.append(int(c61[1][1]))


        if c62[0][0] >= len(self.data_mat):
            c62[0][0] = c62[0][0] - len(self.data_mat)
            c72.append(int(c62[0][0]))
        if c62[0][1] >= len(self.data_mat):
            c62[0][1] = c62[0][1] - len(self.data_mat)
            c72.append(int(c62[0][1]))
        if c62[1][0] >= len(self.data_mat):
            c62[1][0] = c62[1][0] - len(self.data_mat)
            c72.append(int(c62[1][0]))
        if c62[1][1] >= len(self.data_mat):
            c62[1][1] = c62[1][1] - len(self.data_mat)
            c72.append(int(c62[1][1]))


        print(c71)
        print(c72)



        cf1 = self.z[len(self.z) - 1]

        cf2 = cf1[:2]
        cf3 = []
        for i in cf2:
            if i > len(self.data_mat):
                i = i - len(self.data_mat)
                cf3.append(i)

        cf4 = []
        for i in range(len(cf3)):
            cf4.append(self.z[int(cf3[i])][:2])

        cf5 = []
        for j in range(len(cf4)):
            for i in cf4[j]:
                if i > len(self.data_mat):
                    i = i - len(self.data_mat)
                    cf5.append(i)
        cf6 = []
        for i in range(len(cf5)):
            cf6.append(self.z[int(cf5[i])][:2])
        cf7 = []
        for j in range(len(cf6)):
            for i in cf6[j]:
                if i >= len(self.data_mat):
                    i = i - len(self.data_mat)
                    cf7.append(i)
        #print(cf7)






    def run_algorithm(self):
        self.read_excel_file()
        self.clustering()
        self.cluster_visulization()


class Clusters_DrugsAs_Samples():
    """
    This class will be used to make the drug clusters.
    """

    def __init__(self, file_path):
        self.pack_drug_dict = {}
        self.data_mat = []
        self.file_path = file_path
        self.z = []
        self.cluster = []
        self.packs = []
        self.drugs = []
        self.df = pd.DataFrame()
        self.numberOfRobots = 3
        self.cluster_list = []
        self.packs_list = []
        self.drugs_list = []
        self.drug = []

    def read_excel_file(self):
        """
        This method will be used to read csv file, and convert data of csv file into the dictionary form.
        :param csv_file_name:
        :return: data in matrix form
        """
        self.df = pd.read_excel(str(self.file_path))
        self.data_mat = np.array(self.df).transpose()

    def clustering(self):
        """
        this method will make the clusters of the drugs
        :param data_mat:
        :return: clusters of the drugs
        """
        self.z = linkage(self.data_mat, method='ward', metric='euclidean', optimal_ordering=False)

    def cluster_visulization(self):
        """
        this method will help to visulize the clusters
        :param z:
        :return: dendrogram
        """
        dn = dendrogram(self.z)
        plt.show()

    def packsToDrugs(self):
        try:
            for i in range(self.numberOfRobots):
                self.cluster = self.z[len(self.z) - (i + 1)]
                self.cluster_list.append(self.cluster)
            print(self.cluster_list)
            for j in range(len(self.cluster_list)):
                self.packs = self.cluster_list[j][:2]
                self.packs_list.append(self.packs)
            print(self.packs_list)

            for k in range(len(self.packs_list)):
                self.drugs = self.df.iloc[[int(self.packs_list[k][0]), int(self.packs_list[k][1])]]

                self.drugs_list.append(self.drugs)

            print(self.drugs_list)
        except IndexError:
            print("not that much clusters")




    def run_algorithm(self):
        self.read_excel_file()
        self.clustering()
        self.cluster_visulization()
        self.packsToDrugs()
        '''
        [   d0  d1  d2
        2  23  44  65
        3  24  45  66,
            d0  d1  d2
        0  21  42  63
        1  22  43  64,
            d0  d1  d2
        2  23  44  65
        3  24  45  66]

        '''
class Lossfunction():
    def __init__(self):

        self.deviation=np.array()

    def stdeviation(self, clusters):
        self.deviation = np.std(clusters, ddof=1)
        print(self.deviation)









cwd = os.getcwd()
file_path = cwd + '/test2.xlsx'
print ("file_path", file_path)

c =Clusters_PacksAs_Samples(file_path)
c.run_algorithm()
