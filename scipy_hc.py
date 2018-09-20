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
        c81 = []
        c82 = []
        for i in c71:
            c81.append(self.z[i][:2])
        for i in c72:
            c82.append(self.z[i][:2])

        print(c81)
        print(c82)

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

c =Clusters_PacksAs_Samples('test2.xlsx')
c.run_algorithm()
