from pylab import plot,show
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans,vq
from collections import defaultdict


class Clusters_DrugsAs_Samples():
    """
    This class will be used to make the drug clusters.
    """

    def __init__(self,file_path):
        self.data_mat = []
        self.file_path = file_path
        self.k = 3
        self.idx_list= []
        self.color_code = ['ob', 'or', 'oc', 'om', 'oy', 'ok']
        self.plot_list = []
        self.plot_list1 = []
        self.clusterlist = []
        self.clusterdict = {}
        self.df = pd.DataFrame()
    def read_excel_file(self):
        """
        This method will be used to read csv file, and convert data of csv file into the dictionary form.
        :param csv_file_name:
        :return: data in matrix form
        """
        self.df = pd.read_excel(str(self.file_path))
        self.data_mat=np.array(self.df).astype(float)


    def clustering_and_visulization(self):
        """
        this method will make the clusters of the drugs
        :param data_mat:
        :return: clusters of the drugs and it's visulization
        """
        centroids, _ = kmeans(self.data_mat, self.k)
        idx, _ = vq(self.data_mat, centroids)
        for i in range(self.k):

            self.plot_list.append(self.data_mat[idx == i, 0])
            self.plot_list1.append(self.data_mat[idx == i, 1])

        for j in range(self.k):
            plot(self.plot_list[j], self.plot_list1[j], self.color_code[j])
            plot(centroids[:, 0], centroids[:, 1], 'sg', markersize=8)
        show()
        for i in range(self.k):
            self.cluster = self.data_mat[idx == i]
            self.clusterlist.append(self.cluster)

        for i in range(len(self.clusterlist)):
            self.clusterdict[i] = self.clusterlist[i]
        print(self.clusterdict)


        self.indexdict = defaultdict(list)
        print(len(self.clusterdict))
        for i in range(len(idx)):
            for j in range(len(self.clusterdict)):
                if (self.clusterdict[j][:] == self.data_mat[i]).any():
                    self.indexdict[j].append(i)
        print(self.indexdict)
        self.drugdict=defaultdict(list)
        self.drug=[]
        for i in range(len(self.indexdict.keys())):
            for j in range(len(self.indexdict[i])):
                self.drugdict[i].append(self.df.iloc[self.indexdict[i][j]])
        print(self.drugdict)
        '''
        defaultdict(<class 'list'>, {0: [d0    35
        d1    56
        d2    77
        Name: 14, dtype: int64, d0    36
        d1    57
        d2    78
        Name: 15, dtype: int64, d0    37
        d1    58
        d2    79
        Name: 16, dtype: int64, d0    38
        d1    59
        d2    80
        Name: 17, dtype: int64, d0    39
        d1    60
        d2    81
        Name: 18, dtype: int64, d0    40
        d1    61
        d2    82
        Name: 19, dtype: int64, d0    41
        d1    62
        d2    83
        Name: 20, dtype: int64], 1: [d0    28
        d1    49
        d2    70
        Name: 7, dtype: int64, d0    29
        d1    50
        d2    71
        Name: 8, dtype: int64, d0    30
        d1    51
        d2    72
        Name: 9, dtype: int64, d0    31
        d1    52
        d2    73
        Name: 10, dtype: int64, d0    32
        d1    53
        d2    74
        Name: 11, dtype: int64, d0    33
        d1    54
        d2    75
        Name: 12, dtype: int64, d0    34
        d1    55
        d2    76
        Name: 13, dtype: int64], 2: [d0    21
        d1    42
        d2    63
        Name: 0, dtype: int64, d0    22
        d1    43
        d2    64
        Name: 1, dtype: int64, d0    23
        d1    44
        d2    65
        Name: 2, dtype: int64, d0    24
        d1    45
        d2    66
        Name: 3, dtype: int64, d0    25
        d1    46
        d2    67
        Name: 4, dtype: int64, d0    26
        d1    47
        d2    68
        Name: 5, dtype: int64, d0    27
        d1    48
        d2    69
        Name: 6, dtype: int64]})

        '''




    def run_algorithm(self):
        self.read_excel_file()
        self.clustering_and_visulization()


class Clusters_PacksAs_Samples():
    """
    This class will be used to make the packs clusters.
    """

    def __init__(self,file_path):
        self.data_mat = []
        self.file_path = file_path
        self.k = 2
        self.idx_list= []
        self.color_code = ['ob', 'or', 'oc', 'om', 'oy', 'ok']
        self.plot_list = []
        self.plot_list1 = []
        self.cluster_list = []
        self.idx = []
        self.packs_list = []
        self.df = pd.DataFrame()
        self.drugs_list = []
        self.clusterlist = []
        self.clusterdict = {}

    def read_excel_file(self):
        """
        This method will be used to read csv file, and convert data of csv file into the dictionary form.
        :param csv_file_name:
        :return: data in matrix form
        """
        self.df = pd.read_excel(str(self.file_path))
        self.data_mat=np.array(self.df).astype(float).transpose()

    def clustering_and_visulization(self):
        """
        this method will make the clusters of the packs
        :param data_mat:
        :return: clusters of the packs and it's visulization
        """
        try:
            centroids, _ = kmeans(self.data_mat, self.k)
        except ValueError:
            print("The number of clusters is more than the data points")
        self.idx, _ = vq(self.data_mat, centroids)
        for i in range(self.k):

            self.plot_list.append(self.data_mat[self.idx == i, 0])
            self.plot_list1.append(self.data_mat[self.idx == i, 1])

        for j in range(self.k):
            plot(self.plot_list[j], self.plot_list1[j], self.color_code[j])
            plot(centroids[:, 0], centroids[:, 1], 'sg', markersize=8)
        show()

        for i in range(self.k):
            self.cluster = self.data_mat[self.idx == i]
            self.clusterlist.append(self.cluster)
        print(self.clusterlist)
        for i in range(len(self.clusterlist)):
            self.clusterdict[i] = self.clusterlist[i]
        print(self.clusterdict)
        index_dict = defaultdict(list)
        for i in range(len(self.data_mat)):
            for j in range(len(self.clusterdict)):
                if (self.clusterdict[j][:] == self.data_mat[i]).any():
                    index_dict[j].append(i)
        print(index_dict)
        self.drugdict = defaultdict(list)
        for i in range(len(index_dict.keys())):
            for j in range(len(index_dict[i])):
                self.drugdict[i].append(self.df.iloc[index_dict[i][j]])
        print(self.drugdict)
        '''
        defaultdict(<class 'list'>, {0: [d0    21
        d1    42
        d2    63
        Name: 0, dtype: int64, d0    22
        d1    43
        d2    64
        Name: 1, dtype: int64], 1: [d0    23
        d1    44
        d2    65
        Name: 2, dtype: int64]})

        '''

        # temp_all_lists = []
        # for key, values in self.clusterdict.items():
        #     for value in values:
        #         temp_all_lists.append(value)
        # # all lists segmented in temp_all_lists
        # list_dict = defaultdict(list)
        # for i in range(len(temp_all_lists)):
        #     list_dict[i].append(temp_all_lists[i])
        #
        # print(list_dict)
        # index_dict = defaultdict(list)
        # for i in range(len(self.data_mat)):
        #     for j in range(len(list_dict)):
        #         if (list_dict[j][:] == self.data_mat[i]).any():
        #             index_dict[j].append(i)
        # print(index_dict)
        # self.drugdict = defaultdict(list)
        # for i in range(len(index_dict.keys())):
        #     for j in range(len(index_dict[i])):
        #         self.drugdict[i].append(self.df.iloc[index_dict[i][j]])
        # print(self.drugdict)









    def run_algorithm(self):
        self.read_excel_file()
        self.clustering_and_visulization()


c=Clusters_PacksAs_Samples('test2.xlsx')
c.run_algorithm()