from pylab import plot,show
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans,vq
import os
import more_itertools
class Clusters_PacksAs_Samples():
    """
    This class will be used to make the drug clusters.
    """

    def __init__(self,file_path):
        self.data_mat = []
        self.file_path = file_path
        self.k = 2
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


        self.indexdict = {}
        for i in self.clusterdict:
            self.indexdict[i] = []
        print(len(self.clusterdict))
        for i in range(len(idx)):
            for j in range(len(self.clusterdict)):
                if (self.clusterdict[j][:] == self.data_mat[i]).any():
                    self.indexdict[j].append(i)
        print("cluster dict of packs",self.indexdict)

        self.drugdict = {}
        for i in self.clusterdict:
            self.drugdict[i] = []
        self.drug=[]
        for i in range(len(self.indexdict.keys())):
            for j in range(len(self.indexdict[i])):
                self.drugdict[i].append(self.df.iloc[self.indexdict[i][j]].to_dict())
        print("drugs dict with their frequencies",self.drugdict)
        clusterdict_from_df_as_drug_non_O_frequency = {}
        clusterdict_from_as_drugs_only_as_list = {}
        clusterdict_of_non_repeated_drugs ={}
        for i in self.drugdict:
            clusterdict_from_df_as_drug_non_O_frequency[i] = []
        for i in self.drugdict:
            for j in self.drugdict[i]:
                clusterdict_from_df_as_drug_non_O_frequency[i].append({x: y for x, y in j.items() if y != 0})
        print("clusterdict_from_df_as_drug_non_O_frequency", clusterdict_from_df_as_drug_non_O_frequency)
        print('\n')

        for i in self.drugdict:
            clusterdict_from_as_drugs_only_as_list[i] = []

        for i in self.drugdict:
            for j in clusterdict_from_df_as_drug_non_O_frequency[i]:
                clusterdict_from_as_drugs_only_as_list[i].append(j.keys())

        print("only keys drugs with drugs name", clusterdict_from_as_drugs_only_as_list)
        print('\n')


        for i in self.drugdict:
            clusterdict_of_non_repeated_drugs[i]=list(more_itertools.collapse([list(x) for x in set([tuple(x) for x in clusterdict_from_as_drugs_only_as_list[i]])]))


        print("only drugs only", clusterdict_of_non_repeated_drugs)

########################################################################################################################
        try:
            common_drug_list = [x for x in clusterdict_of_non_repeated_drugs[0] if x in clusterdict_of_non_repeated_drugs[1]]
            print('\n')
            print("common drug list", common_drug_list)
            total_frequency_of_drugs_dict = {}
            for i in self.drugdict:
                total_frequency_of_drugs_dict[i] = []

            for drug in common_drug_list:

                for cluster_keys in clusterdict_from_df_as_drug_non_O_frequency.keys():
                    temp_list = []
                    for cluster_values_as_list in clusterdict_from_df_as_drug_non_O_frequency[cluster_keys]:
                        try:
                            temp_list.append(cluster_values_as_list[str(drug)])
                        except KeyError:
                            print("\t")
                    total_frequency_of_drugs_dict[cluster_keys].append(np.sum(temp_list))
            print("total drugs frequency",total_frequency_of_drugs_dict)
            total_frequency_of_drugs_dict_with_drugs = {}
            for i in self.drugdict:
                total_frequency_of_drugs_dict_with_drugs[i] = []
            temp_list1 = []
            temp_list2 = []
            for keys in self.drugdict.keys():
                temp_list1.append(clusterdict_of_non_repeated_drugs[keys])
            for keys in self.drugdict.keys():
                temp_list2.append(total_frequency_of_drugs_dict[keys])
            temp_list3 = []
            for i in temp_list1:
                for j in temp_list2:
                    temp_list3.append(dict(zip(i,j)))
            temp_list4 = temp_list3[:2]
            print('\n')
            for keys in self.drugdict:
                total_frequency_of_drugs_dict_with_drugs[keys].append(temp_list4[keys])
            print("total frequency with drugs dict",total_frequency_of_drugs_dict_with_drugs)

            final_drugs_in_clusters_dict = {}
            for i in self.drugdict:
                final_drugs_in_clusters_dict[i] = []
            compare_list = []
            for drug in common_drug_list:
                compare_list.append(min(total_frequency_of_drugs_dict_with_drugs[0][0][drug], total_frequency_of_drugs_dict_with_drugs[1][0][drug]))
            print("compare list",compare_list)
            for values in total_frequency_of_drugs_dict_with_drugs.values():
                for key1, value1 in values[0].items():
                    if value1 in compare_list:

                        key2 =values[0].keys()[values[0].values().index(value1)]
                        values[0].pop(key2, None)


            print('final dict with deleted keys', total_frequency_of_drugs_dict_with_drugs)

            clusterdict_from_as_drugs_only_as_list = {}
            clusterdict_of_non_repeated_drugs = {}

            for i in self.drugdict:
                clusterdict_from_as_drugs_only_as_list[i] = []

            for i in self.drugdict:
                for j in total_frequency_of_drugs_dict_with_drugs[i]:
                    clusterdict_from_as_drugs_only_as_list[i].append(j.keys())
            print("only keys drugs with drugs name", clusterdict_from_as_drugs_only_as_list)
            print('\n')

            for i in self.drugdict:
                clusterdict_of_non_repeated_drugs[i] = list(more_itertools.collapse([list(x) for x in set([tuple(x) for x in clusterdict_from_as_drugs_only_as_list[i]])]))
            print("only drugs",clusterdict_of_non_repeated_drugs)

            final_robot_packs_dict = {}
            for i in self.drugdict:
                final_robot_packs_dict[i] = []

            winner_drug_dict = {}
            for i in common_drug_list:
                winner_drug_dict[i] = []
            for drug in common_drug_list:
                if drug in clusterdict_of_non_repeated_drugs[0]:
                    winner_drug_dict[str(drug)].append(0)
                if drug in clusterdict_of_non_repeated_drugs[1]:
                    winner_drug_dict[str(drug)].append(1)
            print("winner drug dict",winner_drug_dict)

            for i in self.indexdict:
                print(i)
                for pack in self.indexdict[i]:
                    packdict = self.df.iloc[pack].to_dict()
                    packdict_non_0 = {x: y for x, y in packdict.items() if y != 0}
                    packdict_non_0_key = packdict_non_0.keys()
                    for drug in packdict_non_0_key:
                        if drug in clusterdict_of_non_repeated_drugs[0]:
                            final_robot_packs_dict[0].append(pack)
                        elif drug in clusterdict_of_non_repeated_drugs[1]:
                            final_robot_packs_dict[1].append(pack)

                    final_robot_packs_dict[i].append(pack)
                    for commondrugs in winner_drug_dict:
                        for winnercluster in winner_drug_dict[commondrugs]:
                            if winnercluster==0:
                                loosercluster =1
                            if winnercluster == 1:
                                loosercluster = 0
                        if commondrugs in packdict_non_0_key and i==loosercluster:
                            try:
                                final_robot_packs_dict[i].remove(pack)
                                final_robot_packs_dict[winnercluster].append(pack)
                            except ValueError:
                                print('\t')

            for i in self.indexdict:
                final_robot_packs_dict[i] = set(final_robot_packs_dict[i])

            print("final which pack which robot dict",final_robot_packs_dict)

        except IndexError:
            print("No common drugs")



    def run_algorithm(self):
        self.read_excel_file()
        self.clustering_and_visulization()


class Clusters_DrugsAs_Samples():
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

        index_dict ={}
        for i in self.clusterdict:
            index_dict[i] = []
        for i in range(len(self.data_mat)):
            for j in range(len(self.clusterdict)):
                if (self.clusterdict[j][:] == self.data_mat[i]).any():
                    index_dict[j].append(i)
        print("drugs cluster dict", index_dict)

        self.drugsdict = {}
        for i in index_dict:
            self.drugsdict[i] = []
        drugslist = list(self.df.columns.values)
        print("drugs list from dataframe", drugslist)

        for i in index_dict:
            self.drugsdict[i] = [drugslist[index] for index in index_dict[i]]

        print("drugs cluster dict", self.drugsdict)
########################################################################################################################
        clusterdict_from_df_as_drug_frequency = {}
        clusterdict_from_df_as_drug_non_O_frequency = {}

        print('\n')

        for i in range(len(self.df)):
            clusterdict_from_df_as_drug_frequency[i] = []

        for i in range(len(self.df)):
            clusterdict_from_df_as_drug_frequency[i].append(self.df.iloc[i].to_dict())  #
        print("packs in dict form of drugs frequency", clusterdict_from_df_as_drug_frequency)

        for i in range(len(self.df)):
            clusterdict_from_df_as_drug_non_O_frequency[i] = []

        for i in range(len(self.df)):
            for j in clusterdict_from_df_as_drug_frequency[i]:
                clusterdict_from_df_as_drug_non_O_frequency[i].append({x: y for x, y in j.items() if y != 0})
        print("clusterdict_from_df_as_drug_non_O_frequency", clusterdict_from_df_as_drug_non_O_frequency)
        print('\n')
        clusterdict_from_as_drugs_only_as_list = {}
        clusterdict_of_non_repeated_drugs = {}
        for i in range(len(self.df)):
            clusterdict_from_as_drugs_only_as_list[i] = []

        for i in range(len(self.df)):
            for j in clusterdict_from_df_as_drug_non_O_frequency[i]:
                clusterdict_from_as_drugs_only_as_list[i].append(j.keys())

        print("only keys drugs with drugs name", clusterdict_from_as_drugs_only_as_list)
        print('\n')

        for i in range(len(self.df)):
            clusterdict_of_non_repeated_drugs[i] = list(more_itertools.collapse(
                [list(x) for x in set([tuple(x) for x in clusterdict_from_as_drugs_only_as_list[i]])]))

        print("only drugs only", clusterdict_of_non_repeated_drugs)

########################################################################################################################
        robot_for_packs_dict = {}
        for i in range(len(self.df)):
            robot_for_packs_dict[i] = []

        # for i in range(len(self.df)):
        for i in range(len(self.df)):
            for j in clusterdict_of_non_repeated_drugs[i]:
                if j in self.drugsdict[0]:
                    robot_for_packs_dict[i].append(0)
                elif j in self.drugsdict[1]:
                    robot_for_packs_dict[i].append(1)
        for i in range(len(self.df)):
            robot_for_packs_dict[i] = set(robot_for_packs_dict[i])

        for i in range(len(self.df)):
            robot_for_packs_dict[i] = list(more_itertools.collapse(robot_for_packs_dict[i]))
        print('\n')
        print("clusterdict_of_non_repeated_drugs", robot_for_packs_dict)

    def run_algorithm(self):
        self.read_excel_file()
        self.clustering_and_visulization()



cwd = os.getcwd()
file_path = cwd + '/test3.xlsx'
print ("file_path", file_path)

c =Clusters_DrugsAs_Samples(file_path)
c.run_algorithm()
