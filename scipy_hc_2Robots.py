import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import os
import more_itertools

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
        self.linkage_dict = {}
        self.cluster_dict = {}
        self.clusterdict_from_df_as_drug_frequency = {}
        self.clusterdict_from_df_as_drug_non_O_frequency = {}
        self.clusterdict_from_as_drugs_only_as_list = {}
        self.clusterdict_of_non_repeated_drugs = {}

    def divide_linkage_matrix_in_clusters(self, linkage_matrix, number_of_clusters):
        """
        This method will devide our samples in number_of_clusters specified based on the given linkage matrix.
        :param linkage_matrix:
        :param number_of_clusters:
        :return:
        """
        initial_key = len(linkage_matrix)+1
        for row in linkage_matrix:
            list_needed =  map(int, row[:2])
            self.linkage_dict[initial_key] = list_needed
            initial_key += 1
        end_key_id = initial_key - 1
        for i in range(number_of_clusters):
            self.cluster_dict[i] = []
        print (self.cluster_dict, "cluster_dict", end_key_id)

        # initialising cluster dict
        for i in self.cluster_dict:
            self.cluster_dict[i].append(self.linkage_dict[end_key_id][i])
        # process cluster dict
        for i in self.cluster_dict.keys():
            processing_array = self.cluster_dict[i]
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

        print ("cluster dict", self.cluster_dict)

    def pack_clusters_to_drugs_suggestion_for_robot(self):
        '''
        This function performs several tasks described below.
        1) Checks which type of drugs are used in packs of any cluster.
        2) Makes list of drugs which are used in that cluster.
        3) If drugs are common between two clusters, it gives priority to frequency of particular cluster.
        :return: type of drugs in each cluster
        '''
        self.packs_to_drugs_without_filter()
        # self.clusterdict_of_non_repeated_drugs
        print("in method: suggestion for robot, clusterdict non repeated drug",self.clusterdict_of_non_repeated_drugs)

        try:
            common_drug_list = [x for x in self.clusterdict_of_non_repeated_drugs[0] if x in self.clusterdict_of_non_repeated_drugs[1]]
            print('\n')
            print("common drug list", common_drug_list)
            total_frequency_of_drugs_dict = {}
            for i in self.cluster_dict:
                total_frequency_of_drugs_dict[i] = []

            for drug in common_drug_list:

                for cluster_keys in self.clusterdict_from_df_as_drug_non_O_frequency.keys():
                    temp_list = []
                    for cluster_values_as_list in self.clusterdict_from_df_as_drug_non_O_frequency[cluster_keys]:
                        try:
                            temp_list.append(cluster_values_as_list[str(drug)])
                        except KeyError:
                            print("\t")
                    total_frequency_of_drugs_dict[cluster_keys].append(np.sum(temp_list))
            print("total drugs frequency",total_frequency_of_drugs_dict)
            total_frequency_of_drugs_dict_with_drugs = {}
            for i in self.cluster_dict:
                total_frequency_of_drugs_dict_with_drugs[i] = []
            temp_list1 = []
            temp_list2 = []
            for keys in self.cluster_dict.keys():
                temp_list1.append(self.clusterdict_of_non_repeated_drugs[keys])
            for keys in self.cluster_dict.keys():
                temp_list2.append(total_frequency_of_drugs_dict[keys])
            temp_list3 = []
            for i in temp_list1:
                for j in temp_list2:
                    temp_list3.append(dict(zip(i,j)))
            temp_list4 = temp_list3[:2]
            print('\n')
            for keys in self.cluster_dict:
                total_frequency_of_drugs_dict_with_drugs[keys].append(temp_list4[keys])
            print("total frequency with drugs dict",total_frequency_of_drugs_dict_with_drugs)

            final_drugs_in_clusters_dict = {}
            for i in self.cluster_dict:
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

            for i in self.cluster_dict:
                clusterdict_from_as_drugs_only_as_list[i] = []

            for i in self.cluster_dict:
                for j in total_frequency_of_drugs_dict_with_drugs[i]:
                    clusterdict_from_as_drugs_only_as_list[i].append(j.keys())
            print("only keys drugs with drugs name", clusterdict_from_as_drugs_only_as_list)
            print('\n')

            for i in self.cluster_dict:
                clusterdict_of_non_repeated_drugs[i] = list(more_itertools.collapse([list(x) for x in set([tuple(x) for x in clusterdict_from_as_drugs_only_as_list[i]])]))
            print("only drugs",clusterdict_of_non_repeated_drugs)

            final_robot_packs_dict = {}
            for i in self.cluster_dict:
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

            for i in self.cluster_dict:
                print(i)
                for pack in self.cluster_dict[i]:
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

            for i in self.cluster_dict:
                final_robot_packs_dict[i] = set(final_robot_packs_dict[i])

            print("final which pack which robot dict",final_robot_packs_dict)

        except IndexError:
            print("No common drugs")

    def packs_to_drugs_without_filter(self):
        """
        This function just checks how many types of drugs are used in given pack cluster.
        :param cluster:
        :return: drug_array
        """
        print("without filter's function: clusterdict",self.cluster_dict)
        print('\n')
        for i in self.cluster_dict:
            self.clusterdict_from_df_as_drug_frequency[i] = []

        for i in self.cluster_dict:
            for j in self.cluster_dict[i]:
                self.clusterdict_from_df_as_drug_frequency[i].append(self.df.iloc[j].to_dict())

        for i in self.cluster_dict:
            self.clusterdict_from_df_as_drug_non_O_frequency[i] = []

        for i in self.cluster_dict:
            for j in self.clusterdict_from_df_as_drug_frequency[i]:
                self.clusterdict_from_df_as_drug_non_O_frequency[i].append({x: y for x, y in j.items() if y != 0})
        print("clusterdict_from_df_as_drug_non_O_frequency", self.clusterdict_from_df_as_drug_non_O_frequency)
        print('\n')

        for i in self.cluster_dict:
            self.clusterdict_from_as_drugs_only_as_list[i] = []

        for i in self.cluster_dict:
            for j in self.clusterdict_from_df_as_drug_non_O_frequency[i]:
                self.clusterdict_from_as_drugs_only_as_list[i].append(j.keys())

        print("only keys drugs with drugs name", self.clusterdict_from_as_drugs_only_as_list)
        print('\n')


        for i in self.cluster_dict:
            self.clusterdict_of_non_repeated_drugs[i]=list(more_itertools.collapse([list(x) for x in set([tuple(x) for x in self.clusterdict_from_as_drugs_only_as_list[i]])]))


        print("only drugs only", self.clusterdict_of_non_repeated_drugs)


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
        self.divide_linkage_matrix_in_clusters(self.z, 2)

        self.pack_clusters_to_drugs_suggestion_for_robot()


    def cluster_visulization(self):
        """
        this method will help to visulize the clusters
        :param z:
        :return: dendrogram
        """
        dn = dendrogram(self.z)
        plt.axhline(y=2)
        plt.show()



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

    def divide_linkage_matrix_in_clusters(self, linkage_matrix, number_of_clusters):
        """
        This method will devide our samples in number_of_clusters specified based on the given linkage matrix.
        :param linkage_matrix:
        :param number_of_clusters:
        :return:
        """
        print ("linkage matrix", linkage_matrix, len(linkage_matrix))

        self.linkage_dict = {}
        self.cluster_dict = {}
        initial_key = len(linkage_matrix) + 1
        for row in linkage_matrix:
            list_needed = map(int, row[:2])
            print ("row", list_needed)
            self.linkage_dict[initial_key] = list_needed
            initial_key += 1
        end_key_id = initial_key - 1

        print (self.linkage_dict)
        for i in range(number_of_clusters):
            self.cluster_dict[i] = []
        print (self.cluster_dict, "cluster_dict", end_key_id)

        # initialising cluster dict
        for i in self.cluster_dict:
            print (self.linkage_dict[end_key_id][i])
            self.cluster_dict[i].append(self.linkage_dict[end_key_id][i])
        print (self.cluster_dict)

        # process cluster dict
        for i in self.cluster_dict.keys():
            print (i)
            processing_array = self.cluster_dict[i]
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

        print("cluster of drugs dict",self.cluster_dict)

        self.drugsdict = {}
        for i in self.cluster_dict:
            self.drugsdict[i] = []
        drugslist = list(self.df.columns.values)
        print("drugs list from dataframe", drugslist)

        for i in self.cluster_dict:
            self.drugsdict[i] = [drugslist[index] for index in self.cluster_dict[i]]

        print("drugs cluster dict", self.drugsdict)
########################################################################################################################


        clusterdict_from_df_as_drug_frequency = {}
        clusterdict_from_df_as_drug_non_O_frequency = {}

        print('\n')

        for i in range(len(self.df)):
            clusterdict_from_df_as_drug_frequency[i] = []

        for i in range(len(self.df)):
            clusterdict_from_df_as_drug_frequency[i].append(self.df.iloc[i].to_dict()) #
        print("packs in dict form of drugs frequency",clusterdict_from_df_as_drug_frequency)

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
            clusterdict_of_non_repeated_drugs[i]=list(more_itertools.collapse([list(x) for x in set([tuple(x) for x in clusterdict_from_as_drugs_only_as_list[i]])]))


        print("only drugs only", clusterdict_of_non_repeated_drugs)

########################################################################################################################
        robot_for_packs_dict = {}
        for i in range(len(self.df)):
            robot_for_packs_dict[i] = []

        #for i in range(len(self.df)):
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
        print("clusterdict_of_non_repeated_drugs",robot_for_packs_dict)






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
        self.divide_linkage_matrix_in_clusters(self.z, 2)
        plt.show()





    def run_algorithm(self):
        self.read_excel_file()
        self.clustering()
        self.cluster_visulization()

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
