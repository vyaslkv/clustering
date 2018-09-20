import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import os
from pylab import plot,show
from scipy.cluster.vq import kmeans,vq
import more_itertools
from time_predictor import Pack_Time_Predictor

class Masterclass():
    def __init__(self,file_path):
        self.file_path = file_path

        self.hcp = HC_Clusters_PacksAs_Samples(self.file_path)
        self.hcp.run_algorithm()
        #self.clusterlist_hcp = self.hcp.pack_clusters_to_drugs_suggestion_for_robot()
        #self.totaltime_hcp = self.hcp.packs_to_drugs_without_filter()


        self.hcd = HC_Clusters_DrugsAs_Samples(self.file_path)
        self.hcd.run_algorithm()
        #self.clusterlist_hcd = self.hcd.test()
        #self.totaltime_hcd = self.hcd.test1()

        self.kmp = KMeans_Clusters_PacksAs_Samples(self.file_path)
        self.kmp.run_algorithm()
        #self.clusterlist_kmp = self.kmp.test()
        #self.totaltime_kmp = self.kmp.clustering_and_visulization()

        self.kmd = KMeans_Clusters_DrugsAs_Samples(self.file_path)
        self.kmd.run_algorithm()
        #self.clusterlist_kmd = self.kmd.clustering_and_visulization()
        #self.totaltime_kmd = self.kmd.test()

    def losscalculation(self):
         # self.lf = Lossfunction()
         # std_for_hcp = self.lf.stdeviation(self.clusterlist_hcp)
         # std_for_kmp = self.lf.stdeviation(self.clusterlist_kmp)
         # print("std for std_for_hcp",std_for_hcp)
         # print("std for std_for_kmp", std_for_kmp)
         # std_for_hcd = self.lf.stdeviation(self.clusterlist_hcd)
         # std_for_kmd = self.lf.stdeviation(self.clusterlist_kmd)
         # print("std for std_for_hcd", std_for_hcd)
         # print("std for std_for_kmd", std_for_kmd)
         # print("self.totaltime_hcp cluster one",self.totaltime_hcp[0])
         # print("self.totaltime_hcp cluster two", self.totaltime_hcp[1])
         # print('\n')
         #
         # print("self.totaltime_hcd cluster one", self.totaltime_hcd[0])
         # print("self.totaltime_hcd cluster two", self.totaltime_hcd[1])
         # print('\n')
         #
         # print("self.totaltime_kmp cluster one", self.totaltime_kmp[0])
         # print("self.totaltime_kmp cluster two", self.totaltime_kmp[1])
         # print('\n')
         #
         # print("self.totaltime_kmd cluster one", self.totaltime_kmd[0])
         # print("self.totaltime_kmd cluster two", self.totaltime_kmd[1])
         # print('\n')
         pass

class HC_Clusters_PacksAs_Samples():
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

        print ("cluster dict 0&1 of packs of hc packs", self.cluster_dict)

    def pack_clusters_to_drugs_suggestion_for_robot(self):
        '''
        This function performs several tasks described below.
        1) Checks which type of drugs are used in packs of any cluster.
        2) Makes list of drugs which are used in that cluster.
        3) If drugs are common between two clusters, it gives priority to frequency of particular cluster.
        :return: type of drugs in each cluster
        '''
        self.packs_to_drugs_without_filter()
        for i in self.clusterdict_of_non_repeated_drugs:
            self.clusterdict_of_non_repeated_drugs[i] = set(self.clusterdict_of_non_repeated_drugs[i])
        for i in self.clusterdict_of_non_repeated_drugs:
            self.clusterdict_of_non_repeated_drugs[i] = list(more_itertools.collapse(self.clusterdict_of_non_repeated_drugs[i]))
        print("in method: suggestion for robot, clusterdict non repeated drug of hc as packs",self.clusterdict_of_non_repeated_drugs)

        try:
            common_drug_list = set([x for x in self.clusterdict_of_non_repeated_drugs[0] if x in self.clusterdict_of_non_repeated_drugs[1]])
            print('\n')
            common_drug_list = list(more_itertools.collapse(common_drug_list))
            print("common drug list of hc as packs", common_drug_list)
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
            print("sum of total drugs frequency of hc as packs",total_frequency_of_drugs_dict)
            total_frequency_of_drugs_dict_with_drugs = {}
            for i in self.cluster_dict:
                total_frequency_of_drugs_dict_with_drugs[i] = []
            temp_list1 = []
            temp_list2 = []
            for keys in self.cluster_dict.keys():
                # here just appending the non repeating drugs name
                temp_list1.append(self.clusterdict_of_non_repeated_drugs[keys])
            for keys in self.cluster_dict.keys():
                # here just appending sum of the total frequency of drugs
                temp_list2.append(total_frequency_of_drugs_dict[keys])
            temp_list3 = []
            for i in temp_list1:
                for j in temp_list2:
                    temp_list3.append(dict(zip(i,j)))
            temp_list4 = temp_list3[:2]
            print('\n')
            for keys in self.cluster_dict:
                total_frequency_of_drugs_dict_with_drugs[keys].append(temp_list4[keys])
            print("sum of total drugs frequency with drugs dict of hc as packs",total_frequency_of_drugs_dict_with_drugs)

            final_drugs_in_clusters_dict = {}
            for i in self.cluster_dict:
                final_drugs_in_clusters_dict[i] = []
            compare_list = []
            for drug in total_frequency_of_drugs_dict_with_drugs[0][0].keys():
                compare_list.append(min(total_frequency_of_drugs_dict_with_drugs[0][0][drug], total_frequency_of_drugs_dict_with_drugs[1][0][drug]))
            print("compare list of hc as packs",compare_list)
            for values in total_frequency_of_drugs_dict_with_drugs.values():
                for key1, value1 in values[0].items():
                    if value1 in compare_list:
                        key2 =values[0].keys()[values[0].values().index(value1)]
                        values[0].pop(key2, None)


            print('final dict with deleted keys from the looser cluter of hc as packs', total_frequency_of_drugs_dict_with_drugs)

            clusterdict_from_as_drugs_only_as_list = {}
            clusterdict_of_non_repeated_drugs = {}

            for i in self.cluster_dict:
                clusterdict_from_as_drugs_only_as_list[i] = []

            for i in self.cluster_dict:
                for j in total_frequency_of_drugs_dict_with_drugs[i]:
                    clusterdict_from_as_drugs_only_as_list[i].append(j.keys())
            print("final dict with deleted keys from the looser cluter with removed frequency of hc as packs", clusterdict_from_as_drugs_only_as_list)
            print('\n')

            for i in self.cluster_dict:
                clusterdict_of_non_repeated_drugs[i] = list(more_itertools.collapse([list(x) for x in set([tuple(x) for x in clusterdict_from_as_drugs_only_as_list[i]])]))
            print("final dict with deleted keys from the looser cluter with removed frequency  with beautified result of hc as packs",clusterdict_of_non_repeated_drugs)

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
            print("winner drug dict of hc as packs",winner_drug_dict)

            for i in self.cluster_dict:
                for pack in self.cluster_dict[i]:
                    packdict = self.df.iloc[pack].to_dict()
                    packdict_non_0 = {x: y for x, y in packdict.items() if y != 0}
                    packdict_non_0_key = packdict_non_0.keys()
                    for drug in packdict_non_0_key:
                        if drug in clusterdict_of_non_repeated_drugs[0]:
                            final_robot_packs_dict[0].append(pack)
                        elif drug in clusterdict_of_non_repeated_drugs[1]:
                            final_robot_packs_dict[1].append(pack)
                    loosercluster = int
                    winnercluster = int
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

            print("final which pack which robot dict of hc as packs",final_robot_packs_dict)
            self.hcp_number_of_packs_in_cluster1 = len(self.cluster_dict[0])
            self.hcp_number_of_packs_in_cluster2  = len(self.cluster_dict[1])


        except IndexError:
            print("No common drugs of hc as packs")

        return [self.hcp_number_of_packs_in_cluster1, self.hcp_number_of_packs_in_cluster2]
    def packs_to_drugs_without_filter(self):
        """
        This function just checks how many types of drugs are used in given pack cluster.
        :param cluster:
        :return: drug_array
        """
        print("without filter's function: clusterdict 0&1 of hc packs",self.cluster_dict)
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
        print("clusterdict_ 0&1 from_df_as_drug_non_O_frequency of drugs of hc packs", self.clusterdict_from_df_as_drug_non_O_frequency)
        print('\n')

        for i in self.cluster_dict:
            self.clusterdict_from_as_drugs_only_as_list[i] = []

        for i in self.cluster_dict:
            for j in self.clusterdict_from_df_as_drug_non_O_frequency[i]:
                self.clusterdict_from_as_drugs_only_as_list[i].append(j.keys())

        print("from_df_as_drug_non_O_frequency with removed frequency only drugs of hc packs", self.clusterdict_from_as_drugs_only_as_list)
        print('\n')


        for i in self.cluster_dict:
            self.clusterdict_of_non_repeated_drugs[i]=list(more_itertools.collapse([list(x) for x in set([tuple(x) for x in self.clusterdict_from_as_drugs_only_as_list[i]])]))

        for i in self.clusterdict_of_non_repeated_drugs:
            if __name__ == '__main__':
                self.clusterdict_of_non_repeated_drugs[i] = list(more_itertools.collapse(set(self.clusterdict_of_non_repeated_drugs[i])))
        print("from_df_as_drug_non_O_frequency with removed frequency only drugs, repeted drugs removed of hc packs", self.clusterdict_of_non_repeated_drugs)
        ################################################################################################################
        t_loss = Pack_Time_Predictor()
        timelosslist = []
        timelosslist1 = []
        lenlist = []
        lenlist1 = []
        for packdict in self.clusterdict_from_df_as_drug_non_O_frequency[0]:
                timelosslist.append(packdict.values())
                lenlist.append(len(packdict.values()))
        for packdict in self.clusterdict_from_df_as_drug_non_O_frequency[1]:
                timelosslist1.append(packdict.values())
                lenlist1.append(len(packdict.values()))
        maxlen = np.max(lenlist)
        df = pd.DataFrame(timelosslist)
        print("df of hc packs one before padding", df)
        maxdummydf= t_loss.load_dataset_from_database_type1()
        for i in range(maxlen+1, maxdummydf+1):
            df.insert(maxlen,i,0)
        df = df.fillna(0)
        print("df of hc packs one after padding",df)
        X = df[list(df.columns.values)]
        predictions = t_loss.predict_given_input(X)
        totaltimecluster1= np.sum(list(predictions))
        ################################################################################################################
        maxlen1 = np.max(lenlist1)
        df1 = pd.DataFrame(timelosslist1)
        print("df of hc packs two before padding", df1)
        maxdummydf1 = t_loss.load_dataset_from_database_type1()
        for i in range(maxlen1 + 1, maxdummydf1 + 1):
            df1.insert(maxlen1, i, 0)
        df1 = df1.fillna(0)
        print("df of hc packs two after padding", df1)
        X = df1[list(df1.columns.values)]
        predictions1 = t_loss.predict_given_input(X)
        totaltimecluster2 = np.sum(list(predictions1))

        print("total time in cluster one hc packs",totaltimecluster1)
        print("total time in cluster two hc packs", totaltimecluster2)
        return totaltimecluster1, totaltimecluster2
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


class HC_Clusters_DrugsAs_Samples():
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
        self.linkage_dict = {}
        self.cluster_dict = {}
        initial_key = len(linkage_matrix) + 1
        for row in linkage_matrix:
            list_needed = map(int, row[:2])
            self.linkage_dict[initial_key] = list_needed
            initial_key += 1
        end_key_id = initial_key - 1

        print (self.linkage_dict)
        for i in range(number_of_clusters):
            self.cluster_dict[i] = []
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

        print("cluster of drug's index dict of hc as drugs",self.cluster_dict)

        self.drugsdict = {}
        for i in self.cluster_dict:
            self.drugsdict[i] = []
        drugslist = list(self.df.columns.values)
        print("drugs list from dataframe of hc as drugs", drugslist)

        for i in self.cluster_dict:
            self.drugsdict[i] = [drugslist[index] for index in self.cluster_dict[i]]

        print("drugs cluster dict of hc as drugs", self.drugsdict)
        self.test()
        ################################################################################################################
    def test(self):

        clusterdict_from_df_as_drug_frequency = {}
        clusterdict_from_df_as_drug_non_O_frequency = {}

        print('\n')

        for i in range(len(self.df)):
            clusterdict_from_df_as_drug_frequency[i] = []

        for i in range(len(self.df)):
            clusterdict_from_df_as_drug_frequency[i].append(self.df.iloc[i].to_dict()) #
        print("packs in dict form of drugs frequency of hc as drugs",clusterdict_from_df_as_drug_frequency)

        for i in range(len(self.df)):
            clusterdict_from_df_as_drug_non_O_frequency[i] = []

        for i in range(len(self.df)):
            for j in clusterdict_from_df_as_drug_frequency[i]:
                clusterdict_from_df_as_drug_non_O_frequency[i].append({x: y for x, y in j.items() if y != 0})
        print("clusterdict_from_df_as_drug_non_O_frequency of hc as drugs", clusterdict_from_df_as_drug_non_O_frequency)
        print('\n')
        clusterdict_from_as_drugs_only_as_list = {}
        clusterdict_of_non_repeated_drugs = {}
        for i in range(len(self.df)):
            clusterdict_from_as_drugs_only_as_list[i] = []

        for i in range(len(self.df)):
            for j in clusterdict_from_df_as_drug_non_O_frequency[i]:
                clusterdict_from_as_drugs_only_as_list[i].append(j.keys())

        print("clusterdict_from_df_as_drug_non_O_frequency with removed frequencies of hc as drugs", clusterdict_from_as_drugs_only_as_list)
        print('\n')


        for i in range(len(self.df)):
            clusterdict_of_non_repeated_drugs[i]=list(more_itertools.collapse([list(x) for x in set([tuple(x) for x in clusterdict_from_as_drugs_only_as_list[i]])]))


        print("clusterdict_from_df_as_drug_non_O_frequency with removed frequencies with more beautified results of hc as drugs", clusterdict_of_non_repeated_drugs)

        ################################################################################################################
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
        print("which pack which robot dict hc drugs",robot_for_packs_dict)

        ################################################################################################################
        self.packs_cluster_dict = {}
        for i in range(2):
            self.packs_cluster_dict[i] = []
        for key, value in robot_for_packs_dict.items():
            if value == [1]:
                self.packs_cluster_dict[1].append(key)
            elif value == [0]:
                self.packs_cluster_dict[0].append(key)
            else:
                self.packs_cluster_dict[0].append(key)
                self.packs_cluster_dict[1].append(key)

        print("packs_cluster_dict of hc drugs",self.packs_cluster_dict)
        self.hcd_number_of_packs_in_cluster1 = len(self.packs_cluster_dict[0])
        self.hcd_number_of_packs_in_cluster2 = len(self.packs_cluster_dict[1])
        self.test1()
        return [self.hcd_number_of_packs_in_cluster1, self.hcd_number_of_packs_in_cluster2]
        ################################################################################################################
    def test1(self):
        self.drugdict = {}
        for i in self.packs_cluster_dict:
            self.drugdict[i] = []
        self.drug = []
        for i in range(len(self.packs_cluster_dict.keys())):
            for j in range(len(self.packs_cluster_dict[i])):
                self.drugdict[i].append(self.df.iloc[self.packs_cluster_dict[i][j]].to_dict())
        print("drugs dict with their frequencies of hc drugs", self.drugdict)
        clusterdict_from_df_as_drug_non_O_frequency = {}
        clusterdict_from_as_drugs_only_as_list = {}
        clusterdict_of_non_repeated_drugs = {}
        for i in self.drugdict:
            clusterdict_from_df_as_drug_non_O_frequency[i] = []
        for i in self.drugdict:
            for j in self.drugdict[i]:
                clusterdict_from_df_as_drug_non_O_frequency[i].append({x: y for x, y in j.items() if y != 0})
        print("clusterdict_from_df_as_drug_non_O_frequency of hc drugs", clusterdict_from_df_as_drug_non_O_frequency)
        print('\n')
        ################################################################################################################
        t_loss = Pack_Time_Predictor()
        timelosslist = []
        timelosslist1 = []
        lenlist = []
        lenlist1 = []
        for packdict in clusterdict_from_df_as_drug_non_O_frequency[0]:
            timelosslist.append(packdict.values())
            lenlist.append(len(packdict.values()))
        for packdict in clusterdict_from_df_as_drug_non_O_frequency[1]:
            timelosslist1.append(packdict.values())
            lenlist1.append(len(packdict.values()))
        maxlen = np.max(lenlist)
        df = pd.DataFrame(timelosslist)
        print("df one of hc drugs before padding", df)
        maxdummydf = t_loss.load_dataset_from_database_type1()
        for i in range(maxlen + 1, maxdummydf + 1):
            df.insert(maxlen, i, 0)
        df = df.fillna(0)
        print("df one of hc drugs after padding", df)
        X = df[list(df.columns.values)]
        predictions = t_loss.predict_given_input(X)
        totaltimecluster1 = np.sum(list(predictions))
        ################################################################################################################
        maxlen1 = np.max(lenlist1)
        df1 = pd.DataFrame(timelosslist1)
        print("df two of hc drugs before padding", df1)
        maxdummydf1 = t_loss.load_dataset_from_database_type1()
        for i in range(maxlen1 + 1, maxdummydf1 + 1):
            df1.insert(maxlen1, i, 0)
        df1 = df1.fillna(0)
        print("df two of hc drugs after padding", df1)
        X = df1[list(df1.columns.values)]
        predictions1 = t_loss.predict_given_input(X)
        totaltimecluster2 = np.sum(list(predictions1))

        print("total time in cluster one of hc drugs", totaltimecluster1)
        print("total time in cluster two of hc drugs", totaltimecluster2)
        return [totaltimecluster1,totaltimecluster2]

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
class KMeans_Clusters_PacksAs_Samples():
    """
    This class will be used to make the drug clusters.
    """

    def __init__(self, file_path):
        self.data_mat = []
        self.file_path = file_path
        self.k = 2
        self.idx_list = []
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
        self.data_mat = np.array(self.df).astype(float)

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
        print("cluster dict of packs of kmeans as packs with two keys",self.indexdict)

        self.drugdict = {}
        for i in self.clusterdict:
            self.drugdict[i] = []
        self.drug = []
        for i in range(len(self.indexdict.keys())):
            for j in range(len(self.indexdict[i])):
                self.drugdict[i].append(self.df.iloc[self.indexdict[i][j]].to_dict())
        print("drugs dict with their frequencies of kmeans as packs", self.drugdict)
        self.clusterdict_from_df_as_drug_non_O_frequency = {}

        for i in self.drugdict:
            self.clusterdict_from_df_as_drug_non_O_frequency[i] = []
        for i in self.drugdict:
            for j in self.drugdict[i]:
                self.clusterdict_from_df_as_drug_non_O_frequency[i].append({x: y for x, y in j.items() if y != 0})
        print("clusterdict_from_df_as_drug_non_O_frequency of kmeans packs", self.clusterdict_from_df_as_drug_non_O_frequency)
        print('\n')


        ################################################################################################################
        t_loss = Pack_Time_Predictor()
        timelosslist = []
        timelosslist1 = []
        lenlist = []
        lenlist1 = []
        for packdict in self.clusterdict_from_df_as_drug_non_O_frequency[0]:
                timelosslist.append(packdict.values())
                lenlist.append(len(packdict.values()))
        for packdict in self.clusterdict_from_df_as_drug_non_O_frequency[1]:
                timelosslist1.append(packdict.values())
                lenlist1.append(len(packdict.values()))
        maxlen = np.max(lenlist)
        df = pd.DataFrame(timelosslist)
        print("df one of kmeans packs before padding", df)
        maxdummydf= t_loss.load_dataset_from_database_type1()
        for i in range(maxlen+1, maxdummydf+1):
            df.insert(maxlen,i,0)
        df = df.fillna(0)
        print("df one of kmeans packs after padding",df)
        X = df[list(df.columns.values)]
        predictions = t_loss.predict_given_input(X)
        totaltimecluster1= np.sum(list(predictions))
        print()
        ################################################################################################################
        maxlen1 = np.max(lenlist1)
        df1 = pd.DataFrame(timelosslist1)
        print("df two of kmeans packs before padding", df1)
        maxdummydf1 = t_loss.load_dataset_from_database_type1()
        for i in range(maxlen1 + 1, maxdummydf1 + 1):
            df1.insert(maxlen1, i, 0)
        df1 = df1.fillna(0)
        print("df two of kmeans packs after padding", df1)
        X = df1[list(df1.columns.values)]
        predictions1 = t_loss.predict_given_input(X)
        totaltimecluster2 = np.sum(list(predictions1))

        print("total time in cluster one of kmeans packs",totaltimecluster1)
        print("total time in cluster two of kmeans packs", totaltimecluster2)
        self.test()
        return [totaltimecluster1, totaltimecluster2]
    def test(self):
        clusterdict_from_as_drugs_only_as_list = {}
        clusterdict_of_non_repeated_drugs ={}
        for i in self.drugdict:
            clusterdict_from_as_drugs_only_as_list[i] = []


        for i in self.drugdict:
            for j in self.clusterdict_from_df_as_drug_non_O_frequency[i]:
                clusterdict_from_as_drugs_only_as_list[i].append(j.keys())

        print("only keys drugs with drugs name of kmeans packs", clusterdict_from_as_drugs_only_as_list)
        print('\n')

        for i in self.drugdict:
            clusterdict_of_non_repeated_drugs[i] = list(more_itertools.collapse([list(x) for x in set([tuple(x) for x in clusterdict_from_as_drugs_only_as_list[i]])]))

        for i in clusterdict_of_non_repeated_drugs:
            clusterdict_of_non_repeated_drugs[i] = set(clusterdict_of_non_repeated_drugs[i])
        for i in clusterdict_of_non_repeated_drugs:
            clusterdict_of_non_repeated_drugs[i] = list(more_itertools.collapse(clusterdict_of_non_repeated_drugs[i]))
        print("only drugs only of kmeans packs of non repeated drugs", clusterdict_of_non_repeated_drugs)
        ################################################################################################################
        try:
            common_drug_list = set([x for x in clusterdict_of_non_repeated_drugs[0] if
                                x in clusterdict_of_non_repeated_drugs[1]])
            print('\n')
            print("common drug list of kmeans packs", common_drug_list)
            total_frequency_of_drugs_dict = {}
            for i in self.drugdict:
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
            print("total drugs frequency of kmeans packs", total_frequency_of_drugs_dict)
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
                    temp_list3.append(dict(zip(i, j)))
            temp_list4 = temp_list3[:2]
            print('\n')
            for keys in self.drugdict:
                total_frequency_of_drugs_dict_with_drugs[keys].append(temp_list4[keys])
            print("total frequency with drugs dict of kmeans packs", total_frequency_of_drugs_dict_with_drugs)

            final_drugs_in_clusters_dict = {}
            for i in self.drugdict:
                final_drugs_in_clusters_dict[i] = []
            compare_list = []
            for drug in total_frequency_of_drugs_dict_with_drugs[0][0].keys():
                compare_list.append(min(total_frequency_of_drugs_dict_with_drugs[0][0][drug],total_frequency_of_drugs_dict_with_drugs[1][0][drug]))
            print("compare list of kmeans packs", compare_list)
            for values in total_frequency_of_drugs_dict_with_drugs.values():
                for key1, value1 in values[0].items():
                    if value1 in compare_list:
                        key2 = values[0].keys()[values[0].values().index(value1)]
                        values[0].pop(key2, None)

            print('final dict with deleted keys of kmeans packs', total_frequency_of_drugs_dict_with_drugs)

            clusterdict_from_as_drugs_only_as_list = {}
            clusterdict_of_non_repeated_drugs = {}

            for i in self.drugdict:
                clusterdict_from_as_drugs_only_as_list[i] = []

            for i in self.drugdict:
                for j in total_frequency_of_drugs_dict_with_drugs[i]:
                    clusterdict_from_as_drugs_only_as_list[i].append(j.keys())
            print("only keys drugs with drugs name of kmeans packs", clusterdict_from_as_drugs_only_as_list)
            print('\n')

            for i in self.drugdict:
                clusterdict_of_non_repeated_drugs[i] = list(more_itertools.collapse([list(x) for x in [tuple(x) for x in clusterdict_from_as_drugs_only_as_list[i]]]))
            print("only drugs of kmeans packs", clusterdict_of_non_repeated_drugs)

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
            print("winner drug dict of kmeans packs", winner_drug_dict)

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
                    loosercluster = int
                    winnercluster = int
                    final_robot_packs_dict[i].append(pack)
                    for commondrugs in winner_drug_dict:
                        for winnercluster in winner_drug_dict[commondrugs]:
                            if winnercluster == 0:
                                loosercluster = 1
                            if winnercluster == 1:
                                loosercluster = 0
                        if commondrugs in packdict_non_0_key and i == loosercluster:
                            try:
                                final_robot_packs_dict[i].remove(pack)
                                final_robot_packs_dict[winnercluster].append(pack)
                            except ValueError:
                                print('\t')

            for i in self.indexdict:
                final_robot_packs_dict[i] = set(final_robot_packs_dict[i])

            print("final which pack which robot dict of kmeans packs", final_robot_packs_dict)

            self.kmp_number_of_packs_in_cluster1 = len(self.indexdict[0])
            self.kmp_number_of_packs_in_cluster2 = len(self.indexdict[1])


        except IndexError:
            print("No common drugs of kmeans packs")
        return [self.kmp_number_of_packs_in_cluster1, self.kmp_number_of_packs_in_cluster2]

    def run_algorithm(self):
        self.read_excel_file()
        self.clustering_and_visulization()


class KMeans_Clusters_DrugsAs_Samples():
    """
    This class will be used to make the packs clusters.
    """

    def __init__(self, file_path):
        self.data_mat = []
        self.file_path = file_path
        self.k = 2
        self.idx_list = []
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
        self.data_mat = np.array(self.df).astype(float).transpose()

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

        index_dict = {}
        for i in self.clusterdict:
            index_dict[i] = []
        for i in range(len(self.data_mat)):
            for j in range(len(self.clusterdict)):
                if (self.clusterdict[j][:] == self.data_mat[i]).any():
                    index_dict[j].append(i)
        print("drugs cluster dict of kmeans drugs", index_dict)

        self.drugsdict = {}
        for i in index_dict:
            self.drugsdict[i] = []
        drugslist = list(self.df.columns.values)
        print("drugs list from dataframe of kmeans drugs", drugslist)

        for i in index_dict:
            self.drugsdict[i] = [drugslist[index] for index in index_dict[i]]

        print("drugs cluster dict of kmeans drugs", self.drugsdict)
        ################################################################################################################
        clusterdict_from_df_as_drug_frequency = {}
        clusterdict_from_df_as_drug_non_O_frequency = {}

        print('\n')

        for i in range(len(self.df)):
            clusterdict_from_df_as_drug_frequency[i] = []

        for i in range(len(self.df)):
            clusterdict_from_df_as_drug_frequency[i].append(self.df.iloc[i].to_dict())  #
        print("packs in dict form of drugs frequency of kmeans drugs", clusterdict_from_df_as_drug_frequency)

        for i in range(len(self.df)):
            clusterdict_from_df_as_drug_non_O_frequency[i] = []

        for i in range(len(self.df)):
            for j in clusterdict_from_df_as_drug_frequency[i]:
                clusterdict_from_df_as_drug_non_O_frequency[i].append({x: y for x, y in j.items() if y != 0})
        print("clusterdict_from_df_as_drug_non_O_frequency of kmeans drugs", clusterdict_from_df_as_drug_non_O_frequency)
        print('\n')
        clusterdict_from_as_drugs_only_as_list = {}
        clusterdict_of_non_repeated_drugs = {}
        for i in range(len(self.df)):
            clusterdict_from_as_drugs_only_as_list[i] = []

        for i in range(len(self.df)):
            for j in clusterdict_from_df_as_drug_non_O_frequency[i]:
                clusterdict_from_as_drugs_only_as_list[i].append(j.keys())

        print("only keys drugs with drugs name of kmeans drugs", clusterdict_from_as_drugs_only_as_list)
        print('\n')

        for i in range(len(self.df)):
            clusterdict_of_non_repeated_drugs[i] = list(more_itertools.collapse(
                [list(x) for x in set([tuple(x) for x in clusterdict_from_as_drugs_only_as_list[i]])]))

        print("only drugs only of kmeans drugs", clusterdict_of_non_repeated_drugs)

        ################################################################################################################
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
        print("clusterdict_of_non_repeated_drugs kmeans drugs", robot_for_packs_dict)


        ################################################################################################################
        self.packs_cluster_dict = {}
        for i in range(2):
            self.packs_cluster_dict[i] = []
        for key, value in robot_for_packs_dict.items():
            if value == [1]:
                self.packs_cluster_dict[1].append(key)
            elif value == [0]:
                self.packs_cluster_dict[0].append(key)
            else:
                self.packs_cluster_dict[0].append(key)
                self.packs_cluster_dict[1].append(key)

        print("packs_cluster_dict of kmeans drugs", self.packs_cluster_dict)
        self.kmd_number_of_packs_in_cluster1 = len(self.packs_cluster_dict[0])
        self.kmd_number_of_packs_in_cluster2 = len(self.packs_cluster_dict[1])
        self.test()
        return [self.kmd_number_of_packs_in_cluster1, self.kmd_number_of_packs_in_cluster2]
        ################################################################################################################
    def test(self):
        self.drugdict = {}
        for i in self.packs_cluster_dict:
            self.drugdict[i] = []
        self.drug = []
        for i in range(len(self.packs_cluster_dict.keys())):
            for j in range(len(self.packs_cluster_dict[i])):
                self.drugdict[i].append(self.df.iloc[self.packs_cluster_dict[i][j]].to_dict())
        print("drugs dict with their frequencies of kmeans drugs", self.drugdict)
        clusterdict_from_df_as_drug_non_O_frequency = {}
        for i in self.drugdict:
            clusterdict_from_df_as_drug_non_O_frequency[i] = []
        for i in self.drugdict:
            for j in self.drugdict[i]:
                clusterdict_from_df_as_drug_non_O_frequency[i].append({x: y for x, y in j.items() if y != 0})
        print("clusterdict_from_df_as_drug_non_O_frequency of kmeans drugs", clusterdict_from_df_as_drug_non_O_frequency)
        print('\n')
        ################################################################################################################
        self.timelossdict = {}

        for i in clusterdict_from_df_as_drug_non_O_frequency:
            self.timelossdict[i] = []
        t_loss = Pack_Time_Predictor()
        timelosslist = []
        timelosslist1 = []
        lenlist = []
        lenlist1 = []
        for packdict in clusterdict_from_df_as_drug_non_O_frequency[0]:
            timelosslist.append(packdict.values())
            lenlist.append(len(packdict.values()))
        for packdict in clusterdict_from_df_as_drug_non_O_frequency[1]:
            timelosslist1.append(packdict.values())
            lenlist1.append(len(packdict.values()))
        maxlen = np.max(lenlist)

        print("self.timelossdict", timelosslist)
        df = pd.DataFrame(timelosslist)
        print("df 1 of kmeans drugs", df)
        maxdummydf = t_loss.load_dataset_from_database_type1()
        for i in range(maxlen + 1, maxdummydf + 1):
            df.insert(maxlen, i, 0)
        df = df.fillna(0)
        print("df 1 of kmeans drugs", df)
        X = df[list(df.columns.values)]
        predictions = t_loss.predict_given_input(X)
        totaltimecluster1 = np.sum(list(predictions))
        print()
        ################################################################################################################
        maxlen1 = np.max(lenlist1)
        print("self.timelossdict", timelosslist1)
        df1 = pd.DataFrame(timelosslist1)
        print("df 2 of kmeans drugs", df1)
        maxdummydf1 = t_loss.load_dataset_from_database_type1()
        for i in range(maxlen1 + 1, maxdummydf1 + 1):
            df1.insert(maxlen1, i, 0)
        df1 = df1.fillna(0)
        print("df 2 of kmeans drugs", df1)
        X = df1[list(df1.columns.values)]
        predictions1 = t_loss.predict_given_input(X)
        totaltimecluster2 = np.sum(list(predictions1))

        print("total time in cluster one kmeans drugs", totaltimecluster1)
        print("total time in cluster two kmeans drugs", totaltimecluster2)
        return [totaltimecluster1,totaltimecluster2]

    def run_algorithm(self):
        self.read_excel_file()
        self.clustering_and_visulization()


class Lossfunction():
    def __init__(self):
       pass

    def stdeviation(self, clusterlist):


        self.deviation = np.std(clusterlist)

        return self.deviation

cwd = os.getcwd()
file_path = cwd + '/test3.xlsx'
print ("file_path", file_path)

mc = Masterclass(file_path)
mc.losscalculation()