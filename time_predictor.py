
import MySQLdb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class Pack_Time_Predictor:
    """
    Purpose: We want to predict time taken by any robot in feeling a pack. This class will assist us for the same.
    This class does following task.
    1) Loads dataset in needed format from database.
       -We have different types of methos for the same task which pertains to different types of feature vector.
       -Every type loads data from our database in different kind of feature vector.

    2) Visualise dataset.

    3) Split dataset in train test batch.

    4) Train Model.

    5) Validate Model.

    6) load model.

    7) Save model.

    8) Predict given input.
    """

    def __init__(self):
        self.packdict = {}
        self.df = None
        self.X = None
        self.Y = None



    def load_dataset_from_database_type0(self):
        """
        In this methood, we will extract needed feature vector from database.
        :return:
        """


    def load_dataset_from_database_type1(self):
        """
        In this methood, we will extract needed feature vector from database.
        :return:
        """
        connection = MySQLdb.connect(host="localhost", user="root", passwd="root", db="lossfunction")
        cursor = connection.cursor()
        cursor.execute("select start_time, end_time, pack_id from robot_data where start_time != end_time")
        data = cursor.fetchall()
        templist = []
        for row in data:
            scds = (row[1] - row[0]).seconds
            templist.append(int(row[2]))
            self.packdict[int(row[2])] = scds
        avgtime = np.average(self.packdict.values())
        print("avgtime", avgtime)
        cursor.close()
        connection.close()
        connection = MySQLdb.connect(host="localhost", user="root", passwd="root", db="dpwsqa")
        cursor = connection.cursor()
        cursor.execute("SELECT `t2`.`drug_id_id`, `t3`.`id` AS pack_id, SUM(FLOOR(`t6`.`quantity`)) AS quantity \
                        FROM `pack_rx_link` AS t1 \
                        INNER JOIN `patient_rx` AS t2 ON (`t2`.`id` = `t1`.`patient_rx_id_id`) \
                        INNER JOIN `pack_details` AS t3 ON (`t3`.`id` = `t1`.`pack_id_id`) \
                        INNER JOIN `pack_header` AS t4 ON (`t4`.`id` = `t3`.`pack_header_id_id`) \
                        INNER JOIN `drug_master` AS t5 ON (`t2`.`drug_id_id` = `t5`.`id`) \
                        INNER JOIN `slot_details` AS t6 ON (`t6`.`pack_rx_id_id` = `t1`.`id`) \
                        WHERE (((`t3`.`id` IN (98, 99, 98, 99, 98, 99, 100, 101, 106, 105, 107, 108, 108, 109, 109, 112, 113, 115, 116, 117, 120, 120, 123, 124, 125, 128, 126, 130, 130, 131, 132, 132, 133, 141, 141, 140, 140, 145, 142, 143, 146, 147, 136, 137, 153, 153, 154, 154, 155, 155, 157, 157, 159, 158, 158, 162, 156, 170, 171, 174, 176, 176, 177, 177, 173, 173, 179, 179, 186, 187, 195, 196, 197, 200, 201, 202, 205, 206, 207, 230, 235, 237, 238, 239, 220, 222, 227, 232, 236, 229, 231, 233, 221, 224, 234, 242, 243, 244, 246, 223, 245, 247, 248, 249, 250, 251, 252, 251, 253, 226, 228, 254, 228, 254, 228, 254, 228, 262, 261, 263, 264, 265, 267, 269, 268, 266, 281, 272, 271, 282, 283, 270, 289, 285, 290, 292, 293, 273, 286, 287, 288, 295, 291, 296, 294, 299, 297, 300, 298, 301, 311, 302, 303, 312, 304, 305, 306, 316, 317, 314, 320, 322, 324, 307, 308, 309, 310, 313, 315, 318, 339, 340, 355, 357, 361, 356, 378, 379, 380, 381, 382, 383, 385, 386, 389, 387, 392, 390, 391, 388, 416, 417, 418, 419, 429, 421, 420, 422, 430, 432, 431, 433, 434, 435, 440, 439, 441, 438, 443, 442, 445, 447, 446, 446, 450, 450, 449, 449, 448, 451, 452, 453, 466, 456, 467, 459, 469, 455, 470, 460, 465, 454, 461, 457, 462, 464, 463, 484, 477, 485, 487, 486, 488, 478, 480, 490, 481, 475, 506, 516, 518, 525, 527, 509, 507, 508, 521, 517, 514, 519, 511, 520, 522, 523, 513, 510, 512, 515, 524, 526, 530, 535, 534, 539, 536, 537, 538, 540, 542, 541, 543, 546, 545, 544, 548, 547, 549, 550, 551, 552, 553, 557, 563, 560, 568, 561, 562, 559, 558, 564, 565, 566, 571, 570, 569, 567, 572, 556, 574, 582, 584, 583, 587, 588, 589, 591, 593, 592, 590, 596, 594, 598, 595, 600, 599, 585, 586, 597, 625, 616, 617, 626, 609, 605, 610, 608, 611, 607, 606, 612, 613, 614, 615, 618, 619, 620, 622, 623, 621, 624, 630, 635, 636, 638, 633, 645, 648, 646, 647, 646, 646, 651, 687, 693, 695, 695, 770, 765, 764, 768, 765, 764, 768, 766, 764, 768, 766, 768, 768, 766, 767, 769, 772, 767, 769, 772, 774, 777, 771, 775, 776, 773, 779, 778, 780, 781, 781, 795, 787, 787, 789, 789, 797, 796, 787, 787, 789, 789, 797, 790, 790, 785, 788, 788, 791, 791, 793, 793, 792, 792, 794, 794, 786, 786, 809, 809, 811, 811, 808, 808, 836, 836, 834, 837, 838, 800, 804, 817, 820, 820, 814, 813, 818, 812, 820, 820, 813, 819, 822, 818, 825, 830, 826, 831, 827, 827, 835, 835, 828, 833, 833, 853, 855, 854, 856, 860, 857, 858, 861, 862, 865, 859, 878, 863, 871, 881, 883, 869, 870, 864, 892, 880, 885, 868, 868, 877, 877, 882, 882, 887, 887, 890, 890, 884, 867, 893, 893, 874, 874, 888, 889, 889, 894, 895, 895, 891, 872, 875, 879, 866, 886, 912, 899, 911, 913, 900, 913, 900, 903, 914, 916, 915, 917, 918, 919, 920, 897, 901, 904, 961, 951, 964, 962, 955, 960, 971, 974, 983, 998, 988, 1001, 1000, 999, 1002, 1003, 991, 966, 1004, 968, 966, 1004, 968, 1005, 1026, 1019, 1028, 1030, 1030, 1030, 1032, 1030, 1030, 1019, 1030, 1030, 1032, 1031, 1019, 1029, 1020, 1021, 1034, 1033, 1022, 1035, 1037, 1023, 1036, 1039, 1027, 1024, 1025, 1038, 1040, 1049, 1049, 1048, 1053, 1052, 1060, 1053, 1053, 1052, 1060, 1062, 1053, 1053, 1052, 1060, 1052, 1060, 1052, 1060, 1062, 1063, 1062, 1063, 1063, 1050, 1051, 1054, 1055, 1058, 1057, 1056, 1059, 1061, 1064, 1065, 1066, 1068, 1067, 1071, 1069, 1069, 1070, 1072, 1084, 1074, 1087, 1089, 1090, 1091, 1088, 1076, 1092, 1092, 1076, 1094, 1093, 1097, 1096, 1098, 1075, 1073, 1149, 1160, 1153, 1159, 1154, 1161, 1151, 1150, 1155, 1152, 1158, 1156, 1157, 1168, 1163, 1162, 1164, 1165, 1167, 1169, 1170, 1166, 1171, 1172, 1172, 1176, 1182, 1173, 1177, 1184, 1189, 1185, 1101, 1102, 1101, 1102, 1085, 1099, 1221, 1219, 1218, 1299, 1299, 1298, 1297, 1300, 1295, 1302, 1303, 1305, 1304, 1307, 1307, 1306, 1306, 1310, 1310, 1301, 1307, 1307, 1306, 1306, 1301, 1307, 1307, 1306, 1306, 1307, 1307, 1296, 1347, 1348, 1342, 1342, 1348, 1349, 1344, 1345, 1344, 1346, 1346, 1343, 1350, 1366, 1351, 1374, 1369, 1354, 1354, 1356, 1356, 1352, 1352, 1353, 1353, 1355, 1355, 1361, 1361, 1363, 1363, 1355, 1355, 1355, 1355, 1361, 1361, 1399, 1391, 1399, 1391, 1398, 1401, 1389, 1400, 1401, 1400, 1403, 1402, 1403, 1402, 1405, 1407, 1406, 1407, 1389, 1406, 1408, 1404, 1406, 1406, 1388, 1408, 1404, 1410, 1393, 1390, 1392, 1394, 1395, 1396, 1397, 1409, 1396, 1397, 1396, 1397, 1822, 1832, 1828, 1822, 1835, 1828, 1834, 1823, 1833, 1826, 1836, 1827, 1824, 1838, 1825, 1837, 1825, 1837, 1829, 1830, 1831, 1840, 1841, 1843, 1842, 1839, 1846, 1856, 1857, 1859, 1858, 1849, 1860, 1768, 1768, 1798, 1762, 1773, 1773, 1768, 1773, 1773, 1796, 1878, 1870, 1908, 1879, 1879, 1888, 1888, 1908, 1879, 1879, 1888, 1888, 1880, 1880, 1879, 1879, 1888, 1888, 1880, 1880, 1884, 1884, 1886, 1880, 1880, 1884, 1884, 1886, 1883, 1912, 1889, 1889, 1890, 1890, 1882, 1892, 1881, 1885, 1885, 1887, 1887, 1891, 1900, 1900, 1936, 1936, 1935, 1935, 1937, 1935, 1937, 1939, 1937, 1939, 1934, 1939, 1934, 1939, 1934, 1940, 1938, 1913, 1869, 1872, 1913, 1869, 1872, 1913, 1872, 1871, 1913, 1872, 1871, 1877, 1946, 1955, 1945, 1954, 1947, 1957, 1944, 1958, 1944, 1958, 1956, 1948, 1944, 1958, 1944, 1958, 1948, 1950, 1956, 1949, 1952, 1953, 1951, 1965, 1959, 1960, 1961, 1962, 1964, 1961, 1962, 1964, 1963, 1964, 1963, 1966, 1968, 1974, 1969, 1976, 1970, 1973, 1970, 1976, 1973, 1970, 1976, 1973, 1971, 1977, 1975, 1978, 1980, 1979, 1981, 1866, 1866, 1967, 1967, 1984, 1992, 1982, 1983, 1995, 1986, 1985, 1988, 1990, 1987, 1989, 1991, 1993, 1994, 1996, 1997, 1998, 2000, 2002, 1999, 2003, 2005, 2007, 2005, 2007, 2006, 2004, 2009, 2008, 2011, 2012, 2010, 2013, 2013, 2022, 2035, 2031, 2032, 2028, 2036, 2033, 2029, 2034, 2030, 2037, 2038, 2041, 2040, 2104, 2102, 2104, 2102, 2090, 2090, 2093, 2093, 2108, 2108, 2103, 2103, 1989, 1989, 1992, 1995, 1993, 1994, 1996, 1991, 1997, 1998, 2000, 2113, 2114, 2115, 2116, 2117, 2126, 2123, 2125, 2124, 2131, 2133, 2140, 2150, 2141, 2140, 2150, 2141, 2149, 2139, 2152, 2143, 2151, 2142, 2154, 2144, 2145, 2144, 2144, 2145, 2148, 2146, 2147, 2160, 2153, 2155, 2156, 2158, 2157, 2159, 2161, 2176, 2183, 2174, 2178, 2175, 2173, 2177, 2180, 2184, 2181, 2179, 2186, 2185, 2182, 2194, 2187, 2188, 2189, 2200, 2256, 2206, 2257, 2212, 2258, 2219, 2259, 2196, 2201, 2197, 2198, 2199, 2209, 2225, 2225, 2201, 2225, 2201, 2226, 2229, 2209, 2230, 2262, 2264, 2266, 2267, 2272, 2273, 2332, 2343, 2331, 2360, 2365, 2363, 2362, 2361, 2369, 2371, 2370, 2382, 2383, 2384, 2385, 2374, 2372, 2373, 2375, 2381, 2393, 2386, 2392, 2387, 2390, 2388, 2396, 2391, 2389, 2399, 2401, 2397, 2394, 2398, 2456, 2420, 2417, 2418, 2464, 2419, 2423, 2468, 2424, 2422, 2467, 2421, 2426, 2425, 2428, 2427, 2432, 2430, 2431, 2433, 2429, 2437, 2434, 2436, 2435, 2438, 2490, 2479, 2475, 2491, 2475, 2497, 2475, 2499, 2474, 2497, 2475, 2474, 2499, 2260, 2260, 2261, 2504, 2507, 2506, 2516, 2515, 2517, 2512, 2513, 2510, 2511, 2508, 2501, 2500, 2500, 2503, 2502, 2544, 2547, 2545, 2546, 2548)) )) \
                        GROUP BY `t1`.`pack_id_id`, `t2`.`drug_id_id` \
                        HAVING (SUM(FLOOR(`t6`.`quantity`)) > 0)")
        data = cursor.fetchall()
        self.newpackdict = {}
        for i in self.packdict:
            self.newpackdict[i] = []
        for row in data:
            for key, value in self.packdict.items():
                if key == int(row[1]):
                    self.newpackdict[int(row[1])].append(int(row[2]))
                    self.newpackdict[int(row[1])].append(value)
        cursor.close()
        connection.close()
        self.df = pd.DataFrame(self.newpackdict.values()).fillna(0)
        self.X_x = list(self.df.columns.values)[::2]

        maxlenfordummy = len(self.X_x)
        return maxlenfordummy

    def visualise_dataset(self):
        """
        Visualisation of dataset.
        :return:
        """
        pass
    def split_dataset(self):
        """

        :return:
        """
        scaler = MinMaxScaler()
        self.X = self.df[self.X_x]
        self.Y = self.df[1]
        from faker import Factory
        fake = Factory.create()
        colors = []
        for i in range(len(list(self.X.columns.values))):
            colors.append(fake.hex_color())

        for i, color in zip(list(self.X.columns.values),colors):
            plt.scatter(x=self.X[i],y= self.Y, c=color)
        plt.ylim(self.Y.min(), self.Y.max())
        plt.show()
        print("self.Y.max()",self.Y.max())
        for i in range(len(self.X)):
            np.sum(self.X.iloc[i])
        colors1 =[]
        for i in range(len(self.X)):
            colors1.append(fake.hex_color())
        for i, color in zip(range(len(self.X)),colors1):
            plt.scatter(x=np.average(self.X.iloc[i]),y= self.Y.iloc[i], c=color)
        plt.show()

        self.X1 = self.X.astype(bool).astype(int)
        self.X1.to_csv('testtest.csv')
        self.Y.to_csv('timetaken.csv')
        for i in range(len(self.X1)):
            colors1.append(fake.hex_color())
        for i, color in zip(range(len(self.X1)),colors1):
            plt.scatter(x=np.sum(self.X1.iloc[i]),y= self.Y.iloc[i], c=color)
        plt.show()


        self.X = scaler.fit_transform(self.df[self.X_x])
        self.Y = self.df[1]


        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.5, random_state=101)

    def train_model(self):
        """

        :return:
        """


        self.lm = LinearRegression()

        self.lm.fit(self.X_train, self.y_train)
        self.save_mdoel()



    def save_mdoel(self):
        """

        :return:
        """
        lossmodel = 'loss_model.sav'
        pickle.dump(self.lm, open(lossmodel, 'wb'))


    def load_model(self):
        """

        :return:
        """

        self.lm = joblib.load('loss_model.sav')
        print("loaded model", self.lm)

    def validate_model(self):
        """

        :return:
        """

        predictions = self.lm.predict(self.X_test)
        plt.scatter(self.y_test, predictions)
        plt.xlabel('Y Test')
        plt.ylabel('Predicted Y')
        plt.show()
        print('MAE:', metrics.mean_absolute_error(self.y_test, predictions))
        print('MSE:', metrics.mean_squared_error(self.y_test, predictions))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(self.y_test, predictions)))

    def predict_given_input(self, Xtest):
        """

        :param Xtest: values of which we are gonna predict
        :return:
        """
        self.load_model()
        predictions = self.lm.predict(Xtest)
        return predictions


    def run_algorithm(self):
        self.load_dataset_from_database_type0()
        self.load_dataset_from_database_type1()
        self.split_dataset()
        self.train_model()
        self.save_mdoel()
        self.load_model()
        self.validate_model()

p = Pack_Time_Predictor()
p.run_algorithm()