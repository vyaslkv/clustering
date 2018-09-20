import MySQLdb
import pandas as pd
import numpy as np

#TODO: changes needed in code
"""
1)
We need to make two different code for training/testing and prediction.
In every deep learning/ machine learning model we train our data on different code.
We save parametres of trained model.
At test time we use that parametres and predict the output.
We need to do that.

2)Functions are not clearly defined. Functions should be like this.
-load_data_from_database
-visualise_data
-Split_data_into_train_test.
-Train_model_using_splitted_data.
_Test_model_using_splitted_data.
_predict_given_input.
"""


# open a database connection
# be sure to change the host IP address, username, password and database name to match your own
class Timelossfunction():
    def __init__(self):
        pass

    def packtime(self):
        connection = MySQLdb.connect (host = "localhost", user = "root", passwd = "root", db = "lossfunction")

        # prepare a cursor object using cursor() method
        cursor = connection.cursor ()

        # execute the SQL query using execute() method.
        cursor.execute ("select start_time, end_time, pack_id from robot_data")

        # fetch all of the rows from the query
        data = cursor.fetchall ()
        # print the rows
        # templist = []
        # packidlist = []
        self.packdict = {}
        for row in data :
            # print row[0], row[1]
            # print("start time", row[0])
            # print("end time", row[1])
            # print("time difference", (row[1]-row[0]).seconds)
            scds = (row[1]-row[0]).seconds
            # templist.append(scds)
            # print("scds",scds)
            # packdict[i]=scds
        # print("templist", templist)
            self.packdict[int(row[2])] = scds/60

        #    packidlist.append(row[2])

        # for i in range(len(templist)):
        #     packdict[i] = {}
        # for i in range(len(templist)):
        #    packdict[i] = templist[i]/60

        print("pack dict",self.packdict)
        avgtime = np.average(self.packdict.values())
        print("avgtime",avgtime)
        # print("pack id list",self.packdict.keys())

        # close the cursor object
        cursor.close ()

        # close the connection
        connection.close ()

        # exit the program
        #sys.exit()

    def packdrugquantity(self):
        connection = MySQLdb.connect (host = "localhost", user = "root", passwd = "root", db = "dpwsqa")

        # prepare a cursor object using cursor() method
        cursor = connection.cursor ()

        # execute the SQL query using execute() method.
        cursor.execute ("SELECT `t2`.`drug_id_id`, `t3`.`id` AS pack_id, SUM(FLOOR(`t6`.`quantity`)) AS quantity \
        FROM `pack_rx_link` AS t1 \
        INNER JOIN `patient_rx` AS t2 ON (`t2`.`id` = `t1`.`patient_rx_id_id`) \
        INNER JOIN `pack_details` AS t3 ON (`t3`.`id` = `t1`.`pack_id_id`) \
        INNER JOIN `pack_header` AS t4 ON (`t4`.`id` = `t3`.`pack_header_id_id`) \
        INNER JOIN `drug_master` AS t5 ON (`t2`.`drug_id_id` = `t5`.`id`) \
        INNER JOIN `slot_details` AS t6 ON (`t6`.`pack_rx_id_id` = `t1`.`id`) \
        WHERE (((`t3`.`id` IN (2433, 2387, 2183, 2388, 2090, 2091, 2093, 2102, 2103, 2104, 2184, 2108, 2112, 2113, 2114, 2115, 2116, 2117, 2401, 2123, 2124, 2125, 2126, 2128, 2131, 2133, 2139, 2140, 2141, 2142, 2143, 2144, 2145, 98, 99, 100, 101, 2150, 2151, 2152, 105, 106, 107, 108, 109, 2158, 2159, 112, 113, 115, 116, 117, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 2182, 135, 136, 137, 2186, 2187, 140, 141, 142, 143, 145, 146, 147, 2196, 2197, 2198, 2199, 2200, 153, 154, 155, 156, 157, 158, 159, 2209, 162, 2212, 2215, 168, 170, 171, 173, 174, 176, 177, 2226, 179, 2229, 2230, 186, 187, 195, 196, 197, 200, 201, 202, 205, 206, 207, 2256, 2257, 2258, 2259, 2260, 2261, 2262, 2264, 2266, 2267, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 2464, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 2373, 2374, 2375, 329, 2381, 2382, 2383, 2384, 2385, 2386, 339, 340, 2389, 2390, 2391, 2392, 2393, 2394, 2396, 2397, 2398, 2399, 352, 353, 355, 356, 357, 358, 361, 2430, 2467, 367, 2417, 2418, 2419, 2420, 2421, 2422, 2423, 2424, 2425, 378, 379, 380, 381, 382, 383, 2432, 385, 386, 387, 388, 389, 390, 391, 392, 2441, 2362, 2446, 2456, 415, 416, 417, 418, 419, 420, 421, 422, 2474, 2475, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 2468, 475, 476, 477, 478, 479, 480, 481, 484, 485, 486, 487, 488, 489, 490, 2403, 2544, 2545, 2546, 499, 2548, 503, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 530, 2478, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 574, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 2148, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 630, 631, 633, 635, 636, 2154, 638, 639, 2155, 644, 645, 646, 647, 648, 649, 651, 2272, 2157, 2499, 2500, 2501, 2160, 2502, 2161, 2273, 687, 688, 2504, 692, 693, 695, 2506, 2507, 2508, 2479, 2511, 2512, 2513, 2173, 2515, 2174, 2516, 2175, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 2181, 800, 804, 805, 808, 809, 811, 812, 813, 814, 815, 817, 818, 819, 820, 822, 2185, 825, 826, 827, 828, 830, 831, 833, 834, 835, 836, 837, 838, 2188, 2189, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 874, 875, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 897, 899, 900, 901, 903, 904, 906, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 2547, 2206, 951, 955, 960, 961, 962, 964, 2146, 966, 968, 971, 974, 979, 983, 988, 990, 991, 992, 2147, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1084, 1085, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1096, 1097, 1098, 1099, 1101, 1102, 2219, 2176, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 2153, 1176, 1177, 2426, 1182, 1184, 1185, 1189, 2190, 2427, 1218, 1219, 1221, 1225, 2428, 1262, 2156, 2497, 2429, 2361, 2225, 1295, 1296, 1297, 1298, 1299, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1310, 1314, 2268, 2269, 2431, 2491, 2363, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1360, 1361, 1363, 1366, 1369, 1374, 1384, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1409, 1410, 2434, 2503, 2435, 2436, 2437, 2438, 2371, 2179, 2372, 2517, 2490, 2510, 2331, 2332, 2180, 1762, 1768, 2343, 1773, 2194, 1796, 1798, 2149, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834, 1835, 1836, 1837, 1838, 1839, 1840, 1841, 1842, 1843, 1846, 1849, 1856, 1857, 1858, 1859, 1860, 1866, 1869, 1870, 1871, 1872, 2360, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 2177, 1900, 2365, 1908, 1912, 1913, 1918, 2178, 2369, 2201, 2370, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2022, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2040, 2041, 2043)) )) \
        GROUP BY `t1`.`pack_id_id`, `t2`.`drug_id_id` \
        HAVING (SUM(FLOOR(`t6`.`quantity`)) > 0)")

        # fetch all of the rows from the query
        data = cursor.fetchall ()
        # print the rows
        # templist = []
        # packidlist = []
        self.newpackdict = {}
        for i in self.packdict:
            self.newpackdict[i] = []
        for row in data :
            # print row[0], row[1]
            # print("start time", row[0])
            # print("end time", row[1])
            # print("time difference", (row[1]-row[0]).seconds)

            # templist.append(scds)
            # print("scds",scds)
            # packdict[i]=scds
        # print("templist", templist)

            for key, value in self.packdict.items():
                if key == int(row[1]):
                    #self.newpackdict[int(row[1])].append({int(row[0]): int(row[2])})
                    self.newpackdict[int(row[1])].append(int(row[2]))
                    self.newpackdict[int(row[1])].append(value)
            #newpackdict[int(row[1])].append()

            # packidlist.append(row[2])
        # for i in self.newpackdict.keys():
        #      self.newpackdict[i].append(len(self.newpackdict[i])/2)

        # for i in range(len(templist)):
        #     packdict[i] = {}
        # for i in range(len(templist)):
        #    packdict[i] = templist[i]/60

        # print("pack dict",packdict)
        # avgtime = np.average(packdict.values())
        # print("avgtime",avgtime)
        # print("pack id list",packdict.keys())

        # close the cursor object
        cursor.close ()

        # close the connection
        connection.close ()

        # exit the program
        #sys.exit()
        print("new pack dict with drugs's pill count", self.newpackdict)

    def dict2dataframe(self):
        df = pd.DataFrame(self.newpackdict.values()).fillna(0)
        # df = df.replace(to_replace='None', value=np.nan)
        # df = df.replace(to_replace='NaN', value=0)
        # df = df.dropna(axis=1, how='all')
        # df.to_csv('sample.csv', sep='\t', encoding='utf-8')
        print("dataframe",df)
        import sklearn

        print("df.columns.values",list(df.columns.values))
        X_x = list(df.columns.values)[::2]
        X = df[X_x]
        print("X",X)
        Y = df[1]
        print("Y", Y)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
        from sklearn.linear_model import LinearRegression
        lm = LinearRegression()
        lm.fit(X_train, y_train)
        predictions = lm.predict(X_test)
        print("predictions", predictions)
        from sklearn import metrics

        print('MAE:', metrics.mean_absolute_error(y_test, predictions))
        print('MSE:', metrics.mean_squared_error(y_test, predictions))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


    def run_algorithm(self):
         self.packtime()
         self.packdrugquantity()
         self.dict2dataframe()

t = Timelossfunction()
t.run_algorithm()