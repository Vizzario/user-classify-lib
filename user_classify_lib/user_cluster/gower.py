import matplotlib
import numpy as np
from matplotlib import pyplot as plt


"""
    GOWER'S DISTANCE
"""

# get the range for all the numerical responses in a data survey
def get_range_in_labels(raw_data : dict, labels, set_key = None):
    label_range = dict()
    label_max = dict()
    label_min = dict()
    for user, value in raw_data.items():
        if set_key is None:
            response = value
        else:
            response = value[set_key]
        for label in labels:
            if label in response.keys():
                if response[label] is not str:
                    if label not in label_max.keys():
                        label_max[label] = response[label]
                        label_min[label] = response[label]
                    else:
                        if response[label] > label_max[label]: label_max[label] = response[label]
                        if response[label] < label_min[label]: label_min[label] = response[label]

    for label in label_max.keys():
        label_range[label] = label_max[label] - label_min[label]

    return label_range

# calculate the gower distance between two users
# note that the label_range is a dictionary that lists the range of all quantitative responses
# Missing Data Equidistant: True means same value (dist = 0 ), False means different value ( dist = 1), None means omit
def calculate_gower_distance(raw_data : dict, user1 : str, user2: str, labels: list, label_range: dict, set_key = None,
                             miss_data_equidistant : bool = None):

    if set_key is None:
        user1_set = raw_data[user1]
        user2_set = raw_data[user2]
    else:
        user1_set = raw_data[user1][set_key]
        user2_set = raw_data[user2][set_key]

    gower_val = list()
    gower_dist = 0
    for label in labels:
        if label in user1_set.keys() and label in user2_set.keys():
            if type(user1_set[label]) is not str:
                gower = np.abs(user1_set[label]-user2_set[label]) / label_range[label]
            else:
                if user1_set[label] == user2_set[label]:
                    gower = 0
                else:
                    gower = 1
            gower_dist = gower_dist + gower
            gower_val.append(gower)
        elif miss_data_equidistant is not None:
#            if label in user1_set.keys() or label in user2_set.keys():

            if miss_data_equidistant : gower = 0
            else : gower = 1

            gower_dist = gower_dist + gower
            gower_val.append(gower)
    if(len(gower_val)) == 0 : gower_dist_norm = None
    else : gower_dist_norm = gower_dist / len(gower_val)

    return gower_dist_norm, gower_val


def plot_comparison_table(raw_data : dict, user1 : str, user2: str, labels: list, set_key = None):

    if set_key is None:
        user1_set = raw_data[user1]
        user2_set = raw_data[user2]
    else:
        user1_set = raw_data[user1][set_key]
        user2_set = raw_data[user2][set_key]

    column_label = ["User 1", "User 2"]
    row_label = labels
    response_values = []
    for label in labels:
        val = [None, None]
        if label in user1_set.keys():
            val[0] = user1_set[label]
        if label in user2_set.keys():
            val[1] = user2_set[label]
        response_values.append(val)

    table = plt.table(cellText=response_values, rowLabels = row_label, colLabels = column_label, colWidths=[0.1, 0.1],
                      loc='center',
                      bbox=[0.35, 0.05, 0.55, 0.9])
    table.scale(2.5, 0.5)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10.5, 10.5)
    fig.tight_layout()
    plt.show()


# a brute-force method of finding the minimum, maximum, or all the distance between users
# using the gower distance as a metric.
def find_gower_distance_brute(raw_data: dict, user_list: list, labels: list, label_range: dict,
                              set_key = None, search:str = 'all', miss_data_equidistant: bool = None):
    rank_dist = list()
    rank_user = list()
    for i in range(len(user_list)):
        for j in range(i+1, len(user_list)):

            dist, val = calculate_gower_distance(raw_data, user_list[i], user_list[j], labels, label_range,
                                                 set_key, miss_data_equidistant=miss_data_equidistant)

            if len(rank_dist) == 0 :
                rank_dist.append(dist)
                rank_user.append([user_list[i], user_list[j]])
            else:
                if search == 'all':
                    for k in range(len(rank_dist)):
                        if rank_dist[k] > dist:
                            rank_dist.insert(k, dist)
                            rank_user.insert(k, [user_list[i], user_list[j]])
                            break
                elif search == 'min':
                    if dist < rank_dist[0]:
                        rank_dist[0] = dist
                        rank_user[0] = [user_list[i], user_list[j]]
                elif search == 'max':
                    if dist > rank_dist[0]:
                        rank_dist[0] = dist
                        rank_user[0] = [user_list[i], user_list[j]]


    return rank_dist, rank_user


def cluster_based_on_gower(raw_data:dict, user_list: list, user_centroid: list,
                           labels: list, label_range: dict,
                           set_key = None, miss_data_equidistant: bool = None):
    user_cluster = dict()
    for user in user_centroid : user_cluster[user] = dict()

    for user in user_list:
        closest_dist, closest_centroid = None, None
        for centroid in user_centroid:
            dist, val = calculate_gower_distance(raw_data, user, centroid, labels, label_range,
                                                 set_key, miss_data_equidistant=miss_data_equidistant)
            if closest_dist is None:
                closest_dist = dist
                closest_centroid = centroid
            else:
                if dist < closest_dist:
                    closest_dist = dist
                    closest_centroid = centroid
        user_cluster[closest_centroid][user] = raw_data[user]

    return user_cluster