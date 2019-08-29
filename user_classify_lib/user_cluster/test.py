from user_classify_lib import psy_variables as psy
from user_classify_lib.user_cluster.gower import find_gower_distance_brute, cluster_based_on_gower, \
    plot_comparison_table
from user_classify_lib.util import get_questions, get_user_ids, get_feature_counts

PSY_GROUPS = {
    'basic' : psy.basic_variables,
    'central' :psy.central,
    'color' :psy.color,
    'components' :psy.components,
    'concurrency' :psy.concurrency,
    'contrast' : psy.contrast,
    'divided' : psy.divided,
    'duration' :psy.duration,
    'essentricity' :psy.eccentricity,
    'focussed' :psy.focussed,
    'peripheral' : psy.peripheral,
    'size' :psy.size,
    'speed' :psy.speed,
    'subcomponents' :psy.sub_components
}


"""TESTING GOWER DISTANCE"""

def test_gower(raw_data: dict):
    conditions = get_questions(raw_data['fulcrum_study'], set_key='demographics', anchor_key='eyeConditions | glaucoma',
                               data_type='all', omitted_words=['mild', 'moderate', 'severe'])
    print(conditions)
    user_ids = get_user_ids(raw_data['fulcrum_study'])
    range = dict()
    for label in conditions: range[label] = 1
    range['age'] = 80
    user_list = list(user_ids)

    #gower_dist, gower_val = calculate_gower_distance(raw_data['fulcrum_study'], user_list[0], user_list[1], labels=conditions, label_range=range, set_key='demographics')
    #plot_comparison_table(raw_data['fulcrum_study'], user_list[0], user_list[1], labels=conditions, set_key='demographics')

    rank_dist, rank_user = find_gower_distance_brute(raw_data['fulcrum_study'], user_list,
                                                     labels=conditions, label_range=range,
                                                     set_key='demographics', search='max', miss_data_equidistant=True)



    print(rank_dist)

    print(rank_user[0])

    user_group = cluster_based_on_gower(raw_data['fulcrum_study'], user_list, rank_user[0],
                                        labels=conditions, label_range=range,
                                        set_key='demographics', miss_data_equidistant=None)

    get_feature_counts(user_group[rank_user[0][0]], set_key='demographics',
                       latex_output_path='user_cluster_1.tex', latex_title="Fulcrum Study Data: Cluster 1")

    get_feature_counts(user_group[rank_user[0][1]], set_key='demographics',
                       latex_output_path='user_cluster_2.tex', latex_title="Fulcrum Study Data: Cluster 2")

    plot_comparison_table(raw_data['fulcrum_study'], rank_user[0][0], rank_user[0][1],
                          labels=conditions, set_key='demographics')

    #print(minDist)
    #print(json.dumps(raw_data['fulcrum_study'][closest_user1]['demographics'], indent=4, sort_keys=True))
    #print(json.dumps(raw_data['fulcrum_study'][closest_user2]['demographics'], indent=4, sort_keys=True))