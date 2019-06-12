import os
import json
from typing import List, Any

import numpy as np
#import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import chisquare
import pickle

#import user_classify_lib.corex as ce

import user_classify_lib.corextopic.corextopic as cet
#import user_classify_lib.corextopic.vis_topic as vis

import user_classify_lib.biocorex as ce
import user_classify_lib.vis_corex as vis

import user_classify_lib.psy_variables as psy
import user_classify_lib.demographic_variables as dem
import user_classify_lib.fulcrum_test_functions as test

from user_classify_lib.output import LatexChartGenerator

#get a list of all the questions from the survey set. Use data type to filter for a type of question (binomial,
# multivariate, continuous etc.) based on the data type or set it to all to get all the questions.
def get_questions(raw_data: dict, set_key : str = None,
                  anchor_key : str = None, anchor_set : str = None,
                  data_type : str = 'all', omitted_words : list = None):

    questions = list()
    omit = False

    if anchor_set is None:
        anchor_set = set_key

    for user, value in raw_data.items():
        if set_key is None:
            response = value
        else:
            response = value[set_key]
        if anchor_key is None or \
                (anchor_set is None and anchor_key in raw_data[user].keys()) or \
                (anchor_set is not None and anchor_key in raw_data[user][anchor_set].keys()):

            for question, answer in response.items():

                if data_type == 'all' or \
                        (type(answer) is bool and data_type == 'bool') or \
                        (type(answer) is str and data_type == 'str') or \
                        (type(answer) is int and data_type == 'int') or \
                        (type(answer) is float and data_type == 'float'):
                    if question not in questions:
                        if omitted_words is not None:
                            for word in omitted_words:
                                if word in question:
                                    omit = True
                        if not omit:
                            questions.append(question)
                        else:
                            omit = False

    return questions

#get the user IDs
# (having a list of them reduces randomness when getting the user responses as sample data)
def get_user_ids(raw_data: dict):
    return raw_data.keys()


def get_sample_response_to_cont_value(answer: str, responses: list, values: list):
    for i, response in enumerate(responses):
        if answer == response:
            return values[i]
    return -1


def get_sample_response_to_binary_value(all_questions: dict, sample_question: str, possible_answers: list,
                                        all_sample_responses: list):
    for answer in possible_answers:
        if sample_question not in all_questions.keys():
            # all_sample_responses.append(-1)
            all_sample_responses.append(0)
        elif all_questions[sample_question] == answer:
            all_sample_responses.append(1)
        else:
            all_sample_responses.append(0)


def get_samples_for_input(raw_data: dict, user_id: list, labels: list, set_key: str = None,
                          anchor_key : str = None, #must have this variable to be part of the samples
                          anchor_set : str = None): #in case the anchor is in a different dataset
    input_samples = None
    if anchor_set is None:
        anchor_set = set_key

    for user in user_id:
        if set_key is None:
            response = raw_data[user]
        else:
            response = raw_data[user][set_key]
        if anchor_key is None or \
                (anchor_set is None and anchor_key in raw_data[user].keys()) or \
                (anchor_set is not None and anchor_key in raw_data[user][anchor_set].keys()):
            answers = list()
            for label in labels:
                if label in response.keys():
                    if response[label] is bool:
                        answers.append(1) if response[label] else answers.append(0)
                    else:
                        answers.append(response[label])
                else:
                    # answers_demo.append(-1)
                    answers.append(0)

            if answers is not None:
                # print(value['demographics']['visionPrescription'])
                # get_sample_response_to_binary_value(value['demographics'], 'visionPrescription',
                #                                    dem.vision_prescription, answers)

                answers = np.asarray(answers)
                # print(answers)

                input_samples = answers if input_samples is None else np.vstack((input_samples, answers))
            '''
            response = value['normalized']
            answers_psych = list()
            for label in labels_psych:
                if label in response.keys():
                    answers_psych.append(response[label])
                    # answers.append(response[label])
                else:
                    answers_psych.append(-1)
                    # answers.append(0)
                    # continue
            if answers_psych is not None:
                # add age
                answers_psych.append(
                    float(value['demographics']['age'] + 50))  # add 50 to age for range scaling

                # add 'hours_screen_per_day'
                answers_psych.append(
                    get_sample_response_to_cont_value(value['demographics']['hours_screen_per_day'],
                                                      dem.hours_screen_per_day_responses,
                                                      dem.hours_screen_per_day_values))

                # add 'hours_activity_per_week'
                answers_psych.append(
                    get_sample_response_to_cont_value(value['demographics']['hours_activity_per_week'],
                                                      dem.hours_activity_per_week_responses,
                                                      dem.hours_activity_per_week_values))

                answers_psych = np.asarray(answers_psych)
                # print(answers)
                input_psy = answers_psych if input_psy is None else np.vstack((input_psy, answers_psych))
                '''
    #print(input_samples.shape)

    return input_samples

#A supplement function to add numerical values to categorical responses for samples
def add_response_to_cont_value_samples(samples: np.ndarray, raw_data: dict, user_id: list, label: str,
                                       label_responses: list = None, label_values: list = None, set_key: str = None,
                                       anchor_key : str = None, anchor_set : str = None):

    new_samples = None
    i = 0 #counter for entries with anchors
    if anchor_set is None:
        anchor_set = set_key

    for user in user_id:
        if set_key is None:
            response = raw_data[user]
        else:
            response = raw_data[user][set_key]

        if anchor_key is None or \
                (anchor_set is None and anchor_key in raw_data[user].keys()) or \
                (anchor_set is not None and anchor_key in raw_data[user][anchor_set].keys()):
            answers = samples[i,:].tolist()


            if label in response.keys():
                if(label_responses is not None):
                    answers.append(get_sample_response_to_cont_value(response[label],
                                                      label_responses,
                                                      label_values))
                else:
                    answers.append(response[label])
            else:
                # answers_demo.append(-1)
                answers.append(0)
            answers = np.asarray(answers)
            new_samples = answers if new_samples is None else np.vstack((new_samples, answers))
            i = i + 1

    return new_samples


#A supplement function that converts the possible answers to a question into their own questions with binary responses
# For example: changing the "occupation" question and response to a set of questions for each type of occupation
# and a binary "yes/no" response based on whether they are of that occupation or not
def add_response_to_binary_value_samples(samples: np.ndarray, raw_data: dict, user_id: list, label: str,
                                       label_responses: list,  set_key: str = None,
                                         anchor_key : str = None, anchor_set : str = None):

    new_samples = None
    i = 0 #counter for entries with anchor
    if anchor_set is None:
        anchor_set = set_key

    for user in user_id:
        if set_key is None:
            response = raw_data[user]
        else:
            response = raw_data[user][set_key]

        if anchor_key is None or \
                (anchor_set is None and anchor_key in raw_data[user].keys()) or \
                (anchor_set is not None and anchor_key in raw_data[user][anchor_set].keys()):

            answers = samples[i,:].tolist()
            get_sample_response_to_binary_value(response, label,
                                                label_responses, answers)
            answers = np.asarray(answers)
            new_samples = answers if new_samples is None else np.vstack((new_samples, answers))
            i = i+1

    return new_samples

def test_ce():
    X = np.array([[0, 0, 0, 0, 0],  # A matrix with rows as samples and columns as variables.
                  [0, 0, 0, 1, 1],
                  [1, 1, 1, 0, 0],
                  [1, 1, 1, 1, 1]], dtype=int)

    #layer1 = ce.Corex(n_hidden=2)  # Define the number of hidden factors to use.
    layer1 = ce.Corex(n_hidden=2, dim_hidden=2, marginal_description='discrete', smooth_marginals=False, n_repeat=10)  #for biocorex
    layer1.fit(X)

    print(layer1.clusters)  # Each variable/column is associated with one Y_j
    # array([0, 0, 0, 1, 1])
    print(layer1.labels[0])  # Labels for each sample for Y_0
    # array([0, 0, 1, 1])
    print(layer1.labels[1])  # Labels for each sample for Y_1
    # array([0, 1, 0, 1])
    print(layer1.tcs)  # TC(X;Y_j) (all info measures reported in nats).


def test_ce_topic():
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer
    import scipy.sparse as ss

    # Get 20 newsgroups data
    newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

    # Transform 20 newsgroup data into a sparse matrix
    vectorizer = CountVectorizer(stop_words='english', max_features=20000, binary=True)
    doc_word = vectorizer.fit_transform(newsgroups.data)
    doc_word = ss.csr_matrix(doc_word)

    # Get words that label the columns (needed to extract readable topics and make anchoring easier)
    words = list(np.asarray(vectorizer.get_feature_names()))

    not_digit_inds = [ind for ind, word in enumerate(words) if not word.isdigit()]
    doc_word = doc_word[:, not_digit_inds]
    words = [word for ind, word in enumerate(words) if not word.isdigit()]

    #reduce the number of words interested for simplicity and testing
    doc_word = doc_word[:, 10000:10100].toarray()
    words = words[10000:10100]

    #### CorEx Models testing portion

    n_clusters=50

    for run in range(5):
        print("RUN NUMBER {0}".format(run))
        print("===================================================================")

        #Using CorEx Topic
        #layer1 = ce.Corex(n_hidden=n_clusters)

        #Using BioCorEx
        layer1 = ce.Corex(n_hidden=n_clusters, marginal_description='discrete', smooth_marginals=False)
        layer1.tc_min = 0

        Y1 = layer1.fit(doc_word)

        c_index = layer1.clusters

        clusters = [[] for i in range(n_clusters)]

        #print(c_index)

        [clusters[c_index[i]].append(label) for i, label in enumerate(words)]

        for j in range(n_clusters):
            print("\nCluster {0}:".format(j))
            [print(clusters[j][k]) for k in range(len(clusters[j]))]


def run_corex(input: np.ndarray, conditions:list, n_clusters:int = 10, title:str=None, topic=False):

    # linearcorex
    # layer1 = ce.Corex(n_hidden=n_clusters, gaussianize='None', missing_values=-1, verbose=True)

    if(topic):
        layer1 = cet.Corex(n_hidden=n_clusters, verbose=False)

    # biocorex
    else:
        layer1 = ce.Corex(n_hidden=n_clusters, marginal_description='gaussian', smooth_marginals=True, missing_values=-1,
                      verbose=False, n_repeat=1)
    #layer1.tc_min = 0

    # original
    # layer1 = ce.Corex(n_hidden=n_clusters, verbose=True, count='fraction')

    Y1 = layer1.fit(input)

    c_index = layer1.clusters

    clusters = [[] for i in range(n_clusters)]

    print(c_index)

    [clusters[c_index[i]].append(label) for i, label in enumerate(conditions)]

    for j in range(n_clusters):
        print("\nCluster {0}:".format(j))
        [print(clusters[j][k]) for k in range(len(clusters[j]))]

    print("\nCLUSTER SAMPLE SIZE: ")
    print(layer1.labels.shape)
    print("\nTC FOR EACH CLUSTER:")
    print(layer1.tcs)
    print("\nTOTAL TC FOR THIS COREX RUN:")
    print(layer1.tc)

    #vis.vis_rep(layer1, data=input, column_label=conditions, prefix=title)

    # print(layer1.alpha)
    # np.savetxt('alpha.txt', layer1.alpha[:,:,0].transpose(), fmt='%.3e')

    return layer1, clusters


def run_fulcrum_data(raw_data:dict):


    #conditions = get_conditions_all(raw_data['fulcrum_study'])
    conditions = test.get_conditions_no_scales(raw_data['fulcrum_study'])

    input = test.get_demo_samples(raw_data['fulcrum_study'], conditions)
    #input = get_all_samples_w_scale(raw_data['fulcrum_study'], conditions)

    #also consider age.
    conditions.append('age')

    '''
    conditions.append("age | A")
    conditions.append("age | B")
    conditions.append("age | C")
    conditions.append("age | D")
    conditions.append("age | E")

    
    conditions.append("age | young")
    conditions.append("age | medium")
    conditions.append("age | old")
    '''

    #np.savetxt('conditions.txt', conditions)
    n_clusters = 50 # Define the number of hidden factors to use.

    run_corex(input, conditions, n_clusters, 'fulcrum_demo_w_age_biocorex2')


def run_psychometric_data(raw_data:dict):

    psy_all = True

    if(psy_all):

        psychometrics = test.get_psychometrics_all(raw_data['fulcrum_study'])
        input = test.get_psychometric_samples(raw_data['fulcrum_study'], psychometrics)

        #add age to psychometrics labels
        psychometrics.append('age')

        n_clusters = 50  # Define the number of hidden factors to use.
        run_corex(input, psychometrics, n_clusters, 'fulcrum_psychometrics_all_w_age')

    else:
        psychometrics = {
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

        #psyKey = 'subcomponents'
        for psyKey in psychometrics:
            print("==================================")
            print("=================" + psyKey + "=================")
            print("==================================")
            print(psychometrics[psyKey])
            input = test.get_psychometric_samples(raw_data['fulcrum_study'], psychometrics[psyKey])
            # input = get_all_samples_w_scale(raw_data['fulcrum_study'], conditions)

            # np.savetxt('conditions.txt', conditions)
            n_clusters = 10  # Define the number of hidden factors to use.
            run_corex(input, psychometrics[psyKey], n_clusters, 'fulcrum_psychometrics_'+psyKey)

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

def run_demo_psych_combined(raw_data: dict, psy_group_key : str = None):

    #conditions = test.get_conditions_no_scales(raw_data['fulcrum_study'])
    conditions = get_questions(raw_data['fulcrum_study'], set_key='demographics', anchor_key='eyeConditions | glaucoma',
                               data_type='bool', omitted_words=['mild', 'moderate', 'severe'])
    if psy_group_key is None:
        psychometrics = get_questions(raw_data['fulcrum_study'], set_key='normalized',
                                      anchor_key='eyeConditions | glaucoma', anchor_set='demographics')
        tag = 'all'
        n_clusters = 30 # Define the number of hidden factors to use.
    else:
        psychometrics = PSY_GROUPS[psy_group_key]
        tag = psy_group_key
        n_clusters = 10 # Define the number of hidden factors to use.

    user_ids = get_user_ids(raw_data['fulcrum_study'])
    print(psychometrics)

    input_demo = get_samples_for_input(raw_data['fulcrum_study'], user_ids, conditions, set_key='demographics',
                                       anchor_key='eyeConditions | glaucoma')
    input_psych = get_samples_for_input(raw_data['fulcrum_study'], user_ids, psychometrics, set_key='normalized',
                                       anchor_key='eyeConditions | glaucoma', anchor_set='demographics')

    input_demo = add_response_to_binary_value_samples(input_demo, raw_data['fulcrum_study'], user_ids, 'visionPrescription',
                                         dem.vision_prescription, set_key='demographics', anchor_key= 'eyeConditions | glaucoma')

    input_psych = add_response_to_cont_value_samples(input_psych, raw_data['fulcrum_study'], user_ids, 'age',
                                       set_key='demographics', anchor_key='eyeConditions | glaucoma')

    input_psych = add_response_to_cont_value_samples(input_psych, raw_data['fulcrum_study'], user_ids, 'hours_screen_per_day',
                                       set_key='demographics', label_responses=dem.hours_screen_per_day_responses,
                                       label_values=dem.hours_screen_per_day_values, anchor_key='eyeConditions | glaucoma')
    input_psych = add_response_to_cont_value_samples(input_psych, raw_data['fulcrum_study'], user_ids, 'hours_activity_per_week',
                                       set_key='demographics', label_responses=dem.hours_activity_per_week_responses,
                                       label_values=dem.hours_activity_per_week_values, anchor_key='eyeConditions | glaucoma')

    #input_demo, input_psych = test.get_demo_and_psych_samples(raw_data['fulcrum_study'], conditions, psychometrics)

    print(input_psych.shape)
    print(input_demo.shape)

    #Run the psychometric data analysis first - Layer 1

    #additional continuous variables added alongside the psychometrics data
    psychometrics.append('age')
    psychometrics.append('hours_screen_per_day')
    psychometrics.append('hours_activity_per_week')

    # additional binary variables added alongside the demographics data
    conditions.append("vision | farsighted")
    conditions.append("vision | nearsighted")
    conditions.append("vision | only glasses to read")


    psy_layer, psy_clusters = run_corex(input_psych, psychometrics, n_clusters, 'fulcrum_psychometrics_' + tag + '_w_age')


    #include the psychometric clusters with the demographic variables
    input_all = np.hstack((input_demo, psy_layer.labels))
    print("Input all size:")
    print(input_all.shape)
    [conditions.append("Psy_Cluster_" + str(i)) for i in range(n_clusters)]

    print("labels size:")
    print(len(conditions))
    n_clusters_all = 30

    #Run demographic variables and psychometric clusters analysis together
    dem_layer, dem_clusters = run_corex(input_all, conditions, n_clusters_all, 'fulcrum_dem_and_psych', topic=True)

    psy_cluster_order = []

    for j in range(n_clusters_all):
        [psy_cluster_order.append(int(dem_clusters[j][k][12:])) for k in range(len(dem_clusters[j])) if "Psy_Cluster_" in dem_clusters[j][k] ]

    print(psy_cluster_order)

    #generate graphs
    tex_name = 'fulcrum_dem_and_psych_' + tag + '.tex'
    test_output = LatexChartGenerator(tex_name)
    test_output.writeHeader()
    test_output.writeClusters(psy_clusters, n_clusters, 1, cluster_order=psy_cluster_order, tc=psy_layer.tc)
    test_output.writeClusters(dem_clusters, n_clusters_all, 2, tc=dem_layer.tc)
    test_output.writeArrows(n_clusters,"Psy_Cluster_")
    test_output.writeFooter()



if __name__ == "__main__":

    folder_name = 'data/classification_source/'
    file_names = {
        'fulcrum_study': 'normalized_data.p',
    }

    raw_data = dict()
    for key, value in file_names.items():
        if os.path.isfile(
                os.path.join(folder_name, value)
        ):
            with open(os.path.join(folder_name, value), 'rb') as read_file:
                # raw_data[key] = json.load(read_file)
                print(os.path.join(folder_name, value))
                raw_data[key] = pickle.load(read_file)
                # print(raw_data[key][next(iter(raw_data[key]))]['normalized'].keys())
        else:
            print('File not found: {0}'.format(os.path.join(folder_name, value)))

    #test_ce()

    #test_ce_topic()


    #for run in range(5):
    #    print("RUN NUMBER {0}".format(run))
    #    print("===================================================================")
    #run_fulcrum_data()
    #run_psychometric_data(raw_data)
    run_demo_psych_combined(raw_data)

    #for psyKey in PSY_GROUPS:
    #   run_demo_psych_combined(raw_data, psyKey)

