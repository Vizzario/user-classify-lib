import json

import numpy as np

#get a list of all the questions from the survey set. Use data type to filter for a type of question (binomial,
# multivariate, continuous etc.) based on the data type or set it to all to get all the questions.
from user_classify_lib import psy_variables as psy, demographic_variables as dem
from user_classify_lib.output import LatexFeaturesBarGraphGenerator
from user_classify_lib.question_cluster import biocorex as ce
from user_classify_lib.question_cluster.corextopic import corextopic as cet

#region User Question and Response
"""
    USER QUESTION AND RESPONSE
"""

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
#ADDENDUM: Apparently this is known as "one-hot encoding"
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
#endregion


#region Features Count and Plotting
"""
    FEATURES COUNT LIST
"""
#Feature Count List for continuous (psychometric) variables

def get_feature_counts_cont(desired_data: dict, set_key: str, feature_keys : list = None, output_path: str = None):

    if feature_keys is None:
        # Sorted feature list
        feature_keys = list()
        for agent_data in desired_data.values():
            for feature_key in agent_data[set_key].keys():
                if feature_key not in feature_keys: feature_keys.append(feature_key)
        feature_keys = list(sorted(feature_keys))

    # Sorted possible feature responses
    feature_responses = {
        feature_key: list()
        for feature_key in feature_keys
    }
    for agent_data in desired_data.values():
        for feature_key in agent_data[set_key].keys():

            #psychometric variables are restricted within a certain range
            feature_responses[feature_key] = psy.psy_range

    # Counts
    feature_response_counts = {
        feature_key: {
            feature_response: 0
            for feature_response in feature_key_responses
        }
        for feature_key, feature_key_responses in feature_responses.items()
    }
    for agent_data in desired_data.values():
        for feature_key, feature_response in agent_data[set_key].items():
            feature_response_counts[feature_key] = psy.bin_psy_range(feature_response, feature_response_counts[feature_key])


    #test plot

    #temp_cond = ["eyeConditions | diabetic retinopathy", "eyeConditions | glaucoma | moderate"]

    #generate graphs
    tag = 0
    tex_name = 'fulcrum_psychometrics_chart_' + str(tag) + '.tex'
    latex_chart = LatexFeaturesBarGraphGenerator(tex_name)
    latex_chart.writeHeader("Fulcrum Psychometrics Data")

    for feature_key in sorted(feature_response_counts):
        #if feature_key == 'age' : continue
        if feature_key == 'study_key': continue

        sorted_names = list()
        sorted_vals = list()

        for key, response in sorted(feature_response_counts[feature_key].items()):
            sorted_names.append(key)
            sorted_vals.append(response)

        latex_chart.pageBreak()
        latex_chart.makeFeaturesBarGraph(feature_key, sorted_names, sorted_vals)

    latex_chart.writeFooter()

    #print(json.dumps(
    #            feature_response_counts,
    #            indent=4
    #        ))
    # Save
    if output_path is not None:
        with open(output_path, 'w') as write_file:
            json.dump(
                feature_response_counts,
                write_file,
                indent=4
            )

#Feature Count List, based on script from Kyle
def get_feature_counts(desired_data: dict, set_key: str, feature_keys : list = None,
                       latex_output_path: str = None, latex_title: str = "Categorical Features Count",
                       json_output_path: str = None):

    if feature_keys is None:
        # Sorted feature list
        feature_keys = list()
        for agent_data in desired_data.values():
            for feature_key in agent_data[set_key].keys():
                if feature_key not in feature_keys: feature_keys.append(feature_key)
        feature_keys = list(sorted(feature_keys))

    # Sorted possible feature responses
    feature_responses = {
        feature_key: list()
        for feature_key in feature_keys
    }
    for agent_data in desired_data.values():
        for feature_key, feature_response in agent_data[set_key].items():

            #for demographic data: we consider age as a binned range instead of individual values
            if feature_key == 'age':
                feature_responses[feature_key] = dem.age_range
                continue

            if feature_response not in feature_responses[feature_key]:
                feature_responses[feature_key].append(feature_response)
    for feature_key, feature_values in feature_responses.items():
        feature_responses[feature_key] = list(sorted(feature_values))

    # Counts
    feature_response_counts = {
        feature_key: {
            feature_response: 0
            for feature_response in feature_key_responses
        }
        for feature_key, feature_key_responses in feature_responses.items()
    }
    for agent_data in desired_data.values():
        for feature_key, feature_response in agent_data[set_key].items():
            if feature_key == 'age':
                feature_response_counts[feature_key] = dem.bin_age_range(feature_response, feature_response_counts[feature_key])
            else:
                feature_response_counts[feature_key][feature_response] += 1

    #test plot

    #temp_cond = ["eyeConditions | diabetic retinopathy", "eyeConditions | glaucoma | moderate"]

    #generate graphs
    if latex_output_path is None:
        tag = 0
        latex_output_path = 'fulcrum_features_chart_' + str(tag) + '.tex'
    latex_chart = LatexFeaturesBarGraphGenerator(latex_output_path)
    latex_chart.writeHeader(latex_title)

    offset = 0
    page_break_counter = 0
    #for feature_key in temp_cond:
    omitted = [
        'age',
        'ethnicity',
        'education',
        'occupation',
        'study_key']
    for feature_key in sorted(feature_response_counts):

        if feature_key in omitted : continue
        sorted_names = list()
        sorted_vals = list()

        for key, response in sorted(feature_response_counts[feature_key].items()):
            sorted_names.append(key)
            sorted_vals.append(response)

        latex_chart.makeFeaturesStackedGraph(offset, feature_key, sorted_names, sorted_vals)
        offset = offset - 3

        page_break_counter = page_break_counter + 1
        if page_break_counter > 3:
            latex_chart.pageBreak()
            page_break_counter = 0
            offset = 0

    for feature_key in sorted(omitted):
        #if feature_key == 'age' : continue
        if feature_key == 'study_key': continue

        sorted_names = list()
        sorted_vals = list()

        for key, response in sorted(feature_response_counts[feature_key].items()):
            sorted_names.append(key)
            sorted_vals.append(response)

        latex_chart.pageBreak()
        latex_chart.makeFeaturesBarGraph(feature_key, sorted_names, sorted_vals)

    latex_chart.writeFooter()

    #print(json.dumps(
    #            feature_response_counts,
    #            indent=4
    #        ))
    # Save
    if json_output_path is not None:
        with open(json_output_path, 'w') as write_file:
            json.dump(
                feature_response_counts,
                write_file,
                indent=4
            )
#endregion

#filter out a subset of data based on conditions.
#we assume the data has the format:
#user >> dataset (demographic or psychometric) >> questions >> response
def filter_data_by_condition(raw_data: dict, question: (str, list), answer, set_key: str):
    filtered_data = dict()
    for user, data in raw_data.items():

        if question is str:
            if question in data[set_key].keys() and data[set_key][question] == answer:
                filtered_data[user] = data

        elif question is list:
            include = True
            for i in range(len(question)):
                if question[i] not in data[set_key].keys(): include = False
                if data[set_key][question[i]] != answer[i]: include = False
            if include:
                filtered_data[user] = data

    return filtered_data






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

