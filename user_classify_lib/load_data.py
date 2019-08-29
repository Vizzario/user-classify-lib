import os

#import sklearn
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

from user_classify_lib.user_cluster.test import test_gower
from user_classify_lib.fulcrum_test_functions import generate_supervised_classifier
from user_classify_lib.psy_variables import components

#matplotlib.use('Agg')

def get_responses_from_questions(value: dict, labels: list, set_key:str=None):

    if set_key is None:
        response = value
    else:
        response = value[set_key]

    answers = list()

    for label in labels:
        if label in response.keys():
            # input_answers.append(1) if input_response[label] else input_answers.append(0)
            answers.append(response[label])
        else:
            answers = None
            break
            # input_answers.append(None)
    if answers is not None:
        # print(input_answers)
        answers = np.asarray(answers)

    return answers

# training and testing various supervised learning classifiers
# raw data is the dataset to do the analyses from
# input/output labels are the names of the input parameters and output answers from the classifiers, respectively
# set keys determine which dataset we are using from the user (demographics dataset or psychometrics dataset
# train/test ratio is the partitioning of data to training and testing sets
def supervised_ml_train_test(raw_data: dict,
                             input_labels: list = None, output_labels: list = None,
                             input_set_key: str = None, output_set_key : str = None,
                             test_ratio : float = 0.2):

    input_values = None
    output_values = None
    for user, value in raw_data.items():

        if input_labels is None:
            if input_set_key is None: input_labels = value.keys()
            else: input_labels = value[input_set_key].keys()
            print("==========")
            print("Input Labels")
            print(input_labels)

        if output_labels is None:
            if output_set_key is None: output_labels = value.keys()
            else: output_labels = value[output_set_key].keys()
            print("==========")
            print("Output Labels")
            print(output_labels)

        input_answers = get_responses_from_questions(value, input_labels, input_set_key)
        output_answers = get_responses_from_questions(value, output_labels, output_set_key)

        if input_answers is not None and output_answers is not None:

            input_answers = np.asarray(input_answers)
            input_values = input_answers if input_values is None else np.vstack((input_values, input_answers))

            output_answers = np.asarray(output_answers)
            output_values = output_answers if output_values is None else np.vstack((output_values, output_answers))

    # print(input_values)

    print(input_values.shape)
    print(output_values.shape)

    train_x, test_x, train_y, test_y = train_test_split(input_values, output_values,
                                                        test_size = test_ratio)

    generate_supervised_classifier(train_x, train_y, test_x, test_y,
                                   classifier_type='RandomForest')

    print("Train X shape")
    print(train_x.shape)
    print("Labels length")
    print(len(input_labels))

if __name__ == "__main__":

    folder_name = 'data/classification_source/'
    file_names = {
        'fulcrum_study': 'normalized_data (fulcrum_study).p',
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

    ######Corex Tests#####

    #test_ce()

    #test_ce_topic()


    #for run in range(5):
    #    print("RUN NUMBER {0}".format(run))
    #    print("===================================================================")

    #run_fulcrum_data()
    #run_psychometric_data(raw_data)
    #run_demo_psych_combined(raw_data)

    # for psyKey in PSY_GROUPS:
    #   run_demo_psych_combined(raw_data, psyKey)

    ##### User Clustering Tests #####

    #test_gower(raw_data)

    #filtered_data = filter_data_by_condition(raw_data['fulcrum_study'], "eyeConditions | dry eye", True, set_key='demographics')

    #get_feature_counts(filtered_data, set_key='demographics')
    #get_feature_counts_cont(filtered_data, set_key='normalized')

    ##### Supervised Classifiers Tests #####
    input_labels = components

    output_labels= [
        "eyeConditions | dry eye",
        "eyeConditions | glaucoma"
    ]
    supervised_ml_train_test(raw_data['fulcrum_study'],
                             input_labels=input_labels, output_labels=output_labels,
                             input_set_key='normalized', output_set_key='demographics')






