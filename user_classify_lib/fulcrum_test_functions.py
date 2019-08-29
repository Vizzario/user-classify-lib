import numpy as np

#import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import chisquare

import user_classify_lib.psy_variables as psy
import user_classify_lib.demographic_variables as dem

AGE_DIVIDERS = (15, 30, 45, 60)  # type: List[Any]

def get_age_group(age:int, answers: list):

    A, B, C, D, E = False, False, False, False, False
    if (age < AGE_DIVIDERS[0]):
        A = True
        #print("A")
    elif (age < AGE_DIVIDERS[1]):
        B = True
        #print("B")
    elif (age < AGE_DIVIDERS[2]):
        C = True
        #print("C")
    elif (age < AGE_DIVIDERS[3]):
        D = True
        #print("D")
    else:
        E = True
        #print("E")

    answers.append(1) if A else answers.append(0)
    answers.append(1) if B else answers.append(0)
    answers.append(1) if C else answers.append(0)
    answers.append(1) if D else answers.append(0)
    answers.append(1) if E else answers.append(0)
    return answers
    '''
    young, med, old = False, False, False
    if(age < AGE_DIVIDERS[0]):
        young = True
        print("Young")
    elif(age < AGE_DIVIDERS[1]):
        med = True
        print("Medium")
    else:
        old = True
        print("Old")

    answers.append(1) if young else answers.append(0)
    answers.append(1) if med else answers.append(0)
    answers.append(1) if old else answers.append(0)
    return answers
    '''

def get_input_conditions(raw_data: dict):
    health_conditions = list()
    for user, value in raw_data.items():
        response = value['demographics']
        if 'eyeConditions | glaucoma' in response.keys():

            for question, answer in response.items():
                if 'eyeConditions' not in question and type(answer) is bool:
                    if question not in health_conditions:
                        health_conditions.append(question)

    print(health_conditions)
    return health_conditions

##Remove mild/medium/severe conditions
def get_conditions_no_scales(raw_data: dict):
    health_conditions = list()
    for user, value in raw_data.items():
        response = value['demographics']
        if 'eyeConditions | glaucoma' in response.keys():

            for question, answer in response.items():
                if 'mild' not in question and 'moderate' not in question and 'severe' not in question and type(
                        answer) is bool:
                    if question not in health_conditions:
                        health_conditions.append(question)

    print(health_conditions)
    return health_conditions

def get_conditions_all(raw_data: dict):
    health_conditions = list()
    for user, value in raw_data.items():
        response = value['demographics']
        if 'eyeConditions | glaucoma' in response.keys():

            for question, answer in response.items():
                if type(answer) is bool:
                    if question not in health_conditions:
                        health_conditions.append(question)

    for i, condition in enumerate(health_conditions):
        print(i)
        print(condition)
    return health_conditions

def get_psychometrics_all(raw_data: dict):
    psychometrics = list()
    for user, value in raw_data.items():
        response = value['normalized']
        for question, answer in response.items():
            if question not in psychometrics:
                psychometrics.append(question)

    return psychometrics

def generate_input_matrix(raw_data: dict, labels: list):
    input_all = None
    for user, value in raw_data['demo'].items():
        response = value['demographics']
        answers = list()
        for label in labels:
            if label in response.keys():
                answers.append(1) if response[label] else answers.append(0)
            else:
                answers.append(None)
        print(answers)
        answers = np.asarray(answers)
        input_all = answers if input_all is None else np.vstack((input_all, answers))
    # print(input_all)

def get_demo_samples(raw_data: dict, labels: list):
    input_all = None

    for user, value in raw_data.items():
        response = value['demographics']
        if 'eyeConditions | glaucoma' in response.keys():
            answers = list()
            for label in labels:
                if label in response.keys():
                    answers.append(1) if response[label] else answers.append(0)
                    # answers.append(response[label])
                else:
                    answers.append(-1)
                    # answers.append(0)
                    # continue
            if answers is not None:
                # add in age as a variable
                # answers = get_age_group(response['age'], answers)
                answers.append(float(value['demographics']['age'] / 100))

                answers = np.asarray(answers)
                # print(answers)
                input_all = answers if input_all is None else np.vstack((input_all, answers))
    # print(input_all)
    print(input_all.shape)
    return input_all

def get_demo_samples_w_scale(raw_data: dict, labels: list):
    input_all = None

    for user, value in raw_data.items():
        response = value['demographics']
        if 'eyeConditions | glaucoma' in response.keys():
            answers = list()
            for label in labels:
                if label in response.keys():
                    if response[label] and label + " | mild" in response.keys():
                        if response[label + " | mild"]:
                            answers.append(0.33333)
                        elif response[label + " | moderate"]:
                            answers.append(0.66667)
                        elif response[label + " | severe"]:
                            answers.append(1)
                        else:
                            print("WARNING: Expected a rating (mild/moderate/severe) but has None")
                            answers.append(1)

                    else:
                        answers.append(1) if response[label] else answers.append(0)
                        # answers.append(response[label])
                else:
                    answers.append(0)
                    continue
            if answers is not None:
                answers = np.asarray(answers)
                # print(answers)
                input_all = answers if input_all is None else np.vstack((input_all, answers))
    print(input_all)
    print(input_all.shape)
    return input_all

def get_psychometric_samples(raw_data: dict, labels: list):
    input_all = None

    for user, value in raw_data.items():
        response = value['normalized']
        answers = list()
        for label in labels:
            if label in response.keys():
                answers.append(response[label])
                # answers.append(response[label])
            else:
                answers.append(-1)
                # answers.append(0)
                # continue
        if answers is not None:
            # add age
            answers.append(float(value['demographics']['age'] + 50))  # add 50 to age for range scaling

            answers = np.asarray(answers)
            # print(answers)
            input_all = answers if input_all is None else np.vstack((input_all, answers))
    # print(input_all)
    print(input_all.shape)
    return input_all

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

def get_demo_and_psych_samples(raw_data: dict, labels_demo: list, labels_psych: list, anchor_key=None):

    input_demo = None
    input_psy = None

    if anchor_key is None:
        anchor_key = 'eyeConditions | glaucoma'  # samples must have this key in order to be added to the list

    for user, value in raw_data.items():

        if anchor_key in value['demographics'].keys():
            response = value['demographics']
            answers_demo = list()
            for label in labels_demo:
                if label in response.keys():
                    answers_demo.append(1) if response[label] else answers_demo.append(0)
                    # answers.append(response[label])
                else:
                    # answers_demo.append(-1)
                    answers_demo.append(0)
                    # continue
            if answers_demo is not None:
                # print(value['demographics']['visionPrescription'])
                get_sample_response_to_binary_value(value['demographics'], 'visionPrescription',
                                                    dem.vision_prescription, answers_demo)

                answers_demo = np.asarray(answers_demo)
                # print(answers)
                input_demo = answers_demo if input_demo is None else np.vstack((input_demo, answers_demo))

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

    print(input_demo.shape)
    print(input_psy.shape)
    return input_demo, input_psy

def generate_supervised_classifier(train_x: np.ndarray, train_y: np.ndarray,
                                   test_x: np.ndarray = None, test_y: np.ndarray = None,
                                   classifier_type: str = 'Dummy'):
    print(train_x.shape)
    print(test_x.shape)

    classifier_options = {
        'Dummy' : DummyClassifier(strategy="stratified"),
        'RandomForest' : RandomForestClassifier(n_estimators=1000)
    }

    classifier = classifier_options[classifier_type]
    classifier.fit(train_x, train_y)

    if test_x is not None and test_y is not None:

        print(train_y.shape)
        print(test_y.shape)
        test_data_length = test_y.shape[0]
        num_output_labels = test_y.shape[1]

        pred_val = classifier.predict(test_x)  # > 0.08
        prob_val = classifier.predict_proba(test_x)

        # pred_val = pred_val.astype(int)

        num_correct = [0] * num_output_labels
        for i in range(test_data_length):
            print("{0}  {1} : Probability: {2}".format(str(test_y[i]), str(pred_val[i]), str([p[i] for p in prob_val])))
            for j in range(len(num_correct)):
                if int(test_y[i][j]) == int(pred_val[i][j]) : num_correct[j] += 1

        print("Number Correct: {0}".format(num_correct))
        print("Ratio Correct: {0}".format([n/test_data_length for n in num_correct]))

    return classifier

def test_glaucoma_conditions_chi(raw_data: dict, labels: list):
    input_all = None
    glaucoma_all = list()
    inputs_glaucoma = None
    for user, value in raw_data.items():
        response = value['demographics']
        if 'eyeConditions | glaucoma' in response.keys():
            answers = list()
            for label in labels:
                if label in response.keys():
                    # answers.append(1) if response[label] else answers.append(0)
                    answers.append(response[label])
                else:
                    answers = None
                    continue
            if answers is not None:
                print(answers)
                answers = np.asarray(answers)
                input_all = answers if input_all is None else np.vstack((input_all, answers))
                glaucoma_all.append(response['eyeConditions | glaucoma'])
                if (response['eyeConditions | glaucoma']):
                    inputs_glaucoma = answers if inputs_glaucoma is None else np.vstack(
                        (inputs_glaucoma, answers))
    # print(input_all)
    glaucoma_all = np.asarray(glaucoma_all)
    labels = np.asarray(labels)
    print(input_all.shape)
    print(glaucoma_all.shape)
    print(inputs_glaucoma.shape)

    # Total entries: 1440
    # Total entries with Glaucoma: 119

    inputs_all_sum = np.sum(input_all, axis=0)
    glaucoma_observed = np.sum(inputs_glaucoma, axis=0)

    glaucoma_expected = 119 * (inputs_all_sum / 1440)

    print(inputs_all_sum)
    print(labels)
    print(glaucoma_observed)
    print(glaucoma_expected)
    # print(labels[glaucoma_expected > 5])
    # print(glaucoma_observed[glaucoma_expected > 5])
    # print(glaucoma_expected[glaucoma_expected > 5])

    # print(glaucoma_expected > 5)
    chi = chisquare(glaucoma_observed[glaucoma_expected > 5], f_exp=glaucoma_expected[glaucoma_expected > 5])
    print(chi)

def test_class_conditions_input(raw_data: dict, labels: list):
    input_all = None
    glaucoma_all = list()
    for user, value in raw_data.items():
        response = value['demographics']
        if 'eyeConditions | glaucoma' in response.keys():
            answers = list()
            for label in labels:
                if label in response.keys():
                    # answers.append(1) if response[label] else answers.append(0)
                    answers.append(response[label])
                else:
                    answers = None
                    continue
            if answers is not None:
                print(answers)
                answers = np.asarray(answers)
                input_all = answers if input_all is None else np.vstack((input_all, answers))
                # glaucoma_all.append(1) if response['eyeConditions | glaucoma'] else glaucoma_all.append(0)
                glaucoma_all.append(response['eyeConditions | glaucoma'])
    # print(input_all)
    glaucoma_all = np.asarray(glaucoma_all)
    print(input_all.shape)
    print(glaucoma_all.shape)

    train_x = input_all[0:1000];
    test_x = input_all[1000:];

    train_y = glaucoma_all[0:1000];
    test_y = glaucoma_all[1000:];

    generate_supervised_classifier(train_x, train_y, test_x, test_y)

def test_class_gender_input(raw_data: dict):
    gender_all = None
    glaucoma_all = list()
    for user, value in raw_data.items():
        response = value['demographics']
        if 'eyeConditions | glaucoma' in response.keys() and 'gender' in response.keys():
            gen = list()
            glaucoma = None
            for question, answer in response.items():
                if 'gender' == question:
                    gen.append(1) if answer == 'male' else gen.append(0)
                elif 'eyeConditions | glaucoma' == question:
                    glaucoma = 1 if answer else 0

            gen = np.asarray(gen)
            if gen.size == 0:
                print(user)
            gender_all = gen if gender_all is None else np.vstack((gender_all, gen))
            glaucoma_all.append(glaucoma)
    glaucoma_all = np.asarray(glaucoma_all)

    train_x = gender_all[0:1000];
    test_x = gender_all[1000:];

    train_y = glaucoma_all[0:1000];
    test_y = glaucoma_all[1000:];

    print(gender_all.size)
    print(glaucoma_all.size)
    generate_supervised_classifier(train_x, train_y, test_x, test_y)