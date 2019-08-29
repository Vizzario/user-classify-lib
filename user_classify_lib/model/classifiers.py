from typing import List

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from user_classify_lib.model.performance_model import PerformanceModel

classifier_options = {
    'Dummy': DummyClassifier(strategy="stratified"),
    'RandomForest': RandomForestClassifier(),
    'RandomForestRegressor': RandomForestRegressor()
}

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



class SupervisedClassifier:


    def __init__(self, classifier_type: str = 'Dummy',
                 input_labels: list = None, output_labels: list = None):

        self.model = classifier_options[classifier_type]
        self.classifier_type = classifier_type
        self.input_labels = input_labels
        self.output_labels = output_labels


    # get the values from a dictionary dataset based on the labels
    # type has three values: input, output, and custom. Determines which label to use
    def parse_data_from_dict(self, raw_data: dict,
                             set_key : str = None,
                             type : str = 'input', custom_label : list = None):

        type_options = ['input', 'output', 'custom']
        assert type in type_options

        if type == 'input' : labels = self.input_labels
        elif type == 'output' : labels = self.output_labels
        elif type == 'custom' : labels = custom_label

        assert labels is not None
        all_answers = None

        for user, value in raw_data.items():

            answers = get_responses_from_questions(value, labels, set_key)

            if answers is not None:
                answers = np.asarray(answers)
                all_answers = answers if all_answers is None else np.vstack((all_answers, answers))

        return all_answers


    # derive training and testing data from a dictionary dataset
    # note that the number of input samples (users) and the number of output samples (users) should be the same
    def get_train_test_data(self, raw_data: dict,
                            input_set_key : str = None, output_set_key : str = None,
                            test_ratio : float = 0.2):
        input_values = None
        output_values = None
        for user, value in raw_data.items():

            input_answers = get_responses_from_questions(value, self.input_labels, input_set_key)
            output_answers = get_responses_from_questions(value, self.output_labels, output_set_key)

            if input_answers is not None and output_answers is not None:
                input_answers = np.asarray(input_answers)
                input_values = input_answers if input_values is None else np.vstack((input_values, input_answers))

                output_answers = np.asarray(output_answers)
                output_values = output_answers if output_values is None else np.vstack((output_values, output_answers))

        train_x, test_x, train_y, test_y = train_test_split(input_values, output_values,
                                                            test_size=test_ratio)
        return train_x, test_x, train_y, test_y


    def train_model(self, train_x: np.ndarray, train_y: np.ndarray):

        assert train_x.shape[1] == len(self.input_labels) #make sure the number of inputs for each sample user is the same
        self.model.fit(train_x, train_y)


    def validate_model(self, test_x: np.ndarray, test_y: np.ndarray):

        pred_val = self.model.predict(test_x)  # > 0.08
        prob_val = self.model.predict_proba(test_x)

        test_data_length = test_y.shape[0]
        num_output_labels = test_y.shape[1]

        # pred_val = pred_val.astype(int)

        num_correct = [0] * num_output_labels
        for i in range(test_data_length):
            print("{0}  {1} : Probability: {2}".format(str(test_y[i]), str(pred_val[i]), str([p[i] for p in prob_val])))
            for j in range(len(num_correct)):
                if int(test_y[i][j]) == int(pred_val[i][j]): num_correct[j] += 1

        print("Number Correct: {0}".format(num_correct))
        print("Ratio Correct: {0}".format([n / test_data_length for n in num_correct]))


    def predict_user_condition(self, user : dict, set_key: str = None):

        input_values = get_responses_from_questions(user.values(), self.input_labels, set_key)
        assert input_values is not None
        input_values = np.asarray(input_values)
        prediction = self.model.predict(input_values)

        return prediction


    def predict(self, x : np.ndarray):
        return self.model.predict(x)


    def predict_probability(self, x):
        return self.model.predict_proba(x)


    def __json__(self):
        return self.__dict__

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.model))


class VPISupervisedClassifier(SupervisedClassifier):

    self.vpi_components = ["field_of_view", "accuracy", "multi_tracking", "endurance", "detection"]

    def __init__(self, classifier_type: str = 'Dummy',
                 output_labels : list = None):


        super().__init__(classifier_type, input_labels=self.vpi_components,
                         output_labels=output_labels)

    def predict(self, components : PerformanceModel):

        score_breakdown = components.scores_breakdown
        inputs = get_responses_from_questions(score_breakdown, self.vpi_components)
        result = self.model.predict(inputs)

        return result