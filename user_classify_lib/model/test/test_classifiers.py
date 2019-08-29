import os
import unittest
import pickle
from sklearn.model_selection import train_test_split

from user_classify_lib.psy_variables import components
from user_classify_lib.model.classifiers import SupervisedClassifier

TEST_DIR = os.path.dirname(os.path.abspath(__file__))

class TestSupervisedClassifiers(unittest.TestCase):


    def load_sample_data(self):
        folder_name = TEST_DIR + '/data/classification_source/'
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

        return raw_data['fulcrum_study']

    def test_dummy_classifier(self):

        input_labels = components

        output_labels = [
            "eyeConditions | dry eye",
            "eyeConditions | glaucoma"
        ]

        dummy = SupervisedClassifier(classifier_type='Dummy',
                                     input_labels=input_labels,
                                     output_labels=output_labels)

        raw_data = self.load_sample_data()

        train_x, test_x, train_y, test_y = dummy.get_train_test_data(raw_data, input_set_key='normalized',
                                                                     output_set_key='demographics')
        dummy.train_model(train_x, train_y)
        dummy.validate_model(test_x, test_y)
