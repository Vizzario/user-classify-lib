import numpy as np

from user_classify_lib import fulcrum_test_functions as test, psy_variables as psy, demographic_variables as dem
from user_classify_lib.question_cluster import biocorex as ce
from user_classify_lib.util import run_corex, get_questions, get_user_ids, get_samples_for_input, \
    add_response_to_binary_value_samples, add_response_to_cont_value_samples

from user_classify_lib.output import LatexClusterChartGenerator

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
            run_corex(input, psychometrics[psyKey], n_clusters, 'fulcrum_psychometrics_' + psyKey)


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
    test_output = LatexClusterChartGenerator(tex_name)
    test_output.writeHeader()
    test_output.writeClusters(psy_clusters, n_clusters, 1, cluster_order=psy_cluster_order, tc=psy_layer.tc)
    test_output.writeClusters(dem_clusters, n_clusters_all, 2, tc=dem_layer.tc)
    test_output.writeArrows(n_clusters,"Psy_Cluster_")
    test_output.writeFooter()