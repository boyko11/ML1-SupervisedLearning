from sklearn import tree
from learner import Learner
import data_service
import sys


class DTLearner(Learner):

    def __init__(self, criterion='gini', min_samples_leaf=1, max_depth=None, class_weight=None):
        self.estimator = tree.DecisionTreeClassifier(random_state=78, criterion=criterion,
                                                     min_samples_leaf=min_samples_leaf, max_depth=max_depth,
                                                     class_weight=class_weight)

    def fit_predict_score(self, x_train, y_train, x_test, y_test):

        # print(np.unique(y_train))
        # print(np.unique(y_test))
        return super(DTLearner, self).fit_predict_score(self.estimator, x_train, y_train, x_test, y_test)

    def draw_tree(self, tree_id=''):
        tree.export_graphviz(self.estimator, out_file = 'tree{0}.dot'.format(tree_id))

    @staticmethod
    def gini_vs_entropy():

        min_samples_leaf = 1
        max_depth = None
        class_weight = 'balanced'

        datasets = ['breast_cancer', 'kdd']
        criterions = ['gini', 'entropy']
        for dataset in datasets:
            if dataset == 'kdd':
                transform_data = True
                random_slice = 10000
            else:
                transform_data = False
                random_slice = None
            x_train, x_test, y_train, y_test = data_service.load_and_split_data(scale_data=True,
                                                                                transform_data=transform_data,
                                                                                random_slice=random_slice,
                                                                                random_seed=None,
                                                                                dataset=dataset,
                                                                                test_size=.5)
            for criterion in criterions:
                print(dataset, criterion)
                dt_learner = DTLearner(criterion=criterion, min_samples_leaf=min_samples_leaf, max_depth=max_depth,
                                       class_weight=class_weight)

                dt_accuracy_score, dt_fit_time, dt_predict_time = dt_learner.fit_predict_score(x_train, y_train,
                                                                                               x_test, y_test)
                print('DT set: {0}, criterion: {1}  score: {2}, fit_time: {3}, predict_time: {4}, {2}'
                      .format(dataset, criterion, dt_accuracy_score, dt_fit_time, dt_predict_time))

                dt_learner.draw_tree('-{0}-{1}'.format(dataset, criterion))

    @staticmethod
    def prunning(class_weight='balanced'):

        datasets = ['breast_cancer', 'kdd']
        min_samples_leaf_list = [1, 25]
        max_depth_list = [None, 4]
        for dataset in datasets:
            if dataset == 'kdd':
                transform_data = True
                random_slice = 10000
            else:
                transform_data = False
                random_slice = None
            x_train, x_test, y_train, y_test = data_service.load_and_split_data(scale_data=True,
                                                                                transform_data=transform_data,
                                                                                random_slice=random_slice,
                                                                                random_seed=None,
                                                                                dataset=dataset,
                                                                                test_size=.5)
            for min_samples_leaf in min_samples_leaf_list:
                for max_depth in max_depth_list:
                    dt_learner = DTLearner(criterion='entropy', min_samples_leaf=min_samples_leaf, max_depth=max_depth,
                                           class_weight=class_weight)

                    dt_accuracy_score, dt_fit_time, dt_predict_time = dt_learner.fit_predict_score(x_train, y_train,
                                                                                                   x_test, y_test)
                    print('DT set: {0}, min_samples_leaf: {1}, max_depth: {2},  score: {3}, fit_time: {4}, predict_time: {5}'
                          .format(dataset, min_samples_leaf, max_depth, dt_accuracy_score, dt_fit_time, dt_predict_time))

                    dt_learner.draw_tree('-{0}-min_samples_leaf{1}-max_depth{2}'.format(dataset, min_samples_leaf,
                                                                                        max_depth))


if __name__ == '__main__':

    dtLearner = DTLearner()
    if len(sys.argv) < 2:
        print('Need a second command line argument.')
        exit()

    function_to_run = sys.argv[1]
    if function_to_run == 'attribute_splitting':
        dtLearner.gini_vs_entropy()
    elif function_to_run == 'prunning':
        class_weight = 'balanced'
        if len(sys.argv) > 2:
            class_weight = None if sys.argv[2] != 'balanced' else 'balanced'

        print('class_weight: {0}'.format(class_weight))
        dtLearner.prunning(class_weight)