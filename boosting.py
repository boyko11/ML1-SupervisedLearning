from sklearn import ensemble
from learner import Learner
from sklearn import tree


class BoostingLearner(Learner):

    def __init__(self, n_estimators=50, max_depth=1, class_weight=None):

        self.max_depth = max_depth
        #self.estimator = ensemble.GradientBoostingClassifier()
        self.estimator = ensemble.AdaBoostClassifier(
            base_estimator=tree.DecisionTreeClassifier(max_depth=self.max_depth, class_weight=class_weight),
            n_estimators=n_estimators)
        #self.estimator = ensemble.AdaBoostClassifier()

    def fit_predict_score(self, x_train, y_train, x_test, y_test):

        return super(BoostingLearner, self).fit_predict_score(self.estimator, x_train, y_train, x_test, y_test)

    def display_trees(self):

        print(self.estimator.estimators_)
        # for i, one_tree_in_the_forrest in enumerate(self.estimator.estimators_):
        #     print(one_tree_in_the_forrest.tree_.node_count)
        #     print(one_tree_in_the_forrest.tree_.feature[0])
        #     print('----------------------')
        #     tree.export_graphviz(one_tree_in_the_forrest, out_file='tree{0}.dot'.format(i+1))
