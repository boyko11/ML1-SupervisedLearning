code: https://github.com/boyko11/ML1-SupervisedLearning

Assumes installed:

python 3+,
numpy,
scikit-learn,
scipy,
matplotlib,
tabulate,
graphviz

It also assumes graphviz's 'dot' is on the execution path

For most experiments the 'random_slice' variable is set to 10000 for the kdd dataset.
This is to demo in a quick way. If the varibable is set to None, it would load the entire kdd dataset and for some of
the algorithms execution would take a long time.

1. Decision Tree - gini-vs-entropy

    python decision_tree.py attribute_splitting

    This will print gini-vs-entropy stats as described in 'Attribute Splitting' section
    It will also create '.dot' files for the trees created in the working directory.
    To convert the files to .png:

    dot -Tps tree-breast_cancer-gini.dot -o tree-breast_cancer-gini.png
    dot -Tps tree-breast_cancer-entropy.dot -o tree-breast_cancer-entropy.png
    dot -Tps tree-kdd-gini.dot -o tree-kdd-gini.png
    dot -Tps tree-kdd-entropy.dot -o tree-kdd-entropy.png

-------------------------------------------------------------------------------------------

2. Decision Tree - prunning

    python decision_tree.py prunning

    This will fit,test and print stats for the decision tree under combinations of the following params and their values:
    min_samples_leaf: 1, 25
    max_depth: None, 4

    It will also create a .dot graph files in the working directory for each use case. To convert them to .png

    dot -Tps tree-breast_cancer-min_samples_leaf1-max_depth4.dot -o tree-breast_cancer-min_samples_leaf1-max_depth4.png
    dot -Tps tree-breast_cancer-min_samples_leaf1-max_depthNone.dot -o tree-breast_cancer-min_samples_leaf1-max_depthNone.png
    dot -Tps tree-breast_cancer-min_samples_leaf25-max_depth4.dot -o tree-breast_cancer-min_samples_leaf25-max_depth4.png
    dot -Tps tree-breast_cancer-min_samples_leaf25-max_depthNone.dot -o tree-breast_cancer-min_samples_leaf25-max_depthNone.png
    dot -Tps tree-kdd-min_samples_leaf1-max_depth4.dot -o tree-kdd-min_samples_leaf1-max_depth4.png
    dot -Tps tree-kdd-min_samples_leaf1-max_depthNone.dot -o tree-kdd-min_samples_leaf1-max_depthNone.png
    dot -Tps tree-kdd-min_samples_leaf25-max_depth4.dot -o tree-kdd-min_samples_leaf25-max_depth4.png
    dot -Tps tree-kdd-min_samples_leaf25-max_depthNone.dot -o tree-kdd-min_samples_leaf25-max_depthNone.png

    python decision_tree.py prunning unbalanced
    would do the same, but it would not set the class_weights to be balanced

-------------------------------------------------------------------------------------------

3. Decision Tree - learning curve

    python run_learning_curves_compare.py dt breast_cancer
    python run_learning_curves_compare.py dt kdd

    This will generate learning curve per training samples graphs in the working directory
    Breast_Cancer_DT_Learning_Curve.png
    KDD_DT_Learning_Curve.png

-------------------------------------------------------------------------------------------

4. Boosting

    python run_accuracy_compare.py breast_cancer
    python run_accuracy_compare.py kdd

    These would print the stats discussed in the Boosting section
    For the different permutation discussed, the max_depth variable and max_depth_boost variable would need to be adjusted
    Note: this would also run and display stats for SVM, KNN and the Neural Net

    python run_learning_curves_compare.py boost breast_cancer
    python run_learning_curves_compare.py boost kdd

    This will generate learning curve per training samples graphs in the working directory
    KDD_BOOST_Learning_Curve.png
    Breast_Cancer_BOOST_Learning_Curve.png

-------------------------------------------------------------------------------------------

5. KNN

   python run_accuracy_compare.py breast_cancer
   python run_accuracy_compare.py kdd

   These would print the stats discussed in the KNN section
   For the different permutation discussed, the variables under the ##  KNN comment section would need adjusting
   Note: this would also run and display stats for the other algorithms

   python run_learning_curves_compare.py knn breast_cancer
   python run_learning_curves_compare.py knn kdd

   This will generate learning curve per training samples graphs in the working directory
   KDD_KNN_Learning_Curve.png
   Breast_Cancer_KNN_Learning_Curve.png

-------------------------------------------------------------------------------------------

6. Neural Network

   python run_accuracy_compare.py breast_cancer
   python run_accuracy_compare.py kdd

   These would print the stats discussed in the Neural Network section
   For the different permutation discussed, the variables under the ##  NN comment section would need adjusting
   Note: this would also run and display stats for the other algorithms

   python run_learning_curves_compare.py nn breast_cancer
   python run_learning_curves_compare.py nn kdd

   This will generate learning curve per training samples graphs in the working directory
   This will also plot the learning curve for non-scaled data - dashed line
   KDD_NN_Learning_Curve.png
   Breast_Cancer_NN_Learning_Curve.png
   Note: The KDD Learning curve took one minute to run on my machine for a sample of 10000 training records

   python run_learning_curves_iter_compare.py nn breast_cancer
   python run_learning_curves_iter_compare.py nn kdd

   These will generate learning curve per iterations plot files in the working directory
   Breast_Cancer_NN_Iterations_Learning_Curve.png
   KDD_NN_Iterations_Learning_Curve.png
   Note: KDD is currently set to run with 1000 training samples only, larger more realistic numbers take a long time

-------------------------------------------------------------------------------------------
7. SVM

   python run_accuracy_compare.py breast_cancer
   python run_accuracy_compare.py kdd

   These would print the stats discussed in the SVM section
   For the different permutation discussed, the variables under the ##  SVM comment section would need adjusting
   Note: this would also run and display stats for the other algorithms

   python run_learning_curves_compare.py svm breast_cancer
   python run_learning_curves_compare.py svm kdd

   These will generate learning curve per training samples graphs in the working directory
   These will also plot the learning curve for non-scaled data - dashed line
   KDD_SVM_Learning_Curve.png
   Breast_Cancer_SVM_Learning_Curve.png
   Note: The KDD Learning curve took about 5 minutes to run on my machine for a sample of 10000 training records

   python run_learning_curves_iter_compare.py nn breast_cancer
   python run_learning_curves_iter_compare.py nn kdd

   These will generate learning curve per iterations plot files in the working directory
   Breast_Cancer_SVM_Iterations_Learning_Curve.png
   KDD_SVM_Iterations_Learning_Curve.png
   Note: KDD is currently set to run with 1000 training samples only, larger more realistic numbers take a long time


-------------------------------------------------------------------------------------------
8. Performance Comparison

   python run_accuracy_compare.py breast_cancer
   python run_accuracy_compare.py kdd

   These will run all algorithms and output coparison stats


-------------------------------------------------------------------------------------------
8. Grid Search

   python grid_search.py

   This file was run to report on the findings in the Grid Search section.
   The variables tuned_parameters and estimator would need to be change to do a grid search form SVM vs NN
   The varaible dataset would need to be changed to run breast_cancer vs kdd






