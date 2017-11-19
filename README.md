# adversarial-decision-tree
Adversarial Decision Tree

This is a simple exploration of adversarial examples and attacks against a decision tree. The decision tree is trained on the famous iris dataset. Then, some test sample is used to predict a target. The leaf node returned by the prediction is obtained. Every leaf node has a parent reference to the parent node in the tree. Every internal node (except the root) has a parent reference, a left child node, a right child node, a feature used to split left and right nodes, and a value from which to split (feature values < this value are classified as left children, and >= are right children). 

With this leaf node, a uniform cost search is made against the tree to find the shortest path from this leaf node to another class in the decision tree. This shortest cost path is returned. The path is then traversed from the initial correctly predicted leaf node to the new target (adversarial) leaf node. 

The first entry in the path is of course the initial correct leaf node, the second entry is its parent node in the decision tree (since it cannot have left and right children). The next entry in the path could be yet another parent of the tree-node, a left child, or a right child. If it is a parent, we continue traversing. Once we start traveling down another branch in the decision tree via a left or right child, we know this is the least common ancestor between the correct leaf node and the target leaf node. We want to update our test sample to fall on the target's side of this node. So we find the column (feature) that this common ancestor in the tree is using to split into left and right children, then we look at the value it uses. 

If the target is on the left side of this value, we update the corresponding test sample feature to be equal to the ancestor value plus some small negative number (-0.001) so that the sample will fall on the left side of the decision when it is next predicted, and therefore be closer to the target class. 

If the target is on the right side, we update the sample column to the ancestor value plus some small positive number (+0.001) so that the sample will fall on the right side of the decision when it is next predicted. 

We continue traversing the path in this way -- path nodes that are left children of their parents cause the sample to be updated according to the parent's column (feature) and value, but on the left side (by subtracting a small number); path nodes that are right children of their parents cause the sample to be updated according ot the parent's column (feature) and value, but on the right side (by adding a small number).

An example is:

python3 adversarialDecisionTree.py 

[7.2, 0.805555556, 3.6, 0.666666667, 6.1, 0.86440678, 2.5, 0.9999, 'virginica']

virginica

[7.2, 0.805555556, 3.6, 0.666666667, 6.1, 0.86440678, 1.699, 0.9999, 'virginica']

versicolor

The first line is just the command to run the program. 
The second line is the test sample. It is a vector of a bunch of features and finally a true label (virginica). 
The third line is the predicted class by the tree. The tree gets this test sample correct in the third line: virginica. 
The fourth line is the updated sample after traversing the shortest path from the prediction's outputted leaf node to a tree leaf node with a leaf value of versicolor. So our attack target class is versicolor. Note the only change occurred in column 6 (0-based index) from 2.5 to 1.699. This means our attack found a versicolor leaf node that was a sibling of our initial leaf node. The parent of these two nodes was splitting on the 6th feature with a value of 1.7, and we updated to fall on the left side of that node to 1.699, which caused our sample to be predicted as versicolor instead of virginica. 
The fifth line is the prediction the tree makes when ran against the updated (adversarial) sample, which is indeed versicolor, our target class.
