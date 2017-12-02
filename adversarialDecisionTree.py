from random import seed
from random import randrange
from csv import reader
from queue import PriorityQueue

def formSplit(col, value, dataset):
    left = list()
    right = list()
    for row in dataset:
        if row[col] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def getGini(leftRight, classes):
    #total count of left values and right values
    totalLen = float(sum([len(leftOrRight) for leftOrRight in leftRight]))
    #init gini
    giniCoeff = 0.0
    #loop twice (one for left, one for right)
    for leftOrRight in leftRight:
        size = float(len(leftOrRight)) #count of left or right
        if size == 0: #skip if nothing
            continue
        score = 0.0 #init score for this side
        for c in classes: #for each class, get the proportion of each class
            p = [row[-1] for row in leftOrRight].count(c) / size
            score += p ** 2 #sum squared proportion
        giniCoeff += (1.0 - score) * (size / totalLen) #weight gini by size of split
    return giniCoeff #return gini

def getBestSplit(dataset):
    classes = list(set(row[-1] for row in dataset))
    bestGini = 99999
    bestCol = None
    bestValue = None
    bestLeftRight = None
    for col in range(len(dataset[0])-1):
        for row in dataset:
            valueToSplitOn = row[col]
            leftRight = formSplit(col, valueToSplitOn, dataset)
            giniCoeff = getGini(leftRight, classes)
            if giniCoeff < bestGini:
                bestGini = giniCoeff
                bestCol = col
                bestValue = row[col]
                bestLeftRight = leftRight
    return {'col': bestCol, 'value': bestValue, 'LR': bestLeftRight}

def formLeaf(group, node):
    outcomes = [row[-1] for row in group]
    return {'leaf':max(set(outcomes), key = outcomes.count), 'parent':node}

def split(node, maxDepth, minSize, depth):
    left, right = node['LR'] #get left and right nodes
    del(node['LR']) #delete LR node since we are now splitting it
    if not left or not right: #if no split (data sent to split was empty), make a leaf node
        leaf = formLeaf(left + right, node)
        node['left'] = leaf
        node['right'] = leaf
        return
    if depth >= maxDepth: #if max depth, make a leaf node per side
        node['left'] = formLeaf(left, node)
        node['right'] = formLeaf(right, node)
        return
    if len(left) <= minSize: #if not enough members in left split, form leaf
        node['left'] = formLeaf(left, node)
    else: #split left side recursively
        node['left'] = getBestSplit(left) #get best way to split this node
        node['left']['parent'] = node #set parent of split
        split(node['left'], maxDepth, minSize, depth + 1) #recurse
    if len(right) <= minSize: #if not enough members in right split, form leaf
        node['right'] = formLeaf(right, node)
    else: #split right side recursively
        node['right'] = getBestSplit(right)
        node['right']['parent'] = node
        split(node['right'], maxDepth, minSize, depth + 1)

def buildTree(train, maxDepth, minSize):
    root = getBestSplit(train)
    split(root, maxDepth, minSize, 1)
    return root

def predict(node, sample): 
    if sample[node['col']] < node['value']: #test sample value of split col against tree node value
        #if test falls on left side...
        if 'leaf' not in node['left']: #if not a leaf, continue testing left side
            return predict(node['left'], sample)
        else:
            return node['left'] #return leaf
    else: #if test falls on right side...
        if 'leaf' not in node['right']: #if not a leaf, continue testing right side
            return predict(node['right'], sample)
        else:
            return node['right'] #return leaf

#run a test set through the tree, report accuracy
def treeTest(tree, test):
    correct = 0
    for row in test:
        pred = predict(tree, row)
        if(pred == row[-1]):
            correct = correct + 1
    return correct / len(test)

#uniform cost search from any node to any target class in a tree
def UCS(node, target, tree):
    frontier = PriorityQueue()
    explored = list()
    g = 1 #init g-cost
    
    #init frontier to break-out nodes from initial node (parent, left, right)
    #note: small .0000X additions used to make priority queue order unique (python issue...)
    #note initial frontier nodes have parent of [node], a single value list
    if('parent' in node):
        frontier.put((g + 0.00001, node['parent'], [node])) 
    if('left' in node):
        frontier.put((g + 0.00004, node['left'], [node]))
    if('right' in node):
        frontier.put((g + 0.00008, node['right'], [node]))
        
    #loop while frontier is not empty 
    while(not frontier.empty()):
        newNode = frontier.get() #pop frontier (lowest cost next node)
        candidate = newNode[1] #candidate tree-node (frontier nodes are (g-cost, tree-node, frontier-parent))
        if('leaf' in candidate and candidate['leaf'] == target): #if target class found
            stack = list()
            stack = [newNode[1]] + stack #place target on stack
            while(len(newNode[2]) == 3): #while path member has another parent
                parentNode = newNode[2] #get parent
                stack = [parentNode[1]] + stack #place parent tree-node on stack
                newNode = parentNode #set new node to parent to keep travsersing the path
            stack = [newNode[2][0]] + stack #set the final (now initial) node to top of stack
            return stack #return full path of initilal -> next -> next ... -> target
        else: #target not found
            if(newNode not in enumerate(explored)): #if frontier node not explored
                explored.append(newNode) #add to explored
                g = g + 1 #increment g-cost
                #expand frontier node into parent tree node, left, right
                if('parent' in candidate):
                    frontier.put((g + 0.00001, candidate['parent'], newNode))
                if('left' in candidate):
                    frontier.put((g + 0.00004, candidate['left'], newNode))
                if('right' in candidate):
                    frontier.put((g + 0.00008, candidate['right'], newNode))

#attack strategy: find shortest path from current leaf to target class 
#we want tree to classify sample as. Note fromLeaf is the leaf predict(sample)
#returns. With the shortest path, we find the least common ancestor tree node, 
#and edit the sample to fall on the target's side of that node! Then continue on 
#the path to the target class, editing the sample column at each successive path node
#to fall on the target side of the node. At the end we end up with a minimally edited sample 
#that used to classify as fromLeaf but now classifies as toTarget
def attackTree(fromLeaf, sample, toTarget, tree):
    shortestPath = UCS(fromLeaf, toTarget, tree) #get shortest path from some leaf node to target class in tree
    lastNode = None
    for node in shortestPath: #iterate path
        if(node == fromLeaf): #if start-node
            lastNode = node #set last node ref to start node in path
            continue
        else:
            #if path node is a parent node, do nothing, just update lastNode
            if('parent' in lastNode and node == lastNode['parent']): 
                #print('up')
                lastNode = node 
            #if path node is a left-node...
            elif('left' in lastNode and node == lastNode['left']):
                #print('left')
                #update sample split col to fall on left side of split!
                #note lastNode's left node is the path node in the shortest path to the target
                #so we want to update our sample to fall on the left side of lastNode's value
                #so we the tree will take the left path instead of the right one
                sample[lastNode['col']] = lastNode['value'] - 0.001 #subtract to fall on left side
            #if path node is a right-node...
            elif('right' in lastNode and node == lastNode['right']):
                #update sample split col to fall on right side of split!
                sample[lastNode['col']] = lastNode['value'] + 0.001 #add to fall on right side
                #print('right')
    return sample

seed(1)
filename = 'iris.csv'
dataset = list(reader(open(filename)))

for row in dataset:
    for col in range(len(row) - 1):
        row[col] = float(row[col])

train = dataset[0:75]
test = dataset[75:]
classes = set([x[-1] for x in dataset])
maxDepth = 10
minSize = 15

tree = buildTree(train, maxDepth, minSize)
acc = treeTest(tree, test)

leaf = predict(tree, test[0])
newSample = attackTree(leaf, test[0][:], 'setosa', tree)
pred = predict(tree, newSample)

print(test[0]) #verginica
print(leaf['leaf'])
print(newSample)
print(pred['leaf'])