import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.lang.Math;

/**
 * Fill in the implementation details of the class DecisionTree using this file. Any methods or
 * secondary classes that you want are fine but we will only interact with those methods in the
 * DecisionTree framework.
 */
public class DecisionTreeImpl {
    public DecTreeNode root;
    public List<List<Integer>> trainData;
    public int maxPerLeaf;
    public int maxDepth;
    public int numAttr;

    DecisionTreeImpl(List<List<Integer>> trainDataSet, int mPerLeaf, int mDepth) {
        this.trainData = trainDataSet;
        this.maxPerLeaf = mPerLeaf;
        this.maxDepth = mDepth;
        if (this.trainData.size() > 0) this.numAttr = trainDataSet.get(0).size() - 1;
        this.root = buildTree(trainData, 0);
    }
    
    private int helperGetAttrClass (List<List<Integer>> dataList) {
        int count_zero = 0;
        int count_one = 0;
        for(List<Integer> instance : dataList) {
            if (instance.get(numAttr) == 1) {
                count_one++;
            }
            else {
                count_zero++;
            }
        }
        if (count_zero <= count_one) {
            return 1;
        }
        else {
            return 0;
        }
    }
    
    private double [][] infoGainCalc (List<List<Integer>> trainDataSet) {
        double infoGainList[][] = new double[numAttr][10];
        for(int i = 0; i < numAttr; i++) {
            for(int j = 1; j < 10; j++) {
                List<List<Integer>> leftLowList = new ArrayList<List<Integer>>();
                List<List<Integer>> rightHighList = new ArrayList<List<Integer>>();
                double calcEntropy = calcEntropy(trainDataSet);
                for(List<Integer> instance : trainDataSet) {
                    if (instance.get(i) <= j) {
                        leftLowList.add(instance);
                    }
                    else {
                        rightHighList.add(instance);
                    }
                    
                    double entropyLeftLowList = calcEntropy(leftLowList);
                    double entropyRightHighList = calcEntropy(rightHighList);
                    double conditionalEntropy = (double)leftLowList.size()/trainDataSet.size() * entropyLeftLowList + (double)rightHighList.size()/trainDataSet.size() * entropyRightHighList;
                    infoGainList[i][j] = calcEntropy - conditionalEntropy;
                }
            }
        }
        return infoGainList;
    }
    
    private double calcEntropy(List<List<Integer>> trainData) {
        if (trainData.size() == 0)
            return 0;
        double count_0 = 0;
        double count_1 = 0;
        for(int i = 0; i < trainData.size(); i++) {
            if (trainData.get(i).get(numAttr) == 1) {
                count_1++;
            }
            else {
                count_0++;
            }
        }
        double probability_0 = count_0/(count_0 + count_1);
        double probability_1 = count_1/(count_0 + count_1);
        double calcLog2Prob0 , calcLog2Prob1;
        if (probability_0 != 0) {
            calcLog2Prob0 = Math.log(probability_0)/Math.log(2);            
        }
        else {
            calcLog2Prob0 = 0.0;
        }
        if (probability_1 != 0) {
            calcLog2Prob1 = Math.log(probability_1)/Math.log(2);            
        }
        else {
            calcLog2Prob1 = 0.0;
        }
        double result = -probability_0 * calcLog2Prob0
                        - probability_1 * calcLog2Prob1;
        return result;
    }
    
    private DecTreeNode buildTree(List<List<Integer>> dataList, int depth) {
        int c = 0;
        for(int i = 0; i < dataList.size(); i++) {
            if (dataList.get(i).get(numAttr) == 0) {
                c++;
            }
        }
        if (depth == maxDepth || dataList.size() <= maxPerLeaf || c == 0 || c == dataList.size()) {
            return new DecTreeNode(helperGetAttrClass(dataList), -1, -1);
        }
        double [][] infoGainList = infoGainCalc(dataList);
        double bestIGVal = Double.NEGATIVE_INFINITY;
        int bestAttr = -1;
        int bestT = -1;
        for(int i = 0; i < infoGainList.length; i++) {
            for(int j = 1; j < infoGainList[i].length; j++) {
                if (infoGainList[i][j] > bestIGVal) {
                    bestIGVal = infoGainList[i][j];
                    bestAttr = i;
                    bestT = j;
                }
            }
        }
        if (bestIGVal == 0) {
            return new DecTreeNode (helperGetAttrClass(dataList), -1, -1);
        }
        List<List<Integer>> leftLowList = new ArrayList<List<Integer>>();
        List<List<Integer>> rightHighList = new ArrayList<List<Integer>>();
        
        for (List<Integer> instance : dataList) {
            if (instance.get(bestAttr) <= bestT) {
                leftLowList.add(instance);
            }
            else {
                rightHighList.add(instance);
            }
        }
        
        DecTreeNode rootNode = new DecTreeNode(-1, bestAttr, bestT);
        rootNode.left = buildTree(leftLowList, depth + 1);
        rootNode.right = buildTree(rightHighList, depth + 1);
        return rootNode;
        
        
    }      
         
    public int classify(List<Integer> instance) {
        DecTreeNode rootNode = this.root;
        while(!rootNode.isLeaf()) {
            if (instance.get(rootNode.attribute) <= rootNode.threshold) {
                rootNode = rootNode.left;
            } else {
                rootNode = rootNode.right;
            }
        }
        return rootNode.classLabel;
    }
    
    public void printTree() {
        printTreeNode("", this.root);
    }

    public void printTreeNode(String prefixStr, DecTreeNode node) {
        String printStr = prefixStr + "X_" + node.attribute;
        System.out.print(printStr + " <= " + String.format("%d", node.threshold));
        if(node.left.isLeaf()) {
            System.out.println(" : " + String.valueOf(node.left.classLabel));
        }
        else {
            System.out.println();
            printTreeNode(prefixStr + "|\t", node.left);
        }
        System.out.print(printStr + " > " + String.format("%d", node.threshold));
        if(node.right.isLeaf()) {
            System.out.println(" : " + String.valueOf(node.right.classLabel));
        }
        else {
            System.out.println();
            printTreeNode(prefixStr + "|\t", node.right);
        }
    }
    
    public double printTest(List<List<Integer>> testDataSet) {
        int numEqual = 0;
        int numTotal = 0;
        for (int i = 0; i < testDataSet.size(); i ++)
        {
            int prediction = classify(testDataSet.get(i));
            int groundTruth = testDataSet.get(i).get(testDataSet.get(i).size() - 1);
            System.out.println(prediction);
            if (groundTruth == prediction) {
                numEqual++;
            }
            numTotal++;
        }
        double accuracy = numEqual*100.0 / (double)numTotal;
        System.out.println(String.format("%.2f", accuracy) + "%");
        return accuracy;
    }
}
