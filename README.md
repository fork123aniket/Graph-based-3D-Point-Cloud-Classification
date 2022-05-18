# Graph-based-3D-Point-Cloud-Classification
This repository aims at classify 3D points clouds with the help of graph neural networks (GNNs). Here, the raw input cloud is used as input to the GNN so that GNN will learn to capture meaningful local structures in order to classify the entire point set.
## Requirements
- `Python 3.9`
- `PyTorch 1.11`
- `torch-cluster 1.6`
- `PyTorch Geometric 2.0.4`
- `numpy 1.22.3`
- `matplotlib 3.5.1`
## Usage
### Data
The  `GeometricShapes` dataset from PyTorch Geomtric dataset collection is being used here. The dataset contains 40 different 2D and 3D geometric shapes such as cubes, spheres, pyramids, etc. Moreover, for each shape, there exists two different versions. One is used to train the GNN and the other one is used to evaluate its performance.
### Training and Testing
The current implementation provides three imperative functions:-
- `train()` to train the GNN-based point cloud classifier.
- `test()` to test the trained network.
- `visualize_points()` to accomplish three main visualization related tasks:-
    - to plot the position of points in the point cloud,
    - to plot the farthest points sampled in the point cloud,
    - to plot the *generated dynamic graph* of the point cloud.
- In addition to these, it also provides class `PPFNet` that implements ***Point Pair Feature*** network, ***a rotation-invariant*** version of ***PointNet++*** architecture.
- The average loss and associated test accuracy for the trained model are printed after every epoch. Moreover, upon completion of the training procedure, the best test accuracy for the trained model is also printed.
- All hyperparameters to control training and testing of the model are provided in the given `.py` file.
## Output Samples
`num_graphs` variable is set to 4 in the current implementation, however, it can be set to any number based on the requirements.
| Point Cloud Classes        | Position of Points           | Farthest Points Sampled  | Generated Dynamic Graph  |
| -------------------------- |:----------------------------:| ------------------------:| ------------------------:|
| 3d_cone      | ![alt text](https://github.com/fork123aniket/Graph-based-3D-Point-Cloud-Classification/blob/main/Images/11.PNG) | ![alt text](https://github.com/fork123aniket/Graph-based-3D-Point-Cloud-Classification/blob/main/Images/12.PNG) | ![alt text](https://github.com/fork123aniket/Graph-based-3D-Point-Cloud-Classification/blob/main/Images/13.PNG) |
| 3d_moon      | ![alt text](https://github.com/fork123aniket/Graph-based-3D-Point-Cloud-Classification/blob/main/Images/21.PNG) | ![alt text](https://github.com/fork123aniket/Graph-based-3D-Point-Cloud-Classification/blob/main/Images/22.PNG) | ![alt text](https://github.com/fork123aniket/Graph-based-3D-Point-Cloud-Classification/blob/main/Images/23.PNG) |
| 3d_icecream | ![alt text](https://github.com/fork123aniket/Graph-based-3D-Point-Cloud-Classification/blob/main/Images/31.PNG) | ![alt text](https://github.com/fork123aniket/Graph-based-3D-Point-Cloud-Classification/blob/main/Images/32.PNG) | ![alt text](https://github.com/fork123aniket/Graph-based-3D-Point-Cloud-Classification/blob/main/Images/33.PNG) |
| 3d_ico2      | ![alt text](https://github.com/fork123aniket/Graph-based-3D-Point-Cloud-Classification/blob/main/Images/41.PNG) | ![alt text](https://github.com/fork123aniket/Graph-based-3D-Point-Cloud-Classification/blob/main/Images/42.PNG) | ![alt text](https://github.com/fork123aniket/Graph-based-3D-Point-Cloud-Classification/blob/main/Images/43.PNG) |
