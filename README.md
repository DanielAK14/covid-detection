# covid-detection
## Image Classification of COVID-19 X-rays Using TensorFlow

This project was created to classify X-ray images given a combination of the Kaggle X-ray dataset and the COVID-19 Chest X-ray dataset collected by Dr. Joseph Paul Cohen of the Univeristy of Montreal. 

Two similar tasks were presented in this assignment. The first was to design and train a deep neural network to determine whether a given chest X-ray image was one of a healthy patient or of one with COVID-19. The second task required a deep neural network model be designed and trained to classify a chest X-ray into four different classes: normal, COVID-19, bacterial pneumonia, and viral pneumonia. Three different models were used for the second task classification in order to explore the validity of 
varying architectures. The report of my findings, including t-SNE feature visualization graphs, is attached below. Although this project was completed alone, plural pronouns are used in the report for clarity.


## 1 &nbsp; &nbsp; Model Architectures
For the first task, the model predicted a binary classification from the Kaggle Chest X-ray and COVID-19 Chest X-Ray datasets to determine whether the scan resembled a healthy (normal) patient or one with COVID-19. The top layer of our sequential model was retrofitted with the VGG16 model trained on ImageNet to obtain a base configuration of the weights. This submodel's parameters were set to untrainable in order to aid in test and validation time of our relatively simple classification. The VGG16 submodel's prediction layer was removed and its output was flattened to connect to a dense layer containing 128 neurons with reLU activation. Dropout with a rate of 0.20 was applied on both the output of the flattened layer and the output of the 128-neuron dense layer before connecting a single neuron with a sigmoid activation to act as our binary prediction classifier.

A model implementing three dense layers (including output) was also tried. The first dense layer after flattening was 512-neurons and reLU activated with a dropout rate of 0.33. The second was the same but with 128-neurons. After much experimentation no sizable difference between this larger model and our down-scaled model was observed.

For the second task our model trained and validated on the same dataset to predict whether a given X-ray showed a normal patient, one with Covid-19, one with viral pneumonia, or one with bacterial pneumonia. The second task's model borrows the first task model's architecture. A top VGG16 submodel connected to a flattening layer applied with a 0.20 dropout rate before being fed into a 128-neuron reLU activated dense layer, also with 0.20 droupout rate. The only difference is the last dense layer consisting of four (4) neurons (also sigmoid activated) since our classification is non-binary, separating the data into four cases instead of two.

## 2 &nbsp; &nbsp; Optimization and Loss
For both tasks, the Keras library's built in optimizer *Adam* was used. The learning rate of the first task was set to 0.0005 while the second task's learning rate was set to 0.0003. Since our first model only distinguished between two classes, its loss function was set to binary cross entropy. Since our second task's model had four classes instead of two, its loss function was set to categorical cross entropy. No additional regularization was implemented in either task's model.

## 3 &nbsp; &nbsp; Comparison of Task 2 Models
Three different models were compared when undergoing training and validation of the second task. The first model, as explained previously, implemented a top VGG16 submodel with a 128-neuron dense reLU activated hidden layer before the output. The second model instead used resNet50V2 as the base submodel. The third model used VGG19 as the top submodel to see if there were any large-scale differences in prediction by implementing a 19-layer deep submodel instead of a 16-layer deep submodel. All three models used a dropout rate of 0.20 before and after the appended 128-neuron deep hidden layer. All three models used pretrained and untrainbable weights based on the ImageNet dataset. For all task 2 models tested against each other, the learning rate was set to 0.0003. The *Adam* optimization function was chosen as was the categorical cross entropy loss function since the amount of desired classifications per each model remained constant.

#### VGG16
The VGG16 submodel is comprised of a series of convolutional layers with intermittent pooling-layers making it a common model to use for image-classification. Our abbreviated VGG16 submodel base is comprised of 13 convolutional layers and 5 max pooling layers:

![VGG16](https://user-images.githubusercontent.com/32208581/155856847-f35f8605-55e6-425d-988e-669d226f8734.png)

#### resNet50V2
resNet50V2 was chosen as a comparison submodel since its architecture differs greatly from both VGG16 and VGG19. resNet50V2 is a deep residual neural network with unique 'skip connections' that pass input both to the adjacent layer *and* to later layers of the model. In essence, the model reserves old unchanged input and passes it to layers that would not normally see it. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun equate such models to recurrent neural networks without temporal-based weighting in their 2015 paper *Deep Residual Learning for Image Recognition* [1].

#### VGG19
VGG19 was chosen as a third comparison model to test whether the additional 3 convolutional layers, a total of 16, would pose significant improvement over its 13-convolutional-layer VGG16 counterpart. Both submodels have 5 max pooling layers.

![VGG19](https://user-images.githubusercontent.com/32208581/155856862-9cca66a8-1f81-423e-afde-dfac746eda99.png)

## 4 &nbsp; &nbsp; Accuracy and Loss Results
#### Task 1
The VGG16-based model used in Task 1 achieved an accuracy of 95-100%. As seen in the graph below, upon the last epoch of training and validation, the model achieved a training accuracy 92.5% with a test accuracy of 100%. Most all test accuracies hovered between the 90% and 100% threshold with the exception of early tests and a sudden dip at epoch 18.

As is shown in the graph of loss values over 40 epochs, our VGG16 model achieved a training loss of 0.151 and a validation loss of 0.0319 as recorded on the last epoch. At the 18<sup>th</sup> epoch of the graphed accuracy, we see a spike in the loss graph with a validation loss value of 0.2011.

![Task1_Graphs](https://user-images.githubusercontent.com/32208581/155856871-d8748d4c-afb5-47f5-9b57-b1dd7f828180.png)

#### Task 2
As to be expected, the plots seen below for VGG16 and VGG19 look very similar since they are based off of near-identical models. ResNet50V2, however, is much more sporadic in both its test accuracy and test values. Perhaps this is to do with resNet5V0V2's 'skip connections'; later layers are getting more varied inputs (a sequential input and an untouched input from the bargaining layers of the model) thus the accuracy and loss varies greatly. Even with drastic peaks and valleys in resNet50V2, the test accuracy matched that of VGG16: 72%. VGG19 approached a 70% test accuracy on its hundredth epoch, but varied greatly both in the beginning of training and throughout the bulk of future epochs. VGG16 achieved a loss of 0.6988, resNet50V2 achieved a loss of 0.9349, VGG19 achieved a loss of 0.9466. For a simple classification task like that of Task 2, I would most likely choose the model based on VGG16. Its end performance was comparable to resNet50V2, and its pattern of accuracy and loss values stayed semi-consistent - all while requiring less parameters and less time to train (using an Intel i7-6700HQ CPU @ 2.60GHz).

![Task2_Graphs](https://user-images.githubusercontent.com/32208581/155856919-baf34a67-580c-4216-8b70-5316ea6f8834.png)

## 5 &nbsp; &nbsp; t-SNE Visualizations
#### Task 1
The first dense layer of the model, 128-neuron large, was used as an intermediate model to test acceptable feature extraction. As seen in the plot below, the clusters of features can be easily separated into two segments (with the exception of a single normal case among the COVID-19 cases). Based on this visualization our model seems to be working properly, finding and grouping similarities between the two classes.

![Task1_TSNE](https://user-images.githubusercontent.com/32208581/155857066-1863574c-53f8-4086-8e7f-2163ad3fc93b.png)

#### Task 2
For all models used, the respective 128-neuron large dense layer was used as the intermediate model during feature extraction. Below is the graphed result of t-distributed stochastic neighbor embedding on our model using VGG16 as the base sub-model. Most features are well-defined and well-clustered with the exception of the two classes of pneumonia. This is to be expected since viral and bacterial pneumonia look very similar to one another in an X-ray. This is most probably the reason our best model couldn't surpass a test accuracy of 80%; although it can differentiate between pneumonia, COVID-19, and a normal patient, it has trouble differentiating between the *nature* of the pneumonia that it has succesfully classified.

![Task2_TSNE1](https://user-images.githubusercontent.com/32208581/155857140-f3e2c800-73e7-49b6-80cb-2ddf89dd371c.png)

Below is the plotted t-SNE for the model using resNet50V2 as its base submodel. Once again, the features are well-defined and well-clustered with the exception of bacterial and viral pneumonia. It seems as though changing only the submodel does not result in an extreme change of feature extraction: the same errors are present in both models.

![Task2_TSNE2](https://user-images.githubusercontent.com/32208581/155857169-d3162713-2e21-47fe-be05-37b499a9adea.png)

Below is the plotted t-SNE for the model using VGG19 as its base submodel. Unlike the other two models, the distinction between bacterial pneumonia and Covid-19 becomes muddled in some cases. It can now be seen that every model implemented had trouble determining a distinction between viral pneumonia and bacterial pneumonia. However, every model tested successfully differentiates between normal x-rays, ones of a Covid-19 patient, and ones of pneumonia patients as a whole.

![Task2_TSNE3](https://user-images.githubusercontent.com/32208581/155857200-403d7817-3c28-4fde-839b-2a40d72c72c8.png)

## 6 &nbsp; &nbsp; References
[1] He, K., Zhang, X., Ren, S., Sun, J. Deep Residual Learning for Image Recognition. https://arxiv.org/pdf/1512.03385.pdf, 2-3, 2015.


