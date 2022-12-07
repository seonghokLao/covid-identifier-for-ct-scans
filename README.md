# Covid Identifier for CT Scans: Team 70 Project
## Introduction/Background:
We aim to classify x-ray images of people’s chests to identify which images indicate that one is COVID-19 positive.
 
Through our research, we found that the occurrence of abnormal chest x-rays is highly corresponding to Covid-19 (Rousan et al., 2020). Inspired by the fact that chest imaging is not a first-line test for COVID-19 due to low diagnostic accuracy and confounding with other viral pneumonias, Nikolaou et al. showed that with the aid of machine learning, it is feasible to accurately distinguish COVID-19 from other viral pneumonia and healthy lungs on x-ray pictures (Nikolaou et al., 2021). After determination of the COVID-19 disease course and severity using chest X-ray (CXR) scoring system, Yasin & Gouda showed that Radiographic findings can be utilized to evaluate long-term effects because they are great predictors of the course of the COVID-19 disease (Yasin & Gouda, 2020).

We conclude that chest X-rays are applicable and reliable imaging techniques to identify the coronavirus. With the Chest X-Ray dataset, we aim to develop a Machine Learning Model to classify the X-Rays of Healthy vs. COVID-19 affected patients.
 
## Problem definition: 
The fastest and most effective ways to test Covid-19 are to use PCR and antigen. However, neither of these 2 approaches to covid testing visualize the infection. In this project, the “chest x-ray testing” approach for classifying Covid patients will be investigated if it is an accurate testing procedure with visualization.

## Data Collection:
Our dataset is collected from Kaggle:
<https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset>

This raw data(images) is a Kaggle dataset of CT scans collected from real SARS-CoV-2 patients in Sao Paulo Hospital, Brazil. The dataset contains 1252 CT scans of positive SARS-CoV-2 infection and 1230 scans of non-infected patients; in total 2482 CT-scan data.

Most of the images are shown as black and white CT scans of the infected lungs of different types of Covid-19 patients using X-rays. First of all, the vast majority of these lung images are well-balanced in size and proportion, making them suitable data for analysis. Second, these pictures are very clear, allowing us to efficiently and accurately analyze the data of the lungs of Covid-19 patients. Next, the number of these data sets is very large, and each label has nearly 1,000 images for us to test. Finally, the pictures of each label have been classified into different folders, and we do not need further large-scale screening and classification.

## Methods:
The packages we are going to use are numpy, pandas, and cv2 (OpenCV). Meanwhile, we are using Keras as the interface for TensorFlow. From TensorFlow, we’d like to import Sequential, Conv2D, MaxPooling2D, etc.   
The model is trained by 2 groups of labeled images: Covid and Non-Covid. The model will make classifications on input images.
 
## Results and Discussion:
### Model 1: CNN based on DenseNet121
Densely Connected Convolutional Networks, or DenseNet, is a method to increase the depth of deep convolutional network.  
It simplifies the connectivity pattern between the layers. With the nature of require fewer parameters, DenseNet avoids learning redundant feature map. In our first model, we use DenseNet121, which is a DenseNet model that performs image classification.

#### Preprocessing:
We divided our 2482-image dataset into **train**, **validation**, and **test** subfolders with a ratio of .7:.15:.15 using a package called **split-folders**.
Then, using the `flow_from_directory()` function of ImageDataGenerator from Keras, we resized the images to (64, 64, 3) before passing them into `train_ds` and `val_ds` variables. The data flowed into `train_ds` are shuffled for a possibly better training outcome. The resizing was necessary due to the input size of DenseNet121, which we will further discuss below.

#### Neural Network Details:
![image info](./assets/densenet_model_summary.png)

#### Training Procedure:
Our Dataset comes in 2 folders/labels–**COVID** and **NON-COVID**. We first split our dataset as mentioned in preprocessing, which randomly assigns images to **train**, **validation**, and **test** subfolders regardless of their label.  
Then, we build our model using DenseNet121 with pretrained weights obtained from ImageNet. Our model is backed by TensorFlow and Keras, and DenseNet121 is directly imported from `keras.applications`. As shown in the graph under Neural Network Details, our model has 7,219,414 trainable parameters. The following are some specifics of our model.  
`optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)`  
`model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])`  
As shown, our loss is calculated using categorical_crossentropy.  
We ran our training for 10 epochs for 50 steps per epoch. We then saved the model with the least validation loss throughout training.

#### Results:
![model loss](./assets/densenet_model_accuracy.png)![image info](./assets/densenet_model_loss.png)  
The number of epochs is positively correlated with accuracy and negatively correlated with loss as expected. Validation accuracy and loss seem to fluctuate a lot more than training accuracy and loss. Our model reaches its highest accuracy of 0.9704 after the last epoch which is also when the validation loss reaches its low at 0.1016. Previous model training seesions tend to produce the best model at epoch 9.

Here are sample predictions by passing in images extracted from our test dataset into our model.
![model predictions](./assets/densenet_predictions.png)  
Note that labels `[1,0]` and `[0,1]` represent **COVID** and **NON-COVID** respectively.  
Training our model using DenseNet121 with weights pretrained from ImageNet seems viable given the high accuracy in identifying covid. However, whether this is applicable to covid identification in society is yet to be decided.

---

### Model2: Vision Transformer (ViT)

A Vision Transformer (ViT) is a transformer that is designed specifically at vision processing tasks such as image recognition.  

Transformers measure attention—the relationships between pairs of input tokens. For images, the basic unit of analysis is the pixel. ViT computes relationships among pixels in a variety of tiny picture portions at a significantly lower cost.

#### Vision Transformer Overview:

![image info](./assets/vit_arch.png)

#### Training Procedure:

We first split data using the procedure in preprocessing. Using sklearn.model_selection.train_test_split on the training dataset, we found images belong to 2 classes: COVID and non-COVID. Then, we build our model from vit_base_patch16_224, with Adam as the optimizer. The model details are shown in the graph under Vision Transformer Overview. Fitting the model, we found that as epoch increases, accuracy increases while loss decreases. Applying our model to a few test images, we found all predictions match their corresponding labels.  

#### Results:
![model loss](./assets/vit_acc.png)![image info](./assets/vit_loss.png)  
The number of epochs is roughly positively correlated with accuracy and negatively correlated with loss. Validation accuracy and loss seem to fluctuate a lot more than training accuracy and loss. Meanwhile, the increase in epochs results in a smaller loss. With epochs between 8 and 10, the training accuracy is between 0.88 to 1, with a fair loss. Our validation loss reaches its low at epoch 9, which is where we saved our model. Our model ends up having about 88% validation accuracy.

## Conclusion:

Both our unsupervised and supervised learning algorithms respectively demonstrate the following accuracy data:
|Models|Test Accuracy|
|---|---|
|CNN|95%|
|ViT|87.5%|
|K-means|*AutoEncoder*: 60.76%|

Shown by the above accuracy data, the neural network represented by the CNN based on DenseNet121 is the most favored by our Covid-19 CT dataset, and its accuracy rate reaches nearly 95%. The accuracy of the ViT model follows closely, and the worst one is K-means. The latter two models may be able to improve accuracy in other ways (such as tuning hyperparameters), but it is still difficult to achieve the accuracy of neural networks such as CNN.

Obviously, the neural network has good performance and accuracy in chest CT-image classification for Covid-19 patients, and can help hospitals to identify and distinguish whether most patients are infected with Covid-19 efficiently. However, the neural network system still relies heavily on a large amount of data support to sustain the generalization of the model: if there is lack of many data, neural network model that relies on neurons is likely to be overfitting or underfitting, which may cause the model to be unable to recognize patients' chest CT samples effectively. Therefore, we may conduct more targeted research in the future to provide better data analysis support for both efficient and accurate identification of Covid-19 patients by improving the model or adopting a better one.
 
## Contribution Table:

| Person | Contributions |
| --- | --- |
| Seong Hok Lao | Data Sourcing and Cleaning, Model Selection (Model 1), Data Pre-Processing, Model Coding (Model 1), Results Evaluation and Analysis (Model 1), Model Coding (Model 3), Results Evaluation and Analysis (Model 3), Midterm Report, Final Report |
| Shijie Wang | Data Sourcing and Cleaning, Model Selection (Model 2), Data Pre-Processing, Model Coding (Model 2), Results Evaluation and Analysis (Model 2), Model Coding (Model 3), Results Evaluation and Analysis (Model 3), Midterm Report, Final Report |
| Haoyuan Wei | Results Evaluation and Analysis (Model 1), Midterm Report, Final Report |
| Qihang Hu | Midterm Report, Final Report |
| Zixiang Xu | Midterm Report, Model Coding (Model 3), Results Evaluation and Analysis (Model 3), Final Report |
 
## Link to Dataset:
<https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset>
## Link to Gantt Chart:
<https://docs.google.com/spreadsheets/d/1hWJDLwGgn0_DzW2z-kzkuP65He5jon1U/edit?usp=sharing&ouid=100331825873577630128&rtpof=true&sd=true>

## References
Nikolaou, V., Massaro, S., Fakhimi, S., Stergioulas, M., & Garn, W. (2021). COVID-19 diagnosis from chest x-rays: developing a simple, fast, and accurate neural network. Health information science and systems, 9(1), 36. https://doi.org/10.1007/s13755-021-00166-4

Rousan, L.A., Elobeid, E., Karrar, M., & Khader, Y. (2020). Chest x-ray findings and temporal lung changes in patients with COVID-19 pneumonia. BMC Pulmonary Medicine, 20(1), 245. https://doi.org/10.1186/s12890-020-01286-5

Yasin, R., & Gouda, W. (2020). Chest X-ray findings monitoring COVID-19 disease course and severity. Egyptian Journal of Radiology and Nuclear Medicine, 51(1), 193. https://doi.org/10.1186/s43055-020-00296-x
