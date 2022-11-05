# Chest X-Ray Covid Testing: Team 70 Project Proposal
## Introduction/Background:
 
We aim to classify x-ray images of people’s chests to identify which images indicate that one is COVID-19 positive.
 
Through our research, we found that the occurrence of abnormal chest x-rays is highly corresponding to Covid-19 (Rousan et al., 2020). Inspired by the fact that chest imaging is not a first-line test for COVID-19 due to low diagnostic accuracy and confounding with other viral pneumonias, Nikolaou et al. showed that with the aid of machine learning, it is feasible to accurately distinguish COVID-19 from other viral pneumonia and healthy lungs on x-ray pictures (Nikolaou et al., 2021). After determination of the COVID-19 disease course and severity using chest X-ray (CXR) scoring system, Yasin & Gouda showed that Radiographic findings can be utilized to evaluate long-term effects because they are great predictors of the course of the COVID-19 disease (Yasin & Gouda, 2020).
 
We conclude that chest X-rays are applicable and reliable imaging techniques to identify the coronavirus. With the Chest X-Ray dataset, we aim to develop a Machine Learning Model to classify the X-Rays of Healthy vs. COVID-19 affected patients.
 
## Problem definition:
 
The current most popular way to identify covid-19 is by PCR tests. We wanted to see if alternate methods such as X-Ray scans of people’s chests would be a reliable indicator of covid.
 
The fastest and most effective ways to test Covid-19 are to use PCR and antigen. However, neither of these 2 approaches to covid testing visualize the infection. In this project, the “chest x-ray testing” approach for classifying Covid patients will be investigated if it is an accurate testing procedure with visualization.
 
## Methods:
 
The packages we are going to use are numpy, pandas, and cv2 (OpenCV). Meanwhile, we are using Keras as the interface for TensorFlow. From TensorFlow, we’d like to import Sequential, Conv2D, MaxPooling2D, etc. 
The model is trained by 2 groups of labeled images: Covid Positive and Covid Negative. The model will make classifications on input images. We will use SIFT to detect and describe local features in images and compare the features in two images. Then, we will use Logistic Regression to predict the binary outcome for our observation of the two images. 
 
## Potential Results and Discussion:
 
We plan to use classification metrics from scikit-learn. More precisely, due to the binary nature of our image classifier, where we either identify the image as “Covid Positive” or “Covid Negative”, we plan to conduct binary classification using precision_recall_curve(). We expect that our model will reach at least 90% accuracy in identifying Covid. We will raise a discussion about what other models can be used and why we choose one over the others.
 
Contribution Table:

Person
Contributions
Shijie Wang
Introduction
Haoyuan Wei
Proposed Timeline, Methods, Problem Definition
Qihang Hu
Methods, Introduction, References
Seong Hok Lao
Introduction, Problem Definition, Potential Results, Video
Zixiang Xu
References & Contributions Table

 
## Link to Dataset:
https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset
Link to Gantt Chart:
https://docs.google.com/spreadsheets/d/1hWJDLwGgn0_DzW2z-kzkuP65He5jon1U/edit?usp=sharing&ouid=100331825873577630128&rtpof=true&sd=true

## References
Nikolaou, V., Massaro, S., Fakhimi, S., Stergioulas, M., & Garn, W. (2021). COVID-19 diagnosis from chest x-rays: developing a simple, fast, and accurate neural network. Health information science and systems, 9(1), 36. https://doi.org/10.1007/s13755-021-00166-4
Rousan, L.A., Elobeid, E., Karrar, M., & Khader, Y. (2020). Chest x-ray findings and temporal lung changes in patients with COVID-19 pneumonia. BMC Pulmonary Medicine, 20(1), 245. https://doi.org/10.1186/s12890-020-01286-5
Yasin, R., & Gouda, W. (2020). Chest X-ray findings monitoring COVID-19 disease course and severity. Egyptian Journal of Radiology and Nuclear Medicine, 51(1), 193. https://doi.org/10.1186/s43055-020-00296-x
