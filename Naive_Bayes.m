%%Naive Bayes Classifier
%  Author: Abhilasha Jayaswal
%  Homework 3
%  CS 613
%  Machine Learning

%Clearing the Command Window
clc;

% Clearing all variables
clear; 
close all;

% Reading the data
data_file = 'spambase.data';
dataset_file = 'spambase.mat';

if(exist(dataset_file, 'file'))
    %Loading data file if there is already one created before
    load(dataset_file);
else
    %Loading dataset from csv 
    data = csvread(data_file);
    %Saving the dataset to datafile
    save(dataset_file,'data');
end
% Getting the training and testing dataset
%Randomizing the dataset
rng(0);
data = data( randperm( length(data) ), : );

%Using first 2/3 of the dataset for training
training_data_size = ceil( length(data) * 2 / 3 );
training_data = data(1 : training_data_size, :);

% Using the remaining 1/3 dataset for testing
testing_data = data(training_data_size+1 : end, :);

% Standardizing the data
% Calculating the mean and standard deviation of the training dataset
data_mean = mean(training_data(:, 1:end-1));
data_sigma = std(training_data(:, 1:end-1));

% Standardizing the Data
training_data = [(training_data(:, 1:end-1) - data_mean) ./ data_sigma, training_data(:, end)];
testing_data = [(testing_data(:, 1:end-1) - data_mean) ./ data_sigma, testing_data(:, end)];

%Seperating the training data sample and the label for the sample
training_sample = training_data(:, 1:end-1);
training_label = training_data(:, end);

% Creating arrays to store spam data and non spam data seperately
class_spam = [];
class_non_spam = [];

%Dividing the training data into spam and non spam based on the label
for l=1 : size(training_sample, 1)
    if training_label(l, 1) ==1
        class_spam = [class_spam; training_sample(l, :)];
    else
        class_non_spam = [class_non_spam; training_sample(l, :)];
    end
end

%Calculating Prior for the two classes of data spam and non spam
prior_spam = size(class_spam, 1)/size(training_sample, 1);
prior_non_spam = size(class_non_spam, 1)/size(training_sample, 1);

%Calculating the mean and standard deviation for class spam
class_spam_mean = mean(class_spam(:, 1:end));
class_spam_sigma = std(class_spam(:, 1:end));

%Calculating the mean and standard deviation for class non spam
class_non_spam_mean = mean(class_non_spam(:, 1:end));
class_non_spam_sigma = std(class_non_spam(:, 1:end));

% Seperating the testing data sample and the label for the sample
testing_sample = testing_data(:, 1:end-1);
testing_label = testing_data(:, end);

%Normalizing the testing data
normalized_spam = normpdf(testing_sample, class_spam_mean, class_spam_sigma);
normalized_non_spam = normpdf(testing_sample, class_non_spam_mean, class_non_spam_sigma);

%Calculating posterior of the normalized spam and non spam matrix
spam_posterior = (prod(normalized_spam, 2)) * prior_spam;
non_spam_posterior = (prod(normalized_non_spam, 2)) * prior_non_spam;

%Creating the result matrix
result = [];

%storing the class of the testing sample in reult
for k = 1: size(testing_label, 1)
    if spam_posterior(k) > non_spam_posterior(k)
        result = [result; 1];
    else
        result = [result; 0];
    end
end

T_pos = 0;
T_neg = 0;
F_pos = 0;
F_neg = 0;

%Calculating the value of true positives, true negative, false positives and false
%negatives
for j=1 : size(testing_label, 1)
    if testing_label(j) == 1
        if result(j) == 1
            T_pos = T_pos+1;
        else
            F_neg = F_neg + 1;
        end
    else
        if result(j) == 1
            F_pos = F_pos+1;
        else
            T_neg = T_neg + 1;
        end
    end
end

%Evaluating The Classisfier
acc = (T_pos + T_neg)/size(testing_label,1);
prec = T_pos/(T_pos + F_pos);
recall = T_pos/(T_pos + F_neg);
F_mes = (2*prec*recall)/(prec+recall);


