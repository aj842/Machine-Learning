
%%Multiclass Naive Bayes Classifier
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
data_file = 'CTG.csv';
dataset_file = 'CTG.mat';

if(exist(dataset_file, 'file'))
    %Loading data file if there is already one created before
    load(dataset_file);
else
    %Loading dataset from csv 
    data = csvread(data_file, 2);
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
training_sample = training_data(:, 1:end-2);
training_label = training_data(:, end);

%Creating a matrix of all the unique label for the given data sample
labels = unique(data(:, end));

for l = 1 : size(labels)
    %Creating a temporary matrix using the training sample matrix
    temp_matrix = [];
    for z = 1 : size(training_sample)
        if training_label(z, 1) == labels(l)
            temp_matrix = [temp_matrix; training_sample(z, :)];
        end
    end
    
    %calculating prior matrix for the temporary matrix
    prior_temp_matrix(l, :) = size(temp_matrix, 1)/size(training_sample, 1);
    
    %calculating mean matrix for the temporary matrix
    mean_temp_matrix(l, :) = mean(temp_matrix);
    
    %calculating standard deviation matrix for the temporary matrix
    sigma_temp_matrix(l, :) = std(temp_matrix);
end

%Seperating the testing data sample and the label for the testing sample
testing_sample = testing_data(:,1 : end-2);
testing_label = testing_data(:, end);

%finding narmalized matrix and calculating posterior 
for k = 1 : size(labels)
    normalized = normpdf(testing_sample, mean_temp_matrix(k, :), sigma_temp_matrix(k, :));
    posterior(:, k) = prod(normalized, 2)*prior_temp_matrix(k);
end

%Finding the resultant labels for the testing sample
result = [];
for j=1 : size(testing_label, 1)
    [temp, label_max] = max(posterior(j, :), [], 2);
    result = [result; labels(label_max)];
end

%Calculating the number of data samples classified correctly for
%calculating accuracy
t=0;
for b = 1 : size(testing_label, 1)
    if testing_label(b) == result(b)
        t = t+1;
    end
end

%Calculating accuracy of the model
acc= (t)/size(testing_sample, 1);
