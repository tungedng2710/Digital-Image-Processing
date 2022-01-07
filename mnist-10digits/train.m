clc; clear; close all;

% Get dataset
num_classes = 10;

[XTrain,YTrain,anglesTrain] = digitTrain4DArrayData;
[XTest,YTest,anglesTest] = digitTest4DArrayData;

% Network
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    fullyConnectedLayer(num_classes)
    softmaxLayer
    classificationLayer];

% Training options
options = trainingOptions('sgdm', ...
    'MaxEpochs',10, ...
    'ValidationData',{XTest,YTest}, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
% Train network
lgraph = layerGraph(layers);
net = trainNetwork(XTrain,YTrain,lgraph,options);

% Save network
model = net;
save ('model.mat', 'model');
