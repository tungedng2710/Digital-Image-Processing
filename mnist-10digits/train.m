clc; clear; close all;

% Get dataset
num_classes = 10;

[XTrain,YTrain,anglesTrain] = digitTrain4DArrayData;
[XTest,YTest,anglesTest] = digitTest4DArrayData;
% idx = 2000;
% sample = XTrain(:,:,:,idx);
% imshow(sample);

% Network
layers = [
    imageInputLayer([28 28 1],'Name','input')
    
    convolution2dLayer(3,8,'Padding','same','Name','2dconv_1')
    batchNormalizationLayer('Name','bn_1')
    reluLayer('Name','relu_1')
    
    maxPooling2dLayer(2,'Stride',2, 'Name','maxpool_1')
    
    convolution2dLayer(3,16,'Padding','same','Name','2dconv_2')
    batchNormalizationLayer('Name','bn_2')
    reluLayer('Name','relu_2')   
    
    maxPooling2dLayer(2,'Stride',2, 'Name','maxpool_2')
    
    convolution2dLayer(3,32,'Padding','same','Name','2dconv_3')
    batchNormalizationLayer('Name','bn_3')
    reluLayer('Name','relu_3')
    
    fullyConnectedLayer(64,'Name','fc1')
    batchNormalizationLayer('Name','bn_4')
    reluLayer('Name','relu_4')

    fullyConnectedLayer(num_classes,'Name','fc2')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','output')];

% Training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.1, ...
    'MaxEpochs',30, ...
    'ValidationData',{XTest,YTest}, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'MiniBatchSize',512,...
    'LearnRateDropFactor',0.7, ...
    'LearnRateDropPeriod',60, ...
    'Plots','training-progress');
% Train network
lgraph = layerGraph(layers);
net = trainNetwork(XTrain,YTrain,lgraph,options);

% Save network
model = net;
save ('model3.mat', 'model');

