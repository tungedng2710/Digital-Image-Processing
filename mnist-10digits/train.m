clc; clear; close all;

% Get dataset
num_classes = 10;

[XTrain,YTrain,anglesTrain] = digitTrain4DArrayData;
[XTest,YTest,anglesTest] = digitTest4DArrayData;
idx = 1999;
sample = XTrain(:,:,:,idx);
imshow(sample);

% Network
layers = [
    imageInputLayer([28 28 1],'Name','input')
    
    convolution2dLayer(3,8,'Padding','same','Name','2dconv_1')
    batchNormalizationLayer('Name','bn_1')
    reluLayer('Name','relu_1')
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same','Name','2dconv_2')
    batchNormalizationLayer('Name','bn_2')
    reluLayer('Name','relu_2')   
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same','Name','2dconv_3')
    batchNormalizationLayer('Name','bn_3')
    reluLayer('Name','relu_3')
    
    fullyConnectedLayer(num_classes,'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','output')];

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

