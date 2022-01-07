clc; clear; close all;
num_classes = 10;

[XTrain,YTrain,anglesTrain] = digitTrain4DArrayData;
[XTest,YTest,anglesTest] = digitTest4DArrayData;

idx = 2834;
I = XTest(:, :, :, idx);
dlI = dlarray(I, 'SSCB');
% imshow(sample);

load ('model.mat', 'model');
plot(model);

backbone = removeLayers(layerGraph(model),'classoutput');
backbone = removeLayers(backbone,'softmax');
% backbone = removeLayers(backbone,'fc');
% backbone = removeLayers(backbone,'relu_3');
plot(backbone);
backbone = dlnetwork(backbone);
output = forward(backbone,dlI);
