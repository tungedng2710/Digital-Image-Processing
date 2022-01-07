clc; close all;
num_classes = 10;

[XTrain,YTrain,anglesTrain] = digitTrain4DArrayData;
[XTest,YTest,anglesTest] = digitTest4DArrayData;

% idx = 2834;
% I = XTest(:, :, :, idx);
I = XTest;
dlI = dlarray(I, 'SSCB');

load ('model.mat', 'model');
plot(model);

% Test Feature Extraction
backbone = removeLayers(layerGraph(model),'output');
backbone = removeLayers(backbone,'softmax');
% backbone = removeLayers(backbone,'fc');
% backbone = removeLayers(backbone,'relu_3');
plot(backbone);
backbone = dlnetwork(backbone);
output = forward(backbone,dlI);

