clc; close all; clear;
num_classes = 10;

[XTrain,YTrain,anglesTrain] = digitTrain4DArrayData;
[XTest,YTest,anglesTest] = digitTest4DArrayData;

% idx = 2834;
% I = XTest(:, :, :, idx);
I = XTest;
dlI = dlarray(I, 'SSCB');

load ('model.mat', 'model');
[labels,scores] = classify(model,XTest);
labels = double(labels);

% Test Feature Extraction
backbone = removeLayers(layerGraph(model),'output');
backbone = removeLayers(backbone,'softmax');
% backbone = removeLayers(backbone,'fc');
% backbone = removeLayers(backbone,'relu_3');
% plot(backbone);

backbone = dlnetwork(backbone);

feat_vecs = forward(backbone,dlI);
feat_vecs = double(extractdata(feat_vecs));

rng default % for reproducibility
feat_vecs = tsne(feat_vecs','Algorithm','barneshut','NumPCAComponents',2);

data = [feat_vecs, labels];

cmap = colormap(parula(10));
fh = figure;
ah = axes(fh);
hold(ah,'on');
for i=1:5000
    k = data(i,3);
    plot(ah, data(i,1),data(i,2), '.', 'Color',cmap(k,:));
end

