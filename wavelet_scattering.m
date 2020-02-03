
%% Wavelet Time Scattering + SVM Classifier
%
% Wavelet time scattering consist on using wavelets transforms to extract low-variance 
% features from signals, time series data or image data. Wavelet time scattering yields signal 
% representations insensitive to shifts in the input signal without sacrificing class 
% discriminability.
%
%% Data

data_dir = './data/UCI HAR Dataset/';
save_dir = './data/wavelet_scattering/';  % directory where to save the extracted features

% Load data
load(fullfile(data_dir,'HAR_signals_labels.mat'));


%% Create the wavelet filter banks

fs = 50;
N = size(X_train,2);
invariance_scale = 1;  % seconds
quality_factors = [8 1];  % wavelets per octave

voices = 8;  % voices per 

sf = waveletScattering('SignalLength',N, 'InvarianceScale',1,'SamplingFrequency',fs,...
                       'QualityFactors',quality_factors);


%% Scattering coefficients

all_features_train = [];
all_features_test = [];
for ii=1:9
    sig_comp_train = X_train(:,:,ii);
    sig_comp_test = X_test(:, :, ii);

    % Apply the decomposition framework
    scat_features_train = featureMatrix(sf,sig_comp_train');
    scat_features_test = featureMatrix(sf,sig_comp_test');
    
    % Reshape the multisignal scattering transform into a matrix where each column
    % corresponds to a scattering path and each row is a scattering time window.
    Nwin = size(scat_features_train,2);
    scat_features_train = permute(scat_features_train,[2 3 1]);
    scat_features_train = reshape(scat_features_train,...
    size(scat_features_train,1)*size(scat_features_train,2),[]);

    scat_features_test = permute(scat_features_test,[2 3 1]);
    scat_features_test = reshape(scat_features_test,...
    size(scat_features_test,1)*size(scat_features_test,2),[]);

    all_features_train = [all_features_train scat_features_train];
    all_features_test = [all_features_test scat_features_test];  
end

save(fullfile(save_dir,'train_WTS_features.mat'), 'all_features_train')
save(fullfile(save_dir,'test_WTS_features.mat'), 'all_features_test')

% The wavelet decomposition framework extracts 27 features of each signal component in 
% 8 different scattering time windows: 
%
%   27 features * 9 sign_comp * 8 Nwin --> 1944 total features 


%% Classification
%
% For classification we use a multi-class SVM with a quadratic kernel.
%
% We have features from 8 different scattering time windows, instead of
% passing them together to the classifier we are going to try to classify
% each time window independently and then apply majority vote to get the
% final result. On the "1_Demostration.py" notebook I put another example
% joining all the features together instead. The accuracies achieved are
% similar.
%
% Because for each signal we obtained Nwin scattering windows, we need to create 
% labels to match the number of windows. The helper function createSequenceLabels 
% does this based on the number of windows.

y_train = transpose(y_train);
y_test = transpose(y_test);

sequence_labels_train = repelem(y_train,Nwin,1);
sequence_labels_test = repelem(y_test,Nwin,1);

save(fullfile(save_dir,'train_WTS_labels.mat'), 'sequence_labels_train')
save(fullfile(save_dir,'test_WTS_labels.mat'), 'sequence_labels_test')


%% Model (SVM)

rng(1);
template = templateSVM(...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);

%% Fit

model = fitcecoc(...
     all_features_train, ...
     sequence_labels_train, ...
     'Learners', template, ...
     'Coding', 'onevsone', ...
     'ClassNames', [1,2,3,4,5,6]);

%% Predict

predLabels = predict(model,all_features_test);

% Voting
classes = categorical([1,2,3,4,5,6]);

addpath('utils/') 
[TestVotes,TestCounts] = helperMajorityVote(predLabels,y_test,classes);

% The error rate, or loss, is estimated using 5-fold cross validation.
testaccuracy = sum(eq(TestVotes,categorical(y_test)))/numel(y_test)*100;
fprintf('The test accuracy is %2.2f percent. \n',testaccuracy);
testconfmat = confusionmat(TestVotes,categorical(y_test));
testconfmat
