%% Continuous Wavelet Transform Filter Bank
% 
% Convolution network (2D) on time/frequency representations (scalograms) extracted 
% using the Continuous Wavelet Transform Filter Blank. 
%
% Time/frequency analysis of signals using the wavelet transform allows us to 
% extract information from the signal in both, frequency and time domains. The wavelet
% transform filter bank is also better than other methods as the transformation at different 
% scales achieves the best resolution trade-off between time and frequency (high resolution
% in the frequency domain for small frequency values | high resolution in the time 
% domain for for large frequency values).
%

%% Load data

data_dir = 'data/UCI HAR Dataset/';

% The signal
load(fullfile(data_dir, 'HAR_signals_labels.mat'))

% Location to saves the scalograms
scalograms_dir = 'data/data_scalograms/';


%% Plot scalogram example

% Time/frecuency analysis of a signal using the continuous 
% wavelet transform filter bank.

Fs = 50;  % Hz
signal_length = size(X_train, 2);
voices = 12;

fb = cwtfilterbank('SignalLength',signal_length,...
    'SamplingFrequency',Fs,...
    'VoicesPerOctave',voices);

sig = X_train(203, :, 2);
[cfs,frq] = wt(fb,sig);
t = (0:signal_length-1)/Fs;figure;pcolor(t,frq,abs(cfs))
set(gca,'yscale','log');shading interp;axis tight;
title('Scalogram');xlabel('Time (s)');ylabel('Frequency (Hz)')

%% Generate and save the scalograms (time/frequency representation) 

% The scalograms will be use in keras (python). Saving the images as arrays 
% on independent files we can load the images on small batches using 
% the keras.data_generator without risking running out of memory RAM when
% training the convolutional neural network on large datasets.
% 
% The arrays will have shapes 8x200x200, similar to a RGB picture of 200x200
% pixels but with 9 channels (1 for each signal component) instead of 3.
%
% More info about keras data generators: 
% https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

%% Create directories to save the scalograms


mkdir(fullfile(scalograms_dir, 'train'));
mkdir(fullfile(scalograms_dir, 'test'));


%% Continuous wavelet trasform filter bank

% Time/frequency analysis of the signals. The wavelet transform allows 
% us to extract information from the signal in both, frequency and time domains. 
% The wavelet transform it is also better than other methods as the transformation 
% at different scales achieves the best trade-off of resolution between time 
% and frequency (high resolution in the frequency domain for small frequency 
% values | high resolution in the time domain for large frequency values).

height = 224;
width = 224;
voices = 12;

fb = cwtfilterbank('SignalLength',signal_length,'VoicesPerOctave',voices);

% Plot example of a black&white scalogram image
cfs = abs(fb.wt(X_train(1,:,1)));
im = im2uint8(rescale(cfs));  % For color images --> ind2rgb(im2uint8(rescale(cfs)),jet(128));
im = imresize(im,[height, width]);  
imshow(im);

% Train
rows = size(X_train, 1);
num_channels = size(X_train, 3);  % Number of signal components: total_acc, body_acc, body_gyr (X, Y, Z)
for ii=1:rows
    array = zeros(height, width, 9);
    for jj=1:num_channels
        cfs = abs(fb.wt(X_train(ii,:, jj)));
        img = im2uint8(rescale(cfs));  % Color --> ind2rgb(im2uint8(rescale(cfs)),jet(128));
        img = imresize(img,[height, width]);
        array(:, :, jj) = img;
    end
    % Give the file a unique name, which must also contain the label
    imFileName = strcat(num2str(y_train(ii)),'_',num2str(ii),'.mat');  
    save(fullfile(scalograms_dir,'train',imFileName), 'array');
end

% Test
rows = size(X_test, 1);
num_channels = size(X_test, 3);  % Number of signal components: total_acc, body_acc, body_gyr (X, Y, Z)
for ii=1:rows
    array = zeros(height, width, 9);
    for jj=1:num_channels
        cfs = abs(fb.wt(X_test(ii,:, jj)));
        img = im2uint8(rescale(cfs));  % Color --> ind2rgb(im2uint8(rescale(cfs)),jet(128));
        img = imresize(img,[height, width]);
        array(:, :, jj) = img;
    end
    % Give the file a unique name, which must also contain the label
    imFileName = strcat(num2str(y_test(ii)),'_',num2str(ii),'.mat');  
    save(fullfile(scalograms_dir,'test',imFileName), 'array');
end


%% Triple images concatenating axis x, y and z of each signal component horizontally

% Create directories to save the scalograms
scalograms_dir = 'data/data_scalograms/';
mkdir(fullfile(scalograms_dir, 'train_triple'));
mkdir(fullfile(scalograms_dir, 'test_triple'));

height = 224;
width = 120;
voices = 12;

fb = cwtfilterbank('SignalLength',signal_length,'VoicesPerOctave',voices);

% Plot example of a black&white scalogram image
cfs_x = abs(fb.wt(X_train(1,:,1)));
im_x = im2uint8(rescale(cfs_x));
im_x = imresize(im_x,[height, width]); 

cfs_y = abs(fb.wt(X_train(1,:,2)));
im_y = im2uint8(rescale(cfs_y));
im_y = imresize(im_y,[height, width]); 

cfs_z = abs(fb.wt(X_train(1,:,3)));
im_z = im2uint8(rescale(cfs_z));
im_z = imresize(im_z,[height, width]); 
imshow([im_x im_y im_z]);

% Train
rows = size(X_train, 1);
num_channels = size(X_train, 3);  % Number of signal components: total_acc, body_acc, body_gyr (X, Y, Z)
for ii=1:rows
    array = zeros(height, width*3, 3);
    channel = 1;
    for jj=1:num_channels-1:3
        cfs_x = abs(fb.wt(X_train(ii,:, jj)));
        img_x = im2uint8(rescale(cfs_x));
        img_x = imresize(img_x,[height, width]);
        
        cfs_y = abs(fb.wt(X_train(ii,:, jj+1)));
        img_y = im2uint8(rescale(cfs_y));
        img_y = imresize(img_y,[height, width]);
        
        cfs_z = abs(fb.wt(X_train(ii,:, jj+2)));
        img_z = im2uint8(rescale(cfs_z));
        img_z = imresize(img_z,[height, width]);
        
        array(:, :, channel) = [img_x img_y img_z];
        channel = channel+1;
    end
    % Give the file a unique name, which must also contain the label
    imFileName = strcat(num2str(y_train(ii)),'_',num2str(ii),'.mat');  
    save(fullfile(scalograms_dir,'train_triple',imFileName), 'array');
end


% Test
rows = size(X_test, 1);
num_channels = size(X_test, 3);  % Number of signal components: total_acc, body_acc, body_gyr (X, Y, Z)
for ii=1:rows
    array = zeros(height, width*3, 3);
    channel = 1;
    for jj=1:num_channels-1:3
        cfs_x = abs(fb.wt(X_test(ii,:, jj)));
        img_x = im2uint8(rescale(cfs_x));
        img_x = imresize(img_x,[height, width]);
        
        cfs_y = abs(fb.wt(X_test(ii,:, jj+1)));
        img_y = im2uint8(rescale(cfs_y));
        img_y = imresize(img_y,[height, width]);
        
        cfs_z = abs(fb.wt(X_test(ii,:, jj+2)));
        img_z = im2uint8(rescale(cfs_z));
        img_z = imresize(img_z,[height, width]);
        
        array(:, :, channel) = [img_x img_y img_z];
        channel = channel+1;
    end
    % Give the file a unique name, which must also contain the label
    imFileName = strcat(num2str(y_test(ii)),'_',num2str(ii),'.mat');  
    save(fullfile(scalograms_dir,'test_triple',imFileName), 'array');
end
