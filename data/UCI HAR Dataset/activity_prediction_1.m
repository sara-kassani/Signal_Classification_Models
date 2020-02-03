


N = 128;
half_N = floor(N/2);
f_s = 50;
t_n = 2.56;
T = t_n / N;
sample_rate = 1 / f_s;

numOfPeaks = 5;


%% Apply


func_list = {@fft_peaks, @psd_peaks, @corr_peaks};
signal_list = {bodyaccxtrain, bodyaccytrain, bodyaccztrain, bodygyroxtrain, bodygyroytrain, bodygyroztrain, totalaccxtrain, totalaccytrain, totalaccztrain};


signal_matrix = {};
for i=1:length(signal_list)
    
    row_matrix= {};
    for row=1:height(signal_list{i})
        
        peaks_array = [];
        for func_index=1:length(func_list)
            func = func_list{func_index};
            
            peaks = func(table2array(signal_list{i}(row, 1:end)), N, f_s, half_N, numOfPeaks);
            peaks_array = [peaks_array peaks];
        
        end
        row_matrix{row} = peaks_array;
        
    end
    signal_matrix{i} = row_matrix;
    
end



% Functions

function peaks = fft_peaks(signal, ~, f_s, half_N, numOfPeaks)
    
    % x
    signalamp = abs( fft( signal )/f_s );
    signalamp = signalamp(1:half_N);

    % y
    hz = linspace(0,52,half_N);

    % peaks
    [~, peak_indexes] = findpeaks(signalamp);

    peaks_x = signalamp(peak_indexes); 
    peaks_y = hz(peak_indexes);
    
    if length(peak_indexes) < numOfPeaks
        peaks_x = [peaks_x zeros(1, numOfPeaks)];
        peaks_y = [peaks_y zeros(1, numOfPeaks)];
    end
    
    peaks = [peaks_x(1:numOfPeaks) peaks_y(1:numOfPeaks)];

end


function peaks = psd_peaks(signal, N, ~, half_N, numOfPeaks)

    % x
    signalpower = abs( fft( signal )/N ).^2;
    signalpower = signalpower(1:half_N);

    % y
    hz = linspace(0,52,half_N);

    % peaks
    [~, peak_indexes] = findpeaks(signalpower);

    peaks_x = signalpower(peak_indexes); 
    peaks_y = hz(peak_indexes);
    
    if length(peak_indexes) < numOfPeaks
        peaks_x = [peaks_x zeros(1, numOfPeaks)];
        peaks_y = [peaks_y zeros(1, numOfPeaks)];
    end
    
    peaks = [peaks_x(1:numOfPeaks) peaks_y(1:numOfPeaks)];
end


function peaks = corr_peaks(signal, N, f_s, ~, numOfPeaks)

    [autocor,lags] = xcorr(signal,N,'coeff');

    % extract only the positive part
    autocor = autocor(floor(length(autocor)/2):end); %x
    lags = lags(floor(length(lags)/2):end);

    lag = lags/(f_s); %y

    [~, peak_indexes] = findpeaks(autocor);
    
    peaks_x = autocor(peak_indexes); 
    peaks_y = lag(peak_indexes);
    
    if length(peak_indexes) < numOfPeaks
        peaks_x = [peaks_x zeros(1, numOfPeaks)];
        peaks_y = [peaks_y zeros(1, numOfPeaks)];
    end
    
    peaks = [peaks_x(1:numOfPeaks) peaks_y(1:numOfPeaks)];
    disp(peaks)

end























%% TODO
%- add more features (variance coheficient, signal noise...)

















