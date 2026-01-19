clc;
% Test parameters
fs = 10000;           % Sampling frequency (Hz)
duration = 0.5;        % 1 second
n_x = fs * duration; % 1000 samples
t = (0:n_x-1)' / fs;

% Signals
freq1 = 250 + 50*t./duration;
freq2 = 300 - 250*t;

x = sin(2*pi*freq1.*t);

% Filter parameters
freq_vec = freq1;
p = [1 2 3];

% Test with three bandwidths
bandwidths = [5];

% Call filter for each bandwidth
for bw = bandwidths
    [y, phasor, cost] = vold_kalman_filter(x, freq_vec, fs, bw, p);
    
    % Reconstruct signal
    x_recon = real(y .* phasor);
    
    % Print diagnostic info
    fprintf('\n===== Bandwidth = %d Hz =====\n', bw);
    
    % Add these debug outputs in your MATLAB function:
    % (I'll tell you what to add to the code)
    figure;
plot(t,x)
hold on
plot(t,x_recon)
end

