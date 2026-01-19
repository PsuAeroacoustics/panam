%{

This function implements the multi-shaft Vold-Kalman filter.

The inputs are (x,freq,fs,bandwidth,p,r),
where:
        x - Acoustic time-history signal
     freq - Frequency vector with frequencies/orders of interest to extract
            from the signal, Hz
       fs - Sampling frequency of the acoustic signal and frequency vector
bandwidth - Bandwidth to use for the weighting factor formulation, Hz
        p - Row matrix with the filter order (ex:[1 2])
        r - (Optional) Weighting factor to use for the filter

x and freq vectors must have same length(# of time steps)


The outputs are: [y phasor cost_Mat],
where:
        y - Extracted complex envelopes of each order
   phasor - Complex phasor corrersponding to each order
 cost_Mat - (Optional) cost matric of the iterative solver

Waveforms can be obtained by multiplying the complex envelopes and phasors
    waveform=y.*phasor


If the user chooses to use a weighting factor, input a arbitrary bandwidth
to the filter, it will not affect the extracted signal.

This algotirhm tracks the acoustic signal produced by shafts and/or orders
that are given in the reference frequency vector.

If the weighting factor is not defined, the algorithm uses the bandwidth to
compute a desired weighting factor for the signal.

To use different bandwidth for every order, the bandwidth vector must be a
row vector with length same as number of orders with individual
bandwidths corresponding to frequency of the order respectively.

%}

function varargout = vold_kalman_filter(x,freq,fs,bandwidth,p,r)

%Checking the input and output arguments of the function
switch nargin 
    case or(nargin<5,nargin>6)
    error('Not enough arguments or too many arguments.')

    case 5
    weightfactor = true;

    case 6
    weightfactor = false;
    weigF.r=r;
end

if or(nargout<=1,nargout>=4)
    warning(['This function returns only Complex envelope & Phasor ' ...
        'and optional r-weighting factor'])
    error('Invalid number of outputs, please choose 2 or 3:')
end


% Initializing the variables
dt=1/fs;
n_x=length(x);
[n_f, n_Ord]=size(freq);
n_Tot=n_x*n_Ord;

if n_f ~= n_x
    error('The length of frequency vector must be the same as the signal')
end


% Creating the Phasor Vector
phasor=exp(2i*pi*cumsum(freq)*dt);

bw.scalarBW=false;
bw.vectorBW=false;
bw.scalarOrderWiseBW=false;
bw.vectorOrderWiseBW=false;


% Initializing the bandwidth vector

if length(bandwidth)==1
    bw.band=bandwidth*ones(n_x,1);
    bw.scalarBW=true;
elseif size(bandwidth)==size(x)
    bw.band=bandwidth;
    bw.vectorBW=true;
elseif length(bandwidth)==n_Ord
    bw.band=bandwidth.*ones(n_x,n_Ord);
    bw.scalarOrderWiseBW = true;
elseif size(bandwidth)==size(freq)
    bw.band=bandwidth;
    bw.vectorOrderWiseBW = true;
else
    error(['Bandwidth not chosen properly, Check bandwidth scalar' ...
        '/vector dimensions'])
end     

bw.band_Rad=bw.band.*pi/fs;


% Setting the filter order & coefficients
p_p=max(p);

diffEq.power=pascal(p_p+1,1);
diffEq.mat_Coe=diffEq.power(end,:);

% Defining the matrix of linear equations
diffEq.df=setdiff(p,p_p);
diffEq.nr=numel(diffEq.df);
diffEq.coe_Mat=ones(n_x-p_p,1)*diffEq.mat_Coe;

diffEq.A0=sparse(diffEq.nr,n_x);
diffEq.A0(:,1:p_p)=diffEq.power(diffEq.df+1,1:p_p);

% Defining the matrix of linear equations for 1st order
A_temp=spdiags(diffEq.coe_Mat,0:p_p,n_x-p_p,n_x);

% Debug: Show A0
disp('=== A0 ===');
disp(['Size of A0: ' num2str(size(diffEq.A0))]);
if size(diffEq.A0, 1) > 0
    disp('A0 first row (first 10 elements):');
    disp(full(diffEq.A0(1, 1:min(10, size(diffEq.A0, 2)))));
end

% Debug: Show A before concatenation
disp('=== A (before concatenation) ===');
disp(['Size: ' num2str(size(A_temp))]);
disp('First 10 rows, 10 cols:');
disp(full(A_temp(1:min(10, size(A_temp, 1)), 1:min(10, size(A_temp, 2)))));

% Concatenate
diffEq.A=[diffEq.A0; A_temp; diffEq.A0(end:-1:1,end:-1:1)];

% Debug: Show A after concatenation
disp('=== A (after concatenation) ===');
disp(['Size: ' num2str(size(diffEq.A))]);
disp('First 10 rows, 10 cols:');
disp(full(diffEq.A(1:min(10, size(diffEq.A, 1)), 1:min(10, size(diffEq.A, 2)))));

% This is where the diagonal extraction happens
diffEq.diagonal=spdiags(diffEq.A,-p_p:p_p);

% Debug: Show diagonal matrix
disp('=== Diagonal Matrix ===');
disp(['Size: ' num2str(size(diffEq.diagonal))]);
disp('First 5 rows, all columns:');
disp(full(diffEq.diagonal(1:min(5, size(diffEq.diagonal, 1)), :)));

% Build AA
diffEq.AA=spdiags(repmat(diffEq.diagonal,n_Ord,1),-p_p:p_p,n_Tot,n_Tot);

% Debug: Show AA
disp('=== AA Matrix ===');
disp(['Size: ' num2str(size(diffEq.AA))]);
disp('First 10 rows, 10 cols:');
disp(full(diffEq.AA(1:min(10, size(diffEq.AA, 1)), 1:min(10, size(diffEq.AA, 2)))));
disp('Last 10 rows, last 10 cols:');
n_rows = size(diffEq.AA, 1);
n_cols = size(diffEq.AA, 2);
disp(full(diffEq.AA(max(1, n_rows-9):n_rows, max(1, n_cols-9):n_cols)));
% Solving for r-weighting factor
% If the user specifies a weighting factor then this step will be skipped,
% but if the user doesnt specify weighting factor, then the bandwidth will
% be used to calculate the weighting factor for the equations.

if weightfactor
    weigF.n=0:p_p;
    weigF.sign=(-1).^weigF.n;
    weigF.coecos=ones(p_p+1,p_p+1);
    weigF.coe=zeros(p_p+1,1);
    weigF.coe(1)=2^(2*p_p);
    
    %Structural differential equations
    if or(bw.scalarBW,bw.vectorBW)
        
        for i=1:p_p
            weigF.coecos(i+1,:)=weigF.n.^(2*(i-1)).*weigF.sign;
        end
        coeff=transpose(weigF.coecos\weigF.coe).*weigF.sign;
        weigF.denominator = zeros(size(bw.band_Rad));
        for i=1:length(coeff)
            weigF.denominator=weigF.denominator+coeff(i)*(cos(bw.band_Rad*(i-1)));
        end
        weigF.numerator = sqrt(2)-1;
        weigF.r0 = sqrt(weigF.numerator./weigF.denominator);
        weigF.r = repmat(weigF.r0,n_Ord,1);
        
    elseif or(bw.scalarOrderWiseBW,bw.vectorOrderWiseBW)
        
        for j=1:n_Ord
            for i=1:p_p
                weigF.coecos(i+1,:)=weigF.n.^(2*(i-1)).*weigF.sign;
            end
            coeff=transpose(weigF.coecos\weigF.coe).*weigF.sign;
            weigF.denominator = zeros(size(bw.band_Rad(:,j)));
            for i=1:length(coeff)
                weigF.denominator=weigF.denominator+coeff(i)*(cos(bw.band_Rad(:,j)*(i-1)));
            end
            weigF.numerator = sqrt(2)-1;
            weigF.r0 = sqrt(weigF.numerator./weigF.denominator);
            weigF.r(((j-1)*n_x)+(1:n_x),1)=weigF.r0;
            
        end 
    end
end

% Defining the r-weighting matrix
diffEq.RR=spdiags(ones(n_Tot,1).*weigF.r,0,n_Tot,n_Tot);
disp(['Size of RR: ' num2str(size(diffEq.RR))]);
rr_diag = diag(diffEq.RR);
disp(['RR diagonal(1:5): ' num2str(full(rr_diag(1:5))')]);  

% Combining AA, RR, and Unity to form BB-matrix
diffEq.B0=(diffEq.AA')*(diffEq.RR*diffEq.RR)*diffEq.AA + speye(n_Tot);
disp(['Size of B0: ' num2str(size(diffEq.B0))]);
disp('B0(1:10, 1:10):'), disp(full(diffEq.B0(1:10, 1:10)));


% Initializing the vectors to store the coefficient matrix off-diagonal
% values
diffEq.sumord=(n_Ord*(n_Ord-1))/2;
diffEq.sumordmat=zeros(n_x,diffEq.sumord);


% Vectors to store row and column positions of the off-diagonal elements
% and the conjugate off-diagonal values
diffEq.rowcol1=diffEq.sumordmat;
diffEq.rowcol2=diffEq.sumordmat;
diffEq.value=complex(diffEq.sumordmat,0);

% Debug output
disp(['BW: ' num2str(bw.band(1))]);
disp(['weigF.r0(1): ' num2str(weigF.r0(1))]);
disp(['weigF.coecos: ']);
disp(weigF.coecos);
disp(['coeff: ' num2str(coeff)]);
disp(['denominator(1): ' num2str(weigF.denominator(1))]);
rr_diag = diag(diffEq.RR);
disp(['RR diagonal(1:5): ' num2str(full(rr_diag(1:5))')]);
disp(['B0(1,1): ' num2str(diffEq.B0(1,1))]);


clear weigF bw

arb=1;
for i=1:n_Ord
    for j=(i+1):n_Ord
        diffEq.rowcol1(:,arb) = (i-1)*n_x + (1:n_x);
        diffEq.rowcol2(:,arb) = (j-1)*n_x + (1:n_x);
        diffEq.value(:,arb) = conj(phasor(:,i)).*phasor(:,j);
        arb=arb+1;
    end
end


% Creating a sparse upper matrix using the row&column positions and the
% conjugate off-diagonal coefficient values
diffEq.B_U=sparse(diffEq.rowcol1,diffEq.rowcol2,diffEq.value,n_Tot,n_Tot);


% Creating a sparse coefficient matrix using the diagonal matrix and 
% upper & lower coeficient matrices
B = diffEq.B0 + diffEq.B_U + diffEq.B_U';
disp(['B(1,1): ' num2str(full(B(1,1)))]);

clear diffEq


%Reshape and find complex conjugate of the phasor
phasor_rs=reshape(phasor,n_Tot,1);
conj_Phasor=conj(phasor_rs);

% Debug: Show phasor values
disp(['phasor(1:5): ' num2str(phasor(1:5)')]);
disp(['conj_Phasor(1:5): ' num2str(conj_Phasor(1:5)')]);

%Reshaping the signal to apply the filter
x_rv=repmat(x,n_Ord,1);

% Debug: Show signal values
disp(['x(1:5): ' num2str(x(1:5)')]);
disp(['x_rv(1:5): ' num2str(x_rv(1:5)')]);

%Calculating the Right-hand side of the linear differential equations
cH_x=conj_Phasor.*x_rv;


%Debug statement
%fprintf('Solving Linear System of Equations\n')
% Debug: Show cH_x and B before solve
disp(['cH_x(1:5): ' num2str(cH_x(1:5)')]);
disp(['B(1,1) before solve: ' num2str(B(1,1))]);
disp(['norm(B-B''): ' num2str(norm(full(B-B')))]);  % Check if B is symmetric

%Solving the linear equations using a MATLAB backslash
y_R = B\cH_x;


% Save B and cH_x for Python cross-check
save('matlab_solve_inputs.mat', 'B', 'cH_x', 'y_R');


%Getting an estimate of the cost matrix for the solution
cost_Mat = cH_x - B*y_R;

% Scaling debug
disp(['y_R(1:5): ' num2str(y_R(1:5)')]);
[y_R_max, max_idx] = max(abs(y_R));
disp(['y_R max: ' num2str(y_R_max) ' at index ' num2str(max_idx)]);
disp(['y_R(' num2str(max_idx) '): ' num2str(y_R(max_idx))]);
disp(['|y_R(' num2str(max_idx) ')|: ' num2str(abs(y_R(max_idx)))]);

% Reordering the complex envelope from a column vector to a
% #oforder-by-#oftimesteps matrix
y=2*reshape(y_R,[n_x,n_Ord]);


% Outputs
switch nargout
    case 2
        varargout{1} = y;
        varargout{2} = phasor;

    case 3
        varargout{1} = y;
        varargout{2} = phasor;
        varargout{3} = cost_Mat;
end

end
