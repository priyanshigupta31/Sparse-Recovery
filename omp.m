function [x_omp, support, iteration] = omp(y, A, K, err)
% Orthogonal Matching Pursuit (OMP) is a greedy algorothm that
% provides approximate solution to the problem: min ||x||_0 such that Ax = y.  
% Input:
% y		    : measurements
% A       	: measurement matrix
% K	    	: Sparsity level of underlying signal to be recovered
% err       : residual tolerant
% Output:
% x_omp    : estimated sparse signal
% The set of indices that correspond to the nonzero vector components 
% is also known as the support
% support   : estimated support set
% iteration : # of performed iterations 
 	if nargin < 4
	   err    = 1e-5;
 end 
m = 30
l = 3
k = 10
R=randn(k, m);  % random mxn rows x col
A=orth( R.' ).'; % orthogonal rows
x_sparse = sprand(m,1,l/m)
x = full(x_sparse)
y = A*x


	x_omp	  = zeros(size(A,2), 1);
	residual  = y;
	supp	  = [];
	iteration = 0; 
	
	while (norm(residual) > err) %&& iteration < min(l, floor(size(A,1)))) 
		   iteration          = iteration + 1;
		   [~, idx]           = sort(abs(A' * residual), 'descend');
		   supp_temp          = union(supp, idx(1:1));
	   if (length(supp_temp) ~= length(supp))
           supp	              = supp_temp;
		   x_hat			  = A(:,supp)\y;
		   residual           = y - A(:,supp) * x_hat; 
      else
		   break;
       end
    end
    
 	x_omp(supp)	          = A(:,supp)\y;
	[~, supp_idx]             = sort(abs(x_omp), 'descend');
	support                   = supp_idx(1:l); 
	x_omp                    = zeros(size(A,2), 1);
    x_omp(support)           = A(:,support)\y;


    j                         = immse(x_omp, x)

end




