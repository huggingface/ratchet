import numpy as np 

# Understanding argument reordering for a matmul
# nn.Linear in PyTorch is defined as:
# y = x @ W.t() + b
# Weights in a GGUF are stored as (out_features, in_features)
#
# Argument reordering
# In order to have fast memory access patterns, it can sometimes be prudent to reorder the arguments of a matmul
# Particularly in the case of a vector-matrix multiplication.
# e.g [1, 2560] @ [10240, 2560].t() -> [1, 10240]
# If everything is stored in row-major order, the above matmul will have poor memory access patterns.
# However, we can swap the arguments.
# [10240, 2560] @ [1, 2560].t() -> [10240, 1]
# This will have good access patterns on BOTH A & B.
W = np.random.rand(10240, 2560) #
X = np.random.rand(1, 2560) #

WT = np.transpose(W, (1, 0))

Y = X @ WT
print("Without reordering")
print(f"{X.shape} @ {WT.shape} = {Y.shape}")

XT = np.transpose(X, (1, 0))

Z = W @ XT
print("With reordering")
print(f"{W.shape} @ {XT.shape} = {Z.shape}")

#check if Y and Z are the same
print("Are results the same: ", np.allclose(Y, np.transpose(Z, (1, 0))))

