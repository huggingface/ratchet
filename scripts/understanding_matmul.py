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
X = np.random.rand(2, 2560) #

WT = np.transpose(W, (1, 0))

Y = X @ WT
print("Standard case: y = xWT + b")
print(f"{X.shape} @ {WT.shape} = {Y.shape}\n")

XT = np.transpose(X, (1, 0))

ZT = W @ XT
print("Reordered case: zT = WxT + b")
print(f"{W.shape} @ {XT.shape} = {ZT.shape}\n")

#check if Y and Z are the same
print("Are results the same: ", np.allclose(Y, np.transpose(ZT, (1, 0))))


print("By performing the reordered case, we can avoid transposing W, which is not feasible for quantized W.")
