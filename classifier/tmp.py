# Fake NN output
import torch
import torch.nn as nn

out = torch.FloatTensor(
    [[0.05, 0.9, 0.05, 0.05, 0.9, 0.05], [0.05, 0.9, 0.05, 0.05, 0.9, 0.05]]
)
out = torch.autograd.Variable(out)

# One-hot encoded targets
y1 = torch.FloatTensor([[0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0]])
y1 = torch.autograd.Variable(y1)

loss_fn = nn.BCEWithLogitsLoss()
# Calculating the loss
print(out.size())
print(y1.size())
loss_val1 = loss_fn(out, y1)

print(loss_val1)
