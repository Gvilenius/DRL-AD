import torch
def hook_b(grad):
    print(grad)
a = torch.rand((1,3), requires_grad=True)
print(a)
print(a.requires_grad)
b = 2 * a
print(b)
print(b.requires_grad)
c = a + 12*b.detach()
z = c.sum()
print(c)
z.backward()
print(a.grad)