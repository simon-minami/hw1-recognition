from one_stage_detector import DetectorBackboneWithFPN
import torch
# model = DetectorBackboneWithFPN(64)
# dummy = torch.rand(2,3,224,224)
# model(dummy)

# # test = dummy.view(dummy.shape[:2], -1)
# test = dummy.view(*dummy.shape[:2], -1)

# print(test.shape)


dummy = torch.rand(4)
print(dummy)
mask = torch.tensor([True, True, False, False])
print(dummy[mask])

test = torch.tensor(5, dtype=torch.int16)
test2 = torch.tensor(5, dtype=torch.int64)
print(test, test2, test==test2)