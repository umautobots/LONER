import torch
import pytorch3d.transforms as transf

def aa_to_mat(aa):
    if aa.dim() == 1:
        aa = aa.unsqueeze(0)
    theta = aa.norm(dim=1)

    ss = torch.zeros((aa.shape[0], 3, 3))
    ss[:, 0, 1] = -aa[:,2]
    ss[:, 0, 2] = aa[:,1]
    ss[:, 1, 0] = aa[:,2]
    ss[:, 1, 2] = -aa[:,0]
    ss[:, 2, 0] = -aa[:,1]
    ss[:, 2, 1] = aa[:,0]
    
    theta = theta.view(-1, 1, 1)
    mat = torch.eye(3).repeat(aa.shape[0], 1, 1) + torch.sin(theta) * ss + (1-torch.cos(theta))*(ss@ss)
    return mat

import cProfile

aa = torch.rand((1_000_000, 3))

print("theirs")
print(cProfile.run("transf.axis_angle_to_matrix(aa)"))


print("mine")
print(cProfile.run("aa_to_mat(aa)"))


print("quat")
print(cProfile.run("transf.axis_angle_to_quaternion(aa)"))
