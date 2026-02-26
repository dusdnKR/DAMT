import numpy as np
import torch


def rot_rand(args, x_s, a_s):
    img_n = x_s.size()[0]
    x_aug = x_s.detach().clone()
    a_aug = a_s.detach().clone()
    device = torch.device(f"cuda:{args.local_rank}")
    x_rot = torch.zeros(img_n).long().to(device)
    for i in range(img_n):
        x = x_s[i]
        a = a_s[i]
        orientation = np.random.randint(0, 10)
        if orientation == 0:
            pass
        elif orientation == 1:
            x = x.rot90(1, (2, 3))
            a = a.rot90(1, (2, 3))
        elif orientation == 2:
            x = x.rot90(2, (2, 3))
            a = a.rot90(2, (2, 3))
        elif orientation == 3:
            x = x.rot90(3, (2, 3))
            a = a.rot90(3, (2, 3))
        elif orientation == 4:
            x = x.rot90(1, (1, 3))
            a = a.rot90(1, (1, 3))
        elif orientation == 5:
            x = x.rot90(2, (1, 3))
            a = a.rot90(2, (1, 3))
        elif orientation == 6:
            x = x.rot90(3, (1, 3))
            a = a.rot90(3, (1, 3))
        elif orientation == 7:
            x = x.rot90(1, (1, 2))
            a = a.rot90(1, (1, 2))
        elif orientation == 8:
            x = x.rot90(2, (1, 2))
            a = a.rot90(2, (1, 2))
        elif orientation == 9:
            x = x.rot90(3, (1, 2))
            a = a.rot90(3, (1, 2))
        x_aug[i] = x
        a_aug[i] = a
        x_rot[i] = orientation

    return x_aug, a_aug, x_rot

