
import torch
def second_box_encode(boxes, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box encode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
        encode_angle_to_vector: bool. increase aos performance,
            decrease other performance.
    """
    # need to convert boxes to z-center format
    xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
    xg, yg, zg, wg, lg, hg, rg = torch.split(boxes, 1, dim=-1)
    zg = zg + hg / 2
    za = za + ha / 2
    diagonal = torch.sqrt(la ** 2 + wa ** 2)  # 4.3
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    zt = (zg - za) / ha  # 1.6

    if smooth_dim:
        lt = lg / la - 1
        wt = wg / wa - 1
        ht = hg / ha - 1
    else:
        lt = torch.log(lg / la)
        wt = torch.log(wg / wa)
        ht = torch.log(hg / ha)
    if encode_angle_to_vector:
        rgx = torch.cos(rg)
        rgy = torch.sin(rg)
        rax = torch.cos(ra)
        ray = torch.sin(ra)
        rtx = rgx - rax
        rty = rgy - ray
        return torch.cat([xt, yt, zt, wt, lt, ht, rtx, rty], dim=-1)
    else:
        rt = rg - ra
        return torch.cat([xt, yt, zt, wt, lt, ht, rt], dim=-1)

def second_box_decode(box_encodings, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box decode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
    """
    
    xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
    if encode_angle_to_vector:
        xt, yt, zt, wt, lt, ht, rtx, rty = torch.split(
            box_encodings, 1, dim=-1)

    else:
        xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)

    # xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)
    za = za + ha / 2
    diagonal = torch.sqrt(la**2 + wa**2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za

    if smooth_dim:
        lg = (lt + 1) * la
        wg = (wt + 1) * wa
        hg = (ht + 1) * ha
    else:

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
    if encode_angle_to_vector:
        rax = torch.cos(ra)
        ray = torch.sin(ra)
        rgx = rtx + rax
        rgy = rty + ray
        rg = torch.atan2(rgy, rgx)
    else:
        rg = rt + ra
    zg = zg - hg / 2
    return torch.cat([xg, yg, zg, wg, lg, hg, rg], dim=-1)

def gen_sample_grid(box, window_size=(4, 7), grid_offsets=(0, 0), spatial_scale=1.):
    N = box.shape[0]
    win = window_size[0] * window_size[1]
    xg, yg, wg, lg, rg = torch.split(box, 1, dim=-1)

    xg = xg.unsqueeze_(-1).expand(N, *window_size)
    yg = yg.unsqueeze_(-1).expand(N, *window_size)
    rg = rg.unsqueeze_(-1).expand(N, *window_size)

    cosTheta = torch.cos(rg)
    sinTheta = torch.sin(rg)

    xx = torch.linspace(-.5, .5, window_size[0]).type_as(box).view(1, -1) * wg
    yy = torch.linspace(-.5, .5, window_size[1]).type_as(box).view(1, -1) * lg

    xx = xx.unsqueeze_(-1).expand(N, *window_size)
    yy = yy.unsqueeze_(1).expand(N, *window_size)

    x=(xx * cosTheta + yy * sinTheta + xg)
    y=(yy * cosTheta - xx * sinTheta + yg)

    x = (x.permute(1, 2, 0).contiguous() + grid_offsets[0]) * spatial_scale
    y = (y.permute(1, 2, 0).contiguous() + grid_offsets[1]) * spatial_scale

    return x.view(win, -1), y.view(win, -1)

def bilinear_interpolate_torch_gridsample(image, samples_x, samples_y):
    C, H, W = image.shape
    
    image = image.unsqueeze(1)  # change to:  C x 1 x H x W
    
    samples_x = samples_x.unsqueeze(2)
    samples_x = samples_x.unsqueeze(3)
    
    samples_y = samples_y.unsqueeze(2)
    samples_y = samples_y.unsqueeze(3)
    
    samples = torch.cat([samples_x, samples_y], 3)
    samples[:, :, :, 0] = (samples[:, :, :, 0] / (W - 1))  # normalize to between  0 and 1
    samples[:, :, :, 1] = (samples[:, :, :, 1] / (H - 1))  # normalize to between  0 and 1
    samples = samples * 2 - 1  # normalize to between -1 and 1
    # C, 1, N, 1
    return torch.nn.functional.grid_sample(image, samples)

def bilinear_interpolate_torch_gridsample_rep(image, samples_x, samples_y, parts):
    C, H, W = image.shape

    ch = int(C/parts)
    # image = image.unsqueeze(1)  # change to:  C x 1 x H x W
    image = image.view(parts, ch, H, W)
    samples_x = samples_x.unsqueeze(2)
    samples_x = samples_x.unsqueeze(3)
    
    samples_y = samples_y.unsqueeze(2)
    samples_y = samples_y.unsqueeze(3)
    
    samples = torch.cat([samples_x, samples_y], 3)
    samples[:, :, :, 0] = (samples[:, :, :, 0] / (W - 1))  # normalize to between 0 and 1
    samples[:, :, :, 1] = (samples[:, :, :, 1] / (H - 1))  # normalize to between 0 and 1
    samples = samples * 2 - 1  # normalize to between -1 and 1
    # C, ch, N, 1
    return torch.nn.functional.grid_sample(image, samples)

def bilinear_interpolate_torch_gridsample_rep_2(image, samples_x, samples_y, parts):
    C, H, W = image.shape
    image = image.unsqueeze(0)
    # ch = int(C/parts)
    # image = image.unsqueeze(1)  # change to:  C x 1 x H x W
    # image = image.view(parts, ch, H, W)
    image = image.expand([parts, C, H, W])
    samples_x = samples_x.unsqueeze(2)
    samples_x = samples_x.unsqueeze(3)
    
    samples_y = samples_y.unsqueeze(2)
    samples_y = samples_y.unsqueeze(3)
    
    samples = torch.cat([samples_x, samples_y], 3)
    samples[:, :, :, 0] = (samples[:, :, :, 0] / (W - 1))  # normalize to between 0 and 1
    samples[:, :, :, 1] = (samples[:, :, :, 1] / (H - 1))  # normalize to between 0 and 1
    samples = samples * 2 - 1                              # normalize to between -1 and 1
    # parts, ch, N, 1
    return torch.nn.functional.grid_sample(image, samples)