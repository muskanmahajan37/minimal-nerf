import torch
from torch import nn

class NeRF(nn.Module):

    def __init__(self, device, min_bounds, max_bounds, num_enc_p=10, num_enc_d=4, num_channels=256):
        super(NeRF, self).__init__()
        self.device = device
        self.num_enc_p = num_enc_p
        self.num_enc_d = num_enc_d
        self.num_channels = num_channels
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
        self.layers = nn.ModuleList([
            self.layer(6 * num_enc_p, num_channels),
            self.layer(num_channels, num_channels),
            self.layer(num_channels, num_channels),
            self.layer(num_channels, num_channels),
            self.layer(num_channels, num_channels),
            self.layer(6 * num_enc_p + num_channels, num_channels),
            self.layer(num_channels, num_channels),
            self.layer(num_channels, num_channels),
            self.layer(num_channels, num_channels + 1, act_fn = torch.nn.Identity),
            self.layer(6 * num_enc_d + num_channels, num_channels // 2),
            self.layer(num_channels // 2, 3, act_fn = torch.nn.Sigmoid)
        ])
    
    def layer(self, in_features, out_features, act_fn = torch.nn.ReLU):
        return nn.Sequential(
            torch.nn.Linear(in_features, out_features),
            act_fn()
        )

    def get_rays(self, image, camera_pose, focal):
        W = image.shape[1]
        H = image.shape[0]
        i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
        lat = (i - W/2) / W
        lon = (j - H/2) / H
        dirs = torch.stack([(i - (W - 1) * 0.5) / focal, -(j - (H - 1) * 0.5) / focal, -torch.ones_like(i)], -1)
        rays_d = torch.sum(dirs[..., np.newaxis, :] * camera_pose[:3,:3], -1)
        rays_d = rays_d.permute((1, 0, 2)) # (w, h, ch) -> (h, w, ch)
        rays_d = torch.reshape(rays_d, [-1,3])
        rays_d = rays_d / torch.sqrt(torch.sum(torch.square(rays_d), dim=1))[:,None]
        rays_o = camera_pose[:3,-1].expand(rays_d.shape)
        gt_colors = image.reshape([-1, 3])
        return [rays_o, rays_d, gt_colors]

    def box_intersection(self, positions, directions): 
        inv_directions = 1 / directions
        t0 = (self.min_bounds - positions) * inv_directions
        t1 = (self.max_bounds - positions) * inv_directions
        tmax, _ = torch.min(torch.max(t0, t1), dim=1)
        return tmax
    
    def render_rays(self, positions, directions, num_samples, noise=True):
        batch_size = positions.shape[0]
        path_length = self.box_intersection(positions, directions)
        samples = torch.arange(1, num_samples + 1).to(device) / num_samples
        p = positions[:,None,:] + directions[:,None,:] * samples[None,:,None] * path_length[:,None,None]
        p_flat = torch.reshape(p, (-1, 3)).float()
        d = directions.expand((num_samples, batch_size, 3)).permute((1, 0, 2))
        d_flat = torch.reshape(d, (-1, 3)).float()
        colors, densities = self.forward(p_flat, d_flat)
        colors = colors.reshape((batch_size, num_samples, 3))
        densities = densities.reshape(d.shape[:-1])
        delta = path_length / num_samples
        batch_ones = torch.ones((batch_size, 1)).to(device)
        alpha = 1.0 - torch.exp(-1.0 * densities * delta[:,None])          
        T = torch.cumprod(torch.cat([batch_ones, 1.0 - alpha], -1), -1)[:, :-1]
        weights = T * alpha
        projected_colors = torch.sum(weights[:,:,None] * colors, dim=1)
        depth = torch.sum(weights * samples, dim=1) 
        return [projected_colors, depth, weights]
            
    def encode(self, x, L):
        batch_size = x.shape[0]
        f = ((2.0 ** torch.arange(0, L))).to(device)
        f = f.expand((batch_size, 3, L))
        f = torch.cat([torch.cos(math.pi * f * x[:,:,None]), torch.sin(math.pi * f * x[:,:,None])], dim=2)
        return f.reshape((batch_size, -1))

    def forward(self, p, d):
        p_normalized = -1. + 2. * (p - self.min_bounds) / (self.max_bounds - self.min_bounds)
        p_enc = self.encode(p_normalized, self.num_enc_p);
        d_enc = self.encode(d, self.num_enc_d);
        res1 = self.layers[0](p_enc)
        res2 = self.layers[1](res1)
        res3 = self.layers[2](res2)
        res4 = self.layers[3](res3)
        res5 = self.layers[4](res4)
        res6 = self.layers[5](torch.cat([p_enc, res5], dim=1))
        res7 = self.layers[6](res6)
        res8 = self.layers[7](res7)
        res9 = self.layers[8](res8)
        density = F.relu(res9[:,0])
        res10 = self.layers[9](torch.cat([res9[:,1:], d_enc], dim=1))
        color = self.layers[10](res10)
        return [color, density]