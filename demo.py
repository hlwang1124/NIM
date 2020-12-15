import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os


class CalibInfo():
    """Process KITTI calibration file"""

    def __init__(self, filepath):
        self.data = self._load_calib(filepath)

    def get_cam_param(self):
        return self.data['P2']

    def get_baseline(self):
        T = (self.data['P2'] - self.data['P3']
             ).dot(np.array([[0], [0], [0], [1]], dtype=np.float32))
        return np.sqrt(np.sum((T * T)))

    def _load_calib(self, filepath):
        rawdata = self._read_calib_file(filepath)
        data = {}
        P0 = np.reshape(rawdata['P0'], (3, 4))
        P1 = np.reshape(rawdata['P1'], (3, 4))
        P2 = np.reshape(rawdata['P2'], (3, 4))
        P3 = np.reshape(rawdata['P3'], (3, 4))
        R0_rect = np.reshape(rawdata['R0_rect'], (3, 3))
        Tr_velo_to_cam = np.reshape(rawdata['Tr_velo_to_cam'], (3, 4))

        data['P0'] = P0
        data['P1'] = P1
        data['P2'] = P2
        data['P3'] = P3
        data['R0_rect'] = R0_rect
        data['Tr_velo_to_cam'] = Tr_velo_to_cam

        return data

    def _read_calib_file(self, filepath):
        """Read in a calibration file and parse into a dictionary"""
        data = {}

        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data


class NIM(nn.Module):
    """Normal Inference Module"""

    def __init__(self):
        super(NIM, self).__init__()

    def forward(self, depth, calib, sign_filter):
        """Generate surface normal estimation from depth images

        Args:
            depth (torch.Tensor): depth image
            calib (CalibInfo): calibration parameters
            sign_filter (bool): if True, our NIM will additionally utilize a sign filter

        Returns:
            torch.Tensor: surface normal estimation
        """
        camParam = torch.tensor(calib.get_cam_param(), dtype=torch.float32)

        h, w = depth.size()
        v_map, u_map = torch.meshgrid(torch.arange(h), torch.arange(w))
        v_map = v_map.type(torch.float32)
        u_map = u_map.type(torch.float32)

        Z = depth   # h, w
        Y = Z * (v_map - camParam[1, 2]) / camParam[0, 0]  # h, w
        X = Z * (u_map - camParam[0, 2]) / camParam[0, 0]  # h, w
        Z[Y <= 0] = 0
        Y[Y <= 0] = 0
        Z[torch.isnan(Z)] = 0
        D = torch.ones(h, w) / Z  # h, w

        Gx = torch.tensor([[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
                          dtype=torch.float32)
        Gy = torch.tensor([[0, -1, 0], [0, 0, 0], [0, 1, 0]],
                          dtype=torch.float32)

        Gu = F.conv2d(D.view(1, 1, h, w), Gx.view(1, 1, 3, 3), padding=1)
        Gv = F.conv2d(D.view(1, 1, h, w), Gy.view(1, 1, 3, 3), padding=1)

        nx_t = Gu * camParam[0, 0]   # 1, 1, h, w
        ny_t = Gv * camParam[1, 1]   # 1, 1, h, w

        phi = torch.atan(ny_t / nx_t) + torch.ones([1, 1, h, w]) * 3.141592657
        a = torch.cos(phi)
        b = torch.sin(phi)

        diffKernelArray = torch.tensor([[0, -1, 0, 0, 1, 0, 0, 0, 0],
                                        [0, 0, 0, -1, 1, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 1, -1, 0, 0, 0],
                                        [0, 0, 0, 0, 1, 0, 0, -1, 0]], dtype=torch.float32)

        nx_volume = torch.zeros((1, 4, h, w), dtype=torch.float32)
        ny_volume = torch.zeros((1, 4, h, w), dtype=torch.float32)
        nz_volume = torch.zeros((1, 4, h, w), dtype=torch.float32)

        for i in range(4):
            diffKernel = diffKernelArray[i].view(1, 1, 3, 3)
            X_d = F.conv2d(X.view(1, 1, h, w), diffKernel, padding=1)
            Y_d = F.conv2d(Y.view(1, 1, h, w), diffKernel, padding=1)
            Z_d = F.conv2d(Z.view(1, 1, h, w), diffKernel, padding=1)

            nz_i = -(nx_t * X_d + ny_t * Y_d) / Z_d
            norm = torch.sqrt(nx_t * nx_t + ny_t * ny_t + nz_i * nz_i)
            nx_t_i = nx_t / norm
            ny_t_i = ny_t / norm
            nz_t_i = nz_i / norm

            nx_t_i[torch.isnan(nx_t_i)] = 0
            ny_t_i[torch.isnan(ny_t_i)] = 0
            nz_t_i[torch.isnan(nz_t_i)] = 0

            nx_volume[0, i, :, :] = nx_t_i
            ny_volume[0, i, :, :] = ny_t_i
            nz_volume[0, i, :, :] = nz_t_i

        if sign_filter:
            nz_volume_pos = torch.sum(nz_volume > 0, dim=1, keepdim=True)
            nz_volume_neg = torch.sum(nz_volume < 0, dim=1, keepdim=True)
            pos_mask = (nz_volume_pos >= nz_volume_neg) * (nz_volume > 0)
            neg_mask = (nz_volume_pos < nz_volume_neg) * (nz_volume < 0)
            final_mask = pos_mask | neg_mask
            nx_volume *= final_mask
            ny_volume *= final_mask
            nz_volume *= final_mask

        theta = torch.atan((torch.sum(nx_volume, 1) * a +
                            torch.sum(ny_volume, 1) * b) / torch.sum(nz_volume, 1))
        nx = torch.sin(theta) * torch.cos(phi)
        ny = torch.sin(theta) * torch.sin(phi)
        nz = torch.cos(theta)

        nx[torch.isnan(nz)] = 0
        ny[torch.isnan(nz)] = 0
        nz[torch.isnan(nz)] = -1

        sign_map = torch.ones((1, 1, h, w), dtype=torch.float32)
        sign_map[ny > 0] = -1

        nx = (nx * sign_map).squeeze(dim=0)
        ny = (ny * sign_map).squeeze(dim=0)
        nz = (nz * sign_map).squeeze(dim=0)

        return torch.cat([nx, ny, nz], dim=0)


def normal_visualization(normal):
    normal_vis = (1 + normal) / 2
    return normal_vis


if __name__ == '__main__':
    example_name = 'uu_000000'
    depth = cv2.imread(os.path.join('examples', 'depth_u16',
                                    example_name + '.png'), cv2.IMREAD_ANYDEPTH).astype(np.float32)/1000
    calib = CalibInfo(os.path.join('examples', 'calib', example_name + '.txt'))
    depth = torch.tensor(depth)

    model = NIM()

    normal = model(depth, calib, sign_filter=True)
    normal = normal.cpu().numpy()

    normal_vis = normal_visualization(normal)

    if not os.path.exists(os.path.join('examples', 'normal')):
        os.makedirs(os.path.join('examples', 'normal'))
    cv2.imwrite(os.path.join('examples', 'normal', example_name + '.png'), cv2.cvtColor(
        normal_vis.transpose([1, 2, 0])*255, cv2.COLOR_RGB2BGR))
