from torch.optim import Adam, SGD

from model.loss import LCC, SSD, RegLossL2, EntropyMultivariateNormal
from model.model import SimEnc
from utils.registration import RegistrationModule
from utils.sampler import sample_qf, sample_qv
from utils.transformation import SVF
from utils.util import init_identity_grid_3d, pixel_to_normalised_3d, save_im_to_disk

import numpy as np
import torch
import unittest

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

torch.autograd.set_detect_anomaly(True)


class RegistrationTestMethods(unittest.TestCase):
    def setUp(self):
        print(self._testMethodName)

        """
        utils
        """

        n = 64
        self.dim_x = n
        self.dim_y = n
        self.dim_z = n

        self.dims_im = (1, 1, self.dim_x, self.dim_y, self.dim_z)
        self.dims_v = (1, 3, self.dim_x, self.dim_y, self.dim_z)

        self.identity_grid = init_identity_grid_3d(self.dim_x, self.dim_y, self.dim_z).to('cuda:0')

        """
        modules
        """

        self.s = 7
        self.s_LCC = 5

        self.transformation_model = SVF().to('cuda:0')
        self.registration_module = RegistrationModule().to('cuda:0')

        """
        losses
        """

        self.reg_loss = RegLossL2('GradientOperator').to('cuda:0')
        self.entropy = EntropyMultivariateNormal().to('cuda:0')

        """
        optimiser parameters
        """

        self.no_steps_v = 128
        self.no_samples = 6

    def test_sphere_translation_LCC_data_only(self):
        """
        initialise 3D images of spheres
        """

        im_fixed = -1.0 + torch.zeros(self.dims_im).to('cuda:0')
        im_moving = -1.0 + torch.zeros(self.dims_im).to('cuda:0')

        r = 0.02

        offset_x = 5
        offset_y = 5
        offset_z = 5

        for idx_z in range(im_fixed.shape[2]):
            for idx_y in range(im_fixed.shape[3]):
                for idx_x in range(im_fixed.shape[4]):
                    x, y, z = pixel_to_normalised_3d(idx_x, idx_y, idx_z, self.dim_x, self.dim_y, self.dim_z)

                    if x ** 2 + y ** 2 + z ** 2 <= r:
                        im_fixed[0, 0, idx_x, idx_y, idx_z] = 1.0
                        im_moving[0, 0, idx_x + offset_x, idx_y + offset_y, idx_z + offset_z] = 1.0

        """"
        save the volumetric images to disk
        """

        save_im_to_disk(im_fixed[0, :, :, :, :], './temp/fixed.nii.gz')
        save_im_to_disk(im_moving[0, :, :, :, :], './temp/moving.nii.gz')

        """
        loss, parameters to learn, and optimiser
        """

        data_loss = LCC(self.s_LCC).to('cuda:0')
        mu_v = torch.zeros(self.dims_v).to('cuda:0').requires_grad_(True)
        optimizer_v = Adam([mu_v], lr=1e-3)

        """
        registration
        """

        total_loss = 0.0

        with torch.no_grad():
            warp_field = self.transformation_model.forward_3d(self.identity_grid, mu_v)
            im_moving_warped = self.registration_module(im_moving, warp_field)

            print(f'PRE-REGISTRATION: ' +
                  f'{data_loss(im_fixed, im_moving).item():.2f}' +
                  f', {data_loss(im_fixed, im_moving_warped).item():.2f}\n'
                  )

        """
        q_v
        """

        for iter_no in range(self.no_steps_v):
            optimizer_v.zero_grad()

            warp_field = self.transformation_model.forward_3d(self.identity_grid, mu_v)
            im_moving_warped = self.registration_module(im_moving, warp_field)

            data_term = data_loss(im_fixed, im_moving_warped)
            reg_term = self.reg_loss(mu_v).sum()

            loss_qv = data_term + reg_term
            loss_qv.backward()
            optimizer_v.step()

            print(f'ITERATION ' + str(iter_no) + '/' + str(self.no_steps_v - 1) +
                  f', TOTAL ENERGY: {loss_qv.item():.2f}' +
                  f'\ndata: {data_term.item():.2f}' +
                  f', regularisation: {reg_term.item():.2f}'
                  )

            total_loss += (loss_qv.item() / float(self.no_steps_v))

        mu_v.requires_grad_(False)

        """
        save image of the warped sphere to disk
        """

        with torch.no_grad():
            warp_field = self.transformation_model.forward_3d(self.identity_grid, mu_v)
            im_moving_warped = self.registration_module(im_moving, warp_field)

            save_im_to_disk(im_moving_warped[0, :, :, :, :], './temp/moving_warped_LCC_data_only.nii.gz')

    def test_sphere_translation_SSD_full(self):
        """
        initialise 3D images of spheres
        """
        im_fixed = -1.0 + torch.zeros(self.dims_im).to('cuda:0')
        im_moving = -1.0 + torch.zeros(self.dims_im).to('cuda:0')

        r = 0.02

        offset_x = 5
        offset_y = 5
        offset_z = 5

        for idx_z in range(im_fixed.shape[2]):
            for idx_y in range(im_fixed.shape[3]):
                for idx_x in range(im_fixed.shape[4]):
                    x, y, z = pixel_to_normalised_3d(idx_x, idx_y, idx_z, self.dim_x, self.dim_y, self.dim_z)

                    if x ** 2 + y ** 2 + z ** 2 <= r:
                        im_fixed[0, 0, idx_x, idx_y, idx_z] = 1.0
                        im_moving[0, 0, idx_x + offset_x, idx_y + offset_y, idx_z + offset_z] = 1.0

        """
        parameters to learn
        """

        mu_v = torch.zeros(self.dims_v).to('cuda:0').requires_grad_(True)

        var_v = float(self.dim_x ** (-2)) * torch.ones(self.dims_v)
        log_var_v = torch.log(var_v).to('cuda:0').requires_grad_(True)
        u_v = torch.zeros(self.dims_v).to('cuda:0').requires_grad_(True)

        var_f = float(0.1 ** 2) * torch.ones(self.dims_im)
        log_var_f = torch.log(var_f).to('cuda:0').requires_grad_(True)
        u_f = torch.zeros(self.dims_im).to('cuda:0').requires_grad_(True)

        """
        encoding function
        """

        enc = SimEnc(self.s).to('cuda:0')

        """
        data loss
        """

        data_loss = SSD().to('cuda:0')

        """
        optimisers
        """

        optimizer_v = SGD([mu_v, log_var_v, u_v], lr=1e-3)
        optimizer_f = SGD([log_var_f, u_f], lr=1e-3)
        optimizer_phi = SGD(enc.parameters(), lr=1e-3)

        """
        registration
        """

        total_loss = 0.0

        with torch.no_grad():
            warp_field = self.transformation_model.forward_3d(self.identity_grid, mu_v)
            im_moving_warped = self.registration_module(im_moving, warp_field)

            print(f'PRE-REGISTRATION: ' +
                  f'{data_loss(im_fixed - im_moving).item():.2f}' +
                  f', {data_loss(im_fixed - im_moving_warped).item():.2f}\n'
                  )

        """
        q_v
        """

        enc.eval()
        enc.set_grad_enabled(False)

        for iter_no in range(self.no_steps_v):
            optimizer_v.zero_grad()
            data_term = 0.0

            for _ in range(self.no_samples):
                v_sample = sample_qv(mu_v, log_var_v, u_v)
                warp_field = self.transformation_model.forward_3d(self.identity_grid, v_sample)

                im_moving_warped = self.registration_module(im_moving, warp_field)
                im_out = enc(im_fixed, im_moving_warped)

                data_term_sample = data_loss(im_out).sum() / float(self.no_samples)
                data_term += data_term_sample

            reg_term = self.reg_loss(mu_v).sum()
            entropy_term = self.entropy(log_var_v, u_v).sum()

            loss_qv = data_term + reg_term + entropy_term
            loss_qv.backward()
            optimizer_v.step()

            print(f'ITERATION ' + str(iter_no) + '/' + str(self.no_steps_v - 1) +
                  f', TOTAL ENERGY: {loss_qv.item():.2f}' +
                  f'\ndata: {data_term.item():.2f}' +
                  f', regularisation: {reg_term.item():.2f}' +
                  f', entropy: {entropy_term.item():.2f}'
                  )

            total_loss += (loss_qv.item() / float(self.no_steps_v))

        mu_v.requires_grad_(False)
        log_var_v.requires_grad_(False)
        u_v.requires_grad_(False)

        """
        q_phi
        """

        enc.train()
        enc.set_grad_enabled(True)

        optimizer_phi.zero_grad()
        optimizer_f.zero_grad()

        loss_qphi = 0.0

        for _ in range(self.no_samples):
            # first term
            v_sample = sample_qv(mu_v, log_var_v, u_v)
            warp_field = self.transformation_model.forward_3d(self.identity_grid, v_sample)

            im_moving_warped = self.registration_module(im_moving, warp_field)
            im_out = enc(im_fixed, im_moving_warped)

            data_term_sample = data_loss(im_out).sum() / float(self.no_samples)
            loss_qphi += data_term_sample

            # second term
            for _ in range(self.no_samples):
                f_sample = sample_qf(im_fixed, log_var_f, u_f)
                im_out = enc(f_sample, im_moving_warped)

                data_term_sample = data_loss(im_out).sum() / float(self.no_samples ** 2)
                loss_qphi -= data_term_sample

        loss_qphi.backward()
        optimizer_f.step()
        optimizer_phi.step()

        total_loss += loss_qphi.item()

        log_var_f.requires_grad_(False)
        u_f.requires_grad_(False)

        """
        save image of the warped sphere to disk
        """

        with torch.no_grad():
            warp_field = self.transformation_model.forward_3d(self.identity_grid, mu_v)
            im_moving_warped = self.registration_module(im_moving, warp_field)

            save_im_to_disk(im_moving_warped[0, :, :, :, :], './temp/moving_warped_SSD_all.nii.gz')
