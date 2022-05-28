from __future__ import division, absolute_import, print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pyeit.mesh as mesh
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines
from pyeit.mesh.shape import thorax, circle
from pyeit.eit.interp2d import sim2pts

import pyeit.eit.jac as jac
import pyeit.eit.bp as bp
import pyeit.eit.greit as greit


class Eit:
    def __init__(self, fpath = './coordinate.csv', n_el = 16, model = 'thorax', chunk_size = 1):
        # build mesh
        self.n_el = n_el
        self.model = model
        self.mesh_obj, self.el_pos = mesh.create(n_el, h0=0.1, fd=thorax if model == 'thorax' else circle)
        self.el_dist, self.step = 1, 1

        # extract node, element, alpha
        self.pts = self.mesh_obj["node"]
        self.tri = self.mesh_obj["element"]
        # coordinate data
        self.data = pd.read_csv(fpath)
        self.fpath = fpath
        self.chunk_size = chunk_size

    def set_data(self, fpath):
        self.data = pd.read_csv(fpath)
        self.fpath = fpath

    def set_model(self, model):
        self.mesh_obj, self.el_pos = mesh.create(self.n_el, h0=0.1, fd=thorax if model == 'thorax' else circle)
        self.el_dist, self.step = 1, 1

        # extract node, element, alpha
        self.pts = self.mesh_obj["node"]
        self.tri = self.mesh_obj["element"]

    def solver_bp(self, id):
        num_data = len(self.data.x)
        anomaly = list(
            {"x": self.data.x[i % num_data], "y": self.data.y[i % num_data], "d": self.data.d[i % num_data], "perm": self.data.perm[i % num_data]} 
            for i in range(id, id + self.chunk_size) 
        )
        mesh_new = mesh.set_perm(self.mesh_obj, anomaly=anomaly, background=1.0)

        el_dist, step = 1, 1
        ex_mat = eit_scan_lines(16, el_dist)

        # calculate simulated data
        fwd = Forward(self.mesh_obj, self.el_pos)
        f0 = fwd.solve_eit(ex_mat, step=step, perm=self.mesh_obj["perm"])
        f1 = fwd.solve_eit(ex_mat, step=step, perm=mesh_new["perm"])

        # ? naive inverse solver using back-projection
        eit = bp.BP(self.mesh_obj, self.el_pos, ex_mat=ex_mat, step=1, parser="std")
        eit.setup(weight="none")
        ds = 192.0 * eit.solve(f1.v, f0.v, normalize=False)

        return {
            'mtd': 'BP',
            'id': id,
            'ds': ds
        }

    def solver_greit(self, id):
        num_data = len(self.data.x)
        anomaly = list(
            {"x": self.data.x[i % num_data], "y": self.data.y[i % num_data], "d": self.data.d[i % num_data], "perm": self.data.perm[i % num_data]} 
            for i in range(id, id + self.chunk_size) 
        )
        mesh_new = mesh.set_perm(self.mesh_obj, anomaly=anomaly, background=1.0)
        ex_mat = eit_scan_lines(self.n_el, self.el_dist)

        # calculate simulated data
        fwd = Forward(self.mesh_obj, self.el_pos)
        f0 = fwd.solve_eit(ex_mat, step=self.step, perm=self.mesh_obj["perm"])
        f1 = fwd.solve_eit(ex_mat, step=self.step, perm=mesh_new["perm"])

        """ 3. Construct using GREIT """
        eit = greit.GREIT(self.mesh_obj, self.el_pos, ex_mat=ex_mat, step=self.step, parser="std")
        eit.setup(p=0.50, lamb=0.001)
        ds = eit.solve(f1.v, f0.v)
        _, _, ds = eit.mask_value(ds, mask_value=np.NAN)

        return {
            'mtd': 'GREIT',
            'id': id,
            'ds': ds
        }

    def solver_jac(self, id):
        self.mesh_obj["alpha"] = np.random.rand(self.tri.shape[0]) * 200 + 100
        num_data = len(self.data.x)
        anomaly = list(
            {"x": self.data.x[i % num_data], "y": self.data.y[i % num_data], "d": self.data.d[i % num_data], "perm": self.data.perm[i % num_data]} 
            for i in range(id, id + self.chunk_size) 
        )
        mesh_new = mesh.set_perm(self.mesh_obj, anomaly=anomaly)

        """ 2. FEM simulation """
        ex_mat = eit_scan_lines(self.n_el, self.el_dist)

        # calculate simulated data
        fwd = Forward(self.mesh_obj, self.el_pos)
        f0 = fwd.solve_eit(ex_mat, step=self.step, perm=self.mesh_obj["perm"])
        f1 = fwd.solve_eit(ex_mat, step=self.step, perm=mesh_new["perm"])

        """ 3. JAC solver """
        eit = jac.JAC(
            self.mesh_obj,
            self.el_pos,
            ex_mat=ex_mat,
            step=self.step,
            perm=1.0,
            parser="std",
        )

        eit.setup(p=0.5, lamb=0.01, method="kotre")
        ds = eit.solve(f1.v, f0.v, normalize=True)
        ds_n = sim2pts(self.pts, self.tri, np.real(ds))
        
        return {
            'mtd': 'JAC',
            'id': id,
            'ds': ds,
            'ds_n': ds_n
        }
    
    def solver(self, mtd = 'BP', id = 0):
        if mtd == 'BP':
            return self.solver_bp(id)
        elif mtd == 'GREIT':
            return self.solver_greit(id)
        elif mtd == 'JCA':
            return self.solver_jac(id)

    def plot(self, ax, title='Reconstituted $\Delta$ Conductivities', data={}):
        if data['mtd'] == 'BP':
            im = ax.tripcolor(self.pts[:, 0], self.pts[:, 1], self.tri, data['ds'], cmap=plt.cm.viridis)
            ax.axis("equal")
            ax.set_title(title)

        elif data['mtd'] == 'GREIT':
            im = ax.imshow(np.real(data['ds']), interpolation="none", cmap=plt.cm.viridis)
            ax.axis("equal")
            ax.set_title(title)

        elif data['mtd'] == 'JAC':
            x, y = self.pts[:, 0], self.pts[:, 1]
            im = ax.tripcolor(x, y, self.tri, data['ds_n'], shading="flat")
            for i, e in enumerate(self.el_pos):
                ax.annotate(str(i + 1), xy=(x[e], y[e]), color="r")
            ax.set_aspect("equal")


def solver_bp_mut( id, data, chunk_size, mesh_obj, el_pos):
    num_data = len(data.x)
    anomaly = list(
        {"x": data.x[i % num_data], "y": data.y[i % num_data], "d": data.d[i % num_data], "perm": data.perm[i % num_data]} 
        for i in range(id, id + chunk_size) 
    )

    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)

    el_dist, step = 1, 1
    ex_mat = eit_scan_lines(16, el_dist)

    # calculate simulated data
    fwd = Forward(mesh_obj, el_pos)
    f0 = fwd.solve_eit(ex_mat, step=step, perm=mesh_obj["perm"])
    f1 = fwd.solve_eit(ex_mat, step=step, perm=mesh_new["perm"])

    # ? naive inverse solver using back-projection
    eit = bp.BP(mesh_obj, el_pos, ex_mat=ex_mat, step=1, parser="std")
    eit.setup(weight="none")
    ds = 192.0 * eit.solve(f1.v, f0.v, normalize=False)

    return {
        'mtd': 'BP',
        'id': id,
        'ds': ds
    }

def solver_greit_mut( id, data, chunk_size, mesh_obj, el_pos, n_el, el_dist):
    num_data = len(data.x)
    anomaly = list(
        {"x": data.x[i % num_data], "y": data.y[i % num_data], "d": data.d[i % num_data], "perm": data.perm[i % num_data]} 
        for i in range(id, id + chunk_size) 
    )
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
    ex_mat = eit_scan_lines(n_el, el_dist)

    # calculate simulated data
    fwd = Forward(mesh_obj, el_pos)
    f0 = fwd.solve_eit(ex_mat, step=step, perm=mesh_obj["perm"])
    f1 = fwd.solve_eit(ex_mat, step=step, perm=mesh_new["perm"])

    """ 3. Construct using GREIT """
    eit = greit.GREIT(mesh_obj, el_pos, ex_mat=ex_mat, step=step, parser="std")
    eit.setup(p=0.50, lamb=0.001)
    ds = eit.solve(f1.v, f0.v)
    _, _, ds = eit.mask_value(ds, mask_value=np.NAN)

    return {
        'mtd': 'GREIT',
        'id': id,
        'ds': ds
    }

def solver_jac_mut(id, data, chunk_size, mesh_obj, el_pos, tri, step, n_el, el_dist):
    mesh_obj["alpha"] = np.random.rand(tri.shape[0]) * 200 + 100
    num_data = len(data.x)
    anomaly = list(
        {"x": data.x[i % num_data], "y": data.y[i % num_data], "d": data.d[i % num_data], "perm": data.perm[i % num_data]} 
        for i in range(id, id + chunk_size) 
    )
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly)

    """ 2. FEM simulation """
    ex_mat = eit_scan_lines(n_el, el_dist)

    # calculate simulated data
    fwd = Forward(mesh_obj, el_pos)
    f0 = fwd.solve_eit(ex_mat, step=step, perm=mesh_obj["perm"])
    f1 = fwd.solve_eit(ex_mat, step=step, perm=mesh_new["perm"])

    """ 3. JAC solver """
    eit = jac.JAC(
        mesh_obj,
        el_pos,
        ex_mat=ex_mat,
        step=step,
        perm=1.0,
        parser="std",
    )

    eit.setup(p=0.5, lamb=0.01, method="kotre")
    ds = eit.solve(f1.v, f0.v, normalize=True)
    ds_n = sim2pts(pts, tri, np.real(ds))
    
    return {
        'mtd': 'JAC',
        'id': id,
        'ds': ds,
        'ds_n': ds_n
    }
    

def solver_mut( mtd, id, data, chunk_size, mesh_obj, el_pos, tri, step, n_el, el_dist):
    if mtd == 'BP':
        return solver_bp_mut(id, data, chunk_size, mesh_obj, el_pos)
    elif mtd == 'GREIT':
        return solver_greit_mut(id, data, chunk_size, mesh_obj, el_pos, n_el, el_dist)
    elif mtd == 'JCA':
        return solver_jac_mut(id, data, chunk_size, mesh_obj, el_pos, tri, step, n_el, el_dist)
