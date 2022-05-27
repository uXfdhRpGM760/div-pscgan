import numpy as np

from visualisations_settings import DATA_PATH, TICKS


def load_fids():
    n2c_fid = np.load(DATA_PATH + 'fid_n2c.npy')
    dGAN_fid = np.load(DATA_PATH + 'fids_dgan.npy')
    return n2c_fid, dGAN_fid


def create_plot(ax):
    n2c_fid, dGAN_fid = load_fids()
    ax.plot(TICKS, [n2c_fid]*len(TICKS), '--', color='red', label='fid_N2C')
    ax.plot(TICKS, dGAN_fid, '-', color='blue', label='fid_dGAN')
    ax.legend()
