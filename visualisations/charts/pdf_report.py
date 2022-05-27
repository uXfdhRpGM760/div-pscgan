from matplotlib.backends.backend_pdf import PdfPages
from visualisations_settings import PDF_IMG_NUMBER
from mse_to_gt import mse_to_gt_by_image_idx
from plot_fids import create_plot
from image_grid import get_image_grid


from matplotlib import pyplot as plt

def create_report():
    REPORT_FOLDER = ''
    pp = PdfPages(REPORT_FOLDER + 'report_842_19.pdf')
    for idx in range(PDF_IMG_NUMBER):
        fig, (ax1, ax2) = plt.subplots(2, figsize=[13, 27])
        ax1.axis('off')
        mse_to_gt_by_image_idx(ax2, idx)
        get_image_grid(fig, idx)
        pp.savefig(fig)
    fig, ax = plt.subplots(1, figsize=[13, 8])
    create_plot(ax)
    pp.savefig(fig)
    pp.close()

if __name__ == '__main__':
    create_report()
