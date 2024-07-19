import matplotlib.pyplot as plt
import astra
import numpy as np
from PIL import Image
from os.path import exists
from os import listdir, makedirs
import sys

# Add necessary paths for custom imports
sys.path.append("..")
sys.path.append("../src")

# Import custom modules
from src.algorithms_OhGreat.DART import *
from src.algorithms_OhGreat.SART import *
from src.algorithms_OhGreat.SIRT import *
from src.algorithms_OhGreat.FBP import *
from src.projections_OhGreat.project import *

def main():
    n_det_spacing = [0.8, 1, 1.3, 2, 5]
    # Total iterations for comparison algorithms
    iters = 1000
    # DART parameters
    dart_iters = 10
    rec_alg_iters = 1000
    p_fixed = 0.9
    # Phantom family
    phants_fam = ["bone"] #"butterfly", "checkbox","bat", "text",
    # Define number of projections and angles
    n_projections = [2]  # 36, 18, 9, 4,
    angle_range = []
    """angle_range.append(np.deg2rad([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115,
                   120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175]))
    angle_range.append(np.deg2rad([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]))
    angle_range.append(np.deg2rad([0, 20, 40, 60, 80, 100, 120, 140, 160, 175]))
    angle_range.append(np.deg2rad([0, 45, 90, 135]))
    angle_range.append(np.deg2rad([0, 90]))"""

    noises = [ None, 1500, 3000, 4500, 6000, 7500, 9000]

    for phantoms in phants_fam:
        # Input directory
        in_dir = f"../phantoms/{phantoms}/"

        for phantom_name in sorted(listdir(in_dir)):
            # Output directory
            out_dir_proj = f"../results/n_proj/{phantoms}/{phantom_name}/"
            out_dir_angles = f"../results/angle_range/{phantoms}/{phantom_name}/"
            if not exists(out_dir_proj):
                makedirs(out_dir_proj)
            if not exists(out_dir_angles):
                makedirs(out_dir_angles)

            # Choose a phantom
            phantom = np.array(Image.open(in_dir + phantom_name), dtype=np.uint8)
            img_width, img_height = phantom.shape
            print(f"Curr phantom: {phantom_name}")


            abs_errors_sart = []
            abs_errors_sirt = []
            abs_errors_rbf = []
            abs_errors_dart_sart = []
            abs_errors_dart_fbp = []
            abs_errors_dart_sirt = []

            rel_errors_sart = []
            rel_errors_sirt = []
            rel_errors_rbf = []
            rel_errors_dart_sart = []
            rel_errors_dart_fbp = []
            rel_errors_dart_sirt = []


            for curr_proj in n_projections:
                for det_spacing in n_det_spacing:
                    for noise in noises:
                        print("curr_proj:", curr_proj)
                        print("det_spacing:", det_spacing)
                        print("~current noise level:", noise, "~")

                        n_proj, n_detectors = curr_proj, 512
                        vol_geom = astra.creators.create_vol_geom([img_width, img_height])
                        phantom_id = astra.data2d.create('-vol', vol_geom, data=phantom)
                        angles = np.linspace(0, np.pi, curr_proj, endpoint=False)

                        # for angles in angle_range:
                        # save_dir = out_dir_angles+f"{phantoms}_proj_{curr_proj}_detSpace_{det_spacing}_noise_{noise}"
                        projector_id, sino_id, sinogram = project_from_2D(phantom_id=phantom_id,
                                                                          vol_geom=vol_geom,
                                                                          n_projections=n_proj,
                                                                          n_detectors=n_detectors,
                                                                          detector_spacing=det_spacing,
                                                                          angles=angles,
                                                                          noise_factor=noise,
                                                                          use_gpu=True)

                        proj_geom = astra.create_proj_geom('parallel', det_spacing,
                                                            n_detectors, angles)

                        # DART with SART
                        gray_lvls = np.unique(phantom).astype(np.float32)
                        d = DART(gray_levels=gray_lvls, p=p_fixed, rec_shape=phantom.shape,
                                 proj_geom=proj_geom, projector_id=projector_id,
                                 sinogram=sinogram)
                        # run the algorithm
                        dart_res = d.run(iters=dart_iters, rec_alg="SART_CUDA", rec_iter=rec_alg_iters)
                        # abs_errors_dart_sart.append(np.abs(phantom - dart_res).mean())
                        # rel_errors_dart_sart.append(np.sum(np.abs(phantom - dart_res)) / np.sum(np.abs(phantom)))

                        np.save(out_dir_angles + f"{phantoms}_Dart_sart_proj_{curr_proj}_detSpace_{det_spacing}_noise_{noise}", dart_res)

                        # im = Image.fromarray(dart_res.astype(np.uint8))
                        # im = im.convert('L')
                        # im.save(out_dir_angles+f"Dart_sart_{phantoms}_proj_{curr_proj}_detSpace_{det_spacing}_noise_{noise}.png", "PNG")


                        # Display the reconstructed image
                        fig = plt.figure()
                        plt.imshow(dart_res)
                        plt.title(f'Dart sart {phantoms} proj:{curr_proj} detSpace:{det_spacing} noise:{noise}')
                        plt.savefig(out_dir_angles+f"{phantoms}_Dart_sart_proj_{curr_proj}_detSpace_{det_spacing}_noise_{noise}.png",  dpi=fig.dpi)
                        plt.show()


                        # DART with SIRT
                        gray_lvls = np.unique(phantom).astype(np.float32)
                        d = DART(gray_levels=gray_lvls, p=p_fixed, rec_shape=phantom.shape,
                                 proj_geom=proj_geom, projector_id=projector_id,
                                 sinogram=sinogram)
                        # run the algorithm
                        dart_res = d.run(iters=dart_iters, rec_alg="SIRT_CUDA", rec_iter=rec_alg_iters)
                        # abs_errors_dart_sirt.append(np.abs(phantom - dart_res).mean())
                        # rel_errors_dart_sirt.append(np.sum(np.abs(phantom - dart_res)) / np.sum(np.abs(phantom)))

                        np.save(out_dir_angles + f"{phantoms}_Dart_sirt_proj_{curr_proj}_detSpace_{det_spacing}_noise_{noise}", dart_res)

                        # im = Image.fromarray(dart_res.astype(np.uint8))
                        # im = im.convert('L')
                        # im.save(out_dir_angles+f"Dart sirt_{phantoms}_proj_{curr_proj}_detSpace_{det_spacing}_noise_{noise}.png", "PNG")


                        # Display the reconstructed image
                        fig = plt.figure()
                        plt.imshow(dart_res)
                        plt.title(f'Dart_sirt {phantoms} proj:{curr_proj} detSpace:{det_spacing} noise:{noise}')
                        plt.savefig(out_dir_angles+f"{phantoms}_Dart_sirt_proj_{curr_proj}_detSpace_{det_spacing}_noise_{noise}.png", dpi=fig.dpi)
                        plt.show()


                        """  # DART with FBP
                        d = DART(gray_levels=gray_lvls, p=p_fixed, rec_shape=phantom.shape,
                                 proj_geom=proj_geom, projector_id=projector_id,
                                 sinogram=sinogram)
                        # run the algorithm
                        dart_res = d.run(iters=dart_iters, rec_alg="FBP_CUDA", rec_iter=rec_alg_iters)
                        abs_errors_dart_fbp.append(np.abs(phantom - dart_res).mean())
                        rel_errors_dart_fbp.append(np.sum(np.abs(phantom - dart_res)) / np.sum(np.abs(phantom)))

                        im = Image.fromarray(dart_res.astype(np.uint8))
                        # im = im.convert('L')
                        im.save(out_dir_angles+f"Dart_fbp_{phantoms}_proj_{curr_proj}_detSpace_{det_spacing}_noise_{noise}.png", "PNG")


                        # Display the reconstructed image
                        plt.figure()
                        plt.imshow(dart_res, cmap='gray')
                        plt.title(f'Dart fbp {phantoms} proj:{curr_proj} detSpace:{det_spacing} noise:{noise}')
                        plt.show()"""

                        """
                        # SART
                        _, sart_res = SART(vol_geom, 0, projector_id, sino_id,
                                           iters, use_gpu=True)
                        abs_errors_sart.append(np.abs(phantom - sart_res).mean())
                        rel_errors_sart.append(np.abs(np.sum(np.abs(phantom - dart_res)) / np.sum(np.abs(phantom))))

                        # SIRT
                        _, sirt_res = SIRT(vol_geom, 0, sino_id, iters, use_gpu=True)
                        abs_errors_sirt.append(np.abs(phantom - sirt_res).mean())
                        rel_errors_sirt.append(np.abs(np.sum(np.abs(phantom - dart_res)) / np.sum(np.abs(phantom))))

                        # RBF
                        _, fbp_res = FBP(vol_geom, 0, projector_id, sino_id,
                                         iters, use_gpu=True)
                        abs_errors_rbf.append(np.abs(phantom - fbp_res).mean())
                        rel_errors_rbf.append(np.sum(np.abs(phantom - dart_res)) / np.sum(np.abs(phantom)))
"""
                        astra.data2d.clear()
                        astra.projector.clear()
                        astra.algorithm.clear()

                            # Mean Absolute Error (MAE)
                            # mae = np.abs(phantom - dart_res).mean()
                            # Relative Mean Error (RME)
                            # rme = np.sum(np.abs(phantom - dart_res)) / np.sum(np.abs(phantom))



"""            np.save(out_dir_angles + f"abs_SART_{phantoms}", abs_errors_sart)
            np.save(out_dir_angles + f"abs_SIRT_{phantoms}", abs_errors_sirt)
            # np.save(out_dir_angles + f"abs_RBF_{phantoms}", abs_errors_rbf)
            np.save(out_dir_angles + f"abs_DART_sart_{phantoms}", abs_errors_dart_sart)
            np.save(out_dir_angles + f"abs_DART_sirt_{phantoms}", abs_errors_dart_sirt)
            # np.save(out_dir_angles + f"abs_DART_fbp_{phantoms}", abs_errors_dart_fbp)

            np.save(out_dir_angles + f"rel_SART_{phantoms}", rel_errors_sart)
            np.save(out_dir_angles + f"rel_SIRT_{phantoms}", rel_errors_sirt)
            # np.save(out_dir_angles + f"rel_RBF_{phantoms}", rel_errors_rbf)
            np.save(out_dir_angles + f"rel_DART_sart_{phantoms}", rel_errors_dart_sart)
            np.save(out_dir_angles + f"rel_DART_sirt_{phantoms}", rel_errors_dart_sirt)
            # np.save(out_dir_angles + f"rel_DART_fbp_{phantoms}", rel_errors_dart_fbp)"""


if __name__ == "__main__":
    main()
