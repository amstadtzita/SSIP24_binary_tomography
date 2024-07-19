import os
import numpy as np
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt

def load_npy_files(parent_path, contains_string):
    """Load all .npy files in a directory whose names contain a certain string."""
    loaded_arrays = {}
    for filename in os.listdir(parent_path):
        if filename.endswith('.npy') and contains_string in filename:
            file_path = os.path.join(parent_path, filename)
            loaded_arrays[filename] = np.load(file_path)
    return loaded_arrays

phants_fam = ["butterfly", "checkbox", "bat", "text", "bone"]
parent_path = '../results/angle_range/'

n_projections = [36, 18, 9, 4, 2]
noises = [None, 1500, 3000, 4500, 6000, 7500, 9000]
n_det_spacing = [0.8, 1, 1.3, 2, 5]

fig_size = (15, 15)

tick_labels_noise = [0, 1500, 3000, 4500, 6000, 7500, 9000]
tick_labels_noise = [str(np.round((noise/53418)*100, 1))+"%" if noise is not None else "0%" for noise in tick_labels_noise]
tick_labels_projections = [str(proj) for proj in n_projections]
tick_labels_spacing = [str(spacing) for spacing in n_det_spacing]

fig, axs = plt.subplots(3, 1, figsize=fig_size)

colors = ['b', 'g', 'r', 'c', 'm']  # Different colors for different phantoms

for phantom_idx, phantom_name in enumerate(phants_fam):
    print(phantom_idx, phantom_name)

    in_dir = f"../phantoms/{phantom_name}/"
    out_dir_angles = f"../results/angle_range/{phantom_name}/{phantom_name}_dart.pgm/"
    phantom = np.array(Image.open(in_dir + sorted(listdir(in_dir))[0]), dtype=np.uint8)

    color = colors[phantom_idx % len(colors)]

    # Effect of Poisson noise
    rel_errors_dart_sart_noise = []
    rel_errors_dart_sirt_noise = []

    for curr in noises:
        n_projections = 36
        det_spacing = 1
        contains_string_sart = f"{phantom_name}_Dart_sart_proj_{n_projections}_detSpace_{det_spacing}_noise_{curr}"
        contains_string_sirt = f"{phantom_name}_Dart_sirt_proj_{n_projections}_detSpace_{det_spacing}_noise_{curr}"

        loaded_array_sart = load_npy_files(out_dir_angles, contains_string_sart)
        loaded_array_sirt = load_npy_files(out_dir_angles, contains_string_sirt)

        # Calculate relative errors
        for filename, array in loaded_array_sart.items():
            dart_res = array
            rel_errors_dart_sart_noise.append(np.sum(np.abs(phantom - dart_res)) / np.sum(np.abs(phantom)))

        for filename, array in loaded_array_sirt.items():
            dart_res = array
            rel_errors_dart_sirt_noise.append(np.sum(np.abs(phantom - dart_res)) / np.sum(np.abs(phantom)))

    axs[0].plot(tick_labels_noise, rel_errors_dart_sart_noise, label=f"SART - {phantom_name}", linestyle="solid", linewidth=2, color=color)
    axs[0].plot(tick_labels_noise, rel_errors_dart_sirt_noise, label=f"SIRT - {phantom_name}", linestyle="--", linewidth=2, color=color)

    # Effect of projections
    rel_errors_dart_sart_proj = []
    rel_errors_dart_sirt_proj = []

    for curr in n_projections:
        det_spacing = 1
        noise = None
        contains_string_sart = f"{phantom_name}_Dart_sart_proj_{curr}_detSpace_{det_spacing}_noise_{noise}"
        contains_string_sirt = f"{phantom_name}_Dart_sirt_proj_{curr}_detSpace_{det_spacing}_noise_{noise}"

        loaded_array_sart = load_npy_files(out_dir_angles, contains_string_sart)
        loaded_array_sirt = load_npy_files(out_dir_angles, contains_string_sirt)

        # Calculate relative errors
        for filename, array in loaded_array_sart.items():
            dart_res = array
            rel_errors_dart_sart_proj.append(np.sum(np.abs(phantom - dart_res)) / np.sum(np.abs(phantom)))

        for filename, array in loaded_array_sirt.items():
            dart_res = array
            rel_errors_dart_sirt_proj.append(np.sum(np.abs(phantom - dart_res)) / np.sum(np.abs(phantom)))

    axs[1].plot(n_projections, rel_errors_dart_sart_proj, label=f"SART - {phantom_name}", linestyle="solid", linewidth=2, color=color)
    axs[1].plot(n_projections, rel_errors_dart_sirt_proj, label=f"SIRT - {phantom_name}", linestyle="--", linewidth=2, color=color)

    # Effect of detector spacing
    rel_errors_dart_sart_spacing = []
    rel_errors_dart_sirt_spacing = []

    for curr in n_det_spacing:
        proj = 36
        noise = None
        contains_string_sart = f"{phantom_name}_Dart_sart_proj_{proj}_detSpace_{curr}_noise_{noise}"
        contains_string_sirt = f"{phantom_name}_Dart_sirt_proj_{proj}_detSpace_{curr}_noise_{noise}"

        loaded_array_sart = load_npy_files(out_dir_angles, contains_string_sart)
        loaded_array_sirt = load_npy_files(out_dir_angles, contains_string_sirt)

        # Calculate relative errors
        for filename, array in loaded_array_sart.items():
            dart_res = array
            rel_errors_dart_sart_spacing.append(np.sum(np.abs(phantom - dart_res)) / np.sum(np.abs(phantom)))

        for filename, array in loaded_array_sirt.items():
            dart_res = array
            rel_errors_dart_sirt_spacing.append(np.sum(np.abs(phantom - dart_res)) / np.sum(np.abs(phantom)))

    axs[2].plot(n_det_spacing, rel_errors_dart_sart_spacing, label=f"SART - {phantom_name}", linestyle="solid", linewidth=2, color=color)
    axs[2].plot(n_det_spacing, rel_errors_dart_sirt_spacing, label=f"SIRT - {phantom_name}", linestyle="--", linewidth=2, color=color)

for ax in axs:
    ax.set_ylabel("Relative mean pixel error", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)

axs[0].set_xlabel("Poisson noise %", fontsize=14)
axs[0].set_title("Effect of Poisson noise", fontsize=16)
axs[0].set_xticks(tick_labels_noise)
axs[0].set_xticklabels(tick_labels_noise)

axs[1].set_xlabel("Number of Projections", fontsize=14)
axs[1].set_title("Effect of projections", fontsize=16)
axs[1].set_xticks(n_projections)
axs[1].set_xticklabels(tick_labels_projections)

axs[2].set_xlabel("Detector spacing", fontsize=14)
axs[2].set_title("Effect of detector spacing", fontsize=16)
axs[2].set_xticks(n_det_spacing)
axs[2].set_xticklabels(tick_labels_spacing)

fig.suptitle(f"Effect of Different Parameters for All Phantoms", fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(hspace=0.35)
plt.show()
