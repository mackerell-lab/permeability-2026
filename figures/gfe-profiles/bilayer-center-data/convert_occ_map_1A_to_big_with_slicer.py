# Copyright (C) 2023 SilcsBio, LLC. All Rights Reserved.

import argparse
import sys
import os
import yaml
from glob import glob
from collections import defaultdict
import numpy as np

sys.path.append('/opt/silcsbio/apps/silcsbio/silcsbio.develop/utils/python')
from clusters import *
from griddata import griddata as gd
from align import rms_transform, apply_transform
from scipy import interpolate
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapsdir', help='FragMaps directory to manipulate')
    parser.add_argument('--yamlfile', help='configuration YAML file')
    parser.add_argument('--map_zscan_data', default=True, help='create gfe zscan data file')
    parser.add_argument('--outdir', default='2b_gen_maps/maps_frm_run', help='new FragMaps directory')

    # Map slicer:
    # Select a region of the input map before converting it to the new voxel size.
    # Use zero-based, end-exclusive indices, e.g. --slice_z 92 113 selects
    # Z slabs 92 through 112 (21 slabs).
    parser.add_argument('--slice_x', nargs=2, type=int, metavar=('START', 'END'),
                        default=None,
                        help='input X slab range [START, END), zero-based')
    parser.add_argument('--slice_y', nargs=2, type=int, metavar=('START', 'END'),
                        default=None,
                        help='input Y slab range [START, END), zero-based')
    parser.add_argument('--slice_z', nargs=2, type=int, metavar=('START', 'END'),
                        default=None,
                        help='input Z slab range [START, END), zero-based')
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    with open(args.yamlfile, "r") as configfile:
        config = yaml.safe_load(configfile)

    files = glob(os.path.join(args.mapsdir, '*.map'))

    # make sure at least one maps in the maps directory
    if len(files) == 0:
        print("Not enough FragMaps files are found.")
        sys.exit()

    # Old maps properties
    xmax, ymax, zmax = config['max_dim']['x'], config['max_dim']['y'], config['max_dim']['z']
    xmin, ymin, zmin = config['min_dim']['x'], config['min_dim']['y'], config['min_dim']['z']
    c = lambda xmax, xmin: (xmax+xmin)/2
    center = np.array((c(xmax, xmin), c(ymax, ymin), c(zmax, zmin)))
    print ("center of the map is (", str(center[:]), ")")
    xspacing, yspacing, zspacing = config['spacing_1A']['x'], config['spacing_1A']['y'], config['spacing_1A']['z']
    spacing_1A = [xspacing, yspacing, zspacing]
    s = lambda xmax, xmin, xspacing: int(int((xmax-xmin) / xspacing + 0.5) / 2) * 2 + 1
    shape_1A = [s(xmax, xmin, xspacing), s(ymax, ymin, yspacing), s(zmax, zmin, zspacing)]
    x_bins_1A, y_bins_1A, z_bins_1A = config['bin_num_1A']['x'], config['bin_num_1A']['y'], config['bin_num_1A']['z']

    # New maps properties
    xspacing_new, yspacing_new, zspacing_new = config['spacing_new']['x'], config['spacing_new']['y'], config['spacing_new']['z']
    spacing_new = [xspacing_new, yspacing_new, zspacing_new]
    shape_new = [s(xmax, xmin, xspacing_new), s(ymax, ymin, yspacing_new), s(zmax, zmin, zspacing_new)]
    # using [5:-5] to remove some defect at the edge of input maps - all dimenion lose 1A voxel
    x_bins_new, y_bins_new, z_bins_new = config['bin_num_new']['x'], config['bin_num_new']['y'], config['bin_num_new']['z']

    # sanity check - Old maps
    assert [shape_1A[0], shape_1A[1], shape_1A[2]] == [x_bins_1A, y_bins_1A, z_bins_1A], f"Number of bins calculated by the script are {shape_1A[0]}, {shape_1A[1]}, {shape_1A[2]} respectively given the input spacing in YAML file. These bin numbers are different from the numberof bins you wanted as given in YAML file" 
    assert [shape_new[0], shape_new[1], shape_new[2]] == [x_bins_new, y_bins_new, z_bins_new], f"Number of bins calculated by the script are {shape_new[0]}, {shape_new[1]}, {shape_new[2]} respectively given the input spacing in YAML file. These bin numbers are different from the numberof bins you wanted as given in YAML file" 
    # Validate bin dimensions - New Maps
    if x_bins_1A % x_bins_new != 0 or y_bins_1A % y_bins_new != 0 or z_bins_1A % z_bins_new != 0:
        raise ValueError("bin_num of new maps in each dimension must be divisible by the number of bins in 1A maps in that respective dimension.")
    
    bin_size_x = x_bins_1A // x_bins_new  # Number of points in each bin along X
    bin_size_y = y_bins_1A // y_bins_new  # Number of points in each bin along Y 
    bin_size_z = z_bins_1A // z_bins_new  # Number of points in each bin along Z

    print (bin_size_x, bin_size_y, bin_size_z)
    print (x_bins_new, y_bins_new, z_bins_new)
    
    for mapfile in files:

        # Old map
        print("Processing mapfile: %s" % mapfile)
        g = gd.load(open(mapfile), format='map')
        values = g.ndelements
        print ("Shape of 1A map: ", np.shape(values))

        # ------------------------------------------------------------
        # Map slicer
        # ------------------------------------------------------------
        # The slicer is applied to the loaded 1-Angstrom map BEFORE
        # binning/conversion.  If a slice is not supplied, the full
        # dimension is used.
        input_shape = values.shape

        def validate_slice(slice_arg, n, name):
            if slice_arg is None:
                return 0, n

            start, end = slice_arg

            if start < 0 or end > n or start >= end:
                raise ValueError(
                    f"Invalid --slice_{name} [{start}, {end}) for "
                    f"dimension of size {n}."
                )

            return start, end

        sx0, sx1 = validate_slice(args.slice_x, input_shape[0], 'x')
        sy0, sy1 = validate_slice(args.slice_y, input_shape[1], 'y')
        sz0, sz1 = validate_slice(args.slice_z, input_shape[2], 'z')

        print(
            f"Selected input slabs: "
            f"X={sx0}:{sx1}, Y={sy0}:{sy1}, Z={sz0}:{sz1}"
        )

        values = values[sx0:sx1, sy0:sy1, sz0:sz1]

        print ("Shape after slicing: ", np.shape(values))

        # ------------------------------------------------------------
        # Conversion dimensions
        # ------------------------------------------------------------
        selected_x, selected_y, selected_z = values.shape

        # The selected region must be divisible by the requested
        # binning factors.
        if selected_x % bin_size_x != 0:
            raise ValueError(
                f"Selected X range ({selected_x} voxels) is not divisible "
                f"by bin_size_x ({bin_size_x})."
            )
        if selected_y % bin_size_y != 0:
            raise ValueError(
                f"Selected Y range ({selected_y} voxels) is not divisible "
                f"by bin_size_y ({bin_size_y})."
            )
        if selected_z % bin_size_z != 0:
            raise ValueError(
                f"Selected Z range ({selected_z} voxels) is not divisible "
                f"by bin_size_z ({bin_size_z})."
            )

        # Number of output bins produced by the selected region.
        selected_x_bins = selected_x // bin_size_x
        selected_y_bins = selected_y // bin_size_y
        selected_z_bins = selected_z // bin_size_z

        print(
            f"Output bins from selected region: "
            f"X={selected_x_bins}, Y={selected_y_bins}, Z={selected_z_bins}"
        )

        # New map grid
        grid = gd.Grid()
        # Use the dimensions generated by the selected slice rather than
        # blindly using the dimensions in the YAML file.
        selected_shape_new = [
            selected_x_bins,
            selected_y_bins,
            selected_z_bins
        ]

        grid.n_elements = np.cumprod(selected_shape_new)[-1]
        grid.spacing = [spacing_new, spacing_new, spacing_new]
        grid.shape = selected_shape_new
        grid.elements = np.zeros(grid.n_elements, dtype=np.float32)
        grid.center = g.center
        print ("Shape of new map: ", np.shape(grid.ndelements))

        for i in range(selected_x_bins):
            for j in range(selected_y_bins):
                for k in range(selected_z_bins):
                     grid.ndelements[
                        i * bin_size_x : (i + 1) * bin_size_x,  # Fill the X-bin
                        j * bin_size_y : (j + 1) * bin_size_y,  # Fill the Y-bin
                        k * bin_size_z : (k + 1) * bin_size_z  # Fill the Z-bin

                    ] = np.sum(values[
                        i * bin_size_x : (i + 1) * bin_size_x,  # Fill the X-bin
                        j * bin_size_y : (j + 1) * bin_size_y,  # Fill the Y-bin
                        k * bin_size_z : (k + 1) * bin_size_z  # Fill the Z-bin
                        ])

        prefix = os.path.splitext(os.path.basename(mapfile))[0]

        new_mapfile = '%s.map' % (prefix)
        grid.elements = grid.ndelements.reshape(grid.n_elements) # , order="C")
        print(grid.elements)
        zscan_filename = '%s_map' % (prefix)
        # This is for the metadata for autodock. Else grid.save will give an error
        grid.spacing = [1.0, 1.0, 1.0]
        gd.save(grid, open(os.path.join(args.outdir, new_mapfile), 'w'), format='map')

        if args.map_zscan_data:
            zscan = np.mean(grid.ndelements, axis=(0, 1))
            np.save(str(os.path.join(args.outdir)) + "/" + zscan_filename + '.npy', zscan)
