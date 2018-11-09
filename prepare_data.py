"""Prepare DSTL data: load tiff, create masks from WKT polygons."""
import argparse, os
import numpy as np
import pandas as pd
import tifffile
from shapely.wkt import loads as wkt_loads
import cv2

NUM_CLASSES = 10
CLASS_NAMES = ['Buildings', 'Misc structures', 'Road', 'Track', 'Trees', 'Crops', 'Waterway', 'Standing water', 'Vehicle large', 'Vehicle small']

def create_mask(shape, polygons, grid_sizes):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if not polygons or polygons == 'MULTIPOLYGON EMPTY':
        return mask
    poly = wkt_loads(polygons)
    if poly.is_empty: return mask
    x_max, y_min = grid_sizes
    geoms = list(poly.geoms) if poly.geom_type == 'MultiPolygon' else [poly]
    for g in geoms:
        pts = np.array(g.exterior.coords)
        pts[:, 0] = pts[:, 0] * w / x_max
        pts[:, 1] = pts[:, 1] * h / y_min
        cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/dstl')
    parser.add_argument('--output', default='data/processed')
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    train_wkt = pd.read_csv(os.path.join(args.data_dir, 'train_wkt_v4.csv'))
    grid_sizes = pd.read_csv(os.path.join(args.data_dir, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
    for img_id in train_wkt['ImageId'].unique():
        tiff_path = os.path.join(args.data_dir, 'three_band', f'{img_id}.tif')
        if not os.path.exists(tiff_path): continue
        img = np.transpose(tifffile.imread(tiff_path), (1, 2, 0))
        gs = grid_sizes[grid_sizes['ImageId'] == img_id]
        masks = np.zeros((img.shape[0], img.shape[1], NUM_CLASSES), dtype=np.uint8)
        for _, row in train_wkt[train_wkt['ImageId'] == img_id].iterrows():
            masks[:, :, row['ClassType'] - 1] = create_mask(img.shape, row['MultipolygonWKT'], (gs['Xmax'].values[0], abs(gs['Ymin'].values[0])))
        np.savez(os.path.join(args.output, f'{img_id}.npz'), image=img, masks=masks)
        print(f"  {img_id}: {img.shape}")

if __name__ == '__main__':
    main()
