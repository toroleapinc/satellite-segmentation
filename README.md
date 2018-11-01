# Satellite Segmentation

Multi-class segmentation of satellite images from the DSTL Kaggle competition. Uses U-Net with attention gates for 10 land-use classes (buildings, roads, trees, crops, water, etc).

Get the data from https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection

```bash
pip install -r requirements.txt
python prepare_data.py --data-dir data/dstl/
python run_training.py --epochs 80
```
