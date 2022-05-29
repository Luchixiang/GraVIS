# GraVIS: Grouping Augmented Views from Independent Sources for Dermatology Analysis
This repo is the official implementation of our TMI paper titled "GraVIS: Grouping Augmented Views from Independent Sources for Dermatology Analysis". In this repo, we demonstrate how to use GraVIS to conduct pre-training on ISIC2017 dataset. The employed backbones are ResNet-50.

### Dependency

Please install PyTorch (>=1.1) before you run the code. We strongly recommend you to install Anaconda3 where we use Python 3.6. 

### Run GraVIS

#### Step 1

Download ISIC2017 dataset from [this link](https://challenge.isic-archive.com/data/).

The image folder of ISIC2017 should look like this:

```
./ISIC-2017_Training_Data
	ISIC-2017_Training_Data_metadata.csv
	ISIC_0015295_superpixels.png
	ISIC_0015295.jpg
	ISIC_0015284_superpixels.png
	ISIC_0015284.jpg
	....
```

Besides, we also provide the list of training image in `isic_seg_train.txt`.

#### Step 2

```bash
git clone https://github.com/Luchixiang/GraVIS.git
cd GraVIS
```

#### Step 3

```bash
python main.py --data /home/luchixiang/isic/ISIC-2017_Training_Data/0  --epochs 240 --ratio 1.0 --lr 1e-3 --output ./weight --workers 16 --gpus 0,1,2,3 --b 16
```

or 

```
bash run.sh
```

 Please replace the `/home/luchixiang/isic/ISIC-2017_Training_Data/0` with the your ISIC2017 data path

`--epochs` defines the number of training epochs

`--ratio` determines the percentages of images in the training set for pretraining. Here, `1` means using all training images in the training set to for pretraining.

`--lr` represents the learning rate

`--output` denotes the path to save the weight

`--b` is the batch size. Please note that the true batch size is `16 * 20` where 20 denotes that we apply augment one image 20 times to generate the positive samples.