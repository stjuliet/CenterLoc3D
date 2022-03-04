# SVLD-3D Dataset

## Introduction
Surveillance Vehicle Localization Dataset - 3D vehicle detection and localization for monocular roadside cameras.
- 5 scenes with 3 typical  vehicle types.
- 14592 train/val images, 2237 test images.
- Examples:
![image](imgs/example.png)

## Directory Structure
```
SVLD-3D                             # root directory
    ├── DATA2021                    # trainval set
        ├── Annotations             # annotation
            ├── {image_id}.xml
        ├── Calib                   # calibration parameters (world to image)
            ├── {scene_id}.xml
        ├── ImageSets               # dataset split
            ├── Main
                ├── trainval.txt
                ├── train.txt
                ├── val.txt
                ├── test.txt
        ├── JPEGImages              # image (jpg/png)
            ├── {image_id}.jpg/png
    ├── TESTDATA2021                # test set
        ├── Annotations             # annotation
            ├── {image_id}.xml
        ├── Calib                   # calibration parameters (world to image)
            ├── {scene_id}.xml
        ├── JPEGImages              # image (jpg/png)
            ├── {image_id}.jpg/png
    ├── split_train_val_test.py     # split train/val/test set
    ├── README.md                   # instruction
```

## Annotation Format
```
<annotation>
	<filename>absolute path: {image_id}.jpg/png</filename>
	<calibfile>absolute path: {scene_id}.xml</calibfile>
	<size>
		<width>image_width</width>
		<height>image_height</height>
		<depth>image_depth</depth>
	</size>
	<object>
		<type>vehicle type</type>
		<bbox2d>2D bbox: left, top, right, bottom</bbox2d>
		<vertex2d>3D bbox(image): pt1~pt8</vertex2d>
		<veh_size>3D bbox dimension(m)</veh_size>
		<perspective>view: left/right</perspective>
		<base_point>3D bbox base point(image): pt2</base_point>
		<vertex3d>3D bbox(world): pt1~pt8</vertex3d>
		<veh_loc_2d>3D bbox centroid(image)</veh_loc_2d>
	</object>
	<object>
            ...
	</object>
	...
</annotation>
```

## Vehicle Type

| Type              | Annotation        |
| ----------------- | ----------------- |
| car               | Car               |
| truck             | Truck             |
| bus               | Bus               |

## Download
All the images and annotations in SVLD-3D are available at Google Drive.

### train/val set:

- [DATA2021: Scene A-C](https://drive.google.com/file/d/1yO9av6qtLxOehT-vRivaX_WX6TT-jIvc/view?usp=sharing)
- [DATA2021: Scene D](https://drive.google.com/file/d/13i341cnFtNVlRvVBjSAmUGtxEiHt_Fjz/view?usp=sharing)
- [DATA2021: Scene E](https://drive.google.com/file/d/1Z-lv_NQtBc8EFYJaFapF69H79m2Ce2mO/view?usp=sharing)

### test set:

- [TESTDATA2021: Scene A-E](https://drive.google.com/file/d/10zspX_-gWoQOwzN5Ba-MDQVsV6Omh80A/view?usp=sharing)

## Citation
```
@article{tang2022CenterLoc3D,
  title={CenterLoc3D: Monocular 3D Vehicle Localization Network for Roadside Surveillance Cameras},
  author={Tang, Xinyao and Song, Huansheng and Wang, Wei and Zhao, Chunhui},
  journal={arXiv preprint arXiv:********},
  year={2022}
}
```