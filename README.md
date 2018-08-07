# edge2view-demo

This is a pix2pix demo that learns from edge and translates this into view. A interactive application is also provided that translates edge to view.

## Getting Started

#### 1. Prepare Environment

```
# Clone this repo
git clone git@github.com:GordonRen/edge2view.git

# Create the conda environment from file
conda env create -f environment.yml
```
#### 2. Configure Holistically-Nested Edge Detection

```
https://github.com/s9xie/hed
```
#### 3. Generate Original Data

```
python generate_train_data.py --file Desert.mp4
```

Input:

- `file` is the name of the video file from which you want to create the data set.

Output:

- One folder `original` will be created.

#### 4. Generate Edge Data

- generate edge data by following batch_hed.py and put the edge data into `hed_edge`.

![example](hed.png)

If you want to download my dataset, here is also the [video file](https://dl.dropboxusercontent.com/s/yaxh66n31v2unfc/Desert.mp4) that I used and the generated [training dataset](https://dl.dropboxusercontent.com/s/c7hg8efxey0uxxf/dataset_view.zip) (708 images already split into training and validation).

#### 5. Train Model
```
# Clone the repo from Christopher Hesse's pix2pix TensorFlow implementation
git clone https://github.com/affinelayer/pix2pix-tensorflow.git

# Move the original and hed_edge folder into the pix2pix-tensorflow folder
mv edge2view/hed_edge edge2view/original pix2pix-tensorflow/photos_view

# Go into the pix2pix-tensorflow folder
cd pix2pix-tensorflow/

# Reset to april version
git reset --hard d6f8e4ce00a1fd7a96a72ed17366bfcb207882c7

# Resize original images
python tools/process.py \
  --input_dir photos_view/original \
  --operation resize \
  --output_dir photos_view/original_resized
  
# Resize hed_edge images
python tools/process.py \
  --input_dir photos_view/hed_edge \
  --operation resize \
  --output_dir photos_view/hed_edge_resized
  
# Combine both resized original and hed_edge images
python tools/process.py \
  --input_dir photos_view/hed_edge_resized \
  --b_dir photos_view/original_resized \
  --operation combine \
  --output_dir photos_view/combined
  
# Split into train/val set
python tools/split.py \
  --dir photos_view/combined
  
# Train the model on the data
python pix2pix.py \
  --mode train \
  --output_dir edge2view-model \
  --max_epochs 1000 \
  --input_dir photos_view/combined/train \
  --which_direction AtoB
```

For more information around training, have a look at Christopher Hesse's [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow) implementation.

#### 6. Export Model

1. First, we need to reduce the trained model so that we can use an image tensor as input: 
    ```
    python reduce_model.py --model-input edge2view-model --model-output edge2view-reduced-model
    ```
    
    Input:
    
    - `model-input` is the model folder to be imported.
    - `model-output` is the model (reduced) folder to be exported.
    
    Output:
    
    - It returns a reduced model with less weights file size than the original model.

2. Second, we freeze the reduced model to a single file.
    ```
    python freeze_model.py --model-folder edge2view-reduced-model
    ```

    Input:
    
    - `model-folder` is the model folder of the reduced model.
    
    Output:
    
    - It returns a frozen model file `frozen_model.pb` in the model folder.
    
I have uploaded a pre-trained frozen model [here](https://dl.dropboxusercontent.com/s/b94rggjff91gnzx/edge2view_model_epoch_1000.zip). This model is trained on 708 images with epoch 1000.
    
#### 7. Run Demo

```
python edge2view.py --tf-model edge2view-reduced-model/frozen_model.pb
```

Input:

- `tf-model` is the frozen model file.

Example:

![example](example.gif)

## Requirements
- [TensorFlow 1.0.0](https://www.tensorflow.org/)
- [Holistically-Nested Edge Detection](https://github.com/s9xie/hed)
- [MATLAB](https://www.mathworks.com/)

## Acknowledgments
Kudos to [Christopher Hesse](https://github.com/christopherhesse) for his amazing pix2pix TensorFlow implementation and [Gene Kogan](http://genekogan.com/) for his inspirational workshop. \
Inspired by [Dat Tran](https://github.com/datitran/face2face-demo).

## License
See [LICENSE](LICENSE) for details.
