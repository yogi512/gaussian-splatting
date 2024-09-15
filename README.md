# 3D Gaussian Splatting for Real-Time Radiance Field Rendering

The codebase has 4 main components:
- A PyTorch-based optimizer to produce a 3D Gaussian model from SfM inputs
- A network viewer that allows to connect to and visualize the optimization process
- An OpenGL-based real-time viewer to render trained models in real-time.
- A script to help you turn your own images into optimization-ready SfM data sets

The components have different requirements w.r.t. both hardware and software. They have been tested on Windows 10 and Ubuntu Linux 22.04. Instructions for setting up and running each of them are found in the sections below.

### Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- 24 GB VRAM (to train to paper evaluation quality)
- Please see FAQ for smaller VRAM configurations

### Software Requirements
- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions (we used Visual Studio 2019 for Windows)
- CUDA SDK 11 for PyTorch extensions, install *after* Visual Studio (we used 11.8, **known issues with 11.6**)
- C++ Compiler and CUDA SDK must be compatible



# Custom Dataset Training and Ego-Centric View Rendering


## Steps

### 1. Preprocess the Data using our custom function

Run the following command to preprocess the images from the `input` folder:

```bash
python preprocess_video.py --datadir 'data/custom/<dataset_name>'
```

Example:

```bash
python preprocess_video.py --datadir 'data/custom/work'
```

### 2. Folder Structure

Ensure your custom dataset follows this folder structure, with the images inside the `input` directory:

```
custom_dataset
|-----input
        |---- image1.png 
        |---- image2.png 
        |---- ...
```

Example:

```
work
|-----input
        |---- image1.png 
        |---- image2.png 
```

### 3. Convert the Data

After preprocessing, convert the data with the following command:

```bash
python convert.py -s 'data/custom/<dataset_name>'
```

Example:

```bash
python convert.py -s 'data/custom/work'
```

COLMAP loaders expect the following dataset structure in the source path location:
```
<location>
|---images
|   |---<image 0>
|   |---<image 1>
|   |---...
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```

### 4. Train the Model

Train the model using the `yogi_train.py` script:

```bash
python yogi_train.py -s 'data/custom/<dataset_name>' -m 'output/<model_name>' --eval
```

Example:

```bash
python yogi_train.py -s 'data/custom/work' -m 'output/work' --eval
```

### 5. Render Ego-Centric Views

To render the images in an ego-centric viewpoint, use the following command:

```bash
python render.py -m 'output/<model_name>'
```

Example:

```bash
python render.py -m 'output/work'
```

### 6. Render with Camera Parameters with our rendering algorithm

For generating images using specific camera parameters for necessary viewpoints, add the necessary viewpoints in the list  'my_views' and  run:

```bash
python yogi_render.py -m 'output/<model_name>'
```

Example:

```bash
python yogi_render.py -m 'output/work'
```

### 7. Compute Metrics

After training and rendering, you can compute the evaluation metrics for your model:

```bash
python metrics.py -m 'output/<model_name>'
```

Example:

```bash
python metrics.py -m 'output/work'
```

---

## Notes

- Ensure the dataset is correctly formatted with images inside the `input` folder.
- Replace `<dataset_name>` and `<model_name>` with the appropriate dataset and model names during execution.

