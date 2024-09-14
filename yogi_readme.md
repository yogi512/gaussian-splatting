
# Custom Dataset Training and Ego-Centric View Rendering


## Steps

### 1. Preprocess the Data

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
python yogi_train.py -s 'data/custom/work' -m 'output/work_1' --eval
```

### 5. Render Ego-Centric Views

To render the images in an ego-centric viewpoint, use the following command:

```bash
python render.py -m 'output/<model_name>'
```

Example:

```bash
python render.py -m 'output/work_1'
```

### 6. Yogi Render with Camera Parameters

For generating images using specific camera parameters for necessary viewpoints, add the necessary viewpoints in the list  'my_views' and  run:

```bash
python yogi_render.py -m 'output/<model_name>'
```

Example:

```bash
python yogi_render.py -m 'output/work_1'
```

### 7. Compute Metrics

After training and rendering, you can compute the evaluation metrics for your model:

```bash
python metrics.py -m 'output/<model_name>'
```

Example:

```bash
python compute_metrics.py -m 'output/work_1'
```

---

## Notes

- Ensure the dataset is correctly formatted with images inside the `input` folder.
- Replace `<dataset_name>` and `<model_name>` with the appropriate dataset and model names during execution.

Happy Training, Rendering, and Evaluating!
```

This guide now includes each step in the correct order, with corresponding commands for preprocessing, folder structure, conversion, training, rendering, and computing metrics.