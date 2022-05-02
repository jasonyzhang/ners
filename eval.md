# NeRS Evaluation

## Evaluation Data

Download the evaluation data from [here](https://drive.google.com/file/d/1Nle6efI8HAHIXgvqhBhEb7ZHRn2dg5Mm/view?usp=sharing):

```
gdown https://drive.google.com/uc?id=1Nle6efI8HAHIXgvqhBhEb7ZHRn2dg5Mm
unzip -n ners_evaluation.zip -d data
```

The metadata for the evaluation sequences contains an extra set of camera parameters
(`camera_pretrained`). In total, we provide 3 types of cameras:
* `camera_initial`: the off-the-shelf camera using PoseFromShape
* `camera_pretrained`: the camera parameters after optimizing `camera_initial` to
    minimize silhouette reprojection error using the template car mesh (i.e. the
    cameras after Stage 1 in the paper). This camera serves as reasonably well-aligned
    but inexact camera initialization.
* `camera_optimized`: the camera parameters after optimizing NeRS + some manual tuning.
    These cameras should be **better** than the ones in the main dataset. I manually
    fixed some of the rear viewpoints, which was a common failure mode for PoseFromShape.

We consider two evaluation settings:

1. *Novel view synthesis with fixed cameras (`camera_optimized`)* treats each of the
    N - 1 input cameras as fixed. At inference time, given the Nth target camera, we
    render the Nth target image and compare this with the ground truth image.
2. *In-the-wild novel view sysnthesis (`cameras_pretrained`)* treats each of the N - 1
    input cameras as a noisy initialization. During training, the cameras are also
    optimized. At inference time, given a target view and possibly given a noisy target
    camera, we search for the camera that best aligns with the target view.

The evaluation data contains the images corresponding to the NeRS renderings from both of
these evaluation settings. These can be evaluated using:
```
$  python -m eval.eval --pred-name ners_fixed
Name            MSE   PSNR   SSIM  LPIPS
ners_fixed   0.0254   16.5  0.720  0.172
$  python -m eval.eval --pred-name ners_trained
Name            MSE   PSNR   SSIM  LPIPS
ners_trained 0.0364   15.0  0.673  0.206
```

You can also retrain all models on the evaluation data. For example, to retrain the
fixed camera evaluation:
```
python -m eval.train_evaluation_model --data_dir data/evaluation --fix-cameras \
    --instance-id 7251103879 --camera-index 0 --camera-type camera_optimized
```

and the trainable camera evaluation:
```
python -m eval.train_evaluation_model --data_dir data/evaluation \
    --instance-id 7251103879 --camera-index 0 --camera-type camera_pretrained
```

Note that one model must be trained for each camera index for each instance. This process can be automated by running:

```
python -m eval.eval_driver --data-dir data/evaluation --evaluation-mode fixed

python -m eval.eval_driver --data-dir data/evaluation --evaluation-mode in-the-wild
```