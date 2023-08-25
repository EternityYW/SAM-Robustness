# An Empirical Study on the Robustness of the Segment Anything Model (SAM)

This repository contains open-source SAM implementation code, datasets (sample images), code to generate perturbed images, sample experimental results, and code for results analysis.

## Datasets
Nine datasets are chosen that span across distinct imaging conditions and pose various segmentation challenges. The Remote Sensing and Geographical category comprises datasets with aerial and satellite imagery, which present hurdles such as diverse resolutions, intricate patterns, and large-scale structures that need to be processed. Medical Imaging datasets, featuring ultrasound and X-ray modalities, often grapple with issues like noise, artifacts, and fluctuating contrasts. The Environment and Natural Phenomena category encompass datasets with a multitude of dynamic elements, such as fish schools with overlapping shapes and fire spread with irregular boundaries, requiring the model to adapt to varying shapes and textures. Finally, the Structural and Human Motion category incorporates datasets with detailed structures like cracks and the intricate motion patterns of human dance, necessitating high precision and the ability to capture subtle nuances. The following table presents an overview of the datasets used in our experiments, all of which have binary mask ground truth annotations.

                                                                            | **Category**                                         | **Dataset**       | **Modality**  | **Num. Images** |
                                                                            |------------------------------------------------------|-------------------|---------------|-----------------|
                                                                            | Remote Sensing and Geographical                      | Forest Aerial     | Aerial        | 5,108           |
                                                                            |                                                      | Water Bodies      | Satellite     | 2,841           |
                                                                            |                                                      | Road Extraction   | Satellite     | 8,570           |
                                                                            | Medical Imaging                                      | Breast Ultrasound | Ultrasound    | 780             |
                                                                            |                                                      | Chest X-Ray       | X-Ray         | 18,479          |
                                                                            | Environment and Natural Phenomena                    | Fish              | RGB           | 9,000           |
                                                                            |                                                      | Fire              | RGB           | 110             |
                                                                            | Structural and Human Motion Analysis                 | Crack             | RGB           | 11,200          |
                                                                            |                                                      | TikTok Dancing    | RGB           | 100,000         |

The following figure shows representative raw images along with their corresponding ground truth masks for each dataset.

<div align="center">
    <img width="80%" alt="image" src="https://github.com/EternityYW/SAM-Robustness/blob/main/Sources/sample_images.png">
</div>

For sample images of each dataset, refer to "[./Sample_Images](./Sample_Images/)".

## Prompting Methods
We employ three major types of prompting methods for SAM: point, box, and a combination of both. These methods guide the model in its segmentation task by providing varying levels of information about the target object. For point prompting, we explore two variants: single-point and multiple-point. The choice between them depends on both the dataset and the complexity of the object being segmented. The following figure illustrates examples of the three major types of prompting, showing the raw image, ground truth mask, point, box, and the combination of point and box prompts for two representative image samples.

<div align="center">
    <img width="80%" alt="image" src="https://github.com/EternityYW/SAM-Robustness/blob/main/Sources/prompting_images.png">
</div>

## Perturbation Types
We evaluate the robustness of SAM by considering fifteen image perturbations, each chosen for their frequent occurrence in real-world imaging scenarios, categorized into six distinct groups. The following table provides an overview of different perturbations and parameters used for experiments.

                                                                            | **Category** | **Perturbation**        | **Abbreviation** | **Parameters**                       |
                                                                            |--------------|-------------------------|------------------|--------------------------------------|
                                                                            | Noise       | Gaussian Noise          | GN               | mean=0, std.=30                      |
                                                                            |              | Shot Noise              | SN               | intensity=0.1                        |
                                                                            |              | Salt & Pepper Noise     | SPN              | prob=0.04                            |
                                                                            | Blur        | Gaussian Blur           | GB               | kernel size=15                       |
                                                                            |              | Motion Blur             | MB               | kernel size=20                       |
                                                                            |              | Defocus Blur            | DB               | kernel size=25, sigma_x=sigma_y=8    |
                                                                            | OG          | Chromatic Aberration    | CA               | shift_x= shift_y=15                  |
                                                                            |              | Elastic Transform       | ET               | alpha=100, sigma=10                  |
                                                                            |              | Radial Distortion       | RD               | k1=-0.5, k2=0.05, k3=p1=p2=0         |
                                                                            | IC          | Brightness              | BRT              | factor=1.5                           |
                                                                            |              | Saturation              | SAT              | coefficient=0.5                      |
                                                                            |              | Contrast                | CON              | factor=2                             |
                                                                            | ENV         | Snow                    | SNOW             | coefficient=0.3                      |
                                                                            |              | Fog                     | FOG              | intensity=0.5                        |
                                                                            | CMP         | JPEG Compression        | COM              | quality=15                           |

The figure illustrates the perturbations:

<div align="center">
    <img width="80%" alt="image" src="https://github.com/EternityYW/SAM-Robustness/blob/main/Sources/perturbation_illustration.png">
</div>

For sample perturbation images of each dataset, refer to "[./Sample_Images](./Sample_Images/)".
For code to generate perturbations, refer to "[./Utils](./Utils/)".

## Experimental Results
The sample experimental results for each dataset (10 images for each) can be found in "[./Sample_Results](./Sample_Results/)".
The code for result analysis can be found in "[./Utils](./Utils/)".

Please refer to our full [paper](https://arxiv.org/pdf/2305.06422.pdf) for more details.

## Citation
If you find this work helpful, please consider citing as follows:  

```ruby
@article{wang2023empirical,
  title={An empirical study on the robustness of the segment anything model (sam)},
  author={Wang, Yuqing and Zhao, Yun and Petzold, Linda},
  journal={arXiv preprint arXiv:2305.06422},
  year={2023}
}
```




