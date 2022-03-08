# Explainability-Dermatology

Black box machine learning models that cannot be understood by people, such as deep neural networks and large ensembles, are achieving impressive accuracy on various tasks. However, as machine learning is increasingly used to inform high stakes decisions, explainability and interpretability of the models is becoming essential. The lack of understanding on how neural networks make predictions enables unpredictable/biased models, causing real harm to society and a loss of trust in AI-assisted systems. There are many ways to explain: data vs. model, directly interpretable vs. post hoc explanation, local vs. global, static vs. interactive; the appropriate choice depends on the persona of the consumer of the explanation.

For our particular project, our aim is twofold:
* Increase trustworthiness by delivering better explanations to both physicians and patients.
* Study biases to build more accurate and robust models.

This repository contains code for testing different local and global methods using diverse pytorch libraries and [ISIC 2020 dataset](https://www.kaggle.com/nroman/melanoma-external-malignant-256).

## Methods

### Global vs Local Explanations

Global explanations are for entire models whereas local explanations are for single sample points.

Global directly interpretable models are important for personas that need to understand the entire decision making process and ensure its safety, reliability, or compliance. Such personas include regulators and data scientists responsible for the deployment of systems. Global post hoc explanations are useful for decision maker personas that are being supported by the machine learning model. Physicians, judges, and loan officers develop an overall understanding of how the model works, but there is necessarily a gap between the black box model and the explanation. Therefore, a global post hoc explanation may hide some safety issues but its antecedent black box model may have favorable accuracy. Local models are the most useful for affected user personas such as patients, defendants, and applicants who need to understand the decision on a single sample (theirs).

### LOCAL XAI

#### Attribution Methods 

Attribution methods link a deep neural network's prediction to the input features that most influence that prediction. If the model makes a misprediction, we might want to know which features contributed to the misclassification. We can visualize the attribution scores for image models as a grayscale image with the same dimensions as the original image with brightness corresponding to the importance of the pixel.

* Primary Attribution: Evaluates contribution of each input feature to the output of a model.
* Layer Attribution: Evaluates contribution of each neuron in a given layer to the output of the model.
* Neuron Attribution: Evaluates contribution of each input feature on the activation of a particular hidden neuron.

**Libraries used**:

**[PAIRML Saliency Methods](https://github.com/pair-code/saliency)**

```pip install saliency```

This repository contains code for the following saliency techniques:

*   Guided Integrated Gradients* ([paper](https://arxiv.org/abs/2106.09788), [poster](https://github.com/PAIR-code/saliency/blob/master/docs/CVPR_Guided_IG_Poster.pdf))
*   XRAI* ([paper](https://arxiv.org/abs/1906.02825), [poster](https://github.com/PAIR-code/saliency/blob/master/docs/ICCV_XRAI_Poster.pdf))
*   SmoothGrad* ([paper](https://arxiv.org/abs/1706.03825))
*   Vanilla Gradients
    ([paper](https://scholar.google.com/scholar?q=Visualizing+higher-layer+features+of+a+deep+network&btnG=&hl=en&as_sdt=0%2C22),
    [paper](https://arxiv.org/abs/1312.6034))
*   Guided Backpropogation ([paper](https://arxiv.org/abs/1412.6806))
*   Integrated Gradients ([paper](https://arxiv.org/abs/1703.01365))
*   Occlusion
*   Grad-CAM ([paper](https://arxiv.org/abs/1610.02391))
*   Blur IG ([paper](https://arxiv.org/abs/2004.03383))

**[Pytorch GradCam - Class Activation Map methods implemented in Pytorch](https://github.com/jacobgil/pytorch-grad-cam)** 

```pip install grad-cam```

*  GradCAM ([paper](https://arxiv.org/abs/1610.02391)) - Weight the 2D activations by the average gradient
*  GradCAM++ ([paper](https://arxiv.org/abs/1710.11063)) - Like GradCAM but uses second order gradients
*  XGradCAM ([paper](https://arxiv.org/abs/2008.02312)) - Like GradCAM but scale the gradients by the normalized activations
*  AblationCAM ([paper](https://ieeexplore.ieee.org/abstract/document/9093360/)) - Zero out activations and measure how the output drops 
*  ScoreCAM ([paper](https://arxiv.org/abs/1910.01279)) -  Perbutate the image by the scaled activations and measure how the output drops
*  EigenCAM ([paper](https://arxiv.org/abs/2008.00299)) - Takes the first principle component of the 2D Activations (no class discrimination, but seems to give great results)
*  EigenGradCAM -  Like EigenCAM but with class discrimination: First principle component of Activations*Grad. Looks like GradCAM, but cleaner
*  LayerCAM ([paper](http://mftp.mmcheng.net/Papers/21TIP_LayerCAM.pdf)) - Spatially weight the activations by positive gradients. Works better especially in lower layers 
*  FullGrad ([paper](https://arxiv.org/abs/1905.00780)) - Computes the gradients of the biases from all over the network, and then sums them


**[Captum](https://captum.ai/)**  

```pip install captum```

* Multimodality: Supports interpretability of models across modalities including vision, text, and more.

* Supports most types of PyTorch models and can be used with minimal modification to the original neural network.

* Easy to use and extensible: Open source, generic library for interpretability research. Easily implement and benchmark new algorithms.

* [Captum Insights](https://captum.ai/docs/captum_insights) · Captum interpretability visualization widget built on top of Captum


#### CODE

 [`local_xai_attr.py`](./local_xai_attr.py): Uses Captum and Pytorch GradCam libraries to implement different attribution methods.
 
 [`PAIR_saliency.ipynb`](./PAIR_saliency.ipynb): Uses PAIRML saliency library to test the following methods:
 * Vanilla Gradient
 * SmoothGrad 
 * XRAI
 * Guided Integrated Gradients
 
**XRAI Examples**:
Local methods are used to explore individual instances that are important for some reason, like edge cases. The following images come from a different dataset the model has not seen before during training. In this particular case, the XRAI method was used, as it was considered clearer and easier to interpret.

The last column shows the most salient n% (15% or 5%) of the image. It appears that asymmetry plays an important role in discerning between melanomas and non-melanomas. Rulers seem to be somewhat important to the model, hence inserting a bias.

![ex1](https://github.com/sandracl72/Explainability-Dermatology/blob/master/docs/centroid_3668.png)
![ex2](https://github.com/sandracl72/Explainability-Dermatology/blob/master/docs/xrai_SAM_2.png)
![ex3](https://github.com/sandracl72/Explainability-Dermatology/blob/master/docs/xrai_SAM_5.png)

The following two images showcase the strong bias of the model towards black frames. However, white frames, which are very rare in the dataset, are not a source of bias.

![ex4](https://github.com/sandracl72/Explainability-Dermatology/blob/master/docs/edge_13403.png)
![ex5](https://github.com/sandracl72/Explainability-Dermatology/blob/master/docs/xrai_SAM_6.png)

### GLOBAL XAI

#### TCAV: Testing with Concept Activation Vectors

 [`TCAV.ipynb`](./TCAV.ipynb) - The notebook is self-contained.

#### Projection of XRAI 

 [`XRAI_global_projection.py`](./XRAI_global_projection.py)
 
 1. Apply XRAI to 6k images from ISIC and save the masked images
 2. Apply EffNetB2 as a feature extractor → 500D vectors
 3. Project with UMAP 

We observe two clear clusters corresponding to frames and rulers, as well as one zone with thick hair.
![visualization](https://github.com/sandracl72/Explainability-Dermatology/blob/master/docs/XAI_CNNembs_from_masked_imgs.PNG)

 


