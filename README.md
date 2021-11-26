# Fire-Detection

**Problem framing**: Image Classification

No. of classes: Two (Binary classification)

No. of labels per image: One/Single

Performance-metric chosen: Accuracy

Reasons:
- Though the class distribution in dataset is not perfectly balanced, it's not heavily skewed either.
- False Negatives (e.g. actual fire cases which went undetected) are as much to avoid as False Positives (e.g. false alarms)

Model **assumption**(s): It is **NOT** a smoke-detector. There has to be a flame (preferably erupting) in the image, for the model to classify it as "Fire".

_Curate-Data_

Step 1: Get **images' urls** from Google search e.g. using keywords like 'building on fire', 'bushfires' etc.

_Parallelized Data-Ingestion_

Step 2: **Load images to GCS** through Cloud **DataFlow** pipeline

Step 3: Train a **baseline AutoML model** using Python SDK for **Vertex AI**.
[Caution: This step cost me ~ $30.]

Step 4: Convert image files in GCS, to **TFRecords** format, using Cloud **Dataflow** pipeline.
[Note: This step took almost 14 vCPU hours on GCP, in my case.]

_Model-Training_

Step 5: **Hyper-parameter tuning** for custom NN model, on **Vertex AI**
[Note: Make sure there are enough GPU quotas for the GCP Project-ID.]

Step 6: **Distributed training** (for the chosen hyper-parameter combo) across multiple GPUs
[Note: Access to > 1 GPU is needed for MirroredStrategy in this step. For <= 1 GPU, just get rid of the strategy part of the code.]

_Explainable-AI_

Step 7: Add **explainability** (**instance-level** feature importances) to the model predictions by
- using **Integrated-Gradients** (**pixel-based**) and Explainable Representations through AI (**XRAI**; **region-based**) techniques on AI-Platform
[Note: Choose your [runtime-version](https://cloud.google.com/ai-platform/training/docs/runtime-version-list) carefully.]
- using **Explainable-AI-SDK**

_Inferencing_

Step 8: Setting model signature(s) to infer from:
- image **files**
- image **bytes**

In both cases, we deploy the model by creating a **REST endpoint on Vertex-AI**.

**Why do we need the bytes option at this stage?**: What if we don't have the luxury of first uploading images to GCS, before sending them to our trained/exported model? In such cases, we can keep files locally stored, just send the extracted bytes over to the model in the form of json request, and get a json response in return. 

_Edge-deployment_

Step 9: Deployment on Android smartphone
- convert TF2.x model to tflite version
- post-training quantization/optimization
- deploy on **Android** device using PalletML

https://user-images.githubusercontent.com/35262566/142001318-f42dd5d8-2363-46b6-9c62-e510c7c9b5ad.mp4

Credits:
- [Practical Machine Learning for Computer Vision](https://www.oreilly.com/library/view/practical-machine-learning/9781098102357/), by Valliappa Lakshmanan, Martin Görner, and Ryan Gillard. Copyright 2021 Valliappa Lakshmanan, Martin Görner, and Ryan Gillard, 978-1-098-10236-4.
- [Maven Wave](https://www.mavenwave.com/) (my employer) for giving me ample opportunities to not just learn new concepts, but also to get my hands dirty in cloud sandboxes
- content by [quicktensorflow](https://courses.quicktensorflow.com/courses/)
