# Fire-Detection

**Problem framing**: Image Classification

No. of classes: Two (Binary classification)

No. of labels per image: One/Single

Performance-metric chosen: Accuracy

Reasons:
- Though the class distribution in dataset is not perfectly balanced, it's not heavily skewed either.
- False Negatives (e.g. actual fire cases which went undetected) are as much to avoid as False Positives (e.g. false alarms)

Model **assumption**(s): It is **NOT** a smoke-detector. There has to be a flame (preferably erupting) in the image, for the model to classify it as "Fire".

_Curate Data_

Step 1: Get **images' urls** from Google search e.g. using keywords like 'building on fire', 'bushfires' etc.

_Data Ingestion_

Step 2: **Load images to GCS** through Cloud **DataFlow** pipeline

Step 3: Train a **baseline AutoML model** using Python SDK for **Vertex AI**.
[Caution: This step cost me ~ $30.]

Step 4: Convert image files in GCS, to **TFRecords** format, using Cloud **Dataflow** pipeline.
[Note: This step took almost 14 vCPU hours on GCP, in my case.]

_Model training_

Step 5: **Hyper-parameter tuning** for custom NN model, on **Vertex AI**
[Note: Make sure there are enough GPU quotas for the GCP Project-ID.]

Step 6: **Distributed training** (for the chosen hyper-parameter combo) across multiple GPUs
[Note: Access to > 1 GPU is needed for MirroredStrategy in this step. For <= 1 GPU, just get rid of the strategy part of the code.]

_Explainable-AI_

Step 7: Add **explainability** (**instance-level** feature importances) to the model predictions by
- using **Integrated-Gradients** (**pixel-based**) and Explainable Representations through AI (**XRAI**; **region-based**) techniques on AI-Platform
[Note: Choose you [runtime-version](https://cloud.google.com/ai-platform/training/docs/runtime-version-list) carefully.]
- using **Explainable-AI-SDK**

Step 8: Deploy model on Android smartphone
- convert TF2.x model to tflite version
- post-training quantization/optimization
- deploy on Android device using PalletML
https://youtu.be/VHicKAhkTwI


Credits:
- [Practical Machine Learning for Computer Vision](https://www.oreilly.com/library/view/practical-machine-learning/9781098102357/), by Valliappa Lakshmanan, Martin Görner, and Ryan Gillard. Copyright
2021 Valliappa Lakshmanan, Martin Görner, and Ryan Gillard, 978-1-098-10236-4.
- [Maven Wave](https://www.mavenwave.com/) (my employer) for giving me ample opportunities to not just learn new concepts, but also to get my hands dirty in cloud sandboxes
- quicktensorflow course