# Fire-Detection

**Problem framing**: Image Classification

No. of classes: Two (Binary classification)

No. of labels per image: One/Single

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



Credits:
- Practical Machine Learning for Computer Vision, by Valliappa Lakshmanan, Martin Görner, and Ryan Gillard. Copyright
2021 Valliappa Lakshmanan, Martin Görner, and Ryan Gillard, 978-1-098-10236-4.