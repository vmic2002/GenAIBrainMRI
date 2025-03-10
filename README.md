# Fine tuning Stable diffusion model for Brain MRIs
This repository contains code and workflows to generate synthetic brain MRI images and use them to improve deep neural network (DNN) training for brain tumor classification. The goal is to augment limited real-world brain MRI datasets, enabling more reliable training of DNNs by increasing data diversity and volume. Additionally, we aim to evaluate how much data is needed for effective classification: 248 images, 800 images, or the full training set of 2870 images. The latter two (800 and 2870) required an A100 GPU on Google Colab due to their computational demands.

## Why Generate Synthetic Brain MRIs?

Real brain MRI datasets are often small due to privacy, cost, and acquisition challenges. Generating synthetic MRIs helps overcome this limitation by providing additional training data, allowing deep neural networks to learn more robust features and improve generalization for tasks like tumor classification.

## Approach

1. **Stable Diffusion Fine-Tuning**:
   - We fine-tune the pre-trained **Stable Diffusion v1-5** model (`runwayml/stable-diffusion-v1-5`) from Hugging Face on a brain MRI dataset sourced from Kaggle ([Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)).
   - Three models are fine-tuned with varying dataset sizes: 248, 800, and 2870 images (full training set), producing synthetic MRIs at 3× the real data volume (744, 2400, 8610 images, respectively). The 800 and 2870 models leverage an A100 GPU on Colab for efficient training.

2. **ResNet Fine-Tuning**:
   - A ResNet model is fine-tuned on the combined dataset (real + synthetic MRIs) to classify brain tumors into four categories: glioma, meningioma, pituitary, and no tumor.
   - Performance is evaluated in two settings:
     - **Real Data Only**: Baseline performance on the original Kaggle dataset.
     - **Real + Synthetic Data**: Assesses whether synthetic data improves classification accuracy, robustness, or generalization.
   - We also compare performance across dataset sizes (248, 800, 2870) to determine the minimum data required for effective classification.

## Evaluation

- We evaluate ResNet performance using standard metrics (e.g., accuracy, precision, recall) on a held-out test set of real MRIs in both settings.
- The comparison between real-only and real+synthetic results, across the three data sizes, will reveal if synthetic data enhances model performance and how much data is optimal.

## Notebooks

- **`fine_tune_model_gen_ai_medical_imaging.ipynb`**:
  - Downloads the Kaggle brain MRI dataset and fine-tunes the Stable Diffusion model using a Hugging Face script (with slight modifications for our project). Example command:
    ```bash
    # Run training optimized for A100 with batch 32
    # FOR HIGH QUALITY IMAGES:
    #  20 epochs (num train epochs is overwritten)
    # 2870 ÷ 32 ≈ 89.6875 (rounds to 90 steps per epoch), 90steps/epoch
    # Total steps ÷ Steps per epoch = 1800 ÷ 90 ≈ 20 epochs.
    !python TESTING_train_text_to_image.py \
       --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
       --train_data_dir="." \
       --dataset_name="csv" \
       --dataset_config_name="fine_tuning_metadata_A100_2870.csv" \
       --image_column="image_path" \
       --caption_column="prompt" \
       --resolution=256 \
       --train_batch_size=32 \
       --num_train_epochs=1 \
       --learning_rate=1e-5 \
       --max_train_steps=1800 \
       --output_dir="fine_tuned_model_A100_2870" \
       --checkpointing_steps=900 \
       --mixed_precision="fp16"
    ```
- **`gen_images_fine_tune_model.ipynb`**:
  - Takes pre-trained fine-tuned models and generates synthetic brain MRIs, saving them to directories like `post_fine_tuning_images_{GPU}_{numImagesTrainedOn}/`.
- **`gen_images_fine_tune_model_COLLAB.ipynb`**:
  - Uses CSVs containing the real data (248, 800, 2870 images) to fine-tune three Stable Diffusion models on an A100 GPU in Colab. Then generates three sets of synthetic data (744, 2400, 8610 images), each produced by a corresponding fine-tuned model. It is the same as the previous ipynb file except that it does not expect the fine tuned models to exist yet.

## Results

Results are not yet available but will be added soon—stay tuned!
