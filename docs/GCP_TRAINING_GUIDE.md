# GCP Training Pipeline Guide

This guide provides the steps to configure your Google Cloud environment, build the required Docker image, and run the training pipeline for the brain segmentation model.

## 1. Prerequisites

- You have a Google Cloud Platform (GCP) account.
- You have the `gcloud` CLI installed and authenticated.
- You have Docker installed on your local machine.
- You have a GCP Service Account with the "Vertex AI User" and "Storage Object Admin" roles.

### Creating a Service Account and Key

If you do not have a service account key, follow these steps:

1.  **Navigate to the Service Accounts page** in the GCP Console.
2.  Select your project.
3.  Click **"CREATE SERVICE ACCOUNT"**.
4.  Give the service account a name (e.g., `ai-platform-executor`) and click **"CREATE AND CONTINUE"**.
5.  In the "Grant this service account access to project" screen, grant the following roles. **Use the search bar to find them if they are not immediately visible**:
    *   **Vertex AI User**: This role allows the service account to create and manage AI Platform (Vertex AI) jobs.
    *   **Storage Object Admin**: This role allows the service account to read from and write to your GCS bucket, which is necessary for accessing data and saving model artifacts.
6.  Click **"CONTINUE"**, then **"DONE"**.
7.  Find the service account you just created in the list, click the three-dot menu on the right, and select **"Manage keys"**.
8.  Click **"ADD KEY"** -> **"Create new key"**.
9.  Select **"JSON"** as the key type and click **"CREATE"**.
10. A JSON file will be downloaded. **This is your service account key.**
11. Place this file at `data/credentials/gcp_service_account.json`, replacing the placeholder content with the content of the file you downloaded.

## 2. Installing and Authenticating the `gcloud` CLI

The `gcloud` command-line tool is essential for interacting with your GCP account.

### Installation

1.  **Follow the official Google Cloud documentation** for the most up-to-date instructions for your operating system:
    - [Google Cloud SDK Installation Guide](https://cloud.google.com/sdk/docs/install)

2.  A common method for macOS and Linux is to run the interactive installer:
    ```bash
    curl https://sdk.cloud.google.com | bash
    ```
    After the script finishes, restart your shell or source your `.bashrc`/`.zshrc` file:
    ```bash
    exec -l $SHELL
    ```
    > **Note on Interactive Prompts:** The installer will ask you if you want to update your PATH. You should answer **`Y` (yes)**. It will then ask for the path to your rc file (e.g., `/Users/camdouglas/.zshrc`). You should **press Enter** to accept the default.

### Authentication

Once installed, you need to log in to your Google account.

1.  Run the following command in your terminal:
    ```bash
    gcloud auth login
    ```
2.  This command will open a new browser window or provide a URL for you to log in to your Google account.
3.  After you approve the access request, your credentials will be stored locally, and the CLI will be authenticated.

4.  Next, set your default project:
    ```bash
    gcloud config set project [YOUR_PROJECT_ID]
    ```
    Replace `[YOUR_PROJECT_ID]` with the ID you find in the next step.

## 3. Finding Your Project ID

Your Project ID is a unique identifier for your GCP project.

1.  Run the following command in your terminal:
    ```bash
    gcloud config get-value project
    ```
2.  If a value is returned, that is your **Project ID**.
3.  If no value is returned, visit the [GCP Console Dashboard](https://console.cloud.google.com/home/dashboard) and select your project. The Project ID will be visible in the "Project info" card.

    **Example:** `my-brain-project-12345`

## 3. Creating a Google Cloud Storage (GCS) Bucket

The pipeline requires a GCS bucket to store model artifacts and TensorBoard logs.

1.  Choose a unique name for your bucket (e.g., `your-name-brain-segmentation-bucket`).
2.  Run the following command to create the bucket:
    ```bash
    gsutil mb gs://your-unique-bucket-name
    ```
    Replace `your-unique-bucket-name` with the name you chose.

    Your **GCS Bucket** path is `gs://your-unique-bucket-name`.

## 4. Building and Pushing the Docker Image

The training job runs inside a Docker container. We need to build this container and push it to Google Container Registry (GCR).

1.  **Enable GCR API:** Ensure the Container Registry API is enabled for your project. You can do this from the [GCP Console](https://console.cloud.google.com/apis/library/containerregistry.googleapis.com).

2.  **Create a `Dockerfile`:** I will create a `Dockerfile` in the root of the project for you in the next step.

3.  **Build the Image:** Once the `Dockerfile` is created, run the following command. Replace `[PROJECT_ID]` with your actual Project ID.
    ```bash
    docker build -t gcr.io/[PROJECT_ID]/brain-segmentation-trainer:latest .
    ```

4.  **Configure Docker to Authenticate with GCR:**
    ```bash
    gcloud auth configure-docker
    ```

5.  **Push the Image to GCR:**
    ```bash
    docker push gcr.io/[PROJECT_ID]/brain-segmentation-trainer:latest
    ```

    Your **Container Image URI** is `gcr.io/[PROJECT_ID]/brain-segmentation-trainer:latest`.

## 5. Running the Training Job

Once you have your `PROJECT_ID`, `GCS_BUCKET`, and `CONTAINER_IMAGE_URI`, you can run the training job.

1.  Open the `brain/gcp_training_manager.py` file.
2.  Uncomment and fill in the configuration variables in the `if __name__ == '__main__':` block.
3.  Run the script from your terminal:
    ```bash
    python brain/gcp_training_manager.py
    ```

The script will then submit the job to AI Platform, and you will see monitoring output in your terminal, including the URL for your TensorBoard instance.