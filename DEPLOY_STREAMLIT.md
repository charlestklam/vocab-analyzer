# Deploying to Streamlit Community Cloud

Follow these steps to deploy the Vocabulary Analyzer app to the web for free using Streamlit Community Cloud.

## Prerequisites

1.  **A GitHub Account:** You need a GitHub account to host your code.
2.  **A Streamlit Account:** Sign up for a free Streamlit Community Cloud account at [share.streamlit.io](https://share.streamlit.io/) (you can link your GitHub account).

## Step 1: Push Your Code to GitHub

Your local repository is already initialized and your initial code is committed. Now, you need to push it to a new remote repository on GitHub:

1.  Go to [GitHub](https://github.com/) and create a **New repository**.
2.  Name it (e.g., `vocab-analyzer`).
3.  Do **not** initialize it with a README, .gitignore, or license (leave those unchecked).
4.  Copy the URL of your new repository.
5.  Open your terminal, navigate to the `/home/charles/Documents/Chile_Writing_Corpus/vocab_app` directory, and run the following commands (replace `<YOUR_REPO_URL>` with the URL you just copied):

```bash
git remote add origin <YOUR_REPO_URL>
git branch -M main
git push -u origin main
```

## Step 2: Deploy on Streamlit Community Cloud

Once your code is pushed to GitHub, deploying to Streamlit is a few clicks away:

1.  Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
2.  Click the **"New app"** button.
3.  If prompted to "Deploy an app", select **"Deploy from a GitHub repo"**.
4.  Fill in the deployment details:
    *   **Repository:** Select your newly created repository (e.g., `yourusername/vocab-analyzer`).
    *   **Branch:** Select `main`.
    *   **Main file path:** Enter `app.py`.
    *   **App URL (optional):** You can customize the URL of your app or leave it blank for a random one.
5.  Click **"Deploy!"**.

## Step 3: Wait for Building

Streamlit will now clone your repository, read the `requirements.txt` file, install dependencies (like `spacy`, `wordfreq`, `altair`), and launch your app. The initial deployment might take a couple of minutes to process the spaCy model download included in `analyzer.py`.

Once the deployment completes, your app will be live and accessible via the provided URL!

## Note on Pre-calculated Data

The application includes pre-calculated data (`batch_50_results.json` and `openalex_5525_results.json`) inside the `data/` folder. Ensure these files are correctly pushed to GitHub so the preset functionality works immediately on the deployed app without needing to re-process locally.
