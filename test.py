from huggingface_hub import upload_file

if __name__ == "__main__":
    upload_file(
        path_or_fileobj="Trained_Models/knn.pkl",  # your local file
        path_in_repo="knn.pkl",                    # desired name on the Hub
        repo_id="shreyasshukla/emotion-cnn-single",# your HF repo
        repo_type="model",                         # model repo
        token=True                                 # uses your cached token
    )
    print("✅ knn.pkl uploaded!")
