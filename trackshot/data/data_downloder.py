import kagglehub

# Download latest version
path = kagglehub.dataset_download(
    "michaelmortenson/football-soccer-ball-detection-dfl",
    force_download=True,
    output_dir="/Users/Shared/git/codedesign/Trackshot-CodeDesign/data/raw",
)

print("Path to dataset files:", path)
