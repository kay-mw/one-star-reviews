import os
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


def download_files(url: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    existing_files = os.listdir(output_dir)

    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for link in soup.find_all("a"):
        file_url = link.get("href")
        if file_url:
            full_url = urljoin(url, file_url)

            if "jsonl.gz" not in file_url:
                continue
            elif file_url in existing_files:
                continue

            file_name = os.path.join(output_dir, os.path.basename(file_url))
            print(f"Downloading {full_url} to {file_name}")

            with requests.get(full_url, stream=True) as file_response:
                file_response.raise_for_status()
                with open(file_name, "wb") as file:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        file.write(chunk)

    print("Download complete!")


download_files(
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/",
    "./product_data",
)

download_files(
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/",
    "./review_data",
)
