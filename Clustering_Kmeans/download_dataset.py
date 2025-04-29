from kaggle.api.kaggle_api_extended import KaggleApi


def download_dataset():
    api = KaggleApi()
    api.authenticate()

    dataset_name = 'vjchoudhary7/customer-segmentation-tutorial-in-python'

    api.dataset_download_files(dataset_name, path='.', unzip=True)

    print("✅ Dataset downloaded and extracted successfully!")


if __name__ == '__main__':
    download_dataset()
