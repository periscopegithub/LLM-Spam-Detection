import os
import glob
import tarfile
import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi


def get_dataset(name: str) -> pd.DataFrame:
    """Load a processed dataset based on a name"""
    return pd.read_csv(f"data/processed/{name}/data.csv").dropna()


def preprocess_enron() -> None:
    """Clean and rename the dataset and save it in data/processed"""
    raw_dir = "data/raw/enron"
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    Path("data/processed/enron").mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    # Download and extract
    # url = "https://github.com/MWiechmann/enron_spam_data/raw/master/enron_spam_data.zip"
    # with urlopen(url) as zurl:
    #     with ZipFile(BytesIO(zurl.read())) as zfile:
    #         zfile.extractall("data/raw/enron")

    # Download the dataset using Kaggle API
    dataset_url = f"bayes2003/emails-for-spam-or-ham-classification-enron-2006"
    api.dataset_download_files(dataset_url, path=raw_dir, unzip=True)

    # Load dataset
    # df = pd.read_csv("data/raw/enron/enron_spam_data.csv", encoding="ISO-8859-1")
    # Load the processed email text CSV file
    csv_file = os.path.join(raw_dir, "email_text.csv")
    df = pd.read_csv(csv_file, encoding="ISO-8859-1")

    # Preprocess
    df = df.fillna("")
    # df["text"] = df["Subject"] + df["Message"]
    # df["label"] = df["Spam/Ham"].map({"ham": 0, "spam": 1})
    df = df[["text", "label"]]

    # Remove rows where text is empty or whitespace
    df = df[df["text"].str.strip() != ""]

    # Remove rows where label is not 0 or 1
    df = df[df["label"].isin([0, 1])]

    # Drop rows where label is an empty string
    df = df[df["label"].astype(str).str.strip() != ""]

    df = df.drop_duplicates()

    # Save
    df.to_csv("data/processed/enron/data.csv", index=False)
    print(f"Enron dataset processed and saved.")


def preprocess_ling() -> None:
    """Clean and rename the dataset and save it in data/processed"""
    Path("data/raw/ling").mkdir(parents=True, exist_ok=True)
    Path("data/processed/ling").mkdir(parents=True, exist_ok=True)

    # Download and extract
    url = "https://github.com/oreilly-japan/ml-security-jp/raw/master/ch02/lingspam_public.tar.gz"
    r = urlopen(url)
    t = tarfile.open(name=None, fileobj=BytesIO(r.read()))
    t.extractall("data/raw/ling")
    t.close()

    path = r"data/raw/ling/lingspam_public/bare/*/*"
    data = []

    for fn in glob.glob(path):
        label = 1 if "spmsg" in fn else 0

        with open(fn, "r", encoding="ISO-8859-1") as file:
            text = file.read()
            data.append((text, label))

    df = pd.DataFrame(data, columns=["text", "label"])
    df = df.dropna()
    df = df.drop_duplicates()

    # Save
    df.to_csv("data/processed/ling/data.csv", index=False)
    print(f"Ling dataset processed and saved.")


def preprocess_sms() -> None:
    """Clean and rename the dataset and save it in data/processed"""
    Path("data/raw/sms").mkdir(parents=True, exist_ok=True)
    Path("data/processed/sms").mkdir(parents=True, exist_ok=True)

    # Download and extract
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    with urlopen(url) as zurl:
        with ZipFile(BytesIO(zurl.read())) as zfile:
            zfile.extractall("data/raw/sms")

    # Load dataset
    df = pd.read_csv("data/raw/sms/SMSSpamCollection", sep="\t", header=None)

    # Clean dataset
    df = df.drop_duplicates(keep="first")

    # Rename
    df = df.rename(columns={0: "label", 1: "text"})
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # Preprocessing
    df = df.dropna()
    df = df.drop_duplicates()

    # Save
    df.to_csv("data/processed/sms/data.csv", index=False)
    print(f"SMS dataset processed and saved.")


def preprocess_spamassassin() -> None:
    """Clean and rename the dataset and save it in data/processed"""
    Path("data/raw/spamassassin").mkdir(parents=True, exist_ok=True)
    Path("data/processed/spamassassin").mkdir(parents=True, exist_ok=True)

    urls = [
        "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
        "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2",
        "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2",
        "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
        "https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2",
    ]
    for url in urls:
        r = urlopen(url)
        t = tarfile.open(name=None, fileobj=BytesIO(r.read()))
        t.extractall("data/raw/spamassassin")
        t.close()

    path = r"data/raw/spamassassin/*/*"
    data = []

    for fn in glob.glob(path):
        label = 0 if "ham" in fn else 1

        with open(fn, "r", encoding="ISO-8859-1") as file:
            text = file.read()
            data.append((text, label))

    df = pd.DataFrame(data, columns=["text", "label"])
    df = df.dropna()
    df = df.drop_duplicates()

    # Save
    df.to_csv("data/processed/spamassassin/data.csv", index=False)
    print(f"SpamAssassin dataset processed and saved.")


def preprocess_trec() -> None:
    dataset_versions = ["2005", "2006", "2007"]

    for dataset_version in dataset_versions:
        """Download, clean, and rename a specific TREC dataset and save it in data/processed"""
        raw_dir = f"data/raw/trec-{dataset_version}"
        processed_dir = f"data/processed/trec-{dataset_version}"

        Path(raw_dir).mkdir(parents=True, exist_ok=True)
        Path(processed_dir).mkdir(parents=True, exist_ok=True)

        api = KaggleApi()
        api.authenticate()

        # Download the dataset using Kaggle API
        dataset_url = (
            f"bayes2003/emails-for-spam-or-ham-classification-trec-{dataset_version}"
        )
        api.dataset_download_files(dataset_url, path=raw_dir, unzip=True)

        # Load the processed email text CSV file
        csv_file = os.path.join(raw_dir, "email_text.csv")
        df = pd.read_csv(csv_file, encoding="ISO-8859-1")

        # Preprocess
        df = df.fillna("")
        df = df[["text", "label"]]

        # Remove rows where text is empty or whitespace
        df = df[df["text"].str.strip() != ""]

        # Remove rows where label is not 0 or 1
        df = df[df["label"].isin([0, 1])]

        # Drop rows where label is an empty string
        df = df[df["label"].astype(str).str.strip() != ""]

        df = df.drop_duplicates()

        # Save the preprocessed data
        df.to_csv(os.path.join(processed_dir, "data.csv"), index=False)
        print(f"TREC {dataset_version} dataset processed and saved.")


def init_datasets() -> None:
    preprocess_enron()
    # preprocess_ling()
    # preprocess_sms()
    # preprocess_spamassassin()
    preprocess_trec()


def train_val_test_split(df, train_size=0.8, has_val=True):
    """Return a tuple (DataFrame, DatasetDict) with a custom train/val/split"""
    # Convert int train_size into float
    if isinstance(train_size, int):
        train_size = train_size / len(df)

    # Shuffled train/val/test split
    df = df.sample(frac=1, random_state=0)
    df_train, df_test = train_test_split(
        df, test_size=1 - train_size, stratify=df["label"]
    )

    if has_val:
        df_test, df_val = train_test_split(
            df_test, test_size=0.5, stratify=df_test["label"]
        )
        return (
            (df_train, df_val, df_test),
            datasets.DatasetDict(
                {
                    "train": datasets.Dataset.from_pandas(df_train),
                    "val": datasets.Dataset.from_pandas(df_val),
                    "test": datasets.Dataset.from_pandas(df_test),
                }
            ),
        )

    else:
        return (
            (df_train, df_test),
            datasets.DatasetDict(
                {
                    "train": datasets.Dataset.from_pandas(df_train),
                    "test": datasets.Dataset.from_pandas(df_test),
                }
            ),
        )
