import os
import shutil
from unittest.mock import patch, MagicMock
import pytest
from pipelines.cnn.prepare import get_df
from pipelines.cnn.load_data import load_data_to

@pytest.fixture
def fake_dataset(tmp_path):
    """
    Create a fake dataset directory structure for testing.
    """
    root = tmp_path / "fake_kaggle_data"
    root.mkdir()

    
    file1 = root / "file1.txt"
    file1.write_text("hello")

    
    folder = root / "images"
    folder.mkdir()
    (folder / "img1.jpg").write_text("image")

    return root


@patch("kagglehub.dataset_download")
def test_load_data_to(mock_download, fake_dataset, tmp_path):
    

    
    mock_download.return_value = str(fake_dataset)

    destination = tmp_path / "output"
    os.makedirs(destination, exist_ok=True)

    
    load_data_to(str(destination))

    
    assert (destination / "file1.txt").exists()
    assert (destination / "images").is_dir()
    assert (destination / "images" / "img1.jpg").exists()

    
    mock_download.assert_called_once_with("luluw8071/brain-tumor-mri-datasets")








def create_fake_dataset(root):
    """
    Structure:
    root/
      yes/img1.jpg
      yes/img2.jpg
      no/img3.jpg
    """
    yes_dir = root / "yes"
    no_dir = root / "no"
    yes_dir.mkdir()
    no_dir.mkdir()

    (yes_dir / "img1.jpg").write_text("fake")
    (yes_dir / "img2.jpg").write_text("fake")
    (no_dir / "img3.jpg").write_text("fake")

    return root



def test_get_df(tmp_path):
    dataset = create_fake_dataset(tmp_path)

    images, labels = get_df(str(dataset))

    assert len(images) == 3
    assert len(labels) == 3

    
    assert labels.count("yes") == 2
    assert labels.count("no") == 1

    
    for img in images:
        assert os.path.exists(img)



