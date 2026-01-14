import os
import yaml
import torch
from data.custom_dataset import CustomDataset
from options.train_options import TrainOptions
from PIL import Image
import numpy as np

def create_dummy_images(root):
    os.makedirs(root, exist_ok=True)
    Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)).save(os.path.join(root, 'clear.png'))
    Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)).save(os.path.join(root, 'turbid1.png'))
    Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)).save(os.path.join(root, 'turbid2.png'))
    
    config = {
        'pairs': [
            {
                'good': 'clear.png',
                'bads': ['turbid1.png', 'turbid2.png'],
                'caption': 'A dummy underwater image'
            }
        ]
    }
    with open(os.path.join(root, 'data.yaml'), 'w') as f:
        yaml.dump(config, f)

def test_custom_dataset():
    root = 'test_custom_data'
    create_dummy_images(root)
    
    # Mock options
    class MockOpt:
        dataroot = root
        yaml_path = os.path.join(root, 'data.yaml')
        dataset_mode = 'custom'
        isTrain = True
        batch_size = 1
        serial_batches = True
        num_threads = 1
        max_dataset_size = float('inf')
        preprocess = 'resize'
        load_size = 256
        crop_size = 256
        no_flip = True
        
    opt = MockOpt()
    
    try:
        dataset = CustomDataset(opt)
        print(f"Dataset length: {len(dataset)}")
        item = dataset[0]
        print("Success! Keys in item:", item.keys())
        print("A shape:", item['A'].shape)
        print("B shape:", item['B'].shape)
        print("A range:", item['A'].min().item(), item['A'].max().item())
        print("B range:", item['B'].min().item(), item['B'].max().item())
        print("A path:", item['A_paths'])
        print("B path:", item['B_paths'])
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        import shutil
        if os.path.exists(root):
            shutil.rmtree(root)

if __name__ == '__main__':
    test_custom_dataset()
